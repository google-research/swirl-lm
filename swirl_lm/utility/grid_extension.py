# Copyright 2025 The swirl_lm Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A library for extending the logical grid above the simulation domain."""
from typing import Tuple

import numpy as np
from swirl_lm.communication import halo_exchange
from swirl_lm.utility import common_ops
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf


FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap
GridParametrization = grid_parametrization.GridParametrization

GRID_EXTENSION_KEY_PREFIX = 'extended'


class GridExtension:
  """A library for managing TPU communication on an extended logical grid.

  Often in fluid flow simulations, it is necessary to account for the transport
  of energy far away from the limited domain where the LES dynamics is resolved.
  One such example is radiative transfer in the atmosphere. Although the LES
  attempts to slice a limited portion of the atmosphere and treat it in
  isolation from all the rest, in actuality the energetics of the planetary
  boundary layer is very much influenced by how ultraviolet and thermal
  radiation is being transmitted, scattered, and absorbed even in regions as far
  up as the stratosphere. For this reason, to accurately capture the effects of
  radiative transfer it is necessary to solve a nonlocal equation with the same
  horizontal granularity as the LES, but possibly much coarser vertical
  resolution, extending all the way up to the stratosphere. Such nonlocal
  processes are the main use case for this utility.

  This library allows an extended domain to be appended to the higher end of the
  original domain by overlaying a secondary (extended) grid on the same
  computational TPU topology. The extension is assumed to be along a single
  dimension `dim` and no constraint is enforced on the spacing. In particular,
  it can be much coarser than that of the primary grid.

  The partitions of the extended grid are assigned to the TPU topology in a
  circular scheme whereby moving up from the top level of the computational
  topology takes one back to the first level. Each variable that needs to be
  represented in the extended grid and persisted through many steps must have a
  dedicated `additional_state` with the name of the corresponding state on the
  primary grid prefixed by 'extended_'.

  For instance, if one wishes to represent the state of name 'phi' on the
  extended grid, an additional state of name 'extended_phi' must be created by
  the user, and the replica at level j in the computational topology will hold
  both the usual j-th partition of the grid along `dim` as well as the j-th
  partition of the extended grid. Note that each replica will then hold a pair
  of possibly noncontiguous partitions of the full grid. The following is an
  illustration of a 1x3 computational topology supporting an extended grid:

  |replica_0|replica_1|replica_2|replica_0|replica_1|replica_2|
                                |
                        extension interface

  Similarly, the computational grid laid out sequentially will look like this:
                                |
  |  phi_0  |  phi_1  |  phi_2  |extended_phi_0|extended_phi_1|extended_phi_2|
                                |
                        extension interface

  Note that replica_1 stores both phi_1, which belongs to the primary grid, as
  well as extended_phi_1, which is a patch of the extended grid above.


  Attributes:
    dim: The dimension along which the extension is applied.
    params: An instance of `GridParametrization` containing the grid dimensions
      and information about the TPU computational topology.
    extended_grid_params: An instance of `GridParametrization` containing the
      grid dimensions and information about the TPU computational topology for
      the extended grid.
    num_cores: A 3-tuple containing the number of cores assigned to each
      dimension.
    grid_size: The local grid dimensions per core.
    halos: The number of halo points on each face of the subgrid.
  """

  def __init__(
      self,
      params: GridParametrization,
      extended_coordinate: np.ndarray,
      dim: int,
  ):
    if params.global_xyz[dim].shape[0] != extended_coordinate.shape[0]:
      # TODO(sheide): Support arbitrary extended grid shape and TPU topology as
      # long as the grid size is divisible by the number of cores assigned to
      # the extension.
      raise ValueError(
          'The number of grid points in the extension dimension must be the'
          ' same as the number of grid points in the primary grid in that'
          f' dimension ({params.global_xyz[dim].shape[0]} !='
          f' {extended_coordinate.shape[0]})'
      )
    self._num_cores = (params.cx, params.cy, params.cz)
    self._halos = params.halo_width
    self.dim = dim
    self._extended_coordinate = extended_coordinate
    self.params = params
    self.extended_grid_params = self._create_extended_grid_params()

  def _create_extended_grid_params(
      self,
  ) -> GridParametrization:
    new_grid_lengths = [self.params.lx, self.params.ly, self.params.lz]
    new_grid_lengths[self.dim] = (
        self._extended_coordinate[-1] - self._extended_coordinate[0]
    )
    return GridParametrization.create_from_grid_lengths_and_etc(
        new_grid_lengths,
        self.params.computation_shape,
        (self.params.nx, self.params.ny, self.params.nz),
        self.params.halo_width,
    )

  def get_helper_variables(
      self,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Returns a dictionary with just the grid extension helper variables."""
    return {
        key: additional_states[key]
        for key in additional_states
        if key.startswith(GRID_EXTENSION_KEY_PREFIX)
    }

  def get_layer_across_interface(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      x_primary: FlowFieldVal,
      x_extension: FlowFieldVal,
      layer_idx: int,
  ) -> FlowFieldVal:
    """Gets layer on the other side of the extension interface at `layer_idx`.

    If a TPU replica is adjacent to the extension interface (i.e., it is either
    in the first or last level of the topology along the extension dimension),
    that replica will receive the logical layer or plane (represented as a 3D
    field with a single dimension along `dim`) from the replica adjacent to it
    across the extension interface. For the replicas that are not adjacent to
    the extension interface this simply returns a plane of zeros. The extracted
    layer is at index `layer_idx` away from the interface and users must be
    cautious when specifying this index, e.g. if there are halo layers present
    that must be skipped over.

    For example, consider a 1x3 computational grid containing replicas 0, 1, and
    2. The extended computational topology will be as follows:

                                  |
    |replica_0|replica_1|replica_2|replica_0|replica_1|replica_2|
                                  |
                         extension interface

    Similarly, the computational grid laid out sequentially will look like this:

                                  |
    |subgrid_0|subgrid_1|subgrid_2|ext_grid_0|ext_grid_1|ext_grid_2|
                                  |
                         extension interface

    Note that replica_1, for instance, holds both subgrid_1, which belongs to
    the original grid, and ext_grid_1, which belongs to the extended grid.

    Assume 2 halo layers are used in a simulation and the primary and extended
    subgrids are represented in code by the variables `subgrid` and
    `extended_subgrid`.

    Calling `get_layer_across_interface(replica_id, replicas, subgrid,
    extended_subgrid, 2) will yield the following output in each replica:

    replica_0: The last fluid layer of subgrid_2 from replica_2.
    replica_1: A plane of zeros.
    replica_2: The first fluid layer of ext_grid_0 from replica_0.

    This is useful whenever one is computing a sequential operation that
    requires output from a previous level to be propagated in a given direction,
    as it will allow the propagation to move through the logical interface of
    the grid extension seamlessly. It can also be used for halo exchange, as the
    internal layers of replica_2 need to be communicated to replica_0 and
    vice-versa in a way that is aware of the extension.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      x_primary: A 3D field in the primary grid.
      x_extension: A 3D field in the extended grid.
      layer_idx: The index of the layer to be extracted from the logically
        adjacent replica counting from the face that is adjacent to the
        interface of the grid extension.

    Returns:
      A plane represented as a shallow 3D field with dimension 1 along the
      extension direction that is extracted from a logically adjacent replica
      across the extension interface and at index `layer_idx` counting from
      the interface side.
    """
    core_idx = common_ops.get_core_coordinate(replicas, replica_id)[self.dim]

    groups = common_ops.group_replicas(replicas, axis=self.dim)

    interface_pairs = np.concatenate([groups[:, -1:], groups[:, :1]], axis=-1)
    # Include the same pairs in the opposite direction.
    all_interface_pairs = (
        interface_pairs.tolist() + np.array(interface_pairs)[:, ::-1].tolist()
    )

    def communicate_fn(x):
      return tf.raw_ops.CollectivePermute(
          input=x, source_target_pairs=all_interface_pairs
      )

    last_core_idx = self._num_cores[self.dim] - 1
    # This will only be valid for the top layer of cores in the primary grid,
    # before the logical interface.
    layer_below_interface = common_ops.slice_field(
        x_primary, self.dim, (-layer_idx - 1), size=1
    )
    cond_layer_below_interface = tf.cond(
        pred=tf.equal(core_idx, last_core_idx),
        true_fn=lambda: layer_below_interface,
        false_fn=lambda: tf.nest.map_structure(
            tf.zeros_like, layer_below_interface
        ),
    )
    # This will only be valid for the first logical layer of cores in the
    # extended grid, immediately after the logical interface.
    layer_above_interface = common_ops.slice_field(
        x_extension, self.dim, layer_idx, size=1
    )
    layer_near_interface = tf.cond(
        pred=tf.equal(core_idx, 0),
        true_fn=lambda: layer_above_interface,
        false_fn=lambda: cond_layer_below_interface,
    )
    return tf.nest.map_structure(
        communicate_fn, layer_near_interface
    )

  def _exchange_halos_across_interface(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      x_primary: FlowFieldVal,
      x_extension: FlowFieldVal,
      halo_layer_idx: int,
  ) -> Tuple[FlowFieldVal, FlowFieldVal]:
    """Communicates individual halo layers across the extension interface.

    The `halo_layer_idx` is counted from the face adjacent to the interface.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      x_primary: A 3D field in the primary grid.
      x_extension: A 3D field in the extended grid.
      halo_layer_idx: The index of the halo layer to be exchanged across the
        extension interface, counting from the face adjacent to the interface
        and moving inward into the subgrid.

    Returns:
      A tuple with the primary grid field and the extended grid field, with
      halos for a given layer properly exchanged.
    """
    core_idx = common_ops.get_core_coordinate(replicas, replica_id)[self.dim]
    last_core_idx = self._num_cores[self.dim] - 1

    shape = common_ops.get_shape(x_primary)

    if replicas.shape[self.dim] > 1:
      layer_across_interface = self.get_layer_across_interface(
          replica_id,
          replicas,
          x_primary,
          x_extension,
          self._halos + halo_layer_idx,
      )
      x_primary = tf.cond(
          pred=tf.equal(core_idx, last_core_idx),
          true_fn=lambda: common_ops.tensor_scatter_1d_update(
              x_primary,
              self.dim,
              shape[self.dim] - (self._halos - halo_layer_idx),
              layer_across_interface,
          ),
          false_fn=lambda: x_primary,
      )
      x_extension = tf.cond(
          pred=tf.equal(core_idx, 0),
          true_fn=lambda: common_ops.tensor_scatter_1d_update(
              x_extension,
              self.dim,
              self._halos - 1 - halo_layer_idx,
              layer_across_interface,
          ),
          false_fn=lambda: x_extension,
      )
    # If there is only a single TPU layer in `dim`, no need for communication.
    else:
      index_from_face = self._halos + halo_layer_idx
      layer_below_interface = common_ops.slice_field(
          x_primary, self.dim, (-index_from_face - 1), size=1
      )
      layer_above_interface = common_ops.slice_field(
          x_extension, self.dim, index_from_face, size=1
      )
      x_primary = common_ops.tensor_scatter_1d_update(
          x_primary,
          self.dim,
          shape[self.dim] - (self._halos - halo_layer_idx),
          layer_above_interface,
      )
      x_extension = common_ops.tensor_scatter_1d_update(
          x_extension,
          self.dim,
          self._halos - 1 - halo_layer_idx,
          layer_below_interface,
      )

    return x_primary, x_extension

  def exchange_halos_with_extended_grid(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      x_primary: FlowFieldVal,
      x_extension: FlowFieldVal,
      bc_primary: halo_exchange.BoundaryConditionsSpec,
      bc_extension: halo_exchange.BoundaryConditionsSpec,
  ) -> Tuple[FlowFieldVal, FlowFieldVal]:
    """Exchanges halos between neighboring replicas in the extended topology.

    The halos of replicas that are not on the boundaries of the primary grid or
    the extended grid are exchanged in the usual way, but the ones surrounding
    the extension interface will respect the circular configuration of the
    extended topology.

    The cores on both sides of the logical interface between the primary grid
    and the extended grid are determined, and the layers that are closest to the
    interface are communicated to the replicas on the opposite side.

    For example, consider a 1x2 computational grid containing replicas 0 and 1
    The extended computational topology will be as follows:

                        |
    |replica_0|replica_1|replica_0|replica_1|
                        |
               extension interface

    Similarly, the full computational grid laid out sequentially will be:
                        |
    |subgrid_0|subgrid_1|ext_grid_0|ext_grid_1|
                        |
               extension interface

    The halos will be exchanged as follows:
    replica_0/subgrid_0 ↔ replica_1/subgrid_1
    replica_1/subgrid_1 ↔ replica_0/ext_grid_0 (across extension interface)
    replica_0/ext_grid_0 ↔ replica_1/ext_grid_1

    Note that the top halo layers of the primary grid may end up with different
    spacings than the original. If any computation relies on the values of these
    halo layers, the user of this library must account for it.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      x_primary: A 3D field in the primary grid.
      x_extension: A 3D field in the extended grid.
      bc_primary: Boundary condition spec for the primary grid.
      bc_extension: Boundary condition spec for the extended grid.

    Returns:
      A tuple of the primary subgrid state and the extended subgrid state, with
      updated halos.
    """
    def exchange_halos(f, bc):
      return halo_exchange.inplace_halo_exchange(
          f,
          (0, 1, 2),
          replica_id,
          replicas,
          (0, 1, 2),
          boundary_conditions=bc,
          width=self._halos,
      )

    x_primary = exchange_halos(x_primary, bc_primary)
    x_extension = exchange_halos(x_extension, bc_extension)

    # Still required to exchange halos between the first and last level of
    # replicas to account for the interface between the original and the
    # extended grid.
    for halo_layer_idx in range(self._halos):
      x_primary, x_extension = self._exchange_halos_across_interface(
          replica_id, replicas, x_primary, x_extension, halo_layer_idx,
      )

    return x_primary, x_extension
