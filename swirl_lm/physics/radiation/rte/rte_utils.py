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

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility library for solving the radiative transfer equation (RTE)."""

import inspect
from typing import Callable, Dict, List, Optional, Tuple, Literal

import numpy as np
from swirl_lm.communication import halo_exchange
from swirl_lm.utility import common_ops
from swirl_lm.utility import grid_extension
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf


FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap

X0_KEY = 'x0'
PRIMARY_GRID_KEY = 'primary'
EXTENDED_GRID_KEY = 'extended'


class RTEUtils:
  """A library for distributing radiative transfer computations on TPU's.

  Attributes:
    params: An instance of `GridParametrization` containing the grid dimensions
    and information about the TPU computational topology.
    grid_extension_lib: An instance of `GridExtension`, a library for
      facilitating TPU communication when an extended grid is present.
    num_cores: A 3-tuple containing the number of cores assigned to each
      dimension.
    grid_size: The local grid dimensions per core.
    halos: The number of halo points on each face of the grid.
  """

  def __init__(
      self,
      params: grid_parametrization.GridParametrization,
      grid_extension_lib: Optional[grid_extension.GridExtension] = None,
  ):
    self.params = params
    self.grid_extension_lib = grid_extension_lib
    self.num_cores = (params.cx, params.cy, params.cz)
    self.grid_size = (params.nx, params.ny, params.nz)
    self.halos = params.halo_width

  def _append(
      self,
      a: FlowFieldVal,
      b: FlowFieldVal,
      dim: int,
      forward: bool = True,
  ) -> FlowFieldVal:
    """Appends `a` to `b` along `axis` if `forward` and `b` to `a` otherwise."""
    if not forward:
      a, b = b, a
    a_is_tensor = isinstance(a, tf.Tensor)
    assert a_is_tensor == isinstance(b, tf.Tensor)

    if not a_is_tensor and dim == 2:  # Append to Python list.
      return a + b

    if a_is_tensor:
      # Handles the case of single 3D tensor. Shifts `dim` to conform with the
      # 2-0-1 3D tensor orientation.
      axis = (dim + 1) % 3
    else:
      axis = dim

    return tf.nest.map_structure(
        lambda x, y: tf.concat([x, y], axis=axis), a, b
    )

  def _pad(
      self,
      f: FlowFieldVal,
      low_n: int,
      high_n: int,
      dim: int
  ) -> FlowFieldVal:
    """Pads the field with zeros along the dimension `dim`."""
    paddings = [(0, 0)] * 3
    paddings[dim] = (low_n, high_n)
    return common_ops.pad(f, paddings)

  def _generate_adjacent_pair_assignments(
      self, replicas: np.ndarray, axis: int, forward: bool
  ) -> List[np.ndarray]:
    """Creates groups of source-target TPU devices along `axis`.

    The group assignments are used by `tf.raw.CollectivePermute` to exchange
    data between neighboring replicas along `axis`. There will be one such group
    for every interface of the topology. As an example consider a `replicas` of
    shape `[4, 2, 1]` where the replica id's are 0 through 7. Each replica is
    assigned to a unique 3-tuple `coordinate`. In this example, the mapping of
    coordinates to replica ids is `{(0, 0, 0): 0, (0, 1, 0): 1, (1, 0, 0): 2,
    (1, 1, 0): 3, (2, 0, 0): 4, (2, 1, 0): 5, (3, 0, 0): 6, (3, 1, 0): 7}` with
    each element following the form `(coordinate[0], coordinate[1],
    coordinate[2]): replica_id`. The group assignment along dimension 0 contains
    the replica ids with the same coordinate of dimensions 1 and 2 that neighbor
    each other along `axis` in the direction of increasing index if `forward` is
    set to `True`, and in the direction of decreasing index otherwise. In this
    example, if `axis` is 0 and `forward` is `True`, the assignments are:

    [[[0, 2], [1, 3]],
     [[[2, 4], [3, 5]],
     [[[4, 6], [5, 7]]].

    If `forward` is `False`, the pairs for each interface are reversed:

    [[2, 0], [3, 1]],
     [[4, 2], [5, 3]],
     [[6, 4], [7, 5]]].

    Args:
      replicas: The mapping from the core coordinate to the local replica id.
      axis: The axis along which data will be propagated.
      forward: Whether the data propagation of `tf.raw_ops.CollectivePermute`
        will unravel in the direction of increasing index along `axis`.

    Returns:
      A list of groups of adjacent replica id pairs. There will be one such
      group for every interface of the computational topology along `axis`.
    """
    groups = common_ops.group_replicas(replicas, axis=axis)
    pair_groups = []
    depth = groups.shape[1]
    for i in range(depth - 1):
      pair_group = groups[:, i : i + 2]
      if not forward:
        # Reverse the order of the source-target pairs.
        pair_group = pair_group[:, ::-1]
      pair_groups.append(pair_group.tolist())
    return pair_groups

  def _local_recurrent_op(
      self,
      recurrent_fn: Callable[..., tf.Tensor],
      variables: FlowFieldMap,
      dim: int,
      n: int,
      forward: bool = True,
  ) -> Tuple[FlowFieldVal, FlowFieldVal]:
    """Computes a sequence of recurrent operations along a dimension.

    Each core performs the same operation on data local to it independently.
    Note that the initial input `x0` in `variables` is not included in the final
    output.

    Args:
      recurrent_fn: The local cumulative recurrent operation.
      variables: A dictionary containing the local fields that will be inputs to
        `recurrent_fn`. One of the entries must be `x0`, which should correspond
        to the boundary solution that initiates the recurrence. `x0` has the
        same structure as other fields in `variables` (i.e. either a 3-D
        tf.Tensor or a list of 2D `tf.Tensor`s) but has a size of 1 along the
        axis determined by `dim`.
      dim: The physical dimension along which the sequence of affine
        transformations will be applied.
      n: The number of layers in the final solution.
      forward: Whether the accumulation starts with the first layer of
        coefficients. If set to False, then the recurrence relation unravels
        from the last layer to the first as follows:
        x[i] = recurrent_fn(x[i + 1]).

    Returns:
      A tuple containing 1) A 3D variable with the cumulative output from the
      chain of recurrent transformations having the same structure and shape as
      any field in `variables` that is not `x0`. 2) the 2D output of the last
      recurrent transformation.
    """
    x = variables[X0_KEY]

    for i in range(n):
      prev_idx = i - 1
      slice_idx = i if forward else -i - 1
      plane_args = {
          k: common_ops.slice_field(v, dim, slice_idx, size=1)
          for k, v in variables.items()
          if k != X0_KEY
      }
      prev_slice_idx = prev_idx if forward else -prev_idx - 1
      plane_args[X0_KEY] = (
          x
          if i == 0
          else common_ops.slice_field(x, dim, prev_slice_idx, size=1)
      )
      arg_lst = [
          plane_args[k] for k in inspect.getfullargspec(recurrent_fn).args
      ]
      next_layer = tf.nest.map_structure(recurrent_fn, *arg_lst)
      x = next_layer if i == 0 else self._append(x, next_layer, dim, forward)

    last_layer = -1 if forward else 0
    last_local_layer = common_ops.slice_field(x, dim, last_layer, size=1)

    return x, last_local_layer

  def _cumulative_recurrent_op_sequential(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      recurrent_fn: Callable[..., FlowFieldVal],
      variables: FlowFieldMap,
      dim: int,
      forward: bool = True,
  ) -> FlowFieldVal:
    """Computes a sequence of recurrent affine transformations globally.

    This particular implementation is sequential, so every layer of replicas
    along the accumulation axis needs to wait for the previous computational
    layer to complete before proceeding, which can be slow. On the other hand,
    this approach has a very small memory overhead, since the TPU communication
    only happens between pairs of adjacent computational layers through the
    `tf.raw_ops.CollectivePermute` operation.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      recurrent_fn: The local cumulative recurrent operation.
      variables: A dictionary containing the local fields that will be inputs to
        `recurrent_fn`. One of the entries must be `x0`, which should correspond
        to the boundary solution that initiates the recurrence.
      dim: The physical dimension along which the sequence of affine
        transformations will be applied.
      forward: Whether the accumulation starts with the first layer of
        coefficients. If set to False, then the recurrence relation unravels
        from the last layer to the first as follows:
        x[i] = w[i] * x[i + 1] + b[i].

    Returns:
      A 3D variable with the cumulative output from the chain of recurrent
      transformations having the same structure and shape as any field in
      `variables` that is not `x0`.
    """
    halos = [0] * 3
    halos[dim] = self.halos

    n = self.grid_size[dim] - 2 * self.halos

    # Remove halos along the axis.
    kwargs = {
        k: common_ops.strip_halos(v, halos)
        for k, v in variables.items()
        if k != X0_KEY
    }
    kwargs[X0_KEY] = variables[X0_KEY]

    def local_fn(x0: FlowFieldVal) -> Tuple[FlowFieldVal, FlowFieldVal]:
      """Generates the output of a cumulative operation and its last layer."""
      kwargs[X0_KEY] = x0
      return self._local_recurrent_op(recurrent_fn, kwargs, dim, n, forward)

    def communicate_fn(x):
      return tf.raw_ops.CollectivePermute(
          input=x, source_target_pairs=pair_group
      )

    core_idx = common_ops.get_core_coordinate(replicas, replica_id)[dim]

    # Cumulative local output and its last layer. This result will only be valid
    # for the first level of the computational topology. All the subsequent
    # levels will need to wait for the output from the previous level before
    # evaluating their local function.
    x_cum, x_out = local_fn(x0=variables[X0_KEY])

    pair_groups = self._generate_adjacent_pair_assignments(
        replicas, dim, forward
    )
    n_groups = len(pair_groups)
    interface_iter = range(n_groups) if forward else reversed(range(n_groups))

    # Sequentially evaluate a level of the computational topology and propagate
    # results to the next level.
    for i in interface_iter:
      pair_group = pair_groups[i]
      # Send / receive the last recurrent output layer.
      x_prev = tf.nest.map_structure(communicate_fn, x_out)
      # Index of the next set of cores receiving the data.
      recv_core_idx = i + 1 if forward else i
      x_cum, x_out = tf.cond(
          tf.equal(core_idx, recv_core_idx),
          true_fn=lambda: local_fn(x0=x_prev),  # pylint: disable=cell-var-from-loop
          false_fn=lambda: (x_cum, x_out))

    # Pad the result with halo layers.
    return self._pad(x_cum, self.halos, self.halos, dim)

  def _exchange_halos(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      f: FlowFieldVal,
      x0: FlowFieldVal,
      dim: int,
      x0_face: Literal[0, 1],
  ) -> FlowFieldVal:
    """Exchanges halos along the specified dimension."""
    bc = [[(halo_exchange.BCType.NEUMANN, 0.0)] * 2 for _ in range(3)]
    # Set the boundary plane that initiates the recurrent operation as the
    # boundary values.
    bc[dim][x0_face] = (
        halo_exchange.BCType.DIRICHLET,
        [x0] * self.halos,
    )
    return halo_exchange.inplace_halo_exchange(
        f,
        (0, 1, 2),
        replica_id,
        replicas,
        (0, 1, 2),
        (False, False, False),
        boundary_conditions=bc,
        width=self.halos,
    )

  def _exchange_halos_with_ext_grid(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      f: FlowFieldVal,
      f_ext: FlowFieldVal,
      x0: FlowFieldVal,
      dim: int,
      x0_face: Literal[0, 1],
  ) -> Tuple[FlowFieldVal, FlowFieldVal]:
    """Exchanges halos consistently with the extended computational grid."""
    assert self.grid_extension_lib is not None, (
        'An instance of `GridExtension` is required for handling'
        ' `extended_grid_variables`.'
    )

    neumann_bc = [[(halo_exchange.BCType.NEUMANN, 0.0)] * 2 for _ in range(3)]
    bc_with_boundary_val = [
        [(halo_exchange.BCType.NEUMANN, 0.0)] * 2 for _ in range(3)
    ]
    # Set the plane that initiates the recurrent operation as boundary value.
    bc_with_boundary_val[dim][x0_face] = (
        halo_exchange.BCType.DIRICHLET,
        [x0] * self.halos,
    )
    # The boundary value stays with the logical grid that initiates the
    # recurrence: the primary grid if `forward` is `True` and the extended grid
    # otherwise.
    bc_primary = bc_with_boundary_val if x0_face == 0 else neumann_bc
    bc_extended = neumann_bc if x0_face == 0 else bc_with_boundary_val
    return self.grid_extension_lib.exchange_halos_with_extended_grid(
        replica_id,
        replicas,
        f,
        f_ext,
        bc_primary,
        bc_extended,
    )

  def cumulative_recurrent_op(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      recurrent_fn: Callable[..., FlowFieldVal],
      variables: FlowFieldMap,
      dim: int,
      forward: bool = True,
      extended_grid_variables: Optional[FlowFieldMap] = None,
  ) -> Dict[str, FlowFieldVal]:
    """Applies a recurrent operation globally along a specified dimension.

    This global operation is sequential and will process each layer along `dim`
    at a time in ascending order if `forward` is `True` and in descending order
    otherwise. As a consequence, if there are multiple TPU cores assigned to
    dimension `dim` in the computational topology some cores will experience
    idleness as they wait for previous layers residing in other cores to be
    processed.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      recurrent_fn: The cumulative recurrent operation.
      variables: A dictionary containing the local fields that will be inputs to
        `recurrent_fn`. One of the entries must be `x0`, which should correspond
        to the boundary solution that initiates the recurrence.
      dim: The physical dimension along which the sequence of affine
        transformations will be applied.
      forward: Whether the accumulation starts with the first layer of
        coefficients. If set to False, then the recurrence relation unravels
        from the last layer to the first as follows:
        x[i] = recurrent_fn({'x0': x[i + 1], ...})
      extended_grid_variables: An optional dictionary of state variables for the
        extended grid above the simulation domain. This will be used instead of
        `variables` when the recurrence unrolling reaches the extended domain.

    Returns:
      If `extended_grid_variables` is not provided:
        A dictionary containing a single entry ('primary') for the 3D field with
        the cumulative output from the chain of recurrent transformations having
        the same structure and shape as any field in `variables` that is not
        `x0`, and having `x0` as the boundary value at the face that initiates
        the recurrence.
      If `extended_grid_variables` is provided:
        A dictionary with one 3D field for the primary grid, with key 'primary',
        and one for the extended grid, with key 'extended'.
    """
    # Store the initial plane to be used as the boundary value when updating
    # halos.
    x0 = variables[X0_KEY]
    # Face that initiates the recurrence.
    x0_face = 0 if forward else 1
    # If the boundary condition is set along the list dimension, the halo
    # exchange expects the boundary plane represented as a 2D tensor. Otherwise,
    # a list of thin tensors of dimension (1, ny) or (nx, 1) is expected.
    if not isinstance(x0, tf.Tensor) and dim == 2:
      x0 = x0[0]

    def single_grid_op(variables):
      """Calls the recurrent operation for a single grid."""
      return self._cumulative_recurrent_op_sequential(
          replica_id, replicas, recurrent_fn, variables, dim, forward
      )

    if extended_grid_variables is None:
      val = single_grid_op(variables)
      return {
          'primary': self._exchange_halos(
              replica_id, replicas, val, x0, dim, x0_face
          )
      }

    assert self.grid_extension_lib is not None, (
        'An instance of `GridExtension` is required for handling'
        ' `extended_grid_variables`.'
    )

    # Swap the order of execution if the propagation is not forward.
    ordered_vars = [variables, extended_grid_variables]
    if not forward:
      ordered_vars.reverse()

    first_output = single_grid_op(ordered_vars[0])
    # The last output of the first grid becomes the initiating plane of the
    # second recurrent operation.
    second_vars = dict(ordered_vars[1])
    if self.num_cores[dim] == 1:
      last_plane_idx = -self.halos - 1 if forward else self.halos
      last_plane = common_ops.slice_field(
          first_output, dim, last_plane_idx, size=1
      )
    else:
      last_plane = self.grid_extension_lib.get_layer_across_interface(
          replica_id, replicas, first_output, first_output, layer_idx=self.halos
      )
    second_vars[X0_KEY] = last_plane
    second_output = single_grid_op(second_vars)
    outputs = [first_output, second_output]
    # Revert to natural order of logical grids, where the extended grid is
    # always above the primary grid.
    if not forward:
      outputs.reverse()

    outputs = self._exchange_halos_with_ext_grid(
        replica_id, replicas, outputs[0], outputs[1], x0, dim, x0_face,
    )
    return {
        PRIMARY_GRID_KEY: outputs[0],
        EXTENDED_GRID_KEY: outputs[1],
    }
