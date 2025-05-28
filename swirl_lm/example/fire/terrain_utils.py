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

"""A library for incorporating terrain information in wildfire simulations.

The terrain is currently represented using the immersed boundary (IB) method.
functions in this library with prefix 'ib_' are utility functions for the
immersed boundary method.
"""

import os
from typing import Tuple, Union

import numpy as np
from swirl_lm.base import initializer
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.boundary_condition import immersed_boundary_method
from swirl_lm.communication import halo_exchange
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import init_fn as init_fn_lib
from swirl_lm.utility import types
import tensorflow as tf

gfile = tf.io.gfile
_TF_DTYPE = tf.float32

InitFn = initializer.ValueFunction
ThreeIntTuple = initializer.ThreeIntTuple
RangeIndices = Tuple[int, int]
RangeIndices3D = Tuple[RangeIndices, RangeIndices, RangeIndices]
FlowFieldVal = types.FlowFieldVal


def generate_terrain_map_from_file(
    config: parameters_lib.SwirlLMParameters,
    input_terrain_filepath: str,
) -> tf.Tensor:
  """Creates a 2D terrain map of size (nx, ny).

  Args:
    config: An instance of `SwirlLMParameters` that stores parameters of a
      simulation the with incompressible NS solver.
    input_terrain_filepath: The full path to a data file that contains a 2D
      array representing the terrain of an area.

  Returns:
    A 2D tf.Tensor that has the same size of the full horizontal mesh that
    represents the height of a terrain on each mesh point.

  Raises:
    ValueError: If the terrain file extension is not '.ser' or '.npy'.
  """
  terrain_filepath = input_terrain_filepath

  ext = os.path.splitext(terrain_filepath)

  if ext[-1] == '.ser':
    raw_map = tf.io.parse_tensor(tf.io.read_file(terrain_filepath),
                                 out_type=tf.float32)
    # This is needed to give enough hint of the shape to allow tf.image.resize
    # to work.
    raw_map.set_shape([None, None])
  elif ext[-1] == '.npy':
    with gfile.GFile(terrain_filepath, 'rb') as f:
      raw_map = np.load(f)
  else:
    raise ValueError(
        f'Unsupported terrain file type {ext[-1]}. Available options are:'
        '".ser" ".npy".'
    )

  return tf.image.resize(
      tf.expand_dims(raw_map, axis=-1), [config.fx, config.fy],
      method='bicubic')[:, :, 0]


class TerrainUtils(object):
  """A library for terrain processing in wildfire simulations."""

  def __init__(
      self,
      config: parameters_lib.SwirlLMParameters,
      elevation_map: Union[np.ndarray, tf.Tensor],
  ):
    """Initializes required parameters for terrain processing."""
    self.config = config
    self.elevation_map = elevation_map

  def _get_local_range_indices(
      self,
      coordinates: ThreeIntTuple,
  ) -> RangeIndices3D:
    """Generates the range indices for the local mesh."""
    i_start = coordinates[0] * self.config.core_nx
    i_end = (coordinates[0] + 1) * self.config.core_nx
    j_start = coordinates[1] * self.config.core_ny
    j_end = (coordinates[1] + 1) * self.config.core_ny
    k_start = coordinates[2] * self.config.core_nz
    k_end = (coordinates[2] + 1) * self.config.core_nz
    return ((i_start, i_end), (j_start, j_end), (k_start, k_end))

  def local_elevation_map(
      self,
      coordinates: ThreeIntTuple,
      offset: float = 0.0,
  ) -> tf.Tensor:
    """Generates the elevation map corresponds to the specified coordinates.

    Args:
      coordinates: The location of the current TPU replica in the TPU grid.
      offset: The offset in the z (height) direction with respect to the
        elevation map.

    Returns:
      A 3D tensor with elevation information corrected by `offset` at location
      specified by coordinates. The size of last dimension of the tensor is 1
      for the convenience of broadcasting in conditional operations.
    """
    indices = self._get_local_range_indices(coordinates)
    return tf.expand_dims(
        tf.convert_to_tensor(
            self.elevation_map[slice(indices[0][0], indices[0][1]),
                               slice(indices[1][0], indices[1][1])] + offset,
            dtype=_TF_DTYPE), 2)

  def ib_flow_field_mask_fn(
      self,
      coordinates: ThreeIntTuple,
  ) -> InitFn:
    """Generates a function that initializes the mask for the flow field."""
    local_map = self.local_elevation_map(coordinates)

    def init_fn(xx: tf.Tensor, yy: tf.Tensor, zz: tf.Tensor, lx: float,
                ly: float, lz: float, coord: ThreeIntTuple) -> tf.Tensor:
      """Initializes the flow field mask."""
      del xx, yy, lx, ly, lz, coord
      return tf.compat.v1.where(zz > local_map,
                                tf.ones_like(zz, dtype=_TF_DTYPE),
                                tf.zeros_like(zz, dtype=_TF_DTYPE))

    return init_fn

  def ib_boundary_mask_fn(
      self,
      coordinates: ThreeIntTuple,
  ) -> InitFn:
    """Generates a function that initializes the mask for the flow field."""
    local_map = self.local_elevation_map(coordinates)

    def init_fn(xx: tf.Tensor, yy: tf.Tensor, zz: tf.Tensor, lx: float,
                ly: float, lz: float, coord: ThreeIntTuple) -> tf.Tensor:
      """Initializes the flow field mask."""
      del xx, yy, lx, ly, lz, coord
      return tf.compat.v1.where(
          tf.math.logical_and(zz <= local_map, zz > local_map - self.config.dz),
          tf.ones_like(zz, dtype=_TF_DTYPE), tf.zeros_like(zz, dtype=_TF_DTYPE))

    return init_fn

  def compute_boundary_weights(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      ib_mask: FlowFieldVal,
      ib_boundary_mask: FlowFieldVal,
  ) -> FlowFieldVal:
    """Compute the weights at the boundary for the Cartesian grid method.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The topography of the TPU replicas.
      kernel_op: An object holding a library of kernel operations.
      ib_mask: A mask with binary floating point values. 1 represents the flow
        field, and 0 represents the immersed body.
      ib_boundary_mask: A mask with binary floating point values. 1 represents
        the layer of fluid that is closest to the actual solid boundary, and 0
        represents elsewhere.

    Returns:
      The weights at the boundary points, which is 1/number of neighboring fluid
      nodes where `ib_boundary_mask` = 1, and 0 elsewhere.
    """

    def mask_halo_update(mask: FlowFieldVal) -> FlowFieldVal:
      """Updates the halos with actual boundary treated as Neumann BC."""
      return halo_exchange.inplace_halo_exchange(
          mask,
          [0, 1, 2],
          replica_id,
          replicas,
          [0, 1, 2],
          [False, False, False],
          [[
              (halo_exchange.BCType.NEUMANN, 0.0),
          ] * 2] * 3,
          self.config.halo_width,
      )

    return (immersed_boundary_method
            .update_cartesian_grid_method_boundary_coefficients(
                mask_halo_update(ib_boundary_mask), mask_halo_update(ib_mask),
                kernel_op))

  def blasius_uvw_init_fn(
      self,
      u_inf: float,
      v_inf: float,
      nu: float,
      dx: float,
      dy: float,
      lz: float,
      nz: float,
      x: float,
      apply_transition: bool,
      transition_fraction: float,
      coordinates: ThreeIntTuple,
  ) -> init_fn_lib.InitFnDict:
    """Generates initialization functions for u, v, and w.

    Args:
      u_inf: The free stream velocity in the x direction.
      v_inf: The free stream velocity in the y direction.
      nu: The kinematic viscosity.
      dx: The grid spacing in the x direction.
      dy: The grid spacing in the y direction.
      lz: The height of the domain.
      nz: The number of mesh points in the vertical dimension.
      x: The distance from which the boundary layer profile is computed.
      apply_transition: An indicator of whether the boundary layer profile is
        transitioned from surface normal to coordinates aligned. If `True`, the
        boundary layer profile will be transitioned at a fraction of the domain
        height; otherwise it'll stay as the proflie normal to the wall along the
        vertical direction.
      transition_fraction: The fraction of the boundary layer that's considered
        as normal to the ground.
      coordinates: The location of the current TPU replica in the TPU grid.

    Returns:
      A dictionary of initialization functions for u, v, and w.
    """
    return init_fn_lib.blasius_boundary_layer(
        u_inf,
        v_inf,
        nu,
        dx,
        dy,
        lz,
        nz,
        x,
        self.local_elevation_map(coordinates),
        apply_transition=apply_transition,
        transition_fraction=transition_fraction)


# TODO(b/258264622): Update the function names to represent that these functions
# now supports generic variables.
def terrain_utils_factory(
    config: parameters_lib.SwirlLMParameters,
    input_terrain_filepath: str,
) -> TerrainUtils:
  """Creates a `TerrainUtils` object.

  Args:
    config: An instance of `SwirlLMParameters` that stores parameters of a
      simulation the with incompressible NS solver.
    input_terrain_filepath: The full path to a data file that contains a 2D
      array representing the terrain of an area.

  Returns:
    A `TerrainUtils` object that generates terrain related variables for a
    wildfire simulation.
  """
  actual_map = generate_terrain_map_from_file(config, input_terrain_filepath)

  return TerrainUtils(config, actual_map)
