# Copyright 2026 The swirl_lm Authors.
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

# Copyright 2021 Google LLC
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
"""Stretched grid scale factor computation (JAX port).

Port of `swirl_lm.utility.stretched_grid` to JAX. Computes the metric
scale factors h = ds/dq (derivative of physical coordinate w.r.t.
computational coordinate, where Δq = 1) needed for stretched grids.

Scale factors are computed on both nodes (`h`) and faces (`h_face`) for
each stretched dimension. The face value at index i represents the
value at position i − 1/2 (i.e. the face between nodes i-1 and i).
"""


import jax.numpy as jnp
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.utility import stretched_grid_util

# 3D coordinate field key names: xx for dim 0, yy for dim 1, zz for dim 2.
COORDINATE_KEYS_3D = ('xx', 'yy', 'zz')


def _deriv_centered(s: jnp.ndarray) -> jnp.ndarray:
  """Computes ds/dq on nodes, assuming grid spacing Δq = 1.

  Uses 2nd-order accurate centered finite differences in the interior
  and 2nd-order one-sided stencils at the boundaries.

  Args:
    s: 1D array of coordinate values.

  Returns:
    1D array of ds/dq values at each node.
  """
  left = jnp.array([-1.5 * s[0] + 2.0 * s[1] - 0.5 * s[2]])
  middle = (s[2:] - s[:-2]) / 2.0
  right = jnp.array([1.5 * s[-1] - 2.0 * s[-2] + 0.5 * s[-3]])
  return jnp.concatenate([left, middle, right], axis=0)


def _deriv_node_to_face(s: jnp.ndarray) -> jnp.ndarray:
  """Computes ds/dq on faces, assuming grid spacing Δq = 1.

  Values on faces at coordinate location i − 1/2 are at index i.
  Uses a 2nd-order one-sided stencil for the first face.

  Args:
    s: 1D array of coordinate values at nodes.

  Returns:
    1D array of ds/dq values at each face.
  """
  left = jnp.array([-2.0 * s[0] + 3.0 * s[1] - s[2]])
  middle = s[1:] - s[:-1]
  return jnp.concatenate([left, middle], axis=0)


def compute_h_and_hface_from_coordinate_levels(
    s: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes h = ds/dq on nodes and faces from global coordinate levels `s`.

  Args:
    s: 1D array of coordinate values.

  Returns:
    Tuple (h, h_face) where h is the scale factor on nodes and h_face
    is the scale factor on faces.
  """
  h = _deriv_centered(s)
  h_face = _deriv_node_to_face(s)
  return h, h_face


def _get_h_with_halos_periodic(
    global_coord_no_halos: jnp.ndarray,
    halo_width: int,
    domain_size: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Gets the stretched-grid scale factors for a periodic dimension.

  For periodic dimensions, the coordinate array is extended by wrapping
  points from each end (shifted by domain_size) before computing
  derivatives. This avoids one-sided stencils entirely.

  Args:
    global_coord_no_halos: 1D array of coordinate values (no halos).
    halo_width: Number of halo layers.
    domain_size: Size of the periodic domain.

  Returns:
    Tuple (global_h, global_h_face) with halos included.
  """
  if global_coord_no_halos.shape[0] < halo_width + 1:
    raise ValueError(
        f'Global coordinate array size ({global_coord_no_halos.shape[0]}) '
        f'must be at least halo_width + 1 ({halo_width + 1}).'
    )
  pad_left = global_coord_no_halos[-(1 + halo_width) :] - domain_size
  pad_right = global_coord_no_halos[: halo_width + 1] + domain_size
  global_coord = jnp.concatenate(
      [pad_left, global_coord_no_halos, pad_right], axis=0
  )
  global_h, global_h_face = compute_h_and_hface_from_coordinate_levels(
      global_coord
  )
  # Remove the extra point from each end.
  return global_h[1:-1], global_h_face[1:-1]


def _get_h_with_halos_nonperiodic(
    global_coord_with_halos: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Gets the stretched-grid scale factors for a non-periodic dimension.

  For non-periodic dimensions, the coordinate array already includes
  halo points, so derivatives are computed directly.

  Args:
    global_coord_with_halos: 1D array of coordinate values including halos.

  Returns:
    Tuple (global_h, global_h_face).
  """
  return compute_h_and_hface_from_coordinate_levels(global_coord_with_halos)


def _reshape_to_broadcastable(
    arr: jnp.ndarray,
    dim: int,
) -> jnp.ndarray:
  """Reshapes a 1D array for broadcasting along dimension `dim`.

  Args:
    arr: 1D array of length N.
    dim: Target dimension (0, 1, or 2).

  Returns:
    Array reshaped to be broadcastable: (N,1,1) for dim=0, (1,N,1) for
    dim=1, (1,1,N) for dim=2.
  """
  shape = [1, 1, 1]
  shape[dim] = arr.shape[0]
  return arr.reshape(shape)


def local_stretched_grid_vars_from_global_xyz(
    params: parameters_lib.SwirlLMParameters,
    logical_coordinates: tuple[int, int, int],
) -> dict[str, jnp.ndarray]:
  """Returns the local variables required for stretched grids.

  For dimensions in which stretched grids are used, given global coordinate
  arrays (excluding halos) contained in `params`, computes the scale factors
  needed for stretched grids that are local to this replica.

  Args:
    params: The simulation parameters (must have grid_params with global_xyz,
      global_xyz_with_halos, and use_stretched_grid).
    logical_coordinates: A tuple of logical coordinates for this replica in each
      dimension (replica_x, replica_y, replica_z).

  Returns:
    A dictionary of stretched grid variables local to this replica. For
    each stretched dimension d (0 for x, 1 for y, 2 for z), contains:
      - COORDINATE_KEYS_3D[d]: 3D coordinate field
      - stretched_grid_util.h_key(d): scale factor on nodes (broadcastable)
      - stretched_grid_util.h_face_key(d): scale factor on faces (broadcastable)
  """
  gp = params.grid_params
  core_n_xyz = (gp.core_nx, gp.core_ny, gp.core_nz)
  n_xyz = (gp.nx, gp.ny, gp.nz)
  domain_sizes_xyz = (gp.lx, gp.ly, gp.lz)
  use_stretched_grid_xyz = gp.to_xyz_order((
      params.use_stretched_grid[0],
      params.use_stretched_grid[1],
      params.use_stretched_grid[2],
  ))
  periodic_dims_xyz = gp.to_xyz_order(gp.periodic_dims)
  full_shape = gp.to_data_axis_order(gp.nx, gp.ny, gp.nz)

  local_vars: dict[str, jnp.ndarray] = {}
  for dim in range(3):
    if not use_stretched_grid_xyz[dim]:
      continue

    axis = ('x', 'y', 'z')[dim]
    tensor_axis = gp.get_axis_index(axis)
    assert isinstance(tensor_axis, int)

    global_coord_no_halos = gp.global_xyz[tensor_axis]
    global_coord_with_halos = gp.global_xyz_with_halos[tensor_axis]

    # Compute scale factors from global coordinates.
    if periodic_dims_xyz[dim]:
      global_h, global_h_face = _get_h_with_halos_periodic(
          global_coord_no_halos, gp.halo_width, domain_sizes_xyz[dim]
      )
    else:
      global_h, global_h_face = _get_h_with_halos_nonperiodic(
          global_coord_with_halos
      )

    # Slice to local replica.
    replica_idx = logical_coordinates[dim]
    start = replica_idx * core_n_xyz[dim]

    coord_local = jnp.array(global_coord_with_halos[start : start + n_xyz[dim]])
    h_local = jnp.array(global_h[start : start + n_xyz[dim]])
    h_face_local = jnp.array(global_h_face[start : start + n_xyz[dim]])

    # Create 3D coordinate field by tiling along the other two dimensions.
    coord_3d = _reshape_to_broadcastable(coord_local, tensor_axis)
    coord_3d = jnp.broadcast_to(coord_3d, full_shape)

    # Reshape scale factors for broadcastable use.
    h_local = _reshape_to_broadcastable(h_local, tensor_axis)
    h_face_local = _reshape_to_broadcastable(h_face_local, tensor_axis)

    local_vars[COORDINATE_KEYS_3D[dim]] = coord_3d
    local_vars[stretched_grid_util.h_key(dim)] = h_local
    local_vars[stretched_grid_util.h_face_key(dim)] = h_face_local

  return local_vars
