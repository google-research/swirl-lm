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

"""Library for stretched grid functionality, especially initialization."""

from typing import TypeAlias

from swirl_lm.base import initializer
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.utility import common_ops
from swirl_lm.utility import stretched_grid_util
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldMap: TypeAlias = types.FlowFieldMap
COORDINATE_KEYS_3D = ('xx', 'yy', 'zz')


def _deriv_centered(s: tf.Tensor) -> tf.Tensor:
  """Computes ds/dq on nodes, assuming grid spacing Δq = 1.

  A 2nd-order accurate calculation is used.

  Args:
    s: A 1D array of values.

  Returns:
    A 1D array of values representing the derivative of `s` with respect to
    `q`, where `q` is the coordinate variable.
  """
  left = [-1.5 * s[0] + 2.0 * s[1] - 0.5 * s[2]]
  middle = (s[2:] - s[:-2]) / 2.0
  right = [1.5 * s[-1] - 2.0 * s[-2] + 0.5 * s[-3]]
  return tf.concat([left, middle, right], axis=0)


def _deriv_node_to_face(s: tf.Tensor) -> tf.Tensor:
  """Computes ds/dq on faces, assuming grid spacing Δq = 1.

  A 2nd-order-accurate calculation is used.

  Args:
    s: A 1D array of values on nodes.

  Returns:
    A 1D array of values representing the derivative of `s` with respect to
    `q`, evaluated on faces, where `q` is the coordinate variable (with Δq = 1).
    Values on faces at coordinate location i - 1/2 are at index i.
  """
  left = [-2.0 * s[0] + 3.0 * s[1] - s[2]]
  middle = s[1:] - s[:-1]
  return tf.concat([left, middle], axis=0)


def compute_h_and_hface_from_coordinate_levels(
    s: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
  """Computes h = ds/dq on nodes and faces from global coordinate levels `s`.

  Args:
    s: A 1D array of coordinate values on nodes.

  Returns:
    A tuple of 1D arrays of values representing the scale factors h = ds/dq on
    nodes and faces, respectively.
  """
  h = _deriv_centered(s)
  h_face = _deriv_node_to_face(s)
  return h, h_face


def _get_h_with_halos_periodic(
    global_coord_no_halos: tf.Tensor,
    halo_width: int,
    domain_size: float,
) -> tuple[tf.Tensor, tf.Tensor]:
  """Gets the stretched-grid scale factors for a periodic dimension."""
  # We extend the coordinates into the halos first, then compute the scale
  # factors. When extending, we initially pad each end with one more point than
  # necessary. The reason is that for periodic consistency, we want to avoid
  # using one-sided derivatives, and stick to centered derivatives, which are
  # used for interior points.
  if global_coord_no_halos.shape[0] < halo_width + 1:
    raise ValueError(
        'Using a stretched grid in a periodic dimension requires at least'
        ' `halo_width` + 1 levels to be provided, but only'
        f' {global_coord_no_halos.shape[0]} levels were provided.'
    )

  pad_left = global_coord_no_halos[-(1 + halo_width) :] - domain_size
  pad_right = global_coord_no_halos[: halo_width + 1] + domain_size
  global_coord = tf.concat([pad_left, global_coord_no_halos, pad_right], axis=0)

  global_h, global_h_face = compute_h_and_hface_from_coordinate_levels(
      global_coord
  )

  # Remove the extra point from each end.
  return global_h[1:-1], global_h_face[1:-1]


def _get_h_with_halos_nonperiodic(
    global_coord_with_halos: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
  """Gets the stretched-grid scale factors for a nonperiodic dimension."""
  return compute_h_and_hface_from_coordinate_levels(global_coord_with_halos)


def local_stretched_grid_vars_from_global_xyz(
    params: parameters_lib.SwirlLMParameters,
    logical_coordinates: types.ReplicaCoordinates,
) -> FlowFieldMap:
  """Returns the local variables required for stretched grids.

  For dimensions in which stretched grids are used, given global coordinate
  arrays (excluding halos) contained in `params`, obtain the scale factors
  needed for stretched grids that are local to this replica. In addition, the
  coordinate is saved as a local 3D field.

  Args:
    params: An instance of the `SwirlLMParameters` specifying a simulation
      configuration.
    logical_coordinates: A tuple of logical coordinates for this replica.

  Returns:
    A dictionary of stretched grid variables local to this replica. If stretched
    grid is used in dim 0, the dict contains 3D field `xx`, and 1D fields
    `stretched_grid_h0` and `stretched_grid_h0_face`. If stretched grid is used
    in dim 1, the dict contains keys `yy`, `stretched_grid_h1` and
    `stretched_grid_h1_face`. If stretched grid is used in dim 2, the dict
    contains keys `zz`, `stretched_grid_h2`, and `stretched_grid_h2_face`.
  """
  core_n = (params.core_nx, params.core_ny, params.core_nz)
  n = (params.nx, params.ny, params.nz)

  local_vars = {}
  for dim in (0, 1, 2):
    if not params.use_stretched_grid[dim]:
      continue

    global_coord_no_halos = params.global_xyz[dim]
    global_coord = params.global_xyz_with_halos[dim]

    # From the coordinates without halos, get the global arrays for the
    # coordinates & the scale factors including boundary halos.  The arrays with
    # halos included are needed when extracting a local slice for each replica.
    if params.periodic_dims[dim]:
      # If dim 0 is a periodic dimension and is stretched, then `params.lx` is
      # interpreted as the total domain size, and not the distance between the
      # first and last grid point. This is done because the total domain size
      # is required information to fully specify the periodicity.
      domain_size = (params.lx, params.ly, params.lz)[dim]
      global_h, global_h_face = _get_h_with_halos_periodic(
          global_coord_no_halos, params.halo_width, domain_size
      )
    else:  # Non-periodic dimension
      global_h, global_h_face = _get_h_with_halos_nonperiodic(global_coord)

    # Get local slices from the 1D global arrays.
    coord_local = common_ops.get_local_slice_of_1d_array(
        global_coord, logical_coordinates[dim], core_n[dim], n[dim]
    )
    h_local = common_ops.get_local_slice_of_1d_array(
        global_h, logical_coordinates[dim], core_n[dim], n[dim]
    )
    h_face_local = common_ops.get_local_slice_of_1d_array(
        global_h_face, logical_coordinates[dim], core_n[dim], n[dim]
    )

    coord_local_3d = common_ops.convert_to_3d_tensor_and_tile(
        coord_local, dim, params.nx, params.ny, params.nz
    )

    # Reshape the local 1D arrays for broadcastable form for later use.
    h_local = initializer.reshape_to_broadcastable(h_local, dim)
    h_face_local = initializer.reshape_to_broadcastable(h_face_local, dim)

    local_vars[f'{COORDINATE_KEYS_3D[dim]}'] = coord_local_3d
    local_vars[stretched_grid_util.h_key(dim)] = h_local
    local_vars[stretched_grid_util.h_face_key(dim)] = h_face_local

  return local_vars


def additional_and_helper_var_keys(
    use_stretched_grid: tuple[bool, bool, bool], use_3d_tf_tensor: bool
) -> tuple[list[str], list[str]]:
  """Determines the required additional keys for stretched grids.

  The "additional keys" are for fields that get split into lists of tensors.
  For `dim` = 0, 1, and 2, the required additional keys are `xx`, `yy`, and
  `zz`, respectively. The coordinate fields are saved this way as 3D fields (as
  opposed to 1D fields as part of helper variables) for backwards compatibility
  reasons.

  Args:
    use_stretched_grid: Tuple of bools indicating whether to use stretched grid
      in each dimension.
    use_3d_tf_tensor: Whether 3D fields are represented by 3D tensors or lists
      of 2D tensors.

  Returns:
    The stretched-grid keys for additional and helper variables.
  """
  additional_keys = []
  helper_var_keys = []
  for dim in (0, 1, 2):
    if not use_stretched_grid[dim]:
      continue

    additional_keys.append(COORDINATE_KEYS_3D[dim])

    # For 3D tensors, place the stretched grid factors as helper_var keys.
    # For lists of 2D tensors, place the stretched grid factors as
    # additional_keys so they get unstacked into lists at the beginning of
    # `driver._one_cycle()`.
    if use_3d_tf_tensor:
      helper_var_keys.append(stretched_grid_util.h_key(dim))
      helper_var_keys.append(stretched_grid_util.h_face_key(dim))
    else:
      additional_keys.append(stretched_grid_util.h_key(dim))
      additional_keys.append(stretched_grid_util.h_face_key(dim))

  return additional_keys, helper_var_keys
