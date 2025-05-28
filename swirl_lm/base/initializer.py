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

"""Library for initializing the variables on TPU cores."""

import enum
from typing import Callable, List, Literal, Optional, Sequence, Text, Tuple, Union

import numpy as np
from swirl_lm.utility import common_ops
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf

TensorOrArray = Union[tf.Tensor, np.ndarray]
ThreeIntTuple = Union[np.ndarray, tf.Tensor, Tuple[int, int, int]]
ValueFunction = Callable[[
    TensorOrArray, TensorOrArray, TensorOrArray, float, float, float,
    ThreeIntTuple
], tf.Tensor]

DEFAULT_PERMUTATION = (2, 0, 1)
_DEFAULT_PAD_MODE = 'CONSTANT'
_NP_DTYPE = types.NP_DTYPE


class MeshChoice(enum.Enum):
  """The choice of mesh generation for the full domain."""
  UNKNOWN = 0
  PARAMS = 1
  # The `DERIVED` method will be deprecated in future versions.
  DERIVED = 2


def partial_mesh_for_core(
    params: grid_parametrization.GridParametrization,
    coordinate: ThreeIntTuple,
    value_fn: ValueFunction,
    perm: Optional[ThreeIntTuple] = DEFAULT_PERMUTATION,
    pad_mode: Optional[Text] = _DEFAULT_PAD_MODE,
    mesh_choice: MeshChoice = MeshChoice.DERIVED,
) -> tf.Tensor:
  """Generates a partial mesh of a given value function for a core.

  The full grid spec is provided by `params`. The value function `value_fn`
  takes a 3-D mesh grid and corresponding lengths in three different dimensions
  as arguments. It returns a partial mesh with the corresponding values for the
  core at the coordinate specified by `coordinate`.

  NB: `perm` and `pad_mode` have defaults if the parameters are not provided.
  This is in contrast to passing the value `None`, which means, `do not
  transpose` and `do not pad`, respectively.

  Args:
    params: A grid_parametrization.GridParametrization object containing the
      required grid spec information.
    coordinate: A vector/sequence of integer with length 3 representing the
      logical coordinate of the core in the logical mesh [x, y, z].
    value_fn: A function that takes the local mesh_grid tensor for the core (in
      order x, y, z), the global characteristic length floats (in order x, y, z)
      and the local core coordinate, and returns a 3-D tensor representing the
      value for the local core (without including the margin/overlap between the
      cores).
    perm: A 3-tuple that defines the permutation ordering for the returned
      tensor. The default is (2, 0, 1). If `None`, permutation is not applied.
    pad_mode: Defines the padding applied the returned tensor. Must be
      'CONSTANT', 'REFLECT', 'SYMMETRIC', 'PHYSICAL', or `None`. The default is
      'CONSTANT'. If `None`, padding is not applied. If 'PHYSICAL', the halos
      are filled with physically correct values.
    mesh_choice: Use mesh from `params` if equals `MeshChoice.PARAMS`, and
      derive the mesh from `core_n`, `num_cores`, and `length` if equals
      `MeshChoice.DERIVED`. Note that the `mesh_choice` is `MeshChoice.PARAMS`
      if `pad_mode` is 'PHYSICAL'.

  Returns:
    A 3-D tensor representing the mesh of the local core. The values are valid
    only within the 'core' portion of the sub-grid. The halos are filled with 0.

  Raises:
    ValueError: If arguments are incorrect.
  """

  def get_slice_in_dim(core_n, length, num_cores, core_id, provided_mesh):
    """Returns the portion of the (sub) grid in the given dimension.

    Note that on each side we pad one grid point regardless of the halo width.
    Below, all arguments are implicitly in the given dimension corresponding to
    the context in which this helper is being called.

    Args:
      core_n: The value of params.core_n*.
      length: The spatial extent of the grid.
      num_cores: The total number of cores.
      core_id: The index of the core in {0, 1, ... num_cores - 1}.
      provided_mesh: Global mesh, provided from `params`.

    Returns:
      The subgrid corresponding to the portion of the grid in the given
        dimension assigned to the `core_id`.
    """
    if not core_n:
      return [_NP_DTYPE(0.0)]

    if mesh_choice == MeshChoice.DERIVED:
      mesh = tf.linspace(
          _NP_DTYPE(0.0),
          _NP_DTYPE(length),
          num_cores * core_n,
      )
    else:
      mesh = provided_mesh

    return common_ops.get_local_slice_of_1d_array(mesh, core_id, core_n, core_n)

  lx = params.lx
  ly = params.ly
  lz = params.lz
  cx = params.cx
  cy = params.cy
  cz = params.cz
  core_nx = params.core_nx
  core_ny = params.core_ny
  core_nz = params.core_nz
  nx = params.nx
  ny = params.ny
  nz = params.nz
  padwidth_x = (nx - core_nx) // 2 if core_nx else 0
  padwidth_y = (ny - core_ny) // 2 if core_ny else 0
  padwidth_z = (nz - core_nz) // 2 if core_nz else 0
  gx = coordinate[0]
  gy = coordinate[1]
  gz = coordinate[2]
  # These assert ops will be ignored on TPU. Force to place on CPU in case the
  # function is used outside initialization stage (which is already on CPU).
  with tf.device('CPU'):
    tf.debugging.assert_greater_equal(
        gx, 0,
        'Invalid subgrid coordinate specified with negative x core index.')
    tf.debugging.assert_greater(
        cx, gx,
        'Invalid subgrid coordinate specified with x core index. Must be '
        'smaller than total number of core partitioning in x direction.')
    tf.debugging.assert_greater_equal(
        gy, 0, 'Invalid subgrid coordinate specified with negative y core '
        'index.')
    tf.debugging.assert_greater(
        cy, gy,
        'Invalid subgrid coordinate specified with y core index. Must be '
        'smaller than total number of core partitioning in y direction.')
    tf.debugging.assert_greater_equal(
        gz, 0,
        'Invalid subgrid coordinate specified with negative z core index.')
    tf.debugging.assert_greater(
        cz, gz,
        'Invalid subgrid coordinate specified with z core index. Must be '
        'smaller than total number of core partitioning in z direction.')

  if pad_mode == 'PHYSICAL':
    xs = common_ops.get_local_slice_of_1d_array(
        params.global_xyz_with_halos[0], coordinate[0], core_nx, nx
    )
    ys = common_ops.get_local_slice_of_1d_array(
        params.global_xyz_with_halos[1], coordinate[1], core_ny, ny
    )
    zs = common_ops.get_local_slice_of_1d_array(
        params.global_xyz_with_halos[2], coordinate[2], core_nz, nz
    )
  else:
    xs = get_slice_in_dim(core_nx, lx, cx, gx, params.x)
    ys = get_slice_in_dim(core_ny, ly, cy, gy, params.y)
    zs = get_slice_in_dim(core_nz, lz, cz, gz, params.z)

  xx, yy, zz = tf.meshgrid(xs, ys, zs, indexing='ij')
  val = value_fn(xx, yy, zz, _NP_DTYPE(lx), _NP_DTYPE(ly), _NP_DTYPE(lz),  # pytype: disable=wrong-arg-types  # numpy-scalars
                 coordinate)
  if pad_mode and pad_mode != 'PHYSICAL':
    val = tf.pad(
        val,
        paddings=[
            [padwidth_x, padwidth_x],
            [padwidth_y, padwidth_y],
            [padwidth_z, padwidth_z],
        ],
        mode=pad_mode)
  if perm:
    val = tf.transpose(val, perm=perm)
  return val


def reshape_to_broadcastable(
    f_1d: tf.Tensor, dim: Literal[0, 1, 2]
) -> tf.Tensor:
  """Reshapes a rank-1 tensor to a form broadcastable against 3D fields.

  This function is appropriate for initialization and storing of 1D arrays, to
  be used later on in the simulation.

  Note: do not use this function inside of `partial_mesh_for_core`. That
  function expects dimensions to be ordered (x,y,z), whereas this function
  outputs dimensions with order (z,x,y).


  Here, `dim` is 0, 1, or 2, corresponding to dimension x, y, or z respectively.
  The rank-1 tensor `f_1d` will be reshaped such that it represents a 3D field
  whose values vary only along dimension `dim`. However, for memory efficiency,
  the number of elements do not change. The output can be used in operations
  with 3D fields, with broadcasting occurring.

  The number of elements of `f_1d` must be correct on input (this is NOT
  checked). That is, if `dim`==0, 1, or 2, then len(f_1d) must equal nx, ny, or
  nz, respectively, where `nx`, `ny`, `nz` are the corresponding sizes of 3D
  fields.

  Args:
    f_1d: A rank-1 tensor.
    dim: The dimension of variation of the input tensor `f_1d`.

  Returns:
    The reshaped tensor that can be broadcast against a 3D field.
  """
  assert (
      f_1d.ndim == 1
  ), f'Expecting rank-1 tensor, got rank-{f_1d.ndim} tensor.'
  if dim == 0:
    return f_1d[tf.newaxis, :, tf.newaxis]  # Set tensor shape to (1, nx, 1).
  elif dim == 1:
    return f_1d[tf.newaxis, tf.newaxis, :]  # Set tensor shape to (1, 1, ny).
  else:  # dim == 2
    return f_1d[:, tf.newaxis, tf.newaxis]  # Set tensor shape to (nz, 1, 1).


# Below are convenience wrappers of some initialization functions.
def gen_circular_u(params, coordinate, omega=0.1, perm=DEFAULT_PERMUTATION):
  """A simple wrapper for generating circular U field."""

  def circular_u_fn(xx, yy, zz, lx, ly, lz, coord):
    del xx, zz, lx, lz, coord
    return omega * (yy - 0.5 * ly)

  return partial_mesh_for_core(params, coordinate, circular_u_fn, perm)


def gen_circular_v(params, coordinate, omega=0.1, perm=DEFAULT_PERMUTATION):
  """A simple wrapper for generating circular V field."""

  def circular_v_fn(xx, yy, zz, lx, ly, lz, coord):
    del yy, zz, ly, lz, coord
    return omega * (xx - 0.5 * lx)

  return partial_mesh_for_core(params, coordinate, circular_v_fn, perm)


def gen_cone(params,
             coordinate,
             cone_height=3.87,
             inverse_sqrt_cone_width=0.0032,
             perm=DEFAULT_PERMUTATION):
  """A simple wrapper for generating circular cone function."""

  # Many of the initialization parameters for forcing function and other
  # functions should be encapsulated in a model specific param object.
  def gen_cone_fn(xx, yy, zz, lx, ly, lz, coord):
    del zz, lz, coord
    return tf.clip_by_value(
        cone_height - cone_height *
        tf.math.sqrt(inverse_sqrt_cone_width *
                     (xx - 0.75 * lx)**2 + inverse_sqrt_cone_width *
                     (yy - 0.5 * ly)**2),
        clip_value_min=0.0,
        clip_value_max=cone_height)

  return partial_mesh_for_core(params, coordinate, gen_cone_fn, perm)


def gen_smooth_sin(params, coordinate, amp=1e-4, perm=DEFAULT_PERMUTATION):
  """A simple wrapper for generating smooth sinusoidal input field."""
  init_sin_shift_y = params.init_sin_shift_y
  init_sin_shift_z = params.init_sin_shift_z

  def gen_smooth_sin_fn(xx, yy, zz, lx, ly, lz, coord):
    del coord
    return amp * (
        tf.math.sin(2 * np.pi * (xx / lx - 0.5)) *
        tf.math.sin(2 * np.pi * (yy / ly - 0.5) + init_sin_shift_y) *
        tf.math.sin(2 * np.pi * (zz / lz - 0.5) + init_sin_shift_z))

  return partial_mesh_for_core(params, coordinate, gen_smooth_sin_fn, perm)


def gen_forcing(params, coordinate, alpha_max=0.05, perm=DEFAULT_PERMUTATION):
  """A simple wrapper for generating a forcing field."""
  inverse_sq_jet_size_x = params.inverse_sq_jet_size_x
  inverse_sq_jet_size_y = params.inverse_sq_jet_size_y
  inverse_sq_jet_size_z = params.inverse_sq_jet_size_z
  jet_center_x = params.jet_center_x
  jet_center_y = params.jet_center_y
  jet_center_z = params.jet_center_z

  def gen_forcing_fn(xx, yy, zz, lx, ly, lz, coord):
    del coord
    r = tf.clip_by_value(
        4.0 * tf.math.sqrt(inverse_sq_jet_size_x *
                           (xx - jet_center_x * lx)**2 + inverse_sq_jet_size_y *
                           (yy - jet_center_y * ly)**2 + inverse_sq_jet_size_z *
                           (zz - jet_center_z * lz)**2),
        clip_value_min=0,
        clip_value_max=1)
    return alpha_max * (1 + tf.math.cos(np.pi * r))

  return partial_mesh_for_core(params, coordinate, gen_forcing_fn, perm)


def subgrid_slice_indices(
    subgrid_size: int,
    coordinate: int,
    halo_width: int = 1,
) -> Tuple[int, int]:
  """Determines the start and end indices for slicing."""
  core_subgrid_size = subgrid_size - 2 * halo_width
  start = coordinate * core_subgrid_size
  return start, start + subgrid_size


def subgrid_slice(
    subgrid_size: int, coordinate: int, halo_width: Optional[int] = 1
) -> slice:
  """Returns the slice of a field corresponding to `coordinate`.

  Args:
    subgrid_size: The size of the subgrid (including the halo).
    coordinate: The subgrid's coordinate.
    halo_width: The width of the halo.

  Returns:
    The subgrid slice corresponding to the given subgrid coordinate (including
    halo).
  """
  start, end = subgrid_slice_indices(subgrid_size, coordinate, halo_width)

  return slice(start, end)


def three_d_subgrid_slices(
    subgrid_size: ThreeIntTuple,
    coordinates: ThreeIntTuple,
    halo_width: Optional[int] = 1) -> Tuple[slice, slice, slice]:
  """Returns the 3 slices of a 3D field corresponding to `coordinates`.

  Args:
    subgrid_size: The size of the subgrid (including the halo).
    coordinates: The coordinates of the subgrid.
    halo_width: The width of the halo.

  Returns:
    The subgrid slices corresponding to the given subgrid coordinates.
  """
  return tuple([
      subgrid_slice(ss, c, halo_width)
      for ss, c in zip(subgrid_size, coordinates)
  ])


def subgrid_of_3d_grid(
    full_3d_grid: Union[TensorOrArray, List[TensorOrArray]],
    subgrid_size: ThreeIntTuple,
    coordinates: ThreeIntTuple,
    halo_width: Optional[int] = 1) -> Union[TensorOrArray, List[TensorOrArray]]:
  """Returns the subgrid of `full_3d_grid` corresponding to `coordinates`.

  The `full_3d_grid` can have shape `(nx, ny, nz)` or can be a list of length
    `nz`, where the shape of each element in the list is `(nx, ny)`.

  All the points in the subgrid come from the input. This is as opposed to most
  `gen_*` functions which return a subgrid with invalid values on the borders.

  Args:
    full_3d_grid: A 3D grid as a 3D tensor or array or a list of 2D tensors or
      arrays.
    subgrid_size: The size of the subgrid (including halo).
    coordinates: The core coordinates of the subgrid to return.
    halo_width: The halo width.

  Returns:
    The requested subgrid as a 3D tensor or array or list of 2D tensors or
      arrays.
  """
  x_slice, y_slice, z_slice = three_d_subgrid_slices(subgrid_size, coordinates,
                                                     halo_width)

  if isinstance(full_3d_grid, list):
    return [xy[x_slice, y_slice] for xy in full_3d_grid[z_slice]]
  else:
    return full_3d_grid[x_slice, y_slice, z_slice]


def subgrid_of_3d_grid_from_params(
    full_3d_grid: Union[TensorOrArray, List[TensorOrArray]],
    params: grid_parametrization.GridParametrization,
    coordinates: ThreeIntTuple) -> Union[TensorOrArray, List[TensorOrArray]]:
  """Returns the subgrid of `full_3d_grid` corresponding to `coordinates`."""
  if params.cx * params.cy * params.cz == 1:
    return full_3d_grid

  return subgrid_of_3d_grid(full_3d_grid, (params.nx, params.ny, params.nz),
                            coordinates, params.halo_width)


def subgrid_of_3d_tensor(full_3d_grid: tf.Tensor,
                         subgrid_shape: Sequence[int],
                         coordinates: ThreeIntTuple,
                         halo_width: Optional[int] = 1) -> tf.Tensor:
  """Similar to `subgrid_of_3d_grid`, but for a tensor."""
  core_subgrid_shape = np.array([s - 2 * halo_width for s in subgrid_shape])
  begin = core_subgrid_shape * coordinates

  return tf.slice(full_3d_grid, begin, subgrid_shape)


def subgrid_of_2d_grid(full_2d_grid: TensorOrArray,
                       params: grid_parametrization.GridParametrization,
                       coordinates: ThreeIntTuple) -> TensorOrArray:
  """Returns the subgrid of `full_2d_grid` corresponding to `coordinates`.

  All the points in the subgrid come from the supplied tensor. This is as
  opposed to most other `gen_*` functions which return a subgrid with valid
  values only in the core region, and invalid values on the borders.

  Args:
    full_2d_grid: A 2D tensor.
    params: A GridParametrization instance.
    coordinates: The core coordinates of the subgrid to return.

  Returns:
    The requested subgrid.
  """
  cxi, cyi, _ = coordinates
  core_nx, core_ny = params.core_nx, params.core_ny
  halo_width = params.halo_width

  nx_start = cxi * core_nx
  x_slice = slice(nx_start, nx_start + core_nx + 2 * halo_width)

  ny_start = cyi * core_ny
  y_slice = slice(ny_start, ny_start + core_ny + 2 * halo_width)

  return full_2d_grid[x_slice, y_slice]


def three_d_subgrid_of_2d_grid(
    full_2d_grid: TensorOrArray,
    params: grid_parametrization.GridParametrization,
    coordinates: ThreeIntTuple) -> TensorOrArray:
  """Same as `subgrid_of_2d_grid`, but with an added trivial third dimension."""
  sub_grid = subgrid_of_2d_grid(full_2d_grid, params, coordinates)
  expand_dims = (
      tf.expand_dims if isinstance(full_2d_grid, tf.Tensor) else np.expand_dims)
  return expand_dims(sub_grid, axis=0)


def three_d_subgrid_of_2d_border_strip(
    border_strip: np.ndarray,
    params: grid_parametrization.GridParametrization,
    coordinates: ThreeIntTuple,
    strip_width: int = 1) -> np.ndarray:
  """Returns the subgrid of `border_strip` corresponding to `coordinates`.

  All the points in the subgrid come from the supplied array. This is as opposed
  to most `gen_*` functions which return a subgrid with valid values only in the
  core region, and invalid values on the borders.

  Args:
    border_strip: A 2D array of dimension `(border_strip, fy)` or `(fx,
      border_strip)`, where the full grid has dimension `(fx, fy)`.
    params: A GridParametrization instance.
    coordinates: The core coordinates of the subgrid to return.
    strip_width: The width of the border strip.

  Returns:
    The requested subgrid.
  """
  fx, fy = border_strip.shape
  if strip_width not in (fx, fy):
    raise ValueError('The border strip must have dimension '
                     f'{strip_width} in the x- or y- dimension '
                     f'(got ({fx}, {fy})).')

  cxi, cyi, _ = coordinates
  core_nx, core_ny = params.core_nx, params.core_ny

  if strip_width == fx:
    x_slice = slice(None)
  else:
    nx_start = cxi * core_nx
    x_slice = slice(nx_start, nx_start + params.nx)

  if strip_width == fy:
    y_slice = slice(None)
  else:
    ny_start = cyi * core_ny
    y_slice = slice(ny_start, ny_start + params.ny)
  return np.expand_dims(border_strip[x_slice, y_slice], axis=0)


def gen_partial_from_full_2d(field_2d: np.ndarray,
                             params: grid_parametrization.GridParametrization,
                             coordinates: ThreeIntTuple) -> np.ndarray:
  """Returns a 3d subgrid of a full 2d grid corresponding to `coordinate`.

  All the points in the subgrid come from the supplied 2d matrix. This is as
  opposed to most other `gen_*` functions which return a subgrid with valid
  values only in the core region, and invalid values on the borders.

  Args:
    field_2d: A 2D numpy array.
    params: A GridParametrization instance.
    coordinates: The core coordinates of the subgrid to return.

  Returns:
    The requested subgrid, as a 3d numpy array.
  """
  return np.expand_dims(
      subgrid_of_2d_grid(field_2d, params, coordinates), axis=0)
