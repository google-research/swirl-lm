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
"""Common grid parameterization."""

import itertools
from typing import Any, Literal, TypeAlias

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.utility import file_io
from swirl_lm.jax.utility import file_pb2
from swirl_lm.jax.utility import grid_parametrization_pb2
from swirl_lm.jax.utility import stretched_grid_util
from swirl_lm.jax.utility import types
from swirl_lm.utility import text_util

from google.protobuf import text_format

ScalarField: TypeAlias = types.ScalarField
ScalarFieldMap: TypeAlias = types.ScalarFieldMap


def _validate_grid_size_wrt_halo_width(
    n: int, halo_width: int, axis: str
) -> None:
  """Checks that the grid size is greater than 2 * halo width.

  Args:
    n: The grid size inside a core along a particular axis.
    halo_width: The halo width.
    axis: The axis of the grid.

  Raises:
    ValueError: If the grid size is not greater than 2 * halo width.
  """
  if n <= 2 * halo_width:
    raise ValueError(
        f'n{axis} should be greater than 2*halo_width, but got n{axis}={n} and'
        f' halo_width={halo_width}.'
    )


def _get_core_n(n: int, halo_width: int) -> int:
  core_n = n - 2 * halo_width
  return core_n


def _get_full_grid_size(n: int, halo_width: int, num_cores: int) -> int:
  """The full grid size (includes padding, if any)."""
  return num_cores * _get_core_n(n, halo_width)


def _get_full_uniform_grid(n: int, l: float) -> jax.Array:
  """The full grid without halos for a uniformly spaced mesh.

  Args:
    n: The total number of grid points without halos.
    l: The total length of the domain.

  Returns:
    A equidistant grid for the entire computational domain. The first grid point
    is 0.
  """
  # n_effective = n if n is not None else 1
  return jnp.linspace(0.0, l, n)


def get_physical_full_grid_size(
    params: grid_parametrization_pb2.GridParametrization,
) -> grid_parametrization_pb2.CoordinateInt:
  """Returns the full physical grid size.

  Args:
    params: The grid parametrization.

  Returns:
    The full physical grid size.

  Raises:
    ValueError: If the grid size is not greater than 2 * halo width.
  """
  # Some simulations have a physical grid size mandated externally, and add
  # padding in order that the internal grid has dimensions appropriate for
  # running on TPU (e.g. the grid sizes of dim 0 and 1 should be multiples of
  # 128). The full physical size is set here assuming there is no padding. If
  # there is padding, these values will be overridden.
  for grid_size in (
      params.grid_size.dim_x,
      params.grid_size.dim_y,
      params.grid_size.dim_z,
  ):
    if grid_size <= 2 * params.halo_width:
      raise ValueError(
          'Each entry in grid_size should be greater than 2*halo_width, but'
          f' got grid_size={params.grid_size} and'
          f' halo_width={params.halo_width}.'
      )
  return grid_parametrization_pb2.CoordinateInt(
      dim_x=_get_full_grid_size(
          params.grid_size.dim_x,
          params.halo_width,
          params.computation_shape.dim_x,
      ),
      dim_y=_get_full_grid_size(
          params.grid_size.dim_y,
          params.halo_width,
          params.computation_shape.dim_y,
      ),
      dim_z=_get_full_grid_size(
          params.grid_size.dim_z,
          params.halo_width,
          params.computation_shape.dim_z,
      ),
  )


def _get_grid_spacing(full_grid_size: int, length: float) -> float | None:
  """Get the grid spacing between nodes in a equidistant mesh.

  Args:
    full_grid_size: The total number of nodes in the mesh grid.
    length: The size of the domain in a particular dimension.

  Returns:
    The distance between two adjacent nodes.
  """
  return length / (full_grid_size - 1) if full_grid_size > 1 else None


def _get_stretched_grid_aware_grid_spacing_in_dim(
    dx_dy_dz: tuple[float | None, float | None, float | None],
    dim: Literal[0, 1, 2],
    use_stretched_grid_in_dim: bool,
) -> float:
  """Get the stretched-grid-aware grid spacing in a particular dimension.

  Return the grid spacing of the computational mesh, assumed uniform. When
  using a stretched grid, the grid spacing will be that of the transformed
  coordinate, not of the physical Cartesian coordinate values.

  Note: internally, when using a stretched grid where the coordinate levels
  are passed in, we will treat the grid spacing of the transformed coordinate
  as equal to 1.0, because the coordinate values of the transformed coordinate
  never need be referred to directly.

  Args:
    dx_dy_dz: Grid spacings in x, y and z in that order.
    dim: The dimension for which to get the grid spacing.
    use_stretched_grid_in_dim: Whether a stretched grid is used in the given
      dimension.

  Returns:
    The stretched-grid-aware grid spacing in the given dimension.
  """
  if not use_stretched_grid_in_dim:
    grid_spacing = dx_dy_dz[dim]
    # Note: GridParametrization.{dx,dy,dz} can return None depending on
    # parameter input. If this occurs, a None is not supposed to be used in
    # practice downstream. However, for type checking it is useful to keep the
    # return type as `float`. Setting the grid spacing to 0.0 in that case
    # ensures that if it gets inadvertently used there will be an immediate
    # error.
    if grid_spacing is None:
      grid_spacing = 0.0
  else:
    grid_spacing = 1.0
  return grid_spacing


def domain_size_from_periodic_stretched_grid(
    stretched_grid_file: file_pb2.File,
) -> float:
  """Returns the domain size of a periodic stretched grid.

  The input format for a periodic stretched grid differs from an unperiodic
  stretched grid. If there are N total grid points across all cores, excluding
  halos, then for an unperiodic grid, exactly N points are specified in the
  stretched grid file. For a periodic grid, N+1 points are specified, where the
  last point is the same as the first point. This extra point is used to
  specify the periodic domain length.

  Args:
    stretched_grid_file: The path to the stretched grid file.

  Returns:
    The domain size of the periodic stretched grid.
  """
  global_coord = _load_array_from_file(stretched_grid_file)
  return global_coord[-1] - global_coord[0]


def _validate_global_coord(
    global_coord: jax.Array,
    core_n: int,
    num_core: int,
    dim: Literal[0, 1, 2],
) -> None:
  """Checks that global coordinates have the correct number of elements.

  Global coordinates do not include the halos. The input args `core_n`,
  and `num_core` should come from a `GridParametrization` object. They should
  correspond to the `dim` value.

  Args:
    global_coord: The global coordinates.
    core_n: Number of non-halo points in this dimension.
    num_core: Number of cores in this dimension.
    dim: Which dimension is being considered (only used for an error message).

  Raises:
    ValueError: If `global_coord` does not have the correct number of
      elements.
  """

  global_grid_size = num_core * core_n  # Excludes halos.

  if global_coord.shape[0] != global_grid_size:
    raise ValueError(
        f'Global coordinates in dim {dim} has'
        f' {global_coord.shape[0]} elements, which does not match the'
        f' required number of elements {global_grid_size}.'
    )


def _load_array_from_file(array_file: file_pb2.File) -> np.ndarray:
  """Loads a 1D array from a text file and returns it as a tensor.

  Each element of the array should be on its own line.

  Args:
    array_file: The file to load.

  Returns:
    A 1D tensor of the data from the file.
  """
  contents = text_util.strip_line_comments(file_io.load_file(array_file), '#')
  return np.fromstring(contents, dtype=np.float64, sep='\n')


def _set_data_axis_order(data_axis_order: str) -> tuple[str, str, str]:
  """Returns a tuple of the axis order.

  Example: if data_axis_order is 'zxy', the return value is ('z', 'x', 'y').

  Args:
    data_axis_order: A string of length 3, representing the order of the axes.
      It should be a permutation of 'xyz'.

  Raises:
    ValueError: If `data_axis_order` is not of length 3 or is not a permutation
    of "xyz".

  Returns:
    A tuple of the axis order.
  """
  if tuple(data_axis_order) not in itertools.permutations('xyz'):
    raise ValueError(
        f'`data_axis_order` {data_axis_order} is not a permutation of "xyz".'
    )
  return tuple(data_axis_order)


def _global_xyz_from_config(
    stretched_grid_files: tuple[
        file_pb2.File | None, file_pb2.File | None, file_pb2.File | None
    ],
    full_grid_size: tuple[int, int, int],
    length: tuple[float, float, float],
    core_n: tuple[int, int, int],
    num_core: tuple[int, int, int],
    periodic: tuple[bool, bool, bool],
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Returns global coordinates, excluding halo points.

  For each dimension: if a stretched grid is provided then that grid is used,
  otherwise a uniform grid is used. All tuple arguments should be in
  data_axis_order. The output is also in data_axis_order.

  Args:
    stretched_grid_files: Paths to stretched-grid text files for each dimension.
    full_grid_size: The full grid size excluding halos in each dimension.
    length: The length of the domain in each dimension.
    core_n: The number of non-halo points in each dimension.
    num_core: The number of cores in each dimension.
    periodic: Whether each dimension is periodic.

  Returns:
    A tuple of global coordinates in each dimension.
  """

  def get_full_grid(dim: Literal[0, 1, 2]) -> jax.Array:
    if stretched_grid_files[dim]:
      global_coord = _load_array_from_file(stretched_grid_files[dim])
      logging.info(
          'Loaded stretched grid in dim %d from file `%s`.',
          dim,
          stretched_grid_files[dim],
      )
      # Periodic stretched grids have an extra point at the end of the grid
      # which is equivalent to the first point. This extra point is used to
      # specify the periodic domain length but is otherwise not part of the grid
      # of interior nodes. Here, get rid of this extra point.
      if periodic[dim]:
        global_coord = global_coord[:-1]
        logging.info(
            'Removing final point from periodic stretched grid in dim %d.', dim
        )
      global_coord = jnp.array(global_coord, dtype=jnp.float32)
    else:
      global_coord = _get_full_uniform_grid(full_grid_size[dim], length[dim])
    _validate_global_coord(global_coord, core_n[dim], num_core[dim], dim)
    return global_coord

  return tuple(get_full_grid(dim) for dim in (0, 1, 2))


# TODO(b/368405442): Adjust definition of a periodic uniform mesh to be
# consistent with the length defined in the config, and be consistent with the
# way the stretched mesh is defined.
def _extend_grid(
    grid: np.ndarray,
    halo_width: int,
    periodic_and_stretched: bool,
    domain_size: float,
) -> np.ndarray:
  """Extends a 1D grid with extrapolation into halos on the 2 ends.

  The grid spacing in the halo repeats its nearest neighbor in grid.

  Note that in case the mesh only has 1 point, we copy the value of this point
  to all halos.

  Args:
    grid: A 1D grid.
    halo_width: The number of points to be extended to the grid on each end of
      the grid.
    periodic_and_stretched: True if this dimension is both periodic and
      stretched.
    domain_size: The size of the domain in this dimension.

  Returns:
    The grid extended `halo_width` number of points with linear extrapolation on
    both ends.
  """

  def single_point() -> np.ndarray:
    """Generates a mesh of a same value when `grid` is a single point."""
    return grid[0] * np.ones((halo_width * 2 + 1), dtype=grid.dtype)

  def regular_mesh() -> np.ndarray:
    """Generates a mesh with linearly extrapolated halos."""
    d0 = grid[1] - grid[0]
    d1 = grid[-1] - grid[-2]
    ext = np.linspace(1, halo_width, halo_width, dtype=grid.dtype)
    ext_0 = grid[0] - d0 * ext[::-1]
    ext_1 = grid[-1] + d1 * ext
    return np.concat([ext_0, grid, ext_1], axis=0)

  def stretched_periodic_mesh() -> np.ndarray:
    """Generates a mesh with periodic extension into the halos."""
    assert domain_size > grid[-1] - grid[0], (
        'Domain size of a stretched periodic dimension must be larger than the'
        ' distance between the first and last grid point. In one dimension, the'
        f' domain size is {domain_size} while the distance between the first'
        f' grid point ({grid[0]}) and the last grid point ({grid[-1]}) is'
        f' {grid[-1] - grid[0]}.'
    )
    ext_left = grid[-halo_width:] - domain_size
    ext_right = grid[:halo_width] + domain_size
    return np.concat([ext_left, grid, ext_right], axis=0)

  if grid.shape[0] < 2:
    return single_point()
  elif periodic_and_stretched:
    return stretched_periodic_mesh()
  else:
    return regular_mesh()


def params_from_text_proto(
    text_proto: str,
) -> grid_parametrization_pb2.GridParametrization:
  """Returns a GridParametrization protobuf from a text-formatted proto."""
  # Get default values from flags.
  params = text_format.Parse(
      text_proto, grid_parametrization_pb2.GridParametrization()
  )
  # Clear physical_full_grid_size, which is a computed field. We'll recompute it
  # at the end.
  params.ClearField('physical_full_grid_size')
  params.MergeFrom(
      text_format.Parse(
          text_proto, grid_parametrization_pb2.GridParametrization()
      )
  )
  # Re-compute physical_full_grid_size if necessary.
  if not params.HasField('physical_full_grid_size'):
    params.physical_full_grid_size.CopyFrom(get_physical_full_grid_size(params))
  return params


class GridParametrization(object):
  """Holds configuration parameters.

  For computing dx, dy, dz below, we assume the 'box' boundaries coincide with
  the outer most grid points on each end -- the 'halo' grid. Assuming one halo
  point, this means there are total core * c + 2 points, or core * c + 1
  spacings.

  Note that all the 3-tuples in this class are in the order of data_axis_order.
  For example, if data_axis_order is 'zxy', self.grid_spacings is (dz, dx, dy).
  """

  def __init__(
      self,
      params: grid_parametrization_pb2.GridParametrization,
  ):
    """Creates an object from protobuf."""
    if not params.HasField('physical_full_grid_size'):
      params.physical_full_grid_size.CopyFrom(
          get_physical_full_grid_size(params)
      )
    if params.halo_width <= 0:
      raise ValueError('Halo width must be greater than 0.')
    self.grid_params_proto = params
    self.data_axis_order = _set_data_axis_order(params.data_axis_order)
    self.cx = params.computation_shape.dim_x
    self.cy = params.computation_shape.dim_y
    self.cz = params.computation_shape.dim_z
    self.lx = params.length.dim_x
    self.ly = params.length.dim_y
    self.lz = params.length.dim_z
    self.nx = params.grid_size.dim_x
    self.ny = params.grid_size.dim_y
    self.nz = params.grid_size.dim_z
    self.fx_physical = params.physical_full_grid_size.dim_x
    self.fy_physical = params.physical_full_grid_size.dim_y
    self.fz_physical = params.physical_full_grid_size.dim_z
    self.halo_width = params.halo_width
    self.dt = params.dt
    # "Recover" float64 precision for dt by rounding the 32 bit value to 6
    # significant digits and then converting back to float. The assumption is
    # that dt is user specified with 6 or less significant digits.
    self.dt64 = float(
        np.format_float_positional(params.dt, 6, fractional=False)
    )
    self.kernel_size = params.kernel_size

    self.use_stretched_grid = self.to_data_axis_order(
        params.stretched_grid_files.HasField('dim_x'),
        params.stretched_grid_files.HasField('dim_y'),
        params.stretched_grid_files.HasField('dim_z'),
    )

    # Whether the dimensions are periodic or not.
    self.periodic_dims = self.to_data_axis_order(
        params.periodic.dim_x,
        params.periodic.dim_y,
        params.periodic.dim_z,
    )

    # Get coordinate grid spacings (valid with or without stretched grid). To be
    # compatible with stretched grids, use these values instead of
    # self.{dx,dy,dz}.
    dx_uniform = _get_grid_spacing(self.fx, self.lx)
    dy_uniform = _get_grid_spacing(self.fy, self.ly)
    dz_uniform = _get_grid_spacing(self.fz, self.lz)
    self.grid_spacings: tuple[float, float, float] = tuple(
        _get_stretched_grid_aware_grid_spacing_in_dim(
            (dx_uniform, dy_uniform, dz_uniform),
            dim,
            self.to_xyz_order(self.use_stretched_grid)[dim],
        )
        for dim in (0, 1, 2)
    )
    self.grid_spacings = self.to_data_axis_order(*self.grid_spacings)

    # Tuple of global coordinate values for each dimension, including halos.
    # Consists of the stretched-grid coordinates (if used), otherwise defaults
    # to uniform grid.
    stretched_grid_files_xyz = [None] * 3
    if params.stretched_grid_files.HasField('dim_x'):
      stretched_grid_files_xyz[0] = params.stretched_grid_files.dim_x
    if params.stretched_grid_files.HasField('dim_y'):
      stretched_grid_files_xyz[1] = params.stretched_grid_files.dim_y
    if params.stretched_grid_files.HasField('dim_z'):
      stretched_grid_files_xyz[2] = params.stretched_grid_files.dim_z
    self.global_xyz = _global_xyz_from_config(
        self.to_data_axis_order(*stretched_grid_files_xyz),
        self.to_data_axis_order(self.fx, self.fy, self.fz),
        self.to_data_axis_order(self.lx, self.ly, self.lz),
        self.to_data_axis_order(self.core_nx, self.core_ny, self.core_nz),
        self.to_data_axis_order(self.cx, self.cy, self.cz),
        self.periodic_dims,
    )

    # Overwrite domain sizes for periodic stretched grid. This step must be done
    # before `lx`, `ly`, and `lz` are used in the rest of the constructor.
    for dim in range(3):
      if self.use_stretched_grid[dim] and self.periodic_dims[dim]:
        axis = self.data_axis_order[dim]
        if self.to_data_axis_order(self.lx, self.ly, self.lz)[dim] != 0:
          raise ValueError(
              f'l{axis} in the grid_parametrization proto must be 0 to use'
              f' periodic stretched grid in axis {axis}.'
          )
        value = domain_size_from_periodic_stretched_grid(
            (
                params.stretched_grid_files.dim_x,
                params.stretched_grid_files.dim_y,
                params.stretched_grid_files.dim_z,
            )[dim]
        )
        if axis == 'x':
          self.lx = value
        elif axis == 'y':
          self.ly = value
        elif axis == 'z':
          self.lz = value
        else:
          raise ValueError(f'`axis`: {axis} should be one of x, y, z.')
        logging.info(
            'Overwriting domain size of periodic stretched grid in axis: %s to'
            ' %f',
            axis,
            value,
        )

    global_xyz_with_halos = []
    for dim in range(3):
      coord = self.global_xyz[dim]
      domain_size = self.to_data_axis_order(self.lx, self.ly, self.lz)[dim]
      periodic_and_stretched = (
          self.periodic_dims[dim] and self.use_stretched_grid[dim]
      )
      global_xyz_with_halos.append(
          jnp.array(
              _extend_grid(
                  np.array(coord),
                  self.halo_width,
                  periodic_and_stretched,
                  domain_size,
              ),
              dtype=coord.dtype,
          )
      )
    self.global_xyz_with_halos: tuple[jax.Array, jax.Array, jax.Array] = tuple(
        global_xyz_with_halos
    )

  def get_axis_index(
      self, axis_name: tuple[str, ...] | str
  ) -> tuple[int, ...] | int:
    """Gets the index of the axis in the axis order.

    Args:
      axis_name: The name of the axis. If it's a tuple of str, the output will
        be a tuple of int. Otherwise, it will be an int. For ex., if
        data_axis_order = ('x', 'y', 'z') and the axis_name is ('z', 'x'), the
        output will be (2, 0).

    Returns:
      The index of the axis in the axis order.
    """
    if isinstance(axis_name, str):
      if axis_name not in self.data_axis_order:
        raise ValueError(
            f'`axis_name` {axis_name} is not in the `data_axis_order`'
            f' {self.data_axis_order}.'
        )
      return self.data_axis_order.index(axis_name)
    else:
      if any(a not in self.data_axis_order for a in axis_name):
        raise ValueError(
            f'`axis_name` {axis_name} contains names not in `data_axis_order`'
            f' f`{self.data_axis_order}`'
        )
      return tuple(self.data_axis_order.index(a) for a in axis_name)

  def to_data_axis_order(
      self, x_entry: Any, y_entry: Any, z_entry: Any
  ) -> tuple[Any, Any, Any]:
    """Permutes the entries to match with axis order.

    This function is useful to permute the entries of a tuple to match the
    data axis order. For example, if the axis order is ('z', 'x', 'y'), then
    the output will be (z_entry, x_entry, y_entry). This function is an inverse
    of `to_xyz_order`.

    Args:
      x_entry: The entry for x axis.
      y_entry: The entry for y axis.
      z_entry: The entry for z axis.

    Returns:
      A tuple of 3 entries in the axis order.
    """
    entries = [None] * 3
    entries[self.get_axis_index('x')] = x_entry
    entries[self.get_axis_index('y')] = y_entry
    entries[self.get_axis_index('z')] = z_entry
    return tuple(entries)

  def to_xyz_order(self, entries: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
    """Returns the entries in the axis order.

    This function returns the entries corresponding to x, y, z order. It is an
    inverse of `to_data_axis_order`.

    Args:
      entries: Order of this tuple should match with the axis order. This
        function will not check the order.

    Returns:
      A tuple of 3 entries in x, y, z order.
    """
    return tuple(entries[i] for i in self.get_axis_index(('x', 'y', 'z')))

  def get_axis_entry(
      self,
      x_entry: Any,
      y_entry: Any,
      z_entry: Any,
      axis: str,
  ) -> Any:
    """Returns the entry corresponding to the given axis.

    Args:
      x_entry: The entry for x axis.
      y_entry: The entry for y axis.
      z_entry: The entry for z axis.
      axis: The axis to return the entry for.

    Returns:
      The entry corresponding to the given axis.
    """
    if axis == 'x':
      return x_entry
    elif axis == 'y':
      return y_entry
    elif axis == 'z':
      return z_entry
    else:
      raise ValueError(f'`axis`: {axis} should be one of x, y, z.')

  @classmethod
  def create_from_grid_lengths_and_etc(
      cls,
      grid_lengths: dict[str, float],
      computation_shape: dict[str, int],
      subgrid_shape: dict[str, int] | None,
      halo_width: int,
      data_axis_order: str,
      kernel_size: int = 128,
  ):
    """Creates grid parametrization from specific arguments (grid lengths, etc).

    Args:
      grid_lengths: The full grid lengths in the three dimensions. The keys
        should be 'x', 'y', and 'z'.
      computation_shape: The number of TPU cores assigned to each of three axes.
        The keys should be 'x', 'y', and 'z'.
      subgrid_shape: The subgrid shape in the three dimensions. The keys should
        be 'x', 'y', and 'z'.
      halo_width: The halo width.
      data_axis_order: A string of length 3, representing the order of the axes.
        It should be one of ['xyz', 'xzy', 'yzx', 'yxz', 'zxy', 'zyx'].
      kernel_size: The kernel size.

    Returns:
      The `GridParametrization` encapsulating the input arguments.
    """
    if tuple(sorted(grid_lengths.keys())) != ('x', 'y', 'z'):
      raise ValueError(
          'The keys of `grid_lengths` should be `x`, `y`, and `z`. But'
          f' grid_lengths: {grid_lengths}'
      )
    if tuple(sorted(computation_shape.keys())) != ('x', 'y', 'z'):
      raise ValueError(
          'The keys of `computation_shape` should be `x`, `y`, and `z`. But'
          f' computation_shape: {computation_shape}'
      )
    if subgrid_shape:
      if tuple(sorted(subgrid_shape.keys())) != ('x', 'y', 'z'):
        raise ValueError(
            'The keys of `subgrid_shape` should be `x`, `y`, and `z`. But'
            f' subgrid_shape: {subgrid_shape}'
        )

    proto = grid_parametrization_pb2.GridParametrization()
    proto.data_axis_order = data_axis_order
    proto.length.dim_x = grid_lengths['x']
    proto.length.dim_y = grid_lengths['y']
    proto.length.dim_z = grid_lengths['z']

    proto.computation_shape.dim_x = computation_shape['x']
    proto.computation_shape.dim_y = computation_shape['y']
    proto.computation_shape.dim_z = computation_shape['z']

    if subgrid_shape:
      proto.grid_size.dim_x = subgrid_shape['x']
      proto.grid_size.dim_y = subgrid_shape['y']
      proto.grid_size.dim_z = subgrid_shape['z']

    proto.halo_width = halo_width
    proto.kernel_size = kernel_size

    return cls(proto)

  @classmethod
  def create_from_grid_lengths_and_etc_with_defaults(
      cls,
      grid_lengths: dict[str, float] | None = None,
      computation_shape: dict[str, int] | None = None,
      subgrid_shape: dict[str, int] | None = None,
      halo_width: int = 1,
      data_axis_order: str = 'xyz',
  ):
    """Same as `create_from_grid_lengths_and_etc`, but, with default arguments.

    If the default arguments for `computation_shape` and `subgrid_shape` are
    used, the grid spacings are equal to the provided grid lengths. This can be
    useful in case a parametrization is needed which is only used to encapsulate
    grid spacings.

    Args:
      grid_lengths: The full grid lengths in the three dimensions.
      computation_shape: The number of cores assigned to each of three axes.
      subgrid_shape: The subgrid shape in the three dimensions.
      halo_width: The halo width.
      data_axis_order: A string of length 3, representing the order of the axes.
        It should be a permutation of 'xyz'.

    Returns:
      The `GridParametrization` encapsulating the input arguments.
    """
    # TODO(shantanussh): Remove the default arguments. These are only useful for
    # unit tests. Move these to test utility.
    if grid_lengths is None:
      grid_lengths = {'x': 1.0, 'y': 1.0, 'z': 1.0}
    if computation_shape is None:
      computation_shape = {'x': 1, 'y': 1, 'z': 1}
    if subgrid_shape is None:
      subgrid_shape = {'x': 4, 'y': 4, 'z': 4}
    return cls.create_from_grid_lengths_and_etc(
        grid_lengths,
        computation_shape,
        subgrid_shape,
        halo_width,
        data_axis_order,
    )

  def __str__(self):
    dx_dy_dz = self.to_xyz_order(self.grid_spacings)
    return (
        f'fx_physical: {self.fx_physical}, fy_physical: {self.fy_physical},'
        f' fz_physical: {self.fz_physical}, fx: {self.fx}, fy: {self.fy}, fz:'
        f' {self.fz}, cx: {self.cx}, cy: {self.cy}, cz: {self.cz}, nx:'
        f' {self.nx}, ny: {self.ny}, nz: {self.nz}, core_nx: {self.core_nx},'
        f' core_ny: {self.core_ny}, core_nz: {self.core_nz}, lx: {self.lx}, ly:'
        f' {self.ly}, lz: {self.lz}, dt: {self.dt}, dx: {dx_dy_dz[0]}, dy:'
        f' {dx_dy_dz[1]}, dz: {dx_dy_dz[2]}, computation_shape:'
        f' {self.computation_shape}, halo_width: {self.halo_width},'
        f' kernel_size: {self.kernel_size}'
    )

  @property
  def computation_shape(self) -> jnp.ndarray:
    return jnp.array([self.cx, self.cy, self.cz])

  @property
  def core_nx(self) -> int:
    _validate_grid_size_wrt_halo_width(self.nx, self.halo_width, 'x')
    return _get_core_n(self.nx, self.halo_width)

  @property
  def core_ny(self) -> int:
    _validate_grid_size_wrt_halo_width(self.ny, self.halo_width, 'y')
    return _get_core_n(self.ny, self.halo_width)

  @property
  def core_nz(self) -> int:
    _validate_grid_size_wrt_halo_width(self.nz, self.halo_width, 'z')
    return _get_core_n(self.nz, self.halo_width)

  @property
  def dx(self) -> float | None:
    if self.to_xyz_order(self.use_stretched_grid)[self.get_axis_index('x')]:
      raise ValueError(
          'Calling .dx when using stretched grid in dim x is likely an error!'
      )
    # Note: The final grid should return an outer halo of width 1 due to
    # boundary conditions, but does not. So for now we ignore that in the dx,
    # dy, dz computations.
    return _get_grid_spacing(self.fx, self.lx)

  @property
  def dy(self) -> float | None:
    if self.to_xyz_order(self.use_stretched_grid)[self.get_axis_index('y')]:
      raise ValueError(
          'Calling .dy when using stretched grid in dim y is likely an error!'
      )
    return _get_grid_spacing(self.fy, self.ly)

  @property
  def dz(self) -> float | None:
    if self.to_xyz_order(self.use_stretched_grid)[self.get_axis_index('z')]:
      raise ValueError(
          'Calling .dz when using stretched grid in dim z is likely an error!'
      )
    return _get_grid_spacing(self.fz, self.lz)

  @property
  def fx(self):
    """The full grid size in dim x."""
    _validate_grid_size_wrt_halo_width(self.nx, self.halo_width, 'x')
    return _get_full_grid_size(self.nx, self.halo_width, self.cx)

  @property
  def fy(self):
    """The full grid size in dim y."""
    _validate_grid_size_wrt_halo_width(self.ny, self.halo_width, 'y')
    return _get_full_grid_size(self.ny, self.halo_width, self.cy)

  @property
  def fz(self):
    """The full grid size in dim z."""
    _validate_grid_size_wrt_halo_width(self.nz, self.halo_width, 'z')
    return _get_full_grid_size(self.nz, self.halo_width, self.cz)

  @property
  def x(self) -> jax.Array:
    """The full grid in dim x."""
    return self.to_xyz_order(self.global_xyz)[0]

  @property
  def y(self) -> jax.Array:
    """The full grid in dim y."""
    return self.to_xyz_order(self.global_xyz)[1]

  @property
  def z(self) -> jax.Array:
    """The full grid in dim z."""
    return self.to_xyz_order(self.global_xyz)[2]

  def grid_local_with_coord(
      self,
      axis: str,
      mesh: jax.sharding.Mesh,
      include_halo: bool = True,
  ) -> jax.Array:
    """The local grid in `dim`.

    Args:
      axis: The axis of the grid.
      mesh: A jax Mesh object representing the device topology.
      include_halo: An option of whether to include coordinates of halos in the
        returned grid.

    Returns:
      The grid in axis `axis` local to the current replica.

    Raises:
      AssertionError: If the full grid includes additional boundary points, in
        which case the full grid can not be evenly distributed across all cores
        without halo.
    """
    axis_index = self.get_axis_index(axis)
    n_local = self.get_axis_entry(
        self.core_nx, self.core_ny, self.core_nz, axis
    )
    i_core = jax.lax.axis_index(mesh.axis_names[axis_index])
    if include_halo:
      grid_full = self.global_xyz_with_halos[axis_index]
    else:
      grid_full = self.get_axis_entry(self.x, self.y, self.z, axis)
    start = i_core * n_local
    size = n_local + 2 * include_halo * self.halo_width
    return jax.lax.dynamic_slice_in_dim(grid_full, start, size)

  def x_local(self, mesh: jax.sharding.Mesh) -> jax.Array:
    """The local grid in dim x without halo."""
    return self.grid_local_with_coord('x', mesh, False)

  def y_local(self, mesh: jax.sharding.Mesh) -> jax.Array:
    """The local grid in dim y without halo."""
    return self.grid_local_with_coord('y', mesh, False)

  def z_local(self, mesh: jax.sharding.Mesh) -> jax.Array:
    """The local grid in dim z without halo."""
    return self.grid_local_with_coord('z', mesh, False)

  def x_local_ext(self, mesh: jax.sharding.Mesh) -> jax.Array:
    """The local grid in dim x with halo."""
    return self.grid_local_with_coord('x', mesh, True)

  def y_local_ext(self, mesh: jax.sharding.Mesh) -> jax.Array:
    """The local grid in dim y with halo."""
    return self.grid_local_with_coord('y', mesh, True)

  def z_local_ext(self, mesh: jax.sharding.Mesh) -> jax.Array:
    """The local grid in dim z with halo."""
    return self.grid_local_with_coord('z', mesh, True)

  @property
  def meshgrid(self):
    xs = jnp.linspace(0, self.lx, self.cx * self.core_nx)
    ys = jnp.linspace(0, self.ly, self.cy * self.core_ny)
    zs = jnp.linspace(0, self.lz, self.cz * self.core_nz)
    return jnp.meshgrid(xs, ys, zs, indexing='ij')

  def physical_grid_spacing(
      self,
      axis: str,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Gets the physical grid spacing as a 1D array, allowing nonuniform grid.

    Returns the physical grid spacing (dx, dy, or dz) corresponding to `axis` as
    a 1D tensor.

    This function is to be used in a distributed setting. It returns the grid
    spacing, evaluated on nodes, local to this replica.

    Args:
      axis: The axis under consideration.
      additional_states: Dictionary with helper variables.

    Returns:
      The physical grid spacing in the given axis as a 1D array in
      broadcastable form.
    """
    axis_index = self.get_axis_index(axis)
    if self.use_stretched_grid[axis_index]:
      return additional_states[stretched_grid_util.h_key(axis_index)]
    else:
      n = self.get_axis_entry(self.nx, self.ny, self.nz, axis)
      h = self.to_xyz_order(self.grid_spacings)[axis_index] * jnp.ones(n)
      if axis_index == 0:
        return h[:, jnp.newaxis, jnp.newaxis]
      elif axis_index == 1:
        return h[jnp.newaxis, :, jnp.newaxis]
      else:
        return h[jnp.newaxis, jnp.newaxis, :]
