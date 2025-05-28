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

from typing import List, Literal, Optional, Sequence, Tuple, TypeAlias

from absl import logging
import numpy as np
from swirl_lm.utility import common_ops
from swirl_lm.utility import file_io
from swirl_lm.utility import file_pb2
from swirl_lm.utility import grid_parametrization_pb2
from swirl_lm.utility import stretched_grid_util
from swirl_lm.utility import text_util
from swirl_lm.utility import types
import tensorflow as tf

from google.protobuf import text_format

FlowFieldVal: TypeAlias = types.FlowFieldVal
FlowFieldMap: TypeAlias = types.FlowFieldMap


def _get_core_n(n: int, halo_width: int) -> Optional[int]:
  core_n = n - 2 * halo_width
  return core_n if core_n > 0 else None


def _get_full_grid_size(
    n: int,
    halo_width: int,
    num_cores: int
) -> int:
  """The full grid size (includes padding, if any)."""
  core_n = _get_core_n(n, halo_width)
  if not core_n:
    return 1
  return num_cores * core_n


def _get_full_uniform_grid(n: Optional[int], l: float) -> tf.Tensor:
  """The full grid without halos for a uniformly spaced mesh.

  Args:
    n: The total number of grid points without halos.
    l: The total length of the domain.

  Returns:
    A equidistant grid for the entire computational domain. The first grid point
    is 0.
  """
  n_effective = n if n is not None else 1
  return tf.linspace(0.0, l, n_effective)


def get_physical_full_grid_size(
    params: grid_parametrization_pb2.GridParametrization
) -> grid_parametrization_pb2.CoordinateInt:
  # Some simulations have a physical grid size mandated externally, and add
  # padding in order that the internal grid has dimensions appropriate for
  # running on TPU (e.g. the grid sizes of dim 0 and 1 should be multiples of
  # 128). The full physical size is set here assuming there is no padding. If
  # there is padding, these values will be overridden.
  return grid_parametrization_pb2.CoordinateInt(
      dim_0=_get_full_grid_size(params.grid_size.dim_0, params.halo_width,
                                params.computation_shape.dim_0),
      dim_1=_get_full_grid_size(params.grid_size.dim_1, params.halo_width,
                                params.computation_shape.dim_1),
      dim_2=_get_full_grid_size(params.grid_size.dim_2, params.halo_width,
                                params.computation_shape.dim_2))


def _get_grid_spacing(
    full_grid_size, length
) -> float | None:
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

  Return the grid spacing of the computational mesh, assumed uniform.  When
  using a stretched grid, the grid spacing will be that of the transformed
  coordinate, not of the physical Cartesian coordinate values.

  Note: internally, when using a stretched grid where the coordinate levels
  are passed in, we will treat the grid spacing of the transformed coordinate
  as equal to 1.0, because the coordinate values of the transformed coordinate
  never need be referred to directly.

  Args:
    dx_dy_dz: Tuple of values from GridParametrization.(dx,dy,dz)
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
    stretched_grid_file: file_pb2.File) -> float:
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
    global_coord: tf.Tensor,
    core_n: int | None,
    num_core: int,
    dim: Literal[0, 1, 2],
) -> None:
  """Checks that global coordinates have the correct number of elements.

  Global coordinates do not include the halos. The input args `core_n`,
  and `num_core` should come from a `GridParametrization`  object.  `core_n`
  should be the 3-tuple `(params.core_nx, params.core_ny, params.core_nz)`, and
  `num_core` should be the 3-tuple `(params.nx, params.ny, params.nz)`.

  Args:
    global_coord: The global coordinates.
    core_n: Number of non-halo points in this dimension.
    num_core: Number of cores in this dimension.
    dim: Which dimension is being considered (only used for an error message).

  Raises:
    ValueError: If `global_coord` does not have the correct number of
      elements.
  """
  if core_n is None:
    # There are no non-halo points in this dimension, so there is nothing to
    # validate.
    return

  global_grid_size = num_core * core_n  # Excludes halos.

  if global_coord.shape[0] != global_grid_size:
    raise ValueError(
        f'Global coordinates in dim {dim} has'
        f' {global_coord.shape[0]} elements, which does not match the'
        f' required number of elements {global_grid_size}.'
    )


def _load_array_from_file(array_file: file_pb2.File) -> tf.Tensor:
  """Loads a 1D array from a text file and returns it as a tensor.

  Each element of the array should be on its own line.

  Args:
    array_file: The file to load.

  Returns:
    A 1D tensor of the data from the file.
  """
  contents = text_util.strip_line_comments(file_io.load_file(array_file), '#')
  return tf.convert_to_tensor(
      np.fromstring(contents, sep='\n'), dtype=tf.float32
  )


def _global_xyz_from_config(
    stretched_grid_files: grid_parametrization_pb2.CoordinateFile,
    full_grid_size: tuple[int, int, int],
    length: tuple[float, float, float],
    core_n: tuple[int | None, int | None, int | None],
    num_core: tuple[int, int, int],
    periodic: tuple[bool, bool, bool],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Returns global coordinates, excluding halo points.

  For each dimension: if a stretched grid is provided then that grid is used,
  otherwise a uniform grid is used.

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

  def get_full_grid(dim: Literal[0, 1, 2]) -> tf.Tensor:
    stretched_grid_path = (
        stretched_grid_files.dim_0,
        stretched_grid_files.dim_1,
        stretched_grid_files.dim_2,
    )[dim]
    if stretched_grid_files.HasField(f'dim_{dim}'):
      global_coord = _load_array_from_file(stretched_grid_path)
      logging.info(
          'Loaded stretched grid in dim %d from file `%s`.',
          dim,
          stretched_grid_path,
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
    else:
      global_coord = _get_full_uniform_grid(full_grid_size[dim], length[dim])
    _validate_global_coord(
        global_coord, core_n[dim], num_core[dim], dim
    )
    return global_coord

  return tuple(get_full_grid(dim) for dim in (0, 1, 2))


# TODO(b/368405442): Adjust definition of a periodic uniform mesh to be
# consistent with the length defined in the config, and be consistent with the
# way the stretched mesh is defined.
def _extend_grid(
    grid: tf.Tensor,
    halo_width: int,
    periodic_and_stretched: bool,
    domain_size: float,
) -> tf.Tensor:
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
  def single_point():
    """Generates a mesh of a same value when `grid` is a single point."""
    return grid[0] * tf.ones((halo_width * 2 + 1), dtype=grid.dtype)

  def regular_mesh():
    """Generates a mesh with linearly extrapolated halos."""
    d0 = grid[1] - grid[0]
    d1 = grid[-1] - grid[-2]
    ext = np.linspace(1, halo_width, halo_width)
    ext_0 = grid[0] - d0 * ext[::-1]
    ext_1 = grid[-1] + d1 * ext
    return tf.concat([ext_0, grid, ext_1], axis=0)

  def stretched_periodic_mesh() -> tf.Tensor:
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
    return tf.concat([ext_left, grid, ext_right], axis=0)

  # Note that we have to use the static shape here. Condition with results from
  # tf.shape() will provide a tf.Tensor, which can only be used in the context
  # of tf.cond. Because tf.cond will compile all branches regardless of the
  # condition, and `grid` is valid in only one of the branches at a time, we
  # have to take advantage of the static shape of the grid.
  if grid.shape[0] < 2:
    return single_point()
  elif periodic_and_stretched:
    return stretched_periodic_mesh()
  else:
    return regular_mesh()


def params_from_text_proto(
    text_proto: str) -> grid_parametrization_pb2.GridParametrization:
  """Returns a GridParametrization protobuf from a text-formatted proto."""
  params = text_format.Parse(text_proto,
                             grid_parametrization_pb2.GridParametrization())
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
  """

  def __init__(
      self,
      params: grid_parametrization_pb2.GridParametrization,
  ):
    """Creates an object from protobuf."""
    if not params.HasField('physical_full_grid_size'):
      params.physical_full_grid_size.CopyFrom(
          get_physical_full_grid_size(params))
    if params.halo_width <= 0:
      raise ValueError('Halo width must be greater than 0.')
    self.grid_params_proto = params
    self.cx = params.computation_shape.dim_0
    self.cy = params.computation_shape.dim_1
    self.cz = params.computation_shape.dim_2
    self.lx = params.length.dim_0
    self.ly = params.length.dim_1
    self.lz = params.length.dim_2
    self.nx = params.grid_size.dim_0
    self.ny = params.grid_size.dim_1
    self.nz = params.grid_size.dim_2
    self.fx_physical = params.physical_full_grid_size.dim_0
    self.fy_physical = params.physical_full_grid_size.dim_1
    self.fz_physical = params.physical_full_grid_size.dim_2
    self.halo_width = params.halo_width
    self.dt = params.dt
    # "Recover" float64 precision for dt by rounding the 32 bit value to 6
    # significant digits and then converting back to float. The assumption is
    # that dt is user specified with 6 or less significant digits.
    self.dt64 = float(
        np.format_float_positional(params.dt, 6, fractional=False))
    self.kernel_size = params.kernel_size
    self.input_chunk_size = params.input_chunk_size
    self.num_output_splits = params.num_output_splits

    self.use_stretched_grid = (
        params.stretched_grid_files.HasField('dim_0'),
        params.stretched_grid_files.HasField('dim_1'),
        params.stretched_grid_files.HasField('dim_2'),
    )

    # Whether the dimensions are periodic or not.
    self.periodic_dims = (
        params.periodic.dim_0, params.periodic.dim_1, params.periodic.dim_2
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
            self.use_stretched_grid[dim],
        )
        for dim in (0, 1, 2)
    )

    # Tuple of global coordinate values for each dimension, including halos.
    # Consists of the stretched-grid coordinates (if used), otherwise defaults
    # to uniform grid.
    self.global_xyz = _global_xyz_from_config(
        params.stretched_grid_files,
        (self.fx, self.fy, self.fz),
        (self.lx, self.ly, self.lz),
        (self.core_nx, self.core_ny, self.core_nz),
        (self.cx, self.cy, self.cz),
        self.periodic_dims,
    )
    # Overwrite domain sizes for periodic stretched grid. This step must be done
    # before `lx`, `ly`, and `lz` are used in the rest of the constructor.
    if self.use_stretched_grid[0] and self.periodic_dims[0]:
      if self.lx != 0:
        raise ValueError(
            'lx in the grid_parametrization proto must be 0 to use periodic'
            ' stretched grid in dim 0.'
        )
      self.lx = domain_size_from_periodic_stretched_grid(
          params.stretched_grid_files.dim_0
      )
      logging.info(
          'Overwriting domain size of periodic stretched grid in dim 0 to %f.',
          self.lx,
      )
    if self.use_stretched_grid[1] and self.periodic_dims[1]:
      if self.ly != 0:
        raise ValueError(
            'ly in the grid_parametrization proto must be 0 to use periodic'
            ' stretched grid in dim 1.'
        )
      self.ly = domain_size_from_periodic_stretched_grid(
          params.stretched_grid_files.dim_1
      )
      logging.info(
          'Overwriting domain size of periodic stretched grid in dim 1 to %f.',
          self.ly,
      )
    if self.use_stretched_grid[2] and self.periodic_dims[2]:
      if self.lz != 0:
        raise ValueError(
            'lz in the grid_parametrization proto must be 0 to use periodic'
            ' stretched grid in dim 2.'
        )
      self.lz = domain_size_from_periodic_stretched_grid(
          params.stretched_grid_files.dim_2
      )
      logging.info(
          'Overwriting domain size of periodic stretched grid in dim 2 to %f.',
          self.lz,
      )

    global_xyz_with_halos = []
    for dim in (0, 1, 2):
      coord = self.global_xyz[dim]
      domain_size = (self.lx, self.ly, self.lz)[dim]
      periodic_and_stretched = (
          self.periodic_dims[dim] and self.use_stretched_grid[dim]
      )
      global_xyz_with_halos.append(
          _extend_grid(
              coord, self.halo_width, periodic_and_stretched, domain_size
          )
      )
    self.global_xyz_with_halos: tuple[tf.Tensor, tf.Tensor, tf.Tensor] = tuple(
        global_xyz_with_halos
    )

  @classmethod
  def create_from_grid_lengths_and_etc(
      cls,
      grid_lengths: Sequence[float],
      computation_shape: Sequence[int],
      subgrid_shape: Optional[Sequence[int]],
      halo_width: int,
      kernel_size: int = 128,
  ):
    """Creates grid parametrization from specific arguments (grid lengths, etc).

    Args:
      grid_lengths: The full grid lengths in the three dimensions.
      computation_shape: The number of TPU cores assigned to each of three axes.
      subgrid_shape: The subgrid shape in the three dimensions.
      halo_width: The halo width.
      kernel_size: The kernel size.

    Returns:
      The `GridParametrization` encapsulating the input arguments.
    """
    proto = grid_parametrization_pb2.GridParametrization()
    proto.length.dim_0 = grid_lengths[0]
    proto.length.dim_1 = grid_lengths[1]
    proto.length.dim_2 = grid_lengths[2]

    proto.computation_shape.dim_0 = computation_shape[0]
    proto.computation_shape.dim_1 = computation_shape[1]
    proto.computation_shape.dim_2 = computation_shape[2]

    if subgrid_shape:
      proto.grid_size.dim_0 = subgrid_shape[0]
      proto.grid_size.dim_1 = subgrid_shape[1]
      proto.grid_size.dim_2 = subgrid_shape[2]

    proto.halo_width = halo_width
    proto.kernel_size = kernel_size

    return cls(proto)

  @classmethod
  def create_from_grid_lengths_and_etc_with_defaults(
      cls,
      grid_lengths: Sequence[float],
      computation_shape: Sequence[int] = (1, 1, 1),
      subgrid_shape: Optional[Sequence[int]] = (4, 4, 4),
      halo_width: int = 1):
    """Same as `create_from_grid_lengths_and_etc`, but, with default arguments.

    If the default arguments for `computation_shape` and `subgrid_shape` are
    used, the grid spacings are equal to the provided grid lengths. This can be
    useful in case a parametrization is needed which is only used to encapsulate
    grid spacings.

    Args:
      grid_lengths: The full grid lengths in the three dimensions.
      computation_shape: The number of TPU cores assigned to each of three axes.
      subgrid_shape: The subgrid shape in the three dimensions.
      halo_width: The halo width.

    Returns:
      The `GridParametrization` encapsulating the input arguments.
    """
    return cls.create_from_grid_lengths_and_etc(grid_lengths, computation_shape,
                                                subgrid_shape, halo_width)

  def __str__(self):
    # fmt: off
    return ('fx_physical: {}, fy_physical: {}, fz_physical: {}, fx: {}, '
            'fy: {}, fz: {}, cx: {}, cy: {}, cz: {}, nx: {}, ny: {}, nz: {}, '
            'core_nx: {}, core_ny: {}, core_nz: {}, lx: {}, ly: {}, lz: {}, '
            'dt: {}, dx: {}, dy: {}, dz: {}, computation_shape: {}, '
            'halo_width: {}, kernel_size: {}, input_chunk_size: {}, '
            'num_output_splits: {}'.format(
                self.fx_physical, self.fy_physical, self.fz_physical, self.fx,
                self.fy, self.fz, self.cx, self.cy, self.cz, self.nx, self.ny,
                self.nz, self.core_nx, self.core_ny, self.core_nz, self.lx,
                self.ly, self.lz, self.dt, self.grid_spacings[0],
                self.grid_spacings[1], self.grid_spacings[2],
                self.computation_shape, self.halo_width, self.kernel_size,
                self.input_chunk_size, self.num_output_splits))
    # fmt: on

  @property
  def computation_shape(self) -> np.ndarray:
    return np.array([self.cx, self.cy, self.cz])

  @property
  def core_nx(self) -> Optional[int]:
    return _get_core_n(self.nx, self.halo_width)

  @property
  def core_ny(self) -> Optional[int]:
    return _get_core_n(self.ny, self.halo_width)

  @property
  def core_nz(self) -> Optional[int]:
    return _get_core_n(self.nz, self.halo_width)

  @property
  def dx(self) -> Optional[float]:
    if self.use_stretched_grid[0]:
      raise ValueError(
          'Calling .dx when using stretched grid in dim 0 is likely an error!'
      )
    # Note: The final grid should return an outer halo of width 1 due to
    # boundary conditions, but does not. So for now we ignore that in the dx,
    # dy, dz computations.
    return _get_grid_spacing(self.fx, self.lx)

  @property
  def dy(self) -> Optional[float]:
    if self.use_stretched_grid[1]:
      raise ValueError(
          'Calling .dy when using stretched grid in dim 1 is likely an error!'
      )
    return _get_grid_spacing(self.fy, self.ly)

  @property
  def dz(self) -> Optional[float]:
    if self.use_stretched_grid[2]:
      raise ValueError(
          'Calling .dz when using stretched grid in dim 2 is likely an error!'
      )
    return _get_grid_spacing(self.fz, self.lz)

  @property
  def fx(self):
    """The full grid size in dim 0."""
    return _get_full_grid_size(self.nx, self.halo_width, self.cx)

  @property
  def fy(self):
    """The full grid size in dim 1."""
    return _get_full_grid_size(self.ny, self.halo_width, self.cy)

  @property
  def fz(self):
    """The full grid size in dim 2."""
    return _get_full_grid_size(self.nz, self.halo_width, self.cz)

  @property
  def x(self) -> tf.Tensor:
    """The full grid in dim 0."""
    return self.global_xyz[0]

  @property
  def y(self) -> tf.Tensor:
    """The full grid in dim 1."""
    return self.global_xyz[1]

  @property
  def z(self) -> tf.Tensor:
    """The full grid in dim 2."""
    return self.global_xyz[2]

  def grid_local(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      dim: int,
      include_halo: bool = True,
  ) -> tf.Tensor:
    """The local grid in `dim`.

    Args:
      replica_id: The index of the current replica.
      replicas: A 3D tensor that saves the topology of the partitioning.
      dim: The dimension of the grid.
      include_halo: An option of whether to include coordinates of halos in the
        returned grid.

    Returns:
      The grid in dim `dim` local to `replica_id`.

    Raises:
      AssertionError: If the full grid includes additional boundary points, in
        which case the full grid can not be evenly distributed across all cores
        without halo.
    """
    coord = common_ops.get_core_coordinate(replicas, replica_id)[dim]
    core_n = (self.core_nx, self.core_ny, self.core_nz)[dim]
    n = (self.nx, self.ny, self.nz)[dim]

    if include_halo:
      return common_ops.get_local_slice_of_1d_array(
          self.global_xyz_with_halos[dim], coord, core_n, n
      )
    else:
      return common_ops.get_local_slice_of_1d_array(
          self.global_xyz[dim], coord, core_n, core_n
      )

  def x_local(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
  ) -> tf.Tensor:
    """The local grid in dim 0 without halo."""
    return self.grid_local(replica_id, replicas, 0, False)

  def y_local(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
  ) -> tf.Tensor:
    """The local grid in dim 1 without halo."""
    return self.grid_local(replica_id, replicas, 1, False)

  def z_local(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
  ) -> tf.Tensor:
    """The local grid in dim 2 without halo."""
    return self.grid_local(replica_id, replicas, 2, False)

  def x_local_ext(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
  ) -> tf.Tensor:
    """The local grid in dim 0 with halo."""
    return self.grid_local(replica_id, replicas, 0, True)

  def y_local_ext(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
  ) -> tf.Tensor:
    """The local grid in dim 1 with halo."""
    return self.grid_local(replica_id, replicas, 1, True)

  def z_local_ext(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
  ) -> tf.Tensor:
    """The local grid in dim 2 with halo."""
    return self.grid_local(replica_id, replicas, 2, True)

  @property
  def input_chunks(self) -> List[Tuple[int, int]]:
    """Input split for 2GB limitation workaround.


    Returns:
      A list of integer tuples corresponding to ranges of the input in z.
    """
    input_chunks = [(x, x + self.input_chunk_size if
                     (x + self.input_chunk_size) < self.nz else self.nz)
                    for x in range(0, self.nz, self.input_chunk_size)]
    return input_chunks

  @property
  def chunk_limits(self) -> List[Tuple[int, int]]:
    """Output split for 2GB limitation workaround.


    Returns:
      A list of integer tuples corresponding to ranges of the output in z.
    """
    chunk_size, nz_r = divmod(self.nz, self.num_output_splits)
    chunks = [
        chunk_size + (1 if i < nz_r else 0)
        for i in range(self.num_output_splits)
    ]
    chunk_limits = []
    llimit = 0
    for chunk in chunks:
      chunk_limits.append((llimit, llimit + chunk))
      llimit += chunk
    return chunk_limits

  @property
  def num_replicas(self):
    return self.cx * self.cy * self.cz

  def physical_grid_spacing(
      self,
      dim: Literal[0, 1, 2],
      use_3d_tf_tensor: bool,
      additional_states: FlowFieldMap,
  ) -> FlowFieldVal:
    """Gets the physical grid spacing as a 1D array, allowing nonuniform grid.

    Returns the physical grid spacing (dx, dy, or dz) corresponding to dimension
    `dim` as a 1D tensor.

    This function is to be used in a distributed setting. It returns the grid
    spacing, evaluated on nodes, local to this replica.

    Args:
      dim: The dimension under consideration.
      use_3d_tf_tensor: Whether 3D fields are represented as 3D tensors.
      additional_states: Dictionary with helper variables.

    Returns:
      The physical grid spacing in the given dimension as a 1D array in
      broadcastable form.
    """
    if self.use_stretched_grid[dim]:
      return additional_states[stretched_grid_util.h_key(dim)]
    else:
      n = (self.nx, self.ny, self.nz)[dim]
      h = self.grid_spacings[dim] * tf.ones(n)
      return common_ops.reshape_to_broadcastable(h, dim, use_3d_tf_tensor)
