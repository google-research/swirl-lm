# Copyright 2022 The swirl_lm Authors.
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

from typing import List, Optional, Sequence, Tuple

from absl import flags
import numpy as np
from swirl_lm.utility import grid_parametrization_pb2
import tensorflow as tf

from google.protobuf import text_format

# Set allow_override=True for these flags, so each simulation can have its own
# default values.
flags.DEFINE_integer('cx', 2, 'computation shape x.', allow_override=True)
flags.DEFINE_integer('cy', 2, 'computation shape y.', allow_override=True)
flags.DEFINE_integer('cz', 2, 'computation shape z.', allow_override=True)
flags.DEFINE_float('lx', 8., 'x length.', allow_override=True)
flags.DEFINE_float('ly', 8., 'y length.', allow_override=True)
flags.DEFINE_float('lz', 8., 'z length.', allow_override=True)
flags.DEFINE_integer(
    'nx', 256, 'x grid size for one core.', allow_override=True)
flags.DEFINE_integer(
    'ny', 256, 'y grid size for one core.', allow_override=True)
flags.DEFINE_integer(
    'nz', 256, 'z grid size for one core.', allow_override=True)
flags.DEFINE_integer('halo_width', 2, 'The halo width.', allow_override=True)
flags.DEFINE_float('dt', 0.00003, 'dt.', allow_override=True)
flags.DEFINE_integer(
    'kernel_size', 128, 'The size of the 2D kernel.', allow_override=True)
flags.DEFINE_integer(
    'input_chunk_size', 128, 'chunk size for input.', allow_override=True)
flags.DEFINE_integer(
    'num_output_splits',
    1, 'number of splits for processing '
    'output.',
    allow_override=True)
flags.DEFINE_integer(
    'num_boundary_points',
    1,
    'Number of points to be added to each end of the computational domain.',
    allow_override=True)

FLAGS = flags.FLAGS


def _get_core_n(n: int, halo_width: int) -> Optional[int]:
  core_n = n - 2 * halo_width
  return core_n if core_n > 0 else None


def _get_full_grid_size(
    n: int,
    halo_width: int,
    num_cores: int,
    num_boundary_points: int = 1,
) -> int:
  """The full grid size (includes padding, if any)."""
  core_n = _get_core_n(n, halo_width)
  if not core_n:
    return 1
  return num_cores * core_n + num_boundary_points * 2


def _get_full_grid(n: Optional[int], l: float) -> tf.Tensor:
  """The full grid without halos.

  Args:
    n: The total number of grid points without halos.
    l: The total length of the domain.

  Returns:
    A equidistant grid for the entire computational domain. The first grid point
    is 0.
  """
  n_effective = n if n is not None else 1
  return tf.linspace(0.0, l, n_effective)


def _get_pysical_full_grid_size(
    params: grid_parametrization_pb2.GridParametrization
) -> grid_parametrization_pb2.CoordinateInt:
  # Some simulations have a physical grid size mandated externally, and add
  # padding in order that the internal grid has dimensions appropriate for
  # running on TPU (e.g. the grid sizes of dim 0 and 1 should be multiples of
  # 128). The full physical size is set here assuming there is no padding. If
  # there is padding, these values will be overridden.
  return grid_parametrization_pb2.CoordinateInt(
      dim_0=_get_full_grid_size(params.grid_size.dim_0, params.halo_width,
                                params.computation_shape.dim_0,
                                params.num_boundary_points),
      dim_1=_get_full_grid_size(params.grid_size.dim_1, params.halo_width,
                                params.computation_shape.dim_1,
                                params.num_boundary_points),
      dim_2=_get_full_grid_size(params.grid_size.dim_2, params.halo_width,
                                params.computation_shape.dim_2,
                                params.num_boundary_points))


def params_from_flags() -> grid_parametrization_pb2.GridParametrization:
  """Returns a GridParametrization protobuf from flags."""
  params = grid_parametrization_pb2.GridParametrization()
  params.computation_shape.dim_0 = FLAGS.cx
  params.computation_shape.dim_1 = FLAGS.cy
  params.computation_shape.dim_2 = FLAGS.cz
  params.length.dim_0 = FLAGS.lx
  params.length.dim_1 = FLAGS.ly
  params.length.dim_2 = FLAGS.lz
  params.grid_size.dim_0 = FLAGS.nx
  params.grid_size.dim_1 = FLAGS.ny
  params.grid_size.dim_2 = FLAGS.nz
  params.halo_width = FLAGS.halo_width
  params.dt = FLAGS.dt
  params.kernel_size = FLAGS.kernel_size
  params.input_chunk_size = FLAGS.input_chunk_size
  params.num_output_splits = FLAGS.num_output_splits
  params.num_boundary_points = FLAGS.num_boundary_points
  if not params.HasField('physical_full_grid_size'):
    params.physical_full_grid_size.CopyFrom(_get_pysical_full_grid_size(params))
  return params


def params_from_text_proto(
    text_proto: str) -> grid_parametrization_pb2.GridParametrization:
  """Returns a GridParametrization protobuf from a text-formatted proto."""
  # Get default values from flags.
  params = params_from_flags()
  # Clear physical_full_grid_size, which is a computed field. We'll recompute it
  # at the end.
  params.ClearField('physical_full_grid_size')
  params.MergeFrom(
      text_format.Parse(text_proto,
                        grid_parametrization_pb2.GridParametrization()))
  # Re-compute physical_full_grid_size if necessary.
  if not params.HasField('physical_full_grid_size'):
    params.physical_full_grid_size.CopyFrom(_get_pysical_full_grid_size(params))
  return params


class GridParametrization(object):
  """An object to hold configuration parameters (from flags).

  For computing dx, dy, dz below, we assume the 'box' boundaries coincide with
  the outer most grid points on each end -- the 'halo' grid. Assuming one halo
  point, this means there are total core * c + 2 points, or core * c + 1
  spacings.

  """

  def __init__(
      self,
      params: Optional[grid_parametrization_pb2.GridParametrization] = None,
  ):
    """Creates an object from protobuf."""
    if not params:
      params = params_from_flags()
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
    self.kernel_size = params.kernel_size
    self.input_chunk_size = params.input_chunk_size
    self.num_output_splits = params.num_output_splits
    self.num_boundary_points = params.num_boundary_points

  @classmethod
  def create_from_flags(cls):
    """Creates an object from flags."""
    return cls(params_from_flags())

  @classmethod
  def create_from_grid_lengths_and_etc(
      cls,
      grid_lengths: Sequence[float],
      computation_shape: Sequence[int],
      subgrid_shape: Optional[Sequence[int]],
      halo_width: int,
      num_boundary_points: int = 1,
  ):
    """Creates grid parametrization from specific arguments (grid lengths, etc).

    The parametrization is initially created from flags, then the input
    arguments overwrite the corresponding values. This was created for use in
    applications where `GridParametrization` is only used to encapsulate the
    input arguments. In those cases no other properties are used.

    Args:
      grid_lengths: The full grid lengths in the three dimensions.
      computation_shape: The number of TPU cores assigned to each of three axes.
      subgrid_shape: The subgrid shape in the three dimensions.
      halo_width: The halo width.
      num_boundary_points: The number of boundary points.

    Returns:
      The `GridParametrization` encapsulating the input arguments.
    """
    proto = params_from_flags()

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
    proto.num_boundary_points = num_boundary_points

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
    return ('fx_physical: {}, fy_physical: {}, fz_physical: {}, fx: {}, fy: {},'
            'fz: {}, cx: {}, cy: {}, cz: {}, nx: {}, ny: {}, nz: {}, '
            'core_nx: {}, core_ny: {}, core_nz: {}, lx: {}, ly: {}, lz: {}, '
            'dt: {}, dx: {}, dy: {}, dz: {}, computation_shape: {}, '
            'halo_width: {}, kernel_size: {}, input_chunk_size: {}, '
            'num_output_splits: {}'.format(
                self.fx_physical, self.fy_physical, self.fz_physical, self.fx,
                self.fy, self.fz, self.cx, self.cy, self.cz, self.nx, self.ny,
                self.nz, self.core_nx, self.core_ny, self.core_nz, self.lx,
                self.ly, self.lz, self.dt, self.dx, self.dy, self.dz,
                self.computation_shape, self.halo_width, self.kernel_size,
                self.input_chunk_size, self.num_output_splits))

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

  def _get_grid_spacing(self, full_grid_size, length) -> Optional[float]:
    """Get the grid spacing between nodes in a equidistant mesh.

    Args:
      full_grid_size: The total number of nodes in the mesh grid.
      length: The size of the domain in a particular dimension.

    Returns:
      The distance between two adjacent nodes.
    """
    # The following statement is kept to maintain the behavior of Saint Venant
    # cases.
    full_grid_size -= 2 * self.num_boundary_points
    return length / (full_grid_size - 1) if full_grid_size > 1 else None

  @property
  def dx(self) -> Optional[float]:
    # Note: The final grid should return an outer halo of width 1 due to
    # boundary conditions, but does not. So for now we ignore that in the dx,
    # dy, dz computations.
    return self._get_grid_spacing(self.fx, self.lx)

  @property
  def dy(self) -> Optional[float]:
    return self._get_grid_spacing(self.fy, self.ly)

  @property
  def dz(self) -> Optional[float]:
    return self._get_grid_spacing(self.fz, self.lz)

  @property
  def fx(self):
    """The full grid size in dim 0."""
    return _get_full_grid_size(self.nx, self.halo_width, self.cx,
                               self.num_boundary_points)

  @property
  def fy(self):
    """The full grid size in dim 1."""
    return _get_full_grid_size(self.ny, self.halo_width, self.cy,
                               self.num_boundary_points)

  @property
  def fz(self):
    """The full grid size in dim 2."""
    return _get_full_grid_size(self.nz, self.halo_width, self.cz,
                               self.num_boundary_points)

  @property
  def x(self) -> tf.Tensor:
    """The full grid in dim 0."""
    return _get_full_grid(self.fx, self.lx)

  @property
  def y(self) -> tf.Tensor:
    """The full grid in dim 1."""
    return _get_full_grid(self.fy, self.ly)

  @property
  def z(self) -> tf.Tensor:
    """The full grid in dim 2."""
    return _get_full_grid(self.fz, self.lz)

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
  def meshgrid(self):
    xs = np.linspace(0, self.lx, self.cx * self.core_nx)
    ys = np.linspace(0, self.ly, self.cy * self.core_ny)
    zs = np.linspace(0, self.lz, self.cz * self.core_nz)
    return np.meshgrid(xs, ys, zs, indexing='ij')

  @property
  def num_replicas(self):
    return self.cx * self.cy * self.cz
