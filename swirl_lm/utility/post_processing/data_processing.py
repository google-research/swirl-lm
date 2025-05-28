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

"""A library with tools that processes simulation data."""

import itertools
import math
from multiprocessing import pool
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

_TF_DTYPE = tf.float32

FILE_FMT = '{}-field-{}-xyz-{}-{}-{}-step-{}.ser'


def _get_dimension_from_mode(mode: str) -> List[int]:
  """Gets the ordering of dimensions from `mode`.

  Args:
    mode: A 3-character string consist of 'x', 'y', and 'z' that represents the
      orientation of the tensor.

  Returns:
    A list of integers with length 3 with each element specifying the actual
    axis of this dimension that a tensor is corresponding to.

  Raises:
    ValueError: If there are repeated characters in `mode`, or characters other
      than 'x', 'y', or 'z', or the length of mode is not 3.
  """
  dims = [ord(c) - ord('x') for c in mode]
  if len(mode) != 3 or len(set(dims)) < 3 or max(dims) > 2 or min(dims) < 0:
    raise ValueError(
        f'Invalid mode {mode}. `mode` has to be a string of length 3 '
        f'constructed by "x", "y", and "z" uniquely.')
  return dims


def read_serialized_tensor(filename: str):
  """Reads a serialized tensor from `filename` and convert it to tf.Tensor."""
  return tf.io.parse_tensor(tf.io.read_file(filename), out_type=_TF_DTYPE)


def write_serialized_tensor(filename: str, tensor: tf.Tensor):
  """Writes `tensor` to a data file as a serialized tensor."""
  return tf.io.write_file(filename, tf.io.serialize_tensor(tensor))


def interpolate_data(
    data: tf.Tensor,
    shape_new: Tuple[int, int, int],
) -> tf.Tensor:
  """Interpolates 3D `data` onto a new mesh assuming same physical size.

  Args:
    data: A 3D tensor to be interpolated.
    shape_new: The shape of the interpolated data.

  Returns:
    A 3D tensor with shape `shape_new` interpolated from `data`.
  """
  # Because the mesh coordinates are computed as `linspace(0, l, n)`, the grid
  # spacing is l / (n - 1). Therefore the ratio of grid spacing requires a
  # subtraction by one from the total number of grid points.
  h_ratio = [
      tf.math.divide(s_o - 1, s_n - 1)
      for s_o, s_n in zip(data.shape, shape_new)
  ]

  # Interpolates from the first dimension to the third.
  prev = tf.identity(data)
  for i in range(3):
    shape = prev.get_shape().as_list()
    shape[0] = shape_new[i]
    buf = tf.zeros(shape, dtype=data.dtype)

    for j in range(shape[0]):
      j_0 = tf.cast(tf.math.floor(j * h_ratio[i]), dtype=tf.int32)
      j_1 = tf.math.minimum(j_0 + 1, data.shape[i] - 1)
      factor = tf.cast(tf.math.mod(j * h_ratio[i], 1.0), dtype=tf.float32)
      plane = (1.0 - factor) * tf.gather(prev, j_0) + factor * tf.gather(
          prev, j_1)
      buf = tf.tensor_scatter_nd_update(buf, [[j]], plane[tf.newaxis, ...])

    # Always shifts the next dimension to interpolate to the first dimension
    # to conform with the argument required of the tf.tensor_scatter_nd_update
    # function.
    prev = tf.transpose(buf, (1, 2, 0))

  return prev


def _get_global_interp_info(
    source_nc: Tuple[int, int, int],
    source_halo: Tuple[int, int, int],
    source_grid: Tuple[int, int, int],
    target_nc: Tuple[int, int, int],
    target_halo: Tuple[int, int, int],
    target_grid: Tuple[int, int, int],
) -> Tuple[
    Tuple[int, int, int],
    Tuple[int, int, int],
    Tuple[float, float, float],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """Generates global interpolation mapping information.

  Args:
    source_nc: The partition of the source data.
    source_halo: The halo width of the source data.
    source_grid: The number of data points in each dimension in a single
      partition for source data, including halos.
    target_nc: The partition of the target data.
    target_halo: The halo width of the target data.
    target_grid: The number of data points in each dimension in a single
      partition for the target data, including halos.

  Returns:
    A tuple of the following:
      * A 3-tuple of integers representing the single partition grid size for
        the source data, in the order of `x-y-z`.
      * A 3-tuple of integers representing the single partition grid size for
        the target data, in the order of `x-y-z`.
      * A length 3 float tuple representing the ratios of the target grid
        spacing to the source grid spacing in three dimensions. Always ordered
        in `x-y-z`.
      * A length 3 tuple of 1-D Numpy array representing the global indices
        of the left point on the source grid from which the interpolation is
        calculated. The 3 components are ordered in `x-y-z`.
      * A length 3 tuple of 1-D Numpy array representing the global indices
        of the right point on the source grid from which the interpolation is
        calculated. The 3 components are ordered in `x-y-z`.
      * A length 3 tuple of 1-D Numpy array representing the fraction of the
        distance between the interpolation evaluation point and the left point
        normalized to the interval distance between the left point and the
        right point. The 3 components are ordered in `x-y-z`.
  """
  inner_source_grid = []
  inner_target_grid = []
  h_ratio = []
  j_0 = []
  j_1 = []
  factor = []
  for dim in range(3):
    inner_source_grid.append(source_grid[dim] - 2 * source_halo[dim])
    total_source_grid = source_nc[dim] * inner_source_grid[-1]
    inner_target_grid.append(target_grid[dim] - 2 * target_halo[dim])
    total_target_grid = target_nc[dim] * inner_target_grid[-1]
    h_ratio.append(float(total_source_grid - 1) / float(total_target_grid - 1))
    j_0.append(np.array([math.floor(i * h_ratio[-1])
                         for i in range(total_target_grid)]))
    j_1.append(np.minimum(j_0[-1] + 1, total_source_grid - 1))
    factor.append(np.array([math.fmod(i * h_ratio[-1], 1.0)
                            for i in range(total_target_grid)]))

  return (tuple(inner_source_grid), tuple(inner_target_grid), tuple(h_ratio),
          tuple(j_0), tuple(j_1), tuple(factor))


def _interpolate_for_one_target_core(
    source_prefix: str,
    source_step: int,
    source_nc: Tuple[int, int, int],
    source_halo: Tuple[int, int, int],
    target_halo: Tuple[int, int, int],
    inner_source_grid: Tuple[int, int, int],
    inner_target_grid: Tuple[int, int, int],
    j_0: Tuple[np.ndarray, np.ndarray, np.ndarray],
    j_1: Tuple[np.ndarray, np.ndarray, np.ndarray],
    factor: Tuple[np.ndarray, np.ndarray, np.ndarray],
    target_core_id: Tuple[int, int, int],
    varname: str,
    mode: str = 'zxy',) -> tf.Tensor:
  """Interpolates for a single target core.

  Args:
    source_prefix: The prefix of the data source.
    source_step: The step id of the data source.
    source_nc: The partition of the source data. Ordered in `x-y-z`.
    source_halo: The halo width of the source data. Ordered in `x-y-z`.
    target_halo: The halo width of the target data. Ordered in `x-y-z`.
    inner_source_grid: The grid size of a signel soruce partition excluding the
      halo. Ordered in `x-y-z`.
    inner_target_grid: The grid size of a signel target partition excluding the
      halo. Ordered in `x-y-z`.
    j_0: A length 3 tuple of 1-D Numpy array representing the global indices
      of the left point on the source grid from which the interpolation is
      calculated. The 3 components are ordered in `x-y-z`.
    j_1: A length 3 tuple of 1-D Numpy array representing the global indices
      of the right point on the source grid from which the interpolation is
      calculated. The 3 components are ordered in `x-y-z`.
    factor: A length 3 tuple of 1-D Numpy array representing the fraction of
      the distance between the interpolation evaluation point and the left point
      normalized to the interval distance between the left point and the right
      point. The 3 components are ordered in `x-y-z`.
    target_core_id: The core id for which the partial target interpolation to
      be generated. Ordered in `x-y-z`.
    varname: The name of the variable to be processed.
    mode: The orientation of the tensor. Valid options are 'z-x-y', 'x-y-z'.
      Assumes to be the same for the source and target data.

  Returns:
    The local/partial iterpolation results for the specified target core.
  """
  axis = [mode.find(d) for d in ('x', 'y', 'z')]
  dims = _get_dimension_from_mode(mode)
  start_source_core = []
  end_source_core = []
  partial_factor = []
  local_j_0 = []
  local_j_1 = []
  for dim in range(3):
    target_grid_start = inner_target_grid[dim] * target_core_id[dim]
    partial_j_0 = j_0[dim][
        target_grid_start:(target_grid_start + inner_target_grid[dim])]
    partial_j_1 = j_1[dim][
        target_grid_start:(target_grid_start + inner_target_grid[dim])]
    partial_factor.append(factor[dim][
        target_grid_start:(target_grid_start + inner_target_grid[dim])])

    start_source_core.append(partial_j_0[0] // inner_source_grid[dim])
    end_source_core.append(partial_j_1[-1] // inner_source_grid[dim])
    local_j_0.append(partial_j_0 -
                     (start_source_core[-1] * inner_source_grid[dim]))
    local_j_1.append(partial_j_1 -
                     (start_source_core[-1] * inner_source_grid[dim]))

  core_limits = [(i, j + 1) for i, j in
                 zip(start_source_core, end_source_core)]
  data = tf.transpose(
      load_and_merge_serialized_tensor(source_prefix, varname, source_step,
                                       source_nc, source_halo, mode,
                                       tuple(core_limits)), axis)

  prev = tf.identity(data)
  for dim in range(3):
    shape = prev.get_shape().as_list()
    shape[0] = inner_target_grid[dim]
    buf = (
        # Calculates (1 - f) * y(j_0) + f * y(j_1).
        # Reshapes (1 - f) so it is broadcastable to y(j_0)
        tf.reshape(tf.cast(1.0 - partial_factor[dim], tf.float32), [-1, 1, 1])
        # y(j_0) is created from gathering the data at the indices specified
        # with j_0.
        * tf.gather_nd(prev, np.expand_dims(local_j_0[dim], 1)) +
        # Reshaps f so it is broadcastable to y(j_1)
        tf.reshape(tf.cast(partial_factor[dim], tf.float32), [-1, 1, 1])
        # y(j_1) is created from gathering the data at the indicies specified
        # with j_1.
        * tf.gather_nd(prev, np.expand_dims(local_j_1[dim], 1)))
    prev = tf.transpose(buf, [1, 2, 0])

  return tf.transpose(tf.pad(prev, [[target_halo[0],] * 2,
                                    [target_halo[1],] * 2,
                                    [target_halo[2],] * 2]), dims)


def sequential_interpolate_distributed_serialized_tensor(
    source_prefix: str,
    source_step: int,
    source_nc: Tuple[int, int, int],
    source_halo: Tuple[int, int, int],
    target_prefix: str,
    target_step: int,
    target_nc: Tuple[int, int, int],
    target_halo: Tuple[int, int, int],
    target_grid: Tuple[int, int, int],
    varname: str,
    mode: str = 'zxy',
) -> None:
  """Interpolates distributed tensor and repartitions it.

  This function does the interpolation sequentially through the output cores
  one by one to limit the peak memory usage.

  Args:
    source_prefix: The prefix of the data source.
    source_step: The step id of the data source.
    source_nc: The partition of the source data.
    source_halo: The halo width of the source data.
    target_prefix: The prefix of the target data.
    target_step: The step id of the target data.
    target_nc: The partition of the target data.
    target_halo: The halo width of the target data.
    target_grid: The number of data points in each dimension in a single
      partition, including halos.
    varname: The name of the variable to be processed.
    mode: The orientation of the tensor. Valid options are 'z-x-y', 'x-y-z'.
      Assumes to be the same for the source and target data.
  """
  dims = _get_dimension_from_mode(mode)

  # Load source data from the first core to get the shape of the source.
  # Note we are not stripping the halos here. This will be handled later.
  filename = FILE_FMT.format(source_prefix, varname, 0, 0, 0, source_step)
  single_tensor = read_serialized_tensor(filename)

  tensor_shape = single_tensor.get_shape().as_list()
  source_grid = [None,] * 3
  for i, dim in enumerate(dims):
    source_grid[dim] = tensor_shape[i]
  source_grid = tuple(source_grid)

  # Get the interploation mapping information. Note everything is in the order
  # of x, y, z.
  (inner_source_grid,
   inner_target_grid,
   _,
   j_0,
   j_1,
   factor) = _get_global_interp_info(source_nc,
                                     source_halo,
                                     source_grid,
                                     target_nc,
                                     target_halo,
                                     target_grid)

  for cx in range(target_nc[0]):
    for cy in range(target_nc[1]):
      for cz in range(target_nc[2]):
        interpolated = _interpolate_for_one_target_core(
            source_prefix=source_prefix,
            source_step=source_step,
            source_nc=source_nc,
            source_halo=source_halo,
            target_halo=target_halo,
            inner_source_grid=inner_source_grid,
            inner_target_grid=inner_target_grid,
            j_0=j_0,
            j_1=j_1,
            factor=factor,
            target_core_id=(cx, cy, cz),
            varname=varname,
            mode=mode)

        filename = FILE_FMT.format(
            target_prefix,
            varname,
            cx,
            cy,
            cz,
            target_step)
        write_serialized_tensor(filename, interpolated)


def load_and_merge_serialized_tensor(
    prefix: str,
    varname: str,
    step: int,
    n_core: Tuple[int, int, int],
    halo_width: Tuple[int, int, int],
    mode: str = 'zxy',
    core_limits: Optional[Tuple[Tuple[int, int],
                                Tuple[int, int],
                                Tuple[int, int]]] = None,
) -> tf.Tensor:
  """Merges sharded .ser files for a variable at a specific step.

  Args:
    prefix: The prefix for the data, including the full path to the data.
    varname: The name of the variable for the data to be retrieved.
    step: The step number at which the data is retrieved.
    n_core: A length 3 tuple with elements being the number of cores in the x,
      y, and directions, respectively.
    halo_width: The number of points contained in the halo layer on each side
      of the simulation mesh.
    mode: A 3-character string consisting of 'x', 'y', and 'z' that represents
      the orientation of the tensor. For example:
      If mode is 'zxy', the tensor will be merged following the rule below:
      partition in dimension 2 to be merged in tensor dimension 0;
      partition in dimension 0 to be merged in tensor dimension 1;
      partition in dimension 1 to be merged in tensor dimension 2.
      If mode is 'xyz' the tensor will be merged in correspondence to the
      partition dimensions.
    core_limits: A 3-tuple of ((core_x_start, core_x_end),
                               (core_y_start, core_y_end),
                               (core_z_start, core_z_end)), when set, indicating
      the range of the cores to be merged. Note that the cores included start
      at `core_{x|y|z}_start` (inclusive), but end before `core_{x|y|z}_end` (
      exclusive). When left unset, the range is determined by `n_core`.

  Returns:
    The full 3D data for variable `varname` at `step` without halos.
  """
  axis = [mode.find(dim) for dim in ('x', 'y', 'z')]
  dims = _get_dimension_from_mode(mode)

  halo_width = np.array(halo_width)[dims]

  tensor = []

  if core_limits is not None:
    core_x_range = range(core_limits[0][0], core_limits[0][1])
    core_y_range = range(core_limits[1][0], core_limits[1][1])
    core_z_range = range(core_limits[2][0], core_limits[2][1])
  else:
    core_x_range = range(n_core[0])
    core_y_range = range(n_core[1])
    core_z_range = range(n_core[2])

  for i in core_x_range:
    buf_0 = []
    for j in core_y_range:
      buf_1 = []
      for k in core_z_range:
        filename = FILE_FMT.format(prefix, varname, i, j, k, step)
        buf = read_serialized_tensor(filename)
        n_0, n_1, n_2 = buf.shape
        buf_1.append(buf[halo_width[0]:n_0 - halo_width[0],
                         halo_width[1]:n_1 - halo_width[1],
                         halo_width[2]:n_2 - halo_width[2]])
      buf_0.append(tf.concat(buf_1, axis=axis[2]))
    tensor.append(tf.concat(buf_0, axis=axis[1]))

  return tf.concat(tensor, axis=axis[0])


def distribute_and_write_serialized_tensor(
    tensor: tf.Tensor,
    prefix: str,
    varname: str,
    step: int,
    n_core: Tuple[int, int, int],
    halo_width: Tuple[int, int, int],
    mode: str = 'zxy',
) -> None:
  """Writes the shards of a 3D tensor to files.

  Args:
    tensor: The 3D tensor to be distributed and dumped.
    prefix: The prefix for the data, including the full path to the data.
    varname: The name of the variable for the data to be retrieved.
    step: The number of times steps at which the data is retrieved.
    n_core: A length 3 tuple with elements being the number of cores in the x,
      y, and directions, respectively.
    halo_width: The number of points contained in the halo layer on each side
      of the simulation mesh
    mode: A 3-character string consisting of 'x', 'y', and 'z' that represents
      the orientation of the tensor. For example:
      If mode is 'zxy', the tensor will be partitioned following the rule below:
      tensor dimension 0 will be partitioned in dimension 1;
      tensor dimension 1 will be partitioned in dimension 2;
      tensor dimension 2 will be partitioned in dimension 0.
      If mode is 'xyz' the tensor will be distributed in correspondence to the
      partition dimensions.
  """
  dims = _get_dimension_from_mode(mode)

  orientation_fn = lambda f: np.array(f)[dims]
  n_local = [n / c for n, c in zip(tensor.shape, orientation_fn(n_core))]

  halo_width = orientation_fn(halo_width)
  paddings = [[h,] * 2 for h in halo_width]
  tensor_full = tf.pad(tensor, paddings)

  def write_file(replica):
    c_0, c_1, c_2 = replica
    filename = FILE_FMT.format(prefix, varname, c_0, c_1, c_2, step)
    i, j, k = orientation_fn(replica)
    buf = tensor_full[int(i * n_local[0]):int((i + 1) * n_local[0] +
                                              2 * halo_width[0]),
                      int(j * n_local[1]):int((j + 1) * n_local[1] +
                                              2 * halo_width[1]),
                      int(k * n_local[2]):int((k + 1) * n_local[2] +
                                              2 * halo_width[2])]
    write_serialized_tensor(filename, buf)

  iter_range = itertools.product(*[range(c) for c in n_core])
  replicas = list(iter_range)

  with pool.ThreadPool(np.prod(n_core)) as p:
    p.map(write_file, replicas)
    p.close()
    p.join()


def interpolate_distributed_serialized_tensor(
    source_prefix: str,
    source_step: int,
    source_n_core: Tuple[int, int, int],
    source_halo_width: Tuple[int, int, int],
    target_prefix: str,
    target_step: int,
    target_n_core: Tuple[int, int, int],
    target_halo_width: Tuple[int, int, int],
    target_n_grid: Tuple[int, int, int],
    varname: str,
    mode: str,
) -> None:
  """Interpolates distributed tensor and repartitions it.

  Args:
    source_prefix: The prefix of the data source.
    source_step: The step id of the data source.
    source_n_core: The partition of the source data.
    source_halo_width: The halo width of the source data.
    target_prefix: The prefix of the target data.
    target_step: The step id of the target data.
    target_n_core: The partition of the target data.
    target_halo_width: The halo width of the target data.
    target_n_grid: The number of data points in each dimension in a single
      partition, including halos.
    varname: The name of the variable to be processed.
    mode: The orientation of the tensor. Valid options are 'z-x-y', 'x-y-z'.
      Assumes to be the same for the source and target data.
  """
  dims = _get_dimension_from_mode(mode)

  tensor = load_and_merge_serialized_tensor(source_prefix, varname, source_step,
                                            source_n_core, source_halo_width,
                                            mode)

  shape_new = np.array([
      (n - 2 * h) * c
      for n, h, c in zip(target_n_grid, target_halo_width, target_n_core)
  ])[dims]

  tensor = interpolate_data(tensor, tuple(shape_new))

  distribute_and_write_serialized_tensor(tensor, target_prefix, varname,
                                         target_step, target_n_core,
                                         target_halo_width, mode)


def coordinates_to_indices(
    locations: np.ndarray,
    domain_size: Sequence[float],
    mesh_size_local: Sequence[int],
    partition: Sequence[int],
    halo_width: int,
) -> Tuple[Sequence[int], np.ndarray]:
  """Finds the indices of the partition and physical locations in each core.

  This function assumes that all probes are in the same core of partition.

  Args:
    locations: The indices of the probe locations. Stored in a 2D array of 3
      columns, with the columns being x, y, and z indices, respectively.
    domain_size: A three-element sequence that stores the physical size of the
      full domain.
    mesh_size_local: A length 3 tuple with elements being the number of mesh
      points in the x, y, and z directions in each core, respectively. Including
      halos.
    partition: A length 3 tuple with elements being the number of cores in the
      x, y, and directions, respectively.
    halo_width: The number of points contained in the halo layer on each side
      of the simulation mesh.

  Returns:
    A tuple of 2 elements. The first element is a length three sequence that
    stores the indices of the core in the partition. The second element is a 2D
    np.array. Each row of the array stores the index of the corresponding point
    in `locations` that's local to the core.
  """
  # Effective size of the mesh in each core.
  n = [core_n - 2 * halo_width for core_n in mesh_size_local]

  # Length of the domain in each core.
  core_l = [l_i / nc_i for l_i, nc_i in zip(domain_size, partition)]

  # Grid spacing.
  h = [
      l_i / (n_i * c_i - 1.0)
      for l_i, c_i, n_i in zip(domain_size, partition, n)
  ]

  # Find the indices of the core. Assumes that all probes are in the same
  # partition.
  c_indices = [int(locations[0][i] // core_l[i]) for i in range(3)]

  # Finds the indices of the physical coordinates inside the core.
  indices = np.zeros_like(locations, dtype=int)
  for i in range(3):
    indices[:, i] = np.array(
        (locations[:, i] - c_indices[i] * core_l[i]) // h[i] + halo_width,
        dtype=int)

  return c_indices, indices
