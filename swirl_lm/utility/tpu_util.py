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

"""Helper functions for fluid simulation."""

from typing import Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

TensorOrArray = Union[tf.Tensor, np.ndarray]


def grid_coordinates(computation_shape: np.ndarray) -> np.ndarray:
  """Returns a numpy array containing all grid coordinates.

  Args:
    computation_shape: A sequence of integers giving the shape of the grid.

  Returns:
    A numpy array with shape
    (np.prod(computation_shape), len(computation_shape)) and type np.int32.
  """
  rank = len(computation_shape)
  assert rank > 0
  coords = np.meshgrid(
      *[np.arange(x, dtype=np.int32) for x in computation_shape], indexing='ij')
  return np.stack(coords, axis=-1).reshape(-1, rank)


def gen_computation_stride(computation_shape: np.ndarray,
                           tpu_mesh_shape: np.ndarray) -> np.ndarray:
  """Generates `computation_stride` for TPU `device_assignment`.

  This approach to generate `computation_stride` for TPU `device_assignment`
  supports both the existing and the upcoming generations of TPUs, including
  TPUv2, TPUv3, and TPUv4. The user-defined `computation_shape`
  is recast into the format of `computation_stride` for TPU `device_assignment`.
  The recasting is based on `tpu_mesh_shape`, the TPU topology `mesh_shape`,
  describing the shape of TPU topology, a rank-1 array of size 4, and in the
  format of `[nx, ny, nz, num_cores]` with `ni (i = x, y, z)` denoting the
  number of TPU chips along each dimension and `num_cores` denotes the number of
  cores per requested chip. The recasting finds the `strides` that pack the
  requested number of TPU cores with the given `tpu_mesh_shape`.

  Args:
    computation_shape: A rank 1 array of size 3 representing the shape of the
      user-defined computational grid. Each element in the grid represents the
      requested number of processors, to be precise, TPU cores.
    tpu_mesh_shape: The TPU topology `mesh_shape`, a rank 1 array of size 4
      describing the shape of the TPU topology, which is in the form of
      `[nx, ny, nz, num_cores]` with `ni (i = x, y, z)` denoting the number of
      TPU chips along each dimension and `num_cores` denoting the number of
      cores per requested chip. Note that `num_cores` can be `1` or `2`. In
      TPUv2 and TPUv3 TPUs, the number of TPU chips along the third
      dimension `nz` is always 1.
  Returns:
    The `computation_stride` for TPU `device_assignment`, a rank 1 array of size
    `topology_rank`, describing the inter-core spacing in the TPU topology. Note
    that `topology_rank` is always `4` ensuring the consistency between
    TPUv2/TPUv3 and TPUv4 TPUs.
  Raises:
    ValueError: If `computation_shape` does not fit the TPU topology mesh
    shape `tpu_mesh_shape`.
  """
  computation_stride = np.ones_like(tpu_mesh_shape)
  num_cores_requested = np.prod(computation_shape)

  if num_cores_requested > np.prod(tpu_mesh_shape):
    raise ValueError('Requested {} cores, whereas only {} are available from '
                     'the topology.'.format(num_cores_requested,
                                            np.prod(tpu_mesh_shape)))

  sorted_idx = np.argsort(tpu_mesh_shape, kind='stable')
  idx = 0
  while idx <= 3:
    div, mod = np.divmod(num_cores_requested, tpu_mesh_shape[sorted_idx[idx]])
    if mod == 0:
      num_cores_requested = div
      computation_stride[sorted_idx[idx]] = tpu_mesh_shape[sorted_idx[idx]]
    if div == 0:
      computation_stride[sorted_idx[idx]] = mod
      break
    idx += 1

  if np.prod(computation_stride) < np.prod(computation_shape):
    raise ValueError('Requested computation_shape ({}, {}, {}) does not fit '
                     'into TPU topology mesh_shape ({}, {}, {}, {}).'.format(
                         computation_shape[0], computation_shape[1],
                         computation_shape[2], tpu_mesh_shape[0],
                         tpu_mesh_shape[1], tpu_mesh_shape[2],
                         tpu_mesh_shape[3]))

  return computation_stride


def tpu_device_assignment(
    computation_shape: np.ndarray, tpu_topology: tf.tpu.experimental.Topology
) -> Tuple[tf.tpu.experimental.DeviceAssignment, np.ndarray]:
  """Builds a DeviceAssignment that maps grid coordinates to TPU cores."""
  compute_core_assignment = grid_coordinates(computation_shape)

  computation_stride = gen_computation_stride(computation_shape,
                                              tpu_topology.mesh_shape)

  device_assignment = tf.tpu.experimental.DeviceAssignment.build(
      tpu_topology,
      computation_stride=computation_stride,
      num_replicas=np.prod(computation_stride))

  return device_assignment, compute_core_assignment


def combine_subgrids(subgrids: Sequence[TensorOrArray],
                     replicas: np.ndarray,
                     halo_width: int = 1) -> TensorOrArray:
  """Combines subgrids into a single grid.

  Args:
    subgrids: A sequence of subgrids (3D tensors or numpy arrays), one for each
      replica.
    replicas: A numpy array that maps a replica's grid coordinate to its
        replica_id, e.g. replicas[0, 0, 0] = 0, replicas[0, 0, 1] = 1.
    halo_width: The width of the halo.
  Returns:
    The combined tensor or numpy array.
  """

  if isinstance(subgrids[0], tf.Tensor):
    copy = tf.identity
    concat = tf.concat
  else:
    copy = np.copy
    concat = np.concatenate

  cx, cy, cz = replicas.shape
  left_slice = (
      slice(None, None) if halo_width == 0 else slice(None, -halo_width))
  right_slice = slice(halo_width, None)

  def append_grids(i, j, k, arr_z, arr_yz, arr_xyz):
    """Builds up subgrids and returns the intermediate and final grids."""
    arr = subgrids[replicas[i, j, k]]
    if k == 0:
      arr_z = copy(arr)
    if k > 0 or cz == 1:
      if cz > 1:
        arr_z = concat(
            (arr_z[:, :, left_slice], arr[:, :, right_slice]), axis=2)
        # Assign intermediate arrays to None to force garbage collection to
        # keep memory use low.
        arr = None
      if k == cz - 1:
        if j == 0:
          arr_yz = copy(arr_z)
        if j > 0 or cy == 1:
          if cy > 1:
            arr_yz = concat(
                (arr_yz[:, left_slice, :], arr_z[:, right_slice, :]), axis=1)
            arr_z = None
          if j == cy - 1:
            if i == 0:
              arr_xyz = copy(arr_yz)
            elif cx > 1:
              arr_xyz = concat(
                  (arr_xyz[left_slice, :, :], arr_yz[right_slice, :, :]),
                  axis=0)
              arr_yz = None
    return arr_z, arr_yz, arr_xyz

  arr_z = arr_yz = arr_xyz = None
  for i in range(cx):
    for j in range(cy):
      for k in range(cz):
        arr_z, arr_yz, arr_xyz = append_grids(i, j, k, arr_z, arr_yz, arr_xyz)
  if arr_xyz is None:
    raise ValueError('No replicas.')

  return arr_xyz
