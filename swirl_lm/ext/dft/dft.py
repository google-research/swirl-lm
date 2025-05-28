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

# Copyright 2022 Google LLC
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

"""TPU distributed Fourier transform (DFT).

The implementation of the distributed 2D Discrete Fourier Transform (DFT)
allows the two-way partition of the input 2D `Tensor` along both dimensions
0 and 1. The index notation `[i, j, k]` is used here for a `Tensor`,
even in 2D. `i` denotes dimension 0, `j` denotes dimension 1, and `k`
denotes dimension 2, which in 2D is trivial (size 1). The two-way partition
of the input `Tensor` can be understood as a checkerboard-like partition.
Consequently, the corresponding 2D `Tensor` representing the Vandermonde
matrices that pre- and post-multiply the input `Tensor` are also partitioned,
in turn, along dimension 0 and 1.

The implementation of the distributed 3D DFT allows the three-way partition
of the input 3D `Tensor` along dimensions 0, 1, and 2, respectively. Each TPU
core takes a 3D subgrid `Tensor` as input. There are two major steps
required by the 3D DFT computation: (1) treating the 3D DFT as a sequence of
2D DFTs along dimension 2, in which the partions of the input and the
Vandermonde `Tensor`s remain the same as the 2D DFT; (2) generating the 2D
`Tensor` representing the Vandermonde matrix of the third dimension and
multiplying it with the obtained 2D DFT results.
"""
import collections
import enum

from typing import List, Text, Tuple

import numpy as np
import tensorflow.compat.v1 as tf

ThreeTupleTensor = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]


class Dimension(enum.Enum):
  """A class defining the dimension of an input `Tensor` for DFT."""

  DIM0 = 0
  DIM1 = 1
  DIM2 = 2


def gen_group_assignment(
    computation_shape: np.ndarray,
    dim: Dimension) -> List[List[int]]:
  """Creates a group assignment of TPU devices along dimension 0, 1, and 2.

  The group assignments are used by `tf.raw_ops.AllToAll` for broadcasting data
  across TPU replicas. As an example consider a `computation_shape` of `[2, 2,
  2]`, where the `replica id`s are 0 through 7. Each replica is assigned to a
  unique 3-tuple `coordinate`. In this example, the mapping of coordinates to
  replica ids is `{(0, 0, 0): 0, (0, 0, 1): 1, (0, 1, 0): 2, (0, 1, 1): 3, (1,
  0, 0): 4, (1, 0, 1): 5, (1, 1, 0): 6, (1, 1, 1): 7}` with each element
  following the form `(coordinate[0], coordinate[1], coordinate[2]):
  replica_id`. The group assignment along dimension 0 contains the replica ids
  with the same coordinate of dimensions 1 and 2. In this example it is `[[0,
  4], [2, 6], [1, 5], [3, 7]]`. Similarly, the group assignment along dimension
  1 contains the replica ids with the same coordinate of dimesnions 0 and 2,
  which is `[[0, 2], [4, 6], [1, 3], [5, 7]]` in this example.  And the
  group assignment along dimension 2 in the example is `[[0, 1], [2, 3], [4, 5],
  [6, 7]]`.

  Args:
    computation_shape: A 1D `np.ndarray` representing the shape of the
      computational grid. Each element denotes the number of TPU cores along its
      dimension.
    dim: A `Dimension` enum specifying the dimension of the input `Tensor`.

  Returns:
    A `List` of `replica_group`s with each `replica_group` containing
    a `List` of `replica id`s.
  """

  num_replicas = np.prod(computation_shape)
  replica_ids = (np.arange(num_replicas)).reshape(
      (-1, computation_shape[2]))

  group_assignment = []
  if dim == Dimension.DIM2:
    group_assignment = replica_ids.tolist()
  elif dim == Dimension.DIM1:
    for x in np.transpose(replica_ids):
      for y in x.reshape((-1, computation_shape[1])).tolist():
        group_assignment.append(y)
  elif dim == Dimension.DIM0:
    for x in np.transpose(replica_ids):
      for y in (np.transpose(x.reshape((-1, computation_shape[1])))).tolist():
        group_assignment.append(y)

  return group_assignment


def gen_source_target_pairs(
    computation_shape: np.ndarray,
    dim: Dimension) -> List[Tuple[int, int]]:
  """Creates source-target pairs of TPU devices along dimension 0, 1, and 2.

  The source-target pairs are used by `tf.raw_ops.CollectivePermute` for sending
  and receiving data across TPU replicas. The source-target pairs are created
  based on the `group_assignment` along one specific dimension. As an example
  consider a `computation_shape` of `[3, 2, 2]`, where the `replica id`s are 0
  through 11. Each replica is assigned to a unique 3-tuple `coordinate`. In
  this example, the mapping of coordinates to replica ids is `{(0, 0, 0): 0,
  (0, 0, 1): 1, (0, 1, 0): 2, (0, 1, 1): 3, (1, 0, 0): 4, (1, 0, 1): 5,
  (1, 1, 0): 6, (1, 1, 1): 7, (2, 0, 0): 8, (2, 0, 1): 9, (2, 1, 0): 10,
  (2, 1, 1): 11}` with each element following the form
  `(coordinate[0], coordinate[1], coordinate[2]): replica_id`. The group
  assignment along dimension 0 contains the replica ids with the same coordinate
  of dimensions 1 and 2. In this example it is `[[0, 4, 8], [2, 6, 10],
  [1, 5, 9], [3, 7, 11]]`. Therefore, the source-target pairs along dimension 0
  are `[(4, 0), (8, 4), (0, 8), (6, 2), (10, 6), (2, 10), (5, 1), (9, 5),
  (1, 9), (7, 3), (11, 7), (3, 11)]`.

  Args:
    computation_shape: A 1D `np.ndarray` representing the shape of the
      computational grid. Each element denotes the number of TPU cores along its
      dimension.
    dim: A `Dimension` enum specifying the dimension of the input `Tensor`.

  Returns:
    A `List` of source-target pairs in terms of the replica pairs.
  """

  group_assignment = gen_group_assignment(computation_shape, dim)

  source_target_pairs = []

  for replica_group in group_assignment:
    source_target_pairs += zip(replica_group[1:] + replica_group[:1],
                               replica_group)

  return source_target_pairs


def _get_number_replicas(group_assignment: List[List[int]]) -> int:
  """Returns the number of replicas per group.

  Args:
    group_assignment: A `List` of `replica_group`s where each `replica_group`
      contains a `List` of `replica id`s. It is used by the `tpu all_to_all`
      operation, which is applied within a `replica_group`. All `replica_group`s
      within one `group_assignment` share the same number of replicas. The
      formation of `group_assignment` along dimensions 0, 1, and 2 can be found
      in `gen_group_assignment`.
  Returns:
    The number of replicas in a `replica_group`.
  """

  return len(group_assignment[0])


def _all_to_all_matmul(a: tf.Tensor,
                       b: tf.Tensor,
                       subscripts: Text,
                       split_dimension: int,
                       concat_dimension: int,
                       group_assignment: List[List[int]]) -> tf.Tensor:
  """Multiplies two `Tensor`s over a group of TPU replicas.

  The communication among TPU replicas is through `all_to_all`.

  Args:
    a: A `Tensor` of type `tf.complex64`.
    b: A `Tensor` of type `tf.complex64`.
    subscripts: `String` specifying the subscripts for summation as comma
      separated list of subscript labels, following the same convention in
      tf.einsum.
    split_dimension: An `int` in the interval [0, n) along which a `Tensor` is
      split in `tf.raw_ops.AllToAll()` with `n` denoting the rank of the
      `Tensor`.
    concat_dimension: An `int` in the interval [0, n) along which the split
      blocks of a `Tensor` are concatenated.
    group_assignment: A `List` of `replica_group`s where each `replica_group`
      contains a `List` of `replica id`s.

  Returns:
    A `Tensor` of type `tf.complex64`, of which the number of dimensions is the
    same as the input `Tensor` for the DFT computation..
  """

  def all_to_all(func, ab_expand):
    return tf.raw_ops.AllToAll(
        input=func(ab_expand),
        split_dimension=split_dimension,
        concat_dimension=concat_dimension,
        split_count=_get_number_replicas(group_assignment),
        group_assignment=group_assignment)

  ab = tf.einsum(subscripts, a, b)
  ab_expand = tf.compat.v1.expand_dims(ab, axis=-1)
  ab_sum = tf.complex(
      all_to_all(tf.real, ab_expand),
      all_to_all(tf.imag, ab_expand))

  return tf.math.reduce_sum(ab_sum, keepdims=None, axis=concat_dimension)


def dft_2d(a: tf.Tensor,
           vm: tf.Tensor,
           vn: tf.Tensor,
           computation_shape: np.ndarray) -> tf.Tensor:
  """Computes the 2D discrete Fourier transform.

  Args:
    a: A 2D `Tensor` representing the input signal.
    vm: A 2D `Tensor` representing the Vandermonde matrix that pre-multiplies
      the input `Tensor`.
    vn: A 2D `Tensor` representing the Vandermonde matrix that post-multiplies
      the input `Tensor`.
    computation_shape: A 1D `np.ndarray` representing the shape of the
      computation grid. Each element denotes the number of TPU cores along
      its dimension.

  Returns:
    A 2D `Tensor` of type `tf.complex64`.
  """

  group_assignment_dim0 = gen_group_assignment(computation_shape,
                                               Dimension.DIM0)
  group_assignment_dim1 = gen_group_assignment(computation_shape,
                                               Dimension.DIM1)

  vm_a = _all_to_all_matmul(vm, a, 'ij,jk->ik', 0, 2, group_assignment_dim0)
  vm_a_vn = _all_to_all_matmul(vm_a, vn, 'ij,jk->ik', 1, 2,
                               group_assignment_dim1)

  return vm_a_vn


def dft_3d(a: tf.Tensor,
           vm: tf.Tensor,
           vn: tf.Tensor,
           vs: tf.Tensor,
           computation_shape: np.ndarray) -> tf.Tensor:
  """Computes the 3D discrete Fourier Transform.

  Args:
    a: A 3D `Tensor` representing the input signal.
    vm: A 2D `Tensor` representing the Vandermonde matrix that pre-multiplies
      the input `Tensor`.
    vn: A 2D `Tensor` representing the Vandermonde matrix that post-multiplies
      the input `Tensor`.
    vs: A 2D `Tensor` representing the Vandermonde matrix that multiplies the
      2D DFT results.
    computation_shape: A 1D `np.ndarray` representing the shape of the
      computation grid. Each element denotes the number of TPU cores along its
      dimension.

  Returns:
    A 3D `Tensor` of type `tf.complex64`.
  """

  group_assignment_dim0 = gen_group_assignment(computation_shape,
                                               Dimension.DIM0)
  group_assignment_dim1 = gen_group_assignment(computation_shape,
                                               Dimension.DIM1)
  group_assignment_dim2 = gen_group_assignment(computation_shape,
                                               Dimension.DIM2)

  vm_a = _all_to_all_matmul(vm, a, 'ij,jkl->ikl', 0, 3, group_assignment_dim0)

  vm_a_vn = _all_to_all_matmul(
      vm_a, vn, 'ijl,jk->ikl', 1, 3, group_assignment_dim1)

  vm_a_vn_vs = _all_to_all_matmul(
      vm_a_vn, vs, 'ijk,kl->ijl', 2, 3, group_assignment_dim2)

  return vm_a_vn_vs


def _cross_replica_sum_einsum(v: tf.Tensor,
                              a: tf.Tensor,
                              num_slices: int,
                              slice_size: int,
                              group_assignment_dim: List[List[int]],
                              core_index: tf.Tensor,
                              dim: Dimension) -> tf.Tensor:
  """Computes the discrete Fourier Transform along the specified dimension.

  The communication among TPU cores is through `cross_replica_sum`.

  Args:
    v: A 2D `Tensor` of `tf.complex64` representing the Vandermonde matrix
    a: A 3D `Tensor` of `tf.complex64` containing input or partially computed
      Fourier transform.
    num_slices: The number of slices that tensor `v` is partitioned into,
      which is the same as the number of TPU cores along that specific dimension
      in `computation shape`.
    slice_size: The size of a slice that tensor `v` is partitioned with, which
      is the same as the number of points along that specific dimension and
      handled by the specific TPU core.
    group_assignment_dim:  A `List` of `replica_group`s where each
      `replica_group` contains a `List` of `replica id`s.
    core_index: The index of the TPU core in the logical mesh along
      dimension `dim`.
    dim: A `Dimension` enum specifying the dimension along which the Fourier
      transfrom is computed.
  Returns:
    A 3D `Tensor` of type `tf.complex64` with Fourier transform computed along
    dimension `dim`.
  """

  slice_idx = tf.constant(0, dtype=tf.int32)
  a_replica = tf.zeros_like(a)

  def condition(slice_idx, a_replica):
    """The termination condition of the tf.while_loop."""
    del a_replica
    return slice_idx < num_slices

  def body(slice_idx, a_replica):
    """The function body of the tf.while_loop."""
    pick = tf.cond(tf.math.equal(slice_idx, core_index),
                   lambda: tf.complex(tf.constant(1.0), tf.constant(0.0)),
                   lambda: tf.complex(tf.constant(0.0), tf.constant(0.0)))

    if dim is Dimension.DIM0:
      a_replica_slice = tf.einsum(
          'ij,jkl->ikl',
          tf.slice(v, [slice_idx * slice_size, 0], [slice_size, slice_size]),
          a)
    elif dim is Dimension.DIM1:
      a_replica_slice = tf.einsum(
          'ijl,jk->ikl',
          a,
          tf.slice(v, [0, slice_idx * slice_size], [slice_size, slice_size]))
    elif dim is Dimension.DIM2:
      a_replica_slice = tf.einsum(
          'ijk,kl->ijl',
          a,
          tf.slice(v, [0, slice_idx * slice_size], [slice_size, slice_size]))

    a_replica += pick * tf.complex(
        tf.raw_ops.CrossReplicaSum(
            input=tf.real(a_replica_slice),
            group_assignment=group_assignment_dim),
        tf.raw_ops.CrossReplicaSum(
            input=tf.imag(a_replica_slice),
            group_assignment=group_assignment_dim))

    slice_idx += 1

    return (slice_idx, a_replica)

  _, a_update = tf.while_loop(
      condition, body, loop_vars=(slice_idx, a_replica),
      return_same_structure=True, back_prop=False)

  return a_update


def dft_3d_slice_cross_replica_sum(
    a: tf.Tensor,
    vm: tf.Tensor,
    vn: tf.Tensor,
    vs: tf.Tensor,
    computation_shape: np.ndarray,
    core_indices: ThreeTupleTensor) -> tf.Tensor:
  """Computes the 3D discrete Fourier Transform.

  In the slice mode computation, the action of tensor contraction is between
  one `slice` of the 2D Vandermonde `Tensor` and the 3D input `Tensor` or the
  3D `Tensor` containing partial results of Fourier transform. In slice mode the
  peak memory usage is reduced. The results are updated among TPU cores
  between slices through the `cross_replica_sum` TPU operation.

  Args:
    a: A 3D `Tensor` representing the input signal.
    vm: A 2D `Tensor` representing the Vandermonde matrix that pre-multiplies
      the input `Tensor`.
    vn: A 2D `Tensor` representing the Vandermonde matrix that post-multiplies
      the input `Tensor`.
    vs: A 2D `Tensor` representing the Vandermonde matrix that multiplies the
      2D DFT results.
    computation_shape: A 1D `np.ndarray` representing the shape of the
      computation grid. Each element denotes the number of TPU cores along its
      dimension.
    core_indices: A 3-tuple of `tf.Tensor` representing the indices of the
      TPU core in the 3D logical mesh `[i, j, k]`.

  Returns:
    A 3D `Tensor` of type `tf.complex64`.
  """

  group_assignment_dim0 = gen_group_assignment(computation_shape,
                                               Dimension.DIM0)
  group_assignment_dim1 = gen_group_assignment(computation_shape,
                                               Dimension.DIM1)
  group_assignment_dim2 = gen_group_assignment(computation_shape,
                                               Dimension.DIM2)

  vm_a = _cross_replica_sum_einsum(
      vm, a, computation_shape[0], a.shape[0], group_assignment_dim0,
      core_indices[0], Dimension.DIM0)

  vm_a_vn = _cross_replica_sum_einsum(
      vn, vm_a, computation_shape[1], a.shape[1], group_assignment_dim1,
      core_indices[1], Dimension.DIM1)

  vm_a_vn_vs = _cross_replica_sum_einsum(
      vs, vm_a_vn, computation_shape[2], a.shape[2], group_assignment_dim2,
      core_indices[2], Dimension.DIM2)

  return vm_a_vn_vs


def _collective_permute_einsum(v: tf.Tensor,
                               a: tf.Tensor,
                               num_slices: int,
                               slice_size: int,
                               source_target_pairs: List[Tuple[int, int]],
                               core_index: tf.Tensor,
                               einsum_subscripts_dim: Text) -> tf.Tensor:
  """Computes the discrete Fourier Transform along the specified dimension.

  The communication required by the distributed computation is achieved through
  `collective_permute`. Both the Vandermonde matrices and the input are local
  to the TPU core, containing partial information in the SIMD context. The
  partition to the input tensor is three-way, along dimensions 0, 1, and 2. The
  partition to the Vandermonde matrix is only one way, along dimension 0. The
  tensor representing the Vandermonde matrix has shape (num_r, num_k) where
  `num_k` denotes total number of points in the spectral domain without
  partition and is generally a very large number. Consequently, the tensor
  contraction between the local input tensor and the Vandermonde matrix may lead
  to a giant 3D tensor. This causes a peak high-bandwidth-memory usage,
  preventing from solving larger size problems. In order to reduce the peak HBM
  usage, the idea of slicing through the Vandermonde matrix is implemented. It
  slices the Vandermonde matrix along dimension 1 at the step of `slice_size`
  with a total number of `num_slices`. Therefore, each slice of the Vandermonde
  matrix is contracted with the local input tensor. After the tesnor contraction
  and right before the next slicing, the local input tensor is shuffled with a
  neighbouring TPU core defined in `source_target_pairs` through
  `collective_permute`.

  Args:
    v: A 2D `Tensor` of `tf.complex64` representing the Vandermonde matrix
    a: A 3D `Tensor` of `tf.complex64` containing input or partially computed
      Fourier transform.
    num_slices: The number of slices that tensor `v` is partitioned into,
      which is the same as the number of TPU cores along that specific dimension
      in `computation shape`.
    slice_size: The size of a slice that tensor `v` is partitioned with, which
      is the same as the number of points along that specific dimension and
      handled by the specific TPU core.
    source_target_pairs:  A `List` of `replica_group`s where each
      `replica_group` contains a `List` of `replica id`s.
    core_index: The index of the TPU core in the logical mesh along
      dimension `dim`.
    einsum_subscripts_dim: The subscripts in `tf.einsum` for computing the
      Fourier transform along one dimension.
  Returns:
    A 3D `Tensor` of type `tf.complex64` with Fourier transform computed along
    dimension `dim`.
  """

  iteration_idx = tf.constant(0, tf.int32)
  slice_idx = core_index

  def multiply_with_vandermonde_matrix(a, slice_idx):
    """Multiplies with Vandermonde matrix."""
    return tf.einsum(
        einsum_subscripts_dim,
        a,
        tf.slice(v, [0, slice_idx * slice_size], [slice_size, slice_size]))

  def increase_slice_idx(slice_idx):
    # Increases the slice index one at a time and resets it to the original
    # value before it is larger than (`num_slices` - 1).
    return tf.math.mod(slice_idx + 1, num_slices)

  a_replica = multiply_with_vandermonde_matrix(a, slice_idx)

  slice_idx = increase_slice_idx(slice_idx)

  def condition(iteration_idx, slice_idx, a, a_replica):
    """The termination condition of the tf.while_loop."""
    del slice_idx, a, a_replica
    return iteration_idx < num_slices - 1

  def body(iteration_idx, slice_idx, a, a_replica):
    """The function body of the tf.while_loop."""

    a = tf.complex(
        tf.raw_ops.CollectivePermute(
            input=tf.real(a), source_target_pairs=source_target_pairs),
        tf.raw_ops.CollectivePermute(
            input=tf.imag(a), source_target_pairs=source_target_pairs))

    a_replica += multiply_with_vandermonde_matrix(a, slice_idx)

    return (iteration_idx + 1, increase_slice_idx(slice_idx), a, a_replica)

  _, _, _, a_update = tf.while_loop(
      condition, body, loop_vars=(iteration_idx, slice_idx, a, a_replica),
      return_same_structure=True, back_prop=False)

  return a_update


def dft_3d_slice_one_shuffle(
    a: tf.Tensor,
    vm: tf.Tensor,
    vn: tf.Tensor,
    vs: tf.Tensor,
    computation_shape: np.ndarray,
    core_indices: ThreeTupleTensor) -> tf.Tensor:
  """Computes the 3D discrete Fourier Transform.

  In the slice mode computation, the action of tensor contraction is between
  one `slice` of the 2D Vandermonde `Tensor` and the 3D input `Tensor` or the
  3D `Tensor` containing partial results of Fourier transform. In slice mode the
  peak memory usage is reduced. The results are updated among TPU cores
  between slices through the `cross_replica_sum` TPU operation.

  Args:
    a: A 3D `Tensor` representing the input signal.
    vm: A 2D `Tensor` representing the Vandermonde matrix that pre-multiplies
      the input `Tensor`.
    vn: A 2D `Tensor` representing the Vandermonde matrix that post-multiplies
      the input `Tensor`.
    vs: A 2D `Tensor` representing the Vandermonde matrix that multiplies the
      2D DFT results.
    computation_shape: A 1D `np.ndarray` representing the shape of the
      computation grid. Each element denotes the number of TPU cores along its
      dimension.
    core_indices: A 3-tuple of `tf.Tensor` representing the indices of the
      TPU core in the 3D logical mesh `[i, j, k]`.

  Returns:
    A 3D `Tensor` of type `tf.complex64`.
  """
  einsum_subscripts = collections.OrderedDict([(Dimension.DIM0, 'jkl,ij->ikl'),
                                               (Dimension.DIM1, 'ikl,jk->ijl'),
                                               (Dimension.DIM2, 'ijl,kl->ijk')])

  dims = []
  source_target_pairs = []
  for key, _ in einsum_subscripts.items():
    dims.append(key)
    source_target_pairs.append(gen_source_target_pairs(computation_shape, key))

  v = [vm, vn, vs]

  def compute_dft_per_dim(i, x):
    """Computes DFT along one dimension."""
    return _collective_permute_einsum(
        v[i], x, computation_shape[i], a.shape[i], source_target_pairs[i],
        core_indices[i], einsum_subscripts[dims[i]])

  out = a
  for i in np.arange(3):
    out = compute_dft_per_dim(i, out)

  return out
