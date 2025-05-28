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
# coding=utf-8
r"""A direct solver for a Kronecker sum of three Hermitian matrices.

Use the tensor product method to solve a linear system
(A₁ ⊗ I ⊗ I + I ⊗ A₂ ⊗ I + I ⊗ I ⊗ A₃) x = b directly using
the tensor product
eigendecomposition [1],

  (A₁ ⊗ I ⊗ I + I ⊗ A₂ ⊗ I + I ⊗ I ⊗ A₃)⁻¹
  = (V₁ ⊗ V₂ ⊗ V₃) (Λ₁ ⊗ I ⊗ I + I ⊗ Λ₂ ⊗ I + I ⊗ I ⊗ Λ₃)⁻¹ (V₁ ⊗ V₂ ⊗ V₃)⁻¹,

where, e.g., V₁ and Λ₁ are the matrix of eigenvectors and diagonal matrix of
eigenvalues of A₁,

  A₁ V₁ = V₁ Λ₁.

[1] Lynch, Robert E., John R. Rice, and Donald H. Thomas. "Direct solution of
partial difference equations by tensor product methods." Numerische Mathematik
6.1 (1964): 185-199.
"""

from typing import Callable, Sequence

import numpy as np
from swirl_lm.ext.dft import dft
from swirl_lm.utility import common_ops
from swirl_lm.utility import types
import tensorflow as tf

_N_DIM = 3
_CTYPE = types.TF_COMPLEX_DTYPE
_DTYPE = types.TF_DTYPE
_ModifyTensorFn = Callable[[tf.Tensor], tf.Tensor]


def _get_slice_indices(
    mesh_size: Sequence[int],
    replica_id: tf.Tensor,
    replicas: np.ndarray,
):
  """Finds the global indices of a distributed tensor in a specific core."""

  computation_shape = np.array(replicas.shape)
  core_id = tf.compat.v1.where(
      tf.equal(tf.constant(replicas, dtype=tf.int32), replica_id))[0]

  def get_slice_in_dim(dim: int):
    """Finds the global indices in a specific dimension."""
    if mesh_size[dim] % computation_shape[dim] != 0:
      raise ValueError(
          'Tensor dimension {0} should be a multiple of grid dimension {0}.'
          'Currently n_dim0 is {1} cx is {2}.'.format(dim, mesh_size[dim],
                                                      computation_shape[dim]))
    num_pts_per_core = mesh_size[dim] // computation_shape[dim]
    return (num_pts_per_core * core_id[dim],
            num_pts_per_core * (core_id[dim] + 1))

  return [get_slice_in_dim(dim) for dim in range(_N_DIM)]


def _prep_partitioned_eigen_system(
    a: Sequence[np.ndarray],
    n_local: Sequence[int],
    replica_id: tf.Tensor,
    replicas: np.ndarray,
):
  """Prepares the eigensystem required by the solver by partition.

  Args:
    a: The linear operators stored in a length 3 list, with each element to be
      applied in dimension 0, 1, and 2, respectively.
    n_local: The size of the right hand side tensor in the current TPU replica.
    replica_id: The id of the replica.
    replicas: A numpy array of TPU replicas.

  Returns:
    lam_local_sum: The eigenvalues of the Kronecker sum, the diagonal entries of
      Λ₁ ⊗ I ⊗ I + I ⊗ Λ₂ ⊗ I + I ⊗ I ⊗ Λ₃.
    v_col: A list of [V₁, V₂, V₃] partitioned by column.
    v_row: A list of [V₁, V₂, V₃] partitioned by row.
  """
  if len(a) != _N_DIM:
    raise ValueError(
        'The number of linear operators should be {}, but {} is given.'.format(
            _N_DIM, len(a)))

  # Finds the coordinate indices in the TPU grid for the current replica.
  n_dim = [a_i.shape[0] for a_i in a]

  if any([n_dim[i] % n_local[i] != 0 for i in range(_N_DIM)]):
    raise ValueError('Global tensor dimension should be fully divisible by the '
                     'local tensor dimension. Currently global dimension is '
                     '{} and local dimension is {}.'.format(n_dim, n_local))

  slice_indices = _get_slice_indices(n_dim, replica_id, replicas)

  # Performs eigenvalue decomposition to the Hermitian matrices A₁, A₂, and A₃.
  lam = [None] * _N_DIM
  v = [None] * _N_DIM

  for i in range(_N_DIM):
    # An alternative to `tf.linalg.eigh` is the python op `scipy.linalg.eigh`.
    # The python op will happen during the graph construction. The output of the
    # python op become node of constants in the graph. We may switch back to
    # python op, or implement both.
    lam[i], v[i] = tf.linalg.eigh(tf.convert_to_tensor(a[i]))

  # Gets the eigenvalues and eigenvectors tensor corresponding to the partition.
  lam_local_sum = tf.zeros(n_local, dtype=a[0].dtype)
  v_col = [None] * _N_DIM
  v_row = [None] * _N_DIM

  for dim in range(_N_DIM):
    slice_len = n_local[dim]

    tf_lam_buf = tf.convert_to_tensor([[lam[dim]]])
    lam_buf_slice = tf.slice(tf_lam_buf, [0, 0, slice_indices[dim][0]],
                             [1, 1, slice_len])
    if dim == 0:
      lam_in_dim = tf.reshape(lam_buf_slice, (-1, 1, 1))
    elif dim == 1:
      lam_in_dim = tf.reshape(lam_buf_slice, (1, -1, 1))
    else:  # dim == 2
      lam_in_dim = tf.reshape(lam_buf_slice, (1, 1, -1))
    lam_local_sum += lam_in_dim

    # Partitions the eigenvector matrix column-wise and transpose it.
    v_col[dim] = tf.slice(tf.transpose(v[dim]), [slice_indices[dim][0], 0],
                          [slice_len, n_dim[dim]])
    v_col[dim] = tf.cast(v_col[dim], dtype=_CTYPE)
    # Partitions the eigenvector matrix row-wise.
    v_row[dim] = tf.slice(v[dim], [slice_indices[dim][0], 0],
                          [slice_len, n_dim[dim]])
    v_row[dim] = tf.cast(v_row[dim], dtype=_CTYPE)

  return lam_local_sum, v_col, v_row


def fast_diagonalization_solver(
    a: Sequence[np.ndarray],
    n_local: Sequence[int],
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    cutoff: float,
) -> _ModifyTensorFn:
  """Direct solver for a Hermitian Kronecker sum.

  Args:
    a: The linear operators stored in a length 3 list, with each element to be
      applied in dimension 0, 1, and 2, respectively.
    n_local: The size of the right hand side tensor in the current TPU replica.
    replica_id: The id of the replica.
    replicas: A numpy array of TPU replicas.
    cutoff: The threshold for the absolute eigenvalue to be considered as 0.

  Returns:
    The solution `{x}` of the linear system in the same format as the right hand
    side vector `{b}`.
  """
  computation_shape = np.array(replicas.shape)
  coordinates = common_ops.get_core_coordinate(replicas, replica_id)

  lam_local_sum, v_col, v_row = _prep_partitioned_eigen_system(
      a, n_local, replica_id, replicas)

  def solve(rhs: tf.Tensor) -> tf.Tensor:
    """rhs: The right hand side vector `{b}`."""

    # Checks if the dimension of the right hand side function is the same as
    # `n_local`.
    rhs_dim = rhs.get_shape().as_list()

    if any([rhs_dim[i] != n_local[i] for i in range(_N_DIM)]):
      raise ValueError('Right hand side tensor size mismatch: expected {}'
                       'actual {}.'.format(n_local, rhs_dim))

    # Step 1: Transform to the tensor-product eigenvector basis.
    #   buf := (V₁ᵀ ⊗ V₂ᵀ ⊗ V₃ᵀ) rhs
    # Note that one-shuffle dft_3d expects each factor (vandermonde matrix) to
    # be partitioned along its first dimension.
    buf = dft.dft_3d_slice_one_shuffle(
        tf.cast(rhs, dtype=_CTYPE), v_col[0], v_col[1], v_col[2],
        computation_shape, coordinates)

    # Step 2: Multiply by the pseudoinverse (controlled by cutoff) of the
    # diagonal matrix of eigenvalues,
    #   buf := (Λ₁ ⊗ I ⊗ I + I ⊗ Λ₂ ⊗ I + I ⊗ I ⊗ Λ₃)⁻¹ buf
    lam_local_inv = tf.compat.v1.where(
        abs(lam_local_sum) > cutoff, 1. / lam_local_sum,
        tf.zeros(n_local, dtype=lam_local_sum.dtype))
    lam_local_inv = tf.cast(lam_local_inv, dtype=_CTYPE)
    buf = lam_local_inv * buf

    # Step 3: Transform back from the tensor-product eigenvector basis.
    #   res := (V₁ ⊗ V₂ ⊗ V₃) buf
    res = dft.dft_3d_slice_one_shuffle(
        buf, v_row[0], v_row[1], v_row[2], computation_shape, coordinates)

    return tf.cast(res, dtype=_DTYPE)

  return solve
