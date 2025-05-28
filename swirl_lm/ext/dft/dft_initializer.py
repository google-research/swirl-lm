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

"""Library for initializing DFT on TPUs."""

import enum
import math
from typing import Callable, Tuple

from swirl_lm.utility import common_ops
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow.compat.v1 as tf

_CTYPE = types.TF_COMPLEX_DTYPE
_DTYPE = types.TF_DTYPE

_PI = math.pi
_J = complex(0, 1)


class PartitionDomain(enum.Enum):
  """Defines the domain in which the Vandermonde matrices are partitioned.

  Each TPU core carries partitial information of the Vandermonde matrices,
  either in the spatial/temporal or the spectral domain.
  `SPATIO_TEMPORAL`: the Vandermonde matrix is partitioned in the
  spatial/temporal domain such that each TPU core has the Vandermonde matrix
  with partial spatial/temporal information but full spectral information.
  `SPECTRAL`: the Vandermonde matrix is partitioned in the spectral
  domain such that each TPU core has the Vandermonde matrix with partial
  spectral/temporal information but full spatial/temporal information.
  """
  UNKNOWN = 0
  SPATIO_TEMPORAL = 1
  SPECTRAL = 2


class PartitionDimension(enum.Enum):
  """Defines the dimension along which the Vandermonde matrices are partitioned.

  DIM0 and DIM1 are used by both 2D and 3D DFTs, specifying the dimension
  along which the Vandermonde matrices are partitioned. For example, DIM1 is
  applied to the 2D `Tensor` that pre-multiplies the input 2D `Tensor`
  along dimension 1 (j in [i, j, k] notation). Similaly, DIM0 is applied to
  the 2D `Tensor` that post-multiplies the input `Tensor` along dimension 0
  (i in the [i, j, k] notation). DIM2 is only used by 3D DFT. For now, if
  DIM2 is selected, the 2D `Tensor` is partitioned along dimension 1. DIM0
  partition uses the information in `grid_parametrization.GridParametrization`
  and the coordinate of dimension 1; DIM1 partition uses the information in
  `grid_parametrization.GridParametrization` and the coordinate of dimension 0;
  and DIM2 partition uses the information in
  `grid_parametrization.GridParametrization` and the coordinate of dimension 2.
  """
  DIM0 = 0
  DIM1 = 1
  DIM2 = 2


def _validate_inputs(
    params: grid_parametrization.GridParametrization,
    coordinate: Tuple[int, int, int]) -> None:
  """Validates input prior to mesh generation.

  Args:
    params: A `grid_parametrization.GridParametrization` object containing the
      required grid spec information.
    coordinate: A 3-tuple representing the coordinate of the core in the logical
      mesh [i, j, k].

  Returns:
    None

  Raises:
    ValueError: If arguments are incorrect.
  """
  c0 = params.cx
  c1 = params.cy
  c2 = params.cz
  g0 = coordinate[0]
  g1 = coordinate[1]
  g2 = coordinate[2]

  if not 0 <= g0 < c0 or not 0 <= g1 < c1 or not 0 <= g2 < c2:
    raise ValueError("Invalid subgrid coordinate specified. Subgrid {} "
                     "specified while compute shape is ({}, {}, {})".format(
                         coordinate, c0, c1, c2))


def _dft_partial_mesh_for_core(
    params: grid_parametrization.GridParametrization,
    coordinate: Tuple[int, int, int],
    value_fn: Callable[[tf.Tensor, tf.Tensor, float], tf.Tensor],
    partition_dimension: PartitionDimension,
    partition_domain: PartitionDomain) -> tf.Tensor:
  """Generates a partial mesh per core with partition in the spectral domain.

  The full grid spec is provided by `params`. The value function `value_fn`
  takes a 3D mesh grid and corresponding lengths in three different dimensions
  as arguments (dimension 2 data is not used). It returns the Vandermonde
  matrix for the core at the coordinate specified by `coordinate`.

  Args:
    params: A `grid_parametrization.GridParametrization` object containing the
      required grid spec information.
    coordinate: A 3-tuple representing the coordinate of the core in the logical
      mesh [i, j, k].
    value_fn: A function that takes the local mesh_grid tensor for the core (in
      order i, j) and the number of sampling points along the non-partition
      dimension and returns a 2D `Tensor` representing the value for the local
      core (without including the margin/overlap between the cores).
      Currently it is expected that value_fn produces `tf.complex64`.
    partition_dimension: A `PartitionDimension` enum specifying the dimension
      along which the returned 2D `Tensor` is partitioned.
    partition_domain: A `PartitionDomain` enum specifying the domain in which
      the returned 2D `Tensor` is partitioned.

  Returns:
    A 2D `Tensor` of `tf.complex64` for each core.
  """
  _validate_inputs(params, coordinate)

  c0 = params.cx
  c1 = params.cy
  c2 = params.cz
  n0 = params.nx
  n1 = params.ny
  n2 = params.nz
  g0 = coordinate[0]
  g1 = coordinate[1]
  g2 = coordinate[2]

  if partition_domain == PartitionDomain.SPATIO_TEMPORAL:
    if partition_dimension == PartitionDimension.DIM0:
      num_pts = c1 * n1
      vec_dim0 = tf.cast(tf.range(g1 * n1, (g1 + 1) * n1), dtype=_DTYPE)
      vec_dim1 = tf.cast(tf.range(num_pts), dtype=_DTYPE)
    elif partition_dimension == PartitionDimension.DIM1:
      num_pts = c0 * n0
      vec_dim0 = tf.cast(tf.range(num_pts), dtype=_DTYPE)
      vec_dim1 = tf.cast(tf.range(g0 * n0, (g0 + 1) * n0), dtype=_DTYPE)
    elif partition_dimension == PartitionDimension.DIM2:
      num_pts = c2 * n2
      vec_dim0 = tf.cast(tf.range(g2 * n2, (g2 + 1) * n2), dtype=_DTYPE)
      vec_dim1 = tf.cast(tf.range(num_pts), dtype=_DTYPE)
  elif partition_domain == PartitionDomain.SPECTRAL:
    if partition_dimension == PartitionDimension.DIM0:
      num_pts = c0 * n0
      vec_dim0 = tf.cast(tf.range(g0 * n0, (g0 + 1) * n0), dtype=_DTYPE)
    elif partition_dimension == PartitionDimension.DIM1:
      num_pts = c1 * n1
      vec_dim0 = tf.cast(tf.range(g1 * n1, (g1 + 1) * n1), dtype=_DTYPE)
    elif partition_dimension == PartitionDimension.DIM2:
      num_pts = c2 * n2
      vec_dim0 = tf.cast(tf.range(g2 * n2, (g2 + 1) * n2), dtype=_DTYPE)
    vec_dim1 = tf.cast(tf.range(num_pts), dtype=_DTYPE)

  vec_dim2 = [0.0]
  mat_dim0, mat_dim1, _ = common_ops.meshgrid(vec_dim0, vec_dim1, vec_dim2)
  mat_dim0 = tf.cast(tf.squeeze(mat_dim0, 2), dtype=_CTYPE)
  mat_dim1 = tf.cast(tf.squeeze(mat_dim1, 2), dtype=_CTYPE)

  return value_fn(mat_dim0, mat_dim1, num_pts)


def gen_vandermonde_mat(
    params: grid_parametrization.GridParametrization,
    coordinate: Tuple[int, int, int],
    partition_dimension: PartitionDimension,
    partition_domain: PartitionDomain) -> tf.Tensor:
  """Generates Vandermonde matrix for DFT calculation.

  Args:
    params: A `grid_parametrization.GridParametrization` object containing the
      required grid spec information.
    coordinate: A 3-tuple representing the coordinate of the core in the logical
      mesh [i, j, k].
    partition_dimension: A `PartitionDimension` enum specifying the dimension
      along which the returned 2D `Tensor` is partitioned.
    partition_domain: A `PartitionDomain` enum specifying the dimension along
      which the returned 2D `Tensor` is partitioned.

  Returns:
    A 2D `Tensor` of `tf.complex64` for each core, representing the
    Vandermonde matrix.
  """
  def vandermonde_fn(mat_dim0, mat_dim1, num_pts):
    coeff = -2.0 * _PI * _J / tf.cast(num_pts, dtype=_CTYPE)
    return tf.math.exp(coeff * tf.math.multiply(mat_dim0, mat_dim1))

  return _dft_partial_mesh_for_core(params, coordinate, vandermonde_fn,
                                    partition_dimension, partition_domain)
