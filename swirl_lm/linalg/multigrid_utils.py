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
"""Library of multigrid utility functions that operate on n-dimensional inputs.

Some functions are compatible with numpy and TensorFlow (that is, they can take
one or more numpy array(s) as input and return a numpy array, or same with
`tf.Tensor`), and some are TensorFlow-only. The former return `TensorOrArray`,
while TensorFlow-only functions return `tf.Tensor`.
"""
import functools
import itertools
import math
import re
from typing import Callable, List, Mapping, Optional, Sequence, Text, Tuple, Union

from absl import logging
import numpy as np
import scipy as sp
import scipy.linalg  # pylint: disable=unused-import
from swirl_lm.base import initializer
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.communication import halo_exchange
from swirl_lm.communication import halo_exchange_utils
from swirl_lm.linalg import poisson_solver_pb2
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf

BCType = halo_exchange_utils.BCType
BoundaryConditionsSpec = halo_exchange_utils.BoundaryConditionsSpec
SideType = halo_exchange_utils.SideType
TensorOrArray = initializer.TensorOrArray
ValueFunction = initializer.ValueFunction

ModifyTensorOrArrayFn = Callable[[TensorOrArray], TensorOrArray]
ProlongRestrictMatrices = Tuple[Sequence[Sequence[Optional[TensorOrArray]]],
                                Sequence[Sequence[Optional[TensorOrArray]]]]
ThreeIntTuple = Tuple[int, int, int]

# Function type for smoother or residual calculations. Takes
# `(x, b, full_grid_shape)`, the current iterate, right hand side, and full
# shape, and returns the next iterate or the residual.
SmootherTypeFn = Callable[[TensorOrArray, TensorOrArray], TensorOrArray]
InitFn = types.InitFn
FlowFieldVal = types.FlowFieldVal

HELPER_VARIABLES = ('ps', 'rs')

_NP_DTYPE = types.NP_DTYPE


def get_shape(x: TensorOrArray) -> Sequence[int]:
  shape = x.shape
  if not isinstance(shape, tuple): shape = shape.as_list()
  return shape


def get_subgrid_shape(full_grid_shape: Sequence[int],
                      computation_shape: Optional[Sequence[int]] = None,
                      halo_width: int = 1) -> Sequence[int]:
  if computation_shape is None:
    return full_grid_shape
  return [(s - 2 * halo_width) // c + 2 * halo_width
          for s, c in zip(full_grid_shape, computation_shape)]


def get_full_grid_shape(subgrid_shape: Sequence[int],
                        computation_shape: Optional[Sequence[int]] = None,
                        halo_width: int = 1) -> Sequence[int]:
  if computation_shape is None:
    return subgrid_shape
  return [(s - 2 * halo_width) * c + 2 * halo_width
          for s, c in zip(subgrid_shape, computation_shape)]


def zero_borders(x: TensorOrArray) -> TensorOrArray:
  """Zeros out the borders of the given tensor (width 1)."""
  rank = len(get_shape(x))
  slices = (slice(1, -1),) * rank
  padding = ((1, 1),) * rank
  if isinstance(x, tf.Tensor):
    return tf.pad(x[slices], paddings=padding)
  return np.pad(x[slices], padding, mode='constant')


def add_borders(x: TensorOrArray, y: TensorOrArray) -> TensorOrArray:
  """Returns `x` with the borders of `y` added."""
  ones_like = tf.ones_like if isinstance(x, tf.Tensor) else np.ones_like
  mask = 1 - zero_borders(ones_like(x))
  return x + mask * y


def get_homogeneous_boundary_conditions(
    boundary_conditions: Optional[BoundaryConditionsSpec] = None
) -> BoundaryConditionsSpec:
  """Returns the given boundary conditions with all values set to 0."""
  if boundary_conditions is None:
    return halo_exchange_utils.homogeneous_bcs()

  zero_valued_bcs = []

  for bc_per_dim in boundary_conditions:
    if bc_per_dim is None:
      zero_valued_bcs.append(None)
    else:
      zero_valued_per_side = []
      for bc_per_side in bc_per_dim:
        if bc_per_side is None:
          zero_valued_per_side.append(None)
        else:
          zero_valued_per_side.append((bc_per_side[0], 0.))
      zero_valued_bcs.append(zero_valued_per_side)

  return zero_valued_bcs


def get_apply_one_core_boundary_conditions_fn(
    boundary_conditions: Optional[BoundaryConditionsSpec] = None,
    homogeneous: bool = False
) -> Optional[ModifyTensorOrArrayFn]:
  """Returns a function that applies boundary conditions.

  Args:
    boundary_conditions: The boundary conditions.
    homogeneous: If true, the returnd function applies homogeneous boundary
      conditions.

  Returns:
    A function that returns a given tensor or array with the boundary
      conditions applied. Or returns `None` if `boundary_conditions` is `None`.
  """
  if boundary_conditions is None:
    return None

  if homogeneous:
    boundary_conditions = get_homogeneous_boundary_conditions(
        boundary_conditions)
  subtract_mean = boundary_conditions_all_neumann(boundary_conditions)

  def halo_exchange_fn(x: TensorOrArray) -> TensorOrArray:
    dtype = x.dtype
    rank = len(get_shape(x))
    x = (halo_exchange_utils.
         apply_one_core_boundary_conditions_to_tensor_or_array(
             x, boundary_conditions))
    inner = (slice(1, -1),) * rank
    if subtract_mean:
      x -= (
          tf.math.reduce_mean(x[inner])
          if isinstance(x, tf.Tensor) else np.mean(x[inner]))

    return tf.cast(x, dtype) if isinstance(x, tf.Tensor) else x.astype(dtype)

  return halo_exchange_fn


def boundary_conditions_all_one_type(
    boundary_conditions: BoundaryConditionsSpec,
    boundary_condition_type: BCType) -> bool:
  """Returns `True` if every boundary type is NEUMANN."""
  if boundary_conditions is None:
    return False
  for bcs_per_dim in boundary_conditions:
    for bc_type, _ in bcs_per_dim:
      if bc_type != boundary_condition_type:
        return False
  return True


def boundary_conditions_all_no_touch(
    bcs: BoundaryConditionsSpec) -> bool:
  return boundary_conditions_all_one_type(bcs, BCType.NO_TOUCH)


def boundary_conditions_all_neumann(
    bcs: BoundaryConditionsSpec) -> bool:
  return boundary_conditions_all_one_type(bcs, BCType.NEUMANN)


def jacobi(
    x: TensorOrArray,
    b: TensorOrArray,
    a_operator: ModifyTensorOrArrayFn,
    a_inv_diagonal: ModifyTensorOrArrayFn,
    n: int = 1,
    weight: float = 1,
    halo_exchange_fn: Optional[ModifyTensorOrArrayFn] = None,
    all_no_touch_boundary_conditions: bool = False) -> TensorOrArray:
  """Returns `x` after `n` iterations of the Jacobi method.

  Jacobi's method is used to obtain an approximate solution of

    `A x = b`

  where `A` is an operator, `b` is a source term and `x` are the unknowns.
  In the Jacobi method, `x` in the `k+1`th iteration is

    `x^(k+1) = x^k + w * D_inv ( b - A x^k )`

  where `D_inv` is the inverse of the diagonal of `A`, and `w` is the weight.
  The default `w = 1` corresponds to unweighted Jacobi.

  The boundaries of `x` are preserved in each iteration (Dirichlet boundary
  conditions). The tensors can be in any number of dimensions.

  Args:
    x: A tensor or array with the initial values of `x`.
    b: The source tensor or array.
    a_operator: A function which returns the operator `A` acting on the input
      tensor or array.
    a_inv_diagonal: A function which returns the inverse of the diagonal of `A`
      times the input tensor or array.
    n: The number of Jacobi iterations to perform.
    weight: The weight factor, for weighted Jacobi. The default 1 corresponds to
      the unweighted Jacobi method.
    halo_exchange_fn: A function which, given a field, returns the field with
      boundary conditions and halo exchange applied.
    all_no_touch_boundary_conditions: If True, a mask is used so the borders are
      unchanged. This is used in simple Dirichlet multicore cases where the
      initial borders of `x` should be used as the Dirichlet values. This
      optimizes the case where halo exchange is used and the boundary conditions
      are not all `NO_TOUCH`.

  Returns:
    `x` after `n` Jacobi iterations.
  """
  use_tf = isinstance(x, tf.Tensor)
  ones_like = tf.ones_like if use_tf else np.ones_like
  if halo_exchange_fn and (not all_no_touch_boundary_conditions):
    mask = 1
  else:
    mask = zero_borders(ones_like(x))

  def body(x, i):
    update = weight * mask * a_inv_diagonal(b - a_operator(x))
    x += update
    if halo_exchange_fn:
      x = halo_exchange_fn(x)
    return x, i + 1

  i = 0
  cond = lambda _, i: i < n
  if use_tf:
    x, _ = tf.while_loop(
        cond=cond, body=body, loop_vars=[x, i], back_prop=False)
  else:
    while i < n:
      x, i = body(x, i)

  return x


def laplacian_and_inv_diagonal_fns(
    full_grid_shape: Sequence[int],
    full_grid_lengths: Optional[Sequence[float]] = None
) -> Tuple[ModifyTensorOrArrayFn, ModifyTensorOrArrayFn]:
  """Returns the Laplacian function and its inverse diagonal function.

  The functions return a tensor that is the same shape as the input tensor. For
  both functions the input tensor can be in any number of dimensions. The
  boundary values are not meaningful and should be masked.

  Args:
    full_grid_shape: The full-grid shape of the tensor or array.
    full_grid_lengths: An optional sequence of full-grid lengths. If not given,
      1 is used.

  Returns:
    A tuple of two functions: the Laplacian and its inverse-diagonal
    functions to second-order. The border values returned by the functions are
    not meaningful.

  Raises:
    `ValueError` in case `full_grid_lengths` and `full_grid_shape` have
    unequal lengths (if `full_grid_lengths` is not `None`).
  """
  rank = len(full_grid_shape)
  if not full_grid_lengths:
    full_grid_lengths = (1,) * rank

  if len(full_grid_lengths) != rank:
    raise ValueError('The size of grid_lengths ({}) must be equal to the '
                     'rank ({}).'.format(len(full_grid_lengths), rank))

  # `s` includes the 1 halo pixel on each of the 2 ends. To be consistent with
  # the grid parametrization where the inner grids represent the nodes of the
  # interior domain that spans the `grid_lengths`, the grid space should be
  # computed with the 2 side halo nodes removed.
  dxs = [
      length / (s - 3) for length, s in zip(full_grid_lengths, full_grid_shape)
  ]
  diag = -2 * sum([dx**-2 for dx in dxs])

  def laplacian(x: TensorOrArray) -> TensorOrArray:
    roll = tf.roll if isinstance(x, tf.Tensor) else np.roll
    lap = diag * x
    for i in range(rank):
      lap += ((roll(x, shift=1, axis=i) + roll(x, shift=-1, axis=i))
              / dxs[i]**2)
    return lap

  def inv_diagonal(x: TensorOrArray) -> TensorOrArray:
    return x / diag

  return laplacian, inv_diagonal


def laplacian_inv_diagonal_fn(
    full_grid_shape: Sequence[int],
    full_grid_lengths: Optional[Sequence[float]] = None
) -> ModifyTensorOrArrayFn:
  _, inv_diagonal = laplacian_and_inv_diagonal_fns(full_grid_shape,
                                                   full_grid_lengths)
  return inv_diagonal


def poisson_jacobi(
    x: TensorOrArray,
    b: TensorOrArray,
    params: Optional[grid_parametrization.GridParametrization] = None,
    n: int = 1,
    weight: float = 1,
    halo_exchange_fn: Optional[ModifyTensorOrArrayFn] = None,
    all_no_touch_boundary_conditions: bool = False) -> TensorOrArray:
  """Returns the Jacobi method applied to Poisson's equation (see `jacobi`)."""
  subgrid_shape = get_shape(x)
  if params:
    rank = len(subgrid_shape)
    full_grid_lengths = (params.lx, params.ly, params.lz)[:rank]
    computation_shape = (params.cx, params.cy, params.cz)
  else:
    full_grid_lengths = None
    computation_shape = None

  full_grid_shape = get_full_grid_shape(subgrid_shape, computation_shape)
  laplacian, lap_inv_diagonal = (
      laplacian_and_inv_diagonal_fns(full_grid_shape, full_grid_lengths))

  return jacobi(x, b, laplacian, lap_inv_diagonal, n, weight, halo_exchange_fn,
                all_no_touch_boundary_conditions)


def poisson_jacobi_fn_for_one_core(
    params: Optional[grid_parametrization.GridParametrization] = None,
    n: int = 1,
    weight: float = 1,
    halo_exchange_fn: Optional[ModifyTensorOrArrayFn] = None) -> SmootherTypeFn:
  """Returns a one-core Jacobi fn for Poisson's equation (see `jacobi`)."""
  return functools.partial(poisson_jacobi, params=params, n=n, weight=weight,
                           halo_exchange_fn=halo_exchange_fn)


def poisson_jacobi_step_fn(
    params: Optional[grid_parametrization.GridParametrization] = None,
    n: int = 1,
    weight: float = 1,
    boundary_conditions: Optional[BoundaryConditionsSpec] = None
) -> Callable[[tf.Tensor, np.ndarray], SmootherTypeFn]:
  """Returns a step fn for Poisson Jacobi (see `jacobi`)."""
  all_no_touch = boundary_conditions_all_no_touch(boundary_conditions)

  def step_fn(replica_id: tf.Tensor, replicas: np.ndarray):
    halo_exchange_fn = halo_exchange_step_fn(
        replica_id, replicas, boundary_conditions)

    return functools.partial(
        poisson_jacobi, params=params, n=n, weight=weight,
        halo_exchange_fn=halo_exchange_fn,
        all_no_touch_boundary_conditions=all_no_touch)

  return step_fn


def poisson_residual(
    x: TensorOrArray,
    b: TensorOrArray,
    params: Optional[grid_parametrization.GridParametrization] = None,
    full_grids: bool = False
) -> TensorOrArray:
  """Returns the Poisson residual.

  Returns the residual `b - A x` for the Poisson equation `A x = b`.

  Args:
    x: The input tensor or array.
    b: The source tensor or array.
    params: The grid parametrization.
    full_grids: if `True`, inputs are full grid.

  Returns:
    The residual tensor or array.
  """
  subgrid_shape = get_shape(x)
  if params:
    full_grid_lengths = (params.lx, params.ly, params.lz)[:len(subgrid_shape)]
  else:
    full_grid_lengths = None

  if (full_grids or (params is None) or
      (params and params.cx * params.cy * params.cz == 1)):
    full_grid_shape = subgrid_shape
  else:
    full_grid_shape = get_full_grid_shape(subgrid_shape,
                                          (params.cx, params.cy, params.cz))

  laplacian, _ = laplacian_and_inv_diagonal_fns(full_grid_shape,
                                                full_grid_lengths)

  return b - laplacian(x)


def poisson_residual_norm(
    x: TensorOrArray,
    b: TensorOrArray,
    params: Optional[grid_parametrization.GridParametrization] = None,
    full_grids: bool = True) -> float:
  """Returns the Poisson residual norm."""
  rank = len(get_shape(x))
  inner = (slice(1, -1),) * rank

  res_no_border = poisson_residual(x, b, params, full_grids)[inner]
  if isinstance(x, tf.Tensor):
    return tf.norm(res_no_border)
  return np.linalg.norm(res_no_border)


def prolong_matrix(n2: int, n1: Optional[int] = None,  # pytype: disable=annotation-type-mismatch  # numpy-scalars
                   dtype: np.dtype = _NP_DTYPE) -> np.ndarray:
  """Returns a prolongation matrix (2D numpy array).

  Returns an `n2 x n1` prolongation matrix. If `v` is a vector of length `n1`,
  representing the nodal values on an evenly spaced grid of `n1` points covering
  some interval, and `p` is the returned prolongation matrix, then

    `w = p @ v`

  is a vector of nodal values on an evenly spaced grid of `n2 > n1` points
  covering the same interval, given by bilinear interpolation from the nearest
  nodes on the `n1` length grid. Both grids include the endpoints, so the first
  and last elements of `w` are equal to those of `v`.

  `n1` is optional. If not provided, it is set to roughly `n2 / 2`. `n2` (and
  `n1`, if given) must be greater than 1.

  NB: The restriction matrix from `n2 -> n1` is the transpose of this matrix up
  to a multiplicative constant.

  Args:
    n2: The number of elements to prolongate to.
    n1: The number of elements to prolongate from. If `None`, it is set to
      `(n2 + 1) // 2`.
    dtype: The dtype.

  Returns:
    The prolongation matrix corresponding to prolongating a vector with `n1`
    nodal values to one with `n2` values via bilinear interpolation.
  """
  if (n1 is not None and n1 < 2) or (n2 < 2):
    raise ValueError('n1 and n2 have to be greater than 1 (got {} and {}).'
                     .format(n1, n2))
  if n1 is not None and n1 >= n2:
    raise ValueError('n1 must be less than n2 (got {} and {}).'.format(n1, n2))
  if n1 is None:
    n1 = (n2 + 1) // 2
  i = np.arange(n2)
  x = i / (n2 - 1) * (n1 - 1)
  j = np.floor(x).astype(int)
  p = np.concatenate((1 - (x - j), x - j))
  i = np.concatenate((i, i))
  j = np.concatenate((j, j + 1))
  b = j < n1
  return sp.sparse.coo_matrix((p[b], (i[b], j[b])),
                              (n2, n1)).toarray().astype(dtype)


def _restrict_matrix_from_prolong(p: np.ndarray) -> np.ndarray:
  """Returns the restriction matrix corresponding to a prolongation matrix."""
  n2, n1 = p.shape
  return p.T * (n1 - 1) / (n2 - 1)


def restrict_matrix(n2: int, n1: Optional[int] = None,  # pytype: disable=annotation-type-mismatch  # numpy-scalars
                    dtype: np.dtype = _NP_DTYPE) -> np.ndarray:
  """Returns a restriction matrix from `n2` to `n1`. See `prolong_matrix`."""
  return _restrict_matrix_from_prolong(prolong_matrix(n2, n1, dtype))


def _grid_sizes(n2: int, n1: int) -> List[Tuple[int, int]]:
  """Returns a sequence of grid sizes from `n2` down to `n1`.

  The size usually reduces by about a factor of 2 each iteration.

  Args:
    n2: The starting size.
    n1: The ending size.

  Returns:
    The list of grid sizes.
  """
  sizes = []
  while n2 > n1:
    m1 = max(n1, (n2 + 1) // 2)
    sizes.append((n2, m1))
    n2 = m1
  return sizes


def _prolong_matrices(n2: int, n1: int,  # pytype: disable=annotation-type-mismatch  # numpy-scalars
                      dtype: np.dtype = _NP_DTYPE) -> List[np.ndarray]:
  """Returns a sequence of prolongation matrices from `n2` to `n1`."""
  return [prolong_matrix(m2, m1, dtype) for m2, m1 in _grid_sizes(n2, n1)]


def _restrict_matrices(n2: int, n1: int) -> List[np.ndarray]:
  """Returns a sequence of restriction matrices from `n` to `m`."""
  return [_restrict_matrix_from_prolong(p) for p in _prolong_matrices(n2, n1)]


def log_full_grid_shapes_from_ps(ps: Sequence[Sequence[np.ndarray]]):
  """Logs the sequence of grid shapes from the prolongation matrices."""

  def big_size(p):
    return None if p is None else p.shape[0]

  def small_size(p):
    return None if p is None else p.shape[1]

  shapes = []
  for i, p_per_level in enumerate(ps):
    if i == 0:
      shapes.append(tuple([big_size(p) for p in p_per_level]))
    shapes.append(tuple([small_size(p) for p in p_per_level]))
  logging.info('Full grid shapes: %s', ' -> '.join([str(s) for s in shapes]))


def prolong_restrict_matrices(
    start_shape: Sequence[int], end_shape: Sequence[int]) -> Tuple[
        List[List[Optional[np.ndarray]]],
        List[List[Optional[np.ndarray]]]]:
  """Returns nested lists of prolongation and restriction matrices.

  Given `start_shape` with each element being greater than or equal to its
  corresponding element in `end_shape`, this function returns two lists of
  matrices corresponding to shrinking the dimensions down to `end_shape` by
  roughly factors of 2 each iteration.

  The function returns a 2-tuple. The first (second) element is a list of a list
  of prolongation (restriction) matrices. The inner list corresponds to each
  dimension (so has length `len(shape)`). The outer list corresponds to the
  prolongation or restriction matrix from a given level to the next level
  down. Typically, each dimension is reduced by about a factor of 2. If 4 is
  reached, then the matrices corresponding to 4 <-> 3 are added. If 3 is
  reached, but some other dimension has not reached 3, then `None` is added,
  which signifies that no restriction / prolongation should happen in that
  dimension at that level.

  To give an example, suppose in 3 dimensions `start_shape` is `(7, 21, 4)`
  and `end_shape` is `(4, 4, 4)`. The first element of the returned
  prolongation matrices list will consist of three prolongation matrices, one
  for each dimension, with shapes
    `(7, 4), (21, 11), None`.
  The next and the final element will have shapes or values
    `None, (11, 6), None`,
  The sequence stops because for every dimension the value is `None` or the
  smaller shape is (in this example) 3.

  Args:
    start_shape: The starting grid shape.
    end_shape: The ending grid shape. Each element must be greater than or equal
      to 4.

  Returns:
    A tuple of (prolongation_matrices, restriction_matrices) corresponding to
    the reduction in size from `start_shape` to `end_shape`.
  """
  if min(end_shape) < 4:
    raise ValueError(f'Each element of `end_shape` ({end_shape}) must be '
                     'greater than 3.')
  if any(np.array(start_shape) - np.array(end_shape)) < 0:
    raise ValueError(f'Each element of `start_shape` ({start_shape}) must be '
                     f'greater than or equal to `end_shape` ({end_shape}).')

  ps = list(itertools.zip_longest(
            *[_prolong_matrices(n2, n1)
              for n2, n1 in zip(start_shape, end_shape)]))
  rs = list(itertools.zip_longest(
            *[_restrict_matrices(n2, n1)
              for n2, n1 in zip(start_shape, end_shape)]))

  log_full_grid_shapes_from_ps(ps)

  return (ps, rs)  # pytype: disable=bad-return-type


def full_1d_grid_size_pairs(
    n: int, num_cores: int, coarsest_subgrid_size: int
) -> Sequence[Tuple[int, int]]:
  """Returns a list of full grid size pairs.

  The first number in the first pair is the starting full grid size `n`. The
  second number in each pair is typically about a factor of 2 smaller than the
  first one (counting degrees of freedom). All grid sizes except `n` are
  adjusted to be divisible by `num_cores` (in the sense of dividing a full grid
  into subgrids with halo width 1; `n` must have this property). The first
  number in each pair is the second number of the previous pair. The last number
  in the last pair is as small as it can be, given that it is greater than
  `num_cores + 2` and greater than the full grid size corresponding to
  `coarsest_subgrid_size`.

  Args:
    n: The starting full grid size.
    num_cores: The number of cores.
    coarsest_subgrid_size: The coarsest subgrid size.

  Returns:
    The set of grid pairs.

  Raises:
    `ValueError` in case `n - 2` is not divisible by `num_cores`.
  """
  if (n - 2) % num_cores != 0:
    raise ValueError(f'The starting full grid size minus 2 {n - 2} must be '
                     f'divisble by the number of cores {num_cores}.')

  sizes = []
  coarsest_full_grid_size = (coarsest_subgrid_size - 2) * num_cores + 2
  last_n = n

  while n > num_cores + 2 and last_n > coarsest_full_grid_size:
    # Divide the d.o.f by 2.
    n = max((n - 2) // 2 + 2, num_cores + 2)
    # Adjust the size to correspond to the number of cores. `math.ceil` is
    # needed here to keep the number of non-zero entries per row in the
    # restriction matrix small enough that all the non-zero values fit in every
    # non-boundary row.
    n = int(math.ceil((n - 2) / num_cores)) * num_cores + 2
    if n < coarsest_full_grid_size:
      sizes.append((last_n, coarsest_full_grid_size))
      break
    sizes.append((last_n, n))
    last_n = n

  return sizes


def prolong_restrict_matrices_from_params(  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    params: grid_parametrization.GridParametrization,
    coarsest_subgrid_shape: Optional[Sequence[int]] = None,
    dtype: np.dtype = _NP_DTYPE
) -> ProlongRestrictMatrices:
  """Returns nested lists of prolongation and restriction matrices.

  This is similar to `prolong_restrict_matrices`. The starting subgrid shape
  `(nx, ny, nz)` is reduced down just as in that function, but the prolong and
  restrict matrices returned are the ones corresponding to the full grid at each
  level (halo width 1 is assumed). This is so that the subgrid computations in a
  parallel context match exactly those done in the single core case.

  Args:
    params: a GridParametrization instance.
    coarsest_subgrid_shape: The ending subgrid shape.
    dtype: The dtype.

  Returns:
    A tuple of (prolongation_matrices, restriction_matrices) corresponding to
    the reduction in size from the coarsest full grid shape down to the finiest
    one corresponding to `coarsest_subgrid_shape`.
  """
  if not coarsest_subgrid_shape:
    coarsest_subgrid_shape = (4, 4, 4)

  if min(coarsest_subgrid_shape) < 4:
    raise ValueError('Each element of `coarsest_subgrid_shape` '
                     f'({coarsest_subgrid_shape}) must be greater than 3.')

  if any(
      np.array((params.nx, params.ny, params.nz)) -
      np.array(coarsest_subgrid_shape)) < 0:
    raise ValueError('Each element of the subgrid shape ('
                     f'{params.nx, params.ny, params.nz}) must be '
                     'greater than or equal to `coarsest_subgrid_shape` '
                     f'({coarsest_subgrid_shape}).')

  # Currently the multigrid solver always assumes halo width of 1. When used
  # in case of halo width larger than 1, the solver is still called with an
  # adaptation step to keep only a single halo layer. So here the
  # `full_grid_shape` is calculated with a fixed halo width of 1.
  full_grid_shape = (params.core_nx * params.cx + 2,
                     params.core_ny * params.cy + 2,
                     params.core_nz * params.cz + 2)
  computation_shape = params.cx, params.cy, params.cz

  return prolong_restrict_matrices_from_shapes(full_grid_shape,
                                               coarsest_subgrid_shape,
                                               computation_shape,
                                               dtype)


def prolong_restrict_matrices_from_shapes(  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    full_grid_shape: Sequence[int],
    coarsest_subgrid_shape: Sequence[int],
    computation_shape: Sequence[int],
    dtype: np.dtype = _NP_DTYPE) -> ProlongRestrictMatrices:
  """Returns a set of prolongation and restriction matrices.

  See `prolong_restrict_matrices_from_params`.

  Args:
    full_grid_shape: The full grid shape.
    coarsest_subgrid_shape: The coasrsest subgrid shape.
    computation_shape: The computation shape.
    dtype: The dtype.

  Returns:
    A tuple of (prolongation_matrices, restriction_matrices).
  """
  full_grid_size_pairs_per_level = list(
      itertools.zip_longest(*[
          full_1d_grid_size_pairs(fg, c, csgs) for fg, c, csgs in zip(
              full_grid_shape, computation_shape, coarsest_subgrid_shape)
      ]))
  ps = []
  rs = []
  for sizes_per_dim in full_grid_size_pairs_per_level:
    pss = []
    rss = []
    for f2f1 in sizes_per_dim:
      if f2f1 is None:
        pss.append(None)
        rss.append(None)
      else:
        pss.append(prolong_matrix(*f2f1, dtype))
        rss.append(restrict_matrix(*f2f1, dtype))
    ps.append(pss)
    rs.append(rss)

  log_full_grid_shapes_from_ps(ps)

  return (ps, rs)


def convert_ps_rs_dict_to_tuple(
    psd: Mapping[Text, tf.Tensor],
    rsd: Mapping[Text, tf.Tensor]) -> ProlongRestrictMatrices:
  """Convert maps of prolongation / restriction matrices to lists.

  The maps have keys of the form '0_1' where the first number is the multigrid
  level and the second number is the dimension.

  Args:
    psd: A map from a level/dim key to the prolong matrix.
    rsd: A map from a level/dim key to the restrict matrix.

  Returns:
    Nested lists of prolong/restrict matrices as described in
    `prolong_restrict_matrices`.
  """
  ps = []
  rs = []
  level = 0
  while f'{level}_0' in psd or f'{level}_1' in psd or f'{level}_2' in psd:
    pss = []
    rss = []
    for dim in range(3):
      key = f'{level}_{dim}'
      pss.append(psd[key] if key in psd else None)
      rss.append(rsd[key] if key in rsd else None)

    ps.append(pss)
    rs.append(rss)

    level += 1

  return (ps, rs)


def get_ps_rs_init_fn(params: grid_parametrization.GridParametrization,  # pytype: disable=annotation-type-mismatch  # numpy-scalars
                      coarsest_subgrid_shape: Optional[Sequence[int]] = None,
                      dtype: np.dtype = _NP_DTYPE):
  """Returns an init function for prolongation and restriction matrices.

  The returned function, when called, returns a dictionary with keys `ps` and
  `rs`. Each value is itself a dictionary with keys of the form `(i, j)` where
  the `i` is the multigrid level and `j` is the dimension. The values for those
  keys are the parts of full-grid prolongation and restriction matrices
  appropriate for each core in a multicore context (halo width 1 is assumed).

  Args:
    params: A `GridParametrization` instance.
    coarsest_subgrid_shape: The coarsest subgrid shape.
    dtype: The dtype.

  Returns:
    A function that returns a dictionary as described above for the given
    coordinates.
  """
  computation_shape = params.cx, params.cy, params.cz
  if not coarsest_subgrid_shape:
    coarsest_subgrid_shape = (4, 4, 4)
  ps_per_level, rs_per_level = prolong_restrict_matrices_from_params(
      params, coarsest_subgrid_shape, dtype)

  get_subgrid_size = lambda s, c: int((s - 2) // c + 2)

  def ps_rs_init_fn(coordinates: Union[np.ndarray, Tuple[int, int, int]]):
    ps, rs = {}, {}
    for level, (ps_per_dim,
                rs_per_dim) in enumerate(zip(ps_per_level, rs_per_level)):
      for dim in range(3):
        p, r = ps_per_dim[dim], rs_per_dim[dim]
        if p is not None:
          key = f'{level}_{dim}'
          subgrid_size = get_subgrid_size(p.shape[0], computation_shape[dim])
          dim_0_start, dim_0_end = initializer.subgrid_slice_indices(
              subgrid_size, coordinates[dim]
          )
          subgrid_size = get_subgrid_size(p.shape[1], computation_shape[dim])
          dim_1_start, dim_1_end = initializer.subgrid_slice_indices(
              subgrid_size, coordinates[dim]
          )
          ps[key] = tf.slice(
              p,
              [dim_0_start, dim_1_start],
              [dim_0_end - dim_0_start, dim_1_end - dim_1_start],
          )
          rs[key] = tf.slice(
              r,
              [dim_1_start, dim_0_start],
              [dim_1_end - dim_1_start, dim_0_end - dim_0_start],
          )

    return {'ps': ps, 'rs': rs}

  return ps_rs_init_fn


def get_full_grids_init_fn(
    x: TensorOrArray,
    b: TensorOrArray,
    params: grid_parametrization.GridParametrization,
    coarsest_subgrid_shape: Optional[Sequence[int]] = None,
    boundary_conditions: Optional[BoundaryConditionsSpec] = None,
    a_operator: Optional[ModifyTensorOrArrayFn] = None) -> InitFn:
  """An multigrid init function with full grid inputs.

  Args:
    x: The initial full grid of the field being solved for.
    b: The initial full grid source field.
    params: The grid parametrization.
    coarsest_subgrid_shape: The coarsest subgrid shape.
    boundary_conditions: The boundary conditions specification.
    a_operator: A function which returns the operator `A` acting on the input
      tensor or array.

  Returns:
    The init function.
  """
  x0 = tf.zeros_like(x) if isinstance(x, tf.Tensor) else np.zeros_like(x)
  apply_boundary_conditions_fn = get_apply_one_core_boundary_conditions_fn(
      boundary_conditions)
  xb = apply_boundary_conditions_fn(x) if apply_boundary_conditions_fn else x
  full_grid_shape = get_shape(x)
  rank = len(full_grid_shape)
  full_grid_lengths = (params.lx, params.ly, params.lz)[:rank]
  if a_operator is None:
    a_operator = laplacian_and_inv_diagonal_fns(
        full_grid_shape, full_grid_lengths)[0]
  b_minus_a_xb = b - a_operator(xb)

  def subgrid_of_3d_grid(full_3d_grid, coordinates):
    """Retrieves the subgrid in the partition specified by `coordinates`."""
    if params.cx * params.cy * params.cz == 1:
      return full_3d_grid

    # Note that halo_width is assumed to be 1 in the current multigrid solver
    # implementation.
    halo_width = 1

    return initializer.subgrid_of_3d_grid(
        full_3d_grid, (params.nx, params.ny, params.nz), coordinates, halo_width
    )

  def init_fn(replica_id, coordinates):
    dtype = x.dtype.as_numpy_dtype if isinstance(x, tf.Tensor) else x.dtype
    ps_rs_init_fn = get_ps_rs_init_fn(params, coarsest_subgrid_shape, dtype)
    state = ps_rs_init_fn(coordinates)
    state.update({
        'replica_id': replica_id,
        'coordinates': coordinates,
        'x0': subgrid_of_3d_grid(x0, coordinates),
        'xb': subgrid_of_3d_grid(xb, coordinates),
        'b_minus_a_xb': subgrid_of_3d_grid(b_minus_a_xb, coordinates),
    })

    return state

  return init_fn


def get_init_fn_from_value_fn_for_homogeneous_bcs(
    b_fn: ValueFunction,
    params: grid_parametrization.GridParametrization,
    coarsest_subgrid_shape: Optional[Sequence[int]] = None,
    perm: ThreeIntTuple = initializer.DEFAULT_PERMUTATION) -> InitFn:
  """An multigrid init function with value function input for `b`.

  This function is for the case of homogeneous boundary conditions.

  Args:
    b_fn: The initializer function for `b`.
    params: The grid parametrization.
    coarsest_subgrid_shape: The coarsest subgrid shape.
    perm: A 3-tuple that defines the permutation ordering for the returned
      tensor. The default is `(2, 0, 1)`. If `None`, permutation is not applied.

  Returns:
    The init function.
  """
  def init_fn(replica_id, coordinates):
    b = initializer.partial_mesh_for_core(
        params, coordinates, b_fn, perm=perm)
    x0 = tf.zeros_like(b)
    dtype = b.dtype.as_numpy_dtype
    ps_rs_init_fn = get_ps_rs_init_fn(params, coarsest_subgrid_shape, dtype)
    state = ps_rs_init_fn(coordinates)

    state.update({
        'replica_id': replica_id,
        'coordinates': coordinates,
        'x0': x0,
        # `xb` is zero in this homogeneous case, so `b_minus_a_xb` is `b`.
        'b_minus_a_xb': b
    })

    return state

  return init_fn


def kronecker_einsum_indices(n: int, i: int) -> Text:
  """Returns an einsum string.

  Returns an einsum string appropriate for doing a Kronecker product of a 2d
  matrix into a multidimensional tensor. The string corresponds to matrix
  multiplying a 2d tensor into the `i-th` axis of a tensor.

  It is required that `0 < n < 25` and `0 <= i < n`.

  Examples:
    kronecker_einsum_indices(0, 3) = 'zy,ybc->zbc'
    kronecker_einsum_indices(1, 3) = 'zy,ayc->azc'

  Args:
    n: The number of columns of the 2d matrix.
    i: The axis to multiply the matrix into.

  Returns:
    The einsum string.
  """
  if n > 24 or i >= n or n < 1 or i < 0:
    raise ValueError('It is required that 0 < n < 25 and 0 <= i < n. Got '
                     'n = {}, i = {}.'.format(n, i))
  a_to_n = ''
  for j in range(n):
    a_to_n += chr(ord('a') + j)
  a_to_n_0 = a_to_n[:i] + 'y' + a_to_n[i+1:]
  a_to_n_1 = a_to_n[:i] + 'z' + a_to_n[i+1:]
  return 'zy,' + a_to_n_0 + '->' + a_to_n_1


def kronecker_products(ms: Sequence[Optional[TensorOrArray]],
                       x: TensorOrArray) -> TensorOrArray:
  """Returns the application of the Kronecker product of matrices `ms` on `x`.

  Each matrix in `ms` corresponds to each dimension of `x`. If a matrix in `ms`
  is `None`, nothing is done in the corresponding dimension.

  Args:
    ms: A sequence of 2d matrices, one for each dimension.
    x: The input tensor or array.

  Returns:
    The input tensor or array after doing Kronecker products of each `non-None`
    matrix in `ms`.
  """
  use_tf = isinstance(ms[0], tf.Tensor) or isinstance(x, tf.Tensor)
  einsum = tf.einsum if use_tf else np.einsum
  rank = len(ms)
  for i, m in enumerate(ms):
    if m is not None:
      indices = kronecker_einsum_indices(rank, i)
      x = einsum(indices, m, x)
  return x


def laplacian_matrix(  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    shape: Sequence[int],
    grid_lengths: Optional[Sequence[float]] = None,
    boundary_conditions: Optional[BoundaryConditionsSpec] = None,
    dtype: np.dtype = _NP_DTYPE) -> np.ndarray:
  """Returns the Laplacian 2D matrix for a vector of the given shape.

  Only Dirichlet and Neumann boundary conditions are supported. If boundary
  conditions are not supplied, `(Dirichlet, 0)` conditions are used. In order to
  not apply any boundary conditions, don't provide `boundary_conditions` (or use
  `None`).

  Args:
    shape: The grid shape.
    grid_lengths: The grid lengths (defaults to `(1,) * rank`).
    boundary_conditions: The boundary conditions.
    dtype: The numpy dtype.
  """
  allowed_bc_types = (BCType.DIRICHLET, BCType.NEUMANN)

  rank = len(shape)
  boundary_conditions = (
      boundary_conditions or halo_exchange_utils.homogeneous_bcs(rank))

  def bc_types_are_supported() -> bool:
    if not boundary_conditions: return True

    for bc in boundary_conditions:
      if not bc: continue
      bc_types = [b[0] for b in bc]
      if any([t not in allowed_bc_types for t in bc_types]):
        return False

    return True

  if not bc_types_are_supported():
    raise ValueError('Only certain boundary condition types are supported '
                     f'({allowed_bc_types}).')

  if not grid_lengths:
    grid_lengths = (1,) * rank
  dxs = [length / (s - 1) for length, s in zip(grid_lengths, shape)]

  aa = [(np.eye(n, k=-1) - 2 * np.eye(n) + np.eye(n, k=1)).astype(dtype)
        / dx ** 2 for n, dx in zip(shape, dxs)]

  a_dim = np.prod(shape)
  a = np.zeros((a_dim, a_dim)).astype(dtype)

  for i, b in enumerate(aa):
    a_nd = None
    for j in range(rank):
      a_or_eye = b if j == i else np.eye(shape[j])
      a_nd = a_or_eye if a_nd is None else np.kron(a_nd, a_or_eye)
    a += a_nd

  if not boundary_conditions: return a

  dim_stride = lambda dim, shape: int(np.prod(shape[dim + 1:]))

  def indices_for_row(n: int, shape: Sequence[int]) -> Sequence[int]:
    indices = []
    for dim in range(len(shape)):
      divisor = dim_stride(dim, shape)
      indices.append((n // divisor) % shape[dim])
    return indices

  def border_row_types(n: int) -> Sequence[SideType]:
    lo_indices = indices_for_row(n, shape)
    hi_indices = [(s - 1) - i for i, s in zip(lo_indices, shape)]
    sides = []
    for lo_index, hi_index in zip(lo_indices, hi_indices):
      sides.append(SideType.LOW if lo_index == 0
                   else SideType.HIGH if hi_index == 0
                   else SideType.NONE)
    return sides

  # Replace the border rows per the boundary conditions.
  for n in range(a_dim):
    for dim, border_row_type in enumerate(border_row_types(n)):
      if not boundary_conditions[dim]: continue
      if border_row_type == SideType.NONE: continue
      a[n, :] = 0
      if ((border_row_type == SideType.LOW and
           boundary_conditions[dim][0][0] == BCType.DIRICHLET) or
          (border_row_type == SideType.HIGH and
           boundary_conditions[dim][1][0] == BCType.DIRICHLET)):  # DIRICHLET
        a[n, n] = 1
      elif border_row_type == SideType.LOW:  # NEUMANN LOW.
        a[n, n] = -1
        stride = dim_stride(dim, shape)
        a[n, n + stride] = 1
      else:  # NEUMANN HIGH.
        a[n, n] = 1
        stride = dim_stride(dim, shape)
        a[n, n - stride] = -1

  return a


def inverse_laplacian_matrix(  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    shape: Sequence[int],
    grid_lengths: Optional[Sequence[float]] = None,
    boundary_conditions: Optional[BoundaryConditionsSpec] = None,
    dtype: np.dtype = _NP_DTYPE,
    use_pinv: bool = False) -> np.ndarray:
  """Returns the inverse Laplacian matrix for the given boundary conditions.

  Args:
    shape: The grid shape.
    grid_lengths: The grid lengths (the default value is 1).
    boundary_conditions: The boundary connditins (the default values are
      Dirichlet-0 values).
    dtype: The dtype (the default value is float32).
    use_pinv: The pseudo-inverse needs to be used in the all-Neumann case. It is
      not normally used in other cases. This argument is used by unit tests to
      demonstrate that the pseudo-inverse can be less accurate than `inv`.

  Returns:
    The inverse of the Laplacian matrix. In the all-Neumann case, or if
    `use_pinv` is `True`, the pseudo-inverse is returned.
  """
  lap = laplacian_matrix(shape, grid_lengths, boundary_conditions, dtype)
  # `pinv` needs to be used in the all-Neumann case. In all other cases, `inv`
  # is used. It can be more accurate.
  if use_pinv or boundary_conditions_all_neumann(boundary_conditions):
    return sp.linalg.pinv(lap)
  else:
    return sp.linalg.inv(lap)


def solve(a: TensorOrArray, b: TensorOrArray) -> TensorOrArray:
  """Returns `x` using `scipy.linalg.solve` for `a x = b`."""
  shape = get_shape(b)
  reshape, solve_fn = ((np.reshape, sp.linalg.solve)
                       if isinstance(b, np.ndarray)
                       else (tf.reshape, tf.linalg.solve))
  return reshape(solve_fn(a, reshape(b, [-1, 1])), shape)


def matmul(a: TensorOrArray, x: TensorOrArray) -> TensorOrArray:
  """Returns `a @ x.ravel()` in the shape of `x`."""
  if isinstance(x, np.ndarray):
    return np.reshape(a @ x.ravel(), x.shape)
  return tf.reshape(a @ tf.reshape(x, [-1, 1]), x.shape)


def halo_exchange_step_fn(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    boundary_conditions: Optional[BoundaryConditionsSpec] = None,
    halo_width: int = 1) -> Optional[Callable[[FlowFieldVal], FlowFieldVal]]:
  """Returns a halo exchange function for the given boundary conditions."""
  if boundary_conditions is None:
    return None

  subtract_mean = boundary_conditions_all_neumann(boundary_conditions)

  def halo_exchange_fn(x: FlowFieldVal) -> FlowFieldVal:
    is_tensor = isinstance(x, tf.Tensor)
    shape = get_shape(x) if is_tensor else (*get_shape(x[0]), len(x))
    tiles = tf.unstack(x, axis=-1) if is_tensor else x
    tiles = halo_exchange.inplace_halo_exchange(
        tiles,
        dims=(0, 1, 2),
        replica_id=replica_id,
        replicas=replicas,
        replica_dims=(0, 1, 2),
        boundary_conditions=boundary_conditions)
    if subtract_mean:
      inner = (slice(halo_width, -halo_width),) * 3
      full_grid_shape = get_full_grid_shape(shape, replicas.shape, halo_width)
      num_dof = np.prod([s - halo_width * 2 for s in full_grid_shape])
      single_tensor = tf.stack(tiles)[inner]
      avg = tf.math.reduce_sum(
          tf.compat.v1.tpu.cross_replica_sum(single_tensor)) / num_dof
      tiles = [t - avg for t in tiles]

    return tf.stack(tiles, axis=-1) if is_tensor else tiles

  return halo_exchange_fn


# Helper functions for coupling the multigrid solver with Swirl-LM.
def flatten_dict_with_prefix(
    data: Mapping[str, tf.Tensor],
    prefix: str,
) -> Mapping[str, tf.Tensor]:
  """Adds a prefix to all keys in `data`.

  Args:
    data: A dictionary of `tf.Tensor`.
    prefix: The prefix to be added in front of each key in `data`.

  Returns:
    A dictionary that is identical to `data`, but with its keys being
    `{prefix}-{original_key_in_data}`.
  """
  return {f'{prefix}-{key}': val for key, val in data.items()}


def remove_prefix_in_dict(
    data: Mapping[str, tf.Tensor],
    prefix: str,
) -> Mapping[str, tf.Tensor]:
  """Removes prefix from keys in `data`.

  Args:
    data: A dictionary of tf.Tensor.
    prefix: The prefix to be removed from keys in `data`.

  Returns:
    A dictionary that contains all items that starts with `{prefix}-` in data.
    Matching items with keys specified as `{prefix}-{key}` will be rekeyed as
    `key`.
  """
  res = {}
  for maybe_prefixed_key, val in data.items():
    key = re.split(f'{prefix}-', maybe_prefixed_key)
    if len(key) == 1:
      continue

    res[key[-1]] = val

  return res


def maybe_get_multigrid_params(
    params: parameters_lib.SwirlLMParameters,
) -> Optional[poisson_solver_pb2.PoissonSolver.Multigrid]:
  """Retrieves the parameters for the multigrid solver.

  Args:
    params: An instance of the `SwirlLMParameters` specifying a simulation
      configuration.

  Returns:
    Parameters of the multigrid solver if defined in the simulation config.
    Otherwise returns `None`.
  """
  if (
      params.pressure is None
      or params.pressure.solver.WhichOneof('solver') != 'multigrid'
  ):
    return None

  return params.pressure.solver.multigrid


def get_multigrid_helper_var_keys(
    params: parameters_lib.SwirlLMParameters,
) -> List[str]:
  """Generates a list of helper variable names for the MG solver.

  Args:
    params: An instance of the `SwirlLMParameters` specifying a simulation
      configuration.

  Returns:
    A list of string specifying the names of all helper variables required by
    the multigrid solver.
  """
  mg_params = maybe_get_multigrid_params(params)

  assert mg_params is not None, (
      '`multigrid` parameters are not defined in the config while the multigrid'
      ' solver is used.'
  )

  # Currently the multigrid solver always assumes halo width of 1. When used
  # in case of halo width larger than 1, the solver is still called with an
  # adaptation step to keep only a single halo layer. So here the
  # `full_grid_shape` is calculated with a fixed halo width of 1.
  full_grid_shape = (
      params.core_nx * params.cx + 2,
      params.core_ny * params.cy + 2,
      params.core_nz * params.cz + 2,
  )
  computation_shape = params.cx, params.cy, params.cz
  coarsest_subgrid_shape = (
      mg_params.coarsest_subgrid_shape.dim_0,
      mg_params.coarsest_subgrid_shape.dim_1,
      mg_params.coarsest_subgrid_shape.dim_2,
  )

  full_grid_size_pairs_per_level = list(
      itertools.zip_longest(
          *[
              full_1d_grid_size_pairs(fg, c, csgs)
              for fg, c, csgs in zip(
                  full_grid_shape, computation_shape, coarsest_subgrid_shape
              )
          ]
      )
  )
  return list(
      f'{varname}-{level}_{dim}'  # pylint: disable=g-complex-comprehension
      for varname, level, dim in itertools.product(
          HELPER_VARIABLES, range(len(full_grid_size_pairs_per_level)), range(3)
      )
      if full_grid_size_pairs_per_level[level][dim] is not None
  )


def get_multigrid_helper_var_init_fn(
    params: parameters_lib.SwirlLMParameters,
) -> types.InitFn:
  """Generates a function that initializes helper variables for the MG solver.

  Args:
    params: An instance of the `SwirlLMParameters` specifying a simulation
      configuration.

  Returns:
    A function that initializes the helper variables, i.e. 'ps' and 'rs', that
    are required by the multigrid solver. Note that all subfields of `ps` and
    `rs` are appended with the prefix 'ps' or 'rs', respectively.
  """
  mg_params = maybe_get_multigrid_params(params)

  assert mg_params is not None, (
      '`multigrid` parameters are not defined in the config while the multigrid'
      ' solver is used.'
  )

  coarsest_subgrid_shape = (
      mg_params.coarsest_subgrid_shape.dim_0,
      mg_params.coarsest_subgrid_shape.dim_1,
      mg_params.coarsest_subgrid_shape.dim_2,
  )

  ps_rs_init_fn = get_ps_rs_init_fn(
      params, coarsest_subgrid_shape=coarsest_subgrid_shape
  )

  def multigrid_helper_var_init_fn(
      replica_id: tf.Tensor,
      coordinates: types.ReplicaCoordinates,
  ) -> types.FlowFieldMap:
    """Fits the `ps_rs_init_fn` to the `init_fn` signature."""
    del replica_id
    ps_rs = ps_rs_init_fn(coordinates)

    helper_var = {}
    for prefix, val in ps_rs.items():
      helper_var.update(flatten_dict_with_prefix(val, prefix))

    return helper_var

  return multigrid_helper_var_init_fn
