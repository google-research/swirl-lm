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
"""Library of multigrid utility functions that operate on lists of 2D inputs.

Some functions are compatible with numpy and TensorFlow (that is, they can take
one or more lists of numpy array(s) as input and return a list of numpy array,
or same with `tf.Tensor`), and some are TensorFlow-only. The former return
`Tiles`, while TensorFlow-only functions return `List[tf.Tensor]`.
"""
import functools
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from swirl_lm.communication import halo_exchange_utils
from swirl_lm.linalg import multigrid_utils
from swirl_lm.utility import grid_parametrization
import tensorflow as tf

BoundaryConditionsSpec = halo_exchange_utils.BoundaryConditionsSpec
TensorOrArray = multigrid_utils.TensorOrArray
HaloExchangeFn = Callable[[List[tf.Tensor]], List[tf.Tensor]]
ModifyTensorOrArrayFn = Callable[[List[TensorOrArray]], List[TensorOrArray]]
Tiles = Union[List[tf.Tensor], List[np.ndarray]]
SmootherTypeFn = Callable[[Tiles, Tiles], Tiles]


def get_shape(x: Tiles) -> Tuple[int, int, int]:
  """Returns the shape of a list of tensors or arrays."""
  shape_dim_2 = len(x)
  shape_dims_0_1 = x[0].shape
  if not isinstance(shape_dims_0_1, tuple):
    shape_dims_0_1 = shape_dims_0_1.as_list()
  return (*shape_dims_0_1, shape_dim_2)


def stack(x: Tiles, axis: int = -1) -> TensorOrArray:
  """Stacks the list of tensors or arrays `x` along the given axis."""
  stack_fn = tf.stack if isinstance(x[0], tf.Tensor) else np.stack
  return stack_fn(x, axis)


def unstack(x: TensorOrArray, axis: int = -1) -> Tiles:
  """Unstacks the tensor or array `x` along the given axis."""
  if isinstance(x, tf.Tensor):
    return tf.unstack(x, axis=axis)
  return [np.squeeze(y, axis) for y in np.split(x, x.shape[axis], axis=axis)]


def zero_borders(x: Tiles) -> Tiles:
  """Zeros out the borders of the given tiles (width 1)."""
  slices_2d = (slice(1, -1), slice(1, -1))
  padding = ((1, 1), (1, 1))
  if isinstance(x[0], tf.Tensor):
    zero_tile = tf.zeros_like(x[0])
    return ([zero_tile] +
            [tf.pad(x_2d[slices_2d], paddings=padding) for x_2d in x[1:-1]] +
            [zero_tile])

  zero_tile = np.zeros_like(x[0])
  return ([zero_tile] +
          [np.pad(x_2d[slices_2d], padding, mode='constant')
           for x_2d in x[1:-1]] +
          [zero_tile])


def jacobi(x: Tiles,
           b: Tiles,
           a_operator: Callable[[Tiles], Tiles],
           a_inv_diagonal: Callable[[Tiles], Tiles],
           n: int,
           weight: float = 1,
           halo_exchange_fn: Optional[HaloExchangeFn] = None,
           all_no_touch_boundary_conditions: bool = False) -> Tiles:
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

  In the case of one core and Dirichlet boundary conditions, `halo_exchange_fn`
  is not needed.

  Args:
    x: The initial values of `x` as a list of 2D tensors or arrays.
    b: The source as a list of 2D tensors or arrays.
    a_operator: A function which returns the operator `A` acting on the input
      tiles.
    a_inv_diagonal: A function which returns the inverse of the diagonal of `A`
      times the input tiles.
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
  do_not_change_borders = (
      (halo_exchange_fn is None) or all_no_touch_boundary_conditions)
  if do_not_change_borders:
    inner_xy = (slice(1, -1),) * 2
    if isinstance(x[0], tf.Tensor):
      mask_xy = tf.pad(tf.ones_like(x[0][inner_xy]), paddings=((1, 1),) * 2)
    else:
      mask_xy = np.pad(np.ones_like(x[0][inner_xy]), ((1, 1),) * 2, 'constant')
  else:
    mask_xy = 1

  def body(x, i):
    update = [
        weight * mask_xy * a_inv_diag for a_inv_diag in a_inv_diagonal(
            [b_ - a_x for b_, a_x in zip(b, a_operator(x))])
    ]
    if do_not_change_borders:
      x = ([x[0]] + [x_ + u for x_, u in zip(x[1:-1], update[1:-1])] +
           [x[-1]])
    else:
      x = [x_ + u for x_, u in zip(x, update)]

    if halo_exchange_fn:
      x = halo_exchange_fn(x)

    return x, i + 1

  i = 0
  if isinstance(x[0], tf.Tensor):
    cond = lambda _, i: i < n
    x, _ = tf.while_loop(
        cond=cond, body=body, loop_vars=[x, i], back_prop=False)
  else:
    while i < n:
      x, i = body(x, i)

  return x


def laplacian_and_inv_diagonal_fns(
    shape: Sequence[int],
    grid_lengths: Optional[Sequence[float]] = None
) -> Tuple[Callable[[Tiles], Tiles], Callable[[Tiles], Tiles]]:
  """Returns the Laplacian function and its inverse diagonal function.

  The functions return tiles of the same shape as the input 2D tiles. The
  boundary values are not meaningful and should be masked.

  Args:
    shape: The shape of the tensor or array (length 3).
    grid_lengths: An optional sequence of grid lengths. If not given, 1 is used.

  Returns:
    A tuple of two functions: the Laplacian and its inverse-diagonal
    functions to second-order. The border values returned by the functions are
    not meaningful.

  Raises:
    `ValueError` in case the number of elements of `shape` or `grid_lengths` is
    not equal to 3.
  """
  rank = 3

  if len(shape) != rank:
    raise ValueError(f'The length of shape ({len(shape)}) must be {rank}.')

  if grid_lengths and len(grid_lengths) != rank:
    raise ValueError(f'The length of grid_lengths ({len(grid_lengths)}) must '
                     f'be {rank}.')

  grid_lengths = grid_lengths or (1,) * rank

  # `s` includes the 1 halo pixel on each of the 2 ends. To be consistent with
  # the grid parametrization where the inner grids represent the nodes of the
  # interior domain that spans the `grid_lengths`, the grid space should be
  # computed with the 2 side halo nodes removed.
  dxs = [length / (s - 3) for length, s in zip(grid_lengths, shape)]
  diag = -2 * sum([dx**-2 for dx in dxs])

  def laplacian(x: Tiles) -> Tiles:
    roll = tf.roll if isinstance(x[0], tf.Tensor) else np.roll
    lap = [diag * x_ for x_ in x]
    for i in range(len(x)):
      for jk in range(2):
        lap[i] += (
            (roll(x[i], shift=1, axis=jk) + roll(x[i], shift=-1, axis=jk)) /
            dxs[jk]**2)
      if i > 0:
        lap[i] += x[i - 1] / dxs[2]**2
      if i < len(x) - 1:
        lap[i] += x[i + 1] / dxs[2]**2
    return lap

  def inv_diagonal(x: Tiles) -> Tiles:
    return [x_ / diag for x_ in x]

  return laplacian, inv_diagonal


def laplacian_inv_diagonal_fn(
    shape: Sequence[int],
    grid_lengths: Optional[Sequence[float]] = None) -> Callable[[Tiles], Tiles]:
  _, inv_diagonal = laplacian_and_inv_diagonal_fns(shape, grid_lengths)
  return inv_diagonal


def poisson_jacobi(
    x: Tiles,
    b: Tiles,
    params: Optional[grid_parametrization.GridParametrization] = None,
    n: int = 1,
    weight: float = 1,
    halo_exchange_fn: Optional[HaloExchangeFn] = None,
    all_no_touch_boundary_conditions: bool = False) -> Tiles:
  """Returns the Jacobi method applied to Poisson's equation (see `jacobi`)."""
  subgrid_shape = get_shape(x)
  if params:
    full_grid_lengths = (params.lx, params.ly, params.lz)
    computation_shape = (params.cx, params.cy, params.cz)
  else:
    full_grid_lengths = None
    computation_shape = None

  full_grid_shape = multigrid_utils.get_full_grid_shape(subgrid_shape,
                                                        computation_shape)
  laplacian, lap_inv_diagonal = (
      laplacian_and_inv_diagonal_fns(full_grid_shape, full_grid_lengths))

  return jacobi(x, b, laplacian, lap_inv_diagonal, n, weight, halo_exchange_fn,
                all_no_touch_boundary_conditions)


def poisson_jacobi_fn_for_one_core(
    params: Optional[grid_parametrization.GridParametrization] = None,
    n: int = 1,
    weight: float = 1,
    halo_exchange_fn: Optional[HaloExchangeFn] = None
) -> SmootherTypeFn:
  """Returns a one-core Jacobi fn for Poisson's equation (see `jacobi`)."""
  return functools.partial(poisson_jacobi, params=params, n=n, weight=weight,
                           halo_exchange_fn=halo_exchange_fn)


def poisson_jacobi_step_fn(
    params: grid_parametrization.GridParametrization,
    n: int,
    weight: float = 1,
    boundary_conditions: Optional[BoundaryConditionsSpec] = None
) -> Callable[[tf.Tensor, np.ndarray], SmootherTypeFn]:
  """Returns a step fn for Poisson Jacobi (see `jacobi`)."""
  all_no_touch = multigrid_utils.boundary_conditions_all_no_touch(
      boundary_conditions)

  def step_fn(replica_id: tf.Tensor, replicas: np.ndarray):
    halo_exchange_fn = multigrid_utils.halo_exchange_step_fn(
        replica_id, replicas, boundary_conditions)

    return functools.partial(
        poisson_jacobi, params=params, n=n, weight=weight,
        halo_exchange_fn=halo_exchange_fn,
        all_no_touch_boundary_conditions=all_no_touch)

  return step_fn


def poisson_residual(
    x: Tiles,
    b: Tiles,
    params: Optional[grid_parametrization.GridParametrization] = None) -> Tiles:
  """Returns the Poisson residual.

  Returns the residual `b - A x` for the Poisson equation `A x = b`.

  Args:
    x: The input tiles.
    b: The source tiles.
    params: The grid parametrization.

  Returns:
    The residual tiles.
  """
  subgrid_shape = get_shape(x)

  if params:
    full_grid_lengths = (params.lx, params.ly, params.lz)
    computation_shape = (params.cx, params.cy, params.cz)
  else:
    full_grid_lengths = None
    computation_shape = None

  full_grid_shape = multigrid_utils.get_full_grid_shape(subgrid_shape,
                                                        computation_shape)

  laplacian, _ = laplacian_and_inv_diagonal_fns(full_grid_shape,
                                                full_grid_lengths)

  return [b_ - lap for b_, lap in zip(b, laplacian(x))]


def poisson_residual_norm(
    x: Tiles,
    b: Tiles,
    params: Optional[grid_parametrization.GridParametrization] = None) -> float:
  """Returns the Poisson residual norm (float)."""
  inner = (slice(1, -1),) * 2
  res_no_border = [r[inner]
                   for r in poisson_residual(x, b, params)][1:-1]
  norm = tf.norm if isinstance(x[0], tf.Tensor) else np.linalg.norm

  return norm(stack(res_no_border, axis=0))


def kronecker_products(ms: Sequence[TensorOrArray], x: Tiles) -> Tiles:
  """Returns the application of the Kronecker product of matrices `ms` on `x`.

  Each matrix in `ms` corresponds to each dimension of `x`. If a matrix in `ms`
  is `None`, nothing is done in the corresponding dimension.

  Args:
    ms: A sequence of 2D matrices, one for each dimension.
    x: The input tiles.

  Returns:
    The input tiles after doing Kronecker products of each `non-None`
    matrix in `ms`.
  """
  einsum = tf.einsum if isinstance(x[0], tf.Tensor) else np.einsum

  # First do dims 0 and 1.
  for i, m in enumerate(ms[:2]):
    if m is not None:
      indices = multigrid_utils.kronecker_einsum_indices(n=2, i=i)
      x = [einsum(indices, m, x_) for x_ in x]

  # Then dim 2 (the list dimension).
  mz = ms[2]
  if mz is None:
    return x
  res = []
  for i in range(mz.shape[0]):
    ith_plane = 0.
    for j in range(mz.shape[1]):
      try:
        mz_i_j_non_0 = bool(mz[i, j] != 0.)
      except tf.errors.OperatorNotAllowedInGraphError:
        # In TF graph mode, the tensor is not immediately available.
        mz_i_j_non_0 = True
      if mz_i_j_non_0:
        ith_plane += mz[i, j] * x[j]
    res.append(ith_plane)
  return res  # pytype: disable=bad-return-type


def get_apply_one_core_boundary_conditions_fn(
    boundary_conditions: Optional[BoundaryConditionsSpec] = None,
    homogeneous: bool = False
) -> Optional[ModifyTensorOrArrayFn]:
  """Returns a function that applies boundary conditions.

  Args:
    boundary_conditions: The boundary conditions.
    homogeneous: If true, the returned function applies homogeneous boundary
      conditions.

  Returns:
    A function that returns a given tensor or array with the boundary
      conditions applied. Or returns `None` if `boundary_conditions` is `None`.
  """
  if boundary_conditions is None:
    return None

  if homogeneous:
    boundary_conditions = (
        multigrid_utils.get_homogeneous_boundary_conditions(
            boundary_conditions))
  subtract_mean = (
      multigrid_utils.boundary_conditions_all_neumann(boundary_conditions))

  def apply_boundary_conditions(x: List[TensorOrArray]) -> List[TensorOrArray]:
    use_tf = isinstance(x[0], tf.Tensor)
    x = (halo_exchange_utils.
         apply_one_core_boundary_conditions(x, boundary_conditions))
    if subtract_mean:
      inner = (slice(1, -1), slice(1, -1))
      inner_total = sum([
          tf.math.reduce_sum(x_2d[inner]) if use_tf else np.sum(x_2d[inner])
          for x_2d in x[1:-1]
      ])
      inner_dof = np.prod([s - 2 for s in get_shape(x)])
      avg = inner_total / inner_dof
      x = [x_2d - avg for x_2d in x]

    return x

  return apply_boundary_conditions
