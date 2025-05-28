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
"""Multigrid library using n-dimensional tensors."""

import functools
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from swirl_lm.base import initializer
from swirl_lm.communication import halo_exchange_utils
from swirl_lm.linalg import multigrid_utils
from swirl_lm.utility import common_ops
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import tpu_util
import tensorflow as tf

BoundaryConditionsSpec = halo_exchange_utils.BoundaryConditionsSpec
TensorOrArray = multigrid_utils.TensorOrArray
ModifyTensorOrArrayFn = multigrid_utils.ModifyTensorOrArrayFn
ProlongRestrictMatrices = multigrid_utils.ProlongRestrictMatrices
SmootherTypeFn = multigrid_utils.SmootherTypeFn


def _mg_cycle_internal(
    x: TensorOrArray,
    b: TensorOrArray,
    prs: ProlongRestrictMatrices,
    homogeneous_smoother_fn: SmootherTypeFn,
    residual_fn: SmootherTypeFn,
    dirichlet0_halo_exchange_fn: ModifyTensorOrArrayFn,
    a_inv_for_coarsest_level: Optional[np.ndarray] = None,
    n_coarse: int = 1,
    replica_id: Union[int, tf.Tensor] = 0,
    replicas: Optional[np.ndarray] = None,
    coordinates: Optional[TensorOrArray] = None,
    level: int = 0) -> TensorOrArray:
  """The (internal) implementation of multigrid cycle. See `mg_cycle`."""
  def recurse(x, b):
    return _mg_cycle_internal(
        x, b, prs, homogeneous_smoother_fn, residual_fn,
        dirichlet0_halo_exchange_fn, a_inv_for_coarsest_level, n_coarse,
        replica_id, replicas, coordinates, level + 1)

  subgrid_shape = multigrid_utils.get_shape(x)
  replicas = replicas if replicas is not None else np.array([[[0]]])
  num_replicas = np.prod(replicas.shape)

  if level == len(prs[0]):
    if a_inv_for_coarsest_level is None:
      return homogeneous_smoother_fn(x, b)

    # Both cases below deal with the global residual, and apply zero borders to
    # it.
    if num_replicas == 1:
      return multigrid_utils.matmul(a_inv_for_coarsest_level,
                                    multigrid_utils.zero_borders(b))
    else:
      # Combine the `b` subgrids to get the full `b`. Multiply by `A_inv`, then
      # take the subgrid corresponding to this core.
      b_subgrids = common_ops.cross_replica_gather(b, num_replicas)
      b_full_grid = tpu_util.combine_subgrids(b_subgrids, replicas)
      x_full_grid = multigrid_utils.matmul(
          a_inv_for_coarsest_level, multigrid_utils.zero_borders(b_full_grid))
      return initializer.subgrid_of_3d_tensor(x_full_grid, subgrid_shape,
                                              coordinates)

  x = homogeneous_smoother_fn(x, b)

  pss, rss = prs
  ps = pss[level]
  rs = rss[level]

  def body(i, x, err_c):
    res = dirichlet0_halo_exchange_fn(residual_fn(x, b))
    res_c = multigrid_utils.kronecker_products(rs, res)  # Restrict.
    err_c = recurse(err_c, res_c)
    x += multigrid_utils.kronecker_products(ps, err_c)  # Prolong.
    x = homogeneous_smoother_fn(x, b)

    return i + 1, x, err_c

  coarse_shape = [p.shape[1] if p is not None else subgrid_shape[i]
                  for i, p in enumerate(ps)]
  i = 0

  if isinstance(x, tf.Tensor):
    err_c = tf.zeros(coarse_shape, x.dtype)
    _, x, err_c = tf.while_loop(
        cond=lambda i, *_: i < n_coarse,
        body=body,
        loop_vars=[i, x, err_c],
        back_prop=False)
  else:
    err_c = np.zeros(coarse_shape, x.dtype)
    while i < n_coarse:
      i, x, err_c = body(i, x, err_c)

  return x


def mg_cycle(
    x: TensorOrArray,
    b: TensorOrArray,
    prs: ProlongRestrictMatrices,
    homogeneous_smoother_fn: SmootherTypeFn,
    residual_fn: SmootherTypeFn,
    a_inv_for_coarsest_level: Optional[np.ndarray] = None,
    n_coarse: int = 1,
    replica_id: Union[int, tf.Tensor] = 0,
    replicas: Optional[np.ndarray] = None,
    coordinates: Optional[TensorOrArray] = None,
    num_cycles: int = 1) -> TensorOrArray:
  """Performs `num_cycle` multigrid cycles for the set of equations `A x = b`.

  `x` represents the unknown field with homogeneous boundary conditions.

  The matrix `A` is implicitly specified by the two function agruments. If
  `n_coarse` is 1, a V-cycle is done; if 2, a W-cycle is done. `x` and `b` can
  be in any number of dimensions.

  Args:
    x: The tensor or array of unknowns which have homogeneous boundary
      conditions.
    b: The source tensor or array.
    prs: The nested sequences of prolongation and restriction matrices to
      prolong / restrict uniform grids from the finest size down to size 3 in
      each dimension. See `multigrid_utils.prolong_restrict_matrices`, which
      returns the required matrix sequences.
    homogeneous_smoother_fn: The smoothing function. Takes arguments `x` and `b`
      and returns the smoothed `x`. This function should apply the homogeneous
      boundary conditions.
    residual_fn: A function that, given `x` and `b`, returns the residual
      `b - A x`.
    a_inv_for_coarsest_level: The inverse of the `A` matrix for the coarsest
      level. This is used to solve for `x` exactly via matrix multiply. If not
      provided, the coarsest level returns a smoothed `x`.
    n_coarse: The number of coarse grid iterations. Use 1 for V-cycle and 2 for
      W-cycle.
    replica_id: The replica id.
    replicas: A numpy array that maps a replica's grid coordinate to its
      replica_id, e.g. replicas[0, 0, 0] = 0, replicas[0, 0, 1] = 1.
    coordinates: The coordinates of this replica (as a tensor or array).
    num_cycles: The number of multigrid cycles to perform.

  Returns:
    The input `x` after applying `num_cycles` multigrid cycles.
  """

  if (replicas is None) or (np.prod(replicas.shape) == 1):
    # Single core.
    dirichlet0_halo_exchange_fn = multigrid_utils.zero_borders
  else:
    dirichlet0_halo_exchange_fn = multigrid_utils.halo_exchange_step_fn(
        replica_id, replicas, halo_exchange_utils.homogeneous_bcs())

  def body(i, x):
    x = _mg_cycle_internal(
        x, b, prs, homogeneous_smoother_fn, residual_fn,
        dirichlet0_halo_exchange_fn, a_inv_for_coarsest_level, n_coarse,
        replica_id, replicas, coordinates)
    return i + 1, x

  cond = lambda i, _: i < num_cycles

  i = 0
  if isinstance(x, tf.Tensor):
    _, x = tf.while_loop(
        cond=cond, body=body, loop_vars=[i, x], back_prop=False)
  else:
    while cond(i, x):
      i, x = body(i, x)

  return x


def poisson_mg_cycle_fn_for_one_core(  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    full_grid_shape: Sequence[int],
    n_coarse: int = 1,
    n_smooth: int = 1,
    weight: float = 2 / 3,
    coarsest_full_grid_shape: Optional[Sequence[int]] = None,
    params: Optional[grid_parametrization.GridParametrization] = None,
    prs: Optional[ProlongRestrictMatrices] = None,
    use_a_inv: bool = True,
    boundary_conditions: Optional[
        halo_exchange_utils.BoundaryConditionsSpec] = None,
    num_cycles: int = 1,
    dtype: np.dtype = np.float32
) -> SmootherTypeFn:
  """Returns a Poisson single-core `mg_cycle` function.

  Returns an `mg_cycle` function for the case of the Poisson equation on one
  core. The function takes two input tensors or arrays: the unknowns `x`, and
  the source `b`. The function returns the tensor `x` after a multigrid
  cycle.

  Args:
    full_grid_shape: The shape of the full_grid.
    n_coarse: The number of coarse grid iterations. Use 1 for V-cycle and 2 for
      W-cycle.
    n_smooth: The number of Jacobi smoothing iterations.
    weight: The weight used in the Jacobi smoothing. The default value is
      the canonical `2 / 3` (in many cases this is the optimal value).
    coarsest_full_grid_shape: The shape of the coarsest full grid. If `None`, 3
      in each dimension is used.
    params: The grid parametrization.
    prs: Prolong/restrict matrices per dim per level.
    use_a_inv: If `False`, smooth at the coarsest level instead of solving
      exactly.
    boundary_conditions: The boundary conditions specification.
    num_cycles: The number of multigrid cycles to do.
    dtype: The dtype.
  """
  rank = len(full_grid_shape)
  computation_shape = (1, 1, 1)

  if params is None:
    grid_lengths = (1, 1, 1)
    # params are used only for grid lengths and computation shape.
    params = (grid_parametrization.GridParametrization.
              create_from_grid_lengths_and_etc_with_defaults(
                  grid_lengths, computation_shape))
  elif (params.cx, params.cy, params.cz) != computation_shape:
    raise ValueError('params must have computation_shape (1, 1, 1).')

  full_grid_lengths = (params.lx, params.ly, params.lz)[:rank]
  a_operator = multigrid_utils.laplacian_and_inv_diagonal_fns(
      full_grid_shape, full_grid_lengths)[0]

  homogeneous_halo_exchange_fn = (
      multigrid_utils.get_apply_one_core_boundary_conditions_fn(
          boundary_conditions, homogeneous=True))
  homogeneous_smoother_fn = multigrid_utils.poisson_jacobi_fn_for_one_core(
      params, n_smooth, weight, homogeneous_halo_exchange_fn)

  apply_boundary_conditions_fn = (
      multigrid_utils.get_apply_one_core_boundary_conditions_fn(
          boundary_conditions))

  residual_fn = functools.partial(multigrid_utils.poisson_residual,
                                  params=params)

  if not coarsest_full_grid_shape:
    coarsest_full_grid_shape = (4,) * rank
  if prs is None:
    prs = multigrid_utils.prolong_restrict_matrices_from_shapes(
        full_grid_shape, coarsest_full_grid_shape, computation_shape, dtype)

  if use_a_inv:
    a_inv_for_coarsest_level = multigrid_utils.inverse_laplacian_matrix(
        coarsest_full_grid_shape, full_grid_lengths, boundary_conditions,
        dtype)
  else:
    a_inv_for_coarsest_level = None

  def mg_cycle_fn(x, b):
    x0 = tf.zeros_like(x) if isinstance(x, tf.Tensor) else np.zeros_like(x)
    xb = apply_boundary_conditions_fn(x) if apply_boundary_conditions_fn else x
    b_minus_a_xb = b - a_operator(xb)

    x0 = mg_cycle(
        x0, b_minus_a_xb, prs, homogeneous_smoother_fn, residual_fn,
        a_inv_for_coarsest_level, n_coarse, num_cycles=num_cycles)

    return x0 + xb

  return mg_cycle_fn


def poisson_mg_cycle_fn(  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    params: grid_parametrization.GridParametrization,
    prs: ProlongRestrictMatrices,
    boundary_conditions: halo_exchange_utils.BoundaryConditionsSpec,
    homogeneous_smoother_fn: SmootherTypeFn,
    coarsest_subgrid_shape: Optional[Sequence[int]] = None,
    n_coarse: int = 1,
    use_a_inv: bool = True,
    replica_id: Union[int, tf.Tensor] = 0,
    replicas: Optional[np.ndarray] = None,
    coordinates: Optional[TensorOrArray] = None,
    num_cycles: int = 1,
    dtype: np.dtype = np.float32
) -> Callable[[TensorOrArray, TensorOrArray], TensorOrArray]:
  """Returns a Poisson `mg_cycle` function.

  Returns an `mg_cycle` function for the case of the Poisson equation running on
  multiple cores. The function takes two input tensors or arrays: the unknowns
  `x`, and the source `b`. The function returns the tensor `x` after
  `num_cycles` multigrid cycles. The homogeneous boundary conditions are
  embedded in `homogeneous_smoother_fn`.

  Args:
    params: The grid parametrization.
    prs: Prolong/restrict matrices per dim per level.
    boundary_conditions: The boundary conditions specification.
    homogeneous_smoother_fn: The smoothing function. Takes arguments `x` and `b`
      and returns the smoothed `x`. The boundary conditions associated with halo
       exchange in this function should be homogeneous.
    coarsest_subgrid_shape: The shape of the coarsest subgrid. If `None`, 3 in
      each dimension is used.
    n_coarse: The number of coarse grid iterations. Use 1 for V-cycle and 2 for
      W-cycle.
    use_a_inv: If `False`, smooth at the coarsest level instead of solving
      exactly.
    replica_id: The replica id.
    replicas: A numpy array that maps a replica's grid coordinate to its
      replica_id, e.g. replicas[0, 0, 0] = 0, replicas[0, 0, 1] = 1.
    coordinates: The coordinates of this replica (as a tensor or array).
    num_cycles: The number of cycles.
    dtype: The dtype.
  """
  if not coarsest_subgrid_shape:
    coarsest_subgrid_shape = (4,) * 3
  computation_shape = (params.cx, params.cy, params.cz)
  full_grid_lengths = (params.lx, params.ly, params.lz)

  if use_a_inv:
    coarsest_full_grid_shape = multigrid_utils.get_full_grid_shape(
        coarsest_subgrid_shape, computation_shape)
    a_inv_for_coarsest_level = multigrid_utils.inverse_laplacian_matrix(
        coarsest_full_grid_shape, full_grid_lengths, boundary_conditions,
        dtype)
  else:
    a_inv_for_coarsest_level = None

  residual_fn = functools.partial(multigrid_utils.poisson_residual,
                                  params=params)

  def mg_cycle_fn(x, b):
    return mg_cycle(
        x, b, prs, homogeneous_smoother_fn, residual_fn,
        a_inv_for_coarsest_level, n_coarse, replica_id, replicas, coordinates,
        num_cycles)

  return mg_cycle_fn


def poisson_mg_cycle_step_fn(  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    params: grid_parametrization.GridParametrization,
    coarsest_subgrid_shape: Optional[Sequence[int]] = None,
    n_coarse: int = 1,
    n_smooth: int = 1,
    weight: float = 2 / 3,
    boundary_conditions: Optional[BoundaryConditionsSpec] = None,
    use_a_inv: bool = True,
    num_cycles: int = 1,
    dtype: np.dtype = np.float32
) -> Callable[[ProlongRestrictMatrices,
               Union[int, tf.Tensor],
               Optional[np.ndarray],
               Optional[TensorOrArray]], Any]:
  """Step function version of `poisson_mg_cycle_fn`. Includes Jacobi params."""
  if boundary_conditions is None:
    boundary_conditions = halo_exchange_utils.homogeneous_bcs()
  if not coarsest_subgrid_shape:
    coarsest_subgrid_shape = (4, 4, 4)
  homogeneous_boundary_conditions = (
      multigrid_utils.get_homogeneous_boundary_conditions(
          boundary_conditions))
  homogeneous_smoother_step_fn = multigrid_utils.poisson_jacobi_step_fn(
      params, n_smooth, weight, homogeneous_boundary_conditions)

  def step_fn(prs: ProlongRestrictMatrices,
              replica_id: Union[int, tf.Tensor] = 0,
              replicas: Optional[np.ndarray] = None,
              coordinates: Optional[TensorOrArray] = None):
    homogeneous_smoother_fn = homogeneous_smoother_step_fn(replica_id, replicas)

    return poisson_mg_cycle_fn(
        params, prs, boundary_conditions, homogeneous_smoother_fn,
        coarsest_subgrid_shape, n_coarse, use_a_inv, replica_id, replicas,
        coordinates, num_cycles, dtype)

  return step_fn
