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
"""A library of Jacobi solvers for the Poisson equation.

Methods presented here are used to solve the Poisson equation, i.e.
  ∇²p = b,
as well as variants of the Poisson equations, in a distributed setting.
"""

import functools
from typing import Callable, TypeAlias

import jax
from jax import sharding
import jax.numpy as jnp
from swirl_lm.jax.linalg import base_poisson_solver
from swirl_lm.jax.linalg import poisson_solver_pb2
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types


ScalarFieldVal: TypeAlias = types.ScalarField
ScalarFieldMap: TypeAlias = types.ScalarFieldMap
_HaloUpdateFn: TypeAlias = Callable[[ScalarFieldVal], ScalarFieldVal]
PoissonSolverSolution: TypeAlias = base_poisson_solver.PoissonSolverSolution

X = base_poisson_solver.X
ITERATIONS = base_poisson_solver.ITERATIONS
VARIABLE_COEFF = base_poisson_solver.VARIABLE_COEFF


class BaseJacobiSolver:
  """Base class for weighted Jacobi solvers."""

  def __init__(
      self,
      grid_params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
      mesh: sharding.Mesh,
  ):
    self._grid_params = grid_params
    self._kernel_op = kernel_op
    self._solver_option = solver_option
    self._mesh = mesh

    if isinstance(self._kernel_op, get_kernel_fn.ApplyKernelConvOp):
      self._kernel_op.add_kernel({'weighted_sum_121': ([1.0, 2.0, 1.0], 1)})
    elif isinstance(self._kernel_op, get_kernel_fn.ApplyKernelSliceOp):
      self._kernel_op.add_kernel(
          {'weighted_sum_121': {'coeff': [1.0, 2.0, 1.0], 'shift': [-1, 0, 1]}}
      )

    self._omega = solver_option.jacobi.omega
    self._num_iters = solver_option.jacobi.max_iterations

    self._delta2_inv = (
        1.0 / self._grid_params.grid_spacings[0] ** 2,
        1.0 / self._grid_params.grid_spacings[1] ** 2,
        1.0 / self._grid_params.grid_spacings[2] ** 2,
    )

  def _apply_underrelaxation(
      self, p_star: ScalarFieldVal, p_old: ScalarFieldVal
  ) -> ScalarFieldVal:
    """Apply underrelaxation for weighted Jacobi solver.

    Args:
      p_star: The solution to the Poisson equation.
      p_old: The previous iteration of the solution.

    Returns:
      The next iteration of the solution after underrelaxation.
    """
    return self._omega * p_star + (1.0 - self._omega) * p_old

  def _do_iterations(
      self,
      p0: ScalarFieldVal,
      one_iteration_fn: Callable[[ScalarFieldVal], ScalarFieldVal],
      halo_update_fn: _HaloUpdateFn,
  ) -> PoissonSolverSolution:
    """Runs the iterations for weighted Jacobi solver."""
    stop_condition = lambda iter_p_tuple: iter_p_tuple[0] < self._num_iters
    body = lambda iter_p_tuple: (
        iter_p_tuple[0] + 1,
        one_iteration_fn(iter_p_tuple[1]),
    )
    iterations, p = jax.lax.while_loop(
        cond_fun=stop_condition, body_fun=body, init_val=(jnp.array(0), p0)
    )
    return {
        X: halo_update_fn(p),
        ITERATIONS: iterations,
    }


class PlainPoisson(BaseJacobiSolver):
  """Library for solving the plain Poisson equation.

  Solve the equation

      (∂^2/∂x^2 + ∂^2/∂y^2 + ∂^2/∂z^2) p = f

  using the weighted Jacobi method.
  """

  def __init__(
      self,
      grid_params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
      mesh: sharding.Mesh,
  ):
    """Initializes the Jacobi solver for the Poisson equation."""
    super().__init__(grid_params, kernel_op, solver_option, mesh)

    self._factor_b = 0.5 / (
        self._delta2_inv[0] + self._delta2_inv[1] + self._delta2_inv[2]
    )
    self._factors = tuple(
        [g**2 / self._factor_b for g in self._grid_params.grid_spacings]
    )

  def _poisson_step(
      self,
      p: ScalarFieldVal,
      rhs: ScalarFieldVal,
      halo_update_fn: _HaloUpdateFn,
  ) -> ScalarFieldVal:
    """Computes the pressure correction for the next sub-iteration."""
    # Compute the right hand side function for interior points.
    p = halo_update_fn(p)

    p_terms = [None] * 3
    for axis in ('x', 'y', 'z'):
      p_terms[self._grid_params.get_axis_index(axis)] = (
          self._kernel_op.apply_kernel_op(p, 'kS', axis)
      )
    p_jacobi = (
        p_terms[0] / self._factors[0]
        + p_terms[1] / self._factors[1]
        + p_terms[2] / self._factors[2]
        - self._factor_b * rhs
    )

    return self._apply_underrelaxation(p_jacobi, p)

  def solve(
      self,
      rhs: ScalarFieldVal,
      p0: ScalarFieldVal,
      halo_update_fn: _HaloUpdateFn,
  ) -> PoissonSolverSolution:
    """Solves the Poisson equation."""
    p_next_fn = functools.partial(
        self._poisson_step, rhs=rhs, halo_update_fn=halo_update_fn
    )
    return self._do_iterations(p0, p_next_fn, halo_update_fn)


class VariableCoefficient(BaseJacobiSolver):
  """Library for solving the variable-coefficient Poisson equation with Jacobi.

  Solve the variable-coefficient Poisson equation:

      ∂/∂x (w ∂p/∂x) + ∂/∂y (w ∂p/∂y) + ∂/∂z (w ∂p/∂z) = f

  where w(x,y,z) is the variable coefficent. The weighted Jacobi method is used.
  """

  def _variable_coefficient_poisson_step(
      self,
      p: ScalarFieldVal,
      w: ScalarFieldVal,
      rhs: ScalarFieldVal,
      halo_update_fn: Callable[[ScalarFieldVal], ScalarFieldVal],
  ) -> ScalarFieldVal:
    """Performs one Jacobi iteration."""
    p = halo_update_fn(p)

    def get_kernel_op_sum(array: ScalarFieldVal, kernel_name: str):
      kernel_op_sum = jnp.zeros_like(array)
      for axis in ('x', 'y', 'z'):
        axis_index = self._grid_params.get_axis_index(axis)
        kernel_op_sum += (
            0.5
            * jnp.array(self._delta2_inv[axis_index], dtype=array.dtype)
            * self._kernel_op.apply_kernel_op(array, kernel_name, axis)
        )
      return kernel_op_sum

    # Compute the diagonal factor.
    factor_diag_sum = get_kernel_op_sum(w, 'weighted_sum_121')

    # Compute factors for the off-diagonal terms.
    p_sum = get_kernel_op_sum(p, 'kS')
    w_p_sum = get_kernel_op_sum(w * p, 'kS')

    p_jacobi = (-rhs + w * p_sum + w_p_sum) / factor_diag_sum
    return self._apply_underrelaxation(p_jacobi, p)

  def solve(
      self,
      w: ScalarFieldVal,
      rhs: ScalarFieldVal,
      p0: ScalarFieldVal,
      halo_update_fn: _HaloUpdateFn,
  ) -> PoissonSolverSolution:
    """Solves the variable-coefficient Poisson equation.

    Args:
      w: The variable-coefficient weight in the Poisson equation.
      rhs: The right-hand-side of the Poisson equation.
      p0: The initial guess for the solution.
      halo_update_fn: A function that updates the halos and boundaries of the
        solution.

    Returns:
      A dictionary containing the solution and the number of iterations.
    """
    p_next_fn = functools.partial(
        self._variable_coefficient_poisson_step,
        w=w,
        rhs=rhs,
        halo_update_fn=halo_update_fn,
    )

    return self._do_iterations(p0, p_next_fn, halo_update_fn)


class ThreeWeight(BaseJacobiSolver):
  """Jacobi solver for the three-weight Poisson equation.

  Solves the three-weight Poisson equation:

      ∂/∂x (w_x ∂p/∂x) + ∂/∂y (w_y ∂p/∂y) + ∂/∂z (w_z ∂p/∂z) = f

  where w_x(x,y,z), w_y(x,y,z), w_z(x,y,z) are the three weighting coefficients
  corresponding to the (x, y, z) axes respectively. The equation is solved using
  the weighted Jacobi method.
  """

  def _kernel_op_wrapper(
      self,
      array: ScalarFieldVal,
      delta2_inv: float,
      kernel_name: str,
      axis: str,
  ) -> ScalarFieldVal:
    return (
        0.5
        * jnp.array(delta2_inv, dtype=array.dtype)
        * self._kernel_op.apply_kernel_op(array, kernel_name, axis)
    )

  def _precompute_reciprocal_diagonal_factor(
      self, w_x: ScalarFieldVal, w_y: ScalarFieldVal, w_z: ScalarFieldVal
  ) -> ScalarFieldVal:
    """Computes the reciprocal diagonal factor for the 3-weight Poisson eqn."""
    w_dict = {'x': w_x, 'y': w_y, 'z': w_z}
    factor_diag = jnp.zeros_like(w_x)
    for axis in ('x', 'y', 'z'):
      axis_index = self._grid_params.get_axis_index(axis)
      factor_diag += self._kernel_op_wrapper(
          w_dict[axis], self._delta2_inv[axis_index], 'weighted_sum_121', axis
      )
    return 1.0 / factor_diag

  def _precompute_off_diagonal_factor(
      self,
      p: ScalarFieldVal,
      w_x: ScalarFieldVal,
      w_y: ScalarFieldVal,
      w_z: ScalarFieldVal,
      rhs: ScalarFieldVal,
  ) -> ScalarFieldVal:
    """Computes the off-diagonal factor for the 3-weight Poisson eqn."""
    w_dict = {'x': w_x, 'y': w_y, 'z': w_z}
    w_s_p_dict = {}
    s_w_p_dict = {}
    for axis in ('x', 'y', 'z'):
      axis_index = self._grid_params.get_axis_index(axis)
      w_s_p_dict[axis] = w_dict[axis] * self._kernel_op_wrapper(
          p, self._delta2_inv[axis_index], 'kS', axis
      )
      s_w_p_dict[axis] = self._kernel_op_wrapper(
          w_dict[axis] * p, self._delta2_inv[axis_index], 'kS', axis
      )
    return (
        w_s_p_dict['x']
        + w_s_p_dict['y']
        + w_s_p_dict['z']
        + s_w_p_dict['x']
        + s_w_p_dict['y']
        + s_w_p_dict['z']
        - rhs
    )

  def _three_weight_poisson_step(
      self,
      p: ScalarFieldVal,
      w_x: ScalarFieldVal,
      w_y: ScalarFieldVal,
      w_z: ScalarFieldVal,
      rhs: ScalarFieldVal,
      halo_update_fn: Callable[[ScalarFieldVal], ScalarFieldVal],
      reciprocal_diagonal_factor: ScalarFieldVal,
  ) -> ScalarFieldVal:
    """Performs one Jacobi iteration."""
    p = halo_update_fn(p)
    numerator = self._precompute_off_diagonal_factor(p, w_x, w_y, w_z, rhs)
    return self._apply_underrelaxation(
        numerator * reciprocal_diagonal_factor, p
    )

  def residual(
      self,
      p: ScalarFieldVal,
      w_x: ScalarFieldVal,
      w_y: ScalarFieldVal,
      w_z: ScalarFieldVal,
      rhs: ScalarFieldVal,
  ) -> ScalarFieldVal:
    """Computes the residual (LHS - RHS) of the three-weight Poisson equation.

    Given approximate solution `p` to the Poisson equation, compute the
    residual.  The approximate solution `p` might be obtained, for example, from
    a finite number of Jacobi iterations that are not necessarily fully
    converged.  The residual can be used as a diagnostic to check how well
    converged the solution is.

    The residual calculation very closely follows the computation of the next
    Jacobi iteration, so the computations look very similar.

    Args:
      p: The approximate solution to the Poisson equation.
      w_x: The weighting coefficient for the x axis.
      w_y: The weighting coefficient for the y axis.
      w_z: The weighting coefficient for the z axis.
      rhs: The right hand side of the Poisson equation.

    Returns:
      The residual (LHS - RHS) of the Poisson equation.
    """
    reciprocal_diagonal_factor = self._precompute_reciprocal_diagonal_factor(
        w_x, w_y, w_z
    )
    diagonal_factor = 1 / reciprocal_diagonal_factor
    off_diagonal_factor = self._precompute_off_diagonal_factor(
        p, w_x, w_y, w_z, rhs
    )
    return -diagonal_factor * p + off_diagonal_factor

  def solve(
      self,
      w_x: ScalarFieldVal,
      w_y: ScalarFieldVal,
      w_z: ScalarFieldVal,
      rhs: ScalarFieldVal,
      p0: ScalarFieldVal,
      halo_update_fn: _HaloUpdateFn,
  ) -> PoissonSolverSolution:
    """Solves the three-weight Poisson equation.

    Args:
      w_x: The weighting coefficient for the x axis.
      w_y: The weighting coefficient for the y axis.
      w_z: The weighting coefficient for the z axis.
      rhs: The right-hand side of the Poisson equation.
      p0: The initial guess for the solution.
      halo_update_fn: A function that updates the halos and enforces boundary
        conditions for the solution.

    Returns:
      A dictionary containing the solution and the number of iterations.
    """
    reciprocal_diagonal_factor = self._precompute_reciprocal_diagonal_factor(
        w_x, w_y, w_z
    )
    p_next_fn = functools.partial(
        self._three_weight_poisson_step,
        w_x=w_x,
        w_y=w_y,
        w_z=w_z,
        rhs=rhs,
        halo_update_fn=halo_update_fn,
        reciprocal_diagonal_factor=reciprocal_diagonal_factor,
    )
    return self._do_iterations(p0, p_next_fn, halo_update_fn)
