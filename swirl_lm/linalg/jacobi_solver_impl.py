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

from swirl_lm.linalg import base_poisson_solver
from swirl_lm.linalg import poisson_solver_pb2
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal: TypeAlias = types.FlowFieldVal
FlowFieldMap: TypeAlias = types.FlowFieldMap
_HaloUpdateFn: TypeAlias = Callable[[FlowFieldVal], FlowFieldVal]
PoissonSolverSolution: TypeAlias = base_poisson_solver.PoissonSolverSolution

X = base_poisson_solver.X
ITERATIONS = base_poisson_solver.ITERATIONS
VARIABLE_COEFF = base_poisson_solver.VARIABLE_COEFF


class BaseJacobiSolver:
  """Base class for weighted Jacobi solvers."""

  def __init__(
      self,
      grid_spacings: tuple[float, float, float],
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
  ):
    self._grid_spacings = grid_spacings
    self._kernel_op = kernel_op
    self._solver_option = solver_option

    self._kernel_op.add_kernel({'weighted_sum_121': ([1.0, 2.0, 1.0], 1)})
    self._omega = solver_option.jacobi.omega
    self._num_iters = solver_option.jacobi.max_iterations
    self._halo_width = solver_option.jacobi.halo_width

    self._delta2_inv = (
        1.0 / grid_spacings[0] ** 2,
        1.0 / grid_spacings[1] ** 2,
        1.0 / grid_spacings[2] ** 2,
    )

  def _apply_underrelaxation(
      self, p_star: FlowFieldVal, p: FlowFieldVal
  ) -> FlowFieldVal:
    """Apply underrelaxation for weighted Jacobi solver."""
    return tf.nest.map_structure(
        lambda p_star, p: self._omega * p_star + (1.0 - self._omega) * p,
        p_star,
        p,
    )

  def _do_iterations(
      self,
      p0: FlowFieldVal,
      one_iteration_fn: Callable[[FlowFieldVal], FlowFieldVal],
      halo_update_fn: _HaloUpdateFn,
  ) -> PoissonSolverSolution:
    """Runs the iterations for weighted Jacobi solver."""
    i0 = tf.constant(0)
    stop_condition = lambda i, p: i < self._num_iters
    body = lambda i, p: (i + 1, one_iteration_fn(p))

    iterations, p = tf.while_loop(
        cond=stop_condition, body=body, loop_vars=(i0, p0), back_prop=False
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
      grid_spacings: tuple[float, float, float],
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
  ):
    """Initializes the Jacobi solver for the Poisson equation."""
    super().__init__(grid_spacings, kernel_op, solver_option)

    self._factor_b = 0.5 / (
        self._delta2_inv[0] + self._delta2_inv[1] + self._delta2_inv[2]
    )
    self._factor_x = grid_spacings[0] ** 2 / self._factor_b
    self._factor_y = grid_spacings[1] ** 2 / self._factor_b
    self._factor_z = grid_spacings[2] ** 2 / self._factor_b

  def _poisson_step(
      self,
      p: FlowFieldVal,
      rhs: FlowFieldVal,
      halo_update_fn: Callable[[FlowFieldVal], FlowFieldVal],
  ) -> FlowFieldVal:
    """Computes the pressure correction for the next sub-iteration."""
    # Compute the right hand side function for interior points.
    p = halo_update_fn(p)

    p_terms = (
        self._kernel_op.apply_kernel_op_x(p, 'kSx'),
        self._kernel_op.apply_kernel_op_y(p, 'kSy'),
        self._kernel_op.apply_kernel_op_z(p, 'kSz', 'kSzsh'),
        rhs,
    )

    p_jacobi = tf.nest.map_structure(
        lambda fx, fy, fz, fb: (  # pylint: disable=g-long-lambda
            fx / self._factor_x
            + fy / self._factor_y
            + fz / self._factor_z
            - self._factor_b * fb
        ),
        *p_terms,
    )

    return self._apply_underrelaxation(p_jacobi, p)

  def solve(
      self,
      rhs: FlowFieldVal,
      p0: FlowFieldVal,
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
      p: FlowFieldVal,
      w: FlowFieldVal,
      rhs: FlowFieldVal,
      halo_update_fn: Callable[[FlowFieldVal], FlowFieldVal],
  ) -> FlowFieldVal:
    """Performs one Jacobi iteration."""
    p = halo_update_fn(p)

    # Compute the diagonal factor.
    factor_diag = (
        tf.nest.map_structure(
            lambda ws: 0.5 * self._delta2_inv[0] * ws,
            self._kernel_op.apply_kernel_op_x(w, 'weighted_sum_121x'),
        ),
        tf.nest.map_structure(
            lambda ws: 0.5 * self._delta2_inv[1] * ws,
            self._kernel_op.apply_kernel_op_y(w, 'weighted_sum_121y'),
        ),
        tf.nest.map_structure(
            lambda ws: 0.5 * self._delta2_inv[2] * ws,
            self._kernel_op.apply_kernel_op_z(
                w, 'weighted_sum_121z', 'weighted_sum_121zsh'
            ),
        ),
    )
    factor_diag_sum = tf.nest.map_structure(lambda a, b, c: a + b + c,
                                            *factor_diag)

    # Compute factors for the off-diagonal terms.
    sum_op = (
        lambda f: tf.nest.map_structure(
            lambda fs: 0.5 * self._delta2_inv[0] * fs,
            self._kernel_op.apply_kernel_op_x(f, 'kSx'),
        ),
        lambda f: tf.nest.map_structure(
            lambda fs: 0.5 * self._delta2_inv[1] * fs,
            self._kernel_op.apply_kernel_op_y(f, 'kSy'),
        ),
        lambda f: tf.nest.map_structure(
            lambda fs: 0.5 * self._delta2_inv[2] * fs,
            self._kernel_op.apply_kernel_op_z(f, 'kSz', 'kSzsh'),
        ),
    )

    p_factor = [sum_op_i(p) for sum_op_i in sum_op]
    p_sum = tf.nest.map_structure(lambda a, b, c: a + b + c, *p_factor)

    w_p = tf.nest.map_structure(tf.math.multiply, w, p)
    w_p_factor = [sum_op_i(w_p) for sum_op_i in sum_op]
    w_p_sum = tf.nest.map_structure(lambda a, b, c: a + b + c, *w_p_factor)

    rhs_factor = tf.nest.map_structure(
        lambda rhs_i, w_i, p_s, w_p_s: -rhs_i + w_i * p_s + w_p_s,
        rhs,
        w,
        p_sum,
        w_p_sum,
    )

    p_jacobi = tf.nest.map_structure(
        tf.math.divide, rhs_factor, factor_diag_sum
    )
    return self._apply_underrelaxation(p_jacobi, p)

  def solve(
      self,
      w: FlowFieldVal,
      rhs: FlowFieldVal,
      p0: FlowFieldVal,
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

      ∂/∂x (w0 ∂p/∂x) + ∂/∂y (w1 ∂p/∂y) + ∂/∂z (w2 ∂p/∂z) = f

  where w0(x,y,z), w1(x,y,z), w2(x,y,z) are the three weighting coefficients.
  The equation is solved using the weighted Jacobi method.
  """

  def _precompute_reciprocal_diagonal_factor(
      self,
      w0: FlowFieldVal,
      w1: FlowFieldVal,
      w2: FlowFieldVal,
  ) -> FlowFieldVal:
    """Computes the reciprocal diagonal factor for the 3-weight Poisson eqn."""
    factor_diag = (
        tf.nest.map_structure(
            lambda w0: 0.5 * self._delta2_inv[0] * w0,
            self._kernel_op.apply_kernel_op_x(w0, 'weighted_sum_121x'),
        ),
        tf.nest.map_structure(
            lambda w1: 0.5 * self._delta2_inv[1] * w1,
            self._kernel_op.apply_kernel_op_y(w1, 'weighted_sum_121y'),
        ),
        tf.nest.map_structure(
            lambda w2: 0.5 * self._delta2_inv[2] * w2,
            self._kernel_op.apply_kernel_op_z(
                w2, 'weighted_sum_121z', 'weighted_sum_121zsh'
            ),
        ),
    )
    return tf.nest.map_structure(
        lambda a, b, c: 1 / (a + b + c), *factor_diag
    )

  def _three_weight_poisson_step(
      self,
      p: FlowFieldVal,
      w0: FlowFieldVal,
      w1: FlowFieldVal,
      w2: FlowFieldVal,
      rhs: FlowFieldVal,
      halo_update_fn: Callable[[FlowFieldVal], FlowFieldVal],
      reciprocal_diagonal_factor: FlowFieldVal,
  ) -> FlowFieldVal:
    """Performs one Jacobi iteration."""
    p = halo_update_fn(p)

    # Compute factors for the off-diagonal terms.
    sum_op = (
        lambda f: tf.nest.map_structure(
            lambda fs: 0.5 * self._delta2_inv[0] * fs,
            self._kernel_op.apply_kernel_op_x(f, 'kSx')),
        lambda f: tf.nest.map_structure(
            lambda fs: 0.5 * self._delta2_inv[1] * fs,
            self._kernel_op.apply_kernel_op_y(f, 'kSy')),
        lambda f: tf.nest.map_structure(
            lambda fs: 0.5 * self._delta2_inv[2] * fs,
            self._kernel_op.apply_kernel_op_z(f, 'kSz', 'kSzsh')),
    )
    multiply = lambda a, b: tf.nest.map_structure(tf.math.multiply, a, b)

    w = (w0, w1, w2)
    s_p = [sum_op[dim](p) for dim in (0, 1, 2)]
    w_s_p = [multiply(w[dim], s_p[dim]) for dim in (0, 1, 2)]
    w_p = [multiply(w[dim], p) for dim in (0, 1, 2)]
    s_w_p = [sum_op[dim](w_p[dim]) for dim in (0, 1, 2)]

    numerator = tf.nest.map_structure(
        lambda t1, t2, t3, t4, t5, t6, c: t1 + t2 + t3 + t4 + t5 + t6 - c,
        *w_s_p,
        *s_w_p,
        rhs,
    )
    p_jacobi = multiply(numerator, reciprocal_diagonal_factor)
    return self._apply_underrelaxation(p_jacobi, p)

  def residual(
      self,
      p: tf.Tensor,
      w0: tf.Tensor,
      w1: tf.Tensor,
      w2: tf.Tensor,
      rhs: tf.Tensor,
  ) -> tf.Tensor:
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
      w0: The first weighting coefficient.
      w1: The second weighting coefficient.
      w2: The third weighting coefficient.
      rhs: The right hand side of the Poisson equation.

    Returns:
      The residual (LHS - RHS) of the Poisson equation.
    """
    reciprocal_diagonal_factor = self._precompute_reciprocal_diagonal_factor(
        w0, w1, w2
    )
    diagonal_factor = 1 / reciprocal_diagonal_factor

    # Compute factors for the off-diagonal terms.
    sum_op = (
        lambda f: (0.5 * self._delta2_inv[0])
        * self._kernel_op.apply_kernel_op_x(f, 'kSx'),
        lambda f: (0.5 * self._delta2_inv[1])
        * self._kernel_op.apply_kernel_op_y(f, 'kSy'),
        lambda f: (0.5 * self._delta2_inv[2])
        * self._kernel_op.apply_kernel_op_z(f, 'kSz', 'kSzsh'),
    )

    w = (w0, w1, w2)
    s_p = [sum_op[dim](p) for dim in (0, 1, 2)]
    w_s_p = [w[dim] * s_p[dim] for dim in (0, 1, 2)]
    w_p = [w[dim] * p for dim in (0, 1, 2)]
    s_w_p = [sum_op[dim](w_p[dim]) for dim in (0, 1, 2)]

    return (
        -diagonal_factor * p
        + (w_s_p[0] + w_s_p[1] + w_s_p[2])
        + (s_w_p[0] + s_w_p[1] + s_w_p[2])
        - rhs
    )

  def solve(
      self,
      w0: FlowFieldVal,
      w1: FlowFieldVal,
      w2: FlowFieldVal,
      rhs: FlowFieldVal,
      p0: FlowFieldVal,
      halo_update_fn: _HaloUpdateFn,
  ) -> PoissonSolverSolution:
    """Solves the three-weight Poisson equation.

    Args:
      w0: The first weighting coefficient.
      w1: The second weighting coefficient.
      w2: The third weighting coefficient.
      rhs: The right-hand side of the Poisson equation.
      p0: The initial guess for the solution.
      halo_update_fn: A function that updates the halos and enforces boundary
        conditions for the solution.

    Returns:
      A dictionary containing the solution and the number of iterations.
    """
    reciprocal_diagonal_factor = self._precompute_reciprocal_diagonal_factor(
        w0, w1, w2
    )
    p_next_fn = functools.partial(
        self._three_weight_poisson_step,
        w0=w0,
        w1=w1,
        w2=w2,
        rhs=rhs,
        halo_update_fn=halo_update_fn,
        reciprocal_diagonal_factor=reciprocal_diagonal_factor,
    )
    return self._do_iterations(p0, p_next_fn, halo_update_fn)
