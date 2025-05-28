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
"""A tensorflow library of a distributed conjugate gradient solver.

Uses the conjugate gradient method to iteratively solve a linear system
A x = b, where A is a Hermitian semi-definite operator.

The implementation is "matrix-free", accepting the linear operator `A` as a
`Callable` that evaluates the action of `A` on a given input vector `x`
(which may be a Tensor of any shape, or a list of Tensors). In a distributed
setting, the vectors `x` may be partitioned among cores, and the `Callable`
implementing `A` is responsible for handling any communication/coordination
required (e.g., a halo exchange). Note that the output of `A` must be
partitioned/distributed in exactly the same way as its input.

The implementation also accepts the inner product as a `Callable`. In the
usual distributed use case, this will involve a cross replica sum. In more
generality, `A` must be self-adjoint nonnegative with respect to the provided
inner product. This constraint reduces to `A` being (or rather implementing the
action of) a Hermitian semi-definite matrix when the inner product is the
standard Euclidean one.

As a second distributed use case, it is possible for the vectors `x` and `b` to
be replicated across cores, but for `A` to be distributed. In this case, the
Callable implementing `A` is responsible for somehow ensuring its output is
replicated across cores, just as its input is, and the inner product `Callable`
can just be the local dot product.

Pseudo-code for the first use case:

  def linear_op(x: Sequence[tf.Tensor]) -> Sequence[tf.Tensor]:
    # An operator defining the linear system of equations.
    # In cases where halo exchange is performed, the values in the halo need to
    # be set to zero before returning to the solver, i.e.
    # x = halo_exchange.inplace_halo_exchange(...)
    # f = foo(x)  ## This is a transformation only on the local copy.
    # return halo_exchange.clear_halos(f) -- removing the halo so the operation
    #    is truly linear, and the dot product is correct.
    ...

  # Note that common_ops.local_dot can be used for the second use case instead.
  dot = functools.partial(common_ops.global_dot,
                          group_assignment=group_assignment)

  # Values of the right hand side vector.
  b = ...

  # The maximum number of iterations allowed before the solver terminates.
  max_iteration = ...

  # The tolerance that is used to determine if the solution is converged.
  tol = ...

  # The initial guess of the solution.
  x0 = ...

  # Solver the linear system of equations.
  x = conjugate_gradient_solver(linear_op, dot, b, max_iterations, tol, x0)
"""

import collections
from typing import Callable, Optional
from swirl_lm.linalg import base_poisson_solver
from swirl_lm.utility import common_ops
from swirl_lm.utility import types
import tensorflow as tf

_CGState = collections.namedtuple(
    'CGState', ('residual', 'direction', 'solution', 'sq_norm_res',
                'component_wise_distance', 'gamma'))
_TF_DTYPE = types.TF_DTYPE
_UNUSED_VALUE = 1

X = base_poisson_solver.X
RESIDUAL_L2_NORM = base_poisson_solver.RESIDUAL_L2_NORM
COMPONENT_WISE_DISTANCE = base_poisson_solver.COMPONENT_WISE_DISTANCE
ITERATIONS = base_poisson_solver.ITERATIONS

PoissonSolverSolution = base_poisson_solver.PoissonSolverSolution


def conjugate_gradient_solver(
    linear_operator: common_ops.LinearOp,
    dot: common_ops.Dot,
    b: common_ops.FlowFieldVal,
    max_iterations: int,
    tol: tf.Tensor,
    x0: common_ops.FlowFieldVal,
    l2_norm_reduction: bool = False,
    component_wise_distance_fn: Optional[Callable[[common_ops.FlowFieldVal],
                                                  tf.Tensor]] = None,
    reprojection: Optional[common_ops.LinearOp] = None,
    preconditioner=None,
    internal_dtype=None,
) -> PoissonSolverSolution:
  """Solves `[a]{x} = {b}` with the conjugate gradient method.

  This function is to be used in a SIMD setting across multiple TPUs.
  Either of the following two conditions can terminate the procedure: the
  number of iterations reaches `max_iterations` or `rho` (the squared L2 norm of
  the residual or its L2 norm reduction relative to `b`) is smaller than or
  equal to `tol`.

  Args:
    linear_operator: A `Callable` giving the action of multiplying a Hermitian
      semi-definite `n x n` matrix, with a vector stored in a form of a sequence
      of tf.Tensor, where `n` is also the size of vectors `{b}` and `{x}`. Note
      that `n` is the physical size of the problem, which excludes the halo size
      in a multicore simulation.
    dot: A `Callable` that computes the dot product of two vectors.
    b: The right-hand-side (rhs) vector.
    max_iterations: The maximum number of iterations.
    tol: A predefined tolerance for residual's absolute L2 norm or relative L2
      norm reduction relative to `b`.
    x0: The initial guess to the solution vector. Note that the halos for `x0`
      need to be updated before calling `solve`.
    l2_norm_reduction: Whether to use `tol` as a relative L2 norm for the
      residual, relative to the input rhs `b`
    component_wise_distance_fn: An optional function that takes a `lhs`
      (`A * x`) and returns its componentwise distance to the `rhs` i.e. `b`.
      When not None, this distance is used as an extra convergence criterion,
      indicating convergence when non-positive. The overall convergence
      criterion is a logical OR of L2 norm (through `tol` and
      `l2_norm_reduction`) & component wise distance.
    reprojection: A `Callable` that performs reprojection onto the orthogonal
      complement of the null space of a matrix. This is helpful while dealing
      with a singular system. For example, the singular system arises from the
      situation where only Neumann boundaries are applied; in this case,
      reprojecting the residual and iterate onto the orthogonal complement of
      the null space is equivalent to subtracting off the mean. The reprojection
      resolves the divergence issue arising from round-off error accumulation.
      A reference to reprojection: E.F. Kaasschieter, "Preconditioned conjugate
        gradients for solving singular systems",
      https://doi.org/10.1016/0377-0427(88)90358-5
    preconditioner: An optional efficient preconditioner could dramatically
      improve the rate of convergence. It represents a matrix `M`, and an
      approximation for `A^{-1}`.
    internal_dtype: Which dtype to use inside the function, independent of the
      original dtype of `b` or `x0`, useful if `float64` is needed to avoid
      numeric error accumulation.

  Returns:
    A dict with the following elements:
      1. Solution vector to the linear system `[a]` under the rhs vector `{b}`
      2. L2 norm for the residual vector
      3. Number of iterations used for the computation.
  """
  if isinstance(b, tf.Tensor):
    b = [b]
  if isinstance(x0, tf.Tensor):
    x0 = [x0]

  if internal_dtype is not None and b:
    input_dtype = b[0].dtype  # pytype: disable=attribute-error
  else:
    input_dtype = None

  b = tf.nest.map_structure(
      lambda b_i: common_ops.tf_cast(b_i, internal_dtype), b)
  x0 = tf.nest.map_structure(
      lambda x0_i: common_ops.tf_cast(x0_i, internal_dtype), x0)

  if reprojection:
    x = reprojection(x0)
  else:
    x = x0

  # Computes the residual, r.

  r = tf.nest.map_structure(tf.math.subtract, b,
                            tf.unstack(linear_operator(tf.stack(x))))
  if reprojection:
    r = reprojection(r)

  def get_cg_vars(r, is_initial_step, d_previous=None, gamma_previous=None):
    """CG vars useful in iterations."""
    # Computes the search direction, d.

    q = r if preconditioner is None else tf.unstack(preconditioner(tf.stack(r)))

    gamma = dot(tf.stack(r), tf.stack(q))

    if is_initial_step:
      d = q
    else:
      beta = tf.math.divide(gamma, gamma_previous)
      d = tf.nest.map_structure(lambda q_, d_: q_ + beta * d_, q, d_previous)

    # Computes the squared norm of the residual.
    #
    rho = gamma if preconditioner is None else dot(tf.stack(r), tf.stack(r))

    component_wise_distance = (
        _UNUSED_VALUE  # pylint: disable=g-long-ternary
        if component_wise_distance_fn is None
        else component_wise_distance_fn(
            tf.nest.map_structure(tf.math.subtract, tf.stack(r), tf.stack(b))
        )
    )

    return d, rho, gamma, component_wise_distance

  d, rho, gamma, component_wise_distance = get_cg_vars(r, True)

  state0 = _CGState(
      residual=r,
      direction=d,
      solution=x,
      sq_norm_res=rho,
      component_wise_distance=component_wise_distance,
      gamma=gamma)
  i0 = tf.constant(0)

  def _conjugate_gradient_step(linear_operator: common_ops.LinearOp,
                               dot: common_ops.Dot,
                               state: _CGState) -> _CGState:
    """Computes one step of conjugate gradient solver.

    Args:
      linear_operator: Provides the action of the matrix `[a]`.
      dot: A `Callable` that computes the dot product of two vectors.
      state: A named tuple that holds the following information:
        1. residual: the residual vector, `[a]{x} - {b}`.
        2. direction: the vector of the search direction.
        3. solution: the solution vector.
        4. sq_norm_res: the squared norm of the residual.

    Returns:
      A named tuple consisting of the residual vector, search direction,
      solution, and squared L2 norm of the residual for the next iteration.
    """
    r = state.residual
    d = state.direction
    x = state.solution
    gamma = state.gamma

    a_d = tf.unstack(linear_operator(tf.stack(d)))

    alpha = tf.math.divide(gamma, dot(tf.stack(d), tf.stack(a_d)))

    x_next = tf.nest.map_structure(lambda x_, d_: x_ + alpha * d_, x, d)
    r_next = tf.nest.map_structure(lambda r_, a_d_: r_ - alpha * a_d_, r, a_d)

    if reprojection:
      x_next = reprojection(x_next)
      r_next = reprojection(r_next)

    d_next, rho_next, gamma_next, component_wise_distance = (
        get_cg_vars(r_next, False, d, gamma))

    return _CGState(
        residual=r_next,
        direction=d_next,
        solution=x_next,
        sq_norm_res=rho_next,
        component_wise_distance=component_wise_distance,
        gamma=gamma_next)

  tol_sq = tf.square(tol)
  if internal_dtype is not None:
    tol_sq = tf.cast(tol_sq, internal_dtype)

  if l2_norm_reduction:
    tol_sq *= dot(tf.stack(b), tf.stack(b))

  def condition(i, state):
    cond = tf.math.logical_and(i < max_iterations,
                               tf.math.real(state.sq_norm_res) > tol_sq)

    return (cond if component_wise_distance_fn is None else tf.math.logical_and(
        cond, state.component_wise_distance > 0))

  def body(i, state):
    return i + 1, _conjugate_gradient_step(linear_operator, dot, state)

  iterations, output_state = tf.while_loop(
      cond=condition, body=body, loop_vars=(i0, state0), back_prop=False)

  residual_l2_norm = tf.math.sqrt(output_state.sq_norm_res)
  component_wise_distance = output_state.component_wise_distance
  if input_dtype is not None:
    residual_l2_norm = tf.cast(residual_l2_norm, input_dtype)
    component_wise_distance = tf.cast(component_wise_distance, input_dtype)

  return {
      # Note that the solution vector has specified `internal_dtype`, while the
      # L2 norm has `input_dtype`.
      X: output_state.solution,
      RESIDUAL_L2_NORM: residual_l2_norm,
      COMPONENT_WISE_DISTANCE: component_wise_distance,
      ITERATIONS: iterations,
  }
