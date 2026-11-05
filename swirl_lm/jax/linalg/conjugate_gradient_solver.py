# Copyright 2026 The swirl_lm Authors.
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
"""A JAX library for a distributed conjugate gradient solver.

Uses the conjugate gradient method to iteratively solve a linear system
A x = b, where A is a Hermitian semi-definite operator.

The implementation is "matrix-free", accepting the linear operator `A` as a
`Callable` that evaluates the action of `A` on a given input vector `x`.
In a distributed setting with JAX, the sharding is handled by
`jax.sharding.Mesh`
and the operator `A` is responsible for any required halo exchange.

The inner product is also provided as a `Callable`, typically wrapping
`common_ops.global_dot` with a `jax.sharding.Mesh`.
"""


import functools
from typing import Callable, Optional, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.communication import halo_exchange as halo_ex
from swirl_lm.jax.linalg import base_poisson_solver
from swirl_lm.jax.utility import common_ops
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import types

ScalarField: TypeAlias = types.ScalarField
PoissonSolverSolution: TypeAlias = base_poisson_solver.PoissonSolverSolution

X = base_poisson_solver.X
RESIDUAL_L2_NORM = base_poisson_solver.RESIDUAL_L2_NORM
COMPONENT_WISE_DISTANCE = base_poisson_solver.COMPONENT_WISE_DISTANCE
ITERATIONS = base_poisson_solver.ITERATIONS

# How often to recompute the residual from its definition rather than from the
# recursion `r <- r - alpha A d`. The recursion drifts in finite precision,
# especially in float32: measured against a manufactured solution, asking
# for a relative residual of 1e-10 on a 64^3 grid had the recursive estimate
# reach 8.5e-11 while the true residual was at 1.5e-4.
# Discrepancy grows with the grid size. True residual needs to be recomputed at
# the cost of one extra operator application per interval. The interval has to
# be shorter than a typical solve.
_RESIDUAL_RESTART_INTERVAL = 20

# Type aliases for the linear operator and dot product callables.
LinearOp = Callable[[ScalarField], ScalarField]
DotFn = Callable[[ScalarField, ScalarField], jax.Array]
ComponentWiseDistanceFn = Callable[[ScalarField], jax.Array]

_UNUSED_VALUE = 1.0


def conjugate_gradient_solver(
    linear_operator: LinearOp,
    dot: DotFn,
    b: ScalarField,
    max_iterations: int,
    tol: float,
    x0: ScalarField,
    l2_norm_reduction: bool = False,
    component_wise_distance_fn: Optional[ComponentWiseDistanceFn] = None,
    reprojection: Optional[LinearOp] = None,
    preconditioner: Optional[LinearOp] = None,
    internal_dtype: Optional[jnp.dtype] = None,
) -> PoissonSolverSolution:
  """Solves `[A]{x} = {b}` with the conjugate gradient method.

  Either the number of iterations reaches `max_iterations` or `rho` (the squared
  L2 norm of the residual or its L2 norm reduction relative to `b`) is smaller
  than or equal to `tol` will terminate the iteration.

  Args:
    linear_operator: A `Callable` giving the action of multiplying a Hermitian
      semi-definite operator with a vector (3D `jax.Array`). Must handle halo
      exchange internally, and set halos to zero before returning.
    dot: A `Callable` that computes the inner product of two vectors.
    b: The right-hand-side (rhs) vector (3D `jax.Array`).
    max_iterations: The maximum number of iterations.
    tol: A predefined tolerance for the residual's absolute L2 norm or relative
      L2 norm reduction relative to `b`.
    x0: The initial guess to the solution vector. Halos must be updated before
      calling this function.
    l2_norm_reduction: Whether to use `tol` as a relative L2 norm for the
      residual, relative to the input rhs `b`.
    component_wise_distance_fn: An optional function that takes a `lhs` (`A *
      x`) and returns its componentwise distance to the `rhs` i.e. `b`.
      Convergence is indicated when non-positive. The overall convergence
      criterion is a logical OR of the L2 norm condition and this condition.
    reprojection: A `Callable` that performs reprojection onto the orthogonal
      complement of the null space. Useful for singular systems (e.g., all
      Neumann BCs), where it is equivalent to subtracting off the mean.
    preconditioner: An optional preconditioner M approximating A^{-1}. Applied
      to the residual to improve convergence rate.
    internal_dtype: Optional dtype for internal CG computation (e.g.
      jnp.float64). When set, inputs are cast to this dtype before the CG loop
      and the solution is cast back afterwards. Useful for avoiding numeric
      error accumulation in ill-conditioned problems.

  Returns:
    A dict with the following elements:
      'x': Solution vector.
      'residual_l2_norm': L2 norm for the residual vector.
      'component_wise_distance_from_rhs': Component-wise distance.
      'iterations': Number of iterations used.
  """
  # Cast to internal dtype if specified.
  input_dtype = b.dtype if internal_dtype is not None else None
  if internal_dtype is not None:
    b = b.astype(internal_dtype)
    x0 = x0.astype(internal_dtype)

  x = reprojection(x0) if reprojection else x0

  # Compute the initial residual r = b - A*x.
  r = b - linear_operator(x)
  if reprojection:
    r = reprojection(r)

  # Apply preconditioner: q = M*r or q = r if no preconditioner.
  q = preconditioner(r) if preconditioner else r

  # gamma = <r, q>
  gamma = dot(r, q)

  # Initial search direction d = q.
  d = q

  # Squared norm of residual: rho = gamma if no preconditioner, else <r, r>.
  rho = gamma if preconditioner is None else dot(r, r)

  # Componentwise distance for convergence check.
  if component_wise_distance_fn is not None:
    component_wise_distance = component_wise_distance_fn(r - b)
  else:
    component_wise_distance = jnp.array(_UNUSED_VALUE)

  # Squared tolerance for convergence.
  tol_sq = jnp.array(tol**2)
  if l2_norm_reduction:
    tol_sq = tol_sq * dot(b, b)

  # Pack CG state into a tuple for jax.lax.while_loop.
  # State: (i, r, d, x, rho, gamma, component_wise_distance)
  init_state = (jnp.array(0), r, d, x, rho, gamma, component_wise_distance)

  def cg_cond(state):
    """Checks if the CG iteration should continue."""
    i, _, _, _, rho, _, cw_dist = state
    cond = jnp.logical_and(i < max_iterations, jnp.real(rho) > tol_sq)
    if component_wise_distance_fn is not None:
      cond = jnp.logical_and(cond, cw_dist > 0)
    return cond

  def cg_body(state):
    """One step of conjugate gradient."""
    i, r, d, x, _, gamma, _ = state

    # Compute A*d.
    a_d = linear_operator(d)

    # Step size: alpha = gamma / <d, A*d>.
    alpha = gamma / dot(d, a_d)

    # Update solution and residual.
    x_next = x + alpha * d
    r_next = r - alpha * a_d

    # Periodically recompute the residual from its definition. Without this
    # the iteration stops on an estimate that no longer reflects the true
    # residual, reporting successful convergence while the solution is far from
    # solving the system.
    r_next = jax.lax.cond(
        (i + 1) % _RESIDUAL_RESTART_INTERVAL == 0,
        lambda: b - linear_operator(x_next),
        lambda: r_next,
    )

    if reprojection:
      x_next = reprojection(x_next)
      r_next = reprojection(r_next)

    # Apply preconditioner.
    q_next = preconditioner(r_next) if preconditioner else r_next

    # gamma_next = <r_next, q_next>.
    gamma_next = dot(r_next, q_next)

    # Update search direction: beta = gamma_next / gamma.
    beta = gamma_next / gamma
    d_next = q_next + beta * d

    # Squared norm of residual.
    rho_next = gamma_next if preconditioner is None else dot(r_next, r_next)

    # Componentwise distance.
    if component_wise_distance_fn is not None:
      cw_dist_next = component_wise_distance_fn(r_next - b)
    else:
      cw_dist_next = jnp.array(_UNUSED_VALUE)

    return (i + 1, r_next, d_next, x_next, rho_next, gamma_next, cw_dist_next)

  # Run the CG iteration.
  final_state = jax.lax.while_loop(cg_cond, cg_body, init_state)
  iterations, _, _, x_sol, rho_final, _, cw_dist_final = final_state

  residual_l2_norm = jnp.sqrt(jnp.real(rho_final))

  # Cast back to the original dtype if internal_dtype was used.
  if input_dtype is not None:
    x_sol = x_sol.astype(input_dtype)

  return {
      X: x_sol,
      RESIDUAL_L2_NORM: residual_l2_norm,
      COMPONENT_WISE_DISTANCE: cw_dist_final,
      ITERATIONS: iterations,
  }


class ConjugateGradientSolver(base_poisson_solver.PoissonSolver):
  """A Poisson solver using the Conjugate Gradient method."""

  def __init__(
      self,
      grid_params,
      kernel_op,
      solver_option,
  ):
    """Initializes the CG solver.

    Args:
      grid_params: The grid parametrization.
      kernel_op: An object holding a library of kernel operations.
      solver_option: The option of the selected solver (PoissonSolver proto).
    """
    super().__init__(grid_params, kernel_op, solver_option)
    self._cg_option = solver_option.conjugate_gradient

  def _band_preconditioner(
      self,
      halo_update_fn: Callable[[ScalarField], ScalarField],
  ) -> Callable[[ScalarField], ScalarField]:
    """Builds the band preconditioner the configuration asks for.

    Ported from the TensorFlow implementation in
    `swirl_lm/linalg/poisson_solver.py`. The construction is adapted such
    that it can be applied as a single convolution.

    Let `A_d` be the one-dimensional Laplacian along the direction with the
    smallest grid spacing, and `x` a symmetric band approximation of its
    inverse. The preconditioner is then

        M^-1 = x (1 - O + O^2 - O^3 + ...),   O = (A_j + A_k) x,

    with `j` and `k` the remaining two directions, truncated at
    `taylor_expansion_order`. With no expansion it degenerates to `x` applied
    along the dominating direction alone, or, with `symmetric` set, to the sum
    of `x` applied along all three.

    Args:
      halo_update_fn: The boundary conditions the system is solved against.

    Returns:
      A callable applying `M^-1`.

    Raises:
      ValueError: If the coefficients are not an odd-length, non-trivial
        stencil that fits inside the halos, or if `dominating_direction` is
        not an axis index.
    """
    config = self._cg_option.preconditioner.band_preconditioner
    coefficients = list(config.coefficients)
    halo_width = self._grid_params.halo_width

    # The same constraints the TensorFlow implementation validates, checked
    # here rather than in a separate validator so a bad configuration fails
    # where it is used.
    if len(coefficients) % 2 == 0:
      raise ValueError(
          'The band preconditioner needs an odd number of coefficients so '
          f'that it is centred; got {len(coefficients)}.'
      )
    if len(coefficients) // 2 > halo_width:
      raise ValueError(
          f'The band preconditioner reaches {len(coefficients) // 2} cells, '
          f'further than the halo width of {halo_width}.'
      )
    if sum(abs(c) for c in coefficients) < 1e-6:
      raise ValueError('The band preconditioner coefficients are all zero.')

    axes = ('x', 'y', 'z')
    spacings = self._grid_params.to_xyz_order(self._grid_params.grid_spacings)

    if config.HasField('dominating_direction'):
      dominating = config.dominating_direction
      if dominating not in (0, 1, 2):
        raise ValueError(
            f'dominating_direction must be 0, 1 or 2; got {dominating}.'
        )
    else:
      dominating = int(np.argmin(spacings))

    # Only meaningful when one direction really does dominate: the expansion
    # is in `O`, whose size is set by how much coarser the other two are.
    if config.HasField('dominating_grid_spacing_gap') and spacings[
        dominating
    ] >= sorted(spacings)[1] + config.dominating_grid_spacing_gap:
      raise ValueError(
          'The band preconditioner expands about a dominating direction, but '
          f'the grid spacings {tuple(spacings)} have no clear minimum within '
          f'{config.dominating_grid_spacing_gap}.'
      )

    kernel_op = get_kernel_fn.ApplyKernelConvOp(
        self._grid_params.kernel_size,
        self._grid_params,
        {'band': (coefficients, len(coefficients) // 2)},
    )

    def band(f: ScalarField, indices) -> list[ScalarField]:
      """Applies `x` along each requested axis, scaled by that spacing."""
      return [
          spacings[i] ** 2 * kernel_op.apply_kernel_op(f, 'band', axes[i])
          for i in indices
      ]

    def o_operator(f: ScalarField) -> ScalarField:
      """`O f`, the cross-direction Laplacian of the banded field."""
      banded = band(halo_update_fn(f), [dominating])[0]
      others = [axes[i] for i in range(3) if i != dominating]
      terms = self._laplacian_terms(banded, halo_update=None, axes=others)
      return sum(terms.values())  # pyrefly: ignore[bad-return]

    order = config.taylor_expansion_order

    def preconditioner(f: ScalarField) -> ScalarField:
      # The halos have to be filled before the band filter reads them;
      # without this the operator is not even symmetric, which conjugate
      # gradient cannot survive.
      f = halo_update_fn(f)

      if order < 1:
        if config.symmetric:
          return sum(band(f, (0, 1, 2)))  # pyrefly: ignore[bad-return]
        return band(f, [dominating])[0]

      # Accumulate `f - O f + O^2 f - ...`, smallest terms first, since the
      # powers shrink and adding them in that order loses less.
      powers = []
      current = f
      for _ in range(order):
        current = o_operator(current if not powers else halo_update_fn(current))
        powers.append(current)
      total = jnp.zeros_like(f)
      for power_index in range(len(powers) - 1, -1, -1):
        sign = -1.0 if power_index % 2 == 0 else 1.0
        total = total + sign * powers[power_index]
      return band(halo_update_fn(total + f), [dominating])[0]

    return preconditioner

  def solve(
      self,
      rhs: ScalarField,
      p0: ScalarField,
      mesh: jax.sharding.Mesh,
      halo_update_fn=None,
      additional_states=None,
  ) -> PoissonSolverSolution:
    """Solves the Poisson equation using Conjugate Gradient.

    Args:
      rhs: A 3D array that represents the right hand side tensor.
      p0: A 3D array that provides initial guess.
      mesh: A jax Mesh object representing the device topology.
      halo_update_fn: A function that updates the halo of the input.
      additional_states: Additional static fields needed in the computation.

    Returns:
      A dict with solution, residual L2 norm, componentwise distance, and
      iterations.
    """
    del additional_states  # unused.

    hw = self._grid_params.halo_width

    if halo_update_fn is None:
      halo_update_fn = base_poisson_solver._halo_update_homogeneous_neumann(  # pylint: disable=protected-access
          mesh, self._grid_params
      )

    def linear_op(x: ScalarField) -> ScalarField:
      """Computes the Laplacian of x, with halos cleared."""
      x_with_halos = halo_update_fn(x)
      laplacian = self._laplacian(x_with_halos)
      return halo_ex.set_halos_to_zero(laplacian, hw, self._grid_params)

    dot = functools.partial(common_ops.global_dot, mesh=mesh)

    # Reprojection: subtract the mean, to handle a singular operator whose
    # null space is the constant. Only meaningful when the operator really is
    # singular (i.e., all Neumann boundary conditions). On a problem with a
    # Dirichlet face the mean belongs to the solution and must not be removed.
    reprojection_fn = None
    if self._cg_option.reprojection:

      def reprojection_fn(v: ScalarField) -> ScalarField:  # pylint: disable=function-redefined
        # The halos must be put back to zero. Every vector in the iteration
        # carries zeroed halos, which is what makes `global_dot` an
        # interior-only inner product despite summing the whole local array;
        # subtracting a scalar would leave `-mean` in the halos and corrupt
        # every inner product, and with them `alpha`, `beta` and the
        # convergence test.
        mean = common_ops.global_mean(v, mesh, hw, hw, hw, self._grid_params)
        return halo_ex.set_halos_to_zero(v - mean, hw, self._grid_params)

    # Componentwise convergence check.
    cw_distance_fn = None
    if self._cg_option.HasField('component_wise_convergence'):
      cw_cfg = self._cg_option.component_wise_convergence
      cw_atol = cw_cfg.atol
      cw_rtol = cw_cfg.rtol

      def cw_distance_fn(diff: ScalarField) -> jax.Array:  # pylint: disable=function-redefined
        """Computes the componentwise distance."""
        rhs_cleared = halo_ex.set_halos_to_zero(rhs, hw, self._grid_params)
        tol_field = cw_atol + cw_rtol * jnp.abs(rhs_cleared)
        distance = jnp.abs(diff) - tol_field
        return jax.lax.pmax(jnp.max(distance), axis_name=mesh.axis_names)

    preconditioner_fn = None
    if self._cg_option.HasField(
        'preconditioner'
    ) and self._cg_option.preconditioner.HasField('band_preconditioner'):
      preconditioner_fn = self._band_preconditioner(halo_update_fn)

    # Clear halos on rhs for the solver.
    rhs_cleared = halo_ex.set_halos_to_zero(rhs, hw, self._grid_params)

    # Remove mean from rhs if reprojection is enabled.
    if reprojection_fn is not None:
      rhs_cleared = reprojection_fn(rhs_cleared)

    solution = conjugate_gradient_solver(
        linear_operator=linear_op,
        dot=dot,
        b=rhs_cleared,
        max_iterations=self._cg_option.max_iterations,
        tol=self._cg_option.atol,
        x0=p0,
        l2_norm_reduction=self._cg_option.l2_norm_reduction,
        component_wise_distance_fn=cw_distance_fn,
        reprojection=reprojection_fn,
        preconditioner=preconditioner_fn,
    )

    # Hand back a solution whose halos satisfy the boundary conditions, which
    # is assumed by every caller of `X`. `velocity.correction_step` then takes a
    # centred difference of it which should see a gradient at each halo
    # interface instead of a one-sided jump.
    # `ThreeWeight` and `FastDiagonalization` already return an updated halo.
    solution[X] = halo_update_fn(solution[X])  # pyrefly: ignore[bad-argument-type]
    return solution

  def residual(
      self,
      p: ScalarField,
      rhs: ScalarField,
      mesh: jax.sharding.Mesh,
      additional_states=None,
  ) -> jax.Array:
    """Computes the residual (LHS - RHS) of the Poisson equation.

    Args:
      p: The approximate solution.
      rhs: The right hand side.
      mesh: A jax Mesh object.
      additional_states: Additional fields.

    Returns:
      The residual (Laplacian(p) - rhs).
    """
    halo_update = base_poisson_solver._halo_update_homogeneous_neumann(  # pylint: disable=protected-access
        mesh, self._grid_params
    )
    p_updated = halo_update(p)
    return self._laplacian(p_updated) - rhs
