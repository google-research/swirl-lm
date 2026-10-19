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
"""Multigrid Poisson solver for the JAX framework.

This module provides a geometric multigrid solver for the Poisson equation.
It implements a V-cycle multigrid method with weighted Jacobi smoothing,
standard restriction (full weighting), and prolongation (bilinear
interpolation).

This is the JAX port of `swirl_lm.linalg.multigrid`. The multigrid algorithm
operates on single-device 3D arrays using JAX operations.
"""


import functools
from typing import Callable

import jax
import jax.numpy as jnp
from swirl_lm.jax.linalg import base_poisson_solver
from swirl_lm.jax.linalg import poisson_solver_pb2
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField


def _laplacian(u: jnp.ndarray, spacings: tuple[float, ...]) -> jnp.ndarray:
  """The second-order Laplacian of `u`, with Dirichlet-zero ghosts.

  The shifts are zero padded, which places a homogeneous Dirichlet condition
  one cell outside each face. `u` therefore holds unknowns only: every cell
  is solved for, and nothing in it is pinned.

  Args:
    u: The field, with no halos.
    spacings: The grid spacing along each array axis, in array axis order.

  Returns:
    The Laplacian of `u`.
  """
  lap = jnp.zeros_like(u)
  for axis, spacing in enumerate(spacings):
    forward = jnp.roll(u, -1, axis=axis).at[_last(u, axis)].set(0.0)
    backward = jnp.roll(u, 1, axis=axis).at[_first(axis)].set(0.0)
    lap = lap + (forward - 2.0 * u + backward) / (spacing * spacing)
  return lap


def _first(axis: int):
  """An index selecting the first slice along `axis`."""
  return (slice(None),) * axis + (0,)


def _last(u: jnp.ndarray, axis: int):
  """An index selecting the last slice along `axis`."""
  return (slice(None),) * axis + (u.shape[axis] - 1,)


def _interior_diagonal(
    operator, shape: tuple[int, ...]
) -> jnp.ndarray:
  """The diagonal entry of a translation-invariant stencil operator.

  Weighted Jacobi needs the diagonal of whatever operator it is smoothing. A
  Galerkin operator has no closed form for it, but away from the boundary
  these operators are translation invariant, so one probe suffices: apply the
  operator to a unit impulse and read the value back at the same cell. The
  boundary rows differ, and the damping absorbs that.

  Args:
    operator: The linear operator.
    shape: The grid shape at this level.

  Returns:
    The diagonal entry, as a scalar.
  """
  centre = tuple(extent // 2 for extent in shape)
  impulse = jnp.zeros(shape, jnp.float32).at[centre].set(1.0)
  return operator(impulse)[centre]


def _smooth(
    u: jnp.ndarray,
    b: jnp.ndarray,
    operator,
    diagonal: jnp.ndarray,
    weight: float,
    n_smooth: int,
) -> jnp.ndarray:
  """Applies `n_smooth` weighted Jacobi sweeps against `operator`.

  Args:
    u: The current iterate.
    b: The right hand side.
    operator: The linear operator at this level.
    diagonal: Its diagonal entry, from `_interior_diagonal`.
    weight: The damping factor; 2/3 is the usual choice.
    n_smooth: How many sweeps.

  Returns:
    The smoothed iterate.
  """

  def sweep(_, current):
    return current + weight * (b - operator(current)) / diagonal

  return jax.lax.fori_loop(0, n_smooth, sweep, u)


def _coarsen_operator(error, fine_operator):
  """`R A P`, the Galerkin coarse operator built from `fine_operator`.

  Args:
    error: The coarse field to apply it to.
    fine_operator: The operator one level finer.

  Returns:
    The coarse operator applied to `error`.
  """
  return _restrict(fine_operator(_prolongate(error)))


def _shift_up(u: jnp.ndarray, axis: int) -> jnp.ndarray:
  """`u[i+1]`, with zero beyond the last cell."""
  return jnp.roll(u, -1, axis=axis).at[_last(u, axis)].set(0.0)


def _shift_down(u: jnp.ndarray, axis: int) -> jnp.ndarray:
  """`u[i-1]`, with zero before the first cell."""
  return jnp.roll(u, 1, axis=axis).at[_first(axis)].set(0.0)


def _interleave(even: jnp.ndarray, odd: jnp.ndarray, axis: int) -> jnp.ndarray:
  """Weaves `even` and `odd` back together along `axis`."""
  stacked = jnp.stack([even, odd], axis=axis + 1)
  shape = list(even.shape)
  shape[axis] = even.shape[axis] * 2
  return stacked.reshape(shape)


def _prolongate_axis(coarse: jnp.ndarray, axis: int) -> jnp.ndarray:
  """Interpolates linearly along one axis, doubling its length.

  Cell centred, so each coarse cell covers two fine cells and each of those
  takes three quarters from the cell it sits in and one quarter from the
  neighbour it leans towards. Beyond the ends that neighbour is zero, which
  is the homogeneous Dirichlet condition the rest of the cycle assumes.

  Linear rather than piecewise constant because for Poisson the orders of
  restriction and prolongation must sum to more than two; constant paired
  with averaging sums to exactly two, which leaves the per-cycle convergence
  degrading as the grid grows.

  Args:
    coarse: The coarse field.
    axis: The axis to refine.

  Returns:
    The field with `axis` twice as long.
  """
  lower = 0.75 * coarse + 0.25 * _shift_down(coarse, axis)
  upper = 0.75 * coarse + 0.25 * _shift_up(coarse, axis)
  return _interleave(lower, upper, axis)


def _restrict_axis(fine: jnp.ndarray, axis: int) -> jnp.ndarray:
  """The exact transpose of `_prolongate_axis`, halving the axis.

  Derived rather than assumed. With `lower[i] = 3/4 c[i] + 1/4 c[i-1]` and
  `upper[i] = 3/4 c[i] + 1/4 c[i+1]`, gathering the coefficient of `c[i]`
  in `<f, P c>` gives

      (P^T f)[i] = 3/4 (lower[i] + upper[i])
                 + 1/4 lower[i+1] + 1/4 upper[i-1].

  Args:
    fine: The fine field. `axis` must have even length.
    axis: The axis to coarsen.

  Returns:
    The field with `axis` halved.
  """
  lower = jax.lax.slice_in_dim(fine, 0, fine.shape[axis], stride=2, axis=axis)
  upper = jax.lax.slice_in_dim(fine, 1, fine.shape[axis], stride=2, axis=axis)
  return (
      0.75 * (lower + upper)
      + 0.25 * _shift_up(lower, axis)
      + 0.25 * _shift_down(upper, axis)
  )


def _restrict(fine: jnp.ndarray) -> jnp.ndarray:
  """Restricts by transposing the prolongation along every axis.

  Being the exact transpose is what the coarse-grid correction rests on: it
  makes the correction minimise the error over the coarse space rather than
  merely change it. The operators this replaced were plain injection against
  an interpolating prolongation, giving ratios of 1.12, 0.068 and 4.47 on
  fine grids of 8, 16 and 32 where a consistent pair gives one constant.

  Args:
    fine: The fine field, every axis of even length.

  Returns:
    The coarse field, half the length along each axis.
  """
  coarse = fine
  for axis in range(3):
    coarse = _restrict_axis(coarse, axis)
  return coarse


def _prolongate(coarse: jnp.ndarray) -> jnp.ndarray:
  """Interpolates onto the fine grid along every axis.

  Args:
    coarse: The coarse field.

  Returns:
    The fine field, twice the length along each axis.
  """
  fine = coarse
  for axis in range(3):
    fine = _prolongate_axis(fine, axis)
  return fine


class MultigridSolver(base_poisson_solver.PoissonSolver):
  """Multigrid Poisson solver using V-cycle with Jacobi smoothing."""

  def __init__(
      self,
      grid_params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
  ):
    """Initializes the multigrid solver.

    Args:
      grid_params: The grid parametrization.
      kernel_op: An object holding a library of kernel operations.
      solver_option: The solver configuration proto.
    """
    super().__init__(grid_params, kernel_op, solver_option)
    mg_config = solver_option.multigrid

    self._num_iterations = mg_config.num_iterations
    self._n_smooth = mg_config.n_smooth
    self._weight = mg_config.weight
    self._n_coarse = mg_config.n_coarse

    # Coarsest subgrid shape (stop coarsening when any dim <= this).
    if mg_config.HasField('coarsest_subgrid_shape'):
      cs = mg_config.coarsest_subgrid_shape
      self._coarsest_shape = (cs.dim_z, cs.dim_x, cs.dim_y)
    else:
      self._coarsest_shape = (4, 4, 4)

    self._use_a_inv = mg_config.use_a_inv

    # Kept in array axis order rather than named z, x, y: `grid_spacings` is
    # already in `data_axis_order`, so axis 0 is whichever axis comes first,
    # not necessarily z. Naming them otherwise invites a transposition that
    # would be invisible on a cubic grid.
    self._spacings = tuple(grid_params.grid_spacings)

  def _can_coarsen(self, shape: tuple[int, ...]) -> bool:
    """Whether every axis can be halved and still exceed the coarsest shape.

    Args:
      shape: The current grid shape, unknowns only.

    Returns:
      True if the grid can be coarsened.
    """
    return all(
        extent % 2 == 0 and extent // 2 >= coarsest
        for extent, coarsest in zip(shape, self._coarsest_shape)
    )

  def _hierarchy(self, shape: tuple[int, ...]):
    """Builds the operator for every level, coarsest last.

    The coarse operators are Galerkin, `A_coarse = R A_fine P`, rather than
    the Laplacian rediscretised at twice the spacing. Rediscretisation is
    only consistent when it happens to agree with the Galerkin product, which
    it does for a piecewise constant transfer pair, where the two differ by
    exactly a factor of two and the stencils match. It does not agree for a
    linear prolongation: there the Galerkin stencil is wider than three
    points, the measured ratio drifts with the grid -- 1.41 at 16^3 against
    1.58 at 32^3 -- and no scalar can reconcile them. Defining the coarse
    operator by the transfers removes the question.

    Args:
      shape: The finest grid shape, unknowns only.

    Returns:
      A list of `(shape, operator, diagonal)`, finest first.
    """
    levels = []
    operator = functools.partial(_laplacian, spacings=self._spacings)
    current = shape
    while True:
      levels.append((current, operator, _interior_diagonal(operator, current)))
      if not self._can_coarsen(current):
        return levels
      operator = functools.partial(_coarsen_operator, fine_operator=operator)
      current = tuple(extent // 2 for extent in current)

  def _v_cycle(self, levels, index: int, u, b):
    """Performs one V-cycle starting at `levels[index]`.

    Args:
      levels: The hierarchy from `_hierarchy`.
      index: Which level to start on, 0 being finest.
      u: The current iterate.
      b: The right hand side.

    Returns:
      The iterate after one cycle.
    """
    _, operator, diagonal = levels[index]
    u = _smooth(u, b, operator, diagonal, self._weight, self._n_smooth)

    if index + 1 == len(levels):
      # Nothing coarser to fall back on, so smooth harder instead.
      return _smooth(
          u, b, operator, diagonal, self._weight, self._n_smooth * 4
      )

    coarse_residual = _restrict(b - operator(u))
    coarse_error = jnp.zeros_like(coarse_residual)
    for _ in range(max(self._n_coarse, 1)):
      coarse_error = self._v_cycle(
          levels, index + 1, coarse_error, coarse_residual
      )

    u = u + _prolongate(coarse_error)
    return _smooth(u, b, operator, diagonal, self._weight, self._n_smooth)

  def solve(
      self,
      rhs: ScalarField,
      p0: ScalarField,
      mesh: jax.sharding.Mesh,
      halo_update_fn: Callable[[ScalarField], ScalarField] | None = None,
      additional_states: dict[str, jax.Array] | None = None,
  ) -> base_poisson_solver.PoissonSolverSolution:
    """Solves the Poisson equation with multigrid V-cycles.

    The halos are stripped on the way in and restored on the way out, so the
    cycle works on unknowns alone. The boundary condition is homogeneous
    Dirichlet one cell outside each face, which is what the zero-padded
    shifts in `_laplacian` impose; `halo_update_fn` is not yet honoured, so a
    case with Neumann faces is solving the wrong problem and a case on more
    than one device is solving several unconnected ones.

    Args:
      rhs: The right hand side.
      p0: The initial guess.
      mesh: The device mesh. Not yet used.
      halo_update_fn: The boundary conditions. Not yet used.
      additional_states: Not used.

    Returns:
      The solution, the residual norm and the cycle count.
    """
    del mesh, halo_update_fn, additional_states

    halo = self._grid_params.halo_width
    interior = (slice(halo, -halo),) * 3
    b = rhs[interior]
    x = p0[interior]

    levels = self._hierarchy(b.shape)
    for _ in range(self._num_iterations):
      x = self._v_cycle(levels, 0, x, b)

    residual = b - _laplacian(x, self._spacings)
    return {
        base_poisson_solver.X: jnp.pad(x, ((halo, halo),) * 3),
        base_poisson_solver.RESIDUAL_L2_NORM: jnp.sqrt(
            jnp.sum(residual**2)
        ),
        base_poisson_solver.ITERATIONS: jnp.asarray(
            self._num_iterations, jnp.int32
        ),
    }
