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


def _smooth(
    u: jnp.ndarray,
    b: jnp.ndarray,
    spacings: tuple[float, ...],
    weight: float,
    n_smooth: int,
) -> jnp.ndarray:
  """Applies `n_smooth` weighted Jacobi sweeps.

  Args:
    u: The current iterate.
    b: The right hand side.
    spacings: Grid spacing per array axis.
    weight: The damping factor; 2/3 is the usual choice.
    n_smooth: How many sweeps.

  Returns:
    The smoothed iterate.
  """
  diagonal = -2.0 * sum(1.0 / (s * s) for s in spacings)

  def sweep(_, current):
    return current + weight * (b - _laplacian(current, spacings)) / diagonal

  return jax.lax.fori_loop(0, n_smooth, sweep, u)


def _restrict(fine: jnp.ndarray) -> jnp.ndarray:
  """Averages each 2x2x2 block onto one coarse cell.

  Paired with `_prolongate` this is an exact adjoint up to the constant
  `1 / 2**3`: averaging is the transpose of copying, scaled by the block
  size.

  Args:
    fine: The fine field. Every axis must have even length.

  Returns:
    The coarse field, half the length along each axis.
  """
  shape = fine.shape
  blocked = fine.reshape(
      shape[0] // 2, 2, shape[1] // 2, 2, shape[2] // 2, 2
  )
  return jnp.mean(blocked, axis=(1, 3, 5))


def _prolongate(coarse: jnp.ndarray) -> jnp.ndarray:
  """Copies each coarse cell onto the eight fine cells beneath it.

  Piecewise constant, so that it is exactly the transpose of `_restrict`.
  Trilinear interpolation converges faster per cycle but is only
  approximately adjoint, which is the property worth having first.

  Args:
    coarse: The coarse field.

  Returns:
    The fine field, twice the length along each axis.
  """
  fine = coarse
  for axis in range(3):
    fine = jnp.repeat(fine, 2, axis=axis)
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

  def _v_cycle(
      self,
      u: jnp.ndarray,
      b: jnp.ndarray,
      spacings: tuple[float, ...],
  ) -> jnp.ndarray:
    """Performs one V-cycle.

    Smooth, take the residual to the coarse grid, solve for the error there,
    bring it back and smooth again. The coarse operator is the Laplacian
    rediscretised at twice the spacing rather than a Galerkin product, which
    is consistent here because the transfer pair is adjoint.

    Args:
      u: The current iterate, unknowns only.
      b: The right hand side.
      spacings: Grid spacing per array axis at this level.

    Returns:
      The iterate after one cycle.
    """
    u = _smooth(u, b, spacings, self._weight, self._n_smooth)

    if not self._can_coarsen(u.shape):
      # Nothing coarser to fall back on, so smooth harder instead. This is
      # the one level where the error has no cheaper representation.
      return _smooth(u, b, spacings, self._weight, self._n_smooth * 4)

    # The coarse problem is solved with the Laplacian rediscretised at twice
    # the spacing, which is not the Galerkin operator this transfer pair
    # implies. Measured, `R A_h P` comes to exactly twice `A_2h`, uniformly
    # over axes and grid sizes, because averaging a piecewise constant
    # prolongation loses a factor of two per dimension in the same way for
    # each. Halving the restricted residual is equivalent to solving against
    # the Galerkin operator, and without it every level over-corrects by two
    # and the cycle diverges rather than converges.
    coarse_residual = 0.5 * _restrict(b - _laplacian(u, spacings))
    coarse_spacings = tuple(2.0 * s for s in spacings)

    coarse_error = jnp.zeros_like(coarse_residual)
    for _ in range(max(self._n_coarse, 1)):
      coarse_error = self._v_cycle(
          coarse_error, coarse_residual, coarse_spacings
      )

    u = u + _prolongate(coarse_error)
    return _smooth(u, b, spacings, self._weight, self._n_smooth)

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

    for _ in range(self._num_iterations):
      x = self._v_cycle(x, b, self._spacings)

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
