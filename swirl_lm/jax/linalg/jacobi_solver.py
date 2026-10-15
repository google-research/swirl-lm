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
"""Jacobi-based Poisson solver with stretched grid support (JAX port).

Port of `swirl_lm.linalg.jacobi_solver.ThreeWeightForPressure`. Wraps the
low-level `ThreeWeight` Jacobi implementation with metric-weight generation
for the pressure correction equation on stretched grids.

For a Low-Mach formulation on a stretched grid, the Poisson equation
  nabla^2 p = rhs
is reformulated in computational coordinates with scale factors h0, h1, h2:
  w0 = h1*h2/h0,  w1 = h0*h2/h1,  w2 = h0*h1/h2
  rhs_modified = rhs * h0*h1*h2

For the anelastic case, each weight is additionally multiplied by a reference
density rho_0.
"""


from typing import Callable

import jax
from jax import sharding
import jax.numpy as jnp
from swirl_lm.jax.linalg import base_poisson_solver
from swirl_lm.jax.linalg import jacobi_solver_impl
from swirl_lm.jax.linalg import poisson_solver_pb2
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import stretched_grid_util
from swirl_lm.jax.utility import types

PoissonSolverSolution = base_poisson_solver.PoissonSolverSolution
ScalarField = types.ScalarField

_HaloUpdateFn = Callable[[ScalarField], ScalarField]


def _generate_weights_and_modified_rhs(
    rhs: ScalarField,
    additional_states: dict[str, jax.Array],
    use_stretched_grid: tuple[bool, ...],
    grid_params: grid_parametrization.GridParametrization | None = None,
) -> tuple[ScalarField, ScalarField, ScalarField, ScalarField]:
  """Generates the 3 metric-weight coefficients and modified RHS.

  For each dimension, if stretched grid is used, the scale factor h is
  retrieved from `additional_states`. Otherwise, h = 1 (uniform).

  The weight coefficients are:
    w0 = h1 * h2 / h0
    w1 = h0 * h2 / h1
    w2 = h0 * h1 / h2

  The RHS is modified: rhs_modified = rhs * h0 * h1 * h2.

  Args:
    rhs: The right-hand side of the pressure equation.
    additional_states: Dict containing stretched grid scale factors.
    use_stretched_grid: Whether each dimension uses stretched grid.
    grid_params: Optional grid parametrization for data_axis_order.

  Returns:
    Tuple (w0, w1, w2, rhs_modified).
  """
  data_order = (
      list(grid_params.data_axis_order)
      if grid_params is not None
      else ['z', 'x', 'y']
  )
  h = []
  for dim in range(3):
    if use_stretched_grid[dim]:
      physical_dim = ('x', 'y', 'z').index(data_order[dim])
      h_key = stretched_grid_util.h_key(physical_dim)
      h.append(additional_states[h_key])
    else:
      # For uniform dimensions, h = 1 (unit scale factor). Must be a full
      # 3D array because ThreeWeight applies kernel ops on the weights.
      h.append(jnp.ones_like(rhs))

  h0, h1, h2 = h  # pylint: disable=unbalanced-tuple-unpacking
  w0 = h1 * h2 / h0
  w1 = h0 * h2 / h1
  w2 = h0 * h1 / h2

  rhs_modified = rhs * h0 * h1 * h2

  # For the anelastic case, multiply weights by the reference density.
  if base_poisson_solver.VARIABLE_COEFF in additional_states:
    rho_0 = additional_states[base_poisson_solver.VARIABLE_COEFF]
    w0 = w0 * rho_0
    w1 = w1 * rho_0
    w2 = w2 * rho_0

  return w0, w1, w2, rhs_modified


class JacobiSolver(base_poisson_solver.PoissonSolver):
  """Jacobi Poisson solver with stretched grid support.

  Wraps `jacobi_solver_impl.ThreeWeight` with metric-weight generation for
  the pressure correction equation. On uniform grids, the weights reduce to
  identity (w0 = w1 = w2 = 1) and the solver is equivalent to a standard
  Jacobi iteration.
  """

  def __init__(
      self,
      grid_params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
      use_stretched_grid: tuple[bool, ...] = (False, False, False),
  ):
    super().__init__(grid_params, kernel_op, solver_option)
    self._use_stretched_grid = use_stretched_grid
    # Pre-register the `weighted_sum_121` kernel that ThreeWeight needs.
    # This must happen outside JAX tracing (shard_map/jit) to avoid tracer
    # leaks. When ThreeWeight.__init__ later calls add_kernel, the kernel
    # is already registered and the call is a no-op.
    if isinstance(kernel_op, get_kernel_fn.ApplyKernelConvOp):
      kernel_op.add_kernel({'weighted_sum_121': ([1.0, 2.0, 1.0], 1)})
    elif isinstance(kernel_op, get_kernel_fn.ApplyKernelSliceOp):
      kernel_op.add_kernel(
          {'weighted_sum_121': {'coeff': [1.0, 2.0, 1.0], 'shift': [-1, 0, 1]}}  # pyrefly: ignore[bad-assignment]
      )
    self._three_weight_solver: jacobi_solver_impl.ThreeWeight | None = None

  def solve(
      self,
      rhs: ScalarField,
      p0: ScalarField,
      mesh: sharding.Mesh,
      halo_update_fn: _HaloUpdateFn | None = None,
      additional_states: dict[str, jax.Array] | None = None,
  ) -> PoissonSolverSolution:
    """Solves the Poisson equation using the Jacobi method.

    Args:
      rhs: The right-hand side of the Poisson equation.
      p0: Initial guess for the solution.
      mesh: Device mesh for distributed computation.
      halo_update_fn: Function to update halos and enforce BCs.
      additional_states: Dict with stretched grid scale factors and optional
        variable coefficient (for anelastic).

    Returns:
      Dict with solution and iteration count.
    """
    if additional_states is None:
      additional_states = {}

    w0, w1, w2, rhs_mod = _generate_weights_and_modified_rhs(
        rhs, additional_states, self._use_stretched_grid, self._grid_params
    )

    if halo_update_fn is None:
      halo_update_fn = base_poisson_solver._halo_update_homogeneous_neumann(  # pylint: disable=protected-access
          mesh, self._grid_params
      )

    if self._three_weight_solver is None:
      self._three_weight_solver = jacobi_solver_impl.ThreeWeight(
          self._grid_params,
          self._kernel_op,
          self._solver_option,
          mesh,
      )

    return self._three_weight_solver.solve(
        w0, w1, w2, rhs_mod, p0, halo_update_fn
    )

  def residual(
      self,
      p: ScalarField,
      rhs: ScalarField,
      mesh: sharding.Mesh,
      additional_states: dict[str, jax.Array] | None = None,
  ) -> jax.Array:
    """Computes the residual of the Poisson equation.

    Args:
      p: Approximate solution.
      rhs: Right-hand side.
      mesh: Device mesh.
      additional_states: Dict with stretched grid scale factors.

    Returns:
      The residual (LHS - RHS).
    """
    if additional_states is None:
      additional_states = {}

    w0, w1, w2, rhs_mod = _generate_weights_and_modified_rhs(
        rhs, additional_states, self._use_stretched_grid, self._grid_params
    )

    if self._three_weight_solver is None:
      self._three_weight_solver = jacobi_solver_impl.ThreeWeight(
          self._grid_params,
          self._kernel_op,
          self._solver_option,
          mesh,
      )

    return self._three_weight_solver.residual(p, w0, w1, w2, rhs_mod)
