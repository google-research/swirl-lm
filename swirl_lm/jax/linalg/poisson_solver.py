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
"""Factory for creating Poisson solvers based on proto configuration."""


from swirl_lm.jax.linalg import base_poisson_solver
from swirl_lm.jax.linalg import conjugate_gradient_solver
from swirl_lm.jax.linalg import fast_diagonalization_solver
from swirl_lm.jax.linalg import jacobi_solver
from swirl_lm.jax.linalg import multigrid_solver
from swirl_lm.jax.linalg import poisson_solver_pb2
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization

PoissonSolver = base_poisson_solver.PoissonSolver


def poisson_solver_factory(
    grid_params: grid_parametrization.GridParametrization,
    kernel_op: get_kernel_fn.ApplyKernelOp,
    solver_option: poisson_solver_pb2.PoissonSolver,
    use_stretched_grid: tuple[bool, ...] = (False, False, False),
) -> PoissonSolver:
  """Creates a Poisson solver based on the proto configuration.

  Args:
    grid_params: The grid parametrization.
    kernel_op: An object holding a library of kernel operations.
    solver_option: The solver configuration proto.
    use_stretched_grid: Whether each dimension uses stretched grid.

  Returns:
    A PoissonSolver instance configured for the specified solver type.

  Raises:
    ValueError: If no supported solver type is found in the proto.
  """
  if solver_option.HasField('jacobi'):
    return jacobi_solver.JacobiSolver(
        grid_params, kernel_op, solver_option, use_stretched_grid
    )
  elif solver_option.HasField('conjugate_gradient'):
    return conjugate_gradient_solver.ConjugateGradientSolver(
        grid_params, kernel_op, solver_option
    )
  elif solver_option.HasField('fast_diagonalization'):
    return fast_diagonalization_solver.FastDiagonalizationSolver(
        grid_params, kernel_op, solver_option
    )
  elif solver_option.HasField('multigrid'):
    return multigrid_solver.MultigridSolver(
        grid_params, kernel_op, solver_option, use_stretched_grid
    )
  else:
    raise ValueError(
        'Unsupported Poisson solver type. Supported types: jacobi,'
        ' conjugate_gradient, fast_diagonalization, multigrid.'
    )
