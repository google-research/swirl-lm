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
"""Residual norm per iteration tests for poisson_solver.py."""

import itertools
import os

import numpy as np
from swirl_lm.linalg import base_poisson_solver
from swirl_lm.linalg import multigrid_utils
from swirl_lm.linalg import poisson_solver_pb2
from swirl_lm.linalg import poisson_solver_testutil
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization_pb2
import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.framework import util
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized

_X = base_poisson_solver.X


class PoissonSolverResidualsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._write_dir = (os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'))

  _COMPUTATION_SHAPES = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
                         (2, 2, 1), (2, 1, 2), (2, 2, 1))
  _SOLVER_AND_NEUMANN_AND_EXPECTED_RESIDUAL_NORMS_AND_RTOL = (
      # In multigrid different computation shapes give rise to different
      # prolongation/restriction matrices, so the residual norm values are only
      # approximately equal.
      ('Multigrid', False, (46, 24, 13, 6.8, 3.8), 0.08),
      ('Multigrid', True, (48.6, 33.0, 22.9, 16.1, 11.5), 0.02),
  )

  @parameterized.parameters(*itertools.product(
      _COMPUTATION_SHAPES,
      _SOLVER_AND_NEUMANN_AND_EXPECTED_RESIDUAL_NORMS_AND_RTOL))
  def testResidualNormsPerIteration_(
      self,
      computation_shape,
      solver_and_nemuann_and_expected_residual_norms):
    solver_name, neumann, expected_residual_norms, rtol = (
        solver_and_nemuann_and_expected_residual_norms)
    halo_width = 1  # Multigrid only supports halo width 1.
    runner = TpuRunner(computation_shape=computation_shape)
    replicas = runner.replicas
    num_iterations = 5
    fx, fy, fz = 18, 20, 22
    nx, ny, nz = [(f - 2) // c + 2 for f, c in zip((fx, fy, fz),
                                                   computation_shape)]
    lx = ly = lz = 2.0 * np.pi

    if neumann:
      bc_type = grid_parametrization_pb2.BC_TYPE_NEUMANN
    else:
      bc_type = grid_parametrization_pb2.BC_TYPE_DIRICHLET

    def rhs_fn(xx, yy, zz, lx, ly, lz, coord):
      """Defines the right hand side tensor."""
      del lx, ly, lz, coord  # Not used.
      if neumann:
        return -3. * tf.math.cos(xx) * tf.math.cos(yy) * tf.math.cos(zz)
      else:
        return -3. * tf.math.sin(xx) * tf.math.sin(yy) * tf.math.sin(zz)

    solver_option = poisson_solver_pb2.PoissonSolver()

    if solver_name == 'Multigrid':
      solver_option.multigrid.coarsest_subgrid_shape.dim_0 = 3
      solver_option.multigrid.coarsest_subgrid_shape.dim_1 = 3
      solver_option.multigrid.coarsest_subgrid_shape.dim_2 = 3
      solver_option.multigrid.boundary_condition.dim_0 = bc_type
      solver_option.multigrid.boundary_condition.dim_1 = bc_type
      solver_option.multigrid.boundary_condition.dim_2 = bc_type
      solver_option.multigrid.use_a_inv = True
      solver_option.multigrid.num_iterations = 1

    solver = poisson_solver_testutil.PoissonSolverRunner(
        get_kernel_fn.ApplyKernelSliceOp(),
        rhs_fn,
        replicas,
        nx, ny, nz,
        lx, ly, lz,
        halo_width,
        solver_option)

    coordinates = util.grid_coordinates(computation_shape)
    replicas_range = range(np.prod(computation_shape))
    states = [solver.init_fn_tf2(i, coordinates[i])
              for i in replicas_range]

    if solver_name == 'Multigrid':
      b = util.combine_subgrids(
          [tf.transpose(states[i]['b_minus_a_xb'], (1, 2, 0))
           for i in replicas_range], replicas).numpy()

    def near_exact_solution(xx, yy, zz):
      # Note this is not the exact solution due to boundary effects. This is
      # very close to the expected solution, and is shown as "expected" in the
      # 2D slice plots.
      if neumann:
        return np.cos(xx) * np.cos(yy) * np.cos(zz)
      else:
        return np.sin(xx) * np.sin(yy) * np.sin(zz)

    xl = np.linspace(0, lx, fx)
    yl = np.linspace(0, ly, fy)
    zl = np.linspace(0, lz, fz)
    xx, yy, zz = np.meshgrid(xl, yl, zl, indexing='ij')
    near_exact_solution_np = near_exact_solution(xx, yy, zz)

    actual_residual_norms = []
    for _ in range(num_iterations):
      tpu_res = runner.run_with_replica_args(solver.step_fn_tf2, states)

      if solver_name == 'Multigrid':
        for i in replicas_range:
          states[i]['x0'] = np.stack(tpu_res[i][_X])
        # Normally x = x0 + xb, but xb is 0 in these homogeneous boundary
        # condition tests.
        x = util.combine_subgrids(
            [np.transpose(states[i]['x0'], (1, 2, 0)) for i in replicas_range],
            replicas)
        l2_norm = multigrid_utils.poisson_residual_norm(x, b, solver.params)
        actual_residual_norms.append(l2_norm)

    additional_title = 'Neumann' if neumann else 'Dirichlet'
    additional_file_name = 'neumann' if neumann else 'dirichlet'
    slice_divisor = 2 if neumann else 4
    poisson_solver_testutil.create_2d_slice_pngs(
        near_exact_solution_np, x, xl, yl, zl, solver.params, self._write_dir,
        solver_name, l2_norm, slice_divisor, additional_title,
        additional_file_name)

    self.assertAllClose(expected_residual_norms, actual_residual_norms,
                        rtol=rtol)


if __name__ == '__main__':
  tf.test.main()
