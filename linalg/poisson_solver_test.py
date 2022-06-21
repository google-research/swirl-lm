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
"""Tests for google3.research.simulation.tensorflow.fluid.framework.poisson_solver."""

import itertools
import os

import numpy as np
from swirl_lm.linalg import base_poisson_solver
from swirl_lm.linalg import poisson_solver
from swirl_lm.linalg import poisson_solver_pb2
from swirl_lm.linalg import poisson_solver_testutil
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import grid_parametrization_pb2
import tensorflow as tf

from google3.pyglib import gfile
from google3.research.simulation.tensorflow.fluid.framework import util
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized

_X = base_poisson_solver.X
_RESIDUAL_L2_NORM = base_poisson_solver.RESIDUAL_L2_NORM
_ITERATIONS = base_poisson_solver.ITERATIONS


def get_kernel_op(name):
  if name == 'ApplyKernelConvOp':
    return get_kernel_fn.ApplyKernelConvOp(4)
  elif name == 'ApplyKernelSliceOp':
    return get_kernel_fn.ApplyKernelSliceOp()

  return None


class PoissonSolverTest(tf.test.TestCase, parameterized.TestCase):

  _L2_NORM_01 = 3.0e-06
  _L2_NORM_02 = 6.0e-06

  # TODO(b/190265551): There are some issues with partitions and iterations,
  # ideally L2 norm and number of iterations should depend on the actual problem
  # size only, and independent of number of partitions, halo_width, etc.
  _ITERATIONS_01 = (53, 61)
  _ITERATIONS_02 = 100

  # Actual total problem size is all `24 x 24 x 24`, with different halo_width &
  # partitions, the args is in the format of:
  # (replicas, halo_width, (nx, ny, nz)), where n_{xyz} is the size of one core.
  _REPLICAS_AND_PARTITIONS = (
      # One single partition.
      (np.array([[[0]]], dtype=np.int32), 2, (28,) * 3),
      (np.array([[[0]]], dtype=np.int32), 4, (32,) * 3),
      (np.array([[[0]]], dtype=np.int32), 6, (36,) * 3),
      # Two partitions.
      (np.array([[[0, 1]]], dtype=np.int32), 2, (16, 28, 28)),
      (np.array([[[0]], [[1]]], dtype=np.int32), 2, (28, 16, 28)),
      (np.array([[[0], [1]]], dtype=np.int32), 2, (28, 28, 16)),
  )

  _SOLVER_NAMES_AND_EXPECTED = (
      ('Jacobi', None, 100),
      ('Fast Diagonalization', None, None),
      ('Conjugate Gradient', (_L2_NORM_01, _L2_NORM_02), (_ITERATIONS_01,
                                                          _ITERATIONS_02)),
  )

  def setUp(self):
    super(PoissonSolverTest, self).setUp()

    self._write_dir = (os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'))

  @parameterized.named_parameters(
      ('Periodic', (grid_parametrization_pb2.BC_TYPE_PERIODIC,
                    grid_parametrization_pb2.BC_TYPE_PERIODIC),
       [[-8, 4, 0, 4], [4, -8, 4, 0], [0, 4, -8, 4], [4, 0, 4, -8]]),
      ('Neumann', (grid_parametrization_pb2.BC_TYPE_NEUMANN,
                   grid_parametrization_pb2.BC_TYPE_NEUMANN),
       [[-4, 4, 0, 0], [4, -8, 4, 0], [0, 4, -8, 4], [0, 0, 4, -4]]),
      ('Dirichlet', (grid_parametrization_pb2.BC_TYPE_DIRICHLET,
                     grid_parametrization_pb2.BC_TYPE_DIRICHLET),
       [[-8, 4, 0, 0], [4, -8, 4, 0], [0, 4, -8, 4], [0, 0, 4, -8]]),
      ('Neumann+Dirichlet', (grid_parametrization_pb2.BC_TYPE_NEUMANN,
                             grid_parametrization_pb2.BC_TYPE_DIRICHLET),
       [[-4, 4, 0, 0], [4, -8, 4, 0], [0, 4, -8, 4], [0, 0, 4, -8]]),
      ('Dirichlet+Neumann', (grid_parametrization_pb2.BC_TYPE_DIRICHLET,
                             grid_parametrization_pb2.BC_TYPE_NEUMANN),
       [[-8, 4, 0, 0], [4, -8, 4, 0], [0, 4, -8, 4], [0, 0, 4, -4]]),
  )
  def testLaplacianMatrixWithPeriodicBcIsConstructedCorrectly(
      self, bc_type, expected):
    """Checks if the Laplacian matrix with periodic BC is correct."""
    n = 4
    h = 0.5

    res = poisson_solver._make_laplacian_matrix(n, h, bc_type)

    self.assertAllEqual(expected, res)

  def testLaplacianMatrixWithIncorrectPeriodicBCSpecificationReturnsError(self):
    """Checks if error is triggered if boundary condition is incompatible."""
    with self.assertRaisesRegex(ValueError,
                                'Periodic boundary condition is ambiguous'):
      _ = poisson_solver._make_laplacian_matrix(
          4, 0.5, (grid_parametrization_pb2.BC_TYPE_PERIODIC,
                   grid_parametrization_pb2.BC_TYPE_NEUMANN))

  @parameterized.parameters(*itertools.product(
      _REPLICAS_AND_PARTITIONS,
      _SOLVER_NAMES_AND_EXPECTED,
      (
          # (internal_dtype, expected_x_dtype, expected_l2_norm_dtype,
          #  kernel_op_name)
          (None, np.float32, np.float32, 'ApplyKernelConvOp'),
          (tf.float32, np.float32, np.float32, 'ApplyKernelConvOp'),
          (tf.float64, np.float64, np.float32, 'ApplyKernelSliceOp'),
      )))
  def testPoissonSolversSolveProblemCorrectly(self, replicas_and_partitions,
                                              solver_name_and_expected,
                                              dtypes_and_kernel):

    (internal_dtype, expected_x_dtype, expected_l2_norm_dtype,
     kernel_op) = dtypes_and_kernel

    def exact_solution(xx, yy, zz):
      """Define the solution in a form that its boundary is homogeneous."""
      return np.sin(xx) * np.sin(yy) * np.sin(zz)

    def rhs_fn(xx, yy, zz, lx, ly, lz, coord):
      """Defines the right hand side tensor."""
      del lx, ly, lz, coord  # Not used.
      return -3. * tf.math.sin(xx) * tf.math.sin(yy) * tf.math.sin(zz)

    replicas, halo_width, xyz = replicas_and_partitions
    computation_shape = np.array(replicas.shape)

    nx, ny, nz = xyz
    lx = ly = lz = 2.0 * np.pi

    (solver_name, expected_l2_norm,
     expected_iterations) = solver_name_and_expected
    solver_option = poisson_solver_pb2.PoissonSolver()
    if solver_name == 'Jacobi':
      solver_option.jacobi.max_iterations = 100
      solver_option.jacobi.halo_width = halo_width
    elif solver_name == 'Fast Diagonalization':
      solver_option.fast_diagonalization.halo_width = halo_width
      solver_option.fast_diagonalization.cutoff = 1e-16
    elif solver_name == 'Conjugate Gradient':
      solver_option.conjugate_gradient.max_iterations = 100
      solver_option.conjugate_gradient.halo_width = halo_width
      solver_option.conjugate_gradient.atol = 3e-6
      solver_option.conjugate_gradient.reprojection = False

      index = int(np.prod(computation_shape)) - 1
      expected_l2_norm = expected_l2_norm[index]
      expected_iterations = expected_iterations[index]

    solver = poisson_solver_testutil.PoissonSolverRunner(
        get_kernel_op(kernel_op),
        rhs_fn,
        replicas,
        nx, ny, nz,
        lx, ly, lz,
        halo_width,
        solver_option,
        internal_dtype)

    coordinates = util.grid_coordinates(computation_shape)
    runner = TpuRunner(computation_shape=computation_shape)
    states = [solver.init_fn_tf2(i, coordinates[i])
              for i in range(np.prod(computation_shape))]
    tpu_res = runner.run_with_replica_args(solver.step_fn_tf2, states)
    poisson_solver_solution = tpu_res[0]
    l2_norm = poisson_solver_solution.get(_RESIDUAL_L2_NORM)
    iterations = poisson_solver_solution.get(_ITERATIONS)

    tf.compat.v1.logging.info(
        '(solver, (nx, ny, nz), halo_width) = (%s, (%d, %d, %d), %d): '
        '(l2_norm, iterations) = (%s, %s)', solver_name, nx, ny, nz, halo_width,
        str(l2_norm), str(iterations))

    if expected_l2_norm is None:
      self.assertIsNone(l2_norm)
    else:
      self.assertIsInstance(l2_norm, expected_l2_norm_dtype)
      self.assertLess(l2_norm, expected_l2_norm)

    if expected_iterations is None:
      self.assertIsNone(iterations)
    elif isinstance(expected_iterations, int):
      self.assertIsInstance(iterations, np.int32)
      self.assertEqual(expected_iterations, iterations)
    else:
      self.assertIsInstance(iterations, np.int32)
      self.assertIn(iterations, expected_iterations)

    tpu_res = [
        np.transpose(np.stack(tpu_res_i[_X]), [1, 2, 0])
        for tpu_res_i in tpu_res
    ]
    interior = slice(halo_width, -halo_width, 1)
    if computation_shape[0] == 2:
      res = np.concatenate([
          tpu_res[0][interior, interior, interior],
          tpu_res[1][interior, interior, interior]
      ],
                           axis=0)
    elif computation_shape[1] == 2:
      res = np.concatenate([
          tpu_res[0][interior, interior, interior],
          tpu_res[1][interior, interior, interior]
      ],
                           axis=1)
    elif computation_shape[2] == 2:
      res = np.concatenate([
          tpu_res[0][interior, interior, interior],
          tpu_res[1][interior, interior, interior]
      ],
                           axis=2)
    else:
      res = tpu_res[0][interior, interior, interior]

    # Compute the reference solution.

    # TODO(wqing): Determine `boundary_points_per_side` from
    # `params.num_boundary_points`. Resolve this with the TODO about
    # `DEFAULT_NUM_BOUNDARY_POINTS = 2` in initializer.py.
    boundary_points_per_side = 1
    inner = slice(boundary_points_per_side, -boundary_points_per_side)
    x = np.linspace(0, lx, solver.params.fx)[inner]
    y = np.linspace(0, ly, solver.params.fy)[inner]
    z = np.linspace(0, lz, solver.params.fz)[inner]
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    expected = exact_solution(xx, yy, zz)

    poisson_solver_testutil.create_2d_slice_pngs(
        expected, res, x, y, z, solver.params, self._write_dir, solver_name,
        l2_norm)

    if solver_name == 'Conjugate Gradient':
      self.assertEqual(res.dtype, expected_x_dtype)
    else:
      # TODO(b/192092950): `tf.float64` is not supported yet for other solvers.
      self.assertEqual(res.dtype, np.float32)
    self.assertAllClose(expected, res, atol=1, rtol=1)

    expected_fname = os.path.join(
        self._write_dir, '{}_{}.npy'.format(solver_name, computation_shape))
    with gfile.GFile(expected_fname, 'w') as f:
      np.save(f, np.stack(res))

  _L_1_NORM = 768.0
  _L_2_NORM = 30.98387
  _L_INF_NORM = 3.0

  _NORM_TYPES_AND_VALUES = (
      #
      # Single norm type.
      #
      ((poisson_solver.NormType.L1,), (_L_1_NORM,)),
      ((poisson_solver.NormType.L2,), (_L_2_NORM,)),
      ((poisson_solver.NormType.L_INF,), (_L_INF_NORM,)),
      #
      # Multiple norm types.
      #
      ((poisson_solver.NormType.L1, poisson_solver.NormType.L2), (_L_1_NORM,
                                                                  _L_2_NORM)),
      ((poisson_solver.NormType.L1, poisson_solver.NormType.L_INF),
       (_L_1_NORM, _L_INF_NORM)),
      ((poisson_solver.NormType.L2, poisson_solver.NormType.L_INF),
       (_L_2_NORM, _L_INF_NORM)),
      ((poisson_solver.NormType.L1, poisson_solver.NormType.L2,
        poisson_solver.NormType.L_INF), (_L_1_NORM, _L_2_NORM, _L_INF_NORM)),
      # Reverse ordering.
      ((poisson_solver.NormType.L2, poisson_solver.NormType.L1), (_L_2_NORM,
                                                                  _L_1_NORM)),
      # With duplicates.
      ((poisson_solver.NormType.L1, poisson_solver.NormType.L2,
        poisson_solver.NormType.L1), (_L_1_NORM, _L_2_NORM, _L_1_NORM)),
  )

  _REPLICAS = (
      np.array([[[0]]], dtype=np.int32),
      np.array([[[0]], [[1]]], dtype=np.int32),
      np.array([[[0], [1]]], dtype=np.int32),
      np.array([[[0, 1]]], dtype=np.int32),
  )

  # Data types (input dtype, internal dtype, output dtype): Note that the input
  # dtype always match with output dtype precision, and independent of internal
  # dtype.
  # `ApplyKernelConvOp` doesn't support `tf.float64` yet.
  _DTYPES_AND_KERNEL_OP = (
      (tf.float32, tf.float32, np.float32, 'ApplyKernelConvOp'),
      (tf.float32, tf.float32, np.float32, 'ApplyKernelSliceOp'),
      (tf.float32, tf.float64, np.float32, 'ApplyKernelSliceOp'),
  )

  @parameterized.parameters(*itertools.product(_REPLICAS,
                                               _NORM_TYPES_AND_VALUES,
                                               (True, False),
                                               _DTYPES_AND_KERNEL_OP))
  def testComputeResidualGeneratesTheCorrectResidualNorms(
      self, replicas, norm_types_and_values, remove_mean_from_rhs,
      dtypes_and_kernel_op):
    """Checks if the residual of the Poisson equation is computed correctly."""

    solver_option = poisson_solver_pb2.PoissonSolver()
    halo_width = 2
    # params is used for grid spacings (and halo width after b/199428224).
    # `subgrid_shape` is set only so that grid spacings are correct.
    params = (grid_parametrization.GridParametrization.
              create_from_grid_lengths_and_etc(
                  grid_lengths=(0.1, 0.2, 0.5), computation_shape=(1, 1, 1),
                  subgrid_shape=(6, 6, 6), halo_width=halo_width))
    solver = poisson_solver.PoissonSolver(
        params, get_kernel_op(dtypes_and_kernel_op[-1]), solver_option)

    f = np.zeros((16, 16, 16), dtype=np.float32)
    f[4:-4, 4:-4, 4:-4] = 1.0
    # Offset each derivative by 1.
    ddz = np.zeros((16, 16, 16), dtype=np.float32)
    ddz[3, 4:-4, 4:-4] = 5.0
    ddz[4, 4:-4, 4:-4] = -5.0
    ddz[-4, 4:-4, 4:-4] = 5.0
    ddz[-5, 4:-4, 4:-4] = -5.0
    ddx = np.zeros((16, 16, 16), dtype=np.float32)
    ddx[4:-4, 3, 4:-4] = 101.0
    ddx[4:-4, 4, 4:-4] = -101.0
    ddx[4:-4, -4, 4:-4] = 101.0
    ddx[4:-4, -5, 4:-4] = -101.0
    ddy = np.zeros((16, 16, 16), dtype=np.float32)
    ddy[4:-4, 4:-4, 3] = 26.0
    ddy[4:-4, 4:-4, 4] = -26.0
    ddy[4:-4, 4:-4, -4] = 26.0
    ddy[4:-4, 4:-4, -5] = -26.0
    rhs = ddx + ddy + ddz
    expected_raw_res = np.zeros((16, 16, 16), dtype=np.float32)
    expected_raw_res[3, 4:-4, 4:-4] -= 1.0
    expected_raw_res[4, 4:-4, 4:-4] += 1.0
    expected_raw_res[-4, 4:-4, 4:-4] -= 1.0
    expected_raw_res[-5, 4:-4, 4:-4] += 1.0
    expected_raw_res[4:-4, 3, 4:-4] -= 1.0
    expected_raw_res[4:-4, 4, 4:-4] += 1.0
    expected_raw_res[4:-4, -4, 4:-4] -= 1.0
    expected_raw_res[4:-4, -5, 4:-4] += 1.0
    expected_raw_res[4:-4, 4:-4, 3] -= 1.0
    expected_raw_res[4:-4, 4:-4, 4] += 1.0
    expected_raw_res[4:-4, 4:-4, -4] -= 1.0
    expected_raw_res[4:-4, 4:-4, -5] += 1.0

    dtypes = tuple(dtypes_and_kernel_op[:-1])

    def device_fn(replica_id, f, rhs):
      """The wrapped device function for the residual computation."""

      return solver.compute_residual(
          replica_id,
          replicas,
          common_ops.tf_cast(f, dtypes[0]),
          common_ops.tf_cast(rhs, dtypes[0]),
          norm_types_and_values[0],
          halo_width,
          remove_mean_from_rhs=remove_mean_from_rhs,
          internal_dtype=dtypes[1])

    def _unstack_padded(f):
      """Utility function that pads and unstacks given field."""
      return tf.unstack(tf.pad(f, paddings=paddings, mode='CONSTANT'))

    computation_shape = np.array(replicas.shape)
    paddings = [[halo_width, halo_width], [halo_width, halo_width],
                [halo_width, halo_width]]
    if computation_shape[0] == 2:
      ids = [tf.constant(0), tf.constant(1)]
      fs = [_unstack_padded(f[:, :8, :]), _unstack_padded(f[:, 8:, :])]
      rhss = [_unstack_padded(rhs[:, :8, :]), _unstack_padded(rhs[:, 8:, :])]
      expected_raw_residuals = [np.pad(expected_raw_res[:, :8, :], paddings,
                                       'constant'),
                                np.pad(expected_raw_res[:, 8:, :], paddings,
                                       'constant')]
    elif computation_shape[1] == 2:
      ids = [tf.constant(0), tf.constant(1)]
      fs = [_unstack_padded(f[:, :, :8]), _unstack_padded(f[:, :, 8:])]
      rhss = [_unstack_padded(rhs[:, :, :8]), _unstack_padded(rhs[:, :, 8:])]
      expected_raw_residuals = [np.pad(expected_raw_res[:, :, :8], paddings,
                                       'constant'),
                                np.pad(expected_raw_res[:, :, 8:], paddings,
                                       'constant')]
    elif computation_shape[2] == 2:
      ids = [tf.constant(0), tf.constant(1)]
      fs = [_unstack_padded(f[:8, :, :]), _unstack_padded(f[8:, :, :])]
      rhss = [_unstack_padded(rhs[:8, :, :]), _unstack_padded(rhs[8:, :, :])]
      expected_raw_residuals = [np.pad(expected_raw_res[:8, :, :], paddings,
                                       'constant'),
                                np.pad(expected_raw_res[8:, :, :], paddings,
                                       'constant')]
    else:
      ids = [tf.constant(0)]
      fs = [_unstack_padded(tf.convert_to_tensor(f))]
      rhss = [_unstack_padded(tf.convert_to_tensor(rhs))]
      expected_raw_residuals = [np.pad(expected_raw_res, paddings, 'constant')]

    runner = TpuRunner(replicas=replicas)
    res = runner.run(device_fn, ids, fs, rhss)

    self.assertIsInstance(res[0][0][0], dtypes[2])
    self.assertAllClose(
        np.stack(norm_types_and_values[1]), res[0][0], rtol=1e-5)
    for i in range(len(ids)):
      raw_res_slices = res[i][1]
      raw_res_no_halos = np.stack([s[halo_width:-halo_width,
                                     halo_width:-halo_width] for s in
                                   raw_res_slices[halo_width:-halo_width]])

      self.assertEqual(raw_res_no_halos.dtype, dtypes[2])
      self.assertAllClose(
          expected_raw_residuals[i][halo_width:-halo_width,
                                    halo_width:-halo_width,
                                    halo_width:-halo_width], raw_res_no_halos,
          rtol=1e-5)


if __name__ == '__main__':
  tf.test.main()
