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
"""Tests for the distributed conjugate gradient (CG) solver on TPUs."""

import functools
import itertools

from absl import logging
import numpy as np
from swirl_lm.communication import halo_exchange
from swirl_lm.communication import halo_exchange_utils
from swirl_lm.linalg import conjugate_gradient_solver
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
import tensorflow as tf
from google3.research.simulation.tensorflow.fluid.framework import util
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized

_A1111 = [[1., 1.], [1., 1.]]
_A2112 = [[2., 1.], [1., 2.]]
_B00 = [[0.], [0.]]
_B10 = [[1.], [0.]]
_B11 = [[1.], [1.]]

_X = conjugate_gradient_solver.X
_RESIDUAL_L2_NORM = conjugate_gradient_solver.RESIDUAL_L2_NORM
_ITERATIONS = conjugate_gradient_solver.ITERATIONS


class ConjugateGradientSolverBuilder(object):
  """Defines the init_fn and step_fn of the CG solver for TPUSimulation."""

  def __init__(
      self,
      linear_operator,
      b,
      max_iterations,
      atol,
      x0=None,
      l2_norm_reduction=False,
      dot=None,
      internal_dtype=tf.float32,
  ):
    self._linear_operator = linear_operator
    self._b = [b] if isinstance(b, tf.Tensor) else b
    self._max_iterations = max_iterations
    self._atol = atol
    self._x0 = x0
    self._l2_norm_reduction = l2_norm_reduction
    self._dot = dot
    self._internal_dtype = internal_dtype

  def _maybe_split_linear_operator(self, linear_op, replica_id):
    """Maybe splits the linear operator by replicas.

    If `linear_op` is a `Callable`, all replicas will share the same linear
    operator; if `linear_op` is a `list`, the linear operator in each replica
    is `linear_op[replica_id]`. In this test, a maximum of 2 replicas is
    considered.

    Args:
      linear_op: The (group of ) linear operator(s) to possibly be split.
      replica_id: The id of the replica.

    Returns:
      The split linear operator associated with replica_id.
    """
    if callable(linear_op):
      return linear_op

    return tf.cond(
        pred=tf.equal(replica_id, 0),
        true_fn=lambda: linear_op[0],
        false_fn=lambda: linear_op[1])

  def _maybe_split_field(self, f, replica_id, replicas):
    """Maybe splits field `f` by replicas.

    Note that only a split along dimension 2 is considered in the tests
    (b/144510711). The number of TPUs in that dimension is 1 or 2. If 2, this
    operator splits `f` in half. `replica_id` 0 gets the first half and
    `replica_id` 1 gets the second half.

    Args:
      f: The field to possibly be split.
      replica_id: The id of the replica.
      replicas: The grid of the TPUs.

    Returns:
      The split field associated with replica_id.
    """
    computation_shape = replicas.shape
    if np.prod(computation_shape) == 1 or len(f) == 1:
      return f

    if len(f) % computation_shape[2] != 0:
      raise ValueError(
          'Field to be split need to be fully divided by the number of cores in'
          ' the corresponding dimension. Now has size {} but {} cores.'.format(
              len(f), computation_shape[2]))
    nz = int(len(f) / computation_shape[2])

    # Only 2 TPUs are used in this test.
    return tf.cond(
        pred=tf.equal(replica_id, 0),
        true_fn=lambda: f[:nz],
        false_fn=lambda: f[nz:])

  def init_fn(self, replica_id, coordinates):
    del coordinates
    return {'replica_id': replica_id}

  def step_fn(self, state, replicas, replica_id):
    # Split the right hand side by replicas.
    b = self._maybe_split_field(self._b, replica_id, replicas)

    # Split the initial guess vector by replicas.
    if self._x0 is None:
      x0 = [tf.zeros_like(b_) for b_ in b]
    else:
      x0 = self._maybe_split_field(self._x0, replica_id, replicas)

    # Initialize the dot if no definition is not provided by the caller.
    computation_shape = replicas.shape
    num_replicas = np.prod(computation_shape)
    group_assignment = np.array([range(num_replicas)], dtype=np.int32)
    if self._dot is None:
      self._dot = functools.partial(
          common_ops.global_dot, group_assignment=group_assignment)

    linear_operator = self._maybe_split_linear_operator(self._linear_operator,
                                                        replica_id)
    linear_operator = functools.partial(linear_operator, replica_id, replicas)
    return (conjugate_gradient_solver.conjugate_gradient_solver(
        linear_operator,
        self._dot,
        b,
        self._max_iterations,
        self._atol,
        x0,
        l2_norm_reduction=self._l2_norm_reduction,
        internal_dtype=self._internal_dtype), state)


class ConjugateGradientSolverTest(tf.test.TestCase, parameterized.TestCase):

  def _testSolution(self, linear_operator, x, b, atol=1e-6, atol_raw=None):
    """Test of x as a solution for A * x = b, where A is the linear operator."""
    error = b - linear_operator(None, None, [x])
    error = self.evaluate(error)[0]

    if atol_raw is None:
      atol_raw = atol
    logging.info('Shape = %s: (atol_raw, atol, max) = (%g, %g, %g).',
                 str(error.shape), atol_raw, atol, np.amax(np.abs(error)))

    self.assertAllClose(error, np.zeros(b.shape), atol=atol)

  _ARGS = (
      #
      # 1. Unique solution.
      #
      ('Case00', _A2112, _B10, None, 1e-6, 10, [[2. / 3], [-1. / 3]], 0., 2),
      ('Case01', _A2112, _B10, None, 1e-8, 10, [[2. / 3], [-1. / 3]], 0., 2),
      ('Case02', _A2112, _B10, None, 1e-8, 2, None, 0., 2),
      #   1.1 Initial value without any iteration.
      ('Case03', _A2112, _B10, [[1.], [0.]], .1, 0, None, 1.4142135, 0),
      ('Case04', _A2112, _B10, [[0.], [1.]], .1, 0, None, 2., 0),
      #
      # 2. Non-unique solution, for a singular matrix: homogeneous.
      #
      #   2.1. Initial value is a solution.
      ('Case10', _A1111, _B00, [[0.], [0.]], 1e-6, 2, [[0.], [0.]], 0., 0),
      ('Case11', _A1111, _B00, [[1.], [-1.]], 1e-6, 2, [[1.], [-1.]], 0., 0),
      ('Case12', _A1111, _B00, [[-1.], [1.]], 1e-6, 2, [[-1.], [1.]], 0., 0),
      #   2.2. Initial value is not a solution.
      ('Case13', _A1111, _B00, [[1.], [1.]], .1, 0, None, 2.828427, 0),
      ('Case14', _A1111, _B00, [[1.], [1.]], 1e-6, 1, [[0.], [0.]], 0., 1),
      ('Case15', _A1111, _B00, [[0.3], [-0.7]], 1e-6, 1, [[.5], [-.5]], 0., 1),
      ('Case16', _A1111, _B00, [[0.6], [-0.4]], 1e-6, 1, [[.5], [-.5]], 0., 1),
      ('Case17', _A1111, _B00, [[0.9], [-0.1]], 1e-6, 1, [[.5], [-.5]], 0., 1),
      ('Case18', _A1111, _B00, [[3.], [-7.]], 1e-6, 1, [[5.], [-5.]], 0., 1),
      ('Case19', _A1111, _B00, [[6.], [-4.]], 1e-6, 1, [[5.], [-5.]], 0., 1),
      ('Case20', _A1111, _B00, [[9.], [-1.]], 1e-6, 1, [[5.], [-5.]], 0., 1),
      #
      # 3. Non-unique solution, for a singular matrix: nonhomogeneous.
      #
      #   3.1. Initial value is a solution.
      ('Case30', _A1111, _B11, [[.5], [.5]], 1e-6, 2, [[.5], [.5]], 0., 0),
      ('Case31', _A1111, _B11, [[1.], [0.]], 1e-6, 2, [[1.], [0.]], 0., 0),
      ('Case32', _A1111, _B11, [[0.], [1.]], 1e-6, 2, [[0.], [1.]], 0., 0),
      #   3.2. Initial value is not a solution.
      ('Case33', _A1111, _B11, [[.5], [-.5]], 1e-6, 2, [[1.], [0.]], 0., 1),
      ('Case34', _A1111, _B11, [[5.], [-5.]], 1e-6, 2, [[5.5], [-4.5]], 0., 1),
      ('Case35', _A1111, _B11, [[52.], [-49.]], 1e-6, 2, [[51.],
                                                          [-50.]], 0., 1),
  )

  @parameterized.parameters(*itertools.product(
      _ARGS,
      (True, False),
      (
          # (internal_dtype, expected_np_dtype_x, expected_np_dtype_l2_norm)
          (tf.float32, np.float32, np.float32),
          (tf.float64, np.float64, np.float32),
          (None, np.float32, np.float32),
      )))
  def testConjugateGradientSolverSmallRealSystem(self, args, l2_norm_reduction,
                                                 dtypes):
    (_, a, b, x0, atol, max_iterations, expected_x, expected_l2_norm,
     expected_iterations) = args
    internal_dtype, expected_np_dtype_x, expected_np_dtype_l2_norm = dtypes

    tf_a = tf.constant(a, dtype=tf.float32)
    tf_b = tf.constant(b, dtype=tf.float32)
    tf_atol = tf.constant(atol, dtype=tf.float32)
    computation_shape = np.array([1, 1, 1])

    def linear_op(replica_id, replicas, x):
      del replica_id, replicas

      tf_aa = tf_a if internal_dtype is None else tf.cast(tf_a, internal_dtype)
      return [tf.einsum('ij,jk->ik', tf_aa, x_) for x_ in x]

    cg_solver = ConjugateGradientSolverBuilder(
        linear_op,
        tf_b,
        max_iterations,
        tf_atol,
        x0=None if x0 is None else [tf.constant(x0, dtype=tf.float32)],
        l2_norm_reduction=l2_norm_reduction,
        internal_dtype=internal_dtype)

    coordinates = util.grid_coordinates(computation_shape)
    num_replicas = np.prod(computation_shape)
    runner = TpuRunner(computation_shape=computation_shape)
    replica_outputs = runner.run_with_replica_args(
        cg_solver.step_fn,
        [cg_solver.init_fn(i, coordinates[i]) for i in range(num_replicas)])

    tpu_res = [output[0] for output in replica_outputs]
    poisson_solver_solution = tpu_res[0]
    x = poisson_solver_solution[_X][0]
    l2_norm = poisson_solver_solution[_RESIDUAL_L2_NORM]
    iterations = poisson_solver_solution[_ITERATIONS]

    self.assertIsInstance(l2_norm, expected_np_dtype_l2_norm)
    self.assertAlmostEqual(expected_l2_norm, l2_norm, places=6)

    self.assertIsInstance(iterations, np.int32)
    self.assertEqual(expected_iterations, iterations)

    if max_iterations == 0:
      self.assertAllClose(x, x0, atol=1e-10)
      return

    self.assertEqual(x.dtype, expected_np_dtype_x)
    # 1. Comparing a^-1 * b vs x
    if expected_x is None:
      expected_x = np.linalg.solve(
          np.array(a, dtype=np.float32), np.array(b, dtype=np.float32))
    self.assertAllClose(expected_x, x, atol=atol)

    # 2. Comparing a * x vs b
    self._testSolution(linear_op, x, tf_b)

  @parameterized.named_parameters(
      # n = 10
      ('Case00', 10, 1e-4, 1.5013355e-06, 12),
      ('Case01', 10, 1e-6, 3.431348e-07, 13),
      ('Case02', 10, 1e-8, 7.763538e-09, 19),
      # n = 30
      ('Case10', 30, 1e-4, 4.5180404e-05, 53),
      ('Case11', 30, 1e-6, 9.4588195e-06, 60),
      ('Case12', 30, 1e-8, 9.4588195e-06, 60),
      # n = 50
      ('Case20', 50, 1e-4, 0.00023923842, 100),
      ('Case21', 50, 1e-6, 0.00023923842, 100),
      # n = 100
      ('Case30', 100, 1e-4, 7.414594e-05, 171),
      ('Case31', 100, 1e-6, 5.4911648e-06, 200),
      ('Case32', 100, 1e-8, 5.4911648e-06, 200),
  )
  def testConjugateGradientSolverLargeRealSystem(self, n, atol,
                                                 expected_l2_norm,
                                                 expected_iterations):
    p = np.random.normal(loc=0.1, scale=1, size=(n, n))
    a = np.array(np.dot(p.T, p), dtype=np.float32)
    b = np.ones((n, 1), dtype=np.float32)

    tf_a = tf.constant(a, dtype=tf.float32)
    tf_b = tf.constant(b, dtype=tf.float32)
    max_iterations = 2 * n
    tf_atol = tf.constant(atol, dtype=tf.float32)
    computation_shape = np.array([1, 1, 1])

    def linear_op(replica_id, replicas, x):
      del replica_id, replicas
      return [tf.einsum('ij,jk->ik', tf_a, x_) for x_ in x]

    cg_solver = ConjugateGradientSolverBuilder(linear_op, tf_b, max_iterations,
                                               tf_atol)

    coordinates = util.grid_coordinates(computation_shape)
    num_replicas = np.prod(computation_shape)
    runner = TpuRunner(computation_shape=computation_shape)
    replica_outputs = runner.run_with_replica_args(
        cg_solver.step_fn,
        [cg_solver.init_fn(i, coordinates[i]) for i in range(num_replicas)])

    tpu_res = [output[0] for output in replica_outputs]
    poisson_solver_solution = tpu_res[0]
    x = poisson_solver_solution[_X][0]
    l2_norm = poisson_solver_solution[_RESIDUAL_L2_NORM]
    iterations = poisson_solver_solution[_ITERATIONS]

    self.assertAlmostEqual(expected_l2_norm, l2_norm, places=6)
    self.assertEqual(expected_iterations, iterations)

    # 1. Comparing a^-1 * b vs x
    expected_x = np.linalg.solve(a, b)
    self.assertAllClose(expected_x, x, rtol=1e-3, atol=atol)

    # 2. Comparing a * x vs b
    atol_expected_x5 = np.amax(np.abs(np.matmul(a, expected_x) - b)) * 5
    self._testSolution(linear_op, x, tf_b, atol=atol_expected_x5, atol_raw=atol)

  def testConjugateGradientSolverMultipleTPUCoresDiagonalMatrix(self):
    a0 = 3.0
    b0 = 6.0
    nx = 16
    ny = 16
    nz = 10
    computation_shape = np.array([1, 1, 2])
    max_iterations = 2 * nx
    atol = 1e-6

    def linear_op(replica_id, replicas, x):
      del replica_id, replicas
      return [a0 * x_ for x_ in x]

    rhs = [tf.constant(b0, shape=(nx, ny), dtype=tf.float32)] * nz * 2
    cg_solver = ConjugateGradientSolverBuilder(linear_op, rhs, max_iterations,
                                               atol)

    coordinates = util.grid_coordinates(computation_shape)
    num_replicas = np.prod(computation_shape)
    runner = TpuRunner(computation_shape=computation_shape)
    replica_outputs = runner.run_with_replica_args(
        cg_solver.step_fn,
        [cg_solver.init_fn(i, coordinates[i]) for i in range(num_replicas)])

    tpu_res = [output[0] for output in replica_outputs]
    poisson_solver_solution = tpu_res[0]
    l2_norm = poisson_solver_solution[_RESIDUAL_L2_NORM]
    iterations = poisson_solver_solution[_ITERATIONS]
    f_0 = tpu_res[0][_X]
    f_1 = tpu_res[1][_X]

    self.assertAlmostEqual(0., l2_norm)
    self.assertEqual(2, iterations)

    f = np.concatenate([f_0[1:-1], f_1[1:-1]], axis=0)
    self.assertLen(f, 2 * (nz - 2))

    for i in range(2 * (nz - 2)):
      expected_f = (b0 / a0) * np.ones(shape=(nx, ny))
      self.assertAllClose(
          expected_f,
          f[i],
          rtol=1e-3,
          atol=1e-3,
          msg='The {}th z plane does not match'.format(i))

  def testConjugateGradientSolverMultipleTPUCores(self):
    """Solves a Poisson equation that has a constant right hand side.

       The equation takes the following form:
         d^2 f / dx^2 + d^2 f / dy^2 + d^2f / dz^2 = Const,
         df / dx = Const0 @ x = 0,
         f =  0 @ x = 1,
         df / dy = 0 @ y = 0,
         df / dy = 0 @ y = 1,
         df / dz = 0 @ y = 0,
         df / dz = 0 @ z = 1.
       with x, y, and z ranging from 0 to 1. The exact solution to this problem
       is:
         f(x, y, z) = a0 x^2 + a1 x + b0 y^2 + b1 y + c0 z^2 + c1 z + d
       with:
         b0 = b1 = c0 = c1 = 0
         a0 = 0.5 * Const,
         a1 = Const0,
         d  = -a0 - a1
       The discreitized problem can be expressed as a linear system of equation:
         A x = b
       Because the boundary condition of the variable has to be homogeneous to
       use the CG solver, the problem is reformulated as:
         A x_h = b - A x_b = b_h
       where x_b is the boundary condition of x, and x_h is the solution with a
       homogeneous boundary condition. This test checks the difference between
       A x_h and b_h.
    """
    num_replicas = 2
    halo_width = 1
    nx = 8
    ny = 8
    nz = 10
    lx = 1.0
    dx = lx / (nx - 1)
    dy = dx
    dz = dx
    x = np.linspace(0, lx, nx)
    xy_x = np.tile(np.array([x]).transpose(), (1, ny))

    a1 = 0.0
    rhs_const = 6.0
    a0 = 0.5 * rhs_const
    d = -a0 - a1
    computation_shape = np.array([1, 1, num_replicas])
    max_iterations = 4 * nx
    atol = 1e-6

    def global_linear_op(f):
      kernel_op = get_kernel_fn.ApplyKernelConvOp(4)

      ddx = [g / dx**2 for g in kernel_op.apply_kernel_op_x(f, 'kddx')]
      ddy = [g / dy**2 for g in kernel_op.apply_kernel_op_y(f, 'kddy')]
      ddz = [
          g / dz**2 for g in kernel_op.apply_kernel_op_z(f, 'kddz', 'kddzsh')
      ]

      return [ddx_ + ddy_ + ddz_ for ddx_, ddy_, ddz_ in zip(ddx, ddy, ddz)]

    def linear_op(replica_id, replicas, f):
      # Constants required in this function.
      halo_dims = (0, 1, 2)
      replica_dims = (0, 1, 2)
      periodic_dims = [False, False, False]

      # Make sure all boundary information is correct before solving the system.
      f = halo_exchange.inplace_halo_exchange(
          f,
          halo_dims,
          replica_id,
          replicas,
          replica_dims,
          periodic_dims,
          halo_exchange_utils.homogeneous_bcs(),
          width=halo_width)

      # Apply the matrix multiplication.
      f = global_linear_op(f)

      # Remove the cells on the border to prevent residuals from being polluted.
      f = halo_exchange.clear_halos(f, halo_width)

      return f

    # Get the solution and the boundary values of it.
    def get_boundary(x):
      nx, ny = x.get_shape().as_list()
      bc_left = x[halo_width:-halo_width, 0]
      bc_right = x[halo_width:-halo_width, -1]
      bc_top = x[0, :]
      bc_bottom = x[-1, :]

      x_bc = tf.concat([
          tf.reshape(bc_left, [-1, halo_width]),
          tf.zeros(
              shape=(nx - 2 * halo_width, ny - 2 * halo_width),
              dtype=tf.float32),
          tf.reshape(bc_right, [-1, halo_width])
      ],
                       axis=1)
      x_bc = tf.concat([
          tf.reshape(bc_top, [halo_width, -1]), x_bc,
          tf.reshape(bc_bottom, [halo_width, -1])
      ],
                       axis=0)

      return x_bc

    x0_xy = tf.constant(a0 * xy_x**2 + a1 * xy_x + d, dtype=tf.float32)
    x0_xy_bc = get_boundary(x0_xy)
    x0_bc = [x0_xy
            ] + [x0_xy_bc] * (nz * num_replicas - 2 * halo_width) + [x0_xy]

    # Get the right hand side of the linear system of equation.
    rhs = [tf.constant(rhs_const, shape=(nx, ny), dtype=tf.float32)] * nz
    rhs_0 = [
        rhs_ - rhs_bc_
        for rhs_, rhs_bc_ in zip(rhs, global_linear_op(x0_bc[:nz]))
    ]
    rhs_1 = [
        rhs_ - rhs_bc_
        for rhs_, rhs_bc_ in zip(rhs, global_linear_op(x0_bc[nz:]))
    ]
    rhs_0 = halo_exchange.clear_halos(rhs_0, halo_width)
    rhs_1 = halo_exchange.clear_halos(rhs_1, halo_width)
    rhs = rhs_0 + rhs_1
    rhs_global = rhs_0[:-1] + rhs_1[1:]

    cg_solver = ConjugateGradientSolverBuilder(linear_op, rhs, max_iterations,
                                               atol)

    coordinates = util.grid_coordinates(computation_shape)
    num_replicas = np.prod(computation_shape)
    runner = TpuRunner(computation_shape=computation_shape)
    replica_outputs = runner.run_with_replica_args(
        cg_solver.step_fn,
        [cg_solver.init_fn(i, coordinates[i]) for i in range(num_replicas)])

    tpu_res = [output[0] for output in replica_outputs]
    poisson_solver_solution = tpu_res[0]
    l2_norm = poisson_solver_solution[_RESIDUAL_L2_NORM]
    iterations = poisson_solver_solution[_ITERATIONS]
    f_0 = tpu_res[0][_X]
    f_1 = tpu_res[1][_X]

    rhs_val = self.evaluate(rhs_global)

    x_actual_val = np.concatenate([
        f_0[:-halo_width],
        f_1[halo_width:]
    ],
                                  axis=0)
    x = [tf.constant(x_val_, dtype=tf.float32) for x_val_ in x_actual_val]
    rhs_actual = global_linear_op(x)
    rhs_actual = halo_exchange.clear_halos(rhs_actual, halo_width)
    rhs_actual_val = self.evaluate(rhs_actual)

    self.assertAlmostEqual(1.6139717e-05, l2_norm)
    self.assertEqual(32, iterations)

    self.assertLen(rhs_actual_val, len(rhs_val))
    for i in range(len(rhs_val)):
      self.assertAllClose(
          rhs_val[i],
          rhs_actual_val[i],
          rtol=1e-3,
          atol=1e-3,
          msg='The {}th z plane does not match'.format(i))

  def testConjugateGradientSolverOperatorSplit(self):
    factors = [0.3, 0.7]
    computation_shape = np.array([1, 1, 2])
    nx = 8
    ny = 16
    max_iterations = 2 * ny
    atol = 1e-6

    def linear_op(replica_id, replicas, x):
      computation_shape = replicas.shape
      num_replicas = np.prod(computation_shape)
      group_assignment = np.array([range(num_replicas)], dtype=np.int32)

      factor = tf.cond(
          pred=tf.equal(replica_id, 0),
          true_fn=lambda: factors[0],
          false_fn=lambda: factors[1])
      x_split = [factor * x_ for x_ in x]
      x_total = [
          tf.compat.v1.tpu.cross_replica_sum(x_, group_assignment)
          for x_ in x_split
      ]

      return x_total

    rhs_val = np.random.normal(loc=0.1, scale=1, size=[nx, ny])
    rhs = [tf.constant(rhs_val, dtype=tf.float32)]
    cg_solver = ConjugateGradientSolverBuilder(
        linear_op, rhs, max_iterations, atol, dot=common_ops.local_dot)

    coordinates = util.grid_coordinates(computation_shape)
    num_replicas = np.prod(computation_shape)
    runner = TpuRunner(computation_shape=computation_shape)
    replica_outputs = runner.run_with_replica_args(
        cg_solver.step_fn,
        [cg_solver.init_fn(i, coordinates[i]) for i in range(num_replicas)])

    tpu_res = [output[0] for output in replica_outputs]
    poisson_solver_solution = tpu_res[0]
    l2_norm = poisson_solver_solution[_RESIDUAL_L2_NORM]
    iterations = poisson_solver_solution[_ITERATIONS]
    f_0 = tpu_res[0][_X]
    f_1 = tpu_res[1][_X]

    self.assertAlmostEqual(4.9148656e-14, l2_norm)
    self.assertEqual(2, iterations)

    self.assertLen(
        f_0,
        1,
        msg='The length of solution tensor from replica 0 should be 1. Actual: '
        '{}.'.format(len(f_0)))
    self.assertLen(
        f_1,
        1,
        msg='The length of solution tensor from replica 1 should be 1. Actual: '
        '{}.'.format(len(f_1)))
    self.assertAllClose(
        rhs_val / np.sum(factors),
        f_0[0],
        msg='Results from replica 0 are incorrect.')
    self.assertAllEqual(
        f_0[0], f_1[0], msg='Results from replica 1 are incorrect.')


if __name__ == '__main__':
  tf.test.main()
