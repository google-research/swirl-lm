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
"""Tests for multigrid_utils."""
import functools
import itertools

import numpy as np
from swirl_lm.communication import halo_exchange
from swirl_lm.linalg import multigrid_test_common
from swirl_lm.linalg import multigrid_utils
from swirl_lm.utility import grid_parametrization
import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.framework import initializer
from google3.research.simulation.tensorflow.fluid.framework import util
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized

BCType = halo_exchange.BCType


class MultigridUtilsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests multigrid_utils functions in 1, 2, and 3 dimensions."""

  def set_params(self, rank, grid_lengths):
    computation_shape = (1, 1, 1)
    self.shape = (13, 19, 24)[:rank]
    self.x = np.random.rand(*self.shape)
    self.b = np.random.rand(*self.shape)
    self.params = (grid_parametrization.GridParametrization.
                   create_from_grid_lengths_and_etc_with_defaults(
                       grid_lengths, computation_shape))
    self.dxs = [length / (s - 1)
                for length, s in zip(grid_lengths, self.shape)]
    self.sum_inverse_spacing_squared = sum([dx**-2 for dx in self.dxs])
    self.inner_slices = (slice(1, -1),) * rank

  # Numpy reference implementations.
  def lap_inv_diagonal(self):
    return self.x / (-2 * self.sum_inverse_spacing_squared)

  def lap(self):
    res = -2 * self.sum_inverse_spacing_squared * self.x
    for i in range(self.x.ndim):
      r_slices = tuple([slice(1, -1)] * i + [slice(2, None)] +
                       [slice(1, -1)] * (self.x.ndim - i - 1))
      l_slices = tuple([slice(1, -1)] * i + [slice(0, -2)] +
                       [slice(1, -1)] * (self.x.ndim - i - 1))
      res[self.inner_slices] += (
          (self.x[r_slices] + self.x[l_slices]) / (self.dxs[i]**2))
    return res

  def jac(self, n, weight):
    for _ in range(n):
      self.x[self.inner_slices] -= (
          weight / 2 / self.sum_inverse_spacing_squared *
          (self.b - self.lap())[self.inner_slices])
    return self.x

  PARAMS = itertools.product((1, 2, 3), ((1, 1, 1), (1.1, 5.2, 9.33)))

  @parameterized.parameters(PARAMS)
  def test_laplacian_inv_diagonal_(self, rank, grid_lengths):
    """Laplacian inverse diagonal function test in 1, 2 and 3 dimensions."""
    self.set_params(rank, grid_lengths)
    inv_diag_fn = multigrid_utils.laplacian_and_inv_diagonal_fns(
        self.shape, grid_lengths[:rank])[1]
    x_tf = tf.convert_to_tensor(self.x)
    tpu = TpuRunner(computation_shape=(1,))
    actual = tpu.run(lambda: inv_diag_fn(x_tf))[0]
    expected = self.lap_inv_diagonal()
    self.assertAllClose(actual[self.inner_slices], expected[self.inner_slices],
                        atol=1e-10)

  PARAMS = itertools.product((1, 2, 3), ((1, 1, 1), (1.1, 5.2, 9.33)))

  @parameterized.parameters(PARAMS)
  def test_laplacian_(self, rank, grid_lengths):
    """Tests the Laplacian function in 1, 2 and 3 dimensions."""
    self.set_params(rank, grid_lengths)
    x_tf = tf.convert_to_tensor(self.x)
    laplacian = multigrid_utils.laplacian_and_inv_diagonal_fns(
        self.shape, grid_lengths[:rank])[0]
    tpu = TpuRunner(computation_shape=(1,))
    actual = tpu.run(lambda: laplacian(x_tf))[0]
    expected = self.lap()
    self.assertAllClose(actual[self.inner_slices], expected[self.inner_slices],
                        atol=1e-10)

  PARAMS = itertools.product((1, 2, 3), ((1, 1, 1), (1.1, 5.2, 9.33)),
                             (1, 2 / 3))

  @parameterized.parameters(PARAMS)
  def test_poisson_jacobi_(self, rank, grid_lengths, weight):
    """Tests the Poisson Jacobi function in 1, 2 and 3 dimensions."""
    self.set_params(rank, grid_lengths)
    n = 111

    actual = self.evaluate(
        multigrid_utils.poisson_jacobi(
            tf.convert_to_tensor(self.x), tf.convert_to_tensor(self.b),
            self.params, n, weight))

    expected = self.jac(n, weight)
    self.assertAllClose(actual, expected, atol=1e-10)

  def assertBordersEqual(self, x, b, atol):
    inner = (slice(1, -1),) * len(x.shape)
    x[inner] = 0
    b[inner] = 0
    self.assertAllClose(x, b, atol=atol)

  def test_laplacian_matrix_1d_dirichlet(self):
    shape = (3,)
    grid_lengths = (2,)
    a = multigrid_utils.laplacian_matrix(shape, grid_lengths)
    expected = [[1, 0, 0],
                [1, -2, 1],
                [0, 0, 1]]

    self.assertAllEqual(a, expected)

  def test_laplacian_matrix_1d_dirichlet_size_5(self):
    shape = (5,)
    grid_lengths = (4,)
    a = multigrid_utils.laplacian_matrix(shape, grid_lengths)
    expected = [[1, 0, 0, 0, 0],
                [1, -2, 1, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 1, -2, 1],
                [0, 0, 0, 0, 1]]

    self.assertAllEqual(a, expected)

  def test_laplacian_matrix_1d_neumann_size_5(self):
    shape = (5,)
    grid_lengths = (4,)
    boundary_conditions = [[(BCType.NEUMANN, 0.)] * 2]
    a = multigrid_utils.laplacian_matrix(shape, grid_lengths,
                                         boundary_conditions)
    expected = [[-1, 1, 0, 0, 0],
                [1, -2, 1, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 1, -2, 1],
                [0, 0, 0, -1, 1]]

    self.assertAllEqual(a, expected)

  @parameterized.parameters(itertools.product(range(1, 4), (False, True)))
  def test_inverse_laplacian_matrix_np_f32_(self, rank, use_pinv):
    expected_max_diff = ((2.98e-8, 6.26e-7),
                         (5.79e-6, 4.69e-5),
                         (2.75e-5, 9.77e-4))[rank - 1][use_pinv]

    expected_norm = ((6.56e-7, 1.30e-6),
                     (3.01e-4, 4.18e-5),
                     (6.31e-3, 1.40e-3))[rank - 1][use_pinv]

    borders_equal_atol = ((0, 6.56e-7),
                          (5.66e-6, 4.70e-5),
                          (2.54e-5, 9.72e-4))[rank - 1][use_pinv]

    shape = (5, 7, 11)[:rank]
    b = np.random.rand(*shape).astype(np.float32)

    a = multigrid_utils.laplacian_matrix(shape)
    x_expected = multigrid_utils.solve(a, b)

    a_inv = multigrid_utils.inverse_laplacian_matrix(shape, use_pinv=use_pinv)
    x_actual = multigrid_utils.matmul(a_inv, b)

    max_diff = np.abs(x_actual - x_expected).max()

    self.assertAllClose(max_diff, expected_max_diff, rtol=1e-3)

    residual_norm = multigrid_utils.poisson_residual_norm(x_actual, b)
    self.assertAllClose(residual_norm, expected_norm, rtol=3e-3)

    # Check that x borders are equal to b borders.
    self.assertBordersEqual(x_actual, b, atol=borders_equal_atol)

  @parameterized.parameters(range(1, 4))
  def test_inverse_laplacian_matrix_tf_f32_(self, rank):
    atol = (7e-7, 3e-5, 6e-5)[rank - 1]
    norm_expected = (6.56e-7, 3.07e-4, 6.26e-3)[rank - 1]

    shape = (5, 7, 11)[:rank]
    b = np.random.rand(*shape).astype(np.float32)

    a = multigrid_utils.laplacian_matrix(shape)
    x_expected = multigrid_utils.solve(a, tf.convert_to_tensor(b))

    a_inv = multigrid_utils.inverse_laplacian_matrix(shape)
    x_actual = multigrid_utils.matmul(a_inv, tf.convert_to_tensor(b))

    x_expected, x_actual = self.evaluate((x_expected, x_actual))

    self.assertAllClose(x_actual, x_expected, atol=atol)
    self.assertAllClose(
        multigrid_utils.poisson_residual_norm(x_actual, b),
        norm_expected,
        rtol=1e-3)

    # Check that x borders are equal to b borders.
    self.assertBordersEqual(x_actual, b, atol)

  @parameterized.parameters(range(1, 4))
  def test_inverse_laplacian_matrix_np_f64_(self, rank):
    atol = 1e-12
    norm_expected = (1.48e-15, 3.75e-13, 1.04e-11)[rank -1]

    shape = (5, 7, 11)[:rank]
    b = np.random.rand(*shape).astype(np.float64)

    a = multigrid_utils.laplacian_matrix(shape, dtype=np.float64)
    x_expected = multigrid_utils.solve(a, b)

    a_inv = multigrid_utils.inverse_laplacian_matrix(shape, dtype=np.float64)
    x_actual = multigrid_utils.matmul(a_inv, b)

    self.assertAllClose(x_actual, x_expected, atol=atol)
    self.assertAllClose(
        multigrid_utils.poisson_residual_norm(x_actual, b),
        norm_expected,
        rtol=1e-4)

    # Check that x borders are equal to b borders.
    self.assertBordersEqual(x_actual, b, atol)

  @parameterized.parameters(range(1, 4))
  def test_inverse_laplacian_matrix_tf_f64_(self, rank):
    atol = 1e-12
    norm_tol = 1.3e-11

    shape = (5, 7, 11)[:rank]
    b = np.random.rand(*shape).astype(np.float64)

    a = multigrid_utils.laplacian_matrix(shape, dtype=np.float64)
    x_expected = multigrid_utils.solve(a, tf.convert_to_tensor(b))

    a_inv = multigrid_utils.inverse_laplacian_matrix(shape, dtype=np.float64)
    x_actual = multigrid_utils.matmul(a_inv, tf.convert_to_tensor(b))

    x_expected, x_actual = self.evaluate((x_expected, x_actual))

    self.assertAllClose(x_actual, x_expected, atol=atol)
    self.assertLess(
        multigrid_utils.poisson_residual_norm(x_actual, b), norm_tol)

    self.assertBordersEqual(x_actual, b, atol)

  def test_poisson_jacobi_roughly_equals_a_inv_times_b(self):
    grid_lengths = (2.1, 2.2, 2.33)
    self.set_params(rank=3, grid_lengths=grid_lengths)
    a_inv = multigrid_utils.inverse_laplacian_matrix(self.shape, grid_lengths)
    n = 100
    weight = 1

    # Transfer boundaries of b to x.
    zero_boundary_x = np.pad(
        self.x[self.inner_slices], ((1, 1),) * 3, 'constant')
    boundary_b = np.copy(self.b)
    boundary_b[self.inner_slices] = 0
    self.x = zero_boundary_x + boundary_b

    actual = self.evaluate(
        multigrid_utils.poisson_jacobi(
            tf.convert_to_tensor(self.x), tf.convert_to_tensor(self.b),
            self.params, n, weight))
    exact = self.evaluate(
        multigrid_utils.matmul(a_inv, tf.convert_to_tensor(self.b)))

    self.assertAllClose(np.abs(actual - exact).max(), 0.0376, atol=1e-4)

    # Run Jacobi further to confirm that the error decreases.
    actual = self.evaluate(
        multigrid_utils.poisson_jacobi(
            tf.convert_to_tensor(actual), tf.convert_to_tensor(self.b),
            self.params, n, weight))

    self.assertAllClose(np.abs(actual - exact).max(), 0.008, atol=1e-5)

  BCTYPES = (BCType.DIRICHLET, BCType.NEUMANN)
  PARAMS = itertools.product(BCTYPES, BCTYPES, range(1, 4))

  @parameterized.parameters(PARAMS)
  def test_laplacian_inverse_various_bcs_np_f64_(
      self, bctype_1, bctype_2, rank):
    if bctype_1 == BCType.NEUMANN and bctype_2 == BCType.NEUMANN: return
    atol = 1e-12
    norm_tol = 1e-10

    shape = (5, 7, 11)[:rank]
    grid_lengths = [s - 1 for s in shape]
    b = np.random.rand(*shape).astype(np.float64)

    bcs = [((bctype_1, 1.), (bctype_2, -1.))] * rank

    args = (shape, grid_lengths, bcs, np.float64)
    a = multigrid_utils.laplacian_matrix(*args)
    a_inv = multigrid_utils.inverse_laplacian_matrix(*args)

    x_expected = multigrid_utils.solve(a, b)
    x_actual = multigrid_utils.matmul(a_inv, b)

    self.assertAllClose(x_actual, x_expected, atol=atol)

    # Compute params, which require 3d inputs.
    computation_shape = (1, 1, 1)
    shape_3 = list(shape)
    grid_lengths_3 = list(grid_lengths)
    for _ in range(3 - len(shape)):
      shape_3.append(1)
      grid_lengths_3.append(1)

    params = (grid_parametrization.GridParametrization.
              create_from_grid_lengths_and_etc(
                  grid_lengths_3, computation_shape, shape_3, halo_width=1))

    actual_res_norm = multigrid_utils.poisson_residual_norm(x_actual, b, params)

    self.assertLess(actual_res_norm, norm_tol)


class MultigridUtilsApplyBoundaryConditionsTest(tf.test.TestCase):
  """Tests apply boundary conditions function."""

  def setUp(self):
    super().setUp()
    self.boundary_conditions = [((BCType.DIRICHLET, 1.), (BCType.NEUMANN, 2.)),
                                ((BCType.NO_TOUCH, 3.), (BCType.DIRICHLET, 4.)),
                                ((BCType.NEUMANN, 5.), (BCType.NO_TOUCH, 6.))]
    self.x = np.array([[[1., 2, 3], [4, 5, 6], [7, 8, 9]],
                       [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
                       [[21, 22, 23], [24, 25, 26], [27, 28, 29]]])

  def test_get_apply_one_core_boundary_conditions_fn(self):
    apply_fn = multigrid_utils.get_apply_one_core_boundary_conditions_fn(
        self.boundary_conditions)
    actual = apply_fn(self.x)

    # Expected after dim 0:
    # [[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    #  [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
    #  [[11 + 2, 12 + 2, 13 + 2], [16, 17, 18], [19, 20, 21]]]

    # Expected after dim 1:
    # [[[1, 1, 1], [1, 1, 1], [4, 4, 4]],
    #  [[11, 12, 13], [14, 15, 16], [4, 4, 4]],
    #  [[13, 14, 15], [16, 17, 18], [4, 4, 4]]]

    expected = [[[1 - 5, 1, 1], [1 - 5, 1, 1], [4 - 5, 4, 4]],
                [[12 - 5, 12, 13], [15 - 5, 15, 16], [4 - 5, 4, 4]],
                [[14 - 5, 14, 15], [17 - 5, 17, 18], [4 - 5, 4, 4]]]

    self.assertAllEqual(actual, expected)

  def test_get_apply_one_core_boundary_conditions_fn_homogeneous(self):
    apply_fn = multigrid_utils.get_apply_one_core_boundary_conditions_fn(
        self.boundary_conditions, homogeneous=True)
    actual = apply_fn(self.x)

    # Expected after dim 0:
    # [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #  [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
    #  [[11 + 0, 12 + 0, 13 + 0], [14, 15, 16], [17, 18, 19]]]

    # Expected after dim 1:
    # [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #  [[11, 12, 13], [14, 15, 16], [0, 0, 0]],
    #  [[11, 12, 13], [14, 15, 16], [0, 0, 0]]]

    expected = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[12, 12, 13], [15, 15, 16], [0, 0, 0]],
                [[12, 12, 13], [15, 15, 16], [0, 0, 0]]]

    self.assertAllEqual(actual, expected)

  def test_get_apply_one_core_boundary_conditions_fn_all_neumann(self):
    boundary_conditions = [((BCType.NEUMANN, 1.0), (BCType.NEUMANN, 2.0)),
                           ((BCType.NEUMANN, 3.0), (BCType.NEUMANN, 4.0)),
                           ((BCType.NEUMANN, 5.0), (BCType.NEUMANN, 6.0))]
    apply_fn = multigrid_utils.get_apply_one_core_boundary_conditions_fn(
        boundary_conditions)
    actual = apply_fn(self.x)

    # Expected after dim 0:
    # [[[11 - 1, 12 - 1, 13 - 1], [13, 14, 15], [16, 17, 18]],
    #  [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
    #  [[11 + 2, 12 + 2, 13 + 2], [16, 17, 18], [19, 20, 21]]]

    # Expected after dim 1:
    # [[[13 - 3, 14 - 3, 15 - 3], [13, 14, 15], [13 + 4, 14 + 4, 15 + 4]],
    #  [[11, 12, 13], [14, 15, 16], [18, 19, 20]],
    #  [[13, 14, 15], [16, 17, 18], [20, 21, 22]]]

    expected = np.array(
        [[[11. - 5, 11, 11 + 6], [14 - 5, 14, 14 + 6], [13, 18, 24]],
         [[12 - 5, 12, 12 + 6], [15 - 5, 15, 15 + 6], [14, 19, 25]],
         [[14 - 5, 14, 14 + 6], [17 - 5, 17, 17 + 6], [16, 21, 27]]])
    # In an all-Neumann case the mean (excluding the boundary) is subtracted.
    expected -= 15

    self.assertAllEqual(actual, expected)


class JacobiConvergenceTest(multigrid_test_common.ConvergenceTest,
                            parameterized.TestCase):

  @parameterized.parameters(
      multigrid_test_common.ConvergenceTest.JACOBI_DIFFS_NORMS_5X5X5)
  def test_jacobi_convergence_no_source_5x5x5_(self, tr_axis, params):
    """Jacobi convergence test for 5x5x5."""
    n, expected_diffs, expected_norms = params
    starting, expected = self.starting_and_expected_5x5x5(tr_axis)
    name = f'shape={expected.shape}, n={n}'
    solver = multigrid_utils.poisson_jacobi_fn_for_one_core(n=n)
    self._run_convergence_test(solver, name, starting, expected, expected_diffs,
                               expected_norms, using_tiles=False)

  @parameterized.parameters(
      multigrid_test_common.ConvergenceTest.JACOBI_DIFFS_NORMS_WEIGHT_2_3_5X5X5)
  def test_jacobi_convergence_no_source_weight_2_3_5x5x5_(
      self, tr_axis, params):
    """Jacobi with weight 2/3 convergence test for 5x5x5."""
    n, expected_diffs, expected_norms = params
    starting, expected = self.starting_and_expected_5x5x5(tr_axis)
    name = f'shape={expected.shape}, n={n}'
    weight = 2 / 3
    solver = multigrid_utils.poisson_jacobi_fn_for_one_core(n=n, weight=weight)
    self._run_convergence_test(solver, name, starting, expected, expected_diffs,
                               expected_norms)


class MultigridMiscTest(tf.test.TestCase, parameterized.TestCase):
  """Tests prolong, restrict and einsum related utils."""

  def test_zero_borders_1d(self):
    x = np.array([1, 2, 3, 4, 8, 9])
    expected = [0, 2, 3, 4, 8, 0]
    self.assertAllEqual(multigrid_utils.zero_borders(x), expected)

  def test_zero_borders_2d(self):
    x = np.array([[1, 2, 3, 4, 8, 9], [1, 2, 3, 4, 8, 9], [1, 3, 4, 5, 8, 9],
                  [1, 4, 5, 6, 8, 9], [1, 2, 3, 4, 8, 9]])
    expected = [[0, 0, 0, 0, 0, 0], [0, 2, 3, 4, 8, 0], [0, 3, 4, 5, 8, 0],
                [0, 4, 5, 6, 8, 0], [0, 0, 0, 0, 0, 0]]
    self.assertAllEqual(multigrid_utils.zero_borders(x), expected)

  def test_add_borders_1d(self):
    x = np.array([1, 2, 3, 4, 8, 9])
    y = np.array([2, 3, 4, 8, 9, 5])
    expected = [3, 2, 3, 4, 8, 14]
    self.assertAllEqual(multigrid_utils.add_borders(x, y), expected)

  def test_add_borders_2d(self):
    x = np.array([[1, 2, 3, 4, 8, 9], [1, 2, 3, 4, 8, 9], [1, 3, 4, 5, 8, 9],
                  [1, 4, 5, 6, 8, 9], [-1, 0, 2, 3, 4, 5]])
    expected = [[2, 4, 6, 8, 16, 18], [2, 2, 3, 4, 8, 18], [2, 3, 4, 5, 8, 18],
                [2, 4, 5, 6, 8, 18], [-2, 0, 4, 6, 8, 10]]
    self.assertAllEqual(multigrid_utils.add_borders(x, x), expected)

  def test_prolong_matrix_3_5(self):
    actual0 = multigrid_utils.prolong_matrix(5, 3)
    actual1 = multigrid_utils.prolong_matrix(5)
    expected = np.array([[1, 0, 0],
                         [0.5, 0.5, 0],
                         [0, 1, 0],
                         [0, 0.5, 0.5],
                         [0, 0, 1]])
    self.assertAllEqual(actual0, expected)
    self.assertAllEqual(actual1, expected)

  def test_apply_prolong_matrix_constant_3_5(self):
    p = multigrid_utils.prolong_matrix(5)
    v = np.ones((3,), np.float32).T
    p_v = p @ v
    expected = np.ones((5,), np.float32).T
    self.assertAllEqual(p_v, expected)
    p_v = multigrid_utils.kronecker_products([p], v)
    self.assertAllEqual(p_v, expected)

  def test_apply_prolong_matrix_constant_17_22(self):
    p = multigrid_utils.prolong_matrix(22, 17)
    v = np.ones((17,), np.float32).T
    p_v = p @ v
    expected = np.ones((22,), np.float32).T
    self.assertAllEqual(p_v, expected)
    p_v = multigrid_utils.kronecker_products([p], v)
    self.assertAllEqual(p_v, expected)

  def test_apply_prolong_matrix_linear_3_5(self):
    p = multigrid_utils.prolong_matrix(5)
    v = np.array([1., 2., 3.]).T
    p_v = p @ v
    expected = np.linspace(1., 3., 5).T
    self.assertAllEqual(p_v, expected)
    p_v = multigrid_utils.kronecker_products([p], v)
    self.assertAllEqual(p_v, expected)

  def test_apply_prolong_matrix_linear_17_22(self):
    p = multigrid_utils.prolong_matrix(22, 17)
    v = np.linspace(5., 6., 17).T
    p_v = p @ v
    expected = np.linspace(5., 6., 22).T
    self.assertAllClose(p_v, expected, rtol=1e-15)
    p_v = multigrid_utils.kronecker_products([p], v)
    self.assertAllClose(p_v, expected, rtol=1e-15)

  def test_prolong_2d_constant_linear_3_3_to_6_5(self):
    m = []
    for _ in range(3):
      m.append(np.linspace(1, 5, 3))
    m = np.stack(m)
    ps = (multigrid_utils.prolong_matrix(6, 3),
          multigrid_utils.prolong_matrix(5, 3))
    actual = multigrid_utils.kronecker_products(ps, m)
    expected = ((1, 2, 3, 4, 5),) * 6
    self.assertAllClose(actual, expected)

  def test_apply_restrict_matrix_constant_17_9(self):
    # If n2 == 2 * n1 - 1, a constant is restricted to a constant, except at the
    # endpoints.
    r = multigrid_utils.restrict_matrix(17)
    v = np.ones((17,), np.float32).T
    r_v = r @ v
    expected = np.ones((9,), np.float32).T
    self.assertAllEqual(r_v[1:-1], expected[1:-1])
    self.assertAllClose(r_v, expected, rtol=0.25)
    r_v = multigrid_utils.kronecker_products([r], v)
    self.assertAllEqual(r_v[1:-1], expected[1:-1])
    self.assertAllClose(r_v, expected, rtol=0.25)

  def test_apply_restrict_matrix_constant_22_17(self):
    # If n2 != 2 * n1 - 1, a constant only approximately restricts to a
    # constant.
    r = multigrid_utils.restrict_matrix(22, 17)
    v = np.ones((22,), np.float32).T
    r_v = r @ v
    expected = np.ones((17,), np.float32).T
    self.assertAllClose(r_v, expected, rtol=0.1)
    r_v = multigrid_utils.kronecker_products([r], v)
    self.assertAllClose(r_v, expected, rtol=0.1)

  def test_apply_restrict_matrix_linear_17_9(self):
    # If n2 == 2 * n1 - 1, linear restricts to linear, except at the endpoints.
    r = multigrid_utils.restrict_matrix(17)
    v = np.linspace(5., 6., 17).T
    r_v = r @ v
    expected = np.linspace(5., 6., 9).T
    self.assertAllEqual(r_v[1:-1], expected[1:-1])
    self.assertAllClose(r_v, expected, rtol=0.26)
    r_v = multigrid_utils.kronecker_products([r], v)
    self.assertAllEqual(r_v[1:-1], expected[1:-1])
    self.assertAllClose(r_v, expected, rtol=0.26)

  def test_apply_restrict_matrix_linear_22_17(self):
    # If n2 != 2 * n1 - 1, linear restricts to approximately linear.
    r = multigrid_utils.restrict_matrix(22, 17)
    v = np.linspace(5., 6., 22).T
    r_v = r @ v
    expected = np.linspace(5., 6., 17).T
    self.assertAllClose(r_v, expected, rtol=0.1)
    r_v = multigrid_utils.kronecker_products([r], v)
    self.assertAllClose(r_v, expected, rtol=0.1)

  def test_restrict_2d_constant_linear_17_10_to_6_5(self):
    m = []
    for _ in range(17):
      m.append(np.linspace(1, 5, 10))
    m = np.stack(m)
    rs = (multigrid_utils.restrict_matrix(17, 6),
          multigrid_utils.restrict_matrix(10, 5))
    actual = multigrid_utils.kronecker_products(rs, m)
    expected = np.array(((1, 2, 3, 4, 5),) * 6)
    inner = (slice(1, -1),) * 2
    self.assertAllClose(actual[inner], expected[inner], rtol=0.03)

  def test_full_1d_grid_size_pairs(self):
    actual = multigrid_utils.full_1d_grid_size_pairs(n=66, num_cores=2,
                                                     coarsest_subgrid_size=4)
    expected = [(66, 34), (34, 18), (18, 10), (10, 6)]
    self.assertEqual(actual, expected)

    actual = multigrid_utils.full_1d_grid_size_pairs(n=66, num_cores=4,
                                                     coarsest_subgrid_size=3)
    expected = [(66, 34), (34, 18), (18, 10), (10, 6)]
    self.assertEqual(actual, expected)

  def test_prolong_matrix_3_17(self):
    actual = multigrid_utils.prolong_matrix(17, 3)
    expected = np.array([[1, 0, 0],
                         [0.875, 0.125, 0],
                         [0.75, 0.25, 0],
                         [0.625, 0.375, 0],
                         [0.5, 0.5, 0],
                         [0.375, 0.625, 0],
                         [0.25, 0.75, 0],
                         [0.125, 0.875, 0],
                         [0, 1, 0],
                         [0, 0.875, 0.125],
                         [0, 0.75, 0.25],
                         [0, 0.625, 0.375],
                         [0, 0.5, 0.5],
                         [0, 0.375, 0.625],
                         [0, 0.25, 0.75],
                         [0, 0.125, 0.875],
                         [0, 0, 1]])
    self.assertAllEqual(actual, expected)

  def run_test_prolong_restrict_matrices(self, start_shape, end_shape,
                                         expected_p_shapes):
    shapes = lambda ms: [None if m is None else m.shape for m in ms]
    all_shapes = lambda mss: [shapes(ms) for ms in mss]

    shapes_trans = lambda ms: [None if m is None else m[::-1] for m in ms]
    all_shapes_trans = lambda mss: [shapes_trans(ms) for ms in mss]

    ps, rs = multigrid_utils.prolong_restrict_matrices(start_shape,
                                                       end_shape)
    self.assertEqual(len(ps), len(expected_p_shapes))

    self.assertAllEqual(all_shapes(ps), expected_p_shapes)
    expected_r_shapes = all_shapes_trans(expected_p_shapes)
    self.assertAllEqual(all_shapes(rs), expected_r_shapes)

  def test_prolong_restrict_matrices_17_to_3(self):
    start_shape = (17,)
    end_shape = (3,)
    expected_p_shapes = [[(17, 9)],
                         [(9, 5)],
                         [(5, 3)]]
    self.run_test_prolong_restrict_matrices(start_shape, end_shape,
                                            expected_p_shapes)

  def test_prolong_restrict_matrices_5_17_to_3(self):
    start_shape = (5, 17)
    end_shape = (3, 3)
    expected_p_shapes = [[(5, 3), (17, 9)],
                         [None, (9, 5)],
                         [None, (5, 3)]]
    self.run_test_prolong_restrict_matrices(start_shape, end_shape,
                                            expected_p_shapes)

  def test_prolong_restrict_matrices_3_9_to_3(self):
    start_shape = (3, 9)
    end_shape = (3, 3)
    expected_p_shapes = [[None, (9, 5)],
                         [None, (5, 3)]]
    self.run_test_prolong_restrict_matrices(start_shape, end_shape,
                                            expected_p_shapes)

  def test_prolong_restrict_matrices_4_5_17_to_3(self):
    start_shape = (4, 5, 17)
    end_shape = (3, 3, 3)
    expected_p_shapes = [[(4, 3), (5, 3), (17, 9)],
                         [None, None, (9, 5)],
                         [None, None, (5, 3)]]
    self.run_test_prolong_restrict_matrices(start_shape, end_shape,
                                            expected_p_shapes)

  def test_prolong_restrict_matrices_7_21_4_to_3(self):
    start_shape = (7, 21, 4)
    end_shape = (3, 3, 3)
    expected_p_shapes = [[(7, 4), (21, 11), (4, 3)],
                         [(4, 3), (11, 6), None],
                         [None, (6, 3), None]]
    self.run_test_prolong_restrict_matrices(start_shape, end_shape,
                                            expected_p_shapes)

  def test_prolong_restrict_matrices_17_to_4(self):
    start_shape = (17,)
    end_shape = (4,)
    expected_p_shapes = [[(17, 9)],
                         [(9, 5)],
                         [(5, 4)]]
    self.run_test_prolong_restrict_matrices(start_shape, end_shape,
                                            expected_p_shapes)

  def test_prolong_restrict_matrices_5_17_to_4(self):
    start_shape = (5, 17)
    end_shape = (4, 4)
    expected_p_shapes = [[(5, 4), (17, 9)],
                         [None, (9, 5)],
                         [None, (5, 4)]]
    self.run_test_prolong_restrict_matrices(start_shape, end_shape,
                                            expected_p_shapes)

  def test_prolong_restrict_matrices_4_9_to_4(self):
    start_shape = (4, 9)
    end_shape = (4, 4)
    expected_p_shapes = [[None, (9, 5)],
                         [None, (5, 4)]]
    self.run_test_prolong_restrict_matrices(start_shape, end_shape,
                                            expected_p_shapes)

  def test_prolong_restrict_matrices_4_5_17_to_4(self):
    start_shape = (4, 5, 17)
    end_shape = (4, 4, 4)
    expected_p_shapes = [[None, (5, 4), (17, 9)],
                         [None, None, (9, 5)],
                         [None, None, (5, 4)]]
    self.run_test_prolong_restrict_matrices(start_shape, end_shape,
                                            expected_p_shapes)

  def test_prolong_restrict_matrices_7_21_4_to_4(self):
    start_shape = (9, 21, 6)
    end_shape = (4, 4, 4)
    expected_p_shapes = [[(9, 5), (21, 11), (6, 4)],
                         [(5, 4), (11, 6), None],
                         [None, (6, 4), None]]
    self.run_test_prolong_restrict_matrices(start_shape, end_shape,
                                            expected_p_shapes)

  def test_prolong_restrict_matrices_from_params(self):
    grid_lengths = (1, 1, 1)
    subgrid_shape = (9, 28, 299)
    end_shape = (3, 4, 4)
    computation_shape = (1, 2, 2)
    params = (grid_parametrization.GridParametrization.
              create_from_grid_lengths_and_etc(
                  grid_lengths, computation_shape, subgrid_shape, halo_width=1))
    expected_shapes = (((9, 5), (54, 28), (596, 300)),
                       ((5, 3), (28, 16), (300, 152)),
                       (None, (16, 10), (152, 78)),
                       (None, (10, 6), (78, 40)),
                       (None, None, (40, 22)),
                       (None, None, (22, 12)),
                       (None, None, (12, 8)),
                       (None, None, (8, 6)))
    ps, _ = multigrid_utils.prolong_restrict_matrices_from_params(
        params, end_shape)
    self.assertEqual(len(ps), len(expected_shapes))
    for level in range(len(ps)):
      for dim in range(len(ps[level])):
        actual_shape = None if ps[level][dim] is None else ps[level][dim].shape
        self.assertEqual(expected_shapes[level][dim], actual_shape)

  def test_prolong_restrict_matrices_from_params_more_cores(self):
    grid_lengths = (1, 1, 1)
    subgrid_shape = (9, 29, 114)
    end_shape = (3, 3, 3)
    computation_shape = (16, 4, 1)
    params = (grid_parametrization.GridParametrization.
              create_from_grid_lengths_and_etc(
                  grid_lengths, computation_shape, subgrid_shape, halo_width=1))
    expected_shapes = (((114, 66), (110, 58), (114, 58)),
                       ((66, 34), (58, 30), (58, 30)),
                       ((34, 18), (30, 18), (30, 16)),
                       (None, (18, 10), (16, 9)),
                       (None, (10, 6), (9, 5)),
                       (None, None, (5, 3)))
    ps, _ = multigrid_utils.prolong_restrict_matrices_from_params(
        params, end_shape)
    self.assertEqual(len(ps), len(expected_shapes))
    for level in range(len(ps)):
      for dim in range(len(ps[level])):
        actual_shape = None if ps[level][dim] is None else ps[level][dim].shape
        self.assertEqual(expected_shapes[level][dim], actual_shape)

  def test_get_ps_rs_init_fn(self):
    grid_lengths = (1, 1, 1)
    shape = (24, 17, 15)
    coarsest_subgrid_shape = (5, 5, 5)
    computation_shape = (2, 2, 2)
    params = (grid_parametrization.GridParametrization.
              create_from_grid_lengths_and_etc(
                  grid_lengths, computation_shape, shape, halo_width=1))

    init_fn = multigrid_utils.get_ps_rs_init_fn(params, coarsest_subgrid_shape)
    subgrid = lambda f: (f - 2) // 2 + 2

    d = init_fn((0, 0, 0))

    # Checking first subgrid, level = 0, dim = 0.
    n1, n2 = 46, 24
    p_actual = d['ps']['0_0']
    p_full = multigrid_utils.prolong_matrix(n1, n2)
    p_expected = p_full[:subgrid(n1), :subgrid(n2)]
    self.assertAllEqual(p_actual, p_expected)

    # Checking first subgrid, level = 1, dim = 2.
    n1, n2 = 16, 10
    p_actual = d['ps']['1_2']
    p_full = multigrid_utils.prolong_matrix(n1, n2)
    p_expected = p_full[:subgrid(n1), :subgrid(n2)]
    self.assertAllEqual(p_actual, p_expected)

    d = init_fn((1, 1, 1))

    # Checking second subgrid, level = 0, dim = 0.
    n1, n2 = 46, 24
    p_actual = d['ps']['0_0']
    p_full = multigrid_utils.prolong_matrix(n1, n2)
    p_expected = p_full[subgrid(n1) - 2:, subgrid(n2) - 2:]
    self.assertAllEqual(p_actual, p_expected)

    # Checking second subgrid, level = 1, dim = 2.
    n1, n2 = 16, 10
    p_actual = d['ps']['1_2']
    p_full = multigrid_utils.prolong_matrix(n1, n2)
    p_expected = p_full[subgrid(n1) - 2:, subgrid(n2) - 2:]
    self.assertAllEqual(p_actual, p_expected)

  @parameterized.parameters(
      (2, 0, 'zy,yb->zb'),
      (2, 1, 'zy,ay->az'),
      (3, 0, 'zy,ybc->zbc'),
      (3, 1, 'zy,ayc->azc'),
      (3, 2, 'zy,aby->abz'))
  def test_einsum_indices_(self, n, i, expected):
    actual = multigrid_utils.kronecker_einsum_indices(n, i)
    self.assertEqual(actual, expected)

  RANKS = (1, 2, 3)
  BC_TYPES = (halo_exchange.BCType.DIRICHLET, halo_exchange.BCType.NEUMANN)

  @parameterized.parameters(itertools.product(RANKS, BC_TYPES, BC_TYPES))
  def test_get_homogeneous_boundary_conditions_(self, rank, bc_type_1,
                                                bc_type_2):
    boundary_conditions = [((bc_type_1, 1.), (bc_type_2, 2.)),
                           ((bc_type_1, 3.), (bc_type_2, 4.)),
                           ((bc_type_1, 5.), (bc_type_2, 6.))][:rank]
    expected = [((bc_type_1, 0.), (bc_type_2, 0.))] * rank
    actual = multigrid_utils.get_homogeneous_boundary_conditions(
        boundary_conditions)

    self.assertAllEqual(actual, expected)


class MultigridUtilsMultiCoreTest(tf.test.TestCase, parameterized.TestCase):

  def init_fn(self, coordinates,
              params: grid_parametrization.GridParametrization):
    return {
        'x': initializer.subgrid_of_3d_grid_from_params(
            self._x, params, coordinates),
        'b': initializer.subgrid_of_3d_grid_from_params(
            self._b, params, coordinates),
    }

  def step_fn(self, state, replicas, replica_id):
    x = state['x']
    b = state['b']
    poisson_jacobi = self._jacobi_step_fn(replica_id, replicas)
    return poisson_jacobi(x, b)

  def maybe_combine_xs(self, xs, computation_shape):
    x0 = xs[0]
    if np.prod(computation_shape) == 1:
      return x0
    else:
      x1 = xs[1]
      axis = {(2, 1, 1): 0, (1, 2, 1): 1, (1, 1, 2): 2}[computation_shape]
      if axis == 0:
        return np.concatenate((x0[:-1, :, :], x1[1:, :, :]), axis=axis)
      elif axis == 1:
        return np.concatenate((x0[:, :-1, :], x1[:, 1:, :]), axis=axis)
      else:
        return np.concatenate((x0[:, :, :-1], x1[:, :, 1:]), axis=axis)

  def get_poisson_jacobi_result(
      self, params: grid_parametrization.GridParametrization) -> np.ndarray:
    computation_shape = params.cx, params.cy, params.cz
    runner = TpuRunner(computation_shape=computation_shape)
    init_fn = functools.partial(self.init_fn, params=params)
    coordinates = util.grid_coordinates(computation_shape)
    xs = runner.run_with_replica_args(
        self.step_fn,
        [init_fn(coordinates[i]) for i in range(np.prod(computation_shape))])
    return self.maybe_combine_xs(xs, computation_shape)

  @parameterized.parameters((2, 1, 1), (1, 2, 1), (1, 1, 2))
  def test_poisson_jacobi_compare_one_core_to_two_(self, cx, cy, cz):
    grid_lengths = (13.3, 14.4, 15.5)
    subgrid_shape = (6, 9, 12)
    full_grid_shape = [2 * n - 2 for n in subgrid_shape]

    self._x = np.random.rand(*full_grid_shape).astype(np.float32)
    self._b = np.random.rand(*full_grid_shape).astype(np.float32)
    n_jacobi = 20  # Number of Jacobi iterations.

    # One core params.
    computation_shape_1_core = (1, 1, 1)
    params_1_core = (grid_parametrization.GridParametrization.
                     create_from_grid_lengths_and_etc(
                         grid_lengths, computation_shape_1_core,
                         full_grid_shape, halo_width=1))

    # One core numpy result.
    weight = 1
    one_core_np = multigrid_utils.poisson_jacobi(
        np.copy(self._x), self._b, params_1_core, n_jacobi, weight)
    # One core tensorflow result using multicore code.
    self._jacobi_step_fn = multigrid_utils.poisson_jacobi_step_fn(
        params_1_core, n_jacobi)
    one_core_tf = self.get_poisson_jacobi_result(params_1_core)

    self.assertAllClose(one_core_np, one_core_tf)

    # Two cores.
    boundary_conditions = [[(halo_exchange.BCType.NO_TOUCH, 0.)] * 2] * 3
    computation_shape = (cx, cy, cz)
    subgrid_shape_2_cores = full_grid_shape[:]
    if cx == 2:
      subgrid_shape_2_cores[0] = subgrid_shape[0]
    elif cy == 2:
      subgrid_shape_2_cores[1] = subgrid_shape[1]
    else:
      subgrid_shape_2_cores[2] = subgrid_shape[2]

    params_2_cores = (grid_parametrization.GridParametrization.
                      create_from_grid_lengths_and_etc(
                          grid_lengths, computation_shape,
                          subgrid_shape_2_cores, halo_width=1))
    self._jacobi_step_fn = multigrid_utils.poisson_jacobi_step_fn(
        params_2_cores, n_jacobi,
        boundary_conditions=boundary_conditions)

    two_core_tf = self.get_poisson_jacobi_result(params_2_cores)

    self.assertAllEqual(one_core_tf, two_core_tf)

  def test_poisson_jacobi_neumann_bcs_convergence(self):
    """Tests convergence for Jacobi with a Neumann boundary condition."""
    grid_lengths = (4, 4, 4)
    subgrid_shape = (3, 3, 3)
    computation_shape = (1, 1, 1)
    full_grid_shape = [2 * n - 2 for n in subgrid_shape]

    self._x = np.zeros(full_grid_shape).astype(np.float32)
    self._b = np.zeros(full_grid_shape).astype(np.float32)

    n_jacobi = 10  # Number of Jacobi iterations.

    special_number = 0.2229885
    # Boundary conditions are defined so that, given the Neumann boundary
    # condition, in the solution element [0, 1, 1] is 0 and element [1, 1, 1] is
    # the special number.
    bcs = (((BCType.NEUMANN, special_number), (BCType.DIRICHLET, 1.0)),
           ((BCType.DIRICHLET, 0.0), (BCType.DIRICHLET, 1.0)),
           ((BCType.DIRICHLET, 0.0), (BCType.DIRICHLET, 1.0)))

    params = (grid_parametrization.GridParametrization.
              create_from_grid_lengths_and_etc(
                  grid_lengths, computation_shape,
                  subgrid_shape, halo_width=1))

    self._jacobi_step_fn = multigrid_utils.poisson_jacobi_step_fn(
        params, n_jacobi, boundary_conditions=bcs)

    # Convergence is very fast on a small grid.
    expected_0_1_1_values = (-0.006444171, -4.0993094e-05, -2.2351742e-07, 0)
    expected_1_1_1_values = (0.21654433, 0.22294751, 0.22298828, special_number)

    computation_shape = params.cx, params.cy, params.cz
    runner = TpuRunner(computation_shape=computation_shape)
    init_fn = functools.partial(self.init_fn, params=params)
    coordinates = util.grid_coordinates(computation_shape)
    state = [init_fn(coordinates[i]) for i in range(np.prod(computation_shape))]

    for i in range(4):
      xs = runner.run_with_replica_args(self.step_fn, state)
      for state_i, x in zip(state, xs):
        state_i['x'] = x
      actual = self.maybe_combine_xs(xs, computation_shape)
      self.assertNear(expected_0_1_1_values[i], actual[0, 1, 1], err=1e-7)
      self.assertNear(expected_1_1_1_values[i], actual[1, 1, 1], err=1e-7)

  @parameterized.parameters((2, 1, 1), (1, 2, 1), (1, 1, 2))
  def test_poisson_jacobi_compare_1_core_to_2_neumann_bcs_(self, cx, cy, cz):
    grid_lengths = (4, 44, 444)
    subgrid_shape = (3, 10, 25)
    computation_shape_1_core = (1, 1, 1)
    full_grid_shape = [2 * n - 2 for n in subgrid_shape]

    self._x = np.random.rand(*full_grid_shape).astype(np.float32)
    self._b = np.random.rand(*full_grid_shape).astype(np.float32)

    n_jacobi = 40  # Number of Jacobi iterations.

    bcs = (((BCType.NEUMANN, -0.4), (BCType.DIRICHLET, 0.8)),
           ((BCType.DIRICHLET, 0.0), (BCType.NEUMANN, -1.3)),
           ((BCType.DIRICHLET, 0.6), (BCType.DIRICHLET, 1.1)))

    # One core params.
    params = (grid_parametrization.GridParametrization.
              create_from_grid_lengths_and_etc(
                  grid_lengths, computation_shape_1_core,
                  full_grid_shape, halo_width=1))

    # One core tensorflow result using multicore code.
    self._jacobi_step_fn = multigrid_utils.poisson_jacobi_step_fn(
        params, n_jacobi, boundary_conditions=bcs)

    one_core_result = np.copy(self.get_poisson_jacobi_result(params))

    # Two cores.
    computation_shape = (cx, cy, cz)
    subgrid_shape_2_cores = full_grid_shape[:]
    if cx == 2:
      subgrid_shape_2_cores[0] = subgrid_shape[0]
    elif cy == 2:
      subgrid_shape_2_cores[1] = subgrid_shape[1]
    else:
      subgrid_shape_2_cores[2] = subgrid_shape[2]
    params_2_cores = (grid_parametrization.GridParametrization.
                      create_from_grid_lengths_and_etc(
                          grid_lengths, computation_shape,
                          subgrid_shape_2_cores, halo_width=1))

    self._jacobi_step_fn = multigrid_utils.poisson_jacobi_step_fn(
        params_2_cores, n_jacobi, boundary_conditions=bcs)

    two_core_result = self.get_poisson_jacobi_result(params_2_cores)

    self.assertAllEqual(one_core_result, two_core_result)


if __name__ == '__main__':
  tf.test.main()
