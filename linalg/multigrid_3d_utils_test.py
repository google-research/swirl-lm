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
"""Tests for multigrid_3d_utils."""
import functools
import itertools

import numpy as np
from swirl_lm.communication import halo_exchange
from swirl_lm.linalg import multigrid_3d_utils
from swirl_lm.linalg import multigrid_test_common
from swirl_lm.linalg import multigrid_utils
from swirl_lm.utility import grid_parametrization
import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.framework import initializer
from google3.research.simulation.tensorflow.fluid.framework import util
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized

BCType = halo_exchange.BCType

INNER = (slice(1, -1),) * 3


class Multigrid3dUtilsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests multigrid_3d_utils functions."""

  def compare_np_tf_and_tpu(self, expected, tf_fn, atol, post_fn=None):
    with self.subTest('tf_cpu'):
      actual = self.evaluate(tf_fn)
      if isinstance(actual, list):
        actual = np.stack(actual, axis=-1)
      if post_fn is not None:
        actual = post_fn(actual)
      self.assertAllClose(expected, actual, atol=atol)
    with self.subTest('tf_tpu'):
      tpu = TpuRunner(computation_shape=(1,))
      actual = tpu.run(tf_fn)[0]
      if isinstance(actual, list):
        actual = np.stack(actual, axis=-1)
      if post_fn is not None:
        actual = post_fn(actual)
      self.assertAllClose(expected, actual, atol=atol)

  def set_params(self, grid_lengths):
    self.shape = (13, 19, 24)
    self.x = np.random.rand(*self.shape)
    self.b = np.random.rand(*self.shape)
    self.x_tiles = [self.x[:, :, i] for i in range(self.shape[2])]
    self.b_tiles = [self.b[:, :, i] for i in range(self.shape[2])]

    computation_shape = (1, 1, 1)
    self.params = (
        grid_parametrization.GridParametrization
        .create_from_grid_lengths_and_etc_with_defaults(grid_lengths,
                                                        computation_shape))

    self.dxs = [length / (s - 1) for length, s in zip(grid_lengths, self.shape)]

    self.sum_inverse_spacing_squared = sum([dx**-2 for dx in self.dxs])

  # Numpy reference implementations.
  def lap_inv_diagonal(self):
    return self.x / (-2 * self.sum_inverse_spacing_squared)

  def lap(self):
    res = -2 * self.sum_inverse_spacing_squared * self.x
    for i in range(self.x.ndim):
      r_slices = tuple([slice(1, -1)] * i + [slice(2, None)] + [slice(1, -1)] *
                       (self.x.ndim - i - 1))
      l_slices = tuple([slice(1, -1)] * i + [slice(0, -2)] + [slice(1, -1)] *
                       (self.x.ndim - i - 1))
      res[INNER] += ((self.x[r_slices] + self.x[l_slices]) / (self.dxs[i]**2))
    return res

  def jac(self, n, weight):
    for _ in range(n):
      self.x[INNER] -= (
          weight / 2 / self.sum_inverse_spacing_squared *
          (self.b - self.lap())[INNER])
    return self.x

  @parameterized.parameters(((1, 1, 1),), ((1.1, 5.2, 9.33),))
  def test_laplacian_inv_diagonal_(self, grid_lengths):
    """Laplacian inverse diagonal function test."""
    self.set_params(grid_lengths)
    inv_diag_fn = multigrid_3d_utils.laplacian_and_inv_diagonal_fns(
        self.shape, grid_lengths)[1]
    x_tiles = [tf.convert_to_tensor(xy) for xy in self.x_tiles]
    expected = self.lap_inv_diagonal()
    self.compare_np_tf_and_tpu(
        expected[INNER],
        lambda: inv_diag_fn(x_tiles),
        atol=1e-10,
        post_fn=lambda t: t[INNER])

  @parameterized.parameters(((1, 1, 1),), ((1.1, 5.2, 9.33),))
  def test_laplacian_(self, grid_lengths):
    """Tests the Laplacian function."""
    self.set_params(grid_lengths)
    laplacian = multigrid_3d_utils.laplacian_and_inv_diagonal_fns(
        self.shape, grid_lengths)[0]
    x_tiles = [tf.convert_to_tensor(xy) for xy in self.x_tiles]
    self.compare_np_tf_and_tpu(
        self.lap()[INNER],
        lambda: laplacian(x_tiles),
        atol=1e-10,
        post_fn=lambda t: t[INNER])

  PARAMS = itertools.product(((1, 1, 1), (1.1, 5.2, 9.33)), (1, 2 / 3))

  @parameterized.parameters(PARAMS)
  def test_poisson_jacobi_(self, grid_lengths, weight):
    """Tests the Poisson Jacobi function."""
    self.set_params(grid_lengths)
    n = 111
    x_tiles = [tf.convert_to_tensor(xy) for xy in self.x_tiles]
    b_tiles = [tf.convert_to_tensor(xy) for xy in self.b_tiles]
    tf_fn = functools.partial(multigrid_3d_utils.poisson_jacobi, x_tiles,
                              b_tiles, self.params, n, weight)
    expected = self.jac(n, weight)
    self.compare_np_tf_and_tpu(expected, tf_fn, atol=1e-10)

  def test_zero_borders(self):
    x = np.random.rand(6, 9, 15)
    x_tiles = [x[:, :, i] for i in range(15)]
    actual_tiles = multigrid_3d_utils.zero_borders(x_tiles)
    actual = np.stack(actual_tiles, axis=-1)
    # Check that the interior is the same.
    interior = (slice(1, -1),) * 3
    self.assertAllEqual(x[interior], actual[interior])
    # Check that the borders are zero.
    self.assertAllEqual(np.zeros_like(x[0, :, :]), actual[0, :, :])
    self.assertAllEqual(np.zeros_like(x[:, 0, :]), actual[:, 0, :])
    self.assertAllEqual(np.zeros_like(x[:, :, 0]), actual[:, :, 0])


class JacobiConvergenceTest(multigrid_test_common.ConvergenceTest,
                            parameterized.TestCase):

  @parameterized.parameters(
      multigrid_test_common.ConvergenceTest.JACOBI_DIFFS_NORMS_5X5X5)
  def test_jacobi_convergence_no_source_5x5x5_(self, tr_axis, params):
    """Jacobi convergence test for 5x5x5."""
    n, expected_diffs, expected_norms = params
    starting, expected = self.starting_and_expected_5x5x5(tr_axis)
    name = f'shape={expected.shape}, n={n}'
    solver = multigrid_3d_utils.poisson_jacobi_fn_for_one_core(n=n)
    self._run_convergence_test(
        solver,
        name,
        starting,
        expected,
        expected_diffs,
        expected_norms,
        using_tiles=True)

  @parameterized.parameters(
      multigrid_test_common.ConvergenceTest.JACOBI_DIFFS_NORMS_WEIGHT_2_3_5X5X5)
  def test_jacobi_convergence_no_source_weight_2_3_5x5x5_(
      self, tr_axis, params):
    """Jacobi with weight 2/3 convergence test for 5x5x5."""
    n, expected_diffs, expected_norms = params
    starting, expected = self.starting_and_expected_5x5x5(tr_axis)
    name = f'shape={expected.shape}, n={n}'
    solver = multigrid_3d_utils.poisson_jacobi_fn_for_one_core(n=n, weight=2/3)
    self._run_convergence_test(
        solver,
        name,
        starting,
        expected,
        expected_diffs,
        expected_norms,
        using_tiles=True)


class UtilsComparisonTest(Multigrid3dUtilsTest, parameterized.TestCase):
  """Compares array and tile versions of various util functions."""

  def compare_tiles_and_residuals(self, expected_tiles, tiles_fn,
                                  expected_residual, residual_fn):
    with self.subTest('x_values'):
      self.assertAllClose(
          expected_tiles,
          np.stack(self.evaluate(tiles_fn), axis=-1),
          atol=1e-15)
    with self.subTest('residuals'):
      self.assertAllClose(
          expected_residual[INNER],
          np.stack(self.evaluate(residual_fn), axis=-1)[INNER],
          atol=1.7e-6)

  @parameterized.parameters(((1, 1, 1),), ((1.1, 5.2, 9.33),))
  def test_compare_laplacian_np_tf_(self, grid_lengths):
    """Compares the np multigrid_3d_utils and multigrid_utils Laplacian fns."""
    self.set_params(grid_lengths)
    laplacian, _ = multigrid_3d_utils.laplacian_and_inv_diagonal_fns(
        self.shape, grid_lengths)
    lap_np = np.stack(laplacian(self.x_tiles), axis=-1)
    x_tiles = [tf.convert_to_tensor(tile) for tile in self.x_tiles]
    self.compare_np_tf_and_tpu(
        lap_np[INNER],
        lambda: laplacian(x_tiles),
        atol=1e-15,
        post_fn=lambda x: x[INNER])

  @parameterized.parameters(((1, 1, 1),), ((1.1, 5.2, 9.33),))
  def test_compare_laplacian_3d_np_(self, grid_lengths):
    """Compares the np multigrid_3d_utils and multigrid_utils Laplacian fns."""
    self.set_params(grid_lengths)
    laplacian_3d, _ = multigrid_3d_utils.laplacian_and_inv_diagonal_fns(
        self.shape, grid_lengths)
    laplacian, _ = multigrid_utils.laplacian_and_inv_diagonal_fns(
        self.shape, grid_lengths)
    lap_3d = np.stack(laplacian_3d(self.x_tiles), axis=-1)
    lap = laplacian(self.x)
    self.assertAllClose(lap_3d[INNER], lap[INNER], atol=1e-15)

  @parameterized.parameters(((1, 1, 1),), ((1.1, 5.2, 9.33),))
  def test_compare_laplacian_3d_tf_(self, grid_lengths):
    """Compares the tf multigrid_3d_utils and multigrid_utils Laplacian fns."""
    self.set_params(grid_lengths)
    laplacian_3d, _ = multigrid_3d_utils.laplacian_and_inv_diagonal_fns(
        self.shape, grid_lengths)
    laplacian, _ = multigrid_utils.laplacian_and_inv_diagonal_fns(
        self.shape, grid_lengths)
    x = tf.convert_to_tensor(self.x)
    x_tiles = [tf.convert_to_tensor(xy) for xy in self.x_tiles]
    lap = self.evaluate(laplacian(x))
    self.compare_np_tf_and_tpu(
        lap[INNER],
        lambda: laplacian_3d(x_tiles),
        atol=1e-15,
        post_fn=lambda t: t[INNER])

  PARAMS = itertools.product(((1, 1, 1), (1.1, 5.2, 9.33)), (1, 2 / 3))

  @parameterized.parameters(PARAMS)
  def test_compare_jacobi_np_tf_(self, grid_lengths, weight):
    """Compares np vs. tf results for mutigrid_3d_utils.poisson_jacobi."""
    self.set_params(grid_lengths)
    n = 100

    tiles_np = multigrid_3d_utils.poisson_jacobi(self.x_tiles, self.b_tiles,
                                                 self.params, n, weight)
    res_np = np.stack(
        multigrid_3d_utils.poisson_residual(tiles_np, self.b_tiles,
                                            self.params),
        axis=-1)
    tiles_np = np.stack(tiles_np, axis=-1)

    tiles_tf = [tf.convert_to_tensor(xy) for xy in self.x_tiles]
    b_tiles = [tf.convert_to_tensor(xy) for xy in self.b_tiles]
    tiles_tf_fn = functools.partial(multigrid_3d_utils.poisson_jacobi, tiles_tf,
                                    b_tiles, self.params, n, weight)
    res_tf_fn = functools.partial(multigrid_3d_utils.poisson_residual,
                                  tiles_tf_fn(), b_tiles, self.params)
    self.compare_tiles_and_residuals(tiles_np, tiles_tf_fn, res_np, res_tf_fn)

  PARAMS = itertools.product(((1, 1, 1), (1.1, 5.2, 9.33)), (1, 2 / 3))

  @parameterized.parameters(PARAMS)
  def test_compare_jacobi_3d_np_(self, grid_lengths, weight):
    """Compares np multigrid_utils Jacobi with mutigrid_3d_utils."""
    self.set_params(grid_lengths)
    n = 100

    x1_tiles = multigrid_3d_utils.poisson_jacobi(self.x_tiles, self.b_tiles,
                                                 self.params, n, weight)
    x2 = multigrid_utils.poisson_jacobi(self.x, self.b, self.params, n, weight)
    res1 = np.stack(
        multigrid_3d_utils.poisson_residual(x1_tiles, self.b_tiles,
                                            self.params),
        axis=-1)
    res2 = multigrid_utils.poisson_residual(x2, self.b, self.params)
    with self.subTest('x values'):
      self.assertAllClose(np.stack(x1_tiles, axis=-1), x2, atol=1e-15)
    with self.subTest('residuals'):
      self.assertAllClose(res1[INNER], res2[INNER], atol=1.7e-6)

  PARAMS = itertools.product(((1, 1, 1), (1.1, 5.2, 9.33)), (1, 2 / 3))

  @parameterized.parameters(PARAMS)
  def test_compare_jacobi_3d_tf_(self, grid_lengths, weight):
    """Compares tf multigrid_utils Jacobi with mutigrid_3d_utils."""
    self.set_params(grid_lengths)
    n = 100
    x_tiles = [tf.convert_to_tensor(xy) for xy in self.x_tiles]
    b_tiles = [tf.convert_to_tensor(xy) for xy in self.b_tiles]
    x = tf.convert_to_tensor(self.x)
    b = tf.convert_to_tensor(self.b)

    x1_tiles_f = functools.partial(multigrid_3d_utils.poisson_jacobi, x_tiles,
                                   b_tiles, self.params, n, weight)
    x2 = multigrid_utils.poisson_jacobi(x, b, self.params, n, weight)
    res1_f = functools.partial(multigrid_3d_utils.poisson_residual,
                               x1_tiles_f(), b_tiles, self.params)
    res2 = self.evaluate(multigrid_utils.poisson_residual(x2, b, self.params))
    x2 = self.evaluate(x2)
    self.compare_tiles_and_residuals(x2, x1_tiles_f, res2, res1_f)

  @parameterized.parameters(((1, 1, 1),), ((1.1, 5.2, 9.33),))
  def test_compare_poisson_residual_3d_np_(self, grid_lengths):
    self.set_params(grid_lengths)
    res1 = np.stack(
        multigrid_3d_utils.poisson_residual(self.x_tiles, self.b_tiles,
                                            self.params),
        axis=-1)
    res2 = multigrid_utils.poisson_residual(self.x, self.b, self.params)

    self.assertAllClose(res1[INNER], res2[INNER], atol=1.5e-5)

  @parameterized.parameters(((1, 1, 1),), ((1.1, 5.2, 9.33),))
  def test_compare_kronecker_products_3d_(self, grid_lengths):
    self.set_params(grid_lengths)
    coarsest_shape = (3, 3, 3)
    ps, rs = multigrid_utils.prolong_restrict_matrices(self.shape,
                                                       coarsest_shape)
    x_r = multigrid_utils.kronecker_products(rs[0], self.x)
    x_tiles_r = multigrid_3d_utils.kronecker_products(rs[0], self.x_tiles)
    actual_r = np.stack(x_tiles_r, axis=-1)

    with self.subTest('restrict'):
      self.assertAllClose(actual_r, x_r, atol=1e-15)

    x_p = multigrid_utils.kronecker_products(ps[0], x_r)
    x_tiles_p = multigrid_3d_utils.kronecker_products(ps[0], x_tiles_r)
    actual_p = np.stack(x_tiles_p, axis=-1)

    self.assertEqual(self.shape, actual_p.shape)

    with self.subTest('prolong'):
      self.assertAllClose(actual_p, x_p, atol=1e-15)


class Multigrid3dUtilsApplyBoundaryConditionsTest(tf.test.TestCase):
  """Tests apply boundary conditions function."""

  def setUp(self):
    super().setUp()
    self.boundary_conditions = [((BCType.DIRICHLET, 1.), (BCType.NEUMANN, 2.)),
                                ((BCType.NO_TOUCH, 3.), (BCType.DIRICHLET, 4.)),
                                ((BCType.NEUMANN, 5.), (BCType.NO_TOUCH, 6.))]
    x = np.array([[[1., 2, 3], [4, 5, 6], [7, 8, 9]],
                  [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
                  [[21, 22, 23], [24, 25, 26], [27, 28, 29]]])
    self.x = [x[:, :, i] for i in range(3)]

  def test_get_apply_one_core_boundary_conditions_fn(self):
    apply_fn = multigrid_3d_utils.get_apply_one_core_boundary_conditions_fn(
        self.boundary_conditions)
    actual = np.stack(apply_fn(self.x), axis=-1)

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
    apply_fn = multigrid_3d_utils.get_apply_one_core_boundary_conditions_fn(
        self.boundary_conditions, homogeneous=True)
    actual = np.stack(apply_fn(self.x), axis=-1)

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
    apply_fn = multigrid_3d_utils.get_apply_one_core_boundary_conditions_fn(
        boundary_conditions)
    actual = np.stack(apply_fn(self.x), axis=-1)

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


class Multigrid3dUtilsMultiCoreTest(tf.test.TestCase, parameterized.TestCase):

  def init_fn(self, params: grid_parametrization.GridParametrization,
              coordinates):
    return {
        'x':
            initializer.subgrid_of_3d_grid_from_params(self._x, params,
                                                       coordinates),
        'b':
            initializer.subgrid_of_3d_grid_from_params(self._b, params,
                                                       coordinates),
    }

  def init_states(self, params: grid_parametrization.GridParametrization):
    init_fn = functools.partial(self.init_fn, params)
    computation_shape = params.cx, params.cy, params.cz
    coordinates = util.grid_coordinates(computation_shape)
    return [init_fn(coordinates[i]) for i in range(np.prod(computation_shape))]

  def step_fn(self, state, replicas, replica_id):
    x = tf.unstack(state['x'], axis=-1)
    b = tf.unstack(state['b'], axis=-1)
    poisson_jacobi = self._jacobi_step_fn(replica_id, replicas)
    return poisson_jacobi(x, b)

  def maybe_combine_xs(self, xs, computation_shape):
    x0 = np.stack(xs[0], axis=-1)
    if np.prod(computation_shape) == 1:
      return x0
    else:
      x1 = np.stack(xs[1], axis=-1)
      if computation_shape == (2, 1, 1):
        return np.concatenate((x0[:-1, :, :], x1[1:, :, :]), axis=0)
      elif computation_shape == (1, 2, 1):
        return np.concatenate((x0[:, :-1, :], x1[:, 1:, :]), axis=1)
      else:
        return np.concatenate((x0[:, :, :-1], x1[:, :, 1:]), axis=2)

  def get_poisson_jacobi_result(
      self, params: grid_parametrization.GridParametrization) -> np.ndarray:
    computation_shape = params.cx, params.cy, params.cz
    tpu = TpuRunner(computation_shape=computation_shape)
    xs = tpu.run_with_replica_args(self.step_fn, self.init_states(params))
    return self.maybe_combine_xs(xs, computation_shape)

  @parameterized.parameters((2, 1, 1), (1, 2, 1), (1, 1, 2))
  def test_poisson_jacobi_compare_one_core_to_two_(self, cx, cy, cz):
    grid_lengths = (1, 10, 100)
    subgrid_shape = (7, 14, 26)
    full_grid_shape = [2 * n - 2 for n in subgrid_shape]
    boundary_conditions = [[(halo_exchange.BCType.NO_TOUCH, 0.)] * 2] * 3

    # Full grids.
    self._x = np.random.rand(*full_grid_shape).astype(np.float32)
    self._b = np.random.rand(*full_grid_shape).astype(np.float32)

    n_jacobi = 30

    # One core params.
    computation_shape_1_core = (1, 1, 1)
    params_1_core = (
        grid_parametrization.GridParametrization
        .create_from_grid_lengths_and_etc(
            grid_lengths,
            computation_shape_1_core,
            full_grid_shape,
            halo_width=1))
    self._jacobi_step_fn = multigrid_3d_utils.poisson_jacobi_step_fn(
        params_1_core, n_jacobi)

    # One core result using explicitly one-core numpy code.
    weight = 1
    one_core_np = multigrid_utils.poisson_jacobi(
        np.copy(self._x), self._b, params_1_core, n_jacobi, weight)

    # One core result using tf and multicore code.
    one_core_tf = self.get_poisson_jacobi_result(params_1_core)

    self.assertAllClose(one_core_np, one_core_tf)

    # Two core params.
    subgrid_shape_2_cores = full_grid_shape[:]
    if cx == 2:
      subgrid_shape_2_cores[0] = subgrid_shape[0]
    elif cy == 2:
      subgrid_shape_2_cores[1] = subgrid_shape[1]
    else:
      subgrid_shape_2_cores[2] = subgrid_shape[2]
    computation_shape = (cx, cy, cz)
    params_2_cores = (
        grid_parametrization.GridParametrization
        .create_from_grid_lengths_and_etc(
            grid_lengths,
            computation_shape,
            subgrid_shape_2_cores,
            halo_width=1))
    self._jacobi_step_fn = multigrid_3d_utils.poisson_jacobi_step_fn(
        params_2_cores, n_jacobi, boundary_conditions=boundary_conditions)

    two_core_tf = self.get_poisson_jacobi_result(params_2_cores)

    self.assertAllEqual(one_core_tf, two_core_tf)

  def test_poisson_jacobi_neumann_bcs_convergence(self):
    """Tests convergence for Jacobi with a Neumann boundary condition."""
    grid_lengths = (4, 4, 4)
    subgrid_shape = (3, 3, 3)
    full_grid_shape = [2 * n - 2 for n in subgrid_shape]

    self._x = np.zeros(full_grid_shape).astype(np.float32)
    self._b = np.zeros(full_grid_shape).astype(np.float32)

    n_jacobi = 10  # Number of Jacobi iterations.

    special_number = 0.2229885
    # Boundary conditions are defined so that, given the Neumann boundary
    # condition, in the solution element [0, 1, 1] is 0 and element [1, 1, 1] is
    # the special number.
    boundary_conditions = (
        # pyformat: disable
        ((BCType.NEUMANN, special_number), (BCType.DIRICHLET, 1.0)),
        ((BCType.DIRICHLET, 0.0), (BCType.DIRICHLET, 1.0)),
        ((BCType.DIRICHLET, 0.0), (BCType.DIRICHLET, 1.0)))
    # pyformat: enable

    computation_shape_1_core = (1, 1, 1)

    params = (
        grid_parametrization.GridParametrization
        .create_from_grid_lengths_and_etc_with_defaults(
            grid_lengths, computation_shape_1_core))

    self._jacobi_step_fn = multigrid_3d_utils.poisson_jacobi_step_fn(
        params, n_jacobi, boundary_conditions=boundary_conditions)

    # Convergence is very fast on a small grid.
    # TODO(anudhyan, emilp): The tf1 version of the test converged to 1e-7.
    # Investigate the precision loss with tf2 and restore the low end-tolerance.
    # See the original expected values in the tf1 version here:
    # https://source.corp.google.com/piper///depot/google3/research/simulation/tensorflow/fluid/framework/tf1/multigrid_3d_utils_test.py;l=468;cl=410387335 # pylint: disable=line-too-long
    tols = (1e-2, 5e-4, 2e-4, 1.1e-4)

    computation_shape = params.cx, params.cy, params.cz
    tpu = TpuRunner(computation_shape=computation_shape)
    state = self.init_states(params)

    for i in range(4):
      xs = tpu.run_with_replica_args(self.step_fn, state)
      for state_i, x_i in zip(state, xs):
        state_i['x'] = x_i
      actual = self.maybe_combine_xs(xs, computation_shape)
      self.assertNear(0, actual[0, 1, 1], err=tols[i])
      self.assertNear(special_number, actual[1, 1, 1], err=tols[i])

  @parameterized.parameters((2, 1, 1), (1, 2, 1), (1, 1, 2))
  def test_poisson_jacobi_compare_1_core_to_2_neumann_bcs_(self, cx, cy, cz):
    grid_lengths = (4, 44, 444)
    subgrid_shape = (3, 10, 25)
    full_grid_shape = [2 * n - 2 for n in subgrid_shape]

    self._x = np.random.rand(*full_grid_shape).astype(np.float32)
    self._b = np.random.rand(*full_grid_shape).astype(np.float32)

    n_jacobi = 40  # Number of Jacobi iterations.

    boundary_conditions = (((BCType.NEUMANN, -0.4), (BCType.DIRICHLET, 0.8)),
                           ((BCType.DIRICHLET, 0.0), (BCType.NEUMANN, -1.3)),
                           ((BCType.DIRICHLET, 0.6), (BCType.DIRICHLET, 1.1)))

    # One core params.
    computation_shape_1_core = (1, 1, 1)
    params = (
        grid_parametrization.GridParametrization
        .create_from_grid_lengths_and_etc(
            grid_lengths,
            computation_shape_1_core,
            full_grid_shape,
            halo_width=1))

    # One core tensorflow result using multicore code.
    self._jacobi_step_fn = multigrid_3d_utils.poisson_jacobi_step_fn(
        params, n_jacobi, boundary_conditions=boundary_conditions)

    one_core_result = self.get_poisson_jacobi_result(params)

    # Two cores.
    subgrid_shape_2_cores = full_grid_shape[:]
    if cx == 2:
      subgrid_shape_2_cores[0] = subgrid_shape[0]
    elif cy == 2:
      subgrid_shape_2_cores[1] = subgrid_shape[1]
    else:
      subgrid_shape_2_cores[2] = subgrid_shape[2]
    computation_shape = (cx, cy, cz)
    params_2_cores = (
        grid_parametrization.GridParametrization
        .create_from_grid_lengths_and_etc(
            grid_lengths,
            computation_shape,
            subgrid_shape_2_cores,
            halo_width=1))

    self._jacobi_step_fn = multigrid_3d_utils.poisson_jacobi_step_fn(
        params_2_cores, n_jacobi, boundary_conditions=boundary_conditions)

    two_core_result = self.get_poisson_jacobi_result(params_2_cores)

    self.assertAllEqual(one_core_result, two_core_result)


if __name__ == '__main__':
  tf.test.main()
