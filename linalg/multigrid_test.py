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
"""Tests for multigrid."""

import numpy as np
from swirl_lm.communication import halo_exchange
from swirl_lm.linalg import multigrid
from swirl_lm.linalg import multigrid_test_common
import tensorflow as tf
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized

BCType = halo_exchange.BCType


class MultigridTest(tf.test.TestCase, parameterized.TestCase):
  """Tests multigrid functions in 1, 2, and 3 dimensions."""

  def test_mg_cycle_1d_tf_no_source_start_at_soln(self):
    """mg_cycle 1D tf: No change when starting at the solution."""
    x = np.linspace(1, 10, 10).astype(np.float32)
    x_tf = tf.convert_to_tensor(x)
    expected = np.copy(x)
    b = tf.zeros_like(x_tf)
    mg_cycle_fn = multigrid.poisson_mg_cycle_fn_for_one_core(b.shape)
    tpu = TpuRunner(computation_shape=(1,))
    actual = tpu.run(lambda: mg_cycle_fn(x_tf, b))[0]
    self.assertAllClose(expected, actual)

  def test_mg_cycle_1d_np_no_source_start_at_soln(self):
    """mg_cycle 1D np: No change when starting at the solution."""
    x = np.linspace(1, 10, 10).astype(np.float32)
    expected = np.copy(x)
    b = np.zeros_like(x)
    mg_cycle_fn = multigrid.poisson_mg_cycle_fn_for_one_core(b.shape)
    actual = mg_cycle_fn(x, b)
    self.assertAllClose(actual, expected)

  def test_mg_cycle_1d_tf_no_source_start_away_from_soln(self):
    """mg_cycle 1D tf: Convergence to exact solution."""
    actual = np.array([1, 6, 9, 7, 2, 8, 5, 3, 4, 10]).astype(np.float32)
    expected = np.linspace(1, 10, 10).astype(np.float32)
    rtols = (9.078193E-02, 5.509615E-03, 3.480116E-04, 2.169609E-05)
    b = tf.zeros_like(tf.convert_to_tensor(actual))
    mg_cycle_fn = multigrid.poisson_mg_cycle_fn_for_one_core(
        b.shape, n_coarse=1, n_smooth=3)

    tpu = TpuRunner(computation_shape=(1,))
    for rtol in rtols:
      actual = tpu.run(lambda: mg_cycle_fn(tf.convert_to_tensor(actual), b))[0]
      max_relative_diff = np.max(np.abs(actual - expected) / expected)
      with self.subTest(name='rtol={}'.format(rtol)):
        self.assertAllClose(max_relative_diff, rtol, atol=rtol / 1000)

  def test_mg_cycle_1d_np_no_source_start_away_from_soln(self):
    """mg_cycle 1D np: Convergence to exact solution."""
    actual = np.array([1, 6, 9, 7, 2, 8, 5, 3, 4, 10]).astype(np.float32)
    expected = np.linspace(1, 10, 10).astype(np.float32)
    rtols = (0.09078193, 5.5093765e-03, 3.480116e-04, 2.16960e-05)
    b = np.zeros_like(actual)
    mg_cycle_fn = multigrid.poisson_mg_cycle_fn_for_one_core(
        b.shape, n_coarse=1, n_smooth=3)
    for rtol in rtols:
      actual = mg_cycle_fn(actual, b)
      max_relative_diff = np.max(np.abs(actual - expected) / expected)
      with self.subTest(name='rtol={}'.format(rtol)):
        self.assertAllClose(max_relative_diff, rtol, atol=rtol / 800)

  def test_mg_cycle_2d_tf_no_source_start_at_soln(self):
    """mg_cycle 2D tf: No change when starting at the solution."""
    x = np.stack((np.linspace(1, 10, 10),) * 10)
    x_tf = tf.convert_to_tensor(x)
    b = tf.zeros_like(x_tf)
    expected = np.copy(x)
    mg_cycle_fn = multigrid.poisson_mg_cycle_fn_for_one_core(b.shape)
    tpu = TpuRunner(computation_shape=(1,))
    actual = tpu.run(lambda: mg_cycle_fn(x_tf, b))[0]
    self.assertAllClose(expected, actual)

  def test_mg_cycle_2d_np_no_source_start_at_soln(self):
    """mg_cycle 2D np: No change when starting at the solution."""
    x = np.stack((np.linspace(1, 10, 10),) * 10)
    expected = np.copy(x)
    b = np.zeros_like(x)
    mg_cycle_fn = multigrid.poisson_mg_cycle_fn_for_one_core(b.shape)
    actual = mg_cycle_fn(x, b)
    self.assertAllClose(actual, expected)

  # See hard-coded exact max differences from tf1 version here:
  # https://source.corp.google.com/piper///depot/google3/research/simulation/tensorflow/fluid/framework/multigrid_test.py;l=94;cl=410584531 # pylint: disable=line-too-long
  PARAMS = (
      # pyformat: disable
      (1, 1, (1, 3e-01, 1e-01, 4e-02, 2e-02)),
      (1, 3, (2e-01, 2e-02, 3e-03, 5e-04, 9e-05)),
      (1, 6, (5e-02, 4e-03, 4e-04, 4e-05, 5e-06)),
      (3, 1, (2e-01, 2e-02, 3e-03, 5e-04, 2e-04)),
      (3, 3, (8e-03, 6e-05, 8e-07, 7e-07, 8e-07)),
      (6, 3, (2e-04, 5e-07, 2e-06, 2e-06, 5e-07)),
      (6, 6, (2e-06, 5e-07, 2e-06, 1.2e-06, 9.6e-07)))
  # pyformat: enable

  @parameterized.parameters(*PARAMS)
  def test_mg_cycle_2d_no_source_start_away_from_soln_(self, n_coarse, n_smooth,
                                                       rtols):
    """mg_cycle 2D tf: Convergence to exact solution."""
    actual = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [2, 6, 9, 7, 2, 8, 5, 3, 4, 2],
                       [3, 9, 7, 2, 8, 5, 3, 4, 6, 3],
                       [4, 7, 2, 8, 5, 3, 4, 9, 6, 4],
                       [5, 2, 8, 5, 3, 4, 9, 6, 7, 5],
                       [6, 8, 5, 3, 4, 2, 6, 9, 7, 6],
                       [7, 5, 3, 4, 2, 6, 9, 7, 8, 7],
                       [8, 5, 3, 4, 6, 9, 7, 2, 8, 8],
                       [9, 4, 3, 6, 9, 7, 2, 8, 5, 9],
                       [10, 4, 6, 9, 7, 2, 8, 5, 3, 10],
                       [11, 4, 9, 7, 2, 8, 5, 3, 4, 11],
                       [12, 4, 9, 7, 2, 4, 5, 3, 4, 12],
                       [13, 13, 13, 13, 13, 13, 13, 13, 13,
                        13]]).astype(np.float32)
    expected = []
    for i in range(1, actual.shape[0] + 1):
      expected.append(np.ones(actual.shape[1], np.float32) * i)
    expected = np.stack(expected)
    b = tf.zeros_like(actual)
    mg_cycle_fn = multigrid.poisson_mg_cycle_fn_for_one_core(
        b.shape, n_coarse, n_smooth)
    tpu = TpuRunner(computation_shape=(1,))
    for rtol in rtols:
      actual = tpu.run(lambda: mg_cycle_fn(tf.convert_to_tensor(actual), b))[0]
      max_relative_diff = np.max(np.abs(actual - expected) / expected)
      with self.subTest(name='rtol={}'.format(rtol)):
        self.assertLess(max_relative_diff, rtol)


class MultigridConvergenceTest(multigrid_test_common.ConvergenceTest,
                               parameterized.TestCase):

  @parameterized.parameters(
      multigrid_test_common.ConvergenceTest.MG_DIFFS_NORMS_5X5X5)
  def test_mg_cycle_no_source_5x5x5_(self, tr_axis, params):
    """mg_cycle test for 5x5x5. Illustrates convergence."""
    n_coarse, n_smooth, expected_diffs, expected_norms = params
    starting, expected = self.starting_and_expected_5x5x5(tr_axis)
    name = (f'shape={expected.shape}, n_coarse={n_coarse}, '
            f'n_smooth={n_smooth}')
    solver = multigrid.poisson_mg_cycle_fn_for_one_core(expected.shape,
                                                        n_coarse, n_smooth)
    self._run_convergence_test(solver, name, starting, expected, expected_diffs,
                               expected_norms)

  @parameterized.parameters(
      multigrid_test_common.ConvergenceTest.MG_DIFFS_NORMS_5X10X13)
  def test_mg_cycle_no_source_5x10x13_(self, tr_axis, params):
    """mg_cycle test for 5x10x13. Illustrates convergence."""
    n_coarse, n_smooth, expected_diffs, expected_norms = params
    starting, expected = self.starting_and_expected_5x10x13(tr_axis)
    name = (f'shape={expected.shape}, n_coarse={n_coarse}, '
            f'n_smooth={n_smooth}')
    solver = multigrid.poisson_mg_cycle_fn_for_one_core(expected.shape,
                                                        n_coarse, n_smooth)
    self._run_convergence_test(solver, name, starting, expected, expected_diffs,
                               expected_norms)

  @parameterized.parameters(multigrid_test_common.ConvergenceTest
                            .MG_DIFFS_NORMS_5X10X13_COARSEST_4X4X4)
  def test_mg_cycle_no_source_5x10x13_coarsest_4x4x4_(self, tr_axis, params):
    """mg_cycle test for 5x10x13. Illustrates convergence w/ coarsest 4x4x4."""
    n_coarse, n_smooth, expected_diffs, expected_norms = params
    coarsest_full_grid_shape = (4, 4, 4)
    starting, expected = self.starting_and_expected_5x10x13(tr_axis)
    name = (f'shape={expected.shape}, n_coarse={n_coarse}, '
            f'n_smooth={n_smooth}')
    solver = multigrid.poisson_mg_cycle_fn_for_one_core(
        expected.shape,
        n_coarse,
        n_smooth,
        coarsest_full_grid_shape=coarsest_full_grid_shape)
    self._run_convergence_test(solver, name, starting, expected, expected_diffs,
                               expected_norms)


if __name__ == '__main__':
  tf.test.main()
