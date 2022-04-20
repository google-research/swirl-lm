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
"""Tests for multigrid_3d."""
import numpy as np
from swirl_lm.linalg import multigrid_3d
from swirl_lm.linalg import multigrid_test_common
import tensorflow as tf
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized


class MultigridTest(tf.test.TestCase, parameterized.TestCase):
  """Tests simple cases of multigrid_3d.poisson_mg_cycle_fn_for_one_core."""

  def setUp(self):
    super().setUp()
    shape = (10, 10, 10)
    xy_plane = np.stack((np.linspace(1, 10, 10),) * 10)
    self.x = [np.copy(xy_plane) for _ in range(10)]
    self.b = [np.zeros_like(x_) for x_ in self.x]
    self.expected = np.stack(self.x, axis=-1)
    self.mg_cycle_fn = multigrid_3d.poisson_mg_cycle_fn_for_one_core(shape)

  def test_np_mg_cycle_no_source_start_at_soln(self):
    """mg_cycle test. There is no change when starting from the solution."""
    actual = np.stack(self.mg_cycle_fn(self.x, self.b), axis=-1)
    self.assertAllClose(self.expected, actual)

  def test_tf_mg_cycle_no_source_start_at_soln(self):
    """mg_cycle test. There is no change when starting from the solution."""
    x = [tf.convert_to_tensor(x_) for x_ in self.x]
    b = [tf.zeros_like(x_) for x_ in self.x]
    actual = np.stack(self.evaluate(self.mg_cycle_fn(x, b)), axis=-1)
    self.assertAllClose(self.expected, actual)

  def test_tf_tpu_mg_cycle_no_source_start_at_soln(self):
    """mg_cycle test. There is no change when starting from the solution."""
    tpu = TpuRunner(computation_shape=(1,))
    actual = np.stack(tpu.run(self.mg_cycle_fn, [self.x], [self.b])[0], axis=-1)
    self.assertAllClose(self.expected, actual)


class MultigridConvergenceTest(multigrid_test_common.ConvergenceTest,
                               parameterized.TestCase):
  """Tests convergence of multigrid_3d.poisson_mg_cycle_fn_for_one_core."""

  @parameterized.parameters(
      multigrid_test_common.ConvergenceTest.MG_DIFFS_NORMS_5X5X5)
  def test_mg_cycle_no_source_5x5x5(self, tr_axis, params):
    """mg_cycle test for 5x5x5. Illustrates convergence."""
    n_coarse, n_smooth, expected_diffs, expected_norms = params
    starting, expected = self.starting_and_expected_5x5x5(tr_axis)
    name = (f'shape={expected.shape}, n_coarse={n_coarse}, '
            f'n_smooth={n_smooth}')
    solver = multigrid_3d.poisson_mg_cycle_fn_for_one_core(
        expected.shape, n_coarse, n_smooth)
    self._run_convergence_test(solver, name, starting, expected, expected_diffs,
                               expected_norms, using_tiles=True)

  @parameterized.parameters(
      multigrid_test_common.ConvergenceTest.MG_DIFFS_NORMS_5X10X13)
  def test_mg_cycle_no_source_5x10x13(self, tr_axis, params):
    """mg_cycle test for 5x10x13. Illustrates convergence."""
    n_coarse, n_smooth, expected_diffs, expected_norms = params
    starting, expected = self.starting_and_expected_5x10x13(tr_axis)
    name = (f'shape={expected.shape}, n_coarse={n_coarse}, '
            f'n_smooth={n_smooth}')
    solver = multigrid_3d.poisson_mg_cycle_fn_for_one_core(
        expected.shape, n_coarse, n_smooth)
    self._run_convergence_test(solver, name, starting, expected, expected_diffs,
                               expected_norms, using_tiles=True)


if __name__ == '__main__':
  tf.test.main()
