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
"""Multigrid one core comparison tests."""

import functools
import itertools

import numpy as np
from swirl_lm.communication import halo_exchange_utils
from swirl_lm.linalg import multigrid
from swirl_lm.linalg import multigrid_3d
from swirl_lm.linalg import multigrid_utils
from swirl_lm.utility import grid_parametrization
import tensorflow as tf
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized

BCType = halo_exchange_utils.BCType


class MultigridOneCoreCompareTest(tf.test.TestCase, parameterized.TestCase):

  def init(self, full_grid_shape, boundary_conditions=None):
    self.computation_shape = (1, 1, 1)
    self.full_grid_shape = full_grid_shape
    self.boundary_conditions = boundary_conditions

    self.x = np.random.rand(*full_grid_shape).astype(np.float32)
    self.b = np.random.rand(*full_grid_shape).astype(np.float32)

    self.x_tiles = [self.x[:, :, i] for i in range(full_grid_shape[2])]
    self.b_tiles = [self.b[:, :, i] for i in range(full_grid_shape[2])]

    grid_lengths = (1.2, 13.1, 144.1)

    self.params = (
        grid_parametrization.GridParametrization
        .create_from_grid_lengths_and_etc(
            grid_lengths, self.computation_shape, full_grid_shape, halo_width=1)
    )

    self.prs = multigrid_utils.prolong_restrict_matrices_from_params(
        self.params)

  def step_fn(self, state, n_coarse, replicas, replica_id):
    coordinates = state['coordinates']
    x0 = state['x0']
    xb = state['xb']
    b_minus_a_xb = state['b_minus_a_xb']

    nz = multigrid_utils.get_shape(x0)[2]

    x0 = [x0[:, :, i] for i in range(nz)]
    xb = [xb[:, :, i] for i in range(nz)]
    b_minus_a_xb = [b_minus_a_xb[:, :, i] for i in range(nz)]

    self.prs = multigrid_utils.convert_ps_rs_dict_to_tuple(state['ps'],
                                                           state['rs'])

    mg_cycle_step_fn = multigrid_3d.poisson_mg_cycle_step_fn(
        self.params,
        n_coarse=n_coarse,
        boundary_conditions=self.boundary_conditions)
    mg_cycle = mg_cycle_step_fn(self.prs, replica_id, replicas, coordinates)
    x0 = mg_cycle(x0, b_minus_a_xb)
    return [x0_ + xb_ for x0_, xb_ in zip(x0, xb)]

  def np_mg_cycle(self, mg_lib, n_coarse=1):
    return mg_lib.poisson_mg_cycle_fn_for_one_core(
        self.full_grid_shape,
        n_coarse,
        params=self.params,
        prs=self.prs,
        boundary_conditions=self.boundary_conditions)

  def run_multigrid_np(self, n_coarse=1):
    mg_cycle = self.np_mg_cycle(multigrid, n_coarse)
    return mg_cycle(np.copy(self.x), self.b)

  def run_multigrid_3d_np(self, n_coarse=1):
    mg_cycle = self.np_mg_cycle(multigrid_3d, n_coarse)
    x_tiles = [np.copy(x_tile) for x_tile in self.x_tiles]
    return np.stack(mg_cycle(x_tiles, self.b_tiles), axis=-1)

  def run_multigrid_3d_tf(self, n_coarse=1):
    init_fn = multigrid_utils.get_full_grids_init_fn(
        tf.convert_to_tensor(self.x), self.b, self.params,
        boundary_conditions=self.boundary_conditions)
    state = init_fn(replica_id=0, coordinates=(0, 0, 0))
    replicas = np.array([[[0]]])
    x = self.evaluate(self.step_fn(state, n_coarse, replicas, replica_id=0))
    return np.stack(x, axis=-1)

  def run_multigrid_3d_tpu(self, n_coarse=1):
    runner = TpuRunner(computation_shape=self.computation_shape)
    init_fn = multigrid_utils.get_full_grids_init_fn(
        self.x, self.b, self.params,
        boundary_conditions=self.boundary_conditions)
    state = init_fn(replica_id=0, coordinates=(0, 0, 0))
    step_fn_state_arg = functools.partial(self.step_fn, n_coarse=n_coarse)
    xs = runner.run_with_replica_args(step_fn_state_arg, [state])
    return np.stack(xs[0], axis=-1)

  N_COARSE = (1, 2)
  FULL_GRID_SHAPE = ((4, 4, 4), (6, 6, 6), (34, 32, 30))

  @parameterized.parameters(*itertools.product(N_COARSE, FULL_GRID_SHAPE))
  def test_1_core_multigrid_compare_3d_nd_(self, n_coarse, full_grid_shape):
    self.init(full_grid_shape)

    one_core_np_result = self.run_multigrid_np(n_coarse)
    one_core_np_3d_result = self.run_multigrid_3d_np(n_coarse)

    self.assertAllClose(one_core_np_3d_result, one_core_np_result)

  @parameterized.parameters(*itertools.product(N_COARSE, FULL_GRID_SHAPE))
  def test_1_core_multigrid_compare_3d_np_tf_tpu_(self, n_coarse,
                                                  full_grid_shape):
    boundary_conditions = [[(BCType.DIRICHLET, 0.)] * 2] * 3
    self.init(full_grid_shape, boundary_conditions)

    one_core_np_result = self.run_multigrid_3d_np(n_coarse)
    one_core_tf_result = self.run_multigrid_3d_tf(n_coarse)
    one_core_tpu_result = self.run_multigrid_3d_tpu(n_coarse)

    with self.subTest('np_tf'):
      self.assertAllClose(one_core_np_result, one_core_tf_result, atol=2e-6)
    with self.subTest('np_tpu'):
      self.assertAllClose(one_core_np_result, one_core_tpu_result, atol=2e-6)


if __name__ == '__main__':
  tf.test.main()
