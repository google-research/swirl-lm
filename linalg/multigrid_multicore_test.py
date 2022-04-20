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
"""Tests for multicore multigrid."""

import functools
import itertools

import numpy as np
from swirl_lm.communication import halo_exchange
from swirl_lm.linalg import multigrid
from swirl_lm.linalg import multigrid_utils
from swirl_lm.utility import grid_parametrization
import tensorflow as tf
from google3.research.simulation.tensorflow.fluid.framework import util
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized

BCType = halo_exchange.BCType


class MultigridMulticoreTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.full_grid_shape = (10, 14, 18)
    self.weight = 0.8

    bcs_dirichlet = [((BCType.DIRICHLET, 1.0), (BCType.DIRICHLET, -1.0)),
                     ((BCType.DIRICHLET, 1.0), (BCType.DIRICHLET, -1.0)),
                     ((BCType.DIRICHLET, 1.0), (BCType.DIRICHLET, -1.0))]

    bcs_neumann = [((BCType.NEUMANN, -0.07), (BCType.NEUMANN, -0.07)),
                   ((BCType.NEUMANN, -0.07), (BCType.NEUMANN, -0.07)),
                   ((BCType.NEUMANN, -0.07), (BCType.NEUMANN, -0.07))]

    bcs_mixed = [((BCType.NEUMANN, -0.07), (BCType.DIRICHLET, -1.0)),
                 ((BCType.NEUMANN, -0.07), (BCType.DIRICHLET, -1.0)),
                 ((BCType.NEUMANN, -0.07), (BCType.DIRICHLET, -1.0))]

    self.bcs_map = {'dirichlet': bcs_dirichlet,
                    'mixed': bcs_mixed,
                    'neumann': bcs_neumann}

  def set_x_b_initial_values(self):
    x = np.ones(self.full_grid_shape, self.dtype)
    meshes = np.meshgrid(
        *[np.linspace(0, 1, n) for n in self.full_grid_shape], indexing='ij')
    for i in range(len(self.full_grid_shape)):
      x *= np.cos(np.pi * meshes[i], dtype=self.dtype)

    apply_bcs = multigrid_utils.get_apply_one_core_boundary_conditions_fn(
        self.boundary_conditions)
    x = apply_bcs(x)

    a = multigrid_utils.laplacian_matrix(
        self.full_grid_shape,
        boundary_conditions=self.boundary_conditions,
        dtype=self.dtype)

    self.b = multigrid_utils.matmul(a, x).astype(self.dtype)
    self.x = np.zeros_like(self.b)

  def init_states(self):
    coordinates = util.grid_coordinates(self.computation_shape)
    init_fn = multigrid_utils.get_full_grids_init_fn(
        self.x, self.b, self.many_cores_params, self.coarsest_subgrid_shape,
        self.boundary_conditions)
    return [init_fn(i, coordinates[i])
            for i in range(np.prod(self.computation_shape))]

  def init(self, many_cores_computation_shape, boundary_conditions,
           dtype=np.float32, coarsest_subgrid_shape=(3, 3, 3)):
    self.computation_shape = many_cores_computation_shape
    self.boundary_conditions = boundary_conditions
    self.dtype = dtype
    self.coarsest_subgrid_shape = coarsest_subgrid_shape
    self.set_x_b_initial_values()

    subgrid_shape = multigrid_utils.get_subgrid_shape(
        self.full_grid_shape, many_cores_computation_shape)
    self.coarsest_full_grid_shape = multigrid_utils.get_full_grid_shape(
        coarsest_subgrid_shape, many_cores_computation_shape)

    grid_lengths = (1.0, 1.5, 2.0)

    self.many_cores_params = (
        grid_parametrization.GridParametrization
        .create_from_grid_lengths_and_etc(
            grid_lengths,
            many_cores_computation_shape,
            subgrid_shape,
            halo_width=1))

    one_core_computation_shape = (1, 1, 1)
    self.one_core_params = (
        grid_parametrization.GridParametrization
        .create_from_grid_lengths_and_etc(
            grid_lengths,
            one_core_computation_shape,
            self.full_grid_shape,
            halo_width=1))

    self.prs = multigrid_utils.prolong_restrict_matrices_from_params(
        self.many_cores_params, coarsest_subgrid_shape, dtype)

    self.initial_residual_norm = multigrid_utils.poisson_residual_norm(
        self.x, self.b, self.one_core_params)

    self.tpu = TpuRunner(computation_shape=self.computation_shape)
    self.states = self.init_states()
    self.xb = util.combine_subgrids(
        [self.states[i]['xb'] for i in range(len(self.states))],
        self.tpu.replicas)

  def step_fn(self, state, replicas, replica_id, n_coarse, num_cycles):
    coordinates = state['coordinates']
    prs = multigrid_utils.convert_ps_rs_dict_to_tuple(state['ps'], state['rs'])

    mg_cycle_step_fn = multigrid.poisson_mg_cycle_step_fn(
        self.many_cores_params,
        self.coarsest_subgrid_shape,
        n_coarse,
        weight=self.weight,
        boundary_conditions=self.boundary_conditions,
        num_cycles=num_cycles,
        dtype=self.dtype)
    mg_cycle = mg_cycle_step_fn(prs, replica_id, replicas, coordinates)
    state['x0'] = mg_cycle(state['x0'], state['b_minus_a_xb'])
    return state['x0']

  def run_multigrid_tpu(self, n_coarse, num_cycles):
    def mg_cycle():
      step_fn = functools.partial(self.step_fn, n_coarse=n_coarse,
                                  num_cycles=num_cycles)
      x0s = self.tpu.run_with_replica_args(step_fn, self.states)
      for i in range(np.prod(self.computation_shape)):
        self.states[i]['x0'] = x0s[i]
      return util.combine_subgrids(x0s, self.tpu.replicas) + self.xb

    x = mg_cycle()
    normalized_residual_norm = multigrid_utils.poisson_residual_norm(
        x, self.b, self.one_core_params) / self.initial_residual_norm

    return x, normalized_residual_norm, mg_cycle

  def run_multigrid_np(self, n_coarse, num_cycles):
    mg_cycle_xb = multigrid.poisson_mg_cycle_fn_for_one_core(
        self.full_grid_shape,
        n_coarse,
        weight=self.weight,
        coarsest_full_grid_shape=self.coarsest_full_grid_shape,
        params=self.one_core_params,
        prs=self.prs,
        boundary_conditions=self.boundary_conditions,
        num_cycles=num_cycles,
        dtype=self.dtype)

    def mg_cycle(x):
      return mg_cycle_xb(x, self.b)

    x = mg_cycle(np.copy(self.x))
    normalized_residual_norm = multigrid_utils.poisson_residual_norm(
        x, self.b, self.one_core_params) / self.initial_residual_norm

    return x, normalized_residual_norm, mg_cycle

  def run_test_compare_1_core_np_to_many_core_tpu(
      self, n_coarse, computation_shape, boundary_conditions, atol,
      dtype=np.float32):
    self.init(computation_shape, boundary_conditions, dtype)

    num_cycles = 5
    x_one_core, normalized_residual_norm_one_core, _ = self.run_multigrid_np(
        n_coarse, num_cycles)
    x_many_core, normalized_residual_norm_many_core, _ = self.run_multigrid_tpu(
        n_coarse, num_cycles)

    self.assertAllClose(
        normalized_residual_norm_one_core, normalized_residual_norm_many_core,
        atol=atol)

    self.assertAllClose(x_one_core, x_many_core, atol=atol)

  N_COARSE = (1, 2)
  COMPUTATION_SHAPE = ((1, 1, 1), (2, 1, 1), (1, 2, 2), (1, 4, 1))
  BC = ('dirichlet', 'mixed', 'neumann')

  @parameterized.parameters(*itertools.product(N_COARSE, COMPUTATION_SHAPE, BC))
  def test_compare_1_core_np_to_many_core_tpu_(
      self, n_coarse, computation_shape, bc):
    boundary_conditions = self.bcs_map[bc]
    atol = {'dirichlet': 1e-6, 'mixed': 2e-6, 'neumann': 4e-5}[bc]

    self.run_test_compare_1_core_np_to_many_core_tpu(
        n_coarse, computation_shape, boundary_conditions, atol=atol)

  @parameterized.parameters(('dirichlet', np.float32, 6e-7),
                            ('dirichlet', np.float64, 2e-15),
                            ('mixed', np.float32, 2e-6),
                            ('mixed', np.float64, 5e-14),
                            ('neumann', np.float32, 3e-5),
                            ('neumann', np.float64, 3e-3))
  def test_compare_1_core_np_to_many_core_tf_f32_vs_f64_(
      self, boundary_condition, dtype, atol):
    # The np-tpu correspondence often is near the machine limit when comparing
    # f32 to f64. However, in the all-Neumann case something unexpected
    # happens. The calculation is quirky because the Laplacian matrix is
    # ill-conditioned.
    boundary_conditions = self.bcs_map[boundary_condition]

    self.run_test_compare_1_core_np_to_many_core_tpu(
        n_coarse=1, computation_shape=(1, 2, 2),
        boundary_conditions=boundary_conditions, atol=atol, dtype=dtype)

  def test_different_coarsest_subgrid_shapes_approx_equal(self):
    # Values for different coarsest grids should be approximately equal.
    # They should converge with the number of iterations.
    atols = (0.256, 0.190, 0.107, 0.054, 0.026)

    boundary_conditions = self.bcs_map['neumann']

    self.init(many_cores_computation_shape=(1, 1, 1),
              boundary_conditions=boundary_conditions,
              coarsest_subgrid_shape=(3, 3, 3))
    x_333, _, mg_cycle_333 = self.run_multigrid_np(
        n_coarse=1, num_cycles=5)

    self.init(many_cores_computation_shape=(1, 1, 1),
              boundary_conditions=boundary_conditions,
              coarsest_subgrid_shape=(4, 4, 4))
    x_444, _, mg_cycle_444 = self.run_multigrid_np(
        n_coarse=1, num_cycles=5)

    self.assertAllClose(x_333, x_444, atol=atols[0])

    for i, atol in enumerate(atols[1:]):
      with self.subTest(f'cycle_{i + 2}'):
        x_333 = mg_cycle_333(x_333)
        x_444 = mg_cycle_444(x_444)
        self.assertAllClose(x_333, x_444, atol=atol)


if __name__ == '__main__':
  tf.test.main()
