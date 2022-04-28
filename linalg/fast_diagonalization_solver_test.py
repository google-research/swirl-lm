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
"""Tests for google3.research.simulation.tensorflow.fluid.framework.linalg.fast_diagonalization_solver."""

import itertools
import numpy as np
from swirl_lm.linalg import fast_diagonalization_solver
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
import tensorflow as tf
from google3.research.simulation.tensorflow.fluid.framework import initializer
from google3.research.simulation.tensorflow.fluid.framework import util
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner

from google3.testing.pybase import parameterized


class FastDiagonalizationSolverDemo(object):

  def __init__(self, matrices, rhs_fn, cutoff, replicas, core_shape,
               grid_lengths, halo_width):
    self._matrices = matrices
    self._rhs_fn = rhs_fn
    self._cutoff = cutoff
    self._halo_width = halo_width
    subgrid_shape = [c + 2 * halo_width for c in core_shape]
    computation_shape = np.array(replicas.shape)

    self._params = (grid_parametrization.GridParametrization.
                    create_from_grid_lengths_and_etc(
                        grid_lengths, computation_shape, subgrid_shape,
                        halo_width))

  def init_fn(self, coordinates):
    """Initializes the right-hand side tensor field for the fluid framework."""
    return {
        'rhs':
            initializer.partial_mesh_for_core(self._params, coordinates,
                                              self._rhs_fn, perm=None)
    }

  def step_fn(self, state, replicas, replica_id):
    """Solves the linear system and updates state."""
    n_dim = state['rhs'].get_shape().as_list()

    rhs = [
        state['rhs'][i, self._halo_width:-self._halo_width,
                     self._halo_width:-self._halo_width]
        for i in range(self._halo_width, n_dim[0] - self._halo_width)
    ]

    rhs_dim = [dim - 2 * self._halo_width for dim in n_dim]

    solver = fast_diagonalization_solver.fast_diagonalization_solver(
        self._matrices, rhs_dim, replica_id, replicas, self._cutoff)

    rhs = tf.unstack(solver(tf.stack(rhs)))

    paddings = tf.constant([[self._halo_width, self._halo_width],
                            [self._halo_width, self._halo_width]])
    state['rhs'] = [
        tf.pad(rhs_i, paddings=paddings, mode='CONSTANT') for rhs_i in rhs
    ]
    state['rhs'] = [tf.zeros_like(state['rhs'][0])] * self._halo_width + state[
        'rhs'] + [tf.zeros_like(state['rhs'][0])] * self._halo_width

    return rhs, state


class FastDiagonalizationSolverTest(tf.test.TestCase, parameterized.TestCase):

  _REPLICAS = [
      np.array([[[0]]], dtype=np.int32),
      np.array([[[0]], [[1]]], dtype=np.int32),
      np.array([[[0], [1]]], dtype=np.int32),
      np.array([[[0, 1]]], dtype=np.int32),
      np.array([[[0, 1], [2, 3]]], dtype=np.int32),
      np.array([[[0, 1]], [[2, 3]]], dtype=np.int32),
      np.array([[[0], [1]], [[2], [3]]], dtype=np.int32),
      np.array([[[0, 1, 2, 3]]], dtype=np.int32),
      np.array([[[0], [1], [2], [3]]], dtype=np.int32),
      np.array([[[0]], [[1]], [[2]], [[3]]], dtype=np.int32),
  ]

  _DTYPE = [
      np.float32,
      np.complex64
  ]

  @parameterized.parameters(*itertools.product(_REPLICAS, _DTYPE))
  def testFastDiagonalizationSolverSolvesPoissonEquation(self, replicas, dtype):
    # Provides definitions to the parameters.
    n = 62
    l = 2. * np.pi
    halo_width = 1
    cutoff = 1e-8
    computation_shape = np.array(replicas.shape)

    # Generates the linear operators.
    a = [None] * 3
    n_dim = [None] * 3
    for i in range(3):
      n_dim[i] = n * computation_shape[i]
      a[i] = get_kernel_fn._make_banded_matrix([1., -2., 1.], n_dim[i])
      a[i] = a[i].astype(dtype)

    def rhs_fn(xx, yy, zz, lx, ly, lz, coord):
      """Defines the right hand side tensor."""
      del lx, ly, lz, coord
      return tf.math.cos(xx) * tf.math.sin(yy) * tf.math.sin(zz)

    # Initializes the solver.
    grid_lengths = (l, l, l)
    core_shape = (n, n, n)
    solver = FastDiagonalizationSolverDemo(a, rhs_fn, cutoff, replicas,
                                           core_shape, grid_lengths, halo_width)

    # Actual right hand side for reference.
    x = [np.linspace(0., l, n_dim_i + 2) for n_dim_i in n_dim]
    x = [tf.constant(x_i[1:-1], dtype=tf.float32) for x_i in x]
    xx, yy, zz = util.meshgrid(x[0], x[1], x[2])
    tf_actual_rhs = rhs_fn(xx, yy, zz, l, l, l, None)

    coordinates = util.grid_coordinates(computation_shape)
    num_replicas = np.prod(computation_shape)
    runner = TpuRunner(computation_shape=computation_shape)
    replica_outputs = runner.run_with_replica_args(
        solver.step_fn,
        [solver.init_fn(coordinates[i]) for i in range(num_replicas)])

    tpu_res = [output[0] for output in replica_outputs]
    actual_rhs = self.evaluate(tf_actual_rhs)

    tpu_res = [np.stack(tpu_res_i) for tpu_res_i in tpu_res]
    res = util.combine_subgrids(tpu_res, replicas, halo_width=0)

    res_rhs = np.einsum('ip,pqr->iqr', a[0], res) + np.einsum(
        'jq,pqr->pjr', a[1], res) + np.einsum('kr,pqr->pqk', a[2], res)

    self.assertAllClose(res_rhs, actual_rhs, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
  tf.test.main()
