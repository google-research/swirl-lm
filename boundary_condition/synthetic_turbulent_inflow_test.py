"""Tests for swirl_lm.boundary_condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

import numpy as np
from swirl_lm.boundary_condition import synthetic_turbulent_inflow
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.pyglib import gfile
from google3.pyglib import resources
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized

# When set to `True`, the output will be saved when running the tests. This
# is mostly useful when golden test files need to be updated, but normally
# this should always be `False` when the code is checked in.
_SAVE_OUTPUT = False
_INFLOW_DIM = [0, 1, 2]
_REPLICAS = [
    np.array([[[0, 1]]]),
    np.array([[[0], [1]]]),
    np.array([[[0]], [[1]]])
]
_TESTDATA_DIR = 'google3/third_party/py/swirl_lm/boundary_condition/test_data'


class BoundaryConditionsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(BoundaryConditionsTest, self).setUp()

    self._write_dir = (os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'))

  def set_up_synthetic_turbulence_inflow(self, inflow_dim,
                                         replica_partition_dim):
    """Sets up the SyntheticTurbulenceInflow instance."""
    length_scale = (0.1, 0.1, 0.1)
    delta = (0.025, 0.025, 0.025)
    mesh_size = [32, 32, 32]
    mesh_size[replica_partition_dim] = 16
    m = [mesh_size[i] for i in range(3) if i != inflow_dim]
    velocity_mean = [
        tf.zeros((m[0], m[1]), dtype=tf.float32),
        tf.zeros((m[0], m[1]), dtype=tf.float32),
        tf.zeros((m[0], m[1]), dtype=tf.float32),
    ]
    velocity_mean[inflow_dim] = tf.ones((m[0], m[1]), dtype=tf.float32)
    velocity_rms = (
        0.6 * tf.ones((m[0], m[1]), dtype=tf.float32),
        0.2 * tf.ones((m[0], m[1]), dtype=tf.float32),
        0.3 * tf.ones((m[0], m[1]), dtype=tf.float32),
    )
    return (velocity_mean, velocity_rms,
            synthetic_turbulent_inflow.SyntheticTurbulentInflow(
                length_scale, delta, mesh_size, inflow_dim, 0))

  def testHelperKey(self):
    """Checks if helper keys are generated correctly."""
    _, _, inflow_generator = self.set_up_synthetic_turbulence_inflow(0, 0)

    with self.subTest(name='ValidKeyAttributes'):
      self.assertEqual('rand_w_2_0',
                       inflow_generator.helper_key('rand', 'w', 2, 0))
      self.assertEqual('bc_u_1_0', inflow_generator.helper_key('bc', 'u', 1, 0))
      self.assertEqual('mean_v_0_1',
                       inflow_generator.helper_key('mean', 'v', 0, 1))
      self.assertEqual('rms_u_2_1',
                       inflow_generator.helper_key('rms', 'u', 2, 1))

    with self.subTest(name='InvalidHelperType'):
      with self.assertRaisesRegex(
          ValueError, '`helper_type` has to be "bc", mean", "rms", or "rand"'):
        _ = inflow_generator.helper_key('rad', 'u', 0, 0)

    with self.subTest(name='InvalidVelocity'):
      with self.assertRaisesRegex(ValueError,
                                  '`velocity` has to be "u", "v", or "w"'):
        _ = inflow_generator.helper_key('rand', 't', 0, 0)

    with self.subTest(name='InvalidInflowDim'):
      with self.assertRaisesRegex(ValueError,
                                  '`inflow_dim` has to be 0, 1, or 2'):
        _ = inflow_generator.helper_key('rand', 'u', 3, 0)

    with self.subTest(name='InvalidInflowFace'):
      with self.assertRaisesRegex(ValueError, '`inflow_face` has to be 0 or 1'):
        _ = inflow_generator.helper_key('rand', 'u', 0, 2)

  @test_util.run_in_graph_and_eager_modes
  def testInflowPlaneToBCConvertion(self):
    """Checks if the orientation is correct after converting inflow to BC."""
    inflow = tf.convert_to_tensor(np.reshape(np.arange(6), (2, 3)))

    with self.subTest(name='InflowDim0'):
      _, _, inflow_generator = self.set_up_synthetic_turbulence_inflow(0, 0)
      bc = self.evaluate(inflow_generator._inflow_plane_to_bc(inflow, 2))

      expected = np.array([
          [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
          [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
          [[0, 0, 0, 3, 0, 0], [0, 0, 0, 3, 0, 0], [0, 0, 0, 3, 0, 0]],
          [[0, 0, 1, 4, 0, 0], [0, 0, 1, 4, 0, 0], [0, 0, 1, 4, 0, 0]],
          [[0, 0, 2, 5, 0, 0], [0, 0, 2, 5, 0, 0], [0, 0, 2, 5, 0, 0]],
          [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
          [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
      ])
      self.assertAllEqual(expected, bc)

    with self.subTest(name='InflowDim1'):
      _, _, inflow_generator = self.set_up_synthetic_turbulence_inflow(1, 0)
      bc = self.evaluate(inflow_generator._inflow_plane_to_bc(inflow, 2))

      expected = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0], [3, 3, 3],
                            [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [1, 1, 1], [4, 4, 4],
                            [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [2, 2, 2], [5, 5, 5],
                            [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0], [0, 0, 0]]])
      self.assertAllEqual(expected, bc)

    with self.subTest(name='InflowDim2'):
      _, _, inflow_generator = self.set_up_synthetic_turbulence_inflow(2, 0)
      bc = self.evaluate(inflow_generator._inflow_plane_to_bc(inflow, 2))

      expected = np.array([[[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 2, 0, 0], [0, 0, 3, 4, 5, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
                           [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 2, 0, 0], [0, 0, 3, 4, 5, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
                           [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 2, 0, 0], [0, 0, 3, 4, 5, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]])
      self.assertAllEqual(expected, bc)

  @parameterized.parameters(*itertools.product(_INFLOW_DIM, _REPLICAS))
  def testComputeInflowVelocity(self, inflow_dim, replicas):
    """Checks if the turbulent inflow at different faces in parallel."""
    computation_shape = replicas.shape
    replica_partition_dim = computation_shape.index(2)
    velocity_mean, velocity_rms, inflow_generator = (
        self.set_up_synthetic_turbulence_inflow(inflow_dim,
                                                replica_partition_dim))

    inputs = [
        [inflow_generator.generate_random_fields(seed=(1, 3)),
         tf.constant(0)],
        [inflow_generator.generate_random_fields(seed=(5, 7)),
         tf.constant(1)],
    ]

    def device_fn(r, replica_id):
      """Wraps `inflow_generator.compute_inflow_velocity` for TPU."""
      return inflow_generator.compute_inflow_velocity(r, velocity_mean,
                                                      velocity_rms, replica_id,
                                                      replicas,
                                                      seed=(13, 17))

    computation_shape = replicas.shape

    tf.random.set_seed(0)
    # transpose inputs.
    device_inputs = [list(x) for x in zip(*inputs)]
    computation_shape = replicas.shape
    runner = TpuRunner(computation_shape=computation_shape)
    output = runner.run(device_fn, *device_inputs)

    r = np.stack([output[0]['r'], output[1]['r']], axis=0)
    u = np.stack([output[0]['u'], output[1]['u']], axis=0)

    if _SAVE_OUTPUT:
      save_fname = os.path.join(
          self._write_dir,
          'inflow_{}_replicas{}{}{}.npy'.format(inflow_dim, replicas.shape[0],
                                                replicas.shape[1],
                                                replicas.shape[2]))
      with gfile.GFile(save_fname, 'w') as f:
        np.savez(f, r=r, u=u)

    expected_fname = os.path.join(
        _TESTDATA_DIR,
        'inflow_{}_replicas{}{}{}.npy'.format(inflow_dim, replicas.shape[0],
                                              replicas.shape[1],
                                              replicas.shape[2]))
    with gfile.GFile(resources.GetResourceFilename(expected_fname)) as f:
      file_content = np.load(f)
      self.assertAllClose(file_content['r'], r)
      self.assertAllClose(file_content['u'], u)

  @parameterized.parameters(*itertools.product(_INFLOW_DIM))
  def testSyntheticTurbulenceAdditionalStatesUpdateFn(self, inflow_dim):
    """Checks if the random field and boundary conditions are updated."""
    velocity_mean, velocity_rms, inflow_generator = (
        self.set_up_synthetic_turbulence_inflow(inflow_dim, 0))

    r = inflow_generator.generate_random_fields()

    additional_states = {
        'mean_u_{}_0'.format(inflow_dim): velocity_mean[0],
        'mean_v_{}_0'.format(inflow_dim): velocity_mean[1],
        'mean_w_{}_0'.format(inflow_dim): velocity_mean[2],
        'rms_u_{}_0'.format(inflow_dim): velocity_rms[0],
        'rms_v_{}_0'.format(inflow_dim): velocity_rms[1],
        'rms_w_{}_0'.format(inflow_dim): velocity_rms[2],
        'rand_u_{}_0'.format(inflow_dim): r[0],
        'rand_v_{}_0'.format(inflow_dim): r[1],
        'rand_w_{}_0'.format(inflow_dim): r[2],
        'bc_u_{}_0'.format(inflow_dim): tf.constant(0),
        'bc_v_{}_0'.format(inflow_dim): tf.constant(1),
        'bc_w_{}_0'.format(inflow_dim): tf.constant(2),
    }

    kernel_op = get_kernel_fn.ApplyKernelConvOp(8)
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    states = {}
    params = grid_parametrization.GridParametrization()
    params.halo_width = 2

    computation_shape = replicas.shape
    inputs = [
        [
            additional_states,
        ],
    ]

    def device_fn(additional_states):
      """Wraps `additional_states_update_fn` for TPU."""
      return inflow_generator.additional_states_update_fn(
          kernel_op, replica_id, replicas, states, additional_states, params)

    # transpose inputs.
    device_inputs = [list(x) for x in zip(*inputs)]
    computation_shape = replicas.shape
    runner = TpuRunner(computation_shape=computation_shape)
    output = runner.run(device_fn, *device_inputs)[0]

    if inflow_dim == 0:
      expected_shape = (36, 3, 36)
    elif inflow_dim == 1:
      expected_shape = (36, 20, 3)
    else:
      expected_shape = (3, 20, 36)

    self.assertAllEqual(
        np.stack(output['bc_u_{}_0'.format(inflow_dim)]).shape, expected_shape)
    self.assertAllEqual(
        np.stack(output['bc_v_{}_0'.format(inflow_dim)]).shape, expected_shape)
    self.assertAllEqual(
        np.stack(output['bc_w_{}_0'.format(inflow_dim)]).shape, expected_shape)


if __name__ == '__main__':
  tf.test.main()
