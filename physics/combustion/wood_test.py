"""Tests for google3.research.simulation.tensorflow.fluid.models.combustion.wood."""
import os

from absl import flags
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import parameters_pb2
from swirl_lm.physics.combustion import wood
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.pyglib import gfile
from google3.pyglib import resources
from google3.testing.pybase import parameterized

_TESTDATA_DIR = 'google3/third_party/py/swirl_lm/physics/combustion/test_data'

FLAGS = flags.FLAGS


@test_util.run_all_in_graph_and_eager_modes
class WoodTest(tf.test.TestCase, parameterized.TestCase):

  def set_up_model(self, config_filename='dry_wood.textpb'):
    FLAGS.dt = 0.1
    filename = resources.GetResourceFilename(
        os.path.join(_TESTDATA_DIR, config_filename))
    with gfile.GFile(filename, 'r') as f:
      pbtxt = f.read()

    params = text_format.Parse(pbtxt, parameters_pb2.SwirlLMParameters())
    config = parameters_lib.SwirlLMParameters(params)

    return wood.wood_combustion_factory(config)

  def testBoundScalarRegulerizesScalarBetweenSpecificMinAndMax(self):
    """Checks if scalars are bounded correctly between their min and max."""
    rho_f = tf.convert_to_tensor(
        np.array([-1e-3, 0.5, 1.0, 1.001]), dtype=tf.float32)
    y_o = [
        tf.convert_to_tensor(np.array([-1e-3, 0.5]), dtype=tf.float32),
        tf.convert_to_tensor(np.array([1.0, 1.001]), dtype=tf.float32),
    ]

    with self.subTest(name='RhofGreaterThan0'):
      rho_f = self.evaluate(wood._bound_scalar(rho_f, minval=0.0))
      expected = np.array([0.0, 0.5, 1.0, 1.001], dtype=np.float32)
      self.assertAllClose(expected, rho_f)

    with self.subTest(name='YoBetween0and1'):
      y_o = self.evaluate(wood._bound_scalar(y_o, minval=0.0, maxval=1.0))
      expected = np.array([[0.0, 0.5], [1.0, 1.0]], dtype=np.float32)
      self.assertAllClose(expected, y_o)

  def testReactionRateOfWoodCombustion(self):
    """Checks if the reaction rate of wood combustion is computed correctly."""
    rho_f = tf.constant(2.0, dtype=tf.float32)
    rho_g = tf.constant(1.0, dtype=tf.float32)
    y_o = tf.constant(0.2, dtype=tf.float32)
    tke = tf.constant(16.0, dtype=tf.float32)
    temperature = tf.constant(800.0, dtype=tf.float32)
    s_b = 2.0
    s_x = 0.05
    c_f = 0.9

    f_f = self.evaluate(
        wood._reaction_rate(rho_f, rho_g, y_o, tke, temperature, s_b, s_x,
                            c_f))

    self.assertAllClose(f_f, 1.8297792501)

  @parameterized.named_parameters(
      ('BelowThreshold', 289.0, 0.0, 0.0),
      ('BetweenThresholds', 400.0, 0.2857142857, 0.7142857143),
      ('AboveThreshold', 900.0, 0.4, 1.0))
  def testEvaporationGeneratesCorrectEvaporationRateAndMoistureCDF(
      self, t, expected_f_w, expected_phi):
    """Checks if the evaporation rate and cumulative water updated correctly."""
    t = tf.constant(t, dtype=tf.float32)
    phi_max = tf.constant(0.0, dtype=tf.float32)
    rho_m = tf.constant(0.04, dtype=tf.float32)
    dt = 1e-2
    c_w = 0.1

    f_w, phi = self.evaluate(wood._evaporation(t, phi_max, rho_m, dt, c_w))

    with self.subTest(name='EvaporationRate'):
      self.assertAllClose(expected_f_w, f_w)

    with self.subTest(name='WaterCDF'):
      self.assertAllClose(expected_phi, phi)

  def testDryWoodUpdateFnProducesCorrectUpdatesForDryWoodCombustion(self):
    """Checks if all relevant states are updated correctly."""
    u = np.zeros((16, 16, 16), dtype=np.float32)
    u[8, 8, 8] = 2.7
    v = np.zeros((16, 16, 16), dtype=np.float32)
    v[8, 8, 8] = -5.4
    w = np.zeros((16, 16, 16), dtype=np.float32)
    w[8, 8, 8] = 8.1
    rho_f = np.zeros((16, 16, 16), dtype=np.float32)
    rho_f[8, 8, 8] = 2.0
    tke = np.zeros((16, 16, 16), dtype=np.float32)
    tke[8, 8, 8] = 1.82

    states = {
        'u': tf.unstack(tf.convert_to_tensor(u)),
        'v': tf.unstack(tf.convert_to_tensor(v)),
        'w': tf.unstack(tf.convert_to_tensor(w)),
        'rho': [tf.ones((16, 16), dtype=tf.float32),] * 16,
        'T': [800.0 * tf.ones((16, 16), dtype=tf.float32),] * 16,
        'Y_O': [0.2 * tf.ones((16, 16), dtype=tf.float32),] * 16,
    }
    additional_states = {
        'rho_f': tf.unstack(tf.convert_to_tensor(rho_f)),
        'T_s': [700.0 * tf.ones((16, 16), dtype=tf.float32),] * 16,
        'src_rho': [tf.zeros((16, 16), dtype=tf.float32),] * 16,
        'src_T': [tf.zeros((16, 16), dtype=tf.float32),] * 16,
        'src_Y_O': [tf.zeros((16, 16), dtype=tf.float32),] * 16,
        'tke': tf.unstack(tf.convert_to_tensor(tke)),
    }

    kernel_op = get_kernel_fn.ApplyKernelConvOp(4)
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    params = grid_parametrization.GridParametrization()
    params.dt = 0.1

    model = self.set_up_model('dry_wood.textpb')
    update_fn = model.update_fn(
        tf.unstack(tf.convert_to_tensor(rho_f)))

    reaction_source = self.evaluate(
        update_fn(kernel_op, replica_id, replicas, states, additional_states,
                  params))

    # The expected values are computed manually.
    with self.subTest(name='Rhof'):
      expected = np.zeros((16, 16, 16), dtype=np.float32)
      expected[8, 8, 8] = 1.9980016
      self.assertAllClose(np.stack(reaction_source['rho_f']), expected)

    with self.subTest(name='Ts'):
      expected = 700.0 * np.ones((16, 16, 16), dtype=np.float32)
      expected[8, 8, 8] = 700.53564
      self.assertAllClose(np.stack(reaction_source['T_s']), expected)

    with self.subTest(name='SrcRho'):
      expected = np.zeros((16, 16, 16), dtype=np.float32)
      expected[8, 8, 8] = 0.01998425
      self.assertAllClose(np.stack(reaction_source['src_rho']), expected)

    with self.subTest(name='SrcT'):
      expected = np.zeros((16, 16, 16), dtype=np.float32)
      expected[8, 8, 8] = 582.74164
      self.assertAllClose(np.stack(reaction_source['src_T']), expected)

    with self.subTest(name='SrcYo'):
      expected = np.zeros((16, 16, 16), dtype=np.float32)
      expected[8, 8, 8] = -0.02391787
      self.assertAllClose(np.stack(reaction_source['src_Y_O']), expected)

  def testMoistWoodUpdateFnProducesCorrectUpdatesForMoistWoodCombustion(self):
    """Checks if all relevant states are updated correctly."""
    u = np.zeros((16, 16, 16), dtype=np.float32)
    u[8, 8, 8] = 2.7
    v = np.zeros((16, 16, 16), dtype=np.float32)
    v[8, 8, 8] = -5.4
    w = np.zeros((16, 16, 16), dtype=np.float32)
    w[8, 8, 8] = 8.1
    rho_f = np.zeros((16, 16, 16), dtype=np.float32)
    rho_f[8, 8, 8] = 2.0
    tke = np.zeros((16, 16, 16), dtype=np.float32)
    tke[8, 8, 8] = 1.82

    states = {
        'u': tf.unstack(tf.convert_to_tensor(u)),
        'v': tf.unstack(tf.convert_to_tensor(v)),
        'w': tf.unstack(tf.convert_to_tensor(w)),
        'rho': [tf.ones((16, 16), dtype=tf.float32),] * 16,
        'T': [800.0 * tf.ones((16, 16), dtype=tf.float32),] * 16,
        'Y_O': [0.2 * tf.ones((16, 16), dtype=tf.float32),] * 16,
    }
    additional_states = {
        'rho_f': tf.unstack(tf.convert_to_tensor(rho_f)),
        'rho_m': tf.unstack(tf.convert_to_tensor(0.08 * rho_f)),
        'phi_w': tf.unstack(tf.zeros((16, 16, 16), dtype=tf.float32)),
        'T_s': [700.0 * tf.ones((16, 16), dtype=tf.float32),] * 16,
        'src_rho': [tf.zeros((16, 16), dtype=tf.float32),] * 16,
        'src_T': [tf.zeros((16, 16), dtype=tf.float32),] * 16,
        'src_Y_O': [tf.zeros((16, 16), dtype=tf.float32),] * 16,
        'tke': tf.unstack(tf.convert_to_tensor(tke)),
    }

    kernel_op = get_kernel_fn.ApplyKernelConvOp(4)
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    params = grid_parametrization.GridParametrization()
    params.dt = 0.1

    model = self.set_up_model('moist_wood.textpb')
    update_fn = model.update_fn(
        tf.unstack(tf.convert_to_tensor(rho_f)))

    reaction_source = self.evaluate(
        update_fn(kernel_op, replica_id, replicas, states, additional_states,
                  params))

    # The expected values are computed manually.
    with self.subTest(name='Rhof'):
      expected = np.zeros((16, 16, 16), dtype=np.float32)
      expected[8, 8, 8] = 1.9980016
      self.assertAllClose(np.stack(reaction_source['rho_f']), expected)

    with self.subTest(name='Rhow'):
      expected = np.zeros((16, 16, 16), dtype=np.float32)
      expected[8, 8, 8] = 0.15979785
      self.assertAllClose(np.stack(reaction_source['rho_m']), expected)

    with self.subTest(name='Phiw'):
      expected = 0.63213724 * np.ones((16, 16, 16), dtype=np.float32)
      self.assertAllClose(np.stack(reaction_source['phi_w']), expected)

    with self.subTest(name='Ts'):
      expected = 700.0 * np.ones((16, 16, 16), dtype=np.float32)
      expected[8, 8, 8] = 700.05096
      self.assertAllClose(np.stack(reaction_source['T_s']), expected)

    with self.subTest(name='SrcRho'):
      expected = np.zeros((16, 16, 16), dtype=np.float32)
      expected[8, 8, 8] = 0.02200574
      self.assertAllClose(np.stack(reaction_source['src_rho']), expected)

    with self.subTest(name='SrcT'):
      expected = np.zeros((16, 16, 16), dtype=np.float32)
      expected[8, 8, 8] = 582.7093
      self.assertAllClose(np.stack(reaction_source['src_T']), expected)

    with self.subTest(name='SrcYo'):
      expected = np.zeros((16, 16, 16), dtype=np.float32)
      expected[8, 8, 8] = -0.02391787
      self.assertAllClose(np.stack(reaction_source['src_Y_O']), expected)

  def testRadiativeEmissionIsComputedCorrectly(self):
    """Checks if the radiation term is computed correctly."""
    t = tf.constant([800.0, 300.0], dtype=tf.float32)
    t_a = tf.constant([300.0, 800.0], dtype=tf.float32)
    l = 0.5

    src = self.evaluate(wood._radiative_emission(t, t_a, l))

    expected = [45530.1, -45530.1]

    self.assertAllClose(expected, src)


if __name__ == '__main__':
  tf.test.main()
