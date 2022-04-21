"""Tests for google3.research.simulation.tensorflow.fluid.models.combustion.onestep."""

import functools

from absl import flags
import numpy as np
from swirl_lm.physics.combustion import onestep
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

FLAGS = flags.FLAGS


@test_util.run_all_in_graph_and_eager_modes
class OnestepTest(tf.test.TestCase):

  def testOnestepReactionSource(self):
    """Checks if the onestep reaction source is correctly computed."""
    a_cst = 2.0
    coeff_f = 1.0
    coeff_o = 2.0
    e_a = 8314.0
    q = 300.0
    cp = 1000.0
    w_f = 0.016
    w_o = 0.032
    nu_f = 1.0
    nu_o = 2.0

    nx = 8
    ny = 8
    nz = 1

    kernel_op = get_kernel_fn.ApplyKernelConvOp(4)
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    states = {
        'Y_F': [0.2 * tf.ones((nx, ny), dtype=tf.float32),] * nz,
        'Y_O': [0.8 * tf.ones((nx, ny), dtype=tf.float32),] * nz,
        'T': [2000.0 * tf.ones((nx, ny), dtype=tf.float32),] * nz,
        'rho': [tf.ones((nx, ny), dtype=tf.float32),] * nz,
    }
    additional_states = {
        'src_Y_F': [tf.zeros((nx, ny), dtype=tf.float32)] * nz,
        'src_Y_O': [tf.zeros((nx, ny), dtype=tf.float32)] * nz,
        'src_T': [tf.zeros((nx, ny), dtype=tf.float32)] * nz,
    }
    params = grid_parametrization.GridParametrization()

    update_fn = onestep.reaction_source_update_fn(a_cst, coeff_f, coeff_o, e_a,
                                                  q, cp, w_f, w_o, nu_f, nu_o)

    reaction_source = self.evaluate(
        update_fn(kernel_op, replica_id, replicas, states, additional_states,
                  params))

    self.assertLen(reaction_source, 3)

    with self.subTest(name='sourceForFuel'):
      self.assertAllClose(
          reaction_source['src_Y_F'][0], -151.63722428063275 * np.ones(
              (8, 8), dtype=np.float32))
    with self.subTest(name='sourceForOxidizer'):
      self.assertAllClose(reaction_source['src_Y_O'][0],
                          -606.548897122531 * np.ones((8, 8), dtype=np.float32))
    with self.subTest(name='sourceForTemperature'):
      self.assertAllClose(
          reaction_source['src_T'][0], 2843.1979552618636 * np.ones(
              (8, 8), dtype=np.float32))

  def testIntegratedReactionSourceUpdateFn(self):
    """Checks if the integrated reaction source is computed correctly."""
    a_cst = 1.2e6
    coeff_f = 1.0
    coeff_o = 1.0
    e_a = 1.16e5
    q = 1e5
    cp = 1200.0
    w_f = 0.016
    w_o = 0.032
    w_p = 0.08
    nu_f = 1.0
    nu_o = 2.0
    p_thermal = 1.013e5
    nt = 100
    FLAGS.dt = 1e-3

    nx = 1
    ny = 1
    nz = 1

    kernel_op = get_kernel_fn.ApplyKernelConvOp(4)
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    states = {
        'Y_F': [0.2 * tf.ones((nx, ny), dtype=tf.float32),] * nz,
        'Y_O': [0.8 * tf.ones((nx, ny), dtype=tf.float32),] * nz,
        'T': [1200.0 * tf.ones((nx, ny), dtype=tf.float32),] * nz,
    }
    additional_states = {
        'src_Y_F': [tf.zeros((nx, ny), dtype=tf.float32)] * nz,
        'src_Y_O': [tf.zeros((nx, ny), dtype=tf.float32)] * nz,
        'src_T': [tf.zeros((nx, ny), dtype=tf.float32)] * nz,
    }
    params = grid_parametrization.GridParametrization()

    update_fn = onestep.integrated_reaction_source_update_fn(
        a_cst, coeff_f, coeff_o, e_a, q, cp, w_f, w_o, w_p, nu_f, nu_o,
        p_thermal, nt)

    reaction_source = self.evaluate(
        update_fn(kernel_op, replica_id, replicas, states, additional_states,
                  params))

    self.assertLen(reaction_source, 3)

    with self.subTest(name='sourceForFuel'):
      self.assertAllClose(
          reaction_source['src_Y_F'][0], -20.69343 * np.ones(
              (1, 1), dtype=np.float32))
    with self.subTest(name='sourceForOxidizer'):
      self.assertAllClose(reaction_source['src_Y_O'][0],
                          -82.77373 * np.ones((1, 1), dtype=np.float32))
    with self.subTest(name='sourceForTemperature'):
      self.assertAllClose(
          reaction_source['src_T'][0], 107784.4 * np.ones(
              (1, 1), dtype=np.float32))

  def testHomogeneousReactorWithOnestepChemistry(self):
    """Checks if the onestep chemistry integrates correctly."""
    a_cst = 1.2e6
    coeff_f = 1.0
    coeff_o = 1.0
    e_a = 1.16e5
    q = 1e5
    cp = 1200.0
    w_f = 0.016
    w_o = 0.032
    w_p = 0.08
    nu_f = 1.0
    nu_o = 2.0
    p_thermal = 1.013e5
    t_end = 1e-2
    delta_t = 1e-3
    nt = 100

    y_f = [0.2 * tf.ones((1, 1), dtype=tf.float32)]
    y_o = [0.8 * tf.ones((1, 1), dtype=tf.float32)]
    temperature = [1200.0 * tf.ones((1, 1), dtype=tf.float32)]

    states_new = functools.partial(
        onestep.one_step_reaction_integration,
        delta_t=delta_t,
        a_cst=a_cst,
        coeff_f=coeff_f,
        coeff_o=coeff_o,
        e_a=e_a,
        q=q,
        cp=cp,
        w_f=w_f,
        w_o=w_o,
        w_p=w_p,
        nu_f=nu_f,
        nu_o=nu_o,
        p_thermal=p_thermal,
        nt=nt)

    # Expected values are computed from a homogeneous reactor with the same
    # set of parameters.
    expected = [
        [
            0.17930657, 0.12317445, 0.02183588, 0.0068689, 0.00382878,
            0.00261947, 0.00198139, 0.00158982, 0.00132592, 0.00113636
        ],
        [
            0.71722627, 0.4926978, 0.08734352, 0.02747561, 0.01531513,
            0.01047786, 0.00792557, 0.0063593, 0.00530369, 0.00454543
        ],
        [
            1307.7844, 1600.1462, 2127.9583, 2205.919, 2221.76, 2228.063,
            2231.3945, 2233.441, 2234.8218, 2235.817
        ],
    ]

    for i in range(int(t_end / delta_t)):
      y_f, y_o, temperature = self.evaluate(states_new(y_f, y_o, temperature))

      self.assertAlmostEqual(y_f[0][0, 0], expected[0][i], 3)
      self.assertAlmostEqual(y_o[0][0, 0], expected[1][i], 3)
      self.assertAlmostEqual(temperature[0][0, 0], expected[2][i], 3)


if __name__ == '__main__':
  tf.test.main()
