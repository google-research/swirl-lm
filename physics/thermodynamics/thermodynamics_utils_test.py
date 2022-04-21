"""Tests for google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.physics.thermodynamics.thermodynamics_utils."""

import numpy as np
from swirl_lm.physics.thermodynamics import thermodynamics_utils
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf


@test_util.run_all_in_graph_and_eager_modes
class ThermodynamicsUtilsTest(tf.test.TestCase):

  def testRegularizeScalarBoundForcesScalarsBetweenZeroAndOne(self):
    """Checks if the scalars are bounded between 0 and 1."""
    sc = [tf.constant([-0.1, 0.5, 1.1], dtype=tf.float32),]

    sc_bounded = (
        thermodynamics_utils.regularize_scalar_bound(sc))

    sc_output = self.evaluate(sc_bounded)

    self.assertLen(sc_output, 1)
    self.assertAllEqual(sc_output[0], [0., 0.5, 1.])

  def testRegularizeScalarSumForcesSumOfScalarToBeOne(self):
    """Checks if the sum of all scalars is 1."""
    sc = {
        'Y1': [tf.constant([0.5, 0.2, 0.6], dtype=tf.float32),],
        'Y2': [tf.constant([0.1, 0.3, 0.5], dtype=tf.float32),],
        'Y3': [tf.constant([0.4, 0.0, 0.9], dtype=tf.float32),],
    }

    sc_reg = thermodynamics_utils.regularize_scalar_sum(sc)

    sc_output = self.evaluate(sc_reg)

    sc_expected = {
        'Y1': [np.array([0.5, 0.4, 0.3], dtype=np.float32),],
        'Y2': [np.array([0.1, 0.6, 0.25], dtype=np.float32),],
        'Y3': [np.array([0.4, 0.0, 0.45], dtype=np.float32),],
    }

    self.assertLen(sc_output, 3)
    for key, value in sc_output.items():
      self.assertAllClose(value, sc_expected[key])

  def testComputeAmbientAirFractionProducesCorrectAirMassFraction(self):
    """Checks if the massfraction of ambient air is computed correctly."""
    sc = {
        'Y1': [tf.constant([0.5, 0.2, 0.2], dtype=tf.float32),],
        'Y2': [tf.constant([0.1, 0.3, 0.5], dtype=tf.float32),],
        'Y3': [tf.constant([0.4, 0.0, 0.1], dtype=tf.float32),],
    }

    air = (
        thermodynamics_utils.compute_ambient_air_fraction(sc)
    )

    air_output = self.evaluate(air)

    air_expected = np.array([0.0, 0.5, 0.2], dtype=np.float32)

    self.assertLen(air_output, 1)
    self.assertAllClose(air_output[0], air_expected)

  def testComputeMixtureMolecularWeightProducesCorrectResult(self):
    """Checks if the molecular weight of the mixture is computed correctly."""
    molecular_weights = {
        'Y1': 0.032,
        'Y2': 0.018,
        'Y3': 0.016,
        'Y4': 0.044,
        'ambient': 0.028,
    }
    massfractions = {
        'Y1': [tf.constant(0.16),],
        'Y2': [tf.constant(0.09),],
        'Y3': [tf.constant(0.04),],
        'Y4': [tf.constant(0.11),],
        'ambient': [tf.constant(0.6),],
    }

    w_mix = self.evaluate(
        thermodynamics_utils.compute_mixture_molecular_weight(
            molecular_weights, massfractions))

    expected = 0.02745098039

    self.assertAlmostEqual(expected, w_mix[0])


if __name__ == '__main__':
  tf.test.main()
