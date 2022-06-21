"""Tests for google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.physics.thermodynamics.linear_mixing."""

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import parameters_pb2
from swirl_lm.physics.thermodynamics import linear_mixing
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.net.proto2.python.public import text_format


@test_util.run_all_in_graph_and_eager_modes
class LinearMixingTest(tf.test.TestCase):

  def setUp(self):
    super(LinearMixingTest, self).setUp()

    pbtxt = (
        R'density: 1.0  '
        R'scalars {  '
        R'  name: "Y1" diffusivity: 0.002 density: 0.5  '
        R'}  '
        R'scalars {  '
        R'  name: "Y2" diffusivity: 0.002 density: 2.0  '
        R'}  ')
    config = text_format.Parse(pbtxt, parameters_pb2.SwirlLMParameters())

    params = parameters_lib.SwirlLMParameters(config)

    self.model = linear_mixing.LinearMixing(params)

  def testLinearMixingComputesCorrectMixtureDensity(self):
    """Checks if the mixture density is computed correctly."""
    states = {
        'Y1': [tf.constant([[0, 1], [0.4, 0.8]], dtype=tf.float32),
               tf.constant([[-0.1, 1.1], [0.3, 0.7]], dtype=tf.float32)],

        'Y2': [tf.constant([[1, 0], [0.6, 0.2]], dtype=tf.float32),
               tf.constant([[1.1, -0.1], [0.2, 0.1]], dtype=tf.float32)],
    }
    additional_states = {}

    rho_mix = self.evaluate(
        self.model.update_density(states, additional_states))

    expected_rho_mix = [
        np.array([[2.0, 0.5], [1.4, 0.8]], dtype=np.float32),
        np.array([[2.0, 0.5], [1.05, 0.75]], dtype=np.float32)
    ]

    self.assertLen(rho_mix, 2)
    for i in range(2):
      self.assertAllClose(expected_rho_mix[i], rho_mix[i])


if __name__ == '__main__':
  tf.test.main()
