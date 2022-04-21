"""Tests for google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.physics.thermodynamics.constant_density."""

import numpy as np
from swirl_lm.physics.thermodynamics import constant_density
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_parameters_pb2


@test_util.run_all_in_graph_and_eager_modes
class ConstantDensityTest(tf.test.TestCase):

  def setUp(self):
    super(ConstantDensityTest, self).setUp()

    pbtxt = R'density: 1.2  '
    config = text_format.Parse(
        pbtxt,
        incompressible_structured_mesh_parameters_pb2
        .IncompressibleNavierStokesParameters())

    params = (
        incompressible_structured_mesh_config
        .IncompressibleNavierStokesParameters(config))

    self.model = constant_density.ConstantDensity(params)

  def testConstantDensityReturnsCorrectDensity(self):
    """Checks if the mixture density is computed correctly."""
    states = {
        'rho': [
            0.2 * tf.ones((6, 6), dtype=tf.float32),
            1.6 * tf.ones((6, 6), dtype=tf.float32),
            0.3 * tf.ones((6, 6), dtype=tf.float32),
            1.8 * tf.ones((6, 6), dtype=tf.float32),
            0.6 * tf.ones((6, 6), dtype=tf.float32),
            3.6 * tf.ones((6, 6), dtype=tf.float32),
        ],
    }
    additional_states = {}

    rho = self.evaluate(self.model.update_density(states, additional_states))

    expected = 1.2 * np.ones((6, 6, 6), dtype=np.float32)

    self.assertAllClose(expected, np.stack(rho))


if __name__ == '__main__':
  tf.test.main()
