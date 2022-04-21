"""Tests for google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.physics.thermodynamics.thermodynamics_generic."""

import numpy as np
from swirl_lm.physics.thermodynamics import thermodynamics_generic
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_parameters_pb2


@test_util.run_all_in_graph_and_eager_modes
class ThermodynamicsGenericTest(tf.test.TestCase):

  def setUp(self):
    """Initializes the thermodynamics library."""
    super(ThermodynamicsGenericTest, self).setUp()

    pbtxt = (
        R'density: 1.2  '
        R'p_thermal: 1.013e5  ')
    config = text_format.Parse(
        pbtxt,
        incompressible_structured_mesh_parameters_pb2
        .IncompressibleNavierStokesParameters())

    params = (
        incompressible_structured_mesh_config
        .IncompressibleNavierStokesParameters(config))
    params.nz = 36

    self.model = thermodynamics_generic.ThermodynamicModel(params)

  def testRhoRefProvidesConstantReferenceDensity(self):
    """Checks if `rho_ref` is the constant in config with length `nz`."""
    rho = self.evaluate(self.model.rho_ref())

    expected = [1.2,] * 36
    self.assertAllClose(expected, rho)

  def testPRefProvidesConstantReferencePressure(self):
    """Checks if `p_ref` is the constant in config with length `nz`."""
    zz = tf.convert_to_tensor(np.transpose(
        np.tile(np.linspace(0.0, 1000.0, 8), (8, 8, 1)), (2, 0, 1)))

    p = self.evaluate(self.model.p_ref(tf.unstack(zz)))

    expected = 1.013e5 * np.ones((8, 8, 8), dtype=np.float32)
    self.assertAllClose(expected, p)

  def testPRefGradientIsZero(self):
    """Checks if the gradient of `p_ref` is 0 in all three directions."""
    zz = tf.convert_to_tensor(np.transpose(
        np.tile(np.linspace(0.0, 1000.0, 8), (8, 8, 1)), (2, 0, 1)))

    p = self.model.p_ref(tf.unstack(zz))

    expected = np.zeros((6, 6, 6), dtype=np.float32)

    kernel_op = get_kernel_fn.ApplyKernelConvOp(4)

    with self.subTest(name='dPdxIsZero'):
      dpdx = np.array(self.evaluate(kernel_op.apply_kernel_op_x(p, 'kDx')))

      self.assertAllClose(expected, dpdx[1:-1, 1:-1, 1:-1])

    with self.subTest(name='dPdyIsZero'):
      dpdy = np.array(self.evaluate(kernel_op.apply_kernel_op_y(p, 'kDy')))

      self.assertAllClose(expected, dpdy[1:-1, 1:-1, 1:-1])

    with self.subTest(name='dPdzIsZero'):
      dpdz = np.array(
          self.evaluate(kernel_op.apply_kernel_op_z(p, 'kDz', 'kDzsh')))

      self.assertAllClose(expected, dpdz[1:-1, 1:-1, 1:-1])


if __name__ == '__main__':
  tf.test.main()
