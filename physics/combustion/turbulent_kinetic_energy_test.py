"""Tests for google3.research.simulation.tensorflow.fluid.models.combustion.turbulent_kinetic_energy."""
from absl import flags
import numpy as np
from swirl_lm.physics.combustion import turbulent_kinetic_energy
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

FLAGS = flags.FLAGS


@test_util.run_all_in_graph_and_eager_modes
class TurbulentKineticEnergyTest(tf.test.TestCase):

  def run_update_fn_test(self, update_fn, states, additional_states):
    """Runs the `update_fn` with `states` and `additional_states` provided."""
    kernel_op = get_kernel_fn.ApplyKernelConvOp(4)
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    params = grid_parametrization.GridParametrization()

    res = self.evaluate(
        update_fn(kernel_op, replica_id, replicas, states, additional_states,
                  params))
    return res

  def testLocalBoxFilterSmeresSpike(self):
    """Checks if the local box filter redistributes 27 with its neighbors."""
    value = np.zeros((16, 16, 16), dtype=np.float32)
    value[8, 8, 8] = 27.0

    filtered = self.evaluate(
        turbulent_kinetic_energy._local_box_filter_3d(
            tf.unstack(tf.convert_to_tensor(value))))

    expected = np.zeros((16, 16, 16), dtype=np.float32)
    expected[7:10, 7:10, 7:10] = 1.0

    self.assertAllEqual(np.stack(filtered), expected)

  def testConstantTurbulentKineticEnergy(self):
    """Checks if the turbulent kinetic energy computes correctly at center."""
    FLAGS.tke_constant = 66.0

    update_fn = turbulent_kinetic_energy.tke_update_fn_manager(
        turbulent_kinetic_energy.TkeUpdateOption.CONSTANT)

    states = {}
    additional_states = {
        'tke': tf.unstack(tf.zeros((16, 16, 16), dtype=tf.float32))
    }

    tke = self.run_update_fn_test(update_fn, states, additional_states)['tke']

    self.assertAllClose(66.0 * np.ones((16, 16, 16), dtype=np.float32), tke)

  def testAlgebraicTurbulentKineticEnergy(self):
    """Checks if the turbulent kinetic energy computes correctly at center."""
    u = np.zeros((16, 16, 16), dtype=np.float32)
    u[8, 8, 8] = 27.0
    v = np.zeros((16, 16, 16), dtype=np.float32)
    v[8, 8, 8] = -54.0
    w = np.zeros((16, 16, 16), dtype=np.float32)
    w[8, 8, 8] = 81.0

    update_fn = turbulent_kinetic_energy.tke_update_fn_manager(
        turbulent_kinetic_energy.TkeUpdateOption.ALGEBRAIC)

    states = {'u': tf.unstack(tf.convert_to_tensor(u)),
              'v': tf.unstack(tf.convert_to_tensor(v)),
              'w': tf.unstack(tf.convert_to_tensor(w))}
    additional_states = {'tke': tf.unstack(tf.zeros(u.shape, dtype=tf.float32))}

    tke = self.run_update_fn_test(update_fn, states, additional_states)['tke']

    self.assertLen(tke, 16)
    self.assertAllClose(tke[8][8, 8], 182.0)

  def testTurbulentViscosityTkeModel(self):
    """Checks TKE with the turbulent viscosity model."""
    FLAGS.lx = 3.0
    FLAGS.ly = 8.0
    FLAGS.lz = 15.0
    FLAGS.cx = 1
    FLAGS.cy = 1
    FLAGS.cz = 1
    FLAGS.nx = 13
    FLAGS.ny = 23
    FLAGS.nz = 33
    FLAGS.halo_width = 1

    update_fn = turbulent_kinetic_energy.tke_update_fn_manager(
        turbulent_kinetic_energy.TkeUpdateOption.TURBULENT_VISCOSITY)

    additional_states = {
        'tke': tf.unstack(tf.zeros((6, 6, 6), dtype=tf.float32)),
        'nu_t': tf.unstack(0.01 * tf.ones((6, 6, 6), dtype=tf.float32)),
    }

    tke = self.run_update_fn_test(update_fn, {}, additional_states)['tke']

    self.assertAllClose(tke, 0.06524779402 * np.ones(
        (6, 6, 6), dtype=np.float32))


if __name__ == '__main__':
  tf.test.main()
