"""Tests for thermodynamics_manager."""

import os

import numpy as np
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.physics.thermodynamics import thermodynamics_pb2
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.pyglib import gfile
from google3.pyglib import resources
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_parameters_pb2
from google3.testing.pybase import parameterized

_TF_FLOAT32 = tf.float32
_TESTDATA_DIR = ('google3/third_party/py/swirl_lm/physics/thermodynamics/'
                 'testdata')


class ThermodynamicsManagerTest(tf.test.TestCase, parameterized.TestCase):

  def set_up_thermodynamics_manager(self, config_filename):
    """Creates an instance of the ThermodynamcisManager."""
    with gfile.Open(
        resources.GetResourceFilename(
            os.path.join(_TESTDATA_DIR, config_filename)), 'r') as f:
      config = text_format.Parse(
          f.read(),
          incompressible_structured_mesh_parameters_pb2
          .IncompressibleNavierStokesParameters())

    params = (
        incompressible_structured_mesh_config
        .IncompressibleNavierStokesParameters(config))
    params.halo_width = 2
    params.periodic_dims = [False, False, False]

    return thermodynamics_manager.thermodynamics_factory(params)

  def _run_on_tpu(self, replicas, inputs, device_fn):
    """Wrapper for the Cartesian grid method."""
    runner = TpuRunner(replicas=replicas)
    return runner.run(device_fn, *inputs)

  @parameterized.named_parameters(
      ('WaterGeoStaticAnelastic', 'water_geo_static_anelastic.textpb', {
          'e_t': [5201.06235, 22396.03],
          'q_t': [0.01, 0.003],
          'u': [10.0, 5.0],
          'v': [4.0, 2.0],
          'w': [2.0, 1.0],
          'rho': [1.2, 0.9],
      }, {
          'zz': [100.0, 800.0],
      }, (1.2055893, 1.129458)),
      ('WaterGeoStaticLowMach', 'water_geo_static_low_mach.textpb', {
          'e_t': [5201.06235, 22396.03],
          'q_t': [0.01, 0.003],
          'u': [10.0, 5.0],
          'v': [4.0, 2.0],
          'w': [2.0, 1.0],
          'rho': [1.2, 0.9],
      }, {
          'zz': [100.0, 800.0],
      }, (1.2906898, 1.1277938)),
      ('IdealGasAnelastic', 'ideal_gas_anelastic.textpb', {
          'T': [270.0, 900.0],
          'Y_CO2': [0.1, 0.8],
      }, {
          'zz': [100.0, 800.0],
      }, (1.2055893, 1.129458)),
      ('IdealGasLowMach', 'ideal_gas_low_mach.textpb', {
          'T': [270.0, 900.0],
          'Y_CO2': [0.1, 0.8],
      }, {
          'zz': [100.0, 800.0],
      }, (1.338971, 0.490763)),
  )
  def testUpdateDensityProvidesTheCorrectDensity(self, config_filename,
                                                 states_val,
                                                 additional_states_val,
                                                 expected):
    """Checks if density is corrected updated under different solver modes."""
    model = self.set_up_thermodynamics_manager(config_filename)

    rho = np.ones((8, 8, 8), dtype=np.float32)
    rho[:4, ...] = expected[0]
    rho[4:, ...] = expected[1]

    states = {  # pylint: disable=g-complex-comprehension
        key: [
            val[0] * tf.ones((8, 8), dtype=_TF_FLOAT32),
        ] * 4 + [
            val[1] * tf.ones((8, 8), dtype=_TF_FLOAT32),
        ] * 4 for key, val in states_val.items()
    }
    states.update({
        'rho': tf.unstack(tf.convert_to_tensor(rho, dtype=_TF_FLOAT32))
    })
    states_0 = {
        'rho': tf.unstack(tf.convert_to_tensor(rho + 0.1, dtype=_TF_FLOAT32))
    }
    additional_states = {  # pylint: disable=g-complex-comprehension
        key: [
            val[0] * tf.ones((8, 8), dtype=_TF_FLOAT32),
        ] * 4 + [
            val[1] * tf.ones((8, 8), dtype=_TF_FLOAT32),
        ] * 4 for key, val in additional_states_val.items()
    }

    kernel_op = get_kernel_fn.ApplyKernelConvOp(4)
    replicas = np.array([[[0]]])

    def device_fn(replica_id):
      """Wraps the update density function to run on TPU."""
      return model.update_density(kernel_op, replica_id, replicas, states,
                                  additional_states, states_0)

    inputs = [[tf.constant(0)]]

    output = self._run_on_tpu(replicas, inputs, device_fn)
    rho_new, drho = output[0]

    if model.solver_mode == thermodynamics_pb2.Thermodynamics.LOW_MACH:
      expected_drho = -0.1 * np.ones((8, 8, 8), dtype=np.float32)
    else:
      expected_drho = np.zeros((8, 8, 8), dtype=np.float32)

    with self.subTest(name='Rho'):
      self.assertLen(rho_new, 8)
      expected_rho = rho
      self.assertAllClose(expected_rho, rho_new)

    with self.subTest(name='dRho'):
      self.assertLen(drho, 8)
      self.assertAllClose(expected_drho, drho, atol=1e-5, rtol=1e-5)

  @parameterized.named_parameters(
      ('WaterGeoStaticAnelastic', 'water_geo_static_anelastic.textpb', {
          'e_t': [5201.06235, 22396.03],
          'q_t': [0.01, 0.003],
          'u': [10.0, 5.0],
          'v': [4.0, 2.0],
          'w': [2.0, 1.0],
          'rho': [1.2, 0.9],
      }, {
          'zz': [100.0, 800.0],
      }, (1.290692, 1.127795)),
      ('WaterGeoStaticLowMach', 'water_geo_static_low_mach.textpb', {
          'e_t': [5201.06235, 22396.03],
          'q_t': [0.01, 0.003],
          'u': [10.0, 5.0],
          'v': [4.0, 2.0],
          'w': [2.0, 1.0],
          'rho': [1.2, 0.9],
      }, {
          'zz': [100.0, 800.0],
      }, (1.290692, 1.127795)),
      ('IdealGasAnelastic', 'ideal_gas_anelastic.textpb', {
          'T': [270.0, 900.0],
          'Y_CO2': [0.1, 0.8],
      }, {
          'zz': [100.0, 800.0],
      }, (1.338971, 0.490763)),
      ('IdealGasLowMach', 'ideal_gas_low_mach.textpb', {
          'T': [270.0, 900.0],
          'Y_CO2': [0.1, 0.8],
      }, {
          'zz': [100.0, 800.0],
      }, (1.338971, 0.490763)),
  )
  @test_util.run_all_in_graph_and_eager_modes
  def testUpdateThermalDensityProvidesTheCorrectThermodynamicDensity(
      self, config_filename, states_val, additional_states_val, expected):
    """Checks if thermodyanmic density is corrected updated."""
    model = self.set_up_thermodynamics_manager(config_filename)

    states = {
        key: [tf.convert_to_tensor(val, dtype=tf.float32)]
        for key, val in states_val.items()
    }
    additional_states = {
        key: [tf.convert_to_tensor(val, dtype=tf.float32)]
        for key, val in additional_states_val.items()
    }

    rho = self.evaluate(model.update_thermal_density(states, additional_states))

    self.assertAllClose(expected, rho[0])


if __name__ == '__main__':
  tf.test.main()
