"""Tests for google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.physics.thermodynamics.ideal_gas."""

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import parameters_pb2
from swirl_lm.physics.thermodynamics import ideal_gas
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.net.proto2.python.public import text_format


@test_util.run_all_in_graph_and_eager_modes
class IdealGasTest(tf.test.TestCase):

  def set_up_model(self, pbtxt):
    """Initializes the `IdealGas` object from provided text proto."""
    config = text_format.Parse(pbtxt, parameters_pb2.SwirlLMParameters())

    params = parameters_lib.SwirlLMParameters(config)
    params.nz = 2

    return ideal_gas.IdealGas(params)

  def testDensityUpdatesGeneratesCorrectThermodynamicDensityForAmbientAir(self):
    """Checks if the thermodynamic density for air is correct."""
    pbtxt = (
        R'p_thermal: 1.013e5  '
        R'thermodynamics {  '
        R'  ideal_gas_law {}  '
        R'}  '
        R'scalars {  '
        R'  name: "ambient" molecular_weight: 0.02875 solve_scalar: false  '
        R'}  ')
    model = self.set_up_model(pbtxt)
    states = {
        'T': [tf.constant([[250.0, 260.0], [270.0, 280.0]], dtype=tf.float32),
              tf.constant([[290.0, 300.0], [310.0, 320.0]], dtype=tf.float32)],
    }
    additional_states = {}

    rho_mix = self.evaluate(model.update_density(states, additional_states))

    expected = [
        np.array([[1.4011065007, 1.3472177891], [1.297320834, 1.250987947]],
                 dtype=np.float32),
        np.array([[1.2078504316, 1.1675887506], [1.1299245973, 1.0946144537]],
                 dtype=np.float32)
    ]

    self.assertLen(rho_mix, 2)
    for i in range(2):
      self.assertAllClose(expected[i], rho_mix[i])

  def testDensityUpdatesGeneratesCorrectThermodynamicDensityForGasMixture(self):
    """Checks if the thermodynamic density for air is correct."""
    pbtxt = (
        R'p_thermal: 1.013e5  '
        R'thermodynamics {  '
        R'  ideal_gas_law {}  '
        R'}  '
        R'scalars {  '
        R'  name: "Y_CO2" molecular_weight: 0.044 solve_scalar: true  '
        R'}  '
        R'scalars {  '
        R'  name: "ambient" molecular_weight: 0.02875 solve_scalar: false  '
        R'}  ')
    model = self.set_up_model(pbtxt)
    states = {
        'T': [tf.constant([[250.0, 260.0], [270.0, 280.0]], dtype=tf.float32),
              tf.constant([[290.0, 300.0], [310.0, 320.0]], dtype=tf.float32)],
        'Y_CO2': [tf.constant([[-0.1, 1.1], [0.0, 1.0]], dtype=tf.float32),
                  tf.constant([[0.4, 0.5], [0.6, 0.7]], dtype=tf.float32)],
    }
    additional_states = {}

    rho_mix = self.evaluate(model.update_density(states, additional_states))

    expected = [
        np.array([[1.4011065007, 2.0618289642], [1.297320834, 1.9145554667]],
                 dtype=np.float32),
        np.array([[1.4022537992, 1.4123410317], [1.4265905963, 1.445252392]],
                 dtype=np.float32)
    ]

    self.assertLen(rho_mix, 2)
    for i in range(2):
      self.assertAllClose(expected[i], rho_mix[i])

  def testDensityUpdatesGeneratesCorrectDensityForAmbientAirWithTheta(self):
    """Checks if the thermodynamic density for air is correct."""
    pbtxt = (
        R'p_thermal: 1.013e5  '
        R'thermodynamics {  '
        R'  ideal_gas_law {  '
        R'    const_theta: 300.0  '
        R'    cv_d: 716.9 '
        R'  }  '
        R'}  '
        R'scalars {  '
        R'  name: "ambient" molecular_weight: 0.02875 solve_scalar: false  '
        R'}  ')
    model = self.set_up_model(pbtxt)
    states = {
        'theta': [
            tf.constant([[250.0, 260.0], [270.0, 280.0]], dtype=tf.float32),
            tf.constant([[290.0, 300.0], [310.0, 320.0]], dtype=tf.float32)
        ],
    }
    additional_states = {}

    rho_mix = self.evaluate(model.update_density(states, additional_states))

    expected = [
        np.array([[1.4011065007, 1.3472177891], [1.297320834, 1.250987947]],
                 dtype=np.float32),
        np.array([[1.2078504316, 1.1675887506], [1.1299245973, 1.0946144537]],
                 dtype=np.float32)
    ]

    self.assertLen(rho_mix, 2)
    for i in range(2):
      self.assertAllClose(expected[i], rho_mix[i])

  def testDensityUpdatesGeneratesCorrectDensityForGasMixtureWithTheta(self):
    """Checks if the thermodynamic density for air is correct."""
    pbtxt = (
        R'p_thermal: 1.013e5  '
        R'thermodynamics {  '
        R'  ideal_gas_law {  '
        R'    const_theta: 300.0  '
        R'    cv_d: 716.9 '
        R'  }  '
        R'}  '
        R'scalars {  '
        R'  name: "Y_CO2" molecular_weight: 0.044 solve_scalar: true  '
        R'}  '
        R'scalars {  '
        R'  name: "ambient" molecular_weight: 0.02875 solve_scalar: false  '
        R'}  ')
    model = self.set_up_model(pbtxt)
    states = {
        'theta': [
            tf.constant([[250.0, 260.0], [270.0, 280.0]], dtype=tf.float32),
            tf.constant([[290.0, 300.0], [310.0, 320.0]], dtype=tf.float32)
        ],
        'Y_CO2': [
            tf.constant([[-0.1, 1.1], [0.0, 1.0]], dtype=tf.float32),
            tf.constant([[0.4, 0.5], [0.6, 0.7]], dtype=tf.float32)
        ],
    }
    additional_states = {}

    rho_mix = self.evaluate(model.update_density(states, additional_states))

    expected = [
        np.array([[1.4011065007, 2.0618289642], [1.297320834, 1.9145554667]],
                 dtype=np.float32),
        np.array([[1.4022537992, 1.4123410317], [1.4265905963, 1.445252392]],
                 dtype=np.float32)
    ]

    self.assertLen(rho_mix, 2)
    for i in range(2):
      self.assertAllClose(expected[i], rho_mix[i])

  def testIdealGasLawProvidesCorrectDensity(self):
    """Checks if the ideal gas law computes the density correctly."""
    p = tf.constant(1e5, dtype=tf.float32)
    r = tf.constant(2e2, dtype=tf.float32)
    t = tf.constant(5e2, dtype=tf.float32)

    rho = self.evaluate(ideal_gas.IdealGas.density_by_ideal_gas_law(p, r, t))

    expected = 1.0
    self.assertEqual(expected, rho)

  def testPotentialTemperatureIsConvertedToTemperatureCorrectly(self):
    """Checks if temperature is correctly derived from potential temperature."""
    pbtxt = (
        R'p_thermal: 1.013e5  '
        R'thermodynamics {  '
        R'  ideal_gas_law {  '
        R'    t_s: 300.0 '
        R'    height: 8000.0 '
        R'    delta_t: 60.0 '
        R'    cv_d: 716.9 '
        R'  }  '
        R'}  ')

    theta = [
        tf.constant([320.0, 280.0], dtype=tf.float32),
        tf.constant([500.0, 200.0], dtype=tf.float32)
    ]

    zz = [
        tf.constant([0.0, 500.0], dtype=tf.float32),
        tf.constant([1000.0, 5000.0], dtype=tf.float32),
    ]

    with self.subTest(name='TemperatureNoZSameAsPotentialTemperature'):
      model = self.set_up_model(pbtxt)

      t = self.evaluate(model._potential_temperature_to_temperature(theta))

      expected = [np.array([320.0, 280.0]), np.array((500.0, 200.0))]
      self.assertAllEqual(expected, t)

    with self.subTest(name='TemperatureIsCorrectWithGeopotential'):
      model = self.set_up_model(pbtxt)

      t = self.evaluate(model._potential_temperature_to_temperature(theta, zz))

      expected = [
          np.array([320.0, 275.44871546]),
          np.array((483.77697482, 168.1878625))
      ]
      self.assertAllClose(expected, t)

    with self.subTest(name='TemperatureIsCorrectWithGeopotentialConstTheta'):
      pbtxt = (
          R'p_thermal: 1.013e5  '
          R'thermodynamics {  '
          R'  ideal_gas_law {  '
          R'    const_theta: 300.0  '
          R'    cv_d: 716.9 '
          R'  }  '
          R'}  ')
      model = self.set_up_model(pbtxt)

      t = self.evaluate(model._potential_temperature_to_temperature(theta, zz))

      expected = [
          np.array([320.0, 275.44004063]),
          np.array((483.71443082, 167.42886163))
      ]
      self.assertAllClose(expected, t)

  def testTemperatureIsConvertedToPotentialTemperatureCorrectly(self):
    """Checks if potential temperature is correctly derived from temperature."""
    pbtxt = (R'p_thermal: 1.013e5  '
             R'thermodynamics {  '
             R'  ideal_gas_law {  '
             R'    t_s: 300.0 '
             R'    height: 8000.0 '
             R'    delta_t: 60.0 '
             R'    cv_d: 716.9 '
             R'  }  '
             R'}  ')

    expected = [
        np.array([320.0, 280.0], dtype=np.float32),
        np.array([500.0, 200.0], dtype=np.float32)
    ]

    zz = [
        tf.constant([0.0, 500.0], dtype=tf.float32),
        tf.constant([1000.0, 5000.0], dtype=tf.float32),
    ]

    with self.subTest(name='TemperatureNoZSameAsPotentialTemperature'):
      model = self.set_up_model(pbtxt)

      t = [
          tf.constant([320.0, 280.0], dtype=tf.float32),
          tf.constant((500.0, 200.0), dtype=tf.float32)
      ]

      theta = self.evaluate(model.temperature_to_potential_temperature(t))

      self.assertAllEqual(expected, theta)

    with self.subTest(name='TemperatureIsCorrectWithGeopotential'):
      model = self.set_up_model(pbtxt)

      t = [
          tf.constant([320.0, 275.44871546], dtype=tf.float32),
          tf.constant((483.77697482, 168.1878625), dtype=tf.float32)
      ]

      theta = self.evaluate(model.temperature_to_potential_temperature(t, zz))

      self.assertAllClose(expected, theta)

    with self.subTest(name='TemperatureIsCorrectWithGeopotentialConstTheta'):
      pbtxt = (R'p_thermal: 1.013e5  '
               R'thermodynamics {  '
               R'  ideal_gas_law {  '
               R'    const_theta: 300.0  '
               R'    cv_d: 716.9 '
               R'  }  '
               R'}  ')
      model = self.set_up_model(pbtxt)

      t = [
          tf.constant([320.0, 275.44004063], dtype=tf.float32),
          tf.constant((483.71443082, 167.42886163), dtype=tf.float32)
      ]

      theta = self.evaluate(model.temperature_to_potential_temperature(t, zz))

      self.assertAllClose(expected, theta)

  def testPRefIsComputedCorrectly(self):
    """Checks if the reference pressure is computed correctly."""
    zz = [
        tf.constant([0.0, 500.0], dtype=tf.float32),
        tf.constant([1000.0, 5000.0], dtype=tf.float32),
    ]

    with self.subTest(name='WithNonConstantTheta'):
      pbtxt = (
          R'p_thermal: 1.013e5  '
          R'thermodynamics {  '
          R'  ideal_gas_law {  '
          R'    t_s: 300.0 '
          R'    height: 8000.0 '
          R'    delta_t: 60.0 '
          R'    cv_d: 716.9 '
          R'  }  '
          R'}  ')
      model = self.set_up_model(pbtxt)

      p = self.evaluate(model.p_ref(zz))

      expected = [
          np.array([1.01300e5, 95657.12691099]),
          np.array([90262.96132122, 55268.7819519])
      ]
      self.assertAllClose(expected, p)

    with self.subTest(name='WithConstantTheta'):
      pbtxt = (
          R'p_thermal: 1.013e5  '
          R'thermodynamics {  '
          R'  ideal_gas_law {  '
          R'    const_theta: 300.0  '
          R'    cv_d: 716.9 '
          R'  }  '
          R'}  ')
      model = self.set_up_model(pbtxt)

      p = self.evaluate(model.p_ref(zz))

      expected = [
          np.array([1.01300e5, 95646.5910943]),
          np.array([90222.15497689, 54401.36778622])
      ]
      self.assertAllClose(expected, p)

  def testTRefIsComputedCorrectlyWithConstantPotentialTemperature(self):
    """Checks if the reference temperature is correct with constant theta."""
    pbtxt = (
        R'p_thermal: 1.013e5  '
        R'thermodynamics {  '
        R'  ideal_gas_law {  '
        R'    const_theta: 300.0  '
        R'    cv_d: 716.9 '
        R'  }  '
        R'}  ')
    model = self.set_up_model(pbtxt)

    with self.subTest(name='WithGeoPotential'):
      zz = [
          tf.constant([0.0, 500.0], dtype=tf.float32),
          tf.constant([1000.0, 5000.0], dtype=tf.float32),
      ]

      t = self.evaluate(model.t_ref(zz))

      expected = [
          np.array([300.0, 295.11432924]),
          np.array([290.22865849, 251.14329245])
      ]
      self.assertAllClose(expected, t)

    with self.subTest(name='WithoutGeoPotential'):
      t = self.evaluate(model.t_ref())

      expected = [300.0,] * 2
      self.assertAllClose(expected, t)

  def testTRefIsComputedCorrectlyWithNoneConstantPotentialTemperature(self):
    """Checks if the reference temperature is correct with constant theta."""
    pbtxt = (
        R'p_thermal: 1.013e5  '
        R'thermodynamics {  '
        R'  ideal_gas_law {  '
        R'    t_s: 300.0 '
        R'    height: 8000.0 '
        R'    delta_t: 60.0 '
        R'    cv_d: 716.9 '
        R'  }  '
        R'}  ')
    model = self.set_up_model(pbtxt)

    with self.subTest(name='WithGeoPotential'):
      zz = [
          tf.constant([0.0, 500.0], dtype=tf.float32),
          tf.constant([1000.0, 5000.0], dtype=tf.float32),
      ]

      t = self.evaluate(model.t_ref(zz))

      expected = [
          np.array([300.0, 296.2548752]),
          np.array([292.53881989, 266.72401666])
      ]
      self.assertAllClose(expected, t)

    with self.subTest(name='WithoutGeoPotential'):
      t = self.evaluate(model.t_ref())

      expected = [300.0,] * 2
      self.assertAllClose(expected, t)

  def testRhoRefIsComputedCorrectlyWithConstantPotentialTemperature(self):
    """Checks if the reference density is correct with constant theta."""
    pbtxt = (
        R'p_thermal: 1.013e5  '
        R'thermodynamics {  '
        R'  ideal_gas_law {  '
        R'    const_theta: 300.0  '
        R'    cv_d: 716.9 '
        R'  }  '
        R'}  ')
    model = self.set_up_model(pbtxt)

    with self.subTest(name='WithGeoPotential'):
      zz = [
          tf.constant([0.0, 500.0], dtype=tf.float32),
          tf.constant([1000.0, 5000.0], dtype=tf.float32),
      ]

      t = self.evaluate(model.rho_ref(zz))

      expected = [
          np.array([1.1763081, 1.12904719]),
          np.array([1.0829435, 0.75460753])
      ]
      self.assertAllClose(expected, t)

    with self.subTest(name='WithoutGeoPotential'):
      t = self.evaluate(model.rho_ref())

      expected = [1.1763081,] * 2
      self.assertAllClose(expected, t)

  def testRhoRefIsComputedCorrectlyWithNoneConstantPotentialTemperature(self):
    """Checks if the reference density is correct with constant theta."""
    pbtxt = (
        R'p_thermal: 1.013e5  '
        R'thermodynamics {  '
        R'  ideal_gas_law {  '
        R'    t_s: 300.0 '
        R'    height: 8000.0 '
        R'    delta_t: 60.0 '
        R'    cv_d: 716.9 '
        R'  }  '
        R'}  ')
    model = self.set_up_model(pbtxt)

    with self.subTest(name='WithGeoPotential'):
      zz = [
          tf.constant([0.0, 500.0], dtype=tf.float32),
          tf.constant([1000.0, 5000.0], dtype=tf.float32),
      ]

      t = self.evaluate(model.rho_ref(zz))

      expected = [
          np.array([1.1763081, 1.12482439]),
          np.array([1.07487749, 0.72185616])
      ]
      self.assertAllClose(expected, t)

    with self.subTest(name='WithoutGeoPotential'):
      t = self.evaluate(model.rho_ref())

      expected = [1.1763081,] * 2
      self.assertAllClose(expected, t)


if __name__ == '__main__':
  tf.test.main()
