"""Tests for thermodynamics."""

import os

import numpy as np
from swirl_lm.utility.post_processing import thermodynamics
from google3.pyglib import resources
from google3.testing.pybase import googletest
from google3.testing.pybase import parameterized

_TESTDATA_DIR = ('google3/third_party/py/swirl_lm/utility/post_processing/'
                 'test_data')


class ThermodynamicsTest(parameterized.TestCase):

  def setUp(self):
    """Initializes the thermodynamics library."""
    super(ThermodynamicsTest, self).setUp()

    self.water = self.set_up_water_library('water.textpb')

  def set_up_water_library(self, config_filename):
    """Creates the water thermodynamics library object."""
    filename = resources.GetResourceFilename(
        os.path.join(_TESTDATA_DIR, config_filename))
    return thermodynamics.Water(filename)

  @parameterized.named_parameters(
      ('GeoStatic', 'water.textpb', [35872.957, 1.013e5]),
      ('ConstTheta', 'water_const_theta.textpb', [3.5194547e4, 1.013e5]),
      ('Constant', 'water_constant.textpb', [1.013e5, 1.013e5]))
  def testPRefIsComputedCorrectly(self, config_filename, expected_p_ref):
    """Checks if the reference pressure is computed correctly."""
    model = self.set_up_water_library(config_filename)

    zz = np.array([8000.0, 0.0])

    p_ref = model.p_ref(zz)

    self.assertSequenceAlmostEqual(expected_p_ref, p_ref, places=3)

  @parameterized.named_parameters(
      ('GeoStatic', 'water.textpb', [244.7043506427, 290.4]),
      ('ConstTheta', 'water_const_theta.textpb', [221.8007353601, 300.0]),
      ('Constant', 'water_constant.textpb', [300.0, 300.0]))
  def testTRefIsComputedCorrectly(self, config_filename, expected_t_ref):
    """Checks if the reference temperature is computed correctly."""
    model = self.set_up_water_library(config_filename)

    zz = np.array([8000.0, 0.0])

    t_ref = model.t_ref(zz)

    self.assertSequenceAlmostEqual(expected_t_ref, t_ref, places=4)

  @parameterized.named_parameters(
      ('GeoStatic', 'water.textpb', [0.5113437376, 1.2167470128]),
      ('ConstTheta', 'water_const_theta.textpb', [0.553477413, 1.1778111084]),
      ('Constant', 'water_constant.textpb', [1.1778111084, 1.1778111084]))
  def testRhoRefIsComputedCorrectly(self, config_filename, expected_rho_ref):
    """Checks if the reference density is computed correctly."""
    model = self.set_up_water_library(config_filename)

    zz = np.array([8000.0, 0.0])

    rho_ref = model.rho_ref(zz)

    self.assertSequenceAlmostEqual(expected_rho_ref, rho_ref, places=4)

  def testLiquidPotentialTemperatureIsComputedCorrectly(self):
    """Checks if the liquid potential temperature is computed correctly."""
    t = np.array([271.3176, 284.31808])
    q_t = np.array([0.01, 0.003])
    rho = np.array([1.2, 0.9])
    zz = np.array([100.0, 800.0])

    theta_li = self.water.liquid_ice_potential_temperature(t, q_t, rho, zz)

    expected = [255.42886, 292.1548]
    self.assertSequenceAlmostEqual(expected, theta_li, 4)

  def testVirtualPotentialTemperatureIsComputedCorrectly(self):
    """Checks if the virtual potential temperature is computed correctly."""
    t = np.array([271.3176, 284.31808])
    q_t = np.array([0.01, 0.003])
    rho = np.array([1.2, 0.9])
    zz = np.array([100.0, 800.0])

    theta_v = self.water.virtual_potential_temperature(t, q_t, rho, zz)

    expected = [271.0622, 292.6904]
    self.assertSequenceAlmostEqual(expected, theta_v, 4)

  def testTemperatureIsComputedCorrectly(self):
    """Checks if temperature is computed correctly."""
    e_int = np.array([5042.96235, 14533.03])
    q_t = np.array([0.01, 0.003])
    rho = np.array([1.2, 0.9])

    t = self.water.temperature(e_int, rho, q_t)

    expected = np.array([271.9599, 284.31808])
    self.assertSequenceAlmostEqual(expected, t, 4)

  def testDensityIsComputedCorrectly(self):
    """Checks if density is compputed correctly at saturation."""
    e_t = np.array([4220.06235, 14548.03])
    q_t = np.array([0.01, 0.003])
    u = np.array([10.0, 5.0])
    v = np.array([4.0, 2.0])
    w = np.array([2.0, 1.0])
    rho_0 = np.array([1.0, 1.0])

    rho = self.water.density(e_t, q_t, u, v, w, rho_0)

    expected = [1.3057063, 1.2405005]

    self.assertSequenceAlmostEqual(expected, rho)

  def testLiquidWaterMassFractionIsComputedCorrectly(self):
    """Checks if the liquid phase water mass fraction is computed correctly."""
    t = np.array([266.0, 293.0])
    rho = np.array([1.2, 1.0])
    q_t = np.array([0.01, 0.05])

    q_l = self.water.liquid_mass_fraction(t, rho, q_t)

    expected = [0.0, 0.03500298]
    self.assertSequenceAlmostEqual(expected, q_l)

  def testCondensedWaterMassFractionIsComputedCorrectly(self):
    """Checks if the condensed water mass fraction is computed correctly."""
    t = np.array([266.0, 293.0])
    rho = np.array([1.2, 1.0])
    q_t = np.array([0.01, 0.05])

    q_c = self.water.condensed_mass_fraction(t, rho, q_t)

    expected = [0.00761373, 0.03500298]
    self.assertSequenceAlmostEqual(expected, q_c)

  def testInternalEnergyIsComputedCorrectly(self):
    """Checks if the internal enenrgy is computed correctly."""
    temperature = np.array([266.0, 293.0])
    q_l = np.array([0.001, 0.003])
    q_i = np.array([0.004, 0.0])
    q_t = np.array([0.01, 0.003])

    e_int = self.water.internal_energy(temperature, q_t, q_l, q_i)

    expected = [4220.06235, 14548.03]

    self.assertSequenceAlmostEqual(expected, e_int, places=2)

  def testTotalEnthalpyIsComputedCorrectly(self):
    """Checks if the total enthalpy is computed correctly."""
    e_t = np.array([40000.0, 30000.0])
    temperature = np.array([266.0, 293.0])
    rho = np.array([1.0, 0.8])
    q_t = np.array([0.01, 0.003])

    h_t = self.water.total_enthalpy(e_t, rho, q_t, temperature)

    expected = [115848.7695, 114154.1708]

    self.assertSequenceAlmostEqual(expected, h_t, places=2)


if __name__ == '__main__':
  googletest.main()
