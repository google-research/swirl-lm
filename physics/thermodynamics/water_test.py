"""Tests for google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.physics.thermodynamics.water."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from swirl_lm.physics.thermodynamics import water
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_parameters_pb2
from google3.testing.pybase import parameterized


@test_util.run_all_in_graph_and_eager_modes
class WaterTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(WaterTest, self).setUp()

    self.water = self.set_up_model()

  _DEFAULT_REF_STATE = (
      R'    geo_static_reference_state {  '
      R'      t_s: 290.4 '
      R'      height: 8000.0  '
      R'      delta_t: 60.0  '
      R'    }')

  def set_up_model(self, ref_state_pbtxt=_DEFAULT_REF_STATE):
    """Initializes the water thermodynamics library."""
    prefix = (R'p_thermal: 1.013e5  '
              R'thermodynamics {  '
              R'  water {  '
              R'    r_v: 461.89  '
              R'    t_0: 273.0  '
              R'    t_min: 250.0  '
              R'    t_freeze: 273.15  '
              R'    t_triple: 273.16  '
              R'    p_triple: 611.7  '
              R'    e_int_v0: 2.132e6  '
              R'    e_int_i0: 3.34e5  '
              R'    lh_v0: 2.258e6  '
              R'    lh_s0: 2.592e6  '
              R'    cv_d: 716.9  '
              R'    cv_v: 1397.11  '
              R'    cv_l: 4217.4  '
              R'    cv_i: 2050.0  '
              R'    cp_v: 1859.0  '
              R'    cp_l: 4219.9  '
              R'    cp_i: 2050.0  '
              R'    max_temperature_iterations: 101  '
              R'    temperature_tolerance: 1e-3  '
              R'    num_density_iterations: 10  ')
    suffix = R'}} '
    pbtxt = prefix + ref_state_pbtxt + suffix

    config = text_format.Parse(
        pbtxt,
        incompressible_structured_mesh_parameters_pb2
        .IncompressibleNavierStokesParameters())

    params = (
        incompressible_structured_mesh_config
        .IncompressibleNavierStokesParameters(config))

    return water.Water(params)

  def testCvm(self):
    """Checks if the mixture heat capacity is computed correctly."""
    q_tot = [tf.constant(1.0), tf.constant(0.6)]
    q_liq = [tf.constant(0.3), tf.constant(0.2)]
    q_ice = [tf.constant(0.1), tf.constant(0.3)]

    cv_m = self.evaluate(self.water.cv_m(q_tot, q_liq, q_ice))

    expected = [2308.486, 1884.951]

    self.assertAllClose(expected, cv_m)

  def testRm(self):
    """Checks if the mixture gas constant is computed correctly."""
    temperature = [tf.constant(266.0), tf.constant(293.0)]
    rho = [tf.constant(1.0), tf.constant(0.8)]
    q_t = [tf.constant(0.01), tf.constant(0.003)]

    r_m = self.evaluate(self.water.r_m(temperature, rho, q_t))

    expected = [285.14575, 287.2156]

    self.assertAllClose(expected, r_m)

  @parameterized.named_parameters(
      ('GeoStatic', _DEFAULT_REF_STATE, [35872.957, 1.013e5]),
      ('ConstTheta', R'const_theta_reference_state { theta: 300.0 }',
       [35194.5509748833, 1.013e5]),
      ('Constant', R'const_reference_state { t_ref: 300.0 }',
       [1.013e5, 1.013e5]))
  def testPRef(self, ref_state_pbtxt, expected_p_ref):
    """Checks if the reference pressure is computed correctly."""
    zz = [tf.constant(8000.0), tf.constant(0.0)]

    model = self.set_up_model(ref_state_pbtxt)

    p_ref = self.evaluate(model.p_ref(zz))

    self.assertAllClose(expected_p_ref, p_ref)

  @parameterized.named_parameters(
      ('GeoStatic', _DEFAULT_REF_STATE, [244.7043506427, 290.4]),
      ('ConstTheta', R'const_theta_reference_state { theta: 300.0 }',
       [221.8007353601, 300.0]),
      ('Constant', R'const_reference_state { t_ref: 300.0 }',
       [300.0, 300.0]))
  def testTRef(self, ref_state_pbtxt, expected_t_ref):
    """Checks if the reference temperature is computed correctly."""
    zz = [tf.constant(8000.0), tf.constant(0.0)]

    model = self.set_up_model(ref_state_pbtxt)

    t_ref = self.evaluate(model.t_ref(zz))

    self.assertAllClose(expected_t_ref, t_ref)

  @parameterized.named_parameters(
      ('GeoStatic', _DEFAULT_REF_STATE, [0.5113437376, 1.2167470128]),
      ('ConstTheta', R'const_theta_reference_state { theta: 300.0 }',
       [0.553477413, 1.1778111084]),
      ('Constant', R'const_reference_state { t_ref: 300.0 }',
       [1.1778111084, 1.1778111084]))
  def testRhoRef(self, ref_state_pbtxt, expected_rho_ref):
    """Checks if the reference density is computed correctly."""
    zz = [tf.constant(8000.0), tf.constant(0.0)]

    model = self.set_up_model(ref_state_pbtxt)

    rho_ref = self.evaluate(model.rho_ref(zz))

    self.assertAllClose(expected_rho_ref, rho_ref)

  def testAirTemperature(self):
    """Checks if air temperature is computed correctly."""
    q_tot = [tf.constant(1.0), tf.constant(0.6)]
    q_liq = [tf.constant(0.3), tf.constant(0.2)]
    q_ice = [tf.constant(0.1), tf.constant(0.3)]
    e_int = [tf.constant(4e6), tf.constant(1e5)]

    temperature = self.evaluate(
        self.water.air_temperature(e_int, q_tot, q_liq, q_ice))

    expected = [1466.07632795, 266.103269]

    self.assertAllClose(expected, temperature)

  def testSaturationVaporPressure(self):
    """Checks if the saturation vapor pressure is computed correctly."""
    temperature = [tf.constant(273.0), tf.constant(400.0)]
    lh_0 = [
        tf.constant(self.water._lh_s0),
        tf.constant(self.water._lh_v0)
    ]

    d_cp = [
        tf.constant(self.water._cp_v - self.water._cp_l),
    ] * 2

    p = self.evaluate(
        self.water.saturation_vapor_pressure(temperature, lh_0, d_cp))

    expected = [604.37965064, 128229.58693153]

    self.assertAllClose(expected, p)

  def testLiquidFraction(self):
    """Checks if liquid fractions are computed correctly."""
    temperature = [tf.constant(266.0), tf.constant(293.0)]

    with self.subTest(name='WithInputQs'):
      q_l = [tf.constant(0.1), tf.constant(0.3)]
      q_c = [tf.constant(0.5), tf.constant(0.3)]
      liq_frac = self.evaluate(
          self.water.liquid_fraction(temperature, q_l, q_c))

      expected = [0.2, 1.0]

      self.assertAllClose(expected, liq_frac)

    with self.subTest(name='WithoutInputQs'):
      liq_frac = self.evaluate(self.water.liquid_fraction(temperature))

      expected = [0.0, 1.0]

      self.assertAllClose(expected, liq_frac)

  def testSaturationQVapor(self):
    """Checks if the saturation specific humidity is computed correctly."""
    temperature = [tf.constant(266.0), tf.constant(293.0)]
    rho = [tf.constant(1.2), tf.constant(1.0)]
    q_l = [tf.constant(0.1), tf.constant(0.3)]
    q_c = [tf.constant(0.5), tf.constant(0.3)]

    q_v = self.evaluate(
        self.water.saturation_q_vapor(temperature, rho, q_l, q_c))

    expected = [0.00241974, 0.01499702]

    self.assertAllClose(expected, q_v)

  def testSaturationExcess(self):
    """Checks if the saturation excess in equilibrium is computed correctly."""
    temperature = [tf.constant(266.0), tf.constant(293.0)]
    rho = [tf.constant(1.2), tf.constant(1.0)]
    q_l = [tf.constant(0.1), tf.constant(0.3)]
    q_c = [tf.constant(0.5), tf.constant(0.3)]
    q_t = [tf.constant(0.01), tf.constant(0.01)]

    q_excess = self.evaluate(
        self.water.saturation_excess(temperature, rho, q_t, q_l, q_c))

    expected = [0.00758026, 0.0]

    self.assertAllClose(expected, q_excess)

  def testEquilibriumPhasePartition(self):
    """Checks if phase partition is computed correctly at equilibrium."""
    temperature = [tf.constant(266.0), tf.constant(293.0)]
    rho = [tf.constant(1.2), tf.constant(1.0)]
    q_t = [tf.constant(0.01), tf.constant(0.05)]

    q_l, q_i = self.evaluate(
        self.water.equilibrium_phase_partition(temperature, rho, q_t))

    with self.subTest(name='QL'):
      expected = [0.0, 0.03500298]
      self.assertAllClose(expected, q_l)

    with self.subTest(name='QI'):
      expected = [0.00761373, 0.0]
      self.assertAllClose(expected, q_i)

  def testInternalEnergy(self):
    """Checks if the internal enenrgy is computed correctly."""
    temperature = [tf.constant(266.0), tf.constant(293.0)]
    q_l = [tf.constant(0.001), tf.constant(0.003)]
    q_i = [tf.constant(0.004), tf.constant(0.0)]
    q_t = [tf.constant(0.01), tf.constant(0.003)]

    e_int = self.evaluate(
        self.water.internal_energy(temperature, q_t, q_l, q_i))

    expected = [4220.06235, 14548.03]

    self.assertAllClose(expected, e_int)

  def testSpecificInternalEnergy(self):
    """Checks if the internal enenrgy is computed correctly."""
    temperature = [tf.constant(266.0), tf.constant(293.0)]

    e_v, e_l, e_i = self.evaluate(
        self.water.internal_energy_components(temperature))

    expected_v = [2122124.2, 2159846.2]
    expected_l = [-29521.799, 84348.]
    expected_i = [-348350., -293000.]

    self.assertAllClose(expected_v, e_v)
    self.assertAllClose(expected_l, e_l)
    self.assertAllClose(expected_i, e_i)

  def testSaturationInternalEnergy(self):
    """Checks if the internal energy is computed correctly at saturation."""
    temperature = [tf.constant(266.0), tf.constant(293.0)]
    rho = [tf.constant(1.0), tf.constant(0.8)]
    q_t = [tf.constant(0.01), tf.constant(0.003)]

    e_int = self.evaluate(
        self.water.saturation_internal_energy(temperature, rho, q_t))

    expected = [-1377.08809138, 20774.8126]

    self.assertAllClose(expected, e_int, rtol=1e-05, atol=1e-02)

  def testDEintDt(self):
    """Checks if the derivative of internal energy wrt temperaure is correct."""
    temperature = [tf.constant(266.0), tf.constant(293.0)]
    rho = [tf.constant(1.0), tf.constant(0.8)]
    q_t = [tf.constant(0.01), tf.constant(0.003)]

    dedt = self.evaluate(self.water.de_int_dt(temperature, rho, q_t))

    expected = [1289.44906027, 2934.62306238]

    self.assertAllClose(expected, dedt)

  def testSaturationAdjustment(self):
    """Checks if temperature is compputed correctly at saturation."""
    e_int = [
        tf.constant(4220.06235, dtype=tf.float32),
        tf.constant(14548.03, dtype=tf.float32)
    ]
    rho = [
        tf.constant(1.0, dtype=tf.float32),
        tf.constant(0.8, dtype=tf.float32)
    ]
    q_t = [
        tf.constant(0.01, dtype=tf.float32),
        tf.constant(0.003, dtype=tf.float32)
    ]

    t = self.evaluate(self.water.saturation_adjustment(e_int, rho, q_t))

    expected = [270.16278, 284.33896]

    self.assertAllClose(expected, t, atol=1e-5, rtol=0)

  def testSaturationDensity(self):
    """Checks if density is compputed correctly at saturation."""
    e_t = [
        tf.constant(4220.06235, dtype=tf.float32),
        tf.constant(14548.03, dtype=tf.float32)
    ]
    q_t = [
        tf.constant(0.01, dtype=tf.float32),
        tf.constant(0.003, dtype=tf.float32)
    ]
    u = [
        tf.constant(10.0, dtype=tf.float32),
        tf.constant(5.0, dtype=tf.float32),
    ]
    v = [
        tf.constant(4.0, dtype=tf.float32),
        tf.constant(2.0, dtype=tf.float32),
    ]
    w = [
        tf.constant(2.0, dtype=tf.float32),
        tf.constant(1.0, dtype=tf.float32),
    ]
    rho_0 = [
        tf.constant(1.0, dtype=tf.float32),
        tf.constant(1.0, dtype=tf.float32),
    ]

    t = self.evaluate(self.water.saturation_density(e_t, q_t, u, v, w, rho_0))

    expected = [1.305706, 1.240500]

    self.assertAllClose(expected, t)

  def testInternalEnergyComputedCorrectlyFromTotalEnergy(self):
    """Checks if the internal energy is correctly computed from total energy."""
    e_t = [
        tf.constant(5201.06235, dtype=tf.float32),
        tf.constant(22396.03, dtype=tf.float32)
    ]
    u = [
        tf.constant(10.0, dtype=tf.float32),
        tf.constant(5.0, dtype=tf.float32),
    ]
    v = [
        tf.constant(4.0, dtype=tf.float32),
        tf.constant(2.0, dtype=tf.float32),
    ]
    w = [
        tf.constant(2.0, dtype=tf.float32),
        tf.constant(1.0, dtype=tf.float32),
    ]

    with self.subTest(name='HeightProvidedConsidersGeopotential'):
      zz = [
          tf.constant(100.0, dtype=tf.float32),
          tf.constant(800.0, dtype=tf.float32),
      ]

      e = self.evaluate(
          self.water.internal_energy_from_total_energy(e_t, u, v, w, zz))

      expected = [4160.06235, 14533.03]

      self.assertAllClose(expected, e)

    with self.subTest(name='HeightAbsentIgnoresGeopotential'):
      e = self.evaluate(
          self.water.internal_energy_from_total_energy(e_t, u, v, w))

      expected = [5141.06235, 22381.03]

      self.assertAllClose(expected, e)

  def testTotalEnergyComputedCorrectlyFromInternalEnergy(self):
    """Checks if the total energy is correctly computed from internal energy."""
    e = [
        tf.constant(4160.06235, dtype=tf.float32),
        tf.constant(14533.03, dtype=tf.float32)
    ]
    u = [
        tf.constant(10.0, dtype=tf.float32),
        tf.constant(5.0, dtype=tf.float32),
    ]
    v = [
        tf.constant(4.0, dtype=tf.float32),
        tf.constant(2.0, dtype=tf.float32),
    ]
    w = [
        tf.constant(2.0, dtype=tf.float32),
        tf.constant(1.0, dtype=tf.float32),
    ]

    with self.subTest(name='HeightProvidedConsidersGeopotential'):
      zz = [
          tf.constant(100.0, dtype=tf.float32),
          tf.constant(800.0, dtype=tf.float32),
      ]

      e_t = self.evaluate(self.water.total_energy(e, u, v, w, zz))

      expected = [5201.06235, 22396.03]

      self.assertAllClose(expected, e_t)

    with self.subTest(name='HeightAbsentIgnoresGeopotential'):
      e_t = self.evaluate(self.water.total_energy(e, u, v, w))

      expected = [4220.06235, 14548.03]

      self.assertAllClose(expected, e_t)

  def testUpdateTemperatures(self):
    """Checks if the temperature and potential temperatures are correct."""
    states = {
        'e_t': [
            tf.constant(5201.06235, dtype=tf.float32),
            tf.constant(22396.03, dtype=tf.float32)
        ],
        'q_t': [
            tf.constant(0.01, dtype=tf.float32),
            tf.constant(0.003, dtype=tf.float32)
        ],
        'u': [
            tf.constant(10.0, dtype=tf.float32),
            tf.constant(5.0, dtype=tf.float32),
        ],
        'v': [
            tf.constant(4.0, dtype=tf.float32),
            tf.constant(2.0, dtype=tf.float32),
        ],
        'w': [
            tf.constant(2.0, dtype=tf.float32),
            tf.constant(1.0, dtype=tf.float32),
        ],
        'rho': [
            tf.constant(1.2, dtype=tf.float32),
            tf.constant(0.9, dtype=tf.float32),
        ],
    }

    additional_states = {
        'zz': [
            tf.constant(100.0, dtype=tf.float32),
            tf.constant(800.0, dtype=tf.float32),
        ],
    }

    t = self.evaluate(self.water.update_temperatures(states, additional_states))

    with self.subTest(name='Temperature'):
      expected = [271.3176, 284.31808]
      self.assertAllClose(expected, t['T'])

    with self.subTest(name='LiquidIcePotentialTemperature'):
      expected = [255.42886, 292.1548]
      self.assertAllClose(expected, t['theta_li'])

    with self.subTest(name='VirtualPotentialTemperature'):
      expected = [271.0622, 292.69037]
      self.assertAllClose(expected, t['theta_v'])

  @parameterized.named_parameters(
      ('HotSurface', 41000.0, 0.01, 0.0),
      ('ColdSurface', 20000.0, 0.01, 0.0),
      ('HumidSurface', 36000.0, 0.1, 0.0),
      ('DrySurface', 36000.0, 0.0, 0.0),
      ('HotAir', 60000.0, 0.01, 1000.0),
      ('ColdAir', 20000.0, 0.01, 1000.0),
      ('HumidAir', 50000.0, 0.1, 1000.0),
      ('DryAir', 40000.0, 0.0, 1000.0),
  )
  def testThermodynamicsStatesConsistency(
      self,
      e_t_val,
      q_t_val,
      height_val,
  ):
    """Checks if functions in this library are consistent."""
    # Defines the velocity field.
    u = [tf.constant(0.0, dtype=tf.float32)]
    v = [tf.constant(0.0, dtype=tf.float32)]
    w = [tf.constant(0.0, dtype=tf.float32)]

    # Defines the original thermodynamic states.
    e_t = [tf.constant(e_t_val, dtype=tf.float32)]
    q_t = [tf.constant(q_t_val, dtype=tf.float32)]
    height = [tf.constant(height_val, dtype=tf.float32)]

    # Compute the density.
    rho_0 = [tf.constant(1.0, dtype=tf.float32)]
    rho = self.evaluate(
        self.water.saturation_density(e_t, q_t, u, v, w, rho_0, height))

    e = self.evaluate(
        self.water.internal_energy_from_total_energy(e_t, u, v, w, height))

    # Compute the temperature.
    t = self.evaluate(self.water.saturation_adjustment(e, rho, q_t))
    t = [tf.constant(t[0], dtype=tf.float32)]

    # Compute the liquid and ice phase fractions.
    q_l, q_i = self.evaluate(
        self.water.equilibrium_phase_partition(t, rho, q_t))

    # Compute the internal energy from the temperature.
    e_model = self.evaluate(self.water.internal_energy(t, q_t, q_l, q_i))

    # Compute the total energy.
    e_t_model = self.evaluate(self.water.total_energy(e_model, u, v, w, height))

    # Compute the virtual potential temperature.
    theta = self.evaluate(
        self.water.potential_temperatures(t, q_t, rho, height))

    with self.subTest(name='TemperatureAndTotalEnergy'):
      self.assertAllClose(e_t_val, e_t_model[0], atol=10, rtol=1e-3)

    for varname in ('theta', 'theta_v', 'theta_li'):
      with self.subTest(name=f'T&{varname}'):
        # Compute the temperature from the potential temperature.
        t_model = self.evaluate(
            self.water.potential_temperature_to_temperature(
                varname, [tf.constant(theta[varname][0], dtype=tf.float32)],
                q_t, q_l, q_i, height))

        expected = self.evaluate(t)
        self.assertAllClose(expected, t_model)


if __name__ == '__main__':
  tf.test.main()
