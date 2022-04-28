# coding=utf-8
"""A library of thermodynamics to be used in fluid dynamics simulations."""

from typing import List, Optional, Sequence
from swirl_lm.numerics import root_finder
from swirl_lm.physics import constants
from swirl_lm.physics.thermodynamics import thermodynamics_generic
import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config

_TF_DTYPE = thermodynamics_generic.TF_DTYPE

FlowFieldVar = thermodynamics_generic.FlowFieldVar
FlowFieldMap = thermodynamics_generic.FlowFieldMap

# A small number that is used as a tolerance for the internal energy around the
# freezing point.
_EPS_E_INT = 1e-6
# The precomputed gas constant for dry air, in units of J/kg/K.
_R_D = constants.R_D
# Molecular weights for the dry air, kg/m^3.
_W_D = 0.029
# Molecular weights for the water vapor, kg/m^3.
_W_V = 0.018
# The gravitational acceleration constant, in units of N/kg.
_G = 9.81


class Water(thermodynamics_generic.ThermodynamicModel):
  """A library of thermodynamics for water."""

  def __init__(
      self,
      params: incompressible_structured_mesh_config
      .IncompressibleNavierStokesParameters,
  ):
    """Initializes parameters for the water thermodynamics."""
    super(Water, self).__init__(params)

    model_params = params.thermodynamics
    self._r_v = model_params.water.r_v
    self._t_0 = model_params.water.t_0
    self._t_min = model_params.water.t_min
    self._t_freeze = model_params.water.t_freeze
    self._t_triple = model_params.water.t_triple
    self._p_triple = model_params.water.p_triple
    self._e_int_v0 = model_params.water.e_int_v0
    self._e_int_i0 = model_params.water.e_int_i0
    self._lh_v0 = model_params.water.lh_v0
    self._lh_s0 = model_params.water.lh_s0
    self._cv_d = model_params.water.cv_d
    self._cv_v = model_params.water.cv_v
    self._cv_l = model_params.water.cv_l
    self._cv_i = model_params.water.cv_i
    self._cp_v = model_params.water.cp_v
    self._cp_l = model_params.water.cp_l
    self._cp_i = model_params.water.cp_i
    self._p_thermal = params.p_thermal

    self._t_max_iter = model_params.water.max_temperature_iterations
    self._rho_n_iter = model_params.water.num_density_iterations
    self._f_temperature_atol_and_rtol = model_params.water.temperature_tolerance
    self._temperature_atol_and_rtol = (
        model_params.water.temperature_successive_tol
        if model_params.water.HasField('temperature_successive_tol') else
        self._f_temperature_atol_and_rtol)

    # Get parameters for the reference state.
    self._ref_state_type = model_params.water.WhichOneof('reference_state')
    if self._ref_state_type == 'geo_static_reference_state':
      self._ref_state = model_params.water.geo_static_reference_state
    elif self._ref_state_type == 'const_theta_reference_state':
      self._ref_state = model_params.water.const_theta_reference_state
    elif self._ref_state_type == 'const_reference_state':
      self._ref_state = model_params.water.const_reference_state
    else:
      raise ValueError('Unsupported reference state: {}'.format(
          self._ref_state_type))

  @property
  def cp_d(self):
    """The isobaric specific heat.

    cp,d = cv,d + R,d
    """
    return self._cv_d + _R_D

  @property
  def r_v(self):
    """The gas constant of water vapor."""
    return self._r_v

  def cv_m(
      self,
      q_tot: FlowFieldVar,
      q_liq: FlowFieldVar,
      q_ice: FlowFieldVar,
  ) -> FlowFieldVar:
    """Computes the isovolumetric specific heat capacity of moist air.

    cvₘ = cv,d + (cvᵥ - cv,d) qₜ + (cvₗ - cvᵥ) qₗ + (cvᵢ - cvᵥ)
    qᵢ

    Args:
      q_tot: The total specific humidity.
      q_liq: The liquid phase specific humidity.
      q_ice: The solid phase specific humidity.

    Returns:
      The isovolumetric specific heat capacity of moist air.
    """
    return [
        self._cv_d + (self._cv_v - self._cv_d) * q_tot_i +
        (self._cv_l - self._cv_v) * q_liq_i +
        (self._cv_i - self._cv_v) * q_ice_i
        for q_tot_i, q_liq_i, q_ice_i in zip(q_tot, q_liq, q_ice)
    ]

  def cp_m(
      self,
      q_tot: FlowFieldVar,
      q_liq: FlowFieldVar,
      q_ice: FlowFieldVar,
  ) -> FlowFieldVar:
    """Computes the isobaric specific heat capacity of moist air.

    cpₘ = (1 - qₜ) cp,d + (qₜ - qₗ - qᵢ) cpᵥ

    Args:
      q_tot: The total specific humidity.
      q_liq: The liquid phase specific humidity.
      q_ice: The solid phase specific humidity.

    Returns:
      The isobaric specific heat capacity of moist air.
    """
    return [
        (1 - q_tot_i) * self.cp_d + (q_tot_i - q_liq_i - q_ice_i) * self._cp_v
        for q_tot_i, q_liq_i, q_ice_i in zip(q_tot, q_liq, q_ice)
    ]

  def r_m(
      self,
      temperature: FlowFieldVar,
      rho: FlowFieldVar,
      q_tot: FlowFieldVar,
  ) -> FlowFieldVar:
    """Computes the gas constant for moist air.

    Rₘ = R,d [1 + (𝜀 - 1) qₜ - 𝜀 qc],
    where 𝜀 = Rᵥ / R,d = 1.61.

    Args:
      temperature: The temeprature of the flow field.
      rho: The density of the moist air.
      q_tot: The total specific humidity.

    Returns:
      The gas constant for moist air.
    """
    q_c = self.saturation_excess(temperature, rho, q_tot)

    return self.r_mix(q_tot, q_c)

  def r_mix(self, q_tot: FlowFieldVar, q_c: FlowFieldVar,) -> FlowFieldVar:
    """Computes the gas constant for moist air.

    Rₘ = R,d [1 + (𝜀 - 1) qₜ - 𝜀 qc],
    where 𝜀 = Rᵥ / R,d = 1.61.

    Args:
      q_tot: The total specific humidity.
      q_c: The condensed phase specific humidity.

    Returns:
      The gas constant for moist air.
    """
    eps = self._r_v / _R_D

    return [
        _R_D * (1.0 + (eps - 1.0) * q_tot_i - eps * q_c_i)
        for q_tot_i, q_c_i in zip(q_tot, q_c)
    ]

  def p_ref(
      self,
      zz: FlowFieldVar,
  ) -> FlowFieldVar:
    """Computes the reference pressure considering the geopotential.

    Assuming the virtual temperature profile takes the form:
    T = T_s - 𝚫T tanh(z / Hₜ),
    the hydrostatic pressure is derived from the ideal gas law as:
    p(z) = ps exp(-(z + Hₜ𝚫T'[ln(1 - 𝚫T'tanh(z / Hₜ)) -
        ln(1 + tanh(z / Hₜ)) + z / Hₜ]) / [Hₛ (1 - 𝚫T'²)]),
    where:
    T is the virtual temperature, which is the equivalent temperature assuming
      the air is dry,
    Hₛ = Rd Ts / g is the density scale height at the surface,
    𝚫T' = 𝚫T / Ts is the fractional temperature drop.

    Reference:
    CliMa design doc, p 50, Eq. 7.3.

    Args:
      zz: The geopotential height.

    Returns:
      The reference pressure as a function of height.
    """
    def pressure_with_geo_static(z: tf.Tensor) -> tf.Tensor:
      """Computes the reference pressure."""
      # Compute the fractional temperature drop.
      delta_t_frac = self._ref_state.delta_t / self._ref_state.t_s

      # Compute the density scale height at the surface.
      h_sfc = _R_D * self._ref_state.t_s / _G

      return self._p_thermal * tf.math.exp(
          -(z + self._ref_state.height * delta_t_frac *
            (tf.math.log(1.0 - delta_t_frac *
                         tf.math.tanh(z / self._ref_state.height)) -
             tf.math.log(1.0 + tf.math.tanh(z / self._ref_state.height)) +
             z / self._ref_state.height)) / h_sfc / (1.0 - delta_t_frac**2))

    def pressure_with_const_theta(z: tf.Tensor) -> tf.Tensor:
      """Computes the reference pressure for constant potential temperature."""
      return (self._p_thermal *
              (1.0 - _G * z / self.cp_d / self._ref_state.theta)
              **(self.cp_d / _R_D))

    def pressure_with_constant(z: tf.Tensor) -> tf.Tensor:
      """Returns pressure at ground level regardless of height."""
      return self._p_thermal * tf.ones_like(z, dtype=z.dtype)

    if self._ref_state_type == 'geo_static_reference_state':
      pressure_fn = pressure_with_geo_static
    elif self._ref_state_type == 'const_theta_reference_state':
      pressure_fn = pressure_with_const_theta
    elif self._ref_state_type == 'const_reference_state':
      pressure_fn = pressure_with_constant
    else:
      raise ValueError('Unsupported reference state for pressure: {}'.format(
          self._ref_state_type))

    return [pressure_fn(zz_i) for zz_i in zz]

  def t_ref(self, zz: Optional[FlowFieldVar] = None) -> FlowFieldVar:
    """Generates the reference temperature considering the geopotential.

    The virtual temperature profile is assumed to take the form if the potential
    temperature is not a constant:
    T = T_s - 𝚫T tanh(z / Hₜ),
    otherwise it's computed from the isentropic relationship.

    Args:
      zz: The geopotential height.

    Returns:
      The reference temperature as a function of height.
    """

    def temperature_with_geo_static() -> FlowFieldVar:
      """Computes the reference temperature following the presumed profile."""
      return [
          self._ref_state.t_s -
          self._ref_state.delta_t * tf.math.tanh(z / self._ref_state.height)
          for z in zz
      ]

    def temperature_with_const_theta() -> FlowFieldVar:
      """Computes reference temperature for constant potential temperature."""
      return self.virtual_potential_temperature_to_temperature(
          [self._ref_state.theta] * len(zz), zz)

    def temperature_with_constant() -> FlowFieldVar:
      """Provides a constant temperature as the reference state."""
      return [
          self._ref_state.t_ref * tf.ones_like(z, dtype=z.dtype) for z in zz
      ]

    if self._ref_state_type == 'geo_static_reference_state':
      temperature = temperature_with_geo_static()
    elif self._ref_state_type == 'const_theta_reference_state':
      temperature = temperature_with_const_theta()
    elif self._ref_state_type == 'const_reference_state':
      temperature = temperature_with_constant()
    else:
      raise ValueError('Unsupported reference state for temperature: {}'.format(
          self._ref_state_type))

    return temperature

  def rho_ref(
      self,
      zz: Optional[FlowFieldVar] = None,
  ) -> FlowFieldVar:
    """Generates the reference density considering the geopotential.

    Args:
      zz: The geopotential height.

    Returns:
      The reference density as a function of height.
    """
    return [
        p_ref / _R_D / t_ref
        for p_ref, t_ref in zip(self.p_ref(zz), self.t_ref(zz))
    ]

  def air_temperature(
      self,
      e_int: FlowFieldVar,
      q_tot: FlowFieldVar,
      q_liq: FlowFieldVar,
      q_ice: FlowFieldVar,
  ) -> FlowFieldVar:
    """Computes the air temperature.

    T = T₀ + (e - (qₜ - qₗ) eᵥ₀ + qᵢ (eᵥ₀ + eᵢ₀)) / cvₘ

    Args:
      e_int: The specific internal energy.
      q_tot: The total specific humidity.
      q_liq: The liquid phase specific humidity.
      q_ice: The solid phase specific humidity.

    Returns:
      The air temeprature.
    """
    var_list = zip(e_int, q_tot, q_liq, q_ice, self.cv_m(q_tot, q_liq, q_ice))
    return [
        self._t_0 + (e_int_i - (q_tot_i - q_liq_i) * self._e_int_v0 + q_ice_i *
                     (self._e_int_v0 + self._e_int_i0)) / cv_m_i
        for e_int_i, q_tot_i, q_liq_i, q_ice_i, cv_m_i in var_list
    ]

  def _saturation_vapor_pressure(
      self,
      temperature: tf.Tensor,
      lh_0: tf.Tensor,
      d_cp: tf.Tensor,
  ) -> tf.Tensor:
    """Computes the saturation vapor pressure over a plane surface.

    pₛₐₜ = pₜᵣᵢₚₗₑ (T / Tₜᵣᵢₚₗₑ)^(𝚫cp / Rv)
      exp((LH₀ - 𝚫cp T₀) / Rv (1 / Tₜᵣᵢₚₗₑ - 1 / T))

    Args:
      temperature: The temperature of the flow field.
      lh_0: The latent heat at reference state. lh_0 = self._lh_v0 (by default)
        for the saturation vapor pressure over a plane liquid surface; lh_0 =
        self._lh_s0 for the saturation vapor pressure over a plane ice surface.
      d_cp: The difference between specific heat of vapor and liquid at isobaric
        condition.

    Returns:
      The vapor pressure at saturation condition.
    """
    return self._p_triple * tf.math.pow(
        temperature / self._t_triple, d_cp / self._r_v) * tf.math.exp(
            (lh_0 - d_cp * self._t_0) / self._r_v *
            (1.0 / self._t_triple - 1.0 / temperature))

  def saturation_vapor_pressure(
      self,
      temperature: FlowFieldVar,
      lh_0: Optional[FlowFieldVar] = None,
      d_cp: Optional[FlowFieldVar] = None,
  ) -> FlowFieldVar:
    """Computes the saturation vapor pressure over a plane surface.

    The Clausius-Clapeyron relation is used to compute the saturation vapor
    pressure, which is:
      dlog(pᵥ_sat) / dT = L/ (Rᵥ T²).
    L is the specific laten heat with constant isobaric specific heats of the
    phase, which is represented by the Kirchholf's relation as:
      L = LH₀ + Δcp (T - T₀).
    Note that the linear dependency of L on T allows analytical integration of
    the
    Clausius-Clapeyron equation, so that the saturation vapor pressure is a
    function of the triple point pressure.

    Args:
      temperature: The temperature of the flow field.
      lh_0: The latent heat at reference state. lh_0 = self._lh_v0 (by default)
        for the saturation vapor pressure over a plane liquid surface; lh_0 =
        self._lh_s0 for the saturation vapor pressure over a plane ice surface.
      d_cp: The difference between specific heat of vapor and liquid at isobaric
        condition.

    Returns:
      The saturation vapor pressure at given conditions.
    """
    if lh_0 is None:
      lh_0 = [
          self._lh_v0,
      ] * len(temperature)
    if d_cp is None:
      d_cp = [
          self._cp_v - self._cp_l,
      ] * len(temperature)

    return [
        self._saturation_vapor_pressure(t_i, lh_0_i, d_cp_i)
        for t_i, d_cp_i, lh_0_i in zip(temperature, d_cp, lh_0)
    ]

  def saturation_q_vapor(
      self,
      temperature: FlowFieldVar,
      rho: FlowFieldVar,
      q_liq: Optional[FlowFieldVar] = None,
      q_c: Optional[FlowFieldVar] = None,
  ) -> FlowFieldVar:
    """Computes the saturation specific humidity.

    qᵥ = pₛₐₜ / (ϱ Rᵥ T)

    Args:
      temperature: The temperature of the flow field.
      rho: The density of the moist air.
      q_liq: The specific humidity of the liquid phase.
      q_c: The specific humidity of the condensed phase.

    Returns:
      The saturation specific humidity.
    """
    liquid_frac = self.liquid_fraction(temperature, q_liq, q_c)
    ice_frac = [1.0 - liquid_frac_i for liquid_frac_i in liquid_frac]
    lh_0 = [
        liquid_frac_i * self._lh_v0 + ice_frac_i * self._lh_s0
        for liquid_frac_i, ice_frac_i in zip(liquid_frac, ice_frac)
    ]
    d_cp = [
        liquid_frac_i * (self._cp_v - self._cp_l) + ice_frac_i *
        (self._cp_v - self._cp_i)
        for liquid_frac_i, ice_frac_i in zip(liquid_frac, ice_frac)
    ]
    p_v_sat = self.saturation_vapor_pressure(temperature, lh_0, d_cp)

    return [
        p_v_sat_i / (rho_i * self._r_v * t_i)
        for p_v_sat_i, rho_i, t_i in zip(p_v_sat, rho, temperature)
    ]

  def saturation_excess(
      self,
      temperature: FlowFieldVar,
      rho: FlowFieldVar,
      q_tot: FlowFieldVar,
      q_liq: Optional[FlowFieldVar] = None,
      q_c: Optional[FlowFieldVar] = None,
  ) -> FlowFieldVar:
    """Computes the saturation excess in equilibrium.

    qₑₓ = max(qₜ - qᵥ, 0)

    Args:
      temperature: The temperature of the flow field.
      rho: The density of the moist air.
      q_tot: The total specific humidity.
      q_liq: The specific humidity of the liquid phase.
      q_c: The specific humidity of the condensed phase.

    Returns:
      The saturation excess in equilibrium.
    """
    q_vap_sat = self.saturation_q_vapor(temperature, rho, q_liq, q_c)
    return [
        tf.maximum(0.0, q_tot_i - q_vap_sat_i)
        for q_tot_i, q_vap_sat_i in zip(q_tot, q_vap_sat)
    ]

  def liquid_fraction(
      self,
      temperature: FlowFieldVar,
      q_liq: Optional[FlowFieldVar] = None,
      q_c: Optional[FlowFieldVar] = None,
  ) -> FlowFieldVar:
    """Computes the fraction of liquid in the condensed phase.

    fₗ = qₗ / qc

    Args:
      temperature: The temperature of the flow field.
      q_liq: The specific humidity of the liquid phase.
      q_c: The specific humidity of the condensed phase.

    Returns:
      The fraction of liquid phase over the condensed phase.
    """
    liquid_frac_no_condensate = [
        tf.where(
            tf.greater(t_i, self._t_freeze), tf.ones_like(t_i, dtype=_TF_DTYPE),
            tf.zeros_like(t_i, dtype=_TF_DTYPE)) for t_i in temperature
    ]
    if q_liq is None or q_c is None:
      return liquid_frac_no_condensate

    return [
        tf.where(tf.greater(q_c_i, 0.0), q_liq_i / q_c_i,
                 liquid_frac_i) for q_c_i, q_liq_i, liquid_frac_i in zip(
                     q_c, q_liq, liquid_frac_no_condensate)
    ]

  def equilibrium_phase_partition(
      self,
      temperature: FlowFieldVar,
      rho: FlowFieldVar,
      q_tot: FlowFieldVar,
  ) -> Sequence[FlowFieldVar]:
    """Partitions the water phases in equilibrium.

    Args:
      temperature: The temeprature of the flow field.
      rho: The density of the moist air.
      q_tot: The total specific humidity.

    Returns:
      The specific humidity of the liquid and ice phase.
    """
    liquid_frac = self.liquid_fraction(temperature)
    q_c = self.saturation_excess(temperature, rho, q_tot)
    q_liq = [
        liquid_frac_i * q_c_i for liquid_frac_i, q_c_i in zip(liquid_frac, q_c)
    ]
    q_ice = [(1.0 - liquid_frac_i) * q_c_i
             for liquid_frac_i, q_c_i in zip(liquid_frac, q_c)]
    return q_liq, q_ice

  def internal_energy(
      self,
      temperature: FlowFieldVar,
      q_tot: FlowFieldVar,
      q_liq: FlowFieldVar,
      q_ice: FlowFieldVar,
  ) -> FlowFieldVar:
    """Computes the specific internal energy.

    e = cvₘ (T - T₀) + (qₜ - qₗ) eᵥ₀ - qᵢ (eᵥ₀ + eᵢ₀)

    Args:
      temperature: The temperature of the flow field.
      q_tot: The total specific humidity.
      q_liq: The specific humidity of the liquid phase.
      q_ice: The specific humidity of the solid phase.

    Returns:
      The specific internal energy at the given temperature and humidity
      condition.
    """
    var_list = zip(
        self.cv_m(q_tot, q_liq, q_ice), temperature, q_tot, q_liq, q_ice)
    return [
        cv_m_i * (t_i - self._t_0) + (q_tot_i - q_liq_i) * self._e_int_v0 -
        q_ice_i * (self._e_int_v0 + self._e_int_i0)
        for cv_m_i, t_i, q_tot_i, q_liq_i, q_ice_i in var_list
    ]

  def internal_energy_from_total_energy(
      self,
      e_t: FlowFieldVar,
      u: FlowFieldVar,
      v: FlowFieldVar,
      w: FlowFieldVar,
      zz: Optional[FlowFieldVar] = None,
  ) -> FlowFieldVar:
    """Computes the specific internal energy of the flow field.

    The total energy is the sum of internal energy, kinetic energy, and the geo-
    potential energy.

    Args:
      e_t: The specific total energy, in units of J/(kg/m^3).
      u: The velocity component in the x direction, in units of m/s.
      v: The velocity component in the y direction, in units of m/s.
      w: The velocity component in the z direction, in units of m/s.
      zz: The cooridinates in the vertical direction, in units of m.

    Returns:
      The specific internal energy.
    """
    zz = [tf.zeros_like(e_t_i, dtype=e_t_i.dtype) for e_t_i in e_t
         ] if zz is None else zz
    ke = [0.5 * (u_i**2 + v_i**2 + w_i**2) for u_i, v_i, w_i in zip(u, v, w)]
    pe = [_G * zz_i for zz_i in zz]
    return [e_t_i - ke_i - pe_i for e_t_i, ke_i, pe_i in zip(e_t, ke, pe)]

  def total_energy(
      self,
      e: FlowFieldVar,
      u: FlowFieldVar,
      v: FlowFieldVar,
      w: FlowFieldVar,
      zz: Optional[FlowFieldVar] = None,
  ) -> FlowFieldVar:
    """Computes the specific total energy of the flow field.

    The total energy is the sum of internal energy, kinetic energy, and the geo-
    potential energy.

    Args:
      e: The specific internal energy, in units of J/(kg/m^3).
      u: The velocity component in the x direction, in units of m/s.
      v: The velocity component in the y direction, in units of m/s.
      w: The velocity component in the z direction, in units of m/s.
      zz: The cooridinates in the vertical direction, in units of m.

    Returns:
      The specific total energy.
    """
    zz = [tf.zeros_like(e_i, dtype=e_i.dtype) for e_i in e
         ] if zz is None else zz
    ke = [0.5 * (u_i**2 + v_i**2 + w_i**2) for u_i, v_i, w_i in zip(u, v, w)]
    pe = [_G * zz_i for zz_i in zz]

    # Note, we are doing 1.0 * (e_i + ke_i) + pe_i here to work around a
    # non-deterministic issue. See b/221776082.
    return [1.0 * (e_i + ke_i) + pe_i for e_i, ke_i, pe_i in zip(e, ke, pe)]

  def total_enthalpy(
      self,
      e_tot: FlowFieldVar,
      rho: FlowFieldVar,
      q_tot: FlowFieldVar,
      temperature: FlowFieldVar,
  ) -> FlowFieldVar:
    """Computes the total enthalpy.

    hₜ = eₜ + Rₘ T

    Args:
      e_tot: The total energy, in units of J/(kg/m^3).
      rho: The moist air density, in units of kg/m^3.
      q_tot: The total specific humidity.
      temperature: The temperature obtained from saturation adjustment that's
        consistent with the other 3 input parameters.

    Returns:
      The total enthalpy, in units of J/(kg/m^3).
    """
    r_m = self.r_m(temperature, rho, q_tot)
    return [
        e_tot_i + r_m_i * t_i
        for e_tot_i, r_m_i, t_i in zip(e_tot, r_m, temperature)
    ]

  def saturation_internal_energy(
      self,
      temperature: FlowFieldVar,
      rho: FlowFieldVar,
      q_tot: FlowFieldVar,
  ) -> FlowFieldVar:
    """Computes the specific internal energy at saturation.

    Args:
      temperature: The temperature of the flow field.
      rho: The moist air density.
      q_tot: The total specific humidity.

    Returns:
      The internal energy per unit mass in thermodynamic equilibrium at
      saturation
      condition.
    """
    q_liq, q_ice = self.equilibrium_phase_partition(temperature, rho, q_tot)
    return self.internal_energy(temperature, q_tot, q_liq, q_ice)

  def de_int_dt(
      self,
      temperature: FlowFieldVar,
      rho: FlowFieldVar,
      q_tot: FlowFieldVar,
  ) -> FlowFieldVar:
    """Computes the analytical Jacobian of internal energy wrt temperature.

    Args:
      temperature: The temperature of the flow field.
      rho: The density of moist air.
      q_tot: The total specific humidity.

    Returns:
      The partial derivative of the internal energy wrt temperature.
    """
    q_liq, q_ice = self.equilibrium_phase_partition(temperature, rho, q_tot)
    q_c = [q_liq_i + q_ice_i for q_liq_i, q_ice_i in zip(q_liq, q_ice)]
    cv_mix = self.cv_m(q_tot, q_liq, q_ice)
    q_vap_sat = self.saturation_q_vapor(temperature, rho, q_liq, q_c)
    liquid_frac = self.liquid_fraction(temperature, q_liq, q_c)
    l = [
        liquid_frac_i * self._lh_v0 + (1.0 - liquid_frac_i) * self._lh_s0
        for liquid_frac_i in liquid_frac
    ]
    dq_vap_sat_dt = [
        q_vap_sat_i * l_i / (self._r_v * t_i**2)
        for q_vap_sat_i, l_i, t_i in zip(q_vap_sat, l, temperature)
    ]
    dcvm_dq_vap = [
        self._cv_v - liquid_frac_i * self._cv_l -
        (1.0 - liquid_frac_i) * self._cv_i for liquid_frac_i in liquid_frac
    ]
    var_list = zip(cv_mix, liquid_frac, temperature, dcvm_dq_vap, dq_vap_sat_dt)
    return [
        cv_m_i + (self._e_int_v0 + (1 - liquid_frac_i) * self._e_int_i0 +
                  (t_i - self._t_0) * dcvm_dq_vap_i) * dq_vap_sat_dt_i for
        cv_m_i, liquid_frac_i, t_i, dcvm_dq_vap_i, dq_vap_sat_dt_i in var_list
    ]

  def saturation_temperature(
      self,
      t_guess: FlowFieldVar,
      e_int: FlowFieldVar,
      rho: FlowFieldVar,
      q_tot: FlowFieldVar,
  ) -> FlowFieldVar:
    """Computes the temperature assuming water is at saturation."""

    def internal_energy_error_fn(temperature: FlowFieldVar) -> FlowFieldVar:
      """Computes the error of internal energy for the Newton iterations."""
      e_int_sat = self.saturation_internal_energy(temperature, rho, q_tot)
      return [
          e_int_sat_i - e_int_i
          for e_int_sat_i, e_int_i in zip(e_int_sat, e_int)
      ]

    jacobian_fn = lambda temperature: self.de_int_dt(temperature, rho, q_tot)

    t_sat = root_finder.newton_method(
        internal_energy_error_fn,
        t_guess,
        self._t_max_iter,
        position_tolerance=self._f_temperature_atol_and_rtol,
        value_tolerance=self._f_temperature_atol_and_rtol,
        analytical_jacobian_fn=jacobian_fn,
    )

    return t_sat if isinstance(t_sat, List) else tf.unstack(t_sat)

  def saturation_adjustment(
      self,
      e_int: FlowFieldVar,
      rho: FlowFieldVar,
      q_tot: FlowFieldVar,
  ) -> FlowFieldVar:
    """Computes the temperature that is consistent with the input state.

    Args:
      e_int: The internal energy.
      rho: The density of moist air.
      q_tot: The total specific humidity.

    Returns:
      The temperature at the given state.
    """
    # Case 1: temperature at unsaturated condition.
    q_liq = [tf.zeros_like(q_tot_i, dtype=_TF_DTYPE) for q_tot_i in q_tot]
    q_ice = [tf.zeros_like(q_tot_i, dtype=_TF_DTYPE) for q_tot_i in q_tot]
    t_1 = [
        tf.maximum(self._t_min, t_air)
        for t_air in self.air_temperature(e_int, q_tot, q_liq, q_ice)
    ]
    q_v_sat = self.saturation_q_vapor(t_1, rho)

    # Case 2: temperature at freezing point.
    t_freeze = [
        self._t_freeze * tf.ones_like(e_int_i, dtype=_TF_DTYPE)
        for e_int_i in e_int
    ]
    e_int_freeze = self.internal_energy(t_freeze, q_tot, q_liq, q_ice)

    # Case 3: temperature at saturation condition.
    t_sat = self.saturation_temperature(t_1, e_int, rho, q_tot)

    # Get temperature under correct conditions.
    def unsaturation_cond(
        q_tot_i: tf.Tensor,
        q_v_sat_i: tf.Tensor,
        t_1_i: tf.Tensor,
    ) -> tf.Tensor:
      """Determines if the fluid is unsaturated."""
      return tf.math.logical_and(
          tf.less_equal(q_tot_i, q_v_sat_i), tf.greater(t_1_i, self._t_min))

    t = [
        tf.where(unsaturation_cond(q_tot_i, q_v_sat_i, t_1_i), t_1_i, t_sat_i)
        for q_tot_i, q_v_sat_i, t_1_i, t_sat_i in zip(q_tot, q_v_sat, t_1,
                                                      tf.unstack(t_sat))
    ]

    def freezing_cond(
        e_int_freeze_i: tf.Tensor,
        e_int_i: tf.Tensor,
    ) -> tf.Tensor:
      """Determines if the fluid is frozen."""
      return tf.less(tf.abs(e_int_freeze_i - e_int_i), _EPS_E_INT)

    return [
        tf.where(freezing_cond(e_int_freeze_i, e_int_i), t_freeze_i, t_i)
        for e_int_freeze_i, e_int_i, t_freeze_i, t_i in zip(
            e_int_freeze, e_int, t_freeze, t)
    ]

  def saturation_density(
      self,
      e_tot: FlowFieldVar,
      q_tot: FlowFieldVar,
      u: FlowFieldVar,
      v: FlowFieldVar,
      w: FlowFieldVar,
      rho_0: Optional[FlowFieldVar] = None,
      zz: Optional[FlowFieldVar] = None,
  ) -> FlowFieldVar:
    """Computes the density that is consistent with the input state.

    Args:
      e_tot: The total energy.
      q_tot: The total specific humidity.
      u: The velocity component in the x direction.
      v: The velocity component in the y direction.
      w: The velocity component in the z direction. the Secant solver.
      rho_0: A guess of the density, which is used as the initial condition for
      zz: The cooridinates in the vertical direction.

    Returns:
      The density at the given state.
    """
    e_int = self.internal_energy_from_total_energy(e_tot, u, v, w, zz)
    p = self.p_ref(zz) if zz is not None else [
        self._p_thermal * tf.ones_like(e_i, dtype=e_i.dtype) for e_i in e_tot
    ]

    def density_update_fn(rho):
      """Updates the density for one iteration."""
      temperature_sat = self.saturation_adjustment(e_int, rho, q_tot)
      r_mix = self.r_m(temperature_sat, rho, q_tot)
      return [
          p_i / t_sat / r_m
          for p_i, t_sat, r_m in zip(p, temperature_sat, r_mix)
      ]

    if rho_0 is None:
      rho_0 = [tf.ones_like(e_i, dtype=e_i.dtype) for e_i in e_int]

    i0 = tf.constant(0)
    cond = lambda i, rho: tf.less(i, self._rho_n_iter)
    body = lambda i, rho: (i + 1, density_update_fn(rho))
    _, rho = tf.while_loop(
        cond=cond, body=body, loop_vars=(i0, rho_0), back_prop=False)

    return rho

  def potential_temperatures(
      self,
      t: FlowFieldVar,
      q_t: FlowFieldVar,
      rho: FlowFieldVar,
      zz: Optional[FlowFieldVar] = None,
  ) -> FlowFieldMap:
    """Computes the liquid water potential temperatures and virtual temperature.

    Args:
      t: The temperature, in units of K.
      q_t: The total humidty.
      rho: The density of the moist air, in units of kg/m^3.
      zz: The vertical coordinates.

    Returns:
      A dictionary of the two potential temperatures.
    """
    zz = zz if zz is not None else [
        tf.zeros_like(t_i, dtype=t_i.dtype) for t_i in t
    ]

    q_l, q_i = self.equilibrium_phase_partition(t, rho, q_t)

    r_m = self.r_m(t, rho, q_t)
    cp_m = self.cp_m(q_t, q_l, q_i)

    p_ref = self.p_ref(zz)
    pi = [(p_ref_i / self._p_thermal)**(r_m_i / cp_m_i)
          for p_ref_i, r_m_i, cp_m_i in zip(p_ref, r_m, cp_m)]

    t_l = [(t_i - (self._lh_v0 * q_l_i + self._lh_s0 * q_i_i) / cp_m_i) / pi_i
           for t_i, q_l_i, q_i_i, cp_m_i, pi_i in zip(t, q_l, q_i, cp_m, pi)]

    t_v = [r_m_i / _R_D * t_i / pi_i for r_m_i, t_i, pi_i in zip(r_m, t, pi)]

    return {'theta_l': t_l, 'theta_v': t_v}

  def virtual_potential_temperature_to_temperature(
      self,
      theta_v: FlowFieldVar,
      zz: FlowFieldVar,
  ) -> FlowFieldVar:
    """Computes temperature from virtual potential temperature."""
    p = self.p_ref(zz)
    return [
        theta_v_i * (p_i / self._p_thermal)**(_R_D / self.cp_d)
        for theta_v_i, p_i in zip(theta_v, p)
    ]

  def update_density(self, states: FlowFieldMap,
                     additional_states: FlowFieldMap) -> FlowFieldVar:
    """Updates the density of the flow field with water thermodynamics."""
    zz = additional_states['zz'] if 'zz' in additional_states.keys() else None
    return self.saturation_density(states['e_t'], states['q_t'], states['u'],
                                   states['v'], states['w'], states['rho'], zz)

  def update_temperatures(
      self,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Computes the temperature and potential temperatures.

    Args:
      states: Flow field variables, must contain 'rho', 'u', 'v', 'w', 'e_t',
        and 'q_t'.
      additional_states: Helper variables that are required to compute potential
        temperatures. If field 'zz' is not in `additional_states`, assumes the
        flow field is independent of height, i.e. `zz` = 0.

    Returns:
      A dictionary of flow fields that contains, the temperature 'T', the liquid
      potential temperature 'T_l', and the virtual potential temperature 'T_v'.
    """
    zz = additional_states['zz'] if 'zz' in additional_states.keys() else [
        tf.zeros_like(e_i, dtype=e_i.dtype) for e_i in states['e_t']
    ]
    e = self.internal_energy_from_total_energy(states['e_t'], states['u'],
                                               states['v'], states['w'],
                                               zz)

    temperatures = {
        'T': self.saturation_adjustment(e, states['rho'], states['q_t'])
    }
    temperatures.update(
        self.potential_temperatures(temperatures['T'], states['q_t'],
                                    states['rho'], zz))

    return temperatures
