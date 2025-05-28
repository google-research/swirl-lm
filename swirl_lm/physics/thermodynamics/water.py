# Copyright 2025 The swirl_lm Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
"""A library of thermodynamics to be used in fluid dynamics simulations.

This library supports the following pairs of prognostic variables:
1. 'e_t' (total energy) and 'q_t' (total humidity)
2. 'theta_li' (liquid-ice potential temperature) and 'q_t' (total humidity)
3. 'theta' (potential temperature) and 'q_t' (total humidity)
4. 'theta_v' (virtual potential temperature) and 'q_t' (total humidity)

Reference: CLIMA Atmosphere Model.
"""

from collections.abc import Callable
import enum
from typing import Optional, Sequence, Text, cast

from absl import logging
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.numerics import root_finder
from swirl_lm.physics import constants
from swirl_lm.physics.thermodynamics import thermodynamics_generic
from swirl_lm.physics.thermodynamics import thermodynamics_pb2
from swirl_lm.utility import get_kernel_fn
import tensorflow as tf

FlowFieldVal = thermodynamics_generic.FlowFieldVal
FlowFieldMap = thermodynamics_generic.FlowFieldMap

_TF_DTYPE = thermodynamics_generic.TF_DTYPE

# A small number that is used as a tolerance for the internal energy around the
# freezing point.
_EPS_E_INT = 1e-6
# The precomputed gas constant for dry air, in units of J/kg/K.
_R_D = constants.R_D
# Molecular weights for the dry air, kg/m^3.
_W_D = 0.029
# Molecular weights for the water vapor, kg/m^3.
_W_V = 0.018


class PotentialTemperature(enum.Enum):
  """Defines the name of a potential temperature."""
  # The potential temperature of the humid air.
  THETA = 'theta'
  # The virtual potential temperature.
  THETA_V = 'theta_v'
  # The liquid-ice potential temperature.
  THETA_LI = 'theta_li'


class Water(thermodynamics_generic.ThermodynamicModel):
  """A library of thermodynamics for water."""

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    """Initializes parameters for the water thermodynamics."""
    super(Water, self).__init__(params)

    assert (
        model_params := params.thermodynamics
    ) is not None, 'Thermodynamics model is not defined.'
    model_type = model_params.WhichOneof('thermodynamics_type')
    assert model_type == 'water', (
        '`Water` requires the thermodynamics model to be of type `water` but '
        f' {model_type} is provided.'
    )

    self._solver_mode = params.solver_mode

    self._r_v = model_params.water.r_v
    self._t_0 = model_params.water.t_0
    self._t_min = model_params.water.t_min
    self._t_freeze = model_params.water.t_freeze
    self._t_triple = model_params.water.t_triple
    self._t_icenuc = model_params.water.t_icenuc
    self._p_triple = model_params.water.p_triple
    self._p00 = model_params.water.p00
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
    self._use_fast_thermodynamics = model_params.water.use_fast_thermodynamics
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
    elif self._ref_state_type == 'user_defined_reference_state':
      self._ref_state = model_params.water.user_defined_reference_state
    else:
      raise ValueError('Unsupported reference state: {}'.format(
          self._ref_state_type))

  @property
  def t_freeze(self):
    """The freezing point of water."""
    return self._t_freeze

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

  @property
  def lh_v0(self):
    """The latent heat of vaporization."""
    return self._lh_v0

  def lh_v(self, temperature: FlowFieldVal) -> FlowFieldVal:
    """Computes the latent heat of vaporization at `temperature`.

    Args:
      temperature: The temperature of the flow field [K].

    Returns:
      The latent heat of vaporization at the input temperature.
    """
    return tf.nest.map_structure(
        lambda t: self._lh_v0 + (self._cp_v - self._cp_l) * (t - self._t_0),
        temperature)

  @property
  def lh_s0(self):
    """The latent heat of sublimation."""
    return self._lh_s0

  def lh_s(self, temperature: FlowFieldVal) -> FlowFieldVal:
    """Computes the latent heat of sublimation at `temperature`.

    Args:
      temperature: The temperature of the flow field [K].

    Returns:
      The latent heat of sublimation at the input temperature.
    """
    return tf.nest.map_structure(
        lambda t: self._lh_s0 + (self._cp_v - self._cp_i) * (t - self._t_0),
        temperature)

  def lh_f(self, temperature: FlowFieldVal) -> FlowFieldVal:
    """Computes the latent heat of freezing/fusion at `temperature`.

    Args:
      temperature: The temperature of the flow field [K].

    Returns:
      The latent heat of freezing/fusion at the input temperature.
    """
    lh_f0 = self._lh_s0 - self._lh_v0
    return lh_f0 + (self._cp_l - self._cp_i) * (temperature - self._t_0)

  def humidity_to_volume_mixing_ratio(
      self,
      q_t: FlowFieldVal,
      q_c: FlowFieldVal,
  ) -> FlowFieldVal:
    """Computes the water vapor volume mixing ratio from specific humidities.

    Args:
      q_t: The total specific humidity.
      q_c: The condensed phase specific humidity.

    Returns:
      A field containing the volume mixing ratio of water vapor.
    """
    mol_ratio = self.r_v / constants.R_D

    def vmr_fn(q_t, q_c):
      q_v = q_t - q_c
      mix_ratio = q_v / (1.0 - q_t)
      return mol_ratio * mix_ratio

    return tf.nest.map_structure(vmr_fn, q_t, q_c)

  def air_molecules_per_area(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      p: FlowFieldVal,
      g_dim: int,
      vmr_h2o: Optional[FlowFieldVal] = None,
  ) -> FlowFieldVal:
    """Computes the number of molecules in an atmospheric grid cell per area.

    The computation assumes the atmosphere to be in hydrostatic equilibrium.

    Args:
      kernel_op: An object holding a library of kernel operations.
      p: The hydrostatic pressure variable.
      g_dim: The direction of gravity.
      vmr_h2o: The volume mixing ratio of water vapor.

    Returns:
      A field containing the number of molecules of atmospheric gases per area
        [molecules/m^2].
    """
    grad_central = [
        lambda f: kernel_op.apply_kernel_op_x(f, 'kDx'),
        lambda f: kernel_op.apply_kernel_op_y(f, 'kDy'),
        lambda f: kernel_op.apply_kernel_op_z(f, 'kDz', 'kDzsh'),
    ]
    dp = tf.nest.map_structure(lambda dp: 0.5 * dp, grad_central[g_dim](p))

    def mols_fn(dp, vmr):
      mol_m_air = constants.DRY_AIR_MOL_MASS + constants.WATER_MOL_MASS * vmr
      return -(dp / constants.G) * constants.AVOGADRO / mol_m_air

    return tf.nest.map_structure(
        mols_fn,
        dp,
        vmr_h2o,
    )

  def cv_m(
      self,
      q_tot: FlowFieldVal,
      q_liq: FlowFieldVal,
      q_ice: FlowFieldVal,
  ) -> FlowFieldVal:
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
    def cv_m_fn(q_t, q_l, q_i):
      """Computes the isovolumetric specific heat of moist air."""
      return (
          self._cv_d
          + (self._cv_v - self._cv_d) * q_t
          + (self._cv_l - self._cv_v) * q_l
          + (self._cv_i - self._cv_v) * q_i
      )

    return tf.nest.map_structure(cv_m_fn, q_tot, q_liq, q_ice)

  def cp_m(
      self,
      q_tot: FlowFieldVal,
      q_liq: FlowFieldVal,
      q_ice: FlowFieldVal,
  ) -> FlowFieldVal:
    """Computes the isobaric specific heat capacity of moist air.

    cpₘ = (1 - qₜ) cp,d + (qₜ - qₗ - qᵢ) cpᵥ

    Args:
      q_tot: The total specific humidity.
      q_liq: The liquid phase specific humidity.
      q_ice: The solid phase specific humidity.

    Returns:
      The isobaric specific heat capacity of moist air.
    """
    def cp_m_fn(q_t, q_l, q_i):
      """Computes the isobaric specific heat of moist air."""
      return (1 - q_t) * self.cp_d + (q_t - q_l - q_i) * self._cp_v

    return tf.nest.map_structure(cp_m_fn, q_tot, q_liq, q_ice)

  def r_m(
      self,
      temperature: FlowFieldVal,
      rho: FlowFieldVal,
      q_tot: FlowFieldVal,
  ) -> FlowFieldVal:
    """Computes the gas constant for moist air.

    Rₘ = R,d [1 + (𝜀 - 1) qₜ - 𝜀 q_c],
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

  def r_mix(
      self,
      q_tot: FlowFieldVal,
      q_c: FlowFieldVal,
  ) -> FlowFieldVal:
    """Computes the gas constant for moist air.

    Rₘ = R,d [1 + (𝜀 - 1) qₜ - 𝜀 q_c],
    where 𝜀 = Rᵥ / R,d = 1.61.

    Args:
      q_tot: The total specific humidity.
      q_c: The condensed phase specific humidity.

    Returns:
      The gas constant for moist air.
    """
    eps = self._r_v / _R_D

    r_mix_fn = lambda q_t, q_c: _R_D * (1.0 + (eps - 1.0) * q_t - eps * q_c)

    return tf.nest.map_structure(r_mix_fn, q_tot, q_c)

  def p_ref(
      self,
      zz: FlowFieldVal,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
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
      additional_states: Helper variables including those needed to compute the
        reference pressure.

    Returns:
      The reference pressure as a function of height.
    """
    def pressure_with_geo_static(z: tf.Tensor) -> tf.Tensor:
      """Computes the reference pressure."""
      ref_state = cast(
          thermodynamics_pb2.Water.GeoStaticReferenceState, self._ref_state
      )
      # Compute the fractional temperature drop.
      delta_t_frac = ref_state.delta_t / ref_state.t_s

      # Compute the density scale height at the surface.
      h_sfc = _R_D * ref_state.t_s / constants.G

      return self._p_thermal * tf.math.exp(
          -(z + ref_state.height * delta_t_frac *
            (tf.math.log(1.0 - delta_t_frac *
                         tf.math.tanh(z / ref_state.height)) -
             tf.math.log(1.0 + tf.math.tanh(z / ref_state.height)) +
             z / ref_state.height)) / h_sfc / (1.0 - delta_t_frac**2))

    def pressure_with_const_theta(z: tf.Tensor) -> tf.Tensor:
      """Computes the reference pressure for constant potential temperature."""
      ref_state = cast(
          thermodynamics_pb2.Water.ConstThetaReferenceState, self._ref_state
      )
      return (self._p_thermal *
              (1.0 - constants.G * z / self.cp_d / ref_state.theta)
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
    elif self._ref_state_type == 'user_defined_reference_state':
      if additional_states is None or 'p_ref' not in additional_states:
        raise ValueError(
            '`p_ref` is required in additional_states for the user defined'
            ' reference state.'
        )
      return additional_states['p_ref']
    else:
      raise ValueError('Unsupported reference state for pressure: {}'.format(
          self._ref_state_type))

    return tf.nest.map_structure(pressure_fn, zz)

  def t_ref(
      self,
      zz: Optional[FlowFieldVal] = None,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Generates the reference temperature considering the geopotential.

    The virtual temperature profile is assumed to take the form if the potential
    temperature is not a constant:
    T = T_s - 𝚫T tanh(z / Hₜ),
    otherwise it's computed from the isentropic relationship.

    Args:
      zz: The geopotential height.
      additional_states: Helper variables for computing the reference density.

    Returns:
      The reference temperature as a function of height.
    """

    def temperature_with_geo_static() -> FlowFieldVal:
      """Computes the reference temperature following the presumed profile."""
      ref_state = cast(
          thermodynamics_pb2.Water.GeoStaticReferenceState, self._ref_state
      )
      return tf.nest.map_structure(
          lambda z: ref_state.t_s  # pylint: disable=g-long-lambda
          - ref_state.delta_t * tf.math.tanh(z / ref_state.height),
          zz,
      )

    def temperature_with_const_theta() -> FlowFieldVal:
      """Computes reference temperature for constant potential temperature."""
      ref_state = cast(
          thermodynamics_pb2.Water.ConstThetaReferenceState, self._ref_state
      )
      theta = tf.nest.map_structure(
          lambda z: ref_state.theta * tf.ones_like(z), zz
      )
      q_t = tf.nest.map_structure(
          lambda z: ref_state.q_t * tf.ones_like(z), zz
      )
      q_l = tf.nest.map_structure(
          lambda z: ref_state.q_l * tf.ones_like(z), zz
      )
      q_i = tf.nest.map_structure(
          lambda z: ref_state.q_i * tf.ones_like(z), zz
      )
      # Additional states is not used here by intention because no helper
      # variables are required for this type of reference state.
      return self.potential_temperature_to_temperature(
          PotentialTemperature.THETA.value, theta, q_t, q_l, q_i, zz)

    def temperature_with_constant() -> FlowFieldVal:
      """Provides a constant temperature as the reference state."""
      ref_state = cast(
          thermodynamics_pb2.Water.ConstReferenceState, self._ref_state
      )
      return tf.nest.map_structure(
          lambda z: ref_state.t_ref * tf.ones_like(z), zz
      )

    if self._ref_state_type == 'geo_static_reference_state':
      temperature = temperature_with_geo_static()
    elif self._ref_state_type == 'const_theta_reference_state':
      temperature = temperature_with_const_theta()
    elif self._ref_state_type == 'const_reference_state':
      temperature = temperature_with_constant()
    elif self._ref_state_type == 'user_defined_reference_state':
      if additional_states is None or 'theta_ref' not in additional_states:
        raise ValueError(
            '`theta_ref` is required in additional_states to compute the'
            ' user defined reference state.'
        )
      zeros = tf.nest.map_structure(
          tf.zeros_like, additional_states['theta_ref']
      )

      if 'q_t_init' in additional_states:
        q_t = additional_states['q_t_init']
      elif ('q_c_init' in additional_states and
            'q_v_init' in additional_states):
        q_t = tf.nest.map_structure(tf.add, additional_states['q_c_init'],
                                    additional_states['q_v_init'])
      else:
        # No humidity is included, assuming it is a dry case.
        q_t = zeros

      temperature = self.potential_temperature_to_temperature(
          theta_name='theta',
          theta=additional_states['theta_ref'],
          q_tot=q_t,
          q_liq=zeros,
          q_ice=zeros,
          zz=zz,
          additional_states=additional_states,
      )
    else:
      raise ValueError('Unsupported reference state for temperature: {}'.format(
          self._ref_state_type))

    return temperature

  def rho_ref(
      self,
      zz: Optional[FlowFieldVal] = None,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Generates the reference density considering the geopotential.

    Args:
      zz: The geopotential height.
      additional_states: Helper variables for computing the reference density.

    Returns:
      The reference density as a function of height.
    """
    if additional_states is not None:
      if 'rho_ref' in additional_states:
        return additional_states['rho_ref']
      if 'q_t_init' in additional_states:
        q_t = additional_states['q_t_init']
      elif ('q_c_init' in additional_states and
            'q_v_init' in additional_states):
        q_t = tf.nest.map_structure(
            tf.add, additional_states['q_c_init'],
            additional_states['q_v_init'])
      else:
        q_t = tf.nest.map_structure(tf.zeros_like, zz)

      r_m = self.r_mix(
          q_t,
          tf.nest.map_structure(tf.zeros_like, q_t),
      )
      return tf.nest.map_structure(
          lambda p_ref, r, t_ref: p_ref / r / t_ref,
          self.p_ref(zz, additional_states),
          r_m,
          self.t_ref(zz, additional_states),
      )
    else:
      return tf.nest.map_structure(
          lambda p_ref, t_ref: p_ref / _R_D / t_ref,
          self.p_ref(zz, additional_states),
          self.t_ref(zz, additional_states),
      )

  def dry_exner(
      self,
      zz: FlowFieldVal,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Computes the exner function using the dry air gas constant.

    Args:
      zz: The geopotential height.
      additional_states: Helper variables for computing the reference density.

    Returns:
      The exner function as a function of height.
    """
    return tf.nest.map_structure(
        lambda p: tf.pow(p / self._p00, _R_D / self.cp_d),
        self.p_ref(zz, additional_states),
    )

  def dry_exner_inverse(
      self,
      zz: FlowFieldVal,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Computes the inverse exner function using the dry air gas constant.

    Args:
      zz: The geopotential height.
      additional_states: Helper variables for computing the reference density.

    Returns:
      The inverse exner function as a function of height.
    """
    p_ref = self.p_ref(zz, additional_states)
    return tf.nest.map_structure(
        lambda p: tf.pow(p / self._p00, -_R_D / self.cp_d), p_ref)

  def exner(
      self,
      rho: FlowFieldVal,
      q_t: FlowFieldVal,
      t: FlowFieldVal,
      zz: FlowFieldVal,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Computes the exner function from the moisture-adjusted constants.

    Args:
      rho: The density of the moist air.
      q_t: The total specific humidity.
      t: The temperature of the flow field.
      zz: The geopotential height.
      additional_states: Helper variables for computing the reference density.

    Returns:
      The moisture-adjusted exner function as a function of height.
    """
    p_ref = self.p_ref(zz, additional_states)
    q_l, q_i = self.equilibrium_phase_partition(t, rho, q_t)

    r_m = self.r_m(t, rho, q_t)
    cp_m = self.cp_m(q_t, q_l, q_i)
    return tf.nest.map_structure(
        lambda p, r, cp: tf.pow(p / self._p00, r / cp), p_ref, r_m, cp_m)

  def exner_inverse(
      self,
      rho: FlowFieldVal,
      q_t: FlowFieldVal,
      t: FlowFieldVal,
      zz: FlowFieldVal,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Computes the inverse exner function from the moisture-adjusted constants.

    Args:
      rho: The density of the moist air.
      q_t: The total specific humidity.
      t: The temperature of the flow field.
      zz: The geopotential height.
      additional_states: Helper variables for computing the reference density.

    Returns:
      The moisture-adjusted inverse exner function as a function of height.
    """
    p_ref = self.p_ref(zz, additional_states)
    q_l, q_i = self.equilibrium_phase_partition(t, rho, q_t)

    r_m = self.r_m(t, rho, q_t)
    cp_m = self.cp_m(q_t, q_l, q_i)
    return tf.nest.map_structure(
        lambda p, r, cp: tf.pow(p / self._p00, -r / cp), p_ref, r_m, cp_m)

  def air_temperature(
      self,
      e_int: FlowFieldVal,
      q_tot: FlowFieldVal,
      q_liq: FlowFieldVal,
      q_ice: FlowFieldVal,
  ) -> FlowFieldVal:
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

    def air_temperature_fn(e_int_i, q_tot_i, q_liq_i, q_ice_i, cv_m_i):
      """Computes the air temperature."""
      return (
          self._t_0
          + (
              e_int_i
              - (q_tot_i - q_liq_i) * self._e_int_v0
              + q_ice_i * (self._e_int_v0 + self._e_int_i0)
          )
          / cv_m_i
      )

    return tf.nest.map_structure(
        air_temperature_fn,
        e_int,
        q_tot,
        q_liq,
        q_ice,
        self.cv_m(q_tot, q_liq, q_ice),
    )

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

  def saturation_vapor_pressure_generic(
      self,
      temperature: FlowFieldVal,
      lh_0: Optional[FlowFieldVal] = None,
      d_cp: Optional[FlowFieldVal] = None,
  ) -> FlowFieldVal:
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
      lh_0 = tf.nest.map_structure(lambda t: self._lh_v0 * tf.ones_like(t),
                                   temperature)
    if d_cp is None:
      d_cp = tf.nest.map_structure(
          lambda t: (self._cp_v - self._cp_l) * tf.ones_like(t), temperature)

    return tf.nest.map_structure(self._saturation_vapor_pressure, temperature,
                                 lh_0, d_cp)

  def saturation_vapor_pressure(
      self,
      temperature: FlowFieldVal,
      q_liq: Optional[FlowFieldVal] = None,
      q_c: Optional[FlowFieldVal] = None,
  ) -> FlowFieldVal:
    """Computes the saturation vapor pressure.

    Args:
      temperature: The temperature of the flow field.
      q_liq: The specific humidity of the liquid phase.
      q_c: The specific humidity of the condensed phase.

    Returns:
      The saturation vapor pressure.
    """
    liquid_frac = self.liquid_fraction(temperature, q_liq, q_c)
    ice_frac = tf.nest.map_structure(lambda liquid_frac_i: 1.0 - liquid_frac_i,
                                     liquid_frac)

    # pylint: disable=g-long-lambda
    lh_0 = tf.nest.map_structure(
        lambda liquid_frac_i, ice_frac_i: liquid_frac_i * self._lh_v0 +
        ice_frac_i * self._lh_s0, liquid_frac, ice_frac)
    d_cp = tf.nest.map_structure(
        lambda liquid_frac_i, ice_frac_i: liquid_frac_i *
        (self._cp_v - self._cp_l) + ice_frac_i * (self._cp_v - self._cp_i),
        liquid_frac, ice_frac)
    # pylint: enable=g-long-lambda

    return self.saturation_vapor_pressure_generic(temperature, lh_0, d_cp)

  def saturation_q_vapor(
      self,
      temperature: FlowFieldVal,
      rho: FlowFieldVal,
      q_liq: Optional[FlowFieldVal] = None,
      q_c: Optional[FlowFieldVal] = None,
  ) -> FlowFieldVal:
    """Computes the saturation specific humidity from the equation of states.

    qᵥₛ = pₛₐₜ / (ϱ Rᵥ T)

    Args:
      temperature: The temperature of the flow field.
      rho: The density of the moist air.
      q_liq: The specific humidity of the liquid phase.
      q_c: The specific humidity of the condensed phase.

    Returns:
      The saturation specific humidity.
    """
    p_v_sat = self.saturation_vapor_pressure(temperature, q_liq, q_c)
    return tf.nest.map_structure(
        lambda p_v_sat_i, rho_i, t_i: p_v_sat_i / (rho_i * self._r_v * t_i),
        p_v_sat, rho, temperature)

  def saturation_q_vapor_from_pressure(
      self,
      temperature: FlowFieldVal,
      q_tot: FlowFieldVal,
      zz: FlowFieldVal,
      q_liq: Optional[FlowFieldVal] = None,
      q_c: Optional[FlowFieldVal] = None,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Computes the saturation specific humidity from the pressure.

    Args:
      temperature: The temperature of the flow field.
      q_tot: The total specific humidity.
      zz: The cooridinates in the vertical direction, in units of m.
      q_liq: The specific humidity of the liquid phase.
      q_c: The specific humidity of the condensed phase.
      additional_states: Optional[FlowFieldMap] = None,

    Returns:
      The saturation specific humidity.
    """
    p_v_sat = self.saturation_vapor_pressure(temperature, q_liq, q_c)
    return tf.nest.map_structure(
        lambda q_t, p_v_s, p: (_R_D / self._r_v) * (1.0 - q_t) * p_v_s /  # pylint: disable=g-long-lambda
        (p - p_v_s), q_tot, p_v_sat, self.p_ref(zz, additional_states))

  def saturation_excess(
      self,
      temperature: FlowFieldVal,
      rho: FlowFieldVal,
      q_tot: FlowFieldVal,
      q_liq: Optional[FlowFieldVal] = None,
      q_c: Optional[FlowFieldVal] = None,
  ) -> FlowFieldVal:
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
    return tf.nest.map_structure(
        lambda q_tot_i, q_vap_sat_i: tf.maximum(0.0, q_tot_i - q_vap_sat_i),
        q_tot,
        q_vap_sat,
    )

  def liquid_fraction(
      self,
      temperature: FlowFieldVal,
      q_liq: Optional[FlowFieldVal] = None,
      q_c: Optional[FlowFieldVal] = None,
  ) -> FlowFieldVal:
    """Computes the fraction of liquid in the condensed phase.

    fₗ = qₗ / qc

    Args:
      temperature: The temperature of the flow field.
      q_liq: The specific humidity of the liquid phase.
      q_c: The specific humidity of the condensed phase.

    Returns:
      The fraction of liquid phase over the condensed phase.
    """
    liquid_frac_nuc = lambda t: (t - self._t_icenuc) / (  # pylint: disable=g-long-lambda
        self._t_freeze - self._t_icenuc)

    def liquid_fraction_from_temperature(t: tf.Tensor) -> tf.Tensor:
      """Computes the liquid fraction from tempearture."""
      return tf.where(
          tf.greater(t, self._t_freeze), tf.ones_like(t),
          tf.where(
              tf.less_equal(t, self._t_icenuc), tf.zeros_like(t),
              liquid_frac_nuc(t)))

    liquid_frac_no_condensate = tf.nest.map_structure(
        liquid_fraction_from_temperature, temperature)

    if q_liq is None or q_c is None:
      return liquid_frac_no_condensate

    def liquid_fraction_from_condensate(
        q_l: tf.Tensor,
        q_c: tf.Tensor,
        liquid_frac: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the liquid fraction from condensate fractions."""
      return tf.where(tf.greater(q_c, 0.0), q_l / q_c, liquid_frac)

    return tf.nest.map_structure(
        liquid_fraction_from_condensate, q_liq, q_c, liquid_frac_no_condensate)

  def equilibrium_phase_partition(
      self,
      temperature: FlowFieldVal,
      rho: FlowFieldVal,
      q_tot: FlowFieldVal,
  ) -> tuple[FlowFieldVal, FlowFieldVal]:
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
    q_liq = tf.nest.map_structure(
        lambda liquid_frac_i, q_c_i: liquid_frac_i * q_c_i, liquid_frac, q_c
    )
    q_ice = tf.nest.map_structure(
        lambda liquid_frac_i, q_c_i: (1.0 - liquid_frac_i) * q_c_i,
        liquid_frac,
        q_c,
    )
    return q_liq, q_ice

  def internal_energy_components(
      self,
      temperature: FlowFieldVal,
  ) -> Sequence[FlowFieldVal]:
    """Computes the specific internal energy for vapor, liquid, and ice.

    Args:
      temperature: The temperature of the flow field.

    Returns:
      e_v: specific internal energy for the vapor phase.
      e_l: specific internal energy for the liquid phase.
      e_i: specific internal energy for the ice phase.

    """
    # pylint: disable=invalid-name
    dT = tf.nest.map_structure(lambda t: t - self._t_0, temperature)
    e_v = tf.nest.map_structure(
        lambda dT: self._cv_v * dT + self._lh_v0 - self._r_v * self._t_0, dT)
    e_l = tf.nest.map_structure(lambda dT: self._cv_l * dT, dT)
    l_f0 = self._lh_s0 - self._lh_v0
    e_i = tf.nest.map_structure(lambda dT: self._cv_i * dT - l_f0, dT)
    return e_v, e_l, e_i

  def internal_energy(
      self,
      temperature: FlowFieldVal,
      q_tot: FlowFieldVal,
      q_liq: FlowFieldVal,
      q_ice: FlowFieldVal,
  ) -> FlowFieldVal:
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

    def internal_energy_fn(cv_m_i, t_i, q_tot_i, q_liq_i, q_ice_i):
      """Computes the internal energy."""
      return (
          cv_m_i * (t_i - self._t_0)
          + (q_tot_i - q_liq_i) * self._e_int_v0
          - q_ice_i * (self._e_int_v0 + self._e_int_i0)
      )

    return tf.nest.map_structure(
        internal_energy_fn,
        self.cv_m(q_tot, q_liq, q_ice),
        temperature,
        q_tot,
        q_liq,
        q_ice,
    )

  def internal_energy_from_total_energy(
      self,
      e_t: FlowFieldVal,
      u: FlowFieldVal,
      v: FlowFieldVal,
      w: FlowFieldVal,
      zz: Optional[FlowFieldVal] = None,
  ) -> FlowFieldVal:
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
    zz = tf.nest.map_structure(tf.zeros_like, e_t) if zz is None else zz
    ke = tf.nest.map_structure(
        lambda u_i, v_i, w_i: 0.5 * (u_i**2 + v_i**2 + w_i**2), u, v, w
    )
    pe = tf.nest.map_structure(lambda zz_i: constants.G * zz_i, zz)
    return tf.nest.map_structure(
        lambda e_t_i, ke_i, pe_i: e_t_i - ke_i - pe_i, e_t, ke, pe
    )

  def total_energy(
      self,
      e: FlowFieldVal,
      u: FlowFieldVal,
      v: FlowFieldVal,
      w: FlowFieldVal,
      zz: Optional[FlowFieldVal] = None,
  ) -> FlowFieldVal:
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
    zz = tf.nest.map_structure(tf.zeros_like, e) if zz is None else zz
    ke = tf.nest.map_structure(
        lambda u_i, v_i, w_i: 0.5 * (u_i**2 + v_i**2 + w_i**2), u, v, w
    )
    pe = tf.nest.map_structure(lambda zz_i: constants.G * zz_i, zz)

    # Note, we are doing 1.0 * (e_i + ke_i) + pe_i here to work around a
    # non-deterministic issue. See b/221776082.
    return tf.nest.map_structure(
        lambda e_i, ke_i, pe_i: 1.0 * (e_i + ke_i) + pe_i, e, ke, pe
    )

  def total_enthalpy(
      self,
      e_tot: FlowFieldVal,
      rho: FlowFieldVal,
      q_tot: FlowFieldVal,
      temperature: FlowFieldVal,
  ) -> FlowFieldVal:
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
    return tf.nest.map_structure(
        lambda e_tot_i, r_m_i, t_i: e_tot_i + r_m_i * t_i,
        e_tot,
        r_m,
        temperature,
    )

  def saturation_internal_energy(
      self,
      temperature: FlowFieldVal,
      rho: FlowFieldVal,
      q_tot: FlowFieldVal,
  ) -> FlowFieldVal:
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
      temperature: FlowFieldVal,
      rho: FlowFieldVal,
      q_tot: FlowFieldVal,
  ) -> FlowFieldVal:
    """Computes the analytical Jacobian of internal energy wrt temperature.

    Args:
      temperature: The temperature of the flow field.
      rho: The density of moist air.
      q_tot: The total specific humidity.

    Returns:
      The partial derivative of the internal energy wrt temperature.
    """
    q_liq, q_ice = self.equilibrium_phase_partition(temperature, rho, q_tot)
    q_c = tf.nest.map_structure(tf.math.add, q_liq, q_ice)
    cv_mix = self.cv_m(q_tot, q_liq, q_ice)
    q_vap_sat = self.saturation_q_vapor(temperature, rho, q_liq, q_c)
    liquid_frac = self.liquid_fraction(temperature, q_liq, q_c)

    def l_fn(liquid_frac_i):
      """Computes the latent heat."""
      return liquid_frac_i * self._lh_v0 + (1.0 - liquid_frac_i) * self._lh_s0

    l = tf.nest.map_structure(l_fn, liquid_frac)

    def dq_vap_sat_dt_fn(q_vap_sat_i, l_i, t_i):
      """Computes the time rate of change of saturation vapor fraction."""
      return q_vap_sat_i * l_i / (self._r_v * t_i**2)

    dq_vap_sat_dt = tf.nest.map_structure(
        dq_vap_sat_dt_fn, q_vap_sat, l, temperature
    )

    def dcvm_dq_vap_fn(liquid_frac_i):
      """Computes the gradient of cvm wrt q_vap."""
      return (
          self._cv_v
          - liquid_frac_i * self._cv_l
          - (1.0 - liquid_frac_i) * self._cv_i
      )

    dcvm_dq_vap = tf.nest.map_structure(dcvm_dq_vap_fn, liquid_frac)

    def de_int_dt_fn(
        cv_m_i, liquid_frac_i, t_i, dcvm_dq_vap_i, dq_vap_sat_dt_i
    ):
      """Compute the time rate of change of the internal energy."""
      return (
          cv_m_i
          + (
              self._e_int_v0
              + (1 - liquid_frac_i) * self._e_int_i0
              + (t_i - self._t_0) * dcvm_dq_vap_i
          )
          * dq_vap_sat_dt_i
      )

    return tf.nest.map_structure(
        de_int_dt_fn,
        cv_mix,
        liquid_frac,
        temperature,
        dcvm_dq_vap,
        dq_vap_sat_dt,
    )

  def saturation_temperature(
      self,
      target_var_name: Text,
      t_guess: FlowFieldVal,
      target_var: FlowFieldVal,
      rho: FlowFieldVal,
      q_tot: FlowFieldVal,
      zz: Optional[FlowFieldVal] = None,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Computes the temperature assuming water is at saturation."""

    def internal_energy_error_fn(temperature: FlowFieldVal) -> FlowFieldVal:
      """Computes the error of internal energy for the Newton iterations."""
      e_int_sat = self.saturation_internal_energy(temperature, rho, q_tot)
      return tf.nest.map_structure(tf.math.subtract, e_int_sat, target_var)

    def potential_temperature_error_fn(
        temperature: FlowFieldVal) -> FlowFieldVal:
      """Computes the error of potential temperature for Newton iterations."""
      theta_sat = self.potential_temperatures(
          temperature, q_tot, rho, zz, additional_states
      )[target_var_name]
      return tf.nest.map_structure(tf.math.subtract, theta_sat, target_var)

    if target_var_name == 'e_int':
      error_fn = internal_energy_error_fn
      jacobian_fn = lambda temperature: self.de_int_dt(temperature, rho, q_tot)
    elif target_var_name in (PotentialTemperature.THETA.value,
                             PotentialTemperature.THETA_V.value,
                             PotentialTemperature.THETA_LI.value):
      error_fn = potential_temperature_error_fn
      jacobian_fn = None
    else:
      raise ValueError(
          f'{target_var_name} is not a valid variable for saturation '
          f'temperature computation. Available options are: "e_int", '
          f'"{PotentialTemperature.THETA.value}", '
          f'"{PotentialTemperature.THETA_V.value}", '
          f'"{PotentialTemperature.THETA_LI.value}".')

    return root_finder.newton_method(
        error_fn,
        t_guess,
        self._t_max_iter,
        position_tolerance=self._f_temperature_atol_and_rtol,
        value_tolerance=self._f_temperature_atol_and_rtol,
        analytical_jacobian_fn=jacobian_fn,
    )

  def saturation_adjustment(
      self,
      target_var_name: Text,
      target_var: FlowFieldVal,
      rho: FlowFieldVal,
      q_tot: FlowFieldVal,
      zz: Optional[FlowFieldVal] = None,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Computes the temperature that is consistent with the input state.

    Args:
      target_var_name: The name of the target variable that is used to derive
        the temperature.
      target_var: The value of the variable corresponding to `target_var_name`.
      rho: The density of moist air.
      q_tot: The total specific humidity.
      zz: The vertical coordinates.
      additional_states: Optional[FlowFieldMap] = None,

    Returns:
      The temperature at the given state.
    """
    q_liq = tf.nest.map_structure(tf.zeros_like, q_tot)
    q_ice = tf.nest.map_structure(tf.zeros_like, q_tot)

    if target_var_name == 'e_int':
      t_air = self.air_temperature(target_var, q_tot, q_liq, q_ice)
      target_freeze_fn = lambda t: self.internal_energy(t, q_tot, q_liq, q_ice)
    elif target_var_name in (PotentialTemperature.THETA.value,
                             PotentialTemperature.THETA_V.value,
                             PotentialTemperature.THETA_LI.value):
      t_air = self.potential_temperature_to_temperature(
          target_var_name,
          target_var,
          q_tot,
          q_liq,
          q_ice,
          zz,
          additional_states,
      )
      target_freeze_fn = lambda t: self.potential_temperatures(  # pylint: disable=g-long-lambda
          t, q_tot, rho, zz, additional_states)[target_var_name]
    elif target_var_name == 'T':
      return target_var
    else:
      raise ValueError(
          f'{target_var_name} is not a valid variable for saturation '
          f'temperature computation. Available options are: "e_int", '
          f'"{PotentialTemperature.THETA.value}", '
          f'"{PotentialTemperature.THETA_V.value}", '
          f'"{PotentialTemperature.THETA_LI.value}".')

    # Case 1: temperature at unsaturated condition.
    t_1 = tf.nest.map_structure(lambda t: tf.maximum(self._t_min, t), t_air)
    q_v_sat = self.saturation_q_vapor(t_1, rho)

    # Case 2: temperature at freezing point.
    t_freeze = tf.nest.map_structure(lambda t: self._t_freeze * tf.ones_like(t),
                                     target_var)
    target_freeze = target_freeze_fn(t_freeze)

    # Case 3: temperature at saturation condition.
    t_sat = self.saturation_temperature(target_var_name, t_1, target_var, rho,
                                        q_tot, zz, additional_states)

    # Get temperature under correct conditions.
    def unsaturation_cond(
        q_tot_i: tf.Tensor,
        q_v_sat_i: tf.Tensor,
        t_1_i: tf.Tensor,
    ) -> tf.Tensor:
      """Determines if the fluid is unsaturated."""
      return tf.math.logical_and(
          tf.less_equal(q_tot_i, q_v_sat_i), tf.greater(t_1_i, self._t_min))

    t = tf.nest.map_structure(
        lambda q_tot_i, q_v_sat_i, t_1_i, t_sat_i: tf.where(  # pylint: disable=g-long-lambda
            unsaturation_cond(q_tot_i, q_v_sat_i, t_1_i), t_1_i, t_sat_i),
        q_tot, q_v_sat, t_1, t_sat)

    def freezing_cond(
        target_freeze_i: tf.Tensor,
        target_i: tf.Tensor,
    ) -> tf.Tensor:
      """Determines if the fluid is frozen."""
      return tf.less(tf.abs(target_freeze_i - target_i), _EPS_E_INT)

    return tf.nest.map_structure(
        lambda target_freeze_i, target_i, t_freeze_i, t_i: tf.where(  # pylint: disable=g-long-lambda
            freezing_cond(target_freeze_i, target_i), t_freeze_i, t_i),
        target_freeze, target_var, t_freeze, t)

  def saturation_density(
      self,
      prognostic_var_name: Text,
      prognostic_var: FlowFieldVal,
      q_tot: FlowFieldVal,
      u: FlowFieldVal,
      v: FlowFieldVal,
      w: FlowFieldVal,
      rho_0: Optional[FlowFieldVal] = None,
      zz: Optional[FlowFieldVal] = None,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Computes the density that is consistent with the input state.

    Args:
      prognostic_var_name: The name of the target variable that is used to
        derive the temperature.
      prognostic_var: The value of the variable corresponds to
        `target_var_name`.
      q_tot: The total specific humidity.
      u: The velocity component in the x direction.
      v: The velocity component in the y direction.
      w: The velocity component in the z direction. the Secant solver.
      rho_0: A guess of the density, which is used as the initial condition for
        finding the density that is thermodynamically consistent with the state.
      zz: The cooridinates in the vertical direction.
      additional_states: Optional[FlowFieldMap] = None,

    Returns:
      The density at the given state.
    """
    if prognostic_var_name == 'e_t':
      target_var_name = 'e_int'
      target_var = self.internal_energy_from_total_energy(
          prognostic_var, u, v, w, zz)
    elif prognostic_var_name in (PotentialTemperature.THETA.value,
                                 PotentialTemperature.THETA_V.value,
                                 PotentialTemperature.THETA_LI.value,
                                 'T'):
      target_var_name = prognostic_var_name
      target_var = prognostic_var
    else:
      raise ValueError(
          f'{prognostic_var_name} is not a valid variable for saturation '
          f'temperature computation. Available options are: "e_int", '
          f'"{PotentialTemperature.THETA.value}", '
          f'"{PotentialTemperature.THETA_V.value}", '
          f'"{PotentialTemperature.THETA_LI.value}", and "T".')

    p = (
        self.p_ref(zz, additional_states)  # pylint: disable=g-long-ternary
        if zz is not None
        else tf.nest.map_structure(
            lambda f: self._p_thermal * tf.ones_like(f), prognostic_var
        )
    )

    def density_update_fn(rho: FlowFieldVal) -> FlowFieldVal:
      """Updates the density for one iteration."""
      temperature_sat = self.saturation_adjustment(
          target_var_name, target_var, rho, q_tot, zz, additional_states
      )
      r_mix = self.r_m(temperature_sat, rho, q_tot)
      return tf.nest.map_structure(
          lambda p_i, t_sat, r_m: p_i / t_sat / r_m, p, temperature_sat, r_mix
      )

    if rho_0 is None:
      rho_0 = tf.nest.map_structure(tf.ones_like, target_var)

    i0 = tf.constant(0)
    cond = lambda i, rho: tf.less(i, self._rho_n_iter)
    body = lambda i, rho: (i + 1, density_update_fn(rho))
    _, rho = tf.while_loop(
        cond=cond, body=body, loop_vars=(i0, rho_0), back_prop=False)

    return rho

  def potential_temperatures(
      self,
      t: FlowFieldVal,
      q_t: FlowFieldVal,
      rho: FlowFieldVal,
      zz: Optional[FlowFieldVal] = None,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldMap:
    """Computes the potential temperatures.

    Reference: CliMa Design doc.

    Args:
      t: The temperature, in units of K.
      q_t: The total humidty.
      rho: The density of the moist air, in units of kg/m^3.
      zz: The vertical coordinates.
      additional_states: Optional[FlowFieldMap] = None,

    Returns:
      A dictionary of the three potential temperatures, namely the potential
      temperature for the moist air mixture, the liquid water potential
      temperature, and the virtual potential temperature.
    """
    zz = zz if zz is not None else tf.nest.map_structure(tf.zeros_like, t)

    q_l, q_i = self.equilibrium_phase_partition(t, rho, q_t)

    r_m = self.r_m(t, rho, q_t)
    cp_m = self.cp_m(q_t, q_l, q_i)

    p_ref = self.p_ref(zz, additional_states)
    exner_inv = tf.nest.map_structure(
        lambda p, r, cp: tf.pow(p / self._p00, -r / cp), p_ref, r_m, cp_m)

    theta_li = tf.nest.map_structure(
        lambda t_k, q_l_k, q_i_k, cp_m_k, exner_inv_k:  # pylint: disable=g-long-lambda
        (t_k -
         (self._lh_v0 * q_l_k + self._lh_s0 * q_i_k) / cp_m_k) * exner_inv_k,
        t,
        q_l,
        q_i,
        cp_m,
        exner_inv)

    theta = tf.nest.map_structure(tf.math.multiply, t, exner_inv)

    theta_v = tf.nest.map_structure(lambda r, th: (r / _R_D) * th, r_m, theta)

    return {
        PotentialTemperature.THETA.value: theta,
        PotentialTemperature.THETA_LI.value: theta_li,
        PotentialTemperature.THETA_V.value: theta_v
    }

  def potential_temperature_to_temperature(
      self,
      theta_name: Text,
      theta: FlowFieldVal,
      q_tot: FlowFieldVal,
      q_liq: FlowFieldVal,
      q_ice: FlowFieldVal,
      zz: Optional[FlowFieldVal] = None,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Computes temperature from potential temperature.

    Args:
      theta_name: The name of the potential temperature variable `theta`, should
        be one of the following: 'theta' (the potential temperature of the moist
          air mixture), 'theta_v' (the virtual potential temperature),
          'theta_li' (the liquid-ice potential temperature).
      theta: The potential temperature flow field.
      q_tot: The total humidity.
      q_liq: The liquid humidity.
      q_ice: The ice humidity.
      zz: The vertical coordinates.
      additional_states: Optional[FlowFieldMap] = None,

    Returns:
      The temperature in units of K.

    Raises:
      ValueError: If `theta_name` is not in ('theta', 'theta_v', 'theta_li').
    """
    zz = zz if zz is not None else tf.nest.map_structure(tf.zeros_like, theta)

    q_c = tf.nest.map_structure(tf.math.add, q_liq, q_ice)
    r_m = self.r_mix(q_tot, q_c)
    cp_m = self.cp_m(q_tot, q_liq, q_ice)

    p_ref = self.p_ref(zz, additional_states)
    exner = tf.nest.map_structure(
        lambda p, r, cp: tf.pow(p / self._p00, r / cp), p_ref, r_m, cp_m)

    if theta_name == PotentialTemperature.THETA.value:
      t = tf.nest.map_structure(tf.math.multiply, theta, exner)
    elif theta_name == PotentialTemperature.THETA_V.value:
      t = tf.nest.map_structure(lambda t, exner_i, r: _R_D / r * exner_i * t,
                                theta, exner, r_m)
    elif theta_name == PotentialTemperature.THETA_LI.value:
      t = tf.nest.map_structure(
          lambda t_k, q_l_k, q_i_k, cp_m_k, exner_k:  # pylint: disable=g-long-lambda
          t_k * exner_k +
          (self._lh_v0 * q_l_k + self._lh_s0 * q_i_k) / cp_m_k,
          theta,
          q_liq,
          q_ice,
          cp_m,
          exner)
    else:
      raise ValueError(
          f'`theta_name` has to be either "{PotentialTemperature.THETA.value}" '
          f'(the potential temperature of the air mixture), '
          f'"{PotentialTemperature.THETA_V.value}" (the virtual potential '
          f'temperature), or "{PotentialTemperature.THETA_LI.value}" (the '
          f'liquid-ice potential temperature), but {theta_name} is provided.')

    return t

  def temperature_to_potential_temperature(
      self,
      theta_name: Text,
      temperature: FlowFieldVal,
      q_tot: FlowFieldVal,
      q_liq: FlowFieldVal,
      q_ice: FlowFieldVal,
      zz: Optional[FlowFieldVal] = None,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Computes potential temperature from temperature.

    Args:
      theta_name: The name of the potential temperature variable `theta`, should
        be one of the following: 'theta' (the potential temperature of the moist
          air mixture), 'theta_v' (the virtual potential temperature),
          'theta_li' (the liquid-ice potential temperature).
      temperature: The temperature flow field.
      q_tot: The total humidity.
      q_liq: The liquid humidity.
      q_ice: The ice humidity.
      zz: The vertical coordinates.
      additional_states: Optional[FlowFieldMap] = None,

    Returns:
      The potential temperature in units of K.

    Raises:
      ValueError: If `theta_name` is not in ('theta', 'theta_v', 'theta_li').
    """
    zz = (
        zz
        if zz is not None
        else tf.nest.map_structure(tf.zeros_like, temperature)
    )

    q_c = tf.nest.map_structure(tf.math.add, q_liq, q_ice)
    r_m = self.r_mix(q_tot, q_c)
    cp_m = self.cp_m(q_tot, q_liq, q_ice)

    p_ref = self.p_ref(zz, additional_states)
    exner = tf.nest.map_structure(
        lambda p, r, cp: tf.pow(p / self._p00, r / cp), p_ref, r_m, cp_m)

    if theta_name == PotentialTemperature.THETA.value:
      theta = tf.nest.map_structure(tf.math.divide, temperature, exner)
    elif theta_name == PotentialTemperature.THETA_V.value:
      theta = tf.nest.map_structure(
          lambda t, exner_i, r: t / (_R_D / r * exner_i),
          temperature,
          exner,
          r_m,
      )
    elif theta_name == PotentialTemperature.THETA_LI.value:
      theta = tf.nest.map_structure(
          lambda t_k, q_l_k, q_i_k, cp_m_k, exner_k: (  # pylint: disable=g-long-lambda
              t_k - (self._lh_v0 * q_l_k + self._lh_s0 * q_i_k) / cp_m_k
          )
          / exner_k,
          temperature,
          q_liq,
          q_ice,
          cp_m,
          exner,
      )
    else:
      raise ValueError(
          f'`theta_name` has to be either "{PotentialTemperature.THETA.value}" '
          f'(the potential temperature of the air mixture), '
          f'"{PotentialTemperature.THETA_V.value}" (the virtual potential '
          f'temperature), or "{PotentialTemperature.THETA_LI.value}" (the '
          f'liquid-ice potential temperature), but {theta_name} is provided.')

    return theta

  def _rho_temperature_newton_solver(
      self,
      f2: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
      rho_guess: tf.Tensor,
      t_guess: tf.Tensor,
      q_t: tf.Tensor,
      p_ref: tf.Tensor,
      num_iterations: int,
  ) -> tuple[tf.Tensor, tf.Tensor]:
    """Custom Newton solver for rho and temperature given theta_li, q_t.

    The 2 functions to solve f1 = 0 & f2 = 0 are:
        f1 = Rm * rho * T - p_ref = 0
        f2 = theta_li - theta_li_eqb(rho, T, p_ref, q_t) = 0

    Args:
      f2: The function giving the error in the theta_li equation.
      rho_guess: The initial guess for density.
      t_guess: The initial guess for temperature.
      q_t: The total vapor specific humidity.
      p_ref: The reference pressure.
      num_iterations: The number of Newton iterations to perform.

    Returns:
      A tuple of (rho, temperature) where rho is the density and T is the
      temperature.
    """
    dtype = tf.nest.flatten(t_guess)[0].dtype
    # Compute an eps that is 128 times the machine epsilon (maintaining a clean
    # binary representation) for use in the finite difference approximation of
    # derivatives.
    eps = np.finfo(dtype.as_numpy_dtype).eps * 128

    def body(
        i: tf.Tensor, rho_t: tuple[tf.Tensor, tf.Tensor]
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
      """The main function for one Newton iteration."""
      rho, temperature = rho_t

      # Calculate Rm.
      rm = self.r_m(temperature, rho, q_t)

      # Compute the Jacobian J = [∂f1/∂ρ, ∂f1/∂T; ∂f2/∂ρ, ∂f2/∂T]

      # Analytic derivatives for f1 with respect to rho and T.
      # Note: these derivatives are approximate because we neglect the
      # derivative of Rm with respect to T and rho. However, these derivatives
      # are very small, and so the approximation is quite good.
      j11 = rm * temperature  # ∂f1/∂ρ
      j12 = rm * rho  # ∂f1/∂T

      # Finite difference derivatives for f2 with respect to rho and
      # T.
      # ∂f2/∂ρ
      drho = eps * rho
      rho_plus = rho + drho / 2
      rho_minus = rho - drho / 2
      j21 = (f2(rho_plus, temperature) - f2(rho_minus, temperature)) / drho

      # ∂f2/∂T
      dtemp = eps * temperature
      temperature_plus = temperature + dtemp / 2
      temperature_minus = temperature - dtemp / 2
      j22 = (f2(rho, temperature_plus) - f2(rho, temperature_minus)) / dtemp

      # Invert the Jacobian and proceed with a Newton iteration.
      determinant = j11 * j22 - j12 * j21
      jinv_11 = j22 / determinant
      jinv_12 = -j12 / determinant
      jinv_21 = -j21 / determinant
      jinv_22 = j11 / determinant

      f1val = rho * rm * temperature - p_ref
      f2val = f2(rho, temperature)

      rho_new = rho - (jinv_11 * f1val + jinv_12 * f2val)
      temperature_new = temperature - (jinv_21 * f1val + jinv_22 * f2val)
      return (i + 1, (rho_new, temperature_new))

    i0 = tf.constant(0)
    cond = lambda i, rho_t: tf.less(i, num_iterations)

    _, (rho, temperature) = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(i0, (rho_guess, t_guess)),
        back_prop=False,
    )
    return rho, temperature

  def _theta_li_from_temperature_rho_qt(
      self,
      temperature: tf.Tensor,
      rho: tf.Tensor,
      q_t: tf.Tensor,
      p_ref: tf.Tensor,
  ) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute theta_li from T, rho, q_t, p_ref in equilibrium."""
    q_l, q_i = self.equilibrium_phase_partition(temperature, rho, q_t)
    r_m = self.r_m(temperature, rho, q_t)
    cp_m = self.cp_m(q_t, q_l, q_i)
    exner_inv = tf.pow(p_ref / self._p00, -r_m / cp_m)
    lh_v0 = self._lh_v0
    lh_s0 = self._lh_s0

    theta_li = exner_inv * (temperature - (lh_v0 * q_l + lh_s0 * q_i) / cp_m)
    return theta_li

  def _rho_and_temperature_from_theta_li_qt_fast(
      self,
      theta_li: tf.Tensor,
      q_t: tf.Tensor,
      p_ref: tf.Tensor,
      rho_guess: tf.Tensor,
  ) -> tuple[tf.Tensor, tf.Tensor]:
    """Fast solver to determine (rho, T) from (theta_li, q_t, p_ref).

    Solves the 2 simultaneous equations for thermodynamic equilibrium:
        p_ref = rho * Rm(rho, T, q_t) * T
        theta_li = theta_li_eqb(rho, T, p_ref, q_t)

    Care must be taken around the freezing point because the thermodynamic
    equations are not smooth there.

    Args:
      theta_li: The liquid-ice potential temperature.
      q_t: The total vapor specific humidity.
      p_ref: The reference pressure.
      rho_guess: The initial guess for density.

    Returns:
      A tuple of (rho, T) where rho is the density and T is the temperature.
    """
    # Case 1: temperature at unsaturated condition.
    # Compute the temperature assuming the air is unsaturated.  This T will be
    # the initial guess for the Newton iterations.
    r_m_unsat = (1 - q_t) * _R_D + q_t * self._r_v
    cp_m_unsat = (1 - q_t) * constants.CP + q_t * self._cp_v
    exner = (p_ref / self._p00) ** (r_m_unsat / cp_m_unsat)
    t1 = exner * theta_li
    t1 = tf.maximum(self._t_min, t1)  # Apply a cutoff for numerical reasons.
    t_guess = t1

    def potential_temperature_error_fn(
        rho: tf.Tensor, temperature: tf.Tensor
    ) -> tf.Tensor:
      """Computes the error of potential temperature for Newton iterations."""
      theta_sat = self._theta_li_from_temperature_rho_qt(
          temperature, rho, q_t, p_ref
      )
      return theta_sat - theta_li

    num_iterations = self._t_max_iter
    rho, temperature_saturation = self._rho_temperature_newton_solver(
        potential_temperature_error_fn,
        rho_guess,
        t_guess,
        q_t,
        p_ref,
        num_iterations,
    )

    # Handle the freezing case.
    # Compute theta_li at the freezing point for unsaturated (assuming q_c=0)
    theta_li_freeze = self._t_freeze / exner
    # If at the freezing point, use the freezing point temperature and density.
    temperature_eqb = tf.where(
        tf.abs(theta_li_freeze - theta_li) < _EPS_E_INT,
        self._t_freeze,
        temperature_saturation,
    )
    rho_eqb = tf.where(
        tf.abs(theta_li_freeze - theta_li) < _EPS_E_INT,
        p_ref / (r_m_unsat * temperature_eqb),
        rho,
    )
    return rho_eqb, temperature_eqb

  def _update_density_and_temperature_fast(
      self, states: FlowFieldMap, additional_states: FlowFieldMap
  ) -> tuple[FlowFieldVal, FlowFieldVal]:
    """Fast version of update_density; returns tuple of (rho, T).

    This function implements a faster solver for determining (rho, T) from
    (theta_li, q_t, p). It uses only a single Newton iteration loop to update
    both density and temperature.  This function acts as an interface to
    `rho_and_temperature_from_thetali_qt_fast`.

    Args:
      states: Flow field variables, must contain 'rho', 'theta_li', 'q_t'.
      additional_states: Helper variables that are required to compute potential
        temperatures.  Must contain 'p_ref'.

    Returns:
      A tuple of (rho, T) where rho is the density and T is the temperature.
    """
    if 'theta_li' not in states or 'q_t' not in states:
      raise ValueError(
          'update_density_fast is only implemented when theta_li and q_t are'
          ' prognostic variables.'
      )
    theta_li = states['theta_li']
    q_t = states['q_t']
    zz = additional_states.get('zz', None)
    p_ref = self.p_ref(zz, additional_states)
    rho_guess = states['rho']
    return self._rho_and_temperature_from_theta_li_qt_fast(
        theta_li, q_t, p_ref, rho_guess
    )

  def update_density(
      self, states: FlowFieldMap, additional_states: FlowFieldMap
  ) -> tf.Tensor:
    """Updates the density of the flow field with water thermodynamics."""
    if (
        self._use_fast_thermodynamics
        and 'theta_li' in states
        and 'q_t' in states
    ):
      #  `update_density_fast` is only implemented for determining density
      # and temperature from theta_li and q_t.
      density, _ = self._update_density_and_temperature_fast(
          states, additional_states
      )
      return density
    elif self._use_fast_thermodynamics:
      logging.warning(
          '`use_fast_thermodynamics` is set to True, but `theta_li` and `q_t`'
          ' are not both in `states`. Falling back to the slow solver. Note:'
          ' If using combustion, this behavior is expected to occur even if'
          ' `theta_li` and `q_t` are the prognostic variables.'
      )

    zz = additional_states.get('zz', None)

    # Get the prognostic variable.
    allowed_energy_var_names = set((
        'e_t',
        PotentialTemperature.THETA.value,
        PotentialTemperature.THETA_V.value,
        PotentialTemperature.THETA_LI.value,
        'T',
    ))
    energy_var_found = allowed_energy_var_names & set(states.keys())
    assert len(energy_var_found) == 1, (
        f'One and only one energy variable (one of {allowed_energy_var_names})'
        ' is allowed to update the density with the water thermodynamics, but'
        f' {energy_var_found} are provided.'
    )
    prognostic_var_name = energy_var_found.pop()

    q_t = states['q_t'] if 'q_t' in states else (
        tf.nest.map_structure(tf.math.add, states['q_c'], states['q_v']))
    return self.saturation_density(prognostic_var_name,
                                   states[prognostic_var_name], q_t,
                                   states['u'], states['v'], states['w'],
                                   states['rho'], zz, additional_states)

  def update_temperatures(
      self,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Computes the temperature and potential temperatures.

    Args:
      states: Flow field variables, must contain 'rho', 'u', 'v', 'w', 'q_t',
        and one of 'e_t', 'theta', 'theta_li', 'theta_v'.
      additional_states: Helper variables that are required to compute potential
        temperatures. If field 'zz' is not in `additional_states`, assumes the
        flow field is independent of height, i.e. `zz` = 0.

    Returns:
      A dictionary of flow fields that contains, the temperature 'T', the liquid
      potential temperature 'T_l', and the virtual potential temperature 'T_v'.
    """
    zz = additional_states.get(
        'zz', tf.nest.map_structure(tf.zeros_like, states['rho']))

    # Get the prognostic variable.
    if 'e_t' in states:
      target_var_name = 'e_int'
      target_var = self.internal_energy_from_total_energy(
          states['e_t'], states['u'], states['v'], states['w'], zz)
    elif PotentialTemperature.THETA.value in states:
      target_var_name = PotentialTemperature.THETA.value
      target_var = states[PotentialTemperature.THETA.value]
    elif PotentialTemperature.THETA_V.value in states:
      target_var_name = PotentialTemperature.THETA_V.value
      target_var = states[PotentialTemperature.THETA_V.value]
    elif PotentialTemperature.THETA_LI.value in states:
      target_var_name = PotentialTemperature.THETA_LI.value
      target_var = states[PotentialTemperature.THETA_LI.value]
    else:
      raise ValueError(
          f'No prognostic variable for energy is found. Supported options are'
          f'"e_t" (the total energy), "{PotentialTemperature.THETA.value}" '
          f'(the potential temperature of the air mixture), '
          f'"{PotentialTemperature.THETA_V.value}" (the virtual potential '
          f'temperature), or "{PotentialTemperature.THETA_LI.value}" (the '
          f'liquid-ice potential temperature).')

    if (
        self._use_fast_thermodynamics
        and 'theta_li' in states
        and 'q_t' in states
    ):
      rho_thermal, temperature = self._update_density_and_temperature_fast(
          states, additional_states
      )
      temperatures = {'T': temperature}
      temperatures |= self.potential_temperatures(
          temperature, states['q_t'], rho_thermal, zz, additional_states
      )
      return temperatures

    rho_thermal = (
        self.update_density(states, additional_states) if
        self._solver_mode == thermodynamics_pb2.Thermodynamics.ANELASTIC else
        states['rho'])

    q_t = states['q_t'] if 'q_t' in states else (
        tf.nest.map_structure(tf.math.add, states['q_c'], states['q_v']))

    temperatures = {
        'T': self.saturation_adjustment(
            target_var_name,
            target_var,
            rho_thermal,
            q_t,
            zz,
            additional_states,
        )
    }
    temperatures.update(
        self.potential_temperatures(temperatures['T'], q_t,
                                    rho_thermal, zz, additional_states))

    return temperatures
