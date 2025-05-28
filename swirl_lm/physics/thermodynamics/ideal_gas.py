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
"""A library of modeling thermodynamic quantities with ideal gas."""

from typing import Optional
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics.thermodynamics import thermodynamics_generic
from swirl_lm.physics.thermodynamics import thermodynamics_utils
import tensorflow as tf

TF_DTYPE = thermodynamics_generic.TF_DTYPE

FlowFieldVal = thermodynamics_generic.FlowFieldVal
FlowFieldMap = thermodynamics_generic.FlowFieldMap

INERT_SPECIES = thermodynamics_utils.INERT_SPECIES
# A list of variables that are not considered as chemical species.
# T: the temperature.
# theta: the potential temperature.
NON_SPECIES = ['T', 'theta']

R_U = thermodynamics_utils.R_UNIVERSAL
DRY_AIR_MOLECULAR_WEIGHT = 0.0289647


class IdealGas(thermodynamics_generic.ThermodynamicModel):
  """A library of ideal gas modeling."""

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    """Initializes the ideal gas object."""
    super(IdealGas, self).__init__(params)

    self._molecular_weights = {
        scalar_name: params.molecular_weight(scalar_name)
        for scalar_name in params.scalars_names
        if scalar_name not in NON_SPECIES
    }
    self._p_thermal = params.p_thermal

    assert (
        model_params := params.thermodynamics
    ) is not None, 'Thermodynamics must be set in the config.'
    self._t_s = model_params.ideal_gas_law.t_s
    w_inert = (
        self._molecular_weights[INERT_SPECIES] if INERT_SPECIES
        in self._molecular_weights else DRY_AIR_MOLECULAR_WEIGHT)
    self.r_d = R_U / w_inert
    self.cp_d = model_params.ideal_gas_law.cv_d + self.r_d
    self.kappa = self.r_d / self.cp_d
    self._height = model_params.ideal_gas_law.height
    self._delta_t = model_params.ideal_gas_law.delta_t
    self._const_theta = (
        tf.constant(model_params.ideal_gas_law.const_theta)
        if model_params.ideal_gas_law.HasField('const_theta')
        else None
    )

  @staticmethod
  def density_by_ideal_gas_law(
      p: tf.Tensor,
      r: tf.Tensor | float,
      t: tf.Tensor,
  ) -> tf.Tensor:
    """Computes the density following the ideal gas law.

    The density is computed as:
      rho = p / r / t

    Args:
      p: The thermodynamic pressure, in units of Pa.
      r: The mass specific gas constant, in units of J/kg/K.
      t: The temperature, in units of K.

    Returns:
      The density, in units of kg/m^3.
    """
    return p / r / t

  def _potential_temperature_to_temperature(
      self,
      theta: FlowFieldVal,
      zz: Optional[FlowFieldVal] = None,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Converts the potential temperature to temperature.

    Reference: https://glossary.ametsoc.org/wiki/Potential_temperature

    Args:
      theta: The potential temperature, in units of K.
      zz: The geopotential height, in units of m.
      additional_states: Helper variables including those needed to compute
        reference states.

    Returns:
      The temperature, in units of K.
    """
    # The temperature is the same as the potential temperature if geopotential
    # is not considered.
    if zz is None:
      return theta

    return tf.nest.map_structure(
        lambda theta_i, p_i: theta_i * (p_i / self._p_thermal) ** self.kappa,
        theta,
        self.p_ref(zz, additional_states),
    )

  def temperature_to_potential_temperature(
      self,
      t: FlowFieldVal,
      zz: Optional[FlowFieldVal] = None,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Converts the potential temperature to temperature.

    Reference: https://glossary.ametsoc.org/wiki/Potential_temperature

    Args:
      t: The temperature, in units of K.
      zz: The geopotential height, in units of m.
      additional_states: Helper variables including those needed to compute
        reference states.

    Returns:
      The potential temperature, in units of K.
    """
    # The temperature is the same as the potential temperature if geopotential
    # is not considered.
    if zz is None:
      return t

    return tf.nest.map_structure(
        lambda t_i, p_i: t_i * (p_i / self._p_thermal) ** (-self.kappa),
        t,
        self.p_ref(zz, additional_states),
    )

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
    𝚫T' = 𝚫T / Ts is the fractional temperature drop,
    Hₛ = Rd Ts / g is the density scale height at the surface,
    Hₜ is the height where temperatures drop by 𝚫T.

    Reference:
    CliMa design doc, p 50, Eq. 7.3.

    Args:
      zz: The geopotential height.
      additional_states: Helper variables including those needed to compute the
        reference pressure.

    Returns:
      The reference pressure as a function of height.
    """
    del additional_states


    # Compute the fractional temperature drop.
    delta_t_frac = self._delta_t / self._t_s

    # Compute the density scale height at the surface.
    h_sfc = self.r_d * self._t_s / thermodynamics_utils.G

    def pressure(z: tf.Tensor) -> tf.Tensor:
      """Computes the reference pressure."""
      return self._p_thermal * tf.math.exp(
          -(z + self._height * delta_t_frac *
            (tf.math.log(1.0 - delta_t_frac * tf.math.tanh(z / self._height)) -
             tf.math.log(1.0 + tf.math.tanh(z / self._height)) +
             z / self._height)) / h_sfc / (1.0 - delta_t_frac**2))

    def pressure_const_theta(z: tf.Tensor) -> tf.Tensor:
      """Computes the reference pressure for constant potential temperature."""
      return (self._p_thermal *
              (1.0 - thermodynamics_utils.G * z / self.cp_d / self._const_theta)
              **(1.0 / self.kappa))

    return (
        tf.nest.map_structure(pressure, zz)
        if self._const_theta is None
        else tf.nest.map_structure(pressure_const_theta, zz)
    )

  def t_ref(self, zz: Optional[FlowFieldVal] = None) -> FlowFieldVal:
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
    if zz is None:
      if self._params.use_3d_tf_tensor:
        zz = tf.zeros((self._params.nz, 1, 1), TF_DTYPE)
      else:
        zz = [tf.constant(0, dtype=TF_DTYPE)] * self._params.nz

    def temperature() -> FlowFieldVal:
      """Computes the reference temperature following the presumed profile."""
      return tf.nest.map_structure(
          lambda z: self._t_s - self._delta_t * tf.math.tanh(z / self._height),
          zz,
      )

    def temperature_const_theta() -> FlowFieldVal:
      """Computes reference temperature for constant potential temperature."""
      if self._params.use_3d_tf_tensor:
        theta = self._const_theta * tf.ones((self._params.nz, 1, 1), TF_DTYPE)
      else:
        theta = [self._const_theta] * self._params.nz
      return self._potential_temperature_to_temperature(theta, zz)

    return (
        temperature()
        if self._const_theta is None
        else temperature_const_theta()
    )

  def rho_ref(
      self,
      zz: Optional[FlowFieldVal] = None,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Generates the reference density considering the geopotential.

    Args:
      zz: The geopotential height.
      additional_states: Helper variables including those needed to compute the
        reference density.

    Returns:
      The reference density as a function of height.
    """
    if zz is None:
      if self._params.use_3d_tf_tensor:
        zz = tf.zeros((self._params.nz, 1, 1), TF_DTYPE)
      else:
        zz = [tf.constant(0, dtype=TF_DTYPE)] * self._params.nz

    return tf.nest.map_structure(
        lambda p_ref, t_ref: self.density_by_ideal_gas_law(
            p_ref, self.r_d, t_ref
        ),
        self.p_ref(zz, additional_states),
        self.t_ref(zz),
    )

  def update_density(
      self,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldVal:
    """Updates the density with the ideal gas law."""
    zz = additional_states.get('zz')
    input_3d_tf_tensor = isinstance(list(states.values())[0], tf.Tensor)
    if zz is None:
      if input_3d_tf_tensor:
        zz = tf.zeros((self._params.nz, 1, 1), TF_DTYPE)
      else:
        zz = [tf.constant(0, dtype=TF_DTYPE)] * self._params.nz

    if 'T' in states:
      t = states['T']
    elif 'theta' in states:
      t = self._potential_temperature_to_temperature(states['theta'], zz)
    else:
      raise ValueError(
          'Either temperature or potential temperature is required for the '
          'ideal gas law.'
      )

    scalars = {
        sc_name: thermodynamics_utils.regularize_scalar_bound(states[sc_name])
        for sc_name in self._molecular_weights.keys()
        if sc_name != INERT_SPECIES
    }

    if scalars:
      scalars.update({
          INERT_SPECIES: thermodynamics_utils.compute_ambient_air_fraction(
              scalars
          )
      })
      sc_reg = thermodynamics_utils.regularize_scalar_sum(scalars)
    else:
      sc_reg = {
          INERT_SPECIES: tf.nest.map_structure(
              tf.ones_like, list(states.values())[0]
          )
      }

    mixture_molecular_weight = (
        thermodynamics_utils.compute_mixture_molecular_weight(
            self._molecular_weights, sc_reg))

    return tf.nest.map_structure(
        lambda p_i, w_mix_i, t_i: self.density_by_ideal_gas_law(
            p_i, R_U / w_mix_i, t_i
        ),
        self.p_ref(zz, additional_states),
        mixture_molecular_weight,
        t,
    )
