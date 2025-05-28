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

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
R"""A library for the one-moment microphysics models.

Subscript notations:
l: cloud liquid;
i: cloud ice;
r: rain;
s: snow.

References:
0. 1-moment precipitation microphysics · CloudMicrophysics.jl. (n.d.).
   Retrieved March 24, 2023, from
   https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics1M/
1. Wojciech W Grabowski, Toward cloud resolving modeling of large-scale tropical
   circulations: A simple cloud microphysics parameterization, Journal of the
   Atmospheric Sciences, 55(21), 3283–3298, 1998.
2. John S Marshall, W Mc K Palmer, The distribution of raindrops with size,
   Journal of meteorology, 5(4), 165–166, 1948.
3. Colleen M Kaul, Jo{\~a}o Teixeira, Kentaroh Suzuki, Sensitivities in
   large-eddy simulations of mixed-phase Arctic stratocumulus clouds using a
   simple microphysics approach, Monthly Weather Review, 143(11), 4393–4421,
   2015.
4. Jerry Y Harrington, Michael P Meyers, Robert L Walko, William R Cotton,
   Parameterization of ice crystal conversion processes due to vapor deposition
   for mesoscale models using double-moment basis functions. Part I: Basic
   formulation and parcel model results, Journal of the atmospheric sciences,
   52(23), 4344–4366, 1995.
5. Steven A Rutledge, Peterv Hobbs, The mesoscale and microscale structure and
   organization of clouds and precipitation in midlatitude cyclones. VIII: A
   model for the “seeder-feeder” process in warm-frontal rainbands, Journal of
   the Atmospheric Sciences, 40(5), 1185–1206, 1983.
6. Hugh Morrison, Andrew Gettelman, A new two-moment bulk stratiform cloud
   microphysics scheme in the Community Atmosphere Model, version 3 (CAM3). Part
   I: Description and numerical tests, Journal of Climate, 21(15), 3642–3659,
   2008.
7. Colleen M Kaul, Jo{\~a}o Teixeira, Kentaroh Suzuki, Sensitivities in
   large-eddy simulations of mixed-phase Arctic stratocumulus clouds using a
   simple microphysics approach, Monthly Weather Review, 143(11), 4393–4421,
   2015.
8. Liu, Y. and Hallett, J. (1997), The '1/3' power law between effective radius
   and liquid-water content. Q.J.R. Meteorol. Soc., 123: 1789-1795.
"""

import enum
import functools
from typing import Callable, Optional, Union

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics import constants
from swirl_lm.physics.atmosphere import microphysics_generic
from swirl_lm.physics.atmosphere import microphysics_one_moment_constants as constants_1m
from swirl_lm.physics.atmosphere import microphysics_pb2
from swirl_lm.physics.atmosphere import particles
from swirl_lm.physics.atmosphere import terminal_velocity_chen2022
from swirl_lm.physics.thermodynamics import water
from swirl_lm.utility import types
import tensorflow as tf


Rain = particles.Rain
Snow = particles.Snow
Ice = particles.Ice
Autoconversion = microphysics_pb2.OneMoment.Autoconversion
Accretion = microphysics_pb2.OneMoment.Accretion

TERMINAL_VELOCITY_GAMMA_TYPE = (
    microphysics_pb2.OneMoment.TERMINAL_VELOCITY_GAMMA_TYPE
)
TERMINAL_VELOCITY_POWER_LAW = (
    microphysics_pb2.OneMoment.TERMINAL_VELOCITY_POWER_LAW
)


class Phase(enum.Enum):
  """Defines phases of water."""

  LIQUID = 'l'
  ICE = 'i'
  VAPOR = 'v'


def _v_0(
    particle: Rain | Snow,
    rho: Optional[tf.Tensor] = None,
) -> tf.Tensor:
  """Computes unit terminal velocity."""
  if isinstance(particle, Rain):
    assert rho is not None
    return tf.math.sqrt(
        8.0
        * particle.params.r_0
        * constants.G
        / (3.0 * particle.params.c_d)
        * (constants_1m.RHO_WATER / rho - 1.0)
    )
  elif isinstance(particle, Snow):
    return tf.constant(2.0**2.25 * particle.params.r_0**0.25)
  else:
    raise ValueError(
        f'One of Rain or Snow is required but {type(particle)} was provided.'
    )


class OneMoment(microphysics_generic.Microphysics):
  """A library for the one-moment microphysics models."""

  def __init__(
      self, params: parameters_lib.SwirlLMParameters, water_model: water.Water
  ):
    """Initializes required libraries required by microphysics models."""
    super().__init__(params, water_model)
    assert params.microphysics is not None and params.microphysics.HasField(
        'one_moment'
    ), (
        'The microphysics.one_moment field needs to be set in SwirlLMParameters'
        ' in order to initialize a OneMoment object.'
    )
    one_moment_params = params.microphysics.one_moment
    self._one_moment_params = one_moment_params
    self._rain = Rain.from_config(one_moment_params.rain)
    self._snow = Snow.from_config(one_moment_params.snow)
    self._ice = Ice.from_config(one_moment_params.ice)
    self._aut_coeff = one_moment_params.autoconversion
    self._acc_coeff = one_moment_params.accretion
    self._terminal_velocity_chen2022 = None
    if (
        one_moment_params.terminal_velocity_model_type
        == TERMINAL_VELOCITY_GAMMA_TYPE
    ):
      self._terminal_velocity_chen2022 = (
          terminal_velocity_chen2022.TerminalVelocityChen2022.from_config(
              one_moment_params
          )
      )

  def _saturation(
      self,
      temperature: types.FlowFieldVal,
      rho: types.FlowFieldVal,
      q_v: types.FlowFieldVal,
      q_l: Optional[types.FlowFieldVal] = None,
      q_c: Optional[types.FlowFieldVal] = None,
  ) -> types.FlowFieldVal:
    """Computes the saturation q_v / q_v,sat.

    Args:
      temperature: The temperature of the flow field [K].
      rho: The density of the moist air [kg/m^3].
      q_v: The specific humidity of the gas phase [kg/kg].
      q_l: The specific humidity of the liquid phase [kg/kg].
      q_c: The specific humidity of the condensed phase [kg/kg].

    Returns:
      The saturation q_v / q_v,sat.
    """
    return tf.nest.map_structure(
        tf.math.divide,
        q_v,
        self.water_model.saturation_q_vapor(temperature, rho, q_l, q_c),
    )

  def _conduction_and_diffusion(
      self,
      particle: Union[Rain, Snow],
      temperature: types.FlowFieldVal,
      q_l: Optional[types.FlowFieldVal] = None,
      q_c: Optional[types.FlowFieldVal] = None,
  ) -> types.FlowFieldVal:
    """Computes the combined effect of thermal conduction and water diffusion.

    Args:
      particle: A Rain or Snow dataclass object that stores constant parameters
        for microphysics processes.
      temperature: The temperature of the flow field [K].
      q_l: The specific humidity of the liquid phase [kg/kg].
      q_c: The specific humidity of the condensed phase [kg/kg].

    Returns:
      The rate of change of snow due to autoconversion.
    """
    lh = (
        self.water_model.lh_s(temperature)
        if isinstance(particle, Snow)
        else self.water_model.lh_v(temperature)
    )

    def conduction_and_diffusion_fn(
        lh: tf.Tensor,
        temperature: tf.Tensor,
        p_sat: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the effects of thermal conductivity and water diffusivity."""
      r_v = self.water_model.r_v
      return 1.0 / (
          lh
          / (constants_1m.K_COND * temperature)
          * (lh / (r_v * temperature) - 1.0)
          + (r_v * temperature) / (p_sat * constants_1m.D_VAP)
      )

    return tf.nest.map_structure(
        conduction_and_diffusion_fn,
        lh,
        temperature,
        self.water_model.saturation_vapor_pressure(temperature, q_l, q_c),
    )

  def _accretion(
      self,
      particle: Union[Rain, Snow],
      rho: types.FlowFieldVal,
      q_l: types.FlowFieldVal,
      q_i: types.FlowFieldVal,
      q_p: types.FlowFieldVal,
  ) -> types.FlowFieldVal:
    """Computes the accretion source term due to condensed cloud water.

    Note that a positive sign is imposed in front of this term because it is
    considered as a source term for the precipitation (rain/snow). It needs to
    be subtracted from the specific humidity of the cloud.

    Args:
      particle: A Rain or Snow dataclass object that stores constant parameters
        for microphysics processes.
      rho: The density of the moist air [kg/m^3].
      q_l: The specific humidity of the liquid phase [kg/kg].
      q_i: The specific humidity of the ice phase [kg/kg].
      q_p: The precipitation water mass fraction, which can be rain (q_r) or
        snow (q_s). [kg/kg].

    Returns:
      The rate of change of specific humidity of the precipitation (rain/snow)
      due to the collision with condensed phase water.
    """
    coeff = particle.params
    pi_av_coeff = particle.a_0 * coeff.chi_a * coeff.chi_v
    sigma_av = coeff.a_e + coeff.v_e + coeff.del_a + coeff.del_v

    e_cp_l = (
        self._acc_coeff.e_lr
        if isinstance(particle, Rain)
        else self._acc_coeff.e_ls
    )
    e_cp_i = (
        self._acc_coeff.e_ir
        if isinstance(particle, Rain)
        else self._acc_coeff.e_is
    )

    lam = particles.marshall_palmer_distribution_parameter_lambda(
        particle, rho, q_p
    )

    def accretion_fn(
        rho: tf.Tensor,
        q_l: tf.Tensor,
        q_i: tf.Tensor,
        q_p: tf.Tensor,
        lam: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the accretion rate."""
      return tf.math.divide_no_nan(
          particles.n_0(particle, rho, q_p)
          * pi_av_coeff
          * _v_0(particle, rho)
          * (q_l * e_cp_l + q_i * e_cp_i)
          * particles.gamma(sigma_av + 1.0),
          lam,
      ) * tf.math.pow(tf.math.divide_no_nan(1.0, coeff.r_0 * lam), sigma_av)

    return tf.nest.map_structure(accretion_fn, rho, q_l, q_i, q_p, lam)

  def _autoconversion(
      self,
      q_threshold: float,
      tau: float,
      q: types.FlowFieldVal,
  ) -> types.FlowFieldVal:
    """Computes the increase rate of precipitation due to autoconversion.

    Args:
      q_threshold: Threshold for conversion in kg/kg.
      tau: Timescale for autoconversion in seconds.
      q: The specific humidity of cloud water. If `coeff` is `Rain`, `q` should
        be for the liquid phase (`q_l`); if `coeff` is `Snow`, `q` should be the
        ice phase (`q_i`).

    Returns:
      The rate of increase of precipitation (rain/snow) due to autoconversion.
    """
    return tf.nest.map_structure(
        lambda q_c: tf.maximum(q_c - q_threshold, 0.0) / tau, q
    )

  def _autoconversion_snow(
      self,
      temperature: types.FlowFieldVal,
      rho: types.FlowFieldVal,
      q_v: types.FlowFieldVal,
      q_l: types.FlowFieldVal,
      q_c: types.FlowFieldVal,
  ) -> types.FlowFieldVal:
    """Computes the increase rate of precipitation due to snow autoconversion.

    Args:
      temperature: The temperature of the flow field [K].
      rho: The density of the moist air [kg/m^3].
      q_v: The specific humidity of the gas phase [kg/kg].
      q_l: The specific humidity of the liquid phase [kg/kg].
      q_c: The specific humidity of the condensed phase [kg/kg].

    Returns:
      The rate of increase of snow due to autoconversion.
    """

    s = self._saturation(temperature, rho, q_v, q_l, q_c)

    g = self._conduction_and_diffusion(self._snow, temperature, q_l, q_c)

    q_i = tf.nest.map_structure(tf.math.subtract, q_c, q_l)
    lam = particles.marshall_palmer_distribution_parameter_lambda(
        self._ice, rho, q_i
    )

    def snow_autoconversion_fn(
        rho: tf.Tensor,
        s: tf.Tensor,
        g: tf.Tensor,
        lam: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the autoconversion of snow."""
      r_is = self._aut_coeff.r_is
      aut = (
          4.0
          * np.pi
          / rho
          * (s - 1.0)
          * g
          * particles.n_0(self._ice)
          * tf.math.exp(-lam * r_is)
          * (
              r_is**2 / (self._ice.params.m_e + self._ice.params.del_m)
              + tf.math.divide_no_nan((r_is * lam + 1.0), lam**2)
          )
      )
      return tf.math.maximum(aut, 0.0)

    return tf.nest.map_structure(snow_autoconversion_fn, rho, s, g, lam)

  def _ventilation_factor(
      self,
      particle: Union[Rain, Snow],
      rho: tf.Tensor,
      lam: tf.Tensor,
  ) -> types.FlowFieldVal:
    """Computes the ventilation factor for the given particle.

    Args:
      particle: A Rain or Snow dataclass object that stores constant parameters
        for microphysics processes.
      rho: The density of the moist air [kg/m^3].
      lam: The Marshall Palmer distribution parameter lambda.

    Returns:
      The ventilation factor for the given particle.
    """
    coeff = particle.params
    return coeff.a_vent + coeff.b_vent * (
        constants_1m.NU_AIR / constants_1m.D_VAP
    ) ** (1.0 / 3.0) * tf.math.divide_no_nan(1.0, coeff.r_0 * lam) ** (
        0.5 * (coeff.v_e + coeff.del_v)
    ) * tf.math.sqrt(
        tf.math.divide_no_nan(
            2.0 * coeff.chi_v * _v_0(particle, rho),
            (constants_1m.NU_AIR * lam),
        )
    ) * particles.gamma(
        0.5 * (coeff.v_e + coeff.del_v + 5.0)
    )

  def evaporation_sublimation(
      self,
      particle: Union[Rain, Snow],
      temperature: types.FlowFieldVal,
      rho: types.FlowFieldVal,
      q_v: types.FlowFieldVal,
      q_p: types.FlowFieldVal,
      q_l: Optional[types.FlowFieldVal] = None,
      q_c: Optional[types.FlowFieldVal] = None,
  ) -> types.FlowFieldVal:
    """Computes the rate of change of precipitation by evaporation/sublimation.

    Note that for the case of rain we only consider evaporation (s - 1 < 0). For
    the case of snow we consider both the source term due to vapor deposition on
    snow (s - 1 > 0) and the sink due to vapor sublimation (s - 1 < 0).

    Args:
      particle: A Rain or Snow dataclass object that stores constant parameters
        for microphysics processes.
      temperature: The temperature of the flow field [K].
      rho: The density of the moist air [kg/m^3].
      q_v: The specific humidity of the gas phase [kg/kg].
      q_p: The precipitation water mass fraction, which can be rain (q_r) or
        snow (q_s) [kg/kg].
      q_l: The specific humidity of the liquid phase [kg/kg].
      q_c: The specific humidity of the condensed phase [kg/kg].

    Returns:
      The rate of evaporation/sublimation. If the rate is > 0, then the amount
      of rain or snow is decreasing and the amount of water vapor is increasing.
      Note that deposition is also considered so rate can be < 0 in the case
      of snow.
    """
    if isinstance(particle, Rain):
      s = self._saturation(temperature, rho, q_v, q_l, q_l)
    else:
      q_i = tf.nest.map_structure(tf.math.subtract, q_c, q_l)
      zeros = tf.nest.map_structure(tf.zeros_like, q_i)
      s = self._saturation(temperature, rho, q_v, zeros, q_i)

    g = self._conduction_and_diffusion(particle, temperature, q_l, q_c)
    lam = particles.marshall_palmer_distribution_parameter_lambda(
        particle, rho, q_p
    )

    def evap_subl_fn(
        rho: tf.Tensor,
        q_p: tf.Tensor,
        s: tf.Tensor,
        g: tf.Tensor,
        lam: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the evaporation/sublimation rate."""
      f_vent = self._ventilation_factor(particle, rho, lam)
      evap_subl = (
          tf.math.divide_no_nan(
              -4.0
              * np.pi
              * particles.n_0(particle, rho, q_p)
              / rho
              * (s - 1.0)
              * g,
              lam**2,
          )
          * f_vent
      )
      if isinstance(particle, Rain):
        evap_subl = tf.where(
            tf.math.less(s, 1.0), evap_subl, tf.zeros_like(evap_subl)
        )
      return evap_subl

    return tf.nest.map_structure(evap_subl_fn, rho, q_p, s, g, lam)

  def evaporation(
      self,
      temperature: types.FlowFieldVal,
      rho: types.FlowFieldVal,
      q_v: types.FlowFieldVal,
      q_r: types.FlowFieldVal,
      q_s: types.FlowFieldVal,
      q_l: Optional[types.FlowFieldVal] = None,
      q_c: Optional[types.FlowFieldVal] = None,
  ) -> types.FlowFieldVal:
    """Computes the change rate of cloud water by evaporation and sublimation.

    Args:
      temperature: The temperature of the flow field [K].
      rho: The density of the moist air [kg/m^3].
      q_v: The specific humidity of the gas phase [kg/kg].
      q_r: The rain water mass fraction. [kg/kg].
      q_s: The snow water mass fraction. [kg/kg].
      q_l: The specific humidity of the liquid phase [kg/kg].
      q_c: The specific humidity of the condensed phase [kg/kg].

    Returns:
      The rate of change of cloud water due to evaporation and sublimation.
    """
    evap = self.evaporation_sublimation(
        self._rain, temperature, rho, q_v, q_r, q_l, q_c
    )
    subl = self.evaporation_sublimation(
        self._snow, temperature, rho, q_v, q_s, q_l, q_c
    )

    return tf.nest.map_structure(lambda e, s: e + s, evap, subl)

  def snow_melt(
      self,
      snow: Snow,
      temperature: types.FlowFieldVal,
      rho: types.FlowFieldVal,
      q_s: types.FlowFieldVal,
  ) -> types.FlowFieldVal:
    """Computes the melting rate from snow to rain.

    Args:
      snow: A Snow dataclass object that stores constant parameters for
        microphysics processes.
      temperature: The temperature of the flow field [K].
      rho: The density of the moist air [kg/m^3].
      q_s: The snow water mass fraction [kg/kg].

    Returns:
      The melting rate for conversion from snow to rain.
    """
    lam = particles.marshall_palmer_distribution_parameter_lambda(
        snow, rho, q_s
    )

    def melt_fn(
        rho: tf.Tensor,
        q_s: tf.Tensor,
        t: tf.Tensor,
        lam: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the melt rate."""
      f_vent = self._ventilation_factor(snow, rho, lam)
      # Note that a lower bound of 0 is added to the temperature difference term
      # to avoid negative rates.
      melt = (
          tf.math.divide_no_nan(
              4.0
              * np.pi
              * particles.n_0(snow, rho, q_s)
              * constants_1m.K_COND
              / self.water_model.lh_f(t)
              / rho
              * tf.math.maximum(t - self.water_model.t_freeze, 0.0),
              lam**2,
          )
          * f_vent
      )
      return tf.where(tf.math.greater(q_s, 0.0), melt, tf.zeros_like(q_s))

    return tf.nest.map_structure(melt_fn, rho, q_s, temperature, lam)

  def autoconversion_and_accretion(
      self,
      particle: Union[Rain, Snow],
      temperature: types.FlowFieldVal,
      rho: types.FlowFieldVal,
      q_v: types.FlowFieldVal,
      q_p: types.FlowFieldVal,
      q_l: Optional[types.FlowFieldVal],
      q_c: Optional[types.FlowFieldVal],
  ) -> types.FlowFieldVal:
    """Computes the change of precipitation by autoconversion and accretion.

    Args:
      particle: A Rain or Snow dataclass object that stores constant parameters
        for microphysics processes.
      temperature: The temperature of the flow field [K].
      rho: The density of the moist air [kg/m^3].
      q_v: The specific humidity of the gas phase [kg/kg].
      q_p: The precipitation water mass fraction, which can be rain (q_r) or
        snow (q_s). [kg/kg].
      q_l: The specific humidity of the liquid phase [kg/kg].
      q_c: The specific humidity of the condensed phase [kg/kg].

    Returns:
      The rate of change of precipitation due to autoconversion and accretion.
    """
    q_i = tf.nest.map_structure(tf.math.subtract, q_c, q_l)
    acc = self._accretion(particle, rho, q_l, q_i, q_p)

    aut_coeff = Autoconversion()

    if isinstance(particle, Rain):
      aut = self._autoconversion(aut_coeff.q_l_threshold, aut_coeff.tau_lr, q_l)
    elif isinstance(particle, Snow):
      aut = self._autoconversion(aut_coeff.q_i_threshold, aut_coeff.tau_is, q_i)
    else:
      raise ValueError(
          f'Autoconversion and accretion not supported for {type(particle)}.'
      )

    return tf.nest.map_structure(tf.math.add, acc, aut)

  def _terminal_velocity_power_law(
      self,
      varname: str,
      rho: types.FlowFieldVal,
      q: types.FlowFieldVal,
  ) -> types.FlowFieldVal:
    """Computes the bulk terminal velocity using a power law parameterization.

    Args:
      varname: The name of the humidity variable, either `q_r`, `q_s`, `q_l`,
        or `q_i`.
      rho: The density of the moist air [kg/m^3].
      q: The humidity variable mass fraction, which can be rain (q_r), snow
        (q_s), cloud water (q_l), or cloud ice (q_i) [kg/kg].

    Returns:
      The terminal velocity of rain/snow.
    """
    # Treat sedimentation of cloud droplet and ice crystal as rain and snow,
    # respectively.
    if varname in ('q_r', 'q_l'):
      particle = self._rain
    elif varname in ('q_i', 'q_s'):
      particle = self._snow
    else:
      raise ValueError(f'Terminal velocity not supported for {varname}')
    lam = particles.marshall_palmer_distribution_parameter_lambda(
        particle, rho, q
    )
    coeff = particle.params

    def terminal_velocity_fn(rho: tf.Tensor, lam: tf.Tensor):
      """Computes the terminal velocity."""
      return (
          coeff.chi_v
          * _v_0(particle, rho)
          * tf.math.divide_no_nan(1.0, (coeff.r_0 * lam))
          ** (coeff.v_e + coeff.del_v)
          * particles.gamma(
              coeff.m_e + coeff.v_e + coeff.del_m + coeff.del_v + 1.0
          )
          / particles.gamma(coeff.m_e + coeff.del_m + 1.0)
      )

    return tf.nest.map_structure(terminal_velocity_fn, rho, lam)

  def _terminal_velocity_gamma_type(
      self,
      varname: str,
      rho: types.FlowFieldVal,
      q: types.FlowFieldVal,
  ) -> types.FlowFieldVal:
    """Computes the bulk terminal velocity using a gamma-type parameterization.

    Args:
      varname: The name of the humidity variable, either `q_r`, `q_s`, `q_l`,
        or `q_i`.
      rho: The density of the moist air [kg/m^3].
      q: The humidity variable mass fraction, which can be rain (q_r), snow
        (q_s), cloud water (q_l), or cloud ice (q_i) [kg/kg].

    Returns:
      The terminal velocity of rain/snow.
    """
    assert self._terminal_velocity_chen2022 is not None, (
        'Gamma-type terminal velocity model must be set in one-moment'
        ' microphysics scheme.'
    )
    if varname == 'q_r':
      terminal_vel_fn = self._terminal_velocity_chen2022.rain_terminal_velocity
    elif varname == 'q_s':
      terminal_vel_fn = self._terminal_velocity_chen2022.snow_terminal_velocity
    elif varname == 'q_l':
      terminal_vel_fn = functools.partial(
          self._terminal_velocity_chen2022.condensate_terminal_velocity,
          self._rain,
      )
    elif varname == 'q_i':
      terminal_vel_fn = functools.partial(
          self._terminal_velocity_chen2022.condensate_terminal_velocity,
          self._ice,
      )
    else:
      raise ValueError(f'Terminal velocity not supported for {varname}')

    return tf.nest.map_structure(terminal_vel_fn, rho, q)

  def terminal_velocity(
      self,
      varname: str,
      rho: types.FlowFieldVal,
      q: types.FlowFieldVal,
  ) -> types.FlowFieldVal:
    """Computes the terminal velocity of precipitation or sedimentation.

    Args:
      varname: The name of the humidity variable, either `q_r`, `q_s`, `q_l`,
        or `q_i`.
      rho: The density of the moist air [kg/m^3].
      q: The precipitation water mass fraction, which can be rain (q_r) or
        snow (q_s) [kg/kg].

    Returns:
      The terminal velocity of rain/snow.
    """
    if (
        self._one_moment_params.terminal_velocity_model_type
        == TERMINAL_VELOCITY_GAMMA_TYPE
    ):
      return self._terminal_velocity_gamma_type(varname, rho, q)
    elif (
        self._one_moment_params.terminal_velocity_model_type
        == TERMINAL_VELOCITY_POWER_LAW
    ):
      return self._terminal_velocity_power_law(varname, rho, q)
    else:
      raise ValueError(
          f'{self._one_moment_params.terminal_velocity_model_type} '
          'not supported in one-moment microphysics scheme.'
      )


class Adapter(microphysics_generic.MicrophysicsAdapter):
  """Interface between the one-moment model and rest of Swirl-LM."""

  _one_moment: OneMoment

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
      water_model: water.Water,
  ):
    assert params.microphysics is not None and params.microphysics.HasField(
        'one_moment'
    ), (
        'The microphysics.one_moment field needs to be set in SwirlLMParameters'
        ' in order to initialize a one_moment.Adapter object.'
    )
    self._one_moment_params = params.microphysics.one_moment
    self._one_moment = OneMoment(params, water_model)
    self._water_model = water_model
    self._rain = Rain.from_config(self._one_moment_params.rain)
    self._snow = Snow.from_config(self._one_moment_params.snow)
    self._dt = params.dt

  def _clip_temperature_and_adjust_density(
      self,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
      thermo_states: types.FlowFieldMap,
  ) -> dict[str, types.FlowFieldVal]:
    """Applies an upper bound to the temperature and recomputes density."""
    if not self._one_moment_params.HasField('temperature_max'):
      return {'T': thermo_states['T'], 'rho': states['rho_thermal']}

    temperature = tf.nest.map_structure(
        lambda t: tf.math.minimum(
            t, self._one_moment_params.temperature_max * tf.ones_like(t)
        ),
        thermo_states['T'],
    )
    rho = self._water_model.saturation_density(
        'T',
        temperature,
        states['q_t'],
        states['u'],
        states['v'],
        states['w'],
        zz=thermo_states.get('zz', None),
        additional_states=additional_states,
    )
    return {'T': temperature, 'rho': rho}

  def terminal_velocity(
      self,
      varname: str,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldVal:
    """Computes the terminal velocity for `q_r` or `q_s`.

    Args:
      varname: The name of the humidity variable, either `q_r`, `q_s`,
        `q_l`, or `q_i`.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.

    Returns:
      The terminal velocity for `q_r` or `q_s`.

    Raises:
      NotImplementedError If `varname` is not one of `q_r` or `q_s`.
    """
    del additional_states

    assert varname in (
        'q_r',
        'q_s',
        'q_l',
        'q_i',
    ), (
        'Terminal velocity is for `q_r`, `q_s`, `q_l`, or `q_i` only, but'
        f' {varname} is provided.'
    )

    return self._one_moment.terminal_velocity(
        varname, states['rho_thermal'], states[varname]
    )

  def _humidity_source_for_rain_and_snow(
      self,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
      thermo_states: types.FlowFieldMap,
      particle: Union[Rain, Snow],
      include_autoconversion_and_accretion: bool,
  ) -> tuple[types.FlowFieldVal, types.FlowFieldVal]:
    """Computes humidity source (clipped to (1 - k) * q / dt) and melt term.

    Physically q cannot be negative, but instead of limiting the source such
    that

      q + source * dt >= 0

    we limit the source to reduce q to a fraction k of its current value (to
    leave room for convection, etc.).

      q + source * dt >= kq
      source >= -(1 - k)q / dt

    Args:
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.
      thermo_states: A dictionary that holds all thermodynamics variables.
      particle: A Rain, Snow dataclass object that stores constant parameters.
      include_autoconversion_and_accretion: Whether to include
        autoconversion and accretion in the computation.

    Returns:
      The clipped source term and the snow melt term.
    """
    validated_thermo_states = self._clip_temperature_and_adjust_density(
        states, additional_states, thermo_states
    )

    q_p = (
        thermo_states.get('q_r', states['q_r']) if isinstance(particle, Rain)
        else thermo_states.get('q_s', states['q_s'])
    )
    aut_and_acc = self._one_moment.autoconversion_and_accretion(
        particle,
        validated_thermo_states['T'],
        validated_thermo_states['rho'],
        thermo_states['q_v'],
        q_p,
        thermo_states['q_l'],
        thermo_states['q_c'],
    )
    evap_subl = self._one_moment.evaporation_sublimation(
        particle,
        validated_thermo_states['T'],
        validated_thermo_states['rho'],
        thermo_states['q_v'],
        q_p,
        thermo_states['q_l'],
        thermo_states['q_c'],
    )
    melt = self._one_moment.snow_melt(
        self._snow,
        validated_thermo_states['T'],
        validated_thermo_states['rho'],
        thermo_states.get('q_s', states['q_s']),
    )
    if isinstance(particle, Snow):
      melt = tf.nest.map_structure(tf.math.negative, melt)

    def _add_and_clip(q_p, q_t, aut_and_acc, evap_subl, melt):
      # Clip the humidity source such that
      #
      #   q_p + dt * src >= k * q_p and
      #   q_t - dt * src >= k * q_t
      #
      # Solving these inequalities for src, we get:
      #
      #   -(1 - k) * q_p / dt <= src <= (1 - k) * q_t / dt
      #
      # In case q_p or q_t ends up being negative for some other reason, we also
      # clip the input q_p/q_t to be >= 0. Otherwise, the clipping on the
      # humidity source will introduce an artificial positive source for
      # negative values potentially masking other problems.
      if include_autoconversion_and_accretion:
        source = aut_and_acc - evap_subl
      else:
        source = -evap_subl
      k = self._one_moment_params.humidity_source_term_limiter_k
      source_total = tf.clip_by_value(
          source + melt,
          (-(1 - k) * tf.maximum(q_p, 0.0) / self._dt),
          ((1 - k) * tf.maximum(q_t, 0.0) / self._dt),
      )
      return source_total - melt

    return (
        tf.nest.map_structure(
            _add_and_clip,
            q_p,
            thermo_states.get('q_t', states['q_t']),
            aut_and_acc,
            evap_subl,
            melt,
        ),
        melt,
    )

  def humidity_source_fn(
      self,
      varname: str,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
      thermo_states: types.FlowFieldMap,
  ) -> types.FlowFieldVal:
    """Computes the source term in a humidity equation.

    Supported types of humidity are `q_t`, `q_r`, and `q_s`.

    Args:
      varname: The name of the humidity variable, which should be one of `q_r`,
        `q_s`, and `q_t`.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.
      thermo_states: A dictionary that holds all thermodynamics variables.

    Returns:
      The source term in a humidity equation due to microphysics.

    Raises:
      NotImplementedError If `varname` is not one of `q_t`, `q_r`, or `q_s`.
    """
    q_v = tf.nest.map_structure(
        tf.math.subtract, thermo_states['q_t'], thermo_states['q_c']
    )

    if varname == 'q_r':
      src, melt = self._humidity_source_for_rain_and_snow(
          states,
          additional_states,
          thermo_states | {'q_v': q_v},
          self._rain,
          include_autoconversion_and_accretion=True,
      )
      return tf.nest.map_structure(tf.math.add, src, melt)
    elif varname == 'q_s':
      src, melt = self._humidity_source_for_rain_and_snow(
          states,
          additional_states,
          thermo_states | {'q_v': q_v},
          self._snow,
          include_autoconversion_and_accretion=True,
      )
      return tf.nest.map_structure(tf.math.add, src, melt)
    elif varname == 'q_t':
      src_rain, _ = self._humidity_source_for_rain_and_snow(
          states,
          additional_states,
          thermo_states | {'q_v': q_v},
          self._rain,
          include_autoconversion_and_accretion=True,
      )
      src_snow, _ = self._humidity_source_for_rain_and_snow(
          states,
          additional_states,
          thermo_states | {'q_v': q_v},
          self._snow,
          include_autoconversion_and_accretion=True,
      )
      return tf.nest.map_structure(
          lambda src_r, src_s: tf.math.negative(src_r + src_s),
          src_rain,
          src_snow,
      )
    else:
      raise NotImplementedError(
          f'{varname} is not a valid humidity type for the 1-moment'
          ' microphysics. Available options are: `q_t`, `q_r`, and `q_s`.'
      )

  def potential_temperature_source_fn(
      self,
      varname: str,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
      thermo_states: types.FlowFieldMap,
  ) -> types.FlowFieldVal:
    """Computes the source term in a potential temperature equation.

    Supported types of potential temperature are `theta` and `theta_li`.

    Args:
      varname: The name of the potential temperature variable, either `theta` or
        `theta_li`.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.
      thermo_states: A dictionary that holds all thermodynamics variables.

    Returns:
      The source term in a potential temperature equation due to microphysics.

    Raises:
      AssertionError If `varname` is not one of `theta` or `theta_li`.
    """
    assert varname in ('theta', 'theta_li'), (
        'Only `theta` and `theta_li` are supported for the microphysics source'
        f' term computation, but `{varname}` is provided.'
    )

    validated_thermo_states = self._clip_temperature_and_adjust_density(
        states, additional_states, thermo_states
    )

    source = tf.nest.map_structure(tf.zeros_like, states['rho'])

    phase_params = (
        (self._rain, self._one_moment.water_model.lh_v0),
        (self._snow, self._one_moment.water_model.lh_s0),
    )

    cp = self._one_moment.water_model.cp_m(
        thermo_states['q_t'], thermo_states['q_l'], thermo_states['q_i']
    )

    def energy_source_fn(
        lh: float,
    ) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
      """Generates a function that computes the energy source term."""

      def source_fn(
          rho: tf.Tensor,
          cp: tf.Tensor,
          exner_inv: tf.Tensor,
          water_source: tf.Tensor,
      ) -> tf.Tensor:
        """Converts water conversion rate to an energy source term."""
        return rho * lh / cp * exner_inv * water_source

      return source_fn

    for particle, lh in phase_params:
      # Calculate source terms for vapor and liquid conversions, respectively.
      # Use that c_{q_v->q_r/s} = -c_{q_r/s->q_v}, i.e., the negation of the
      # evaporation/sublimation rate.
      water_source, melt = self._humidity_source_for_rain_and_snow(
          states,
          additional_states,
          thermo_states,
          particle,
          include_autoconversion_and_accretion=(varname == 'theta_li'),
      )

      # Converts water mass fraction conversion rates to an energy source term.
      exner_inv = self._one_moment.water_model.exner_inverse(
          validated_thermo_states['rho'],
          thermo_states['q_t'],
          validated_thermo_states['T'],
          thermo_states['zz'],
          additional_states,
      )
      theta_source = tf.nest.map_structure(
          energy_source_fn(lh),
          states['rho'],
          cp,
          exner_inv,
          water_source,
      )

      # Add the melt term to the source term. Because the melt has the same
      # magnitude for rain and snow, with snow having a negative sign, we only
      # add the melt term for snow.
      if isinstance(particle, Snow):
        melt_heat_src_fn = (
            lambda rho, lh, cp, exner_inv, melt: rho
            * lh
            / cp
            * exner_inv
            * melt
        )
        melt_heat_sink = tf.nest.map_structure(
            melt_heat_src_fn,
            states['rho'],
            self._water_model.lh_f(validated_thermo_states['T']),
            cp,
            exner_inv,
            melt,
        )
        theta_source = tf.nest.map_structure(
            tf.math.add, theta_source, melt_heat_sink
        )

      source = tf.nest.map_structure(tf.math.add, source, theta_source)

    return source

  def total_energy_source_fn(
      self,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
      thermo_states: types.FlowFieldMap,
  ) -> types.FlowFieldVal:
    """Computes the source term in the total energy equation.

    Args:
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.
      thermo_states: A dictionary that holds all thermodynamics variables.

    Returns:
      The source term in the total energy equation due to microphysics.
    """
    validated_thermo_states = self._clip_temperature_and_adjust_density(
        states, additional_states, thermo_states
    )

    # Get potential energy.
    pe = tf.nest.map_structure(
        lambda zz_i: constants.G * zz_i, thermo_states['zz']
    )

    # Get conversion rates and energy source from cloud water/ice to rain/snow.
    source_li = tf.nest.map_structure(tf.zeros_like, states['rho'])
    particle_types = [self._rain, self._snow]
    q_p = [thermo_states['q_r'], thermo_states['q_s']]
    e_int = [thermo_states['e_l'], thermo_states['e_i']]

    for particle, q, e in zip(particle_types, q_p, e_int):
      aut_acc = self._one_moment.autoconversion_and_accretion(
          particle,
          validated_thermo_states['T'],
          validated_thermo_states['rho'],
          thermo_states['q_v'],
          q,
          thermo_states['q_l'],
          thermo_states['q_c'],
      )
      source_li = tf.nest.map_structure(
          lambda src, e, pe, rho, c_li_rs: src + (e + pe) * rho * c_li_rs,
          source_li,
          e,
          pe,
          states['rho'],
          aut_acc,
      )

    # Calculate source terms for vapor and liquid conversions, respectively.
    # Use that c_{q_v->q_l} = -c_{q_l->q_v}, i.e. minus the evaporation
    # rate.
    evap_subl = self._one_moment.evaporation(
        validated_thermo_states['T'],
        validated_thermo_states['rho'],
        thermo_states['q_v'],
        thermo_states['q_r'],
        thermo_states['q_s'],
        thermo_states['q_l'],
        thermo_states['q_c'],
    )
    source_v = tf.nest.map_structure(
        lambda e, pe, rho, c_lv: (e + pe) * rho * (-c_lv),
        thermo_states['e_v'],
        pe,
        states['rho'],
        evap_subl,
    )

    # TODO(wqing): Add the source term for snow melt.

    return tf.nest.map_structure(tf.math.add, source_v, source_li)

  def condensation(
      self,
      rho: water.FlowFieldVal,
      temperature: water.FlowFieldVal,
      q_v: water.FlowFieldVal,
      q_l: water.FlowFieldVal,
      q_c: water.FlowFieldVal,
      zz: Optional[water.FlowFieldVal] = None,
      additional_states: Optional[water.FlowFieldMap] = None,
  ) -> water.FlowFieldVal:
    """Computes the condensation rate using Bryan & Fritsch (2002).

    Args:
      rho: The moist air density, in kg/m^3.
      temperature: The temperature.
      q_v: The cloud vapor fraction (kg/kg).
      q_l: The specific humidity of the cloud liquid phase (kg/kg).
      q_c: The specific humidity of the cloud humidity condensed phase,
        including ice and liquid (kg/kg).
      zz: The vertical coordinates (m). Not used.
      additional_states: Helper variables including those needed to compute
        reference states. Not used.

    Returns:
      The condensation rate.
    """
    return self._one_moment.condensation_bf2002(
        rho,
        temperature,
        q_v,
        q_l,
        q_c,
        zz,
        additional_states,
    )

  def cloud_particle_effective_radius(
      self,
      rho: types.FlowFieldVal,
      q_c: types.FlowFieldVal,
      phase: str,
  ) -> types.FlowFieldVal:
    """Computes the 1-moment approximation of cloud particle effective radius.

    This follows the formulation from Liu and Hallett (1997) equation 8. The
    concentration of cloud particles is assumed to be constant and the 1/3 Power
    Law between effective radius and water content is used. Particles are
    assumed to be spherical for both liquid and ice, which is an
    oversimplification for ice because ice crystal shapes can be complex. The
    same asymmetry factor is used for liquid and ice.

    Args:
      rho: The density of the moist air [kg/m^3].
      q_c: The condensed-phase specific humidity [kg/kg]. Note that this is the
        specific humidity of ice or liquid and not their sum.
      phase: The phase of the cloud particles (whether 'solid' or 'liquid').

    Returns:
      The effective radius (m) of the cloud droplets or ice particles.
    """
    assert phase in (
        Phase.ICE.value,
        Phase.LIQUID.value,
    ), 'Effective radius calculation is only valid for liquid or ice.'

    # Density of the condensate.
    rho_c = (
        constants_1m.RHO_WATER
        if phase == Phase.LIQUID.value
        else constants_1m.RHO_ICE
    )

    def r_eff_fn(rho: tf.Tensor, q_c: tf.Tensor) -> tf.Tensor:
      """Equation 8 from Liu and hallett (1997)."""
      alpha = (4.0 / 3.0 * np.pi * rho_c * constants_1m.ASYMMETRY_CLOUD) ** (
          -1 / 3
      )
      return alpha * (rho * q_c / constants_1m.DROPLET_N) ** (1 / 3)

    return tf.nest.map_structure(r_eff_fn, rho, q_c)
