# Copyright 2023 The swirl_lm Authors.
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
"""

import dataclasses
from typing import Callable, Optional

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics import constants
from swirl_lm.physics.atmosphere import microphysics_generic
from swirl_lm.utility import types
import tensorflow as tf

# The density of liquid water [kg/m^3].
RHO_WATER = 1e3
# The density of crystal ice [kg/m^3].
RHO_ICE = 0.917e3
# The density of typical air [kg/m^3].
RHO_AIR = 1.0
# The thermal conductivity of air [J/(m s K)].
K_COND = 2.4e-2
# The kinematic visocity of air [m^2/s].
NU_AIR = 1.6e-5
# The molecular diffusivity of water vapor [m^2/s].
D_VAP = 2.26e-5


def _gamma(x: float | tf.Tensor) -> tf.Tensor:
  """Computes the Gamma function of x."""
  with tf.control_dependencies(
      [
          tf.assert_greater(
              x, 0.0, message='The Gamma function takes positive inputs only.'
          )
      ]
  ):
    return tf.math.exp(tf.math.lgamma(x))


def compute_sphere_mass(rho: float, r: float) -> float:
  """Computes the mass of a sphere."""
  return 4.0 / 3.0 * np.pi * rho * r**3


@dataclasses.dataclass(init=False, frozen=True)
class Ice:
  """Constants for ice related quantities."""

  # The density of an ice crystal [kg/m^3].
  rho: float = RHO_ICE

  # Typical ice crystal radius [m].
  r_0: float = 1e-5

  # Unit mass of an ice crystal [kg].
  m_0: float = compute_sphere_mass(RHO_ICE, r_0)
  # Exponent to the radius ratio in the mass equation.
  m_e: float = 3.0
  # The calibration coefficients in the mass equation.
  chi_m: float = 1.0
  del_m: float = 0.0


@dataclasses.dataclass(init=False, frozen=True)
class Rain:
  """Constants for rain related quantities."""

  # The density of a rain drop [kg/m^3].
  rho: float = RHO_WATER

  # The drag coefficient of a rain drop [1].
  c_d: float = 0.55

  # Typical rain drop radius [m].
  r_0: float = 1e-3

  # Unit mass of a rain drop [kg].
  m_0: float = compute_sphere_mass(rho, r_0)
  # Exponent to the radius ratio in the mass equation.
  m_e: float = 3.0
  # The calibration coefficients in the mass equation.
  chi_m: float = 1.0
  del_m: float = 0.0

  # Unit cross section area of a rain drop [m^2].
  a_0: float = np.pi * r_0**2
  # Exponent to the radius ratio in the cross section area equation.
  a_e: float = 2.0
  # The calibration coefficients in the cross section area equation.
  chi_a: float = 1.0
  del_a: float = 0.0

  # Exponent to the radius ratio in the terminal velocity equation.
  v_e: float = 0.5
  # The calibration coefficients in the terminal velocity equation.
  chi_v: float = 1.0
  del_v: float = 0.0

  # The ventilation factor coefficients [1].
  a_vent: float = 1.5
  b_vent: float = 0.53


@dataclasses.dataclass(init=False, frozen=True)
class Snow:
  """Constants for snow related quantities."""

  # The density of a snow crystal [kg/m^3].
  rho: float = RHO_ICE

  # Typical snow crystal radius [m].
  r_0: float = 1e-3

  # Unit mass of a snow crystal [kg] [1].
  m_0: float = 0.1 * r_0**2
  # Exponent to the radius ratio in the mass equation [1].
  m_e: float = 2.0
  # The calibration coefficients in the mass equation.
  chi_m: float = 1.0
  del_m: float = 0.0

  # Unit cross section area of a snow crystal [m^2] [1].
  a_0: float = 0.3 * np.pi * r_0**2
  # Exponent to the radius ratio in the cross section area equation.
  a_e: float = 2.0
  # The calibration coefficients in the cross section area equation.
  chi_a: float = 1.0
  del_a: float = 0.0

  # Exponent to the radius ratio in the terminal velocity equation [1].
  v_e: float = 0.25
  # The calibration coefficients in the terminal velocity equation.
  chi_v: float = 1.0
  del_v: float = 0.0

  # The snow size distribution parameter exponent [3].
  nu: float = 0.63
  # The snow size distribution parameter coefficient [m^-4] [3].
  mu: float = 4.36e9 * RHO_AIR**nu

  # The ventilation factor coefficients [7].
  a_vent: float = 0.65
  b_vent: float = 0.44


@dataclasses.dataclass(init=False, frozen=True)
class Autoconversion:
  """Constant coefficients in autoconversion."""

  # Timescale for cloud liquid to rain water autoconversion [s] [1].
  tau_lr: float = 1e3
  # Timescale for cloud ice to snow autoconversion [s].
  tau_is: float = 1e2
  # Threshold for cloud liquid to rain water autoconversion [1].
  q_l_threshold: float = 5e-4
  # Threshold for cloud ice to snow autoconversion.
  q_i_threshold: float = 1e-6
  # Threshold particle radius between ice and snow [m] [4].
  r_is: float = 6.25e-5


@dataclasses.dataclass(init=False, frozen=True)
class Accretion:
  """Collision efficiencies in accretion."""

  # Collision efficiency between rain drops and cloud droplets [1].
  e_lr: float = 0.8
  # Collision efficiency between snow and cloud droplets [5].
  e_ls: float = 0.1
  # Collision efficiency between rain drops and cloud ice [5].
  e_ir: float = 1.0
  # Collision efficiency between snow and cloud ice [6].
  e_is: float = 0.1
  # Collision efficiency between rain drops and snow [6].
  e_rs: float = 1.0


def _n_0(
    coeff: Rain | Snow | Ice,
    rho: Optional[tf.Tensor] = None,
    q_s: Optional[tf.Tensor] = None,
) -> tf.Tensor:
  """Computes the Marshall-Palmer distribution parameter [m^-4]."""
  if isinstance(coeff, Snow):
    assert q_s is not None, 'q_s is required for Snow, but None was provided.'
    assert rho is not None, 'rho is required for Snow, but None was provided.'
    return coeff.mu * tf.math.pow(rho * q_s / RHO_AIR, coeff.nu)
  elif isinstance(coeff, Ice):
    return tf.constant(2e7)
  elif isinstance(coeff, Rain):
    return tf.constant(1.6e7)
  else:
    raise ValueError(
        f'One of Snow, Ice, or Rain is required but {type(coeff)} was provided.'
    )


def _v_0(coeff: Rain | Snow, rho: Optional[tf.Tensor] = None) -> tf.Tensor:
  """Computes unit terminal velocity."""
  if isinstance(coeff, Rain):
    assert rho is not None
    return tf.math.sqrt(
        8.0
        * coeff.r_0
        * constants.G
        / (3.0 * coeff.c_d)
        * (RHO_WATER / rho - 1.0)
    )
  elif isinstance(coeff, Snow):
    return tf.constant(2.0**2.25 * coeff.r_0**0.25)
  else:
    raise ValueError(
        f'One of Rain or Snow is required but {type(coeff)} was provided.'
    )


class OneMoment(microphysics_generic.Microphysics):
  """A library for the one-moment microphysics models."""

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    """Initializes required libraries required by microphysics models."""
    super().__init__(params)

    self._rain_coeff = Rain()
    self._snow_coeff = Snow()
    self._ice_coeff = Ice()

  def _marshall_palmer_distribution_parameter_lambda(
      self,
      coeff: Rain | Snow | Ice,
      rho: types.FlowFieldVal,
      q: types.FlowFieldVal,
  ) -> types.FlowFieldVal:
    """Computes lambda in the Marshall-Palmer distribution parameters.

    Args:
      coeff: A Rain, Snow, or Ice dataclass object that stores constant
        parameters.
      rho: The density of the moist air [kg/m^3].
      q: The water mass fraction [kg/kg].

    Returns:
      The lambda parameter in the Marshall-Palmer distribution.
    """
    m = coeff.m_e + coeff.del_m + 1.0

    def lambda_fn(rho: tf.Tensor, q: tf.Tensor) -> tf.Tensor:
      """Computes the lambda parameter for a single tf.Tensor q."""
      # The denominator is 0 when the water mass fraction is 0. In this case,
      # the distribution parameter is 0, which is provided by the behavior of
      # divide_no_nan.
      return tf.math.pow(
          tf.math.divide_no_nan(
              _gamma(m) * coeff.chi_m * coeff.m_0 * _n_0(coeff, rho, q),
              tf.maximum(q, 0.0)
              * rho
              * tf.math.pow(coeff.r_0, coeff.m_e + coeff.del_m),
          ),
          1.0 / m,
      )

    return tf.nest.map_structure(lambda_fn, rho, q)

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
        self._water_model.saturation_q_vapor(temperature, rho, q_l, q_c),
    )

  def _conduction_and_diffusion(
      self,
      coeff: Rain | Snow,
      temperature: types.FlowFieldVal,
      q_l: Optional[types.FlowFieldVal] = None,
      q_c: Optional[types.FlowFieldVal] = None,
  ) -> types.FlowFieldVal:
    """Computes the combined effect of thermal conduction and water diffusion.

    Args:
      coeff: A Rain or Snow dataclass object that stores constant parameters for
        microphysics processes.
      temperature: The temperature of the flow field [K].
      q_l: The specific humidity of the liquid phase [kg/kg].
      q_c: The specific humidity of the condensed phase [kg/kg].

    Returns:
      The rate of change of snow due to autoconversion.
    """
    lh = (
        self._water_model.lh_s(temperature)
        if isinstance(coeff, Snow)
        else self._water_model.lh_v(temperature)
    )

    def conduction_and_diffusion_fn(
        lh: tf.Tensor,
        temperature: tf.Tensor,
        p_sat: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the effects of thermal conductivity and water diffusivity."""
      r_v = self._water_model.r_v
      return 1.0 / (
          lh / (K_COND * temperature) * (lh / (r_v * temperature) - 1.0)
          + (r_v * temperature) / (p_sat * D_VAP)
      )

    return tf.nest.map_structure(
        conduction_and_diffusion_fn,
        lh,
        temperature,
        self._water_model.saturation_vapor_pressure(temperature, q_l, q_c),
    )

  def _accretion(
      self,
      coeff: Rain | Snow,
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
      coeff: A Rain or Snow dataclass object that stores constant parameters for
        microphysics processes.
      rho: The density of the moist air [kg/m^3].
      q_l: The specific humidity of the liquid phase [kg/kg].
      q_i: The specific humidity of the ice phase [kg/kg].
      q_p: The precipitation water mass fraction, which can be rain (q_r) or
        snow (q_s). [kg/kg].

    Returns:
      The rate of change of specific humidity of the precipitation (rain/snow)
      due to the collision with condensed phase water.
    """
    pi_av_coeff = coeff.a_0 * coeff.chi_a * coeff.chi_v
    sigma_av = coeff.a_e + coeff.v_e + coeff.del_a + coeff.del_v

    acc_coeff = Accretion()
    e_cp_l = acc_coeff.e_lr if isinstance(coeff, Rain) else acc_coeff.e_ls
    e_cp_i = acc_coeff.e_ir if isinstance(coeff, Rain) else acc_coeff.e_is

    lam = self._marshall_palmer_distribution_parameter_lambda(coeff, rho, q_p)

    def accretion_fn(
        rho: tf.Tensor,
        q_l: tf.Tensor,
        q_i: tf.Tensor,
        q_p: tf.Tensor,
        lam: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the accretion rate."""
      return tf.math.divide_no_nan(
          _n_0(coeff, rho, q_p)
          * pi_av_coeff
          * _v_0(coeff, rho)
          * (q_l * e_cp_l + q_i * e_cp_i)
          * _gamma(sigma_av + 1.0),
          lam,
      ) * tf.math.pow(tf.math.divide_no_nan(1.0, coeff.r_0 * lam), sigma_av)

    return tf.nest.map_structure(accretion_fn, rho, q_l, q_i, q_p, lam)

  def _autoconversion(
      self,
      coeff: Rain | Snow,
      q: types.FlowFieldVal,
  ) -> types.FlowFieldVal:
    """Computes the increase rate of precipitation due to autoconversion.

    Args:
      coeff: A Rain or Snow dataclass object that stores constant parameters for
        microphysics processes.
      q: The specific humidity of cloud water. If `coeff` is `Rain`, `q` should
        be for the liquid phase (`q_l`); if `coeff` is `Snow`, `q` should be the
        ice phase (`q_i`).

    Returns:
      The rate of increase of precipitation (rain/snow) due to autoconversion.

    Raises:
      ValueError if `coeff` is neither Rain nor Snow.
    """
    aut_coeff = Autoconversion()

    if isinstance(coeff, Rain):
      q_threshold = aut_coeff.q_l_threshold
      tau = aut_coeff.tau_lr
    elif isinstance(coeff, Snow):
      q_threshold = aut_coeff.q_i_threshold
      tau = aut_coeff.tau_is
    else:
      raise ValueError(
          f'{Rain} or {Snow} coefficients are required. {coeff} is provided.'
      )

    return tf.nest.map_structure(
        lambda q_c: tf.maximum(q_c - q_threshold, 0.0) / tau, q
    )

  def _autoconversion_snow(
      self,
      temperature: types.FlowFieldVal,
      rho: types.FlowFieldVal,
      q_v: types.FlowFieldVal,
      q_l: Optional[types.FlowFieldVal],
      q_c: Optional[types.FlowFieldVal],
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
    aut_coeff = Autoconversion()

    s = self._saturation(temperature, rho, q_v, q_l, q_c)

    g = self._conduction_and_diffusion(self._snow_coeff, temperature, q_l, q_c)

    q_i = tf.nest.map_structure(tf.math.subtract, q_c, q_l)
    lam = self._marshall_palmer_distribution_parameter_lambda(
        self._ice_coeff, rho, q_i
    )

    def snow_autoconversion_fn(
        rho: tf.Tensor,
        s: tf.Tensor,
        g: tf.Tensor,
        lam: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the autoconversion of snow."""
      r_is = aut_coeff.r_is
      aut = (
          4.0
          * np.pi
          / rho
          * (s - 1.0)
          * g
          * _n_0(self._ice_coeff)
          * tf.math.exp(-lam * r_is)
          * (
              r_is**2 / (self._ice_coeff.m_e + self._ice_coeff.del_m)
              + tf.math.divide_no_nan((r_is * lam + 1.0), lam**2)
          )
      )
      return tf.math.maximum(aut, 0.0)

    return tf.nest.map_structure(snow_autoconversion_fn, rho, s, g, lam)

  def evaporation_sublimation(
      self,
      coeff: Rain | Snow,
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
      coeff: A Rain or Snow dataclass object that stores constant parameters for
        microphysics processes.
      temperature: The temperature of the flow field [K].
      rho: The density of the moist air [kg/m^3].
      q_v: The specific humidity of the gas phase [kg/kg].
      q_p: The precipitation water mass fraction, which can be rain (q_r) or
        snow (q_s) [kg/kg].
      q_l: The specific humidity of the liquid phase [kg/kg].
      q_c: The specific humidity of the condensed phase [kg/kg].

    Returns:
      The rate of increase of precipitation due to evaporation/sublimation.
    """
    if isinstance(coeff, Rain):
      s = self._saturation(temperature, rho, q_v, q_l, q_l)
    else:
      q_i = tf.nest.map_structure(tf.math.subtract, q_c, q_l)
      zeros = tf.nest.map_structure(tf.zeros_like, q_i)
      s = self._saturation(temperature, rho, q_v, zeros, q_i)

    g = self._conduction_and_diffusion(coeff, temperature, q_l, q_c)
    lam = self._marshall_palmer_distribution_parameter_lambda(coeff, rho, q_p)

    def evap_subl_fn(
        rho: tf.Tensor,
        q_p: tf.Tensor,
        s: tf.Tensor,
        g: tf.Tensor,
        lam: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the evaporation/sublimation rate."""
      f_vent = coeff.a_vent + coeff.b_vent * (NU_AIR / D_VAP) ** (
          1.0 / 3.0
      ) * tf.math.divide_no_nan(1.0, coeff.r_0 * lam) ** (
          0.5 * (coeff.v_e + coeff.del_v)
      ) * tf.math.sqrt(
          tf.math.divide_no_nan(
              2.0 * coeff.chi_v * _v_0(coeff, rho), (NU_AIR * lam)
          )
      ) * _gamma(
          0.5 * (coeff.v_e + coeff.del_v + 5.0)
      )
      evap_subl = (
          tf.math.divide_no_nan(
              4.0 * np.pi * _n_0(coeff, rho, q_p) / rho * (s - 1.0) * g,
              lam**2,
          )
          * f_vent
      )
      if isinstance(coeff, Rain):
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
        self._rain_coeff, temperature, rho, q_v, q_r, q_l, q_c
    )
    subl = self.evaporation_sublimation(
        self._snow_coeff, temperature, rho, q_v, q_s, q_l, q_c
    )

    return tf.nest.map_structure(lambda e, s: -e - s, evap, subl)

  def autoconversion_and_accretion(
      self,
      coeff: Rain | Snow,
      temperature: types.FlowFieldVal,
      rho: types.FlowFieldVal,
      q_v: types.FlowFieldVal,
      q_p: types.FlowFieldVal,
      q_l: Optional[types.FlowFieldVal],
      q_c: Optional[types.FlowFieldVal],
  ) -> types.FlowFieldVal:
    """Computes the change of precipitation by autoconversion and accretion.

    Args:
      coeff: A Rain or Snow dataclass object that stores constant parameters for
        microphysics processes.
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
    acc = self._accretion(coeff, rho, q_l, q_i, q_p)

    if isinstance(coeff, Rain):
      aut = self._autoconversion(coeff, q_l)
    elif isinstance(coeff, Snow):
      aut = self._autoconversion_snow(temperature, rho, q_v, q_l, q_c)

    return tf.nest.map_structure(tf.math.add, acc, aut)

  def terminal_velocity(
      self,
      coeff: Rain | Snow,
      rho: types.FlowFieldVal,
      q_p: types.FlowFieldVal,
  ) -> types.FlowFieldVal:
    """Computes the terminal velocity of rain or snow.

    Args:
      coeff: A Rain or Snow dataclass object that stores constant parameters for
        microphysics processes.
      rho: The density of the moist air [kg/m^3].
      q_p: The precipitation water mass fraction, which can be rain (q_r) or
        snow (q_s) [kg/kg].

    Returns:
      The terminal velocity of rain/snow.
    """
    lam = self._marshall_palmer_distribution_parameter_lambda(coeff, rho, q_p)

    def terminal_velocity_fn(rho: tf.Tensor, lam: tf.Tensor):
      """Computes the terminal velocity."""
      return (
          coeff.chi_v
          * _v_0(coeff, rho)
          * tf.math.divide_no_nan(1.0, (coeff.r_0 * lam))
          ** (coeff.v_e + coeff.del_v)
          * _gamma(coeff.m_e + coeff.v_e + coeff.del_m + coeff.del_v + 1.0)
          / _gamma(coeff.m_e + coeff.del_m + 1.0)
      )

    return tf.nest.map_structure(terminal_velocity_fn, rho, lam)


def terminal_velocity(
    varname: str,
    microphysics: OneMoment,
    states: types.FlowFieldMap,
    additional_states: types.FlowFieldMap,
) -> types.FlowFieldVal:
  """Computes the terminal velocity for `q_r` or `q_s`.

  Args:
    varname: The name of the humidity variable, either `q_r` or `q_s`.
    microphysics: An instance of the `OneMoment` class.
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
  ), f'Terminal velocity is for `q_r` or `q_s` only, but {varname} is provided.'

  coeff = Rain() if varname == 'q_r' else Snow()

  return microphysics.terminal_velocity(
      coeff, states['rho_thermal'], states[varname]
  )


def humidity_source_fn(
    varname: str,
    microphysics: OneMoment,
    states: types.FlowFieldMap,
    additional_states: types.FlowFieldMap,
    thermo_states: types.FlowFieldMap,
) -> types.FlowFieldVal:
  """Computes the source term in a humidity equation.

  Supported types of humidity are `q_t`, `q_r`, and `q_s`.

  Args:
    varname: The name of the humidity variable, which should be one of `q_r`,
      `q_s`, and `q_t`.
    microphysics: An instance of the `OneMoment` class.
    states: A dictionary that holds all flow field variables.
    additional_states: A dictionary that holds all helper variables.
    thermo_states: A dictionary that holds all thermodynamics variables.

  Returns:
    The source term in a humidity equation due to microphysics.

  Raises:
    NotImplementedError If `varname` is not one of `q_t`, `q_r`, or `q_s`.
  """
  del additional_states

  q_v = tf.nest.map_structure(
      tf.math.subtract, thermo_states['q_t'], thermo_states['q_c']
  )

  source = tf.nest.map_structure(tf.zeros_like, states[varname])

  if varname == 'q_r':
    coeffs = [Rain()]
  elif varname == 'q_s':
    coeffs = [Snow()]
  elif varname == 'q_t':
    coeffs = [Rain(), Snow()]
  else:
    raise NotImplementedError(
        f'{varname} is not a valid humidity type for the 1-moment microphysics.'
        ' Available options are: `q_t`, `q_r`, and `q_s`.'
    )

  for coeff in coeffs:
    q_p = (
        thermo_states['q_r']
        if isinstance(coeff, Rain)
        else thermo_states['q_s']
    )
    aut_and_acc = microphysics.autoconversion_and_accretion(
        coeff,
        thermo_states['T'],
        states['rho_thermal'],
        q_v,
        q_p,
        thermo_states['q_l'],
        thermo_states['q_c'],
    )

    evap_subl = microphysics.evaporation_sublimation(
        coeff,
        thermo_states['T'],
        states['rho_thermal'],
        q_v,
        q_p,
        thermo_states['q_l'],
        thermo_states['q_c'],
    )

    source = tf.nest.map_structure(
        lambda s, a, e: s + a + e, source, aut_and_acc, evap_subl
    )

  if varname == 'q_t':
    source = tf.nest.map_structure(tf.math.negative, source)

  return tf.nest.map_structure(tf.math.multiply, source, states['rho'])


def potential_temperature_source_fn(
    varname: str,
    microphysics: OneMoment,
    states: types.FlowFieldMap,
    additional_states: types.FlowFieldMap,
    thermo_states: types.FlowFieldMap,
) -> types.FlowFieldVal:
  """Computes the source term in a potential temperature equation.

  Supported types of potential temperature are `theta` and `theta_li`.

  Args:
    varname: The name of the potential temperature variable, either `theta` or
      `theta_li`.
    microphysics: An instance of the `OneMoment` class.
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

  source = tf.nest.map_structure(tf.zeros_like, states['rho'])

  phase_params = (
      (Rain(), microphysics.water_model.lh_v0, states['q_r']),
      (Snow(), microphysics.water_model.lh_s0, states['q_s']),
  )

  cp = microphysics.water_model.cp_m(
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

  for coeff, lh, q in phase_params:
    # Calculate source terms for vapor and liquid conversions, respectively.
    # Use that c_{q_v->q_r/s} = -c_{q_r/s->q_v}, i.e. minus the evaporation/
    # sublimation rate.
    water_source = tf.nest.map_structure(
        tf.math.negative,
        microphysics.evaporation_sublimation(
            coeff,
            thermo_states['T'],
            states['rho_thermal'],
            thermo_states['q_v'],
            q,
            thermo_states['q_l'],
            thermo_states['q_c'],
        ),
    )

    if varname == 'theta_li':
      # Get conversion rates and energy source from cloud water/ice to rain/
      # snow. This source term applies to the liquid-ice potential temperature
      # only.
      water_source = tf.nest.map_structure(
          tf.math.add,
          water_source,
          microphysics.autoconversion_and_accretion(
              coeff,
              thermo_states['T'],
              states['rho_thermal'],
              thermo_states['q_v'],
              q,
              thermo_states['q_l'],
              thermo_states['q_c'],
          ),
      )

    # Converts water mass fraction conversion rates to an energy source term.
    theta_source = tf.nest.map_structure(
        energy_source_fn(lh),
        states['rho'],
        cp,
        microphysics.water_model.exner_inverse(
            states['rho_thermal'],
            thermo_states['q_t'],
            thermo_states['T'],
            thermo_states['zz'],
            additional_states,
        ),
        water_source,
    )

    source = tf.nest.map_structure(tf.math.add, source, theta_source)

  return source


def total_energy_source_fn(
    microphysics: OneMoment,
    states: types.FlowFieldMap,
    additional_states: types.FlowFieldMap,
    thermo_states: types.FlowFieldMap,
) -> types.FlowFieldVal:
  """Computes the source term in the total energy equation.

  Args:
    microphysics: An instance of the `OneMoment` class.
    states: A dictionary that holds all flow field variables.
    additional_states: A dictionary that holds all helper variables.
    thermo_states: A dictionary that holds all thermodynamics variables.

  Returns:
    The source term in the total energy equation due to microphysics.
  """
  del additional_states

  # Get potential energy.
  pe = tf.nest.map_structure(
      lambda zz_i: constants.G * zz_i, thermo_states['zz']
  )

  # Get conversion rates and energy source from cloud water/ice to rain/snow.
  source_li = tf.nest.map_structure(tf.zeros_like, states['rho'])
  coeffs = [Rain(), Snow()]
  q_p = [thermo_states['q_r'], thermo_states['q_s']]
  e_int = [thermo_states['e_l'], thermo_states['e_i']]

  for coeff, q, e in zip(coeffs, q_p, e_int):
    aut_acc = microphysics.autoconversion_and_accretion(
        coeff,
        thermo_states['T'],
        states['rho_thermal'],
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
  evap_subl = microphysics.evaporation(
      thermo_states['T'],
      states['rho_thermal'],
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

  return tf.nest.map_structure(tf.math.add, source_v, source_li)
