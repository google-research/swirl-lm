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

# Copyright 2022 Google LLC
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
"""A library of the microphysics from Klemp & Wilhelmson, 1978."""

from typing import Optional

from swirl_lm.base import parameters as parameters_lib
from swirl_lm.equations import common
from swirl_lm.physics import constants
from swirl_lm.physics.atmosphere import microphysics_generic
from swirl_lm.physics.thermodynamics import water
import tensorflow as tf


class MicrophysicsKW1978(microphysics_generic.Microphysics):
  """An object for handling precipitation modeling."""

  def evaporation(
      self,
      rho: water.FlowFieldVal,
      temperature: water.FlowFieldVal,
      q_r: water.FlowFieldVal,
      q_v: water.FlowFieldVal,
      q_l: water.FlowFieldVal,
      q_c: water.FlowFieldVal,
      additional_states: water.FlowFieldMap,
  ) -> water.FlowFieldVal:
    r"""The rain evaporation rate.

    This is based on J. B. Klemp and R. B. Wilhelmson, 1978: `The Simulation of
    Three-Dimensional Covective Storm Dynamics`, J. Atmos. Sci, 35, 1070-1096.
    Specifically from eqs. (2.14a) & (2.14b).
    The calculation here has been converted to MKS.

       E_r = \frac{1}{\rho} \frac{(1-q_v/q_{vs})C(\rho q_r)^0.5252}
             {2.03 \times 10^4 + 9.584 \times 10^6/(p_{vs})}

       where

       C = 1.6 + 30.3922(\rho q_r)^0.2046

    Here \rho is the ambient moist air density (which is provided by
    argument `rho`), q_v is the cloud vapor fraction (provided by argument
    `q_v`), q_{vs} is the saturation cloud vapor fraction (calculated from
    `temperature`, `q_l`; the specific humidity of the liquid phase, and
    `q_c`; the specific humidity of the condensed phase), q_r is the rain water
    mixture fraction (provided by argument `q_r`) and p_{vs} is the saturation
    cloud vapor pressure (calculated from q_{vs}, `temperature` and specific gas
    constant of water vapor `r_v`).
    Note that `q_c` is the sum of liquid phase `q_l` and ice phase `q_i `(not
    used here) and it is used for the saturation vapor pressure calculation.

    Args:
      rho: The mosit air density, in kg/m^3.
      temperature: The temperature.
      q_r: The rain water mixture fraction (kg/kg).
      q_v: The cloud vapor fraction (kg/kg).
      q_l: The specific humidity of the cloud liquid phase (kg/kg).
      q_c: The specific humidity of the cloud humidity condensed phase,
        including ice and liquid (kg/kg).
      additional_states: Helper variables in the simulation. It is used to
        compute the reference density and pressure in this function.

    Returns:
      The evaporation rate of the rain drops in the unit of 1/sec.

    Raises:
      ValueError: If the thermodynamics model is not `Water`.
    """
    zz = additional_states.get('zz', tf.nest.map_structure(tf.zeros_like, q_r))
    rho_bar = self.water_model.rho_ref(zz, additional_states)
    p_bar = self.water_model.p_ref(zz, additional_states)
    q_vs = self.water_model.saturation_q_vapor(temperature, rho, q_l, q_c)
    p_vs = tf.nest.map_structure(tf.math.multiply, q_vs, p_bar)

    precipitation_bulk_density = tf.nest.map_structure(
        lambda q_r_i, rho_i: rho_i  # pylint: disable=g-long-lambda
        * tf.clip_by_value(q_r_i, clip_value_min=0.0, clip_value_max=1.0),
        q_r,
        rho_bar,
    )

    def ventilation_factor(rho_qr):
      return 1.6 + 30.3922 * tf.math.pow(rho_qr, 0.2046)

    c = tf.nest.map_structure(ventilation_factor, precipitation_bulk_density)

    def evaporation_rate(rho, q_v, q_vs, c, rho_qr, p_vs):
      return (
          (1.0 / rho)
          * (1.0 - q_v / q_vs)
          * c
          * tf.math.pow(rho_qr, 0.525)
          / (2.03e4 + 9.584e6 / p_vs)
      )

    e_r = tf.nest.map_structure(
        evaporation_rate,
        rho_bar,
        q_v,
        q_vs,
        c,
        precipitation_bulk_density,
        p_vs,
    )
    return e_r

  def autoconversion_and_accretion(
      self,
      q_r: water.FlowFieldVal,
      q_l: water.FlowFieldVal,
  ) -> water.FlowFieldVal:
    r"""The conversion rate from cloud liquid humidity to rain water.

    This is based on J. B. Klemp and R. B. Wilhelmson, 1978: `The Simulation of
    Three-Dimensional Covective Storm Dynamics`, J. Atmos. Sci, 35, 1070-1096.
    Specifically from eqs. (2.13a) & (2.13b).

    Note also the notation in the paper `q_c` is the mixture fraction of the
    cloud liquid phase humidity, which in our context is represented by `q_l`.

    Auto coversion rate (from cloud liquid phase to rain water):

        A_r = k_1 (q_l - a), with k_1 = 0.001 (1/sec), a = 0.001 kg/kg

    Conversion due to rain colliding with cloud liquid:

        C_r = k_2 q_l q_r ^ 0.875, with k_2 = 2.2 (1/sec).

    Total conversion rate: A_r + C_r.

    Args:
      q_r: The rain water mixture fraction (kg/kg).
      q_l: The specific humidity of the cloud liquid phase (kg/kg).

    Returns:
      The conversation rate of cloud liquid phase humidity to rain drops
      in the unit of 1/sec.
    """
    a = 0.001
    k_1 = 0.001
    k_2 = 2.2

    def autoconversion(q_l):
      return k_1 * tf.math.maximum(q_l - a, 0.0)

    a_r = tf.nest.map_structure(autoconversion, q_l)

    def accretion(q_r, q_l):
      return (
          k_2
          * q_l
          * tf.math.pow(
              tf.clip_by_value(q_r, clip_value_min=0.0, clip_value_max=1.0),
              0.875,
          )
      )

    c_r = tf.nest.map_structure(accretion, q_r, q_l)

    return tf.nest.map_structure(tf.math.add, a_r, c_r)

  def terminal_velocity(
      self,
      rho: water.FlowFieldVal,
      q_r: water.FlowFieldVal,
      additional_states: water.FlowFieldMap,
      rho_ref: float = 1.15,
  ) -> water.FlowFieldVal:
    r"""Terminal velocity used for rain water convection term.


    This is based on J. B. Klemp and R. B. Wilhelmson, 1978: `The Simulation of
    Three-Dimensional Covective Storm Dynamics`, J. Atmos. Sci, 35, 1070-1096.
    Specifically from eqs. (2.15).

    In MKS units, this is given as
        14.34 (rho q_r)^0.1346 (rho / rho_ref)^(-0.5) [m/s]

    Args:
       rho: density [kg/m^3]
       q_r: rain water mixture fraction [kg/kg]
       additional_states: Helper variables in the simulation. It is used to
         compute the reference density in this function.
       rho_ref: Reference density at ground level. Default is 1.15 [kg/m^3]

    Returns:
      The terminal velocity of rain water in units of m/s.
    """
    # Temporarity disabling `rho` instead of removing it from the argument list.
    # It will be cleaned up after the formulation is finalized.
    del rho

    zz = additional_states.get('zz', tf.nest.map_structure(tf.zeros_like, q_r))
    rho_bar = self.water_model.rho_ref(zz, additional_states)

    k1 = 14.34
    k2 = 0.1346
    sqrt_rho_ref = tf.math.sqrt(rho_ref)

    def factor1(rho, q_r):
      return k1 * tf.math.pow(
          rho * tf.clip_by_value(q_r, clip_value_min=0.0, clip_value_max=1.0),
          k2,
      )

    f1 = tf.nest.map_structure(factor1, rho_bar, q_r)

    def factor2(rho):
      # Take advantage of fast reciprocal square root op.
      return sqrt_rho_ref * tf.math.rsqrt(rho)

    f2 = tf.nest.map_structure(factor2, rho_bar)

    return tf.nest.map_structure(tf.math.multiply, f1, f2)


class Adapter(microphysics_generic.MicrophysicsAdapter):
  """Interface between the Kessler microphysics model and rest of Swirl-LM."""

  _kessler: MicrophysicsKW1978

  def __init__(
      self, params: parameters_lib.SwirlLMParameters, water_model: water.Water
  ):
    assert params.microphysics is not None and params.microphysics.HasField(
        'kessler'
    ), 'The microphysics.kessler field needs to be set in SwirlLMParameters.'
    self._kessler_params = params.microphysics.kessler
    self._kessler = MicrophysicsKW1978(params, water_model)

  def terminal_velocity(
      self,
      varname: str,
      states: water.FlowFieldMap,
      additional_states: water.FlowFieldMap,
  ) -> water.FlowFieldVal:
    """Computes the terminal velocity for `q_r`, `q_s`, `q_l`, or `q_i`.

    Args:
      varname: The name of the humidity variable, either `q_r` or `q_s`.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.

    Returns:
      The terminal velocity for `q_r` or `q_s`.

    Raises:
      NotImplementedError If `varname` is not one of `q_r` or `q_s`.
    """
    assert varname in (
        'q_r',
        'q_s',
        'q_l',
        'q_i',
    ), (
        f'Terminal velocity is for `q_r` or `q_s` only, but {varname} is'
        ' provided.'
    )

    return self._kessler.terminal_velocity(
        states['rho_thermal'], states[varname], additional_states
    )

  def humidity_source_fn(
      self,
      varname: str,
      states: water.FlowFieldMap,
      additional_states: water.FlowFieldMap,
      thermo_states: water.FlowFieldMap,
  ) -> water.FlowFieldVal:
    """Computes the source term in a humidity equation.

    Supported types of humidity are `q_t` and `q_r`.

    Args:
      varname: The name of the humidity variable, either `q_r` or `q_t`.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.
      thermo_states: A dictionary that holds all thermodynamics variables.

    Returns:
      The source term in a humidity equation due to microphysics.

    Raises:
      NotImplementedError If `varname` is not one of `q_t` `q_c`, `q_v`, or
      `q_r`.
    """
    q_r = thermo_states['q_r']
    q_l = thermo_states['q_l']
    aut_and_acc = self._kessler.autoconversion_and_accretion(q_r, q_l)
    rain_water_evaporation_rate = self._kessler.evaporation(
        states['rho_thermal'],
        thermo_states['T'],
        q_r,
        thermo_states['q_v'],
        q_l,
        thermo_states['q_c'],
        additional_states,
    )
    # Net vapor to rain water rate is
    #   (vapor to rain water rate) - (evaporation rate).
    net_cloud_liquid_to_rain_water_rate = tf.nest.map_structure(
        tf.math.subtract, aut_and_acc, rain_water_evaporation_rate
    )
    cloud_liquid_to_water_source = tf.nest.map_structure(
        tf.math.multiply,
        net_cloud_liquid_to_rain_water_rate,
        states[common.KEY_RHO],
    )
    # Add term for q_r, subtract for q_t.
    if varname == 'q_t':
      return tf.nest.map_structure(
          tf.math.negative, cloud_liquid_to_water_source
      )
    elif varname == 'q_r':
      return cloud_liquid_to_water_source
    # aut_and_acc process consumes cloud liquid.
    elif varname == 'q_c':
      rho_aut_acc = tf.nest.map_structure(
          tf.math.multiply, aut_and_acc, states[common.KEY_RHO]
      )
      return tf.nest.map_structure(tf.math.negative, rho_aut_acc)
    # rain evaporation contributes to add cloud vapor.
    elif varname == 'q_v':
      rho_evap_rate = tf.nest.map_structure(
          tf.math.multiply, rain_water_evaporation_rate, states[common.KEY_RHO]
      )
      return rho_evap_rate
    else:
      raise NotImplementedError(
          f'Precipitation for {varname} is not implemented. Only'
          ' `q_t`, `q_r`, `q_c` and `q_v` are supported.'
      )

  def potential_temperature_source_fn(
      self,
      varname: str,
      states: water.FlowFieldMap,
      additional_states: water.FlowFieldMap,
      thermo_states: water.FlowFieldMap,
  ) -> water.FlowFieldVal:
    """Computes the source term in a potential temperature equation.

    Supported types of potential temperature are `theta` and `theta_li`.

    Args:
      varname: The name of the potential temperature variable, either `theta` or
        `theta_li`.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.
      thermo_states: A dictionary that holds all thermodynamics variables.

    Returns:
      The source term in a potential equation due to microphysics.

    Raises:
      AssertionError If `varname` is not one of `theta` or `theta_li`.
    """
    assert varname in ('theta', 'theta_li'), (
        'Only `theta` and `theta_li` are supported for the microphysics source'
        f' term computation, but {varname} is provided.'
    )

    # Calculate source terms for vapor and liquid conversions, respectively.
    # Use that c_{q_v->q_r} = -c_{q_r->q_v}, i.e. minus the evaporation
    # rate.
    source = tf.nest.map_structure(
        tf.math.negative,
        self._kessler.evaporation(
            states['rho_thermal'],
            thermo_states['T'],
            states['q_r'],
            thermo_states['q_v'],
            thermo_states['q_l'],
            thermo_states['q_c'],
            additional_states,
        ),
    )

    if varname == 'theta_li':
      # Get conversion rates and energy source from cloud water/ice to rain.
      # This source term applies to the liquid-ice potential temperature only.
      source = tf.nest.map_structure(
          tf.math.add,
          source,
          self._kessler.autoconversion_and_accretion(
              states['q_r'], thermo_states['q_l']
          ),
      )

    # Converts water mass fraction conversion rates to an energy source term.
    t_0 = self._kessler.water_model.t_ref(
        thermo_states['zz'], additional_states
    )
    zeros = tf.nest.map_structure(tf.zeros_like, t_0)
    theta_0 = self._kessler.water_model.temperature_to_potential_temperature(
        'theta',
        t_0,
        zeros,
        zeros,
        zeros,
        thermo_states['zz'],
        additional_states,
    )
    cp = self._kessler.water_model.cp_m(
        thermo_states['q_t'], thermo_states['q_l'], thermo_states['q_i']
    )

    def energy_source_fn(
        rho: tf.Tensor,
        cp: tf.Tensor,
        t_0: tf.Tensor,
        theta_0: tf.Tensor,
        s: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the condensation source term."""
      return rho * (self._kessler.water_model.lh_v0 / cp) * (theta_0 / t_0) * s

    return tf.nest.map_structure(
        energy_source_fn, states['rho'], cp, t_0, theta_0, source
    )

  def total_energy_source_fn(
      self,
      states: water.FlowFieldMap,
      additional_states: water.FlowFieldMap,
      thermo_states: water.FlowFieldMap,
  ) -> water.FlowFieldVal:
    """Computes the source term in the total energy equation.

    Args:
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.
      thermo_states: A dictionary that holds all thermodynamics variables.

    Returns:
      The source term in a total energy equation due to microphysics.
    """
    # Get conversion rates from cloud water to rain water (for liquid and
    # vapor phase).
    cloud_liquid_to_rain_water_rate = (
        self._kessler.autoconversion_and_accretion(
            thermo_states['q_r'], thermo_states['q_l']
        )
    )
    # Find q_v from the invariant q_t = q_c + q_v = q_l + q_i + q_v.
    rain_water_evaporation_rate = self._kessler.evaporation(
        states['rho_thermal'],
        thermo_states['T'],
        thermo_states['q_r'],
        thermo_states['q_v'],
        thermo_states['q_l'],
        thermo_states['q_c'],
        additional_states,
    )
    # Get potential energy.
    pe = tf.nest.map_structure(
        lambda zz_i: constants.G * zz_i, thermo_states['zz']
    )
    # Calculate source terms for vapor and liquid conversions, respectively.
    # Use that c_{q_v->q_l} = -c_{q_l->q_v}, i.e. minus the evaporation
    # rate.
    source_v = tf.nest.map_structure(
        lambda e, pe, rho, c_lv: (e + pe) * rho * (-c_lv),
        thermo_states['e_v'],
        pe,
        states['rho'],
        rain_water_evaporation_rate,
    )
    source_l = tf.nest.map_structure(
        lambda e, pe, rho, c_lr: (e + pe) * rho * c_lr,
        thermo_states['e_l'],
        pe,
        states['rho'],
        cloud_liquid_to_rain_water_rate,
    )
    return tf.nest.map_structure(tf.math.add, source_v, source_l)

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
    return self._kessler.condensation_bf2002(
        rho,
        temperature,
        q_v,
        q_l,
        q_c,
        zz,
        additional_states,
    )
