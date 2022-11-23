# Copyright 2022 The swirl_lm Authors.
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
"""A class for handling precipitation modeling."""


from swirl_lm.physics.thermodynamics import water
import tensorflow as tf


class Precipitation(object):
  """An object for handling precipitation modeling."""

  def __init__(self, water_model: water.Water) -> None:
    """Initialize with a water thermodynamics model."""
    self._water_model = water_model

  @property
  def water_model(self) -> water.Water:
    """The underlying water thermodynamics model."""
    return self._water_model

  def rain_evaporation_rate_kw1978(
      self,
      rho: water.FlowFieldVal,
      temperature: water.FlowFieldVal,
      q_r: water.FlowFieldVal,
      q_v: water.FlowFieldVal,
      q_l: water.FlowFieldVal,
      q_c: water.FlowFieldVal,
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

    Returns:
      The evaporation rate of the rain drops in the unit of 1/sec.

    Raises:
      ValueError: If the thermodynamics model is not `Water`.
    """
    q_vs = self.water_model.saturation_q_vapor(temperature, rho, q_l, q_c)

    def saturation_vapor_from_mixture_fraction(q_vs, rho, t):
      return self.water_model.r_v * q_vs * rho * t

    p_vs = tf.nest.map_structure(saturation_vapor_from_mixture_fraction, q_vs,
                                 rho, temperature)

    precipitation_bulk_density = tf.nest.map_structure(
        lambda q_r_i, rho_i: rho_i * tf.clip_by_value(  # pylint: disable=g-long-lambda
            q_r_i, clip_value_min=0.0, clip_value_max=1.0), q_r, rho)

    def c_coefficient(rho_qr):
      return 1.6 + 30.3922 * tf.math.pow(rho_qr, 0.2046)

    c = tf.nest.map_structure(c_coefficient, precipitation_bulk_density)

    def evaporation_rate(rho, q_v, q_vs, c, rho_qr, p_vs):
      return ((1.0 / rho) * (1.0 - q_v / q_vs) * c * tf.math.pow(
          rho_qr, 0.525) / (2.03e4 + 9.584e6 / p_vs))

    e_r = tf.nest.map_structure(evaporation_rate, rho, q_v, q_vs, c,
                                precipitation_bulk_density, p_vs)
    return e_r

  def cloud_liquid_to_rain_conversion_rate_kw1978(
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

    def auto_conversion(q_l):
      return k_1 * tf.math.maximum(q_l - a, 0.0)

    a_r = tf.nest.map_structure(auto_conversion, q_l)

    def colliding_conversion(q_r, q_l):
      return k_2 * q_l * tf.math.pow(
          tf.clip_by_value(q_r, clip_value_min=0.0, clip_value_max=1.0), 0.875)

    c_r = tf.nest.map_structure(colliding_conversion, q_r, q_l)

    return tf.nest.map_structure(tf.math.add, a_r, c_r)

  def rain_water_terminal_velocity_kw1978(
      self,
      rho: water.FlowFieldVal,
      q_r: water.FlowFieldVal,
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
       rho_ref: Reference density at ground level. Default is 1.15 [kg/m^3]

    Returns:
      The terminal velocity of rain water in units of m/s.
    """

    k1 = 14.34
    k2 = 0.1346
    sqrt_rho_ref = tf.math.sqrt(rho_ref)

    def factor1(rho, q_r):
      return k1 * tf.math.pow(
          rho * tf.clip_by_value(q_r, clip_value_min=0.0, clip_value_max=1.0),
          k2)

    f1 = tf.nest.map_structure(factor1, rho, q_r)

    def factor2(rho):
      # Take advantage of fast reciprocal square root op.
      return sqrt_rho_ref * tf.math.rsqrt(rho)

    f2 = tf.nest.map_structure(factor2, rho)

    return tf.nest.map_structure(tf.math.multiply, f1, f2)
