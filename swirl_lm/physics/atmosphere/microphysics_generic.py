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
"""A library for generic microphysics models."""
import abc
from typing import Optional

from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics.thermodynamics import water
from swirl_lm.utility import types
import tensorflow as tf


class MicrophysicsAdapter(abc.ABC):
  """Interface for microphysics models."""

  @abc.abstractmethod
  def terminal_velocity(
      self,
      varname: str,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldVal:
    """Computes the terminal velocity for `q_r`, `q_s`, `q_l`, or `q_i`.

    Args:
      varname: The name of the humidity variable, either `q_r` or `q_s`.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.

    Returns:
      The terminal velocity for `q_r`, `q_s`, `q_l`, or `q_i`.

    Raises:
      NotImplementedError If `varname` is not one of `q_r`, `q_s`, `q_l`, or
      `q_i`.
    """
    raise NotImplementedError

  @abc.abstractmethod
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
    raise NotImplementedError

  @abc.abstractmethod
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
    raise NotImplementedError

  @abc.abstractmethod
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
    raise NotImplementedError

  @abc.abstractmethod
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
    """Computes the condensation rate.

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
    raise NotImplementedError


class Microphysics:
  """Microphysics code common to different microphysics schemes."""

  def __init__(
      self, params: parameters_lib.SwirlLMParameters, water_model: water.Water
  ):
    """Sets up params and water model for microphysics libraries."""
    self.params = params
    self.water_model = water_model

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
    """Computes the condensation rate.

    Reference:
    Grabowski, W. W., & Smolarkiewicz, P. K. (1990). Monotone finite-difference
    approximations to the advection-condensation problem. Monthly Weather
    Review, 118(10), 2082–2098.

    Args:
      rho: The moist air density, in kg/m^3.
      temperature: The temperature.
      q_v: The cloud vapor fraction (kg/kg).
      q_l: The specific humidity of the cloud liquid phase (kg/kg).
      q_c: The specific humidity of the cloud humidity condensed phase,
        including ice and liquid (kg/kg).
      zz: The vertical coordinates (m).
      additional_states: Helper variables including those needed to compute
        reference states.

    Returns:
      The condensation rate.
    """
    q_t = tf.nest.map_structure(tf.math.add, q_v, q_c)
    q_i = tf.nest.map_structure(tf.math.subtract, q_c, q_l)
    q_vs = self.water_model.saturation_q_vapor(temperature, rho, q_l, q_c)
    t_0 = self.water_model.t_ref(zz, additional_states)
    # Here we assume that the air is dry in the reference state.
    zeros = tf.nest.map_structure(tf.zeros_like, zz)
    theta_0 = self.water_model.temperature_to_potential_temperature(
        'theta', t_0, zeros, zeros, zeros, zz, additional_states
    )
    theta = self.water_model.temperature_to_potential_temperature(
        'theta', temperature, q_t, q_l, q_i, zz, additional_states
    )
    cp = self.water_model.cp_m(q_t, q_l, q_i)

    def condensation_fn(q_v, q_vs, t_0, theta_0, theta, cp, q_c):
      """Computes the condensation rate."""
      d_q_v = (q_v - q_vs) / (
          1.0
          + q_vs
          * (self.water_model.lh_v0 / cp / t_0)
          * (theta_0 / theta)
          * (
              (self.water_model.lh_v0 / self.water_model.r_v / t_0)
              * (theta_0 / theta)
              - 1.0
          )
      )
      return tf.maximum(d_q_v, -q_c) / self.params.dt

    return tf.nest.map_structure(
        condensation_fn, q_v, q_vs, t_0, theta_0, theta, cp, q_c
    )

  def condensation_bf2002(
      self,
      rho: water.FlowFieldVal,
      temperature: water.FlowFieldVal,
      q_v: water.FlowFieldVal,
      q_l: water.FlowFieldVal,
      q_c: water.FlowFieldVal,
      zz: Optional[water.FlowFieldVal] = None,
      additional_states: Optional[water.FlowFieldMap] = None,
  ) -> water.FlowFieldVal:
    """Computes the condensation rate.

    Reference:
    Bryan, G.H., & Fritsch, J. M., (2002). A Benchmark Simulation for Moist
    Nonhydrostatic Numerical Models. Monthly Weather Review, 130, 2917-2928.
    Specifically, eq. 27.

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
    del zz, additional_states
    q_t = tf.nest.map_structure(tf.math.add, q_v, q_c)
    q_i = tf.nest.map_structure(tf.math.subtract, q_c, q_l)
    q_vs = self.water_model.saturation_q_vapor(temperature, rho, q_l, q_c)
    cp = self.water_model.cp_m(q_t, q_l, q_i)

    def condensation_fn(q_v, q_vs, temperature, cp, q_c):
      """Computes the condensation rate."""
      d_q_v = (q_v - q_vs) / (
          1.0
          + (q_vs
             * ((self.water_model.lh_v0 / temperature) ** 2)
             / self.water_model.r_v / cp)
      )
      return tf.maximum(d_q_v, -q_c) / self.params.dt

    return tf.nest.map_structure(
        condensation_fn, q_v, q_vs, temperature, cp, q_c
    )
