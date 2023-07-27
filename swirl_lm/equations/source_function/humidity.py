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

"""A library for the source functions in the humidity equation."""
from typing import Tuple

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.equations import common
from swirl_lm.equations import utils as eq_utils
from swirl_lm.equations.source_function import scalar_generic
from swirl_lm.physics.atmosphere import microphysics_utils
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.physics.thermodynamics import water
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf

HUMIDITY_VARNAME = ['q_t', 'q_r', 'q_s', 'q_c', 'q_v']


class Humidity(scalar_generic.ScalarGeneric):
  """Defines functions for source terms in humidity equation.

  This class defines terms that computes the rhs `f(q)` of the humidity update
  equation `d rho_q / dt = f(q)`, where `q` is one of the following:
  1. total humidity `q_t`;
  2. rain mass fraction `q_r`;
  3. snow mass fraction `q_s`;
  4. vapor phase cloud water mass fraction `q_v`;
  5. condensed phase cloud water mass fraction `q_c`;
  """

  def __init__(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      params: parameters_lib.SwirlLMParameters,
      scalar_name: str,
      thermodynamics: thermodynamics_manager.ThermodynamicsManager,
  ):
    """Retrieves context information for the humidity source."""
    super().__init__(kernel_op, params, scalar_name, thermodynamics)

    assert scalar_name in HUMIDITY_VARNAME, (
        f'Source term function for {scalar_name} is not implemented. Supported'
        f' humidity types are: {HUMIDITY_VARNAME}.'
    )

    self._include_subsidence = (
        self._scalar_params.HasField('humidity') and
        self._scalar_params.humidity.include_subsidence and
        self._g_dim is not None)

    self._include_precipitation = (
        self._scalar_params.HasField('humidity') and
        self._scalar_params.humidity.include_precipitation)

    self._include_condensation = (
        self._scalar_params.HasField('humidity') and
        self._scalar_params.humidity.include_condensation)

    if scalar_name in ('q_r', 'q_s'):
      assert self._include_precipitation, (
          'Calculating q_r without setting include_precipitation to True is not'
          ' well-defined.'
      )

    self._microphysics = None
    if (
        self._include_precipitation
        or self._include_condensation
        or scalar_name in ('q_r', 'q_s')
    ):
      assert self._scalar_params.humidity.HasField('microphysics'), (
          'A microphysics model is required to consider precipitation or '
          'condensation in the humidity equation, and to compute the humidity '
          'variables `q_r` and `q_s`.'
      )
      microphysics_model_name = self._scalar_params.humidity.WhichOneof(
          'microphysics'
      )
      self._microphysics = (
          microphysics_utils.select_microphysics(
              microphysics_model_name, self._params, self._thermodynamics
          )
      )

  def _get_thermodynamic_variables(
      self,
      phi: types.FlowFieldVal,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldMap:
    """Computes thermodynamic variables required to evaluate terms in equation.

    Args:
      phi: The variable `self._scalar_name` at the present iteration.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.

    Returns:
      A dictionary of thermodynamic variables.
    """
    if not isinstance(self._thermodynamics.model, water.Water):
      raise ValueError(
          '`water` thermodynamics is required for the humidity equation, but'
          f' {self._thermodynamics.model} is used.'
      )

    if (
        'e_t' not in states
        and 'theta_li' not in states
        and 'theta' not in states
    ):
      raise ValueError(
          'Expected one of e_t, theta_li, and theta in states used'
          'for the total humidity equation, got: {}'.format(states.keys())
      )

    thermo_states = {self._scalar_name: phi}

    if self._include_precipitation:
      thermo_states['q_r'] = (phi if self._scalar_name == 'q_r' else
                              states['q_r'])
      if self._scalar_name == 'q_s':
        thermo_states['q_s'] = phi
      elif 'q_s' in states:
        thermo_states['q_s'] = states['q_s']
      else:
        # Assumes no snow in the simulation.
        thermo_states['q_s'] = tf.nest.map_structure(tf.zeros_like, phi)

    if 'q_t' in states:
      # We solve `q_t`. Other quantities such as `q_c`, `q_v`, are derived from
      # it. In this case, we need to first calculate `temperature` before
      # deriving these quantities.
      thermo_states['q_t'] = (phi if self._scalar_name == 'q_t' else
                              states['q_t'])
    else:
      # We solve `q_c` and `q_v` with their own transport equations. Here we
      # assume the ice phase is absent.
      thermo_states['q_c'] = (phi if self._scalar_name == 'q_c' else
                              states['q_c'])
      thermo_states['q_v'] = (phi if self._scalar_name == 'q_v' else
                              states['q_v'])
      thermo_states['q_l'] = thermo_states['q_c']
      thermo_states['q_i'] = tf.nest.map_structure(
          tf.zeros_like, thermo_states['q_c'])
      thermo_states['q_t'] = tf.nest.map_structure(
          tf.math.add, thermo_states['q_v'], thermo_states['q_c'])

    q_t = thermo_states['q_t']
    rho_thermal = states['rho_thermal']
    zz = additional_states.get('zz', tf.nest.map_structure(tf.zeros_like, phi))
    thermo_states['zz'] = zz

    # Compute the temperature.
    if 'theta_li' in states:
      temperature = self._thermodynamics.model.saturation_adjustment(
          'theta_li',
          states['theta_li'],
          rho_thermal,
          q_t,
          zz=zz,
          additional_states=additional_states,
      )
    elif 'theta' in states:
      if 'q_c' in states:
        temperature = (
            self._thermodynamics.model.potential_temperature_to_temperature(
                'theta', states['theta'], q_t, thermo_states['q_c'],
                tf.nest.map_structure(tf.zeros_like, phi), zz,
                additional_states))
      else:
        temperature = self._thermodynamics.model.saturation_adjustment(
            'theta',
            states['theta'],
            rho_thermal,
            q_t,
            zz=zz,
            additional_states=additional_states,
        )
    elif 'e_t' in states:
      e = self._thermodynamics.model.internal_energy_from_total_energy(
          states['e_t'],
          states[common.KEY_U],
          states[common.KEY_V],
          states[common.KEY_W],
          zz,
      )
      temperature = self._thermodynamics.model.saturation_adjustment(
          'e_int', e, rho_thermal, q_t, additional_states=additional_states)
    elif 'T' in additional_states:
      temperature = additional_states['T']
    thermo_states.update({'T': temperature})

    # Compute the potential temperature.
    if 'theta' in states:
      theta = states['theta']
    elif 'theta' in additional_states:
      theta = additional_states['theta']
    else:
      buf = self._thermodynamics.model.potential_temperatures(
          temperature, q_t, rho_thermal, zz, additional_states)
      theta = buf['theta']
    thermo_states.update({'theta': theta})

    if 'q_t' in states:
      # Compute the liquid and ice humidity for the case `q_t` is a prognostic
      # variable.
      q_l, q_i = self._thermodynamics.model.equilibrium_phase_partition(
          temperature, rho_thermal, thermo_states['q_t']
      )
      thermo_states.update({'q_l': q_l, 'q_i': q_i})
      thermo_states['q_c'] = tf.nest.map_structure(tf.math.add, q_l, q_i)
      thermo_states['q_v'] = tf.nest.map_structure(
          tf.math.subtract, q_t, thermo_states['q_c'])

    return thermo_states

  def _get_wall_diffusive_flux_helper_variables(
      self,
      phi: types.FlowFieldVal,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldMap:
    """Prepares the helper variables for the diffusive flux in wall models.

    Args:
      phi: The humidity (one of `q_t`, `q_r`, `q_s`, `q_v`, and `q_c` that is
        defined in the ctor) at the present iteration.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.

    Returns:
      A dictionary of variables required by wall diffusive flux closure models.
    """
    helper_variables = {
        key: states[key] for key in common.KEYS_VELOCITY
    }
    helper_variables.update(
        self._get_thermodynamic_variables(phi, states, additional_states)
    )
    return helper_variables

  def _get_momentum_for_convection(
      self,
      phi: types.FlowFieldVal,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> Tuple[types.FlowFieldVal, types.FlowFieldVal, types.FlowFieldVal]:
    """Determines the momentum to be used to compute the convection term."""
    momentum = [states[key] for key in common.KEYS_MOMENTUM]

    if self._scalar_name in ('q_r', 'q_s'):
      # Subtract term for rain terminal velocity from the vertical momentum.
      assert self._microphysics is not None, (
          f'Terminal velocity for {self._scalar_name} requires a microphysics'
          ' model but is undefined.'
      )
      rain_water_terminal_velocity = self._microphysics.terminal_velocity(
          self._scalar_name,
          {'rho_thermal': states['rho_thermal'], self._scalar_name: phi},
          additional_states,
      )
      rain_water_momentum = tf.nest.map_structure(
          tf.math.multiply, rain_water_terminal_velocity, states['rho']
      )
      momentum[self._g_dim] = tf.nest.map_structure(
          tf.math.subtract, momentum[self._g_dim], rain_water_momentum
      )

    return tuple(momentum)

  def source_fn(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      phi: types.FlowFieldVal,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldVal:
    """Computes the source term in the humidity equation.

    Args:
      replica_id: The index of the local core replica.
      replicas: A 3D array specifying the topology of the partition.
      phi: The humidity (one of `q_t`, `q_r`, `q_s`, `q_v`, and `q_c` that is
        defined in the ctor) at the present iteration.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.

    Returns:
      The source term of the humidity at the present step.
    """
    del replica_id, replicas

    if not isinstance(self._thermodynamics.model, water.Water):
      raise ValueError(
          '`water` thermodynamics is required for the humidity equation, but'
          f' {self._thermodynamics.model} is used.'
      )

    source = tf.nest.map_structure(tf.zeros_like, phi)

    thermo_states = self._get_thermodynamic_variables(
        phi, states, additional_states
    )

    # Compute vapor to rain water conversion term if needed.
    if self._include_precipitation:
      cloud_liquid_to_water_source = self._microphysics.humidity_source_fn(
          self._scalar_name,
          states,
          additional_states,
          thermo_states,
      )
      source = tf.nest.map_structure(
          tf.math.add, source, cloud_liquid_to_water_source
      )

    # Compute source terms
    if self._scalar_name == 'q_t' and self._include_subsidence:
      subsidence_source = eq_utils.source_by_subsidence_velocity(
          self._kernel_op,
          states[common.KEY_RHO],
          thermo_states['zz'],
          self._h[self._g_dim],
          thermo_states['q_c'],
          self._g_dim,
      )

      # Add external source, e.g. sponge forcing and subsidence.
      source = tf.nest.map_structure(tf.math.add, source, subsidence_source)

    if self._include_condensation:
      assert isinstance(self._thermodynamics.model, water.Water), (
          '`water` thermodynamics model is required to consider condensation in'
          ' the humidity equation.'
      )
      assert self._microphysics is not None, (
          'A microphysics model is required to consider condensation in the'
          ' humidity equation.'
      )
      cond = self._microphysics.condensation(
          states['rho_thermal'],
          thermo_states['T'],
          thermo_states['q_v'],
          thermo_states['q_l'],
          thermo_states['q_c'],
          thermo_states['zz'],
          additional_states,
      )
      cond_source = tf.nest.map_structure(
          tf.math.multiply, states[common.KEY_RHO], cond)
      op = None
      if self._scalar_name == 'q_c':
        op = tf.math.add
      elif self._scalar_name == 'q_v':
        op = tf.math.subtract
      else:
        raise NotImplementedError(
            f'Precipitation for {self._scalar_name} is not implemented. Only'
            ' "q_c" and "q_v" are supported.'
        )
      source = tf.nest.map_structure(op, source, cond_source)

    return source
