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

"""A library for the source functions in the potential temeprature equation."""

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.equations import common
from swirl_lm.equations import utils as eq_utils
from swirl_lm.equations.source_function import scalar_generic
from swirl_lm.physics.atmosphere import cloud
from swirl_lm.physics.atmosphere import microphysics_kw1978
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.physics.thermodynamics import water
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf

POTENTIAL_TEMPERATURE_VARNAME = ('theta', 'theta_li')


class PotentialTemperature(scalar_generic.ScalarGeneric):
  """Defines functions for source terms in potential temperature equation."""

  def __init__(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      params: parameters_lib.SwirlLMParameters,
      scalar_name: str,
      thermodynamics: thermodynamics_manager.ThermodynamicsManager,
  ):
    """Retrieves context information for the potential temperature source."""
    super().__init__(kernel_op, params, scalar_name, thermodynamics)

    assert scalar_name in POTENTIAL_TEMPERATURE_VARNAME, (
        f'Source term function for {scalar_name} is not implemented. Supported'
        f' potential temperature types are: {POTENTIAL_TEMPERATURE_VARNAME}.'
    )

    if isinstance(self._thermodynamics.model, water.Water):
      self._microphysics = microphysics_kw1978.MicrophysicsKW1978(
          params, self._thermodynamics.model
      )
      self._cloud = cloud.Cloud(self._thermodynamics.model)
    else:
      self._microphysics = None
      self._cloud = None

    self._include_radiation = (
        self._scalar_params.HasField('potential_temperature')
        and self._scalar_params.potential_temperature.include_radiation
    )
    if self._include_radiation:
      assert self._g_dim is not None, (
          'The direction for gravity needs to be defined to include cloud'
          ' radiation in the potential temperature equation.'
      )

    self._include_subsidence = (
        self._scalar_params.HasField('potential_temperature')
        and self._scalar_params.potential_temperature.include_subsidence
    )
    if self._include_radiation:
      assert self._g_dim is not None, (
          'The direction for gravity needs to be defined to include cloud'
          ' subsidence in the potential temperature equation.'
      )

    self._include_condensation = (
        self._scalar_params.HasField('potential_temperature')
        and self._scalar_params.potential_temperature.include_condensation
    )

    self._include_evaporation = (
        self._scalar_params.HasField('potential_temperature')
        and self._scalar_params.potential_temperature.include_evaporation
    )

    self._grad_central = [
        lambda f: self._kernel_op.apply_kernel_op_x(f, 'kDx'),
        lambda f: self._kernel_op.apply_kernel_op_y(f, 'kDy'),
        lambda f: self._kernel_op.apply_kernel_op_z(f, 'kDz', 'kDzsh'),
    ]

  def _get_thermodynamic_variables(
      self,
      phi: types.FlowFieldVal,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldMap:
    """Computes thermodynamic variables required to evaluate terms in equation.

    Args:
      phi: The variable `scalar_name` at the present iteration.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.

    Returns:
      A dictionary of thermodynamic variables.
    """
    thermo_states = {self._scalar_name: phi}

    # No other thermodynamic states needs to be derived if the `water`
    # thermodynamics model is not used.
    if not isinstance(self._thermodynamics.model, water.Water):
      return thermo_states

    q_t = states['q_t']
    rho_thermal = states['rho_thermal']
    zz = additional_states.get('zz', tf.nest.map_structure(tf.zeros_like, phi))
    thermo_states.update({'zz': zz})

    # Compute the temperature.
    temperature = self._thermodynamics.model.saturation_adjustment(
        self._scalar_name, phi, rho_thermal, q_t, zz, additional_states)
    thermo_states.update({'T': temperature})

    # Compute the potential temperature.
    if self._scalar_name == 'theta_li':
      buf = self._thermodynamics.model.potential_temperatures(
          temperature, q_t, rho_thermal, zz
      )
      theta = buf['theta']
      thermo_states.update({'theta': theta})

    # Compute the liquid and ice humidity.
    q_l, q_i = self._thermodynamics.model.equilibrium_phase_partition(
        temperature, rho_thermal, q_t
    )
    thermo_states.update({'q_l': q_l, 'q_i': q_i})

    thermo_states['q_c'] = tf.nest.map_structure(tf.math.add, q_l, q_i)
    thermo_states['q_v'] = tf.nest.map_structure(
        tf.math.subtract, q_t, thermo_states['q_c']
    )

    return thermo_states

  def _get_wall_diffusive_flux_helper_variables(
      self,
      phi: types.FlowFieldVal,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldMap:
    """Prepares the helper variables for the diffusive flux in wall models.

    Args:
      phi: The variable `scalar_name` at the present iteration.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.

    Returns:
      A dictionary of variables required by wall diffusive flux closure models.
    """
    helper_variables = {key: states[key] for key in common.KEYS_VELOCITY}
    helper_variables.update(
        self._get_thermodynamic_variables(phi, states, additional_states)
    )
    return helper_variables

  def source_fn(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      phi: types.FlowFieldVal,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldVal:
    """Computes all possible source terms in potential temperature equation.

    Args:
      replica_id: The index of the local core replica.
      replicas: A 3D array specifying the topology of the partition.
      phi: The variable `scalar_name` at the present iteration.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.

    Returns:
      The source term of this scalar transport equation.
    """
    thermo_states = self._get_thermodynamic_variables(
        phi, states, additional_states
    )

    source = tf.nest.map_structure(tf.zeros_like, phi)

    if self._include_radiation:
      assert isinstance(self._thermodynamics.model, water.Water), (
          '`water` thermodynamics model is required to consider cloud radiation'
          ' in the potential temperature equation.'
      )
      halos = [self._params.halo_width] * 3
      f_r = self._cloud.source_by_radiation(
          thermo_states['q_l'],
          states['rho_thermal'],
          thermo_states['zz'],
          self._h[self._g_dim],
          self._g_dim,
          halos,
          replica_id,
          replicas,
      )

      radiation_source_fn = lambda rho, f_r: -rho * f_r

      rad_src = tf.nest.map_structure(
          radiation_source_fn, states[common.KEY_RHO], f_r
      )
      source = tf.nest.map_structure(
          lambda f: f / (2.0 * self._h[self._g_dim]),
          self._grad_central[self._g_dim](rad_src),
      )

      cp_m = self._thermodynamics.model.cp_m(
          states['q_t'], thermo_states['q_l'], thermo_states['q_i']
      )
      cp_m_inv = tf.nest.map_structure(tf.math.reciprocal, cp_m)
      exner_inv = self._thermodynamics.model.exner_inverse(
          states['rho_thermal'],
          states['q_t'],
          thermo_states['T'],
          thermo_states['zz'],
          additional_states,
      )
      cp_m_exner_inv = tf.nest.map_structure(
          tf.math.multiply, cp_m_inv, exner_inv
      )
      source = tf.nest.map_structure(tf.math.multiply, cp_m_exner_inv, source)

    if self._include_subsidence:
      assert isinstance(self._thermodynamics.model, water.Water), (
          '`water` thermodynamics model is required to consider cloud '
          ' subsidence in the potential temperature equation.'
      )
      if self._scalar_name == 'theta':
        theta_li = (
            self._thermodynamics.model.temperature_to_potential_temperature(
                'theta_li',
                thermo_states['T'],
                states['q_t'],
                thermo_states['q_l'],
                thermo_states['q_i'],
                thermo_states['zz'],
                additional_states,
            )
        )
      elif self._scalar_name == 'theta_li':
        theta_li = phi
      else:
        raise NotImplementedError(
            'Cloud subsidence in the potential temperature equation is'
            ' implemented for `theta` and `theta_li` only, but'
            f' {self._scalar_name} is provided.'
        )
      src_subsidence = eq_utils.source_by_subsidence_velocity(
          self._kernel_op,
          states[common.KEY_RHO],
          thermo_states['zz'],
          self._h[self._g_dim],
          theta_li,
          self._g_dim,
      )
      source = tf.nest.map_structure(tf.math.add, source, src_subsidence)

    def cond_or_evap_source_fn(
        cond_or_evap: types.FlowFieldVal,
    ) -> types.FlowFieldVal:
      """Computes the source term due to condensation or evaporation."""
      assert isinstance(self._thermodynamics.model, water.Water), (
          '`water` thermodynamics model is required for condensation and'
          ' evaporation. Current thermodynamics model:'
          f' {self._thermodynamics.model}'
      )
      t_0 = self._thermodynamics.model.t_ref(
          thermo_states['zz'], additional_states
      )
      zeros = tf.nest.map_structure(tf.zeros_like, t_0)
      theta_0 = self._thermodynamics.model.temperature_to_potential_temperature(
          'theta',
          t_0,
          zeros,
          zeros,
          zeros,
          thermo_states['zz'],
          additional_states,
      )
      cp = self._thermodynamics.model.cp_m(
          states['q_t'], thermo_states['q_l'], thermo_states['q_i']
      )

      def get_cond_or_evap(
          rho: tf.Tensor,
          cp: tf.Tensor,
          t_0: tf.Tensor,
          theta_0: tf.Tensor,
          s: tf.Tensor,
      ) -> tf.Tensor:
        """Computes the condensation or evaporation source term."""
        return (
            rho * (self._thermodynamics.model.lh_v0 / cp) * (theta_0 / t_0) * s
        )

      return tf.nest.map_structure(
          get_cond_or_evap,
          states[common.KEY_RHO],
          cp,
          t_0,
          theta_0,
          cond_or_evap,
      )

    if self._include_condensation:
      assert isinstance(self._thermodynamics.model, water.Water), (
          '`water` thermodynamics model is required to consider condensation in'
          ' the potential temperature equation.'
      )
      assert self._microphysics is not None, (
          'A microphysics model is required to consider condensation in the'
          ' potential temperature equation.'
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
      source = tf.nest.map_structure(
          tf.math.subtract, source, cond_or_evap_source_fn(cond)
      )

    if self._include_evaporation:
      assert isinstance(self._thermodynamics.model, water.Water), (
          '`water` thermodynamics model is required to consider evaporation in'
          ' the potential temperature equation.'
      )
      assert self._microphysics is not None, (
          'A microphysics model is required to consider evaporation in the'
          ' potential temperature equation.'
      )
      assert 'q_r' in states, '`q_r` is required for evaporation.'
      evap = self._microphysics.evaporation(
          states['rho_thermal'],
          thermo_states['T'],
          states['q_r'],
          thermo_states['q_v'],
          thermo_states['q_l'],
          thermo_states['q_c'],
      )
      source = tf.nest.map_structure(
          tf.math.subtract, source, cond_or_evap_source_fn(evap)
      )

    return source
