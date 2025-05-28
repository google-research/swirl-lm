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

"""A library for the source functions in the potential temeprature equation."""

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.equations import common
from swirl_lm.equations import utils as eq_utils
from swirl_lm.equations.source_function import scalar_generic
from swirl_lm.physics.atmosphere import cloud
from swirl_lm.physics.atmosphere import microphysics_generic
from swirl_lm.physics.radiation import rrtmgp_common
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
      microphysics: microphysics_generic.MicrophysicsAdapter | None = None,
  ):
    """Retrieves context information for the potential temperature source."""
    super().__init__(kernel_op, params, scalar_name, thermodynamics)

    assert scalar_name in POTENTIAL_TEMPERATURE_VARNAME, (
        f'Source term function for {scalar_name} is not implemented. Supported'
        f' potential temperature types are: {POTENTIAL_TEMPERATURE_VARNAME}.'
    )

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
    if self._include_subsidence:
      assert self._g_dim is not None, (
          'The direction for gravity needs to be defined to include cloud'
          ' subsidence in the potential temperature equation.'
      )

    self._include_precipitation = (
        params.microphysics is not None and
        params.microphysics.include_precipitation)

    self._include_condensation = (
        params.microphysics is not None and
        params.microphysics.include_condensation)

    self._microphysics = microphysics
    self._cloud = None
    if isinstance(self._thermodynamics.model, water.Water):
      self._cloud = cloud.Cloud(self._thermodynamics.model)

    if self._include_precipitation or self._include_condensation:
      assert self._microphysics is not None, (
          'A microphysics model is required to consider evaporation or '
          'condensation in the potential temperature equation.'
      )

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

    if 'q_t' in states:
      q_t = states['q_t']
    else:
      q_t = tf.nest.map_structure(tf.math.add, states['q_c'], states['q_v'])

    thermo_states.update({'q_t': q_t})

    rho_thermal = states['rho_thermal']
    zz = additional_states.get('zz', tf.nest.map_structure(tf.zeros_like, phi))
    thermo_states.update({'zz': zz})

    # Compute the temperature.
    if 'q_c' in states and 'theta' in states:
      temperature = (
          self._thermodynamics.model.potential_temperature_to_temperature(
              'theta',
              states['theta'],
              q_t,
              states['q_c'],
              tf.nest.map_structure(tf.zeros_like, phi),
              zz,
              additional_states,
          )
      )
    else:
      temperature = self._thermodynamics.model.saturation_adjustment(
          self._scalar_name, phi, rho_thermal, q_t, zz, additional_states
      )
    thermo_states.update({'T': temperature})

    # Compute the potential temperature.
    if self._scalar_name == 'theta_li':
      buf = self._thermodynamics.model.potential_temperatures(
          temperature, q_t, rho_thermal, zz, additional_states
      )
      theta = buf['theta']
      thermo_states.update({'theta': theta})

    # We solve 'q_c' with a transport equation. Here we assume the ice phase is
    # absent.
    if 'q_c' in states:
      thermo_states['q_c'] = states['q_c']
      thermo_states['q_l'] = states['q_c']
      thermo_states['q_i'] = tf.nest.map_structure(tf.zeros_like, states['q_c'])
    else:
      # Compute the liquid and ice humidity.
      q_l, q_i = self._thermodynamics.model.equilibrium_phase_partition(
          temperature, rho_thermal, q_t
      )
      thermo_states.update({'q_l': q_l, 'q_i': q_i})
      thermo_states['q_c'] = tf.nest.map_structure(tf.math.add, q_l, q_i)

    if 'q_v' in states:
      thermo_states['q_v'] = states['q_v']
    else:
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
  ) -> types.ScalarSource:
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
      if self._params.radiative_transfer is not None:
        # Access the temperature tendency due to radiation, in K/s.
        radiative_heating_rate = additional_states[
            rrtmgp_common.KEY_APPLIED_RADIATION
        ]

        # Convert to potential temperature tendency due to radiation, in K/s.
        q_t = thermo_states['q_t']
        exner_inv = self._thermodynamics.model.exner_inverse(
            states['rho_thermal'],
            q_t,
            thermo_states['T'],
            thermo_states['zz'],
            additional_states,
        )
        radiative_heating_rate_theta = tf.nest.map_structure(
            tf.math.multiply, radiative_heating_rate, exner_inv
        )

        # The scalar equation is implemented in the form
        #     d/dt(rho*theta_li) = ... + source
        # Therefore, the units of the source term must be (kg/m^3) * K/s. Hence
        # we must multiply the potential temperature tendency by the density.
        source = tf.nest.map_structure(
            tf.math.multiply,
            radiative_heating_rate_theta,
            states[common.KEY_RHO],
        )
      else:
        # Default to a simplified radiation model that captures only longwave
        # radiative fluxes as a function of height and liquid water content
        # (Stevens et al. 2005, https://doi.org/10.1175/MWR2930.1).
        assert (
            self._g_dim is not None
        ), 'Gravity dimension must be 0, 1, or 2, but it is None.'
        if self._params.use_stretched_grid[self._g_dim]:
          raise NotImplementedError(
              'Stretched grid is not yet supported for radiation in the method'
              ' `Cloud.source_by_radiation`.'
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
        source = self._deriv_lib.deriv_centered(
            rad_src, self._g_dim, additional_states
        )

        q_t = thermo_states['q_t']
        cp_m = self._thermodynamics.model.cp_m(
            q_t, thermo_states['q_l'], thermo_states['q_i']
        )
        cp_m_inv = tf.nest.map_structure(tf.math.reciprocal, cp_m)
        exner_inv = self._thermodynamics.model.exner_inverse(
            states['rho_thermal'],
            q_t,
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
        q_t = thermo_states['q_t']
        theta_li = (
            self._thermodynamics.model.temperature_to_potential_temperature(
                'theta_li',
                thermo_states['T'],
                q_t,
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
          self._deriv_lib,
          states[common.KEY_RHO],
          thermo_states['zz'],
          theta_li,
          self._g_dim,
          additional_states,
      )
      source = tf.nest.map_structure(tf.math.add, source, src_subsidence)

    def condensation_source_fn(
        cond_or_precip: types.FlowFieldVal,
    ) -> types.FlowFieldVal:
      """Computes the source term due to condensation."""
      assert isinstance(self._thermodynamics.model, water.Water), (
          '`water` thermodynamics model is required for condensation. Current '
          'thermodynamics model:'
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

      q_t = thermo_states['q_t']
      cp = self._thermodynamics.model.cp_m(
          q_t, thermo_states['q_l'], thermo_states['q_i']
      )

      def get_condensation(
          rho: tf.Tensor,
          cp: tf.Tensor,
          t_0: tf.Tensor,
          theta_0: tf.Tensor,
          s: tf.Tensor,
      ) -> tf.Tensor:
        """Computes the condensation source term."""
        return (
            rho * (self._thermodynamics.model.lh_v0 / cp) * (theta_0 / t_0) * s
        )

      return tf.nest.map_structure(
          get_condensation,
          states[common.KEY_RHO],
          cp,
          t_0,
          theta_0,
          cond_or_precip,
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
          tf.math.add, source, condensation_source_fn(cond)
      )

    if self._include_precipitation:
      assert self._microphysics is not None, (
          'A microphysics model is required to consider precipitation in the'
          ' potential temperature equation.'
      )
      precip = self._microphysics.potential_temperature_source_fn(
          self._scalar_name,
          states,
          additional_states,
          thermo_states,
      )
      source = tf.nest.map_structure(tf.math.add, source, precip)

    return types.ScalarSource(total=source)
