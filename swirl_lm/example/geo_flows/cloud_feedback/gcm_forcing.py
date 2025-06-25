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

"""A library for the source functions in the humidity equation."""

from typing import Optional

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.example.geo_flows.cloud_feedback import gcm_settings
from swirl_lm.numerics import convection
from swirl_lm.physics import constants
from swirl_lm.physics.radiation import rrtmgp
from swirl_lm.physics.thermodynamics import water
from swirl_lm.utility import grid_extension
from swirl_lm.utility import types
import tensorflow as tf

GRID_EXTENSION_KEY_PREFIX = grid_extension.GRID_EXTENSION_KEY_PREFIX


class GCMForcing:
  """Defines source terms to drive the LES with large-scale GCM forcings.

  This library defines source terms for the specific humidity and liquid-ice
  potential temperature transport equations that are a function of large-scale,
  time-averaged states resolved by a global circulation model (GCM).

  The large-scale forcing is derived from the Coupled Model Intercomparison
  Project Phase 5 (CMIP5) archive. The setup is very similar to the framework
  proposed in Shen et al. (2022), except that here we solve the anelastic
  equations with liquid-ice potential temperature as a prognostic variable
  instead of specific entropy.
  """

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
      sim_params: gcm_settings.GCMSettings,
      grid_extension_lib: Optional[grid_extension.GridExtension] = None,
  ):
    self._params = params
    self._kernel_op = params.kernel_op
    self._deriv_lib = params.deriv_lib
    # A thermodynamics manager that handles moisture related physics.
    self._thermodynamics = water.Water(params)
    self._microphysics = None
    self._g_dim = params.g_dim
    self.rrtmgp_lib = None
    if params.radiative_transfer is not None:
      self.rrtmgp_lib = rrtmgp.RRTMGP(params, grid_extension_lib)
    self._grid_spacing = (params.dx, params.dy, params.dz)[self._g_dim]
    self._sim_params = sim_params
    self._z_i = sim_params.z_i
    self._z_r = sim_params.z_r
    self._tau_r_tropo_sec = sim_params.tau_r_tropo_sec
    self._tau_r_wind_sec = sim_params.tau_r_wind_sec
    # Grid extension is only assumed to be necessary for non-stretched grids.
    self._use_grid_extension = not params.use_stretched_grid[self._g_dim]
    # Upper atmosphere states that are required for RRTMGP computations.
    self.upper_atmosphere_state_names = ('rho', 'q_t', 'p_ref', 'T', 'zz')

  def _gcm_vertical_advection(
      self,
      phi: types.FlowFieldVal,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldVal:
    """Computes the vertical advection due to the GCM large-scale subsidence.

    A first-order upwinding scheme is used.

    Args:
      phi: The scalar field that is advected.
      additional_states: A dictionary that holds all helper variables, in
        particular the GCM subsidence.

    Returns:
      A 3D field for the large-scale vertical advective tendency of `phi`.
    """
    deriv_phi = convection.first_order_upwinding(
        self._deriv_lib,
        phi,
        additional_states[gcm_settings.GCM_W_KEY],
        self._g_dim,
        additional_states,
    )
    return tf.nest.map_structure(
        tf.math.multiply, additional_states[gcm_settings.GCM_W_KEY], deriv_phi
    )

  def _q_t_horizontal_advective_tendency(
      self,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldVal:
    """Derives tendency of GCM specific humidity due to horizontal advection.

    The horizontal advective tendency is derived as the residual from the total
    advection after subtraction of the vertical advection.

    Args:
      additional_states: A dictionary that holds all helper variables, in
        particular the GCM advective tendency of total-water specific humidity.

    Returns:
      A 3D field for the tendency of the GCM total-water specific humidity due
      to large-scale horizontal advection only, in units of s⁻¹.
    """
    total_adv_src = additional_states[
        gcm_settings.GCM_ADV_TENDENCY_HUMIDITY_KEY
    ]
    # Remove an estimate of the vertical component from the total advective
    # tendency. The residual will be an estimate of the horizontal advective
    # tendency. Note that the GCM advective tendency corresponds to the negative
    # of the GCM advection term, so to remove the vertical contribution the
    # vertical advection term needs to be added; not subtracted.
    vadv_src = self._gcm_vertical_advection(
        additional_states[gcm_settings.GCM_Q_T_KEY],
        additional_states,
    )
    return tf.nest.map_structure(tf.math.add, total_adv_src, vadv_src)

  def _theta_li_horizontal_advective_tendency(
      self,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldVal:
    """Derives the tendency of the GCM `theta_li` due to horizontal advection.

    Args:
      additional_states: A dictionary that holds all helper variables, in
        particular the GCM advective tendency of total-water specific humidity.

    Returns:
      A 3D field for the tendency of the GCM's liquid-ice potential temperature
      due to large-scale horizontal advection only, in units of K s⁻¹.
    """
    temp_total_adv_src = additional_states[
        gcm_settings.GCM_ADV_TENDENCY_TEMPERATURE_KEY
    ]
    temp_vadv_src = self._gcm_vertical_advection(
        additional_states[gcm_settings.GCM_TEMPERATURE_KEY],
        additional_states,
    )
    vertical_coord_key = ['xx', 'yy', 'zz'][self._g_dim]
    exner_inv = self._thermodynamics.dry_exner_inverse(
        additional_states[vertical_coord_key], additional_states
    )

    def theta_li_adv_fn(temp_adv, temp_vadv, w_gcm, exner_inv):
      """Removes contributions of the pressure work and vertical advection."""
      pressure_work_term = w_gcm * constants.G / self._thermodynamics.cp_d
      j_adv = temp_adv + temp_vadv + pressure_work_term
      # Divide by the exner function to convert an absolute temperature tendency
      # to a liquid-ice potential temperature tendency. In theory, the fluxes of
      # the condensate mass fractions should also be taken into account, but
      # those are not available from the GCM output.
      return exner_inv * j_adv

    return tf.nest.map_structure(
        theta_li_adv_fn,
        temp_total_adv_src,
        temp_vadv_src,
        additional_states[gcm_settings.GCM_W_KEY],
        exner_inv,
    )

  def _inverse_relaxation_time_scale_fn(
      self,
      z: tf.Tensor,
  ) -> tf.Tensor:
    """Height-dependent relaxation coefficient for the free troposphere.

    The spatially variable relaxation coefficient is based on equation 11 of
    Shen et al. (2022). It vanishes in the lower troposphere, below z_i, and
    gradually increases to the user-defined 1 / `tau_r_topo_sec` over the height
    range z_r - z_i above z_i.

    Args:
      z: The vertical coordinate `tf.Tensor`.

    Returns:
      A `tf.Tensor` with the pointwise free troposphere relaxation coefficient.
    """
    cld_layer = 0.5 * (
        1.0 - tf.math.cos(np.pi * (z - self._z_i) / (self._z_r - self._z_i))
    )
    return (
        1.0
        / self._tau_r_tropo_sec
        * tf.where(
            condition=tf.less(z, self._z_i),
            x=tf.zeros_like(z),
            y=tf.where(
                condition=tf.less_equal(z, self._z_r),
                x=cld_layer,
                y=tf.ones_like(z),
            ),
        )
    )

  def _troposphere_relaxation_src(
      self,
      phi: types.FlowFieldVal,
      target: types.FlowFieldVal,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldVal:
    """Computes the relaxation to GCM states in the free troposphere.

    Args:
      phi: A 3D thermodynamic state from the large-eddy simulation.
      target: A time-averaged thermodynamic state from the GCM.
      additional_states: A dictionary that holds all helper variables, in
        particular the vertical coordinate `zz`.

    Returns:
      A 3D field for the spatially variable free troposphere relaxation source.
    """

    def relaxation_src_fn(phi: tf.Tensor, target: tf.Tensor, zz: tf.Tensor):
      relax_coeff = self._inverse_relaxation_time_scale_fn(zz)
      return relax_coeff * (target - phi)

    vertical_coord_key = ['xx', 'yy', 'zz'][self._g_dim]
    assert vertical_coord_key in additional_states, (
        'Vertical coordinate required in `additional_states` to compute'
        ' spatially variable relaxation coefficient.'
    )

    return tf.nest.map_structure(
        relaxation_src_fn, phi, target, additional_states[vertical_coord_key]
    )

  def q_t_source_fn(
      self,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldVal:
    """Computes the source term in the humidity equation.

    Args:
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.

    Returns:
      The source term of the total-water specific humidity equation at the
      present step.
    """
    subsidence_src = self._gcm_vertical_advection(
        states['q_t'], additional_states
    )
    horizontal_adv_src = self._q_t_horizontal_advective_tendency(
        additional_states
    )
    troposphere_src = self._troposphere_relaxation_src(
        states['q_t'],
        additional_states[gcm_settings.GCM_Q_T_KEY],
        additional_states,
    )
    combined_srcs = tf.nest.map_structure(
        lambda subs, adv, trop: -subs + adv + trop,
        subsidence_src,
        horizontal_adv_src,
        troposphere_src,
    )
    return tf.nest.map_structure(tf.math.multiply, combined_srcs, states['rho'])

  def theta_li_source_fn(
      self,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldVal:
    """Computes the forcing term in liquid-ice potential temperature equation.

    Args:
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.

    Returns:
      The source term of the liquid-ice potential temperature equation at the
      present step.
    """
    subsidence_src = self._gcm_vertical_advection(
        states['theta_li'],
        additional_states,
    )
    horizontal_advective_src = self._theta_li_horizontal_advective_tendency(
        additional_states
    )
    troposphere_src = self._troposphere_relaxation_src(
        states['theta_li'],
        additional_states[gcm_settings.GCM_THETA_LI_KEY],
        additional_states,
    )
    combined_srcs = tf.nest.map_structure(
        lambda subs, adv, trop: -subs + adv + trop,
        subsidence_src,
        horizontal_advective_src,
        troposphere_src,
    )
    return tf.nest.map_structure(tf.math.multiply, combined_srcs, states['rho'])

  def geostrophic_wind_forcing_fn(
      self,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldMap:
    """Forcing function for the horizontal winds."""
    def wind_relaxation_fn(rho: tf.Tensor, phi: tf.Tensor, target: tf.Tensor):
      return rho * (target - phi) / self._tau_r_wind_sec

    src_u = tf.nest.map_structure(
        wind_relaxation_fn,
        states['rho'],
        states['u'],
        additional_states[gcm_settings.GCM_U_KEY],
    )
    src_v = tf.nest.map_structure(
        wind_relaxation_fn,
        states['rho'],
        states['v'],
        additional_states[gcm_settings.GCM_V_KEY])

    return {
        'src_u': src_u,
        'src_v': src_v,
    }

  def radiative_heat_src(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldMap:
    """Computes the heating rate due to radiative transfer."""
    if self.rrtmgp_lib is None:
      return tf.nest.map_structure(tf.zeros_like, states['rho'])
    upper_atmosphere_states = None
    if self._use_grid_extension:
      upper_atmosphere_states = {
          k: additional_states[f'{GRID_EXTENSION_KEY_PREFIX}_{k}']
          for k in self.upper_atmosphere_state_names
      }
    return self.rrtmgp_lib.compute_heating_rate(
        replica_id,
        replicas,
        states,
        additional_states,
        sfc_temperature=self._sim_params.sfc_temperature,
        upper_atmosphere_states=upper_atmosphere_states,
    )
