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

"""Implementation of a radiative transfer solver."""

from typing import Any, Dict, Optional

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics.atmosphere import microphysics_one_moment
from swirl_lm.physics.radiation import rrtmgp_common
from swirl_lm.physics.radiation.rte import two_stream
from swirl_lm.physics.thermodynamics import water
from swirl_lm.utility import common_ops
from swirl_lm.utility import grid_extension
from swirl_lm.utility import stretched_grid_util
from swirl_lm.utility import types
import tensorflow as tf


FlowFieldMap = types.FlowFieldMap
FlowFieldVal = types.FlowFieldVal

PRIMARY_GRID_KEY = two_stream.PRIMARY_GRID_KEY
EXTENDED_GRID_KEY = two_stream.EXTENDED_GRID_KEY


class RRTMGP:
  """Rapid Radiative Transfer Model for General Circulation Models (RRTMGP)."""

  def __init__(
      self,
      config: parameters_lib.SwirlLMParameters,
      grid_extension_lib: Optional[grid_extension.GridExtension] = None,
  ):
    self._kernel_op = config.kernel_op
    self._kernel_op.add_kernel({
        'shift_up': ([1.0, 0.0, 0.0], 1),
        'shift_dn': ([0.0, 0.0, 1.0], 1)
    })
    self._config = config
    # A thermodynamics manager that handles moisture related physics.
    self._water = water.Water(config)
    # The vertical dimension.
    self._g_dim = config.g_dim
    # The number of ghost points on a side of the subgrid.
    self._halos = config.halo_width
    # The vertical grid spacing used in computing the local water path for an
    # atmospheric grid cell.
    self._dh = config.grid_spacings[self._g_dim]
    # Whether stretched grid is used in each dimension.
    self._use_stretched_grid = config.use_stretched_grid
    # The two-stream radiative transfer solver.
    self._two_stream_solver = two_stream.TwoStreamSolver(
        config.radiative_transfer,
        config,
        self._kernel_op,
        self._g_dim,
        grid_extension_lib
    )
    # Data library containing atmospheric gas concentrations.
    self._atmospheric_state = self._two_stream_solver.atmospheric_state
    # Library for 1-moment microphysics.
    self._microphysics_lib = microphysics_one_moment.Adapter(
        config, self._water
    )
    self._vertical_coord_name = ('xx', 'yy', 'zz')[self._g_dim]

  def _compute_cloud_path(
      self,
      rho: FlowFieldVal,
      q_c: FlowFieldVal,
      additional_states: FlowFieldMap,
  ) -> FlowFieldVal:
    """Computes the cloud water/ice path in an atmospheric grid cell."""
    if self._use_stretched_grid[self._g_dim]:
      h = additional_states[stretched_grid_util.h_key(self._g_dim)]
      return common_ops.map_structure_3d(lambda a, b, c: a * b * c, rho, q_c, h)
    else:
      def cloud_path_fn(rho: tf.Tensor, q_c: tf.Tensor) -> tf.Tensor:
        return rho * q_c * self._dh
      return tf.nest.map_structure(cloud_path_fn, rho, q_c)

  def _prepare_states(
      self,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> Dict[str, Any]:
    """Prepares the states for the two-stream radiative transfer solver."""
    assert 'rho' in states, (
        'RRTMGP requires the density (`rho`) to be present in `states`.'
    )
    assert 'q_t' in states, (
        'RRTMGP requires the total specific humidity (`q_t`) to be present in'
        ' `states`.'
    )
    assert 'T' in additional_states, (
        'RRTMGP requires the temperature (`T`) to be present in'
        ' `additional_states`.'
    )
    # On very rare occasions, the total-water specific humidity may be a small
    # negative value, likely because the source term is not bounded properly.
    # This is usually innocuous for the evolution of the transport equation, but
    # here may lead to negative water vapor, which leads to negative relative
    # abundance of certain gas species. This results in negative interpolations
    # of quantities like optical depth and Planck fraction that are inherently
    # nonnegative. As a precaution, we clip the total-water specific humidity at
    # 0.
    q_t = tf.nest.map_structure(
        tf.maximum,
        states['q_t'],
        tf.nest.map_structure(tf.zeros_like, states['q_t']),
    )

    # Condensed phase specific humidity required for cloud optics.
    if 'q_c' in additional_states:
      q_c = additional_states['q_c']
      liq_frac = self._water.liquid_fraction(additional_states['T'])
      q_liq = tf.nest.map_structure(tf.math.multiply, liq_frac, q_c)
      q_ice = tf.nest.map_structure(tf.math.subtract, q_c, q_liq)
    else:
      q_liq, q_ice = self._water.equilibrium_phase_partition(
          additional_states['T'], states['rho'], q_t
      )
      q_c = tf.nest.map_structure(tf.math.add, q_liq, q_ice)

    pressure = self._water.p_ref(
        additional_states[self._vertical_coord_name], additional_states
    )

    # Reconstructs volume mixing ratio (vmr) fields of relevant gas species.
    vmr_lib = self._atmospheric_state.vmr
    vmr_fields = vmr_lib.reconstruct_vmr_fields_from_pressure(pressure)

    # Derive the water vapor vmr from the simulation state itself.
    vmr_fields.update(
        {'h2o': self._water.humidity_to_volume_mixing_ratio(q_t, q_c)}
    )
    molecules_per_area = self._water.air_molecules_per_area(
        self._kernel_op, pressure, self._g_dim, vmr_fields['h2o']
    )
    lwp = self._compute_cloud_path(states['rho'], q_liq, additional_states)
    iwp = self._compute_cloud_path(states['rho'], q_ice, additional_states)
    cloud_r_eff_liq = self._microphysics_lib.cloud_particle_effective_radius(
        states['rho'], q_liq, 'l'
    )
    cloud_r_eff_ice = self._microphysics_lib.cloud_particle_effective_radius(
        states['rho'], q_ice, 'i'
    )
    return dict(
        pressure=pressure,
        temperature=additional_states['T'],
        molecules=molecules_per_area,
        vmr_fields=vmr_fields,
        cloud_r_eff_liq=cloud_r_eff_liq,
        cloud_path_liq=lwp,
        cloud_r_eff_ice=cloud_r_eff_ice,
        cloud_path_ice=iwp,
    )

  def _clear_sky_states(
      self,
      states: dict[str, Any],
  ) -> dict[str, Any]:
    """Removes all the cloud states from the input states."""
    cloud_state_names = (
        'cloud_r_eff_liq',
        'cloud_path_liq',
        'cloud_r_eff_ice',
        'cloud_path_ice',
    )
    return {k: v for k, v in states.items() if k not in cloud_state_names}

  def compute_heating_rate(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      sfc_temperature: Optional[FlowFieldVal | float] = None,
      upper_atmosphere_states: Optional[Dict[str, FlowFieldVal]] = None,
  ):
    """Computes the local heating rate due to radiative transfer.

    The optical properties of the layered atmosphere are computed using RRTMGP
    and the two-stream radiative transfer equation is solved for the net fluxes
    at the atmospheric grid cell faces. Based on the overall net radiative flux
    of the grid cell, a local heating rate is determined.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      states: A dictionary that holds all flow field variables and must include
        the total specific humidity (`q_t`) and density (`rho`).
      additional_states: A dictionary that holds all helper variables and must
        include temperatue (`T`), and the vertical coordinates (`zz`) if the
        reference states is height dependent.
      sfc_temperature: The optional surface temperature [K] represented as
        either a 3D field having a single vertical dimension or as a scalar.
      upper_atmosphere_states: An optional dictionary containing all the
        required states for computing radiative transfer in the extended grid
        above the simulation domain. These states will typically come directly
        from a single column of a global circulation model (GCM) in equilibrium.

    Returns:
      A dictionary containing any subset of the following entries, as long as
      the corresponding keys are present in `additional_states`:
      'rad_heat_src' -> The heating rate due to radiative transfer, in K/s.
      'rad_flux_lw' -> The net longwave radiative flux in the upper atmosphere,
        in W/m².
      'rad_flux_sw' -> The net shortwave radiative flux in the upper atmosphere,
        W/m².
      'rad_flux_lw_clear' -> The net longwave radiative flux in the upper
        atmosphere with cloud effects removed, in W/m².
      'rad_flux_sw_clear' -> The net shortwave radiative flux in the upper
        atmosphere with cloud effects removed, in W/m².
    """
    primary_grid_states = self._prepare_states(states, additional_states)
    extended_grid_states = None
    if upper_atmosphere_states is not None:
      state_keys = ('rho', 'q_t')
      additional_state_keys = [
          k
          for k in (self._vertical_coord_name, 'T', 'p_ref')
          if k in additional_states
      ]
      states_ext = {k: upper_atmosphere_states[k] for k in state_keys}
      additional_states_ext = {
          k: upper_atmosphere_states[k] for k in additional_state_keys
      }
      extended_grid_states = self._prepare_states(
          states_ext, additional_states_ext
      )

    lw_fluxes = self._two_stream_solver.solve_lw(
        replica_id,
        replicas,
        **primary_grid_states,
        sfc_temperature=sfc_temperature,
        extended_grid_states=extended_grid_states,
    )
    sw_fluxes = self._two_stream_solver.solve_sw(
        replica_id,
        replicas,
        **primary_grid_states,
        extended_grid_states=extended_grid_states,
    )
    flux_net = tf.nest.map_structure(
        tf.math.add, lw_fluxes['flux_net'], sw_fluxes['flux_net']
    )
    # Heating rate in (K / s).
    heating_rate = self._two_stream_solver.compute_heating_rate(
        flux_net, primary_grid_states['pressure']
    )
    output = {
        rrtmgp_common.KEY_STORED_RADIATION: heating_rate
    }
    rrtmgp_keys = rrtmgp_common.additional_keys(
        self._config.radiative_transfer, self._config.additional_state_keys
    )
    # Select only the diagnostic flux keys.
    diagnostic_flux_keys = [
        k
        for k in rrtmgp_keys
        if k not in rrtmgp_common.required_keys(self._config.radiative_transfer)
    ]

    if not diagnostic_flux_keys:
      return output

    # Construct input states with cloud properties removed in case clear sky
    # fluxes are requested.
    primary_grid_states_clr = self._clear_sky_states(primary_grid_states)
    extended_grid_states_clr = None
    if upper_atmosphere_states is not None:
      extended_grid_states_clr = self._clear_sky_states(extended_grid_states)

    # Get net fluxes (upwelling - downwelling), including those from the
    # extended grid, if requested.
    for k in diagnostic_flux_keys:
      flux_net_key = (
          f'{EXTENDED_GRID_KEY}_flux_net'
          if k.startswith(EXTENDED_GRID_KEY)
          else 'flux_net'
      )
      if k in (
          rrtmgp_common.KEY_RADIATIVE_FLUX_LW,
          rrtmgp_common.KEY_EXT_RADIATIVE_FLUX_LW,
      ):
        output[k] = lw_fluxes[flux_net_key]
      elif k in (
          rrtmgp_common.KEY_RADIATIVE_FLUX_SW,
          rrtmgp_common.KEY_EXT_RADIATIVE_FLUX_SW,
      ):
        output[k] = sw_fluxes[flux_net_key]
      # Compute clear sky fluxes by removing all cloud water and executing a
      # second pass of the two-stream solver.
      elif k in (
          rrtmgp_common.KEY_RADIATIVE_FLUX_LW_CLEAR,
          rrtmgp_common.KEY_EXT_RADIATIVE_FLUX_LW_CLEAR,
      ):
        lw_fluxes_clear = self._two_stream_solver.solve_lw(
            replica_id,
            replicas,
            **primary_grid_states_clr,
            sfc_temperature=sfc_temperature,
            extended_grid_states=extended_grid_states_clr,
        )
        output[k] = lw_fluxes_clear[flux_net_key]
      elif k in (
          rrtmgp_common.KEY_RADIATIVE_FLUX_SW_CLEAR,
          rrtmgp_common.KEY_EXT_RADIATIVE_FLUX_SW_CLEAR,
      ):
        sw_fluxes_clear = self._two_stream_solver.solve_sw(
            replica_id,
            replicas,
            **primary_grid_states_clr,
            extended_grid_states=extended_grid_states_clr,
        )
        output[k] = sw_fluxes_clear[flux_net_key]
      else:
        raise ValueError(f'Unknown RRTMGP flux key: {k}')

    return output
