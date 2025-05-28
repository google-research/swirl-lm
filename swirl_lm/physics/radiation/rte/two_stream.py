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
"""A library for solving the two-stream radiative transfer equation."""

from typing import Any, Dict, Optional

import numpy as np
from swirl_lm.physics import constants
from swirl_lm.physics.radiation.config import radiative_transfer_pb2
from swirl_lm.physics.radiation.optics import atmospheric_state
from swirl_lm.physics.radiation.optics import lookup_gas_optics_base
from swirl_lm.physics.radiation.optics import optics
from swirl_lm.physics.radiation.rte import monochromatic_two_stream
import swirl_lm.physics.radiation.rte.rte_utils as utils
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_extension
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf


AbstractLookupGasOptics = lookup_gas_optics_base.AbstractLookupGasOptics
AtmosphericState = atmospheric_state.AtmosphericState
FlowFieldMap = types.FlowFieldMap
FlowFieldVal = types.FlowFieldVal

PRIMARY_GRID_KEY = utils.PRIMARY_GRID_KEY
EXTENDED_GRID_KEY = utils.EXTENDED_GRID_KEY


class TwoStreamSolver:
  """A library for solving the two-stream radiative transfer equation.

  Attributes:
    atmospheric_state: An instance of `AtmosphericState` containing volume
      mixing ratio profiles for prevalent atmospheric gases and flux boundary
      conditions at particular times and geographic locations.
  """

  def __init__(
      self,
      radiation_params: radiative_transfer_pb2.RadiativeTransfer,
      grid_params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      g_dim: int,
      grid_extension_lib: Optional[grid_extension.GridExtension] = None,
  ):
    self.atmospheric_state = AtmosphericState.from_proto(
        radiation_params.atmospheric_state
    )
    self._g_dim = g_dim
    self._halos = grid_params.halo_width
    self._optics_lib = optics.optics_factory(
        radiation_params.optics,
        kernel_op,
        g_dim,
        self._halos,
        self.atmospheric_state.vmr,
    )
    self._monochrom_solver = (
        monochromatic_two_stream.MonochromaticTwoStreamSolver(
            grid_params, kernel_op, g_dim, grid_extension_lib
        )
    )
    self._rte_utils = utils.RTEUtils(grid_params, grid_extension_lib)

    # Operators used when computing heating rate from fluxes.
    self._grad_central = (
        lambda f: kernel_op.apply_kernel_op_x(f, 'kDx'),
        lambda f: kernel_op.apply_kernel_op_y(f, 'kDy'),
        lambda f: kernel_op.apply_kernel_op_z(f, 'kDz', 'kDzsh'),
    )[g_dim]
    self._grad_forward_fn = (
        lambda f: kernel_op.apply_kernel_op_x(f, 'kdx+'),
        lambda f: kernel_op.apply_kernel_op_y(f, 'kdy+'),
        lambda f: kernel_op.apply_kernel_op_z(f, 'kdz+', 'kdz+sh'),
    )[g_dim]

    # Longwave parameters.
    self._top_flux_down_lw = self.atmospheric_state.toa_flux_lw
    self._sfc_emissivity_lw = self.atmospheric_state.sfc_emis

    # Shortwave parameters.
    self._sfc_albedo = self.atmospheric_state.sfc_alb
    self._zenith = self.atmospheric_state.zenith
    self._total_solar_irrad = self.atmospheric_state.irrad
    self._solar_fraction_by_gpt = self._optics_lib.solar_fraction_by_gpt
    self._flux_keys = ['flux_up', 'flux_down', 'flux_net']
    if grid_extension_lib is not None:
      self._flux_keys += [
          f'{EXTENDED_GRID_KEY}_{k}' for k in self._flux_keys
      ]

  def _compute_local_properties_lw(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      pressure: FlowFieldVal,
      temperature: FlowFieldVal,
      molecules: FlowFieldVal,
      igpt: tf.Tensor,
      vmr_fields: Optional[Dict[str, FlowFieldVal]] = None,
      sfc_temperature: Optional[FlowFieldVal | float] = None,
      cloud_r_eff_liq: Optional[FlowFieldVal] = None,
      cloud_path_liq: Optional[FlowFieldVal] = None,
      cloud_r_eff_ice: Optional[FlowFieldVal] = None,
      cloud_path_ice: Optional[FlowFieldVal] = None,
  ) -> FlowFieldMap:
    """Computes local optical properties for longwave radiative transfer."""
    if isinstance(sfc_temperature, float):
      # Create a plane for the surface temperature representation.
      sfc_temperature = tf.nest.map_structure(
          lambda x: sfc_temperature * tf.ones_like(x),
          common_ops.slice_field(pressure, self._g_dim, 0, size=1)
      )
    lw_optical_props = dict(
        self._optics_lib.compute_lw_optical_properties(
            pressure,
            temperature,
            molecules,
            igpt,
            vmr_fields=vmr_fields,
            cloud_r_eff_liq=cloud_r_eff_liq,
            cloud_path_liq=cloud_path_liq,
            cloud_r_eff_ice=cloud_r_eff_ice,
            cloud_path_ice=cloud_path_ice,
        )
    )
    planck_srcs = dict(
        self._optics_lib.compute_planck_sources(
            replica_id,
            replicas,
            pressure,
            temperature,
            igpt,
            vmr_fields,
            sfc_temperature=sfc_temperature,
        )
    )
    sfc_src = planck_srcs.get(
        'planck_src_sfc',
        common_ops.slice_field(
            planck_srcs['planck_src_bottom'],
            self._g_dim,
            self._halos,
            size=1,
        ),
    )
    combined_srcs = self._monochrom_solver.lw_combine_sources(planck_srcs)
    lw_optical_props['level_src_bottom'] = combined_srcs['planck_src_bottom']
    lw_optical_props['level_src_top'] = combined_srcs['planck_src_top']
    src_and_props = dict(
        self._monochrom_solver.lw_cell_source_and_properties(**lw_optical_props)
    )
    src_and_props['sfc_src'] = sfc_src
    return src_and_props

  def _reindex_vmr_fields(
      self,
      vmr_fields: Dict[str, FlowFieldVal],
      gas_optics_lib: AbstractLookupGasOptics,
  ) -> Dict[int, FlowFieldVal]:
    """Converts the chemical formulas of the gas species to RRTM indices."""
    return {gas_optics_lib.idx_gases[k]: v for k, v in vmr_fields.items()}

  def solve_lw(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      pressure: FlowFieldVal,
      temperature: FlowFieldVal,
      molecules: FlowFieldVal,
      vmr_fields: Optional[Dict[str, FlowFieldVal]] = None,
      sfc_temperature: Optional[FlowFieldVal | float] = None,
      cloud_r_eff_liq: Optional[FlowFieldVal] = None,
      cloud_path_liq: Optional[FlowFieldVal] = None,
      cloud_r_eff_ice: Optional[FlowFieldVal] = None,
      cloud_path_ice: Optional[FlowFieldVal] = None,
      extended_grid_states: Optional[Dict[str, Any]] = None,
  ) -> FlowFieldMap:
    """Solves two-stream radiative transfer equation over the longwave spectrum.

    Local optical properties like optical depth, single-scattering albedo, and
    asymmetry factor are computed using an optics library and transformed to
    two-stream approximations of reflectance and transmittance. The sources of
    longwave radiation are the Planck sources, which are a function only of
    temperature. To obtain the cell-centered directional Planck sources, the
    sources are first computed at the cell boundaries and the net source
    emanating from the grid cell is determined. Each spectral interval,
    represented by a g-point, is a separate radiative transfer problem, and can
    be computed in parallel. Finally, the independently solved fluxes are summed
    over the full spectrum to yield the final upwelling and downwelling fluxes.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      pressure: The pressure field [Pa].
      temperature: The temperature field [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules/m²].
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by the chemical formula.
      sfc_temperature: The optional surface temperature represented as either a
        3D field having a single vertical dimension or as a scalar [K].
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m²].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m²].
      extended_grid_states: An optional dictionary of states for the extended
        grid above the simulation domain with the same states as the ones in the
        argument list of this function and keyed by the same argument names:
        'pressure', 'temperature', etc.

    Returns:
      A dictionary with the following entries (in units of W/m²):
      `flux_up`: The upwelling longwave radiative flux at cell face i - 1/2.
      `flux_down`: The downwelling longwave radiative flux at face i - 1/2.
      `flux_net`: The net longwave radiative flux at face i - 1/2.
      If an extended grid is present, the following entries are also added:
      'extended_flux_up' -> The upwelling radiative flux in the extended grid.
      'extended_flux_down' -> The downwelling radiative flux in extended grid.
      'extended_flux_net' -> The net radiative flux in the extended grid.
    """
    # Convert the chemical formulas of the gas species to RRTM-consistent
    # numerical identifiers.
    if vmr_fields is not None and self._optics_lib.gas_optics_lw is not None:
      gas_optics_lib = self._optics_lib.gas_optics_lw
      vmr_fields = self._reindex_vmr_fields(vmr_fields, gas_optics_lib)
      if extended_grid_states is not None:
        # Create shallow copy to prevent altering the original dictionary.
        extended_grid_states = dict(extended_grid_states)
        extended_grid_states['vmr_fields'] = self._reindex_vmr_fields(
            extended_grid_states['vmr_fields'], gas_optics_lib
        )

    def step_fn(igpt, cumulative_flux):
      optical_props_2stream = self._compute_local_properties_lw(
          replica_id,
          replicas,
          pressure,
          temperature,
          molecules,
          igpt,
          vmr_fields,
          sfc_temperature,
          cloud_r_eff_liq,
          cloud_path_liq,
          cloud_r_eff_ice,
          cloud_path_ice,
      )
      # Handle extended grid if one is present.
      optical_props_2stream_ext = None
      if extended_grid_states is not None:
        optical_props_2stream_ext = self._compute_local_properties_lw(
            replica_id,
            replicas,
            **extended_grid_states,
            igpt=igpt,
        )

      # Boundary conditions.
      sfc_src = optical_props_2stream['sfc_src']
      top_flux_down = tf.nest.map_structure(
          lambda x: self._top_flux_down_lw * tf.ones_like(x), sfc_src
      )
      sfc_emissivity = tf.nest.map_structure(
          lambda x: self._sfc_emissivity_lw * tf.ones_like(x), sfc_src
      )
      fluxes = self._monochrom_solver.lw_transport(
          replica_id,
          replicas,
          top_flux_down=top_flux_down,
          sfc_emissivity=sfc_emissivity,
          **optical_props_2stream,
          extended_grid_optical_props=optical_props_2stream_ext,
      )
      return igpt + 1, tf.nest.map_structure(
          tf.math.add, fluxes, cumulative_flux,
      )

    stop_condition = lambda i, states: i < self._optics_lib.n_gpt_lw
    lw_fluxes0 = {
        k: tf.nest.map_structure(tf.zeros_like, pressure)
        for k in self._flux_keys
    }
    i0 = tf.constant(0)
    _, fluxes = tf.nest.map_structure(
        tf.stop_gradient,
        tf.while_loop(
            cond=stop_condition,
            body=step_fn,
            loop_vars=(i0, lw_fluxes0),
            parallel_iterations=self._optics_lib.n_gpt_lw,
        ),
    )
    return fluxes

  def solve_sw(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      pressure: FlowFieldVal,
      temperature: FlowFieldVal,
      molecules: FlowFieldVal,
      vmr_fields: Optional[Dict[str, FlowFieldVal]] = None,
      cloud_r_eff_liq: Optional[FlowFieldVal] = None,
      cloud_path_liq: Optional[FlowFieldVal] = None,
      cloud_r_eff_ice: Optional[FlowFieldVal] = None,
      cloud_path_ice: Optional[FlowFieldVal] = None,
      extended_grid_states: Optional[Dict[str, Any]] = None,
  ) -> FlowFieldMap:
    """Solves the two-stream radiative transfer equation for shortwave.

    Local optical properties like optical depth, single-scattering albedo, and
    asymmetry factor are computed using an optics library and transformed to
    two-stream approximations of reflectance and transmittance. The sources of
    shortwave radiation are determined by the diffuse propagation of direct
    solar radiation through the layered atmosphere. Each spectral interval,
    represented by a g-point, is a separate radiative transfer problem, and can
    be computed in parallel. Finally, the independently solved fluxes are summed
    over the full spectrum to yield the final upwelling and downwelling fluxes.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      pressure: The pressure field [Pa].
      temperature: The temperature field [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules/m²].
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index.
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m²].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m²].
      extended_grid_states: An optional dictionary of states for the extended
        grid above the simulation domain with the same states as the ones in the
        argument list of this function and keyed by the same argument names:
        'pressure', 'temperature', etc.

    Returns:
      A dictionary with the following entries (in units of W/m²):
      `flux_up`: The upwelling shortwave radiative flux at cell face i - 1/2.
      `flux_down`: The downwelling shortwave radiative flux at face i - 1/2.
      `flux_net`: The net shortwave radiative flux at face i - 1/2.
      If an extended grid is present, the following entries are also added:
      'extended_flux_up' -> The upwelling radiative flux in the extended grid.
      'extended_flux_down' -> The downwelling radiative flux in extended grid.
      'extended_flux_net' -> The net radiative flux in the extended grid.
    """

    def field_like(f: FlowFieldVal, val):
      return tf.nest.map_structure(lambda x: val * tf.ones_like(x), f)

    # Convert the chemical formulas of the gas species to RRTM-consistent
    # numerical identifiers.
    if vmr_fields is not None and self._optics_lib.gas_optics_sw is not None:
      gas_optics_lib = self._optics_lib.gas_optics_sw
      vmr_fields = self._reindex_vmr_fields(
          vmr_fields, gas_optics_lib
      )
      if extended_grid_states is not None:
        # Create shallow copy to prevent altering the original dictionary.
        extended_grid_states = dict(extended_grid_states)
        extended_grid_states['vmr_fields'] = self._reindex_vmr_fields(
            extended_grid_states['vmr_fields'], gas_optics_lib
        )

    def step_fn(igpt, partial_fluxes):
      sw_optical_props = self._optics_lib.compute_sw_optical_properties(
          pressure,
          temperature,
          molecules,
          igpt,
          vmr_fields=vmr_fields,
          cloud_r_eff_liq=cloud_r_eff_liq,
          cloud_path_liq=cloud_path_liq,
          cloud_r_eff_ice=cloud_r_eff_ice,
          cloud_path_ice=cloud_path_ice,
      )
      optical_props_2stream = self._monochrom_solver.sw_cell_properties(
          zenith=self._zenith,
          **sw_optical_props,
      )
      sfc_albedo = tf.nest.map_structure(
          lambda x: self._sfc_albedo * tf.ones_like(x),
          common_ops.slice_field(
              sw_optical_props['optical_depth'], self._g_dim, 0, size=1
          ),
      )
      # Monochromatic top of atmosphere flux.
      solar_flux = self._total_solar_irrad * self._solar_fraction_by_gpt[igpt]
      toa_flux = field_like(sfc_albedo, val=solar_flux)

      # Handle extended grid, if one is present.
      use_extended_grid = extended_grid_states is not None
      optical_props_2stream_ext = None

      if use_extended_grid:
        optical_props_2stream_ext = dict(
            self._optics_lib.compute_sw_optical_properties(
                **extended_grid_states,
                igpt=igpt,
            )
        )
        optical_props_2stream_ext.update(
            dict(
                self._monochrom_solver.sw_cell_properties(
                    zenith=self._zenith,
                    **optical_props_2stream_ext,
                )
            )
        )

      sources_2stream_full_grid = self._monochrom_solver.sw_cell_source(
          replica_id,
          replicas,
          t_dir=optical_props_2stream['t_dir'],
          r_dir=optical_props_2stream['r_dir'],
          optical_depth=sw_optical_props['optical_depth'],
          toa_flux=toa_flux,
          sfc_albedo_direct=sfc_albedo,
          zenith=self._zenith,
          extended_grid_optical_props=optical_props_2stream_ext
      )

      # Extract out primary and extended grid sources.
      sources_2stream = sources_2stream_full_grid[PRIMARY_GRID_KEY]
      if optical_props_2stream_ext is not None:
        sources_2stream_ext = sources_2stream_full_grid[EXTENDED_GRID_KEY]
        optical_props_2stream_ext.update(sources_2stream_ext)

      sw_fluxes = self._monochrom_solver.sw_transport(
          replica_id,
          replicas,
          t_diff=optical_props_2stream['t_diff'],
          r_diff=optical_props_2stream['r_diff'],
          src_up=sources_2stream['src_up'],
          src_down=sources_2stream['src_down'],
          sfc_src=sources_2stream['sfc_src'],
          sfc_albedo=sfc_albedo,
          flux_down_dir=sources_2stream['flux_down_dir'],
          extended_grid_optical_props=optical_props_2stream_ext,
      )
      total_sw_fluxes = tf.nest.map_structure(
          tf.math.add, sw_fluxes, partial_fluxes
      )
      return igpt + 1, total_sw_fluxes

    stop_condition = lambda i, states: i < self._optics_lib.n_gpt_sw

    fluxes_0 = {
        k: tf.nest.map_structure(tf.zeros_like, pressure)
        for k in self._flux_keys
    }
    if self._zenith >= 0.5 * np.pi:
      return fluxes_0
    else:
      i0 = tf.constant(0)
      _, fluxes = tf.nest.map_structure(
          tf.stop_gradient,
          tf.while_loop(
              cond=stop_condition,
              body=step_fn,
              loop_vars=(i0, fluxes_0),
              parallel_iterations=self._optics_lib.n_gpt_sw,
          ),
      )
      return fluxes

  def compute_heating_rate(
      self,
      flux_net: FlowFieldVal,
      pressure: FlowFieldVal,
  ) -> FlowFieldVal:
    """Computes cell-center heating rate from pressure and net radiative flux.

    The net radiative flux corresponds to the bottom cell face. The difference
    of the net flux at the top face and that at the bottom face gives the total
    net flux out of the grid cell. Using the pressure difference across the grid
    cell, the net flux can be converted to a heating rate, in K/s.

    Args:
      flux_net: The net flux at the bottom face [W/m²].
      pressure: The pressure field [Pa].

    Returns:
      The heating rate of the grid cell [K/s].
    """
    # Pressure difference across the atmospheric grid cell.
    dp = tf.nest.map_structure(
        lambda dp_: dp_ / 2.0, self._grad_central(pressure)
    )

    def heating_rate_fn(dflux: tf.Tensor, dp: tf.Tensor):
      """Computes the heating rate at the grid cell center [W]."""
      return constants.G * dflux / dp / constants.CP

    return tf.nest.map_structure(
        heating_rate_fn,
        self._grad_forward_fn(flux_net),
        dp)
