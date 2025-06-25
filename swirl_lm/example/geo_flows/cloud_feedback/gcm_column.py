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

"""Setup for emulating a single grid column of a Global Circulation Model."""

import functools
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np
from swirl_lm.base import initializer
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.example.geo_flows import geophysical_flow_common as common_lib
from swirl_lm.example.geo_flows.cloud_feedback import gcm_forcing
from swirl_lm.example.geo_flows.cloud_feedback import gcm_settings
from swirl_lm.example.shared import cloud_utils
from swirl_lm.example.shared import geophysical_flow_utils
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_extension
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import init_fn as init_fn_lib
from swirl_lm.utility import types
import tensorflow as tf


FlowFieldMap = types.FlowFieldMap
FlowFieldVal = types.FlowFieldVal
GRID_EXTENSION_KEY_PREFIX = grid_extension.GRID_EXTENSION_KEY_PREFIX

# Used to seed the theta_li random noise.
THETA_LI_SEED: int = 96234

# The step size used to solve for potential temperature iteratively.
_STEP_SIZE = 0.3

_HaloUpdateFn = Callable[..., FlowFieldVal]


class GCMColumn(common_lib.GeophysicalFlowSetup):
  """Defines initial conditions and essential fields for a single GCM column."""

  def __init__(
      self,
      config: parameters_lib.SwirlLMParameters,
      sim_params: gcm_settings.GCMSettings,
      cloud_util: cloud_utils.CloudUtils,
  ):
    super().__init__(config)

    self.sim_params = sim_params
    self.cloud_utils = cloud_util
    assert (
        thermodynamics := self.cloud_utils.config.thermodynamics
    ) is not None, 'Thermodynamics must be set in the config.'
    self.reference_state_type = thermodynamics.water.WhichOneof(
        'reference_state'
    )

    zz = np.asarray(self.config.global_xyz[self.g_dim], dtype=np.float64)
    domain_height = [self.config.lx, self.config.ly, self.config.lz][self.g_dim]

    # Use an extended grid to represent the upper atmosphere with a coarser grid
    # to enable solving the nonlocal radiative transfer equations.
    # If a stretched grid is being used in the vertical direction, assume that
    # it extends all the way to the top-of-atmosphere boundary of the radiative
    # transfer model, and the extended grid is not necessary.
    self.grid_extension_lib = None
    self._use_grid_extension = not config.use_stretched_grid[self.g_dim]
    if self._use_grid_extension:
      # Top of atmosphere level where radiative transfer boundary conditions are
      # specified.
      toa_height = np.max(
          common_lib._load_csv(sim_params.sounding_csv_filename)['z']
      )
      assert toa_height > domain_height, (
          'The top of atmosphere height must be larger than the domain height'
          ' when using an extended grid.'
      )
      self._dh = config.grid_spacings[self.g_dim]
      # To be compatible with the top bounday condition of the primary grid,
      # preserve the uniform spacing of the primary grid in the first levels of
      # the extended grid that fall under the halo range of the primary grid.
      extended_grid_buffer = [
          domain_height + self._dh * i
          for i in range(1, config.halo_width)
      ]
      zz_extended = np.linspace(
          extended_grid_buffer[-1] + self._dh,
          toa_height,
          len(zz) - len(extended_grid_buffer),
          endpoint=True,
      )
      zz_extended = np.concatenate([extended_grid_buffer, zz_extended], axis=0)
      # Use the last two levels of the extended grid to extract its coarser
      # spacing.
      self._extended_dh = float((zz_extended[-1] - zz_extended[-2]))
      self.grid_extension_lib = grid_extension.GridExtension(
          config, zz_extended, self.g_dim
      )
      # The full grid including the extension.
      zz = np.concatenate([zz, zz_extended], axis=0)
    # Attach boundary heights to the full grid so the sounding can be properly
    # extrapolated at the boundaries.
    dz_low = zz[1] - zz[0]
    dz_high = zz[-1] - zz[-2]
    lower_boundary = [
        zz[0] - dz_low * i for i in range(1, config.halo_width + 1)
    ][::-1]
    upper_boundary = [
        zz[-1] + dz_high * i for i in range(1, config.halo_width + 1)
    ]
    zz = np.concatenate([lower_boundary, zz, upper_boundary], axis=0)
    sounding = common_lib.load_sounding(
        sim_params.sounding_csv_filename.split(','), zz
    )
    sounding['zz'] = zz
    # Since the pressure gradient is not always available in the GCM output, we
    # take the horizontal winds at altitude `z_i` as the geostrophic winds that
    # are used in the Coriolis forcing function. Note that in addition to the
    # Coriolis forcing we also relax the horizontal winds to the GCM winds
    # uniformly across the entire domain.
    geostrophic_slice = common_lib.interpolate_sounding(
        sounding, np.asarray([self.sim_params.z_i]), 'zz'
    )
    self.geostrophic_wind = {
        'u': np.mean(geostrophic_slice['u']),
        'v': np.mean(geostrophic_slice['v']),
        'w': 0.0,
    }

    if self.sim_params.latitude is not None:
      self.coriolis_force_fn = cloud_utils.coriolis_force(
          self.sim_params.latitude, self.geostrophic_wind, self.g_dim)

    if 'theta_li' in sounding:
      sounding.update(
          self.compute_reference_state_from_theta_li(
              sounding['theta_li'], sounding['q_t'], zz
          )
      )
    elif 'temperature' in sounding:
      sounding.update(
          self.compute_reference_state_from_temperature(
              sounding['temperature'], sounding['q_t'], zz
          )
      )
    else:
      raise NotImplementedError(
          'Unsupported energy variable provided in GCM data.'
      )

    # If an extended grid is used, partition the sounding variables into the
    # primary grid (bottom half) and the extended grid (top half).
    if self._use_grid_extension:
      n = len(zz)
      for k in list(sounding.keys()):
        # Primary grid.
        primary_grid_sounding = sounding[k][: n // 2 + config.halo_width]
        # Extended grid.
        extended_state_key = f'{GRID_EXTENSION_KEY_PREFIX}_{k}'
        sounding[extended_state_key] = sounding[k][n // 2 - config.halo_width:]
        sounding[k] = primary_grid_sounding

    self.sounding = {
        k: tf.cast(v, dtype=types.TF_DTYPE) for k, v in sounding.items()
    }
    self.init_wind = {
        'u': sounding['u'],
        'v': sounding['v'],
        'w': 0.0,
    }

    self._gcm_forcing_lib = gcm_forcing.GCMForcing(
        config, sim_params, self.grid_extension_lib
    )
    # Initialize radiation update function if configuration is available.
    self.radiation_src_update_fn = None
    if self._gcm_forcing_lib.rrtmgp_lib is not None:
      self.radiation_src_update_fn = self._gcm_forcing_lib.radiative_heat_src

  def _iterative_solve(self, iterative_fn, states_0):
    """Solves for a variable iteratively considering arbitrary states."""
    max_iters = 20
    tol = 1e-4

    def cond(i: tf.Tensor, states: FlowFieldMap) -> tf.Tensor:
      """The continue condition of the temperature iteration."""
      return tf.math.logical_and(
          tf.less(i, max_iters),
          tf.math.reduce_any(
              tf.nest.map_structure(
                  lambda res: tf.greater(tf.math.abs(res), tol),
                  states['res'],
              )
          ),
      )

    i0 = tf.constant(0)
    return tf.while_loop(
        cond=cond,
        body=iterative_fn,
        loop_vars=(i0, states_0),
        back_prop=False,
    )[1]

  def compute_reference_state_from_theta_li(
      self,
      theta_li: np.ndarray,
      q_t: np.ndarray,
      zz: np.ndarray,
  ) -> FlowFieldMap:
    """Computes reference state consistent with initial theta_li from sounding.

    Args:
      theta_li: The initial liquid-ice potential temperature, in units of K.
      q_t: The initial total water specific humidity, in units of kg/kg.
      zz: The vertical coordinates, in units of m.

    Returns:
      The reference potential temperature and pressure that are consistent with
      the initial `theta_li` and `q_t`.
    """
    # Variables in the loop are:
    # 'p': The reference pressure consistent with initial conditions.
    # 'theta': Reference theta consistent with initial conditions.
    # 'q_l': The liquid-phase specific humidity.
    # 'q_i': The ice-phase specific humidity.
    zeros = tf.nest.map_structure(tf.zeros_like, zz)
    states_0 = {
        'p': tf.convert_to_tensor(
            self.hydrostatic_pressure_from_energy_variable(
                'theta', theta_li, zz, self.config.p_thermal)
        ),
        'theta': tf.convert_to_tensor(theta_li),
        'q_l': zeros,
        'q_i': zeros,
        'res': tf.convert_to_tensor(theta_li),
    }

    def body(
        i: tf.Tensor, states: FlowFieldMap
    ) -> Tuple[tf.Tensor, FlowFieldMap]:
      """Solves the potential temperature iteratively."""

      helper_states = {'p_ref': states['p']}
      thermal_states = self.cloud_utils.temperature(
          tf.convert_to_tensor(theta_li), tf.convert_to_tensor(q_t),
          tf.convert_to_tensor(zz), 'theta_li', helper_states
      )
      temperature = thermal_states['temperature']
      q_l = states['q_l']
      q_i = states['q_i']
      theta = (
          self.cloud_utils.thermodynamics.temperature_to_potential_temperature(
              'theta',
              temperature,
              tf.convert_to_tensor(q_t),
              q_l,
              q_i,
              tf.convert_to_tensor(zz),
              helper_states,
          )
      )

      theta = tf.nest.map_structure(
          lambda theta_new, theta_old: theta_old
          + _STEP_SIZE * (theta_new - theta_old),
          theta,
          states['theta'],
      )

      p = tf.convert_to_tensor(
          self.hydrostatic_pressure_from_energy_variable(
              'theta', theta, zz, self.config.p_thermal
          ))

      res = tf.nest.map_structure(tf.math.subtract, theta, states['theta'])

      return i + 1, {'p': p, 'theta': theta, 'q_l': q_l, 'q_i': q_i, 'res': res}

    states = self._iterative_solve(body, states_0)
    return {
        'theta_ref': states['theta'],
        'p_ref': states['p'],
    }

  def hydrostatic_pressure_from_energy_variable(
      self,
      var_name: Literal['temperature', 'theta'],
      phi: np.ndarray,
      zz: np.ndarray,
      pressure_0: float,
  ) -> np.ndarray:
    """Numerically integrates energy variable to compute pressure.

    The integration includes all the halo points to ensure that pressure
    differences are well behaved near the boundaries.

    Args:
      var_name: The name of the energy variable to be integrated.
      phi: The temperature or potential temperature field, in units of K.
      zz: The vertical coordinates, in units of m.
      pressure_0: The pressure at z = 0, in units of Pa.

    Returns:
      The hydrostatic pressure profile, in units of Pa.
    """
    if var_name == 'temperature':
      pressure_fn = common_lib.compute_hydrostatic_pressure_from_temperature
    elif var_name == 'theta_li':
      pressure_fn = common_lib.compute_hydrostatic_pressure_from_theta
    else:
      raise ValueError(f'Unsupported energy variable: {var_name}')

    p = pressure_fn(zz, phi, pressure_0)
    # Shift the pressure profile so p[halo_width] == pressure_0.
    return p * self.config.p_thermal / p[self.config.halo_width]

  def compute_reference_state_from_temperature(
      self,
      temperature: np.ndarray,
      q_t: np.ndarray,
      zz: np.ndarray,
  ) -> FlowFieldMap:
    """Computes reference state consistent with initial temperature.

    Args:
      temperature: The initial temperature, in units of K.
      q_t: The initial total water specific humidity, in units of kg/kg.
      zz: The vertical coordinates, in units of m.

    Returns:
      The reference potential temperature and pressure that are consistent with
      the initial `theta_li` and `q_t`.
    """
    # 'p': The reference pressure consistent with initial conditions.
    # 'theta': Reference theta consistent with initial conditions.
    # 'q_l': The liquid-phase specific humidity.
    # 'q_i': The ice-phase specific humidity.
    zeros = tf.nest.map_structure(tf.zeros_like, zz)
    p_init = tf.convert_to_tensor(
        self.hydrostatic_pressure_from_energy_variable(
            'temperature', temperature, zz, self.config.p_thermal
        )
    )
    theta_init = (
        self.cloud_utils.thermodynamics.temperature_to_potential_temperature(
            'theta',
            tf.convert_to_tensor(temperature),
            tf.convert_to_tensor(q_t),
            zeros,
            zeros,
            tf.convert_to_tensor(zz),
            {'p_ref': p_init},
        )
    )
    states_0 = {
        'p': p_init,
        'theta': theta_init,
        'q_l': zeros,
        'q_i': zeros,
        'res': q_t,
    }

    def body(
        i: tf.Tensor,
        states: FlowFieldMap,
    ) -> Tuple[tf.Tensor, FlowFieldMap | dict[str, np.ndarray]]:
      """Solves the potential temperature iteratively."""
      q_c = tf.nest.map_structure(tf.math.add, states['q_l'], states['q_i'])

      r_m = self.cloud_utils.thermodynamics.r_mix(
          tf.convert_to_tensor(q_t), q_c
      )

      rho = tf.nest.map_structure(
          lambda p_i, r_m_i, t_i: p_i / r_m_i / t_i,
          states['p'],
          r_m,
          temperature,
      )

      q_l, q_i = self.cloud_utils.thermodynamics.equilibrium_phase_partition(
          tf.convert_to_tensor(temperature), rho, tf.convert_to_tensor(q_t)
      )
      helper_states = {'p_ref': states['p']}
      theta = (
          self.cloud_utils.thermodynamics.temperature_to_potential_temperature(
              'theta',
              tf.convert_to_tensor(temperature),
              tf.convert_to_tensor(q_t),
              q_l,
              q_i,
              tf.convert_to_tensor(zz),
              helper_states,
          )
      )

      theta = tf.nest.map_structure(
          lambda theta_new, theta_old: theta_old
          + _STEP_SIZE * (theta_new - theta_old),
          theta,
          states['theta'],
      )

      p = self.hydrostatic_pressure_from_energy_variable(
          'theta', theta, zz, self.config.p_thermal
      )

      res = tf.nest.map_structure(tf.math.subtract, theta, states['theta'])

      return i + 1, {'p': p, 'theta': theta, 'q_l': q_l, 'q_i': q_i, 'res': res}

    states = self._iterative_solve(body, states_0)
    theta_li = (
        self.cloud_utils.thermodynamics.temperature_to_potential_temperature(
            'theta_li',
            tf.convert_to_tensor(temperature),
            tf.convert_to_tensor(q_t),
            states['q_l'],
            states['q_i'],
            tf.convert_to_tensor(zz),
            {'p_ref': states['p']},
        )
    )
    return {
        'theta_li': theta_li,
        'theta_ref': states['theta'],
        'p_ref': states['p'],
    }

  def _perturbed_init_fn(self, varname: str) -> initializer.ValueFunction:
    """Returns an initial value function that perturbs the initial field."""
    seed = {
        'u': geophysical_flow_utils.U_SEED,
        'v': geophysical_flow_utils.V_SEED,
        'w': geophysical_flow_utils.W_SEED,
        'theta_li': THETA_LI_SEED,
    }[varname]
    rms = {
        'u': self.sim_params.u_rms,
        'v': self.sim_params.v_rms,
        'w': self.sim_params.w_rms,
        'theta_li': self.sim_params.theta_li_rms,
    }[varname]
    local_grid_size = (
        self.core_n if self.init_mode == 'PAD' else self.core_n_full
    )
    mean = 0.0
    mean_init_fn = None
    if varname in ('u', 'v', 'theta_li'):
      mean_init_fn = self._init_fn_from_sounding(varname)
    return geophysical_flow_utils.perturbed_constant_init_fn(
        seed + self.sim_params.random_seed,
        mean,
        self.g_dim,
        local_grid_size,
        rms,
        mean_init_fn=mean_init_fn,
    )

  def thermodynamics_states(
      self,
      zz: FlowFieldVal,
      xx: Optional[FlowFieldVal] = None,
      yy: Optional[FlowFieldVal] = None,
      lx: Optional[float] = None,
      ly: Optional[float] = None,
      coord: Optional[initializer.ThreeIntTuple] = None,
  ) -> FlowFieldMap:
    """Provides the initial thermodynamic state for the GCM column."""

    def sounding_init_fn(varname: str):
      return lambda z: self._init_from_sounding(
          varname, coord[self.g_dim]
      )

    theta_li = tf.nest.map_structure(
        sounding_init_fn('theta_li'), zz
    )
    q_t = tf.nest.map_structure(sounding_init_fn('q_t'), zz)

    thermal_states = {
        'q_t': q_t,
    }

    helper_states = None
    if 'p_ref' in self.sounding:
      helper_states = {
          'p_ref': tf.nest.map_structure(sounding_init_fn('p_ref'), zz)
      }

    thermal_states.update(
        self.cloud_utils.temperature(
            theta_li, q_t, zz, 'theta_li', helper_states
        )
    )
    xx, yy, zz = (
        geophysical_flow_utils.reorder_vertical_horizontal_coordinates_to_xyz(
            zz, xx, yy, self.g_dim
        )
    )
    thermal_states['theta_li'] = tf.nest.map_structure(
        lambda x, y, z: self._perturbed_init_fn('theta_li')(
            x, y, z, self.config.lx, self.config.ly, self.config.lz, coord),
        xx, yy, zz)

    return thermal_states

  def velocity_init_fn(self, varname: str) -> initializer.ValueFunction:
    """Generates the init velocity functions for `varname` based on GCM winds.

    The velocity specified by `varname` should be one of 'u', 'v', or 'w'. The
    horizontal components are initialized from the sounding, while the vertical
    component is always set to 0. Perturbation of all 3 components is supported.

    Args:
      varname: The name of the velocity component.

    Returns:
      The initial states for variable `varname`.
    """
    assert varname in ('u', 'v', 'w'), (
        f'{varname} is not a valid option. Available options are "u", "v", and'
        ' "w".'
    )
    return self._perturbed_init_fn(varname)

  def helper_states_fn(self) -> Dict[str, initializer.ValueFunction]:
    """Provides `init_fn` of helper variables for GCM states and tendencies."""

    gcm_mean_states = {
        # GCM initial thermodynamic profiles.
        gcm_settings.GCM_THETA_LI_KEY: self._init_fn_from_sounding('theta_li'),
        gcm_settings.GCM_Q_T_KEY: self._init_fn_from_sounding('q_t'),
        gcm_settings.GCM_TEMPERATURE_KEY: self._init_fn_from_sounding('T'),
        # GCM advective tendencies.
        gcm_settings.GCM_ADV_TENDENCY_TEMPERATURE_KEY: (
            self._init_fn_from_sounding('T_adv_src')
        ),
        gcm_settings.GCM_ADV_TENDENCY_HUMIDITY_KEY: self._init_fn_from_sounding(
            'q_t_adv_src'
        ),
        # GCM velocity field.
        gcm_settings.GCM_U_KEY: self._init_fn_from_sounding('u'),
        gcm_settings.GCM_V_KEY: self._init_fn_from_sounding('v'),
        gcm_settings.GCM_W_KEY: self._init_fn_from_sounding('w'),
    }

    if self.reference_state_type == 'user_defined_reference_state':
      gcm_mean_states.update({
          'theta_ref': self._init_fn_from_sounding('theta_ref'),
          'p_ref': self._init_fn_from_sounding('p_ref'),
      })

    if self._use_grid_extension:
      for k in self._gcm_forcing_lib.upper_atmosphere_state_names:
        extended_state_key = f'{GRID_EXTENSION_KEY_PREFIX}_{k}'
        gcm_mean_states[extended_state_key] = self._init_fn_from_sounding(
            extended_state_key
        )

    # Allocate variables for source terms required by the GCM forcing framework.
    src_vars = {
        k: init_fn_lib.constant_init_fn(0)
        for k in ('src_u', 'src_v', 'src_q_t', 'src_theta_li')
    }

    return gcm_mean_states | src_vars

  def scalar_source_fn(
      self,
  ) -> parameters_lib.SourceUpdateFnLib:
    """Constructs the forcing source functions for transported scalar equations.

    These source terms will relax the thermodynamics states to the GCM time-
    averaged thermodynamic states over a user-defined time scale.

    Returns:
      A dictionary containing functions for updating the source terms of the
      transported scalar equations.
    """

    def src_fn(
        varname: str,
        kernel_op: get_kernel_fn.ApplyKernelOp,
        replica_id: tf.Tensor,
        replicas: np.ndarray,
        step_id: tf.Tensor,
        states: types.FlowFieldMap,
        additional_states: types.FlowFieldMap,
        params: grid_parametrization.GridParametrization,
    ) -> types.FlowFieldMap:
      del kernel_op, replica_id, replicas, step_id, params
      if varname == 'theta_li':
        fn = self._gcm_forcing_lib.theta_li_source_fn
      elif varname == 'q_t':
        fn = self._gcm_forcing_lib.q_t_source_fn
      else:
        raise ValueError(f'No source function available for variable {varname}')
      key = f'src_{varname}'
      return {key: fn(states, additional_states)}

    return {
        k: functools.partial(src_fn, k) for k in ('theta_li', 'q_t')
    }

  def momentum_source_fn(
      self,
  ) -> parameters_lib.SourceUpdateFnLib:
    """Constructs the forcing source functions for the momentum equations.

    These source terms will relax the LES horizontal winds to the GCM winds
    uniformly across the whole domain over the user-defined time scale
    `tau_r_wind_sec`.

    Returns:
      A dictionary containing functions for updating the source terms of the
      horizontal momentum equations.
    """
    def uv_fn(
        field_name: str,
        kernel_op: get_kernel_fn.ApplyKernelOp,
        replica_id: tf.Tensor,
        replicas: np.ndarray,
        states: types.FlowFieldMap,
        additional_states: types.FlowFieldMap,
        params: grid_parametrization.GridParametrization,
    ) -> types.FlowFieldMap:
      del kernel_op, replica_id, replicas, params
      key = f'src_{field_name}'
      src = self._gcm_forcing_lib.geostrophic_wind_forcing_fn(
          states, additional_states
      )[key]
      return {key: src}
    return {k: functools.partial(uv_fn, k) for k in ('u', 'v')}
