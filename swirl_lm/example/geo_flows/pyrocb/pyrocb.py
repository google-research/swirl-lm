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

"""Setup for the PyroCb simulation."""

import dataclasses
from typing import Dict, Optional

from absl import logging
import numpy as np
from swirl_lm.base import initializer
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.example.geo_flows import geophysical_flow_common
from swirl_lm.example.shared import cloud_utils
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import init_fn as init_fn_lib
from swirl_lm.utility import types
import tensorflow as tf


FlowFieldMap = types.FlowFieldMap
FlowFieldVal = types.FlowFieldVal

# The name of the variable that traces the product of combustion.
TRACER_NAME = 'tracer'


@dataclasses.dataclass
class PyroCbSettings:
  """Defines parameters to set up the PyroCb simulation."""

  # Comma separated list of paths to the pyrocb soundings. The files
  # will be read in order and for each variable, only the values from the latest
  # file in which the variable is defined will be used.
  # The files don't need to all have the same number of rows or z values -
  # variable values are interpolated to the grid as the files are being loaded.
  sounding_csv_filename: str = ''
  # The option off whether to start the simulation from a no-wind condition or
  # the wind condition in the pyrocb sounding. If set to `False`, all
  # 3 velocity components are set to 0.
  velocity_from_sounding: bool = True
  # The free stream velocity magnitude in m/s. These numbers are used only when
  # `velocity_from_sounding` is `False`.
  u_inf: float = 20.0
  v_inf: float = -14.0
  # The perturbation of potential temperature in the warm bubble [K].
  theta_pert: float = 1.0
  # The height of the center of the warm bubble [m].
  z_bubble: float = 1.4e3
  # The horizontal radius of the warm bubble [m].
  rh_bubble: float = 1e4
  # The vertical radius of the warm bubble [m].
  rv_bubble: float = 1.4e3
  # The option of whether to add temperature perturbation to the initial
  # condition.
  apply_init_pert: bool = False
  # The height below which the temperature perturbation will be applied if
  # `apply_init_pert` is `True`.
  init_pert_height: float = 1e3
  # The amplitude of the perturbation to the temperature initial condition. The
  # perturbation will be randomly selected from a uniform distribution between
  # [-init_per_amp, +init_per_amp] for each grid cell.
  init_pert_amp: float = 0.1
  # The option of whether to include Coriolis force.
  apply_coriolis_force: bool = False
  # The latitude that is used to compute the Coriolis force in unites of
  # degrees. Default is set to 46.8797° N, which is the latitude of Montana.
  latitude: float = 46.8797


class PyroCb(geophysical_flow_common.GeophysicalFlowSetup):
  """Defines initial conditions and essential fields for PyroCb."""

  def __init__(
      self,
      config: parameters_lib.SwirlLMParameters,
      sim_params: PyroCbSettings,
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

    self._potential_temperature_warm_bubble = geophysical_flow_common.bubble(
        self.sim_params.theta_pert,
        self.sim_params.rh_bubble,
        self.sim_params.rv_bubble,
        self.sim_params.z_bubble,
    )

    full_grid = (
        self.config.global_xyz_with_halos
        if self.init_mode == 'PHYSICAL'
        else self.config.global_xyz
    )
    zz = np.asarray(full_grid[self.g_dim])
    sounding = geophysical_flow_common.load_sounding(
        sim_params.sounding_csv_filename.split(','), zz
    )
    sounding['q_v'] = sounding['q_v'] * 1e-3
    sounding['p_ref'] = (
        geophysical_flow_common.compute_hydrostatic_pressure_from_theta(
            zz, sounding['theta'], self.cloud_utils.config.p_thermal
        )
    )
    self.sounding = {
        k: tf.constant(v, dtype=types.TF_DTYPE) for k, v in sounding.items()
    }

    self.vel_inf = {
        'u': self.sim_params.u_inf,
        'v': self.sim_params.v_inf,
        'w': 0.0,
    }

    latitude = self.sim_params.latitude * np.pi / 180.0
    self.coriolis_force_fn = (
        cloud_utils.coriolis_force(latitude, self.vel_inf, self.g_dim, True)
        if self.sim_params.apply_coriolis_force
        else None
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
    """Provides initial thermodynamic state for PyroCb."""

    def sounding_init_fn(varname: str):
      return lambda z: self._init_from_sounding(varname, coord[self.g_dim])

    theta = tf.nest.map_structure(sounding_init_fn('theta'), zz)
    q_t = tf.nest.map_structure(sounding_init_fn('q_v'), zz)
    thermal_states = {'theta_ref': theta, 'q_t': q_t}

    if self.sim_params.apply_init_pert:
      thermal_states['theta'] = tf.nest.map_structure(
          lambda t, z: tf.where(  # pylint:disable=g-long-lambda
              tf.less(z, self.sim_params.init_pert_height),
              t
              + tf.random.uniform(
                  tf.shape(t),
                  minval=-self.sim_params.init_pert_amp,
                  maxval=self.sim_params.init_pert_amp,
                  dtype=t.dtype,
              ),
              t,
          ),
          theta,
          zz,
      )
    else:
      thermal_states['theta'] = tf.nest.map_structure(tf.identity, theta)

    helper_states = (
        {'p_ref': tf.nest.map_structure(sounding_init_fn('p_ref'), zz)}
        if self.reference_state_type == 'user_defined_reference_state'
        else {}
    )
    thermal_states.update(
        self.cloud_utils.temperature(
            thermal_states['theta'], q_t, zz, 'theta', helper_states
        )
    )

    return thermal_states

  def velocity_init_fn(self, varname: str) -> initializer.ValueFunction:
    """Generates the velocity init functions for `varname`.

    Args:
      varname: The name of the velocity component for which the initial state is
        generated. Should be one of 'u', 'v', or 'w'.

    Returns:
      The initial states for variable `varname`.
    """
    if varname == 'w':
      return init_fn_lib.constant_init_fn(0)
    elif varname in ('u', 'v'):
      return (
          self._init_fn_from_sounding(varname)
          if self.sim_params.velocity_from_sounding
          else init_fn_lib.constant_init_fn(self.vel_inf[varname])
      )
    else:
      raise ValueError(
          f'{varname} is not a valid option. Available options are "u", "v", '
          'and "w".'
      )

  def post_simulation_update_fn(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Updates flow field variables at a specific step in the simulation.

    This function is invoked only once after the step specified in the
    commandline flag.

    Args:
      kernel_op: An object holding a library of kernel operations.
      replica_id: The id of the replica.
      replicas: The replicas. In particular, a numpy array that maps grid
        coordinates to replica id numbers.
      states: A keyed dictionary of states that will be updated.
      additional_states: A list of states that are needed by the update fn, but
        will not be updated by the main governing equations.
      params: An instance of `grid_parametrization.GridParametrization`.

    Returns:
      A dictionary that is a union of updated `states` and `additional_states`.
      Here the temperature perturbation is introduced in the energy variable.
    """
    del kernel_op, replica_id, replicas, params
    output = dict(states)
    output.update(additional_states)

    # Get the name of the energy prognostic variable.
    energy_variable = list(states.keys() & {'theta', 'theta_li'})
    assert (
        len(energy_variable) == 1
    ), f'Expected exactly 1 energy variable, found {energy_variable}'
    energy_variable = energy_variable[0]

    if 'theta_pert' in additional_states:
      logging.info('Imposing a thermal bubble in the flow field.')

      if energy_variable in ('theta', 'theta_li'):
        output[energy_variable] = tf.nest.map_structure(
            tf.math.add,
            states[energy_variable],
            additional_states['theta_pert'],
        )
      else:
        raise ValueError(
            'Energy variable not found for the PyroCb simulation. '
            'Available options are `theta_li`, `theta`.'
        )

    return output

  def initial_states(
      self,
      replica_id: tf.Tensor,
      coordinates: initializer.ThreeIntTuple,
  ) -> FlowFieldMap:
    """Initializes additional variables in the PyroCb simulation."""

    def bubble_fn(xx, yy, zz, lx, ly, lz, coord):
      """Initializes the potential temperature of the warm bubble."""
      del lz, coord

      return self._potential_temperature_warm_bubble(xx, yy, zz, lx, ly)

    output = {}
    if 'theta_pert' in self.config.additional_state_keys:
      output['theta_pert'] = self.initialize_states(bubble_fn, coordinates)

    if TRACER_NAME in self.config.transport_scalars_names:
      tracer_bc_name = f'bc_{TRACER_NAME}_{self.g_dim}_0'
      assert tracer_bc_name in self.config.additional_state_keys, (
          f'The boundary condition variable ({tracer_bc_name}) is not'
          ' configured.'
      )
      output[TRACER_NAME] = self.initialize_states(
          init_fn_lib.constant_init_fn(0.0), coordinates
      )
      if self.g_dim == 0:
        output[tracer_bc_name] = output[TRACER_NAME][
            :, : self.config.halo_width + 1, :
        ]
      elif self.g_dim == 1:
        output[tracer_bc_name] = output[TRACER_NAME][
            ..., : self.config.halo_width + 1
        ]
      elif self.g_dim == 2:
        output[tracer_bc_name] = output[TRACER_NAME][
            : self.config.halo_width + 1, ...
        ]
      else:
        raise ValueError(
            '`g_dim` has to be one of 0, 1, and 2. Invalid value found'
            f' {self.g_dim}'
        )

    return output

  def helper_states_fn(self) -> Dict[str, initializer.ValueFunction]:
    """Provides `init_fn` of helper variables for thermodynamic states."""

    if self.reference_state_type != 'user_defined_reference_state':
      return {}

    def thermo_states_init_fn(varname):
      """Provides a function that initializes the flow field of `varname`."""

      def init_fn(xx, yy, zz, lx, ly, lz, coord):
        """Initializes the thermodynamic state."""
        del lz
        return self.thermodynamics_states(zz, xx, yy, lx, ly, coord)[varname]

      return init_fn

    return {
        'theta_ref': thermo_states_init_fn('theta_ref'),
        'q_t_init': thermo_states_init_fn('q_t'),
        'p_ref': self._init_fn_from_sounding('p_ref'),
    }

  def additional_states_update(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      step_id: tf.Tensor,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Updates additional states for the simulation."""
    del replica_id, replicas, step_id, states

    output = {}

    tracer_bc_name = f'bc_{TRACER_NAME}_{self.g_dim}_0'
    if tracer_bc_name in additional_states:
      t_s = common_ops.get_face(
          additional_states['T_s'], self.g_dim, 0, self.config.halo_width
      )[0]
      # Here we consider the use of 3D tf.Tensor only.
      tracer_bc = tf.where(
          tf.greater(t_s, 400.0), tf.ones_like(t_s), tf.zeros_like(t_s)
      )
      multiples = [1, 1, 1]
      g_axis = (self.g_dim + 1) % 3
      multiples[g_axis] = self.config.halo_width + 1
      output[tracer_bc_name] = tf.tile(tracer_bc, multiples)

    return output
