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

# coding=utf-8
"""A library for the geophysical flow simulation.


Simulations considered in this library include:
1. Pyrocumulonimbus cloud (pyrocumulus cloud)
2. GCM-driven LES (single column)
3. DYCOMS-II, RF01
"""

import collections
import dataclasses
import enum
import functools
import re
from typing import Callable, Dict, Optional, Tuple

from absl import flags
from absl import logging
import fancyflags as ff
import ml_collections
import numpy as np
from swirl_lm.base import initializer
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import physical_variable_keys_manager
from swirl_lm.boundary_condition import monin_obukhov_similarity_theory
from swirl_lm.boundary_condition import rayleigh_damping_layer
from swirl_lm.boundary_condition import simulated_turbulent_inflow
from swirl_lm.communication import halo_exchange
from swirl_lm.equations import common
from swirl_lm.example.geo_flows import geophysical_flow_common
from swirl_lm.example.geo_flows.cloud_feedback import gcm_column
from swirl_lm.example.geo_flows.cloud_feedback import gcm_settings
from swirl_lm.example.geo_flows.dycoms import dycoms2_rf01
from swirl_lm.example.geo_flows.pyrocb import pyrocb
from swirl_lm.example.shared import cloud_utils
from swirl_lm.example.shared import wildfire_utils
from swirl_lm.physics import constants
from swirl_lm.physics.combustion import combustion
from swirl_lm.physics.radiation import rrtmgp_common
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.physics.thermodynamics import water
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import init_fn as init_fn_lib
from swirl_lm.utility import monitor
from swirl_lm.utility import types
import tensorflow as tf


FlowFieldMap = types.FlowFieldMap

AMBIENT_STATE_TEMPLATE = '{varname}_ambient'


class SimulationType(enum.Enum):
  """Defines the type of the simulation."""
  # Pyrocumulonimbus cloud simulation.
  PYROCB = 'pyrocb'

  # GCM driven simulation based on the equilibrium state of a single GCM column.
  GCM_COLUMN = 'gcm_column'

  # The stratocumulus cloud simulation following the setup in Stevens et. al.
  # 2005.
  DYCOMS = 'dycoms'


@dataclasses.dataclass
class ConstantHeatSource:
  """Defines a constant heat source applied to the energy variable."""
  # A toggle of whether a constant heat source is included in the simulation.
  include_const_heat_src: bool = False

  # A toggle of whether a constant heat flux is applied in the simulation.
  include_const_heat_flux: bool = False

  # Parameters that determine heat flux from combustion. The heat flux in W/m^2
  # is equal to the product of heat_of_combustion, burning_rate,
  # canopy_bulk_density and fuel_height.

  # https://en.wikipedia.org/wiki/Heat_of_combustion
  heat_of_combustion: float = 20e6  # J/kg

  burning_rate: float = 0.01  # 1/s

  # https://www.nwcg.gov/publications/pms437/fuels/canopy-fuel-characteristics#TOC-Canopy-Bulk-Density-kg-m3-or-lb-ft3
  canopy_bulk_density: float = 0.25  # kg/m^3

  # fuel_height of 0.97m is chosen because it results in heat flux of
  # 48.5 kW/m^2 given the parameters above.
  fuel_height: float = 0.97  # m

  # The heat source takes the shape of a cylinder. The unit of these dimensional
  # parameters is m. The height is rounded *up* to the nearest grid point.
  x_center: float = 6e4
  y_center: float = 6e4
  r: float = 1.2e3
  height: float = 250.0

  # u-component of the velocity of the heat source in m/s.
  u: float = 0.0

  # Define the duration of the heat source to be positioned in the flow field.
  # The procedure for the source term enforcement is shown as follows:
  # t_0 < t <= t_1: src(t) = (t - t_0) / (t_1 - t_0) * src_max;
  # t_1 < t <= t_2: src(t) = src_max;
  # t_2 < t <= t_3: src(t) = (t_3 - t) / (t_3 - t_2) * src_max;
  # t > t_3: src(t) = 0.
  t_0: float = 0.0
  t_1: float = 300.0
  t_2: float = 2100.0
  t_3: float = 2400.0

  @property
  def heat_flux_due_to_combustion(self) -> float:
    """Returns heat flux magnitude in W/m^2."""
    return (self.heat_of_combustion * self.burning_rate *
            self.canopy_bulk_density * self.fuel_height)

  def heat_src_due_to_combustion(self, src_height: float) -> float:
    """Returns heat flux from combustion as a source spread over src_height.

    Args:
      src_height: The height along which the heat flux from combustion is
        uniformly spread. This is not necessarily the same as self.height
        because currently we are spreading heat to a small number of grid cells
        and self.height is not necessarily an integer multiple of dx. Having
        src_height allows adjusting self.height as needed to correctly account
        for rounding errors so the total amount of heat added to the simulation
        is accurate.

    Returns:
      Heat source magnitude in W/m^3.
    """
    return self.heat_flux_due_to_combustion / src_height

  def at_time_t(self, t: tf.Tensor) -> 'ConstantHeatSource':
    # Note that we break the type of `x_center` here by setting it to a
    # tf.Tensor value. The type has to be a simple type (like float) because we
    # pass `x_center` to fancyflags which allows only a limited subset of
    # types. This is ok for now because pytype doesn't catch this as an error.
    return dataclasses.replace(self, x_center=self.x_center + t * self.u)


_SIMULATION_TYPE = flags.DEFINE_enum_class(
    'simulation_type', SimulationType.DYCOMS, SimulationType,
    f'Defines the type of the simulation. '
    f'Supported types: '
    f'{[sim_type.name for sim_type in SimulationType]}.',
    allow_override=True)

_ENFORCE_MOST_CONSISTENT_BC = flags.DEFINE_boolean(
    'enforce_most_consistent_bc', False,
    'Use the Monin-Obukhov Similarity theory to enforce '
    'dynamic Neumann boundary conditions that are consistent '
    'with the surface fluxes. This requires that `most` be '
    'set in the config and that the active scalars have a '
    'Neumann boundary condition type at the surface.',
    allow_override=True)

_INFLOW_INIT_OPTION = flags.DEFINE_enum(
    'inflow_init_option',
    'no_touch',
    ['no_touch', 'init_based', 'inflow_based', 'sounding_based'],
    'Options for preprocessing the inflow and initial conditions of velocity.'
    ' "no_touch" will keep both the inflow and initial conditions of velocity'
    ' without modifications; "init_based" will set the mean of the inflow'
    ' to be the same as the initial condition; "init_with_inflow" will set the'
    ' mean of the initial condition to be the same as the inflow; and'
    ' "with_sounding" will set both inflow and the initial condition to be the'
    ' sounding profile (requires the presence of the sounding).',
)

_INFLOW_INIT_RESCALE_FACTOR = flags.DEFINE_string(
    'inflow_init_rescale_factor',
    '',
    'Factors that rescale the initial and inflow variables. The factors should'
    ' be specified as a dictionary of pairs of tuples, with the first element'
    ' in the tuple being the height and the second element being scaling factor'
    ' below the height specified by the first element. The variable name and'
    ' the values are separated by ":", and different variable items are'
    ' separated by "|". The tuples are separated by ";". For example:'
    ' "u:1e3,1;8e3,0.25;1e4,0.1|q_t:5e2,0.25;8e3,0.5" provides piecewise'
    ' linear profiles for "u" and "q_t". The scaling factor for "u" is 1'
    ' between 0 and 1 km; linearly decaying from 1 to 0.25 from 1 to 8 km;'
    ' lienarly decaying from 0.25 to 0.1 between 8 to 10 km; stays constant at'
    ' 0.1 above 1 km. The scaling factor for "q_t" is 0.25 below 500 m,'
    ' linearly increases from 0.25 to 0.5 between 500 m and 8 km, and kept as a'
    ' constant above 8 km. Note that it is active when `inflow_init_option =='
    ' "init_based"` only.',
)

_PYROCB_FLAG = ff.DEFINE_auto('pyrocb', pyrocb.PyroCbSettings)
_GCM_COLUMN_FLAG = ff.DEFINE_auto('gcm_column', gcm_settings.GCMSettings)
_DYCOMS_FLAG = ff.DEFINE_auto('dycoms', dycoms2_rf01.DycomsSettings)


_CONST_HEAT_SRC_FLAG = ff.DEFINE_auto('const_heat_src', ConstantHeatSource)


def _parse_inflow_rescale_factor(info: str):
  """Processes the string that specifies the inflow scaling factors."""
  output = []
  for scale_factor_pair in info.split(';'):
    buf = re.fullmatch(r'^(.+),(.+)$', scale_factor_pair)
    if buf is None:
      continue
    output.append((float(buf.group(1)), float(buf.group(2))))
  return output


def _parse_inflow_rescale_info(info: str):
  """Processes the string of inflow scaling factors for multiple variables."""
  if not info:
    return {}

  output = {}
  for scale_factor_item in info.split('|'):
    varname, scale_factor = scale_factor_item.split(':')
    output[varname] = _parse_inflow_rescale_factor(scale_factor)

  return output


def _get_inflow_init_rescale_factor_fn(
    rescale_factors: list[tuple[float, float]],
) -> Callable[[tf.Tensor], tf.Tensor]:
  """Generates a function that computes height-dependent rescale factors."""

  rescale_fn = (
      lambda z, z_0, z_1, f_0, f_1: (z - z_0) / (z_1 - z_0) * (f_1 - f_0) + f_0
  )

  def inflow_rescale_factor_fn(zz: tf.Tensor):
    """Generates inflow rescale factors as a function of `zz`."""
    if not rescale_factors:
      # If no rescale factor is provided, no rescaling will be performed to the
      # inflow data.
      return tf.ones_like(zz)

    factor = rescale_factors[0][1] * tf.ones_like(zz)
    for i in range(len(rescale_factors) - 1):
      z_0, f_0 = rescale_factors[i]
      z_1, f_1 = rescale_factors[i + 1]
      factor = tf.where(
          tf.math.logical_and(tf.greater_equal(zz, z_0), tf.less(zz, z_1)),
          rescale_fn(zz, z_0, z_1, f_0, f_1),
          factor,
      )
    return tf.where(
        tf.greater_equal(zz, rescale_factors[-1][0]),
        rescale_factors[-1][1] * tf.ones_like(zz),
        factor,
    )

  return inflow_rescale_factor_fn


def _mesh_grid(x: tf.Tensor, y: tf.Tensor, z: tf.Tensor, g_dim: int) -> Tuple[
    types.FlowFieldVal, types.FlowFieldVal, types.FlowFieldVal]:
  """Returns the mesh-grid of x, y, and z respecting `g_dim`.

  Args:
    x: The 1D x coordinates.
    y: The 1D y coordinates.
    z: The 1D z coordinates.
    g_dim: Gravity direction.

  Returns:
    Tuple of three tensors where the first and second are 3D tensors
    representing the horizontal coordinates and the third is a 3D tensor
    representing the vertical coordinate as determined by `g_dim`.
  """
  height = (x, y, z)[g_dim]
  horizontal_coordinates = [x, y, z]
  del horizontal_coordinates[g_dim]
  height_len = tf.shape(height)[0]
  assert g_dim == 2, ('_mesh_grid() is only supported for g_dim = 2 '
                      f'but got g_dim = {g_dim}.')
  return (
      tf.unstack(tf.tile(horizontal_coordinates[0][tf.newaxis, :, tf.newaxis],
                         (height_len, 1, 1))),
      tf.unstack(tf.tile(horizontal_coordinates[1][tf.newaxis, tf.newaxis, :],
                         (height_len, 1, 1))),
      tf.unstack(height))


class GeophysicalFlow():
  """A library for the geophysical flow simulation setup."""

  def __init__(self,
               config: parameters_lib.SwirlLMParameters,
               experiment_config: Optional[ml_collections.ConfigDict] = None):
    """Initializes the simulation setup."""
    # pylint: disable=g-long-ternary
    self.config = config if config is not None else (
        parameters_lib.params_from_config_file_flag())
    # pylint: enable=g-long-ternary

    computation_shape = np.array(
        [self.config.cx, self.config.cy, self.config.cz])
    self.replicas = np.arange(
        np.prod(computation_shape), dtype=np.int32).reshape(computation_shape)

    self.cloud_utils = cloud_utils.cloud_utils_factory(config=self.config)

    self.fire_utils = (
        wildfire_utils.WildfireUtils(self.config)  # pylint: disable=g-long-ternary
        if self.config.combustion is not None
        else None
    )

    self.sim_type = _SIMULATION_TYPE.value
    if experiment_config is not None:
      # Initialize parameters from the experiment config.
      if self.sim_type == SimulationType.DYCOMS:
        self.sim_setup = dycoms2_rf01.Dycoms2RF01(
            self.config, experiment_config.dycoms, self.cloud_utils)
      else:
        raise NotImplementedError(
            f'Unsupported simulation type {self.sim_type.name} for Vizier'
            ' experiment.'
        )
    else:
      # Initialize parameters from flags.
      if self.sim_type == SimulationType.DYCOMS:
        self.sim_setup = dycoms2_rf01.Dycoms2RF01(
            self.config, _DYCOMS_FLAG.value(), self.cloud_utils)
      elif self.sim_type == SimulationType.PYROCB:
        self.sim_setup = pyrocb.PyroCb(
            self.config, _PYROCB_FLAG.value(), self.cloud_utils)
      elif self.sim_type == SimulationType.GCM_COLUMN:
        self.sim_setup = gcm_column.GCMColumn(
            self.config, _GCM_COLUMN_FLAG.value(), self.cloud_utils)
      else:
        raise NotImplementedError(
            f'Unsupported simulation type {self.sim_type.name}.'
        )

    self.monitor = monitor.Monitor(self.config)

    self._g_dim = self.sim_setup.g_dim

    self.radiation_src_update_fn = self.sim_setup.radiation_src_update_fn
    self.radiation_state_keys = rrtmgp_common.additional_keys(
        self.config.radiative_transfer,
        self.config.additional_state_keys,
    )

    if self.config.boundary_models is not None:
      self.sponge = (
          rayleigh_damping_layer.RayleighDampingLayer(
              self.config.boundary_models.sponge, self.config.periodic_dims)
          if self.config.boundary_models.sponge else None)
      self.abl = (
          monin_obukhov_similarity_theory
          .monin_obukhov_similarity_theory_factory(self.config)
          if self.config.boundary_models.HasField('most') else None)
    else:
      self.sponge = None
      self.abl = None

    self._domain_lens = [self.config.lx, self.config.ly, self.config.lz]
    self._local_grid = [self.config.nx, self.config.ny, self.config.nz]

    # Define the initial wind and the coriolis force function.
    self.init_wind = self.sim_setup.init_wind
    self.coriolis_force_fn = self.sim_setup.coriolis_force_fn

    self.bc_manager = (
        physical_variable_keys_manager.BoundaryConditionKeysHelper())

    self.thermodynamics = thermodynamics_manager.thermodynamics_factory(
        self.config)

    # Determine the energy variable used in the simulation.
    allowed_energy_variables = ('theta_li', 'theta', 'e_t')
    self.energy_variable = None
    for var in allowed_energy_variables:
      if var not in self.config.transport_scalars_names:
        continue

      if self.energy_variable is None:
        self.energy_variable = var
      else:
        raise ValueError('More than one energy variable is defined.')

    # Get parameters for a constant heat source if requested.
    self.const_heat_src = _CONST_HEAT_SRC_FLAG.value()

    self.ramp_up_down_fn = combustion.ramp_up_down_function(
        self.const_heat_src.t_0,
        self.const_heat_src.t_1,
        self.const_heat_src.t_2,
        self.const_heat_src.t_3,
    )

    # Defines the ignition kernel if requested.
    if self.const_heat_src.include_const_heat_src:
      if self.config.use_stretched_grid[self._g_dim]:
        # Rounding of fuel height to full grid is ignore for stretched mesh
        # because grid spacing is ambiguous is this case.
        src_height = self.const_heat_src.height
      else:
        dz = [self.config.dx, self.config.dy, self.config.dz][self._g_dim]
        src_height = tf.math.ceil(self.const_heat_src.height / dz) * dz
      heat_src_magnitude = self.const_heat_src.heat_src_due_to_combustion(
          src_height
      )
      if self.energy_variable in ('theta', 'theta_li'):
        # Assuming air density near the ground is 1 kg/m^3.
        heat_src_magnitude /= self.cloud_utils.thermodynamics.cp_d
      else:
        raise ValueError(
            'Expected `theta` or `theta_li` as energy variable, but '
            f'got `{self.energy_variable}`.'
        )
      self.ignition_heat_src_fn = combustion.ignition_with_heat_source(
          heat_src_magnitude, self.ramp_up_down_fn
      )

    # Add turbulent inflow if available.
    self.inflow = (
        simulated_turbulent_inflow.simulated_turbulent_inflow_factory(
            self.config))
    self.inflow_init_rescale_factors = _parse_inflow_rescale_info(
        _INFLOW_INIT_RESCALE_FACTOR.value
    )

  def _thermal_states_init_fn(
      self,
      varname: str,
      helper_states_fn: Dict[str, initializer.ValueFunction],
  ):
    """Generates the init functions for `varname`.

    The thermodynamic state specified by `varname` should be one of the
    following:
      'theta': The potential temperature of the moist mixture;
      'theta_v': The virtual potential temperature;
      'theta_li':  The liquid-ice potential temperature;
      'T':  The temperature;
      'rho': The density;
      'e': The internal energy;
      'q_t': The total humidity mass fraction;
      'q_c': The condensed liquid mass fraction.

    Args:
      varname: The name of the thermodynamic states for which the initial state
        is generated.
      helper_states_fn: A dictionary of `init_fn` for helper variables required
        to determine thermodynamic states.

    Returns:
      The initial states for variable `varname`.

    Raises:
      ValueError: If varname is not in the list of valid options.
    """

    def init_fn(xx: tf.Tensor, yy: tf.Tensor, zz: tf.Tensor, lx: float,
                ly: float, lz: float,
                coord: initializer.ThreeIntTuple) -> tf.Tensor:
      """Initializes the specified thermodynamic state."""
      valid_varnames = [
          'T', 'rho', 'e', 'q_t', 'q_c', 'theta', 'theta_v', 'theta_li', 'q_r',
          'q_s', 'q_v',
      ]
      if varname not in valid_varnames:
        raise ValueError(
            '{} is not a valid option. Available options are: {})'.format(
                varname, valid_varnames))

      height = (xx, yy, zz)[self._g_dim]

      horizontal_domain_size = [lx, ly, lz]
      horizontal_coordinates = [xx, yy, zz]
      del horizontal_coordinates[self._g_dim], horizontal_domain_size[
          self._g_dim]

      sim_states = self.sim_setup.thermodynamics_states(
          height,
          horizontal_coordinates[0],
          horizontal_coordinates[1],
          horizontal_domain_size[0],
          horizontal_domain_size[1],
          coord,
      )

      if varname in sim_states:
        return sim_states[varname]

      q_t = sim_states['q_t']

      helper_states = {
          varname: fn(xx, yy, zz, lx, ly, lz, coord)
          for varname, fn in helper_states_fn.items()
      }

      p = self.cloud_utils.thermodynamics.p_ref(height, helper_states)
      t = sim_states['temperature']
      if varname == 'T':
        return t

      r_m = sim_states['r_m']
      rho = p / r_m / t
      if varname == 'rho':
        return rho

      # Everything is initialized to equilibrium in the beginning.
      q_l, q_i = self.cloud_utils.thermodynamics.equilibrium_phase_partition(
          t, rho, q_t)
      if varname == 'q_c':
        return q_l + q_i

      if varname == 'q_v':
        return q_t - q_l - q_i

      e = self.cloud_utils.thermodynamics.internal_energy(t, q_t, q_l, q_i)
      if varname == 'e':
        return tf.stack(e)

      if varname in ('theta', 'theta_v', 'theta_li'):
        thetas = self.cloud_utils.thermodynamics.potential_temperatures(
            t, q_t, rho, height, helper_states
        )
        return thetas[varname]

      # Initializes the precipitation. We assume no rain/snow initially if no
      # precipitation condition is provided by the simulation-specific setup.
      if varname in ('q_r', 'q_s'):
        return tf.zeros_like(height)

      # The given varname was one of the valid_varnames but we couldn't
      # initialize it presumably because self.sim_setup.thermodynamics_states
      # didn't return the initialization function for this variable.
      raise NotImplementedError(
          f'Attempted to initialize {varname} but initialization code was not '
          'found.')

    return init_fn

  def _preprocess_init_and_inflow(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> tuple[FlowFieldMap, FlowFieldMap]:
    """Preprocesses initial and inflow velocity."""
    updated_states = {}
    updated_additional_states = {}

    if _INFLOW_INIT_OPTION.value == 'no_touch':
      return updated_states, updated_additional_states

    if not self.config.use_3d_tf_tensor:
      raise NotImplementedError(
          'Only 3D tf.Tensor is supported for the inflow/initial condition'
          ' adjustment.'
      )
    grid_size = [self.config.nz, self.config.nx, self.config.ny]
    grid_size[(self._g_dim + 1) % 3] = 1

    def update_mean(u, mean_old, mean_new, halos):
      """Updates the horizontal mean of `u`."""
      u_interior = common_ops.strip_halos(u, halos) - mean_old
      paddings = [(h, h) for h in halos]
      u_zero_mean = common_ops.pad(u_interior, paddings, value=0.0)
      return u_zero_mean + mean_new

    def get_local_1d_profile_along_gdim(u):
      """Retrieves part of a 1D variable `u` that is local to the replica."""
      coord = common_ops.get_core_coordinate(replicas, replica_id)[self._g_dim]
      n = [self.config.nx, self.config.ny, self.config.nz][self._g_dim]
      core_n = [
          self.config.core_nx,
          self.config.core_ny,
          self.config.core_nz,
      ][self._g_dim]
      s = core_n * coord
      u_idx = tf.range(u.shape[0])
      return tf.squeeze(
          tf.scatter_nd(
              tf.range(n)[:, tf.newaxis],
              tf.gather(
                  u,
                  tf.where(
                      tf.logical_and(
                          tf.greater_equal(u_idx, s),
                          tf.less(u_idx, s + n),
                      )
                  ),
              ),
              (n, 1),
          )
      )

    dims = (0, 1, 2)
    replica_dims = (0, 1, 2)
    partition_dims = [0, 1, 2]
    del partition_dims[self._g_dim]

    # The mean velocity profile for inflow is computed along the time (2
    # because the time dimension is always the 0th dimension of a 3D tensor,
    # which is denoted as "dimension 2" in Swirl-LM) and the lateral (the
    # dimension that is neither time nor vertical, e.g. when inflow is along x
    # and vertical is along z, the lateral dimension is y that corresponds to
    # dimension 1 in Swirl-LM) dimensions, which are partitioned along the
    # stream and lateral dimensions, respectively.
    # Means for the inflow are taken with halos excluded along the lateral
    # direction only, because the inflow dimension corresponds to time and
    # no halos is present. Value in halos of the inflow along the lateral
    # direction can be 0 if the inflow profile is generated from interpolation,
    # which will pollute the mean operation along the time and lateral
    # dimensions.
    if isinstance(
        self.inflow, simulated_turbulent_inflow.SimulatedTurbulentInflow
    ):
      lateral_dim = [0, 1, 2]
      lateral_dim.remove(self.inflow.inflow_dim)
      lateral_dim.remove(self._g_dim)
      lateral_dim = lateral_dim[0]
      if lateral_dim == 1:
        # t(2)-z(0)-y(1) or t(2)-x(0)-y(1).
        mean_dims_inflow = [1, 2]
        halos_inflow = [0, 2, 0]
      elif lateral_dim == 2:
        # t(2)-z(0)-x(1) or t(2)-z(0)-y(1).
        mean_dims_inflow = [0, 2]
        halos_inflow = [2, 0, 0]
      else:  # lateral_dim == 0:
        if self._g_dim == 1:
          # t(2)-x(0)-y(1).
          mean_dims_inflow = [0, 2]
          halos_inflow = [2, 0, 0]
        elif self._g_dim == 2:
          # t(2)-z(0)-x(1).
          mean_dims_inflow = [1, 2]
          halos_inflow = [0, 2, 0]
        else:
          raise ValueError('Both inflow and gravity are in the x direction!')

      inflow_mean_fn = functools.partial(
          common_ops.global_mean,
          replicas=replicas,
          halos=halos_inflow,
          axis=mean_dims_inflow,
          partition_axis=partition_dims,
      )
      inflow_bcast_fn = functools.partial(
          geophysical_flow_common.broadcast_vertical_profile_for_inflow,
          g_dim=self._g_dim,
          inflow_dim=self.inflow.inflow_dim,
      )
      inflow_update_fn = functools.partial(update_mean, halos=halos_inflow)
    elif self.inflow is None:
      inflow_mean_fn = None
      inflow_bcast_fn = None
      inflow_update_fn = None
    else:
      raise NotImplementedError(
          'Only `SimulatedTurbulentInflow` is supported, but'
          f' {type(self.inflow)} is provided.'
      )

    # Means for the initial condition are taken with halos except for the
    # vertical dimension.
    halos_flow_field = [2] * 3
    halos_flow_field[self._g_dim] = 0

    for vel in list(common.KEYS_VELOCITY) + ['q_t', 'theta_li']:
      inflow_name = f'INFLOW_{vel.upper()}'
      ambient_name = AMBIENT_STATE_TEMPLATE.format(varname=vel)

      if _INFLOW_INIT_OPTION.value == 'init_with_inflow':
        if inflow_name not in additional_states:
          continue
        inflow_mean = inflow_mean_fn(additional_states[inflow_name])
        u_mean_old = (
            geophysical_flow_common.broadcast_vertical_profile_for_flow_field(
                inflow_mean, self._g_dim
            )
        )
        u = tf.tile(u_mean_old, grid_size)
        # Homogeneus Neumann BC is applied to the velocity as initial
        # condition, which conforms with the methods specified in the
        # initialization function.
        bc_u = [
            [(halo_exchange.BCType.NEUMANN, 0.0)] * 2,
        ] * 3
        updated_states[vel] = halo_exchange.inplace_halo_exchange(
            tensor=u,
            dims=dims,
            replica_id=replica_id,
            replicas=replicas,
            replica_dims=replica_dims,
            periodic_dims=self.config.periodic_dims,
            boundary_conditions=bc_u,
            width=self.config.halo_width,
        )
      elif _INFLOW_INIT_OPTION.value == 'init_based':
        if vel not in self.inflow_init_rescale_factors:
          scaling_factor = 1.0
        else:
          scaling_factor_fn = _get_inflow_init_rescale_factor_fn(
              self.inflow_init_rescale_factors[vel]
          )
          scaling_factor = scaling_factor_fn(
              get_local_1d_profile_along_gdim(
                  self.config.global_xyz_with_halos[self._g_dim]
              )
          )
        u_mean = tf.squeeze(
            common_ops.global_mean(
                states[vel],
                replicas,
                halos_flow_field,
                partition_dims,
                partition_dims,
            )
        )
        u_mean_old = (
            geophysical_flow_common.broadcast_vertical_profile_for_flow_field(
                u_mean, self._g_dim
            )
        )
        u_mean_new = (
            geophysical_flow_common.broadcast_vertical_profile_for_flow_field(
                scaling_factor * u_mean, self._g_dim
            )
        )
        updated_states[vel] = update_mean(
            states[vel],
            u_mean_old,
            u_mean_new,
            halos_flow_field,
        )
        if vel == 'q_t':
          updated_states[vel] = tf.nest.map_structure(
              lambda x: tf.maximum(x, 0.0), updated_states[vel]
          )
        if inflow_name in additional_states:
          inflow_mean = inflow_mean_fn(additional_states[inflow_name])
          updated_additional_states[inflow_name] = inflow_update_fn(
              additional_states[inflow_name],
              inflow_mean,
              inflow_bcast_fn(scaling_factor * u_mean),
          )
          if inflow_name == 'INFLOW_Q_T':
            updated_additional_states[inflow_name] = tf.nest.map_structure(
                lambda x: tf.maximum(x, 0.0),
                updated_additional_states[inflow_name],
            )
      elif _INFLOW_INIT_OPTION.value == 'with_sounding':
        if vel not in ('u', 'v', 'q_t', 'theta_li'):
          continue

        if vel == 'q_t':
          sounding_vel = 'q_v'
        elif vel == 'theta_li':
          sounding_vel = 'theta'
        else:
          sounding_vel = vel

        sounding = get_local_1d_profile_along_gdim(
            self.sim_setup.sounding[sounding_vel]
        )

        u_mean = common_ops.global_mean(
            states[vel],
            replicas,
            halos_flow_field,
            partition_dims,
            partition_dims,
        )
        u_mean_old = (
            geophysical_flow_common.broadcast_vertical_profile_for_flow_field(
                sounding, self._g_dim
            )
        )

        updated_states[vel] = update_mean(
            states[vel],
            u_mean,
            u_mean_old,
            halos_flow_field,
        )
        if inflow_name in additional_states:
          inflow_mean = inflow_mean_fn(additional_states[inflow_name])
          updated_additional_states[inflow_name] = inflow_update_fn(
              additional_states[inflow_name],
              inflow_mean,
              inflow_bcast_fn(sounding),
          )
      else:
        raise NotImplementedError(
            f'{_INFLOW_INIT_OPTION.value} is not a valid option for inflow'
            ' and init preprocessing. Available options are: "no_touch",'
            ' "init_based", "init_with_inflow", "with_sounding".'
        )

      if ambient_name in list(self.config.additional_state_keys) + list(
          self.config.helper_var_keys
      ):
        updated_additional_states[ambient_name] = u_mean_old

    return updated_states, updated_additional_states

  def _preprocess_with_outflow(
      self,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Preprocesses outflow related helper variables."""
    updated_additional_states = {}

    halos = [self.config.halo_width] * 3
    halos[self._g_dim] = 0
    mean_dims = [0, 1, 2]
    del mean_dims[self._g_dim]

    for varname in list(common.KEYS_VELOCITY) + list(
        self.config.transport_scalars_names
    ):
      # Update the target state for sponge at the outflow in case it's used.
      sponge_varname = f'{varname}_out'
      if sponge_varname not in additional_states:
        continue

      vel_mean = common_ops.global_mean(
          states[varname], replicas, halos, mean_dims, mean_dims
      )
      updated_additional_states[sponge_varname] = (
          geophysical_flow_common.broadcast_vertical_profile_for_flow_field(
              vel_mean, self._g_dim
          )
      )

    return updated_additional_states

  def pre_simulation_update_fn(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Updates the hydrodynamic pressure to balance the gravity.

    This function is invoked only once at the step specified in the commandline
    flag.

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
      A dictionary that is a union of `states` and `additional_states` with
      updated `p`.
    """
    updated_states = dict(states)
    updated_additional_states = dict(additional_states)

    dims = (0, 1, 2)
    replica_dims = (0, 1, 2)

    zz = additional_states.get('zz', None)
    if not self.config.use_stretched_grid[self._g_dim]:
      # When not using stretched grid, updates the vertical coordinates in the
      # halos. When stretched grid is used, this is unnecessary because `zz`
      # saved into additional states already includes valid coordinates in the
      # halos.
      h = self.config.grid_spacings[self._g_dim]
      if zz is not None:
        bc_zz = [
            [(halo_exchange.BCType.NEUMANN, 0.0)] * 2,
        ] * 3
        bc_zz[self._g_dim] = [(halo_exchange.BCType.NEUMANN, h)] * 2
        zz = halo_exchange.inplace_halo_exchange(
            tensor=additional_states['zz'],
            dims=dims,
            replica_id=replica_id,
            replicas=replicas,
            replica_dims=replica_dims,
            periodic_dims=self.config.periodic_dims,
            boundary_conditions=bc_zz,
            width=self.config.halo_width,
        )
        updated_additional_states['zz'] = zz

    # Update either the velocity initial condition or the mean inflow so that
    # the mean profile of the inflow and initial condition always match with
    # each other.
    inflow_init_states, inflow_init_additional_states = (
        self._preprocess_init_and_inflow(
            replica_id, replicas, states, additional_states
        )
    )
    updated_states.update(inflow_init_states)
    updated_additional_states.update(inflow_init_additional_states)

    # Update the outflow related helper variables.
    outflow_additional_states = self._preprocess_with_outflow(
        replicas, updated_states, updated_additional_states
    )
    updated_additional_states.update(outflow_additional_states)

    # Update halos for all scalars and recompute the density to make sure values
    # in halos are valid before integration.
    for sc_name in self.config.transport_scalars_names:
      updated_states[sc_name] = halo_exchange.inplace_halo_exchange(
          tensor=updated_states[sc_name],
          dims=dims,
          replica_id=replica_id,
          replicas=replicas,
          replica_dims=replica_dims,
          periodic_dims=self.config.periodic_dims,
          boundary_conditions=self.config.bc[sc_name],
          width=self.config.halo_width,
      )
    rho = self.thermodynamics.update_thermal_density(
        updated_states, updated_additional_states
    )
    rho_ref = self.cloud_utils.thermodynamics.rho_ref(zz, additional_states)
    p = cloud_utils.compute_buoyancy_balanced_hydrodynamic_pressure(
        kernel_op,
        replica_id,
        replicas,
        rho,
        rho_ref,
        self._g_dim,
        self.config,
        additional_states,
    )
    updated_states['p'] = p

    # Application-specific initial halo updates.
    halo_update_fns = self.sim_setup.initial_halo_update_fn()
    for k, halo_update_fn in halo_update_fns.items():
      if k not in updated_additional_states:
        continue
      updated_additional_states[k] = halo_update_fn(
          kernel_op,
          replica_id,
          replicas,
          updated_states,
          updated_additional_states,
      )

    # Update the helper temperature states for consistency.
    updated_additional_states.update(
        self.cloud_utils.temperature_update_fn(
            kernel_op,
            replica_id,
            replicas,
            updated_states,
            updated_additional_states,
            params,
        )
    )
    if 'T_s' in additional_states:
      updated_additional_states['T_s'] = updated_additional_states['T']

    output = updated_additional_states
    output.update(updated_states)

    return output

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

    This function is invoked only once at the step specified in the commandline
    flag.

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
      A dictionary that is a union of `states` and `additional_states`.
    """
    # If combustion is included in the simulation, we are omitting other
    # modifications to the flow field.
    if self.config.combustion is not None and self.fire_utils is not None:
      logging.info('Igniting the flow field.')
      ignition_fn = combustion.ignition_with_hot_kernel(
          self.energy_variable,
          self.fire_utils.ignition_temperature,
          self.config,
      )
      output = ignition_fn(
          kernel_op, replica_id, replicas, states, additional_states, params
      )
    else:
      logging.info('Running the default post_simulation_update_fn.')
      output = self.sim_setup.post_simulation_update_fn(
          kernel_op, replica_id, replicas, states, additional_states, params
      )

    return output

  # Source term for the mass loading from q_r and q_s in the momentum equation.
  def source_term_update_fn(self) -> parameters_lib.SourceUpdateFnLib:
    """Provides the mass loading source term in the momentum equation."""

    def src_w_mass_loading_fn(
        kernel_op: get_kernel_fn.ApplyKernelOp,
        replica_id: tf.Tensor,
        replicas: np.ndarray,
        states: FlowFieldMap,
        additional_states: FlowFieldMap,
        params: grid_parametrization.GridParametrization,
    ) -> FlowFieldMap:
      del kernel_op, replica_id, replicas, params, additional_states
      q_r = states['q_r']
      q_s = (
          states['q_s']
          if 'q_s' in states
          else tf.nest.map_structure(tf.zeros_like, q_r)
      )

      # Reference: The simulation of three-dimensional convective storm
      # dynamics, Klemp & Wilhelmson, J. Atmos. Sci, 1978.
      # Note that the effects of water vapor and condensate mass fraction on
      # buoyancy are already accounted for when computing the equilibrium state.
      # The only adjustment needed here is to account for the mass loading due
      # to precipitation.
      mass_loading = tf.nest.map_structure(
          lambda rho_i, q_r_i, q_s_i: -1.0
          * constants.G
          * rho_i
          * (q_r_i + q_s_i),
          states['rho'],
          q_r,
          q_s,
      )
      return {'src_w': mass_loading}

    def canopy_drag_fn(velocity_component: str):
      """Generates a function to compute the drag for `velocity_component`."""

      def source_fn(
          kernel_op: get_kernel_fn.ApplyKernelOp,
          replica_id: tf.Tensor,
          replicas: np.ndarray,
          states: FlowFieldMap,
          additional_states: FlowFieldMap,
          params: grid_parametrization.GridParametrization,
      ) -> FlowFieldMap:
        """Computes the drag force for `velocity_component`."""
        src_name = f'src_{velocity_component}'
        src_fn = {
            src_name: tf.nest.map_structure(
                tf.zeros_like, states[velocity_component]
            )
        }

        # Currently a combustion model needs to be defined to compute the
        # canopy drag. In case only fuel is initialized but no combustion model
        # is activated, we set the canopy drag to 0.
        if self.fire_utils is None:
          return src_fn

        drag_states = {vel: states[vel] for vel in ('u', 'v', 'w')}
        drag_states.update(
            {
                'rho': states.get(
                    'rho_thermal',
                    self.thermodynamics.update_thermal_density(
                        states, additional_states
                    ),
                )
            }
        )
        src_fn['rho_f'] = additional_states['rho_f']
        return self.fire_utils.drag_force_fn(
            kernel_op, replica_id, replicas, drag_states, src_fn, params
        )

      return source_fn

    source_fns = collections.defaultdict(list)

    # Add the mass loading to the vertical velocity component if rain is
    # present.
    if (
        self.sim_type
        in (
            SimulationType.PYROCB,
            SimulationType.GCM_COLUMN,
        )
        and 'q_r' in self.config.transport_scalars_names
    ):
      source_fns['w'].append(src_w_mass_loading_fn)

    # Add the canopy drag force.
    if 'rho_f' in self.config.additional_state_keys:
      source_fns['u'].append(canopy_drag_fn('u'))
      source_fns['v'].append(canopy_drag_fn('v'))
      source_fns['w'].append(canopy_drag_fn('w'))

    # Application-specific momentum source functions.
    for source_name, fn in self.sim_setup.momentum_source_fn().items():
      source_fns[source_name].append(fn)
    return {
        source_name: cloud_utils.add_source_fns(fns)
        for source_name, fns in source_fns.items()
    }

  def additional_states_update_fn(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      step_id: tf.Tensor,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Updates source terms for thermodynamic states `e_t` and `q_t`."""
    thermodynamics = self.cloud_utils.thermodynamics

    additional_states = dict(additional_states)
    additional_states.update(self.cloud_utils.temperature_update_fn(
        kernel_op, replica_id, replicas, states, additional_states, params))

    if self.sponge is not None:
      additional_states.update(self.sponge.additional_states_update_fn(
          kernel_op, replica_id, replicas, states, additional_states, params))

    if self.config.combustion is not None:
      assert self.energy_variable in ('theta', 'theta_li'), (
          'The combustion model is supported for energy variable `theta` and'
          f' `theta_li` only. Currently `{self.energy_variable}`.'
      )
      additional_states.update(
          _combustion_update(kernel_op, replica_id, replicas, step_id,
                             self.thermodynamics, thermodynamics, states,
                             additional_states, self.energy_variable,
                             (self.sponge is not None and
                              self.energy_variable in self.sponge.varnames),
                             (self.sponge is not None and
                              'Y_O' in self.sponge.varnames),
                             self.config))
    if self.abl is not None and _ENFORCE_MOST_CONSISTENT_BC.value:
      # Only update the surface boundary condition using the most flux if all
      # active scalars' surface boundary conditions are of type NEUMANN.
      assert (
          self.config.boundary_models is not None
          and self.config.boundary_models.HasField('most')
      ), 'MOST compatible BC is requested but the model is not configured.'

      for active_scalar in self.config.boundary_models.most.active_scalar:
        if self.config.bc[active_scalar][
            self._g_dim][0][0] != halo_exchange.BCType.NEUMANN:
          raise ValueError('The surface boundary condition of MOST-active '
                           'scalars must be of type NEUMANN when enabling '
                           '\'FLAGS.enforce_most_consistent_bc\'.')
      additional_states.update(self.abl.neumann_bc_update_fn(
          kernel_op, states, additional_states))

    if self.coriolis_force_fn is not None:
      src_states = {
          f'src_{var}': tf.nest.map_structure(tf.zeros_like, states[var])
          for var in ('u', 'v', 'w')
      }
      for var in common.KEYS_VELOCITY:
        ambient_name = AMBIENT_STATE_TEMPLATE.format(varname=var)
        if ambient_name not in additional_states:
          continue
        src_states[ambient_name] = additional_states[ambient_name]
      src_states = self.coriolis_force_fn(kernel_op, replica_id, replicas,
                                          states, src_states, params)
      # Convert source term for primitive variables to conservative variables,
      # and add them to the existing source terms.
      additional_states.update({  # pylint: disable=g-complex-comprehension
          var: tf.nest.map_structure(
              lambda sponge, coriolis, rho, b: tf.where(
                  tf.greater(b, 0.0), sponge, rho * coriolis),
              additional_states[var], src_states[var], states['rho'],
              additional_states['sponge_beta'])
          for var in ('src_u', 'src_v', 'src_w')
      })

    if 'q_c' in self.config.additional_state_keys:
      additional_states.update({
          'q_c': thermodynamics.saturation_excess(
              additional_states['T'], states['rho'], states['q_t']
          )
      })

    # This is the case where we should have `q_c` and `q_v` as scalar transport
    # states.
    if ('q_t' in self.config.additional_state_keys and
        'q_c' in states and 'q_v' in states):
      additional_states.update({
          'q_t': tf.nest.map_structure(
              tf.math.add, states['q_c'], states['q_v'])
      })

    for varname in ('u', 'v', 'w', 'theta_li', 'q_t', 'q_c', 'q_v'):
      mean_varname = f'mean_rho_{varname}_vol'
      if mean_varname in self.config.helper_var_keys:
        val = tf.nest.map_structure(tf.math.multiply, states['rho'],
                                    states[varname])
        additional_states.update({
            mean_varname:
                common_ops.global_mean(val, replicas,
                                       [self.config.halo_width] * 3)
        })

    if self.const_heat_src.include_const_heat_src:
      t = self.config.dt * tf.cast(step_id, types.TF_DTYPE)
      if 'ignition_kernel' not in additional_states:
        raise ValueError('"ignition_kernel" not found in `additional_states`.')
      heat_source = self.ignition_heat_src_fn(
          additional_states['ignition_kernel'], t
      )
      src_energy_updated = self.config.combustion is not None or (
          self.sponge is not None
          and self.energy_variable in self.sponge.varnames
      )
      if src_energy_updated:
        logging.info(
            'Source term for %s updated by sponge or combustion. Artificial'
            ' heat source is added on top.',
            self.energy_variable,
        )
        additional_states[f'src_{self.energy_variable}'] = (
            tf.nest.map_structure(
                tf.math.add,
                additional_states[f'src_{self.energy_variable}'],
                heat_source,
            )
        )
      else:
        logging.info(
            'The artificial heat source is the only source term for %s.'
            ' Overriding its source term from previous step.',
            self.energy_variable,
        )
        additional_states[f'src_{self.energy_variable}'] = heat_source

    if self.const_heat_src.include_const_heat_flux:
      additional_states['heat_flux'] = self._heat_flux_update_fn(
          params.x_local_ext(replica_id, replicas),
          params.y_local_ext(replica_id, replicas),
          params.z_local_ext(replica_id, replicas),
          self.config.dt * tf.cast(step_id, types.TF_DTYPE),
      )

    if (
        self.radiation_src_update_fn is not None
        and self.config.radiative_transfer is not None
    ):
      update_cycle_sec = self.config.radiative_transfer.update_cycle_seconds
      update_cycle_steps = tf.cast(update_cycle_sec / self.config.dt, tf.int32)
      update_condition = tf.equal(
          tf.math.floormod(step_id, update_cycle_steps), 0
      )

      # If the apply_cadence is left empty, or at the default value of 0, we
      # do no supercycling and apply the radiative heating at every step.
      if self.config.radiative_transfer.apply_cadence_seconds == 0:
        apply_cadence_steps = 1
      else:
        apply_cadence_sec = self.config.radiative_transfer.apply_cadence_seconds
        apply_cadence_steps = tf.cast(
            apply_cadence_sec / self.config.dt, tf.int32
        )

      # The radiative heating rate is recomputed every `update_cycle_steps` and
      # in between updates the most recently computed heating rate is carried
      # over. This enables the radiative heating rate to have a persistent
      # contribution to the energy equation even if it is being updated somewhat
      # infrequently. The staleness is justified because the time scale of
      # radiative flux profile changes is very large compared to the typical
      # timestep of the LES.
      def filter_radiation_states(m: FlowFieldMap) -> FlowFieldMap:
        return {
            k: m[k]
            for k in self.radiation_state_keys
            if k != rrtmgp_common.KEY_APPLIED_RADIATION
        }

      def radiation_update_fn() -> FlowFieldMap:
        output = self.radiation_src_update_fn(
            replica_id,
            replicas,
            states,
            additional_states,
        )
        return filter_radiation_states(output)

      additional_states.update(
          tf.cond(
              pred=update_condition,
              true_fn=radiation_update_fn,
              false_fn=lambda: filter_radiation_states(additional_states),
          )
      )
      # The radiative heating rate is applied every `apply_cadence_steps`. This
      # is a form of supercycling of the radiation heating, where it is applied
      # not every step only every certain number of steps. This is necessary
      # because the radiative heating rate can be so small that the updates are
      # lost under single precision. The radiative heating is applied based on
      # `additional_states[rrtmgp_common.KEY_APPLIED_RADIATION]`.  When we
      # supercycle, we multiply the radiative heating rate by the number of
      # steps that have passed since the last time it was applied.
      apply_condition = tf.equal(
          tf.math.floormod(step_id, apply_cadence_steps), 0
      )
      additional_states[rrtmgp_common.KEY_APPLIED_RADIATION] = (
          tf.cast(apply_cadence_steps, types.TF_DTYPE)
          * tf.cond(
              pred=apply_condition,
              true_fn=lambda: additional_states[
                  rrtmgp_common.KEY_STORED_RADIATION
              ],
              false_fn=lambda: tf.zeros_like(
                  additional_states[rrtmgp_common.KEY_STORED_RADIATION]
              ),
          )
      )

    if isinstance(
        self.inflow, simulated_turbulent_inflow.SimulatedTurbulentInflow
    ):
      additional_states = dict(
          self.inflow.additional_states_update_fn(
              replica_id, replicas, step_id, states, additional_states
          )
      )
    else:
      assert self.inflow is None, (
          'Only simulated turbulent inflow is supported. Unknown option:'
          f' {self.inflow}'
      )

    source_fns = self.sim_setup.scalar_source_fn()
    for sc_name in source_fns:
      if sc_name in self.config.transport_scalars_names:
        src = source_fns[sc_name](
            kernel_op,
            replica_id,
            replicas,
            step_id,
            states,
            additional_states,
            params,  # pylint: disable=too-many-function-args
        )
        for src_key, src_value in src.items():
          additional_states[src_key] = tf.nest.map_structure(
              tf.math.add, additional_states[src_key], src_value
          )

    # Add simulation-specific updates.
    additional_states |= self.sim_setup.additional_states_update(
        replica_id, replicas, step_id, states, additional_states
    )

    return additional_states  # pytype: disable=bad-return-type

  def initial_states(
      self,
      replica_id: tf.Tensor,
      coordinates: initializer.ThreeIntTuple,
  ) -> FlowFieldMap:
    """Initializes the simulation with the geophysical flow setup."""

    initialize_states = lambda value_fn: self.sim_setup.initialize_states(
        value_fn, coordinates
    )

    # Get helper variables required by the specific simulation configuration.
    helper_states_fn = self.sim_setup.helper_states_fn()
    helper_states = {
        varname: initialize_states(init_fn)
        for varname, init_fn in helper_states_fn.items()
    }

    # Initialize the flow quantities.
    u = initialize_states(self.sim_setup.velocity_init_fn('u'))
    v = initialize_states(self.sim_setup.velocity_init_fn('v'))
    w = initialize_states(self.sim_setup.velocity_init_fn('w'))
    output = {
        'replica_id': replica_id,
        'u': u,
        'v': v,
        'w': w,
        'p': initialize_states(
            lambda xx, yy, zz, lx, ly, lz, coord: tf.zeros_like(zz)
        ),
        'rho': initialize_states(
            self._thermal_states_init_fn('rho', helper_states_fn)
        ),
    }

    # Initializes helper variables.
    zz_fn = lambda xx, yy, zz, lx, ly, lz, coord: (xx, yy, zz)[self._g_dim]
    output.update({
        'zz': initialize_states(zz_fn),
        'nu_t': initialize_states(init_fn_lib.constant_init_fn(0.0)),
    })

    # Initialize thermodynamical states. Note, in `theta` formulation, we use
    # `q_v` and `q_c` explicitly as the prognostic variables. In other
    # formulation, we use `q_t`.
    mixing_ratio_vars = ('q_t', 'q_v', 'q_c')

    if set(self.config.transport_scalars_names).issuperset(mixing_ratio_vars):
      raise ValueError('Overconstrained mixing ratio transport scalars '
                       'specified in the configuration. At most 2 of `q_t`, '
                       '`q_v` and `q_c` can be specified.')

    for var in mixing_ratio_vars:
      if var not in self.config.transport_scalars_names:
        continue

      output.update(
          {
              var: initialize_states(
                  self._thermal_states_init_fn(var, helper_states_fn)
              ),
          }
      )

    if self.energy_variable == 'e_t':
      e = initialize_states(self._thermal_states_init_fn('e', helper_states_fn))
      output['e_t'] = self.cloud_utils.thermodynamics.total_energy(
          e, u, v, w, output['zz']
      )

      def compute_e_t_bc(dim, face, halo_height: tf.Tensor) -> tf.Tensor:
        """Computes the total energy at the boundary."""
        t_states = self.sim_setup.thermodynamics_states(halo_height)
        q_t = t_states['q_t']

        p = self.cloud_utils.thermodynamics.p_ref(halo_height, helper_states)
        t = t_states['temperature']
        r_m = t_states['r_m']

        rho = p / r_m / t

        q_l, q_i = self.cloud_utils.thermodynamics.equilibrium_phase_partition(
            t, rho, q_t
        )
        e_int = self.cloud_utils.thermodynamics.internal_energy(
            t, q_t, q_l, q_i
        )
        # The initial wind profile can be either a constant or a profile of
        # height. In the later case `init_wind` is a tuple of length 2, with
        # the first element being the wind speed on the lower side of the
        # domain, and the second is for the higher end.
        init_wind = {
            var: self.init_wind[var] if isinstance(self.init_wind[var], float)
                 else self.init_wind[var][face] for var in ('u', 'v', 'w')
        }

        u = init_wind['u'] * tf.ones_like(halo_height)
        v = init_wind['v'] * tf.ones_like(halo_height)
        w = init_wind['w'] * tf.ones_like(halo_height)
        bc_e_t = self.cloud_utils.thermodynamics.total_energy(
            e_int, u, v, w, halo_height
        )
        shifted_dim = (dim + 1) % 3

        def get_halos_tensor(bc_halos):
          """Returns a partial tensor containing only the halo values."""
          halo_shape = [1, 1, 1]
          halo_shape[shifted_dim] = self.config.halo_width
          bc_halos = tf.reshape(bc_halos, halo_shape)
          tile_multipliers = [self.config.nz, self.config.nx, self.config.ny]
          tile_multipliers[shifted_dim] = 1
          return tf.tile(bc_halos, tile_multipliers)

        def get_partial_tensor(state):
          """Returns the partial tensor excluding the halos at the BC face."""
          slice_idx = [0, 0, 0]
          slice_lens = [self.config.nz, self.config.nx, self.config.ny]
          slice_lens[shifted_dim] -= self.config.halo_width
          if face == 0:
            slice_idx[shifted_dim] = self.config.halo_width
          return tf.slice(state, slice_idx, slice_lens)

        # Full shape except for the vertical direction, which will have a
        # dimension of `halo_width`.
        bc_e_t_halos = get_halos_tensor(bc_e_t)

        # Full shape except for the vertical direction, which will be
        # `halo_width` short of the full vertical dimension.
        bc_e_t_partial = get_partial_tensor(output['e_t'])

        if face == 0:
          bc_e_t_tensor = tf.concat(
              [bc_e_t_halos, bc_e_t_partial], axis=shifted_dim
          )
        else:
          bc_e_t_tensor = tf.concat(
              [bc_e_t_partial, bc_e_t_halos], axis=shifted_dim
          )
        return bc_e_t_tensor

      bc_e_t_lst = [
          bc for bc in self.config.additional_state_keys
          if bc.startswith('bc_e_t')
      ]
      for bc_e_t_k in bc_e_t_lst:
        _, dim, face = self.bc_manager.parse_key(bc_e_t_k)
        if any(self.config.use_stretched_grid):
          raise ValueError(
              'Stretched grid is not yet supported for using the total energy'
              ' prognostic variable with custom BC.'
          )
        dh = self.config.grid_spacings[dim]
        lk = self._domain_lens[dim]
        if face == 0:
          height = tf.constant([-2.0 * dh, -dh], dtype=tf.float32)
        else:
          height = tf.constant([lk + dh, lk + 2.0 * dh], dtype=tf.float32)
        output[bc_e_t_k] = compute_e_t_bc(dim, face, height)

    elif self.energy_variable == 'theta_li':
      output.update(
          {
              'theta_li': initialize_states(
                  self._thermal_states_init_fn('theta_li', helper_states_fn)
              )
          }
      )
    elif self.energy_variable == 'theta':
      output.update(
          {
              'theta': initialize_states(
                  self._thermal_states_init_fn('theta', helper_states_fn)
              )
          }
      )
    else:
      raise ValueError(
          'One and only one energy variable is required but 0 is defined.')

    # Initialize the precipitation mass fraction if required.
    for varname in ('q_r', 'q_s'):
      if varname in self.config.transport_scalars_names:
        output.update(
            {
                varname: initialize_states(
                    self._thermal_states_init_fn(varname, helper_states_fn)
                )
            }
        )

    if ('q_c' in self.config.additional_state_keys or
        'theta_li' in self.config.additional_state_keys):
      if 'T' not in self.config.additional_state_keys:
        raise KeyError(
            'Some additional states depend on T, which is not among the '
            'additional_state_keys.')

    if (self.config.use_sgs and
        self.config.sgs_model.WhichOneof('sgs_model_type')
        == 'smagorinsky_lilly'):
      output.update(
          {
              'theta_v': initialize_states(
                  self._thermal_states_init_fn('theta_v', helper_states_fn)
              ),
          }
      )

    # Initialize sponge required data.
    if self.sponge is not None:
      klrc = rayleigh_damping_layer.klemp_lilly_relaxation_coeff_fns_for_sponges
      assert (
          self.config.boundary_models is not None
          and self.config.boundary_models.sponge
      ), 'Sponge is requested but the model is not configured.'
      beta_fns_by_name = klrc(
          self.config.boundary_models.sponge, self.config.x[0],
          self.config.y[0], self.config.z[0])
      output.update(self.sponge.init_fn(self.config, coordinates,
                                        beta_fns_by_name))
    for var in list(self.config.transport_scalars_names) + ['u', 'v', 'w']:
      sponge_target_key = f'{var}_init'
      if sponge_target_key in self.config.additional_state_keys:
        output[sponge_target_key] = output[var]

    output.update(self.monitor.data)

    # Add combustion related variables if a combustion model is included in the
    # config.
    output.update(self.add_combustion_states(initialize_states))

    # Add a constant heat source if requested from the input flag.
    if self.const_heat_src.include_const_heat_src:
      logging.info('Heat flux due to combustion: %g W/m^2',
                   self.const_heat_src.heat_flux_due_to_combustion)
      output[f'src_{self.energy_variable}'] = initialize_states(
          init_fn_lib.constant_init_fn(0.0)
      )

    # Add a constant heat flux if requested from the input flag.
    if self.const_heat_src.include_const_heat_flux:
      logging.info('Heat flux due to combustion: %g W/m^2',
                   self.const_heat_src.heat_flux_due_to_combustion)
      output['heat_flux'] = initialize_states(
          lambda xx, yy, zz, lx, ly, lz, coord: tf.zeros_like(zz))[:1]

    # Add a radiation heat source.
    for rad_key in self.radiation_state_keys:
      output[rad_key] = initialize_states(
          lambda xx, yy, zz, lx, ly, lz, coord: tf.zeros_like(zz)
      )

    # Add simulation specific helper variables.
    output.update(self.sim_setup.initial_states(replica_id, coordinates))
    output.update(helper_states)

    # Add inflow related variables.
    if isinstance(
        self.inflow, simulated_turbulent_inflow.SimulatedTurbulentInflow
    ):
      output.update(self.inflow.initialize_inflow())
    else:
      assert self.inflow is None, (
          'Only simulated turbulent inflow is supported. Unknown option:'
          f' {self.inflow}'
      )
    for varname in ('u', 'theta_li', 'q_t'):
      bc_name = f'bc_{varname}_0_0'
      if bc_name not in self.config.additional_state_keys:
        continue
      output[bc_name] = output[varname][:, : self.config.halo_width + 1, :]

    # Add outflow related variables.
    # Note that we assume the target state name for the sponge of `varname` is
    # `varname_out`.
    for varname in list(self.config.transport_scalars_names) + ['u']:
      sponge_target_name = f'{varname}_out'
      if sponge_target_name not in self.config.additional_state_keys:
        continue
      output[sponge_target_name] = tf.math.reduce_mean(
          output[varname], axis=(1, 2), keepdims=True
      )

    # Add Coriolis force related variables.
    if self.coriolis_force_fn is not None:
      for varname in common.KEYS_VELOCITY:
        src_name = f'src_{varname}'
        if src_name not in self.config.additional_state_keys:
          raise ValueError(
              'Coriolis force requires source terms for velocity.'
              f' {src_name} is missing in the config as an additional state.'
          )
        if src_name in output:
          continue
        output[src_name] = tf.zeros_like(output[varname])

    # Adding additional auxiliary states.
    output = self.add_auxiliary_states(output, helper_states, initialize_states)

    return output

  def add_combustion_states(
      self,
      initialize_states_fn: Callable[[initializer.ValueFunction], tf.Tensor],
  ) -> FlowFieldMap:
    """Initializes variables for combustion."""
    output = {}

    if self.config.combustion is None:
      return output

    fire_utils = self.fire_utils
    assert fire_utils is not None, 'Fire utils is not initialized.'

    def init_rho_f(xx, yy, zz, lx, ly, lz, coord):
      """Generates initial `rho_f` field."""
      # Note that we assume the fuel is piled along the z dimension.
      return fire_utils.generate_fuel_layer_init_fn('rho_f')(
          xx, yy, zz, lx, ly, lz, coord
      )

    def init_rho_m(xx, yy, zz, lx, ly, lz, coord):
      """Generates initial moisture `rho_m` field."""
      # Note that we assume the fuel is piled along the z dimension.
      return fire_utils.generate_fuel_layer_init_fn('rho_m')(
          xx, yy, zz, lx, ly, lz, coord
      )

    output['rho_f'] = initialize_states_fn(init_rho_f)
    if 'rho_f_init' in self.config.additional_state_keys:
      output.update({'rho_f_init': output['rho_f']})

    output.update({
        'Y_O': initialize_states_fn(
            init_fn_lib.constant_init_fn(fire_utils.y_o_init)
        ),
        # Note that when `self.fire_utils` is None, `T_s` is not active in the
        # simulation. Setting it to a default value at 300 K just to avoid the
        # "no attribute" error from pylint.
        'T_s': initialize_states_fn(
            init_fn_lib.constant_init_fn(fire_utils.t_init)
        ),
        'src_rho': initialize_states_fn(init_fn_lib.constant_init_fn(0.0)),
        'src_{}'.format(self.energy_variable): initialize_states_fn(
            init_fn_lib.constant_init_fn(0.0)
        ),
        'src_Y_O': initialize_states_fn(init_fn_lib.constant_init_fn(0.0)),
        'nu_t': initialize_states_fn(init_fn_lib.constant_init_fn(0.0)),
    })
    if 'ignition_kernel' in self.config.additional_state_keys:
      output['ignition_kernel'] = initialize_states_fn(
          fire_utils.ignition_kernel_init_fn(
              tf.constant(fire_utils.ignition_depth, dtype=types.TF_DTYPE),
              tf.constant(0.0, dtype=types.TF_DTYPE),
          )
      )
    if 'src_T' in self.config.additional_state_keys:
      output['src_T'] = initialize_states_fn(init_fn_lib.constant_init_fn(0.0))

    # Add additional states required if moisture is considered in the
    # vegetation.
    if fire_utils.combustion_model_type == 'wood':
      if (
          self.config.combustion.wood.WhichOneof('combustion_model_option')
          == 'moist_wood'
      ):
        output.update({
            'rho_m': initialize_states_fn(init_rho_m),
            'phi_w': initialize_states_fn(init_fn_lib.constant_init_fn(0.0)),
        })
    elif fire_utils.combustion_model_type == 'biofuel_multistep':
      output.update({
          'rho_m': initialize_states_fn(init_rho_m),
          'src_q_t': initialize_states_fn((init_fn_lib.constant_init_fn(0.0))),
      })

    # Add the background temperature as the far field temperature if required
    # by the combustion model.
    if fire_utils.wood.WhichOneof('t_far_field') == 't_variable':
      t_far_field_varname = fire_utils.wood.t_variable
      assert t_far_field_varname in self.config.additional_state_keys, (
          f'{t_far_field_varname} is required as the far field temperature in'
          ' the combustion model, but is not included as an additional state.'
      )
      output[t_far_field_varname] = initialize_states_fn(
          self._thermal_states_init_fn('T', self.sim_setup.helper_states_fn())
      )

    return output

  def _heat_source_2d_mask(
      self, xx: tf.Tensor, yy: tf.Tensor, const_heat_src: ConstantHeatSource
  ) -> tf.Tensor:
    """Derives the mask for the heat source in the horizontal plane."""
    r = tf.math.sqrt(
        tf.math.subtract(xx, const_heat_src.x_center) ** 2
        + tf.math.subtract(yy, const_heat_src.y_center) ** 2
    )

    return tf.where(
        tf.less(r, const_heat_src.r),
        tf.math.cos(0.5 * np.pi * r / const_heat_src.r) ** 2,
        tf.zeros_like(r),
    )

  def _get_max_heat_flux(
      self,
      horizontal_0: tf.Tensor,
      horizontal_1: tf.Tensor,
      const_heat_src: ConstantHeatSource,
  ) -> tf.Tensor:
    """Computes the maximum heat flux as a 3D tensor."""
    horizontal_mask = self._heat_source_2d_mask(
        horizontal_0, horizontal_1, const_heat_src
    )
    heat_flux = self.const_heat_src.heat_flux_due_to_combustion
    if self.energy_variable in ('theta', 'theta_li'):
      # Assuming air density near the ground is 1 kg/m^3.
      src_magnitude = heat_flux / self.cloud_utils.thermodynamics.cp_d
    else:
      raise ValueError(
          'Expected `theta` or `theta_li` as energy variable, but '
          f'got `{self.energy_variable}`.'
      )

    return -src_magnitude * horizontal_mask

  def _heat_flux_update_fn(
      self,
      x: tf.Tensor,
      y: tf.Tensor,
      z: tf.Tensor,
      t: tf.Tensor,
  ) -> types.FlowFieldVal:
    """Computes the heat flux at `t` given x, y, z.

    Args:
      x: The 1D x coordinates.
      y: The 1D y coordinates.
      z: The 1D z coordinates.
      t: The time at the current step, in units of s.

    Returns:
      The heat flux at time `t`.
    """
    horizontal_0, horizontal_1, _ = _mesh_grid(x, y, z, self._g_dim)
    const_heat_src = self.const_heat_src.at_time_t(t)
    coeff = self.ramp_up_down_fn(t)
    heat_flux = tf.nest.map_structure(
        lambda h_0, h_1: coeff * self._get_max_heat_flux(h_0, h_1,
                                                         const_heat_src),
        horizontal_0, horizontal_1)
    # NOTE(bcg): The flux variable needs to be a proper 3D tensor with a
    # z-dimension of length 1 and not a list of size 1 which is why we can't
    # just return heat_flux[:1] here.
    return tf.expand_dims(heat_flux[0], 0)

  def add_auxiliary_states(
      self,
      output: types.TensorMap,
      helper_states: types.TensorMap,
      initialize_states_fn: Callable[[initializer.ValueFunction], tf.Tensor]
  ) -> FlowFieldMap:
    """Initializes variables for invariant reference state diagnostics."""
    all_states = {}
    all_states.update(output)
    # Initializes relevant additional states requested in the config.
    maybe_additional_states = ('T', 'q_c', 'theta_li', 'theta', 'q_t')
    helper_states_fn = self.sim_setup.helper_states_fn()
    all_states.update(
        {
            varname: initialize_states_fn(
                self._thermal_states_init_fn(varname, helper_states_fn)
            )
            for varname in maybe_additional_states
            if varname in self.config.additional_state_keys
        }
    )

    # Using `q_v` as explicit prognostic variable. For the calculation of
    # mass loading, we will need keep the initial value of `q_v` around.
    if 'q_v' in self.config.scalar_lib:
      all_states.update({'q_v_init': all_states['q_v']})

    for varname in ('u', 'v', 'w', 'theta_li', 'q_t', 'q_c', 'q_v'):
      mean_varname = f'mean_rho_{varname}_vol'
      if mean_varname in self.config.helper_var_keys:
        all_states.update(
            {mean_varname: tf.constant(0, dtype=tf.float32)})

    # This allows the reserved diagnostic variables exposed by the inner solver
    # to be activated and added to the output through `additional_state_keys` in
    # the configurations. These are defined in `simulation.py`.
    reserved_diagnostic_vars = (
        'rho_thermal', 'buoyancy_u', 'buoyancy_v', 'buoyancy_w')
    for diagnostic_var in reserved_diagnostic_vars:
      if diagnostic_var in self.config.additional_state_keys:
        all_states.update({
            diagnostic_var: tf.nest.map_structure(tf.zeros_like, output['zz'])})

    if 'rho_ref' in self.config.additional_state_keys:
      all_states.update({
          'rho_ref': self.cloud_utils.thermodynamics.rho_ref(
              output['zz'], helper_states)})
    if 'p_ref' in self.config.additional_state_keys:
      all_states.update({
          'p_ref': self.cloud_utils.thermodynamics.p_ref(
              output['zz'], helper_states)})
    if 'T_ref' in self.config.additional_state_keys:
      all_states.update({
          'T_ref': self.cloud_utils.thermodynamics.t_ref(
              output['zz'], helper_states)})

    # `theta_background` will be initialized to the initial `theta`, which can
    # either be in `output` or in `auxiliary_states`.
    if ('theta_background' in self.config.additional_state_keys and
        'theta' in all_states.keys()):
      all_states.update({'theta_background': all_states['theta']})

    # Add ambient state from the mean initial condition. These states are saved
    # as 1D variables along the `g_dim`, and are made broadcastable to 3D
    # states. Note that halos are safe here because they are initialized with
    # valid values.
    mean_axis = [0, 1, 2]
    g_axis = (self._g_dim + 1) % 3
    mean_axis.remove(g_axis)
    for varname in list(common.KEYS_VELOCITY) + list(
        self.config.transport_scalars_names
    ):
      ambient_varname = AMBIENT_STATE_TEMPLATE.format(varname=varname)
      if ambient_varname not in list(self.config.additional_state_keys) + list(
          self.config.helper_var_keys
      ):
        continue
      all_states[ambient_varname] = tf.math.reduce_mean(
          output[varname], mean_axis, True
      )

    return all_states


def _combustion_update(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    step_id: tf.Tensor,
    thermodynamics: thermodynamics_manager.ThermodynamicsManager,
    cloud_thermodynamics: water.Water,
    states: FlowFieldMap,
    additional_states: FlowFieldMap,
    energy_variable: str,
    energy_variable_has_sponge: bool,
    y_o_has_sponge: bool,
    config: parameters_lib.SwirlLMParameters,
):
  """Computes additional_state updates for combustion."""

  assert energy_variable in ('theta', 'theta_li'), (
      'Only energy variables "theta" and "theta_li" are supported for the'
      f' combustion model currently, but {energy_variable} is used.'
  )

  # Replace the energy variable in states by temperature to account for the
  # temperature variation by geopotential in the vertical direction.
  states_combustion = dict(states)
  zz = config.maybe_grid_vertical(replica_id, replicas)
  states_combustion['rho'] = states.get(
      'rho_thermal',
      thermodynamics.update_thermal_density(states, additional_states),
  )
  states_combustion['T'] = additional_states.get(
      'T',
      cloud_thermodynamics.saturation_adjustment(
          energy_variable,
          states[energy_variable],
          states_combustion['rho'],
          states['q_t'],
          zz,
          additional_states,
      ),
  )
  del states_combustion[energy_variable]

  additional_states_combustion = dict(additional_states)
  # Clear the source term from previous steps so that the updated source term
  # contains only results from the combustion at the present time step.
  additional_states_combustion.update({
      'src_T': tf.nest.map_structure(
          tf.zeros_like, states[energy_variable]
      ),
      'src_Y_O': tf.nest.map_structure(tf.zeros_like, states['Y_O']),
      'src_rho': tf.nest.map_structure(tf.zeros_like, states['rho']),
  })

  additional_states_combustion.update(
      combustion.combustion_step(
          kernel_op,
          replica_id,
          replicas,
          step_id,
          states_combustion,
          additional_states_combustion,
          config,
      )
  )

  # Updates the source terms for Y_O and the energy variable. If sponge is
  # not used, source terms from combustion will override values in
  # additional_states previously; otherwise it will be added to the source
  # term computed in the sponge update step.
  if y_o_has_sponge:
    additional_states_combustion['src_Y_O'] = tf.nest.map_structure(
        tf.math.add,
        additional_states['src_Y_O'],
        additional_states_combustion['src_Y_O'],
    )

  src_energy = f'src_{energy_variable}'
  if energy_variable_has_sponge:
    additional_states_combustion[src_energy] = tf.nest.map_structure(
        tf.math.add,
        additional_states[src_energy],
        additional_states_combustion['src_T'],
    )
  else:
    additional_states_combustion[src_energy] = additional_states_combustion[
        'src_T'
    ]

  return additional_states_combustion
