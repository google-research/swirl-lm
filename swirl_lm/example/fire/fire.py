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

"""A simulation of flow over various types of terrains (with or without fire).

Historical note: this file was originally modeling fire over a flat surface and
was named flat_surface.py. It has evolved to be more general and can now handle
different types of terrains so we renamed it to fire.py. We also changed many of
the names accordingly, but not some names (for example flags) still use the
phrase "flat surface".

Remarks:
* The Rayleigh damping layer (sponge)
In this simulation, sponge can be applied as the boundary treatment. Supported
target values for the sponge are:
(1) A constant;
(2) `{variable_name}_init`, which uses the initial condition as the target;
(3) `[uvw]_log`, which uses a log profile to represent the mean boundary layer
of the flow.
Note that the (2) and (3) option require additional variables to be specified
as `additional_states` in the config.
"""

import enum
from typing import Callable, Optional, Sequence, Tuple

from absl import flags
from absl import logging
import numpy as np
from swirl_lm.base import initializer
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.boundary_condition import immersed_boundary_method
from swirl_lm.boundary_condition import simulated_turbulent_inflow
from swirl_lm.boundary_condition import synthetic_turbulent_inflow
from swirl_lm.communication import halo_exchange
from swirl_lm.example.fire import terrain_utils
from swirl_lm.example.shared import cloud_utils
from swirl_lm.example.shared import geophysical_flow_utils
from swirl_lm.example.shared import wildfire_utils
from swirl_lm.numerics import filters
from swirl_lm.physics.combustion import igniter
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.utility import common_ops
from swirl_lm.utility import components_debug
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import init_fn as init_fn_lib
from swirl_lm.utility import probe
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldMap = types.FlowFieldMap
InitFnDict = init_fn_lib.InitFnDict


class TerrainType(enum.Enum):
  """The type of the flat surface terrain."""
  # A flat surface without elevation.
  NONE = 0
  # A flat surface with constant slope.
  RAMP = 1
  # A bump in the center of the domain that is symmetric in the streamwise
  # direction.
  BUMP = 2
  # A leveled flat surface transition to a constant angle slope. The transition
  # point is at the center of the x axis.
  WEDGE = 3
  # Read the terrain map from a file.
  FILE = 4


# Flags for the terrain initialization.
_TERRAIN_FILEPATH = flags.DEFINE_string(
    'terrain_filepath', None,
    'The full path to a data file that contains a 2D array representing the '
    'elevation of an area. This can be in `.npy` or `.ser` format.',
    allow_hide_cpp=True
)
_FLAT_SURFACE_INITIAL_HEIGHT = flags.DEFINE_float(
    'flat_surface_initial_height',
    0.0,
    'The height of the flat surface at the inlet.',
    allow_override=True)
_FLAT_SURFACE_MAX_HEIGHT = flags.DEFINE_float(
    'flat_surface_max_height',
    1e4,
    'The maximum height of the flat surface. Terrain above this height is set '
    'to this constant.',
    allow_override=True)
_FLAT_SURFACE_RAMP_START_POINT = flags.DEFINE_float(
    'flat_surface_ramp_start_point',
    0.0,
    'The starting location of the ramp along the x axis.',
    allow_override=True,
)
_FLAT_SURFACE_RAMP_LENGTH = flags.DEFINE_float(
    'flat_surface_ramp_length',
    1e4,
    'The length of the ramp starting from the inlet. Terrain at the end of the'
    'ramp is set to a constant. Note that the height at the end of the ramp is'
    'constrained by the smaller one of the number determined by this parameter'
    'and `flat_surface_max_height`. This parameter is only used when `RAMP` is'
    'the terrain type.',
    allow_override=True)
flags.DEFINE_float(
    'flat_surface_slope',
    0.0,
    'The slope of the flat surface with respect to the horizontal plane in the '
    'x direction (in degrees). Should be a number between -90 and 90.',
    allow_override=True)

flags.DEFINE_bool(
    'flat_surface_turbulent_inflow',
    False,
    'A option of using sythetic turbulence in the inflow boundary condition.',
    allow_override=True)
flags.DEFINE_bool(
    'flat_surface_include_fire',
    False,
    'A option of including fire dynamics in the simulation.',
    allow_override=True)
flags.DEFINE_bool(
    'flat_surface_ignite',
    False,
    'Set a high temperature spot at the specified ignition location at the '
    'start of a simulation.',
    allow_override=True)
flags.DEFINE_bool(
    'flat_surface_init_bl',
    False,
    'Set to `True` if to use a boundary layer profile as the inflow and '
    'initial conditions for u and v.',
    allow_override=True)
flags.DEFINE_float(
    'flat_surface_blasius_bl_distance',
    1.0,
    'The distance where the Blasius boundary layer profile is computed.',
    allow_override=True)
flags.DEFINE_bool(
    'flat_surface_blasius_bl_transition',
    True,
    'An indicator of whether to apply the transition of boundary layer from '
    'normal-to-ground to coordinate-aligned.',
    allow_override=True)
flags.DEFINE_float(
    'flat_surface_blasius_bl_fraction',
    0.5,
    'The fraction of Blasius boundary layer normal to the ground.',
    allow_override=True)
_TERRAIN_TYPE = flags.DEFINE_enum_class(
    'flat_surface_terrain_type',
    TerrainType.RAMP,
    TerrainType,
    'The type of the terrain. Should be one of the following: RAMP, BUMP.',
    allow_override=True)
flags.DEFINE_bool(
    'flat_surface_use_dynamic_igniter',
    False,
    'Whether ignition will be done as a ignition sequence through multiple '
    'steps. If `False` the ignition will be done at the step specified by '
    '`preprocess_step_id` instantaneously, otherwise ignition is performed in '
    'a sequence that is defined by `flat_surface_ignition_speed`, '
    '`flat_surface_ignition_start_point`, `flat_surface_ignition_duration`, '
    '`flat_surface_igniter_radius`, and `flat_surface_ignition_start_step_id`.',
    allow_override=True)
flags.DEFINE_float(
    'flat_surface_ignition_speed',
    0.8,
    'The speed that the ignition kernel moves. Only takes effect when '
    '`flat_surface_use_dynamic_igniter` is True. ',
    allow_override=True)
flags.DEFINE_list(
    'flat_surface_ignition_start_point',
    [],
    'The (x, y, z) coordinates of the starting point of the ignition. '
    'Only takes effect when `flat_surface_use_dynamic_igniter` is True. ',
    allow_override=True)
flags.DEFINE_float(
    'flat_surface_ignition_duration',
    110.0,
    'The duration of the ignition event, in units of seconds. '
    'Only takes effect when `flat_surface_use_dynamic_igniter` is True. ',
    allow_override=True)
flags.DEFINE_float(
    'flat_surface_igniter_radius',
    2.5,
    'The radius (in units of meter) of the ignition kernel in the ignition '
    'event. Only takes effect when `flat_surface_use_dynamic_igniter` is '
    'True. ',
    allow_override=True)
flags.DEFINE_float(
    'flat_surface_ignition_start_step_id',
    0,
    'The step id at which the ignition starts. Only takes effect when '
    '`flat_surface_use_dynamic_igniter` is True. ',
    allow_override=True)
INCLUDE_CORIOLIS_FORCE = flags.DEFINE_bool(
    'flat_surface_include_coriolis_force',
    False,
    'Whether to include the Coriolis force that drives the flow.',
    allow_override=True)

# Defines the cubic obstacles for the boundary layer transition to turbulent.
INCLUDE_OBSTACLES = flags.DEFINE_bool(
    'flat_surface_include_obstacles',
    False,
    (
        'Whether to include cube obstacles in the flow field for the boundary'
        ' layer to transition to turbulent. The locations of the obstacles are'
        ' specified with `flat_surface_obstacles_x/y` and the sizes of the'
        ' obstacles are specified with `flat_surface_obstacles_w/h`.'
    ),
    allow_override=True,
)
OBSTACLES_X = flags.DEFINE_list(
    'flat_surface_obstacles_x',
    [],
    'The x coordinates of the center of cubic obstacles.',
    allow_override=True)
OBSTACLES_Y = flags.DEFINE_list(
    'flat_surface_obstacles_y',
    [],
    'The y coordinates of the center of cubic obstacles.',
    allow_override=True)
OBSTACLES_W = flags.DEFINE_list(
    'flat_surface_obstacles_w',
    [],
    'The widths of cubic obstacles.',
    allow_override=True)
OBSTACLES_H = flags.DEFINE_list(
    'flat_surface_obstacles_h',
    [],
    'The heights of cubic obstacles.',
    allow_override=True)

_OPT_FILTER_FUEL_DENSITY = flags.DEFINE_bool(
    'opt_filter_fuel_density',
    False,
    (
        'The option of filtering the fuel density spacially with all'
        ' neighboring nodes at the preprocessing step.'
    ),
    allow_override=True,
)


FLAGS = flags.FLAGS


def velocity_bulk_init_fn(
    stream_velocity: float, elevation: tf.Tensor
) -> wildfire_utils.InitFn:
  """Generates a init function for velocity with elevation."""

  def init_fn(xx, yy, zz, lx, ly, lz, coord):
    """Generates the initial velocity component.

    The velocity initial condition with plug flow, i.e. constant magnitude.

    Args:
      xx: The sub-mesh in dimension 0 in the present replica.
      yy: The sub-mesh in dimension 1 in the present replica.
      zz: The sub-mesh in dimension 2 in the present replica.
      lx: Length in dimension 0.
      ly: Length in dimension 1.
      lz: Length in dimension 2.
      coord: The coordinate of the local core.

    Returns:
      The 3D velocity field in dimension 0 in the present replica.
    """
    del xx, yy, lx, ly, lz, coord

    if stream_velocity == 0.0:
      return tf.zeros_like(zz, dtype=zz.dtype)

    return tf.compat.v1.where(
        tf.greater_equal(zz, elevation),
        stream_velocity * tf.ones_like(zz, dtype=zz.dtype),
        tf.zeros_like(zz, dtype=zz.dtype))

  return init_fn


def cubic_obstacles(
    lx: float,
    ly: float,
    nx: int,
    ny: int,
    x_locs: Sequence[float],
    y_locs: Sequence[float],
    widths: Sequence[float],
    heights: Sequence[float],
) -> np.ndarray:
  """Generates a 2D elevation map of a series of cubic blocks.

  Args:
    lx: The length of the domain.
    ly: The width of the domain.
    nx: The number of grid points in along the x direction.
    ny: The number of grid points in along the y direction.
    x_locs: The x coordinates of the center of the cubic obstacles.
    y_locs: The y coordinates of the center of the cubic obstacles.
    widths: The widths of the cubic obstacles.
    heights: The heights of the cubic obstacles.

  Returns:
    A 2D numpy array with values being the height of the obstacles.

  Raises:
    ValueError: If the lengths of the arguments are not all the same.
  """
  n = len(x_locs)
  if len(y_locs) != n or len(widths) != n or len(heights) != n:
    raise ValueError(
        f'The length of inputs are not all equal: '
        f'({n} {len(y_locs)} {len(widths)} {len(heights)}).'
    )

  dx = lx / (nx - 1)
  dy = ly / (ny - 1)

  obstacles_map = np.zeros((nx, ny), dtype=np.float32)
  for i in range(n):
    xl = int((x_locs[i] - 0.5 * widths[i]) // dx)
    xh = int((x_locs[i] + 0.5 * widths[i]) // dx)
    yl = int((y_locs[i] - 0.5 * widths[i]) // dy)
    yh = int((y_locs[i] + 0.5 * widths[i]) // dy)
    obstacles_map[xl:xh, yl:yh] = heights[i]

  return obstacles_map

FirebenchStatesUpdateFn = Callable[
    [tf.Tensor, np.ndarray, tf.Tensor, types.FlowFieldMap,
     types.FlowFieldMap, grid_parametrization.GridParametrization, float],
    types.FlowFieldMap]


def get_init_rho_f(
    ground_elevation: tf.Tensor,
    fuel_bed_height: float,
    fuel_density: float,
    fuel_start_x: float,
    dz: float | tf.Tensor,
) -> wildfire_utils.InitFn:
  """Returns the initializer function for rho_f."""

  # We assume grid coordinates in zz are integer multiples of dz and use +/-0.1
  # * dz offsets to select cells at specific z coordinates while avoiding
  # round-off errors. For example, in a grid with dz=2m, to select the grid
  # points bove z=3.5m, we will first quantize 3.5m with floor(3.5 / 2) * 2 to
  # 2m, and then filter for 2 < zz - 0.1 * dz.

  # `quantized_ground_elevation` is the z coordinate of the *second* grid point
  # that is strictly below ground_elevation. We'll put the lowest fuel at the
  # grid point right above this point, which will have a z-coordinate that
  # satisties
  #
  #   z < ground_elevation <= z + dz
  #
  # For example, in a grid with dz=2m, if ground_elevation is in (0, 2],
  # quantized_ground_elevation will be -2m and the fuel will be placed at the
  # grid point with z=0. Note that if the ground elevation is 0 than the first
  # fuel cell will be in the halo.

  quantized_ground_elevation = tf.math.ceil(ground_elevation / dz) * dz - 2 * dz
  num_full_cells = tf.cast(
      tf.floor(fuel_bed_height / dz), quantized_ground_elevation.dtype
  )
  # Note that the number of full cells is one more than given by
  # fuel_bed_height because we also put fuel into the boundary cell (i.e.,
  # the cell that intersects with the terrain).
  quantized_full_fuel_height = (quantized_ground_elevation +
                                (num_full_cells + 1) * dz)
  rho_f_top_val = fuel_density * (fuel_bed_height - num_full_cells * dz) / dz

  def init_rho_f(xx, yy, zz, lx, ly, lz, coord):
    """Generates initial `rho_f` field."""
    del yy, lx, ly, lz, coord
    # Bottom grid points are fully filled with the given fuel density.
    rho_f_bottom = tf.compat.v1.where(
        tf.math.logical_and(
            zz > quantized_ground_elevation + 0.1 * dz,
            zz < quantized_full_fuel_height + 0.1 * dz
        ),
        fuel_density * tf.ones_like(zz),
        tf.zeros_like(zz),
    )
    # The top grid point is filled proportionally to how much of the remaining
    # fuel bed height extends into the last grid cell.
    rho_f_top = tf.where(
        tf.math.logical_and(
            zz >= quantized_full_fuel_height + 0.1 * dz,
            zz < quantized_full_fuel_height + 1.1 * dz
        ),
        tf.cast(rho_f_top_val, tf.float32),
        tf.zeros_like(zz),
    )

    rho_f = rho_f_bottom + rho_f_top

    return tf.where(tf.less(xx, fuel_start_x), tf.zeros_like(xx), rho_f)

  return init_rho_f


class Fire:
  """A library for simulation of flow over a flat surface."""

  def __init__(
      self,
      fire_utils: wildfire_utils.WildfireUtils,
      uvw_init_fn: Optional[InitFnDict] = None,
      perturb_init_velocity_rand_seed: Optional[int] = None,
      firebench_initialization: Optional[Callable[[int],
                                                  types.FlowFieldMap]] = None,
      firebench_states_update_fn: Optional[FirebenchStatesUpdateFn] = None,
  ):
    """Initializes the library."""
    self.fire_utils = fire_utils
    self.config = fire_utils.config
    self.init_fn_uvw = uvw_init_fn
    self.perturb_init_velocity_rand_seed = perturb_init_velocity_rand_seed
    self.firebench_initialization = firebench_initialization
    self.firebench_states_update_fn = firebench_states_update_fn

    self.coriolis_force_fn = cloud_utils.coriolis_force(0.5497607357, {
        'u': self.fire_utils.u_init,
        'v': self.fire_utils.v_init,
        'w': 0.0
    }, 2)

    self._include_coriolis_force = INCLUDE_CORIOLIS_FORCE.value

    if _TERRAIN_TYPE.value == TerrainType.FILE:
      elevation = terrain_utils.generate_terrain_map_from_file(
          self.config, _TERRAIN_FILEPATH.value
      )
    else:
      # Initializes a simple terrain in the computational domain.
      self.h_0 = _FLAT_SURFACE_INITIAL_HEIGHT.value
      self.slope = FLAGS.flat_surface_slope
      x = tf.linspace(0.0, self.config.lx, self.config.fx)

      if _TERRAIN_TYPE.value == TerrainType.RAMP:
        profile = self.h_0 + tf.clip_by_value(
            x - _FLAT_SURFACE_RAMP_START_POINT.value,
            clip_value_min=0.0,
            clip_value_max=_FLAT_SURFACE_RAMP_LENGTH.value,
        ) * tf.tan(self.slope * np.pi / 180.0)
      elif _TERRAIN_TYPE.value == TerrainType.BUMP:
        forward = self.h_0 + x * tf.tan(tf.abs(self.slope) * np.pi / 180.0)
        backward = forward[::-1]
        profile = tf.where(x <= 0.5 * self.config.lx, forward, backward)
      elif _TERRAIN_TYPE.value == TerrainType.WEDGE:
        profile = self.h_0 + tf.math.maximum(
            (x - 0.5 * self.config.lx) * tf.tan(self.slope * np.pi / 180.0), 0.0
        )
      else:
        profile = tf.zeros_like(x)

      profile = tf.clip_by_value(
          profile,
          clip_value_max=_FLAT_SURFACE_MAX_HEIGHT.value,
          clip_value_min=0.0,
      )

      elevation = tf.transpose(
          tf.maximum(tf.tile(profile[tf.newaxis, :], [self.config.fy, 1]), 0.0)
      )

    # Add obstacles if requested.
    if INCLUDE_OBSTACLES.value:
      x_locs = [float(val) for val in OBSTACLES_X.value]
      y_locs = [float(val) for val in OBSTACLES_Y.value]
      widths = [float(val) for val in OBSTACLES_W.value]
      heights = [float(val) for val in OBSTACLES_H.value]
      obstacles_map = cubic_obstacles(self.config.lx, self.config.ly,
                                      self.config.fx, self.config.fy, x_locs,
                                      y_locs, widths, heights)
      elevation += tf.convert_to_tensor(obstacles_map)

    self.map_utils = terrain_utils.TerrainUtils(self.config, elevation)

    self.thermodynamics = thermodynamics_manager.thermodynamics_factory(
        self.config)

    # Initializes the inflow library. If simulated turbulent inflow is found in
    # the config file, the SimulatedTurbulentInflow model will be used
    # regardless of the value of the `flat_surface_turbulent_inflow` flag.
    if (self.config.boundary_models is not None and
        self.config.boundary_models.HasField('simulated_inflow')):
      self.inflow = (
          simulated_turbulent_inflow.simulated_turbulent_inflow_factory(
              self.config))
      assert self.inflow is not None
      self.inflow_update_fn = self.inflow.additional_states_update_fn
    elif FLAGS.flat_surface_turbulent_inflow:
      self.inflow = self.fire_utils.inflow_generator_init(0)
      self.inflow_update_fn = self.inflow.generate_inflow_update_fn(
          self.fire_utils.inflow_seed) if self.inflow else None
    else:
      self.inflow = None
      self.inflow_update_fn = None

    logging.info('Inflow type: %r.', type(self.inflow))

    # Initializes the combustion related options.
    self.include_fuel = self.fire_utils.fuel_bed_height > 0.0
    self.include_fire = FLAGS.flat_surface_include_fire
    self.ignite = FLAGS.flat_surface_ignite
    if FLAGS.flat_surface_use_dynamic_igniter:
      ignition_start_point = [
          float(p) for p in FLAGS.flat_surface_ignition_start_point
      ]
      self.igniter = igniter.Igniter(FLAGS.flat_surface_ignition_speed,
                                     ignition_start_point,
                                     FLAGS.flat_surface_ignition_duration,
                                     FLAGS.flat_surface_ignition_start_step_id,
                                     FLAGS.flat_surface_igniter_radius,
                                     self.config.dt)
    else:
      self.igniter = None

    # Initializes the immersed boundary method for terrain if in use.
    self.ib = immersed_boundary_method.immersed_boundary_method_factory(
        self.config)
    if (
        self.config.boundary_models is not None
        and self.config.boundary_models.HasField('ib')
    ):
      self.ib_info = immersed_boundary_method.ib_info_map(
          self.config.boundary_models.ib
      )
    else:
      self.ib_info = {}

    # Helper variables required by the initial condition of velocity.
    self.init_bl = FLAGS.flat_surface_init_bl
    self.blasius_bl_distance = FLAGS.flat_surface_blasius_bl_distance
    self.blasius_bl_transition = FLAGS.flat_surface_blasius_bl_transition
    self.blasius_bl_fraction = FLAGS.flat_surface_blasius_bl_fraction

    # Initializes parameters that handles the location of the fuel distribution
    # in the domain.
    self.fuel_start_x = 0.0
    if (
        self.fire_utils.ignition_option
        == wildfire_utils.IgnitionKernelType.SLANT_LINE
    ):
      # Set the fuel layer starting location to be the trailing edge of ignition
      # kernel if the ignition kernel is a SLANT_LINE perpendicular to the
      # inflow direction.
      if (
          np.abs(self.fire_utils.ignition_line.angle - 90.0)
          < np.finfo(np.float32).resolution
      ):
        self.fuel_start_x = (
            self.fire_utils.ignition_line.center_x
            - 0.5 * self.fire_utils.ignition_line.thickness
        )

    self.dbg = components_debug.ComponentsDebug(self.config)

    self.probe = probe.probe_factory(self.config)

  def _ignite(
      self,
      ignition_kernel: tf.Tensor,
      temperature: tf.Tensor,
  ) -> tf.Tensor:
    """Sets a high temperature region at the specified location."""
    return temperature + (self.fire_utils.ignition_temperature -
                          temperature) * ignition_kernel

  def _rescale_horizontal_velocity(
      self,
      replicas: np.ndarray,
      u: types.FlowFieldVal,
      v: types.FlowFieldVal,
      dims: Sequence[int],
      partition_dims: Optional[Sequence[int]] = None,
  ) -> Tuple[types.FlowFieldVal, types.FlowFieldVal]:
    """Rescales the velocity with inputs from config."""
    # Compute the velocity magnitude in the horizontal direction.
    u_mag = tf.nest.map_structure(
        lambda u_i, v_i: tf.math.sqrt(u_i**2 + v_i**2), u, v
    )

    u_mean = common_ops.global_mean(
        u_mag, replicas, [0] * 3, dims, partition_dims
    )

    # Use the maximum velocity as the free stream velocity in the original
    # inflow profile.
    u_max = common_ops.global_reduce(
        tf.stack(u_mean),
        tf.math.reduce_max,
        common_ops.group_replicas(replicas),
    )

    def u_rescale_preserve_tke_fn(u_new: float):  # pylint: disable=unused-variable
      """Rescales the velocity with turbulence kinetic energy preserved."""

      def rescale_u(u_0, u_mean):
        return u_0 + (tf.math.divide_no_nan(u_new, u_max) - 1.0) * u_mean

      return rescale_u

    def u_rescale_preserve_intensity_fn(u_new: float):
      """Rescales the velocity with turbulence intensity preserved.

      To preserve the turbulence intensity, the ratio between the standard
      deviation of the velocity flucutation and the mean velocity needs stay the
      same. For u' = u - <u>:
      <u_new> = <u> * u_new / <u>_max,
      u'_new = u' * <u_new> / <u> = u' * (<u> * u_new / <u>_max) / <u>
             = u' * u_new / <u>_max,
      u_new = <u_new> + u'_new = (<u> + u') * u_new / <u>_max
            = u * u_new / <u>_max.

      Args:
        u_new: The magnitude of the new mean flow velocity.

      Returns:
        A function that takes the 3D velocity field and computes the rescaled
        velocity.
      """

      def rescale_u(u_0):
        return u_0 * tf.math.divide_no_nan(u_new, u_max)

      return rescale_u

    return (
        tf.nest.map_structure(
            u_rescale_preserve_intensity_fn(self.fire_utils.u_mean), u_mag
        ),
        tf.nest.map_structure(
            u_rescale_preserve_intensity_fn(self.fire_utils.v_mean), u_mag
        ),
    )

  def pre_simulation_update_fn(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Updates the IB boundary weights and the turbulent inflow.

    This function is invoked only once at the step spefified in the commandline
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
      updated `ib_interior_mask` and `ib_boundary`. If simulated turbulent
      inflow is used, `INFLOW_U`, `INFLOW_V`, `u`, `v`, `u_init`, and `v_init`
      will be updated so that the mean flow matches with the value specified in
      the config.
    """
    del params
    output = {}
    output.update(states)
    output.update(additional_states)

    # Rescale the inflow velocity so that the mean velocity profile matches with
    # the input if simulated turbulent inflow is used.
    if isinstance(
        self.inflow, simulated_turbulent_inflow.SimulatedTurbulentInflow
    ):
      # The mean velocity profile for inflow is computed along the time (2) and
      # y (1) dimensions, which are partitioned along dimension 0 and 1,
      # respectively.
      inflow_new = self._rescale_horizontal_velocity(
          replicas,
          additional_states['INFLOW_U'],
          additional_states['INFLOW_V'],
          [1, 2],
          [0, 1],
      )
      vel_new = self._rescale_horizontal_velocity(
          replicas, states['u'], states['v'], [0, 1]
      )
      if 'ib_interior_mask' in additional_states:
        vel_new = [
            tf.nest.map_structure(
                tf.math.multiply, vel, additional_states['ib_interior_mask']
            )
            for vel in vel_new
        ]

      output.update({
          'INFLOW_U': inflow_new[0],
          'INFLOW_V': inflow_new[1],
          'u': vel_new[0],
          'v': vel_new[1],
      })
      if 'u_init' in additional_states:
        output.update({'u_init': vel_new[0]})
      if 'v_init' in additional_states:
        output.update({'v_init': vel_new[1]})

    if (self.config.boundary_models is not None and
        self.config.boundary_models.HasField('ib')):
      if self.config.boundary_models.ib.WhichOneof('type') == 'cartesian_grid':
        logging.info(
            'Adding IB masks initialization in simulation graph for the '
            'Cartesian grid method.'
        )

        boundary_weights = self.map_utils.compute_boundary_weights(
            replica_id, replicas, kernel_op,
            additional_states['ib_interior_mask'],
            additional_states['ib_boundary'])
        output.update({'ib_boundary': boundary_weights})
      elif self.config.boundary_models.ib.WhichOneof('type') == 'sponge':
        logging.info(
            'Adding IB masks initialization in simulation graph for the sponge '
            'method.'
        )

        if 'ib_boundary' in self.config.additional_state_keys:
          output.update({
              'ib_boundary':
                  immersed_boundary_method.get_fluid_solid_interface_z(
                      kernel_op, replica_id, replicas,
                      additional_states['ib_interior_mask'],
                      self.config.halo_width)
          })

    if 'rho_f' in additional_states and _OPT_FILTER_FUEL_DENSITY.value:
      # Update the halos before and after the filtering to make sure fuel
      # at the borders between cores are valid. Note that the actual boundary
      # condition doesn't matter for this purpose.
      halo_update_fn = lambda f: halo_exchange.inplace_halo_exchange(  # pylint: disable=g-long-lambda
          f,
          (0, 1, 2),
          replica_id,
          replicas,
          (0, 1, 2),
          periodic_dims=self.config.periodic_dims,
          boundary_conditions=[[(halo_exchange.BCType.NEUMANN, 0.0)] * 2] * 3,
          width=self.config.halo_width,
      )
      rho_f = filters.filter_op(
          self.config,
          halo_update_fn(additional_states['rho_f']), additional_states,
      )
      output.update({'rho_f': halo_update_fn(rho_f)})

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
    """Ignites the fire.

    This function is invoked only once at the step spefified in the commandline
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
      updated `theta` (or `T`), and `T_s` with elevated temperature in
      locations specified by `ignition_kernel`. `rho`, `u`, `v`, and `w` will
      be rescaled for momentum conservation.
    """

    if (
        self.include_fire
        and self.ignite
        and self.igniter is None
        and self.fire_utils.ignition_with_hot_kernel is not None
    ):
      output = self.fire_utils.ignition_with_hot_kernel(
          kernel_op, replica_id, replicas, states, additional_states, params
      )
    else:
      output = {}

    return output

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
    """Updates `additional_states` for drag forces and fire physics if required.

    Args:
      kernel_op: An object holding a library of kernel operations.
      replica_id: The id of the replica.
      replicas: The replicas. In particular, a numpy array that maps grid
        coordinates to replica id numbers.
      step_id: The index of the current time step.
      states: A keyed dictionary of states that will be updated.
      additional_states: A list of states that are needed by the update fn, but
        will not be updated by the main governing equations.
      params: An instance of `grid_parametrization.GridParametrization`.

    Returns:
      Updated `additional_states`.
    """
    states_updated = {}
    states_updated.update(states)
    additional_states_updated = {}
    additional_states_updated.update(additional_states)

    # Clear source terms computed from the previous step.
    for varname in additional_states_updated:
      if not varname.startswith('src_'):
        continue
      additional_states_updated[varname] = tf.nest.map_structure(
          tf.zeros_like, additional_states_updated
      )

    if self.probe is not None:
      additional_states_updated.update(
          self.probe.additional_states_update_fn(kernel_op, replica_id,
                                                 replicas, step_id, states,
                                                 additional_states, params))

    if self.igniter is not None:
      ignition_kernel = self.igniter.ignition_kernel(
          step_id, additional_states['ignition_kernel'])
      temperature = tf.nest.map_structure(
          self._ignite, ignition_kernel, states[self.fire_utils.t_var]
      )
      t_s = tf.nest.map_structure(
          self._ignite, ignition_kernel, additional_states['T_s']
      )
      states_updated.update({self.fire_utils.t_var: temperature})
      additional_states_updated.update({'T_s': t_s})

    if self.include_fire:
      additional_states_updated.update(
          self.fire_utils.combustion_step_fn(
              kernel_op,
              replica_id,
              replicas,
              step_id,
              states_updated,
              additional_states_updated,
              self.config,
          )
      )

    if self.fire_utils.ignition_with_heat_source is not None:
      t = self.config.dt * tf.cast(step_id, types.TF_DTYPE)
      heat_source = self.fire_utils.ignition_with_heat_source(
          additional_states['ignition_kernel'], t
      )
      src_name = f'src_{self.fire_utils.t_var}'
      additional_states_updated[src_name] = tf.nest.map_structure(
          tf.math.add, additional_states_updated[src_name], heat_source
      )

    if self.inflow_update_fn is not None:
      if isinstance(self.inflow,
                    simulated_turbulent_inflow.SimulatedTurbulentInflow):
        additional_states_updated = dict(self.inflow_update_fn(
            replica_id, replicas, step_id, states_updated,
            additional_states_updated))
      elif isinstance(self.inflow,
                      synthetic_turbulent_inflow.SyntheticTurbulentInflow):
        additional_states_updated.update(
            self.inflow_update_fn(kernel_op, replica_id, replicas,
                                  states_updated, additional_states_updated,
                                  params))

    # Note that similar logic is applied for updating forcing terms due to
    # sponge and IB. Only variables that are included in the config are
    # considered for the source term evaluations. In addition, the forcing terms
    # from these functions are independent from other forcing terms, hence are
    # added to the overall source terms directly.

    if self.fire_utils.use_sponge:
      maybe_sponge_scalar = (self.fire_utils.t_var, 'Y_O')
      sponge_states = {
          varname: states_updated[varname]
          for varname in maybe_sponge_scalar
          if varname in self.fire_utils.all_sponge_vars.keys()
      }
      if sponge_states:
        sponge_additional_states = {
            'sponge_beta': additional_states['sponge_beta'],
        }
        sponge_additional_states.update(
            {
                f'src_{varname}': tf.nest.map_structure(tf.zeros_like, val)
                for varname, val in sponge_states.items()
            }
        )
        # TODO(b/217254717): Move ad hoc functions like this to a wildfire
        # utility library.
        for varname in sponge_states.keys():
          if varname not in self.fire_utils.sponge_target_names:
            continue
          target_name = self.fire_utils.sponge_target_names[varname]
          if target_name in additional_states:
            sponge_additional_states.update(
                {target_name: additional_states[target_name]}
            )
        sponge_additional_states = self.fire_utils.sponge_forcing_update_fn(
            kernel_op,
            replica_id,
            replicas,
            sponge_states,
            sponge_additional_states,
            params,
        )
        for varname in [f'src_{varname}' for varname in sponge_states.keys()]:
          additional_states_updated.update(
              {
                  varname: tf.nest.map_structure(
                      tf.math.add,
                      additional_states_updated[varname],
                      sponge_additional_states[varname],
                  )
              }
          )

    if self.ib is not None:
      maybe_ib_scalar = (self.fire_utils.t_var, 'Y_O')
      ib_states = {
          varname: states_updated[varname]
          for varname in maybe_ib_scalar
          if varname in self.ib_info.keys()
      }
      if ib_states:
        ib_additional_states = {
            'ib_interior_mask': additional_states['ib_interior_mask'],
        }
        ib_additional_states.update(
            {
                f'src_{varname}': tf.nest.map_structure(tf.zeros_like, val)
                for varname, val in ib_states.items()
            }
        )
        if 'ib_boundary' in additional_states:
          ib_additional_states.update(
              {'ib_boundary': additional_states['ib_boundary']}
          )
        ib_additional_states = self.ib.update_additional_states(
            kernel_op, replica_id, replicas, ib_states, ib_additional_states
        )
        for varname in [f'src_{varname}' for varname in ib_states.keys()]:
          additional_states_updated.update(
              {
                  varname: tf.nest.map_structure(
                      tf.math.add,
                      additional_states_updated[varname],
                      ib_additional_states[varname],
                  )
              }
          )

    if self.firebench_states_update_fn is not None:
      additional_states_updated.update(
          self.firebench_states_update_fn(
              replica_id,
              replicas,
              step_id - self.config.start_step,
              states,
              additional_states,
              params,
              self.fire_utils.fuel_density,
          )
      )

    return additional_states_updated

  def source_term_update_fn(self) -> parameters_lib.SourceUpdateFnLib:
    """Generates a library of functions that updates requested source terms."""

    # Define a helper function that adds rho multiplied by a new source term to
    # the existing one.
    a_plus_bx = lambda a, b, x: a + b * x

    def src_uvw_fn(kernel_op, replica_id, replicas, states, additional_states,
                   params):
      """Updates the source terms for u, v, and w."""
      src = {
          'src_u': tf.nest.map_structure(tf.zeros_like, states['u']),
          'src_v': tf.nest.map_structure(tf.zeros_like, states['v']),
          'src_w': tf.nest.map_structure(tf.zeros_like, states['w']),
      }

      if self.include_fuel:
        helper_states = {
            'rho_f': additional_states['rho_f'],
            'src_u': None,
            'src_v': None,
            'src_w': None,
        }
        drag = self.fire_utils.drag_force_fn(kernel_op, replica_id, replicas,
                                             states, helper_states, params)
        src.update({
            'src_u': tf.nest.map_structure(
                tf.math.add, src['src_u'], drag['src_u']
            ),
            'src_v': tf.nest.map_structure(
                tf.math.add, src['src_v'], drag['src_v']
            ),
            'src_w': tf.nest.map_structure(
                tf.math.add, src['src_w'], drag['src_w']
            ),
        })

      if self.fire_utils.use_sponge:
        sponge_states = {key: states[key] for key in ('u', 'v', 'w')}
        sponge_states['rho'] = states['rho']
        helper_states = {
            'sponge_beta': additional_states['sponge_beta'],
            'src_u': tf.nest.map_structure(tf.zeros_like, states['u']),
            'src_v': tf.nest.map_structure(tf.zeros_like, states['v']),
            'src_w': tf.nest.map_structure(tf.zeros_like, states['w']),
        }
        for varname in ('u', 'v', 'w'):
          if varname not in self.fire_utils.sponge_target_names:
            continue
          target_name = self.fire_utils.sponge_target_names[varname]
          if target_name in additional_states:
            helper_states.update({target_name: additional_states[target_name]})
        sponge_force = self.fire_utils.sponge_forcing_update_fn(
            kernel_op,
            replica_id,
            replicas,
            sponge_states,
            helper_states,
            params,
        )
        src.update({
            'src_u': tf.nest.map_structure(
                a_plus_bx, src['src_u'], states['rho'], sponge_force['src_u']
            ),
            'src_v': tf.nest.map_structure(
                a_plus_bx, src['src_v'], states['rho'], sponge_force['src_v']
            ),
            'src_w': tf.nest.map_structure(
                a_plus_bx, src['src_w'], states['rho'], sponge_force['src_w']
            ),
        })

      if self.ib is not None:
        ib_states = {key: states[key] for key in ('u', 'v', 'w')}
        helper_states = {
            'ib_interior_mask': additional_states['ib_interior_mask'],
            'src_u': tf.nest.map_structure(tf.zeros_like, states['u']),
            'src_v': tf.nest.map_structure(tf.zeros_like, states['v']),
            'src_w': tf.nest.map_structure(tf.zeros_like, states['w']),
        }
        if 'ib_boundary' in additional_states:
          helper_states.update(
              {'ib_boundary': additional_states['ib_boundary']}
          )
        ib_force = self.ib.update_additional_states(
            kernel_op, replica_id, replicas, ib_states, helper_states
        )
        src.update({
            'src_u': tf.nest.map_structure(
                a_plus_bx, src['src_u'], states['rho'], ib_force['src_u']
            ),
            'src_v': tf.nest.map_structure(
                a_plus_bx, src['src_v'], states['rho'], ib_force['src_v']
            ),
            'src_w': tf.nest.map_structure(
                a_plus_bx, src['src_w'], states['rho'], ib_force['src_w']
            ),
        })

      if self._include_coriolis_force:
        helper_states = {
            f'src_{key}': tf.nest.map_structure(tf.zeros_like, states[key])
            for key in ('u', 'v', 'w')
        }
        coriolis_force = self.coriolis_force_fn(kernel_op, replica_id, replicas,
                                                states, helper_states, params)
        src.update({  # pylint: disable=g-complex-comprehension
            src_name: tf.nest.map_structure(
                lambda src_i, rho_i, force_i: src_i + rho_i * force_i,
                src[src_name], states['rho'], coriolis_force[src_name])
            for src_name in ('src_u', 'src_v', 'src_w')
        })

      return src

    def src_u_fn(kernel_op, replica_id, replicas, states, additional_states,
                 params):
      """Updates the source term for u."""
      src_uvw = src_uvw_fn(kernel_op, replica_id, replicas, states,
                           additional_states, params)
      return {'src_u': src_uvw['src_u']}

    def src_v_fn(kernel_op, replica_id, replicas, states, additional_states,
                 params):
      """Updates the source term for v."""
      src_uvw = src_uvw_fn(kernel_op, replica_id, replicas, states,
                           additional_states, params)
      return {'src_v': src_uvw['src_v']}

    def src_w_fn(kernel_op, replica_id, replicas, states, additional_states,
                 params):
      """Updates the source term for w."""
      src_uvw = src_uvw_fn(kernel_op, replica_id, replicas, states,
                           additional_states, params)
      return {'src_w': src_uvw['src_w']}

    return {'u': src_u_fn, 'v': src_v_fn, 'w': src_w_fn}

  def initialization(
      self,
      replica_id: tf.Tensor,
      coordinates: initializer.ThreeIntTuple,
  ) -> types.FlowFieldMap:
    """Initializes states for the simulation.

    Args:
      replica_id: The ID number of the replica.
      coordinates: A tuple that specifies the replica's grid coordinates in
        physical space.

    Returns:
      A dictionary of states and values that are stored as string and 3D tensor
      pairs.
    """

    def init_y_o(xx, yy, zz, lx, ly, lz, coord):
      """Generates initial `Y_O` field."""
      del xx, yy, lx, ly, lz, coord
      return self.fire_utils.y_o_init * tf.ones_like(zz)

    def init_rho_m(xx, yy, zz, lx, ly, lz, coord):
      """Generates initial moisture `rho_m` field."""
      del xx, yy, lx, ly, lz, coord
      # In case of stretched grid, we use the first grid spacing as reference
      # here.
      return tf.compat.v1.where(
          tf.math.logical_and(
              zz <= ground_elevation + self.fire_utils.fuel_bed_height,
              zz >= ground_elevation - self.config.z[1],
          ),
          self.fire_utils.moisture_density * tf.ones_like(zz),
          tf.zeros_like(zz),
      )

    def init_fuel_height_mask(xx, yy, zz, lx, ly, lz, coord):
      """Generates a binary field with 0 indicating fuel and 1 no fuel."""
      del xx, yy, lx, ly, lz, coord
      return tf.compat.v1.where(
          zz <= fuel_top_elevation, tf.zeros_like(zz), tf.ones_like(zz)
      )

    def init_ignition_kernel(xx, yy, zz, lx, ly, lz, coord):
      """Generates the ignition kernel."""
      ignition_shape_fn = self.fire_utils.ignition_kernel_init_fn(
          fuel_top_elevation, ground_elevation
      )

      ignition_kernel_fn = (
          ignition_shape_fn
          if self.igniter is None
          else self.igniter.ignition_schedule_init_fn(ignition_shape_fn)
      )
      return ignition_kernel_fn(xx, yy, zz, lx, ly, lz, coord)

    fuel_top_elevation = self.map_utils.local_elevation_map(
        coordinates, self.fire_utils.fuel_bed_height)
    ground_elevation = self.map_utils.local_elevation_map(coordinates)

    if self.init_fn_uvw is None:
      if self.init_bl:
        self.init_fn_uvw = self.map_utils.blasius_uvw_init_fn(
            self.fire_utils.u_init, self.fire_utils.v_init, self.config.nu,
            self.config.dx, self.config.dy, self.config.lz, self.config.fz,
            self.blasius_bl_distance, self.blasius_bl_transition,
            self.blasius_bl_fraction, coordinates)
      else:
        self.init_fn_uvw = {
            'u':
                velocity_bulk_init_fn(self.fire_utils.u_init, ground_elevation),
            'v':
                velocity_bulk_init_fn(self.fire_utils.v_init, ground_elevation),
            'w':
                velocity_bulk_init_fn(0.0, ground_elevation),
        }

    if self.perturb_init_velocity_rand_seed is not None:
      seed = {
          'u': geophysical_flow_utils.U_SEED,
          'v': geophysical_flow_utils.V_SEED,
          'w': geophysical_flow_utils.W_SEED,
      }
      mean_velocity = {
          'u': self.fire_utils.u_init,
          'v': self.fire_utils.v_init,
          'w': 0.0,
      }
      rms_velocity = {
          'u': self.fire_utils.u_rms,
          'v': self.fire_utils.v_rms,
          'w': self.fire_utils.w_rms,
      }
      halo_width = self.config.halo_width
      self.init_fn_uvw = {  # pylint: disable=g-complex-comprehension
          key: geophysical_flow_utils.perturbed_constant_init_fn(
              seed=seed[key] + self.perturb_init_velocity_rand_seed,
              mean=mean_velocity[key],
              g_dim=2,
              local_grid_no_halos=(
                  self.config.nx - 2 * halo_width,
                  self.config.ny - 2 * halo_width,
                  self.config.nz - 2 * halo_width,
              ),
              rms=rms_velocity[key],
              mean_init_fn=self.init_fn_uvw[key],
              cloud_base=self.config.lz) for key in ('u', 'v', 'w')
      }

    output = {
        'replica_id':
            replica_id,
        'u':
            self.fire_utils.states_init(coordinates, self.init_fn_uvw['u']),
        'v':
            self.fire_utils.states_init(coordinates, self.init_fn_uvw['v']),
        'w':
            self.fire_utils.states_init(coordinates, self.init_fn_uvw['w']),
        'p':
            self.fire_utils.states_init(coordinates,
                                        self.fire_utils.init_fn_zeros),
    }

    if self.fire_utils.t_var in self.config.transport_scalars_names:
      output.update({
          self.fire_utils.t_var:
              self.fire_utils.states_init(coordinates,
                                          self.fire_utils.init_fn_t)
      })

    if 'Y_O' in self.config.transport_scalars_names:
      output.update({
          'Y_O': self.fire_utils.states_init(coordinates, init_y_o),
      })

    if self.fire_utils.t_var in self.config.transport_scalars_names:
      thermo_states = {self.fire_utils.t_var: output[self.fire_utils.t_var]}
      if 'Y_O' in self.config.transport_scalars_names:
        thermo_states.update({'Y_O': output['Y_O']})
      thermo_additional_states = {}
      output.update({
          'rho': self.thermodynamics.update_thermal_density(
              thermo_states, thermo_additional_states
          )
      })
    else:
      output.update({
          'rho':
              self.config.rho * self.fire_utils.states_init(
                  coordinates, self.fire_utils.init_fn_ones)
      })

    if self.config.use_sgs:
      output.update(
          common_ops.gen_field('nu_t', self.config.nx, self.config.ny,
                               self.config.nz)
      )

    if self.ib is not None:
      ib_boundary_fn = (
          immersed_boundary_method.interp_1d_coeff_init_fn(  # pylint: disable=g-long-ternary
              self.map_utils.elevation_map,
              self.config.g_dim,
              (self.config.cx, self.config.cy, self.config.cz),
          )
          if self.ib.type
          in ('direct_forcing_1d_interp', 'feedback_force_1d_interp')
          else self.map_utils.ib_boundary_mask_fn(coordinates)
      )
      output.update(
          self.ib.generate_initial_states(
              coordinates,
              self.map_utils.ib_flow_field_mask_fn(coordinates),
              ib_boundary_fn,
          )
      )
      # Add an IB boundary field for the support of other modules in case it is
      # not used by the particular IB method per se.
      if 'ib_boundary' in self.config.additional_state_keys:
        output.update(
            {
                'ib_boundary': self.fire_utils.states_init(
                    coordinates, ib_boundary_fn
                )
            }
        )

      # Update the initial velocity so that it is 0 inside the solid.
      output.update(
          {
              key: output[key] * output['ib_interior_mask']
              for key in ('u', 'v', 'w')
          }
      )
      for key in ('src_u', 'src_v', 'src_w'):
        if key in output:
          output.pop(key)

    if self.include_fuel:
      assert (
          'rho_f' in self.config.additional_state_keys
      ), 'Fuel height is none zero but rho_f is not included in the config.'

      # In case of stretched grid, we assign the same fuel density at all node
      # points below the fuel height.
      if self.config.use_stretched_grid[2]:

        def init_rho_f(xx, yy, zz, lx, ly, lz, coord):
          """Generates initial fuel density `rho_f` field."""
          del yy, lx, ly, lz, coord
          rho_f = tf.where(
              tf.math.logical_and(
                  zz <= ground_elevation + self.fire_utils.fuel_bed_height,
                  zz >= ground_elevation - self.config.z[1],
              ),
              self.fire_utils.fuel_density * tf.ones_like(zz),
              tf.zeros_like(zz),
          )
          return tf.where(
              tf.greater_equal(xx, self.fuel_start_x), rho_f, tf.zeros_like(xx)
          )

      else:
        init_rho_f = get_init_rho_f(
            ground_elevation,
            self.fire_utils.fuel_bed_height,
            self.fire_utils.fuel_density,
            self.fuel_start_x,
            self.config.dz,
        )
      output.update({
          'rho_f':
              self.fire_utils.states_init(coordinates, init_rho_f, 'CONSTANT'),
      })

    if self.include_fire:
      if 'rho_f_init' in self.config.additional_state_keys:
        output.update({'rho_f_init': output['rho_f']})

      if (self.fire_utils.t_var not in self.config.transport_scalars_names or
          'Y_O' not in self.config.transport_scalars_names):
        raise ValueError(
            '`{}` and `Y_O` need to be included in the config file to consider '
            'fire modeling.'.format(self.fire_utils.t_var)
        )
      output.update({
          'T_s':
              self.fire_utils.states_init(coordinates,
                                          self.fire_utils.init_fn_t),
          'src_rho':
              self.fire_utils.states_init(coordinates,
                                          self.fire_utils.init_fn_zeros),
          'src_{}'.format(self.fire_utils.t_var):
              self.fire_utils.states_init(coordinates,
                                          self.fire_utils.init_fn_zeros),
          'src_Y_O':
              self.fire_utils.states_init(coordinates,
                                          self.fire_utils.init_fn_zeros),
          'nu_t':
              self.fire_utils.states_init(coordinates,
                                          self.fire_utils.init_fn_zeros),
          'tke':
              self.fire_utils.states_init(coordinates,
                                          self.fire_utils.init_fn_zeros),
          'ignition_kernel':
              self.fire_utils.states_init(coordinates, init_ignition_kernel),
      })
      if self.ib is not None:
        output.update({
            'ignition_kernel':
                output['ignition_kernel'] * output['ib_interior_mask']
        })

      # Add additional states required if moisture is considered in the
      # vegetation.
      assert (
          combustion := self.config.combustion
      ) is not None, 'Combustion must be set in the config.'
      if combustion.wood.WhichOneof('combustion_model_option') == 'moist_wood':
        output.update({
            'rho_m':
                self.fire_utils.states_init(coordinates, init_rho_m,
                                            'CONSTANT'),
            'phi_w':
                self.fire_utils.states_init(coordinates,
                                            self.fire_utils.init_fn_zeros),
        })

    if isinstance(self.inflow,
                  simulated_turbulent_inflow.SimulatedTurbulentInflow):
      output.update(self.inflow.initialize_inflow())
    elif isinstance(self.inflow,
                    synthetic_turbulent_inflow.SyntheticTurbulentInflow):
      halo_width = self.fire_utils.config.halo_width
      fuel_mask_3d = self.fire_utils.states_init(coordinates,
                                                 init_fuel_height_mask)

      def cut_yz_plane(f):
        """Extracts the yz plane from a 3D tensor."""
        return tf.transpose(
            f[halo_width:halo_width + self.config.core_nz,
              halo_width, halo_width:halo_width + self.config.core_ny])

      fuel_mask_yz = cut_yz_plane(fuel_mask_3d)

      output.update(
          self.fire_utils.inflow_states_init(self.inflow, 0, fuel_mask_yz))
      # Overrides the mean profiles with the analytical profile from the initial
      # condition.
      output.update({
          'bc_u_0_0':
              output['u'][:, :self.config.halo_width + 1, :],
          'bc_v_0_0':
              output['v'][:, :self.config.halo_width + 1, :],
          'bc_w_0_0':
              output['w'][:, :self.config.halo_width + 1, :],
          'mean_u_0_0': cut_yz_plane(output['u']),
          'mean_v_0_0': cut_yz_plane(output['v']),
          'mean_w_0_0': cut_yz_plane(output['w']),
      })
    elif not self.config.periodic_dims[0]:
      # Use the laminar profile that is the same as the initial condition as the
      # inflow boundary condition if no inflow model is used.
      output.update({
          'bc_u_0_0':
              output['u'][:, :self.config.halo_width + 1, :],
          'bc_v_0_0':
              output['v'][:, :self.config.halo_width + 1, :],
          'bc_w_0_0':
              output['w'][:, :self.config.halo_width + 1, :],
      })

    if self.fire_utils.use_sponge:
      output.update(self.fire_utils.sponge_init(coordinates))
      # TODO(bcg): Move sponge initialization code to rayleigh_damping_layer.py.
      # We have similar code elsewhere, too.
      for sponge in self.config.sponge:
        for variable in sponge.variable_info:
          # Add the target state if requested.
          if variable.WhichOneof('target') == 'target_state_name':
            target_name = variable.target_state_name
            if target_name not in self.config.additional_state_keys:
              raise ValueError(
                  f'Target state {target_name} is requested for sponge'
                  f' {variable.name} but is not provided.'
              )
            if target_name == f'{variable.name}_init':
              logging.info(
                  'Enforcing the initial condition as sponge for %s.',
                  variable.name,
              )
              output.update({target_name: output[variable.name]})
            elif target_name == f'{variable.name}_log':
              if variable.name not in ('u', 'v', 'w'):
                raise NotImplementedError(
                    'Log-profiled sponge is implemented for u, v, and w only.'
                    f' {variable.name} is requested.'
                )
              logging.info(
                  'Enforcing a terrain following log profile as sponge for %s.',
                  variable.name,
              )
              z_0 = 0.15
              sponge_fn = init_fn_lib.logarithmic_boundary_layer(
                  self.fire_utils.u_mean,
                  self.fire_utils.v_mean,
                  z_0,
                  self.map_utils.local_elevation_map(coordinates),
              )[variable.name]
              output.update(
                  {
                      target_name: self.fire_utils.states_init(
                          coordinates, sponge_fn
                      )
                  }
              )
            else:
              raise NotImplementedError(
                  f'{target_name} is not a supported target sponge type.'
              )
            self.fire_utils.sponge_target_names[variable.name] = target_name

          if variable.name in ('u', 'v', 'w'):
            continue
          output.update({
              'src_{}'.format(variable.name):
                  self.fire_utils.states_init(coordinates,
                                              self.fire_utils.init_fn_zeros)
          })

    if self.fire_utils.use_geo:
      output.update({
          'zz':
              self.fire_utils.states_init(
                  coordinates, lambda xx, yy, zz, lx, ly, lz, coord: zz)
      })

    if self.config.dbg:
      output.update(self.dbg.generate_initial_states())

    if self.probe is not None:
      output.update(self.probe.initialization(replica_id, coordinates))

    if self.firebench_initialization is not None:
      output.update(self.firebench_initialization(
          self.config.num_cycles * self.config.num_steps))

    return output
