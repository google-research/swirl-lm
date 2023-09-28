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

"""A library of utilities that are shared for wildfire simulations."""

import collections
import enum
from typing import Callable, Mapping, Optional

from absl import flags
from absl import logging
import numpy as np
from swirl_lm.base import initializer
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.boundary_condition import rayleigh_damping_layer
from swirl_lm.boundary_condition import synthetic_turbulent_inflow
from swirl_lm.physics.combustion import combustion
from swirl_lm.utility import composite_types
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf

# The universal gas constant, unit J/K/mol.
# Reference: United States Committee on Extension to the Standard Atmosphere,
# "U.S. Standard Atmosphere, 1976", National Oceanic and Atmospheric
# Administration, National Aeronautics and Space Administration, United States
# Air Force, Washington D.C., 1976.
R_UNIVERSAL = 8.3145
# The threshold below which the fuel is treated as depleted.
EPSILON = 1e-6
# The Tensorflow data type.
_TF_DTYPE = tf.float32

FlowFieldMap = types.FlowFieldMap
InitFn = initializer.ValueFunction
ThreeIntTuple = initializer.ThreeIntTuple

TemperatureInitFn = Callable[..., InitFn]


class IgnitionKernelType(enum.Enum):
  """The type of the ignition kernel to initiate the fire."""
  # The ignition kernel is a box defined by the location of its edges. This
  # kernel coincides with the fuel.
  BOX = 0
  # The ignition kernel is a sphere, which is defined by the center and radius.
  # The edge of the sphere is smoothed out by a tanh function.
  SPHERE = 1
  # The ignition kernel is a line that's not aligned with the axis orientation.
  # This kernel is defined based on its center, length, width, and angle with
  # respect to the x axis.
  SLANT_LINE = 2
  # This option constructs multiple slant line ignition kernels with random
  # orientations.
  MULTI_LINE = 3


class VelocityInitAndInflowOption(enum.Enum):
  """Options for the velocity initialization and inflow boundary condition."""
  # Get the velocity initial and inflow conditions from `boundary_condition`
  # defined in the config file.
  FROM_CONFIG = 'from_config'
  # Get the velocity initial and inflow conditions from flags `u_init`, `v_init`
  # and `w_init`.
  FROM_COMPONENT_FLAGS = 'from_component_flags'
  # Get the velocity initial and inflow conditions from flags `wind_speed` and
  # `wind_angle`.
  FROM_SPEED_ANGLE_FLAGS = 'from_speed_angle_flags'


# A slant line defined by its angle with respect to the x axis.
SlantLine = collections.namedtuple(
    'SlaneLine', ['center_x', 'center_y', 'length', 'thickness', 'angle'])

# TODO(b/145002624): Replace flags with configs.
_FUEL_DENSITY = flags.DEFINE_float(
    'fuel_density', 1.0, 'The density of the fuel.', allow_override=True
)
_FUEL_BED_HEIGHT = flags.DEFINE_float(
    'fuel_bed_height', 1.0, 'The height of the fuel bed.', allow_override=True
)
_MOISTURE_DENSITY = flags.DEFINE_float(
    'moisture_density',
    0.01,
    'The bulk density of the moisture in a unit volume, in units of kg/m^3.',
    allow_override=True,
)
_C_D = flags.DEFINE_float(
    'c_d', 1.0, 'The drag coefficient of the vegetation.', allow_override=True
)
# Initial velocity.
_VELOCITY_INIT_INFLOW_OPT = flags.DEFINE_enum_class(
    'velocity_init_inflow_opt',
    VelocityInitAndInflowOption.FROM_CONFIG,
    VelocityInitAndInflowOption,
    (
        'The option for velocity initialization and inflow boundary condition.'
        ' Available options are `FROM_CONFIG`, `FROM_COMPONENT_FLAGS`,'
        ' `FROM_SPEED_ANGLE_FLAGS`.'
    ),
    allow_override=True,
)
_WIND_SPEED = flags.DEFINE_float(
    'wind_speed', 1.0, 'The speed of the wind in m/s.', allow_override=True
)
_WIND_ANGLE = flags.DEFINE_float(
    'wind_angle',
    0.0,
    'The angle of the wind with respect to the x axis in degrees.',
    allow_override=True,
)
_U_INIT = flags.DEFINE_float(
    'u_init',
    1.0,
    'The mean inflow and initial velocity in the streamwise direction.',
    allow_override=True,
)
_V_INIT = flags.DEFINE_float(
    'v_init',
    1.0,
    'The mean inflow and initial velocity in the lateral direction.',
    allow_override=True,
)
_U_RMS = flags.DEFINE_float(
    'u_rms',
    0.1,
    'The root mean square of inflow velocity in the streamwise direction.',
    allow_override=True,
)
_V_RMS = flags.DEFINE_float(
    'v_rms',
    1.0,
    'The root mean square of inflow velocity in the lateral direction.',
    allow_override=True,
)
_W_RMS = flags.DEFINE_float(
    'w_rms',
    1.0,
    'The root mean square of inflow velocity in the vertical direction.',
    allow_override=True,
)
_T_INIT = flags.DEFINE_float(
    't_init', 300.0, 'The initial temperature of the air.', allow_override=True
)
_Y_O_INIT = flags.DEFINE_float(
    'y_o_init', 1.0, 'The initial oxidizer mass fraction.', allow_override=True
)
_IGNITION_OPTION = flags.DEFINE_enum_class(
    'ignition_option',
    IgnitionKernelType.BOX,
    IgnitionKernelType,
    (
        'The type of the ignition kernel. Should be one of the following: '
        'BOX, SPHERE, SLANT_LINE, MULTI_LINE.'
    ),
    allow_override=True,
)
_IGNITION_X_LOW = flags.DEFINE_float(
    'ignition_x_low',
    0.0,
    (
        'The lower end of the ignition kernel in the steamwise direction.'
        'Use only when `ignition_option` is `IgnitionKernelType.BOX`.'
    ),
    allow_override=True,
)
_IGNITION_X_HIGH = flags.DEFINE_float(
    'ignition_x_high',
    0.0,
    (
        'The higher end of the ignition kernel in the steamwise direction.'
        'Use only when `ignition_option` is `IgnitionKernelType.BOX`.'
    ),
    allow_override=True,
)
_IGNITION_Y_LOW = flags.DEFINE_float(
    'ignition_y_low',
    0.0,
    (
        'The lower end of the ignition kernel in the lateral direction.'
        'Use only when `ignition_option` is `IgnitionKernelType.BOX`.'
    ),
    allow_override=True,
)
_IGNITION_Y_HIGH = flags.DEFINE_float(
    'ignition_y_high',
    0.0,
    (
        'The higher end of the ignition kernel in the lateral direction.'
        'Use only when `ignition_option` is `IgnitionKernelType.BOX`.'
    ),
    allow_override=True,
)
_IGNITION_CENTER_X = flags.DEFINE_float(
    'ignition_center_x',
    0.0,
    (
        'The x coordinate of the center of the ignition kernel. '
        'Used only when `ignition_option` is `IgnitionKernelType.SPHERE` or '
        '`IgnitionKernelType.SLANT_LINE`.'
    ),
    allow_override=True,
)
_IGNITION_CENTER_Y = flags.DEFINE_float(
    'ignition_center_y',
    0.0,
    (
        'The y coordinate of the center of the ignition kernel. '
        'Used only when `ignition_option` is `IgnitionKernelType.SPHERE` or '
        '`IgnitionKernelType.SLANT_LINE`.'
    ),
    allow_override=True,
)
_IGNITION_CENTER_Z = flags.DEFINE_float(
    'ignition_center_z',
    0.0,
    (
        'The z coordinate of the center of the ignition kernel. '
        'Used only when `ignition_option` is `IgnitionKernelType.SPHERE`.'
    ),
    allow_override=True,
)
_IGNITION_RADIUS = flags.DEFINE_float(
    'ignition_radius',
    0.0,
    (
        'The radius of the ignition kernel. Used only when `ignition_option` is'
        ' `IgnitionKernelType.SPHERE`.'
    ),
    allow_override=True,
)
_IGNITION_SCALE = flags.DEFINE_float(
    'ignition_scale',
    1.0,
    (
        'The smoothness factor at the edge of the ignition kernel. '
        'Used only when `ignition_option` is `IgnitionKernelType.SPHERE`.'
    ),
    allow_override=True,
)
# For the slant line ignition kernel.
_IGNITION_LINE_LENGTH = flags.DEFINE_float(
    'ignition_line_length',
    0.0,
    (
        'The length of the ignition line. Used only when `ignition_option` is '
        '`IgnitionKernelType.SLANT_LINE`.'
    ),
    allow_override=True,
)
_IGNITION_LINE_THICKNESS = flags.DEFINE_float(
    'ignition_line_thickness',
    0.0,
    (
        'The thickness of the ignition line. Used only when `ignition_option`'
        ' is `IgnitionKernelType.SLANT_LINE`.'
    ),
    allow_override=True,
)
_IGNITION_LINE_ANGLE = flags.DEFINE_float(
    'ignition_line_angle',
    90.0,
    (
        'The angle of the ignition line with respect to the x axis in degree.'
        ' Used only when `ignition_option` is `IgnitionKernelType.SLANT_LINE`.'
    ),
    allow_override=True,
)
# For double-line ignition kernels.
_IGNITION_CENTERS_X = flags.DEFINE_list(
    'ignition_centers_x',
    [],
    (
        'The x coordinates of the centers of ignition kernels. Used only when '
        '`ignition_option` is `IgnitionKernelType.MULTI_LINE`.'
    ),
    allow_override=True,
)
_IGNITION_CENTERS_Y = flags.DEFINE_list(
    'ignition_centers_y',
    [],
    (
        'The y coordinates of the centers of ignition kernels. Used only when '
        '`ignition_option` is `IgnitionKernelType.MULTI_LINE`.'
    ),
    allow_override=True,
)
_IGNITION_LINES_LENGTH = flags.DEFINE_list(
    'ignition_lines_length',
    [],
    (
        'The lengths of ignition lines. Used only when `ignition_option` is '
        '`IgnitionKernelType.MULTI_LINE`.'
    ),
    allow_override=True,
)
_IGNITION_LINES_THICKNESS = flags.DEFINE_list(
    'ignition_lines_thickness',
    [],
    (
        'The thicknesses of ignition lines. Used only when `ignition_option` is'
        ' `IgnitionKernelType.MULTI_LINE`.'
    ),
    allow_override=True,
)
_IGNITION_LINES_ANGLE = flags.DEFINE_list(
    'ignition_lines_angle',
    [],
    (
        'The angles of ignition lines with respect to the x axis in degree.'
        ' Used only when `ignition_option` is `IgnitionKernelType.MULTI_LINE`.'
    ),
    allow_override=True,
)
_IGNITION_TEMPERATURE = flags.DEFINE_float(
    'ignition_temperature',
    800.0,
    'The temperature of the ignition kernel.',
    allow_override=True,
)
# Flags for synthetic turbulent inflow.
_INFLOW_SEED = flags.DEFINE_list(
    'turbulent_inflow_seed',
    default=[],
    help='The seed used to generate the random numbers when computing the '
    'turbulent inflow profile. Should be either empty or has length 2. If '
    'empty, the seed will be generated from a Normal distribution.')
_INFLOW_X_LX = flags.DEFINE_float(
    'inflow_x_lx',
    10.0,
    (
        'The characteristic length scale in dimension 0 for inflow turbulence '
        'generation in dimension 0 with units m/s.'
    ),
    allow_override=True,
)
_INFLOW_X_LY = flags.DEFINE_float(
    'inflow_x_ly',
    10.0,
    (
        'The characteristic length scale in dimension 1 for inflow turbulence '
        'generation in dimension 0 with units m/s.'
    ),
    allow_override=True,
)
_INFLOW_X_LZ = flags.DEFINE_float(
    'inflow_x_lz',
    10.0,
    (
        'The characteristic length scale in dimension 2 for inflow turbulence '
        'generation in dimension 0 with units m/s.'
    ),
    allow_override=True,
)
_INFLOW_Y_LX = flags.DEFINE_float(
    'inflow_y_lx',
    10.0,
    (
        'The characteristic length scale in dimension 0 for inflow turbulence '
        'generation in dimension 1 with units m/s.'
    ),
    allow_override=True,
)
_INFLOW_Y_LY = flags.DEFINE_float(
    'inflow_y_ly',
    10.0,
    (
        'The characteristic length scale in dimension 1 for inflow turbulence '
        'generation in dimension 1 with units m/s.'
    ),
    allow_override=True,
)
_INFLOW_Y_LZ = flags.DEFINE_float(
    'inflow_y_lz',
    10.0,
    (
        'The characteristic length scale in dimension 2 for inflow turbulence '
        'generation in dimension 1 with units m/s.'
    ),
    allow_override=True,
)
# Flags for applying height-dependent geopotential in the simulation.
_USE_GEOPOTENTIAL = flags.DEFINE_bool(
    'use_geopotential',
    False, 'The option of whether to use height-dependent geopotential for the '
    'reference states.',
    allow_override=True)


class WildfireUtils():
  """A library of utilities that are useful for wildfire simulations."""

  def __init__(
      self,
      config: Optional[parameters_lib.SwirlLMParameters] = None,
      temperature_init_fn: Optional[TemperatureInitFn] = None,
  ):
    """Initializes parameters from flags."""
    # pylint: disable=g-long-ternary
    self.config = config if config is not None else (
        parameters_lib.params_from_config_file_flag())
    # pylint: enable=g-long-ternary

    if (self.config.combustion is None or
        not self.config.combustion.HasField('wood')):
      raise ValueError('Wood model is not defined as a combustion model.')

    # The update function for reactive variables and their source terms.
    self.combustion_step_fn = combustion.combustion_step

    # Option of applying height-dependent geopotential for reference states.
    self.use_geo = _USE_GEOPOTENTIAL.value
    if self.use_geo and 'zz' not in self.config.additional_state_keys:
      raise ValueError('`zz` has to be included as an additional state to '
                       'represent the vertical coordinates when the height-'
                       'dependent geopotential is used.')
    self.t_var = 'theta' if self.use_geo else 'T'

    # The update function for igntion.
    self.ignition_step_fn = combustion.ignition_step_fn(
        self.t_var, _IGNITION_TEMPERATURE.value, self.config
    )

    # Parameters for the initial conditions is set to the same as the inflow
    # boundary condition. Note that this assumes that the inflow is on the lower
    # index face of the x direction.
    velocity_init_opt = _VELOCITY_INIT_INFLOW_OPT.value
    if (
        velocity_init_opt == VelocityInitAndInflowOption.FROM_CONFIG
        and self.config.periodic_dims[0]
    ):
      # Set the velocity init and inflow option to `FROM_COMPONENT_FLAGS` when
      # the inflow direction is periodic for backward compatibility.
      logging.warning(
          (
              'Velocity init and inflow option changed from'
              ' %s to FROM_COMPONENT_FLAGS because the'
              ' inflow direction is periodic.'
          ),
          velocity_init_opt.name,
      )
      velocity_init_opt = VelocityInitAndInflowOption.FROM_COMPONENT_FLAGS

    if velocity_init_opt == VelocityInitAndInflowOption.FROM_CONFIG:
      self.u_init = self.config.bc['u'][0][0][1]
      self.v_init = self.config.bc['v'][0][0][1]
    elif velocity_init_opt == VelocityInitAndInflowOption.FROM_COMPONENT_FLAGS:
      self.u_init = _U_INIT.value
      self.v_init = _V_INIT.value
    elif (
        velocity_init_opt == VelocityInitAndInflowOption.FROM_SPEED_ANGLE_FLAGS
    ):
      self.u_init = _WIND_SPEED.value * np.cos(
          _WIND_ANGLE.value * np.pi / 180.0
      )
      self.v_init = _WIND_SPEED.value * np.sin(
          _WIND_ANGLE.value * np.pi / 180.0
      )
    else:
      raise ValueError(
          f'{velocity_init_opt} is not a valid option for velocity'
          ' initialization. Available options are'
          f' {[v.name for v in VelocityInitAndInflowOption]}'
      )
    self.u_init = np.float32(self.u_init)
    self.v_init = np.float32(self.v_init)

    self.t_init = (
        self.config.bc[self.t_var][0][0][1]
        if not self.config.periodic_dims[0] and self.t_var in self.config.bc
        else _T_INIT.value
    )
    self.y_o_init = (
        self.config.bc['Y_O'][0][0][1]
        if not self.config.periodic_dims[0] and 'Y_O' in self.config.bc
        else _Y_O_INIT.value
    )

    # Parameters for the drag force.
    self.drag_force_fn = self.vegetation_drag_update_fn(
        _C_D.value, self.config.combustion.wood.a_v
    )

    # Parameters for the inflow.
    self.u_mean = self.u_init
    self.v_mean = self.v_init
    self.u_rms = _U_RMS.value
    self.v_rms = _V_RMS.value
    self.w_rms = _W_RMS.value

    self.inflow_seed = (None if not _INFLOW_SEED.value else
                        tuple([int(s) for s in _INFLOW_SEED.value]))
    self.inflow_length_scales_x = (
        _INFLOW_X_LX.value,
        _INFLOW_X_LY.value,
        _INFLOW_X_LZ.value,
    )
    self.inflow_length_scales_y = (
        _INFLOW_Y_LX.value,
        _INFLOW_Y_LY.value,
        _INFLOW_Y_LZ.value,
    )
    self.delta = (self.config.dx, self.config.dy, self.config.dz)
    self.mesh_size = (self.config.core_nx, self.config.core_ny,
                      self.config.core_nz)

    # Parameters for the vegetation and ignition kernel.
    self.fuel_density = _FUEL_DENSITY.value
    self.fuel_bed_height = np.maximum(_FUEL_BED_HEIGHT.value, 0.0)
    self.moisture_density = _MOISTURE_DENSITY.value

    self.ignition_option = _IGNITION_OPTION.value
    if self.ignition_option == IgnitionKernelType.BOX:
      self.ignition_x_low = _IGNITION_X_LOW.value
      self.ignition_x_high = _IGNITION_X_HIGH.value
      self.ignition_y_low = _IGNITION_Y_LOW.value
      self.ignition_y_high = _IGNITION_Y_HIGH.value
    elif self.ignition_option == IgnitionKernelType.SPHERE:
      self.ignition_center_x = _IGNITION_CENTER_X.value
      self.ignition_center_y = _IGNITION_CENTER_Y.value
      self.ignition_center_z = _IGNITION_CENTER_Z.value
      self.ignition_radius = _IGNITION_RADIUS.value
      self.ignition_scale = _IGNITION_SCALE.value
    elif self.ignition_option == IgnitionKernelType.SLANT_LINE:
      self.ignition_line = SlantLine(
          center_x=_IGNITION_CENTER_X.value,
          center_y=_IGNITION_CENTER_Y.value,
          length=_IGNITION_LINE_LENGTH.value,
          thickness=_IGNITION_LINE_THICKNESS.value,
          angle=_IGNITION_LINE_ANGLE.value,
      )
    elif self.ignition_option == IgnitionKernelType.MULTI_LINE:
      # Assumes lists for all requested fields have the same length.
      self.ignition_lines = [
          SlantLine(  # pylint: disable=g-complex-comprehension
              center_x=float(_IGNITION_CENTERS_X.value[i]),
              center_y=float(_IGNITION_CENTERS_Y.value[i]),
              length=float(_IGNITION_LINES_LENGTH.value[i]),
              thickness=float(_IGNITION_LINES_THICKNESS.value[i]),
              angle=float(_IGNITION_LINES_ANGLE.value[i]),
          )
          for i in range(len(_IGNITION_CENTERS_X.value))
      ]

    self.ignition_temperature = _IGNITION_TEMPERATURE.value

    # Parameters for the sponge layer.
    self.use_sponge = bool(self.config.sponge)
    if self.use_sponge:
      self.sponge = rayleigh_damping_layer.RayleighDampingLayer(
          self.config.sponge
      )
      self.all_sponge_vars = rayleigh_damping_layer.sponge_info_map(
          self.config.sponge
      )
      self.sponge_target_names = {}

    # Get the function for temperature initialization.
    if temperature_init_fn is None:
      self.init_fn_t = (
          lambda xx, yy, zz, lx, ly, lz, coord:  # pylint: disable=g-long-lambda
          self.t_init * tf.ones_like(xx, dtype=_TF_DTYPE))
    else:
      self.init_fn_t = temperature_init_fn(self.config, self.t_var)

  def vegetation_drag_update_fn(
      self,
      c_d: float,
      a_v: float,
  ) -> composite_types.StatesUpdateFn:
    """Generates an update function for drag forces due to vegetation.

    The drag force in direction i is computed as [1]:
      fᵢ = -3 Cd ϱf ϱg aᵥ|u|uᵢ / 8|ϱf|

    [1] Linn, Rodman, Jon Reisner, Jonah J. Colman, and Judith Winterkamp. 2002.
        “Studying Wildfire Behavior Using FIRETEC.” International Journal of
        Wildland Fire 11 (4): 233–46.

    Args:
      c_d: The drag coefficient of the vegetation.
      a_v: The ratio of the surface area of the fuel to the resolved volume (
        surface area per unit volume of fuel times the volume fraction).

    Returns:
      A function that updates the `additional_states` with the following keys:
      'src_u', 'src_v', 'src_w'.
    """

    def additional_states_update_fn(
        kernel_op: get_kernel_fn.ApplyKernelOp,
        replica_id: tf.Tensor,
        replicas: np.ndarray,
        states: FlowFieldMap,
        additional_states: FlowFieldMap,
        params: grid_parametrization.GridParametrization,
    ) -> FlowFieldMap:
      """Updates 'src_u', 'src_v', 'src_w'."""
      del kernel_op, replica_id, replicas, params

      def drag_force(u):
        """Computes the drag force for the specified velocity."""
        drag = [
            -3.0 * c_d * rho_g * a_v * u_mag_i * u_i / 8.0
            for rho_g, u_mag_i, u_i in zip(states['rho'], u_mag, u)
        ]
        return [
            tf.compat.v1.where(rho_f < EPSILON,
                               tf.zeros_like(drag_i, dtype=tf.float32), drag_i)
            for rho_f, drag_i in zip(additional_states['rho_f'], drag)
        ]

      u_mag = [
          tf.math.sqrt(u**2 + v**2 + w**2)
          for u, v, w in zip(states['u'], states['v'], states['w'])
      ]

      updated_additional_states = {}
      for key, value in additional_states.items():
        if key == 'src_u':
          updated_additional_states.update({'src_u': drag_force(states['u'])})
        elif key == 'src_v':
          updated_additional_states.update({'src_v': drag_force(states['v'])})
        elif key == 'src_w':
          updated_additional_states.update({'src_w': drag_force(states['w'])})
        else:
          updated_additional_states.update({key: value})

      return updated_additional_states

    return additional_states_update_fn

  def sponge_forcing_update_fn(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Generates an update function for forces due the sponge layer.

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
      `additional_states` with the items associated the following keys updated:
      `src_[varname]` with `varname` specified in the sponge target list.
    """
    return self.sponge.additional_states_update_fn(kernel_op, replica_id,
                                                   replicas, states,
                                                   additional_states, params)

  def inflow_generator_init(
      self, inflow_dim: int
  ) -> Optional[synthetic_turbulent_inflow.SyntheticTurbulentInflow]:
    """Initializes the synthetic turbulent inflow generator.

    Args:
      inflow_dim: The dimension of the inflow. Should be either 0 or 1.

    Returns:
      The `SyntheticTurbulentInflow` object corresponds to the `inflow_dim`.

    Raises:
      ValueError: If `inflow_dim` is neither 0 nor 1.
    """
    if inflow_dim == 0:
      velocity_mean = self.u_mean
      length_scale = self.inflow_length_scales_x
    elif inflow_dim == 1:
      velocity_mean = self.v_mean
      length_scale = self.inflow_length_scales_y
    else:
      raise ValueError(
          'Inflow dimension should be 0 or 1. {} is not allowed.'.format(
              inflow_dim))

    periodic = self.config.periodic_dims[inflow_dim]

    if np.abs(velocity_mean) <= EPSILON or periodic:
      return None

    inflow_face = 0 if velocity_mean > 0 else 1
    return synthetic_turbulent_inflow.SyntheticTurbulentInflow(
        length_scale, self.delta, self.mesh_size, inflow_dim, inflow_face)

  def init_fn_ones(self, xx: tf.Tensor, yy: tf.Tensor, zz: tf.Tensor, lx: float,
                   ly: float, lz: float, coord: ThreeIntTuple) -> tf.Tensor:
    """Creates a 3D tensor with value 1 that has the same size as `xx`.

    Args:
      xx: A 3D tensor of the mesh in dimension 0.
      yy: A 3D tensor of the mesh in dimension 1.
      zz: A 3D tensor of the mesh in dimension 2.
      lx: The length of dimension 0.
      ly: The length of dimension 1.
      lz: The length of dimension 2.
      coord: The coordinate of the local core.

    Returns:
      A 3D tensor of value 1.
    """
    del yy, zz, lx, ly, lz, coord
    return tf.ones_like(xx, dtype=_TF_DTYPE)

  def init_fn_zeros(self, xx: tf.Tensor, yy: tf.Tensor, zz: tf.Tensor,
                    lx: float, ly: float, lz: float,
                    coord: ThreeIntTuple) -> tf.Tensor:
    """Creates a 3D tensor with value 0 that has the same size as `xx`.

    Args:
      xx: A 3D tensor of the mesh in dimension 0.
      yy: A 3D tensor of the mesh in dimension 1.
      zz: A 3D tensor of the mesh in dimension 2.
      lx: The length of dimension 0.
      ly: The length of dimension 1.
      lz: The length of dimension 2.
      coord: The coordinate of the local core.

    Returns:
      A 3D tensor of value 0.
    """
    del yy, zz, lx, ly, lz, coord
    return tf.zeros_like(xx, dtype=_TF_DTYPE)

  def box_ignition_kernel_init_fn(
      self,
      fuel_top_elevation: tf.Tensor,
      ground_elevation: tf.Tensor,
  ) -> InitFn:
    """Generates an initialization function for a box ignition kernel."""

    def init_fn(xx, yy, zz, lx, ly, lz, coord):
      """Generates a box shaped ignition kernel."""
      del lx, ly, lz, coord
      location_x = tf.math.logical_and(
          tf.greater_equal(xx, self.ignition_x_low),
          tf.less_equal(xx, self.ignition_x_high))
      location_y = tf.math.logical_and(
          tf.greater_equal(yy, self.ignition_y_low),
          tf.less_equal(yy, self.ignition_y_high))
      location_z = tf.math.logical_and(
          tf.less_equal(zz, fuel_top_elevation),
          tf.greater_equal(zz, ground_elevation))
      location_xy = tf.math.logical_and(location_x, location_y)
      location = tf.math.logical_and(location_xy, location_z)
      return tf.compat.v1.where(location, tf.ones_like(zz, dtype=tf.float32),
                                tf.zeros_like(zz, dtype=tf.float32))

    return init_fn

  def slant_line_ignition_kernel_init_fn(
      self,
      fuel_top_elevation: tf.Tensor,
      ground_elevation: tf.Tensor,
      line_info: Optional[SlantLine] = None,
  ) -> InitFn:
    """Generates a init function for a slant ignition line.

    The ignition kernel is defined by its center in the x-y plane, the length,
    width, and angle with respect to the x-axis.

    Args:
      fuel_top_elevation: A 2D tf.Tensor that represent the physical coordinates
        of the fuel top in the vertical (z) direction.
      ground_elevation: A 2D tf.Tensor that represents the physical coordinates
        of the ground in the vertical (z) direction.
      line_info: A SlantLine object that stores information of the ignition line
        orientation.

    Returns:
      A function that takes the coordinates information and generates a 3D
      tf.Tensor with binary values, where the ignition kernel is set to 1 and
      elsewhere set to 0.
    """
    if line_info is None:
      line_info = self.ignition_line

    def init_fn(xx, yy, zz, lx, ly, lz, coord):
      """Generates a binary mask with ignition kernel 1 and 0 elsewhere."""
      del lx, ly, lz, coord

      # Rotate the original (x, y) coordinates by the angle of the slant
      # ignition line around its center, so that the new coordinate system is
      # aligned with the ignition line.
      deg = np.radians(line_info.angle)
      cos = np.cos(deg)
      sin = np.sin(deg)

      x_new = cos * (xx - line_info.center_x) + sin * (yy - line_info.center_y)
      y_new = -sin * (xx - line_info.center_x) + cos * (yy - line_info.center_y)

      location_x = tf.math.logical_and(
          tf.less_equal(x_new, line_info.length / 2.0),
          tf.greater_equal(x_new, -line_info.length / 2.0))
      location_y = tf.math.logical_and(
          tf.less_equal(y_new, line_info.thickness / 2.0),
          tf.greater_equal(y_new, -line_info.thickness / 2.0))
      location_xy = tf.math.logical_and(location_x, location_y)

      location_z = tf.math.logical_and(
          tf.less_equal(zz, fuel_top_elevation),
          tf.greater_equal(zz, ground_elevation))
      location = tf.math.logical_and(location_xy, location_z)

      return tf.where(location, tf.ones_like(zz, dtype=tf.float32),
                      tf.zeros_like(zz, dtype=tf.float32))

    return init_fn

  def multi_line_ignition_kernel_init_fn(
      self,
      fuel_top_elevation: tf.Tensor,
      ground_elevation: tf.Tensor,
  ) -> InitFn:
    """Generates a init function for multiple ignition lines.

    The ignition kernels are defined by their centers in the x-y plane, the
    lengths, widths, and angles with respect to the x-axis.

    Args:
      fuel_top_elevation: A 2D tf.Tensor that represent the physical coordinates
        of the fuel top in the vertical (z) direction.
      ground_elevation: A 2D tf.Tensor that represents the physical coordinates
        of the ground in the vertical (z) direction.

    Returns:
      A function that takes the coordinates information and generates a 3D
      tf.Tensor with binary values, where the ignition kernel is set to 1 and
      elsewhere set to 0.
    """
    ignition_kernel_fn = [
        self.slant_line_ignition_kernel_init_fn(fuel_top_elevation,
                                                ground_elevation, line)
        for line in self.ignition_lines
    ]

    def init_fn(xx, yy, zz, lx, ly, lz, coord):
      """Generates a binary mask with ignition kernel 1 and 0 elsewhere."""
      location = tf.zeros_like(xx)
      for line_fn in ignition_kernel_fn:
        location += line_fn(xx, yy, zz, lx, ly, lz, coord)
      return tf.math.divide_no_nan(location, location)

    return init_fn

  def init_spherical_ignition_kernel(
      self,
      xx: initializer.TensorOrArray,
      yy: initializer.TensorOrArray,
      zz: initializer.TensorOrArray,
      lx: float,
      ly: float,
      lz: float,
      coord: ThreeIntTuple,
  ) -> tf.Tensor:
    """Generates a spherical ignition kernel with smooth interface."""
    del lx, ly, lz, coord
    if self.ignition_option != IgnitionKernelType.SPHERE:
      raise ValueError(
          'Spherical ignition kernel is only available for '
          '`IgnitionKernelType.SPHERE`. {} is not feasible.'.format(
              self.ignition_option))

    rr = tf.math.sqrt(
        tf.math.pow(xx - self.ignition_center_x, 2) +
        tf.math.pow(yy - self.ignition_center_y, 2) +
        tf.math.pow(zz - self.ignition_center_z, 2))

    return 0.5 * (
        tf.math.tanh(self.ignition_scale * (rr + self.ignition_radius)) -
        tf.math.tanh(self.ignition_scale * (rr - self.ignition_radius)))

  def ignition_kernel_init_fn(
      self,
      fuel_top_elevation: tf.Tensor,
      ground_elevation: tf.Tensor,
  ) -> InitFn:
    """Generates an init function with the selected ignition kernel.

    Args:
      fuel_top_elevation: A 2D tf.Tensor that represents the physical
        coordinates of the fuel top in the vertical (z) direction.
      ground_elevation: A 2D tf.Tensor that represents the physical coordinates
        of the ground in the vertical (z) direction.

    Returns:
      A function that takes the coordinates information and generates a 3D
      tf.Tensor with binary values, where the ignition kernel is set to 1 and
      elsewhere set to 0.
    """
    if self.ignition_option == IgnitionKernelType.BOX:
      ignition_shape_fn = self.box_ignition_kernel_init_fn(
          fuel_top_elevation, ground_elevation
      )
    elif self.ignition_option == IgnitionKernelType.SPHERE:
      ignition_shape_fn = self.init_spherical_ignition_kernel
    elif self.ignition_option == IgnitionKernelType.SLANT_LINE:
      ignition_shape_fn = self.slant_line_ignition_kernel_init_fn(
          fuel_top_elevation, ground_elevation
      )
    elif self.ignition_option == IgnitionKernelType.MULTI_LINE:
      ignition_shape_fn = self.multi_line_ignition_kernel_init_fn(
          fuel_top_elevation, ground_elevation
      )
    return ignition_shape_fn

  def states_init(
      self,
      coordinates: ThreeIntTuple,
      init_fn: InitFn,
      pad_mode: Optional[str] = 'SYMMETRIC',
  ) -> tf.Tensor:
    """Assigns value to a tensor with `init_fn`.

    Args:
      coordinates: A tuple that specifies the replica's grid coordinates in
        physical space.
      init_fn: A function that takes the local mesh_grid tensor for the core (in
        order x, y, z) and the global characteristic length floats (in order x,
        y, z) and returns a 3-D tensor representing the value for the local core
        (without including the margin/overlap between the cores).
      pad_mode: The mode for filling in values in the halo layers. Should be one
        of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive). If
        'CONSTANT' is used, 0 will be assigned in the halos.

    Returns:
      A 3D tensor with values assigned by `init_fn`.
    """
    return initializer.partial_mesh_for_core(
        self.config,
        coordinates,
        init_fn,
        pad_mode=pad_mode,
        num_boundary_points=0,
        mesh_choice=initializer.MeshChoice.PARAMS,
    )

  def inflow_states_init(
      self,
      inflow_generator: synthetic_turbulent_inflow.SyntheticTurbulentInflow,
      inflow_dim: int,
      mask: tf.Tensor,
  ) -> Mapping[str, tf.Tensor]:
    """Initializes states required by synthetic turbulent inflow.

    Args:
      inflow_generator: An `SyntheticTurbulentInflow` object.
      inflow_dim: The dimension of the inflow. Should be either 0 or 1.
      mask: A 2D tensor with values being 0 or 1. Regions with value 1 indicates
        fluid, and 0 for non-fluid (e.g. fuel elements, solid bodies, etc.). The
        shape of the 2D tensor has to be (core_n0, core_n1), where core_n0 and
        core_n1 are the number of nodes along directions other than
        `inflow_dim`, with halos excluded.

    Returns:
      A dictionary of states or helper variables that are required by the
      synthetic inflow generation.

    Raises:
      ValueError: If `inflow_dim` is neither 0 nor 1.
    """
    if inflow_dim == 0:
      velocity_mean = self.u_mean
      ones = tf.ones(
          (self.config.nz, self.config.halo_width + 1, self.config.ny),
          dtype=_TF_DTYPE)
      zeros = tf.zeros(
          (self.config.nz, self.config.halo_width + 1, self.config.ny),
          dtype=_TF_DTYPE)
    elif inflow_dim == 1:
      velocity_mean = self.v_mean
      ones = tf.ones(
          (self.config.nz, self.config.nx, self.config.halo_width + 1),
          dtype=_TF_DTYPE)
      zeros = tf.zeros(
          (self.config.nz, self.config.nx, self.config.halo_width + 1),
          dtype=_TF_DTYPE)
    else:
      raise ValueError(
          'Inflow dimension should be 0 or 1. {} is not allowed.'.format(
              inflow_dim))

    r = inflow_generator.generate_random_fields(self.inflow_seed)
    inflow_face = 0 if velocity_mean > 0 else 1

    def varname(var_type: str, velocity: str) -> str:
      """Generates the variable name for `key`."""
      return inflow_generator.helper_key(var_type, velocity, inflow_dim,
                                         inflow_face)

    return {
        varname('bc', 'u'): self.u_mean * ones,
        varname('bc', 'v'): self.v_mean * ones,
        varname('bc', 'w'): zeros,
        varname('mean', 'u'): self.u_mean * mask,
        varname('mean', 'v'): self.v_mean * mask,
        varname('mean', 'w'): tf.zeros_like(mask, dtype=_TF_DTYPE),
        varname('rms', 'u'): self.u_rms * mask,
        varname('rms', 'v'): self.v_rms * mask,
        varname('rms', 'w'): self.w_rms * mask,
        varname('rand', 'u'): r[0],
        varname('rand', 'v'): r[1],
        varname('rand', 'w'): r[2],
    }

  def sponge_init(
      self,
      coordinates: ThreeIntTuple,
  ) -> Mapping[str, tf.Tensor]:
    """Initializes additional states required by the sponge layer.

    Args:
      coordinates: A tuple that specifies the replica's grid coordinates in
        physical space.

    Returns:
      A dictionary containing the value of `sponge_beta`, which will be used to
      compute the sponge forcing term.
    """
    klrc = rayleigh_damping_layer.klemp_lilly_relaxation_coeff_fns_for_sponges
    beta_fns_by_name = klrc(self.config.dt, self.config.sponge)
    return self.sponge.init_fn(self.config, coordinates, beta_fns_by_name)
