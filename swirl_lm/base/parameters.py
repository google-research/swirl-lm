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

"""Library for input config for the incompressible Navier-Stokes solver."""

import copy
import json
import os
import os.path
import sys
from typing import Callable, List, Literal, Mapping, Optional, Sequence, Tuple, TypeAlias

from absl import flags
from absl import logging
import numpy as np
from swirl_lm.base import parameters_pb2
from swirl_lm.base import physical_variable_keys_manager
from swirl_lm.boundary_condition import boundary_condition_utils
from swirl_lm.boundary_condition import boundary_conditions_pb2
from swirl_lm.communication import halo_exchange
from swirl_lm.equations import pressure_pb2
from swirl_lm.numerics import derivatives
from swirl_lm.numerics import numerics_pb2
from swirl_lm.physics.thermodynamics import thermodynamics_pb2
from swirl_lm.physics.turbulent_combustion import turbulent_combustion
from swirl_lm.utility import file_io
from swirl_lm.utility import file_pb2
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import grid_parametrization_pb2
from swirl_lm.utility import types
import tensorflow as tf

from google.protobuf import text_format

FlowFieldVal: TypeAlias = types.FlowFieldVal
FlowFieldMap: TypeAlias = types.FlowFieldMap

# The threshold of the difference between the absolute value of the
# gravitational vector along a dimension and one. Below this threshold the
# corresponding dimension is the gravity (vertical) dimension.
_G_THRESHOLD = 1e-6

flags.DEFINE_string(
    'config_filepath', None,
    'The full path to the text proto file that stores all input parameters.'
)
flags.DEFINE_bool(
    'simulation_debug',
    False,
    'Toggles if to run the simulation with the debug mode.',
    allow_override=True)
# Note that these flags are ineffective if they are set in the config.
_NUM_CYCLES = flags.DEFINE_integer(
    'num_cycles',
    1,
    'number of cycles to run. Each cycle generates a set of output',
)
_NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 1, 'number of steps to run before generating an output.'
)
_APPLY_PREPROCESS = flags.DEFINE_bool(
    'apply_preprocess',
    False,
    (
        'If True and the `preprocessing_states_update_fn` is defined in'
        ' `params`, data from initial condition are processed before the'
        ' simulation.'
    ),
)
_PREPROCESS_STEP_ID = flags.DEFINE_integer(
    'preprocess_step_id',
    0,
    (
        'The `step_id` for the preprocessing function to be executed at, or if '
        '`preprocess_periodic` is `True`, the period in steps to perform '
        'preprocessing.'
    ),
)
_PREPROCESS_PERIODIC = flags.DEFINE_bool(
    'preprocess_periodic', False, 'Whether to do preprocess periodically.'
)
_APPLY_POSTPROCESS = flags.DEFINE_bool(
    'apply_postprocess',
    False,
    (
        'If True and the `postprocessing_states_update_fn` is defined in'
        ' `params`, a post processing will be executed after the update.'
    ),
)
_POSTPROCESS_STEP_ID = flags.DEFINE_integer(
    'postprocess_step_id',
    0,
    (
        'The `step_id` for the postprocessing function to be executed at, or if'
        ' `postprocess_periodic` is `True`, the period in steps to perform'
        ' postprocessing.'
    ),
)
_POSTPROCESS_PERIODIC = flags.DEFINE_bool(
    'postprocess_periodic', False, 'Whether to do postprocess periodically.'
)
_START_STEP = flags.DEFINE_integer(
    'start_step', 0, 'The beginning step count for the current simulation.'
)
_LOADING_STEP = flags.DEFINE_integer(
    'loading_step',
    None,
    (
        'When this is set, it is the step count from which to '
        'load the initial states.'
    ),
)

KernelOpType = parameters_pb2.SwirlLMParameters.KernelOpType
SolverProcedure = parameters_pb2.SwirlLMParameters.SolverProcedureType
ConvectionScheme = numerics_pb2.ConvectionScheme
DiffusionScheme = numerics_pb2.DiffusionScheme
TimeIntegrationScheme = numerics_pb2.TimeIntegrationScheme
DensityUpdateOption = parameters_pb2.SwirlLMParameters.DensityUpdateOption
SourceUpdateFn = Callable[[
    get_kernel_fn.ApplyKernelOp, tf.Tensor, np.ndarray, types
    .FlowFieldMap, types.FlowFieldMap, grid_parametrization.GridParametrization
], types.FlowFieldMap]

SourceUpdateFnLib = Mapping[str, SourceUpdateFn]

_BCInfo = boundary_conditions_pb2.BoundaryCondition.BoundaryInfo
_BCType = grid_parametrization_pb2.BoundaryConditionType
_BCParams = boundary_conditions_pb2.BoundaryCondition.BoundaryConditionParams

FLAGS = flags.FLAGS


def _get_gravity_direction(
    config: parameters_pb2.SwirlLMParameters,
) -> Sequence[float]:
  """Derives the gravitational vector from the configuration.

  Args:
    config: An instance of the `SwirlLMParameters` proto.

  Returns:
    A 3-component vector that represents the direction of the gravity
    (normalized) if `gravity_direction` is defined in the simulation
    configuration and the magnitude is non-trivial; otherwise a vector with all
    zeros is returned.
  """
  if config.HasField('gravity_direction'):
    gravity_direction = [
        config.gravity_direction.dim_0, config.gravity_direction.dim_1,
        config.gravity_direction.dim_2
    ]
    # Normalize the gravitational vector.
    g_magnitude = np.linalg.norm(gravity_direction)
    if g_magnitude > _G_THRESHOLD:
      gravity_direction = [
          g_dir / g_magnitude for g_dir in gravity_direction
      ]
    else:
      gravity_direction = [0.] * 3
  else:
    gravity_direction = [0.] * 3

  return gravity_direction


class SwirlLMParameters(grid_parametrization.GridParametrization):
  """Parameters for running the incompressible Navier-Stokes solver."""

  def __init__(self, config: parameters_pb2.SwirlLMParameters):
    """Initializes the SwirlLMParameters object.

    Args:
      config: An instance of the `SwirlLMParameters` proto.

    Raises:
      ValueError: If the kernel operator type or scheme used for discretizing
        the diffusion term is not recognized.
    """
    super(SwirlLMParameters, self).__init__(config.grid_params)

    self.swirl_lm_parameters_proto = config
    self.bc_manager = (
        physical_variable_keys_manager.BoundaryConditionKeysHelper())

    self._start_step = _START_STEP.value
    self._loading_step = (
        _LOADING_STEP.value
        if _LOADING_STEP.value is not None
        else self._start_step
    )

    self.kernel_op_type = config.kernel_op_type
    if config.kernel_op_type == KernelOpType.KERNEL_OP_CONV:
      self._kernel_op = get_kernel_fn.ApplyKernelConvOp(self.kernel_size)
    elif config.kernel_op_type == KernelOpType.KERNEL_OP_SLICE:
      self._kernel_op = get_kernel_fn.ApplyKernelSliceOp()
    elif config.kernel_op_type == KernelOpType.KERNEL_OP_MATMUL:
      self._kernel_op = get_kernel_fn.ApplyKernelMulOp(self.nx, self.ny)
    else:
      raise ValueError(
          'Unknown kernel operator {}'.format(config.kernel_op_type)
      )

    self.solver_procedure = config.solver_procedure
    self.convection_scheme = config.convection_scheme
    self.numerical_flux = config.numerical_flux
    self.diffusion_scheme = config.diffusion_scheme
    self.time_integration_scheme = config.time_integration_scheme

    self.diff_stab_crit = (
        config.diff_stab_crit if config.HasField('diff_stab_crit') else None
    )

    self.enable_scalar_recorrection = config.enable_scalar_recorrection
    self.enable_rhie_chow_correction = config.enable_rhie_chow_correction

    logging.info('Convection scheme: %s, Diffusion scheme: %s, Time scheme: %s',
                 ConvectionScheme.Name(self.convection_scheme),
                 DiffusionScheme.Name(self.diffusion_scheme),
                 TimeIntegrationScheme.Name(self.time_integration_scheme))

    self.thermodynamics = config.thermodynamics if config.HasField(
        'thermodynamics') else None
    self.radiative_transfer = config.radiative_transfer if config.HasField(
        'radiative_transfer') else None
    self.microphysics = config.microphysics if config.HasField(
        'microphysics') else None
    self.lpt = config.lpt if config.HasField('lpt') else None

    if (self.thermodynamics is not None and
        self.thermodynamics.HasField('solver_mode')):
      self.solver_mode = self.thermodynamics.solver_mode
    else:
      self.solver_mode = thermodynamics_pb2.Thermodynamics.LOW_MACH

    self.combustion = config.combustion if config.HasField(
        'combustion') else None

    turbulent_combustion_params = (
        config.turbulent_combustion
        if config.HasField('turbulent_combustion')
        else None
    )
    if turbulent_combustion_params is not None:
      assert (
          self.combustion is not None
      ), 'A turbulent combustion model is defined without a combustion model.'
    self.turbulent_combustion = (
        turbulent_combustion.turbulent_combustion_model_factory(
            turbulent_combustion_params
        )
    )

    self.additional_state_keys = config.additional_state_keys
    self.helper_var_keys = config.helper_var_keys
    self.debug_variables = config.debug_variables
    self.states_from_file = config.states_from_file
    self.states_to_file = list(config.states_to_file)
    self.monitor_spec = config.monitor_spec
    self.probe = config.probe if config.HasField('probe') else None

    self.use_sgs = config.use_sgs
    self.sgs_model = config.sgs_model
    self.use_3d_tf_tensor = config.use_3d_tf_tensor

    self.deriv_lib = derivatives.Derivatives(
        self.kernel_op,
        self.use_3d_tf_tensor,
        self.grid_spacings,
        self.use_stretched_grid,
    )

    # Initialize the direction vector of gravitational force. Set to 0 if
    # gravity is not defined in the simulation.
    self.gravity_direction = _get_gravity_direction(config)

    g_dim = np.unique(
        np.nonzero(np.abs(np.abs(self.gravity_direction) - 1.0) < _G_THRESHOLD)
    )
    assert len(g_dim) <= 1, (
        'Gravity dimension is ambiguous if it is not aligned with an axis.'
        f' {g_dim} is provided.'
    )
    self.g_dim: Literal[0, 1, 2] | None = (
        g_dim.item() if len(g_dim) == 1 else None
    )

    # Get the scalar related quantities if scalars are solved as a
    # `List[SwirlLMParameters.Scalar]`.
    self.scalars = config.scalars

    self.scalar_lib = {scalar.name: scalar for scalar in self.scalars}

    self.bc = {'u': None, 'v': None, 'w': None, 'p': None}
    self.bc.update({scalar.name: None for scalar in self.scalars})
    self.bc_params = {'u': None, 'v': None, 'w': None, 'p': None}
    self.bc_params.update({scalar.name: None for scalar in self.scalars})

    for input_bc in config.boundary_conditions:
      bc, bc_params = self._parse_boundary_conditions(input_bc.boundary_info)
      self.bc[input_bc.name] = bc
      self.bc_params[input_bc.name] = bc_params

    # Find the type of boundary, e.g. wall, open boundary, periodic, from the
    # boundary conditions.
    self.bc_type = boundary_condition_utils.find_bc_type(
        self.bc, self.periodic_dims)
    logging.info('Boundary conditions from the input: %r', self.bc)

    logging.info('Boundary conditions for `u`, `v`, `w`, and all scalars are '
                 'retrieved from the config file. Boundary condition for `p` '
                 'is derived based on boundary types.')

    # Adding new additional keys to hold boundary conditions.
    self.bc_keys = (
        boundary_condition_utils.get_keys_for_boundary_condition(
            self.bc, halo_exchange.BCType.NONREFLECTING))
    self.additional_state_keys.extend(self.bc_keys)

    # Get the number of sub-iterations at each time step.
    self.corrector_nit = config.num_sub_iterations

    # Get the number of iterations of the pressure solver. This option will be
    # deprecated.
    self.nit = config.num_pressure_iterations

    # Get the options for pressure solving.
    self.pressure = config.pressure if config.HasField('pressure') else None

    # Get the physical quantities.
    self.rho = config.density
    self.nu = config.kinematic_viscosity
    self.p_thermal = config.p_thermal

    # Allocate a space for a function that updates the additional states.
    self._additional_states_update_fn = None

    # Allocate a space for functions that update source terms for requested
    # variables.
    self._source_update_fn_lib = {}

    # Allocate a space for a function that preprocesses `states`.
    self._preprocessing_states_update_fn = None

    # Allocate a space for a function that postprocesses `states`.
    self._postprocessing_states_update_fn = None

    # Get the option for density update.
    self.density_update_option = config.density_update_option

    # Get the information required for sponge layers if applied.
    if config.HasField('boundary_models') and config.boundary_models.sponge:
      self.sponge = config.boundary_models.sponge
    elif config.HasField('sponge_layer'):
      self.sponge = [config.sponge_layer]
    else:
      self.sponge = None

    # Get models for boundary treatments.
    self.boundary_models = config.boundary_models if config.HasField(
        'boundary_models') else None

    # Set the diffusion scheme to CENTRAL_3 if the Monin-Obukhov similarity
    # theory is used as the atmospheric-boundary layer closure.
    if self.boundary_models is not None:
      if self.boundary_models.HasField('most'):
        if (self.diffusion_scheme !=
            numerics_pb2.DiffusionScheme.DIFFUSION_SCHEME_CENTRAL_3):
          raise ValueError(
              f'{numerics_pb2.DiffusionScheme.Name(self.diffusion_scheme)} is '
              f'used as the diffusion scheme in the momentum equations but only'
              f' DIFFUSION_SCHEME_CENTRAL_3 supports the Monin-Obukhov '
              f'similarity theory.')

    if any(self.use_stretched_grid):
      _validate_config_for_stretched_grid(
          config, self.use_stretched_grid, self.global_xyz, self.g_dim
      )

    # Toggle if to run with the debug mode.
    self.dbg = FLAGS.simulation_debug

    # Get the number of cycles and steps the simulation needs to run. Member
    # variables `_num_cycles` and `_num_steps` will be initialized.
    self._set_simulation_time_info()

    # Determine the pre- and post-process options.
    self._set_pre_post_process_info()

  def __str__(self) -> str:
    return super(SwirlLMParameters, self).__str__() + (
        ', rho: {}, nu: {}, nit: {}'.format(self.rho, self.nu, self.nit))

  def _get_inflow_velocity(self, inflow_dim: int, inflow_face: int) -> float:
    """Retrieves the inflow velocity from the simulation setup."""
    assert (
        self.bc_type[inflow_dim][inflow_face]
        == boundary_condition_utils.BoundaryType.INFLOW
    ), (
        'Simulation setup with `from_flow_through_time` requires a boundary'
        '  type of `INFLOW` in the dimension specified'
        f' ({inflow_dim}), but is'
        f' {self.bc_type[inflow_dim][inflow_face]}.'
    )

    inflow_velocity_name = ('u', 'v', 'w')[inflow_dim]

    return np.abs(
        self.bc[inflow_velocity_name][inflow_dim][inflow_face][1]
    )

  def _get_flow_through_time(self, inflow_dim: int, inflow_face: int) -> float:
    """Computes the flow through time with an inflow boundary condition."""
    inflow_domain_length = (self.lx, self.ly, self.lz)[inflow_dim]
    inflow_val = self._get_inflow_velocity(inflow_dim, inflow_face)
    return inflow_domain_length / inflow_val

  def _set_simulation_time_info(self):
    """Sets `_num_cycles` and `_num_steps` with a lazy approach.

    Note that values for `num_steps` and `num_cycles` from the flags are not
    used if the `simulation_time_method` is `from_config` or
    `from_flow_through_time` even they are provided.
    """
    config = self.swirl_lm_parameters_proto
    if (
        not config.HasField('simulation_time_info')
        or config.simulation_time_info.WhichOneof('simulation_time_method')
        == 'from_flags'
    ):
      self._num_cycles = _NUM_CYCLES.value
      self._num_steps = _NUM_STEPS.value
    elif (
        config.simulation_time_info.WhichOneof('simulation_time_method')
        == 'from_config'
    ):
      self._num_cycles = config.simulation_time_info.from_config.num_cycles
      self._num_steps = config.simulation_time_info.from_config.num_steps
    elif (
        config.simulation_time_info.WhichOneof('simulation_time_method')
        == 'from_flow_through_time'
    ):
      inflow_dim = (
          config.simulation_time_info.from_flow_through_time.mean_flow_dim
      )
      inflow_face = (
          config.simulation_time_info.from_flow_through_time.inflow_face
      )
      self._num_cycles = (
          config.simulation_time_info.from_flow_through_time.num_cycles
      )
      n_flow_through_time = (
          config.simulation_time_info.from_flow_through_time.n_flow_through_time
      )
      t_per_cycle = (
          n_flow_through_time
          * self._get_flow_through_time(inflow_dim, inflow_face)
      ) / float(self._num_cycles)
      self._num_steps = int(np.ceil(t_per_cycle / self.dt))
    else:
      raise ValueError(
          'Unknown simulation time info:'
          f' {config.simulation_time_info.WhichOneof("simulation_time_method")}'
      )

  def _set_pre_post_process_info(self):
    """Determines the pre- and post-process options.

    Note that values for `apply_[pre,post]process`, `[pre,post]process_step_id`,
    and `[pre,post]process_periodic` from the flags are not used if the
    `pre_post_process_option` is `from_config` or `from_flow_through_time` even
    they are provided.
    """
    config = self.swirl_lm_parameters_proto
    if (
        not config.HasField('pre_post_process_info')
        or config.pre_post_process_info.WhichOneof('pre_post_process_option')
        == 'from_flags'
    ):
      self._apply_preprocess = _APPLY_PREPROCESS.value
      self._preprocess_step_id = _PREPROCESS_STEP_ID.value
      self._preprocess_periodic = _PREPROCESS_PERIODIC.value
      self._apply_postprocess = _APPLY_POSTPROCESS.value
      self._postprocess_step_id = _POSTPROCESS_STEP_ID.value
      self._postprocess_periodic = _POSTPROCESS_PERIODIC.value
    elif (
        config.pre_post_process_info.WhichOneof('pre_post_process_option')
        == 'from_config'
    ):
      opt = config.pre_post_process_info.from_config
      self._apply_preprocess = opt.apply_preprocess
      self._preprocess_step_id = opt.preprocess_step_id
      self._preprocess_periodic = opt.preprocess_periodic
      self._apply_postprocess = opt.apply_postprocess
      self._postprocess_step_id = opt.postprocess_step_id
      self._postprocess_periodic = opt.postprocess_periodic
    elif (
        config.pre_post_process_info.WhichOneof('pre_post_process_option')
        == 'from_flow_through_time'
    ):
      opt = config.pre_post_process_info.from_flow_through_time
      self._apply_preprocess = opt.apply_preprocess
      self._preprocess_periodic = opt.preprocess_periodic
      self._apply_postprocess = opt.apply_postprocess
      self._postprocess_periodic = opt.postprocess_periodic

      flow_through_time = self._get_flow_through_time(
          opt.mean_flow_dim, opt.inflow_face
      )

      def get_step_id(t: float) -> int:
        """Computes the closest integer multiple of `num_step` to `t`."""
        return (
            self.start_step
            + round(t / (float(self.num_steps) * self.dt)) * self.num_steps
        )

      self._preprocess_step_id = get_step_id(
          opt.preprocess_flow_through_time * flow_through_time
      )
      self._postprocess_step_id = get_step_id(
          opt.postprocess_flow_through_time * flow_through_time
      )
    else:
      pre_post_opt = config.pre_post_process_info.WhichOneof(
          'pre_post_process_option'
      )
      raise ValueError(f'Unknown pre-post process info: {pre_post_opt}')

  def _parse_boundary_info(
      self, boundary_info: _BCInfo
  ) -> Tuple[
      Optional[Tuple[halo_exchange.BCType, float]],
      Optional[_BCParams]]:
    """Retrieves the boundary condition from proto to fit the framework."""
    bc_type_value = None
    if boundary_info.type == _BCType.BC_TYPE_DIRICHLET:
      bc_type_value = (halo_exchange.BCType.DIRICHLET, boundary_info.value)
    elif boundary_info.type == _BCType.BC_TYPE_NEUMANN:
      bc_type_value = (halo_exchange.BCType.NEUMANN, boundary_info.value)
    elif boundary_info.type == _BCType.BC_TYPE_NEUMANN_2:
      bc_type_value = (halo_exchange.BCType.NEUMANN_2, boundary_info.value)
    elif boundary_info.type == _BCType.BC_TYPE_NO_TOUCH:
      bc_type_value = (halo_exchange.BCType.NO_TOUCH, 0.0)
    elif boundary_info.type == _BCType.BC_TYPE_NONREFLECTING:
      bc_type_value = (halo_exchange.BCType.NONREFLECTING, boundary_info.value)

    return (bc_type_value, boundary_info.bc_params)

  def _parse_boundary_conditions(
      self, boundary_conditions: Sequence[_BCInfo]
  ) -> Tuple[
      List[List[Optional[Tuple[halo_exchange.BCType, float]]]],
      List[List[Optional[_BCParams]]]]:
    """Parses the boundary conditions.

    Args:
      boundary_conditions: A proto that stores all boundary information for a
        specific variable.

    Returns:
      bc: The boundary condition in the fluid framework format.
    """

    bc = [[None, None], [None, None], [None, None]]
    bc_params = [[None, None], [None, None], [None, None]]
    for bc_info in boundary_conditions:
      (
          bc[bc_info.dim][bc_info.location],
          bc_params[bc_info.dim][bc_info.location],
      ) = self._parse_boundary_info(bc_info)
    return bc, bc_params

  @classmethod
  def config_from_text_proto(
      cls,
      text_proto: str,
  ) -> 'SwirlLMParameters':
    """Parses the config proto in text format into SwirlLMParameters."""
    config = parse_text_proto(text_proto)
    if not config.HasField('grid_params'):
      raise ValueError(
          'The `grid_params` field is now required and flags to set these '
          'parameters are no longer available (cl/681027966).')

    # Sanity check for the solver procedure.
    if config.solver_procedure not in (SolverProcedure.SEQUENTIAL,
                                       SolverProcedure.VARIABLE_DENSITY):
      raise NotImplementedError(
          'Solver procedure needs to be specified for a simulation.')

    # Sanity check for the numerical schemes.
    if config.convection_scheme == (ConvectionScheme.CONVECTION_SCHEME_UNKNOWN):
      raise NotImplementedError('Convection scheme is not specified.')

    if config.time_integration_scheme == (
        TimeIntegrationScheme.TIME_SCHEME_UNKNOWN):
      raise NotImplementedError('Time integration scheme is not specified.')

    return cls(config)

  @classmethod
  def config_from_proto(
      cls,
      config_filepath: str,
  ) -> 'SwirlLMParameters':
    """Reads the config text proto file."""
    text_proto = file_io.load_file(file_pb2.File(path=config_filepath))
    return cls.config_from_text_proto(text_proto)

  @property
  def num_cycles(self) -> int:
    """Provides the number of cycles for the simulation to run."""
    return self._num_cycles

  @property
  def num_steps(self) -> int:
    """Provides the number of steps in each simulation cycle."""
    return self._num_steps

  @property
  def start_step(self) -> int:
    """Provides the step id to start the simulation."""
    return self._start_step

  @property
  def loading_step(self):
    """Provides the data load id to start the simulation."""
    return self._loading_step

  @property
  def apply_preprocess(self) -> bool:
    """Provides the option for whether pre-process is applied."""
    return self._apply_preprocess

  @property
  def apply_postprocess(self) -> bool:
    """Provides the option for whether post-process is applied."""
    return self._apply_postprocess

  @property
  def preprocess_step_id(self) -> int:
    """The step id at which preprocess is applied."""
    return self._preprocess_step_id

  @property
  def postprocess_step_id(self) -> int:
    """The step id at which postprocess is applied."""
    return self._postprocess_step_id

  @property
  def preprocess_periodic(self) -> bool:
    """The option of whether preprocess function is applied periodically."""
    return self._preprocess_periodic

  @property
  def postprocess_periodic(self) -> bool:
    """The option of whether postprocess function is applied periodically."""
    return self._postprocess_periodic

  @property
  def max_halo_width(self) -> int:
    """Determines the halo width based on the selected convection scheme.

    The halo width should be the maximum number of halo cells required by all
    schemes required by the simulation configuration.

    Returns:
      An integer that specifies the halo width to be used throughout the
      simulation.
    """
    if self.convection_scheme == ConvectionScheme.CONVECTION_SCHEME_UPWIND_1:
      return max(1, self.halo_width)
    elif self.convection_scheme == ConvectionScheme.CONVECTION_SCHEME_QUICK:
      return max(2, self.halo_width)
    elif self.convection_scheme == ConvectionScheme.CONVECTION_SCHEME_CENTRAL_2:
      return max(1, self.halo_width)
    elif self.convection_scheme == ConvectionScheme.CONVECTION_SCHEME_CENTRAL_4:
      return max(2, self.halo_width)
    else:
      raise ValueError(
          'Halo width is ambiguous because convection scheme is not recognized.'
      )

  @property
  def scalars_names(self) -> List[str]:
    """Retrieves the names of all scalars in the flow system.

    Returns:
      A list that contains all scalars' name that is involved in the simulation.
      If no scalar is included, it returns an empty list.
    """
    return [scalar.name for scalar in self.scalars]

  @property
  def transport_scalars_names(self) -> List[str]:
    """Retrieves the names of transported scalars in the flow system.

    Returns:
      A list that contains all transported scalars' name that is involved in the
      simulation. If no scalar is included, it returns an empty list.
    """
    return [scalar.name for scalar in self.scalars if scalar.solve_scalar]

  @property
  def additional_states_update_fn(self):
    """A function that updates the additional states.

    The additional states can be updated by functions that are not part of the
    solution system, e.g. boundary conditions, forcing terms etc. This function
    needs to be defined outside of the main governing equations. If it's not
    defined, the `additional_states` is not updated.

    Returns:
      The function that updates `additional_states`, which takes the following
      arguments:
      `kernel_op`, `states`, `additional_states`, `params`,
      and it should return the updated `additional_states`.
    """
    return self._additional_states_update_fn

  @additional_states_update_fn.setter
  def additional_states_update_fn(self, update_fn):
    """Sets the function that updates the additional states."""
    self._additional_states_update_fn = update_fn

  @property
  def source_update_fn_lib(self):
    """A library of functions that updates the source terms.

    This library of functions will be called at each subiteration to update the
    source terms by requested variables.

    Returns:
      A library of source term update functions. The source term update function
      takes the following arguments:
      `kernel_op`, `replica_id`, `replicas`, states`, `additional_states`,
      `params`, and it should return the updated source term.
    """
    return self._source_update_fn_lib

  @source_update_fn_lib.setter
  def source_update_fn_lib(self, source_lib: SourceUpdateFnLib):
    """Sets the library of functions that updates the source termss."""
    self._source_update_fn_lib = source_lib

  def source_update_fn(self, varname: str):
    """Retrieves the source term update function for `varname` if available.

    Args:
      varname: The name of the variable for which the source term update
        function is requested.

    Returns:
      If `varname` is in `source_update_fn_lib`, the corresponding function is
      returned. Otherwise `None` will be returned.
    """
    if varname not in self._source_update_fn_lib.keys():
      return None

    return self._source_update_fn_lib[varname]

  @property
  def preprocessing_states_update_fn(self):
    """A function that preprocesses `states` and/or `additional_states`.

    `states` and/or `additional_states` will be updated by an externally
    provided function before the simulation update starts. This preprocess
    function can be called once at a specific step or periodically.

    Returns:
      The function that updates `states` and `additional_states`, which takes
      the following arguments:
      `kernel_op`, `states`, `additional_states`, `params`,
      and it should returns a dictionary containing the updated `states` and/or
      `additional_states`.
    """
    return self._preprocessing_states_update_fn

  @property
  def postprocessing_states_update_fn(self):
    """A function that postprocesses `states` and/or `additional_states`.

    `states` and/or `additional_states` will be updated by an externally
    provided function after the simulation step. This preprocess function can
    be called once at a specific step or periodically.

    Returns:
      The function that updates `states` and `additional_states`, which takes
      the following arguments:
      `kernel_op`, `states`, `additional_states`, `params`,
     and it should return a dictionary containing the updated `states` and/or
      `additional_states`.
    """
    return self._postprocessing_states_update_fn

  @property
  def kernel_op(self):
    """A shallow copy of the `ApplyKernelOp` instance."""
    return copy.copy(self._kernel_op)

  @preprocessing_states_update_fn.setter
  def preprocessing_states_update_fn(self, update_fn):
    """Sets the function that preprocesses `states`."""
    self._preprocessing_states_update_fn = update_fn

  @postprocessing_states_update_fn.setter
  def postprocessing_states_update_fn(self, update_fn):
    """Sets the function that postprocesses `states`."""
    self._postprocessing_states_update_fn = update_fn

  def maybe_grid_vertical(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
  ) -> types.FlowFieldVal:
    """The vertical grid local to `replica_id` if `g_dim` is not None.

    Note that this function supports flow field variables that are represented
    as List[tf.Tensor] only.

    Args:
      replica_id: The index of the current replica.
      replicas: A 3D tensor that saves the topology of the partitioning.

    Returns:
      The local grid (with halo) along the gravity direction. If no gravity
      direction is specified, returns zeros that has the same structure as the
      flow field.
    """
    if self.g_dim is not None:
      if self.g_dim == 0:
        grid_vertical = self.x_local_ext(replica_id, replicas)
        return grid_vertical[tf.newaxis, :, tf.newaxis]
      elif self.g_dim == 1:
        grid_vertical = self.y_local_ext(replica_id, replicas)
        return grid_vertical[tf.newaxis, tf.newaxis, :]
      elif self.g_dim == 2:
        grid_vertical = self.z_local_ext(replica_id, replicas)
        return grid_vertical[:, tf.newaxis, tf.newaxis]
      else:
        raise ValueError(
            f'If set, g_dim should be 0, 1, or 2 but is {self.g_dim}.')
    else:
      logging.info('Gravity direction is not set, grid vertical will be 0.')
      return tf.zeros((self.nz, 1, 1), dtype=types.TF_DTYPE)

  def diffusivity(self, scalar_name: str) -> float:
    """Retrieves the diffusivity of a scalar.

    Args:
      scalar_name: The name of the scalar with which the diffusivity is
        associated.

    Returns:
      The diffusivity of the scalar named by the input.

    Raises:
      ValueError: If the input scalar name is not part of the flow system.
    """
    for scalar in self.scalars:
      if scalar_name == scalar.name:
        return scalar.diffusivity

    raise ValueError(
        '{} is not in the flow field. Valid scalars are {}.'.format(
            scalar_name, self.scalars_names))

  def density(self, scalar_name: str) -> float:
    """Retrieves the density of a scalar.

    Args:
      scalar_name: The name of the scalar with which the density is associated.

    Returns:
      The density of the scalar named by the input.

    Raises:
      ValueError: If the input scalar name is not part of the flow system.
    """
    for scalar in self.scalars:
      if scalar_name == scalar.name:
        return scalar.density

    raise ValueError(
        '{} is not in the flow field. Valid scalars are {}.'.format(
            scalar_name, self.scalars_names))

  def molecular_weight(self, scalar_name: str) -> float:
    """Retrieves the molecular_weight of a scalar.

    Args:
      scalar_name: The name of the scalar with which the density is associated.

    Returns:
      The molecular weight of the scalar named by the input.

    Raises:
      ValueError: If the input scalar name is not part of the flow system.
    """
    for scalar in self.scalars:
      if scalar_name == scalar.name:
        return scalar.molecular_weight

    raise ValueError(
        '{} is not in the flow field. Valid scalars are {}.'.format(
            scalar_name, self.scalars_names))

  def scalar_time_integration_scheme(self,
                                     scalar_name: str) -> TimeIntegrationScheme:
    """Retrieves the time integration scheme of a scalar.

    Args:
      scalar_name: The name of the scalar with which the time integration scheme
        is associated.

    Returns:
      The time integration scheme of the scalar named by the input, falling back
        to the shared scheme if absent.
    """
    for scalar in self.scalars:
      if scalar_name == scalar.name:
        if scalar.HasField('time_integration_scheme'):
          return scalar.time_integration_scheme
        else:
          break

    return self.time_integration_scheme

  def save_to_file(self, output_prefix: str) -> None:
    """Saves configuration protos as text to files."""
    output_dir = f'{output_prefix}_config_files'
    tf.io.gfile.makedirs(output_dir)
    with tf.io.gfile.GFile(get_cmdline_json_path(output_prefix), 'w') as f:
      f.write(json.dumps(sys.argv))
    with tf.io.gfile.GFile(get_swirl_lm_pbtxt_path(output_prefix), 'w') as f:
      f.write(text_format.MessageToString(self.swirl_lm_parameters_proto))
    file_io.copy_files(
        file_io.find_referred_files(self.swirl_lm_parameters_proto),
        output_dir)


def get_swirl_lm_pbtxt_path(output_prefix: str) -> str:
  """Returns the path where SwirlLMParams proto is saved."""
  return f'{output_prefix}_swirl_lm.pbtxt'


def get_cmdline_json_path(output_prefix: str) -> str:
  """Returns the path where SwirlLMParams proto is saved."""
  return f'{output_prefix}_cmdline_args.json'


def load_params_from_output_dir(output_prefix: str) -> SwirlLMParameters:
  """Loads configuration proto from location determined by `output_prefix`."""
  swirl_lm_pbtxt_path = get_swirl_lm_pbtxt_path(output_prefix)
  with tf.io.gfile.GFile(swirl_lm_pbtxt_path, 'r') as f:
    text_proto = f.read()
  try:
    return SwirlLMParameters(parse_text_proto(text_proto))
  except Exception:
    logging.error('Exception while parsing "%s".', swirl_lm_pbtxt_path)
    raise


def params_from_config_file_flag() -> SwirlLMParameters:
  """Returns parameters loaded from --config_filepath flag."""
  if not FLAGS.config_filepath:
    raise ValueError('Flag --config_filepath is not set.')

  return SwirlLMParameters.config_from_proto(FLAGS.config_filepath)


def _validate_config_for_stretched_grid(
    config: parameters_pb2.SwirlLMParameters,
    use_stretched_grid: tuple[bool, bool, bool],
    global_xyz: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    g_dim: Literal[0, 1, 2] | None,
) -> None:
  """Validates the config for features available with stretched grid.

  Caution: This is a list of *known* features that are not yet supported, but
  the list is not necessarily exhaustive.

  Args:
    config: An instance of the `SwirlLMParameters` proto.
    use_stretched_grid: A tuple of booleans indicating whether the grid is
      stretched in each dimension.
    global_xyz: A tuple of tensors representing the global coordinates in each
      dimension, excluding halos.
    g_dim: The dimension along which the gravity force acts. If not None, then
      buoyancy is present.

  Raises:
    NotImplementedError: If the config has features turned on that are not
    supported by stretched grid.
    ValueError: If gravity is present, the pressure-buoyancy-balancing boundary
      condition is used, and the 2nd coordinate level is not exactly 3 times the
      first coordinate level.
  """
  if (
      g_dim is not None
      and use_stretched_grid[g_dim]
      and config.pressure.vertical_bc_treatment
      == pressure_pb2.Pressure.VerticalBCTreatment.PRESSURE_BUOYANCY_BALANCING
  ):
    coord = global_xyz[g_dim]
    # If gravity is present and the pressure-buoyancy-balancing boundary
    # condition is used, it is required that z0=coord[0] is the height
    # above the ground of the first level, and that the first halo node has
    # coordinate value of z_{-1} = -coord[0]. Let z1=coord[1]. The requirement
    # for boundary conditions is that z1 - z0 = z0 - z_{-1}. Thus, z1 = 3 * z0
    # is required.
    if not np.isclose(coord[1], 3 * coord[0], rtol=1e-4):
      raise ValueError(
          'When using stretched grid in the direction with gravity, the second'
          ' coordinate level must be exactly 3 times the first coordinate'
          ' level.'
      )

  if config.HasField('boundary_models') and config.boundary_models.HasField(
      'ib'
  ):
    raise NotImplementedError(
        'Immersed boundary method is not yet supported with stretched grid.'
    )

  if config.enable_rhie_chow_correction:
    raise NotImplementedError(
        'Rhie-Chow correction is not supported with stretched grid.'
    )

  if (
      config.diffusion_scheme
      == numerics_pb2.DiffusionScheme.DIFFUSION_SCHEME_STENCIL_3
  ):
    raise NotImplementedError(
        f'Diffusion scheme {config.diffusion_scheme} is not supported with'
        ' stretched grid.'
    )


def parse_text_proto(text_proto: str) -> parameters_pb2.SwirlLMParameters:
  return text_format.Parse(text_proto, parameters_pb2.SwirlLMParameters())
