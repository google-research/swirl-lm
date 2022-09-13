# Copyright 2022 The swirl_lm Authors.
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

from typing import Callable, List, Mapping, Optional, Sequence, Tuple

from absl import flags
from absl import logging
import numpy as np
from swirl_lm.base import parameters_pb2
from swirl_lm.boundary_condition import boundary_condition_utils
from swirl_lm.boundary_condition import boundary_conditions_pb2
from swirl_lm.communication import halo_exchange
from swirl_lm.numerics import numerics_pb2
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import grid_parametrization_pb2
from swirl_lm.utility import types
import tensorflow as tf

from google.protobuf import text_format

flags.DEFINE_string(
    'config_filepath', None,
    'The full path to the text proto file that stores all input parameters.')
flags.DEFINE_bool(
    'simulation_debug',
    False,
    'Toggles if to run the simulation with the debug mode.',
    allow_override=True)

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

FLAGS = flags.FLAGS


class SwirlLMParameters(grid_parametrization.GridParametrization):
  """Parameters for running the incompressible Navier-Stokes solver."""

  def __init__(
      self,
      config,
      grid_params: Optional[
          grid_parametrization_pb2.GridParametrization] = None,
  ):
    super(SwirlLMParameters, self).__init__(grid_params)

    self.kernel_op_type = config.kernel_op_type

    self.solver_procedure = config.solver_procedure
    self.convection_scheme = config.convection_scheme
    self.diffusion_scheme = config.diffusion_scheme
    self.time_integration_scheme = config.time_integration_scheme

    self.enable_scalar_recorrection = config.enable_scalar_recorrection

    logging.info('Convection scheme: %s, Diffusion scheme: %s, Time scheme: %s',
                 ConvectionScheme.Name(self.convection_scheme),
                 DiffusionScheme.Name(self.diffusion_scheme),
                 TimeIntegrationScheme.Name(self.time_integration_scheme))

    self.thermodynamics = config.thermodynamics if config.HasField(
        'thermodynamics') else None

    self.combustion = config.combustion if config.HasField(
        'combustion') else None

    self.additional_state_keys = config.additional_state_keys
    self.helper_var_keys = config.helper_var_keys
    self.states_from_file = config.states_from_file
    self.monitor_spec = config.monitor_spec
    self.probe = config.probe if config.HasField('probe') else None

    self.use_sgs = config.use_sgs
    self.sgs_model = config.sgs_model

    # Initialize the direction vector of gravitational force. Set to 0 if
    # gravity is not defined in the simulation.
    if config.HasField('gravity_direction'):
      self.gravity_direction = [
          config.gravity_direction.dim_0, config.gravity_direction.dim_1,
          config.gravity_direction.dim_2
      ]
    else:
      self.gravity_direction = [0.] * 3

    # Get the scalar related quantities if scalars are solved as a
    # `List[SwirlLMParameters.Scalar]`.
    self.scalars = config.scalars

    self.scalar_lib = {scalar.name: scalar for scalar in self.scalars}

    # Boundary conditions.
    self.periodic_dims = [
        config.periodic.dim_0, config.periodic.dim_1, config.periodic.dim_2
    ]

    self.bc = {'u': None, 'v': None, 'w': None, 'p': None}
    self.bc.update({scalar.name: None for scalar in self.scalars})

    for input_bc in config.boundary_conditions:
      self.bc.update({
          input_bc.name: self._parse_boundary_conditions(input_bc.boundary_info)
      })

    # Find the type of boundary, e.g. wall, open boundary, periodic, from the
    # boundary conditions.
    self.bc_type = boundary_condition_utils.find_bc_type(
        self.bc, self.periodic_dims)
    logging.info('Boundary conditions from the input: %r', self.bc)

    logging.info('Boundary conditions for `u`, `v`, `w`, and all scalars are '
                 'retrieved from the config file. Boundary condition for `p` '
                 'is derived based on boundary types.')

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

    # Get the option for density upadte.
    self.density_update_option = config.density_update_option

    # Get the information required for sponge layers if applied.
    if (config.HasField('boundary_models') and
        config.boundary_models.HasField('sponge')):
      self.sponge = config.boundary_models.sponge
    elif config.HasField('sponge_layer'):
      self.sponge = config.sponge_layer
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

    # Toggle if to run with the debug mode.
    self.dbg = FLAGS.simulation_debug

  def __str__(self) -> str:
    return super(SwirlLMParameters, self).__str__() + (
        ', rho: {}, nu: {}, nit: {}'.format(self.rho, self.nu, self.nit))

  def _parse_boundary_info(
      self,
      boundary_info: _BCInfo) -> Optional[Tuple[halo_exchange.BCType, float]]:
    """Retrieves the boundary condition from proto to fit the framework."""
    if boundary_info.type == _BCType.BC_TYPE_DIRICHLET:
      return (halo_exchange.BCType.DIRICHLET, boundary_info.value)
    elif boundary_info.type == _BCType.BC_TYPE_NEUMANN:
      return (halo_exchange.BCType.NEUMANN, boundary_info.value)
    elif boundary_info.type == _BCType.BC_TYPE_NO_TOUCH:
      return (halo_exchange.BCType.NO_TOUCH, 0.0)
    else:
      return None

  def _parse_boundary_conditions(
      self, boundary_conditions: Sequence[_BCInfo]
  ) -> List[List[Optional[Tuple[halo_exchange.BCType, float]]]]:
    """Parses the boundary conditions.

    Args:
      boundary_conditions: A proto that stores all boundary information for a
        specific variable.

    Returns:
      bc: The boundary condition in the fluid framework format.
    """

    bc = [[None, None], [None, None], [None, None]]
    for bc_info in boundary_conditions:
      bc[bc_info.dim][bc_info.location] = self._parse_boundary_info(bc_info)
    return bc

  @staticmethod
  def config_from_text_proto(
      text_proto: str,
      grid_params: Optional[
          grid_parametrization_pb2.GridParametrization] = None,
  ) -> 'SwirlLMParameters':
    """Parses the config proto in text format into SwirlLMParameters."""
    config = text_format.Parse(text_proto, parameters_pb2.SwirlLMParameters())

    # Sanity check for the solver procedure.
    if config.solver_procedure not in (SolverProcedure.SEQUENTIAL,
                                       SolverProcedure.PREDICTOR_CORRECTOR,
                                       SolverProcedure.VARIABLE_DENSITY):
      raise NotImplementedError(
          'Solver procedure needs to be specified for a simulation.')

    # Sanity check for the numerical schemes.
    if config.convection_scheme == (ConvectionScheme.CONVECTION_SCHEME_UNKNOWN):
      raise NotImplementedError('Convection scheme is not specified.')

    if config.time_integration_scheme == (
        TimeIntegrationScheme.TIME_SCHEME_UNKNOWN):
      raise NotImplementedError('Time integration scheme is not specified.')

    return SwirlLMParameters(config, grid_params)

  @staticmethod
  def config_from_proto(
      config_filepath: str,
      grid_params: Optional[
          grid_parametrization_pb2.GridParametrization] = None,
  ) -> 'SwirlLMParameters':
    """Reads the config text proto file."""
    with tf.io.gfile.GFile(config_filepath, 'r') as f:
      text_proto = f.read()

    return SwirlLMParameters.config_from_text_proto(text_proto, grid_params)

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

  @preprocessing_states_update_fn.setter
  def preprocessing_states_update_fn(self, update_fn):
    """Sets the function that preprocesses `states`."""
    self._preprocessing_states_update_fn = update_fn

  @postprocessing_states_update_fn.setter
  def postprocessing_states_update_fn(self, update_fn):
    """Sets the function that postprocesses `states`."""
    self._postprocessing_states_update_fn = update_fn

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


def params_from_config_file_flag() -> SwirlLMParameters:
  """Returns parameters loaded from --config_filepath flag."""
  if not FLAGS.config_filepath:
    raise ValueError('Flag --config_filepath is not set.')

  return SwirlLMParameters.config_from_proto(FLAGS.config_filepath)
