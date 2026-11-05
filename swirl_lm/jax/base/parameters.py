# Copyright 2026 The swirl_lm Authors.
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

# Copyright 2021 Google LLC
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
"""Parameters for the JAX incompressible Navier-Stokes solver.

This module provides `SwirlLMParameters`, the JAX equivalent of the TF
`swirl_lm.base.parameters.SwirlLMParameters`. It wraps the same
`SwirlLMParameters` proto and creates JAX-native runtime objects
(kernel_op, deriv_lib).

It is designed to support all simulation types (TGV, geophysical flows,
combustion, atmosphere, etc.) from day one by reusing the existing
`parameters.proto`.
"""


from typing import Callable, Sequence

from absl import logging
from google.protobuf import text_format
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
import numpy as np
from swirl_lm.base import parameters_pb2
from swirl_lm.jax.base import physical_variable_keys_manager
from swirl_lm.jax.boundary_condition import boundary_condition_utils
from swirl_lm.jax.communication import halo_exchange
from swirl_lm.jax.numerics import derivatives as jax_derivatives
from swirl_lm.jax.numerics import numerics_pb2
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization as grid_parametrization_lib
from swirl_lm.jax.utility import grid_parametrization_pb2 as jax_gp_pb2
from swirl_lm.jax.utility import types
from swirl_lm.physics.thermodynamics import thermodynamics_pb2

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

# Type aliases for callback signatures.
SourceUpdateFn = Callable[..., ScalarFieldMap]
SourceUpdateFnLib = dict[str, SourceUpdateFn]
AdditionalStatesUpdateFn = Callable[..., ScalarFieldMap]

# Aliases for proto enums to use short names.
SolverProcedure = parameters_pb2.SwirlLMParameters.SolverProcedureType
ConvectionScheme = numerics_pb2.ConvectionScheme
DiffusionScheme = numerics_pb2.DiffusionScheme
TimeIntegrationScheme = numerics_pb2.TimeIntegrationScheme
NumericalFlux = numerics_pb2.NumericalFlux
KernelOpType = parameters_pb2.SwirlLMParameters.KernelOpType

# Threshold for detecting gravity-aligned axis.
_G_THRESHOLD = 1e-6


def _convert_grid_params_to_jax(
    tf_gp: parameters_pb2.SwirlLMParameters,
) -> jax_gp_pb2.GridParametrization:
  """Converts TF GridParametrization proto to JAX GridParametrization proto.

  The TF and JAX protos share the same logical structure but use different
  coordinate naming conventions (dim_0/1/2 vs dim_x/y/z) and have slightly
  different field numbering.

  Args:
    tf_gp: The TF SwirlLMParameters proto containing grid_params.

  Returns:
    A JAX GridParametrization proto.
  """
  src = tf_gp.grid_params
  jax_gp = jax_gp_pb2.GridParametrization()

  # Computation shape.
  jax_gp.computation_shape.dim_x = src.computation_shape.dim_0
  jax_gp.computation_shape.dim_y = src.computation_shape.dim_1
  jax_gp.computation_shape.dim_z = src.computation_shape.dim_2

  # Grid lengths.
  jax_gp.length.dim_x = src.length.dim_0
  jax_gp.length.dim_y = src.length.dim_1
  jax_gp.length.dim_z = src.length.dim_2

  # Grid sizes per core.
  jax_gp.grid_size.dim_x = src.grid_size.dim_0
  jax_gp.grid_size.dim_y = src.grid_size.dim_1
  jax_gp.grid_size.dim_z = src.grid_size.dim_2

  # Physical full grid size.
  if src.HasField('physical_full_grid_size'):
    jax_gp.physical_full_grid_size.dim_x = src.physical_full_grid_size.dim_0
    jax_gp.physical_full_grid_size.dim_y = src.physical_full_grid_size.dim_1
    jax_gp.physical_full_grid_size.dim_z = src.physical_full_grid_size.dim_2

  # Scalar fields.
  jax_gp.halo_width = src.halo_width
  jax_gp.dt = src.dt
  jax_gp.kernel_size = src.kernel_size

  # Stretched grid files — copy path strings to avoid cross-proto type issues.
  if src.HasField('stretched_grid_files'):
    if src.stretched_grid_files.HasField('dim_0'):
      jax_gp.stretched_grid_files.dim_x.path = (
          src.stretched_grid_files.dim_0.path
      )
    if src.stretched_grid_files.HasField('dim_1'):
      jax_gp.stretched_grid_files.dim_y.path = (
          src.stretched_grid_files.dim_1.path
      )
    if src.stretched_grid_files.HasField('dim_2'):
      jax_gp.stretched_grid_files.dim_z.path = (
          src.stretched_grid_files.dim_2.path
      )

  # Periodic dimensions.
  if src.HasField('periodic'):
    jax_gp.periodic.dim_x = src.periodic.dim_0
    jax_gp.periodic.dim_y = src.periodic.dim_1
    jax_gp.periodic.dim_z = src.periodic.dim_2

  return jax_gp


def _get_gravity_direction(
    config: parameters_pb2.SwirlLMParameters,
) -> list[float]:
  """Extracts and normalizes the gravity direction vector from the config."""
  if config.HasField('gravity_direction'):
    gravity_direction = [
        config.gravity_direction.dim_0,
        config.gravity_direction.dim_1,
        config.gravity_direction.dim_2,
    ]
    g_magnitude = np.sqrt(sum(g**2 for g in gravity_direction))
    if g_magnitude > 0:
      gravity_direction = [g / g_magnitude for g in gravity_direction]
    else:
      gravity_direction = [0.0] * 3
  else:
    gravity_direction = [0.0] * 3
  return gravity_direction


def _validate_stretched_grid_config(
    config: parameters_pb2.SwirlLMParameters,
    use_stretched_grid: tuple[bool, ...],
    g_dim: int | None,
) -> None:
  """Validates features not yet supported with stretched grids.

  This is a port of TF `_validate_config_for_stretched_grid`. The checks
  are not exhaustive but cover the known-problematic combinations.

  Args:
    config: The SwirlLMParameters proto.
    use_stretched_grid: Per-dimension stretched grid flags.
    g_dim: Gravity-aligned dimension, or None.

  Raises:
    NotImplementedError: If an unsupported feature is enabled with stretched
      grid.
  """
  del g_dim  # Unused.

  if not any(use_stretched_grid):
    return

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


class SwirlLMParameters:
  """Parameters for the JAX incompressible Navier-Stokes solver.

  This class wraps the `SwirlLMParameters` proto and creates JAX-native runtime
  objects. It uses composition (has-a `GridParametrization`) rather than
  inheritance, to keep the grid and simulation parameter boundaries clear.

  The field names match the TF version wherever possible, so equation modules
  can use the same access patterns (e.g., `params.convection_scheme`,
  `params.nu`, `params.bc`).
  """

  def __init__(self, config: parameters_pb2.SwirlLMParameters):
    """Initializes the SwirlLMParameters from a proto.

    Args:
      config: An instance of the `SwirlLMParameters` proto.

    Raises:
      ValueError: If the kernel operator type is not recognized.
    """
    self.swirl_lm_parameters_proto = config

    # === Grid ===
    # Convert TF GridParametrization proto to JAX GridParametrization proto.
    jax_gp = _convert_grid_params_to_jax(config)
    self.grid_params = grid_parametrization_lib.GridParametrization(jax_gp)

    # === BC Manager ===
    self.bc_manager = (
        physical_variable_keys_manager.BoundaryConditionKeysHelper()
    )

    # === Kernel operator ===
    self.kernel_op_type = config.kernel_op_type
    if config.kernel_op_type in (
        KernelOpType.KERNEL_OP_CONV,
        KernelOpType.KERNEL_OP_UNKNOWN,
    ):
      self.kernel_op: get_kernel_fn.ApplyKernelOp = (
          get_kernel_fn.ApplyKernelConvOp(
              self.grid_params.kernel_size, self.grid_params
          )
      )
    elif config.kernel_op_type == KernelOpType.KERNEL_OP_SLICE:
      self.kernel_op = get_kernel_fn.ApplyKernelSliceOp(self.grid_params)
    else:
      raise ValueError(
          f'Unknown or unsupported kernel operator: {config.kernel_op_type}.'
          ' KERNEL_OP_MATMUL is not supported in JAX.'
      )

    # === Numerical schemes ===
    self.solver_procedure = config.solver_procedure
    self.convection_scheme = config.convection_scheme
    self.numerical_flux = config.numerical_flux
    self.diffusion_scheme = config.diffusion_scheme
    self.time_integration_scheme = config.time_integration_scheme
    self.diff_stab_crit: float | None = (
        config.diff_stab_crit if config.HasField('diff_stab_crit') else None
    )
    self.enable_scalar_recorrection = config.enable_scalar_recorrection
    self.enable_rhie_chow_correction = config.enable_rhie_chow_correction

    logging.info(
        'Convection scheme: %s, Diffusion scheme: %s, Time scheme: %s',
        ConvectionScheme.Name(self.convection_scheme),
        DiffusionScheme.Name(self.diffusion_scheme),
        TimeIntegrationScheme.Name(self.time_integration_scheme),
    )

    # === Derivatives ===
    self.deriv_lib = jax_derivatives.Derivatives(
        self.kernel_op, self.grid_params
    )

    # === Physics models (all optional) ===
    self.thermodynamics = (
        config.thermodynamics if config.HasField('thermodynamics') else None
    )
    self.radiative_transfer = (
        config.radiative_transfer
        if config.HasField('radiative_transfer')
        else None
    )
    self.microphysics = (
        config.microphysics if config.HasField('microphysics') else None
    )
    self.lpt = config.lpt if config.HasField('lpt') else None
    self.combustion = (
        config.combustion if config.HasField('combustion') else None
    )

    # Solver mode: LOW_MACH (default) or ANELASTIC.
    if self.thermodynamics is not None and self.thermodynamics.HasField(
        'solver_mode'
    ):
      self.solver_mode = self.thermodynamics.solver_mode
    else:
      self.solver_mode = thermodynamics_pb2.Thermodynamics.LOW_MACH

    # === SGS / Turbulence ===
    self.use_sgs = config.use_sgs
    self.sgs_model = config.sgs_model if config.HasField('sgs_model') else None

    # === Gravity ===
    self.gravity_direction = _get_gravity_direction(config)
    g_dim = np.unique(
        np.nonzero(np.abs(np.abs(self.gravity_direction) - 1.0) < _G_THRESHOLD)
    )
    assert len(g_dim) <= 1, (
        'Gravity dimension is ambiguous if it is not aligned with an axis.'
        f' {g_dim} is provided.'
    )
    self.g_dim: int | None = int(g_dim.item()) if len(g_dim) == 1 else None

    # === Physical constants ===
    self.rho = config.density
    self.nu = config.kinematic_viscosity
    self.p_thermal = config.p_thermal

    # === Scalars ===
    self.scalars = list(config.scalars)
    self.scalar_lib = {scalar.name: scalar for scalar in self.scalars}

    # === Boundary conditions ===
    self.bc: boundary_condition_utils.BoundaryConditionDict = {
        'u': None,
        'v': None,
        'w': None,
        'p': None,
    }
    self.bc.update({scalar.name: None for scalar in self.scalars})
    self.bc_params: dict[str, list | None] = {  # pylint: disable=g-bare-generic
        'u': None,
        'v': None,
        'w': None,
        'p': None,
    }
    self.bc_params.update({scalar.name: None for scalar in self.scalars})

    for input_bc in config.boundary_conditions:
      bc, bc_params = self._parse_boundary_conditions(input_bc.boundary_info)
      self.bc[input_bc.name] = bc  # pyrefly: ignore[unsupported-operation]
      self.bc_params[input_bc.name] = bc_params

    self.bc_type = boundary_condition_utils.find_bc_type(
        self.bc,
        list(self.grid_params.to_xyz_order(self.grid_params.periodic_dims)),
    )
    logging.info('Boundary conditions from the input: %r', self.bc)

    # Additional keys for nonreflecting BCs.
    self.bc_keys = boundary_condition_utils.get_keys_for_boundary_condition(
        self.bc, halo_exchange.BCType.NONREFLECTING
    )

    # === Pressure solver ===
    self.corrector_nit = config.num_sub_iterations
    self.pressure = config.pressure if config.HasField('pressure') else None

    # === Boundary models ===
    self.boundary_models = (
        config.boundary_models if config.HasField('boundary_models') else None
    )

    if config.HasField('boundary_models') and config.boundary_models.sponge:
      self.sponge = list(config.boundary_models.sponge)
    elif config.HasField('sponge_layer'):
      self.sponge = [config.sponge_layer]
    else:
      self.sponge = None

    # === State key management ===
    self.additional_state_keys = list(config.additional_state_keys)
    self.additional_state_keys.extend(self.bc_keys)
    self.helper_var_keys = list(config.helper_var_keys)
    self.debug_variables = list(config.debug_variables)
    self.states_from_file = list(config.states_from_file)
    self.states_to_file = list(config.states_to_file)

    # === Monitoring ===
    self.monitor_spec = (
        config.monitor_spec if config.HasField('monitor_spec') else None
    )
    self.probe = config.probe if config.HasField('probe') else None

    # === Callbacks (set externally by simulation apps) ===
    self._additional_states_update_fn: AdditionalStatesUpdateFn | None = None
    self._source_update_fn_lib: SourceUpdateFnLib = {}
    self._preprocessing_states_update_fn: Callable | None = None  # pylint: disable=g-bare-generic
    self._postprocessing_states_update_fn: Callable | None = None  # pylint: disable=g-bare-generic

    # === Diagnostics (set externally by simulation apps) ===
    # Whether the pressure solve reports its relative residual. Read at trace
    # time, so when it is false neither the extra operator application nor the
    # host callback that prints it enters the compiled graph at all.
    self.log_poisson_residual: bool = False

    # === Additional state partition specs and distributors ===
    self.additional_state_partition_specs: dict[str, P] = {}
    self.additional_state_distributors: dict[str, Callable] = {}  # pylint: disable=g-bare-generic

    # === Pre/Post Processing Options ===
    self._parse_pre_post_process_info(config)

    # === Validation ===
    _validate_stretched_grid_config(config, self.use_stretched_grid, self.g_dim)

  def _parse_pre_post_process_info(
      self,
      config: parameters_pb2.SwirlLMParameters,
  ) -> None:
    """Parses pre/post process configuration from the proto.

    Mirrors TF `_parse_pre_post_process_options`. Supports both
    `from_config` proto option and defaults (disabled).

    Args:
      config: The SwirlLMParameters proto.
    """
    if (
        config.HasField('pre_post_process_info')
        and config.pre_post_process_info.WhichOneof('pre_post_process_option')
        == 'from_config'
    ):
      opt = config.pre_post_process_info.from_config
      self._apply_preprocess = opt.apply_preprocess
      self._preprocess_step_id = opt.preprocess_step_id
      self._preprocess_periodic = opt.preprocess_periodic
      self._apply_postprocess = opt.apply_postprocess
      self._postprocess_step_id = opt.postprocess_step_id
      self._postprocess_periodic = opt.postprocess_periodic
    else:
      # Defaults: pre/post processing disabled.
      self._apply_preprocess = False
      self._preprocess_step_id = 0
      self._preprocess_periodic = False
      self._apply_postprocess = False
      self._postprocess_step_id = 0
      self._postprocess_periodic = False

  def _parse_boundary_info(
      self,
      boundary_info: object,
  ) -> tuple[tuple[halo_exchange.BCType, float] | None, object | None]:
    """Retrieves the boundary condition from a proto BoundaryInfo."""
    bc_type = boundary_info.type  # pyrefly: ignore[missing-attribute]
    bc_type_value = None

    if bc_type == 1:  # BC_TYPE_DIRICHLET
      bc_type_value = (halo_exchange.BCType.DIRICHLET, boundary_info.value)  # pyrefly: ignore[missing-attribute]
    elif bc_type == 2:  # BC_TYPE_NEUMANN
      bc_type_value = (halo_exchange.BCType.NEUMANN, boundary_info.value)  # pyrefly: ignore[missing-attribute]
    elif bc_type == 5:  # BC_TYPE_NEUMANN_2
      bc_type_value = (halo_exchange.BCType.NEUMANN_2, boundary_info.value)  # pyrefly: ignore[missing-attribute]
    elif bc_type == 3:  # BC_TYPE_NO_TOUCH
      bc_type_value = (halo_exchange.BCType.NO_TOUCH, 0.0)
    elif bc_type == 6:  # BC_TYPE_NONREFLECTING
      bc_type_value = (
          halo_exchange.BCType.NONREFLECTING,
          boundary_info.value,  # pyrefly: ignore[missing-attribute]
      )

    bc_params = (
        boundary_info.bc_params  # pyrefly: ignore[missing-attribute]
        if boundary_info.HasField('bc_params')  # pyrefly: ignore[missing-attribute]
        else None
    )
    return bc_type_value, bc_params

  def _parse_boundary_conditions(
      self,
      boundary_conditions: Sequence[object],
  ) -> tuple[list, list]:  # pylint: disable=g-bare-generic
    """Parses the boundary conditions from a repeated BoundaryInfo."""
    bc: list[list] = [[None, None], [None, None], [None, None]]  # pylint: disable=g-bare-generic
    bc_params: list[list] = [[None, None], [None, None], [None, None]]  # pylint: disable=g-bare-generic
    for bc_info in boundary_conditions:
      dim = bc_info.dim  # pyrefly: ignore[missing-attribute]
      location = bc_info.location  # pyrefly: ignore[missing-attribute]
      bc[dim][location], bc_params[dim][location] = self._parse_boundary_info(
          bc_info
      )
    return bc, bc_params

  # ---------------------------------------------------------------------------
  # Factory methods
  # ---------------------------------------------------------------------------

  @classmethod
  def config_from_text_proto(
      cls,
      text_proto: str,
  ) -> 'SwirlLMParameters':
    """Parses a text proto into SwirlLMParameters.

    Args:
      text_proto: The text-format `SwirlLMParameters` proto.

    Returns:
      A `SwirlLMParameters` instance.

    Raises:
      ValueError: If required fields are missing.
      NotImplementedError: If solver procedure or schemes are unspecified.
    """
    config = text_format.Parse(text_proto, parameters_pb2.SwirlLMParameters())

    if not config.HasField('grid_params'):
      raise ValueError(
          'The `grid_params` field is required in the config proto.'
      )

    if config.solver_procedure not in (
        SolverProcedure.SEQUENTIAL,
        SolverProcedure.VARIABLE_DENSITY,
    ):
      raise NotImplementedError(
          'Solver procedure must be SEQUENTIAL or VARIABLE_DENSITY.'
      )

    if config.convection_scheme == ConvectionScheme.CONVECTION_SCHEME_UNKNOWN:
      raise NotImplementedError('Convection scheme is not specified.')

    if (
        config.time_integration_scheme
        == TimeIntegrationScheme.TIME_SCHEME_UNKNOWN
    ):
      raise NotImplementedError('Time integration scheme is not specified.')

    return cls(config)

  # ---------------------------------------------------------------------------
  # Properties — same API as TF version
  # ---------------------------------------------------------------------------

  @property
  def scalars_names(self) -> list[str]:
    """Names of all scalars in the flow system."""
    return [scalar.name for scalar in self.scalars]

  @property
  def transport_scalars_names(self) -> list[str]:
    """Names of transported (solved) scalars."""
    return [scalar.name for scalar in self.scalars if scalar.solve_scalar]

  def diffusivity(self, scalar_name: str) -> float:
    """Retrieves the diffusivity of a scalar.

    Args:
      scalar_name: The name of the scalar.

    Returns:
      The diffusivity of the named scalar.

    Raises:
      ValueError: If the scalar name is not in the flow system.
    """
    for scalar in self.scalars:
      if scalar_name == scalar.name:
        return scalar.diffusivity
    raise ValueError(
        f'{scalar_name} is not in the flow field. Valid scalars are'
        f' {self.scalars_names}.'
    )

  def density(self, scalar_name: str) -> float:
    """Retrieves the density associated with a scalar.

    Args:
      scalar_name: The name of the scalar.

    Returns:
      The density of the named scalar.

    Raises:
      ValueError: If the scalar name is not in the flow system.
    """
    for scalar in self.scalars:
      if scalar_name == scalar.name:
        return scalar.density
    raise ValueError(
        f'{scalar_name} is not in the flow field. Valid scalars are'
        f' {self.scalars_names}.'
    )

  def molecular_weight(self, scalar_name: str) -> float:
    """Retrieves the molecular weight associated with a scalar.

    Args:
      scalar_name: The name of the scalar.

    Returns:
      The molecular weight of the named scalar.

    Raises:
      ValueError: If the scalar name is not in the flow system.
    """
    for scalar in self.scalars:
      if scalar_name == scalar.name:
        return scalar.molecular_weight
    raise ValueError(
        f'{scalar_name} is not in the flow field. Valid scalars are'
        f' {self.scalars_names}.'
    )

  @property
  def max_halo_width(self) -> int:
    """Determines the halo width based on the selected convection scheme."""
    if self.convection_scheme == ConvectionScheme.CONVECTION_SCHEME_UPWIND_1:
      return max(1, self.grid_params.halo_width)
    elif self.convection_scheme == ConvectionScheme.CONVECTION_SCHEME_QUICK:
      return max(2, self.grid_params.halo_width)
    elif self.convection_scheme == ConvectionScheme.CONVECTION_SCHEME_CENTRAL_2:
      return max(1, self.grid_params.halo_width)
    elif self.convection_scheme == ConvectionScheme.CONVECTION_SCHEME_CENTRAL_4:
      return max(2, self.grid_params.halo_width)
    else:
      raise ValueError(
          'Halo width is ambiguous because convection scheme is not recognized.'
      )

  # --- Convenience delegations to grid_params ---

  @property
  def halo_width(self) -> int:
    """The halo width from grid parameters."""
    return self.grid_params.halo_width

  @property
  def dt(self) -> float:
    """The time step."""
    return self.grid_params.dt

  @property
  def periodic_dims(self) -> tuple[bool, ...]:
    """Whether each dimension is periodic."""
    return self.grid_params.periodic_dims

  @property
  def grid_spacings(self) -> tuple[float, ...]:
    """Grid spacings (dx, dy, dz)."""
    return self.grid_params.grid_spacings

  @property
  def use_stretched_grid(self) -> tuple[bool, ...]:
    """Whether stretched grid is used per tensor axis, ordered in data_axis_order."""
    return self.grid_params.use_stretched_grid

  @property
  def global_xyz(self) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Global coordinates along each dimension, stored in data_axis_order."""
    return self.grid_params.global_xyz

  @property
  def global_xyz_with_halos(self) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Global coordinates along each dimension with halos, in data_axis_order."""
    return self.grid_params.global_xyz_with_halos

  # --- Callbacks ---

  @property
  def additional_states_update_fn(self) -> AdditionalStatesUpdateFn | None:
    """A function that updates the additional states."""
    return self._additional_states_update_fn

  @additional_states_update_fn.setter
  def additional_states_update_fn(self, fn: AdditionalStatesUpdateFn) -> None:
    """Sets the function that updates the additional states."""
    self._additional_states_update_fn = fn

  @property
  def source_update_fn_lib(self) -> SourceUpdateFnLib:
    """A library of functions that update source terms by variable name."""
    return self._source_update_fn_lib

  @source_update_fn_lib.setter
  def source_update_fn_lib(self, source_lib: SourceUpdateFnLib) -> None:
    """Sets the library of functions that update source terms."""
    self._source_update_fn_lib = source_lib

  def source_update_fn(self, varname: str) -> SourceUpdateFn | None:
    """Retrieves the source term update function for `varname` if available."""
    return self._source_update_fn_lib.get(varname)

  @property
  def preprocessing_states_update_fn(self) -> Callable | None:  # pylint: disable=g-bare-generic
    """A function that preprocesses states before the simulation step."""
    return self._preprocessing_states_update_fn

  @preprocessing_states_update_fn.setter
  def preprocessing_states_update_fn(self, fn: Callable) -> None:  # pylint: disable=g-bare-generic
    """Sets the preprocessing function."""
    self._preprocessing_states_update_fn = fn

  @property
  def postprocessing_states_update_fn(self) -> Callable | None:  # pylint: disable=g-bare-generic
    """A function that postprocesses states after the simulation step."""
    return self._postprocessing_states_update_fn

  @postprocessing_states_update_fn.setter
  def postprocessing_states_update_fn(self, fn: Callable) -> None:  # pylint: disable=g-bare-generic
    """Sets the postprocessing function."""
    self._postprocessing_states_update_fn = fn

  # --- Pre/Post Processing Flags ---

  @property
  def apply_preprocess(self) -> bool:
    """Whether pre-processing is applied."""
    return self._apply_preprocess

  @property
  def apply_postprocess(self) -> bool:
    """Whether post-processing is applied."""
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
    """Whether preprocess function is applied periodically."""
    return self._preprocess_periodic

  @property
  def postprocess_periodic(self) -> bool:
    """Whether postprocess function is applied periodically."""
    return self._postprocess_periodic

  def scalar_time_integration_scheme(self, scalar_name: str) -> int:
    """Retrieves the time integration scheme for a scalar.

    Falls back to the shared (global) scheme if the scalar does not specify
    its own.

    Args:
      scalar_name: The name of the scalar.

    Returns:
      The time integration scheme for the named scalar.
    """
    for scalar in self.scalars:
      if scalar_name == scalar.name:
        if scalar.HasField('time_integration_scheme'):
          return scalar.time_integration_scheme
        break
    return self.time_integration_scheme

  def maybe_grid_vertical(self) -> ScalarField:
    """Returns the vertical grid coordinates as a broadcastable 3D array.

    If gravity is configured (g_dim is not None), returns the global grid
    coordinates along the gravity direction, shaped so that they broadcast
    correctly with 3D fields (e.g., shape (N, 1, 1) if the gravity dimension
    is the first data axis). If no gravity direction is specified, returns
    zeros with shape (nz+2*halo, 1, 1).

    Returns:
      A 3D array of vertical grid coordinates.
    """
    gp = self.grid_params
    if self.g_dim is not None:
      # g_dim is in physical (x=0, y=1, z=2) order. Convert to data axis.
      axis_name = ('x', 'y', 'z')[self.g_dim]
      data_dim = gp.data_axis_order.index(axis_name)
      coords_1d = jnp.array(gp.global_xyz[data_dim])
      # Shape for broadcasting: size along data_dim, 1 elsewhere.
      shape = [1, 1, 1]
      shape[data_dim] = coords_1d.shape[0]
      return coords_1d.reshape(shape)
    else:
      logging.info('Gravity direction is not set, grid vertical will be 0.')
      n = gp.nz + 2 * gp.halo_width
      return jnp.zeros((n, 1, 1))
