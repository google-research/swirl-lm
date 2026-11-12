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
"""A library of the immersed boundary method for the JAX solver.

This is the JAX port of `swirl_lm.boundary_condition.immersed_boundary_method`.
It provides Rayleigh damping (sponge layer) and direct forcing methods for
enforcing no-slip or target-value boundary conditions on immersed surfaces.

Supported IB types:
  - sponge: Rayleigh damping forcing term inside the solid region.
  - direct_forcing: Modifies the RHS to force values toward a target in solid.
  - cartesian_grid: Stub that returns states unchanged (full port requires
      multi-device halo exchange patterns).
  - mac: Stub (full port requires multi-device support).
  - direct_forcing_1d_interp: Stub (full port requires kernel_op patterns).
  - feedback_force_1d_interp: Stub (full port requires kernel_op patterns).
"""


from typing import Optional

from absl import logging
import jax.numpy as jnp
import numpy as np
from swirl_lm.boundary_condition import immersed_boundary_method_pb2
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.boundary_condition import immersed_boundary_ghost_point as ghost_point
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap


def ib_info_map(
    ib_info: immersed_boundary_method_pb2.ImmersedBoundaryMethod,
) -> dict[str, immersed_boundary_method_pb2.IBVariableInfo]:
  """Flattens the variable information in the IB config.

  Args:
    ib_info: The immersed boundary method proto configuration.

  Returns:
    A dictionary mapping variable names to their IB configuration.
  """

  def flatten_ib_info(
      ib_vars: list[immersed_boundary_method_pb2.IBVariableInfo],
  ) -> dict[str, immersed_boundary_method_pb2.IBVariableInfo]:
    """Flattens a sequence of IB info."""
    return {info.name: info for info in ib_vars}

  ib_type = ib_info.WhichOneof('type')
  if ib_type == 'cartesian_grid':
    return flatten_ib_info(list(ib_info.cartesian_grid.variables))
  elif ib_type == 'mac':
    return flatten_ib_info(list(ib_info.mac.variables))
  elif ib_type == 'sponge':
    return flatten_ib_info(list(ib_info.sponge.variables))
  elif ib_type == 'direct_forcing':
    return flatten_ib_info(list(ib_info.direct_forcing.variables))
  elif ib_type == 'direct_forcing_1d_interp':
    return flatten_ib_info(list(ib_info.direct_forcing_1d_interp.variables))
  elif ib_type == 'feedback_force_1d_interp':
    return flatten_ib_info(list(ib_info.feedback_force_1d_interp.variables))
  else:
    raise NotImplementedError(
        f'{ib_type} is not a valid IB type. Available options are:'
        ' "cartesian_grid", "mac", "sponge", "direct_forcing", '
        '"direct_forcing_1d_interp", "feedback_force_1d_interp".'
    )


class ImmersedBoundaryMethod:
  """A library of the immersed boundary method for the JAX solver."""

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    """Initializes the immersed boundary method library.

    Args:
      params: The simulation parameters.
    """
    self._params = params
    assert (
        boundary_models := params.boundary_models
    ) is not None, '`boundary_models` must be set in the config.'
    self._ib_params = boundary_models.ib

  @property
  def type(self) -> str:
    """Provides the type of immersed boundary method for this instance."""
    return self._ib_params.WhichOneof('type')

  def update_states(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> dict[str, ScalarField]:
    """Updates `states` in the immersed boundary at each sub-iteration.

    Args:
      states: Flow field variables.
      additional_states: Helper variables including masks and forcing terms.

    Returns:
      Updated states dictionary.
    """
    if self.type in ('cartesian_grid', 'mac'):
      # Full cartesian_grid/mac methods require multi-device halo exchange.
      # For now, return states unchanged.
      logging.warning(
          'IB type %s is not fully supported in JAX single-device mode.'
          ' States returned unchanged.',
          self.type,
      )
      return dict(states)
    elif self.type == 'ghost_point':
      config = self._ib_params.ghost_point
      if (
          config.enforcement
          != immersed_boundary_method_pb2.GhostPointMethod.DIRECT
      ):
        # The forcing variant contributes through `update_forcing` instead.
        return dict(states)
      return self._apply_ghost_point(states, additional_states)
    elif self.type in (
        'sponge',
        'direct_forcing',
        'feedback_force_1d_interp',
        'direct_forcing_1d_interp',
    ):
      return dict(states)
    else:
      raise ValueError(f'{self.type} is not a valid IB type.')

  def update_additional_states(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> dict[str, ScalarField]:
    """Updates `additional_states` at the beginning of each time step.

    Args:
      states: Flow field variables.
      additional_states: Helper variables including masks and forcing terms.

    Returns:
      Updated additional_states dictionary.
    """
    if self.type in (
        'cartesian_grid',
        'mac',
        'direct_forcing',
        'direct_forcing_1d_interp',
        # The ghost-point geometry is static, so nothing is recomputed here.
        'ghost_point',
    ):
      return dict(additional_states)
    elif self.type == 'sponge':
      return self._apply_rayleigh_damping_method(states, additional_states)
    elif self.type == 'feedback_force_1d_interp':
      # Full implementation requires kernel_op patterns.
      logging.warning(
          'feedback_force_1d_interp is not fully supported in JAX.'
          ' Additional states returned unchanged.'
      )
      return dict(additional_states)
    else:
      raise ValueError(f'{self.type} is not a valid IB type.')

  def update_forcing(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> dict[str, ScalarField]:
    """Updates `additional_states` during each subiteration.

    Args:
      states: Flow field variables.
      additional_states: Helper variables.

    Returns:
      Updated additional_states dictionary.
    """
    if self.type in (
        'cartesian_grid',
        'mac',
        'sponge',
        'feedback_force_1d_interp',
    ):
      return dict(additional_states)
    elif self.type == 'ghost_point':
      config = self._ib_params.ghost_point
      if (
          config.enforcement
          != immersed_boundary_method_pb2.GhostPointMethod.FORCING
      ):
        return dict(additional_states)
      return self._apply_ghost_point_forcing(states, additional_states)
    elif self.type == 'direct_forcing':
      return self._apply_direct_forcing_method(states, additional_states)
    elif self.type == 'direct_forcing_1d_interp':
      # Full implementation requires kernel_op patterns.
      logging.warning(
          'direct_forcing_1d_interp is not fully supported in JAX.'
          ' Additional states returned unchanged.'
      )
      return dict(additional_states)
    else:
      raise ValueError(f'{self.type} is not a valid IB type.')

  def generate_initial_states(
      self,
      grid_shape: tuple[int, int, int],
      ib_interior_mask: ScalarField,
      ib_boundary_mask: Optional[ScalarField] = None,
  ) -> dict[str, ScalarField]:
    """Generates initial states required by the IB model.

    Args:
      grid_shape: The shape of the 3D computational grid (nz, nx, ny).
      ib_interior_mask: A 3D array with 1 in fluid and 0 in solid.
      ib_boundary_mask: Optional 3D array with 1 at fluid-solid interface.

    Returns:
      A dictionary of required states by the selected IB method.
    """
    zeros = jnp.zeros(grid_shape, dtype=ib_interior_mask.dtype)

    output: dict[str, ScalarField] = {'ib_interior_mask': ib_interior_mask}

    if self.type == 'sponge':
      ib_boundary_included = False
      for variable in self._ib_params.sponge.variables:
        force_name = self.ib_force_name(variable.name)
        output[force_name] = zeros

        if (
            not ib_boundary_included
            and variable.bc
            == immersed_boundary_method_pb2.IBVariableInfo.NEUMANN_Z
        ):
          output['ib_boundary'] = (
              ib_boundary_mask if ib_boundary_mask is not None else zeros
          )
          ib_boundary_included = True

    elif self.type == 'feedback_force_1d_interp':
      if ib_boundary_mask is not None:
        output['ib_boundary'] = ib_boundary_mask
      for variable in self._ib_params.sponge.variables:
        force_name = self.ib_force_name(variable.name)
        output[force_name] = zeros

    elif self.type in ('cartesian_grid', 'mac'):
      if ib_boundary_mask is not None:
        output['ib_boundary'] = ib_boundary_mask

    elif self.type == 'ghost_point':
      # The geometry and any enabled output fields come from
      # `GhostPointGeometry.initial_states`, which the example calls. Only
      # the force buffers belong here, and only for the forcing variant.
      config = self._ib_params.ghost_point
      if (
          config.enforcement
          == immersed_boundary_method_pb2.GhostPointMethod.FORCING
      ):
        for variable in config.variables:
          output[self.ib_force_name(variable.name)] = zeros

    elif self.type == 'direct_forcing':
      for variable in self._ib_params.direct_forcing.variables:
        rhs_name = self.ib_rhs_name(variable.name)
        output[rhs_name] = zeros

    return output

  def _bc_name(self, variable) -> str:
    """Maps an `IBVariableInfo` boundary type onto a ghost-point condition."""
    info = immersed_boundary_method_pb2.IBVariableInfo
    if variable.bc == info.DIRICHLET:
      return 'dirichlet'
    if variable.bc == info.NEUMANN:
      return 'neumann'
    raise ValueError(
        f'The ghost-point method supports DIRICHLET and NEUMANN; variable '
        f'{variable.name!r} asks for '
        f'{info.BoundaryConditionType.Name(variable.bc)}. NEUMANN_Z assumes a '
        'single fluid-solid interface along z, which is exactly the terrain '
        'assumption this method exists to remove; use NEUMANN.'
    )

  def _apply_ghost_point(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> dict[str, ScalarField]:
    """Assigns the ghost-cell values, Ghias et al. (2007) Eq. 26.

    Args:
      states: Flow field variables.
      additional_states: Helper variables, carrying the geometry.

    Returns:
      `states` with each configured variable's ghost cells overwritten.
    """
    config = self._ib_params.ghost_point
    geom = ghost_point.from_additional_states(additional_states)
    updated = dict(states)

    clamp = None
    if config.clamp_interior:
      if ghost_point.INTERIOR_MASK_KEY not in additional_states:
        raise ValueError(
            '`clamp_interior` needs the fluid mask, so it also needs '
            f'`write_interior_mask`. {ghost_point.INTERIOR_MASK_KEY!r} is '
            'not in `additional_states`.'
        )
      clamp = ghost_point.deep_interior_mask(
          additional_states[ghost_point.INTERIOR_MASK_KEY], geom
      )

    for variable in config.variables:
      if variable.name not in updated:
        continue
      zeta, gamma = ghost_point.coefficients_for(
          geom, self._bc_name(variable), variable.value
      )
      updated[variable.name] = ghost_point.apply_local(
          updated[variable.name],
          geom,
          zeta,
          gamma,
          num_sweeps=config.num_sweeps,
      )
      if clamp is not None:
        # After the sweeps, so the interpolation still reads whatever the
        # previous application left in the deep cells rather than a value
        # this one just wrote.
        updated[variable.name] = jnp.where(
            clamp, variable.value, updated[variable.name]
        )
    return updated

  def _apply_ghost_point_forcing(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> dict[str, ScalarField]:
    """Relaxes towards the ghost-cell values instead of assigning them.

    Args:
      states: Flow field variables.
      additional_states: Helper variables, carrying the geometry.

    Returns:
      `additional_states` with each configured variable's force term set.
    """
    config = self._ib_params.ghost_point
    geom = ghost_point.from_additional_states(additional_states)
    updated = dict(additional_states)
    for variable in config.variables:
      if variable.name not in states:
        continue
      zeta, gamma = ghost_point.coefficients_for(
          geom, self._bc_name(variable), variable.value
      )
      damping = (
          variable.damping_coeff
          if variable.HasField('damping_coeff')
          else config.damping_coeff
      )
      updated[self.ib_force_name(variable.name)] = ghost_point.forcing(
          states[variable.name],
          geom,
          zeta,
          gamma,
          damping,
          num_sweeps=config.num_sweeps,
      )
    return updated

  @staticmethod
  def ib_force_name(var_name: str) -> str:
    """Generates the name of the force term for `var_name` in the solid.

    Args:
      var_name: The name of the variable.

    Returns:
      The name of the force term for the input variable.
    """
    return f'src_{var_name}'

  @staticmethod
  def ib_rhs_name(var_name: str) -> str:
    """Generates the name of the right hand side for `var_name` in the solid.

    Args:
      var_name: The name of the variable.

    Returns:
      The name of the right hand side for the input variable.
    """
    return f'rhs_{var_name}'

  def _apply_rayleigh_damping_method(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> dict[str, ScalarField]:
    """Generates the Rayleigh Damping forcing term inside the solid.

    Args:
      states: Flow field variables.
      additional_states: Helper variables including masks and forcing terms.

    Returns:
      Updated additional_states with sponge forces computed.
    """

    def update_sponge_force(
        value: ScalarField,
        target_value: ScalarField | float,
        original_force: ScalarField,
        damping_coeff: float,
        mask: ScalarField,
        override: bool,
    ) -> ScalarField:
      """Generates the sponge force for a variable within the solid."""
      a_max = np.power(damping_coeff * self._params.dt, -1)
      force = -a_max * (value - target_value) * (1.0 - mask)
      return force if override else original_force + force

    additional_states_new: dict[str, ScalarField] = dict(additional_states)

    for variable in self._ib_params.sponge.variables:
      if variable.name not in states:
        logging.warning(
            '%s is not a valid state. Available states are: %r',
            variable.name,
            list(states.keys()),
        )
        continue

      force_name = self.ib_force_name(variable.name)
      if force_name not in additional_states:
        raise ValueError(
            f'{force_name} needs to be initialized to use the Rayleigh '
            'damping approach of the immersed boundary method.'
        )

      # For NEUMANN_Z, the target value would come from the fluid-solid
      # interface. In single-device mode, we sum over the z-axis.
      if variable.bc == immersed_boundary_method_pb2.IBVariableInfo.NEUMANN_Z:
        # Extract the interface value: sum(value * ib_boundary) over z-axis.
        z_axis = self._params.grid_params.get_axis_index('z')
        interface_val = jnp.sum(
            states[variable.name] * additional_states['ib_boundary'],
            axis=z_axis,
        )
        target_value: ScalarField | float = jnp.expand_dims(
            interface_val, axis=z_axis
        )
      else:
        target_value = variable.value

      damping_coeff = (
          variable.damping_coeff
          if variable.HasField('damping_coeff')
          else self._ib_params.sponge.damping_coeff
      )

      additional_states_new[force_name] = update_sponge_force(
          states[variable.name],
          target_value,
          additional_states[force_name],
          damping_coeff,
          additional_states['ib_interior_mask'],
          variable.override,
      )

    return additional_states_new

  def _apply_direct_forcing_method(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> dict[str, ScalarField]:
    """Updates the equation right hand side with the direct forcing method.

    Reference:
    [1] Zhang, N., and Z. C. Zheng. 2007. "An Improved Direct-Forcing
       Immersed-Boundary Method for Finite Difference Applications." Journal of
       Computational Physics 221 (1): 250-68.

    Args:
      states: Field variables to which the immersed boundary method are applied.
      additional_states: Helper states that are required to compute the new
        right hand side function. Must contain 'ib_interior_mask' and
        'rhs_[w+]', where 'w+' is the name of the state.

    Returns:
      A dictionary of right hand side functions updated by the direct forcing
      immersed boundary method.

    Raises:
      ValueError: If 'ib_interior_mask' is not in `additional_states`, or no
        `rhs_[w+]` variable found for that variable.
    """
    if 'ib_interior_mask' not in additional_states:
      raise ValueError(
          '"ib_interior_mask" is not found in `additional_states`.'
      )

    def update_rhs(
        value: ScalarField,
        target_value: float,
        damping_coeff: float,
        rhs: ScalarField,
        mask: ScalarField,
    ) -> ScalarField:
      """Updates the right hand side function with direct forcing."""
      coeff = np.power(damping_coeff * self._params.dt, -1)
      return rhs * mask - coeff * (value - target_value) * (1.0 - mask)

    var_dict = {
        variable.name: variable
        for variable in self._ib_params.direct_forcing.variables
    }

    rhs_updated: dict[str, ScalarField] = {}

    for key, value in states.items():
      rhs_name = f'rhs_{key}'
      if rhs_name not in additional_states:
        raise ValueError(f'RHS for {key} is not provided.')

      if key not in var_dict:
        logging.warning(
            'States information for %s is not provided in the IB. Available '
            'states are: %r. Right hand side for %s is not updated and IB not '
            'applied.',
            key,
            list(var_dict.keys()),
            key,
        )
        rhs_updated[rhs_name] = additional_states[rhs_name]
      else:
        damping_coeff = (
            var_dict[key].damping_coeff
            if var_dict[key].HasField('damping_coeff')
            else self._ib_params.direct_forcing.damping_coeff
        )

        rhs_updated[rhs_name] = update_rhs(
            value,
            var_dict[key].value,
            damping_coeff,
            additional_states[rhs_name],
            additional_states['ib_interior_mask'],
        )

    return rhs_updated
