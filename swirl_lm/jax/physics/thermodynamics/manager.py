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
"""Thermodynamics manager for JAX solver.

This is the JAX port of
`swirl_lm.physics.thermodynamics.thermodynamics_manager`.
It provides the density update interface for different thermodynamic models.

Supported modes:
  - ANELASTIC: Reference density is constant; thermodynamic density only
    affects buoyancy via the Boussinesq approximation.
  - LOW_MACH: Full density coupling with scalar transport via equation of
    state (ideal gas, linear mixing, or constant density).
  - Constant density: A special case where rho = const.
"""


import jax
import jax.numpy as jnp
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.communication import halo_exchange
from swirl_lm.jax.communication import halo_exchange_utils
from swirl_lm.jax.numerics import filters
from swirl_lm.jax.physics.thermodynamics import models as thermodynamics_models
from swirl_lm.jax.utility import types
from swirl_lm.physics.thermodynamics import thermodynamics_pb2

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap


class ThermodynamicsManager:
  """Manager for thermodynamic models in the JAX solver.

  Provides a clean interface for density updates. The specific model
  (constant density, ideal gas, linear mixing) is determined from proto config.
  """

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
  ):
    """Initializes the thermodynamics manager.

    Args:
      params: The simulation parameters.
    """
    self._params = params
    self._thermo_config = params.thermodynamics

    if self._thermo_config is not None:
      self._solver_mode = self._thermo_config.solver_mode
      self._model_type = self._thermo_config.WhichOneof('thermodynamics_type')
    else:
      # Default: LOW_MACH with constant density (matching params default).
      self._solver_mode = thermodynamics_pb2.Thermodynamics.LOW_MACH
      self._model_type = 'constant_density'

    # Create the underlying thermodynamic model.
    self.model = _create_model(params, self._model_type)

  @property
  def solver_mode(self) -> int:
    """Returns the solver mode (ANELASTIC or LOW_MACH)."""
    return self._solver_mode

  @property
  def model_type(self) -> str | None:
    """Returns the thermodynamics model type string."""
    return self._model_type

  def rho_ref(
      self,
      ref_field: ScalarField,
      additional_states: ScalarFieldMap | None = None,
  ) -> ScalarField:
    """Returns the reference density field.

    Args:
      ref_field: A reference field for shape/device placement.
      additional_states: Helper variables.

    Returns:
      The reference density as a 3D array.
    """
    return self.model.rho_ref(ref_field, additional_states)

  def p_ref(
      self,
      zz: ScalarField,
      additional_states: ScalarFieldMap | None = None,
  ) -> ScalarField:
    """Returns the reference pressure field.

    Args:
      zz: The geopotential height field.
      additional_states: Helper variables.

    Returns:
      The reference pressure as a 3D array.
    """
    return self.model.p_ref(zz, additional_states)

  def update_thermal_density(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes the thermodynamic density from the equation of state.

    Args:
      states: Flow field variables.
      additional_states: Helper variables.

    Returns:
      The thermodynamic density field.
    """
    return self.model.update_density(states, additional_states)

  def update_density(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      mesh: jax.sharding.Mesh,
      states_0: ScalarFieldMap | None = None,
  ) -> tuple[ScalarField, ScalarField]:
    """Updates the density based on the thermodynamic model.

    For ANELASTIC mode: density = rho_ref (constant), drho = 0.
    For LOW_MACH mode: density is computed from the equation of state.
      If states_0 is provided, the density change is filtered (2nd order)
      and applied incrementally to avoid spurious heat release from
      dispersion errors in scalar convection.

    Args:
      states: Flow field variables. Must include 'rho'.
      additional_states: Helper variables.
      mesh: JAX mesh for device topology.
      states_0: Optional flow field variables at the previous time step.

    Returns:
      Tuple of (rho, drho) where drho is the density change from the
      previous step (zero for constant density and anelastic).

    Raises:
      ValueError: If 'rho' is not in states.
    """
    if 'rho' not in states:
      raise ValueError('"rho" is not found in `states`.')

    rho_field = states['rho']

    if self._solver_mode == thermodynamics_pb2.Thermodynamics.LOW_MACH:
      rho = self.model.update_density(states, additional_states)

      if states_0 is not None:
        if 'rho' not in states_0:
          raise ValueError(
              'Density change from state_0 is requested but "rho" is not found.'
          )
        # Filter the density change to eliminate spurious heat release
        # caused by dispersion errors in scalar convection.
        drho_raw = rho - states_0['rho']
        drho_filtered = filters.filter_op(
            self._params.kernel_op,
            self._params.grid_params,
            drho_raw,
            additional_states,
            order=2,
        )
        # Apply homogeneous Neumann BC on drho and halo exchange.
        bc_neumann: tuple[
            tuple[
                tuple[halo_exchange_utils.BCType, float] | None,
                tuple[halo_exchange_utils.BCType, float] | None,
            ]
            | None,
            ...,
        ] = tuple(
            (
                (halo_exchange_utils.BCType.NEUMANN, 0.0),
                (halo_exchange_utils.BCType.NEUMANN, 0.0),
            )
            for _ in range(3)
        )
        drho = halo_exchange.inplace_halo_exchange(
            drho_filtered,
            ('x', 'y', 'z'),
            mesh,
            self._params.grid_params,
            list(self._params.grid_params.to_xyz_order(self._params.periodic_dims)),  # pyrefly: ignore[bad-argument-type]
            bc_neumann,
        )
        rho = states_0['rho'] + drho
      else:
        drho = jnp.zeros_like(rho_field)

      return rho, drho

    elif self._solver_mode == thermodynamics_pb2.Thermodynamics.ANELASTIC:
      rho_ref = self.rho_ref(rho_field, additional_states)
      return rho_ref, jnp.zeros_like(rho_field)

    else:
      # Default: constant density.
      return rho_field, jnp.zeros_like(rho_field)


def _create_model(
    params: parameters_lib.SwirlLMParameters,
    model_type: str | None,
) -> thermodynamics_models.ThermodynamicModel:
  """Creates a thermodynamic model instance from the config.

  Args:
    params: The simulation parameters.
    model_type: The model type string from proto WhichOneof.

  Returns:
    A ThermodynamicModel instance.

  Raises:
    NotImplementedError: If the model type is not supported.
  """
  if model_type == 'constant_density' or model_type is None:
    return thermodynamics_models.ConstantDensity(params)
  elif model_type == 'linear_mixing':
    return thermodynamics_models.LinearMixing(params)
  elif model_type == 'ideal_gas_law':
    return thermodynamics_models.IdealGas(params)
  elif model_type == 'water':
    from swirl_lm.jax.physics.thermodynamics import water  # pylint: disable=g-import-not-at-top

    return water.Water(params)
  else:
    raise NotImplementedError(
        f'{model_type} is not a valid thermodynamics model.'
    )
