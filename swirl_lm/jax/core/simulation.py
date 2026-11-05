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
"""Variable-density low Mach number Navier-Stokes solver (JAX).

This is the JAX port of `swirl_lm.core.simulation`. It implements the
predictor-corrector time-stepping loop for incompressible / low-Mach flows.

Stretched grid support:
  Stretched grid scale factors are provided via `additional_states` with keys
  defined in `stretched_grid_util` (e.g., `stretched_grid_h0`,
  `stretched_grid_h0_face`). They are consumed internally by the `Derivatives`
  library -- no other module needs to know whether a stretched grid is active.
  The `GridParametrization.physical_grid_spacing()` method provides a clean
  interface for any code that needs physical (non-uniform) grid spacings.

Supported features:
  - Velocity prediction + pressure correction loop.
  - Scalar transport (temperature, humidity, passive tracers).
  - Scalar correction for Low Mach conservative form.
  - Sub-grid scale (SGS) turbulence model (Smagorinsky, Vreman).
  - Thermodynamics-driven density update (Low Mach / Anelastic).
  - Equation of state: constant density, ideal gas, linear mixing.
"""


import jax
import jax.numpy as jnp
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.boundary_condition import immersed_boundary_method as ibm_lib
from swirl_lm.jax.equations import common
from swirl_lm.jax.equations import pressure as pressure_lib
from swirl_lm.jax.equations import scalars as scalars_lib
from swirl_lm.jax.equations import velocity as velocity_lib
from swirl_lm.jax.linalg import poisson_solver_pb2 as jax_ps_pb2
from swirl_lm.jax.physics import thermodynamics as thermodynamics_lib
from swirl_lm.jax.physics.turbulence import sgs_model as sgs_model_lib
from swirl_lm.jax.utility import types
from swirl_lm.physics.thermodynamics import thermodynamics_pb2

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

_KEY_RHO = common.KEY_RHO
_KEY_U = common.KEY_U
_KEY_V = common.KEY_V
_KEY_W = common.KEY_W
_KEY_P = common.KEY_P
_KEY_DP = common.KEY_DP
_KEY_RHO_U = common.KEY_RHO_U
_KEY_RHO_V = common.KEY_RHO_V
_KEY_RHO_W = common.KEY_RHO_W


# `swirl_lm.CoordinateInt` numbers its fields while the JAX message of the
# same name names them, so a copy by field name needs this one translation.
_COORDINATE_ALIASES = {'dim_0': 'dim_x', 'dim_1': 'dim_y', 'dim_2': 'dim_z'}


def _is_repeated(descriptor) -> bool:
  """Whether `descriptor` describes a repeated field.

  Protobuf replaced `label` and `LABEL_REPEATED` with `is_repeated`. Implemented
  this function for backward compatibility.

  Args:
    descriptor: A protobuf field descriptor.

  Returns:
    True if the field is repeated.
  """
  if hasattr(descriptor, 'is_repeated'):
    return descriptor.is_repeated
  return descriptor.label == descriptor.LABEL_REPEATED


def _copy_set_fields(source, target) -> None:
  """Copies every field explicitly set on `source` onto `target`, by name.

  Fields the target does not declare are skipped (e.g., `halo_width`, read from
  grid parametrization in JAX). Fields the source left unset are intentionally
  not touched.

  Args:
    source: The message to copy from.
    target: The message to copy onto, modified in place.
  """
  for descriptor, value in source.ListFields():
    name = descriptor.name
    if name not in target.DESCRIPTOR.fields_by_name:
      name = _COORDINATE_ALIASES.get(name, name)
      if name not in target.DESCRIPTOR.fields_by_name:
        continue

    if descriptor.type == descriptor.TYPE_MESSAGE:
      if _is_repeated(descriptor):
        for element in value:
          _copy_set_fields(element, getattr(target, name).add())
      else:
        nested = getattr(target, name)
        nested.SetInParent()
        _copy_set_fields(value, nested)
    elif _is_repeated(descriptor):
      getattr(target, name).extend(value)
    else:
      setattr(target, name, value)


class Simulation:
  """Defines the step function for a variable-density low Mach solver."""

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
  ):
    """Initializes the simulation.

    Args:
      params: The simulation parameters wrapping the proto config.
    """
    self._params = params
    self.thermodynamics = thermodynamics_lib.ThermodynamicsManager(params)

    # Create immersed boundary method if configured.
    self._has_ib = (
        params.boundary_models is not None
        and params.boundary_models.HasField('ib')
    )
    ib = ibm_lib.ImmersedBoundaryMethod(params) if self._has_ib else None

    # Velocity module. SGS model is created internally by Velocity if enabled.
    # IBM is injected for applying constraints in prediction/correction steps.
    self.velocity = velocity_lib.Velocity(params, self.thermodynamics, ib=ib)
    self.pressure = pressure_lib.Pressure(
        params,
        self.thermodynamics,
        solver_option=self._pressure_solver_option(params),
    )

    # Only create scalar solver if transport scalars are configured.
    # SGS model for scalars is created via scalar model factory.
    self._use_sgs = params.use_sgs
    self._has_scalars = bool(params.transport_scalars_names)
    if self._has_scalars:
      sgs_for_scalars = (
          sgs_model_lib.SgsModel(params) if self._use_sgs else None
      )
      self.scalars = scalars_lib.Scalars(params, sgs_for_scalars, ib=ib)

    # Track which additional_states keys are updated by the solver.
    # Variables NOT in this set will be reverted to their pre-step values
    # after each step, preventing solver internals from leaking out.
    self._updated_additional_states_keys: list[str] = []
    if params.use_sgs:
      self._updated_additional_states_keys += ['nu_t', 'drho']
    self._diagnostic_var_names = common.KEYS_DIAGNOSTICS_BUOYANCY
    self._updated_additional_states_keys += list(self._diagnostic_var_names)
    # Transient variables used during the inner simulation step.
    self._transient_var_names = ('rho_thermal', 'drho', 'dp')
    self._updated_additional_states_keys += list(self._transient_var_names)

  @staticmethod
  def _pressure_solver_option(params):
    """Converts the TensorFlow pressure solver option into the JAX one.

    Copies whatever the configuration actually set.
    An unsupported solver raises instead of defaulting to a different one.

    Args:
      params: The simulation parameters.

    Returns:
      The JAX solver option, or None if the configuration selects none.

    Raises:
      ValueError: If the selected solver has no counterpart on the JAX side.
    """
    if params.pressure is None or not params.pressure.HasField('solver'):
      return None

    tf_solver = params.pressure.solver
    selected = tf_solver.WhichOneof('solver')
    if selected is None:
      return None

    jax_solver = jax_ps_pb2.PoissonSolver()
    if selected not in jax_solver.DESCRIPTOR.fields_by_name:
      raise ValueError(
          f'The pressure configuration selects the {selected} solver, which '
          'the JAX port does not provide. Silently running a different one '
          'is worse than stopping here.'
      )

    target = getattr(jax_solver, selected)
    # Presence even when every field is left at its default, so that
    # `HasField('solver')` below sees the choice.
    target.SetInParent()
    _copy_set_fields(getattr(tf_solver, selected), target)

    return jax_solver if jax_solver.HasField('solver') else None

  def _init_states(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      mesh: jax.sharding.Mesh,
  ) -> dict[str, ScalarField]:
    """Prepares states_0 from the input states with BCs and momentum.

    Mirrors TF `_update_initial_states`: exchanges velocity/scalar halos,
    updates density via thermodynamics (if variable-density), initializes
    momentum, buoyancy, and pressure halos.

    Args:
      states: Must contain rho, u, v, w, p, and any transport scalars.
      additional_states: Helper variables including boundary conditions and
        stretched grid scale factors.
      mesh: JAX mesh for device topology.

    Returns:
      states_0 with rho, u, v, w, p, rho_u, rho_v, rho_w, dp, buoyancy,
      and all transport scalars initialized.
    """
    rho = states[_KEY_RHO]
    u = states[_KEY_U]
    v = states[_KEY_V]
    w = states[_KEY_W]
    p = states[_KEY_P]

    # Exchange velocity halos with boundary conditions.
    u = self.velocity.exchange_velocity_halos(u, _KEY_U, mesh)
    v = self.velocity.exchange_velocity_halos(v, _KEY_V, mesh)
    w = self.velocity.exchange_velocity_halos(w, _KEY_W, mesh)

    states_0: dict[str, ScalarField] = {
        _KEY_RHO: rho,
        _KEY_U: u,
        _KEY_V: v,
        _KEY_W: w,
        _KEY_P: p,
    }

    # Exchange scalar halos.
    if self._has_scalars:
      for sc_name in self._params.transport_scalars_names:
        if sc_name in states:
          states_0[sc_name] = self.scalars.exchange_scalar_halos(
              states[sc_name], sc_name, mesh
          )

    # For variable-density flows, update density from equation of state and
    # initialize thermal density, drho, and buoyancy fields.
    thermo_model_type = self.thermodynamics.model_type
    has_variable_density = (
        self._params.thermodynamics is not None
        and thermo_model_type != 'constant_density'
    )
    if has_variable_density:
      if self._params.solver_mode == thermodynamics_pb2.Thermodynamics.LOW_MACH:
        rho_0, _ = self.thermodynamics.update_density(
            states_0, additional_states, mesh
        )
        states_0[_KEY_RHO] = rho_0
        rho = rho_0
      states_0['rho_thermal'] = self.thermodynamics.update_thermal_density(
          states_0, additional_states
      )
      states_0['drho'] = jnp.zeros_like(rho)
    else:
      states_0['drho'] = jnp.zeros_like(rho)

    # Initialize buoyancy diagnostic fields.
    states_0['buoyancy_u'] = jnp.zeros_like(rho)
    states_0['buoyancy_v'] = jnp.zeros_like(rho)
    states_0['buoyancy_w'] = jnp.zeros_like(rho)

    # Compute momentum.
    states_0[_KEY_RHO_U] = rho * states_0[_KEY_U]
    states_0[_KEY_RHO_V] = rho * states_0[_KEY_V]
    states_0[_KEY_RHO_W] = rho * states_0[_KEY_W]

    # Compute conserved scalar momentum.
    if self._has_scalars:
      for sc_name in self._params.transport_scalars_names:
        if sc_name in states_0:
          states_0[f'rho_{sc_name}'] = rho * states_0[sc_name]

    # Exchange pressure halos with flow-dependent BCs.
    pressure_halos = self.pressure.update_pressure_halos(
        states_0, additional_states, mesh
    )
    states_0.update(pressure_halos)
    states_0[_KEY_DP] = jnp.zeros_like(states_0[_KEY_P])

    # Carry over nu_t from additional_states if available.
    if 'nu_t' in additional_states:
      states_0['nu_t'] = additional_states['nu_t']

    return states_0

  def step(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      mesh: jax.sharding.Mesh,
  ) -> dict[str, ScalarField]:
    """Advances the simulation by one time step.

    Uses a predictor-corrector loop matching the TF implementation:
      Step 1: Scalar prediction (convection + diffusion + source).
      Step 2: Density update from thermodynamic equation of state.
              Includes pressure halo update with flow-dependent BCs.
      Step 3: Scalar correction (Low Mach only).
      Step 4: Velocity prediction (convection + diffusion + SGS + IBM).
      Step 5: Pressure Poisson solve for pressure correction dp.
      Step 6: Velocity correction using grad(dp), with IBM constraints.

    SGS turbulent viscosity is computed internally by the velocity module.
    IBM constraints are applied internally by velocity and scalar modules.

    Stretched grid scale factors are passed transparently via
    `additional_states` and consumed internally by the derivatives library.

    Args:
      states: Flow field variables: rho, u, v, w, p, and transport scalars.
      additional_states: Helper variables including boundary conditions and
        stretched grid scale factors (if any).
      mesh: JAX mesh for device topology.

    Returns:
      Updated states: rho, u, v, w, p, and transport scalars.
    """
    if self._has_scalars:
      self.scalars.prestep(additional_states)
    self.velocity.prestep(additional_states, mesh)
    self.pressure.prestep(additional_states)

    states_0 = self._init_states(states, additional_states, mesh)

    def update_step(
        states_k: dict[str, ScalarField],
    ) -> dict[str, ScalarField]:
      """One predictor-corrector iteration."""
      # Step 0: Recompute momentum from current velocity and mid-point density.
      rho_mid = 0.5 * (states_k[_KEY_RHO] + states_0[_KEY_RHO])
      states_k[_KEY_RHO_U] = rho_mid * states_k[_KEY_U]
      states_k[_KEY_RHO_V] = rho_mid * states_k[_KEY_V]
      states_k[_KEY_RHO_W] = rho_mid * states_k[_KEY_W]

      # Step 1: Scalar prediction.
      # IBM constraints are applied internally by scalars.prediction_step.
      mass_source = None
      if self._has_scalars:
        scalar_prediction, mass_source = self.scalars.prediction_step(
            states_k, states_0, additional_states, mesh
        )
        states_k.update(scalar_prediction)

      # Step 2: Density update from equation of state.
      # Only update density when a non-trivial thermodynamics model is
      # configured. For constant density, rho is unchanging and drho = 0.
      thermo_model_type = self.thermodynamics.model_type
      has_variable_density = (
          self._params.thermodynamics is not None
          and thermo_model_type != 'constant_density'
      )

      if has_variable_density:
        if (
            self._params.solver_mode
            == thermodynamics_pb2.Thermodynamics.LOW_MACH
        ):
          rho, drho = self.thermodynamics.update_density(
              states_k, additional_states, mesh, states_0
          )
          rho_thermal = self.thermodynamics.update_thermal_density(
              states_k, additional_states
          )
          states_k[_KEY_RHO] = rho
          states_k['rho_thermal'] = rho_thermal
          states_k['drho'] = drho
        else:
          # For ANELASTIC, compute thermal density for buoyancy.
          rho_thermal = self.thermodynamics.update_thermal_density(
              states_k, additional_states
          )
          states_k['rho_thermal'] = rho_thermal

      # Step 2 (continued): Update pressure halos with flow-dependent BCs.
      # In TF this is part of the density update step: after density is
      # corrected, pressure halos are re-derived from the current buoyancy
      # field using rho_mid (not rho_k) to prevent spurious forcing.
      pressure_halos = self.pressure.update_pressure_halos(
          states_k, additional_states, mesh
      )
      states_k.update(pressure_halos)

      # Step 3: Scalar correction (Low Mach only).
      # After density update, re-derive primitive scalars from conserved form:
      # phi = rho_phi / rho_corrected. Only meaningful if rho_phi was advanced
      # in the prediction step (Low Mach conservative form).
      # IBM constraints are applied internally by scalars.correction_step.
      if (
          self._has_scalars
          and has_variable_density
          and self._params.enable_scalar_recorrection
          and self._params.solver_mode
          == thermodynamics_pb2.Thermodynamics.LOW_MACH
      ):
        scalar_correction = self.scalars.correction_step(
            states_k, additional_states, mesh
        )
        states_k.update(scalar_correction)

      # Step 4: Velocity prediction.
      # SGS turbulent viscosity is computed internally by velocity.
      # IBM constraints are applied internally after halo exchange.
      velocity_prediction = self.velocity.prediction_step(
          states_k, states_0, additional_states, mesh
      )
      states_k.update(velocity_prediction)

      # Step 5: Pressure solve.
      # Pass mass_source from scalar models to the pressure solver.
      pressure_additional = additional_states
      if mass_source is not None:
        pressure_additional = dict(additional_states)
        pressure_additional['mass_source'] = mass_source
      pressure_step = self.pressure.step(
          states_k, states_0, pressure_additional, mesh
      )
      states_k.update(pressure_step)

      # Step 6: Velocity correction.
      # IBM constraints are applied internally before halo exchange.
      velocity_correction = self.velocity.correction_step(
          states_k, states_0, additional_states, mesh
      )
      states_k.update(velocity_correction)

      return states_k

    # Run predictor-corrector iterations using jax.lax.fori_loop (mirrors
    # TF's tf.while_loop) to avoid graph unrolling inside JIT.
    def _loop_body(_, states_k):
      return update_step(states_k)

    states_k = jax.lax.fori_loop(
        0, self._params.corrector_nit, _loop_body, dict(states_0)
    )

    # For additional states that are NOT meant to be changed by the inner
    # solver, revert their values back to the original (pre-step) values.
    states_k.update({
        key: val
        for key, val in additional_states.items()
        if key not in self._updated_additional_states_keys
    })

    # Remove temporary momentum variables (rho_u, rho_v, rho_w).
    for varname in [_KEY_U, _KEY_V, _KEY_W]:
      states_k.pop(f'rho_{varname}', None)
    for sc_name in self._params.transport_scalars_names:
      states_k.pop(f'rho_{sc_name}', None)

    # Remove diagnostic and transient variables that are not explicitly
    # specified as additional_states in the config.
    for var_name in list(self._diagnostic_var_names) + list(
        self._transient_var_names
    ):
      if var_name not in additional_states:
        states_k.pop(var_name, None)

    return states_k
