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
"""A library for solving scalar transport equations (JAX).

This is the JAX port of `swirl_lm.equations.scalars`. It solves the generic
scalar transport equation:

  d(rho * phi) / dt = -div(rho * u * phi) + div(D * grad(phi)) + S

where:
  - phi is the scalar field (e.g. temperature, humidity, or passive tracer),
  - D is the scalar diffusivity (molecular + SGS turbulent),
  - S is the source term (model-dependent).

Each scalar has an associated `ScalarModel` that provides its diffusivity and
source terms. The model is created via `scalar_model_factory`, which dispatches
to specialised models for known scalar types (PotentialTemperature, Humidity,
TotalEnergy) and falls back to `GenericScalarModel` for passive scalars.

Supported modes:
  - Anelastic (primitive variable form): dphi/dt = f(phi)/rho.
  - Low Mach (conservative form): d(rho*phi)/dt = f(phi).
"""


import jax
import jax.numpy as jnp
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.base import physical_variable_keys_manager
from swirl_lm.jax.boundary_condition import immersed_boundary_method as ibm_lib
from swirl_lm.jax.communication import halo_exchange
from swirl_lm.jax.equations import common
from swirl_lm.jax.equations import scalar_model as scalar_model_lib
from swirl_lm.jax.numerics import convection
from swirl_lm.jax.numerics import diffusion
from swirl_lm.jax.physics.turbulence import sgs_model as sgs_model_lib
from swirl_lm.jax.utility import types
from swirl_lm.physics.thermodynamics import thermodynamics_pb2

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

_KEY_RHO = common.KEY_RHO
_KEY_P = common.KEY_P
_KEYS_MOMENTUM = common.KEYS_MOMENTUM


class Scalars:
  """A library for solving scalar transport equations."""

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
      sgs: sgs_model_lib.SgsModel | None = None,
      ib: ibm_lib.ImmersedBoundaryMethod | None = None,
  ):
    """Initializes the scalar transport library.

    Args:
      params: The simulation parameters.
      sgs: Optional SGS model for turbulent diffusivity.
      ib: Immersed boundary method instance. If provided, IBM constraints are
        applied after scalar prediction and correction steps.
    """
    self._params = params
    self._kernel_op = params.kernel_op
    self._deriv_lib = params.deriv_lib
    self._grid_params = params.grid_params

    # Store immersed boundary method.
    self._ib = ib

    # Collect boundary conditions for transport scalars.
    self._bc = {
        varname: bc_val
        for varname, bc_val in params.bc.items()
        if varname in params.transport_scalars_names
    }

    self._bc_manager = (
        physical_variable_keys_manager.BoundaryConditionKeysHelper()
    )
    self._src_manager = physical_variable_keys_manager.SourceKeysHelper()

    self._source: dict[str, ScalarField | None] = {
        sc.name: None for sc in params.scalars if sc.solve_scalar
    }

    # Create a scalar model per transport scalar.
    self._models: dict[str, scalar_model_lib.ScalarModel] = {}
    for sc_name in params.transport_scalars_names:
      self._models[sc_name] = scalar_model_lib.scalar_model_factory(
          params, sc_name, sgs
      )

    # Create diffusion function with MOST and prescribed flux BC support.
    self._diffusion_fn = diffusion.diffusion_scalar_factory(params)

  def exchange_scalar_halos(
      self,
      f: ScalarField,
      name: str,
      mesh: jax.sharding.Mesh,
  ) -> ScalarField:
    """Performs halo exchange for a scalar field with boundary conditions.

    Args:
      f: The 3D scalar field.
      name: Variable name for BC lookup.
      mesh: JAX device mesh.

    Returns:
      The field with halos updated.
    """
    bc = self._bc.get(name)
    periodic = list(self._grid_params.to_xyz_order(self._params.periodic_dims))  # pyrefly: ignore[bad-argument-type]
    return halo_exchange.inplace_halo_exchange(
        f,
        ('x', 'y', 'z'),
        mesh,
        self._grid_params,
        periodic,
        bc,
        halo_width=self._params.halo_width,
    )

  def prestep(
      self,
      additional_states: ScalarFieldMap,
  ) -> None:
    """Updates additional information required for scalars step.

    This function is called before the beginning of each time step. It updates
    the boundary conditions of all scalars. It also updates the source term of
    each scalar. These information will be held within this helper object.

    Args:
      additional_states: A dictionary that holds constants that will be used in
        the simulation, e.g. boundary conditions, forcing terms.
    """
    # Parse additional states to extract boundary conditions.
    self._bc = self._bc_manager.update_helper_variable_from_additional_states(
        additional_states, self._bc, self._grid_params
    )

    # Parse additional states to extract external source/forcing terms.
    self._source.update(
        self._src_manager.update_helper_variable_from_additional_states(
            additional_states
        )
    )

  def _scalar_rhs(
      self,
      scalar_name: str,
      phi: ScalarField,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes the RHS of the scalar transport equation.

    f(phi) = -(conv_x + conv_y + conv_z) + (diff_x + diff_y + diff_z) + source

    Args:
      scalar_name: Name of the scalar being solved.
      phi: The scalar field at the current iteration.
      states: Flow field variables (rho, u, v, w, p, rho_u, rho_v, rho_w).
      additional_states: Helper variables (diffusivity, BCs, etc.).

    Returns:
      The RHS of the scalar transport equation.
    """
    rho = states[_KEY_RHO]
    axes = self._grid_params.data_axis_order
    gs = self._params.grid_spacings
    dt = self._params.dt

    # Convection: each dimension computes -d(rho_u_i * phi)/dx_i.
    conv = []
    for i, axis in enumerate(axes):
      conv_term = convection.convection_term(
          self._kernel_op,
          self._deriv_lib,
          phi,
          states[_KEYS_MOMENTUM[i]],
          states[_KEY_P],
          gs[i],
          dt,
          axis,
          additional_states,
          self._grid_params,
          scheme=self._params.convection_scheme,  # pyrefly: ignore[bad-argument-type]
          flux_scheme=self._params.numerical_flux,  # pyrefly: ignore[bad-argument-type]
          src=None,
          apply_correction=False,
      )
      conv.append(conv_term)

    # Diffusion: computes d(rho * D * dphi/dx_i)/dx_i for each dim.
    model = self._models[scalar_name]
    diffusivity_field = model.get_diffusivity(phi, states, additional_states)

    # Build helper variables for diffusion. Include stretched grid variables
    # and velocity/theta for MOST if available.
    diff_helper: dict[str, ScalarField] = dict(additional_states)
    for key in ('u', 'v', 'w'):
      if key in states:
        diff_helper[key] = states[key]
    # Potential temperature for MOST: look for theta/theta_li/T.
    for theta_key in ('theta', 'theta_li', 'T'):
      if theta_key in states:
        diff_helper['theta'] = states[theta_key]
        break
      elif theta_key in additional_states:
        diff_helper['theta'] = additional_states[theta_key]
        break

    diff = self._diffusion_fn(
        self._kernel_op,
        self._deriv_lib,
        phi,
        rho,
        diffusivity_field,
        scalar_name=scalar_name,
        helper_variables=diff_helper,
    )

    # Source: model-specific source + external source from prestep().
    source = model.source_fn(phi, states, additional_states)
    external_source = self._source.get(scalar_name)
    if external_source is not None:
      source = source + external_source

    # Assemble RHS: -convection + diffusion + source.
    rhs = -(conv[0] + conv[1] + conv[2]) + (diff[0] + diff[1] + diff[2])
    rhs = rhs + source

    return rhs

  def prediction_step(
      self,
      states: ScalarFieldMap,
      states_0: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      mesh: jax.sharding.Mesh,
  ) -> tuple[dict[str, ScalarField], ScalarField]:
    """Predicts all transport scalars using the generic transport equation.

    Supports two modes:
      - ANELASTIC (primitive form): phi_new = phi_old + dt * f(phi_mid) / rho.
      - LOW_MACH (conservative form): rho_phi_new = rho_phi_old + dt *
      f(phi_mid),
        then phi_new = rho_phi_new / rho.

    Args:
      states: Current flow field variables (latest prediction).
      states_0: Previous time step flow field variables.
      additional_states: Helper variables (BCs, sources, etc.).
      mesh: JAX device mesh.

    Returns:
      A tuple of (updated_scalars, mass_source):
        - updated_scalars: Dict of updated scalar fields (and rho_<scalar>
          for Low Mach).
        - mass_source: Accumulated mass source for the pressure Poisson
          equation (zero if no scalar model contributes).
    """
    updated_scalars = {}
    is_low_mach = (
        self._params.solver_mode == thermodynamics_pb2.Thermodynamics.LOW_MACH
    )
    mass_source = jnp.zeros_like(states[_KEY_RHO])

    # Construct midpoint states for Crank-Nicolson time integration.
    # Average rho, rho_thermal, and all scalars between current and initial.
    states_mid = dict(states)
    states_mid[_KEY_RHO] = 0.5 * (states[_KEY_RHO] + states_0[_KEY_RHO])
    if 'rho_thermal' in states and 'rho_thermal' in states_0:
      states_mid['rho_thermal'] = 0.5 * (
          states['rho_thermal'] + states_0['rho_thermal']
      )
    for sc_name in self._params.transport_scalars_names:
      states_mid[sc_name] = 0.5 * (states[sc_name] + states_0[sc_name])

    for sc_name in self._params.transport_scalars_names:
      # Mid-point scalar for time integration.
      sc_mid = states_mid[sc_name]

      # Compute RHS at mid-point.
      rhs = self._scalar_rhs(
          sc_name,
          sc_mid,
          states_mid,
          additional_states,
      )

      if is_low_mach:
        # Conservative form: d(rho*phi)/dt = f(phi).
        rho_sc_key = f'rho_{sc_name}'
        rho_phi_old = states_0.get(
            rho_sc_key, states_0[_KEY_RHO] * states_0[sc_name]
        )
        rho_phi_new = rho_phi_old + self._params.dt * rhs
        updated_scalars[rho_sc_key] = rho_phi_new

        # Recover primitive scalar: phi = rho_phi / rho.
        new_sc = rho_phi_new / states[_KEY_RHO]
      else:
        # Anelastic mode (primitive variable form):
        # phi_new = phi_old + dt * rhs / rho.
        alpha = 1.0 / states[_KEY_RHO]
        new_sc = states_0[sc_name] + self._params.dt * rhs * alpha

      # Exchange halos with BCs.
      new_sc = self.exchange_scalar_halos(new_sc, sc_name, mesh)
      updated_scalars[sc_name] = new_sc

      # Accumulate mass source from this scalar model.
      model = self._models[sc_name]
      src_rho = model.mass_source_fn(sc_mid, states, additional_states)
      if src_rho is not None:
        mass_source = mass_source + src_rho

    # Apply immersed boundary constraints to all updated scalars.
    if self._ib is not None:
      updated_scalars = dict(
          self._ib.update_states(updated_scalars, additional_states)
      )

    return updated_scalars, mass_source

  def correction_step(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,  # pylint: disable=unused-argument
      mesh: jax.sharding.Mesh,
  ) -> dict[str, ScalarField]:
    """Updates the primitive scalars after the density correction.

    After the pressure solve corrects the density, the primitive scalars
    must be recomputed from the conserved form: phi = rho_phi / rho.

    This step is only meaningful in LOW_MACH mode where the conserved
    scalar (rho_phi) was advanced during prediction. For ANELASTIC mode,
    this is a no-op.

    Args:
      states: Current flow field variables (must contain rho_<scalar>).
      additional_states: Helper variables.
      mesh: JAX device mesh.

    Returns:
      Dict of corrected primitive scalar fields.
    """
    corrected = {}

    for sc_name in self._params.transport_scalars_names:
      rho_sc_key = f'rho_{sc_name}'
      if rho_sc_key in states:
        sc_buf = states[rho_sc_key] / states[_KEY_RHO]

        # Apply IBM constraints per scalar before halo exchange.
        if self._ib is not None:
          sc_buf = self._ib.update_states({sc_name: sc_buf}, additional_states)[
              sc_name
          ]

        sc_buf = self.exchange_scalar_halos(sc_buf, sc_name, mesh)
        corrected[sc_name] = sc_buf

    return corrected
