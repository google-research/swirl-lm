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
"""A library for solving the momentum equation for velocity.

This is the JAX port of `swirl_lm.equations.velocity`. It solves the momentum
equation using the predictor-corrector approach, with optional subgrid-scale
(SGS) turbulence modeling and immersed boundary method (IBM) support.
"""


import copy
from typing import Any

import jax
import jax.numpy as jnp
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.base import physical_variable_keys_manager
from swirl_lm.jax.boundary_condition import boundary_condition_utils
from swirl_lm.jax.boundary_condition import immersed_boundary_method as ibm_lib
from swirl_lm.jax.communication import halo_exchange
from swirl_lm.jax.communication import halo_exchange_utils
from swirl_lm.jax.equations import common
from swirl_lm.jax.equations import utils as eq_utils
from swirl_lm.jax.numerics import convection
from swirl_lm.jax.numerics import diffusion
from swirl_lm.jax.numerics import time_integration
from swirl_lm.jax.physics import thermodynamics as thermodynamics_lib
from swirl_lm.jax.physics.turbulence import sgs_model as sgs_model_lib
from swirl_lm.jax.utility import common_ops
from swirl_lm.jax.utility import types
from swirl_lm.physics.thermodynamics import thermodynamics_pb2

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

_G_THRESHOLD = 1e-6

# Key aliases.
_KEY_RHO = common.KEY_RHO
_KEY_P = common.KEY_P
_KEY_DP = common.KEY_DP
_KEY_U = common.KEY_U
_KEY_V = common.KEY_V
_KEY_W = common.KEY_W
_KEYS_VELOCITY = common.KEYS_VELOCITY
_KEY_RHO_U = common.KEY_RHO_U
_KEY_RHO_V = common.KEY_RHO_V
_KEY_RHO_W = common.KEY_RHO_W
_KEYS_MOMENTUM = common.KEYS_MOMENTUM


class Velocity:
  """A library for advancing velocity to the next time step."""

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
      thermodynamics: thermodynamics_lib.ThermodynamicsManager | None = None,
      ib: ibm_lib.ImmersedBoundaryMethod | None = None,
  ):
    """Initializes the velocity update library.

    Args:
      params: The simulation parameters.
      thermodynamics: Thermodynamics manager for reference density and solver
        mode. If None, a default manager is created from params.
      ib: Immersed boundary method instance. If provided, IBM constraints are
        applied after velocity prediction and correction steps.
    """
    self._params = params
    self._kernel_op = params.kernel_op
    self._deriv_lib = params.deriv_lib
    self._grid_params = params.grid_params

    if thermodynamics is not None:
      self._thermodynamics = thermodynamics
    else:
      self._thermodynamics = thermodynamics_lib.ThermodynamicsManager(params)

    # Create SGS model if turbulence modeling is enabled.
    self._use_sgs = params.use_sgs
    if self._use_sgs:
      self._sgs_model = sgs_model_lib.SgsModel(params)

    # Store immersed boundary method.
    self._ib = ib

    # Create the diffusion function via the factory, which dispatches to the
    # appropriate scheme (CENTRAL_5, CENTRAL_3, or STENCIL_3).
    self._diffusion_fn = diffusion.diffusion_momentum(params)

    self._gravity_vec = (
        self._params.gravity_direction
        if self._params.gravity_direction
        else [0.0, 0.0, 0.0]
    )

    self._bc = {
        varname: bc_val
        for varname, bc_val in self._params.bc.items()
        if varname in _KEYS_VELOCITY
    }

    self._bc_manager = (
        physical_variable_keys_manager.BoundaryConditionKeysHelper()
    )
    self._src_manager = physical_variable_keys_manager.SourceKeysHelper()

    # Mapping from shear stress names to direction labels for tau BC updates.
    self._tau_name_map = {
        'tau00': 'xx',
        'tau01': 'xy',
        'tau02': 'xz',
        'tau10': 'yx',
        'tau11': 'yy',
        'tau12': 'yz',
        'tau20': 'zx',
        'tau21': 'zy',
        'tau22': 'zz',
    }
    self._tau_bc_update_fn: dict[str, Any] = {}
    self._source: dict[str, ScalarField | None] = {
        _KEY_U: None,
        _KEY_V: None,
        _KEY_W: None,
    }

  def _update_wall_bc(
      self,
      states: ScalarFieldMap,
  ) -> dict[str, Any]:
    """Computes updated boundary conditions for velocity at walls.

    The wall is assumed to be at the midpoint between the first halo layer and
    the first fluid layer. For a velocity component to be 0 at this face, the
    halo layers are set to mirrored values of the fluid layers so that the
    interpolated value on the face between them is 0.

    For non-slip walls, all velocity components are set to zero at the wall.
    For free-slip and shear walls, only the wall-normal velocity component is
    set to zero (tangential components keep their configured Neumann BCs).

    Args:
      states: A dictionary that holds flow field variables from the latest
        prediction. Must contain 'u', 'v', and 'w'.

    Returns:
      A new BC dict with wall entries updated from the current velocity fields.
    """
    bc = copy.deepcopy(self._bc)

    hw = self._params.halo_width
    wall_types = (
        boundary_condition_utils.BoundaryType.NON_SLIP_WALL,
        boundary_condition_utils.BoundaryType.SHEAR_WALL,
        boundary_condition_utils.BoundaryType.SLIP_WALL,
    )

    for dim in range(3):
      for face in range(2):
        if self._params.bc_type[dim][face] not in wall_types:
          continue

        # For non-slip walls, enforce zero for all velocity components.
        # For slip/shear walls, only enforce the wall-normal component.
        if (
            self._params.bc_type[dim][face]
            == boundary_condition_utils.BoundaryType.NON_SLIP_WALL
        ):
          velocity_keys = list(_KEYS_VELOCITY)
        else:
          velocity_keys = [_KEYS_VELOCITY[dim]]

        for vel_key in velocity_keys:
          val = states[vel_key]
          axis = ('x', 'y', 'z')[dim]

          # Generate BC planes that mirror the fluid layer so the wall-face
          # interpolated value is zero. The plane ordering is from low to high
          # along the dimension:
          #   - For face=0 (low): innermost plane is at index hw-1
          #   - For face=1 (high): innermost plane is at index 0
          bc_planes = []
          for i in range(hw):
            # idx determines the extrapolation factor.
            idx = i if face == 1 else hw - 1 - i
            scaling = -1.0 * (2 * idx + 1)
            # Get the first fluid layer adjacent to the halo region.
            fluid_plane = common_ops.get_face(
                val, axis, face, hw, self._grid_params  # pyrefly: ignore[bad-argument-type]
            )
            bc_planes.append(scaling * fluid_plane)

          bc[vel_key][dim][face] = (  # pyrefly: ignore[unsupported-operation]
              halo_exchange_utils.BCType.DIRICHLET,
              bc_planes,
          )

    return bc

  def exchange_velocity_halos(
      self,
      f: ScalarField,
      name: str,
      mesh: jax.sharding.Mesh,
      bc: dict[str, Any] | None = None,
  ) -> ScalarField:
    """Performs halo exchange for velocity `f`."""
    if bc is None:
      bc = self._bc
    bc_for_name = bc.get(name)
    periodic = list(self._grid_params.to_xyz_order(self._params.periodic_dims))  # pyrefly: ignore[bad-argument-type]
    return halo_exchange.inplace_halo_exchange(
        f,
        ('x', 'y', 'z'),
        mesh,
        self._grid_params,
        periodic,
        bc_for_name,
        halo_width=self._params.halo_width,
    )

  def _momentum_rhs(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      mu: ScalarField,
      p: ScalarField,
      forces: tuple[ScalarField | None, ...] = (None, None, None),
  ) -> tuple[ScalarField, ScalarField, ScalarField]:
    """Computes the RHS of the momentum equations for all 3 dimensions.

    Args:
      states: Must contain u, v, w, rho_u, rho_v, rho_w.
      additional_states: Helper variables.
      mu: Dynamic viscosity.
      p: Pressure.
      forces: External forces for each dimension.

    Returns:
      Tuple of (rhs_u, rhs_v, rhs_w).
    """
    dt = self._params.dt
    gs = self._params.grid_spacings
    grid_spacings_3 = (gs[0], gs[1], gs[2])
    axes = self._grid_params.data_axis_order

    # Compute diffusion terms for all velocity components using the configured
    # scheme (CENTRAL_5, CENTRAL_3, or STENCIL_3).
    states_for_diff = dict(states)
    states_for_diff[_KEY_RHO] = states.get(
        _KEY_RHO, self._params.rho * jnp.ones_like(p)
    )

    # Enrich helper variables with theta for MOST closure (if configured).
    helper_variables: dict[str, ScalarField] = dict(additional_states)
    if (
        self._params.boundary_models is not None
        and self._params.boundary_models.HasField('most')
    ):
      for theta_key in ('theta', 'theta_li', 'T'):
        if theta_key in states:
          helper_variables['theta'] = states[theta_key]
          break

    diff_all = self._diffusion_fn(
        self._kernel_op,
        self._deriv_lib,
        self._params.diffusion_scheme,
        mu,
        grid_spacings_3,
        states_for_diff,
        helper_variables,
    )

    # Compute buoyancy source for reference density (if gravity is present).
    rho_mix = states.get('rho_thermal', self._params.rho * jnp.ones_like(p))
    zz = additional_states.get('zz', jnp.zeros_like(p))
    rho_ref = self._thermodynamics.rho_ref(zz, additional_states)

    rhs_list = []
    # Iterate over velocity components in physical order: u→x, v→y, w→z.
    # The axis for pressure gradient and buoyancy must match the velocity
    # component, not the data_axis_order.
    for dim, vel_key in enumerate(_KEYS_VELOCITY):
      axis = ('x', 'y', 'z')[dim]

      # Buoyancy source.
      gravity = eq_utils.buoyancy_source(
          rho_mix, rho_ref, self._params, dim, additional_states
      )

      # Convection terms along each axis.
      # The transport momentum for convection axis j is rho_u_j (the momentum
      # in the direction of transport), not rho_u_i (the equation's momentum).
      _axis_to_momentum = {
          'x': _KEYS_MOMENTUM[0],
          'y': _KEYS_MOMENTUM[1],
          'z': _KEYS_MOMENTUM[2],
      }
      conv_terms = []
      conv_dim_map = {'x': 0, 'y': 1, 'z': 2}
      for i, conv_axis in enumerate(axes):
        g_corr = None if abs(self._gravity_vec[dim]) < _G_THRESHOLD else gravity
        conv_dim = conv_dim_map[conv_axis]
        conv_term = convection.convection_term(
            self._kernel_op,
            self._deriv_lib,
            states[vel_key],
            states[_axis_to_momentum[conv_axis]],
            p,
            grid_spacings_3[i],
            dt,
            conv_axis,
            additional_states,
            self._grid_params,
            scheme=self._params.convection_scheme,  # pyrefly: ignore[bad-argument-type]
            flux_scheme=self._params.numerical_flux,  # pyrefly: ignore[bad-argument-type]
            bc_types=tuple(self._params.bc_type[conv_dim]),  # pyrefly: ignore[bad-argument-type]
            varname=_axis_to_momentum[conv_axis],
            halo_width=self._params.halo_width,
            src=g_corr,
            apply_correction=self._params.enable_rhie_chow_correction,
        )
        conv_terms.append(conv_term)

      # Diffusion terms.
      diff = diff_all[vel_key]

      # Pressure gradient.
      dp_dh = self._deriv_lib.deriv_centered(p, axis, additional_states)
      if (
          self._params.solver_mode
          == thermodynamics_pb2.Thermodynamics.ANELASTIC
      ):
        dp_dh = rho_ref * dp_dh

      # External force.
      force = (
          forces[dim]
          if forces[dim] is not None
          else jnp.zeros_like(states[vel_key])
      )

      # RHS = -conv + diff - dp/dx + gravity + force
      rhs = (
          -conv_terms[0]
          - conv_terms[1]
          - conv_terms[2]
          + diff[0]
          + diff[1]
          + diff[2]
          - dp_dh
          + gravity
          + force
      )
      rhs_list.append(rhs)

    return rhs_list[0], rhs_list[1], rhs_list[2]

  def prestep(
      self,
      additional_states: ScalarFieldMap,
      mesh: jax.sharding.Mesh,
  ) -> None:
    """Updates additional information required for velocity step.

    This function is called before the beginning of each time step. It updates
    the boundary conditions of 'u', 'v', 'w', and the shear stresses if
    required. It also updates the forcing term of each momentum component.
    These information will be held within this helper object.

    Args:
      additional_states: A dictionary that holds constants that will be used in
        the simulation, e.g. boundary conditions, forcing terms.
      mesh: JAX mesh for device topology.
    """
    # Parse additional states to extract boundary conditions.
    self._bc = self._bc_manager.update_helper_variable_from_additional_states(
        additional_states, self._bc, self._grid_params
    )
    for key, val in self._bc.items():
      if key not in self._tau_name_map:
        continue
      self._tau_bc_update_fn[self._tau_name_map[key]] = (
          lambda f, bc_f=val: self._exchange_velocity_halos_with_bc(
              f, bc_f, mesh
          )
      )

    # Parse additional states to extract external source/forcing terms.
    self._source.update(
        self._src_manager.update_helper_variable_from_additional_states(
            additional_states
        )
    )

  def _exchange_velocity_halos_with_bc(
      self,
      f: ScalarField,
      bc: Any,
      mesh: jax.sharding.Mesh,
  ) -> ScalarField:
    """Performs halo exchange for a variable with explicit boundary conditions."""
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

  def prediction_step(
      self,
      states: ScalarFieldMap,
      states_0: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      mesh: jax.sharding.Mesh,
  ) -> dict[str, ScalarField]:
    """Predicts the velocity from the momentum equation.

    SGS turbulent viscosity is computed internally if enabled. IBM constraints
    are applied after the velocity halo exchange.

    Args:
      states: Flow field variables from the latest prediction.
      states_0: Flow field variables from the previous time step.
      additional_states: Helper variables.
      mesh: JAX mesh for device topology.

    Returns:
      Dictionary with updated u, v, w, rho_u, rho_v, rho_w, and optionally
      nu_t (if SGS is enabled).
    """
    rho_mid = 0.5 * (states[_KEY_RHO] + states_0[_KEY_RHO])
    u_mid = 0.5 * (states[_KEY_U] + states_0[_KEY_U])
    v_mid = 0.5 * (states[_KEY_V] + states_0[_KEY_V])
    w_mid = 0.5 * (states[_KEY_W] + states_0[_KEY_W])

    states_mid = dict(states_0)
    states_mid.update({
        _KEY_U: u_mid,
        _KEY_V: v_mid,
        _KEY_W: w_mid,
    })

    # Compute effective viscosity: nu_eff = nu + nu_t (if SGS is enabled).
    nu_t = None
    if self._use_sgs:
      nu_t = self._sgs_model.turbulent_viscosity(
          [u_mid, v_mid, w_mid], additional_states
      )
      nu = self._params.nu + nu_t
    else:
      nu = self._params.nu
    nu = eq_utils.bound_viscosity(nu, additional_states, self._params)
    mu = nu * states_0[_KEY_RHO]

    # Use external forcing terms populated by prestep() from additional_states,
    # supplemented by source_update_fn if configured.
    forces: list[ScalarField | None] = [
        self._source[_KEY_U],
        self._source[_KEY_V],
        self._source[_KEY_W],
    ]
    for dim, vel_key in enumerate((_KEY_U, _KEY_V, _KEY_W)):
      src_fn = self._params.source_update_fn(vel_key)
      if src_fn is not None:
        src_result = src_fn(states, additional_states)
        src_val = src_result[self._src_manager.generate_src_key(vel_key)]
        if forces[dim] is not None:
          forces[dim] = forces[dim] + src_val  # pyrefly: ignore[unsupported-operation]
        else:
          forces[dim] = src_val

    # Build momentum RHS function for time integration.
    def rhs_fn(
        rho_u: ScalarField,
        rho_v: ScalarField,
        rho_w: ScalarField,
    ) -> tuple[ScalarField, ScalarField, ScalarField]:
      s = dict(states_mid)
      s.update({
          _KEY_RHO_U: rho_u,
          _KEY_RHO_V: rho_v,
          _KEY_RHO_W: rho_w,
          _KEY_U: u_mid,
          _KEY_V: v_mid,
          _KEY_W: w_mid,
          _KEY_P: states[_KEY_P],
      })
      return self._momentum_rhs(
          s, additional_states, mu, states[_KEY_P], tuple(forces)
      )

    rho_u, rho_v, rho_w = time_integration.time_advancement_explicit(
        rhs_fn,
        self._params.dt,
        self._params.time_integration_scheme,
        (states_0[_KEY_RHO_U], states_0[_KEY_RHO_V], states_0[_KEY_RHO_W]),
        (states[_KEY_RHO_U], states[_KEY_RHO_V], states[_KEY_RHO_W]),
    )

    # Compute velocity from momentum.
    u = rho_u / rho_mid
    v = rho_v / rho_mid
    w = rho_w / rho_mid

    # Update wall boundary conditions before halo exchange.
    wall_bc = self._update_wall_bc({_KEY_U: u, _KEY_V: v, _KEY_W: w})

    # Exchange velocity halos.
    u = self.exchange_velocity_halos(u, _KEY_U, mesh, wall_bc)
    v = self.exchange_velocity_halos(v, _KEY_V, mesh, wall_bc)
    w = self.exchange_velocity_halos(w, _KEY_W, mesh, wall_bc)

    # Apply immersed boundary constraints after halo exchange.
    if self._ib is not None:
      velocity_ib_updated = self._ib.update_states(
          {_KEY_U: u, _KEY_V: v, _KEY_W: w}, additional_states
      )
      u = velocity_ib_updated[_KEY_U]
      v = velocity_ib_updated[_KEY_V]
      w = velocity_ib_updated[_KEY_W]

    updated_velocity: dict[str, Any] = {
        _KEY_U: u,
        _KEY_V: v,
        _KEY_W: w,
        _KEY_RHO_U: rho_mid * u,
        _KEY_RHO_V: rho_mid * v,
        _KEY_RHO_W: rho_mid * w,
    }

    # Output nu_t if SGS is enabled and the caller tracks it.
    if nu_t is not None and 'nu_t' in additional_states:
      updated_velocity['nu_t'] = nu_t

    return updated_velocity

  def correction_step(
      self,
      states: ScalarFieldMap,
      states_0: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      mesh: jax.sharding.Mesh,
  ) -> dict[str, ScalarField]:
    """Updates momentum and velocity from the pressure correction.

    Args:
      states: Flow field variables from the latest prediction. Must have 'dp'.
      states_0: Flow field variables from the previous time step.
      additional_states: Helper variables.
      mesh: JAX mesh for device topology.

    Returns:
      Dictionary with corrected u, v, w, rho_u, rho_v, rho_w.
    """
    dt = self._params.dt

    if self._params.solver_mode == thermodynamics_pb2.Thermodynamics.ANELASTIC:
      rho_mid = states[_KEY_RHO]
    else:
      rho_mid = 0.5 * (states[_KEY_RHO] + states_0[_KEY_RHO])

    dp = states[_KEY_DP]

    # Correct momentum: rho_u <- rho_u - dt * grad(dp)
    # Each momentum component pairs with its physical axis: rho_u→x, rho_v→y,
    # rho_w→z, regardless of data_axis_order.
    momentum_keys = (_KEY_RHO_U, _KEY_RHO_V, _KEY_RHO_W)
    states_new: dict[str, ScalarField] = {}

    for mom_key, axis in zip(momentum_keys, ('x', 'y', 'z')):
      grad_dp = self._deriv_lib.deriv_centered(dp, axis, additional_states)
      if (
          self._params.solver_mode
          == thermodynamics_pb2.Thermodynamics.ANELASTIC
      ):
        states_new[mom_key] = states[mom_key] - dt * rho_mid * grad_dp
      else:
        states_new[mom_key] = states[mom_key] - dt * grad_dp

    # Compute velocity from corrected momentum.
    u = states_new[_KEY_RHO_U] / rho_mid
    v = states_new[_KEY_RHO_V] / rho_mid
    w = states_new[_KEY_RHO_W] / rho_mid

    # Apply immersed boundary constraints before halo exchange.
    if self._ib is not None:
      ib_updated = self._ib.update_states(
          {_KEY_U: u, _KEY_V: v, _KEY_W: w, _KEY_RHO: states[_KEY_RHO]},
          additional_states,
      )
      u = ib_updated[_KEY_U]
      v = ib_updated[_KEY_V]
      w = ib_updated[_KEY_W]

    # Update wall boundary conditions before halo exchange.
    wall_bc = self._update_wall_bc({_KEY_U: u, _KEY_V: v, _KEY_W: w})

    # Exchange velocity halos.
    u = self.exchange_velocity_halos(u, _KEY_U, mesh, wall_bc)
    v = self.exchange_velocity_halos(v, _KEY_V, mesh, wall_bc)
    w = self.exchange_velocity_halos(w, _KEY_W, mesh, wall_bc)

    states_new.update({
        _KEY_U: u,
        _KEY_V: v,
        _KEY_W: w,
    })

    return states_new
