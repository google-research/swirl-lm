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
"""Monin-Obukhov Similarity Theory (MOST) for atmospheric boundary layers.

Provides a wall model for computing surface shear stress, heat flux, and
scalar diffusive flux. Used in LES of atmospheric boundary layers.

The implementation follows Stoll & Porte-Agel (2008, 2009) with stability
corrections for stable, neutral, and unstable regimes.

Key functions:
  - surface_shear_stress_and_heat_flux_update_fn: Computes tau_13, tau_23,
    and q_3 for a given flow state. Used in the shear flux computation.
  - surface_flux_update_fn: Computes prescribed diffusive flux for scalars.
  - neumann_bc_update_fn: Computes Neumann boundary conditions for variables.
  - monin_obukhov_similarity_theory_factory: Creates a MOST instance from
    simulation parameters.
"""


from typing import Any
import jax.numpy as jnp
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.base import physical_variable_keys_manager
from swirl_lm.jax.numerics import root_finder
from swirl_lm.jax.utility import common_ops
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import types
from swirl_lm.physics import constants

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

# The von Karman constant.
_KAPPA = 0.4

# Threshold for height / surface roughness ratio.
_HEIGHT_TO_SURFACE_ROUGHNESS_RATIO_THRESHOLD = 1.1

# Key for constant exchange coefficient for momentum flux.
_MOMENTUM_FLUX_EXCHANGE_COEFF_KEY = 'momentum'


class MoninObukhovSimilarityTheory:
  """Monin-Obukhov Similarity Theory wall model.

  Computes surface shear stress, heat flux, and scalar diffusive flux
  for atmospheric boundary layer simulations.
  """

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
      vertical_dim: int,
  ):
    """Initializes the MOST model.

    Args:
      params: Simulation parameters.
      vertical_dim: The dimension index (0, 1, or 2) aligned with gravity.
    """
    self.params = params
    self.nu = params.nu
    self.halo_width = params.halo_width

    # Height of first fluid layer above the ground.
    use_stretched_grid = params.grid_params.to_xyz_order(
        params.use_stretched_grid  # pyrefly: ignore[bad-argument-type]
    )
    if use_stretched_grid[vertical_dim]:
      global_xyz_raw = (
          params.global_xyz
          if getattr(params, 'global_xyz', None) is not None
          else getattr(params.grid_params, 'global_xyz', None)
      )
      if global_xyz_raw is not None:
        global_xyz = params.grid_params.to_xyz_order(
            global_xyz_raw
        )
        self.height = global_xyz[vertical_dim][0]
      else:
        raise ValueError(
            'Stretched grid with MOST requires `global_xyz` in params.'
        )
    else:
      grid_spacings = params.grid_params.to_xyz_order(
          params.grid_spacings  # pyrefly: ignore[bad-argument-type]
      )
      self.height = 0.5 * grid_spacings[vertical_dim]

    assert (
        boundary_models := params.boundary_models
    ) is not None, '`boundary_models` must be provided.'
    most_params = boundary_models.most

    self.z_0 = most_params.z_0
    self.z_t = most_params.z_t
    self.u_star = most_params.u_star
    self.t_0 = most_params.t_0
    self.t_s = most_params.t_s
    self.heat_flux = most_params.heat_flux
    self.beta_m = most_params.beta_m
    self.beta_h = most_params.beta_h
    self.gamma_m = most_params.gamma_m
    self.gamma_h = most_params.gamma_h
    self.alpha = most_params.alpha
    self._active_scalars = list(most_params.active_scalar)

    self.enable_theta_reg = most_params.enable_theta_reg
    self.theta_max = most_params.theta_max
    self.theta_min = most_params.theta_min
    self.surface_gustiness = most_params.surface_gustiness

    # Vertical and horizontal dimension metadata.
    self.vertical_dim = vertical_dim
    self.horizontal_dims = [0, 1, 2]
    self.horizontal_dims.remove(vertical_dim)
    self._velocity_keys = ('u', 'v', 'w')
    self.bc_manager = (
        physical_variable_keys_manager.BoundaryConditionKeysHelper()
    )

    # Map axes for get_face.
    self._g_axis = ('x', 'y', 'z')[vertical_dim]

    self.sea_level_ref: dict[str, float] = {
        var.name: var.value for var in most_params.sea_level_ref
    }
    self.exchange_coeff: dict[str, float] = {
        var.name: var.value for var in most_params.exchange_coeff
    }

  def is_active_scalar(self, scalar_name: str) -> bool:
    """Checks if MOST is applied to a specific scalar."""
    return scalar_name in self._active_scalars

  def _stability_correction_function(
      self,
      zeta: ScalarField,
      theta: ScalarField,
  ) -> tuple[ScalarField, ScalarField]:
    """Computes stability correction functions psi_m and psi_h.

    Based on Stoll & Porte-Agel (2008, 2009):
    - Stable (theta > t_s): psi_m = -beta_m * zeta, psi_h = -beta_h * zeta
    - Neutral (theta == t_s): psi_m = psi_h = 0
    - Unstable (theta < t_s): Businger-Dyer formulation with gamma_m, gamma_h

    Args:
      zeta: Normalized height z/L.
      theta: Potential temperature.

    Returns:
      Tuple of (psi_m, psi_h) stability correction functions.
    """
    b = theta - self.t_s

    # Stable: psi = -beta * zeta
    psi_m_stable = -self.beta_m * zeta
    psi_h_stable = -self.beta_h * zeta

    # Neutral: psi = 0
    psi_m_neutral = jnp.zeros_like(zeta)
    psi_h_neutral = jnp.zeros_like(zeta)

    # Unstable: Businger-Dyer formulation.
    x_m = jnp.maximum(1.0 - self.gamma_m * zeta, 0.0) ** 0.25
    psi_m_unstable = (
        2.0 * jnp.log((1.0 + x_m) / 2.0)
        + jnp.log((1.0 + x_m**2) / 2.0)
        - 2.0 * jnp.arctan(x_m)
        + jnp.pi / 2.0
    )
    x_h = jnp.maximum(1.0 - self.gamma_h * zeta, 0.0) ** 0.5
    psi_h_unstable = 2.0 * jnp.log((1.0 + x_h) / 2.0)

    # Select based on buoyancy condition.
    psi_m = jnp.where(
        b > 0.0,
        psi_m_stable,
        jnp.where(b < 0.0, psi_m_unstable, psi_m_neutral),
    )
    psi_h = jnp.where(
        b > 0.0,
        psi_h_stable,
        jnp.where(b < 0.0, psi_h_unstable, psi_h_neutral),
    )
    return psi_m, psi_h

  def _richardson_number(
      self,
      theta: ScalarField,
      u1: ScalarField,
      u2: ScalarField,
      height: float,
  ) -> ScalarField:
    """Computes the bulk Richardson number Rb = g*z*(theta - t_s) / (|u|^2 * theta)."""
    u_sq = u1**2 + u2**2
    return jnp.where(
        u_sq * theta != 0.0,
        constants.G * height * (theta - self.t_s) / (u_sq * theta),
        0.0,
    )

  def _normalized_height(
      self,
      theta: ScalarField,
      u1: ScalarField,
      u2: ScalarField,
      height: float,
  ) -> ScalarField:
    """Computes the height normalized by the Obukhov length: zeta = z/L.

    Iteratively solves:
      Rb = zeta * [ln(z/z_0) - psi_h(zeta)] / [ln(z/z_0) - psi_m(zeta)]^2

    using Newton's method.

    Args:
      theta: Potential temperature at the first node above ground.
      u1: First horizontal velocity component.
      u2: Second horizontal velocity component.
      height: Height of the first grid point.

    Returns:
      The normalized height zeta = z/L.
    """
    ln_z_by_z0 = jnp.log(height / self.z_0)
    r_b = self._richardson_number(theta, u1, u2, height)

    def rhs_fn(zeta: ScalarField) -> ScalarField:
      """Residual: Rb - zeta * (ln(z/z0) - psi_h) / (ln(z/z0) - psi_m)^2."""
      psi_m, psi_h = self._stability_correction_function(zeta, theta)
      return r_b - zeta * (ln_z_by_z0 - psi_h) / (ln_z_by_z0 - psi_m) ** 2

    zeta_init = jnp.zeros_like(theta)
    return root_finder.newton_method(rhs_fn, zeta_init, max_iterations=10)

  def _maybe_regularize_potential_temperature(
      self, theta: ScalarField
  ) -> ScalarField:
    """Clips potential temperature to [theta_min, theta_max] if enabled."""
    if self.enable_theta_reg:
      return jnp.clip(theta, self.theta_min, self.theta_max)
    return theta

  def _surface_shear_stress_and_heat_flux(
      self,
      theta: ScalarField,
      u1: ScalarField,
      u2: ScalarField,
      rho: ScalarField,
      height: float,
  ) -> tuple[ScalarField, ScalarField, ScalarField]:
    """Computes surface shear stress and heat flux.

    Args:
      theta: Potential temperature at first node above ground.
      u1: First horizontal velocity.
      u2: Second horizontal velocity.
      rho: Density at first node above ground.
      height: Height of first grid point.

    Returns:
      (tau_13, tau_23, q_3): Surface shear stresses for u1 and u2, and the
      surface heat flux.
    """
    u_mag = jnp.sqrt(self.surface_gustiness**2 + u1**2 + u2**2)
    zeta = self._normalized_height(theta, u1, u2, height)
    phi_m, phi_h = self._stability_correction_function(zeta, theta)

    ln_z = jnp.log(height / self.z_0)

    if _MOMENTUM_FLUX_EXCHANGE_COEFF_KEY in self.exchange_coeff:
      drag_coefficient = self.exchange_coeff[_MOMENTUM_FLUX_EXCHANGE_COEFF_KEY]
    else:
      denom = rho * (ln_z - phi_m) ** 2
      drag_coefficient = jnp.where(denom != 0.0, _KAPPA**2 / denom, 0.0)

    tau_13 = -rho * drag_coefficient * u1 * u_mag
    tau_23 = -rho * drag_coefficient * u2 * u_mag

    u_s = (tau_13**2 + tau_23**2) ** 0.25
    phi_denom = ln_z - phi_h
    q_3 = jnp.where(
        phi_denom != 0.0,
        (self.t_s - theta) * u_s * _KAPPA / phi_denom,
        0.0,
    )

    return tau_13, tau_23, q_3

  def surface_shear_stress_and_heat_flux_update_fn(
      self,
      states: ScalarFieldMap,
  ) -> tuple[ScalarField, ScalarField, ScalarField]:
    """Computes wall shear stress and heat flux from current flow state.

    Extracts the first fluid layer above the ground for horizontal velocities,
    potential temperature, and density, then calls the core MOST computation.

    Args:
      states: Must include 'u', 'v', 'w', 'theta', 'rho'.

    Returns:
      (tau_s1, tau_s2, q_3): Surface shear stresses for the two horizontal
      velocity components and the surface heat flux.
    """
    velocity_keys = list(self._velocity_keys)
    del velocity_keys[self.vertical_dim]

    u1 = common_ops.get_face(
        states[velocity_keys[0]],
        self._g_axis,
        0,
        self.halo_width,
        self.params.grid_params,
    )
    u2 = common_ops.get_face(
        states[velocity_keys[1]],
        self._g_axis,
        0,
        self.halo_width,
        self.params.grid_params,
    )
    theta = self._maybe_regularize_potential_temperature(
        common_ops.get_face(
            states['theta'],
            self._g_axis,
            0,
            self.halo_width,
            self.params.grid_params,
        )
    )
    rho = common_ops.get_face(
        states['rho'],
        self._g_axis,
        0,
        self.halo_width,
        self.params.grid_params,
    )

    return self._surface_shear_stress_and_heat_flux(
        theta, u1, u2, rho, self.height
    )

  def _exchange_coefficient(
      self,
      theta: ScalarField,
      u1: ScalarField,
      u2: ScalarField,
      height: float,
      varname: str | None = None,
  ) -> ScalarField:
    """Computes the exchange coefficient for scalar/velocity transport.

    Args:
      theta: Potential temperature.
      u1: First horizontal velocity.
      u2: Second horizontal velocity.
      height: Height of first grid point.
      varname: Variable name. If a velocity/momentum key, uses phi_m; else uses
        phi_h.

    Returns:
      The exchange coefficient.
    """
    zeta = self._normalized_height(theta, u1, u2, height)
    phi_m, phi_h = self._stability_correction_function(zeta, theta)

    ln_z = jnp.log(height / self.z_0)

    momentum_keys = ('rho_u', 'rho_v', 'rho_w', 'u', 'v', 'w')
    phi_val = phi_m if varname in momentum_keys else phi_h

    denom = (ln_z - phi_val) * (ln_z - phi_m)
    return jnp.where(
        denom != 0.0,
        _KAPPA**2 / jnp.where(denom != 0.0, denom, 1.0),
        0.0,
    )

  def surface_flux_update_fn(
      self,
      states: ScalarFieldMap,
      varname: str | None = None,
  ) -> ScalarField:
    """Computes the diffusive flux at the surface for a scalar.

    Args:
      states: Must include 'u', 'v', 'w', 'theta', 'rho', 'phi'.
      varname: Variable name for exchange coefficient selection.

    Returns:
      The flux of 'phi' at the surface.
    """
    velocity_keys = list(self._velocity_keys)
    del velocity_keys[self.vertical_dim]

    u1 = common_ops.get_face(
        states[velocity_keys[0]],
        self._g_axis,
        0,
        self.halo_width,
        self.params.grid_params,
    )
    u2 = common_ops.get_face(
        states[velocity_keys[1]],
        self._g_axis,
        0,
        self.halo_width,
        self.params.grid_params,
    )
    theta = self._maybe_regularize_potential_temperature(
        common_ops.get_face(
            states['theta'],
            self._g_axis,
            0,
            self.halo_width,
            self.params.grid_params,
        )
    )
    rho = common_ops.get_face(
        states['rho'],
        self._g_axis,
        0,
        self.halo_width,
        self.params.grid_params,
    )
    phi_zm = common_ops.get_face(
        states['phi'],
        self._g_axis,
        0,
        self.halo_width,
        self.params.grid_params,
    )

    # Use the user-defined sea surface reference value if available,
    # otherwise use values at the first halo layer.
    phi_z0: ScalarField | float = self.sea_level_ref.get(  # pyrefly: ignore[no-matching-overload]
        varname,
        common_ops.get_face(
            states['phi'],
            self._g_axis,
            0,
            self.halo_width - 1,
            self.params.grid_params,
        ),
    )

    c_h: ScalarField | float = self.exchange_coeff.get(  # pyrefly: ignore[no-matching-overload]
        varname,
        self._exchange_coefficient(theta, u1, u2, self.height, varname),
    )

    u_mag = jnp.sqrt(self.surface_gustiness**2 + u1**2 + u2**2)
    return -rho * c_h * u_mag * (phi_zm - phi_z0)

  def neumann_bc_update_fn(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      states: ScalarFieldMap,
      additional_states: dict[str, Any],
  ) -> dict[str, Any]:
    """Computes the Neumann BC for all variables.

    Note: While this function is still used in geophysical_flow.py, it only sets
    values in halos, so that the gradient computed from these values is
    consistent with the shear stress/flux obtained from the MO similarity model.
    However, its functionality has been superseded by the preceding functions
    that set the diffusive fluxes directly. So in effect, this function is no
    longer necessary and could be removed and deprecated.

    Args:
      kernel_op: An object holding a library of kernel operations.
      states: A keyed dictionary of state variables.
      additional_states: A list of states that are needed by the update fn, but
        will not be updated by the main governing equations.

    Returns:
      An update function for `additional_states` that updates the boundary
      condition.
    """
    # Computes the boundary condition for all variables in states except for
    # those listed here.
    excluded_vars = ('p', 'rho', self._velocity_keys[self.vertical_dim])

    helper_states = {
        'u': states['u'],
        'v': states['v'],
        'w': states['w'],
        'rho': states['rho'],
    }

    def get_potential_temperature(variables):
      """Retrieves potential temperature from a dictionary of variables."""
      if 'T' in variables:
        # Temperature is used interchangeably with potential temperature because
        # they are almost identical on the ground.
        return variables['T']
      elif 'theta' in variables:
        return variables['theta']
      elif 'theta_li' in variables:
        return variables['theta_li']
      else:
        return None

    theta = get_potential_temperature(states)
    if theta is None:
      theta = get_potential_temperature(additional_states)
    if theta is None:
      raise ValueError(
          'Potential temperature is required by the MOST model but is not'
          ' provided.'
      )
    helper_states.update({'theta': theta})

    # Get the turbulent viscosity and diffusivity.
    nu_t = additional_states.get('nu_t', 0.0)
    if (
        self.params.sgs_model is not None
        and self.params.sgs_model.WhichOneof('sgs_model_type') == 'smagorinsky'
    ):
      pr_t = self.params.sgs_model.smagorinsky.pr_t
    elif (
        self.params.sgs_model is not None
        and self.params.sgs_model.WhichOneof('sgs_model_type')
        == 'smagorinsky_lilly'
    ):
      pr_t = self.params.sgs_model.smagorinsky_lilly.pr_t
    elif (
        self.params.sgs_model is not None
        and self.params.sgs_model.WhichOneof('sgs_model_type') == 'vreman'
    ):
      pr_t = self.params.sgs_model.vreman.pr_t
    else:
      # The turbulent Prandtl number is set to 1 for other SGS models.
      pr_t = 1.0

    d_t = nu_t / pr_t
    nu_total = self.params.nu + nu_t

    interp_fn = lambda f: 0.5 * kernel_op.apply_kernel_op(f, 'ks', self._g_axis)

    additional_states_new = dict(additional_states)
    for key, val in states.items():
      bc_key = self.bc_manager.generate_bc_key(key, self.vertical_dim, 0)
      if key in excluded_vars or bc_key not in additional_states:
        continue

      helper_states.update({'phi': val})
      flux = self.surface_flux_update_fn(helper_states, key)

      # Compute the gradient of the variable at the surface. Note that the
      # diffusive flux is computed as: flux = -\rho D \nabla\phi \delta_{ik},
      # where `k` indicates the vertical direction.
      if key in self._velocity_keys:
        d_total = nu_total
      else:
        d_total = self.params.diffusivity(key) + d_t
      rho_d = states['rho'] * d_total
      rho_d_face = -1.0 * common_ops.get_face(
          interp_fn(rho_d),
          self._g_axis,
          0,
          self.halo_width,
          self.params.grid_params,
      )
      grad_phi = flux / rho_d_face

      # Assume the outer halo layers retains the same value.
      zeros = jnp.zeros_like(grad_phi)
      bc = [zeros] * (self.halo_width - 1) + [grad_phi]
      bc = [b * (self.height * 2.0) for b in bc]

      additional_states_new[bc_key] = bc

    return additional_states_new


def monin_obukhov_similarity_theory_factory(
    params: parameters_lib.SwirlLMParameters,
) -> MoninObukhovSimilarityTheory:
  """Creates a MOST instance from simulation parameters.

  Args:
    params: Simulation parameters with `boundary_models.most` configured.

  Returns:
    A `MoninObukhovSimilarityTheory` instance.

  Raises:
    ValueError: If MOST or gravity is not configured.
    AssertionError: If the first fluid layer is below the surface roughness.
  """
  assert (
      boundary_models := params.boundary_models
  ) is not None, '`boundary_models` must be provided.'
  if not boundary_models.HasField('most'):
    raise ValueError(
        'Parameters for the Monin-Obukhov boundary layer model are not '
        'defined in the config.'
    )

  vertical_dim = params.g_dim
  if vertical_dim is None:
    raise ValueError(
        'Gravity must be defined to use the Monin-Obukhov boundary layer model.'
    )

  # Verify height > surface roughness.
  use_stretched_grid = params.grid_params.to_xyz_order(
      params.use_stretched_grid  # pyrefly: ignore[bad-argument-type]
  )
  if use_stretched_grid[vertical_dim]:
    global_xyz_raw = (
        params.global_xyz
        if getattr(params, 'global_xyz', None) is not None
        else getattr(params.grid_params, 'global_xyz', None)
    )
    if global_xyz_raw is not None:
      global_xyz = params.grid_params.to_xyz_order(
          global_xyz_raw
      )
      height = global_xyz[vertical_dim][0]
    else:
      # Stretched grid without global_xyz: skip height check.
      height = None
  else:
    grid_spacings = params.grid_params.to_xyz_order(
        params.grid_spacings  # pyrefly: ignore[bad-argument-type]
    )
    height = 0.5 * grid_spacings[vertical_dim]

  z_0_threshold = (
      _HEIGHT_TO_SURFACE_ROUGHNESS_RATIO_THRESHOLD * boundary_models.most.z_0
  )
  assert height is None or height > z_0_threshold, (
      f'The height of the first fluid layer ({height} m) is below the'
      f' tolerated surface roughness ({z_0_threshold} m). MOST model should'
      ' be disabled and replaced by a non-slip wall BC.'
  )

  return MoninObukhovSimilarityTheory(params, vertical_dim)
