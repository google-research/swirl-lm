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
"""Sub-grid scale (SGS) models for large-eddy simulation (JAX).

This is the JAX port of `swirl_lm.physics.turbulence.sgs_model`. It provides
SGS turbulent viscosity and diffusivity models for large-eddy simulation.

Supported models:
  - Smagorinsky (static): nu_t = (c_s * delta)^2 * |S|
  - Smagorinsky-Lilly (stability-corrected): delta scaled by f_b(Ri)
  - Vreman: nu_t = c_s * sqrt(B_beta / (alpha_ij * alpha_ij))

Current scope:
  - Smagorinsky model (diagonal and geometric mean delta formulas).
  - Smagorinsky-Lilly model (stratification correction via Richardson number).
  - Vreman model.
  - Clamping of nu_t and diff_t (min/max bounds from proto config).

Future extensions:
  - Dynamic Smagorinsky (Germano procedure) -- needs global averaging.
"""


from collections.abc import Sequence
import itertools

from absl import logging
import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.numerics import calculus
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

# Default Smagorinsky constant for isotropic, homogeneous turbulence.
_CS_CLASSICAL = 0.18
# Gravitational acceleration [m/s^2] for Richardson number in Smagorinsky-Lilly.
_G = 9.81


def _strain_rate_tensor(
    du_dx: Sequence[Sequence[ScalarField]],
) -> list[list[ScalarField]]:
  """Computes the strain rate tensor.

  S_ij = 0.5*(du_i/dx_j + du_j/dx_i) - 1/3 div(u) delta_ij.

  Args:
    du_dx: Velocity gradient tensor. du_dx[i][j] = du_i/dx_j.

  Returns:
    The deviatoric strain rate tensor.
  """
  s_ij = [
      [0.5 * (du_dx[i][j] + du_dx[j][i]) for j in range(3)] for i in range(3)
  ]
  div_u = du_dx[0][0] + du_dx[1][1] + du_dx[2][2]

  return [
      [s_ij[i][j] - div_u / 3.0 if i == j else s_ij[i][j] for j in range(3)]
      for i in range(3)
  ]


def _strain_rate_magnitude(
    strain_rate: Sequence[Sequence[ScalarField]],
) -> ScalarField:
  """Computes |S| = sqrt(2 * S_ij * S_ij).

  Args:
    strain_rate: The strain rate tensor.

  Returns:
    The magnitude of the strain rate tensor.
  """
  s_sq = jnp.zeros_like(strain_rate[0][0])
  for i in range(3):
    for j in range(3):
      s_sq = s_sq + 2.0 * strain_rate[i][j] ** 2
  return jnp.sqrt(s_sq)


class SgsModel:
  """Sub-grid scale model for large-eddy simulation."""

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
  ):
    """Initializes the SGS model.

    Args:
      params: The simulation parameters.
    """
    self._params = params
    self._deriv_lib = params.deriv_lib
    self._sgs_params = params.sgs_model

    # Extract clamping bounds (None if not set in proto).
    if self._sgs_params is not None:
      self._nu_t_max = (
          self._sgs_params.nu_t_max
          if self._sgs_params.HasField('nu_t_max')
          else None
      )
      self._diff_t_max = (
          self._sgs_params.diff_t_max
          if self._sgs_params.HasField('diff_t_max')
          else None
      )
      self._nu_t_min = self._sgs_params.nu_t_min
      self._diff_t_min = self._sgs_params.diff_t_min
    else:
      self._nu_t_max = None
      self._diff_t_max = None
      self._nu_t_min = 0.0
      self._diff_t_min = 0.0

    if self._sgs_params is None or not self._sgs_params.HasField(
        'sgs_model_type'
    ):
      logging.warning(
          'SGS is used but no model is specified. Using Smagorinsky with '
          'default constants (C_s = 0.18, Pr_t = 0.3).'
      )
    else:
      logging.info('SGS model: %r', self._sgs_params)

  def _delta_square(
      self,
      ref_field: ScalarField,
      formula: str = 'DIAGONAL',
  ) -> ScalarField:
    """Computes the filter width squared.

    Args:
      ref_field: Reference field for shape.
      formula: 'DIAGONAL' for sqrt(dx^2+dy^2+dz^2)^2, 'GEOMETRIC_MEAN' for
        (dx*dy*dz)^(2/3).

    Returns:
      The filter width squared.
    """
    gs = self._params.grid_spacings
    if formula == 'GEOMETRIC_MEAN':
      return (gs[0] * gs[1] * gs[2]) ** (2.0 / 3.0) * jnp.ones_like(ref_field)
    else:  # DIAGONAL
      return (gs[0] ** 2 + gs[1] ** 2 + gs[2] ** 2) * jnp.ones_like(ref_field)

  def smagorinsky(
      self,
      field_vars: Sequence[ScalarField],
      additional_states: ScalarFieldMap,
      c_s: float | ScalarField | None = None,
      delta_formula: str = 'DIAGONAL',
  ) -> ScalarField:
    """Computes turbulent viscosity from the Smagorinsky model.

    nu_t = (c_s * delta)^2 * |S|, where |S| = sqrt(2 * S_ij * S_ij).

    Args:
      field_vars: Variables to compute SGS for. If length == 3, treated as
        velocity (uses strain rate tensor). Otherwise, uses gradient magnitude.
      additional_states: Helper variables.
      c_s: Smagorinsky constant. If None, uses _CS_CLASSICAL (0.18).
      delta_formula: 'DIAGONAL' or 'GEOMETRIC_MEAN'.

    Returns:
      The turbulent viscosity (or diffusivity).
    """
    if c_s is None:
      c_s_field = _CS_CLASSICAL * jnp.ones_like(field_vars[0])
    elif isinstance(c_s, (int, float)):
      c_s_field = c_s * jnp.ones_like(field_vars[0])
    else:
      c_s_field = c_s

    # Compute velocity gradient tensor.
    du_dx = calculus.grad(self._deriv_lib, list(field_vars), additional_states)

    # Use strain rate for velocity (3 components), gradient magnitude otherwise.
    if len(field_vars) == 3:
      s_ij = _strain_rate_tensor(du_dx)  # type: ignore[arg-type]
      s_mag = _strain_rate_magnitude(s_ij)  # type: ignore[arg-type]
    else:
      # For scalar fields, compute gradient magnitude |∇φ| directly.
      # du_dx is [[dφ/d0, dφ/d1, dφ/d2]] — a list of 1 list of 3 components.
      grad_phi = du_dx[0]
      s_mag = jnp.sqrt(
          sum(g**2 for g in grad_phi)
      )

    delta_sq = self._delta_square(field_vars[0], delta_formula)

    return c_s_field**2 * delta_sq * s_mag

  def vreman(
      self,
      velocity: Sequence[ScalarField],
      c_s: float,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes turbulent viscosity from the Vreman model.

    Vreman model: nu_t = c_s * sqrt(B_beta / (alpha_ij * alpha_ij))

    Reference: Vreman, 2004. An eddy-viscosity subgrid-scale model for
    turbulent shear flow: Algebraic theory and applications.

    Args:
      velocity: Tuple of (u, v, w) velocity components.
      c_s: Vreman model constant (typically ~0.07).
      additional_states: Helper variables.

    Returns:
      The turbulent viscosity.
    """
    gs = self._params.grid_spacings

    # Compute velocity gradient tensor.
    du_dx = calculus.grad(self._deriv_lib, list(velocity), additional_states)

    # beta_ij = sum_m (delta_m^2 * alpha_mi * alpha_mj)
    beta = [[jnp.zeros_like(velocity[0]) for _ in range(3)] for _ in range(3)]
    for i, j in itertools.product(range(3), range(3)):
      for m in range(3):
        beta[i][j] = beta[i][j] + gs[m] ** 2 * du_dx[m][i] * du_dx[m][j]  # type: ignore[index]

    # B_beta = beta_11*beta_22 - beta_12^2 + beta_11*beta_33 - beta_13^2
    #        + beta_22*beta_33 - beta_23^2
    b_beta = (
        beta[0][0] * beta[1][1]
        - beta[0][1] ** 2
        + beta[0][0] * beta[2][2]
        - beta[0][2] ** 2
        + beta[1][1] * beta[2][2]
        - beta[1][2] ** 2
    )

    # alpha_ij * alpha_ij
    alpha_sq = jnp.zeros_like(velocity[0])
    for i, j in itertools.product(range(3), range(3)):
      alpha_sq = alpha_sq + du_dx[i][j] ** 2  # type: ignore[index]

    # Avoid division by zero.
    safe_alpha_sq = jnp.maximum(alpha_sq, 1e-30)
    b_beta_positive = jnp.maximum(b_beta, 0.0)

    return c_s * jnp.sqrt(b_beta_positive / safe_alpha_sq)

  def smagorinsky_lilly(
      self,
      field_vars: Sequence[ScalarField],
      velocity: Sequence[ScalarField],
      temperature: ScalarField,
      c_s: float,
      pr_t: float,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes turbulent viscosity from the Smagorinsky-Lilly model.

    Adds a stratification correction f_b to the standard Smagorinsky model.
    The effective filter width is scaled by f_b, which depends on the Richardson
    number Ri = (g/T * dT/dz) / |S|^2.

    For unstable stratification (Ri <= 0): f_b = 1 (no correction).
    For stable stratification (Ri > 0): f_b = max(0, 1 - Ri/Pr_t)^0.25.

    Reference:
    Lilly, D. K. 1962. "On the Numerical Simulation of Buoyant Convection."
    Tellus 14 (2): 148-72.

    Args:
      field_vars: Variables for SGS computation. If length is 3, treated as
        velocity and strain rate is used; otherwise treated as scalars.
      velocity: The 3 velocity components (for computing strain rate |S|).
      temperature: The temperature field (for computing Richardson number).
      c_s: The Smagorinsky constant.
      pr_t: The turbulent Prandtl number.
      additional_states: Helper variables.

    Returns:
      The turbulent viscosity/diffusivity.
    """
    axes = self._params.grid_params.data_axis_order
    g_axis = axes[2]  # Vertical dimension axis name.

    # Compute strain rate magnitude from velocity.
    du_dx = calculus.grad(self._deriv_lib, list(velocity), additional_states)
    s_ij = _strain_rate_tensor(du_dx)  # pyrefly: ignore[bad-argument-type]
    strain_rate_mag = _strain_rate_magnitude(s_ij)

    # Compute Richardson number: Ri = (g/T * dT/dz) / |S|^2.
    dt_dz = self._deriv_lib.deriv_centered(
        temperature, g_axis, additional_states
    )
    buoyancy_freq_sq = _G / temperature * dt_dz
    ri = jnp.where(
        strain_rate_mag > 0,
        buoyancy_freq_sq / strain_rate_mag**2,
        jnp.zeros_like(strain_rate_mag),
    )

    # Stratification correction f_b.
    f_b = jnp.where(
        ri <= 0.0,
        jnp.ones_like(ri),
        jnp.maximum(0.0, 1.0 - ri / pr_t) ** 0.25,
    )

    # Compute the gradient magnitude of field_vars.
    df_dx = calculus.grad(self._deriv_lib, list(field_vars), additional_states)
    if len(field_vars) == 3:
      df_ij = _strain_rate_tensor(df_dx)  # pyrefly: ignore[bad-argument-type]
    else:
      df_ij = df_dx
    df_mag = _strain_rate_magnitude(df_ij)  # pyrefly: ignore[bad-argument-type]

    # delta_lilly = (dx * dy * dz)^(1/3) * f_b
    gs = self._params.grid_spacings
    delta = (gs[0] * gs[1] * gs[2]) ** (1.0 / 3.0) * f_b

    return (c_s * delta) ** 2 * df_mag

  def turbulent_viscosity(
      self,
      velocity: Sequence[ScalarField],
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes the turbulent viscosity for the momentum equations.

    Args:
      velocity: Tuple of (u, v, w) velocity components.
      additional_states: Helper variables.

    Returns:
      The turbulent viscosity field.
    """
    if self._sgs_params is None or not self._sgs_params.HasField(
        'sgs_model_type'
    ):
      nu_t = self.smagorinsky(velocity, additional_states)
    elif self._sgs_params.WhichOneof('sgs_model_type') == 'smagorinsky':
      c_s = additional_states.get(
          'c_s',
          self._sgs_params.smagorinsky.c_s * jnp.ones_like(velocity[0]),
      )
      # DeltaFormula enum: DIAGONAL=1, GEOMETRIC_MEAN=2.
      delta_formula = (
          'GEOMETRIC_MEAN'
          if self._sgs_params.smagorinsky.delta_formula == 2
          else 'DIAGONAL'
      )
      nu_t = self.smagorinsky(velocity, additional_states, c_s, delta_formula)
    elif self._sgs_params.WhichOneof('sgs_model_type') == 'vreman':
      nu_t = self.vreman(
          velocity, self._sgs_params.vreman.c_s, additional_states
      )
    elif self._sgs_params.WhichOneof('sgs_model_type') == 'smagorinsky_lilly':
      if 'theta_v' not in additional_states:
        raise ValueError(
            '`theta_v` is required in `additional_states` for the '
            'Smagorinsky-Lilly model.'
        )
      nu_t = self.smagorinsky_lilly(
          velocity,
          velocity,
          additional_states['theta_v'],
          self._sgs_params.smagorinsky_lilly.c_s,
          self._sgs_params.smagorinsky_lilly.pr_t,
          additional_states,
      )
    else:
      raise ValueError(
          'Unsupported SGS model:'
          f' {self._sgs_params.WhichOneof("sgs_model_type")}'
      )

    # Clamp nu_t.
    nu_t = jnp.maximum(nu_t, self._nu_t_min)
    if self._nu_t_max is not None:
      nu_t = jnp.minimum(nu_t, self._nu_t_max)
    return nu_t

  def turbulent_diffusivity(
      self,
      field_vars: Sequence[ScalarField],
      additional_states: ScalarFieldMap,
      velocity: Sequence[ScalarField] | None = None,
  ) -> ScalarField:
    """Computes the turbulent diffusivity for scalar transport.

    Args:
      field_vars: Scalars based on which the model is computed.
      additional_states: Helper variables.
      velocity: Velocity components (required for models using Pr_t).

    Returns:
      The turbulent diffusivity field.
    """
    if self._sgs_params is None or not self._sgs_params.HasField(
        'sgs_model_type'
    ):
      return self.smagorinsky(field_vars, additional_states)

    model_type = self._sgs_params.WhichOneof('sgs_model_type')

    if model_type == 'smagorinsky':
      use_pr_t = self._sgs_params.smagorinsky.use_pr_t
      pr_t = self._sgs_params.smagorinsky.pr_t
      coeff = np.sqrt(pr_t) if use_pr_t else 1.0
      c_s_base = additional_states.get(
          'c_s',
          self._sgs_params.smagorinsky.c_s * jnp.ones_like(field_vars[0]),
      )
      if isinstance(c_s_base, (int, float)):
        c_s = c_s_base / coeff
      else:
        c_s = c_s_base / coeff
      sgs_vars = velocity if use_pr_t and velocity is not None else field_vars
      delta_formula = (
          'GEOMETRIC_MEAN'
          if self._sgs_params.smagorinsky.delta_formula == 2
          else 'DIAGONAL'
      )
      diff_t = self.smagorinsky(sgs_vars, additional_states, c_s, delta_formula)
    elif model_type == 'vreman':
      if velocity is None:
        raise ValueError('Velocity is required for Vreman SGS diffusivity.')
      nu_t = self.vreman(
          velocity, self._sgs_params.vreman.c_s, additional_states
      )
      diff_t = nu_t / self._sgs_params.vreman.pr_t
    elif model_type == 'smagorinsky_lilly':
      if 'theta_v' not in additional_states:
        raise ValueError(
            '`theta_v` is required in `additional_states` for the '
            'Smagorinsky-Lilly model.'
        )
      if velocity is None:
        raise ValueError(
            'Velocity is required for Smagorinsky-Lilly SGS diffusivity.'
        )
      use_pr_t = self._sgs_params.smagorinsky_lilly.use_pr_t
      pr_t = self._sgs_params.smagorinsky_lilly.pr_t
      coeff = np.sqrt(pr_t) if use_pr_t else 1.0
      c_s = self._sgs_params.smagorinsky_lilly.c_s / coeff
      sgs_vars = velocity if use_pr_t else field_vars
      diff_t = self.smagorinsky_lilly(
          sgs_vars,
          velocity,
          additional_states['theta_v'],
          c_s,
          pr_t,
          additional_states,
      )
    else:
      raise ValueError(f'Unsupported SGS model for diffusivity: {model_type}')

    # Clamp diff_t.
    diff_t = jnp.maximum(diff_t, self._diff_t_min)
    if self._diff_t_max is not None:
      diff_t = jnp.minimum(diff_t, self._diff_t_max)
    return diff_t
