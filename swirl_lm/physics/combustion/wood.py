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

# coding=utf-8
"""A library for the combustion modeling of wood.

The governing equations for the solid phase are [2]:
  ∂ϱf/∂t = -Nf Ff,
  ∂ϱw/∂t = -Fw,
  (Cpf ϱf + Cpw ϱw) ∂Tₛ/∂t = Qradₛ + h aᵥ (Tg - Tₛ) - Fw (Hw + Cpw Tᵥₐₚ) +
      Ff(𝚹 Hf - Cpf Tpyr Nf).
These reactions result in source terms in the Navier-Stokes equations:
  𝜔ₘₐₛₛ = Nf Ff + Fw,
  𝜔ₜₑₘₚₑᵣₐₜᵤᵣₑ = 1 / Cpg [h aᵥ (Tₛ - Tg) + Qrad,g + (1 - 𝚹) Ff Hf],
  𝜔ₒ = -Nₒ Ff,
where the effective stoichiometric coefficients for fuel and oxidizer are
  Nf = 0.4552 and
  Nₒ = 0.5448.

In case where the reaction of wood is considered local, the reaction rate is
modeled as [1] (Eq. 4.9):
  Ff = cF ϱf ϱo σcm 𝚿s 𝛌of / (ϱref sₓ²),
where:
  ϱf is the density of the fuel, e.g. 2 kg/m³,
  ϱref = 1.0 kg/m³ is the reference density,
  cF = 0.07 is an empirical scaling coefficient with which a fire in a 1 m/s
    wind can barely sustain itself,
  sₓ is the scale of the smallest fuel elements. Candidate values for sₓ
  are:
    4.0 m: For the features of the crown of the fuel bed (A scale);
    2.0 m: For the distance between branches or trunks of vegetation (B scale);
    0.05 m: For the clumps of leaves or needles on the small limbs (C scale),
  𝚿s = min((T - 300) / 400, 1.0) is a linear temperature function that
    represents ignited volume fraction,
  𝛌of = ϱf ϱo / (ϱf / Nf + ϱo / No)², and
  σcm = 0.09 ϱg sB √K is the turbulent diffusivity [2], K = Rii/2ϱg is the
    turbulenct kinetic energy of the B scale.

The effective heat of reaction is 8440 kJ/kg.

References:
[1] Linn, Rodman Ray. 1997. “A Transport Model for Prediction of Wildfire
    Behavior (No. LA-13334-T).” Edited by Francis H. Harlow. Ph.D, Los Alamos
    National Lab., NM (United States).
[2] Linn, Rodman R. 2005. “Numerical Simulations of Grass Fires Using a Coupled
    Atmosphere–fire Model: Basic Fire Behavior and Dependence on Wind Speed.”
    Journal of Geophysical Research 110 (D13): 287.
"""

import functools
import logging
from typing import Callable, Optional, Sequence

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.communication import halo_exchange
from swirl_lm.numerics import time_integration
from swirl_lm.physics.combustion import wood_pb2
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.physics.turbulent_combustion import turbulent_combustion_generic
from swirl_lm.utility import composite_types
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap
StatesUpdateFn = composite_types.StatesUpdateFn
# The Stefan-Boltzman constant, in unit of W/m^2/K^4.
_SIGMA = 5.67e-8
# Effective stoichiometric coefficients for fuel and oxidizer.
_N_F = 0.4552
_N_O = 0.5448
# The reference density, kg/m³.
_RHO_REF = 1.0
# The specific heat for solid fuel and water, J/kg/K.
_CP_F = 1850.0
_CP_W = 4182.0
# The relative threshold with respect to the maximum fuel density below which
# the fuel is considered depleted.
_EPSILON = 1e-3

_TF_DTYPE = types.TF_DTYPE


def _bound_scalar(
    phi: FlowFieldVal,
    minval: float = 0.0,
    maxval: Optional[float] = None,
) -> FlowFieldVal:
  """Applies physical bounds to the scalar `phi`.

  Combustion related scalars, such as mass/mole fractions, are typically bounded
  between 0 and 1. Enforcing physical bounds to these scalars will improve the
  stability of the combustion model.

  Args:
    phi: The scalar to which the bounds are applied.
    minval: The lower bound of the scalar.
    maxval: The upper bound of the scalar.

  Returns:
    The regularized scalar `phi` so that it is within the physical bounds.

  Raises:
    ValueError: If `maxval` is smaller than `minval`.
  """
  if maxval is not None and maxval < minval:
    raise ValueError(
        'The upper bound for scalar needs to be greater than the lower one. {} '
        'is provided while the lower bound is {}.'.format(maxval, minval))

  def apply_bound(phi_i: tf.Tensor) -> tf.Tensor:
    """Applies bound to a tf.Tensor."""
    phi_i = tf.maximum(phi_i, minval * tf.ones_like(phi_i, dtype=_TF_DTYPE))
    if maxval is not None:
      phi_i = tf.minimum(phi_i, maxval * tf.ones_like(phi_i, dtype=_TF_DTYPE))
    return phi_i

  return apply_bound(phi) if isinstance(
      phi, tf.Tensor) else [apply_bound(phi_i) for phi_i in phi]


def _reaction_rate(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    rho_f: tf.Tensor,
    rho_g: tf.Tensor,
    y_o: tf.Tensor,
    tke: tf.Tensor,
    temperature: tf.Tensor,
    s_b: float,
    s_x: float,
    c_f: float,
    t_0_ivf: float,
    t_1_ivf: float,
    periodic_dims: list[bool],
    halo_width: int,
    w_axis: float = 0.15,
    w_center: float = 0.95,
    turbulent_combustion_model: Optional[
        turbulent_combustion_generic.TurbulentCombustionGeneric
    ] = None,
    apply_temperature_filter: bool = True,
) -> tf.Tensor:
  """Computes the reation rate of the wood combustion.

  The equation for the reaction rate is:
    Ff = cF ϱf ϱo σcm 𝚿s 𝛌of / (ϱref sₓ²).

  Args:
    replica_id: The index of the current TPU replica.
    replicas: A numpy array that maps grid coordinates to replica id numbers.
    rho_f: The bulk density of the fuel in a unit volume, kg/m³.
    rho_g: The density of the surrounding gas, kg/m³.
    y_o: The mass fraction of the oxidizer.
    tke: The turbulent kinetic energy.
    temperature: The bulk temperature in a unit volume, K.
    s_b: The B scale of the fuel elements.
    s_x: The scale of the smallest fuel elements.
    c_f: An empirical scaling coefficient in local fire reaction rates.
    t_0_ivf: Start temperature for the ramp up.
    t_1_ivf: End temperature for the ramp up.
    periodic_dims: A list of booleans indicating whether the variable `psi` is
      periodic along each dimension.
    halo_width: The width of the halo.
    w_axis: The fraction of the weights to be applied along the axes.
    w_center: The fraction of the weights to be applied at the center in the
      3 x 3 stencil.
    turbulent_combustion_model: The turbulence closure for the reaction source
      term.
    apply_temperature_filter: Whether to apply the temperature filter to the
      temperature field before computing the reaction source term.

  Returns:
    The reaction rate due to wood combustion.
  """

  def sigma_cm():
    """Computes the turbulent diffusivity sigma_cm = 0.09 ϱg sB √K."""
    return 0.09 * rho_g * s_b * tf.math.sqrt(tke)

  def psi_s():
    """Computes the ignited volume fraction."""
    return tf.clip_by_value(
        (temperature - t_0_ivf) / (t_1_ivf - t_0_ivf), 0.0, 1.0
    )

  def lambda_of():
    """Computes 𝛌of = ϱf ϱo / (ϱf / Nf + ϱo / No)2."""
    return tf.math.divide_no_nan(
        rho_f * rho_g * y_o, (rho_f / _N_F + rho_g * y_o / _N_O) ** 2
    )

  rho_f = _bound_scalar(rho_f, minval=0.0)
  y_o = _bound_scalar(y_o, minval=0.0, maxval=1.0)
  if apply_temperature_filter:
    temperature = dim12_filter(
        replica_id,
        replicas,
        temperature,
        periodic_dims,
        halo_width,
        w_axis,
        w_center,
    )

  src = (
      c_f
      * rho_f
      * rho_g
      * y_o
      * sigma_cm()
      * psi_s()
      * lambda_of()
      / (_RHO_REF * s_x**2)
  )

  if turbulent_combustion_model is not None:
    src = turbulent_combustion_model.update_source_term(src)
    # The following assertion is required to avoid the "incorrect return type"
    # error.
    assert isinstance(src, tf.Tensor), (
        'Source term dtype changed unexpectedly. A `tf.Tensor` is expected,'
        f' but got {type(src)}.'
    )

  return src


def _radiative_emission(
    t: tf.Tensor,
    t_ambient: tf.Tensor,
    l: float,
    k: float = 1.0,
) -> tf.Tensor:
  """Computes the radiation source for emission.

  Args:
    t: The temperature of the source of emission, in units of K.
    t_ambient: The ambient temperature, in units of K.
    l: The length scale of radiation, in units of m.
    k: A scaling factor that balances the sub-grid effect (> 1) and the
      emissivity (< 1).

  Returns:
    The radiation source term due to emission. If `t` is less than `t_ambient`,
    the radiation term is 0, i.e., radiation energy can only be lost to
    ambient conditions.
  """
  return tf.maximum(_SIGMA * k / l * (t**4 - t_ambient**4), 0.0)


def _evaporation(
    t: tf.Tensor,
    phi_max: tf.Tensor,
    rho_m: tf.Tensor,
    dt: float,
    c_w: float,
) -> tuple[FlowFieldVal, FlowFieldVal]:
  """Computes the evaporation rate and update the moisture CDF in fuel.

  Args:
    t: The temperature of the fuel, in units of K.
    phi_max: The cumulative density function of the maximum amount of water that
      has been evaporated.
    rho_m: The volume averaged moisture density, in units of kg/m^3.
    dt: The time step size, in units of s.
    c_w: An empirical scaling coefficient for the evaporation rate.

  Returns:
    A tuple with its first element being the evaporation rate of the moisture in
    fuel, in units of kg/m^3/s, and its second element being the updated
    moisture CDF.
  """
  phi = tf.minimum(tf.maximum((t - 310.0) / 126.0, 0.0), 1.0)
  return (c_w * rho_m * tf.maximum(phi - phi_max, 0.0) / dt,
          tf.maximum(phi, phi_max))


def _src_oxidizer(f_f: tf.Tensor) -> tf.Tensor:
  """Computes the oxidizer mass fraction source term.

  Args:
    f_f: The reaction rate of wood combustion, in units of kg/m^3/s.

  Returns:
    The rate of consumption of the oxidizer, in units of kg/m^3/s.
  """
  return -_N_O * f_f


def _src_fuel(f_f: tf.Tensor) -> tf.Tensor:
  """Computes the fuel source term.

  Args:
    f_f: The reaction rate of wood combustion, in units of kg/m^3/s.

  Returns:
    The rate of consumption of the fuel, in units of kg/m^3/s.
  """
  return -_N_F * f_f


def _theta(
    rho_f: FlowFieldVal,
    rho_f_init: Optional[FlowFieldVal] = None,
) -> FlowFieldVal:
  """Computes the fraction of heat feedback to solid after combustion.

  Args:
    rho_f: The current fuel density in a unit volume, in units of kg/m^3.
    rho_f_init: The initial fuel density in a unit volume, in units of kg/m^3.

  Returns:
    The fraction of heat feedback to the solid after combustion.
  """
  if rho_f_init is None:
    # Assume reaction heat transfer to gas and solid with equal probability
    # if the initial fuel state is missing.
    return tf.nest.map_structure(
        lambda rho_f_i: 0.5 * tf.ones_like(rho_f_i), rho_f
    )

  rho_f_0 = rho_f_init

  # Note that `divide_no_nan` is used here, which returns 0 when the initial
  # fuel density is 0. This suggests that no heat is transferred to the solid
  # when there is no fuel.
  return tf.nest.map_structure(
      lambda rho_f_i, rho_f_0_i: 1.0  # pylint: disable=g-long-lambda
      - tf.math.divide_no_nan(rho_f_i, rho_f_0_i),
      rho_f,
      rho_f_0,
  )


def _localize_by_fuel(
    rho_f: FlowFieldVal,
    src: FlowFieldVal,
) -> FlowFieldVal:
  """Sets source term to zero where there is no fuel.

  Args:
    rho_f: The current fuel density in a unit volume, in units of kg/m^3.
    src: The source term of a scalar solved in the simulation.

  Returns:
    The source term of a scalar, which only exists where there fuel is non-zero.
  """
  return tf.nest.map_structure(
      lambda f, s: tf.where(  # pylint: disable=g-long-lambda
          tf.math.less_equal(f, _EPSILON), tf.zeros_like(s),
          s),
      rho_f,
      src)


def _compute_mid_state(
    state_old: FlowFieldVal,
    state_new: FlowFieldVal,
) -> FlowFieldVal:
  """Computes the states at the middle step with linear interpolation.

  Args:
    state_old: The state at the beginning of the time step.
    state_new: The state at the end of the time step.

  Returns:
    The state at the middle of the time step.
  """
  return tf.nest.map_structure(
      lambda state_new_i, state_old_i: 0.5 * (state_new_i + state_old_i),
      state_new,
      state_old,
  )


def dim12_filter(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    psi: tf.Tensor,
    periodic_dims: list[bool],
    halo_width: int,
    w_axis: float = 0.15,
    w_center: float = 0.95,
) -> tf.Tensor:
  """Applies a 2D filter along dimensions 1 and 2 for the variable `psi`.

  Args:
    replica_id: The index of the current TPU replica.
    replicas: A numpy array that maps grid coordinates to replica id numbers.
    psi: The 3D variable `psi` to be filtered.
    periodic_dims: A list of booleans indicating whether the variable `psi` is
      periodic along each dimension.
    halo_width: The width of the halo.
    w_axis: The fraction of the weights to be applied along the axes.
    w_center: The fraction of the weights to be applied at the center in the
      3 x 3 stencil.

  Returns:
    The filtered `psi`.
  """
  logging.info('Temperature filter: w_axis: %s, w_center: %s', w_axis, w_center)
  halo_dims = (0, 1, 2)
  replica_dims = (0, 1, 2)
  bc = [
      [(halo_exchange.BCType.NEUMANN, 0.0)] * 2,
  ] * 3
  psi = halo_exchange.inplace_halo_exchange(
      psi,
      halo_dims,
      replica_id,
      replicas,
      replica_dims,
      periodic_dims,
      bc,
      halo_width,
  )
  w_corner = (1 - w_center) * (1 - w_axis) * 0.25
  w_axis = (1 - w_center) * w_axis * 0.25
  filters = tf.constant(
      [
          [w_corner, w_axis, w_corner],
          [w_axis, w_center, w_axis],
          [w_corner, w_axis, w_corner],
      ],
      dtype=tf.float32,
  )[..., tf.newaxis, tf.newaxis]
  # Here we assume that the 0th dimension of `psi` is the batch dimension, which
  # corresponds to the height of the 3D domain.
  inputs = psi[..., tf.newaxis]
  return tf.nn.conv2d(inputs, filters, strides=[1, 1, 1, 1], padding='SAME')[
      ..., 0
  ]


class Wood(object):
  """A library of wood combustion."""

  def __init__(
      self,
      model_params: wood_pb2.Wood,
      thermodynamics_model: thermodynamics_manager.ThermodynamicsManager,
      swirl_lm_params: parameters_lib.SwirlLMParameters,
  ):
    """Initializes the wood combustion library.

    Args:
      model_params: The parameters for the wood combustion model.
      thermodynamics_model: The thermodynamics model.
      swirl_lm_params: The parameters for the Swirl-LM simulation.
    """
    self.model_params = model_params
    params = self.model_params
    assert swirl_lm_params.use_3d_tf_tensor, (
        '3D TF tensor is required to apply the temperature filter.'
    )

    self.s_b = params.s_b
    self.s_x = params.s_x
    self.h_conv = params.h_conv
    self.a_v = params.a_v
    self.cp_g = params.cp_g
    self.h_f = params.h_f
    self.t_pyr = params.t_pyr
    self.n_step = params.n_step
    self.include_radiation = params.include_radiation
    self.efficiency = params.efficiency
    self.c_f = params.c_f
    self.reaction_integration_scheme = (
        params.reaction_integration_scheme)

    self.thermodynamics_model = thermodynamics_model

    self.reaction_rate = functools.partial(
        _reaction_rate,
        s_b=self.s_b,
        s_x=self.s_x,
        c_f=self.c_f,
        t_0_ivf=params.t_0_ivf,
        t_1_ivf=params.t_1_ivf,
        periodic_dims=swirl_lm_params.periodic_dims,
        halo_width=swirl_lm_params.halo_width,
        w_axis=params.w_axis,
        w_center=params.w_center,
        apply_temperature_filter=params.apply_temperature_filter,
    )

    self.combustion_model_option = params.WhichOneof(
        'combustion_model_option'
    )
    if self.combustion_model_option == 'dry_wood':
      self.update_fn = self.dry_wood_update_fn
    elif self.combustion_model_option == 'moist_wood':
      self.update_fn = self.moist_wood_update_fn
    else:
      raise NotImplementedError(
          f'{self.combustion_model_option} is not a valid combustion model.'
          ' Available options are: `dry_wood`, `moist_wood`.'
      )

  def _src_t_g(
      self,
      t_s: tf.Tensor,
      t_g: tf.Tensor,
      theta: tf.Tensor,
      f_f: tf.Tensor,
      rho_f: tf.Tensor,
      t_far_field: Optional[tf.Tensor],
  ) -> tf.Tensor:
    """Computes the temperature source term due to reactions.

    The gas temperature source term is computed as:
      𝜔ₜₑₘₚₑᵣₐₜᵤᵣₑ = 1 / Cpg [h aᵥ (Tₛ - Tg) +
      Qrad,g + (1 - 𝚹) Ff Hf],

    Args:
      t_s: The solid temperature, in units of K.
      t_g: The gas temperature, in units of K.
      theta: The fraction of heat of reaction feed into the solid.
      f_f: The reaction rate, in units of 1/s.
      rho_f: The fuel density in a unit volume, in units of kg/m^3.
      t_far_field: The far-field temperature for the radiation sink, in units of
        K.

    Returns:
      The source term to the gas temperature in conservative form, i.e. rho T.

    Raises:
      ValueError: If radiation model is activiated in the config but
        `t_far_field` is not set.
    """
    if self.include_radiation and t_far_field is None:
      raise ValueError(
          'Radiation is included in the combustion model but `t_far_field` is'
          ' not set.'
      )

    q_rad = (
        _radiative_emission(t_g, t_far_field, self.s_b, self.efficiency)
        if self.include_radiation
        else 0.0
    )
    q_conv = _localize_by_fuel(rho_f, self.h_conv * self.a_v * (t_s - t_g))
    q_comb = _localize_by_fuel(rho_f, (1.0 - theta) * f_f * self.h_f)
    return (q_conv + q_comb - q_rad) / self.cp_g

  def _src_t_s(
      self,
      t_s: tf.Tensor,
      t_g: tf.Tensor,
      theta: tf.Tensor,
      f_f: tf.Tensor,
      rho_f: tf.Tensor,
      f_w: Optional[tf.Tensor] = None,
      rho_m: Optional[tf.Tensor] = None,
      t_far_field: Optional[tf.Tensor] = None,
  ) -> tf.Tensor:
    """Computes the source term for the fuel temperature.

    The gas temperature source term is computed as:
      (Cpf ϱf + Cpw ϱw) ∂Tₛ/∂t = Qradₛ + h aᵥ (Tg - Tₛ) - Fw (Hw +
      Cpw Tᵥₐₚ) +
          Ff(𝚹 Hf - Cpf Tpyr Nf).

    Args:
      t_s: The solid temperature, in units of K.
      t_g: The gas temperature, in units of K.
      theta: The fraction of heat of reaction feed into the solid.
      f_f: The reaction rate, in units of 1/s.
      rho_f: The fuel density in a unit volume, in units of kg/m^3.
      f_w: The evaporation rate, in units of kg/m^3/s.
      rho_m: The moisture density in a unit volume, in units of kg/m^3.
      t_far_field: The far-field temperature for the radiation sink, in units of
        K.

    Returns:
      The source term to the solid temperature.

    Raises:
      ValueError: If radiation model is activiated in the config but
        `t_far_field` is not set.
    """
    if self.include_radiation and t_far_field is None:
      raise ValueError(
          'Radiation is included in the combustion model but `t_far_field` is'
          ' not set.'
      )

    q_rad = (
        _radiative_emission(t_s, t_far_field, self.s_b, self.efficiency)
        if self.include_radiation
        else 0.0
    )
    q_conv = self.h_conv * self.a_v * (t_g - t_s)
    q_comb = f_f * (theta * self.h_f - _CP_F * self.t_pyr * _N_F)
    rhs = _localize_by_fuel(rho_f, q_conv + q_comb - q_rad)

    if (f_w is not None and
        self.model_params.WhichOneof('combustion_model_option')
        == 'moist_wood'):
      rhs -= f_w * (
          self.model_params.moist_wood.h_w +
          _CP_W * self.model_params.moist_wood.t_vap)

      cp = _CP_F * _bound_scalar(rho_f, minval=0.0)

      if rho_m is not None:
        cp += _CP_W * _bound_scalar(rho_m, minval=0.0)
    else:
      cp = _CP_F

    return tf.math.divide_no_nan(rhs, cp)

  def _get_temperature_from_states(self, states):
    """Retrieves temperature from a library of states.

    When height-dependent geopotential is used, the potential temperature
    `theta` is solved; otherwise `T` will be solved. Because we assume that
    combustion is happening close to the ground where the variation in
    hydrostatic pressure is small, the potential temperature is assumed to be
    the same as temperature.

    Args:
      states: A library of flow-field variables.

    Returns:
      The gas phase temeperature.
    """
    if 'T' in states:
      return states['T']
    elif 'theta' in states:
      return states['theta']
    else:
      raise ValueError('Temperature (`theta` or `T`) needs to be included for '
                       'fire simulations.')

  def get_temperature_source_key(self, states):
    """Generates the key for temperature source term from a library of states.

    Args:
      states: A library of flow-field variables.

    Returns:
      The key of the temperature source term.
    """
    if 'T' in states:
      return 'src_T'
    elif 'theta' in states:
      return 'src_theta'
    else:
      raise ValueError('Temperature (`theta` or `T`) needs to be included for '
                       'fire simulations.')

  def _get_far_field_temperature(
      self,
      states: FlowFieldMap,
  ) -> Optional[FlowFieldVal]:
    """Retrieves the far-field temperature for radiation."""
    if self.model_params.WhichOneof('t_far_field') == 't_ambient':
      return tf.constant(self.model_params.t_ambient)
    elif self.model_params.WhichOneof('t_far_field') == 't_variable':
      assert self.model_params.t_variable in states, (
          f'{self.model_params.t_variable} is required to compute the radiation'
          f' sink, but is missing from the states ({states.keys()}).'
      )
      return states[self.model_params.t_variable]
    else:
      return None

  def required_additional_states_keys(
      self, states: types.FlowFieldMap
  ) -> Sequence[str]:
    """Provides keys of required additional states for the combustion model."""
    required_keys = ['rho_f', 'T_s', 'src_rho', 'src_Y_O', 'tke']
    required_keys.append(self.get_temperature_source_key(states))

    if self.combustion_model_option == 'moist_wood':
      required_keys += ['rho_m', 'phi_w']

    if self.model_params.WhichOneof('t_far_field') == 't_variable':
      required_keys += [self.model_params.t_variable]

    return required_keys

  def dry_wood_update_fn(
      self,
      rho_f_init: Optional[FlowFieldVal] = None,
  ) -> StatesUpdateFn:
    """Generates an update function for states in dry wood combustion.

    In this function, the water content is assumed to be zero. The governing
    equations then becomes:
      ∂ϱf/∂t = -Nf Ff,
      Cpf ∂Tₛ/∂t = Qradₛ + h aᵥ (Tg - Tₛ) + Ff(𝚹 Hf - Cpf Tpyr Nf).
    These reactions results in source terms in the Navier-Stokes equations:
      𝜔ₘₐₛₛ = Nf Ff,
      𝜔ₜₑₘₚₑᵣₐₜᵤᵣₑ = 1 / Cpg [h aᵥ (Tₛ - Tg) + Qrad,g + (1 - 𝚹) Ff Hf],
      𝜔ₒ = -Nₒ Ff.

    Args:
      rho_f_init: The initial state of the fuel density.

    Returns:
      A function that updates the `additional_states` with the following keys:
      'rho_f', 'T_s', 'src_rho', 'src_T', 'src_Y_O'.
    """

    def additional_states_update_fn(
        kernel_op: get_kernel_fn.ApplyKernelOp,
        replica_id: tf.Tensor,
        replicas: np.ndarray,
        states: FlowFieldMap,
        additional_states: FlowFieldMap,
        params: grid_parametrization.GridParametrization,
    ) -> FlowFieldMap:
      """Updates 'rho_f', 'T_s', 'src_rho', 'src_T', and 'src_Y_O'."""
      del kernel_op

      t_far_field = self._get_far_field_temperature(additional_states)
      combustion_states = dict(states)

      def reaction_rate(rho_f, rho, y_o, tke, t_s):
        """Computes the reaction rate."""
        return self.reaction_rate(
            replica_id, replicas, rho_f, rho, y_o, tke, t_s
        )

      def rhs_t_g_fn(t_s, t_g, theta_val, f_f, rho_f, rho, t_far_field):
        """Computes the mass-specific source term for gas phase temperature."""
        return tf.math.divide_no_nan(
            self._src_t_g(t_s, t_g, theta_val, f_f, rho_f, t_far_field), rho
        )

      def rhs_solid_phase(rho_f, t_s, t_g, y_o):
        """Computes the right hand side of equations for `rho_f` and `T_s`."""
        rho_f = _bound_scalar(rho_f, minval=0.0)
        y_o = _bound_scalar(y_o, minval=0.0, maxval=1.0)

        combustion_states.update({'Y_O': y_o, 'T': t_g, 'rho': states['rho']})
        rho = self.thermodynamics_model.update_thermal_density(
            combustion_states, additional_states
        )
        f_f = reaction_rate(rho_f, rho, y_o, tke, t_s)
        theta_val = _theta(rho_f, rho_f_init)

        rhs_rho_f = tf.nest.map_structure(_src_fuel, f_f)

        if self.model_params.WhichOneof('t_far_field') == 't_variable':
          rhs_t_s = tf.nest.map_structure(
              self._src_t_s, t_s, t_g, theta_val, f_f, rho_f, t_far_field
          )
          rhs_t_g = tf.nest.map_structure(
              rhs_t_g_fn, t_s, t_g, theta_val, f_f, rho_f, rho, t_far_field
          )
        else:
          rhs_t_s = tf.nest.map_structure(
              functools.partial(self._src_t_s, t_far_field=t_far_field),
              t_s,
              t_g,
              theta_val,
              f_f,
              rho_f,
          )
          rhs_t_g = tf.nest.map_structure(
              functools.partial(rhs_t_g_fn, t_far_field=t_far_field),
              t_s,
              t_g,
              theta_val,
              f_f,
              rho_f,
              rho,
          )

        rhs_y_o = tf.nest.map_structure(
            lambda f_f_i, rho_i: _src_oxidizer(f_f_i) / rho_i, f_f, rho
        )

        return (
            rhs_rho_f,
            rhs_t_s,
            rhs_t_g,
            _localize_by_fuel(rho_f, rhs_y_o),
        )

      def substep_integration(scalars):
        """Integrates all fueld scalars by one substep."""
        return time_integration.time_advancement_explicit(
            rhs_solid_phase,
            dt,
            self.reaction_integration_scheme,
            scalars,
            scalars,
        )

      dt = params.dt / self.n_step
      i_0 = tf.constant(0)
      loop_condition = lambda i, _: i < self.n_step
      body = lambda i, scalars: (i + 1, substep_integration(scalars))

      tke = additional_states['tke']
      t_gas = self._get_temperature_from_states(states)
      scalars_0 = [
          additional_states['rho_f'], additional_states['T_s'], t_gas,
          states['Y_O']
      ]
      _, scalars_new = tf.while_loop(
          cond=loop_condition,
          body=body,
          loop_vars=(i_0, scalars_0),
          back_prop=False,
      )

      # Because the time advancements for the solid phase and gas phase are
      # performed in a staggered step, i.e. the gas phase states are 0.5 dt
      # ahead of the solid phase states, the midpoint of the solid phase time
      # integration step is consistent with the current gas phase step.
      # Therefore, a first order source term estimation is computed based on the
      # current gas phasestates and the mid point of the solid phase states.
      scalars_new[0] = _bound_scalar(scalars_new[0], minval=0.0)
      rho_f_mid = _compute_mid_state(additional_states['rho_f'], scalars_new[0])
      t_s_mid = _compute_mid_state(additional_states['T_s'], scalars_new[1])

      def f_f_mid_fn(rho_f_prev, rho_f_new):
        """Computes the rate of fuel consumption at middle of a time step."""
        return -(rho_f_new - rho_f_prev) / params.dt / _N_F

      f_f_mid = tf.nest.map_structure(
          f_f_mid_fn, additional_states['rho_f'], scalars_new[0]
      )
      f_f_mid = _localize_by_fuel(rho_f_mid, f_f_mid)

      theta_mid = _theta(rho_f_mid, rho_f_init)

      src_rho = tf.nest.map_structure(lambda f_f_i: _N_F * f_f_i, f_f_mid)
      if isinstance(t_far_field, tf.Tensor) or t_far_field is None:
        src_t = tf.nest.map_structure(
            functools.partial(self._src_t_g, t_far_field=t_far_field),
            t_s_mid,
            t_gas,
            theta_mid,
            f_f_mid,
            rho_f_mid,
        )
      else:
        src_t = tf.nest.map_structure(
            self._src_t_g,
            t_s_mid,
            t_gas,
            theta_mid,
            f_f_mid,
            rho_f_mid,
            t_far_field,
        )
      src_y_o = tf.nest.map_structure(_src_oxidizer, f_f_mid)

      src_t_key = self.get_temperature_source_key(states)
      updated_additional_states = dict(additional_states)
      for key in additional_states.keys():
        new_value = None
        if key == 'rho_f':
          new_value = scalars_new[0]
        elif key == 'T_s':
          new_value = scalars_new[1]
        elif key == 'src_rho':
          new_value = _localize_by_fuel(rho_f_mid, src_rho)
        elif key == src_t_key:
          new_value = src_t
        elif key == 'src_Y_O':
          new_value = _localize_by_fuel(rho_f_mid, src_y_o)

        if new_value is not None:
          updated_additional_states.update({key: new_value})

      return updated_additional_states

    return additional_states_update_fn

  def moist_wood_update_fn(
      self,
      rho_f_init: Optional[FlowFieldVal] = None,
      evaporation_fn: Optional[Callable[..., types.FlowFieldVal]] = None,
      add_evap_to_air_mass: bool = True,
  ) -> StatesUpdateFn:
    """Generates an update function for states in wood combustion with moisture.

    The governing equations for the solid phase are [2]:
      ∂ϱf/∂t = -Nf Ff,
      ∂ϱw/∂t = -Fw,
      (Cpf ϱf + Cpw ϱw) ∂Tₛ/∂t = Qradₛ + h aᵥ (Tg - Tₛ) - Fw (Hw +
      Cpw Tᵥₐₚ) +
          Ff(𝚹 Hf - Cpf Tpyr Nf).
    These reactions result in source terms in the Navier-Stokes equations:
      𝜔ₘₐₛₛ = Nf Ff + Fw,
      𝜔ₜₑₘₚₑᵣₐₜᵤᵣₑ = 1 / Cpg [h aᵥ (Tₛ - Tg) +
      Qrad,g + (1 - 𝚹) Ff Hf],
      𝜔ₒ = -Nₒ Ff,

    Args:
      rho_f_init: The initial state of the fuel density.
      evaporation_fn: The function that computes the rate of evaporation of the
        fuel moisture. The default evaporation model will be used if it is not
        provided.
      add_evap_to_air_mass: The option for adding the mass of the evaporated
        fuel moisture to the total mass of the air.

    Returns:
      A function that updates the `additional_states` with the following keys:
      'rho_f', 'rho_m', 'phi_w' 'T_s', 'src_rho', 'src_T', and 'src_Y_O'.
    """

    def additional_states_update_fn(
        kernel_op: get_kernel_fn.ApplyKernelOp,
        replica_id: tf.Tensor,
        replicas: np.ndarray,
        states: FlowFieldMap,
        additional_states: FlowFieldMap,
        params: grid_parametrization.GridParametrization,
    ) -> FlowFieldMap:
      """Updates wood combustion associated states."""
      del kernel_op

      t_far_field = self._get_far_field_temperature(additional_states)
      combustion_states = dict(states)

      def reaction_rate(rho_f, rho, y_o, tke, t_s):
        """Computes the reaction rate of the fuel."""
        return self.reaction_rate(
            replica_id, replicas, rho_f, rho, y_o, tke, t_s
        )

      def rhs_t_g_fn(t_s, t_g, theta_val, f_f, rho_f, rho, t_far_field):
        """Compute sthe mass-specific source term for gas phase temperature."""
        return tf.math.divide_no_nan(
            self._src_t_g(t_s, t_g, theta_val, f_f, rho_f, t_far_field), rho
        )

      def rhs_solid_phase(rho_f, rho_m, t_s, t_g, y_o, phi_w):
        """Computes the right hand side of the equations in the docstring."""
        rho_f = _bound_scalar(rho_f, minval=0.0)
        rho_m = _bound_scalar(rho_m, minval=0.0)
        y_o = _bound_scalar(y_o, minval=0.0, maxval=1.0)

        combustion_states.update({'Y_O': y_o, 'T': t_g, 'rho': states['rho']})
        rho = self.thermodynamics_model.update_thermal_density(
            combustion_states, additional_states
        )
        f_f = reaction_rate(rho_f, rho, y_o, tke, t_s)
        if evaporation_fn is None:
          evap_buf = tf.nest.map_structure(
              functools.partial(
                  _evaporation,
                  dt=params.dt,
                  c_w=self.model_params.moist_wood.c_w,
              ),
              t_s,
              phi_w,
              rho_m,
          )
          if isinstance(t_s, Sequence):
            f_w, phi_w_new = map(list, zip(*evap_buf))
          else:
            f_w, phi_w_new = evap_buf
        else:
          f_w = evaporation_fn(t_s=t_s, rho_m=rho_m)
          phi_w_new = phi_w

        theta_val = _theta(rho_f, rho_f_init)

        rhs_rho_f = tf.nest.map_structure(_src_fuel, f_f)
        rhs_rho_m = tf.nest.map_structure(tf.math.negative, f_w)
        if self.model_params.WhichOneof('t_far_field') == 't_variable':
          rhs_t_s = tf.nest.map_structure(
              self._src_t_s,
              t_s,
              t_g,
              theta_val,
              f_f,
              rho_f,
              f_w,
              rho_m,
              t_far_field,
          )
          rhs_t_g = tf.nest.map_structure(
              rhs_t_g_fn, t_s, t_g, theta_val, f_f, rho_f, rho, t_far_field
          )
        else:
          rhs_t_s = tf.nest.map_structure(
              functools.partial(self._src_t_s, t_far_field=t_far_field),
              t_s,
              t_g,
              theta_val,
              f_f,
              rho_f,
              f_w,
              rho_m,
          )
          rhs_t_g = tf.nest.map_structure(
              functools.partial(rhs_t_g_fn, t_far_field=t_far_field),
              t_s,
              t_g,
              theta_val,
              f_f,
              rho_f,
              rho,
          )
        rhs_y_o = tf.nest.map_structure(
            lambda f_f_i, rho_i: _src_oxidizer(f_f_i) / rho_i, f_f, rho
        )
        rhs_phi_w = tf.nest.map_structure(
            lambda phi_w_new_i, phi_w_i: (phi_w_new_i - phi_w_i) / params.dt,
            phi_w_new,
            phi_w,
        )

        return (
            rhs_rho_f,
            rhs_rho_m,
            rhs_t_s,
            rhs_t_g,
            _localize_by_fuel(rho_f, rhs_y_o),
            rhs_phi_w,
        )

      def substep_integration(scalars):
        """Integrates all fueld scalars by one substep."""
        return time_integration.time_advancement_explicit(
            rhs_solid_phase,
            dt,
            self.reaction_integration_scheme,
            scalars,
            scalars,
        )

      dt = params.dt / self.n_step
      i_0 = tf.constant(0)
      loop_condition = lambda i, _: i < self.n_step
      body = lambda i, scalars: (i + 1, substep_integration(scalars))

      tke = additional_states['tke']
      t_gas = self._get_temperature_from_states(states)
      phi_w = additional_states.get(
          'phi_w',
          tf.nest.map_structure(tf.zeros_like, additional_states['rho_m']),
      )
      scalars_0 = [
          additional_states['rho_f'],
          additional_states['rho_m'],
          additional_states['T_s'],
          t_gas,
          states['Y_O'],
          phi_w,
      ]
      _, scalars_new = tf.while_loop(
          cond=loop_condition,
          body=body,
          loop_vars=(i_0, scalars_0),
          back_prop=False,
      )

      # Because the time advancements for the solid phase and gas phase are
      # performed in a staggered step, i.e. the gas phase states are 0.5 dt
      # ahead of the solid phase states, the midpoint of the solid phase time
      # integration step is consistent with the current gas phase step.
      # Therefore, a first order source term estimation is computed based on the
      # current gas phasestates and the mid point of the solid phase states.
      scalars_new[0] = _bound_scalar(scalars_new[0], minval=0.0)
      scalars_new[1] = _bound_scalar(scalars_new[1], minval=0.0)

      rho_f_mid = _compute_mid_state(additional_states['rho_f'], scalars_new[0])
      t_s_mid = _compute_mid_state(additional_states['T_s'], scalars_new[2])

      def f_mid_fn(rho_prev, rho_new, coeff):
        """Computes the rate of consumption at middle of a time step."""
        return -(rho_new - rho_prev) / params.dt / coeff

      f_f_mid = tf.nest.map_structure(
          functools.partial(f_mid_fn, coeff=_N_F),
          additional_states['rho_f'],
          scalars_new[0],
      )
      f_f_mid = _localize_by_fuel(rho_f_mid, f_f_mid)
      f_w_mid = tf.nest.map_structure(
          functools.partial(f_mid_fn, coeff=1.0),
          additional_states['rho_m'],
          scalars_new[1],
      )
      f_w_mid = _localize_by_fuel(rho_f_mid, f_w_mid)

      theta_mid = _theta(rho_f_mid, rho_f_init)

      if add_evap_to_air_mass:
        src_rho = tf.nest.map_structure(
            lambda f_f_i, f_w_i: _N_F * f_f_i + f_w_i, f_f_mid, f_w_mid
        )
      else:
        src_rho = tf.nest.map_structure(lambda f_f_i: _N_F * f_f_i, f_f_mid)

      if isinstance(t_far_field, tf.Tensor) or t_far_field is None:
        src_t = tf.nest.map_structure(
            functools.partial(self._src_t_g, t_far_field=t_far_field),
            t_s_mid,
            t_gas,
            theta_mid,
            f_f_mid,
            rho_f_mid,
        )
      else:
        src_t = tf.nest.map_structure(
            self._src_t_g,
            t_s_mid,
            t_gas,
            theta_mid,
            f_f_mid,
            rho_f_mid,
            t_far_field,
        )
      src_y_o = tf.nest.map_structure(_src_oxidizer, f_f_mid)

      src_t_key = self.get_temperature_source_key(states)
      updated_additional_states = dict(additional_states)
      for key in additional_states.keys():
        new_value = None
        if key == 'rho_f':
          new_value = scalars_new[0]
        elif key == 'rho_m':
          new_value = scalars_new[1]
        elif key == 'T_s':
          new_value = scalars_new[2]
        elif key == 'src_rho':
          new_value = _localize_by_fuel(rho_f_mid, src_rho)
        elif key == src_t_key:
          new_value = src_t
        elif key == 'src_Y_O':
          new_value = _localize_by_fuel(rho_f_mid, src_y_o)
        elif key == 'phi_w':
          new_value = scalars_new[5]
        elif key == 'src_q_t':
          # The source term for the total humidity is composed of the
          # evaporation of fuel moisture and the reaction product of the wood
          # combustion. The Chemical composition of the reaction product follows
          # C6H10O5 + 6O2 = 6CO2 + 5H2O
          # Ref: Cunningham, P., & Reeder, M. J. (2009). Severe convective
          # storms initiated by intense wildfires: Numerical simulations of
          # pyro‐convection and pyro‐tornadogenesis. Geophysical Research
          # Letters, 36(12). https://doi.org/10.1029/2009gl039262
          src_p = tf.nest.map_structure(
              lambda f_f_i: (_N_F + _N_O) * f_f_i, f_f_mid
          )
          src_p = _localize_by_fuel(rho_f_mid, src_p)
          # Note that the molecular weight of CO2 and H2O are 44 and 18,
          # respectively.
          w_tot = 6 * 44.0 + 5 * 18.0
          f_w_comb = 5 * 18.0 / w_tot * src_p
          new_value = tf.nest.map_structure(tf.math.add, f_w_comb, f_w_mid)

        if new_value is not None:
          updated_additional_states.update({key: new_value})

      return updated_additional_states

    return additional_states_update_fn


def wood_combustion_factory(config: parameters_lib.SwirlLMParameters) -> Wood:
  """Constructs an object of the wood combustion model.

  Args:
    config: The configuration context of the simulation.

  Returns:
    An instance of the wood combustion library.

  Raises:
    ValueError: If `wood` is not defined in the simulation context `config`.
  """
  if config.combustion is None or not config.combustion.HasField('wood'):
    raise ValueError('Wood model is not defined as a combustion model.')

  thermodynamics_model = thermodynamics_manager.thermodynamics_factory(
      config)

  return Wood(config.combustion.wood, thermodynamics_model, config)
