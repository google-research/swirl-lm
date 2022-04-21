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
from typing import List, Optional, Sequence, Union

import numpy as np
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization  # pylint: disable=line-too-long
from swirl_lm.utility import types
import tensorflow as tf
from google3.research.simulation.tensorflow.fluid.framework.tf1 import model_function  # pylint: disable=line-too-long
from google3.research.simulation.tensorflow.fluid.framework.tf1 import step_updater
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_numerics  # pylint: disable=line-too-long

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
# The threshold below which the fuel is considered depleted.
_EPSILON = 1e-6

_TF_DTYPE = types.TF_DTYPE


def _bound_scalar(
    phi: Union[tf.Tensor, Sequence[tf.Tensor]],
    minval: float = 0.0,
    maxval: Optional[float] = None,
) -> Union[tf.Tensor, Sequence[tf.Tensor]]:
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
        'is provided while the lower bound is {}.'
        .format(maxval, minval))

  def apply_bound(phi_i: tf.Tensor) -> tf.Tensor:
    """Applies bound to a tf.Tensor."""
    phi_i = tf.maximum(phi_i, minval * tf.ones_like(phi_i, dtype=_TF_DTYPE))
    if maxval is not None:
      phi_i = tf.minimum(phi_i, maxval * tf.ones_like(phi_i, dtype=_TF_DTYPE))
    return phi_i

  return apply_bound(phi) if isinstance(
      phi, tf.Tensor) else [apply_bound(phi_i) for phi_i in phi]


def _reaction_rate(
    rho_f: tf.Tensor,
    rho_g: tf.Tensor,
    y_o: tf.Tensor,
    tke: tf.Tensor,
    temperature: tf.Tensor,
    s_b: float,
    s_x: float,
    c_f: float,
) -> tf.Tensor:
  """Computes the reation rate of the wood combustion.

  The equation for the reaction rate is:
    Ff = cF ϱf ϱo σcm 𝚿s 𝛌of / (ϱref sₓ²).

  Args:
    rho_f: The bulk density of the fuel in a unit volume, kg/m³.
    rho_g: The density of the surrounding gas, kg/m³.
    y_o: The mass fraction of the oxidizer.
    tke: The turbulent kinetic energy.
    temperature: The bulk temperature in a unit volume, K.
    s_b: The B scale of the fuel elements.
    s_x: The scale of the smallest fuel elements.
    c_f: An empirical scaling coefficient in local fire reaction rates.

  Returns:
    The reaction rate due to wood combustion.
  """

  def sigma_cm():
    """Computes the turbulent diffusivity sigma_cm = 0.09 ϱg sB √K."""
    return 0.09 * rho_g * s_b * tf.sqrt(tke)

  def psi_s():
    """Computes the ignition volume fraction."""
    return tf.math.minimum(
        tf.math.maximum((temperature - 300.0) / 400.0,
                        tf.zeros_like(temperature, dtype=_TF_DTYPE)),
        tf.ones_like(temperature, dtype=_TF_DTYPE))

  def lambda_of():
    """Computes 𝛌of = ϱf ϱo / (ϱf / Nf + ϱo / No)2."""
    return tf.math.divide_no_nan(rho_f * rho_g * y_o,
                                 (rho_f / _N_F + rho_g * y_o / _N_O)**2)

  rho_f = _bound_scalar(rho_f, minval=0.0)
  y_o = _bound_scalar(y_o, minval=0.0, maxval=1.0)

  return c_f * rho_f * rho_g * y_o * sigma_cm() * psi_s() * lambda_of() / (
      _RHO_REF * s_x**2)


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
    The radiation source term due to emission.
  """
  return _SIGMA * k / l * (t**4 - t_ambient**4)


def _evaporation(
    t: tf.Tensor,
    phi_max: tf.Tensor,
    rho_m: tf.Tensor,
    dt: float,
    c_w: float,
) -> Sequence[tf.Tensor]:
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
    rho_f: Sequence[tf.Tensor],
    rho_f_init: Optional[Sequence[tf.Tensor]] = None,
) -> List[tf.Tensor]:
  """Computes the fraction of heat feedback to solid after combustion.

  Args:
    rho_f: The current fuel density in a unit volume, in units of kg/m^3.
    rho_f_init: The initial fuel density in a unit volume, in units of kg/m^3.

  Returns:
    The fraction of heat feedback to the solid after combustion.
  """
  if not rho_f_init:
    # Assume reaction heat transfer to gas and solid with equal probability
    # if the initial fuel state is missing.
    return [0.5 * tf.ones_like(rho_f_i, dtype=_TF_DTYPE) for rho_f_i in rho_f]

  rho_f_0 = rho_f_init
  return [
      1.0 - tf.math.divide_no_nan(rho_f_i, rho_f_0_i)
      for rho_f_i, rho_f_0_i in zip(rho_f, rho_f_0)
  ]


def _localize_by_fuel(
    rho_f: Sequence[tf.Tensor],
    src: Sequence[tf.Tensor],
) -> List[tf.Tensor]:
  """Sets source term to zero where there is no fuel.

  Args:
    rho_f: The current fuel density in a unit volume, in units of kg/m^3.
    src: The source term of a scalar solved in the simulation.

  Returns:
    The source term of a scalar, which only exists where there fuel is non-zero.
  """
  return [
      tf.compat.v1.where(
          rho_f_i < _EPSILON, tf.zeros_like(src_i, dtype=_TF_DTYPE),
          src_i) for rho_f_i, src_i in zip(rho_f, src)
  ]


def _compute_mid_state(
    state_old: Sequence[tf.Tensor],
    state_new: Sequence[tf.Tensor],
) -> Sequence[tf.Tensor]:
  """Computes the states at the middle step with linear interpolation.

  Args:
    state_old: The state at the beginning of the time step.
    state_new: The state at the end of the time step.

  Returns:
    The state at the middle of the time step.
  """
  return [
      0.5 * (state_new_i + state_old_i)
      for state_new_i, state_old_i in zip(state_new, state_old)
  ]


class Wood(object):
  """A library of wood combustion."""

  def __init__(
      self,
      config: incompressible_structured_mesh_config
      .IncompressibleNavierStokesParameters,
  ):
    """Initializes the wood combustion library.

    Args:
      config: The context that provides parameterized information for the
        simulation.
    """
    self.model_params = config.combustion.wood

    self.s_b = self.model_params.s_b
    self.s_x = self.model_params.s_x
    self.h_conv = self.model_params.h_conv
    self.a_v = self.model_params.a_v
    self.cp_g = self.model_params.cp_g
    self.h_f = self.model_params.h_f
    self.t_pyr = self.model_params.t_pyr
    self.n_step = self.model_params.n_step
    self.include_radiation = self.model_params.include_radiation
    self.t_ambient = self.model_params.t_ambient
    self.efficiency = self.model_params.efficiency
    self.c_f = self.model_params.c_f
    self.reaction_integration_scheme = (
        self.model_params.reaction_integration_scheme)

    self.thermodynamics_model = thermodynamics_manager.thermodynamics_factory(
        config)

    self.reaction_rate = functools.partial(
        _reaction_rate, s_b=self.s_b, s_x=self.s_x, c_f=self.c_f)

    if self.model_params.WhichOneof('combustion_model_option') == 'dry_wood':
      self.combustion_model_option = self.model_params.dry_wood
      self.update_fn = self.dry_wood_update_fn
    elif self.model_params.WhichOneof(
        'combustion_model_option') == 'moist_wood':
      self.combustion_model_option = self.model_params.moist_wood
      self.update_fn = self.moist_wood_update_fn
    else:
      self.combustion_model_option = None
      self.update_fn = None

  def _src_t_g(
      self,
      t_s: tf.Tensor,
      t_g: tf.Tensor,
      theta: tf.Tensor,
      f_f: tf.Tensor,
      rho_f: tf.Tensor,
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

    Returns:
      The source term to the gas temperature in conservative form, i.e. rho T.
    """
    q_rad = _radiative_emission(
        t_g, self.t_ambient, self.s_b,
        self.efficiency) if self.include_radiation else 0.0
    src = tf.compat.v1.where(
        rho_f < _EPSILON, tf.zeros_like(t_g, dtype=t_g.dtype),
        1.0 / self.cp_g * (self.h_conv * self.a_v * (t_s - t_g) +
                           (1.0 - theta) * f_f * self.h_f))
    return src - q_rad / self.cp_g

  def _src_t_s(
      self,
      t_s: tf.Tensor,
      t_g: tf.Tensor,
      theta: tf.Tensor,
      f_f: tf.Tensor,
      rho_f: Optional[tf.Tensor] = None,
      f_w: Optional[tf.Tensor] = None,
      rho_m: Optional[tf.Tensor] = None,
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

    Returns:
      The source term to the solid temperature.
    """
    q_rad = _radiative_emission(
        t_s, self.t_ambient, self.s_b,
        self.efficiency) if self.include_radiation else 0.0
    rhs = self.h_conv * self.a_v * (t_g - t_s) - q_rad + f_f * (
        theta * self.h_f - _CP_F * self.t_pyr * _N_F)

    if (f_w is not None and
        self.model_params.WhichOneof('combustion_model_option')
        == 'moist_wood'):
      rhs -= f_w * (
          self.combustion_model_option.h_w +
          _CP_W * self.combustion_model_option.t_vap)

    cp = _CP_F

    if rho_f is not None:
      cp = _CP_F * _bound_scalar(rho_f, minval=0.0)

    if rho_m is not None:
      cp += _CP_W * _bound_scalar(rho_m, minval=0.0)

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

  def _get_temperature_source_key(self, states):
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

  def dry_wood_update_fn(
      self,
      rho_f_init: Optional[Sequence[tf.Tensor]] = None,
  ) -> step_updater.StatesUpdateFn:
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
        states: model_function.StatesMap,
        additional_states: model_function.StatesMap,
        params: grid_parametrization.GridParametrization,
    ) -> model_function.StatesMap:
      """Updates 'rho_f', 'T_s', 'src_rho', 'src_T', and 'src_Y_O'."""

      def rhs_solid_phase(rho_f, t_s, t_g, y_o):
        """Computes the right hand side of the equation for `rho_f` and `T_s`."""
        rho_f = _bound_scalar(rho_f, minval=0.0)
        y_o = _bound_scalar(y_o, minval=0.0, maxval=1.0)

        rho, _ = self.thermodynamics_model.update_density(
            kernel_op, replica_id, replicas, {
                'Y_O': y_o,
                'T': t_g,
                'rho': states['rho'],
            }, additional_states)
        f_f = [
            self.reaction_rate(rho_f_i, rho_i, y_o_i, tke_i, t_s_i)
            for rho_f_i, rho_i, y_o_i, tke_i, t_s_i in zip(
                rho_f, rho, y_o, tke, t_s)
        ]
        theta_val = _theta(rho_f, rho_f_init)

        rhs_rho_f = [_src_fuel(f_f_i) for f_f_i in f_f]
        rhs_t_s = [
            self._src_t_s(t_s_i, t_g_i, theta_val_i, f_f_i)
            for t_s_i, t_g_i, theta_val_i, f_f_i, in zip(
                t_s, t_g, theta_val, f_f)
        ]
        # pylint: disable=g-complex-comprehension
        rhs_t_g = [
            tf.math.divide_no_nan(
                self._src_t_g(t_s_i, t_g_i, theta_val_i, f_f_i, rho_f_i), rho_i)
            for t_s_i, t_g_i, theta_val_i, f_f_i, rho_f_i, rho_i in zip(
                t_s, t_g, theta_val, f_f, rho_f, rho)
        ]
        # pylint: enable=g-complex-comprehension
        rhs_y_o = [
            _src_oxidizer(f_f_i) / rho_i for f_f_i, rho_i in zip(f_f, rho)
        ]

        return (
            rhs_rho_f,
            _localize_by_fuel(rho_f, rhs_t_s),
            rhs_t_g,
            _localize_by_fuel(rho_f, rhs_y_o),
        )

      def substep_integration(scalars):
        """Integrates all fueld scalars by one substep."""
        return (
            incompressible_structured_mesh_numerics.time_advancement_explicit(
                rhs_solid_phase,
                dt,
                self.reaction_integration_scheme,
                scalars,
                scalars,
            ))

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
      f_f_mid = [
          -(rho_f_new - rho_f_prev) / params.dt / _N_F for rho_f_prev, rho_f_new
          in zip(additional_states['rho_f'], scalars_new[0])
      ]
      f_f_mid = _localize_by_fuel(rho_f_mid, f_f_mid)

      theta_mid = _theta(rho_f_mid, rho_f_init)

      src_rho = [_N_F * f_f_i for f_f_i in f_f_mid]
      src_t = [
          self._src_t_g(t_s_i, t_g_i, theta_i, f_f_i, rho_f_i)
          for t_s_i, t_g_i, theta_i, f_f_i, rho_f_i in zip(
              t_s_mid, t_gas, theta_mid, f_f_mid, rho_f_mid)
      ]
      src_y_o = [_src_oxidizer(f_f_i) for f_f_i in f_f_mid]

      src_t_key = self._get_temperature_source_key(states)
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
      rho_f_init: Optional[Sequence[tf.Tensor]] = None,
  ) -> step_updater.StatesUpdateFn:
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

    Returns:
      A function that updates the `additional_states` with the following keys:
      'rho_f', 'rho_m', 'phi_w' 'T_s', 'src_rho', 'src_T', and 'src_Y_O'.
    """

    def additional_states_update_fn(
        kernel_op: get_kernel_fn.ApplyKernelOp,
        replica_id: tf.Tensor,
        replicas: np.ndarray,
        states: model_function.StatesMap,
        additional_states: model_function.StatesMap,
        params: grid_parametrization.GridParametrization,
    ) -> model_function.StatesMap:
      """Updates wood combustion associated states."""

      evaporation = functools.partial(
          _evaporation, dt=params.dt, c_w=self.combustion_model_option.c_w)

      def rhs_solid_phase(rho_f, rho_m, t_s, t_g, y_o, phi_w):
        """Computes the right hand side of the equations in the docstring."""
        rho_f = _bound_scalar(rho_f, minval=0.0)
        rho_m = _bound_scalar(rho_m, minval=0.0)
        y_o = _bound_scalar(y_o, minval=0.0, maxval=1.0)

        rho, _ = self.thermodynamics_model.update_density(
            kernel_op, replica_id, replicas, {
                'Y_O': y_o,
                'T': t_g,
                'rho': states['rho'],
            }, additional_states)
        f_f = [
            self.reaction_rate(rho_f_i, rho_i, y_o_i, tke_i, t_s_i)
            for rho_f_i, rho_i, y_o_i, tke_i, t_s_i in zip(
                rho_f, rho, y_o, tke, t_s)
        ]
        f_w = []
        phi_w_new = []
        for t_s_i, phi_w_i, rho_m_i in zip(t_s, phi_w, rho_m):
          f_w_i, phi_w_new_i = evaporation(t_s_i, phi_w_i, rho_m_i)
          f_w.append(f_w_i)
          phi_w_new.append(phi_w_new_i)

        theta_val = _theta(rho_f, rho_f_init)

        rhs_rho_f = [_src_fuel(f_f_i) for f_f_i in f_f]
        rhs_rho_m = [-f_w_i for f_w_i in f_w]
        rhs_t_s = [
            self._src_t_s(t_s_i, t_g_i, theta_i, f_f_i, rho_f_i, f_w_i, rho_m_i)
            for t_s_i, t_g_i, theta_i, f_f_i, rho_f_i, f_w_i, rho_m_i in zip(
                t_s, t_g, theta_val, f_f, rho_f, f_w, rho_m)
        ]
        rhs_t_g = [
            self._src_t_g(t_s_i, t_g_i, theta_val_i, f_f_i, rho_f_i) / rho_i
            for t_s_i, t_g_i, theta_val_i, f_f_i, rho_f_i, rho_i in zip(
                t_s, t_g, theta_val, f_f, rho_f, rho)
        ]
        rhs_y_o = [
            _src_oxidizer(f_f_i) / rho_i for f_f_i, rho_i in zip(f_f, rho)
        ]
        rhs_phi_w = [(phi_w_new_i - phi_w_i) / params.dt
                     for phi_w_new_i, phi_w_i in zip(phi_w_new, phi_w)]

        return (
            rhs_rho_f,
            rhs_rho_m,
            _localize_by_fuel(rho_f, rhs_t_s),
            rhs_t_g,
            _localize_by_fuel(rho_f, rhs_y_o),
            rhs_phi_w,
        )

      def substep_integration(scalars):
        """Integrates all fueld scalars by one substep."""
        return (
            incompressible_structured_mesh_numerics.time_advancement_explicit(
                rhs_solid_phase,
                dt,
                self.reaction_integration_scheme,
                scalars,
                scalars,
            ))

      dt = params.dt / self.n_step
      i_0 = tf.constant(0)
      loop_condition = lambda i, _: i < self.n_step
      body = lambda i, scalars: (i + 1, substep_integration(scalars))

      tke = additional_states['tke']
      t_gas = self._get_temperature_from_states(states)
      scalars_0 = [
          additional_states['rho_f'], additional_states['rho_m'],
          additional_states['T_s'], t_gas, states['Y_O'],
          additional_states['phi_w']
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
      f_f_mid = [
          -(rho_f_new - rho_f_prev) / params.dt / _N_F for rho_f_prev, rho_f_new
          in zip(additional_states['rho_f'], scalars_new[0])
      ]
      f_f_mid = _localize_by_fuel(rho_f_mid, f_f_mid)
      f_w_mid = [
          -(rho_m_new - rho_m_prev) / params.dt for rho_m_prev, rho_m_new in
          zip(additional_states['rho_m'], scalars_new[1])
      ]
      f_w_mid = _localize_by_fuel(rho_f_mid, f_w_mid)

      theta_mid = _theta(rho_f_mid, rho_f_init)

      src_rho = [_N_F * f_f_i + f_w_i for f_f_i, f_w_i in zip(f_f_mid, f_w_mid)]
      src_t = [
          self._src_t_g(t_s, t_g, theta_i, f_f_i,
                        rho_f_i) for t_s, t_g, theta_i, f_f_i, rho_f_i in zip(
                            t_s_mid, t_gas, theta_mid, f_f_mid, rho_f_mid)
      ]
      src_y_o = [_src_oxidizer(f_f_i) for f_f_i in f_f_mid]

      src_t_key = self._get_temperature_source_key(states)
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

        if new_value is not None:
          updated_additional_states.update({key: new_value})

      return updated_additional_states

    return additional_states_update_fn


def wood_combustion_factory(
    config: incompressible_structured_mesh_config
    .IncompressibleNavierStokesParameters
) -> Wood:
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

  return Wood(config)
