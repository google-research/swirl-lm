# coding=utf-8
"""A library for computing reaction source terms with a one-step mechanism.

The one-step chemistry model for gaseous phase reaction is represented as:
  F + O -> P,
where:
  F is the fuel,
  O is the oxidizer, and
  P is the reaction product.
The reaction source term is a function of the mass fractions of the fuel and
oxidizer, as well as the temperature, which takes the form of the Arrhenius law:
  ω(F, O, T) = A[F]ᵃ[O]ᵇexp(-Eₐ/R/T),
where:
  A is a scaling constant,
  [F] = ϱ Y_F / W_F is the volume concentration of the fuel,
  [O] = ϱ Y_O / W_O is the volume concentration of the oxidizer,
  Eₐ is the activation energy,
  R is the universal gas constant, and
  T is the temperature.
The source term for Y_F, Y_O, and T are then computed as:
  ω_F = -𝛎_F W_F ω(F, O, T) / ϱ,
  ω_O = -𝛎_O W_O ω(F, O, T) / ϱ,
  ω_T = Q / Cₚ ω(F, O, T) / ϱ, where Q is the heat of combustion.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from typing import List, Sequence

import numpy as np
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization  # pylint: disable=line-too-long
import tensorflow as tf
from google3.research.simulation.tensorflow.fluid.framework.tf1 import model_function  # pylint: disable=line-too-long
from google3.research.simulation.tensorflow.fluid.framework.tf1 import step_updater
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_numerics  # pylint: disable=line-too-long
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_updates  # pylint: disable=line-too-long

# The universal gas constant that is used to compute the density in the onestep
# chemistry model, in units of J/mol/K.
R_UNIVERSAL = 8.3145
# The lower bound of temperature considered in the onestep chemistry model, in
# units of K.
T_MIN = 273.0
# The upper bound of temperature considered in the onestep chemistry model, in
# unist of K.
T_MAX = 2500.0


def _arrhenius_law(
    c_f: tf.Tensor,
    c_o: tf.Tensor,
    temperature: tf.Tensor,
    a_cst: float,
    coeff_f: float,
    coeff_o: float,
    e_a: float,
) -> tf.Tensor:
  """Computes the Arrhenius law."""
  return a_cst * tf.math.pow(c_f, coeff_f) * tf.math.pow(
      c_o, coeff_o) * tf.math.exp(-e_a / R_UNIVERSAL / temperature)


def _concentration(
    y_species: tf.Tensor,
    w_species: float,
    rho: tf.Tensor,
) -> tf.Tensor:
  """Computes the volume concentration of species."""
  return rho * y_species / w_species


def one_step_reaction_source(
    y_f: Sequence[tf.Tensor],
    y_o: Sequence[tf.Tensor],
    temperature: Sequence[tf.Tensor],
    rho: Sequence[tf.Tensor],
    a_cst: float,
    coeff_f: float,
    coeff_o: float,
    e_a: float,
    q: float,
    cp: float,
    w_f: float,
    w_o: float,
    nu_f: float = 1.0,
    nu_o: float = 1.0,
) -> List[List[tf.Tensor]]:
  """Computes the reaction source term using onestep chemistry.

  Args:
    y_f: The massfraction of fuel.
    y_o: The massfraction of oxidizer.
    temperature: The temperature, in units of K.
    rho: The density of the flow field, in units of kg/m^3.
    a_cst: The constant A in the Arrhenius law.
    coeff_f: The power law coefficient of the fuel volume concentration.
    coeff_o: The power law coefficient of the oxidizer volume concentration.
    e_a: The activation energy.
    q: The heat of combustion.
    cp: The specific heat.
    w_f: The molecular weight of the fuel.
    w_o: The molecular weight of the oxidizer.
    nu_f: The stoichiometric coefficient of the fuel.
    nu_o: The stoichiometric coefficient of the oxidizer.

  Returns:
    The rate of change of the `y_f`, `y_o`, and `temperature` due to the onestep
    chemical reaction.
  """

  def bound_scalar(value: tf.Tensor, minval: float, maxval: float):
    """Enforces bounds for `value` so that `minval` <= `value` <= `maxval`."""
    return tf.minimum(
        tf.maximum(value, minval * tf.ones_like(value)),
        maxval * tf.ones_like(value))

  c_f = [
      _concentration(bound_scalar(y_f_i, 0.0, 1.0), w_f, rho_i)
      for y_f_i, rho_i in zip(y_f, rho)
  ]
  c_o = [
      _concentration(bound_scalar(y_o_i, 0.0, 1.0), w_o, rho_i)
      for y_o_i, rho_i in zip(y_o, rho)
  ]

  omega = [
      _arrhenius_law(c_f_i, c_o_i, bound_scalar(t_i, T_MIN, T_MAX), a_cst,
                     coeff_f, coeff_o, e_a)
      for c_f_i, c_o_i, t_i in zip(c_f, c_o, temperature)
  ]

  return [[-nu_f * w_f * omega_i / rho for omega_i, rho in zip(omega, rho)],
          [-nu_o * w_o * omega_i / rho for omega_i, rho in zip(omega, rho)],
          [q * omega_i / cp / rho for omega_i, rho in zip(omega, rho)]]


def one_step_reaction_integration(
    y_f: Sequence[tf.Tensor],
    y_o: Sequence[tf.Tensor],
    temperature: Sequence[tf.Tensor],
    delta_t: float,
    a_cst: float,
    coeff_f: float,
    coeff_o: float,
    e_a: float,
    q: float,
    cp: float,
    w_f: float,
    w_o: float,
    w_p: float,
    nu_f: float = 1.0,
    nu_o: float = 1.0,
    p_thermal: float = 1.0e5,
    nt: int = 100,
) -> List[List[tf.Tensor]]:
  """Integrates `y_f`, `y_o`, and `temperature` by `delta_t`.

  Args:
    y_f: The massfraction of fuel.
    y_o: The massfraction of oxidizer.
    temperature: The temperature, in units of K.
    delta_t: The interval of time for the integration.
    a_cst: The constant A in the Arrhenius law.
    coeff_f: The power law coefficient of the fuel volume concentration.
    coeff_o: The power law coefficient of the oxidizer volume concentration.
    e_a: The activation energy.
    q: The heat of combustion.
    cp: The specific heat.
    w_f: The molecular weight of the fuel.
    w_o: The molecular weight of the oxidizer.
    w_p: The molecular weight of the reaction product.
    nu_f: The stoichiometric coefficient of the fuel.
    nu_o: The stoichiometric coefficient of the oxidizer.
    p_thermal: The thermal dynamic pressure, in units of Pa.
    nt: The number of sub-iterations for the integration.

  Returns:
    The new states of the `y_f`, `y_o`, and `temperature` due to the onestep
    chemical reaction integrated over `delta_t`.
  """

  def substep_integration(states):
    """Integrates all variables by one substep."""
    rho = incompressible_structured_mesh_updates.density_update(
        states, scalars_info)
    rhs = functools.partial(
        one_step_reaction_source,
        rho=rho,
        a_cst=a_cst,
        coeff_f=coeff_f,
        coeff_o=coeff_o,
        e_a=e_a,
        q=q,
        cp=cp,
        w_f=w_f,
        w_o=w_o,
        nu_f=nu_f,
        nu_o=nu_o,
    )
    scalars = [states['Y_F'], states['Y_O'], states['T']]
    scalars_new = (
        incompressible_structured_mesh_numerics.time_advancement_explicit(
            rhs,
            dt,
            incompressible_structured_mesh_numerics.TimeIntegrationScheme
            .TIME_SCHEME_RK3,
            scalars,
            scalars,
        ))
    return {'Y_F': scalars_new[0], 'Y_O': scalars_new[1], 'T': scalars_new[2]}

  dt = delta_t / nt

  molecular_weights = {'Y_F': w_f, 'Y_O': w_o, 'ambient': w_p}
  scalars_info = (
      incompressible_structured_mesh_updates.ThermodynamicScalarsInfo(
          p_thermal, molecular_weights))

  states0 = {'Y_F': y_f, 'Y_O': y_o, 'T': temperature}
  i0 = tf.constant(0)
  stop_condition = lambda i, _: i < nt
  body = lambda i, states: (i + 1, substep_integration(states))

  _, states_new = tf.while_loop(
      cond=stop_condition,
      body=body,
      loop_vars=(i0, states0),
      back_prop=False,
  )

  return [states_new['Y_F'], states_new['Y_O'], states_new['T']]


def integrated_reaction_source_update_fn(
    a_cst: float,
    coeff_f: float,
    coeff_o: float,
    e_a: float,
    q: float,
    cp: float,
    w_f: float,
    w_o: float,
    w_p: float,
    nu_f: float = 1.0,
    nu_o: float = 1.0,
    p_thermal: float = 1.0e5,
    nt: int = 100,
) -> step_updater.StatesUpdateFn:
  """Generates an update function of reaction source  integrated change.

  Args:
    a_cst: The constant A in the Arrhenius law.
    coeff_f: The power law coefficient of the fuel volume concentration.
    coeff_o: The power law coefficient of the oxidizer volume concentration.
    e_a: The activation energy.
    q: The heat of combustion.
    cp: The specific heat.
    w_f: The molecular weight of the fuel.
    w_o: The molecular weight of the oxidizer.
    w_p: The molecular weight of the reaction product.
    nu_f: The stoichiometric coefficient of the fuel.
    nu_o: The stoichiometric coefficient of the oxidizer.
    p_thermal: The thermal dynamic pressure, in units of Pa.
    nt: The number of sub-iterations for the integration.

  Returns:
    A function that updates the additional_states `src_Y_F`, `src_Y_O`, `src_T`.
  """

  def additional_states_update_fn(
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: model_function.StatesMap,
      additional_states: model_function.StatesMap,
      params: grid_parametrization.GridParametrization,
  ) -> model_function.StatesMap:
    """Computes the reaction source term for Y_F, Y_O, and T."""
    del kernel_op, replica_id, replicas

    updated_states = one_step_reaction_integration(states['Y_F'], states['Y_O'],
                                                   states['T'], params.dt,
                                                   a_cst, coeff_f, coeff_o, e_a,
                                                   q, cp, w_f, w_o, w_p, nu_f,
                                                   nu_o, p_thermal, nt)

    updated_additional_states = {}
    for varname, value in additional_states.items():
      if varname == 'src_Y_F':
        updated_additional_states.update({
            varname: [
                (y_f - y_f_old) / params.dt
                for y_f, y_f_old in zip(updated_states[0], states['Y_F'])
            ]
        })
      elif varname == 'src_Y_O':
        updated_additional_states.update({
            varname: [
                (y_o - y_o_old) / params.dt
                for y_o, y_o_old in zip(updated_states[1], states['Y_O'])
            ]
        })
      elif varname == 'src_T':
        updated_additional_states.update({
            varname: [
                (temp - temp_old) / params.dt
                for temp, temp_old in zip(updated_states[2], states['T'])
            ]
        })
      else:
        updated_additional_states.update({varname: value})

    return updated_additional_states

  return additional_states_update_fn


def reaction_source_update_fn(
    a_cst: float,
    coeff_f: float,
    coeff_o: float,
    e_a: float,
    q: float,
    cp: float,
    w_f: float,
    w_o: float,
    nu_f: float = 1.0,
    nu_o: float = 1.0,
) -> step_updater.StatesUpdateFn:
  """Generates an update function of reaction source terms and heat release.

  Args:
    a_cst: The constant A in the Arrhenius law.
    coeff_f: The power law coefficient of the fuel volume concentration.
    coeff_o: The power law coefficient of the oxidizer volume concentration.
    e_a: The activation energy.
    q: The heat of combustion.
    cp: The specific heat.
    w_f: The molecular weight of the fuel.
    w_o: The molecular weight of the oxidizer.
    nu_f: The stoichiometric coefficient of the fuel.
    nu_o: The stoichiometric coefficient of the oxidizer.

  Returns:
    A function that updates the additional_states `src_Y_F`, `src_Y_O`, `src_T`.
  """

  def additional_states_update_fn(
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: model_function.StatesMap,
      additional_states: model_function.StatesMap,
      params: grid_parametrization.GridParametrization,
  ) -> model_function.StatesMap:
    """Computes the reaction source term for Y_F, Y_O, and T."""
    del kernel_op, replica_id, replicas, params

    source_terms = one_step_reaction_source(states['Y_F'], states['Y_O'],
                                            states['T'], states['rho'], a_cst,
                                            coeff_f, coeff_o, e_a, q, cp, w_f,
                                            w_o, nu_f, nu_o)

    updated_additional_states = {}
    for varname, value in additional_states.items():
      if varname == 'src_Y_F':
        updated_additional_states.update({varname: source_terms[0]})
      elif varname == 'src_Y_O':
        updated_additional_states.update({varname: source_terms[1]})
      elif varname == 'src_T':
        updated_additional_states.update({varname: source_terms[2]})
      else:
        updated_additional_states.update({varname: value})

    return updated_additional_states

  return additional_states_update_fn
