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

import functools
from typing import List

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.numerics import time_integration
from swirl_lm.physics.thermodynamics import ideal_gas
from swirl_lm.utility import composite_types
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf


FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap
StatesUpdateFn = composite_types.StatesUpdateFn
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
    y_f: FlowFieldVal,
    y_o: FlowFieldVal,
    temperature: FlowFieldVal,
    rho: FlowFieldVal,
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
) -> List[FlowFieldVal]:
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

  c_f = _concentration(bound_scalar(y_f, 0.0, 1.0), w_f, rho)
  c_o = _concentration(bound_scalar(y_o, 0.0, 1.0), w_o, rho)

  omega = _arrhenius_law(
      c_f,
      c_o,
      bound_scalar(temperature, T_MIN, T_MAX),
      a_cst,
      coeff_f,
      coeff_o,
      e_a,
  )

  return [
      -nu_f * w_f * omega / rho,
      -nu_o * w_o * omega / rho,
      q * omega / cp / rho,
  ]


def one_step_reaction_integration(
    params: parameters_lib.SwirlLMParameters,
    y_f: FlowFieldVal,
    y_o: FlowFieldVal,
    temperature: FlowFieldVal,
    delta_t: float,
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
    nt: int = 100,
) -> List[FlowFieldVal]:
  """Integrates `y_f`, `y_o`, and `temperature` by `delta_t`.

  Args:
    params: A context object for the simulation.
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
    nu_f: The stoichiometric coefficient of the fuel.
    nu_o: The stoichiometric coefficient of the oxidizer.
    nt: The number of sub-iterations for the integration.

  Returns:
    The new states of the `y_f`, `y_o`, and `temperature` due to the onestep
    chemical reaction integrated over `delta_t`.

  Raises:
    ValueError: If the thermodynamics model in `params` is not `ideal_gas_law`.
  """

  def substep_integration(states):
    """Integrates all variables by one substep."""
    rho = thermodynamics.update_density(states, {})
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
        time_integration.time_advancement_explicit(
            rhs,
            dt,
            time_integration.TimeIntegrationScheme.TIME_SCHEME_RK3,
            scalars,
            scalars,
        ))
    return {
        'Y_F': scalars_new[0],
        'Y_O': scalars_new[1],
        'T': scalars_new[2],
        'rho': rho
    }

  dt = delta_t / nt

  assert (
      thermodynamics_ := params.thermodynamics
  ) is not None, 'Thermodynamics must be set in the config.'
  if thermodynamics_.WhichOneof('thermodynamics_type') != 'ideal_gas_law':
    raise ValueError(
        'The thermodynamics model has to be `ideal_gas_law` to use the one-step'
        ' chemistry model.'
    )
  thermodynamics = ideal_gas.IdealGas(params)

  states0 = {
      'Y_F': tf.nest.map_structure(tf.identity, y_f),
      'Y_O': tf.nest.map_structure(tf.identity, y_o),
      'T': tf.nest.map_structure(tf.identity, temperature),
      'rho': tf.nest.map_structure(tf.ones_like, y_f)
  }
  i0 = tf.constant(0)
  stop_condition = lambda i, _: i < nt
  body = lambda i, states: (i + 1, substep_integration(states))

  _, states_new = tf.nest.map_structure(
      tf.stop_gradient,
      tf.while_loop(
          cond=stop_condition,
          body=body,
          loop_vars=(i0, states0),
      ))

  return [
      states_new['Y_F'], states_new['Y_O'], states_new['T'], states_new['rho']
  ]


def integrated_reaction_source_update_fn(
    params: parameters_lib.SwirlLMParameters,
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
    nt: int = 100,
) -> StatesUpdateFn:
  """Generates an update function of reaction source  integrated change.

  Args:
    params: A context object for the simulation.
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
    nt: The number of sub-iterations for the integration.

  Returns:
    A function that updates the additional_states `src_Y_F`, `src_Y_O`, `src_T`.
  """

  def additional_states_update_fn(
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      grid_params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Computes the reaction source term for Y_F, Y_O, and T."""
    del kernel_op, replica_id, replicas

    dt = grid_params.dt
    updated_states = one_step_reaction_integration(params, states['Y_F'],
                                                   states['Y_O'], states['T'],
                                                   dt, a_cst, coeff_f,
                                                   coeff_o, e_a, q, cp, w_f,
                                                   w_o, nu_f, nu_o, nt)

    updated_additional_states = {}
    for varname, value in additional_states.items():
      if varname == 'src_Y_F':
        updated_additional_states.update({
            varname: (updated_states[0] - states['Y_F']) / dt,
        })
      elif varname == 'src_Y_O':
        updated_additional_states.update({
            varname: (updated_states[1] - states['Y_O']) / dt,
        })
      elif varname == 'src_T':
        updated_additional_states.update({
            varname: (updated_states[2] - states['T']) / dt,
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
) -> StatesUpdateFn:
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
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
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
