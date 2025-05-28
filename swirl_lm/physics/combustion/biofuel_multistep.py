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

"""Computes the source terms from a multistep biofuel combustion model."""

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics.combustion import biofuel_multistep_pb2
from swirl_lm.physics.combustion import wood
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf


class Dehydration:
  """Computes source terms of fuel moisture dehydration."""

  def __init__(self, params: biofuel_multistep_pb2.Dehydration):
    """Initializes the dehydration model."""
    self.arrhenius_factor = params.arrhenius_factor
    self.activation_temperature = params.activation_temperature
    self.min_dehydration_temperature = params.min_dehydration_temperature

  def dehydration_rate(self, **kwargs) -> types.FlowFieldVal:
    """Computes the rate of dehydration following an Arrhenius law.

    Args:
      **kwargs: 3D scalar variables that are required to compute the evaporation
        rate. Must contain the following arguments:
          t_s -> The temperature of the solid phase, in units of K.
          rho_m -> The bulk density of the moisture, in units of kg / m^3.

    Returns:
      The rate of dehydration in units of kg / m^3 / s.
    """
    t_s = kwargs['t_s']
    rho_m = kwargs['rho_m']
    rate_fn = lambda rho, t: tf.where(
        tf.greater(t, self.min_dehydration_temperature),
        self.arrhenius_factor
        / tf.math.sqrt(t)
        * rho
        * tf.math.exp(-self.activation_temperature / t),
        tf.zeros_like(rho),
    )

    return tf.nest.map_structure(rate_fn, rho_m, t_s)


class PyrolysisAndCharOxidation:
  """Computes source terms due to pyrolysis and char oxydation.

  Currently we are reusing the FireTec model for the pyrolysis and char
  oxidation process.
  """

  def __init__(
      self,
      swirl_lm_params: parameters_lib.SwirlLMParameters,
  ):
    """Initializes the pyrolysis and char oxidation model."""
    if (
        swirl_lm_params.combustion is None
        or not swirl_lm_params.combustion.HasField('biofuel_multistep')
    ):
      raise ValueError(
          'Multistep biofuel model is not defined as a combustion model.'
      )

    model_params = (
        swirl_lm_params.combustion.biofuel_multistep.pyrolysis_char_oxidation
    )
    thermodynamics_model = thermodynamics_manager.thermodynamics_factory(
        swirl_lm_params
    )

    self.combustion_model = wood.Wood(
        model_params.wood, thermodynamics_model, swirl_lm_params
    )

    assert self.combustion_model.combustion_model_option == 'moist_wood', (
        'The combustion model has to be configured with the moist option'
        ' (`moist_wood`) with the multistep biofuel model.'
    )


class BiofuelMultistep:
  """Advances the solid states and computes source terms due to combustion."""

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    """Initializes the multistep biofuel combustion model."""
    if params.combustion is None or not params.combustion.HasField(
        'biofuel_multistep'
    ):
      raise ValueError('The multistep biofuel model is not configured.')

    model_params = params.combustion.biofuel_multistep
    self.dehydration = Dehydration(model_params.dehydration)
    self.pyrolysis_char_oxidation = PyrolysisAndCharOxidation(params)

  def required_additional_states_keys(
      self, states: types.FlowFieldMap
  ) -> list[str]:
    """Provides keys of required additional states for the combustion model."""
    required_keys = [
        'rho_f',
        'rho_m',
        'T_s',
        'src_rho',
        'src_Y_O',
        'src_q_t',
        'tke',
    ]
    combustion_model = self.pyrolysis_char_oxidation.combustion_model
    required_keys.append(combustion_model.get_temperature_source_key(states))

    if combustion_model.model_params.WhichOneof('t_far_field') == 't_variable':
      required_keys += [combustion_model.model_params.t_variable]

    return required_keys

  def additional_states_update_fn(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> types.FlowFieldMap:
    """Updates states associated with biofuel combustion.

    The governing equations for the solid phase are:
      ∂ϱf/∂t = -Nf Ff,
      ∂ϱw/∂t = -Fw,
      (Cpf ϱf + Cpw ϱw) ∂Tₛ/∂t = Qradₛ + h aᵥ (Tg - Tₛ) - Fw (Hw + Cpw Tᵥₐₚ) +
          Ff(𝚹 Hf - Cpf Tpyr Nf).
    These reactions result in source terms in the Navier-Stokes equations:
      𝜔ₘₐₛₛ = Nf Ff + Fw,
      𝜔ₜₑₘₚₑᵣₐₜᵤᵣₑ = 1 / Cpg [h aᵥ (Tₛ - Tg) + Qrad,g + (1 - 𝚹) Ff Hf],
      𝜔ₒ = -Nₒ Ff.

    Args:
      kernel_op: A library for the kernel operations.
      replica_id: The index of the current core replica.
      replicas: The topology of the replicas.
      states: A dictionary of the flow field variables.
      additional_states: A dictionary of the auxiliary states in the simulation,
        including all biofuel related states.
      params: An instance of the grid parameters.

    Returns:
      A dictionary of updated biofuel states and source terms for the gas phase
      temperature, species mass fractions (e.g. oxygen), and humidity.
    """
    combustion_model = self.pyrolysis_char_oxidation.combustion_model
    rho_f_init = additional_states.get('rho_f_init', None)

    combustion_fn = combustion_model.moist_wood_update_fn(
        rho_f_init, self.dehydration.dehydration_rate, False
    )
    return combustion_fn(
        kernel_op, replica_id, replicas, states, additional_states, params
    )
