# Copyright 2023 The swirl_lm Authors.
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

"""Updates flow field variables and source terms by combustion."""

from typing import Callable

from absl import logging
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics.combustion import turbulent_kinetic_energy
from swirl_lm.physics.combustion import wood
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf


def combustion_step(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    step_id: tf.Tensor,
    states: types.FlowFieldMap,
    additional_states: types.FlowFieldMap,
    params: parameters_lib.SwirlLMParameters,
) -> types.FlowFieldMap:
  """Updates states and computes source terms due to combustion.

  Args:
    kernel_op: An object holding a library of kernel operations.
    replica_id: The id of the replica.
    replicas: The replicas. In particular, a numpy array that maps grid
      coordinates to replica id numbers.
    step_id: The index of the current time step.
    states: A keyed dictionary of states that will be updated.
    additional_states: A list of states that are needed by the update fn, but
      will not be updated by the main governing equations.
    params: An instance of the Swirl-LM simulation global context.

  Returns:
    A dictionary with updated flow field variables and/or source terms due to
    combustion.

  Raises:
    AssertionError: If not all required additional states are provided.
    ValueError: If TKE model is not defined in the config for wood combustion.
  """
  del step_id
  if params.combustion is None:
    return {}

  additional_states_combustion = {}

  if params.combustion.HasField('wood'):
    model = wood.wood_combustion_factory(params)

    required_additional_states = list(
        model.required_additional_states_keys(states)
    )
    # Removes `tke` from the list of required additional states because it will
    # always be provided by this function.
    required_additional_states.remove('tke')
    assert set(required_additional_states).issubset(additional_states.keys()), (
        'Required additional states missing for the wood combustion model.'
        f' Required: {required_additional_states}. Missing:'
        f' {set(required_additional_states) - set(additional_states.keys())}.'
    )

    if (
        params.combustion is not None
        and params.combustion.HasField('wood')
        and params.combustion.wood.HasField('tke')
    ):
      tke_update_fn = turbulent_kinetic_energy.tke_update_fn_manager(
          params.combustion.wood.tke
      )
    else:
      raise ValueError(
          'A TKE model is required for wood combustion, but is undefined in the'
          ' config.'
      )

    # Determines the turbulent kinetic energy that is required to compute the
    # reaction source term due to wood combustion.
    additional_states_tke = {}
    if 'nu_t' in additional_states:
      additional_states_tke['nu_t'] = additional_states['nu_t']
    additional_states_tke['tke'] = additional_states.get(
        'tke', tf.nest.map_structure(tf.zeros_like, states['u'])
    )
    additional_states_tke = tke_update_fn(
        kernel_op, replica_id, replicas, states, additional_states_tke, params
    )

    # Computes the reaction source term. Note that only related additional
    # states are passed as inputs to reduce overhead and ambiguity. Source terms
    # in the original `additional_states` will be overridden by the newly
    # computed reaction source terms.
    additional_states_combustion.update(
        {
            key: val
            for key, val in additional_states.items()
            if key in required_additional_states
        }
    )
    additional_states_combustion['tke'] = additional_states_tke['tke']
    # Add extra additional states that may be used for density computation.
    if 'p_ref' in additional_states:
      additional_states_combustion['p_ref'] = additional_states['p_ref']
    if 'theta_ref' in additional_states:
      additional_states_combustion['theta_ref'] = additional_states['theta_ref']
    additional_states_combustion.update(
        model.update_fn(additional_states.get('rho_f_init'))(
            kernel_op,
            replica_id,
            replicas,
            states,
            additional_states_combustion,
            params,
        )
    )

  return additional_states_combustion


def ignition_step_fn(
    energy_varname: str,
    ignition_temperature: float,
    params: parameters_lib.SwirlLMParameters,
) -> Callable[
    [
        get_kernel_fn.ApplyKernelOp,
        tf.Tensor,
        np.ndarray,
        types.FlowFieldMap,
        types.FlowFieldMap,
        grid_parametrization.GridParametrization,
    ],
    types.FlowFieldMap,
]:
  """Generates a function that updates related variables with ignition.

  The returned function can be used as a 'preprocessing_step_fn' or a
  'postprocessing_step_fn'.

  Args:
    energy_varname: The name of the energy prognostic variable.
    ignition_temperature: The temperature of the ignition kernel [K].
    params: An instance of the Swirl-LM simulation global context.

  Returns:
    A pre/post_processing_step_fn that updates related variables, such as
    temperature, density, and the velocity field, incorporating the ignition
    condition.
  """

  thermo_model = thermodynamics_manager.thermodynamics_factory(params)

  def ignite(
      ignition_kernel: tf.Tensor,
      temperature: tf.Tensor,
  ) -> tf.Tensor:
    """Sets a high temperature region at the specified location."""
    return temperature + (ignition_temperature - temperature) * ignition_kernel

  def ignition_step(
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
      grid_params: grid_parametrization.GridParametrization,
  ) -> types.FlowFieldMap:
    """Updates related flow field variables with ignition.

    This function is invoked only once at the step specified in the commandline
    flag.

    Args:
      kernel_op: An object holding a library of kernel operations.
      replica_id: The id of the replica.
      replicas: The replicas. In particular, a numpy array that maps grid
        coordinates to replica id numbers.
      states: A string-keyed dictionary of states that will be updated.
      additional_states: A list of states that are needed by the update fn, but
        will not be updated by the main governing equations.
      grid_params: An instance of `grid_parametrization.GridParametrization`.

    Returns:
      A dictionary that is a union of `states` and `additional_states` with
      updated `theta` (or `T`), and `T_s` with elevated temperature in
      locations specified by `ignition_kernel`. `rho`, `u`, `v`, and `w` will
      be rescaled for momentum conservation.
    """
    del kernel_op, replica_id, replicas, grid_params
    output = {}
    output.update(states)
    output.update(additional_states)

    if params.combustion is None:
      return output

    if params.combustion.HasField('wood'):
      logging.info('Igniting the flow field with the wood combustion model.')

      assert energy_varname in ('T', 'theta', 'theta_li'), (
          'Ignition for the wood combustion model allows energy variables `T`,'
          f' `theta`, and `theta_li` only, but {energy_varname} is provided.'
      )

      temperature = tf.nest.map_structure(
          ignite, additional_states['ignition_kernel'], states[energy_varname]
      )
      t_s_ign = tf.nest.map_structure(
          ignite, additional_states['ignition_kernel'], additional_states['T_s']
      )
      temperature = tf.nest.map_structure(
          lambda rho_f, t_ign, t_0: tf.where(  # pylint: disable=g-long-lambda
              tf.greater(rho_f, 0.0), t_ign, t_0
          ),
          additional_states['rho_f'],
          temperature,
          states[energy_varname],
      )
      t_s = tf.nest.map_structure(
          lambda rho_f, t_ign, t_0: tf.where(  # pylint: disable=g-long-lambda
              tf.greater(rho_f, 0.0), t_ign, t_0
          ),
          additional_states['rho_f'],
          t_s_ign,
          additional_states['T_s'],
      )
      thermo_states = dict(states)
      thermo_states.update({energy_varname: temperature, 'Y_O': states['Y_O']})
      rho = thermo_model.update_thermal_density(
          thermo_states, additional_states
      )
      output.update({'rho': rho, energy_varname: temperature, 'T_s': t_s})
      # Rescales the velocity so that the momentum stays the same as before
      # ignition where mass conservation is enforced.
      output.update(
          {  # pylint: disable=g-complex-comprehension
              vel: tf.nest.map_structure(
                  lambda rho_0, rho_ign, u: rho_0 * u / rho_ign,
                  states['rho'],
                  rho,
                  states[vel],
              )
              for vel in ('u', 'v', 'w')
          }
      )

    return output

  return ignition_step
