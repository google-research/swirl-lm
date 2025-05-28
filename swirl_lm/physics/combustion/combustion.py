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

"""Updates flow field variables and source terms by combustion."""

import dataclasses
from typing import Callable

from absl import logging
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics.combustion import biofuel_multistep
from swirl_lm.physics.combustion import turbulent_kinetic_energy
from swirl_lm.physics.combustion import turbulent_kinetic_energy_pb2
from swirl_lm.physics.combustion import wood
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf


@dataclasses.dataclass
class IgnitionWithHotKernel():
  """Defines parameters for the high-temperature ignition kernel."""
  # The temperature of the ignition kernel.
  ignition_temperature: float = 800.0


@dataclasses.dataclass
class IgnitionWithHeatSource():
  """Defines parameters for heat-source ignition."""
  # The peak magnitude of the ignition heat source. It has a unit of J/m^3/s if
  # the prognostic variable is in forms of energy, and the unit s K/s if the
  # prognostic variable is in forms of temperature. Default value corresponds to
  # fuel with heat of combustion being 20e6 J/kg, fuel bulk density 0.25 kg/m^3,
  # with a burning rate of 0.01 1/s.
  heat_source_magnitude: float = 50.0
  # Define the duration of the heat source to be positioned in the flow field.
  # The procedure for the source term enforcement is shown as follows:
  # t_0 < t <= t_1: src(t) = (t - t_0) / (t_1 - t_0) * src_max;
  # t_1 < t <= t_2: src(t) = src_max;
  # t_2 < t <= t_3: src(t) = (t_3 - t) / (t_3 - t_2) * src_max;
  t_0: float = 0.0
  t_1: float = 300.0
  t_2: float = 2100.0
  t_3: float = 2400.0


def _compute_tke(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    states: types.FlowFieldMap,
    additional_states: types.FlowFieldMap,
    params: parameters_lib.SwirlLMParameters,
    tke_params: turbulent_kinetic_energy_pb2.TKE,
) -> types.FlowFieldVal:
  """Computes the turbulent kinetic energy."""
  tke_update_fn = turbulent_kinetic_energy.tke_update_fn_manager(
      tke_params
  )

  # Determines the turbulent kinetic energy that is required to compute the
  # reaction source term due to combustion.
  additional_states_tke = dict(additional_states)
  additional_states_tke['tke'] = additional_states.get(
      'tke', tf.nest.map_structure(tf.zeros_like, states['u'])
  )
  additional_states_tke = tke_update_fn(
      kernel_op, replica_id, replicas, states, additional_states_tke, params
  )
  return additional_states_tke['tke']


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
      tke = _compute_tke(
          kernel_op,
          replica_id,
          replicas,
          states,
          additional_states,
          params,
          params.combustion.wood.tke,
      )
    else:
      raise ValueError(
          'A TKE model is required for wood combustion, but is undefined in the'
          ' config.'
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
    additional_states_combustion['tke'] = tke
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

  if params.combustion.HasField('biofuel_multistep'):
    model = biofuel_multistep.BiofuelMultistep(params)

    required_additional_states = list(
        model.required_additional_states_keys(states)
    )
    # Removes `tke` from the list of required additional states because it will
    # always be provided by this function.
    required_additional_states.remove('tke')
    assert set(required_additional_states).issubset(additional_states.keys()), (
        'Required additional states missing for the biofuel combustion model.'
        f' Required: {required_additional_states}. Missing:'
        f' {set(required_additional_states) - set(additional_states.keys())}.'
    )

    combustion_model = (
        params.combustion.biofuel_multistep.pyrolysis_char_oxidation.wood
    )
    if combustion_model.HasField('tke'):
      tke = _compute_tke(
          kernel_op,
          replica_id,
          replicas,
          states,
          additional_states,
          params,
          params.combustion.biofuel_multistep.pyrolysis_char_oxidation.wood.tke,
      )
    else:
      raise ValueError(
          'A TKE model is required for wood combustion, but is undefined in the'
          ' config.'
      )

    # Computes the reaction source term. Note that only related additional
    # states are passed as inputs to reduce overhead and ambiguity. Source terms
    # in the original `additional_states` will be overridden by the newly
    # computed reaction source terms.
    additional_states_combustion.update({
        key: val
        for key, val in additional_states.items()
        if key in required_additional_states
    })
    additional_states_combustion['tke'] = tke
    if 'rho_f_init' in additional_states:
      additional_states_combustion['rho_f_init'] = additional_states[
          'rho_f_init'
      ]

    # Add extra additional states that may be used for density computation.
    if 'p_ref' in additional_states:
      additional_states_combustion['p_ref'] = additional_states['p_ref']
    if 'theta_ref' in additional_states:
      additional_states_combustion['theta_ref'] = additional_states['theta_ref']
    additional_states_combustion.update(
        model.additional_states_update_fn(
            kernel_op,
            replica_id,
            replicas,
            states,
            additional_states_combustion,
            params,
        )
    )
  return additional_states_combustion


def ignition_with_hot_kernel(
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

    if params.combustion.HasField('wood') or params.combustion.HasField(
        'biofuel_multistep'
    ):
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
      # Remove moisture from the region of the ignition kernel assuming
      # instantaneous ignition. This prevents the unphysical heat sink provided
      # by instantaneous evaporation of the moisture.
      if 'rho_m' in output:
        output['rho_m'] = tf.nest.map_structure(
            lambda rho_m, ignition_kernel: tf.where(
                tf.greater(ignition_kernel, 0), tf.zeros_like(rho_m), rho_m
            ),
            additional_states['rho_m'],
            additional_states['ignition_kernel'],
        )

    return output

  return ignition_step


def ramp_up_down_function(
    t_0: float,
    t_1: float,
    t_2: float,
    t_3: float,
) -> Callable[[tf.Tensor], tf.Tensor]:
  """Returns the value of the ramp up/down function at time `t`.

  The function is defined as:
    t <= t_0: 0
    t_0 < t <= t_1: linearly ramp up from 0 to 1
    t_1 < t <= t_2: 1
    t_2 < t <= t_3: linearly ramp down from 1 to 0
    t > t_3: 0

  Args:
    t_0: Time at the start of the ramp up.
    t_1: Time at the end of the ramp up.
    t_2: Time at the start of the ramp down.
    t_3: Time at the end of the ramp down.

  Returns:
    A function that takes time as the input, and returns a scaling factor
    following the ramp function.
  """
  assert t_0 <= t_1 <= t_2 <= t_3, (
      '`t_0`, `t_1`, `t_2`, and `t_3` has to be in the ascending order, but '
      f'got {t_0}, {t_1} {t_2}, {t_3}'
  )

  def scaling_factor(t: tf.Tensor) -> tf.Tensor:
    """Computes the scaling factor following the ramp up and down function.

    Args:
      t: Time `t` at which to evaluate the function.

    Returns:
      The scaling factor at `t` following the ramp function.
    """
    t_0_to_t_1 = (
        lambda: (t - t_0) / (t_1 - t_0) if t_1 > 0.0 else tf.ones_like(t)
    )
    t_1_to_t_2 = lambda: tf.constant(1.0, dtype=types.TF_DTYPE)
    t_2_to_t_3 = lambda: (t - t_3) / (t_2 - t_3)
    outside_intervals = lambda: tf.constant(0.0, dtype=types.TF_DTYPE)

    t_interval = lambda t_l, t_r: tf.logical_and(
        tf.greater(t, t_l), tf.less_equal(t, t_r)
    )

    return tf.cond(
        t_interval(t_0, t_1),
        t_0_to_t_1,
        lambda: tf.cond(
            t_interval(t_1, t_2),
            t_1_to_t_2,
            lambda: tf.cond(
                t_interval(t_2, t_3),
                t_2_to_t_3,
                outside_intervals,
            ),
        ),
    )

  return scaling_factor


def ignition_with_heat_source(
    heat_source_magnitude: float,
    heat_source_coeff_fn: Callable[[tf.Tensor], tf.Tensor],
) -> Callable[[types.FlowFieldVal, tf.Tensor], types.FlowFieldVal]:
  """Generates a function that produces a heat source for ignition.

  The shape of the source term is prescribed by a helper variable named
  `ignition_kernel`. A source term will be added to the energy equation
  with the following procedure defined `by heat_source_coeff_fn`, i.e. the
  magnitude of the heat source added to the energy equation is the product of
  `heat_source_magnitude` and the scaling factor obtained from the
  `heat_source_coeff_fn`.

  Args:
    heat_source_magnitude: The maximum value of the heat source. It has a unit
      of J/m^3/s if the prognostic variable is in forms of energy, and the unit
      is K/s if the prognostic variable is in forms of temperature.
    heat_source_coeff_fn: A function that computes a scaling factor for the
      heat source at a specific time.

  Returns:
    A function that takes a 3D tensor that represents the ignition kernel, and a
    `tf.Tensor` representing the time as input, returns a heat source for
    ignition.
  """

  def heat_source_update_fn(
      ignition_kernel: types.FlowFieldVal,
      t: tf.Tensor,
  ) -> types.FlowFieldVal:
    """Computes the heat source at `t` given x, y, z.

    Args:
      ignition_kernel: A 3D tensor of binary floating point values (0 and 1),
        with 1 specifying the ignition location.
      t: The time at the current step, in units of s.

    Returns:
      The heat source at time `t`.
    """
    coeff = heat_source_coeff_fn(t)
    return tf.nest.map_structure(
        lambda mask: coeff * heat_source_magnitude * mask, ignition_kernel
    )

  return heat_source_update_fn
