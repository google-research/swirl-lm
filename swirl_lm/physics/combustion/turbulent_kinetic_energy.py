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

"""A library for the turbulent kinetic energy (TKE) modeling."""

import functools

import numpy as np
from swirl_lm.numerics import filters
from swirl_lm.physics.combustion import turbulent_kinetic_energy_pb2
from swirl_lm.utility import common_ops
from swirl_lm.utility import composite_types
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap
StatesUpdateFn = composite_types.StatesUpdateFn


def _update_local_halos(value: FlowFieldVal, halo_width: int) -> FlowFieldVal:
  """Pads halos with symmetric condition to `value`."""
  if halo_width < 1:
    raise ValueError(
        '`halo_width` has to be greater than 1. {} is provided.'.format(
            halo_width
        )
    )

  if not isinstance(value, tf.Tensor):
    # pylint: disable=g-complex-comprehension
    value_xy_valid = [
        tf.pad(
            value_i[halo_width:-halo_width, halo_width:-halo_width],
            paddings=((halo_width, halo_width), (halo_width, halo_width)),
            mode='SYMMETRIC',
        )
        for value_i in value[halo_width:-halo_width]
    ]
    # pylint: enable=g-complex-comprehension
    return (
        [value_xy_valid[0]] * halo_width
        + value_xy_valid
        + [value_xy_valid[-1]] * halo_width
    )
  else:
    return tf.pad(
        value[
            halo_width:-halo_width,
            halo_width:-halo_width,
            halo_width:-halo_width,
        ],
        paddings=[(halo_width, halo_width)] * 3,
        mode='SYMMETRIC',
    )


def _local_box_filter_3d(state: FlowFieldVal) -> FlowFieldVal:
  """Applies a box filter of width 3 to `state` locally.

  This is a function that is useful to methods in this file only, which assumes
  the halos of the input are valid. For formal filter operations please use the
  `global_box_filter_3d` function in `incompressible_structured_mesh_numerics`.
  Note that the `kernel_op` that is used to perform the filter operation is
  defined within this function, which has a minimum width of 8.

  Args:
    state: A 3D tensor to be filtered.

  Returns:
    The filtered 3D tensor.
  """
  halo_update_fn = functools.partial(_update_local_halos, halo_width=1)

  return filters.global_box_filter_3d(
      state, halo_update_fn, filter_width=3, num_iter=1
  )


def constant_tke_update_function(tke_value: float) -> StatesUpdateFn:
  """Generates an update function for TKE with a constant value.

  Args:
    tke_value: A constant that specifies the value of the TKE.

  Returns:
    A function that updates TKE to the constant value that equals `tke_value`.
  """

  def additional_states_update_fn(
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Updates 'tke' in `additional_states`."""
    del kernel_op, replica_id, replicas, states, params

    tke = tf.nest.map_structure(
        lambda tke_i: tke_value * tf.ones_like(tke_i), additional_states['tke']
    )

    updated_additional_states = {}
    for key, value in additional_states.items():
      if key == 'tke':
        updated_additional_states.update({'tke': tke})
      else:
        updated_additional_states.update({key: value})

    return updated_additional_states

  return additional_states_update_fn


def algebraic_tke_update_function() -> StatesUpdateFn:
  """Generates an function that updates TKE algebraically.

  With this model, the TKE is computed with velocity fluctuations estimated by
  the filtered quantities following a scale similarity approximation in
  turbulence.

  Returns:
    A function that updates TKE with the current velocity field.
  """

  def additional_states_update_fn(
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Updates 'tke' in `additional_states`."""
    del kernel_op, replica_id, replicas

    if any(params.use_stretched_grid):
      raise ValueError('Algebraic TKE model does not support stretched grids.')

    u_mean = _local_box_filter_3d(states['u'])
    v_mean = _local_box_filter_3d(states['v'])
    w_mean = _local_box_filter_3d(states['w'])

    tke = tf.nest.map_structure(
        lambda u_i, u_mean_i, v_i, v_mean_i, w_i, w_mean_i: 0.5
        * (
            (u_i - u_mean_i) ** 2
            + (v_i - v_mean_i) ** 2
            + (w_i - w_mean_i) ** 2
        ),
        states['u'],
        u_mean,
        states['v'],
        v_mean,
        states['w'],
        w_mean,
    )

    updated_additional_states = {}
    for key, value in additional_states.items():
      if key == 'tke':
        updated_additional_states.update({'tke': _local_box_filter_3d(tke)})
      else:
        updated_additional_states.update({key: value})

    return updated_additional_states

  return additional_states_update_fn


def turbulent_viscosity_tke_update_function() -> StatesUpdateFn:
  """Generates a function that updates TKE from turbulent viscosity.

  Reference to the model:
  Pressel, Kyle G., Colleen M. Kaul, Tapio Schneider, Zhihong Tan, and
  Siddhartha Mishra. 2015. “Large-Eddy Simulation in an Anelastic Framework with
  Closed Water and Entropy Balances: LARGE-EDDY SIMULATION FRAMEWORK.” Journal
  of Advances in Modeling Earth Systems 7 (3): 1425–56.

  Note that `nu_t` and `tke` are required in the `additional_states` to use this
  function.

  Returns:
    A function that updates TKE with the current velocity field.
  """
  # An empirical constant provided in Pressel et. al. that is mentioned in the
  # docstring above.
  c_k = 0.1

  def additional_states_update_fn(
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Updates 'tke' in `additional_states`."""
    del kernel_op, replica_id, replicas, states

    if 'nu_t' not in additional_states.keys():
      raise ValueError(
          '`nu_t` is required to use the turbulence viscosity to '
          'compute the TKE.'
      )

    use_3d_tf_tensor = isinstance(additional_states['nu_t'], tf.Tensor)
    dx_dy_dz = tuple(
        params.physical_grid_spacing(dim, use_3d_tf_tensor, additional_states)
        for dim in (0, 1, 2)
    )
    delta = common_ops.map_structure_3d(
        lambda dx, dy, dz: (dx * dy * dz) ** (1 / 3), *dx_dy_dz
    )
    tke = tf.nest.map_structure(
        lambda nu_t_i, delta_i: tf.square(nu_t_i / (c_k * delta_i)),
        additional_states['nu_t'],
        delta,
    )

    updated_additional_states = {}
    for key, value in additional_states.items():
      if key == 'tke':
        updated_additional_states.update(
            {'tke': _update_local_halos(tke, params.halo_width)}
        )
      else:
        updated_additional_states.update({key: value})

    return updated_additional_states

  return additional_states_update_fn


def tke_update_fn_manager(
    tke_update_option: turbulent_kinetic_energy_pb2.TKE,
) -> StatesUpdateFn:
  """Generates the TKE update function requested by `tke_update_option`.

  Args:
    tke_update_option: The method that is used to update the TKE.

  Returns:
    The update function for TKE.

  Raises:
    ValueError: If `tke_update_option` is not one of the following: "CONSTANT",
      "ALGEBRAIC", "TURBULENT_VISCOSITY"
  """

  if tke_update_option.WhichOneof('tke_model_option') == 'constant':
    update_fn = constant_tke_update_function(
        tke_update_option.constant.tke_constant
    )
  elif tke_update_option.WhichOneof('tke_model_option') == 'algebraic':
    update_fn = algebraic_tke_update_function()
  elif (
      tke_update_option.WhichOneof('tke_model_option') == 'turbulent_viscosity'
  ):
    update_fn = turbulent_viscosity_tke_update_function()
  else:
    raise ValueError(
        f'Undefined TKE model {tke_update_option}. Available models are: '
        '"constant", "algebraic", "turbulent_viscosity".'
    )

  return update_fn
