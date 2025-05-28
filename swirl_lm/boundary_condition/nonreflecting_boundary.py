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

"""A library to handle nonreflecting boundary condition."""

from collections import abc

from absl import logging
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.boundary_condition import boundary_condition_utils
from swirl_lm.communication import halo_exchange
from swirl_lm.equations import common
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldMap = types.FlowFieldMap
BCParams = parameters_lib._BCParams  # pylint: disable=protected-access


def nonreflecting_bc_state_init_fn(
    params: parameters_lib.SwirlLMParameters) -> FlowFieldMap:
  """Initializes states used in nonreflecting boundary calculations.

  Args:
    params: An instance of `SwirlLMParameters`.

  Returns:
    A string-keyed dictionary (a `FlowFieldMap`) containing the initialized
    boundary condition slices.
  """
  bc_map = {}
  base_shape = (params.nz, params.nx, params.ny)
  for k in boundary_condition_utils.get_keys_for_boundary_condition(
      params.bc,
      halo_exchange.BCType.NONREFLECTING
  ):
    bc_info = params.bc_manager.parse_key(k)
    dim = bc_info[1]
    shape = list(base_shape)
    shape[(dim + 1) % 3] = params.halo_width
    bc_map[k] = tf.zeros(shape, types.TF_DTYPE)
  return bc_map


def nonreflecting_bc_state_update_fn(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    params: parameters_lib.SwirlLMParameters,
    states: FlowFieldMap,
    additional_states: FlowFieldMap,
    step_id: tf.Tensor,
) -> FlowFieldMap:
  """Updates states used in nonreflecting boundary calculations.

  Using the "high" side of the domain as an example, a forward Euler with
  upwinding scheme is used to solve the boundary:

     ∂ϕ/∂t = - U* ∂ϕ/∂x

  In discrete form:

     ϕⱼⁿ⁺¹ = (1 - Δt U* /Δx) ϕⱼⁿ + (Δt U* /Δx) ϕⱼ₋₁ⁿ

  Similarly for "low" side of the domain boundary, we have a symmetric case:

     ∂ϕ/∂t = U* ∂ϕ/∂x

  In discrete form:

     ϕⱼⁿ⁺¹ = (1 - Δt U* /Δx) ϕⱼⁿ + (Δt U* /Δx) ϕⱼ₊₁ⁿ

  In the current implementation, U* is specified by the configuration
  through the value in the boundary condition and the mode (specified
  through `boundary_info.bc_params.nonreflecting_bc_mode`) setting as:

  For NONREFLECTING_LOCAL_MAX mode:

     U* = abs(max(u + u0, 0)) for the "high" side of the domain, and
     U* = abs(min(u - u0, 0)) for the "low" side of the domain.

  For NONREFLECTING_GLOBAL_MEAN mode:

     U* = abs(max(global_mean(u), 0) + u0), for the "high" side of the domain,
     U* = abs(min(global_mean(u), 0) - u0), for the "low" side of the domain.

  For NONREFLECTING_GLOBAL_MAX mode:

     U* = abs(max(global_max(u), 0) + u0), for the "high" side of the domain,
     U* = abs(min(global_min(u), 0) - u0), for the "low" side of the domain.

  where u is the (spatially dependent) velocity field at the boundary
  (first inner fluid layer), while u0 is value specified for the boundary
  condition configuration.

  Args:
    kernel_op: An object holding a library of kernel operations.
    replica_id: The id of the replica.
    replicas: The replicas. In particular, a numpy array that maps grid
      coordinates to replica id numbers.
    params: An instance of `SwirlLMParameters`.
    states: Essential states. Must the prognostic variables the nonreflecting
      boundary is applied to and the corresponding velocity fields.
    additional_states: Additional states that contain the bc value states.
    step_id: The step id when this update is being called.

  Returns:
    A string-keyed dictionary (a `FlowFieldMap`) containing the initialized
    boundary condition slices.
  """
  del kernel_op, replica_id
  updated_additional_states = {}
  spaces = params.grid_spacings
  halo_width = params.halo_width
  for k in boundary_condition_utils.get_keys_for_boundary_condition(
      params.bc,
      halo_exchange.BCType.NONREFLECTING
  ):
    if any(params.use_stretched_grid):
      raise NotImplementedError(
          'Stretched grid is not yet supported for nonreflecting boundary'
          ' condition.'
      )
    varname, dim, face = params.bc_manager.parse_key(k)
    bc_type, u_threshold = params.bc[varname][dim][face]
    bc_params = params.bc_params[varname][dim][face]
    logging.info('Variable: %s, dimension: %d, face: %d is specified '
                 'with nonreflecting bc_type: %s, with velocity threshold '
                 'value: %f and bc_params %s', varname, dim, face, bc_type,
                 u_threshold, bc_params)
    phi = states[varname]
    # Check whether it's 3D tensor mode or list of 2D tensor mode.
    is_list = isinstance(phi, abc.Sequence)
    group_assignment = np.array(
        [np.reshape(x, -1) for x in
         np.split(replicas, replicas.shape[dim], axis=dim)])

    def phase_velocity(u, varname, dim, face, mode, group_assignment):
      """Calculates the convection phase velocity for the boundary at face."""
      if mode == BCParams.NONREFLECTING_LOCAL_MAX:
        if face == 0:
          return tf.nest.map_structure(
              lambda u_i: tf.math.minimum(u_i - u_threshold, 0.0), u)
        else:
          return tf.nest.map_structure(
              lambda u_i: tf.math.maximum(u_i + u_threshold, 0.0), u)
      else:
        halos_to_strip = [halo_width] * 3
        halos_to_strip[dim] = 0

        u_inner = common_ops.strip_halos(u, halos_to_strip)
        sign = -1.0 if face == 0 else 1.0
        if mode == BCParams.NONREFLECTING_GLOBAL_MEAN:
          reduce_op = tf.math.reduce_mean
        elif mode == BCParams.NONREFLECTING_GLOBAL_MAX:
          reduce_op = tf.math.reduce_min if face == 0 else tf.math.reduce_max
        else:
          raise ValueError(
              f'Unsupported Mode: {mode} for NONREFLECTING boundary condition '
              f'for variable: {varname}, dim: {dim}, face: {face}')

        reduced_velocity = common_ops.global_reduce(
            u_inner, reduce_op, group_assignment)

        # Preventing backflow.
        reduced_velocity = (tf.math.minimum(reduced_velocity, 0) if face == 0
                            else tf.math.maximum(reduced_velocity, 0))

        return tf.nest.map_structure(
            lambda u_i: tf.ones_like(u_i) * (  # pylint: disable=g-long-lambda
                reduced_velocity + sign * u_threshold), u)

    def get_face(phi, dim, face, halo_width, is_list):
      u = common_ops.get_face(phi, dim, face, halo_width)
      if not is_list or dim in (0, 1):
        return u[0]
      else:
        return u

    velocity_keys = [common.KEY_U, common.KEY_V, common.KEY_W]
    u = get_face(
        states[velocity_keys[dim]], dim, face, halo_width, is_list)

    phase_u = phase_velocity(u, varname, dim, face,
                             bc_params.nonreflecting_bc_mode, group_assignment)

    def get_cfl(phase_u, dim):
      return tf.nest.map_structure(
          lambda u_i: tf.abs(u_i) * params.dt / spaces[dim], phase_u)

    cfl = get_cfl(phase_u, dim)
    phi_inner = get_face(phi, dim, face, halo_width, is_list)
    phi_inner_scaled = tf.nest.map_structure(
        lambda phi_inner_i, cfl_i: phi_inner_i * cfl_i, phi_inner, cfl)
    phi_i_1 = tf.nest.map_structure(
        lambda phi_i_1_i, coeff_i: phi_i_1_i * coeff_i,
        get_face(additional_states[k], dim, 1 - face, 0, is_list),
        tf.nest.map_structure(lambda cfl_i: 1.0 - cfl_i, cfl)
    )

    if step_id == bc_params.buffer_init_step:
      # This redundancy is to get around the type annotation check.
      # This just broadcast the first layer of the inner part into the
      # additional_state that persists the BC.
      updated = tf.nest.map_structure(tf.identity, phi_inner)
    else:
      updated = tf.nest.map_structure(tf.add, phi_inner_scaled, phi_i_1)

    if isinstance(additional_states[k], tf.Tensor):
      updated_additional_states[k] = tf.concat(
          [updated,] * halo_width, axis=(dim + 1) % 3)
    else:  # list expression
      if dim == 2:
        updated_additional_states[k] = updated * halo_width
      else:  # dim == 0 or 1
        tile_multiples = [1, 1]
        tile_multiples[dim] = halo_width
        updated_additional_states[k] = [
            tf.tile(val, tile_multiples) for val in updated]

  return updated_additional_states
