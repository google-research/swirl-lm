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

"""Library for the field exchange communication scheme.

In this approach, particles remain controlled by their home device, even
if they are physically located in the region owned by another device. The
local fluid properties at the particle locations are obtained by exchanging
field data between devices. This is in contrast to the particle exchange
approach, where particles are exchanged between devices as they cross over
from one device's region to another.
"""

from typing import TypeAlias
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics.lpt import lpt
from swirl_lm.physics.lpt import lpt_comm
from swirl_lm.physics.lpt import lpt_pb2
from swirl_lm.physics.lpt import lpt_types
from swirl_lm.physics.lpt import lpt_utils
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldMap: TypeAlias = types.FlowFieldMap


FIELD_VALS = ["w", "u", "v"]

LPT_INTS_KEY = lpt_types.LPT_INTS_KEY
LPT_FLOATS_KEY = lpt_types.LPT_FLOATS_KEY
LPT_COUNTER_KEY = lpt_types.LPT_COUNTER_KEY


class FieldExchange(lpt.LPT):
  """Class for the field exchange communication scheme."""

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    super().__init__(params)

    if params.lpt is None:
      raise ValueError("FieldExchange init called but lpt params are None.")

    # Selecting the field exchange communication function.
    if params.lpt.field_exchange.communication_mode == (
        lpt_pb2.LagrangianParticleTracking.FieldExchange.ONE_SHUFFLE
    ):
      self.exchange_fluid_data_fn = lpt_comm.one_shuffle
    elif (
        params.lpt.field_exchange.communication_mode
        == lpt_pb2.LagrangianParticleTracking.FieldExchange.PAIRWISE
    ):
      self.exchange_fluid_data_fn = lpt_comm.pairwise
    else:
      raise NotImplementedError(
          "Field Exchange communication function"
          f" {params.lpt.field_exchange.communication_mode} not supported."
      )

  def update_particles(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Updates the particle trajectories and locations.

    Args:
      replica_id: The ID of the replica that is calling this function.
      replicas: A 3D numpy array containing the replica ID with the global
        domain.
      states: The fluid state fields. See `swirl_lm.base.parameters` for more
        details.
      additional_states: A dictionary of additional particle and fluid states.
        This includes the Lagrangian particle (lpt) states.

    Returns:
      A `FlowFieldMap` containing the updated particle states.
    """
    particles_generated_per_replica = additional_states[LPT_COUNTER_KEY]

    lpt_field_floats = additional_states[LPT_FLOATS_KEY]
    locs = lpt_field_floats[:, :3]

    # Exchange fluid data at particle locations with other replicas.
    with tf.name_scope("communicate_fluid_data"):
      local_min_loc = self._get_local_min_loc(replicas, replica_id)
      fluid_vels = self.exchange_fluid_data_fn(
          locs,
          states,
          replica_id,
          replicas,
          FIELD_VALS,
          self.grid_spacings_zxy,
          self.core_spacings,
          local_min_loc,
          self.global_min_pt,
          self.n_max,
      )

    # TODO(ntricard): Add mass consumption rate function.
    omegas = tf.zeros_like(lpt_field_floats[:, 0], dtype=lpt_types.LPT_FLOAT)

    # Time step the particles, updating their attributes.
    with tf.name_scope("time_step_particles"):
      lpt_field_ints, lpt_field_floats = self.increment_time(
          replica_id, replicas, additional_states, fluid_vels, omegas
      )

    # TODO(ntricard): Account for particles influence on fluid motion.

    # Modulus the locations across periodic boundaries.
    with tf.name_scope("apply_periodic_boundary_conditions"):
      lpt_field_floats = self._apply_periodic_boundary_conditions(
          lpt_field_floats
      )

    # Removing particles that have exited the domain or have vaporized.
    new_locs = lpt_field_floats[:, :3]
    dest_replicas = lpt_utils.get_particle_replica_id(
        new_locs, self.core_spacings, replicas, self.global_min_pt
    )
    lpt_field_ints = self._remove_particles(
        lpt_field_ints, lpt_field_floats, dest_replicas
    )

    return {
        LPT_INTS_KEY: lpt_field_ints,
        LPT_FLOATS_KEY: lpt_field_floats,
        LPT_COUNTER_KEY: particles_generated_per_replica,
    }

  def _apply_periodic_boundary_conditions(
      self, lpt_field_floats: lpt_types.LptFieldFloats
  ) -> lpt_types.LptFieldFloats:
    """Applies periodic boundary conditions to the particle field."""
    locs = lpt_field_floats[:, :3]
    dim_lengths = (self.params.lz, self.params.lx, self.params.ly)
    grid_params = self.params.grid_params_proto
    periodic = (
        grid_params.periodic.dim_2,
        grid_params.periodic.dim_0,
        grid_params.periodic.dim_1,
    )
    for dim, dim_length in enumerate(dim_lengths):
      if not periodic[dim]:
        continue
      # TODO(ntricard): Performance can be improved by only modularizing the
      # locations that hit a periodic boundary.
      new_particle_dim_locs = locs[:, dim] % dim_length
      lpt_field_floats = tf.tensor_scatter_nd_update(
          lpt_field_floats,
          tf.stack([tf.range(self.n_max), tf.fill((self.n_max,), dim)], axis=1),
          new_particle_dim_locs,
      )

    return lpt_field_floats
