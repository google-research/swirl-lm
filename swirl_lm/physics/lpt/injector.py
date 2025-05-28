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

"""Functions that control how particles are injected into the field."""

import abc
import dataclasses
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics.lpt import lpt_pb2
from swirl_lm.physics.lpt import lpt_types
from swirl_lm.utility import types
import tensorflow as tf

LPT_FLOAT = lpt_types.LPT_FLOAT
LPT_INT = lpt_types.LPT_INT

LPT_FLOATS_KEY = lpt_types.LPT_FLOATS_KEY
LPT_INTS_KEY = lpt_types.LPT_INTS_KEY
LPT_COUNTER_KEY = lpt_types.LPT_COUNTER_KEY

ParticleAttributes = (
    lpt_pb2.LagrangianParticleTracking.LptInjector.ParticleAttributes
)


@dataclasses.dataclass
@abc.abstractmethod
class LptInjector(abc.ABC):
  """Defines a region where particles can be injected.

  Attributes:
    injector_params: The LptInjector proto containing the parameters for this
      injector region.
  """

  injector_params: lpt_pb2.LagrangianParticleTracking.LptInjector

  def inject(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
      step_id: tf.Tensor,
      params: parameters_lib.SwirlLMParameters,
  ) -> types.FlowFieldMap:
    """Injects particles into the field.

    This function is called at the time step defined by
    `initial_injection_step`. If the injection schedule defined in the proto
    specifies a period, the function checks if the current time step is a
    multiple of the period and calls the injector if it is.

    Args:
      replica_id: The ID of the replica that is calling this function.
      replicas: A 3D numpy array containing the replica ID with the global
        domain.
      states: The fluid state fields. See `swirl_lm.base.parameters` for more
        details.
      additional_states: A dictionary of additional particle and fluid states.
        This includes the Lagrangian particle (lpt) states.
      step_id: The current time step.
      params: The SwirlLMParameters object containing the LPT parameters.

    Returns:
      A FlowFieldMap with injected particle field values.
    """
    del states
    if params.lpt is None:
      raise ValueError("LPT inject called but lpt params are None.")

    # Unpacking injector parameters.
    injection_schedule = self.injector_params.injection_schedule
    attributes = self.injector_params.attributes
    initial_inject_step = injection_schedule.initial_injection_step
    period = (
        injection_schedule.period
        if injection_schedule.HasField("period")
        else None
    )
    n_particles = tf.constant(attributes.n_particles, LPT_INT)

    # Early return if we do not inject at this time step.
    if attributes.n_particles == 0 or (
        step_id != initial_inject_step
        and (period is None or (step_id - initial_inject_step) % period != 0)
    ):
      return {
          LPT_INTS_KEY: tf.zeros((0, 2), LPT_INT),
          LPT_FLOATS_KEY: tf.zeros((0, 7), LPT_FLOAT),
          LPT_COUNTER_KEY: tf.constant(0, LPT_INT),
      }
    num_replicas = tf.size(replicas, LPT_INT)

    # Number of new particles total in the domain.
    exchange_method = params.lpt.WhichOneof("lpt_parallel_approach")
    if exchange_method == "field_exchange":
      check_op = tf.debugging.assert_equal(
          n_particles % num_replicas,
          0,
          "`n_particles` for each injector must be a multiple of the number"
          " of replicas in the `field_exhange` approach: "
          f" {n_particles=} and {num_replicas=}.",
      )
      with tf.control_dependencies([check_op]):
        replica_n_particles = tf.cast(n_particles / num_replicas, LPT_INT)
    else:
      raise ValueError(
          f"LPT parallel approach not supported: {exchange_method}."
      )

    # Generating particle IDs.
    particle_replica_counter = additional_states[LPT_COUNTER_KEY]
    ids = self._get_particle_ids(
        replica_n_particles, particle_replica_counter, num_replicas, replica_id
    )

    # Random seeds.
    op_seed = ids[0]
    core_seed = self.injector_params.seed * replica_id

    # Generating random attributes.
    locs = self._get_locs(replica_n_particles, (core_seed, op_seed))
    with tf.name_scope("stretched_grid_injection"):
      locs = self._correct_for_stretched_grid(locs, params)
    vels = self._get_vels(
        replica_n_particles, attributes, (core_seed, op_seed + 3)
    )
    masses = self._get_masses(replica_n_particles, attributes)

    new_particle_ints = tf.stack([tf.ones_like(ids, LPT_INT), ids], axis=1)
    new_particle_floats = tf.concat([locs, vels, masses[:, tf.newaxis]], axis=1)

    return {
        LPT_INTS_KEY: new_particle_ints,
        LPT_FLOATS_KEY: new_particle_floats,
        LPT_COUNTER_KEY: replica_n_particles,
    }

  def _get_particle_ids(
      self,
      n_particles: tf.Tensor,
      particles_generated_per_replica: tf.Tensor,
      num_replicas: tf.Tensor,
      replica_id: tf.Tensor,
  ) -> tf.Tensor:
    """Generates particle IDs that are unique across all replicas.

    This equation ensures that particle IDs are unique across all replicas
    without requiring a global counter.

    Args:
      n_particles: Number of particles to inject.
      particles_generated_per_replica: A number tracking the total number of
        particles that have been injected into the field.
      num_replicas: The total number of replicas used globally in the domain.
      replica_id: The ID of the replica that is calling this function.

    Returns:
      A tensor of shape `(n_particles,)` containing the particle IDs.
    """
    return (
        particles_generated_per_replica + tf.range(n_particles, dtype=LPT_INT)
    ) * num_replicas + replica_id

  def _get_vels(
      self,
      n_particles: tf.Tensor,
      attributes: ParticleAttributes,
      seed: tuple[int, int],
  ) -> tf.Tensor:
    """Generates velocities for the particles.

    If a velocity is specified in the proto file, all particles are assigned
    that velocity. Otherwise, the directions are uniformly distributed across
    the unit sphere with magnitudes uniformly random between 0 and `speed_max`.

    Args:
      n_particles: Number of particles to inject.
      attributes: The attribute proto of the injector.
      seed: A size 2 tuple containing seed and operation seed. See
        https://www.tensorflow.org/guide/random_numbers#stateless_rngs.

    Returns:
      A tensor of shape `(n_particles, 3)` containing the particle directions.
    """
    if attributes.HasField("velocity"):
      vel = attributes.velocity
      return tf.tile(
          tf.constant((vel.z, vel.x, vel.y), LPT_FLOAT, shape=(1, 3)),
          (n_particles, 1),
      )

    randoms = tf.random.stateless_uniform(
        (n_particles, 3), seed, dtype=LPT_FLOAT
    )
    thetas = tf.cast(tf.math.acos(1 - 2 * randoms[:, 0]), LPT_FLOAT)
    phis = tf.cast(2 * np.pi * randoms[:, 1], LPT_FLOAT)
    speeds = tf.cast(attributes.speed_max * randoms[:, 2], LPT_FLOAT)

    cos_thetas = tf.math.cos(thetas)
    u = cos_thetas * tf.math.cos(phis) * speeds
    v = cos_thetas * tf.math.sin(phis) * speeds
    w = tf.math.sin(thetas) * speeds
    return tf.stack([w, u, v], axis=1)

  def _get_masses(
      self,
      n_particles: tf.Tensor,
      attributes: ParticleAttributes,
  ) -> tf.Tensor:
    """Generates masses for the particles.

    Args:
      n_particles: Number of particles to inject.
      attributes: The attributes of the particles.

    Returns:
      A tensor of shape `(n_particles,)` containing the particle masses.
    """
    return tf.fill((n_particles,), tf.constant(attributes.mass, LPT_FLOAT))

  def _correct_for_stretched_grid(
      self,
      locs: tf.Tensor,
      params: parameters_lib.SwirlLMParameters,
  ) -> tf.Tensor:
    """Maps locations from physical to mapped grid when stretched grid is used.

    This function is a no-op for any dimensions where stretched grid is not
    used.

    Args:
      locs: A tensor of shape `(n, 3)` containing `n` physical locations in
        order z, x, and y.
      params: The SwirlLMParameters object containing stretched grid parameters.

    Returns:
      A tensor of shape `(n, 3)` with dimensions z, x, and y, where the
      locations are mapped to the mapped grid along dimensions where a stretched
      grid is used.
    """

    def _make_idx_func(grid):
      def _get_loc_indices(loc, grid):
        return tf.cast(
            tf.where(
                tf.logical_and(
                    tf.greater_equal(loc, grid[:-1]),
                    tf.less(loc, grid[1:]),
                )
            )[0, 0],
            LPT_INT,
        )

      return lambda loc: _get_loc_indices(loc, grid)

    for dim in (0, 1, 2):
      dim_xyz = (dim + 2) % 3  # Is zxy for lpt by default.
      if not params.use_stretched_grid[dim_xyz]:
        continue

      dim_grid = params.global_xyz[dim_xyz]

      # Determining the nodal locations.
      with tf.name_scope("gathering_indices_in_stretched_grid"):

        indices = tf.map_fn(
            _make_idx_func(dim_grid),
            locs[:, dim],
            fn_output_signature=LPT_INT,
        )

      # Determining how far between the node and the next node each particle is.
      with tf.name_scope("calculating_weights"):
        prev_loc = tf.gather(dim_grid, indices)
        next_loc = tf.gather(dim_grid, indices + 1)
        w = (locs[:, dim] - prev_loc) / (next_loc - prev_loc)

      # Adding the two to get the overall new mapped particle locs.
      with tf.name_scope("calculating_new_locs"):
        new_dim_locs = tf.cast(indices, LPT_FLOAT) + w

        locs = tf.tensor_scatter_nd_update(
            locs,
            tf.stack(
                [
                    tf.range(tf.shape(locs)[0]),
                    tf.fill((tf.shape(locs)[0],), dim),
                ],
                axis=1,
            ),
            new_dim_locs,
        )

    return locs

  @abc.abstractmethod
  def _get_locs(self, n: tf.Tensor, seed: tuple[int, int]) -> tf.Tensor:
    """Evaluates locations of injected particles.

    Args:
      n: Number of particles to inject.
      seed: A size 2 tuple containing seed and operation seed. See
        https://www.tensorflow.org/guide/random_numbers#stateless_rngs.

    Returns:
      A tensor of shape `(n, 3)` containing the particle locations.
    """
    pass


class Box(LptInjector):
  """Defines a box where particles can be injected.

  Attributes:
    bounding_box: A size 2 tuple containing the minimum point and maximum point
      of the box to inject particles uniformly into.
  """

  def __init__(
      self,
      injector_params: lpt_pb2.LagrangianParticleTracking.LptInjector,
  ):
    super().__init__(injector_params)

    min_pt = injector_params.box.min_pt
    max_pt = injector_params.box.max_pt
    self.bounding_box = (
        (min_pt.z, min_pt.x, min_pt.y),
        (max_pt.z, max_pt.x, max_pt.y),
    )

  def _get_locs(self, n: tf.Tensor, seed: tuple[int, int]) -> tf.Tensor:
    """Injects particles into box uniformly in space and direction.

    Args:
      n: Number of particles to inject.
      seed: A size 2 tuple containing seed and operation seed. See
        https://www.tensorflow.org/guide/random_numbers#stateless_rngs.

    Returns:
      A tensor of shape `(n, 3)` containing the particle locations.
    """
    random_locs = tf.random.stateless_uniform((n, 3), seed, dtype=LPT_FLOAT)
    min_pt = tf.stack(self.bounding_box[0])
    max_pt = tf.stack(self.bounding_box[1])
    return tf.cast(random_locs * (max_pt - min_pt) + min_pt, LPT_FLOAT)


def injector_factory(
    injector_params: lpt_pb2.LagrangianParticleTracking.LptInjector,
) -> LptInjector:
  """Creates an injector region from the proto definition.

  Args:
    injector_params: The LptInjector proto containing the parameters for this
      injector region.

  Returns:
    An LptInjector object.
  """
  injector_type = injector_params.WhichOneof("injector_type")

  # Determining injection geometry.
  if injector_type == "box":
    return Box(injector_params)
  else:
    raise NotImplementedError(
        f"Injector region type not supported: {injector_type}."
    )
