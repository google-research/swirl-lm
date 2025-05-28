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

"""Lagrangian particle tracking models."""

import abc
from typing import TypeAlias
import numpy as np
import six
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.numerics import time_integration
from swirl_lm.physics import constants
from swirl_lm.physics.lpt import injector
from swirl_lm.physics.lpt import lpt_pb2
from swirl_lm.physics.lpt import lpt_types
from swirl_lm.utility import common_ops
from swirl_lm.utility import stretched_grid_util
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldMap: TypeAlias = types.FlowFieldMap


@six.add_metaclass(abc.ABCMeta)
class LPT:
  """Methods to manage particle fields, including positions, velocities, etc.

  Particles are modeled as points in space with one-way coupling to the
  surrounding fluid. Governing equations are:

  ```
  d(x_p) / dt = v_p,
  d(v_p) / dt = -c_d / tau_p * (v_p - v_f),
  d(m_p) / dt = -omega,
  ```

  where `x_p` is particle location, `v_p` is particle velocity, `m_p` is
  particle mass.  Also, `c_d` is the drag coefficient, `tau_p` is the relaxation
  time, `omega` is the mass consumption rate, and `v_f` is the fluid velocity at
  the location of the particle. Each velocity `v` can be broken down into three
  components `v0`, `v1`, and `v2`. The parameters `c_d` and `tau_p` are
  constants and are defined in initializing this class. `omega` is provided as a
  parameter of `increment_time` and can update at run-time.  For any
  stretched-dimensions, those dimensions will be tracked in the mapped domain.

  The particles states are assumed to be stored in two 2D tensors:

    `lpt_field_ints`: (n, 2) for n total particle spaces (see below about n).
      active_particles: 1=active, 0=inactive.
      particle_ids: Global ID of the particles.

    `lpt_field_floats`: (n, 7) for n total spaces.
      particle_locations: x0, x1, x2.
      particle_velocities: v0, v1, v2.
      particle_masses: Mass of the particles.

  To define n:
    The number of particles in each replica changes as particles are generated,
    terminated, or transferred between replicas. In order to maintain statically
    sized field tensors, we pre-allocate excess space, so the value of `n` is
    be some factor larger than the initial number of particles in the replica.
    The functions in this module are responsible for managing the excess space
    as particles are added or removed.

  Here, we introduce methods to:
    - Add new particles to the field.
    - Remove particles from the field.
    - Update particle states (position, velocity, mass) by time integration.

  Attributes:
    dt: Time step size [s].
    grid_spacings_zxy: Uniform grid spacings in the z, x, y directions.
    use_stretched_grid_zxy: Whether to use stretched grid in the z, x, y
      directions.
    global_min_pt: The global minimum point of the simulation domain in order z,
      x, y.
    core_spacings: Each core's partial domain size in the z, x, y directions.
    num_replicas: The number of replicas used globally in the domain.
    c_d: Drag coefficient [-].
    tau_p: Relaxation time [s].
    mass_threshold: The mass below which a particle is considered terminated
      [kg].
    n_max: The maximum number of particles each replica can have.
    gravity_direction: The gravity direction in the z, x, y directions.
    params: The grid and simulation parameters of type `SwirlLMParameters`.
    exchange_fluid_data_fn: The function used to exchange fluid data with other
      replicas.
    injectors: The LptInjector types that define regions where particles are
      injected.

  Raises:
    ValueError if init is called but `lpt` is not set in `params`.
  """

  def __init__(self, params: parameters_lib.SwirlLMParameters):

    if params.lpt is None:
      raise ValueError("LPT init called but lpt params are None.")

    # Extracting grid parameters.
    self.dt = params.dt
    self.grid_spacings_zxy = tf.convert_to_tensor((
        params.grid_spacings[2],
        params.grid_spacings[0],
        params.grid_spacings[1],
    ))
    self.use_stretched_grid_zxy = np.array((
        params.use_stretched_grid[2],
        params.use_stretched_grid[0],
        params.use_stretched_grid[1],
    ))

    self.global_min_pt = tf.convert_to_tensor(
        (
            0.0 if params.use_stretched_grid[2] else params.z[0],
            0.0 if params.use_stretched_grid[0] else params.x[0],
            0.0 if params.use_stretched_grid[1] else params.y[0],
        ),
        lpt_types.LPT_FLOAT,
    )
    self.core_spacings = tf.convert_to_tensor(
        (
            (len(params.z) - 1.0) / params.cz
            if params.use_stretched_grid[2]
            else params.lz / params.cz,
            (len(params.x) - 1.0) / params.cx
            if params.use_stretched_grid[0]
            else params.lx / params.cx,
            (len(params.y) - 1.0) / params.cy
            if params.use_stretched_grid[1]
            else params.ly / params.cy,
        ),
        lpt_types.LPT_FLOAT,
    )
    self.num_replicas = params.num_replicas

    self.c_d = params.lpt.c_d
    self.tau_p = params.lpt.tau_p
    self.mass_threshold = params.lpt.mass_threshold
    self.n_max = params.lpt.n_max
    self.gravity_direction = np.array(
        [
            params.gravity_direction[2],
            params.gravity_direction[0],
            params.gravity_direction[1],
        ],
        np.float32,
    )
    self.params = params

    # Initializing the injectors.
    self.injectors = []
    for injector_params in params.lpt.injector:
      self.injectors.append(injector.injector_factory(injector_params))

  def _add_new_particles(
      self,
      lpt_field_ints: lpt_types.LptFieldInts,
      lpt_field_floats: lpt_types.LptFieldFloats,
      new_lpt_field_ints: lpt_types.LptFieldInts,
      new_lpt_field_floats: lpt_types.LptFieldFloats,
  ) -> tuple[lpt_types.LptFieldInts, lpt_types.LptFieldFloats]:
    """Adds particles to this particle field.

    Empty particle locations marked by `lpt_field_ints[:, 0] == 0` are filled
    with the new particles. This function does not increase
    `particles_generated_per_replica` because that value is used to calculate
    new particle IDs and should only be incremented when new particles are
    generated.

    Args:
      lpt_field_ints: An (n, 2) destination tensor of integer particle fields,
        the 2 columns are particle status (1=active, 0=inactive) and the
        particle IDs.
      lpt_field_floats: An (n, 7) destination tensor of floatint point particle
        fields, the 7 columns are particle locations (x0, x1, x2), particle
        velocities (v0, v1, v2), and particle masses.
      new_lpt_field_ints: An (n, 2) tensor of new particle integer particle
        fields to be added, the 2 columns are row active/deactive (1=active,
        0=inactive), particle IDs.
      new_lpt_field_floats: An (n, 7) tensor of floating point particle fields
        to be added, the 7 columns are particle locations (x0, x1, x2), particle
        velocities (v0, v1, v2), and particle masses.

    Returns:
      lpt_field_ints: An (n, 2) tensor of integer particle fields with the new
        particles added, the 2 columns are row active/deactive (1=active,
        0=inactive), particle IDs.
      lpt_field_floats: An (n, 7) tensor of floating point particle fields with
        the new particles added, the 7 columns are particle locations (x0, x1,
        x2), particle velocities (v0, v1, v2), and particle masses.

    Raises:
      InvalidArgumentError: No remaining space in particle field arrays. Add
        more space by increasing the "extension" argument.
    """
    particle_status = lpt_field_ints[:, 0]
    n_particles = tf.math.reduce_sum(particle_status)
    n_new_particles = tf.shape(new_lpt_field_ints)[0]
    available_spots = tf.shape(particle_status)[0] - n_particles
    check_op = tf.debugging.assert_less_equal(
        n_new_particles,
        available_spots,
        f"Not enough space to add new particles. Currently have {n_particles}"
        f" particles with {available_spots} available spots, but seeking to"
        f" add {n_new_particles} new particles. Please increase the extension"
        " argument to pre-allocate more space for new particles.",
    )

    with tf.control_dependencies([check_op]):
      # Determining free locations in the field tensors.
      free_locations = tf.where(particle_status == 0)
      free_locations = free_locations[:n_new_particles]

      # Insert each row of new particle fields into the available rows in the
      # particle field arrays.
      lpt_field_ints = tf.tensor_scatter_nd_update(
          lpt_field_ints, free_locations, new_lpt_field_ints
      )
      lpt_field_floats = tf.tensor_scatter_nd_update(
          lpt_field_floats, free_locations, new_lpt_field_floats
      )

      return lpt_field_ints, lpt_field_floats

  def _remove_particles(
      self,
      lpt_field_ints: lpt_types.LptFieldInts,
      lpt_field_floats: lpt_types.LptFieldFloats,
      particle_replicas: tf.Tensor,
  ) -> lpt_types.LptFieldInts:
    """Removes particles from the particle field.

    Particles are removed if they are vanishing (mass below threshold) or
    exiting the domain.  This function does not update
    `particles_generated_per_replica` because that value is used to calculate
    new globally unique particle IDs and should never be decremented, even upon
    particle termination or exit from the domain. For the below definitions,
    x0 is z, x1 is x, and x2 is y. The same for v0, v1, and v2.

    Args:
      lpt_field_ints: Integer particle parameters, the 2 columns are particle
        statuses (1=active, 0=inactive) and global particle ids.
      lpt_field_floats: Float particle parameters, the 7 columns are particle
        locations (x0, x1, x2), particle velocities (v0, v1, v2), and particle
        masses.
      particle_replicas: A tensor of length `n` containing the replica ID of
        each particle.

    Returns:
      Integer and float particle parameters updated to exclude the removed
        particles.
    """
    masses = lpt_field_floats[:, lpt_types.COL_MASS]
    particles_terminating = tf.reshape(
        tf.where(
            tf.logical_or(particle_replicas == -1, masses < self.mass_threshold)
        ),
        [-1],
    )
    indices = tf.stack(
        [particles_terminating, tf.zeros_like(particles_terminating)], axis=1
    )
    return tf.tensor_scatter_nd_update(
        lpt_field_ints,
        indices,
        tf.zeros_like(particles_terminating, dtype=lpt_types.LPT_INT),
    )

  def increment_time(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      additional_states: FlowFieldMap,
      fluid_speeds: tf.Tensor,
      omegas: tf.Tensor,
  ) -> tuple[lpt_types.LptFieldInts, lpt_types.LptFieldFloats]:
    """Updates the particles states through time integration.

    Args:
      replica_id: The ID of the replica that is calling this function.
      replicas: A 3D numpy array containing the replica ID with the global
        domain.
      additional_states: A dictionary of additional particle and fluid states.
        This includes the Lagrangian particle (lpt) states.
      fluid_speeds: A tensor (n, 3) of fluid speeds at the n particle locations
        [m/s].
      omegas: Mass consumption rates for each particle [kg/s].

    Returns:
      lpt_field_ints: Integer particle parameters, the 2 columns are particle
        statuses (1=active, 0=inactive), particle ids (global).
      lpt_field_floats: Float particle parameters, the 7 columns are particle
        locations (x0, x1, x2), particle velocities (v0, v1, v2), particle
        masses.
    """
    lpt_field_ints = additional_states[lpt_types.LPT_INTS_KEY]
    lpt_field_floats = additional_states[lpt_types.LPT_FLOATS_KEY]

    local_min_loc = self._get_local_min_loc(replicas, replica_id)

    def particle_evolution(part_locs, part_vels, part_masses):
      del part_locs, part_masses
      # In a stretched grid, dxdt becomes mapped coordinates, while dvdt and
      # dmdt remain in physical domain.
      if np.any(self.use_stretched_grid_zxy):
        # For any non-stretched dimensions, the returned value for
        # `grid_spacings`` is 1.0 because the dxdt equation remains in physical
        # domain for those dimensions.
        with tf.name_scope("lpt_getting_stretched_grid_spacings"):
          grid_spacings = self._get_grid_spacings(
              additional_states, local_min_loc
          )
        dxdt = part_vels / grid_spacings
      else:
        dxdt = part_vels
      dvdt = (
          self.c_d / self.tau_p * (fluid_speeds - part_vels)
          + tf.constant(self.gravity_direction) * constants.G
      )
      dmdt = -omegas
      return (dxdt, dvdt, dmdt)

    locs = lpt_field_floats[:, 0:3]
    vels = lpt_field_floats[:, 3:6]
    masses = lpt_field_floats[:, 6]

    locs, vels, masses = time_integration.time_advancement_explicit(
        particle_evolution,
        self.dt,
        time_integration.TimeIntegrationScheme.TIME_SCHEME_RK3,
        (locs, vels, masses),
        (locs, vels, masses),
    )

    lpt_field_floats = tf.concat([locs, vels, masses[:, tf.newaxis]], axis=1)

    return lpt_field_ints, lpt_field_floats

  def _get_local_min_loc(
      self, replicas: np.ndarray, replica_id: tf.Tensor
  ) -> tf.Tensor:
    """Returns the local minimum location for each replica including halos.

    This function returns the minimum location for each replica including halos
    in the physical domain or mapped domain if the stretched grid is used.
    The minimum location is calculated by subtracting the halo width from the
    minimum mapped location.

    Args:
      replicas: A 3D numpy array containing the replica ID with the global
        domain.
      replica_id: The ID of the replica that is calling this function.

    Returns:
      A tensor of shape (3,) containing the minimum location for each replica
      including halos.
    """
    min_mapped_loc = tf.cast(
        tf.convert_to_tensor(
            common_ops.get_core_coordinate(replicas, replica_id)
        )
        * tf.constant(
            (self.params.core_nx, self.params.core_ny, self.params.core_nz)
        )
        - self.params.halo_width,
        lpt_types.LPT_FLOAT,
    )

    min_physical_loc = [
        self.params.grid_local(replica_id, replicas, dim, include_halo=True)[0]
        for dim in (0, 1, 2)
    ]

    return tf.stack(
        [
            min_mapped_loc[dim]
            if self.params.use_stretched_grid[dim]
            else min_physical_loc[dim]
            for dim in (2, 0, 1)
        ],
        axis=0,
    )

  def _get_grid_spacings(
      self, additional_states: FlowFieldMap, local_min_loc: tf.Tensor
  ) -> tf.Tensor:
    """Returns the grid spacings at the particle locations.

    Args:
      additional_states: A dictionary of additional particle and fluid states.
        This includes the Lagrangian particle (lpt) states.
      local_min_loc: A tensor of shape (3,) containing the minimum location for
        each replica including halos. This should be the minimum location in
        mapped coordinates if the stretched grid is used.

    Returns:
      A tensor of shape (n, 3) containing the grid spacings at the particle
      locations.
    """
    lpt_field_floats = additional_states[lpt_types.LPT_FLOATS_KEY]
    part_locs = lpt_field_floats[:, :3]

    part_grid_sizes = tf.zeros((self.n_max, 3))
    for dim in (0, 1, 2):
      dim_xyz = (dim + 2) % 3  # Is zxy for lpt by default.
      if not self.use_stretched_grid_zxy[dim]:
        part_loc_grid_spacings = tf.ones((self.n_max,), lpt_types.LPT_FLOAT)
      else:
        part_dim_locs = part_locs[:, dim]
        # In a stretched grid, particle locations are tracked in a mapped grid.
        # Therefore, the rounded value of their locations correspond to the
        # index in the grid spacings array.
        indices = (
            tf.cast(tf.round(part_dim_locs), lpt_types.LPT_INT)
            - tf.cast(local_min_loc[dim_xyz], lpt_types.LPT_INT)
            + self.params.halo_width
        )
        # Particles which have escaped the domain and are marked as deactive
        # may still pass through this function. This clip ensures that there is
        # no out of bounds indexing.
        dim_spacings = tf.reshape(
            additional_states[stretched_grid_util.h_key(dim_xyz)], [-1]
        )
        indices = tf.clip_by_value(
            indices, tf.zeros_like(indices), tf.shape(dim_spacings) - 1
        )
        part_loc_grid_spacings = tf.gather(dim_spacings, indices)

      part_grid_sizes = tf.tensor_scatter_nd_update(
          part_grid_sizes,
          tf.stack(
              [tf.range(self.n_max), tf.fill((self.n_max,), dim)],
              axis=1,
          ),
          part_loc_grid_spacings,
      )

    return part_grid_sizes

  def step(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      step_id: tf.Tensor,
  ) -> FlowFieldMap:
    """Updates the particle trajectories and locations.

    Args:
      replica_id: The ID of the replica that is calling this function.
      replicas: A 3D numpy array containing the replica ID with the global
        domain.
      states: The fluid state fields. See `swirl_lm.base.parameters` for more
        details.
      additional_states: A dictionary of additional particle and fluid states.
        This includes the Lagrangian particle (lpt) statuses, global IDs,
        locations, velocities, masses, and a replica particle generation
        counter.
      step_id: The current time step.

    Returns:
      A FlowFieldMap containing the updated particle states.
    """

    # Injecting new particles using user-defined injectors.
    with tf.name_scope("particle_injection"):
      lpt_states = self.inject_particles(
          replica_id, replicas, states, additional_states, step_id, self.params
      )
      additional_states = dict(additional_states)
      additional_states.update(lpt_states)

    with tf.name_scope("update_particles"):
      return self.update_particles(
          replica_id, replicas, states, additional_states
      )

  def inject_particles(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
      step_id: tf.Tensor,
      params: parameters_lib.SwirlLMParameters,
  ) -> FlowFieldMap:
    """Calls each injector region and injects particles into the field.

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
      A dictionary with injected particle field values.
    """
    if params.lpt is None:
      raise ValueError("LPT inject loop called but lpt params are None.")

    new_lpt_ints = tf.zeros((0, 2), lpt_types.LPT_INT)
    new_lpt_floats = tf.zeros((0, 7), lpt_types.LPT_FLOAT)
    particles_generated_per_replica = additional_states[
        lpt_types.LPT_COUNTER_KEY
    ]
    for injector_region in self.injectors:
      lpt_states = injector_region.inject(
          replica_id, replicas, states, additional_states, step_id, params
      )

      new_lpt_ints = tf.concat(
          [new_lpt_ints, lpt_states[lpt_types.LPT_INTS_KEY]], axis=0
      )
      new_lpt_floats = tf.concat(
          [new_lpt_floats, lpt_states[lpt_types.LPT_FLOATS_KEY]], axis=0
      )
      particles_generated_per_replica += lpt_states[lpt_types.LPT_COUNTER_KEY]

    # Adding injected particles to the particle field.
    lpt_field_ints = additional_states[lpt_types.LPT_INTS_KEY]
    lpt_field_floats = additional_states[lpt_types.LPT_FLOATS_KEY]
    lpt_field_ints, lpt_field_floats = self._add_new_particles(
        lpt_field_ints, lpt_field_floats, new_lpt_ints, new_lpt_floats
    )
    lpt_states = {
        lpt_types.LPT_INTS_KEY: lpt_field_ints,
        lpt_types.LPT_FLOATS_KEY: lpt_field_floats,
        lpt_types.LPT_COUNTER_KEY: particles_generated_per_replica,
    }

    return lpt_states

  @abc.abstractmethod
  def update_particles(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Moves the particles and communicates with other replicas.

    Args:
      replica_id: The ID of the replica that is calling this function.
      replicas: A 3D numpy array containing the replica ID with the global
        domain.
      states: The fluid state fields. See `swirl_lm.base.parameters` for more
        details.
      additional_states: A dictionary of additional particle and fluid states.
        This includes the Lagrangian particle (lpt) statuses, global IDs,
        locations, velocities, masses, and a replica particle generation
        counter.

    Returns:
      A FlowFieldMap containing the updated particle states.
    """
    pass


def init_fn(params: parameters_lib.SwirlLMParameters) -> types.FlowFieldMap:
  """Allocates space for the `params.lpt.n_max` particles.

  Args:
    params: The SwirlLMParameters object containing the LPT parameters.

  Returns:
    A dictionary containing zero-declared LPT fields tensors and injector
    regions.
  """
  if params.lpt is None:
    raise ValueError("LPT init called but lpt params are None.")

  n_max = params.lpt.n_max
  return {
      lpt_types.LPT_INTS_KEY: tf.zeros((n_max, 2), lpt_types.LPT_INT),
      lpt_types.LPT_FLOATS_KEY: tf.zeros((n_max, 7), lpt_types.LPT_FLOAT),
      lpt_types.LPT_COUNTER_KEY: tf.constant(0, lpt_types.LPT_INT),
  }


def required_keys(lpt_config: lpt_pb2.LagrangianParticleTracking) -> list[str]:
  """Returns the required keys for the lagrangian particle tracking library."""
  if lpt_config is None:
    return []
  else:
    return [
        lpt_types.LPT_INTS_KEY,
        lpt_types.LPT_FLOATS_KEY,
        lpt_types.LPT_COUNTER_KEY,
    ]
