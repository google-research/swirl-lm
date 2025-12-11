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

"""Tests for lpt and lpt_comm modules."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_testcase import TensorflowTestCase

import itertools
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics import constants
from swirl_lm.physics.lpt import lpt
from swirl_lm.physics.lpt import lpt_comm
from swirl_lm.physics.lpt import lpt_types
from swirl_lm.physics.lpt import field_exchange
from swirl_lm.utility import types
from typing import Tuple


def create_basic_params(
    c_d: float = 0.1,
    tau_p: float = 0.01,
    density_p : float = 10.0,
    density_f : float = 1.0,
    nu_f : float =  1e-4,
    two_way: bool = False,
    grid_sizes: Tuple[int, int, int] = (16, 16, 16),
    computational_shape: Tuple[int, int, int] = (1, 1, 1),
    n_max : int = 100,
) -> parameters_lib.SwirlLMParameters:
  """Creates basic SwirlLMParameters for testing."""
  coupling_type = (
      "TWO_WAY" if two_way else "ONE_WAY"
  )
  params_pbtxt = f"""
      solver_procedure: VARIABLE_DENSITY
      convection_scheme: CONVECTION_SCHEME_QUICK
      time_integration_scheme: TIME_SCHEME_CN_EXPLICIT_ITERATION
      grid_params {{
        computation_shape {{
          dim_0: {computational_shape[0]}
          dim_1: {computational_shape[1]}
          dim_2: {computational_shape[2]}
        }}
        length {{
          dim_0: 1.0
          dim_1: 1.0
          dim_2: 1.0
        }}
        grid_size {{
          dim_0: {grid_sizes[0]}
          dim_1: {grid_sizes[1]}
          dim_2: {grid_sizes[2]}
        }}
        halo_width: 2
        dt: 0.001
        kernel_size: 8
        periodic {{
          dim_0: true dim_1: true dim_2: true
        }}
      }}
      gravity_direction {{
        dim_0: 0. dim_1: 0. dim_2: 0.0
      }}
      pressure {{
        solver {{
          jacobi {{
            max_iterations: 10
            halo_width: 2
            omega: 0.67
          }}
        }}
      }}
      density: {density_f}
      kinematic_viscosity: {nu_f}
      use_sgs: false
      lpt {{
        c_d: {c_d}
        tau_p: {tau_p}
        mass_threshold: 1e-6
        n_max: {n_max}
        density: {density_p}
        omega_const: 0.0
        coupling: {coupling_type}
        field_exchange {{
          communication_mode: ONE_SHUFFLE
        }}
      }}
  """
  return parameters_lib.SwirlLMParameters.config_from_text_proto(params_pbtxt)

def create_constant_velocity_states(
    w_const: float = 1.0,
    u_const: float = 2.0,
    v_const: float = 3.0,
    rho_const: float = 1.2,
    grid_shape: tuple = (8, 8, 8),
) -> types.FlowFieldMap:
  """Creates a states dictionary with constant velocity fields."""
  nz, nx, ny = grid_shape
  states = {
      'w': tf.constant(
          np.full((nz, nx, ny), w_const, dtype=np.float32),
          dtype=tf.float32,
      ),
      'u': tf.constant(
          np.full((nz, nx, ny), u_const, dtype=np.float32),
          dtype=tf.float32,
      ),
      'v': tf.constant(
          np.full((nz, nx, ny), v_const, dtype=np.float32),
          dtype=tf.float32,
      ),
      'rho': tf.constant(
          np.full((nz, nx, ny), rho_const, dtype=np.float32),
          dtype=tf.float32,
      ),
  }
  return states

class SimpleForceFunction:
  """Simple force function for testing - returns (k, 3) tensor."""

  def __init__(self, drag_coefficient: float = 0.1):
    self.drag_coefficient = drag_coefficient

  def __call__(
      self,
      fluid_vels: tf.Tensor,
      fluid_densities: tf.Tensor,
      particle_locs: tf.Tensor,
      particle_vels: tf.Tensor,
      particle_masses: tf.Tensor,
  ) -> tf.Tensor:
    """Computes simple drag force: F = drag_coef * rho * (v_f - v_p).

    Args:
      fluid_vels: (k, 3) tensor of fluid velocities.
      fluid_densities: (k,) tensor of fluid densities.
      particle_locs: (k, 3) tensor of particle locations.
      particle_vels: (k, 3) tensor of particle velocities.
      particle_masses: (k,) tensor of particle masses.

    Returns:
      (k, 3) tensor of forces.
    """
    del particle_locs
    vel_diff = fluid_vels - particle_vels
    rho_expanded = tf.expand_dims(fluid_densities, 1)
    force = self.drag_coefficient * rho_expanded * vel_diff
    return force

class ParticleForcesTest(TensorflowTestCase):
  """Tests for particle_forces_function from LPT class."""

  def test_particle_forces_function_returns_callable(self):
    """Test that particle_forces_function returns a callable."""
    params = create_basic_params()
    try:
      fe_instance = field_exchange.FieldExchange(params)
    except Exception as e:
      self.skipTest(f"Could not initialize FieldExchange: {e}")
    force_fn = fe_instance.particle_forces_function()
    self.assertTrue(callable(force_fn))

  def test_particle_force_with_relative_velocity_constant_tau(self):
    """Test force calculation with relative velocity and constant tau_p."""
    params = create_basic_params(
      c_d=0.1,
      tau_p=0.01,
      density_p=1.1,
      density_f = 1.2,
      nu_f = 2.0,
      two_way=True
    )
    try:
      fe_instance = field_exchange.FieldExchange(params)
    except Exception as e:
      self.skipTest(f"Could not initialize FieldExchange: {e}")

    force_fn = fe_instance.particle_forces_function()

    # Single particle with relative velocity
    fluid_speeds = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
    fluid_densities = tf.constant([1.2], dtype=tf.float32)
    part_locs = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
    part_vels = tf.constant([[2.0, 0.0, 0.0]], dtype=tf.float32)
    part_masses = tf.constant([0.01], dtype=tf.float32)

    forces = force_fn(
        fluid_speeds, fluid_densities, part_locs, part_vels, part_masses
    )

    # Verify force shape (k, 3)
    self.assertEqual(forces.shape, (1, 3))

    # Force = mass * (c_d/tau_p * (v_fluid - v_particle) + gravity)
    # F_0_no_gravity = 0.01 * (0.1/0.01 * (1.0 - 2.0)) = 0.01 * (10 * -1) = -0.1
    # F_0 = F_0_no_gravity + 0.01 * lpt_instance.gravity_direction[0] * constants.G
    # gravity is zero
    self.assertAllClose(forces, tf.constant([-0.1, 0., 0.], tf.float32))

  def test_particle_force_zero_relative_velocity(self):
    """Test force with zero relative velocity."""
    params = create_basic_params(
      c_d=0.1,
      tau_p=0.01,
      density_p=1.1,
      density_f = 1.2,
      nu_f = 2.0,
      two_way=True
    )
    try:
      fe_instance = field_exchange.FieldExchange(params)
    except Exception as e:
      self.skipTest(f"Could not initialize FieldExchange: {e}")

    force_fn = fe_instance.particle_forces_function()

    fluid_speeds = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
    fluid_densities = tf.constant([1.2], dtype=tf.float32)
    part_locs = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
    part_vels = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
    part_masses = tf.constant([0.01], dtype=tf.float32)

    forces = force_fn(
        fluid_speeds, fluid_densities, part_locs, part_vels, part_masses
    )

    # With zero relative velocity, only gravity remains but gravity is zero
    self.assertEqual(forces.shape, (1,3))
    self.assertAllClose(forces, tf.zeros((1,3), dtype=tf.float32))

  def test_particle_force_multiple_particles(self):
    """Test force calculation with multiple particles."""
    params = create_basic_params(
      c_d=0.1,
      tau_p=0.01,
      density_p=1.1,
      density_f = 1.2,
      nu_f = 2.0,
      two_way=True
    )
    try:
      fe_instance = field_exchange.FieldExchange(params)
    except Exception as e:
      self.skipTest(f"Could not initialize FieldExchange: {e}")

    force_fn = fe_instance.particle_forces_function()

    fluid_speeds = tf.constant(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=tf.float32,
    )
    fluid_densities = tf.constant([1.2, 1.0, 1.1], dtype=tf.float32)
    part_locs = tf.constant(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        dtype=tf.float32,
    )
    part_vels = tf.constant(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=tf.float32,
    )
    part_masses = tf.constant([0.01, 0.02, 0.03], dtype=tf.float32)

    forces = force_fn(
        fluid_speeds, fluid_densities, part_locs, part_vels, part_masses
    )

    # Verify shape
    self.assertEqual(forces.shape, (3, 3))

    # Verify magnitude of forces
    forces_v = tf.constant(
      [[-0.1,  0.        ,  0.        ],
       [ 0.        , -0.2,  0.        ],
       [ 0.        ,  0.        , -0.3]],
       dtype = tf.float32
    )

    self.assertAllClose(forces, forces_v)

  def test_particle_force_variable_tau_p(self):
    """Test force calculation with variable tau_p (tau_p = -1.0)."""
    params = create_basic_params(
      c_d=0.1,
      tau_p=-1.0,
      density_p=1.1,
      density_f = 1.2,
      nu_f = 2.0
    )

    try:
      fe_instance = field_exchange.FieldExchange(params)
    except Exception as e:
      self.skipTest(f"Could not initialize FieldExchange: {e}")

    force_fn = fe_instance.particle_forces_function()

    # Single particle
    fluid_speeds = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
    fluid_densities = tf.constant([1.2], dtype=tf.float32)
    part_locs = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
    part_vels = tf.constant([[2.0, 0.0, 0.0]], dtype=tf.float32)
    part_masses = tf.constant([0.01], dtype=tf.float32)

    forces = force_fn(
        fluid_speeds, fluid_densities, part_locs, part_vels, part_masses
    )

    # Verify force shape
    self.assertEqual(forces.shape, (1, 3))

    # Verify magnitude of forces
    # gravity is zero
    forces_v = tf.constant([[-0.5857132,  0.       ,  0.       ]], dtype=tf.float32)

    self.assertAllClose(forces, forces_v)

  def test_particle_force_output_shape(self):
    """Test force output has correct shape."""
    params = create_basic_params()
    try:
      fe_instance = field_exchange.FieldExchange(params)
    except Exception as e:
      self.skipTest(f"Could not initialize FieldExchange: {e}")


    force_fn = fe_instance.particle_forces_function()

    n_particles = 10
    fluid_speeds = tf.random.normal((n_particles, 3), dtype=tf.float32)
    fluid_densities = tf.ones((n_particles,), dtype=tf.float32)
    part_locs = tf.random.normal((n_particles, 3), dtype=tf.float32)
    part_vels = tf.random.normal((n_particles, 3), dtype=tf.float32)
    part_masses = tf.ones((n_particles,), dtype=tf.float32) * 0.01

    forces = force_fn(
        fluid_speeds, fluid_densities, part_locs, part_vels, part_masses
    )

    # Forces should be (n_particles, 3)
    self.assertEqual(forces.shape, (n_particles, 3))

  # test a case where some of the particle masses are zero

class OneShuffleFluidDataAndTwoWayForcesTest(TensorflowTestCase):
  """Tests for one_shuffle_fluid_data_and_two_way_forces function for a single replica. """

  def setUp(self):
    super().setUp()
    self.grid_shape = (8, 8, 8)
    self.nz, self.nx, self.ny = self.grid_shape
    self.n_max = 10

    self.replica_id = tf.constant(0, dtype=tf.int32)
    self.replicas = np.array([[[0]]], dtype=np.int32)
    self.variables = ['w', 'u', 'v', 'rho']

    self.grid_spacings = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)
    self.core_spacings = tf.constant(
        [float(self.nz), float(self.nx), float(self.ny)],
        dtype=tf.float32,
    )
    self.local_min_pt = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    self.global_min_pt = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)

  def test_single_particle_on_fluid_point_in_constant_velocity_field(self):
    """Test single particle in constant velocity field."""
    w_val, u_val, v_val, rho_val = 1.0, 2.0, 3.0, 1.2
    states = create_constant_velocity_states(
        w_const=w_val,
        u_const=u_val,
        v_const=v_val,
        rho_const=rho_val,
        grid_shape=self.grid_shape,
    )

    particle_loc = tf.constant([[4.0, 4.0, 4.0]], dtype=tf.float32)
    particle_vel = tf.constant([[1.0, 1.0, 1.0]], dtype=tf.float32)
    particle_mass = tf.constant([0.01], dtype=tf.float32)
    particle_active = tf.constant([1.0], dtype=tf.float32)

    locs = tf.concat(
        [particle_loc, tf.zeros((self.n_max - 1, 3), dtype=tf.float32)],
        axis=0,
    )
    vels = tf.concat(
        [particle_vel, tf.zeros((self.n_max - 1, 3), dtype=tf.float32)],
        axis=0,
    )
    masses = tf.concat(
        [particle_mass, tf.zeros((self.n_max - 1,), dtype=tf.float32)],
        axis=0,
    )
    active = tf.concat(
        [particle_active, tf.zeros((self.n_max - 1,), dtype=tf.float32)],
        axis=0,
    )

    force_fn = SimpleForceFunction(drag_coefficient=0.1)

    fluid_data, carrier_indices, forces = (
        lpt_comm.one_shuffle_fluid_data_and_two_way_forces(
            locs=locs,
            vels=vels,
            masses=masses,
            active=active,
            force_fn=force_fn,
            states=states,
            replica_id=self.replica_id,
            replicas=self.replicas,
            variables=self.variables,
            grid_spacings=self.grid_spacings,
            core_spacings=self.core_spacings,
            local_min_pt=self.local_min_pt,
            global_min_pt=self.global_min_pt,
            n_max=self.n_max,
        )
    )

    # Verify shapes
    self.assertEqual(fluid_data.shape, (self.n_max, len(self.variables)))

    # carrier_indices and forces should have matching dimensions

    # before the update
    # for a single replica case, all particles (including inactive)
    # are on this replica.
    # so that means that the size of the array will be 8*nmax + 1
    self.assertEqual(carrier_indices.shape, (8*self.n_max + 1, 3))
    self.assertEqual(forces.shape, (8*self.n_max + 1, 3))

    # after the update
    # self.assertEqual(carrier_indices.shape, (8 + 1, 3))
    # self.assertEqual(forces.shape, (8 + 1, 3))


    # Verify the values in the outputs
    # Verify fluid data
    self.assertAllClose(fluid_data[:, 0], tf.constant(w_val, dtype = fluid_data.dtype))
    self.assertAllClose(fluid_data[:, 1], tf.constant(u_val, dtype = fluid_data.dtype))
    self.assertAllClose(fluid_data[:, 2], tf.constant(v_val, dtype = fluid_data.dtype))
    self.assertAllClose(fluid_data[:, 3], tf.constant(rho_val, dtype = fluid_data.dtype))

    # before the update
    # Verify force data
    carrier_indices_expected = tf.concat(
      [
        tf.zeros((1,3), dtype=tf.int32),
        tf.constant(
          [
            [4, 4, 4],
            [4, 4, 5],
            [4, 5, 4],
            [4, 5, 5],
            [5, 4, 4],
            [5, 4, 5],
            [5, 5, 4],
            [5, 5, 5]
          ],
          dtype = tf.int32
        ),
        tf.concat(
            [
                tf.stack(
                [
                    tf.constant([p,q,l])
                    for p, q, l in itertools.product(range(2), range(2), range(2))
                ]
                )
                for i in range(self.n_max - 1)
            ]
        , axis = 0
        )
      ],
      axis = 0
    )

    self.assertAllEqual(carrier_indices, carrier_indices_expected)

    forces_expected = tf.concat(
        [
          tf.constant([[0.        , 0.      , 0.]],dtype = tf.float32),
          tf.constant([[0.        , 0.12      , 0.24]],dtype = tf.float32),
          tf.zeros((8*self.n_max -1, 3), dtype=tf.float32)
        ],
        axis=0,
    )
    self.assertAllClose(forces, forces_expected)

    # After the update
    # Verify force data
    # carrier_indices_expected = tf.concat(
    #   [
    #     tf.zeros((1,3), dtype=tf.int32),
    #     tf.constant(
    #       [
    #         [4, 4, 4],
    #         [4, 4, 5],
    #         [4, 5, 4],
    #         [4, 5, 5],
    #         [5, 4, 4],
    #         [5, 4, 5],
    #         [5, 5, 4],
    #         [5, 5, 5]
    #       ],
    #       dtype = tf.int32
    #     )
    #   ],
    #   axis = 0
    # )

    # self.assertAllEqual(carrier_indices, carrier_indices_expected)

    # forces_expected = tf.concat(
    #     [
    #       tf.constant([[0.        , 0.      , 0.]],dtype = tf.float32),
    #       tf.constant([[0.        , 0.12      , 0.24]],dtype = tf.float32),
    #       tf.zeros((7, 3), dtype=tf.float32)
    #     ],
    #     axis=0,
    # )
    # self.assertAllClose(forces, forces_expected)

  # add an additional test for if the particle is slightly shifted off

  def test_single_particle_on_fluid_point_with_field_exchange_variable_tau_p(self):
    """Test with FieldExchange using variable tau_p (tau_p = -1.0)."""
    params = create_basic_params(
      c_d=0.1,
      tau_p=-1.0,
      density_p=1.1,
      density_f = 1.2,
      nu_f = 2.0,
      two_way=True
    )

    try:
      fe_instance = field_exchange.FieldExchange(params)
      force_fn = fe_instance.particle_forces_function()
    except Exception as e:
      self.skipTest(f"Could not initialize FieldExchange: {e}")

    states = create_constant_velocity_states(
        w_const=1.0,
        u_const=2.0,
        v_const=3.0,
        rho_const=1.2,
        grid_shape=self.grid_shape,
    )

    particle_loc = tf.constant([[4.0, 4.0, 4.0]], dtype=tf.float32)
    particle_vel = tf.constant([[1.0, 1.0, 1.0]], dtype=tf.float32)
    particle_mass = tf.constant([0.01], dtype=tf.float32)
    particle_active = tf.constant([1.0], dtype=tf.float32)

    locs = tf.concat(
        [particle_loc, tf.zeros((self.n_max - 1, 3), dtype=tf.float32)],
        axis=0,
    )
    vels = tf.concat(
        [particle_vel, tf.zeros((self.n_max - 1, 3), dtype=tf.float32)],
        axis=0,
    )
    masses = tf.concat(
        [particle_mass, tf.zeros((self.n_max - 1,), dtype=tf.float32)],
        axis=0,
    )
    active = tf.concat(
        [particle_active, tf.zeros((self.n_max - 1,), dtype=tf.float32)],
        axis=0,
    )

    fluid_data, carrier_indices, forces = (
        lpt_comm.one_shuffle_fluid_data_and_two_way_forces(
            locs=locs,
            vels=vels,
            masses=masses,
            active=active,
            force_fn=force_fn,
            states=states,
            replica_id=self.replica_id,
            replicas=self.replicas,
            variables=self.variables,
            grid_spacings=self.grid_spacings,
            core_spacings=self.core_spacings,
            local_min_pt=self.local_min_pt,
            global_min_pt=self.global_min_pt,
            n_max=self.n_max,
        )
    )

    # Verify shapes
    self.assertEqual(fluid_data.shape, (self.n_max, len(self.variables)))

    # carrier_indices and forces should have matching dimensions
    # before the update
    self.assertEqual(carrier_indices.shape, (8*self.n_max + 1, 3))
    self.assertEqual(forces.shape, (8*self.n_max + 1, 3))
    # after the update
    # self.assertEqual(carrier_indices.shape, (8 + 1, 3))
    # self.assertEqual(forces.shape, (8 + 1, 3))


    # Verify the values in the outputs
    # Verify fluid data
    self.assertAllClose(fluid_data[:, 0], tf.constant(1.0, dtype = fluid_data.dtype))
    self.assertAllClose(fluid_data[:, 1], tf.constant(2.0, dtype = fluid_data.dtype))
    self.assertAllClose(fluid_data[:, 2], tf.constant(3.0, dtype = fluid_data.dtype))
    self.assertAllClose(fluid_data[:, 3], tf.constant(1.2, dtype = fluid_data.dtype))

    # before the update
    # Verify force data
    carrier_indices_expected = tf.concat(
      [
        tf.zeros((1,3), dtype=tf.int32),
        tf.constant(
          [
            [4, 4, 4],
            [4, 4, 5],
            [4, 5, 4],
            [4, 5, 5],
            [5, 4, 4],
            [5, 4, 5],
            [5, 5, 4],
            [5, 5, 5]
          ],
          dtype = tf.int32
        ),
        tf.concat(
            [
                tf.stack(
                [
                    tf.constant([p,q,l])
                    for p, q, l in itertools.product(range(2), range(2), range(2))
                ]
                )
                for i in range(self.n_max - 1)
            ]
        , axis = 0
        )
      ],
      axis = 0
    )

    self.assertAllEqual(carrier_indices, carrier_indices_expected)

    forces_expected = tf.concat(
        [
          tf.constant([[0.        , 0.      , 0.]],dtype = tf.float32),
          tf.constant([[0.       , 0.5857132, 1.1714264]],dtype = tf.float32),
          tf.zeros((8*self.n_max -1, 3), dtype=tf.float32)
        ],
        axis=0,
    )
    self.assertAllClose(forces, forces_expected, atol = tf.constant(1e-5))

    # after the update
    # carrier_indices_expected = tf.concat(
    #   [
    #     tf.zeros((1,3), dtype=tf.int32),
    #     tf.constant(
    #       [
    #         [4, 4, 4],
    #         [4, 4, 5],
    #         [4, 5, 4],
    #         [4, 5, 5],
    #         [5, 4, 4],
    #         [5, 4, 5],
    #         [5, 5, 4],
    #         [5, 5, 5]
    #       ],
    #       dtype = tf.int32
    #     )
    #   ],
    #   axis = 0
    # )

    # self.assertAllEqual(carrier_indices, carrier_indices_expected)

    # forces_expected = tf.concat(
    #     [
    #       tf.constant([[0.        , 0.      , 0.]],dtype = tf.float32),
    #       tf.constant([[0.       , 0.5857132, 1.1714264]],dtype = tf.float32),
    #       tf.zeros((7, 3), dtype=tf.float32)
    #     ],
    #     axis=0,
    # )
    # self.assertAllClose(forces, forces_expected, atol = tf.constant(1e-5))

  def test_output_shapes_are_correct(self):
    """Test that output shapes are correct."""
    states = create_constant_velocity_states(grid_shape=self.grid_shape)
    force_fn = SimpleForceFunction(drag_coefficient=0.1)

    locs = tf.zeros((self.n_max, 3), dtype=tf.float32)
    vels = tf.zeros((self.n_max, 3), dtype=tf.float32)
    masses = tf.zeros((self.n_max,), dtype=tf.float32)
    active = tf.ones((self.n_max,), dtype=tf.float32)

    fluid_data, carrier_indices, forces = (
        lpt_comm.one_shuffle_fluid_data_and_two_way_forces(
            locs=locs,
            vels=vels,
            masses=masses,
            active=active,
            force_fn=force_fn,
            states=states,
            replica_id=self.replica_id,
            replicas=self.replicas,
            variables=self.variables,
            grid_spacings=self.grid_spacings,
            core_spacings=self.core_spacings,
            local_min_pt=self.local_min_pt,
            global_min_pt=self.global_min_pt,
            n_max=self.n_max,
        )
    )

    # fluid_data: (n_max, len(variables))
    self.assertEqual(fluid_data.shape, (self.n_max, len(self.variables)))

    # carrier_indices and forces should have matching dimensions
    # before the update
    self.assertEqual(carrier_indices.shape, (8*self.n_max + 1,3))
    self.assertEqual(forces.shape, (8*self.n_max + 1,3))

    # after the update
    # self.assertEqual(carrier_indices.shape, (8*self.n_max + 1,3))
    # self.assertEqual(forces.shape, (8*self.n_max + 1,3))

  def test_multiple_particles_on_fluid_point(self):
    """Test with multiple particles."""
    states = create_constant_velocity_states(
        w_const=0.5,
        u_const=1.5,
        v_const=2.5,
        rho_const=1.0,
        grid_shape=self.grid_shape,
    )

    particle_locs = tf.constant(
        [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [6.0, 6.0, 6.0]],
        dtype=tf.float32,
    )
    particle_vels = tf.constant(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
        dtype=tf.float32,
    )
    particle_masses = tf.constant([0.01, 0.02, 0.03], dtype=tf.float32)
    particle_active = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)

    locs = tf.concat(
        [particle_locs, tf.zeros((self.n_max - 3, 3), dtype=tf.float32)],
        axis=0,
    )
    vels = tf.concat(
        [particle_vels, tf.zeros((self.n_max - 3, 3), dtype=tf.float32)],
        axis=0,
    )
    masses = tf.concat(
        [particle_masses, tf.zeros((self.n_max - 3,), dtype=tf.float32)],
        axis=0,
    )
    active = tf.concat(
        [particle_active, tf.zeros((self.n_max - 3,), dtype=tf.float32)],
        axis=0,
    )

    force_fn = SimpleForceFunction(drag_coefficient=0.1)

    fluid_data, carrier_indices, forces = (
        lpt_comm.one_shuffle_fluid_data_and_two_way_forces(
            locs=locs,
            vels=vels,
            masses=masses,
            active=active,
            force_fn=force_fn,
            states=states,
            replica_id=self.replica_id,
            replicas=self.replicas,
            variables=self.variables,
            grid_spacings=self.grid_spacings,
            core_spacings=self.core_spacings,
            local_min_pt=self.local_min_pt,
            global_min_pt=self.global_min_pt,
            n_max=self.n_max,
        )
    )

    # Verify shapes
    self.assertEqual(fluid_data.shape, (self.n_max, len(self.variables)))

    # carrier_indices and forces should have matching dimensions
    # before the update
    self.assertEqual(carrier_indices.shape, (8*self.n_max + 1, 3))
    self.assertEqual(forces.shape, (8*self.n_max + 1, 3))
    # after the update
    # self.assertEqual(carrier_indices.shape, (8*3 + 1, 3))
    # self.assertEqual(forces.shape, (8*3 + 1, 3))

    # Verify the values in the outputs
    # Verify fluid data
    self.assertAllClose(fluid_data[:, 0], tf.constant(0.5, dtype = fluid_data.dtype))
    self.assertAllClose(fluid_data[:, 1], tf.constant(1.5, dtype = fluid_data.dtype))
    self.assertAllClose(fluid_data[:, 2], tf.constant(2.5, dtype = fluid_data.dtype))
    self.assertAllClose(fluid_data[:, 3], tf.constant(1.0, dtype = fluid_data.dtype))

    # before the update
    # Verify carrier index
    carrier_indices_expected = tf.concat(
      [
        tf.zeros((1,3), dtype=tf.int32),
        tf.constant(
          [
            [2, 2, 2],
            [2, 2, 3],
            [2, 3, 2],
            [2, 3, 3],
            [3, 2, 2],
            [3, 2, 3],
            [3, 3, 2],
            [3, 3, 3],
            [4, 4, 4],
            [4, 4, 5],
            [4, 5, 4],
            [4, 5, 5],
            [5, 4, 4],
            [5, 4, 5],
            [5, 5, 4],
            [5, 5, 5],
            [6, 6, 6],
            [6, 6, 7],
            [6, 7, 6],
            [6, 7, 7],
            [7, 6, 6],
            [7, 6, 7],
            [7, 7, 6],
            [7, 7, 7]
          ],
          dtype = tf.int32
        ),
        tf.concat(
            [
                tf.stack(
                [
                    tf.constant([p,q,l])
                    for p, q, l in itertools.product(range(2), range(2), range(2))
                ]
                )
                for i in range(self.n_max - 3)
            ]
        , axis = 0
        )
      ],
      axis = 0
    )
    self.assertAllEqual(carrier_indices, carrier_indices_expected)

    # verify force data
    forces_expected = tf.concat(
        [
          tf.constant([[0.        , 0.      , 0.]],dtype = tf.float32),
          tf.constant(
            [
              [-0.05,  0.05,  0.15],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [-0.15, -0.05,  0.05],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [-0.25, -0.15, -0.05],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
            ]
            ,
            dtype = tf.float32
          ),
          tf.zeros((8*(self.n_max -3), 3), dtype=tf.float32)
        ],
        axis=0,
    )
    self.assertAllClose(forces, forces_expected)

    # after the update
    # carrier_indices_expected = tf.concat(
    #   [
    #     tf.zeros((1,3), dtype=tf.int32),
    #     tf.constant(
    #       [
    #         [2, 2, 2],
    #         [2, 2, 3],
    #         [2, 3, 2],
    #         [2, 3, 3],
    #         [3, 2, 2],
    #         [3, 2, 3],
    #         [3, 3, 2],
    #         [3, 3, 3],
    #         [4, 4, 4],
    #         [4, 4, 5],
    #         [4, 5, 4],
    #         [4, 5, 5],
    #         [5, 4, 4],
    #         [5, 4, 5],
    #         [5, 5, 4],
    #         [5, 5, 5],
    #         [6, 6, 6],
    #         [6, 6, 7],
    #         [6, 7, 6],
    #         [6, 7, 7],
    #         [7, 6, 6],
    #         [7, 6, 7],
    #         [7, 7, 6],
    #         [7, 7, 7]
    #       ],
    #       dtype = tf.int32
    #     )
    #   ],
    #   axis = 0
    # )
    # self.assertAllEqual(carrier_indices, carrier_indices_expected)

    # # verify force data
    # forces_expected = tf.concat(
    #     [
    #       tf.constant([[0.        , 0.      , 0.]],dtype = tf.float32),
    #       tf.constant(
    #         [
    #           [-0.05,  0.05,  0.15],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [-0.15, -0.05,  0.05],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [-0.25, -0.15, -0.05],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #           [0., 0., 0.],
    #         ]
    #         ,
    #         dtype = tf.float32
    #       )
    #     ],
    #     axis=0,
    # )
    # self.assertAllClose(forces, forces_expected)

class SimpleForceFunctionTest(TensorflowTestCase):
  """Tests for SimpleForceFunction."""

  def test_simple_force_with_relative_velocity(self):
    """Test simple force calculation with relative velocity."""
    force_fn = SimpleForceFunction(drag_coefficient=0.1)

    fluid_vels = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
    fluid_densities = tf.constant([1.2], dtype=tf.float32)
    particle_locs = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
    particle_vels = tf.constant([[2.0, 0.0, 0.0]], dtype=tf.float32)
    particle_masses = tf.constant([0.01], dtype=tf.float32)

    forces = force_fn(
        fluid_vels,
        fluid_densities,
        particle_locs,
        particle_vels,
        particle_masses,
    )

    # Relative velocity: v_fluid - v_particle = [1, 0, 0] - [2, 0, 0] = [-1, 0, 0]
    # Force = drag_coef * rho * vel_diff = 0.1 * 1.2 * [-1, 0, 0] = [-0.12, 0, 0]
    self.assertAllClose(forces, tf.constant([[-0.12, 0., 0.]], dtype = tf.float32))

  def test_simple_force_zero_relative_velocity(self):
    """Test simple force is zero with zero relative velocity."""
    force_fn = SimpleForceFunction(drag_coefficient=0.1)

    fluid_vels = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
    fluid_densities = tf.constant([1.2], dtype=tf.float32)
    particle_locs = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
    particle_vels = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
    particle_masses = tf.constant([0.01], dtype=tf.float32)

    forces = force_fn(
        fluid_vels,
        fluid_densities,
        particle_locs,
        particle_vels,
        particle_masses,
    )

    self.assertAllClose(forces, tf.constant([[0., 0., 0.]], dtype = tf.float32))

  def test_simple_force_multiple_particles(self):
    """Test simple force with multiple particles."""
    force_fn = SimpleForceFunction(drag_coefficient=0.1)

    fluid_vels = tf.constant(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=tf.float32,
    )
    fluid_densities = tf.constant([1.2, 1.0], dtype=tf.float32)
    particle_locs = tf.constant(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=tf.float32,
    )
    particle_vels = tf.constant(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=tf.float32,
    )
    particle_masses = tf.constant([0.01, 0.01], dtype=tf.float32)

    forces = force_fn(
        fluid_vels,
        fluid_densities,
        particle_locs,
        particle_vels,
        particle_masses,
    )

    self.assertEqual(forces.shape, (2, 3))

    # Verify force function works for 2 particles
    self.assertAllClose(forces, tf.constant([[-0.12, 0., 0.], [0., -0.1, 0.]], dtype = tf.float32))

class TestProtoFile(TensorflowTestCase):
  def test_proto(self):
    """Test creation of params."""
    try:
      create_basic_params()
    except:
      self.assertTrue(False)

if __name__ == '__main__':
  absltest.main()
