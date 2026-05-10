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

"""Tests for lpt and lpt_comm modules for 4 TPU cores."""

from absl.testing import absltest
from absl.testing import parameterized
from absl import logging
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
from swirl_lm.utility import tpu_util
from swirl_lm.base import driver_tpu
from swirl_lm.utility import text_util
from swirl_lm.base import initializer
from typing import Tuple, Any, Optional, Sequence, TypeAlias, TypeVar, Union

import functools
import time

Array: TypeAlias = Any
PerReplica: TypeAlias = tf.types.experimental.distributed.PerReplica

U_MAG = 1.0
V_MAG = 2.0
W_MAG = 3.0
RHO_MAG = 5.0

COMPUTATIONAL_SHAPE = (2, 2, 1)
NMAX = 10

def create_constant_velocity_states(
    w_const: float = 1.0,
    u_const: float = 2.0,
    v_const: float = 3.0,
    rho_const: float = RHO_MAG,
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

def create_basic_params(
    c_d: float = 0.1,
    tau_p: float = 0.01,
    density_p : float = 10.0,
    density_f : float = 1.0,
    nu_f : float =  1e-4,
    two_way: bool = False,
    grid_sizes: Tuple[int, int, int] = (16, 16, 16),
    computational_shape: Tuple[int, int, int] = (1, 1, 1),
    n_max : int = 100
) -> parameters_lib.SwirlLMParameters:
  """Creates basic SwirlLMParameters for testing. The domain size and grid size are equal"""
  halo_width = 2
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
          dim_0: {(grid_sizes[0]-halo_width*2)*computational_shape[0] - 1.0}
          dim_1: {(grid_sizes[1]-halo_width*2)*computational_shape[1] - 1.0}
          dim_2: {(grid_sizes[2]-halo_width*2)*computational_shape[2] - 1.0}
        }}
        grid_size {{
          dim_0: {grid_sizes[0]}
          dim_1: {grid_sizes[1]}
          dim_2: {grid_sizes[2]}
        }}
        halo_width: {halo_width}
        dt: 0.001
        kernel_size: 16
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

def get_stored_particle_loc(replica_id : tf.Tensor) -> tf.Tensor:
  """Returns a list of particle locations stored in the replica"""

  # replica -> coordinate index
  #array([[0, 0, 0],
      #  [0, 1, 0],
      #  [1, 0, 0],
      #  [1, 1, 0]], dtype=int32)

  condition = tf.math.reduce_any( \
    tf.stack(
        [
        tf.math.equal(replica_id, tf.constant(0, dtype=tf.int32)),
        tf.math.equal(replica_id, tf.constant(1, dtype=tf.int32)),
        tf.math.equal(replica_id, tf.constant(2, dtype=tf.int32)),
        tf.math.equal(replica_id, tf.constant(3, dtype=tf.int32))
        ]
        ,
        axis = 0
    )
  )

  def case_true():
    if replica_id == tf.constant(0, dtype=tf.int32):
      # stores particle in replica coordinate x, y, z = 0, 1, 0
      return tf.constant(
        [[0.0, 20.0, 150.0]],
        dtype=tf.float32,
      )
    elif replica_id == tf.constant(1, dtype=tf.int32):
      # stores particle in replica coordinate x, y, z = 1, 0, 0
      return tf.constant(
        [[0.0, 130.0, 50.0]],
        dtype=tf.float32,
      )

    elif replica_id == tf.constant(2, dtype=tf.int32):
      # stores particle in replica coordinate x, y, z = 1, 1, 0
      return tf.constant(
        [[0.0, 140.0, 180.0]],
        dtype=tf.float32,
      )

    else:
      # stores particle in replica coordinate x, y, z = 0, 0, 0
      return tf.constant(
        [[0.0, 30.0, 100.0]],
        dtype=tf.float32,
      )

  def case_false():
    tf.Assert(False, ["Unexpected Replica id"])
    return  tf.constant(
            [[0.0, 30.0, 100.0]],
            dtype=tf.float32,
        )

  try:
    if tf.executing_eagerly():
      print("warning executing eagerly")

    val =  tf.cond(condition, case_true, case_false)

    return val
  except tf.errors.InvalidArgumentError as e:
    print(f"Unknown Replica ID Error : {e}")

def get_physical_particle_index_and_velocity(logical_coordinate : tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """ Returns the location of the particle physically in the replica"""

  condition = tf.math.reduce_any( \
    tf.stack(
      [
        tf.reduce_all(logical_coordinate == tf.constant([0,1,0],dtype = tf.int32)),
        tf.reduce_all(logical_coordinate == tf.constant([1,0,0],dtype = tf.int32)),
        tf.reduce_all(logical_coordinate == tf.constant([1,1,0],dtype = tf.int32)),
        tf.reduce_all(logical_coordinate == tf.constant([0,0,0],dtype = tf.int32))
      ]
      ,
      axis = 0
    )
  )

  def case_true():
    if tf.reduce_all(logical_coordinate == tf.constant([0,1,0],dtype = tf.int32)):
      # gives particle located in replica coordinate x, y, z = 0, 1, 0
      return tf.constant(
            [[2, 22, 28]], # physical location = [[0.0, 20.0, 150.0]],
            dtype=tf.int32,
        ), tf.constant(0.0, dtype = tf.float32)
    elif tf.reduce_all(logical_coordinate == tf.constant([1,0,0],dtype = tf.int32)):
      #  gives particle located in replica coordinate x, y, 1 = 1, 0, 0
      return tf.constant(
            [[2, 8, 52]], # physical location = [[0.0, 130.0, 50.0]],
            dtype=tf.int32,
        ), tf.constant(1.0, dtype = tf.float32)

    elif tf.reduce_all(logical_coordinate == tf.constant([1,1, 0],dtype = tf.int32)):
      # gives particle located in replica coordinate x, y, 1 = 1, 1, 0
      return tf.constant(
            [[2, 18, 58]], # physical location = [[0.0, 140.0, 180.0]],
            dtype=tf.int32,
        ), tf.constant(2.0, dtype = tf.float32)

    else:
      #  gives particle located in replica coordinate x, y, z = 0, 0, 0
      return tf.constant(
            [[2, 32, 102]], # same as physical location
            dtype=tf.int32,
        ), tf.constant(3.0, dtype = tf.float32)

  def case_false():
    tf.Assert(False, ["Unexpected Logical Coordinate"])
    return  tf.constant(
            [[2, 32, 102]],
            dtype=tf.int32,
        ), tf.constant(3.0, dtype = tf.float32)

  try:
    val = tf.cond(condition, case_true, case_false)

    return val

  except tf.errors.InvalidArgumentError as e:
    print(f"Unknown Logical Coordinate Error : {e}")

def get_expected_carrier_index_and_forces(logical_coordinate : tf.Tensor,
                                          n_max : tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

  """gets the expected carrier index and forces for the 4 replica test. """
  particle_loc, particle_vel = get_physical_particle_index_and_velocity(
    logical_coordinate=logical_coordinate
  )
  cd_expected = 1.0
  # del particle_locs
  # vel_diff = fluid_vels - particle_vels
  # rho_expanded = tf.expand_dims(fluid_densities, 1)
  # force = self.drag_coefficient * rho_expanded * vel_diff
  vel_diff = tf.stack([W_MAG - particle_vel,
                       U_MAG- particle_vel,
                       V_MAG - particle_vel], axis = 0)
  vel_diff = tf.reshape(vel_diff, shape = (1,3))
  nonzer_force_replica = cd_expected*RHO_MAG*vel_diff


  condition = tf.math.reduce_any( \
    tf.stack(
      [
        tf.reduce_all(logical_coordinate == tf.constant([0,1,0],dtype = tf.int32)),
        tf.reduce_all(logical_coordinate == tf.constant([1,0,0],dtype = tf.int32)),
        tf.reduce_all(logical_coordinate == tf.constant([1,1,0],dtype = tf.int32)),
        tf.reduce_all(logical_coordinate == tf.constant([0,0,0],dtype = tf.int32))
      ]
      ,
      axis = 0
    )
  )

  def case_true():
    # before the update
    if tf.reduce_all(logical_coordinate == tf.constant([0,0,0],dtype = tf.int32)):
      # gives carrier indices and forces expected for 0, 0, 0
      # in this case because the inactive particles were initialized with 0, 0, 0
      # all forces from inactive particles will be on this TPU
      # based on the order the swaps are performed,should see 297 elements with
      # the 144th-151st indices being associated withe the particles in the TPU
      # the 0,0,0 positions have shifted coordinates because of the halos
      # exchange goes [0 -> 3, 1 -> 0, 2 -> 1, 3 -> 2]
      carrier_expected = tf.concat(
          [
            tf.zeros((1,3), dtype=tf.int32),
            tf.concat(
                [
                    tf.stack(
                    [
                        tf.constant([2,2,2], dtype= tf.int32) + tf.constant([p,q,l])
                        for p, q, l in itertools.product(range(2), range(2), range(2))
                    ]
                    )
                    for i in range( (n_max - 1)*2 )
                ]
            , axis = 0
            ),
            tf.stack(
            [
                tf.cast(particle_loc[0], dtype=tf.int32) + tf.constant([p,q,l])
                for p, q, l in itertools.product(range(2), range(2), range(2))
            ]
            ),

            tf.concat(
                [
                    tf.stack(
                    [
                        tf.constant([2,2,2], dtype= tf.int32) +tf.constant([p,q,l])
                        for p, q, l in itertools.product(range(2), range(2), range(2))
                    ]
                    )
                    for i in range( (n_max - 1)*2 )
                ]
            , axis = 0
            ),
          ],
          axis = 0
        )

      forces_expected = tf.concat(
          [
            tf.constant([[0.        , 0.      , 0.]],dtype = tf.float32),
            tf.zeros((144, 3), dtype=tf.float32),
            nonzer_force_replica,
            tf.zeros((151, 3), dtype=tf.float32)
          ],
          axis=0,
      )



      return carrier_expected, forces_expected
    else:
      carrier_expected = tf.concat(
          [
            tf.zeros((1,3), dtype=tf.int32),
            tf.stack(
            [
                tf.cast(particle_loc[0], dtype=tf.int32) + tf.constant([p,q,l])
                for p, q, l in itertools.product(range(2), range(2), range(2))
            ]
            )
          ],
          axis = 0
        )

      forces_expected = tf.concat(
          [
            tf.constant([[0.        , 0.      , 0.]],dtype = tf.float32),
            nonzer_force_replica,
            tf.zeros((7, 3), dtype=tf.float32)
          ],
          axis=0,
      )

      return carrier_expected, forces_expected

  def case_false():
    tf.Assert(False, ["Unexpected Logical Coordinate"])
    carrier_expected = tf.concat(
          [
            tf.zeros((1,3), dtype=tf.int32),
            tf.stack(
            [
                tf.cast(particle_loc, dtype=tf.int32) + tf.constant([p,q,l])
                for p, q, l in itertools.product(range(2), range(2), range(2))
            ]
            )
          ],
          axis = 0
        )

    forces_expected = tf.concat(
        [
          tf.constant([[0.        , 0.      , 0.]],dtype = tf.float32),
          nonzer_force_replica,
          tf.zeros((7, 3), dtype=tf.float32)
        ],
        axis=0,
    )
    return  carrier_expected, forces_expected

  try:
    val = tf.cond(condition, case_true, case_false)

    return val

  except tf.errors.InvalidArgumentError as e:
    print(f"Unknown Logical Coordinate Error : {e}")

def state_fun_for_testing(
    replica_id: tf.Tensor,
    coordinates: initializer.ThreeIntTuple,
    params: parameters_lib.SwirlLMParameters,
) -> types.FlowFieldMap:
  """Initialize the u, v, w, .
  Args:
    replica_id: The ID number of the replica.
    coordinates: A tuple that specifies the replica's grid coordinates in
      physical space.

  Returns:
    A dictionary of states and values that are stored as string and 3D
    tensor pairs.
  """

  output = {'replica_id': replica_id}
  nx = params.nx
  ny = params.ny
  nz = params.nz

  output.update(
    create_constant_velocity_states(
        w_const=W_MAG,
        u_const=U_MAG,
        v_const=V_MAG,
        rho_const=RHO_MAG,
        grid_shape=(nz, nx, ny),
    )
  )

  # initialize the lpt float fields
  # location of the particle is different from where it is physically stored
  particle_locs = get_stored_particle_loc(replica_id)

  # particle velocity is equal to the index of the replica it is physically stored on
  replica_id_vel = tf.cast(replica_id, dtype=tf.float32)
  particle_vels = tf.expand_dims(
      tf.stack([replica_id_vel, replica_id_vel, replica_id_vel])
      ,
      axis = 0
  )

  particle_masses = tf.constant([0.01], dtype=tf.float32)

  locs = tf.concat(
      [particle_locs, tf.zeros((params.lpt.n_max - 1, 3), dtype=tf.float32)],
      axis=0,
  )
  vels = tf.concat(
      [particle_vels, tf.zeros((params.lpt.n_max - 1, 3), dtype=tf.float32)],
      axis=0,
  )
  masses = tf.concat(
      [particle_masses, tf.zeros((params.lpt.n_max- 1,), dtype=tf.float32)],
      axis=0,
  )
  lpt_floats = tf.concat([locs, vels, masses[:, tf.newaxis]], axis=1)


  # initialize the lpt int fields
  particle_active = tf.constant([1], dtype=tf.int32)

  active = tf.concat(
      [particle_active, tf.zeros((params.lpt.n_max - 1,), dtype=tf.int32)],
      axis=0,
  )
  particle_id = tf.stack([replica_id])
  ids = tf.concat(
      [particle_id, tf.zeros((params.lpt.n_max - 1,), dtype=tf.int32)],
      axis=0,
  )
  lpt_ints = tf.stack([active, ids], axis=1)

  # initialize the particle counter
  lpt_counter = tf.cast(1, tf.int32)

  # update the lpt fields
  output.update(
    {
       lpt_types.LPT_INTS_KEY:lpt_ints,
        lpt_types.LPT_FLOATS_KEY:lpt_floats,
        lpt_types.LPT_COUNTER_KEY:lpt_counter,
        lpt_types.LPT_FORCE_U_KEY: np.zeros((nz, nx, ny), dtype=np.float32),
        lpt_types.LPT_FORCE_V_KEY: np.zeros((nz, nx, ny), dtype=np.float32),
        lpt_types.LPT_FORCE_W_KEY: np.zeros((nz, nx, ny), dtype=np.float32),
    }
  )

  return output

def get_test_state(
    state_fun: types.InitFn,
    strategy: tf.distribute.TPUStrategy,
    params: parameters_lib.SwirlLMParameters,
    logical_coordinates: Array,
) -> dict[str, PerReplica]:
  """Creates the initial state using `state_fun`."""
  t_start = time.time()

  # Wrapping `init_fn` with tf.function so it is not retraced unnecessarily for
  # every core/device.
  state = driver_tpu.distribute_values(
      strategy,
      value_fn=tf.function(state_fun),
      logical_coordinates=logical_coordinates,
  )

  # Accessing the values in state to synchronize the client so the main thread
  # will wait here until the `state` is initialized and all remote operations
  # are done.
  replica_values = state['replica_id'].values
  logging.info('State initialized. Replicas are : %s', str(replica_values))
  t_post_init = time.time()
  logging.info(
      'Initialization stage took %s.',
      text_util.seconds_to_string(t_post_init - t_start),
  )

  return state

def initialize_tpu(params):
  """initializes the local tpu

  Args:
    params: an instance of lpt params


  Returns:
    A tuple containing
      tpu_strategy,
      the logical coordinates : A tuple that specifies the replica's grid
        coordinates in physical space.
      logical replicas : a 3D tensor of with shape equal to the computational
        shape that contains the replica numbers
  """
  computation_shape = np.array([params.cx, params.cy, params.cz])

  try:
    # TPU detection.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
  except ValueError:
    raise BaseException('ERROR: Not connected to a TPU runtime.')

  tf.config.experimental_connect_to_cluster(tpu)
  topology = tf.tpu.experimental.initialize_tpu_system(tpu)
  device_assignment, _ = tpu_util.tpu_device_assignment(
      computation_shape=computation_shape, tpu_topology=topology
  )
  tpu_strategy = tf.distribute.experimental.TPUStrategy(
      tpu, device_assignment=device_assignment
  )
  logical_coordinates = tpu_util.grid_coordinates(computation_shape).tolist()

  print('All devices: ', tf.config.list_logical_devices('TPU'))

  logical_replicas = np.arange(
      tpu_strategy.num_replicas_in_sync, dtype=np.int32
  ).reshape(computation_shape)


  return tpu_strategy, logical_coordinates, logical_replicas

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

class OneShuffleFluidDataAndTwoWayForcesTestMulticore(TensorflowTestCase):
  """Tests for one_shuffle_fluid_data_and_two_way_forces function for multiple replica. """

  def setUp(self):
    super().setUp()
    self.c_d = 1.0
    self.density_p = 1.1
    self.density_f = RHO_MAG
    self.nu_f = 2.0

    self.grid_shape = (128, 128, 6) # x, y, z order
    self.nx, self.ny, self.nz = self.grid_shape
    self.n_max = NMAX

    self.variables = ['w', 'u', 'v', 'rho']

    self.params_tau_neg = create_basic_params(
      c_d=self.c_d,
      tau_p=-1.0,
      density_p=self.density_p,
      density_f = self.density_f,
      nu_f = self.nu_f,
      two_way = True,
      grid_sizes= self.grid_shape,
      computational_shape = COMPUTATIONAL_SHAPE,
      n_max = NMAX,
    )

    self.core_spacings = tf.convert_to_tensor(
      (
          self.params_tau_neg.lz / self.params_tau_neg.cz,
          self.params_tau_neg.lx / self.params_tau_neg.cx,
          self.params_tau_neg.ly / self.params_tau_neg.cy,
      ),
      lpt_types.LPT_FLOAT,
    )

    self.global_min_pt = tf.convert_to_tensor(
        (
            self.params_tau_neg.z[0],
            self.params_tau_neg.x[0],
            self.params_tau_neg.y[0],
        ),
        lpt_types.LPT_FLOAT,
    )

    self.grid_spacings_zxy = tf.convert_to_tensor((
        self.params_tau_neg.grid_spacings[2],
        self.params_tau_neg.grid_spacings[0],
        self.params_tau_neg.grid_spacings[1],
    ))

  def test_spacings(self):
    """Tests if the input parameters are consistent with what future tests exect. """
    self.assertAllClose(self.grid_spacings_zxy, tf.constant([1.0, 1.0, 1.0], dtype = tf.float32))
    self.assertAllClose(self.global_min_pt, tf.constant([0.0, 0.0, 0.0], dtype = tf.float32))
    self.assertAllClose(self.core_spacings, tf.constant([1.0, 123.5, 123.5], dtype = tf.float32))

  def test_one_shuffle_2wc(self):
    """Tests one_shuffle_fluid_data_and_two_way_forces with 4 replicas, checking
       if the fluid data, force, locations of the force, and shapes of the ouputs
       match the expected output.
    """
    tpu_strategy, logical_coordinates, replicas = initialize_tpu(self.params_tau_neg)

    # generates the state on each replica
    state = get_test_state(
        state_fun=functools.partial(
            state_fun_for_testing, params=self.params_tau_neg
        ),
        strategy=tpu_strategy,
        params=self.params_tau_neg,
        logical_coordinates=logical_coordinates,
    )

    def one_shuffle_2wc_runner(state):
      lpt_field_ints = state[lpt_types.LPT_INTS_KEY]
      active = tf.cast(lpt_field_ints[:, 0], dtype=lpt_types.LPT_FLOAT)

      lpt_field_floats = state[lpt_types.LPT_FLOATS_KEY]
      locs = lpt_field_floats[:, :3]
      vels = lpt_field_floats[:, 3:6]
      masses = lpt_field_floats[:, 6]

      grids_local_x = self.params_tau_neg.grid_local(state['replica_id'],
                                         replicas,
                                         0,
                                         include_halo=True)
      grids_local_y = self.params_tau_neg.grid_local(state['replica_id'],
                                         replicas,
                                         1,
                                         include_halo=True)
      grids_local_z = self.params_tau_neg.grid_local(state['replica_id'],
                                         replicas,
                                         2,
                                         include_halo=True)



      min_physical_loc = tf.convert_to_tensor(
        [
          self.params_tau_neg.grid_local(state['replica_id'],
                                         replicas,
                                         dim,
                                         include_halo=True)[0]
          for dim in (2, 0, 1)
        ],
        tf.float32
      )

      force_fn = SimpleForceFunction(drag_coefficient=self.c_d)

      fluid_data, carrier_indices, forces = (
          lpt_comm.one_shuffle_fluid_data_and_two_way_forces(
              locs=locs,
              vels=vels,
              masses=masses,
              active=active,
              force_fn=force_fn,
              states=state,
              replica_id=state['replica_id'],
              replicas=replicas,
              variables=self.variables,
              grid_spacings=self.grid_spacings_zxy,
              core_spacings=self.core_spacings,
              local_min_pt=min_physical_loc,
              global_min_pt=self.global_min_pt,
              n_max=self.n_max,
          )
      )

      return {"fluid_data":fluid_data,
              "carrier_indices":carrier_indices,
              "forces":forces,
              "grids_local_x":grids_local_x,
              "grids_local_y":grids_local_y,
              "grids_local_z":grids_local_z,
              "min_physical_loc":min_physical_loc}

    perReplicaOuput = tpu_strategy.run(tf.function(one_shuffle_2wc_runner), args=(state,))

    # perReplicaOuput[key].values[replica_index]

    fluid_data_rep = perReplicaOuput["fluid_data"].values # w, u, v, rho data
    carrier_indices_rep = perReplicaOuput["carrier_indices"].values # indicies
    forces_rep = perReplicaOuput["forces"].values # forces

    # Verify for each replica
    for i_rep in range(len(fluid_data_rep)):
      self.assertEqual(fluid_data_rep[i_rep].shape, (self.n_max, len(self.variables)))

      # Verify the values in the outputs
      # Verify fluid data
      self.assertAllClose(fluid_data_rep[i_rep][:, 0], tf.constant(W_MAG, dtype = tf.float32))
      self.assertAllClose(fluid_data_rep[i_rep][:, 1], tf.constant(U_MAG, dtype = tf.float32))
      self.assertAllClose(fluid_data_rep[i_rep][:, 2], tf.constant(V_MAG, dtype = tf.float32))
      self.assertAllClose(fluid_data_rep[i_rep][:, 3], tf.constant(RHO_MAG, dtype = tf.float32))

      # Verify force and carrier index data
      carrier_indices_expected, forces_expected = \
          get_expected_carrier_index_and_forces(
            logical_coordinate=logical_coordinates[i_rep],
            n_max=self.n_max)

      self.assertEqual(carrier_indices_rep[i_rep].shape, tf.shape(carrier_indices_expected))
      self.assertEqual(forces_rep[i_rep].shape, tf.shape(forces_expected))


      self.assertAllEqual(carrier_indices_rep[i_rep], carrier_indices_expected)
      self.assertAllClose(forces_rep[i_rep], forces_expected)

if __name__ == '__main__':
  absltest.main()
