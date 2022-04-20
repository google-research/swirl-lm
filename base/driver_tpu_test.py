# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for driver_tpu."""

import functools
import os

from absl import flags
from absl.testing import parameterized
import numpy as np
from swirl_lm.base import driver_tpu
import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.framework import util


class DriverTpuTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(([1, 1, 1],), ([2, 1, 1],))
  def test_initialize_tpu(self, computation_shape):
    strategy = driver_tpu.initialize_tpu(
        tpu_address='', computation_shape=computation_shape)
    num_replicas = np.prod(computation_shape)
    self.assertEqual(strategy.num_replicas_in_sync, num_replicas)

    @tf.function
    def add_fn(x, y):
      z = x + y
      return z

    x = tf.constant(3.)
    y = tf.constant(4.)
    z = strategy.run(add_fn, args=(x, y))
    self.assertEqual(
        strategy.experimental_local_results(z), tuple([7.] * num_replicas))

  def test_distributed_values(self):

    def init_fn(replica_id, coordinates):
      del coordinates
      return {'replica_id': replica_id, 'p': tf.ones([2])}

    computation_shape = np.array([2, 1, 1])
    strategy = driver_tpu.initialize_tpu(
        tpu_address='', computation_shape=computation_shape)
    logical_coordinates = util.grid_coordinates(computation_shape).tolist()
    initial_state = driver_tpu.distribute_values(strategy, init_fn,
                                                 logical_coordinates)

    local_state = strategy.experimental_local_results(initial_state)

    tf.nest.assert_same_structure(local_state, ({
        'replica_id': tf.constant(0),
        'p': tf.zeros([2])
    }, {
        'replica_id': tf.constant(1),
        'p': tf.ones([2])
    }))

  def test_distributed_read_state(self):
    # Set up strategy and the initial state.
    def init_fn(replica_id, coordinates):
      del coordinates
      return {'p': tf.cast(replica_id, tf.float32) * tf.ones([2])}

    output_dir = os.path.join(flags.FLAGS.test_tmpdir, 'output_dir')
    filename_prefix = 'test'
    computation_shape = np.array([2, 1, 1])
    strategy = driver_tpu.initialize_tpu(
        tpu_address='', computation_shape=computation_shape)
    logical_coordinates = util.grid_coordinates(computation_shape).tolist()
    state = driver_tpu.distribute_values(strategy, init_fn, logical_coordinates)

    with strategy.scope():
      step_id = tf.Variable(0)

    @tf.function
    def step_fn(state):
      return {'p': state['p'] + tf.ones_like(state['p'])}

    # Advance the 'simulation' to step 1.
    step_id.assign_add(1)
    state = strategy.run(step_fn, args=(state,))

    # Verify the state at step 1.
    self.assertEqual(step_id, 1)
    self.assertAllEqual(
        tf.nest.flatten(strategy.experimental_local_results(state)),
        (tf.ones([2]), 2 * tf.ones([2])))

    # Now, write the state at step 1.
    driver_tpu.distributed_write_state(
        strategy=strategy,
        local_state=strategy.experimental_local_results(state),
        logical_coordinates=logical_coordinates,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        step_id=step_id)

    # Advance the 'simulation' to step 2.
    step_id.assign_add(1)
    state = strategy.run(step_fn, args=(state,))

    # Verify the state at step 2.
    self.assertEqual(step_id, 2)
    self.assertAllEqual(
        tf.nest.flatten(strategy.experimental_local_results(state)),
        (2 * tf.ones([2]), 3 * tf.ones([2])))

    # Restore to the state at step 1.
    step_id.assign_sub(1)
    state = driver_tpu.distributed_read_state(
        strategy=strategy,
        local_state=strategy.experimental_local_results(state),
        logical_coordinates=logical_coordinates,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        step_id=step_id)

    # Verify that we're at the previous state as step 1.
    self.assertEqual(step_id, 1)
    self.assertAllEqual(
        tf.nest.flatten(strategy.experimental_local_results(state)),
        (tf.ones([2]), 2 * tf.ones([2])))

  def test_distributed_write_state_writes_in_parallel(self):
    # We test that distributed_write_state executes the writes in parallel using
    # the available multiple threads. We create a huge number of states so that
    # writing them sequentially would time out.
    num_states = 1024
    num_writes = 10
    shape = (64, 64, 64)

    def init_fn(replica_id, coordinates):
      del replica_id
      del coordinates
      return {f'state_{i}': tf.ones(shape) for i in range(num_states)}

    computation_shape = np.array([2, 1, 1])
    strategy = driver_tpu.initialize_tpu(
        tpu_address='', computation_shape=computation_shape)
    logical_coordinates = util.grid_coordinates(computation_shape).tolist()
    initial_state = driver_tpu.distribute_values(strategy, init_fn,
                                                 logical_coordinates)

    # Set up step_id.
    with strategy.scope():
      step_id = tf.Variable(10)

    output_dir = os.path.join(flags.FLAGS.test_tmpdir, 'output_dir')
    filename_prefix = 'test'

    # Write the initial state `num_writes` times. We overwrite the same
    # directory to prevent OOMs.
    for _ in range(num_writes):
      driver_tpu.distributed_write_state(
          strategy=strategy,
          local_state=strategy.experimental_local_results(initial_state),
          logical_coordinates=logical_coordinates,
          output_dir=output_dir,
          filename_prefix=filename_prefix,
          step_id=step_id)

  def test_distributed_write_state(self):
    # Set up strategy and the initial state.
    def init_fn(replica_id, coordinates):
      del coordinates
      return {'replica_id': replica_id}

    computation_shape = np.array([2, 1, 1])
    strategy = driver_tpu.initialize_tpu(
        tpu_address='', computation_shape=computation_shape)
    logical_coordinates = util.grid_coordinates(computation_shape).tolist()
    initial_state = driver_tpu.distribute_values(strategy, init_fn,
                                                 logical_coordinates)

    # Set up step_id.
    with strategy.scope():
      step_id = tf.Variable(10)

    # Write the initial state.
    output_dir = os.path.join(flags.FLAGS.test_tmpdir, 'output_dir')
    filename_prefix = 'test'
    driver_tpu.distributed_write_state(
        strategy=strategy,
        local_state=strategy.experimental_local_results(initial_state),
        logical_coordinates=logical_coordinates,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        step_id=step_id)

    # Verify written state.
    for replica_id in range(2):
      output_filepath = os.path.join(
          output_dir, str(step_id.numpy()),
          f'{filename_prefix}-field-replica_id-xyz-{replica_id}-0-0-'
          f'step-{step_id.numpy()}.ser')
      with tf.io.gfile.GFile(output_filepath) as f:
        self.assertEqual(
            tf.io.parse_tensor(f.read(), out_type=tf.int32), replica_id)

  def test_distributed_write_state_single_device(self):
    # Set up strategy and the initial state.
    def init_fn(replica_id, coordinates):
      del coordinates
      return {'replica_id': replica_id}

    # We use just one core out of the available cores.
    computation_shape = np.array([1, 1, 1])
    strategy = driver_tpu.initialize_tpu(
        tpu_address='', computation_shape=computation_shape)
    self.assertEqual(strategy.num_replicas_in_sync, 1)

    logical_coordinates = util.grid_coordinates(computation_shape).tolist()
    initial_state = driver_tpu.distribute_values(strategy, init_fn,
                                                 logical_coordinates)
    # A dummy computation. In the single core case, strategy.run just returns
    # the Tensor instead of a PerReplica object.
    final_state = strategy.run(
        tf.function(functools.partial(tf.nest.map_structure, tf.identity)),
        args=(initial_state,))

    # Set up step_id.
    with strategy.scope():
      step_id = tf.Variable(10)

    # Write the initial state.
    output_dir = os.path.join(flags.FLAGS.test_tmpdir, 'output_dir')
    filename_prefix = 'test'
    # The state to write should be a tuple of length 1 in the single core case.
    driver_tpu.distributed_write_state(
        strategy=strategy,
        local_state=(final_state,),
        logical_coordinates=logical_coordinates,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        step_id=step_id)

    # Verify written state.
    for replica_id in range(1):
      output_filepath = os.path.join(
          output_dir, str(step_id.numpy()),
          f'{filename_prefix}-field-replica_id-xyz-{replica_id}-0-0-'
          f'step-{step_id.numpy()}.ser')
      with tf.io.gfile.GFile(output_filepath) as f:
        self.assertEqual(
            tf.io.parse_tensor(f.read(), out_type=tf.int32), replica_id)

  def test_distributed_write_invalid_state(self):
    # Set up strategy and the initial state.
    def init_fn(replica_id, coordinates):
      del coordinates
      return {
          'replica_id': replica_id,
          'invalid_output': tf.constant(1.0) / tf.range(-3.0, 3.0),
      }

    computation_shape = np.array([2, 1, 1])
    strategy = driver_tpu.initialize_tpu(
        tpu_address='', computation_shape=computation_shape)
    logical_coordinates = util.grid_coordinates(computation_shape).tolist()
    initial_state = driver_tpu.distribute_values(strategy, init_fn,
                                                 logical_coordinates)

    # Set up step_id.
    with strategy.scope():
      step_id = tf.Variable(10)

    # Write the initial state.
    output_dir = os.path.join(flags.FLAGS.test_tmpdir, 'output_dir')
    filename_prefix = 'test_with_invaid_output'
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                'invalid_output'):
      driver_tpu.distributed_write_state(
          strategy=strategy,
          local_state=strategy.experimental_local_results(initial_state),
          logical_coordinates=logical_coordinates,
          output_dir=output_dir,
          filename_prefix=filename_prefix,
          step_id=step_id)


if __name__ == '__main__':
  tf.test.main()
