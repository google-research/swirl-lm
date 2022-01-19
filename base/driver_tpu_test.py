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

import os

from absl import flags
import numpy as np
from swirl_lm.base import driver_tpu
import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.framework import util


class DriverTpuTest(tf.test.TestCase):

  def test_initialize_tpu(self):
    strategy = driver_tpu.initialize_tpu(
        tpu_address='', computation_shape=[2, 1, 1])
    self.assertEqual(strategy.num_replicas_in_sync, 2)

    @tf.function
    def add_fn(x, y):
      z = x + y
      return z

    x = tf.constant(3.)
    y = tf.constant(4.)
    z = strategy.run(add_fn, args=(x, y))
    self.assertEqual(strategy.experimental_local_results(z), (7., 7.))

  def test_distributed_values(self):

    def init_fn(replica_id, coordinates):
      del coordinates
      return {'replica_id': replica_id, 'p': tf.ones([2])}

    computation_shape = np.array([2, 1, 1])
    strategy = driver_tpu.initialize_tpu(
        tpu_address='', computation_shape=computation_shape)
    logical_coordinates = util.grid_coordinates(computation_shape)
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

  def test_distributed_write_state(self):
    # Set up strategy and the initial state.
    def init_fn(replica_id, coordinates):
      del coordinates
      return {'replica_id': replica_id}

    computation_shape = np.array([2, 1, 1])
    strategy = driver_tpu.initialize_tpu(
        tpu_address='', computation_shape=computation_shape)
    logical_coordinates = util.grid_coordinates(computation_shape)
    initial_state = driver_tpu.distribute_values(strategy, init_fn,
                                                 logical_coordinates)

    # Set up step_id with a checkpoint.
    with strategy.scope():
      step_id = tf.Variable(10)
    checkpoint = tf.train.Checkpoint(step_id=step_id)

    # Write the initial state.
    output_dir = os.path.join(flags.FLAGS.test_tmpdir, 'output_dir')
    filename_prefix = 'test'
    driver_tpu.distributed_write_state(
        strategy=strategy,
        state=initial_state,
        logical_coordinates=logical_coordinates,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        step_id=step_id,
        checkpoint=checkpoint)

    # Verify written state.
    for replica_id in range(2):
      output_filepath = os.path.join(
          output_dir, str(step_id.numpy()),
          f'{filename_prefix}-field-replica_id-xyz-{replica_id}-0-0-'
          f'step-{step_id.numpy()}.ser')
      with tf.io.gfile.GFile(output_filepath) as f:
        self.assertEqual(
            tf.io.parse_tensor(f.read(), out_type=tf.int32), replica_id)


if __name__ == '__main__':
  tf.test.main()
