# Copyright 2023 The swirl_lm Authors.
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
"""A module providing common TF2 tf.distribute-related utility functions."""

import os
import time
from typing import Any, Callable, List, Tuple

from absl import logging
from swirl_lm.utility import tpu_util
import tensorflow as tf


FILENAME_FORMAT = '{prefix}-field-{fieldname}-xyz-{rx}-{ry}-{rz}-step-{{}}.ser'

Array = Any  # An array convertible to TF tensors or numpy arrays.
# A structure with atoms convertible to tf.Tensors.
# (See https://www.tensorflow.org/api_docs/python/tf/nest?version=nightly)
Structure = Any
PerReplica = Any  # A tf.distribute PerReplica (not part of the public API yet.)
# A Callable mapping (replica_id, coordinates) to a Structure.
ValueFn = Callable[[Array, Array], Structure]


def initialize_tpu(
    tpu_address: str,
    computation_shape: Array,
    num_attempts: int = 180,
    back_off_time_sec: int = 60,
) -> tf.distribute.TPUStrategy:
  """Initializes the TPU with the logical coordinates and returns a TPUStrategy.

  Args:
    tpu_address: The address of the TPU cluster to connect to.
    computation_shape: An array of three positive integers denoting the logical
      shape of the TPU mesh.
    num_attempts: The max number of times to attempt to connect to the TPU
      workers. In `headless` launch configuration, the worker jobs might not be
      ready right away and we might need to attempt a few times.
    back_off_time_sec: The time in secs between the attempts to connect to the
      TPU workers.

  Returns:
    A TPUStrategy which can run computations on the initialized TPU cluster.
  """
  attempt = 0
  success = False
  while attempt < num_attempts and not success:
    try:
      # Note: all three actions below have to be retried together if any of
      # them is unsuccessful.
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          tpu=tpu_address)
      tf.config.experimental_connect_to_cluster(resolver)
      topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    except Exception as e:  # pylint: disable=broad-except
      success = False
      logging.info(
          'Attempt #%d to initialize the TPU workers was not successful. Error '
          ': %s. TPU workers might still be starting up. Backing off for %d '
          'secs before trying to connect again.',
          attempt, str(e), back_off_time_sec)
      attempt += 1
      time.sleep(back_off_time_sec)
    else:
      success = True
      logging.info('TPU Workers initialization successful at attempt #%d.',
                   attempt)
  device_assignment, _ = tpu_util.tpu_device_assignment(
      computation_shape=computation_shape, tpu_topology=topology)
  return tf.distribute.TPUStrategy(
      resolver, experimental_device_assignment=device_assignment)


def distribute_values(strategy: tf.distribute.TPUStrategy, value_fn: ValueFn,
                      logical_coordinates: Array) -> PerReplica:
  """Populates a PerReplica object containing values specified by `value_fn`.

  Args:
    strategy: The TPUStrategy used to run the distributed computations.
    value_fn: A function accepting `replica_id` and `logical_coordinates` which
      should return the Structure corresponding to the device at `replica_id`.
    logical_coordinates: The `logical_coordinates` is a 2D array whose `i`th
      slice along the first dimension contains the 3 logical mesh coordinates of
      the `i`th replica of the TPU cluster. This is passed as the second
      argument of `value_fn`.

  Returns:
    A PerReplica object containing the Structure created on each device.
  """

  def _wrapped_value_fn(replica_context):
    replica_id = replica_context.replica_id_in_sync_group

    # Places the initialization on the corresponding host for each replica.
    # Without this, all the initialization is done through worker 0 and then
    # dispatched to the corresponding TPU replicas, which requires a large peak
    # memory on worker 0 and puts a bottleneck on the size of the grid we can
    # squeeze in. With this placement, in one example, the peak memory on the
    # hosts is reduced from 220G to ~ 20G and it is clearly seen from the
    # profiling and the memory usage that all worker hosts are now involved in
    # the initialization.
    worker_device = strategy.extended._device_assignment.tpu_device(  # pylint: disable=protected-access
        replica=replica_id)
    with tf.device(worker_device):
      values = value_fn(
          tf.constant(replica_id, tf.int32), logical_coordinates[replica_id])
    return values

  return strategy.experimental_distribute_values_from_function(
      _wrapped_value_fn)


@tf.function
def distributed_read_state(
    strategy: tf.distribute.TPUStrategy,
    local_state: Tuple[Structure],
    logical_coordinates: List[Tuple[int, int, int]],
    output_dir: str,
    filename_prefix: str,
    step_id: Array,
) -> tf.distribute.DistributedValues:
  """Read a DistributedValues structure from the filesystem.

  Args:
    strategy: The strategy from which to obtain the `state`.
    local_state: A tuple of length `strategy.num_replicas_in_sync` containing a
      Structure on each replica. Only the keys and dtypes are used from this to
      parse the read state.
    logical_coordinates: The `logical_coordinates` is a list whose `i`th entry
      contains the 3D logical mesh coordinates of the `i`th replica of the TPU
      cluster. These coordinates are added to the filenames using the string
      template `FILENAME_FORMAT`.
    output_dir: The output directory to read the files from.
    filename_prefix: A prefix added to the filenames. See `FILENAME_FORMAT`.
    step_id: An integer denoting the current step. This is added to the filename
      according to `FILENAME_FORMAT`.

  Returns:
    The parsed state, a dictionary of Tensors on each replica.
  """
  per_replica_states = []
  for replica_id, (rx, ry, rz) in enumerate(logical_coordinates):
    worker_device = strategy.extended._device_assignment.host_device(  # pylint: disable=protected-access
        replica=replica_id)
    with tf.device(worker_device):
      # Place the current replica's value on the appropriate device.
      state = {}
      for fieldname, initial_tensor in local_state[replica_id].items():
        filename_template = FILENAME_FORMAT.format(
            prefix=filename_prefix, fieldname=fieldname, rx=rx, ry=ry, rz=rz)
        filepath_template = os.path.join(output_dir, '{}', filename_template)
        filepath = tf.strings.format(filepath_template, (step_id, step_id))
        state[fieldname] = tf.io.parse_tensor(
            tf.io.read_file(filepath), out_type=initial_tensor.dtype)
      # Append the current state to be distributed
      per_replica_states.append(state)

  # Wrap the per-replica states into a `tf.distribute.DistributedValues`.
  return strategy.experimental_distribute_values_from_function(
      lambda ctx: per_replica_states[ctx.replica_id_in_sync_group])


@tf.function
def distributed_write_state(
    strategy: tf.distribute.TPUStrategy,
    local_state: Tuple[Structure],
    logical_coordinates: List[Tuple[int, int, int]],
    output_dir: str,
    filename_prefix: str,
    step_id: Array,
) -> None:
  """Write a PerReplica structure to the filesystem.

  This also verifies that the content written is all valid and does not contain
  `nan` or `infinity`.

  Args:
    strategy: The strategy from which to obtain the `state`.
    local_state: A tuple of length `strategy.num_replicas_in_sync` containing a
      Structure on each replica.
    logical_coordinates: The `logical_coordinates` is a list whose `i`th entry
      contains the 3D logical mesh coordinates of the `i`th replica of the TPU
      cluster. These coordinates are added to the filenames using the string
      template `FILENAME_FORMAT`.
    output_dir: The output directory to write the files to.
    filename_prefix: A prefix added to the filenames. See `FILENAME_FORMAT`.
    step_id: An integer denoting the current step. This is added to the filename
      according to `FILENAME_FORMAT`.
  """
  for replica_id, (rx, ry, rz) in enumerate(logical_coordinates):
    worker_device = strategy.extended._device_assignment.host_device(  # pylint: disable=protected-access
        replica=replica_id)
    with tf.device(worker_device):
      for fieldname, tensor in local_state[replica_id].items():
        filename_template = FILENAME_FORMAT.format(
            prefix=filename_prefix, fieldname=fieldname, rx=rx, ry=ry, rz=rz)
        filepath_template = os.path.join(output_dir, '{}', filename_template)
        filepath = tf.strings.format(filepath_template, (step_id, step_id))
        if tensor.dtype.is_floating:
          check_op = (
              tf.debugging.assert_equal(
                  tf.math.logical_or(
                      tf.reduce_any(tf.math.is_nan(tensor)),
                      tf.reduce_any(tf.math.is_inf(tensor))), False,
                  f'Unexpected non-finite value in variable `{fieldname}`'))
        else:
          check_op = tf.no_op()
        with tf.control_dependencies([check_op]):
          tf.compat.v1.get_default_graph().experimental_acd_manager.run_independently(  # pylint: disable=line-too-long
              tf.io.write_file(filepath, tf.io.serialize_tensor(tensor)))
