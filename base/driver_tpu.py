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
from typing import Any, Callable

import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.framework import util

FILENAME_FORMAT = '{prefix}-field-{fieldname}-xyz-{rx}-{ry}-{rz}-step-{{}}.ser'

Array = Any  # An array convertible to TF tensors or numpy arrays.
# A structure with atoms convertible to tf.Tensors.
# (See https://www.tensorflow.org/api_docs/python/tf/nest?version=nightly)
Structure = Any
PerReplica = Any  # A tf.distribute PerReplica (not part of the public API yet.)
# A Callable mapping (replica_id, coordinates) to a Structure.
ValueFn = Callable[[Array, Array], Structure]


def initialize_tpu(tpu_address: str,
                   computation_shape: Array) -> tf.distribute.TPUStrategy:
  """Initializes the TPU with the logical coordinates and returns a TPUStrategy.

  Args:
    tpu_address: The address of the TPU cluster to connect to.
    computation_shape: An array of three positive integers denoting the logical
      shape of the TPU mesh.

  Returns:
    A TPUStrategy which can run computations on the initialized TPU cluster.
  """
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
  tf.config.experimental_connect_to_cluster(resolver)
  topology = tf.tpu.experimental.initialize_tpu_system(resolver)
  device_assignment, _ = util.tpu_device_assignment(
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
      slice along the first dimension contains the 3 logical mesh coordinates
      of the `i`th replica of the TPU cluster. This is passed as the second
      argument of `value_fn`.

  Returns:
    A PerReplica object containing the Structure created on each device.
  """
  def _wrapped_value_fn(replica_context):
    replica_id = replica_context.replica_id_in_sync_group
    return value_fn(tf.constant(replica_id, tf.int32),
                    logical_coordinates[replica_id])

  return strategy.experimental_distribute_values_from_function(
      _wrapped_value_fn)


def distributed_write_state(strategy: tf.distribute.TPUStrategy,
                            state: PerReplica, logical_coordinates: Array,
                            output_dir: str, filename_prefix: str,
                            step_id: Array,
                            checkpoint: tf.train.Checkpoint) -> Array:
  """Write a PerReplica structure to the filesystem.

  Args:
    strategy: The strategy from which to obtain the `state`.
    state: A PerReplica object containing a Structure on each worker device.
    logical_coordinates: The `logical_coordinates` is a 4D array whose `i`th
      entry contains the 3D logical mesh coordinates of the `i`th replica of the
      TPU cluster. These coordinates are added to the filenames using the string
      template `FILENAME_FORMAT`.
    output_dir: The output directory to write the files to.
    filename_prefix: A prefix added to the filenames. See `FILENAME_FORMAT`.
    step_id: An integer denoting the current step. This is added to the filename
      according to `FILENAME_FORMAT`.
    checkpoint: A checkpointing object written to after completion of all the
      writes.

  Returns:
    A sentinel `tf.Tensor` whose control dependencies include all the write ops.
  """
  local_state = strategy.experimental_local_results(state)
  write_ops = []

  for replica_id, (rx, ry, rz) in enumerate(logical_coordinates):
    worker_device = strategy.extended._device_assignment.host_device(  # pylint: disable=protected-access
        replica=replica_id)
    with tf.device(worker_device):
      for fieldname, tensor in local_state[replica_id].items():
        filename_template = FILENAME_FORMAT.format(
            prefix=filename_prefix, fieldname=fieldname, rx=rx, ry=ry, rz=rz)
        filepath_template = os.path.join(output_dir, '{}', filename_template)
        filepath = tf.strings.format(filepath_template, (step_id, step_id))
        write_ops.append(
            tf.io.write_file(filepath, tf.io.serialize_tensor(tensor)))

  with tf.control_dependencies(write_ops):
    checkpoint_file_prefix = os.path.join(output_dir, f'{filename_prefix}-ckpt')
    with tf.control_dependencies([checkpoint.write(checkpoint_file_prefix)]):
      return tf.constant(1.)
