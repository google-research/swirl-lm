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

import collections
import os
import typing
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeAlias

from absl import flags
from absl import logging
from swirl_lm.utility import tpu_util
import tensorflow as tf

_WORKER_JOB_PREFIX = flags.DEFINE_string(
    'worker_job_prefix', '',
    'The prefix for worker job. This is for the explicit device placement '
    'used in TPU driver. If left empty the default behavior from '
    'canonicalization will be used. For Google internal jobs, this usually should be '
    'set to `/job:worker/replica:0`.')

FLAGS = flags.FLAGS

# Template to generate filename path. Used for both read and write files.
# The format is:
# {prefix}-field-{fieldname}-xyz-{x_core}-{y_core}-{z_core}-step-{step_id}.ser
FILENAME_FORMAT = '{}-field-{}-xyz-{}-{}-{}-step-{}.ser'

Array = Any  # An array convertible to TF tensors or numpy arrays.
PerReplica: TypeAlias = tf.types.experimental.distributed.PerReplica
# A Callable mapping (replica_id, coordinates) to a tf.Tensor.
ValueFn = Callable[[Array, Array], dict[str, tf.Tensor]]


def replica_groups_by_host(strategy: tf.distribute.TPUStrategy):
  """Groups all device ids by their host."""
  replica_groups = collections.defaultdict(list)
  for replica_id in range(strategy.num_replicas_in_sync):
    # pylint: disable=protected-access
    host_device = strategy.extended._device_assignment.host_device(
        replica=replica_id)
    # pylint: enable=protected-access
    replica_groups[host_device].append(replica_id)

  return replica_groups


#  For the underlying functionality this simply calls the
# `experimental_distribute_values_from_function` that `pins` the values from
# `per_replica_states` to the corresponding TPU device. `tf.device` is added
# to also enforce the assignment ops are done on the corresponding host device.
def _distribute_values(
    strategy: tf.distribute.TPUStrategy,
    per_replica_states: List[dict[str, tf.Tensor]]):
  """Distributes values to the corresponding devices."""
  def distribute(ctx):
    replica_id = ctx.replica_id_in_sync_group
    # pylint: disable=protected-access
    host_device = strategy.extended._device_assignment.host_device(
        replica=replica_id)
    # pylint: enable=protected-access
    device_name = _WORKER_JOB_PREFIX.value + host_device
    logging.info('distribute: replica id: %d will be executed on %s',
                 replica_id, device_name)
    with tf.device(device_name):
      return per_replica_states[replica_id]
  return strategy.experimental_distribute_values_from_function(distribute)


def initialize_tpu(
    tpu_address: str,
    computation_shape: Array,
) -> tf.distribute.TPUStrategy:
  """Initializes the TPU with the logical coordinates and returns a TPUStrategy.

  Args:
    tpu_address: The address of the TPU cluster to connect to.
    computation_shape: An array of three positive integers denoting the logical
      shape of the TPU mesh.

  Returns:
    A TPUStrategy which can run computations on the initialized TPU cluster.
  """
  logging.info('Starting to acquire cluster_resolver at address %s',
               tpu_address)
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=tpu_address)
  logging.info('Resolver obtained at address %s', tpu_address)
  tf.config.experimental_connect_to_cluster(resolver)
  topology = tf.tpu.experimental.initialize_tpu_system(resolver)
  device_assignment, _ = tpu_util.tpu_device_assignment(
      computation_shape=computation_shape, tpu_topology=topology)
  return tf.distribute.TPUStrategy(
      resolver, experimental_device_assignment=device_assignment)


def _inner_wrapped_value_fn(
    replica_ids: Array,
    value_fn: ValueFn,
    logical_coordinates: Array) -> List[dict[str, tf.Tensor]]:
  """Calculates the initial values for all devices with the same host."""
  logging.info('_inner_wrapped_value_fn tracing starts.')
  result = []
  for replica_id, coordinate in zip(replica_ids, logical_coordinates):
    result.append(value_fn(replica_id, coordinate))
  logging.info('_inner_wrapped_value_fn traced.')
  return result


def distribute_values(
    strategy: tf.distribute.TPUStrategy,
    value_fn: ValueFn,
    logical_coordinates: Array,
) -> dict[str, PerReplica]:
  """Populates a PerReplica object containing values specified by `value_fn`.

  Args:
    strategy: The TPUStrategy used to run the distributed computations.
    value_fn: A function accepting `replica_id` and `logical_coordinates` which
      should return a dictionary mapping field names to tensors corresponding
      to the device at `replica_id`.
    logical_coordinates: The `logical_coordinates` is a 2D array whose `i`th
      slice along the first dimension contains the 3 logical mesh coordinates of
      the `i`th replica of the TPU cluster. This is passed as the second
      argument of `value_fn`.

  Returns:
    A dictionary from field names to PerReplica objects containing the
    values created on each device.
  """
  logging.info('Entering `distribute_values`')

  num_replicas = strategy.num_replicas_in_sync
  per_replica_states = [None] * num_replicas

  def _wrapped_value_fn(host_device, replica_ids):
    device_name = _WORKER_JOB_PREFIX.value + host_device
    logging.info('`distribute_values`, replica_ids %s to be placed on %s',
                 str(replica_ids), device_name)
    with tf.device(device_name):
      replica_id_tensors = [tf.constant(i, tf.int32) for i in replica_ids]
      logical_coordinate_tensors = [tf.convert_to_tensor(
          logical_coordinates[i], tf.int32) for i in replica_ids]
      result = _inner_wrapped_value_fn(replica_id_tensors,
                                       value_fn, logical_coordinate_tensors)
      for i, replica_id in enumerate(replica_ids):
        per_replica_states[replica_id] = result[i]

  replica_groups = replica_groups_by_host(strategy)
  logging.info('For init value, replica_groups: %s', str(replica_groups))

  for host_device, replica_ids in replica_groups.items():
    _wrapped_value_fn(host_device, replica_ids)

  logging.info('Exiting `distribute_values`.')
  # Because `per_replica_states` is initialized with Nones and then filled with
  # actual values, pytype infers that it contains tf.Tensor | None instead of
  # just tf.Tensor, so we explicitly cast the type back to tf.Tensors. This
  # has no runtime effect and is only a hint to pytype.
  return _distribute_values(
      strategy, typing.cast(list[dict[str, tf.Tensor]], per_replica_states))


@tf.function
def _read_input_at_step(
    state: dict[str, tf.Tensor],
    rx: Array,
    ry: Array,
    rz: Array,
    output_dir: str,
    filename_prefix: str,
    step_id: Array,
    states_from_file: Optional[Sequence[str]] = None,
) -> dict[str, tf.Tensor]:
  """Reads files for a single device."""
  logging.info('_read_input_at_step tracing starts.')
  read_state = {}
  for fieldname, initial_tensor in state.items():
    if states_from_file and fieldname not in states_from_file:
      read_state[fieldname] = initial_tensor
      continue

    filepath_template = FILENAME_FORMAT.format(
        filename_prefix, fieldname, '{}', '{}', '{}', '{}')
    filepath_template = os.path.join(
        output_dir, '{}', filepath_template)
    filepath = tf.strings.format(filepath_template,
                                 (step_id, rx, ry, rz, step_id))
    read_state[fieldname] = tf.io.parse_tensor(
        tf.io.read_file(filepath), out_type=initial_tensor.dtype)
  logging.info('_read_input_at_step traced.')
  return read_state


@tf.function
def _inner_read_step_fn(
    state: tuple[dict[str, tf.Tensor], ...],
    logical_coordinates: Array,
    output_dir: str,
    filename_prefix: str,
    step_id: Array,
    states_from_file: Optional[Sequence[str]] = None,
) -> List[dict[str, tf.Tensor]]:
  """Reads files for all devices with the same host."""
  logging.info('_inner_read_step_fn tracing starts.')
  result = []
  for i, coordinate in enumerate(logical_coordinates):
    replica_state = state[i]
    result.append(
        _read_input_at_step(
            replica_state,
            coordinate[0],
            coordinate[1],
            coordinate[2],
            output_dir,
            filename_prefix,
            step_id,
            states_from_file=states_from_file,
        )
    )
  logging.info('_inner_read_step_fn traced.')
  return result


def distributed_read_state(
    strategy: tf.distribute.TPUStrategy,
    state: tuple[dict[str, tf.Tensor], ...],
    logical_coordinates: List[Tuple[int, int, int]],
    output_dir: str,
    filename_prefix: str,
    step_id: Array,
    states_from_file: Optional[Sequence[str]] = None,
) -> dict[str, PerReplica]:
  """Read simulator state from the filesystem.

  Args:
    strategy: The strategy from which to obtain the `state`.
    state: A tuple where each element represents the local state for each
      device. Only the keys and dtypes are used from this to parse the read
      state.
    logical_coordinates: The `logical_coordinates` is 2D Tensor whose `i`th
      row contains the 3D logical mesh coordinates of the `i`th replica of the
      TPU cluster. These coordinates are added to the filenames.
    output_dir: The output directory to read the files from.
    filename_prefix: A prefix added to the filenames. See
      `FILENAME_FORMAT`.
    step_id: An integer scalar tf.Tensor denoting the current step. This is
      added to the filename.
    states_from_file: A sequence of strings that specifies the names of
      variables to be loaded from checkpoint files. If not provided, all
      variables in `state` will be loaded from files.

  Returns:
    The parsed state as a PerReplica object.
  """

  logging.info('Entering `distributed_read_state`')

  # In case global distributed step_id tf.Variable is used, this decouples what
  # is passed into the subgraph from it and improves the performance by
  # preventing the unnecessary synchronization/locking between cores.
  step_id = tf.constant(step_id.numpy(), tf.int32)

  num_replicas = strategy.num_replicas_in_sync
  per_replica_states = [None] * num_replicas

  def _read_step_fn(state, host_device, replica_ids, step_id):
    device_name = _WORKER_JOB_PREFIX.value + host_device
    logging.info('`distributed_read_state`, replica_ids %s to be placed on %s',
                 str(replica_ids), device_name)
    with tf.device(device_name):
      # We do this so the unrelated replicas' inputs won't be included into
      # the _inner_read_step_fn, reducing the redundant operations.
      partial_state = [state[i] for i in replica_ids]

      # This makes everything goes into _inner_read_step_fn tensor, so no
      # retracing will happen for different devices.
      partial_coordinates = [
          tf.constant(logical_coordinates[i], tf.int32) for i in replica_ids]
      result = _inner_read_step_fn(
          partial_state,
          partial_coordinates,
          output_dir,
          filename_prefix,
          step_id,
          states_from_file=states_from_file,
      )
      for i, replica_id in enumerate(replica_ids):
        per_replica_states[replica_id] = result[i]

  replica_groups = replica_groups_by_host(strategy)
  logging.info('For read value, replica_groups: %s', str(replica_groups))

  for host_device, replica_ids in replica_groups.items():
    _read_step_fn(state, host_device, replica_ids, step_id)

  logging.info('Exiting `distributed_read_state`')
  # See note elsewhere in this file about the type cast of `per_replica_states`.
  return _distribute_values(
      strategy, typing.cast(list[dict[str, tf.Tensor]], per_replica_states))


@tf.function
def _write_output_at_step(
    state: dict[str, tf.Tensor],
    rx: Array,
    ry: Array,
    rz: Array,
    output_dir: str,
    filename_prefix: str,
    step_id: Array,
    data_dump_filter: Optional[Sequence[str]] = None,
    fields_allowing_non_finite_values: Optional[Sequence[str]] = None,
) -> dict[str, bool]:
  """Writes output for one single device."""
  logging.info('_write_output_at_step tracing starts.')
  write_status = {}
  for fieldname, tensor in state.items():
    if data_dump_filter and fieldname not in data_dump_filter:
      # We still want to fill the return status for all the fields.
      write_status[fieldname] = True
      continue
    if (not tensor.dtype.is_floating or
        (fields_allowing_non_finite_values and
         fieldname in fields_allowing_non_finite_values)):
      any_nan_inf = False
    else:
      any_nan_inf = tf.math.logical_or(tf.reduce_any(tf.math.is_nan(tensor)),
                                       tf.reduce_any(tf.math.is_inf(tensor)))

    filepath_template = FILENAME_FORMAT.format(
        filename_prefix, fieldname, '{}', '{}', '{}', '{}')
    filepath_template = os.path.join(output_dir, '{}', filepath_template)
    filepath = tf.strings.format(filepath_template,
                                 (step_id, rx, ry, rz, step_id))

    check_op = (tf.debugging.assert_equal(
        any_nan_inf, False,
        f'Unexpected non-finite value in variable `{fieldname}`'))
    with tf.control_dependencies([check_op]):
      tf.io.write_file(filepath, tf.io.serialize_tensor(tensor))
      write_status[fieldname] = True
  logging.info('_write_output_at_step traced.')
  return write_status


@tf.function
def _inner_write_step_fn(
    state: tuple[dict[str, tf.Tensor], ...],
    logical_coordinates: Array,
    output_dir: str,
    filename_prefix: str,
    step_id: Array,
    data_dump_filter: Optional[Sequence[str]] = None,
    fields_allowing_non_finite_values: Optional[Sequence[str]] = None,
) -> list[dict[str, tf.Tensor]]:
  """Writes output for all devices with the same host."""
  logging.info('_inner_write_step_fn tracing starts.')
  output: list[dict[str, tf.Tensor]] = []
  for i, coordinate in enumerate(logical_coordinates):
    replica_state = state[i]
    output.append(_write_output_at_step(
        replica_state, coordinate[0],
        coordinate[1],
        coordinate[2],
        output_dir, filename_prefix, step_id,
        data_dump_filter,
        fields_allowing_non_finite_values))
  logging.info('_inner_write_step_fn traced.')
  return output


def distributed_write_state(
    strategy: tf.distribute.TPUStrategy,
    state: tuple[dict[str, tf.Tensor], ...],
    logical_coordinates: List[Tuple[int, int, int]],
    output_dir: str,
    filename_prefix: str,
    step_id: Array,
    data_dump_filter: Optional[Sequence[str]] = None,
    fields_allowing_non_finite_values: Optional[Sequence[str]] = None,
) -> list[dict[str, tf.Tensor]]:
  """Write simulator state to the filesystem.

  This also verifies that the content written is all valid and does not contain
  `nan` or `infinity`.

  Args:
    strategy: The strategy from which to obtain the `state`.
    state: A tuple where each element represents the local state for each
      device.
    logical_coordinates: The `logical_coordinates` is a 2D Tensor whose `i`th
      row contains the 3D logical mesh coordinates of the `i`th replica of the
      TPU cluster. These coordinates are added to the filenames.
    output_dir: The output directory to write the files to.
    filename_prefix: A prefix added to the filenames. See
      `FILENAME_FORMAT`.
    step_id: An integer scalar tf.tensor denoting the current step. This is
      added to the filename.
    data_dump_filter: List of fields that should be written to files when set.
      If this is not provided (None), then all fields will be written.
    fields_allowing_non_finite_values: List of fields that are allowed to
      contain non-finite values. This is normally used for debug variables.

  Returns:
    A distributed status_state with one boolean per field per replica. This is
    used for the client code to synchronize with the remote write operations.
  """

  logging.info('Entering `distributed_write_state`')

  # In case global distributed step_id tf.Variable is used, this decouples what
  # is passed into the subgraph from it and improves the performance by
  # preventing the unnecessary synchronization/locking between cores.
  step_id = tf.constant(step_id.numpy(), tf.int32)

  num_replicas = strategy.num_replicas_in_sync
  per_replica_status = [None] * num_replicas

  def _write_step_fn(state, step_id, host_device, replica_ids):
    device_name = _WORKER_JOB_PREFIX.value + host_device
    logging.info('`distributed_write_state`, replica_ids %s to be placed on %s',
                 str(replica_ids), device_name)
    with tf.device(device_name):
      # We do this so the unrelated replicas' inputs won't be included into
      # the _inner_write_step_fn, reducing the redundant execution.
      partial_state = [state[i] for i in replica_ids]

      # This makes everything goes into _inner_write_step_fn as tensor, so no
      # retracing will happen for different devices.
      partial_coordinates = [
          tf.constant(logical_coordinates[i], tf.int32) for i in replica_ids]
      write_status: list[dict[str, tf.Tensor]] = _inner_write_step_fn(
          partial_state, partial_coordinates, output_dir, filename_prefix,
          step_id, data_dump_filter, fields_allowing_non_finite_values)
      for i, replica_id in enumerate(replica_ids):
        per_replica_status[replica_id] = write_status[i]

  replica_groups = replica_groups_by_host(strategy)
  logging.info('For write_output, replica_groups: %s', str(replica_groups))

  for host_device, replica_ids in replica_groups.items():
    _write_step_fn(state, step_id, host_device, replica_ids)

  logging.info('Exiting `distributed_write_state`')
  # See note elsewhere in this file about the type cast of `per_replica_status`.
  return typing.cast(list[dict[str, tf.Tensor]], per_replica_status)
