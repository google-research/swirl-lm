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

"""Library for the incompressible structured mesh Navier-Stokes solver."""

import functools
import os
import time
from typing import Any, Dict, Optional, Tuple, TypeVar

from absl import flags
from absl import logging
import numpy as np
from swirl_lm.base import driver_tpu
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import target_flag  # pylint: disable=unused-import
from swirl_lm.core import simulation
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import tpu_util
import tensorflow as tf

flags.DEFINE_integer(
    'min_steps_for_output', 1, 'Total number of steps before '
    'the output start to be generated.', allow_override=True)
flags.DEFINE_string(
    'data_dump_prefix', '/tmp/data', 'The output `ser` or `h5` '
    'files prefix. This will be suffixed with the field '
    'components and step count.', allow_override=True)
flags.DEFINE_string(
    'data_load_prefix', '/tmp/data', 'The input `ser` or `h5` '
    'files prefix. This will be suffixed with the field '
    'components and step count.', allow_override=True)
flags.DEFINE_bool(
    'apply_data_load_filter', False,
    'If True, only variables with names provided in field `variable_from_file` '
    'in the config file will be loaded from files (at step specified by flag '
    '`loading_step`.', allow_override=True)
RESTART_DUMP_CYCLE = flags.DEFINE_integer(
    'restart_dump_cycle',
    1,
    'The number of cycles between which full variables may be dumped to file.',
    allow_override=True,
)

FLAGS = flags.FLAGS

CKPT_DIR_FORMAT = '{filename_prefix}-ckpts/'

Array = Any
PerReplica = Any
Structure = Any
T = TypeVar('T')
S = TypeVar('S')


def _stateless_update_if_present(mapping: Dict[T, S],
                                 updates: Dict[T, S]) -> Dict[T, S]:
  """Returns a copy of `mapping` with only existing keys updated."""
  mapping = mapping.copy()
  mapping.update({key: val for key, val in updates.items() if key in mapping})
  return mapping


def _local_state(
    strategy: tf.distribute.Strategy,
    distributed_state: tf.distribute.DistributedValues,
) -> Tuple[Structure]:
  return strategy.experimental_local_results(distributed_state)


def get_checkpoint_manager(
    step_id: Array,
    output_dir: str,
    filename_prefix: str,
) -> tf.train.CheckpointManager:
  """Returns a `tf.train.CheckpointManager` that records just the step id.

  Args:
    step_id: The step id associated with the checkpoint. Usually a tf.Variable.
    output_dir: The directory of the simulation output.
    filename_prefix: The prefix used in the simulation output files.

  Returns:
    A checkpoint manager that can be used to restore the previous state or to
    save the new state.
  """
  checkpoint = tf.train.Checkpoint(step_id=step_id)
  checkpoint_dir = os.path.join(
      output_dir, CKPT_DIR_FORMAT.format(filename_prefix=filename_prefix))
  return tf.train.CheckpointManager(
      checkpoint, directory=checkpoint_dir, max_to_keep=3)


def _get_state_keys(params):
  """Returns essential, additional and helper var state keys."""
  # Essential flow field variables:
  # u: velocity in dimension 0;
  # v: velocity in dimension 1;
  # w: velocity in dimension 2;
  # p: pressure.
  essential_keys = ['u', 'v', 'w', 'p'] + params.transport_scalars_names
  if params.solver_procedure == parameters_lib.SolverProcedure.VARIABLE_DENSITY:
    essential_keys += ['rho']
  additional_keys = list(
      params.additional_state_keys if params.additional_state_keys else [])
  helper_var_keys = list(
      params.helper_var_keys if params.helper_var_keys else [])

  # Check to make sure we don't have keys duplicating / overwriting each other.
  if (len(set(essential_keys)) + len(set(additional_keys)) +
      len(set(helper_var_keys)) !=
      len(set(essential_keys + additional_keys + helper_var_keys))):
    raise ValueError(
        f'Duplicated keys detected between the three types of states: '
        f'essential states: {essential_keys}, additional states: '
        f'{additional_keys}, and helper vars: {helper_var_keys}')

  for state_analytics_info in params.monitor_spec.state_analytics:
    for analytics_spec in state_analytics_info.analytics:
      helper_var_keys.append(analytics_spec.key)

  return essential_keys, additional_keys, helper_var_keys


def _get_model(kernel_op, params):
  """Returns the appropriate Navier-Stokes model from `params`."""
  if params.solver_procedure == parameters_lib.SolverProcedure.VARIABLE_DENSITY:
    return simulation.Simulation(kernel_op, params)
  raise ValueError('Solver procedure not recognized: '
                   f'{params.solver_procedure}')


def _process_at_step_id(process_fn, essential_states, additional_states,
                        step_id, process_step_id, is_periodic):
  """Executes `process_fn` conditionally depending on `step_id`.

  Args:
    process_fn: Function accepting `essential_states` and `additional_states`,
      and returning the updated values of individual states in a dictionary.
    essential_states: The essential states, corresponds to the`states` keyword
      argument of `process_fn`.
    additional_states: The additional states, corresponds to the
      `additional_states` keyword argument of `process_fn`.
    step_id: The current step id.
    process_step_id: The fixed value at which to trigger execution of
      `process_fn`. In other words, if the current step id is equal to
      `process_step_id`, then the execution of `process_fn` is triggered. If,
      additionally, `is_periodic` is True, then the execution is also triggered
      whenever `step_id % process_step_id == 0`.
    is_periodic: If True, then `process_fn` is executed whenever `step_id %
      process_step_id == 0`.

  Returns:
    The updated `essential_states` and `additional_states`.
  """
  should_process = ((step_id % process_step_id == 0)
                    if is_periodic else step_id == process_step_id)
  if should_process:
    updated_states = process_fn(
        states=essential_states, additional_states=additional_states)
    essential_states = _stateless_update_if_present(essential_states,
                                                    updated_states)
    additional_states = _stateless_update_if_present(additional_states,
                                                     updated_states)

  return essential_states, additional_states


@tf.function
def _one_cycle(
    strategy: tf.distribute.Strategy,
    init_state: PerReplica,
    init_step_id: Array,
    num_steps: Array,
    params: parameters_lib.SwirlLMParameters,
) -> PerReplica:
  """Runs one cycle of the Navier-Stokes solver.

  Args:
    strategy: A distributed strategy to run computations. Currently, only
      `TPUStrategy` is supported.
    init_state: The initial state on each device.
    init_step_id: An integer scalar denoting the initial step id at which to
      start the cycle.
    num_steps: An integer scalar; the number of steps to run in the cycle.
    params: The solver configuration.

  Returns:
    The final state at the end of the cycle.
  """
  logging.info('Tracing and compiling of _one_cycle starts. '
               'This can take up to 30 min.')
  essential_keys, additional_keys, helper_var_keys = _get_state_keys(params)
  computation_shape = np.array([params.cx, params.cy, params.cz])
  logical_replicas = np.arange(
      strategy.num_replicas_in_sync, dtype=np.int32).reshape(computation_shape)

  if (params.kernel_op_type ==
      parameters_lib.KernelOpType.KERNEL_OP_CONV):
    kernel_op = get_kernel_fn.ApplyKernelConvOp(params.kernel_size)
  elif (params.kernel_op_type ==
        parameters_lib.KernelOpType.KERNEL_OP_SLICE):
    kernel_op = get_kernel_fn.ApplyKernelSliceOp()
  elif (params.kernel_op_type ==
        parameters_lib.KernelOpType.KERNEL_OP_MATMUL):
    kernel_op = get_kernel_fn.ApplyKernelMulOp(params.nx,
                                               params.ny)
  else:
    raise ValueError('Unknown kernel operator {}'.format(
        params.kernel_op_type))

  model = _get_model(kernel_op, params)

  def step_fn(state):
    # Common keyword arguments to various step functions.
    common_kwargs = dict(
        kernel_op=kernel_op,
        replica_id=state['replica_id'],
        replicas=logical_replicas,
        params=params)
    # Split essential/additional states into lists of 2D Tensors.
    keys_to_split = (
        set.union(set(essential_keys), set(additional_keys)) -
        set(helper_var_keys))
    for key in keys_to_split:
      state[key] = tf.unstack(state[key])

    for cycle_step_id in tf.range(num_steps):
      step_id = init_step_id + cycle_step_id
      # Split the state into essential states and additional states. Note that
      # `additional_states` consists of both additional state keys and helper
      # var keys.
      additional_states = dict(
          (key, state[key]) for key in additional_keys + helper_var_keys)
      essential_states = dict((key, state[key]) for key in essential_keys)

      # Perform a preprocessing step, if configured.
      if params.apply_preprocess:
        essential_states, additional_states = _process_at_step_id(
            process_fn=functools.partial(params.preprocessing_states_update_fn,
                                         **common_kwargs),
            essential_states=essential_states,
            additional_states=additional_states,
            step_id=step_id,
            process_step_id=params.preprocess_step_id,
            is_periodic=params.preprocess_periodic)

      # Perform the additional states update, if present.
      if params.additional_states_update_fn is not None:
        additional_states = params.additional_states_update_fn(
            states=essential_states,
            additional_states=additional_states,
            step_id=step_id,
            **common_kwargs)

      # Perform one step of the Navier-Stokes model. The updated state should
      # contain both the essential and additional states.
      updated_state = model.step(
          replica_id=state['replica_id'],
          replicas=logical_replicas,
          step_id=step_id,
          states=essential_states,
          additional_states=additional_states)

      # Perform a postprocessing step, if configured.
      if params.apply_postprocess:
        # Split the updated_state into essential states and additional states.
        additional_states = _stateless_update_if_present(
            additional_states, updated_state)
        essential_states = _stateless_update_if_present(essential_states,
                                                        updated_state)

        essential_states, additional_states = _process_at_step_id(
            process_fn=functools.partial(params.postprocessing_states_update_fn,
                                         **common_kwargs),
            essential_states=essential_states,
            additional_states=additional_states,
            step_id=step_id,
            process_step_id=params.postprocess_step_id,
            is_periodic=params.postprocess_periodic)

        # Merge the essential states and additional states into updated_state.
        updated_state = _stateless_update_if_present(updated_state,
                                                     additional_states)
        updated_state = _stateless_update_if_present(updated_state,
                                                     essential_states)

      # Some state keys such as `replica_id` may not lie in either of the three
      # categories. Just pass them through.
      state = _stateless_update_if_present(state, updated_state)

    # Unsplit the keys that were previously split.
    for key in keys_to_split:
      state[key] = tf.stack(state[key])

    return state

  return strategy.run(step_fn, args=(init_state,))


def solver(
    init_fn,
    params_input: Optional[parameters_lib.SwirlLMParameters] = None,
):
  """Runs the Navier-Stokes Solver with TF2 Distribution strategy.

  Args:
    init_fn: The function that initializes the flow field. The function needs to
      be replica dependent.
    params_input: An instance of parameters that will be used in the simulation,
      e.g. the mesh size, fluid properties.

  Returns:
    A tuple of the final state on each replica.
  """

  # Obtain params either from the provided input or the flags.
  params = params_input
  if params is None:
    params = parameters_lib.params_from_config_file_flag()
  params.save_to_file(FLAGS.data_dump_prefix)

  # Initialize the TPU.
  logging.info('Entering solver.')
  computation_shape = np.array([params.cx, params.cy, params.cz])
  logging.info('Computation_shape is %s', str(computation_shape))
  strategy = driver_tpu.initialize_tpu(
      tpu_address=FLAGS.target, computation_shape=computation_shape)
  logging.info('TPU is initialized.')
  logical_coordinates = tpu_util.grid_coordinates(computation_shape).tolist()
  # In order to save and restore from the filesystem, we use tf.train.Checkpoint
  # on the step id. The `step_id` Variable should be placed on the TPU device so
  # that we don't block TPU execution when writing the state to filenames
  # formatted dynamically with the step id.
  with strategy.scope():
    step_id = tf.Variable(params.start_step, dtype=tf.int32)
  output_dir, filename_prefix = os.path.split(FLAGS.data_dump_prefix)

  logging.info('Getting checkpoint_manager.')
  ckpt_manager = get_checkpoint_manager(
      step_id=step_id, output_dir=output_dir, filename_prefix=filename_prefix)

  # Check if only part of the data will be dumped.
  data_dump_filter = params.states_to_file if params.states_to_file else None
  checkpoint_interval = RESTART_DUMP_CYCLE.value * params.num_steps
  logging.info('Got checkpoint_manager.')

  # Use this to access step_id, instead of directly using tf.Variable, which
  # will trigger the need for synchronization and will slow down/dead-lock for
  # multi-devices I/O.
  def step_id_value():
    return tf.constant(step_id.numpy(), tf.int32)

  write_state = functools.partial(
      driver_tpu.distributed_write_state,
      strategy,
      logical_coordinates=logical_coordinates,
      output_dir=output_dir,
      filename_prefix=filename_prefix)
  logging.info('write_state function created.')

  read_state = functools.partial(
      driver_tpu.distributed_read_state,
      strategy,
      logical_coordinates=logical_coordinates,
      output_dir=output_dir,
      filename_prefix=filename_prefix)
  logging.info('read_state function created.')
  t_start = time.time()

  # Wrapping `init_fn` with tf.function so it is not retraced unnecessarily for
  # every core/device.
  state = driver_tpu.distribute_values(
      strategy, value_fn=tf.function(init_fn),
      logical_coordinates=logical_coordinates)

  # Accessing the values in state to synchornize the client so the main thread
  # will wait here until the `state` is initialized and all remote operations
  # are done.
  replica_values = state['replica_id'].values
  logging.info('State initialized. Replicas are : %s', str(replica_values))
  t_post_init = time.time()
  logging.info('Initialization stage done. Took %f secs.',
               t_post_init - t_start)

  # Restore from an existing checkpoint if present.
  if ckpt_manager.latest_checkpoint:
    # The checkpoint restore updates the `step_id` variable; which is then used
    # to read in the state.
    logging.info('Detected checkpoint. Starting `restore_or_initialize`.')
    ckpt_manager.restore_or_initialize()
    state = read_state(state=_local_state(strategy, state),
                       step_id=step_id_value())
    # This is to use to sync the client code to the worker execution.
    replica_id_values = state['replica_id'].values
    logging.info(
        '`restoring-checkpoint-if-necessary` stage '
        'done with reading checkpoint. Replicas are: %s',
        str(replica_id_values))
  else:
    # In case we're not restoring from a checkpoint, write the initial state.
    logging.info('No checkpoint detected. Starting `write_state`.')
    write_status = write_state(state=_local_state(strategy, state),
                               step_id=step_id_value())
    # This is used to sync the client code to the remote workers asynchronous
    # executions.
    replica_id_values = write_status[0]['replica_id'].numpy()
    logging.info(
        '`restoring-checkpoint-if-necessary` stage '
        'done with writing initial steps. Replicas are: %s',
        str(replica_id_values))
  t_post_restore = time.time()
  logging.info('restore-if-necessary or write. Took %f secs.',
               t_post_restore - t_post_init)

  if params.num_steps < 0:
    raise ValueError('`num_steps` should not be negative but it was set to ' +
                     f'{params.num_steps}.')
  if (params.num_steps > 0 and
      (step_id_value() - params.start_step) % params.num_steps != 0):
    raise ValueError('Incompatible step_id detected. `step_id` is expected '
                     'to be `start_step` + N * `num_steps` but (step_id: {}, '
                     'start_step: {}, num_steps: {}) is detected. Maybe the '
                     'checkpoint step is inconsistent?'.format(
                         step_id_value(), params.start_step, params.num_steps))
  logging.info(
      'Simulation iteration starts. Total %d steps, starting from %d, '
      'with %d steps per cycle.', params.num_steps * params.num_cycles,
      step_id_value(), params.num_steps)

  while step_id_value() < (params.start_step +
                           params.num_steps * params.num_cycles):
    cycle = (step_id_value() - params.start_step) // params.num_steps
    logging.info('Step %d (cycle %d) is starting.', step_id_value(), cycle)
    t0 = time.time()
    state = _one_cycle(
        strategy=strategy,
        init_state=state,
        init_step_id=step_id_value(),
        num_steps=params.num_steps,
        params=params)

    step_id.assign_add(params.num_steps)
    replica_id_values = _local_state(strategy, state)[0]['replica_id'].numpy()
    logging.info('One cycle computation is done. Replicas are: %s',
                 str(replica_id_values))
    t1 = time.time()
    logging.info('Completed total %d steps (%d cycles) so far. Took %f secs '
                 'for the last cycle (%d steps).',
                 step_id_value(), cycle + 1, t1 - t0, params.num_steps)

    # Save checkpoint if the current step, from the start of the simulation,
    # is a multiple of the checkpoint interval, else just record, a possibly
    # shortened version of the current state.
    if (step_id_value() - params.start_step) % checkpoint_interval == 0:
      write_status = write_state(_local_state(strategy, state),
                                 step_id=step_id_value())
      replica_id_values = write_status[0]['replica_id'].numpy()
      logging.info('`Post cycle writing full state done. '
                   'Replicas are: %s', str(replica_id_values))
      ckpt_manager.save()
    else:
      # Note, the first time this is called retracing will occur for the
      # subgraphs in `distribted_write_state` if data_dump_filter is not `None`.
      write_status = write_state(_local_state(strategy, state),
                                 step_id=step_id_value(),
                                 data_dump_filter=data_dump_filter)
      replica_id_values = write_status[0]['replica_id'].numpy()
      logging.info('`Post cycle writing filtered state done. '
                   'Replicas are: %s', str(replica_id_values))
    t2 = time.time()
    logging.info('Writing output & checkpoint took %f secs.', t2 - t1)

  return strategy.experimental_local_results(state)
