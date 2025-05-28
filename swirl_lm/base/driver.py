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

"""Library for the incompressible structured mesh Navier-Stokes solver."""

import functools
import os
import time
from typing import Any, Optional, Sequence, TypeAlias, TypeVar, Union

from absl import flags
from absl import logging
import numpy as np
from swirl_lm.base import driver_tpu
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import target_flag  # pylint: disable=unused-import
from swirl_lm.boundary_condition import nonreflecting_boundary
from swirl_lm.core import simulation
from swirl_lm.linalg import poisson_solver
from swirl_lm.physics.lpt import lpt
from swirl_lm.physics.lpt import lpt_manager
from swirl_lm.physics.radiation import rrtmgp_common
from swirl_lm.utility import common_ops
from swirl_lm.utility import debug_output
from swirl_lm.utility import stretched_grid
from swirl_lm.utility import text_util
from swirl_lm.utility import tpu_util
from swirl_lm.utility import types
import tensorflow as tf


flags.DEFINE_integer(
    'min_steps_for_output',
    1,
    'Total number of steps before the output start to be generated.',
    allow_override=True,
)
flags.DEFINE_string(
    'data_dump_prefix',
    '/tmp/data',
    'The output `ser` or `h5` '
    'files prefix. This will be suffixed with the field '
    'components and step count.',
    allow_override=True,
)
flags.DEFINE_string(
    'data_load_prefix',
    '',
    'If non-empty, the input `ser` or `h5` '
    'files prefix from where the initial state will be loaded. This will be '
    'suffixed with the field components and step count. If set, the directory '
    'portion of the prefix has to be different from the directory portion of '
    '--data_dump_prefix.',
    allow_override=True,
)
RESTART_DUMP_CYCLE = flags.DEFINE_integer(
    'restart_dump_cycle',
    1,
    'The number of cycles between which full variables may be dumped to file.',
    allow_override=True,
)

SAVE_LAST_VALID_STEP = flags.DEFINE_bool(
    'save_last_valid_step',
    False,
    'If true, the solver will record and output the last non-NAN step before '
    'crashing. Default is `False` as this might have some performance impacts.',
    allow_override=True,
)

SAVE_MAX_UVW_AND_CFL = flags.DEFINE_bool(
    'save_max_uvw_and_cfl',
    False,
    'If true, the solver will record and output max values of abs(u)/dx, '
    'abs(v)/dy and abs(w)/dz as well as '
    'max(sum(abs(u)/dx + abs(v)/dy + abs(w)/dz)) for each timestep. The output '
    'will be written as a [num_steps, 4] tensor named max_uvw_cfl at the end '
    'of each cycle. Note that currently it is not possible to restart a '
    'simulation with a flipped value for this flag.',
    allow_override=True,
)

FLAGS = flags.FLAGS

CKPT_DIR_FORMAT = '{filename_prefix}-ckpts/'
COMPLETION_FILE = 'DONE'
_MAX_UVW_CFL = 'max_uvw_cfl'
TIME_VARNAME = simulation.TIME_VARNAME

Array: TypeAlias = Any
PerReplica: TypeAlias = tf.types.experimental.distributed.PerReplica
FlowFieldVal: TypeAlias = types.FlowFieldVal
T = TypeVar('T')
S = TypeVar('S')


class _NonRecoverableError(Exception):
  """Errors that cannot be recovered from via restarts, e.g., config issues."""

  pass


def _stateless_update_if_present(
    mapping: dict[T, S], updates: dict[T, S]
) -> dict[T, S]:
  """Returns a copy of `mapping` with only existing keys updated."""
  mapping = mapping.copy()
  mapping.update({key: val for key, val in updates.items() if key in mapping})
  return mapping


def _local_state_dict(
    strategy: tf.distribute.Strategy,
    distributed_state: dict[str, PerReplica],
) -> tuple[dict[str, tf.Tensor], ...]:
  return strategy.experimental_local_results(distributed_state)


def _local_state_value(
    strategy: tf.distribute.Strategy,
    distributed_state: PerReplica,
) -> tuple[PerReplica, ...]:
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
      output_dir, CKPT_DIR_FORMAT.format(filename_prefix=filename_prefix)
  )
  return tf.train.CheckpointManager(
      checkpoint, directory=checkpoint_dir, max_to_keep=3
  )


def _write_completion_file(output_dir: str) -> None:
  """Writes the completion file to the `output_dir`."""
  with tf.io.gfile.GFile(f'{output_dir}/{COMPLETION_FILE}', 'w') as f:
    f.write('')


def _get_state_keys(params: parameters_lib.SwirlLMParameters):
  """Returns essential, additional and helper var state keys."""
  # Essential flow field variables:
  # u: velocity in dimension 0;
  # v: velocity in dimension 1;
  # w: velocity in dimension 2;
  # p: pressure. p in LOW_MACH mode and p/ρ₀ in ANELASTIC mode.
  essential_keys = ['u', 'v', 'w', 'p'] + params.transport_scalars_names
  if params.solver_procedure == parameters_lib.SolverProcedure.VARIABLE_DENSITY:
    essential_keys += ['rho']
  additional_keys = list(
      params.additional_state_keys if params.additional_state_keys else []
  )
  helper_var_keys = list(
      params.helper_var_keys if params.helper_var_keys else []
  )

  additional_keys.append(TIME_VARNAME)

  # Add additional and helper_var keys required for stretched grids.
  stretched_grid_additional_keys, stretched_grid_helper_var_keys = (
      stretched_grid.additional_and_helper_var_keys(
          params.use_stretched_grid, params.use_3d_tf_tensor
      )
  )
  additional_keys.extend(stretched_grid_additional_keys)
  helper_var_keys.extend(stretched_grid_helper_var_keys)

  # Add additional keys required by the radiative transfer library.
  additional_keys += rrtmgp_common.required_keys(params.radiative_transfer)

  # Add additional keys required by the lagrangian particle tracking library.
  additional_keys += lpt.required_keys(params.lpt)

  # Check to make sure we don't have keys duplicating / overwriting each other.
  if len(set(essential_keys)) + len(set(additional_keys)) + len(
      set(helper_var_keys)
  ) != len(set(essential_keys + additional_keys + helper_var_keys)):
    raise _NonRecoverableError(
        'Duplicated keys detected between the three types of states: '
        f'essential states: {essential_keys}, additional states: '
        f'{additional_keys}, and helper vars: {helper_var_keys}'
    )

  for state_analytics_info in params.monitor_spec.state_analytics:
    for analytics_spec in state_analytics_info.analytics:
      helper_var_keys.append(analytics_spec.key)

  # Add helper variables required by the Poisson solver.
  helper_var_keys += poisson_solver.poisson_solver_helper_variable_keys(params)

  return essential_keys, additional_keys, helper_var_keys


def _init_fn(
    params: parameters_lib.SwirlLMParameters,
    customized_init_fn: Optional[types.InitFn] = None,
) -> types.InitFn:
  """Generates a function that initializes all variables."""

  def init_fn(
      replica_id: tf.Tensor,
      coordinates: types.ReplicaCoordinates,
  ) -> types.FlowFieldMap:
    """Initializes all variables required in the simulation."""
    states = {}


    # Add helper variables for stretched grids.
    states.update(
        stretched_grid.local_stretched_grid_vars_from_global_xyz(
            params, coordinates
        )
    )

    # Add helper variables from Poisson solver.
    poisson_solver_helper_var_fn = (
        poisson_solver.poisson_solver_helper_variable_init_fn(params)
    )
    if poisson_solver_helper_var_fn is not None:
      states.update(poisson_solver_helper_var_fn(replica_id, coordinates))

    # Helper variable for CFL tracking.
    if SAVE_MAX_UVW_AND_CFL.value:
      # Initialize max_uvw_cfl as a tensor of zeros of length equal to the
      # number of steps in a cycle. At the end of each step, we'll update the
      # entries corresponding to that step with the computed values. The
      # tensor will be written to disk together with the rest of the state
      # at the end of each cycle.
      states[_MAX_UVW_CFL] = tf.zeros(
          (params.num_steps, 4), dtype=types.TF_DTYPE
      )

    states[TIME_VARNAME] = tf.convert_to_tensor(0, dtype=tf.float64)

    if params.lpt is not None:
      states.update(lpt.init_fn(params))

    # Apply the user defined `init_fn` in the end to allow it to override
    # default initializations.
    if customized_init_fn is not None:
      states.update(customized_init_fn(replica_id, coordinates))

    states.update(nonreflecting_boundary.nonreflecting_bc_state_init_fn(params))
    return states

  return init_fn


def _get_model(kernel_op, params):
  """Returns the appropriate Navier-Stokes model from `params`."""
  if params.solver_procedure == parameters_lib.SolverProcedure.VARIABLE_DENSITY:
    return simulation.Simulation(kernel_op, params)
  raise _NonRecoverableError(
      f'Solver procedure not recognized: {params.solver_procedure}'
  )


def _process_at_step_id(
    process_fn,
    essential_states,
    additional_states,
    step_id,
    process_step_id,
    is_periodic,
):
  """Executes `process_fn` conditionally depending on `step_id`.

  Args:
    process_fn: Function accepting `essential_states` and `additional_states`,
      and returning the updated values of individual states in a dictionary.
    essential_states: The essential states, corresponds to the `states` keyword
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
  should_process = (
      (step_id % process_step_id == 0)
      if is_periodic
      else step_id == process_step_id
  )
  if should_process:
    updated_states = process_fn(
        states=essential_states, additional_states=additional_states
    )
    essential_states = _stateless_update_if_present(
        essential_states, updated_states
    )
    additional_states = _stateless_update_if_present(
        additional_states, updated_states
    )

  return essential_states, additional_states


def _update_additional_states(
    essential_states, additional_states, step_id, **common_kwargs
):
  """Updates additional_states."""
  # Perform the additional states update, if present.
  updated_additional_states = additional_states
  params = common_kwargs['params']

  # Clear source terms computed in the previous step.
  for varname in updated_additional_states:
    if not varname.startswith('src_'):
      continue
    zeros = tf.nest.map_structure(
        tf.zeros_like, updated_additional_states[varname]
    )
    updated_additional_states[varname] = zeros

  # Update BC additional states. Note currently this is only done
  # for the nonreflecting BC and will be  a no-op if there is no nonreflecting
  # BC present.
  with tf.name_scope('bc_additional_states_update'):
    updated_additional_states.update(
        nonreflecting_boundary.nonreflecting_bc_state_update_fn(
            states=essential_states,
            additional_states=additional_states,
            step_id=step_id,
            **common_kwargs,
        )
    )

  # Update lagrangian particle additional states.
  with tf.name_scope('lpt_additional_states_update'):
    lpt_field = lpt_manager.lpt_factory(params)
    if lpt_field is not None:
      updated_additional_states.update(
          lpt_field.step(
              replica_id=common_kwargs['replica_id'],
              replicas=common_kwargs['replicas'],
              states=essential_states,
              additional_states=additional_states,
              step_id=step_id,
          ),
      )

  if params.additional_states_update_fn is not None:
    with tf.name_scope('additional_states_update'):
      updated_additional_states = params.additional_states_update_fn(
          states=essential_states,
          additional_states=additional_states,
          step_id=step_id,
          **common_kwargs,
      )

  return updated_additional_states


def _state_has_nan_inf(state: dict[str, PerReplica], replicas: Array) -> bool:
  """Checks whether any field in the `state` contains `nan` or `inf`."""
  local_has_nan_inf = False
  for _, v in state.items():
    if (v.dtype.is_floating and tf.reduce_any(
        tf.math.logical_or(tf.math.is_nan(v), tf.math.is_inf(v)))):
      local_has_nan_inf = True
      # Graph compilation doesn't allow early break.

  # We want to make sure every core is seeing the same problem so the
  # termination of all cores is synchronized, thus a global reduce operation
  # is needed.
  if common_ops.global_reduce(
      tf.convert_to_tensor(local_has_nan_inf),
      tf.reduce_any,
      replicas.reshape([1, -1]),
  ):
    return True
  else:
    return False


def _compute_max_uvw_and_cfl(
    state: dict[str, PerReplica],
    grid_spacings: tuple[FlowFieldVal, FlowFieldVal, FlowFieldVal],
    replicas: Array,
) -> tf.Tensor:
  """Computes global maximum values of abs(u_i)/d_i and sum(abs(u_i)/d_i).

  Maximum value (across cells) of sum(abs(u_i)/d_i) is used in computing CFL as
  in:

  https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition#The_two_and_general_n-dimensional_case

  Args:
    state: State dictionary container per-replica values.
    grid_spacings: Physical grid spacing as 1D field for each dimension.
    replicas: Mapping from grid coordinates to replica id numbers.

  Returns:
    A tensor of length 4 with maximum values for abs(u)/dx, abs(v)/dy and
    abs(w)/dz, and sum(abs(u_i)/d_i).
  """
  abs_divide = lambda a, b: common_ops.map_structure_3d(
      lambda c, d: tf.math.abs(c) / d, a, b
  )

  out = []
  for u, d in zip(['u', 'v', 'w'], grid_spacings):
    out.append(
        common_ops.global_reduce(
            abs_divide(state[u], d),
            tf.math.reduce_max,
            replicas.reshape([1, -1]),
        )
    )

  dx, dy, dz = grid_spacings
  out.append(
      common_ops.global_reduce(
          (
              abs_divide(state['u'], dx)
              + abs_divide(state['v'], dy)
              + abs_divide(state['w'], dz)
          ),
          tf.math.reduce_max,
          replicas.reshape([1, -1]),
      )
  )
  return tf.convert_to_tensor(out)


@tf.function
def _one_cycle(
    strategy: tf.distribute.Strategy,
    init_state: dict[str, PerReplica],
    init_step_id: Array,
    num_steps: Array,
    params: Union[parameters_lib.SwirlLMParameters, Any],
    model: Any,
) -> tuple[dict[str, PerReplica], PerReplica, dict[str, PerReplica],
           PerReplica]:
  """Runs one cycle of the Navier-Stokes solver.

  Args:
    strategy: A distributed strategy to run computations. Currently, only
      `TPUStrategy` is supported.
    init_state: The initial state on each device.
    init_step_id: An integer scalar denoting the initial step id at which to
      start the cycle.
    num_steps: An integer scalar; the number of steps to run in the cycle.
    params: The solver configuration.
    model: The underlying model where the simulation calculation procedures are
      defined.

  Returns:
    A (final_state, completed_steps, previous_state, has_non_finite) where
    final_state is the state at the end of the cycle, completed_steps is the
    number of steps completed in this cycle, previous_state is the state one
    step before the final_state (which is useful in blow-ups), and
    has_non_finite is a boolean indicating whether the final_state contains
    non-finite values.
  """
  logging.info(
      'Tracing and compiling of _one_cycle starts. This can take up to 30 min.'
  )
  essential_keys, additional_keys, helper_var_keys = _get_state_keys(params)

  init_state_keys = set(init_state.keys())
  keys_declared_in_params = (
      set(essential_keys) | set(additional_keys) | set(helper_var_keys)
  )
  logging.info(
      'Keys in init_state but not in params: %s',
      sorted(init_state_keys - keys_declared_in_params),
  )
  logging.info(
      'Keys in params but not in init_state: %s',
      sorted(keys_declared_in_params - init_state_keys),
  )

  computation_shape = np.array([params.cx, params.cy, params.cz])
  logical_replicas = np.arange(
      strategy.num_replicas_in_sync, dtype=np.int32
  ).reshape(computation_shape)

  kernel_op = params.kernel_op

  def step_fn(state):
    # Common keyword arguments to various step functions.
    common_kwargs = dict(
        kernel_op=kernel_op,
        replica_id=state['replica_id'],
        replicas=logical_replicas,
        params=params,
    )
    keys_to_split = set.union(set(essential_keys), set(additional_keys)) - set(
        helper_var_keys
    )
    if not params.use_3d_tf_tensor:
      # Split essential/additional states into lists of 2D Tensors.
      for key in keys_to_split:
        state[key] = tf.unstack(state[key])

    if SAVE_MAX_UVW_AND_CFL.value:
      state[_MAX_UVW_CFL] = tf.zeros_like(state[_MAX_UVW_CFL])

    cycle_step_id = 0
    prev_state = state
    has_non_finite = False
    for _ in tf.range(num_steps):
      step_id = init_step_id + cycle_step_id
      # Split the state into essential states and additional states. Note that
      # `additional_states` consists of both additional state keys and helper
      # var keys.
      additional_states = dict(
          (key, state[key]) for key in additional_keys + helper_var_keys
      )
      essential_states = dict((key, state[key]) for key in essential_keys)

      # Perform a preprocessing step, if configured.
      if params.apply_preprocess:
        with tf.name_scope('preprocess_update'):
          essential_states, additional_states = _process_at_step_id(
              process_fn=functools.partial(
                  params.preprocessing_states_update_fn, **common_kwargs
              ),
              essential_states=essential_states,
              additional_states=additional_states,
              step_id=step_id,
              process_step_id=params.preprocess_step_id,
              is_periodic=params.preprocess_periodic,
          )

      # Perform additional_states update.
      additional_states.update(
          _update_additional_states(
              essential_states=essential_states,
              additional_states=additional_states,
              step_id=step_id,
              **common_kwargs,
          )
      )

      # Perform one step of the Navier-Stokes model. The updated state should
      # contain both the essential and additional states.
      updated_state = model.step(
          replica_id=state['replica_id'],
          replicas=logical_replicas,
          step_id=step_id,
          states=essential_states,
          additional_states=additional_states,
      )

      # Perform a postprocessing step, if configured.
      if params.apply_postprocess:
        with tf.name_scope('postprocess_update'):
          # Split the updated_state into essential states and additional states.
          additional_states = _stateless_update_if_present(
              additional_states, updated_state
          )
          essential_states = _stateless_update_if_present(
              essential_states, updated_state
          )

          essential_states, additional_states = _process_at_step_id(
              process_fn=functools.partial(
                  params.postprocessing_states_update_fn, **common_kwargs
              ),
              essential_states=essential_states,
              additional_states=additional_states,
              step_id=step_id,
              process_step_id=params.postprocess_step_id,
              is_periodic=params.postprocess_periodic,
          )

          # Merge the essential states and additional states into updated_state.
          updated_state = _stateless_update_if_present(
              updated_state, additional_states
          )
          updated_state = _stateless_update_if_present(
              updated_state, essential_states
          )

      if SAVE_MAX_UVW_AND_CFL.value:
        with tf.name_scope('retrieve_grid_spacings'):
          grid_spacings_1d: tuple[FlowFieldVal, FlowFieldVal, FlowFieldVal] = (
              tuple(
                  params.physical_grid_spacing(
                      dim, params.use_3d_tf_tensor, additional_states
                  )
                  for dim in (0, 1, 2)
              )
          )
        with tf.name_scope('save_max_cfl'):
          updated_state[_MAX_UVW_CFL] = tf.tensor_scatter_nd_add(
              state[_MAX_UVW_CFL],
              tf.convert_to_tensor([
                  [cycle_step_id, 0],
                  [cycle_step_id, 1],
                  [cycle_step_id, 2],
                  [cycle_step_id, 3],
              ]),
              _compute_max_uvw_and_cfl(
                  updated_state, grid_spacings_1d, logical_replicas
              ),
          )

      # Simulation time will accumulate precision errors with this approach.
      # For now, the converter deals with this by rounding off to a fewer
      # number of significant digits.
      updated_state[TIME_VARNAME] = state[TIME_VARNAME] + params.dt64

      prev_state = state
      # Some state keys such as `replica_id` may not lie in either of the three
      # categories. Just pass them through.
      state = _stateless_update_if_present(state, updated_state)
      cycle_step_id += 1
      if SAVE_LAST_VALID_STEP.value:
        with tf.name_scope('save_last_valid_step'):
          if _state_has_nan_inf(state, logical_replicas):
            # Detected nan/inf, skip the update of state by early-exiting from
            # the for loop.
            has_non_finite = True
            break

    if not params.use_3d_tf_tensor:
      # Unsplit the keys that were previously split.
      for key in keys_to_split:
        state[key] = tf.stack(state[key])

    return state, cycle_step_id, prev_state, has_non_finite

  return strategy.run(step_fn, args=(init_state,))


def get_strategy_and_coordinates(params: parameters_lib.SwirlLMParameters):
  """Prepares the TPU strategy and the coordinates of the partition."""
  computation_shape = np.array([params.cx, params.cy, params.cz])
  logging.info('Computation_shape is %s', str(computation_shape))
  strategy = driver_tpu.initialize_tpu(
      tpu_address=FLAGS.target, computation_shape=computation_shape
  )
  num_replicas = strategy.num_replicas_in_sync
  logging.info('TPU is initialized. Number of replicas is %d.', num_replicas)
  logical_coordinates = tpu_util.grid_coordinates(computation_shape).tolist()
  return strategy, logical_coordinates


def get_init_state(
    customized_init_fn: Union[types.InitFn, Any],
    strategy: tf.distribute.TPUStrategy,
    params: parameters_lib.SwirlLMParameters,
    logical_coordinates: Array,
) -> dict[str, PerReplica]:
  """Creates the initial state using `customized_init_fn`."""
  t_start = time.time()

  init_fn = _init_fn(params, customized_init_fn)
  # Wrapping `init_fn` with tf.function so it is not retraced unnecessarily for
  # every core/device.
  state = driver_tpu.distribute_values(
      strategy,
      value_fn=tf.function(init_fn),
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


def solver(
    customized_init_fn: Union[types.InitFn, Any],
    params: Optional[Union[parameters_lib.SwirlLMParameters, Any]] = None,
):
  """Runs the Navier-Stokes Solver with TF2 Distribution strategy.

  Initialization of `state`:

     1) The solver initializes `state` with internal variables (e.g., those
        needed by the poisson solver). Additional variables are added by the
        custom initialization function passed into the solver. The solver then
        adds variables related to nonreflecting boundaries.

    2a) If a checkpoint is found, `state` is updated with values read from the
        checkpoint. Only the variables already initialized in step 1 above will
        be loaded from the checkpoint. The simulation will fail with an error if
        any variable is not found in the checkpoint (which could happen if code
        or config is modified before a restart). The data in the checkpoint will
        override the values from step 1.

    2b) If no checkpoint is found and the flags `--data_load_prefix` and
        `--loading_step` are specified, some variables are loaded from the given
        step. The set of variables to be loaded is determined by the repeated
        field `states_from_file` in the config. If `states_from_file` is empty,
        the set of variables initialized in step 1 is loaded from the loading
        step overwriting the initialization values; otherwise only the subset in
        `states_from_file` is loaded. The simulation will fail with an error if
        any variable that needs to be loaded is not found in the loading
        step. Note that if a simulation is restarted with `--loading_step`
        (either manually or because of a preemption), the new run will start
        from the checkpoint saved in the initial run and will not load data
        again from `--data_load_prefix`.

    2c) If no checkpoint is found and `--data_load_prefix` is not specified,
        then the simulation starts with variables initialized in step 1 above.

  Saving of `state` to disk:

    `state` is saved to disk at the end of each cycle except that in a newly
    started simulation (i.e., that is not restarting from a checkpoint), it is
    also saved after initialization just before starting the first cycle.

    Some (by default all) cycles are saved as checkpoints. At a checkpoint,
    all variables in `state` are saved so that the simulation can be restarted
    from the saved data. How often this happens is controlled by the flag
    `--restart_dump_cycle`, e.g., setting it to 3 will cause every third cycle
    to be saved as a checkpoint. The initial `state` is always saved as a
    checkpoint.

    If a cycle is not a checkpoint, then only some of the variables will be
    saved. The set of variables that are saved are determined by the repeated
    field `states_to_file` in the config.

    When a simulation is started with a `--loading_step`, the initial run is
    considered a new simulation and will save its `state` after initialization
    as a checkpoint.

  Categories of tensors in `state`:

    The tensors in `state` consist of 3 categories, which `driver._one_cycle()`
    treats differently. There are `essential_keys`, `additional_keys`, and
    `helper_var_keys`.  Tensors corresponding to `essential_keys` and
    `additional_keys` are unstacked from tensors into lists of tensors at the
    beginning of a cycle and restacked into tensors at the end of a cycle.
    Tensors corresponding to `helper_var_keys` are left as is.

    When passing the tensors to the simulation model's step function, the
    tensors corresponding to `essential_keys` are put into one dictionary,
    whereas the tensors corresponding to `additional_keys` and `helper_var_keys`
    are put into a separate dictionary.

  Termination:

    In the normal case, the solver will return after it has run the simulation
    for the requested number of cycles. The output directory will contain data
    for `num_cycles + 1` steps, more specifically, for step numbers [0, 1 *
    num_steps, 2 * num_steps, ..., num_cycles * num_steps].

    If the simulation reaches a state where any variable has a non-finite value
    (NaN or Inf), then the simulation will stop early and save both the
    non-finite state and the one before it. As a result, there will most likely
    be fewer steps saved than `num_cycles + 1` and the final saved step number
    will not necessarily be a multiple of num_steps.

    In both of the these cases (`num_cycles` reached or non-finite value seen),
    the solver will write an empty `DONE` file to the output directory to
    indicate that the simulation is complete.

    The simulation can also terminate by raising an exception (e.g., because of
    input errors, resource issues, etc.). In this case, the solver will exit
    before writing an empty `DONE` file.

    In case of restarts (e.g., by increasing `num_cycles` to continue a
    simulation), the `DONE` file should be removed before starting the solver.

  Args:
    customized_init_fn: The function that initializes the flow field. The
      function needs to be replica dependent.
    params: An instance of parameters that will be used in the simulation, e.g.
      the mesh size, fluid properties.

  Returns:
    A tuple of the final state on each replica.
  """
  try:
    if params is None:
      params = parameters_lib.params_from_config_file_flag()
    params.save_to_file(FLAGS.data_dump_prefix)

    # Initialize the TPU.
    strategy, logical_coordinates = get_strategy_and_coordinates(params)
    init_state = get_init_state(
        customized_init_fn, strategy, params, logical_coordinates
    )
    debug_output.initialize(params, strategy)
    return solver_loop(strategy, logical_coordinates, init_state, params)
  except _NonRecoverableError:
    logging.exception(
        'Non-recoverable error in solve - returning None '
        'instead of raising an exception to avoid automatic '
        'restarts.'
    )


def solver_loop(
    strategy: tf.distribute.TPUStrategy,
    logical_coordinates: Array,
    init_state: dict[str, PerReplica],
    params: Union[parameters_lib.SwirlLMParameters, Any],
):
  """Runs the solver on an initialized TPU system starting with `init_state`."""
  logging.info('Entering solver_loop.')
  t_pre_restore = time.time()

  num_replicas = strategy.num_replicas_in_sync
  # In order to save and restore from the filesystem, we use tf.train.Checkpoint
  # on the step id. The `step_id` Variable should be placed on the TPU device so
  # that we don't block TPU execution when writing the state to filenames
  # formatted dynamically with the step id.
  with strategy.scope():
    step_id = tf.Variable(params.start_step, dtype=tf.int32)
  output_dir, filename_prefix = os.path.split(FLAGS.data_dump_prefix)

  if FLAGS.data_load_prefix:
    input_dir, input_filename_prefix = os.path.split(FLAGS.data_load_prefix)
    # This is a potential user error where both read and output directories
    # are pointing to the same place. We do not allow this to happen as some
    # files could get overwritten and it can be very confusing.
    if input_dir == output_dir:
      raise _NonRecoverableError(
          'Please check your configuration. The loading directory is '
          f'set to be the same as the output directory {output_dir}, this '
          'will cause confusion and potentially over-write important data. '
          'If you are trying to continue the simulation run using a previous '
          'simulation step from a different run, please use a separate '
          'output directory. To have a separate output directory, the '
          'directory portions of --data_load_prefix and --data_dump_prefix '
          'need to be different.'
      )
    # If a loading directory is specified, we check if the step directory to
    # read from exists. The step id for data to read from the input directory is
    # provided by the `loading_step` in `params`. If it exists, we *assume* the
    # needed files are there and will proceed to read (if it turns out files are
    # missing, the job will just fail).
    loading_subdir = os.path.join(input_dir, str(params.loading_step))
    states_from_file = list(params.states_from_file)
    if not tf.io.gfile.exists(loading_subdir):
      raise _NonRecoverableError(
          f'--data_load_prefix was set to {FLAGS.data_load_prefix} and '
          f'loading step is {params.loading_step} but no restart files are '
          f'found in {loading_subdir}.'
      )
    read_state_from_input_dir = functools.partial(
        driver_tpu.distributed_read_state,
        strategy,
        logical_coordinates=logical_coordinates,
        output_dir=input_dir,
        filename_prefix=input_filename_prefix,
        states_from_file=states_from_file,
    )
    logging.info('read_state_from_input_dir function created.')
  else:
    input_dir = None
    read_state_from_input_dir = None

  logging.info('Getting checkpoint_manager.')
  ckpt_manager = get_checkpoint_manager(
      step_id=step_id, output_dir=output_dir, filename_prefix=filename_prefix
  )

  # Check if only part of the data will be dumped.
  data_dump_filter = list(params.states_to_file)
  checkpoint_interval = RESTART_DUMP_CYCLE.value * params.num_steps
  logging.info('Got checkpoint_manager.')

  # Use this to access step_id, instead of directly using tf.Variable, which
  # will trigger the need for synchronization and will slow down/dead-lock for
  # multi-devices I/O.
  def step_id_value():
    return tf.constant(step_id.numpy(), tf.int32)

  def write_state_and_sync(
      state: dict[str, PerReplica],
      step_id: Array,
      data_dump_filter: Optional[Sequence[str]] = None,
      allow_non_finite_values: bool = False,
      use_zeros_for_debug_values: bool = False,
  ):
    if use_zeros_for_debug_values:
      debug_vars = debug_output.zeros_like_vars(strategy, state.keys())
    else:
      debug_vars = debug_output.get_vars(strategy, state.keys())

    write_state = dict(state) | debug_vars
    if allow_non_finite_values:
      fields_allowing_non_finite_values = list(write_state.keys())
    else:
      fields_allowing_non_finite_values = list(debug_vars.keys())
    write_status = driver_tpu.distributed_write_state(
        strategy,
        _local_state_dict(strategy, write_state),
        logical_coordinates=logical_coordinates,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        step_id=step_id,
        data_dump_filter=data_dump_filter,
        fields_allowing_non_finite_values=fields_allowing_non_finite_values,
    )

    # This will block until all replicas are done writing.
    replica_id_write_status = []
    for i in range(num_replicas):
      replica_id_write_status.append(write_status[i]['replica_id'].numpy())
    return replica_id_write_status

  read_state = functools.partial(
      driver_tpu.distributed_read_state,
      strategy,
      logical_coordinates=logical_coordinates,
      output_dir=output_dir,
      filename_prefix=filename_prefix,
  )
  logging.info('read_state function created.')

  state = init_state
  write_initial_state = False

  # Restore from an existing checkpoint if present.
  if ckpt_manager.latest_checkpoint:
    # The checkpoint restore updates the `step_id` variable; which is then used
    # to read in the state.
    logging.info('Detected checkpoint. Starting `restore_or_initialize`.')
    if input_dir is not None:
      logging.info(
          '--data_load_prefix was set to %s but not using it '
          'because checkpoint was detected in the data dump '
          'directory %s.',
          FLAGS.data_load_prefix,
          FLAGS.data_dump_prefix,
      )
    ckpt_manager.restore_or_initialize()
    state = read_state(
        state=_local_state_dict(strategy, state), step_id=step_id_value()
    )
    # This is to sync the client code to the worker execution.
    replica_id_values = state['replica_id'].values
    logging.info(
        '`restoring-checkpoint-if-necessary` stage '
        'done with reading checkpoint. Replicas are: %s',
        str(replica_id_values),
    )
  # Override initial state with state from a previous run if requested.
  elif input_dir is not None:
    logging.info(
        '--data_load_prefix is set to %s, loading from %s at step %s, '
        'and overriding the default initialized state.',
        FLAGS.data_load_prefix,
        input_dir,
        params.loading_step,
    )
    state = read_state_from_input_dir(
        state=_local_state_dict(strategy, state),
        step_id=tf.constant(params.loading_step),
    )
    write_initial_state = True
    # This is to sync the client code to the worker execution.
    replica_id_values = state['replica_id'].values
    logging.info(
        '`restoring-checkpoint-if-necessary` stage '
        'done with reading from load directory. Replicas are: %s',
        str(replica_id_values),
    )
    logging.info('Read states from %s at %i', input_dir, params.loading_step)
  # Use default initial state.
  else:
    write_initial_state = True
    logging.info(
        'No checkpoint was found and --data_load_prefix was not set. '
        'Proceeding with default initializations for all variables.'
    )

  if write_initial_state:
    logging.info('Starting `write_state` for the initial state.')
    write_status = write_state_and_sync(state=state, step_id=step_id_value())
    logging.info(
        '`restoring-checkpoint-if-necessary` stage '
        'done with writing initial steps. Write status: %s',
        write_status,
    )
    # Only after the logging, which forces the `write_status` to be
    # materialized, we can guarantee that the actual write actions are
    # completed, and we update the saved completed step here.
    ckpt_manager.save()

  t_post_restore = time.time()
  logging.info(
      'restore-if-necessary-or-write took %s.',
      text_util.seconds_to_string(t_post_restore - t_pre_restore),
  )

  if params.num_steps < 0:
    raise _NonRecoverableError(
        '`num_steps` should not be negative but it was set to'
        f' {params.num_steps}.'
    )
  if (
      params.num_steps > 0
      and (step_id_value() - params.start_step) % params.num_steps != 0
  ):
    raise _NonRecoverableError(
        'Incompatible step_id detected. `step_id` is expected '
        'to be `start_step` + N * `num_steps` but (step_id: {}, '
        'start_step: {}, num_steps: {}) is detected. Maybe the '
        'checkpoint step is inconsistent?'.format(
            step_id_value(), params.start_step, params.num_steps
        )
    )
  logging.info(
      'Simulation iteration starts. Total %d steps, starting from %d, '
      'with %d steps per cycle.',
      params.num_steps * params.num_cycles,
      step_id_value(),
      params.num_steps,
  )

  # Get the model that defines the concrete simulation calculation procedures.
  # Since we are allowing some model object's methods to be decorated with
  # `tf.function`, calling `_get_model` outside the loop ensures that these
  # methods are traced only once.
  model = _get_model(params.kernel_op, params)

  while step_id_value() < (
      params.start_step + params.num_steps * params.num_cycles
  ):
    cycle = (step_id_value() - params.start_step) // params.num_steps
    logging.info('Step %d (cycle %d) is starting.', step_id_value(), cycle)
    t0 = time.time()
    state, num_steps_completed, prev_state, has_non_finite = _one_cycle(
        strategy=strategy,
        init_state=state,
        init_step_id=step_id_value(),
        num_steps=params.num_steps,
        params=params,
        model=model,
    )
    # num_steps_completed and has_non_finite are guaranteed to be identical for
    # all replicas, so we are just taking replica 0 value.
    with tf.name_scope('check_states_validity'):
      num_steps_completed = _local_state_value(
          strategy, num_steps_completed)[0].numpy()
      has_non_finite = _local_state_value(
          strategy, has_non_finite)[0].numpy()

    step_id.assign_add(num_steps_completed)

    if SAVE_MAX_UVW_AND_CFL.value:
      with tf.name_scope('print_max_cfl'):
        # CFL number is guaranteed to be identical for all replicas, so take
        # replica 0 value.
        cfl_values = (
            params.dt
            * _local_state_value(strategy, state[_MAX_UVW_CFL])[0].numpy()[:, 3]
        )
        max_cfl_number_from_cycle = tf.reduce_max(cfl_values)
        cfl_number_from_last_step = cfl_values[num_steps_completed - 1]
        logging.info(
            'max CFL number from last cycle: %.3f.  CFL number from last step:'
            ' %.3f',
            max_cfl_number_from_cycle,
            cfl_number_from_last_step,
        )

    # If we just attempted the first cycle, log information about available
    # debug variables to help with debugging.
    with tf.name_scope('log_debug_variable'):
      if cycle == 0:
        debug_output.log_variable_use()

    # Check if we did not complete a full cycle.
    if has_non_finite:
      with tf.name_scope('logging_non_finite_states'):
        logging.info(
            'Non-convergence detected. Early exit from cycle %d at step %d.',
            cycle, step_id_value())
        if num_steps_completed > 1:
          write_status = write_state_and_sync(prev_state,
                                              step_id=step_id_value() - 1,
                                              use_zeros_for_debug_values=True)
          logging.info(
              'Dumping last valid state at step %d done. Write status: %s',
              step_id_value() - 1, write_status)
        write_status = write_state_and_sync(state, step_id=step_id_value(),
                                            allow_non_finite_values=True)
        logging.info(
            'Dumping final non-finite state done. Write status: %s',
            write_status
        )
        # Save checkpoint to update the completed step.
        # Note: Only after the logging, which forces the `write_status` to be
        # materialized, we can guarantee that the actual write actions are
        # completed, and we update the saved completed step here.
        ckpt_manager.save()
        # Wait for checkpoint manager before marking completion.
        ckpt_manager.sync()
        # Mark simulation as complete.
        _write_completion_file(output_dir)
        raise _NonRecoverableError(
            f'Non-convergence detected. Early exit from cycle {cycle} at step'
            f' {step_id_value()}. The last valid state at step'
            f' {step_id_value() - 1} has been saved in the specified output'
            ' path.'
        )

    # Consider explicitly deleting prev_state here to free its memory because
    # after its written to disk it is no longer needed.
    with tf.name_scope('logging_time_info'):
      replica_id_values = []
      replica_id_values.extend(
          _local_state_value(strategy, state['replica_id'])
      )
      logging.info(
          'One cycle computation is done. Replicas are: %s',
          str([v.numpy() for v in replica_id_values]),
      )
      t1 = time.time()
      logging.info(
          'Completed total %d steps (%d cycles, %s simulation time) so far. '
          'Took %s for the last cycle (%d steps).',
          step_id_value(),
          cycle + 1,
          text_util.seconds_to_string(
              int(step_id_value()) * params.dt64, precision=params.dt64
          ),
          text_util.seconds_to_string(t1 - t0),
          params.num_steps,
      )

    # Save checkpoint if the current step, from the start of the simulation,
    # is a multiple of the checkpoint interval, else just record, a possibly
    # shortened version of the current state.
    with tf.name_scope('writing_checkpoints'):
      if (step_id_value() - params.start_step) % checkpoint_interval == 0:
        write_status = write_state_and_sync(
            state=state, step_id=step_id_value()
        )
        logging.info(
            '`Post cycle writing full state done. Write status: %s',
            write_status,
        )
        # Only after the logging, which forces the `write_status` to be
        # materialized, we can guarantee that the actual write actions are
        # completed, and we update the saved completed step here.
        ckpt_manager.save()
      else:
        # Note, the first time this is called retracing will occur for the
        # subgraphs in `distributed_write_state` if data_dump_filter is not
        # `None`.
        write_status = write_state_and_sync(
            state=state,
            step_id=step_id_value(),
            data_dump_filter=data_dump_filter,
        )
        logging.info(
            '`Post cycle writing filtered state done. Write status: %s',
            write_status,
        )
      t2 = time.time()
      logging.info(
          'Writing output & checkpoint took %s.',
          text_util.seconds_to_string(t2 - t1),
      )

  # Wait for checkpoint manager before marking completion.
  ckpt_manager.sync()
  _write_completion_file(output_dir)

  return strategy.experimental_local_results(state)
