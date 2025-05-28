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

r"""Library to help save arbitrary tensor values as solver output.

This library allows arbitrary values to be saved as if they were part of the
solver's state but without having to add plumbing to actually add them to the
state. It does this by saving values in distributed TF Variables and then
moving these values to the solver's state just before the state is saved.

To use the library:

1) Modify the code where you want to save the value of some computation.
   For example to save the computed `q_v`:

   from swirl_lm.utility import debug_output
   ...
   q_v = tf.nest.map_structure(
       tf.math.subtract, thermo_states['q_t'], thermo_states['q_c']
   )
   debug_output.dump_value('debug_q_v_in_one_moment', q_v)

   By convention, the name should start with the prefix 'debug_', but this is
   not enforced. If the name collides with an existing state name, an assertion
   error will be raised - so use 'debug_' as a prefix.

   If dump_value() is called multiple times with the same name (i.e., within a
   python loop or because the enclosing function, e.g., update_step is called
   multiple times), then the second and consecutive calls will attach a unique
   integer to the name. For example, the call above will output
   'debug_q_v_in_one_moment', 'debug_q_v_in_one_moment_1', and
   'debug_q_v_in_one_moment_2'.

2) Modify the SwirlLMParameters config to add the name of the variable(s) (e.g.,
   'debug_q_v_in_one_moment', 'debug_q_v_in_one_moment_2') to the
   `debug_variables` repeated field.

   If you don't know if your new code will be called multiple times, you can
   just run the simulation without updating SwirlLMParameters and search for
   'Available debug variables' in the logs which will be output during graph
   tracing. The message will list all of the variables including the integer
   suffixes.

If the debug output can potentially be useful in the future, consider submitting
the debug output code. If you do this, also update the documentation for the
list of available debug variables in
swirl_lm/base/parameters.proto.

The debug values are saved once at the end of each cycle. It is fine to update
the same variable multiple times during a cycle or even a step (e.g., inside a
function called from a tf.while loop), but only the last value will be saved to
disk.

If a simulation exits early by reaching a non-finite state, then the debug
variables in the state prior to the non-finite state will be all zeros. This is
because the values for that step will have been overwritten with the values
stored in computing the non-finite state, so zeros are output instead.

Unlike normal simulation output, debug output can contain non-finite values. For
example, if an intermediate value has NaNs in the halo, is output as debug,
and the halo later gets overwritten, the debug output will contain these NaNs.
"""
import collections
from collections.abc import Iterable
import logging

from swirl_lm.base import parameters
from swirl_lm.utility import types
import tensorflow as tf

_VARS = {}
_COUNT_BY_NAME_PROVIDED_BY_CODE = collections.Counter()


def initialize(
    params: parameters.SwirlLMParameters, strategy: tf.distribute.TPUStrategy
) -> None:
  """Creates distributed Variables for debug variables listed in `params`."""
  with strategy.scope():
    for name in params.debug_variables:
      _VARS[name] = tf.Variable(
          tf.zeros((params.nz, params.nx, params.ny), dtype=tf.float32),
          trainable=False,
      )


def _unique_var_name(name: str, index: int) -> str:
  return name if index == 0 else f'{name}_{index}'


def _unique_var_names_from_counts(count_by_name: dict[str, int]) -> set[str]:
  out = set()
  for name, count in count_by_name.items():
    for i in range(count):
      out.add(_unique_var_name(name, i))
  return out


def dump_value(name: str, x: types.FlowFieldVal) -> None:
  """Saves `x` under a key generated from `name`.

  The value is only saved if the key generated from `name` is listed as one of
  the 'debug_variables' in SwirlLMParameters.

  If dump_value() is called multiple times with the same `name`, the second and
  consecutive calls will generate unique keys from `name` by appending integer
  suffixes. For example the key will be 'debug_x_1' when dump_value() is called
  a second time with 'debug_x'.

  Args:
    name: The name from which the key will be generated.
    x: The value to save.
  """
  var_name = _unique_var_name(name, _COUNT_BY_NAME_PROVIDED_BY_CODE[name])
  _COUNT_BY_NAME_PROVIDED_BY_CODE[name] += 1
  if var_name not in _VARS:
    return
  _VARS[var_name].assign(x)


def is_debug_enabled(name: str) -> bool:
  return name in _VARS


def get_vars(
    strategy: tf.distribute.TPUStrategy,
    disallowed_var_names: Iterable[str]
) -> dict[str, tf.distribute.DistributedValues]:
  """Returns a copy of debug variables as a dictionary of DistributedValues."""

  # We explicitly "copy" Variables into a dictionary via a tf.function called in
  # the context of the TPUStrategy because directly saving Variables with
  # tpu_driver.distributed_write_state causes the Variables to be synchronized
  # on one host. More technically, calling strategy.experimental_local_results()
  # on Variables appears to cause synchronization either immediately or
  # downstream. The explicit copying is distributed because it's run with
  # TPUStrategy.
  @tf.function
  def _f():
    return dict(_VARS)

  debug_vars = strategy.run(_f)
  duplicate_names = set(debug_vars) & set(disallowed_var_names)
  assert (
      not duplicate_names
  ), f'Debug variables {duplicate_names} conflict with other variables.'
  return debug_vars


def zeros_like_vars(
    strategy: tf.distribute.TPUStrategy,
    disallowed_var_names: Iterable[str]
) -> dict[str, tf.distribute.DistributedValues]:
  """Like get_vars() but returns zeros."""
  @tf.function
  def _f():
    return {k: tf.zeros_like(v) for k, v in _VARS.items()}

  debug_vars = strategy.run(_f)
  duplicate_names = set(debug_vars) & set(disallowed_var_names)
  assert (
      not duplicate_names
  ), f'Debug variables {duplicate_names} conflict with other variables.'
  return debug_vars


def log_variable_use() -> None:
  """Logs messages about variable use."""
  provided_names = _unique_var_names_from_counts(
      _COUNT_BY_NAME_PROVIDED_BY_CODE)
  requested_but_not_written = set(_VARS) - provided_names
  if requested_but_not_written:
    logging.warning(
        'Configuration has debug variables %s that are not provided by the '
        'solver!',
        requested_but_not_written,
    )
  logging.info('Available debug variables are: %s', provided_names)
