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

"""A utility to print debug information from TPU to stderr stream.

For performance consideration, the functionality is guarded behind a
flag. More sophisticated mechanism can be added.
"""
import enum
import inspect
import sys
from typing import Any, Optional

from absl import flags
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal


class LogLevel(enum.Enum):
  INFO = 1
  WARNING = 2
  ERROR = 3
  FATAL = 4


_LOG_LEVEL = flags.DEFINE_enum(
    'swirl_debug_log_level', None,
    [level.name for level in LogLevel],
    'Debug logging level. If the flag is not set (None), the debugg logging '
    'is disabled. Allowed levels (in the ascending order of severity) are: '
    '`INFO`, `WARNING`, `ERROR`, and `FATAL`. When set, the logging that is '
    'at or above the set severity level will be enabled.')


def log_enabled(log_level):
  if _LOG_LEVEL.value is None:
    return False
  log_flag_level = LogLevel[_LOG_LEVEL.value].value
  return log_level.value >= log_flag_level


def _get_log_msg(log_level, message, stack_info=True):
  frame = inspect.stack()[2]
  filename = frame.filename
  line = frame.lineno
  log_level_name = log_level.name
  message = (f'TPU debug logging level: {log_level_name}. File: {filename} '
             f'at line: {line}, message: {message}') if stack_info else (
                 f'TPU debug: {message}')
  return message


def log(
    t: FlowFieldVal,
    message: Optional[str] = '',
    log_level: LogLevel = LogLevel.INFO,
    summarize: Optional[int] = None,
    stack_info: bool = True,
) -> Any:
  """Logging function for debug information that works on TPU.

  Args:
    t: The `tf.Tensor` to be logged. This can be any `tf.Tensor`.
    message: The optional message to be logged together with `t`.
    log_level: The logging level that indicates the severity of the message.
    summarize: If not specified, the first 3 and the last 3 elements of the
      tensor in each dimension will be printed recursively. When specified to be
      `n`, then the first `n` and the last `n` elements will be printed. If set
      to `-1`, then the full tensor (all elements) will be printed.
    stack_info: Whether to include stack information in the log message.

  Returns:
    The same input tensor `t` wired through tf.print (wrapped inside the outside
    compilation). If logging is not enabled, a dummy const tensor with value 0
    is returned.
  """

  if not log_enabled(log_level):
    return tf.constant(0)

  message = _get_log_msg(log_level, message, stack_info)
  def _cpu_print(t):
    print_op = tf.print(
        message, [t], output_stream=sys.stderr,
        summarize=summarize)
    with tf.control_dependencies([print_op]):
      return tf.nest.map_structure(tf.identity, t)
  return tf.compat.v1.tpu.outside_compilation(
      _cpu_print, tf.nest.map_structure(tf.identity, t))


def log_mean_min_max(
    t: FlowFieldVal,
    step_id: Optional[tf.Tensor] = None,
    replica_id: Optional[tf.Tensor] = None,
    message: Optional[str] = '',
    log_level: LogLevel = LogLevel.INFO,
    stack_info: bool = True,
) -> Any:
  """Logging function for debug information that works on TPU.

  Args:
    t: The field from which the local mean, min and max value to be logged.
    step_id: Optional. Representing the `step`.
    replica_id: Optional. Representing the `replica_id`.
    message: The optional message to be logged together with `t`.
    log_level: The logging level that indicates the severity of the message.
    stack_info: Whether to include stack information in the log message.

  Returns:
    The same input `t` wired through tf.print (wrapped inside the outside
    compilation). If logging is not enabled, a dummy const tensor with value 0
    is returned.
  """

  if not log_enabled(log_level):
    return tf.constant(0)

  message = _get_log_msg(log_level, message, stack_info)
  def _cpu_print(t):
    mean_val = tf.math.reduce_mean(tf.stack(t))
    max_val = tf.math.reduce_max(tf.stack(t))
    min_val = tf.math.reduce_min(tf.stack(t))
    out_msg = tf.strings.join([
        'step_id: ', '-1' if step_id is None else tf.strings.as_string(step_id),
        ', replica_id: ',
        '-1' if replica_id is None else tf.strings.as_string(replica_id),
        ', mean: ', tf.strings.as_string(mean_val),
        ', max: ', tf.strings.as_string(max_val),
        ', min: ', tf.strings.as_string(min_val)])
    print_op = tf.print(
        message, [out_msg], output_stream=sys.stderr)
    with tf.control_dependencies([print_op]):
      return tf.nest.map_structure(tf.identity, t)
  return tf.compat.v1.tpu.outside_compilation(
      _cpu_print, tf.nest.map_structure(tf.identity, t))
