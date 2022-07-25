# Copyright 2022 The swirl_lm Authors.
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
import tensorflow as tf


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


def log(
    t: tf.Tensor,
    message: Optional[str] = None,
    log_level: LogLevel = LogLevel.INFO,
    summarize: Optional[int] = None,
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

  Returns:
    The same input tensor `t` wired through tf.print (wrapped inside the outside
    compilation). If logging is not enabled, a dummy const tensor with value 0
    is returned.
  """

  def log_enabled(log_flag):
    if log_flag is None:
      return False
    log_flag_level = LogLevel[log_flag].value
    return log_level.value >= log_flag_level

  if not log_enabled(_LOG_LEVEL.value):
    return tf.constant(0)

  if message is None:
    message = t.name

  frame = inspect.stack()[1]
  filename = frame.filename
  line = frame.lineno
  message = (
      'TPU debug logging level: {}. File: {} at line: {}, message: {}'.format(
          log_level.name, filename, line, message))
  def _cpu_print(t):
    print_op = tf.print(
        message, [tf.shape(t), t], output_stream=sys.stderr,
        summarize=summarize)
    with tf.control_dependencies([print_op]):
      return tf.identity(t)
  return tf.compat.v1.tpu.outside_compilation(_cpu_print, tf.identity(t))
