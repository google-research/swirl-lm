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

"""Utility functions for text."""

import re


def _decimal_digits(numerator: int, denominator: int) -> str:
  """Returns the digits after the decimal point, e.g, '05' for 50 / 1000.

  Args:
    numerator: In [0, denominator).
    denominator: 1, 10, 100, ...

  Returns:
    Digits in the fractional part of numerator / denominator, or the empty
    string if numerator is 0.
  """
  digits = []
  while numerator > 0:
    denominator //= 10
    digits.append(str(numerator // denominator))
    numerator %= denominator
  return ''.join(digits)


def seconds_to_string(total_seconds: float, separator: str = ' ',
                      precision: float = 1.0, zero: str = '0') -> str:
  """Formats a duration into a human-readable string.

  Args:
    total_seconds: Duration in seconds to format into a string.
    separator: String to join the parts of the formatted string.
    precision: The duration will be rounded to an integer multiple of
      `precision`.
    zero: String to return when duration is 0.

  Returns:
    Formatted string like '5d 12h 30m 45s' or '-1d 14h 30m 32s'.
  """
  # Handle negative durations.
  if total_seconds < 0:
    positive_str = seconds_to_string(-total_seconds, separator, precision, zero)
    return '-' + positive_str if positive_str != zero else zero

  # Convert durations to integer nanoseconds to avoid floating errors.
  # (2^63 nanoseconds is approximately 300 years).
  s = 1_000_000_000  # One second, measured in nanoseconds.

  total_ns = round(total_seconds * s)
  precision_ns = round(precision * s)

  rounded_total_ns = (round(total_ns / precision_ns) * precision_ns)

  units = [('d', 24 * 3600 * s), ('h', 3600 * s), ('m', 60 * s),
           ('s', s), ('ms', s // 1000), ('us', s // 1_000_000),
           ('ns', s // 1_000_000_000)]

  # Array of tuples representing the formatted duration so far. Each tuple
  # is of the form (value_str, symbol, place_value), e.g., while formatting
  # 62.3 seconds, parts will be [('1', 'm', 60 * ns), ('2', 's', ns)] at
  # some point. Note that value_str is a string and not an integer because
  # for the last unit we might have a fractional part, e.g., ('2.3', 's', ns).
  parts = []
  for i, (symbol, place_value) in enumerate(units):
    v = rounded_total_ns // place_value
    rounded_total_ns %= place_value
    if v > 0:
      if rounded_total_ns == 0:
        # This is the last unit to use. We'll try to fold the value into the
        # previous unit as a decimal fraction if we can.
        if parts and parts[-1][2] // place_value == 1000:
          # We can just use the previous unit and append the fractional part
          # to its value.
          parts[-1] = (f'{parts[-1][0]}.{_decimal_digits(v, 1000)}',
                       parts[-1][1], parts[-1][2])
        elif i > 0 and units[i - 1][1] // place_value == 1000:
          # The previous unit didn't have a place value of 1000x, but maybe we
          # had skipped it because its value was 0 in which case we'll just use
          # a '0' for it, e.g., 60.1s -> '1m 0.1s.'
          parts.append((f'0.{_decimal_digits(v, 1000)}', *units[i - 1]))
        else:
          # Can't use fractional part, just use the current unit, e.g.,
          # 90s -> '1m 30s' not '1.5m'.
          parts.append((f'{v}', symbol, place_value))
      else:
        parts.append((f'{v}', symbol, place_value))
    if rounded_total_ns == 0:
      break

  if not parts:
    return zero

  return separator.join(f'{k}{v}' for k, v, _ in parts)


def strip_line_comments(text: str, comment_marker: str) -> str:
  strip_re = r'\s*' + re.escape(comment_marker) + r'.*$'
  return re.sub(strip_re, '', text, flags=re.MULTILINE)
