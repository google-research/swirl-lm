# Copyright 2024 The swirl_lm Authors.
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

"""Library for common file IO operations."""

import csv
from typing import Dict

import numpy as np
import tensorflow as tf


def parse_csv_file(
    path: str,
) -> Dict[str, np.ndarray]:
  """Reads and returns the columns of the input csv file (with header row).

  Args:
    path: Path of the csv file.

  Returns:
    A dictionary mapping column names to a `tf.Tensor` of the column values.
  """
  out = {}
  with tf.io.gfile.GFile(path, 'r') as f:
    for i, row in enumerate(csv.DictReader(f)):
      for key, value in row.items():
        out.setdefault(key, []).append(float(value))
      assert all(
          len(values) == i + 1 for values in out.values()
      ), f'Missing values in {path} while processing {row}.'
  return {
      key: np.array(values, dtype=np.float32) for key, values in out.items()
  }
