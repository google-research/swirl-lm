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

"""Tests for lpt_utils modules."""

from absl.testing import absltest
from absl.testing import parameterized
from tensorflow_testcase import TensorflowTestCase
import numpy as np
import tensorflow as tf

import itertools
from swirl_lm.utility import types
from swirl_lm.physics.lpt import lpt_utils
from swirl_lm.physics.lpt import lpt_types
from typing import Tuple

class TestProtoFile(TensorflowTestCase):
  def test_tensor_scatter_update_ints(self):
    """Test tensor_scatter_update_ints for dynamic sized tensor."""
    tensor_with_updates = tf.constant([[110,2,3],[40,5,6],[74,8,9]],
                                      dtype=lpt_types.LPT_INT)
    tensor = tf.constant([[1,2,3],[4,5,6],[7,8,9]],dtype=lpt_types.LPT_INT)
    indices = tf.reshape(tf.where(tensor[:, 0] == 1), [-1])

    updates = tf.einsum(
          "qj,ji->qi", tf.one_hot(indices, 3, dtype = lpt_types.LPT_INT),
          tensor_with_updates
      )

    out = lpt_utils.tensor_scatter_update_ints(tensor, indices, updates)

    expected_out = tf.constant([[110,2,3],[4,5,6],[7,8,9]],dtype=lpt_types.LPT_INT)

    self.assertAllEqual(expected_out, out)


if __name__ == '__main__':
  absltest.main()
