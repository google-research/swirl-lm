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

# Copyright 2021 Google LLC
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
"""Commonly used types that include simulation framework object types."""

from typing import Callable

import numpy as np
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf

# (diff_ops_kernel, replica_id, replica maps, states,
#  additional_states, grid parametrization) --> states + additional_states
StatesUpdateFn = Callable[[
    get_kernel_fn.ApplyKernelOp, tf.Tensor, np.ndarray, types
    .FlowFieldMap, types.FlowFieldMap, grid_parametrization.GridParametrization
], types.FlowFieldMap]
# (diff_ops_kernel, replica_id, replica maps, step_id, states,
#  additional_states, grid parametrization) --> states + additional_states
AdditionalStatesUpdateFn = Callable[[
    get_kernel_fn.ApplyKernelOp, tf.Tensor, np.ndarray, tf.Tensor, types
    .FlowFieldMap, types.FlowFieldMap, grid_parametrization.GridParametrization
], types.FlowFieldMap]
