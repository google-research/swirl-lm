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
"""Commonly used types in the simulation framework."""

from typing import Callable, List, Mapping, MutableMapping, NamedTuple, Optional, Sequence, Text, Tuple, TypeAlias, Union

import numpy as np
import tensorflow as tf

# Note that tf.float{n} needs to match with tf.complex{m} here, and m == 2 * n.
TF_DTYPE = tf.float32
TF_COMPLEX_DTYPE = tf.complex64

# Note that np.float{n} and np.complex{m} needs to match with tf.* version.
NP_DTYPE = np.float32
NP_COMPLEX_DTYPE = np.complex64

FlowFieldVal: TypeAlias = tf.Tensor
FlowFieldMap: TypeAlias = Mapping[str, tf.Tensor]

VectorField = Tuple[FlowFieldVal, FlowFieldVal, FlowFieldVal]
ReplicaCoordinates = Tuple[int, int, int]

InitFn = Callable[[Union[int, tf.Tensor], ReplicaCoordinates], FlowFieldMap]

TensorMap = Mapping[Text, tf.Tensor]
VariableMap = Mapping[Text, tf.Variable]
DimensionMap = Mapping[Text, int]

BoolMap = Mapping[Text, bool]
MutableTensorMap = MutableMapping[Text, tf.Tensor]

FloatSequence = Sequence[float]
IntSequence = Sequence[int]

FnOutput = Tuple[List[tf.Tensor], MutableTensorMap]
StepOutput = List[List[tf.Tensor]]
StepInfeedHandle = NamedTuple(
    'StepInfeedHandle',
    [('enqueue_ops', List[tf.Operation]),
     ('enqueue_placeholders', List[Mapping[Text, tf.Tensor]])])
StepInfeedHandle.__new__.__defaults__ = ([], [])
StepHandle = NamedTuple('StepHandle',
                        [('step_output', StepOutput),
                         ('step_infeed_handle', StepInfeedHandle)])
StepHandle.__new__.__defaults__ = (None, StepInfeedHandle())
StepHandleBuilder = Callable[[], StepHandle]
SingleStepFn = Callable[
    [Sequence[tf.Tensor], TensorMap, MutableTensorMap, np.ndarray], FnOutput]
StepFn = Union[Sequence[SingleStepFn], SingleStepFn]
InfeedElementSpec = NamedTuple('InfeedElementSpec', [('type', tf.DType),
                                                     ('shape', tf.TensorShape)])
SingleInfeedSpec = Optional[Mapping[Text, InfeedElementSpec]]
InfeedSpec = Union[Sequence[SingleInfeedSpec], SingleInfeedSpec]


class ScalarSource(NamedTuple):
  """Defines the source term in a scalar transport equation."""
  # The total source term.
  total: FlowFieldVal
  # The source term to be added to the right hand side of the Poisson equation
  # when the thermodynamics mode is LOW_MACH.
  mass: Optional[FlowFieldVal] = None
