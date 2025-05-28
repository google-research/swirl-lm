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

"""Defines a factory method for the turbulent combustion model."""

from swirl_lm.physics.turbulent_combustion import thickened_flame
from swirl_lm.physics.turbulent_combustion import turbulent_combustion_pb2


def turbulent_combustion_model_factory(
    model_params: turbulent_combustion_pb2.TurbulentCombustion,
):
  """Creates an instance of the selected turbulent combustion model."""
  if model_params is None:
    return None

  model_type = model_params.WhichOneof('turbulent_combustion_model')
  if model_type == 'const_tf':
    return thickened_flame.ThickenedFlame(model_params)
  else:
    raise NotImplementedError(
        f'{model_type} is not supported. Available options for turbulent'
        ' combustion are: "const_tf".'
    )
