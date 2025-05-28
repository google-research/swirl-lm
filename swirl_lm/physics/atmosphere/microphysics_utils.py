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

# Copyright 2023 Google LLC
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
"""A library of helper functions for the microphysics models."""

from typing import TypeAlias

from google.protobuf import message
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics.atmosphere import microphysics_generic
from swirl_lm.physics.atmosphere import microphysics_kw1978
from swirl_lm.physics.atmosphere import microphysics_one_moment
from swirl_lm.physics.atmosphere import microphysics_pb2
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.physics.thermodynamics import water

ModelParamsType: TypeAlias = (microphysics_pb2.Kessler |
                              microphysics_pb2.OneMoment)


def _get_model_params_from_proto(msg: message.Message) -> ModelParamsType:
  """Returns the proto that has been set in the microphysics 'oneof'."""
  return getattr(msg, msg.WhichOneof('model_type'))


def select_microphysics(
    params: parameters_lib.SwirlLMParameters,
    thermodynamics: thermodynamics_manager.ThermodynamicsManager,
) -> microphysics_generic.MicrophysicsAdapter:
  """Selects the microphysics model by `model_name`."""
  model_params = _get_model_params_from_proto(params.microphysics)
  assert isinstance(thermodynamics.model, water.Water), (
      '`water` is required as the thermodynamics model to use'
      f' microphysics models, but {thermodynamics.model} is provided.'
  )
  if isinstance(model_params, microphysics_pb2.Kessler):
    return microphysics_kw1978.Adapter(params, thermodynamics.model)
  elif isinstance(model_params, microphysics_pb2.OneMoment):
    return microphysics_one_moment.Adapter(params, thermodynamics.model)
  else:
    raise NotImplementedError(
        f'{model_params} is not a known params type for microphysics model.'
        ' Available options are `microphysics_pb2.Kessler` and'
        ' `microphysics_pb2.OneMoment`.'
    )
