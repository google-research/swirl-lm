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

"""The manager of the LPT communication models."""

from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics.lpt import field_exchange


def lpt_factory(params: parameters_lib.SwirlLMParameters):
  """Returns the LPT model corresponding to the given parameters."""

  if params.lpt is None:
    return None

  model_params = params.lpt
  model_type = model_params.WhichOneof("lpt_parallel_approach")

  if model_type == "field_exchange":
    model = field_exchange.FieldExchange(params)
  else:
    raise NotImplementedError(
        f"Unknown LPT parallel exchange approach: {model_type}."
    )

  return model
