# Copyright 2023 The swirl_lm Authors.
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

from typing import Any, Tuple

from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics.atmosphere import microphysics_generic
from swirl_lm.physics.atmosphere import microphysics_kw1978
from swirl_lm.physics.atmosphere import microphysics_one_moment
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.physics.thermodynamics import water


def select_microphysics(
    model_name: str,
    params: parameters_lib.SwirlLMParameters,
    thermodynamics: thermodynamics_manager.ThermodynamicsManager,
) -> Tuple[microphysics_generic.Microphysics, Any]:
  """Selects the microphysics model by `model_name`."""
  if model_name == 'kessler':
    assert isinstance(thermodynamics.model, water.Water), (
        '`water` is required as the thermodynamics model to use the Kessler'
        f' microphysics, but {thermodynamics.model} is provided.'
    )
    microphysics = microphysics_kw1978.MicrophysicsKW1978(
        params, thermodynamics.model
    )
    microphysics_lib = microphysics_kw1978
  elif model_name == 'one_moment':
    microphysics = microphysics_one_moment.OneMoment(params)
    microphysics_lib = microphysics_one_moment
  else:
    raise NotImplementedError(
        f'{model_name} is not a valid microphysics model.'
        ' Available options are `kessler`, `one_moment`.'
    )

  return microphysics, microphysics_lib
