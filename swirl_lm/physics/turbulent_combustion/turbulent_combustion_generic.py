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

"""Defines an abstract class for the turbulent-combustion models."""

import abc
from typing import Optional

from swirl_lm.physics.turbulent_combustion import turbulent_combustion_pb2
from swirl_lm.utility import types


class TurbulentCombustionGeneric(abc.ABC):
  """Defines an abstract class for the turbulent-combustion models."""

  def __init__(
      self, model_params: turbulent_combustion_pb2.TurbulentCombustion
  ):
    """Initializes the turbulent combustion model."""
    self.model_params = model_params

  def update_diffusivity(
      self,
      diffusivity: types.FlowFieldVal,
      states: Optional[types.FlowFieldMap] = None,
      additional_states: Optional[types.FlowFieldMap] = None,
  ) -> types.FlowFieldVal:
    """Updates the diffusivity if provided by the turbulent combustion model."""
    del states, additional_states  # unused.

    return diffusivity

  def update_source_term(
      self,
      reaction_rate: types.FlowFieldVal,
      states: Optional[types.FlowFieldMap] = None,
      additional_states: Optional[types.FlowFieldMap] = None,
  ) -> types.FlowFieldVal:
    """Updates the reaction rate with the turbulence closure model."""
    del states, additional_states  # unused.

    return reaction_rate
