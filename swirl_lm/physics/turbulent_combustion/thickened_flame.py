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

"""A library for the Thickened Flame turbulent-combustion closure."""

from typing import Optional

from swirl_lm.physics.turbulent_combustion import turbulent_combustion_generic
from swirl_lm.physics.turbulent_combustion import turbulent_combustion_pb2
from swirl_lm.utility import types
import tensorflow as tf


class ThickenedFlame(turbulent_combustion_generic.TurbulentCombustionGeneric):
  """A library for the Thickened Flame model."""

  def __init__(
      self, model_params: turbulent_combustion_pb2.TurbulentCombustion
  ):
    """Initializes the Thickened Flame model."""
    super().__init__(model_params)

    if model_params.WhichOneof('turbulent_combustion_model') != 'const_tf':
      raise ValueError(
          'The Thickened Flame model is not configured. It should be defined'
          ' through `const_tf` in the config file.'
      )

    self.thickening_factor = self.model_params.const_tf.thickening_factor

  def update_diffusivity(
      self,
      diffusivity: types.FlowFieldVal,
      states: Optional[types.FlowFieldMap] = None,
      additional_states: Optional[types.FlowFieldMap] = None,
  ) -> types.FlowFieldVal:
    """Updates the diffusivity if provided by the turbulent combustion model."""
    del states, additional_states  # unused.

    return tf.nest.map_structure(
        lambda d: self.thickening_factor * d, diffusivity
    )

  def update_source_term(
      self,
      reaction_rate: types.FlowFieldVal,
      states: Optional[types.FlowFieldMap] = None,
      additional_states: Optional[types.FlowFieldMap] = None,
  ) -> types.FlowFieldVal:
    """Updates the reaction rate with the turbulence closure model."""
    del states, additional_states  # unused.

    return tf.nest.map_structure(
        lambda s: s / self.thickening_factor, reaction_rate
    )
