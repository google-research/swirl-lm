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

"""A library of density update with a constant."""

from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics.thermodynamics import thermodynamics_generic
import tensorflow as tf

TF_DTYPE = thermodynamics_generic.TF_DTYPE

FlowFieldVal = thermodynamics_generic.FlowFieldVal
FlowFieldMap = thermodynamics_generic.FlowFieldMap


class ConstantDensity(thermodynamics_generic.ThermodynamicModel):
  """A library of constant density."""

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    """Initializes the constant density object."""
    super(ConstantDensity, self).__init__(params)

    if params.use_3d_tf_tensor:
      self.rho = params.rho * tf.ones(
          (params.nz, params.nx, params.ny), dtype=TF_DTYPE
      )
    else:
      self.rho = params.rho * tf.ones(
          (params.nz, params.nx, params.ny), dtype=TF_DTYPE
      )

  def update_density(
      self,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldVal:
    """Updates the density with the stored constant density."""
    del states, additional_states
    return self.rho
