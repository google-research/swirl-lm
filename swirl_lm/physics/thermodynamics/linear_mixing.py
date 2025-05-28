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

"""A library of modeling thermodynamic quantities with linear mixing."""

from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics.thermodynamics import thermodynamics_generic
from swirl_lm.physics.thermodynamics import thermodynamics_utils
import tensorflow as tf


TF_DTYPE = thermodynamics_generic.TF_DTYPE

FlowFieldVal = thermodynamics_generic.FlowFieldVal
FlowFieldMap = thermodynamics_generic.FlowFieldMap

INERT_SPECIES = thermodynamics_utils.INERT_SPECIES
NON_SPECIES = ['T']


class LinearMixing(thermodynamics_generic.ThermodynamicModel):
  """A library of linear mixing rule."""

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    """Initialize components required by the linear mixing rule."""
    super(LinearMixing, self).__init__(params)

    self._rho_sc = {
        scalar_name: params.density(scalar_name)
        for scalar_name in params.scalars_names
        if scalar_name not in NON_SPECIES
    }
    self._rho_sc.update({INERT_SPECIES: params.rho})

  def update_density(
      self,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldVal:
    """Updates the density with the linear mixing rule."""
    del additional_states
    scalars = {
        sc_name: thermodynamics_utils.regularize_scalar_bound(states[sc_name])
        for sc_name in self._rho_sc.keys()
        if sc_name != INERT_SPECIES
    }

    if scalars:
      scalars.update({
          INERT_SPECIES:
              thermodynamics_utils.compute_ambient_air_fraction(scalars)
      })
      sc_reg = thermodynamics_utils.regularize_scalar_sum(scalars)
    else:
      sc_reg = {
          INERT_SPECIES: [
              tf.ones_like(sc_i, dtype=TF_DTYPE)
              for sc_i in list(states.values())[0]
          ]
      }
    rho_mix = tf.nest.map_structure(tf.zeros_like, list(sc_reg.values())[0])
    for sc_name, sc_val in sc_reg.items():
      rho_mix = tf.nest.map_structure(
          lambda rho_mix_i, sc_val_i: rho_mix_i  # pylint: disable=g-long-lambda
          + sc_val_i * self._rho_sc[sc_name],  # pylint: disable=cell-var-from-loop
          rho_mix,
          sc_val,
      )

    return rho_mix
