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

"""An abstract class for thermodynamic models."""

import abc
from typing import Optional

import six
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics.thermodynamics import thermodynamics_utils
import tensorflow as tf

TF_DTYPE = thermodynamics_utils.TF_DTYPE

FlowFieldVal = thermodynamics_utils.FlowFieldVal
FlowFieldMap = thermodynamics_utils.FlowFieldMap


@six.add_metaclass(abc.ABCMeta)
class ThermodynamicModel(object):
  """A generic class for thermodynamic models."""

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    """Initializes the thermodynamics library."""
    self._params = params
    self._rho = params.rho

  def rho_ref(
      self,
      zz: Optional[FlowFieldVal] = None,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Generates the reference density.

    The default reference density is a constant whose value is specified in the
    input config.

    Args:
      zz: The coordinates along the direction of height/gravitation. Useful in
        geophysical flows.
      additional_states: Helper variables including those needed to compute the
        reference density.

    Returns:
      The reference density in the simulation.
    """
    del zz, additional_states
    return tf.constant(self._rho, dtype=TF_DTYPE)

  def p_ref(
      self,
      zz: FlowFieldVal,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Generates the reference pressure.

    The default reference pressure is a constant whose value is specified in the
    input config.

    Args:
      zz: The coordinates along the direction of height/gravitation. Useful in
        geophysical flows.
      additional_states: Helper variables including those needed to compute the
        reference pressure.

    Returns:
      The reference pressure in the simulation.
    """
    del additional_states

    return tf.nest.map_structure(
        lambda zz_i: self._params.p_thermal * tf.ones_like(zz_i), zz
    )

  def update_density(
      self,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> tf.Tensor:
    """Defines a pure virtual interface for the density update function."""
    raise NotImplementedError(
        'A thermodynamic model needs to provide a definition for the density '
        'update function.'
    )
