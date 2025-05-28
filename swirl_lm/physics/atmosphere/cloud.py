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

"""A class for handling cloud specific parametrizations."""

from typing import Sequence

import numpy as np
from swirl_lm.physics.thermodynamics import water
from swirl_lm.utility import common_ops
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap

# Parameters required by the radiation model. Reference:
# Stevens, Bjorn, Chin-Hoh Moeng, Andrew S. Ackerman, Christopher S.
# Bretherton, Andreas Chlond, Stephan de Roode, James Edwards, et al. 2005.
# “Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine
# Stratocumulus.” Monthly Weather Review 133 (6): 1443–62.
_F0 = 70.0
_F1 = 22.0
_KAPPA = 85.0
_ALPHA_Z = 1.0
# The subsidence velocity coefficient.
_D = 3.75e-6
# The initial height of the cloud, in units of m
_ZI = 840.0


class Cloud(object):
  """An object for handling cloud parametrizations."""

  def __init__(self, water_model: water.Water) -> None:
    """Initialize with a water thermodynamics model."""
    self._water_model = water_model

  @property
  def water_model(self) -> water.Water:
    """The underlying water thermodynamics model."""
    return self._water_model

  def _radiation(
      self,
      q_h: tf.Tensor,
      q_l: tf.Tensor,
      rho: tf.Tensor,
      z: tf.Tensor,
  ) -> tf.Tensor:
    """Computes the radiation term based on given parameters.

    Args:
      q_h: The integral of the liquid water specific mass from `z` to the
        maximum height of the simulation.
      q_l: The integral of the liquid water specific mass from 0 to `z`.
      rho: The density of air at `z`
      z: The current height.

    Returns:
      The radiation source term.
    """
    return (_F0 * tf.math.exp(-_KAPPA * q_h) +
            _F1 * tf.math.exp(-_KAPPA * q_l) +
            rho * self._water_model.cp_d * _D * _ALPHA_Z *
            (0.25 * tf.math.pow(tf.maximum(z - _ZI, 0.0), 4.0 / 3.0) +
             _ZI * tf.math.pow(tf.maximum(z - _ZI, 0.0), 1.0 / 3.0)))

  def source_by_radiation(
      self,
      q_l: FlowFieldVal,
      rho: FlowFieldVal,
      zz: FlowFieldVal,
      h: float,
      g_dim: int,
      halos: Sequence[int],
      replica_id: tf.Tensor,
      replicas: np.ndarray,
  ) -> FlowFieldVal:
    """Computes the energy source term due to radiation.

    Reference:
    Stevens, Bjorn, Chin-Hoh Moeng, Andrew S. Ackerman, Christopher S.
    Bretherton, Andreas Chlond, Stephan de Roode, James Edwards, et al. 2005.
    “Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine
    Stratocumulus.” Monthly Weather Review 133 (6): 1443–62.

    Args:
      q_l: The liquid humidity.
      rho: The density.
      zz: The height values.
      h: The vertical grid spacing.
      g_dim: The dimension of the gravity.
      halos: A sequence of int representing the halo points in each dimension.
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.

    Returns:
      The source term in the total energy equation due to radiation.
    """
    vertical_halos = [
        halo if dim == g_dim else 0 for dim, halo in enumerate(halos)
    ]
    vertical_paddings = [(0, 0)] * 3
    vertical_paddings[g_dim] = [halos[g_dim]] * 2

    def zero_out_vertical_halos(f):
      return common_ops.pad(
          common_ops.strip_halos(f, vertical_halos), vertical_paddings)

    # Remove the vertical halos to avoid polluting the vertical integration.
    rho_q_l = zero_out_vertical_halos(
        tf.nest.map_structure(tf.math.multiply, rho, q_l))
    q_below, q_above = common_ops.integration_in_dim(replica_id, replicas,
                                                     rho_q_l, h, g_dim)

    f_r = tf.nest.map_structure(self._radiation, q_above, q_below, rho, zz)

    return f_r
