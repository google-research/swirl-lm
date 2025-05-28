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

"""A library of boundary layer functions to be used during post-processing.

In this library, all variables are computed with the underlying modules used in
the simulation. No duplication of real logic is introduced.

This library can be imported from colab with adhoc_import. For example:
from colabtools import adhoc_import

"""

from typing import Optional, Tuple

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.boundary_condition import monin_obukhov_similarity_theory
import tensorflow as tf

_TF_DTYPE = tf.float32


class BoundaryLayer():
  """A library of utility functions for boundary layers."""

  def __init__(self, config_filepath: str, tf1: bool = False):
    """Initializes the boundary layer related libraries in the NS solver."""
    params = parameters_lib.SwirlLMParameters.config_from_proto(config_filepath)

    self.most = (
        monin_obukhov_similarity_theory.monin_obukhov_similarity_theory_factory(
            params))

    def tf1_return_fn(result):
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        return sess.run(result)

    def tf2_return_fn(result):
      return tf.nest.map_structure(lambda x: x.numpy(), result)

    self.return_fn = tf1_return_fn if tf1 else tf2_return_fn

  def surface_momentum_and_heat_flux(
      self,
      u: np.ndarray,
      v: np.ndarray,
      w: np.ndarray,
      theta: np.ndarray,
      rho: np.ndarray,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the momentum and heat fluxes at the ground level.

    Args:
      u: The velocity component in the x direction.
      v: The velocity component in the y direction.
      w: The velocity component in the z direction.
      theta: The potential temperature.
      rho: The density.

    Returns:
      A tuple with the first and second components being the wall shear stresses
      along the first and second dimension in the horizontal plane,
      respectively, and the last component being the wall heat flux.
    """
    states = {
        'u': tf.convert_to_tensor(u, dtype=_TF_DTYPE),
        'v': tf.convert_to_tensor(v, dtype=_TF_DTYPE),
        'w': tf.convert_to_tensor(w, dtype=_TF_DTYPE),
        'theta': tf.convert_to_tensor(theta, dtype=_TF_DTYPE),
        'rho': tf.convert_to_tensor(rho, dtype=_TF_DTYPE),
    }

    return self.return_fn(
        self.most.surface_shear_stress_and_heat_flux_update_fn(states))

  def surface_momentum_and_heat_flux_generic(
      self,
      theta: np.ndarray,
      u1: np.ndarray,
      u2: np.ndarray,
      rho: np.ndarray,
      height: Optional[float] = None,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the momentum and heat fluxes at the ground level.

    Args:
      theta: The potential temperature.
      u1: The first velocity component in the horizontal plane (perpendicular
        to the vertical direction).
      u2: The second velocity component in the horizontal plane (perpendicular
        to the vertical direction).
      rho: The density.
      height: The distance from the wall to the first grid point. If not
        provided, it will be set according to the grid spacing in the
        simulation configuration provided in the constructor.


    Returns:
      A tuple with the first and second components being the wall shear stresses
      along the first and second dimension in the horizontal plane,
      respectively, and the last component being the wall heat flux.
    """
    z = self.most.height if height is None else height

    return self.return_fn(
        self.most._surface_shear_stress_and_heat_flux(  # pylint: disable=protected-access
            tf.convert_to_tensor(theta, dtype=_TF_DTYPE),
            tf.convert_to_tensor(u1, dtype=_TF_DTYPE),
            tf.convert_to_tensor(u2, dtype=_TF_DTYPE),
            tf.convert_to_tensor(rho, dtype=_TF_DTYPE),
            z,
        )
    )

  def obukhov_length_scale(
      self,
      theta: np.ndarray,
      u1: np.ndarray,
      u2: np.ndarray,
      height: Optional[float] = None,
  ) -> np.ndarray:
    """Computes the Obukhov length scale.

    Args:
      theta: The potential temperature.
      u1: The first velocity component in the horizontal plane (perpendicular
        to the vertical direction).
      u2: The second velocity component in the horizontal plane (perpendicular
        to the vertical direction).
      height: The distance from the wall to the first grid point. If not
        provided, it will be set according to the grid spacing in the
        simulation configuration provided in the constructor.

    Returns:
      The Obukhov length scale.
    """
    theta = tf.convert_to_tensor(theta, dtype=_TF_DTYPE)
    u1 = tf.convert_to_tensor(u1, dtype=_TF_DTYPE)
    u2 = tf.convert_to_tensor(u2, dtype=_TF_DTYPE)

    z = 0.5 * self.most.height if height is None else height

    zeta = self.return_fn(
        self.most._normalized_height(theta, u1, u2, z))  # pylint: disable=protected-access

    return z / zeta
