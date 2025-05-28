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

# coding=utf-8
# Copyright 2022 Google LLC
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
"""Defines a utility class for computing hydrostatic states."""

import enum

from typing import Text

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.communication import halo_exchange
from swirl_lm.communication import halo_exchange_utils
from swirl_lm.physics import constants
from swirl_lm.utility import common_ops
from swirl_lm.utility import types
import tensorflow as tf


FlowFieldVal = types.FlowFieldVal
BCType = halo_exchange_utils.BCType


class InputType(enum.Enum):
  """Defines the name of a temperature variable."""
  # The temperature of dry air, or virtual temperature of moist air.
  TEMPERATURE = 'temperature'
  # The potential temperature of dry air, or virtual potential temperature of
  # moist air.
  POTENTIAL_TEMPERATURE = 'potential_temperature'


class HydrostaticEquilibrium():
  """Utility functions for computing states in hydrostatic equilibrium.

    In many applications, the background state or initial state of a geophysical
    flow simulation is specified via a realistic atmospheric temperature profile
    from which it is possible to derive a corresponding hydrostatically balanced
    pressure and density. Here we assume the air mixture satisfies the ideal gas
    law.

    Using this library, the hydrostatic pressure can be computed either from a
    (virtual) temperature or from a (virtual) potential temperature profile.

    When a (virtual) temperature is provided, the hydrostatic pressure needs to
    satisfy two equations:
    1) p = 𝜚RT (ideal gas law)
    2) dp/dz = -G 𝜚 (hydrostatic equilibrium)
    which, combined, yield:
    dp/dz = -G p / R / T
    or
    dlog(p) = -G / R / T dz.
    Integrating both sides and solving for p:
    p = p₀exp{- ∫ (G / R / T) dz}

    When a (virtual) potential temperature is provided instead, then 3 equations
    must be satisfied:
    1) p = 𝜚RT (ideal gas law)
    2) T = 𝜃(p₀ / p)**(Cₚ / R) (potential temperature definition)
    3) dp/dz = -G𝜚 (hydrostatic equilibrium)
    which, combined, yield:
    dp/dz = -Gp(p₀ / p)**(R / Cₚ) / R / 𝜃
    or
    p**(-1 / 𝛾)dp = -Gp₀**(R / Cₚ) / R / 𝜃 dz.
    Integrating both sides and solving for p:
    p = p₀[1 - ∫(G / Cₚ / 𝜃)dz]**(Cₚ / R)

    From the given temperature and hydrostatic pressure, the hydrostatic density
    follows straightfowardly from the ideal gas law:

    𝜚 = p / R / T

    Note that the pressure is computed via numerical integration of a function
    of temperature so its accuracy will depend on the grid resolution and on the
    smoothness of the given temperature profile.
  """

  def __init__(self, params: parameters_lib.SwirlLMParameters,):
    self._params = params
    g_vec = (
        params.gravity_direction if params.gravity_direction else [0.0,] * 3)
    # Find the direction of gravity. Only vector along a particular dimension is
    # supported currently.
    self._g_dim = None
    for i in range(3):
      if np.abs(np.abs(g_vec[i]) - 1.0) < np.finfo(np.float32).resolution:
        self._g_dim = i
        break
    self._dh = (self._params.dx, self._params.dy, self._params.dz)[self._g_dim]

  def _temperature_integration_fn(self, t):
    """Computes the integrand in the pressure expression given a T profile.

    Specifically, the integrand in question is the one found in the following
    expression for the hydrostatic pressure as a function of temperature:
    p = p₀exp{- ∫ (G / R / T(z)) dz}

    Args:
      t: The temperature field, in K.

    Returns:
      The integrand in the pressure expression as a function of temperature.
    """
    return constants.G  / constants.R_D * tf.math.reciprocal(t)

  def _theta_integration_fn(self, t):
    """Computes the integrand in the pressure expression given a 𝜃 profile.

    Specifically, the integrand in question is the one found in the following
    expression for the hydrostatic pressure as a function of temperature:
    p = p₀[1 - ∫(G / Cₚ / 𝜃)dz]**(Cₚ / R)

    Args:
      t: The potential temperature field, in K.

    Returns:
      The integrand in the pressure expression as a function of potential
      temperature.
    """
    return constants.G  / constants.CP * tf.math.reciprocal(t)

  def _p_fn_from_temperature(self, integral):
    """Computes the pressure given an integral in terms of temperature."""
    return self._params.p_thermal * tf.math.exp(-integral)

  def _p_fn_from_theta(self, integral):
    """Computes the pressure given an integral in terms of 𝜃."""
    return self._params.p_thermal * (1 - integral)**(
        constants.CP / constants.R_D)

  def pressure(
      self,
      varname: Text,
      t: FlowFieldVal,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
  ) -> FlowFieldVal:
    """Computes the hydrostatic pressure from a given profile of T or 𝜃.

    Note that the pressure calculation assumes the air is dry. If accounting
    for moisture is desired, the caller should pass the virtual temperature
    or virtual potential temperature instead.

    Args:
      varname: One of 'temperature' or 'theta'.
      t: A temperature, or potential temperature, field in units of K,
        or its virtual equivalent if moisture should be accounted for.
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.

    Returns:
      The hydrostatic pressure that conforms to the given (potential)
      temperature profile.
    """
    dims = (0, 1, 2)
    replica_dims = (0, 1, 2)

    if varname == InputType.POTENTIAL_TEMPERATURE.value:
      integration_fn = self._theta_integration_fn
      p_fn = self._p_fn_from_theta
    elif varname == InputType.TEMPERATURE.value:
      integration_fn = self._temperature_integration_fn
      p_fn = self._p_fn_from_temperature
    else:
      raise ValueError(
          f'{varname} is not a valid variable for hydrostatic pressure '
          f'computation. Available options are: '
          f'\'{InputType.TEMPERATURE.value}\' '
          f'or \'{InputType.POTENTIAL_TEMPERATURE.value}\'.')

    if self._g_dim == -1:
      return tf.nest.map_structure(
          lambda t: self._params.p_thermal * tf.ones_like(t), t)

    def strip_halos(f):
      """Removes ghost cells in the vertical direction."""
      vertical_halos = [0, 0, 0]
      vertical_halos[self._g_dim] = self._params.halo_width
      return common_ops.strip_halos(f, vertical_halos)

    # Performs integration to points in the interior domain only.
    integrand = tf.nest.map_structure(integration_fn, strip_halos(t))
    buf, _ = common_ops.integration_in_dim(replica_id, replicas, integrand,
                                           self._dh, self._g_dim)

    p_interior = tf.nest.map_structure(p_fn, buf)

    # Performs integration in the ghost cells.
    def get_pressure_bc(face):
      """Computes the pressure at the boundaries in the vertical direction."""
      # Because the integration is performed from the two ends of the domain
      # outwards, the integral needs to be reversed on the lower end.
      sign = -1.0 if face == 0 else 1.0

      buf_0 = common_ops.get_face(buf, self._g_dim, face, 0)[0]
      p_bc = []

      for i in range(self._params.halo_width):
        t_lim = [
            common_ops.get_face(t, self._g_dim, face,
                                self._params.halo_width - i - j)[0]
            for j in range(2)
        ]
        integrand_lim = [
            tf.nest.map_structure(integration_fn, t_i)
            for t_i in t_lim
        ]
        integral = tf.nest.map_structure(
            lambda a, b: 0.5 * (a + b) * self._dh, *integrand_lim)
        buf_1 = tf.nest.map_structure(
            lambda b_0_i, int_i: b_0_i + sign * int_i, buf_0, integral)
        p_1 = tf.nest.map_structure(p_fn, buf_1)
        p_bc.append(p_1)
        buf_0 = buf_1

      # The order of the ghost cell values needs to follow the coordinates.
      # Because the integration is performed outwards, the sequence needs to be
      # reversed on the lower end.
      if face == 0:
        p_bc.reverse()

      return p_bc

    # Update pressure in the ghost cells.
    vertical_paddings = [(0, 0)] * 3
    vertical_paddings[self._g_dim] = [self._params.halo_width,] * 2
    p = common_ops.pad(p_interior, vertical_paddings, 0.0)

    bc = [[
        (BCType.NEUMANN, 0.0),
    ] * 2] * 3
    bc[self._g_dim] = [(BCType.DIRICHLET, get_pressure_bc(i)) for i in range(2)]

    periodic_dims = [False] * 3
    return halo_exchange.inplace_halo_exchange(p, dims, replica_id, replicas,
                                               replica_dims, periodic_dims, bc,
                                               self._params.halo_width)
