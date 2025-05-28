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

"""Utilities and wrappers for parameters of rain, snow, and ice particles."""

import dataclasses
import numpy as np
from scipy import special
from swirl_lm.physics.atmosphere import microphysics_one_moment_constants as constants
from swirl_lm.physics.atmosphere import microphysics_pb2
from swirl_lm.utility import types
import tensorflow as tf


def compute_sphere_mass(rho: float, r: float) -> float:
  """Computes the mass of a sphere."""
  return 4.0 / 3.0 * np.pi * rho * r ** 3


@dataclasses.dataclass(frozen=True)
class Rain:
  """Parameters for rain related quantities."""

  # A reference to the config proto containing rain parameters.
  params: microphysics_pb2.OneMoment.Rain

  # Unit mass of a rain drop [kg].
  m_0: float

  # Unit cross section area of a rain drop [m^2].
  a_0: float

  @classmethod
  def from_config(cls, params: microphysics_pb2.OneMoment.Rain) -> 'Rain':
    m_0 = compute_sphere_mass(params.rho, params.r_0)
    a_0 = np.pi * params.r_0 ** 2
    return cls(params=params, m_0=m_0, a_0=a_0)


@dataclasses.dataclass(frozen=True)
class Snow:
  """Constants for snow related quantities."""

  # A reference to the config proto containing snow parameters.
  params: microphysics_pb2.OneMoment.Snow

  # Unit mass of a snow crystal [kg] [1].
  m_0: float

  # Unit cross section area of a snow crystal [m^2] [1].
  a_0: float

  # The snow size distribution parameter coefficient [m^-4] [3].
  mu: float

  # The exponent of the radius in the aspect ratio approximation.
  alpha: float

  # The constant factor of the aspect ratio (independent of radius).
  phi_0: float

  @classmethod
  def from_config(cls, params: microphysics_pb2.OneMoment.Snow) -> 'Snow':
    """Creates an instance of Snow from config proto."""
    m_0 = 0.1 * params.r_0 ** 2
    a_0 = 0.3 * np.pi * params.r_0 **  2
    mu = 4.36e9 * constants.RHO_AIR ** params.nu
    alpha = (
        params.m_e + params.del_m - 1.5 * (params.a_e + params.del_a)
    )
    # 3-rd order moment for volume- or mass-weighted average.
    k = 3
    # Precomputed scale factor of the mass-weighted average of the aspect ratio.
    phi_0 = (
        special.gamma(alpha + k + 1) * 3.0 * np.sqrt(np.pi)
        / (special.gamma(k + 1) * 4.0 * params.rho)
        * params.chi_m * m_0
        / (params.chi_a * a_0) ** (3/2)
        / (2.0 * params.r_0) ** alpha
    )
    return cls(
        params=params,
        m_0=m_0,
        a_0=a_0,
        mu=mu,
        alpha=alpha,
        phi_0=phi_0,
    )


@dataclasses.dataclass(frozen=True)
class Ice:
  """Parameters for ice related quantities."""

  # A reference to the config proto containing ice parameters.
  params: microphysics_pb2.OneMoment.Ice

  # Unit mass of an ice crystal [kg].
  m_0: float

  @classmethod
  def from_config(cls, params: microphysics_pb2.OneMoment.Ice):
    """Creates an instance of Ice from config proto."""
    m_0 = compute_sphere_mass(params.rho, params.r_0)
    return cls(params=params, m_0=m_0)


def gamma(x: float | tf.Tensor) -> tf.Tensor:
  """Computes the Gamma function of x."""
  with tf.control_dependencies(
      [
          tf.assert_greater(
              x, 0.0, message='The Gamma function takes positive inputs only.'
          )
      ]
  ):
    return tf.math.exp(tf.math.lgamma(x))


def n_0(
    particle: Rain | Snow | Ice,
    rho: tf.Tensor | None = None,
    q_s: tf.Tensor | None = None,
) -> tf.Tensor:
  """Computes `n_0` in the Marshall-Palmer distribution parameter [m^-4].

  The `n_0` parameter is assumed to be constant for rain and ice and is a
  function of density and mass fraction for snow.

  Args:
    particle: A Rain, Snow, or Ice dataclass object that stores constant
      parameters.
    rho: The density of the moist air [kg/m^3].
    q_s: The snow mass fraction [kg/kg].

  Returns:
    The `n_0` parameter in the Marshall-Palmer distribution.
  """
  if isinstance(particle, Snow):
    assert q_s is not None, 'q_s is required for Snow, but None was provided.'
    assert rho is not None, 'rho is required for Snow, but None was provided.'
    params = particle.params
    assert isinstance(params, microphysics_pb2.OneMoment.Snow), (
        'Snow.params must be set to compute the size distribution parameter.'
    )
    return particle.mu * tf.math.pow(
        rho * q_s / constants.RHO_AIR, params.nu
    )
  elif isinstance(particle, (Ice, Rain)):
    return tf.constant(particle.params.n_0)
  else:
    raise ValueError(
        f'One of Snow, Ice, or Rain is required but {type(particle)} was'
        ' provided.'
    )


def marshall_palmer_distribution_parameter_lambda(
    particle: Rain | Snow | Ice,
    rho: types.FlowFieldVal,
    q: types.FlowFieldVal,
) -> types.FlowFieldVal:
  """Computes `lambda` in the Marshall-Palmer distribution parameters.

  Args:
    particle: A Rain, Snow, or Ice dataclass object that stores constant
      parameters.
    rho: The density of the moist air [kg/m^3].
    q: The water mass fraction [kg/kg].

  Returns:
    The lambda parameter in the Marshall-Palmer distribution.
  """
  m_0 = particle.m_0
  coeff = particle.params
  m = coeff.m_e + coeff.del_m + 1.0

  def lambda_fn(rho: tf.Tensor, q: tf.Tensor) -> tf.Tensor:
    """Computes the lambda parameter for a single tf.Tensor q."""
    # The denominator is 0 when the water mass fraction is 0. In this case,
    # the distribution parameter is 0, which is provided by the behavior of
    # divide_no_nan.
    return tf.math.pow(
        tf.math.divide_no_nan(
            gamma(m) * coeff.chi_m * m_0 * n_0(particle, rho, q),
            tf.maximum(q, 0.0)
            * rho
            * tf.math.pow(coeff.r_0, coeff.m_e + coeff.del_m),
        ),
        1.0 / m,
    )

  return tf.nest.map_structure(lambda_fn, rho, q)
