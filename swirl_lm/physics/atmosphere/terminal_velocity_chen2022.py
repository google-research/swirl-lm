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

"""Data class wrapping terminal velocity parameterization from Chen et al."""

import dataclasses

import numpy as np
from swirl_lm.physics.atmosphere import microphysics_one_moment_constants as constants
from swirl_lm.physics.atmosphere import microphysics_pb2
from swirl_lm.physics.atmosphere import particles
from swirl_lm.physics.atmosphere import terminal_velocity_chen2022_tables
import tensorflow as tf

Rain = particles.Rain
Snow = particles.Snow
Ice = particles.Ice

TableB1Coeffs = terminal_velocity_chen2022_tables.TableB1Coeffs
TableB3Coeffs = terminal_velocity_chen2022_tables.TableB3Coeffs
TableB5Coeffs = terminal_velocity_chen2022_tables.TableB5Coeffs


@dataclasses.dataclass(frozen=True)
class IceVelocity:
  """Computed parameters from table B3 of Chen et al. (2022)."""
  a: tf.Tensor
  b: tf.Tensor
  c: tf.Tensor
  e: tf.Tensor
  f: tf.Tensor
  g: tf.Tensor


@dataclasses.dataclass(frozen=True)
class SnowVelocity:
  """Computed parameters from table B5 of Chen et al. (2022)."""
  a: tf.Tensor
  b: tf.Tensor
  c: tf.Tensor
  e: tf.Tensor
  f: tf.Tensor
  g: tf.Tensor
  h: tf.Tensor


@dataclasses.dataclass(frozen=True)
class TerminalVelocityCoefficients:
  """Final coefficients to be used in the terminal velocity equation."""
  a: tuple[tf.Tensor, ...]
  b: tuple[tf.Tensor, ...]
  c: tuple[tf.Tensor, ...]


@dataclasses.dataclass(frozen=True)
class TerminalVelocityChen2022:
  """Coefficients for gamma-type terminal velocity parameterization."""

  # The correction factor applied to the terminal velocity of cloud droplets.
  # This is needed because we use the same coefficients for cloud droplets that
  # we use for rain, but those are only directly applicable to particle
  # diameters greater than 100 μm.
  _CLOUD_DROPLET_CORRECTION_FACTOR = 0.1
  # The exponent applied to the mass-weighted aspect ratio of snow (ϕ) in the
  # bulk fall speed equation. We assume snow particles are shaped like oblate
  # spheroids, which corresponds to  κ = 1/3 as derived from the relationship
  # between the volume-equivalent diameter and the axial half-lengths of the
  # spheroid.
  _KAPPA = 1 / 3

  # Particle parameters.
  _rain: Rain
  _snow: Snow
  _ice: Ice

  # Terminal velocity coefficients.
  rain_velocity: TableB1Coeffs
  ice_velocity: IceVelocity
  snow_velocity: SnowVelocity

  @classmethod
  def _precompute_ice_coeffs(
      cls,
      ice: Ice,
  ) -> IceVelocity:
    """Precomputes the coefficient formulas from table B3 of Chen et al."""
    rho = ice.params.rho
    # Use the raw coefficients from table B3 to evaluate the formulas in that
    # table as a function of the apparent density.
    table_b3 = TableB3Coeffs()
    a = (
        table_b3.a[0]
        + table_b3.a[1] * np.log(rho) ** 2
        + table_b3.a[2] * np.log(rho)
    )
    b = (
        table_b3.b[0]
        + table_b3.b[1] * np.log(rho)
        + table_b3.b[2] / np.sqrt(rho)
    ) ** -1
    c = (
        table_b3.c[0]
        + table_b3.c[1] * np.exp(table_b3.c[2] * rho)
        + table_b3.c[3] * np.sqrt(rho)
    )
    e = (
        table_b3.e[0]
        + table_b3.e[1] * np.log(rho) ** 2
        + table_b3.e[2] * np.sqrt(rho)
    )
    f = -np.exp(
        table_b3.f[0]
        + table_b3.f[1] * np.log(rho) ** 2
        + table_b3.f[2] * np.log(rho)
    )
    g = (
        table_b3.g[0]
        + table_b3.g[1] / np.log(rho)
        + table_b3.g[2] * np.log(rho) / rho
    ) ** -1
    coeffs = (tf.constant(x, tf.float32) for x in (a, b, c, e, f, g))
    return IceVelocity(*coeffs)

  @classmethod
  def _precompute_snow_coeffs(
      cls,
      snow: Snow,
  ) -> SnowVelocity:
    """Precomputes the coefficient formulas from table B5 of Chen et al."""
    rho = snow.params.rho
    # Read in the raw coefficients from table B5 to evaluate the formulas as a
    # function of the apparent density.
    table_b5 = TableB5Coeffs()
    a = (
        table_b5.a[0]
        + table_b5.a[1] * np.log(rho)
        + table_b5.a[2] * rho ** (-3 / 2)
    )
    b = np.exp(
        table_b5.b[0]
        + table_b5.b[1] * np.log(rho) ** 2
        + table_b5.b[2] * np.log(rho)
    )
    c = np.exp(
        table_b5.c[0] + table_b5.c[1] / np.log(rho) + table_b5.c[2] / rho
    )
    e = (
        table_b5.e[0]
        + table_b5.e[1] * np.log(rho) * np.sqrt(rho)
        + table_b5.e[2] * np.sqrt(rho)
    )
    f = (
        table_b5.f[0]
        + table_b5.f[1] * np.log(rho)
        # Slightly rewritten to convert 10^19 * exp(-rho_i)
        # into exp(19 * log(10) - rho_i), which is more stable.
        + table_b5.f[2] * np.exp(table_b5.f[3] - rho)
    )
    g = (
        table_b5.g[0]
        + table_b5.g[1] * np.log(rho) * np.sqrt(rho)
        + table_b5.g[2] / np.sqrt(rho)
    ) ** -1
    h = (
        table_b5.h[0]
        + table_b5.h[1] * rho ** (5.0 / 2.0)
        # Slightly rewritten to convert 10^20 * exp(-rho_i)
        # into exp(20 * log(10) - rho_i), which is more stable.
        + table_b5.h[2] * np.exp(table_b5.h[3] - rho)
    )
    coeffs = (tf.constant(x, tf.float32) for x in (a, b, c, e, f, g, h))
    return SnowVelocity(*coeffs)

  @classmethod
  def from_config(
      cls,
      one_moment_params: microphysics_pb2.OneMoment,
  ) -> 'TerminalVelocityChen2022':
    """Creates an instance of TerminalVelocityChen2022 from config proto."""
    # Precompute the coefficients as a function of each particle's apparent
    # density.
    rain = Rain.from_config(one_moment_params.rain)
    snow = Snow.from_config(one_moment_params.snow)
    ice = Ice.from_config(one_moment_params.ice)
    ice_velocity = cls._precompute_ice_coeffs(ice)
    snow_velocity = cls._precompute_snow_coeffs(snow)
    return cls(
        _rain=rain,
        _snow=snow,
        _ice=ice,
        rain_velocity=TableB1Coeffs(),
        ice_velocity=ice_velocity,
        snow_velocity=snow_velocity,
    )

  def _convert_coefficients_to_si_units_and_wrap(
      self,
      a: tuple[tf.Tensor, ...],
      b: tuple[tf.Tensor, ...],
      c: tuple[tf.Tensor, ...],
  ) -> TerminalVelocityCoefficients:
    """Converts the coefficients to SI units and puts them in a dataclass."""
    # Convert a from mm^-b to m^-b.
    a = tuple(a_i * tf.math.pow(1e3, b_i) for a_i, b_i in zip(a, b))
    # Convert c from mm^-1 to m^-1.
    c = tuple(c_i * 1e3 for c_i in c)
    return TerminalVelocityCoefficients(a, b, c)

  def _compute_raindrop_coefficients(
      self, rho: tf.Tensor
  ) -> TerminalVelocityCoefficients:
    """Computes raindrop coefficients as a function of density (table b1)."""
    coeffs = self.rain_velocity
    q = tf.math.exp(coeffs.q_coeff * rho)
    a = (
        coeffs.a[0] * q,
        coeffs.a[1] * q,
        coeffs.a[2] * q * tf.math.pow(rho, coeffs.rho_exp),
    )
    b = tuple(coeffs.b[i][0] + coeffs.b[i][1] * rho for i in range(3))
    c = tuple(tf.constant(c_i, tf.float32) for c_i in coeffs.c)
    return self._convert_coefficients_to_si_units_and_wrap(a, b, c)

  def _compute_ice_coefficients(
      self, rho: tf.Tensor
  ) -> TerminalVelocityCoefficients:
    """Computes the ice coefficients as a function of density."""
    coeffs = self.ice_velocity
    a = (
        coeffs.e * rho ** coeffs.a,
        coeffs.f * rho ** coeffs.a
    )
    b = (
        coeffs.b + coeffs.c * rho,
        coeffs.b + coeffs.c * rho,
    )
    c = (tf.zeros_like(coeffs.g), coeffs.g)
    return self._convert_coefficients_to_si_units_and_wrap(a, b, c)

  def _compute_snow_coefficients(
      self, rho: tf.Tensor
  ) -> TerminalVelocityCoefficients:
    """Computes the snow coefficients as a function of density."""
    coeffs = self.snow_velocity
    a = (
        coeffs.b * rho ** coeffs.a,
        coeffs.e * rho ** coeffs.a * tf.math.exp(coeffs.h * rho),
    )
    b = (coeffs.c, coeffs.f)
    c = (tf.zeros_like(coeffs.g), coeffs.g)
    return self._convert_coefficients_to_si_units_and_wrap(a, b, c)

  def _fall_speed_gamma_type(
      self,
      coeffs: TerminalVelocityCoefficients,
      lam: tf.Tensor,
  ) -> tf.Tensor:
    """Computes the gamma-type mass-weighted bulk fall speed formula.

    The equation is given by equation 20 of Chen et al. (2022) and corresponds
    to an integration of a gamma-type function over the particle size spectrum.
    Here we assume the particle size follows the Marshall-Palmer distribution.
    Since the size distribution is exponential, we set `mu` to 0; and since this
    is a mass-weighted average we fix `k`, the moment, at 3.

    Args:
      coeffs: A wrapper for the terminal velocity coefficients `a`, `b`, and `c`
        of the gamma-type function in equation 19.
      lam: The Marshall-Palmer distribution rate parameter lambda.

    Returns:
      The mass-weighted bulk fall speed for a particle group. Note that this
      result should still be scaled by the volume-weighted mean aspect ratio of
      the particle group.
    """
    # Exponential particle size distribution implies mu = 0 (equation 2).
    mu = 0
    # The volume-weighted, or mass-weighted, fall speed corresponds to the third
    # moment (equation 3).
    k = 3
    delta = mu + k + 1.0

    def compute_addend(a, b, c):
      """Returns addend of the bulk fall speed for given set of coefficients."""
      lambda_independent_factor = tf.math.divide_no_nan(
          a * particles.gamma(b + delta),
          particles.gamma(delta),
      )
      # Rewriting the lambda-dependent factor to allow division of the lambda
      # terms directly before exponentiation, thus preventing overflow.
      lambda_ratio = 1.0 / (1.0 + tf.math.divide_no_nan(c, lam))
      lambda_dependent_factor = tf.math.divide_no_nan(
          lambda_ratio ** delta,
          (lam + c) ** b
      )
      return lambda_independent_factor * lambda_dependent_factor

    return tf.math.add_n([
        compute_addend(a_i, b_i, c_i)
        for a_i, b_i, c_i in zip(coeffs.a, coeffs.b, coeffs.c)
    ])

  def _fall_speed_gamma_type_individual(
      self,
      coeffs: TerminalVelocityCoefficients,
      diameter: tf.Tensor,
  ) -> tf.Tensor:
    """Computes the terminal velocity of a single particle.

    This evaluates the multi-term gamma-type function given by equation 19 of
    Chen et al. (2022) excluding the aspect ratio factor, which must be computed
    separately.

    Args:
      coeffs: A wrapper for the terminal velocity coefficients `a`, `b`, and `c`
        of the gamma-type function in equation 19.
      diameter: Pointwise estimated group droplet diameter.

    Returns:
      The mass-weighted bulk fall speed for a particle group. Note that this
      result should still be scaled by the volume-weighted mean aspect ratio of
      the particle group.
    """

    def compute_addend(a, b, c):
      """Returns addend of the bulk fall speed for given set of coefficients."""
      return a * tf.math.pow(diameter, b) * tf.math.exp(-c * diameter)

    return tf.math.add_n([
        compute_addend(a_i, b_i, c_i)
        for a_i, b_i, c_i in zip(coeffs.a, coeffs.b, coeffs.c)
    ])

  def rain_terminal_velocity(
      self,
      rho: tf.Tensor,
      q_r: tf.Tensor,
  ) -> tf.Tensor:
    """Computes the terminal velocity of raindrops.

    Args:
      rho: The density of air [kg/m^3].
      q_r: The rain mass fraction [kg/kg].

    Returns:
      The terminal velocity of rain drops [m/s].
    """
    lam = particles.marshall_palmer_distribution_parameter_lambda(
        self._rain, rho, q_r
    )
    coeffs = self._compute_raindrop_coefficients(rho)
    return tf.math.maximum(self._fall_speed_gamma_type(coeffs, lam), 0.0)

  def snow_terminal_velocity(
      self,
      rho: tf.Tensor,
      q_s: tf.Tensor,
  ) -> tf.Tensor:
    """Computes the terminal velocity of snow.

    Args:
      rho: The density of air [kg/m^3].
      q_s: The snow mass fraction [kg/kg].

    Returns:
      The terminal velocity of snow flakes [m/s].
    """
    lam = particles.marshall_palmer_distribution_parameter_lambda(
        self._snow, rho, q_s
    )
    # Mass-weighted aspect ratio.
    psi_avg = tf.math.divide_no_nan(
        self._snow.phi_0, tf.math.pow(lam, self._snow.alpha)
    )
    coeffs = self._compute_snow_coefficients(rho)
    fall_speed = (
        tf.math.pow(psi_avg, self._KAPPA)
        * self._fall_speed_gamma_type(coeffs, lam)
    )
    return tf.math.maximum(fall_speed, 0.0)

  def condensate_terminal_velocity(
      self,
      particle: Rain | Ice,
      rho: tf.Tensor,
      q_sed: tf.Tensor,
  ) -> tf.Tensor:
    """Computes the sedimentation terminal velocity of cloud droplets or ice.

    Args:
      particle: Dataclass object that stores constant parameters for the
        sediment particle (either `Rain` or `Ice`). Note that `Rain` is reused
        for cloud droplets, as the parameters are the same.
      rho: The density of air [kg/m^3].
      q_sed: The sediment mass fraction [kg/kg].

    Returns:
      The terminal velocity of the sediment [m/s].
    """
    if isinstance(particle, Rain):
      coeffs = self._compute_raindrop_coefficients(rho)
      correction_factor = self._CLOUD_DROPLET_CORRECTION_FACTOR
    elif isinstance(particle, Ice):
      coeffs = self._compute_ice_coefficients(rho)
      correction_factor = 1.0
    else:
      raise ValueError(
          f'Sediment must be of type Rain or Ice, but got {type(particle)}.'
      )
    q_sed = tf.maximum(q_sed, tf.zeros_like(q_sed))
    diameter = tf.math.pow(
        rho * q_sed / constants.DROPLET_N / particle.params.rho, 1.0 / 3.0
    )
    fall_speed = self._fall_speed_gamma_type_individual(coeffs, diameter)
    return tf.math.maximum(correction_factor * fall_speed, 0.0)
