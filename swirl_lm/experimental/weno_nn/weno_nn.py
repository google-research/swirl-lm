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

"""WENO NN scheme written in Jax.

We implement the Neural Network for computing interpolation weights in [1].

This library assumes that the double precision is turned on particularly during
training. Once the weights are trained one can use lower accuracy.

Turning double precision on can achieved by setting the value of the
flag jax_enable_x64 to true by adding

>>> from jax import config
>>> config.update("jax_enable_x64", True)

to the beginning of the code.

Refs:

[1] D. Bezgin, S.J. Schmidt and N.A. Klaus, "WENO3-NN: A maximum-order
three-point data-driven weighted essentially non-oscillatory scheme",
Journal of Computational Physics, Volume 452, Issue C, Mar 2022.

[2] G.S. Jiang, C.W. Shu, "Efficient implementation of weighted ENO schemes",
Journal of Computational Physics, 126 (1) (1996) 202-228.
"""

import functools
from typing import Any, Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_lm.experimental.weno_nn import rational_networks

PyTree = Any

# Activation functions with unshared trainable parameters.
_UNSHARED_ACTIVATION_FUNCTIONS = {
    "rational_act_fun": rational_networks.RationalLayer,
    "unshared_rational_act_fun": rational_networks.UnsharedRationalLayer,
}


def upwind_weights(order: int = 3) -> tuple[jax.Array, jax.Array]:
  """Computes the upwind weights.

  Args:
    order: Order of the interpolation of the ENO polynomials.

  Returns:
    The optimal weights for the ENO polynomials for third order reconstruction
    for smooth functions.
  """

  if order not in [3, 5]:
    raise ValueError(
        "Only 3rd and 5th order polynomials are supported, not ",
        f"order {order}.",
    )

  d_minus, d_plus = jnp.array([0.0]), jnp.array([0.0])

  if order == 3:
    d_minus = jnp.array([2.0, 1.0], dtype=jnp.float64) / 3.0
    d_plus = jnp.array([1.0, 2.0], dtype=jnp.float64) / 3.0

  elif order == 5:
    d_minus = 0.1 * jnp.array([3.0, 6.0, 1.0], dtype=jnp.float64)
    d_plus = 0.1 * jnp.array([1.0, 6.0, 3.0], dtype=jnp.float64)

  return (d_minus, d_plus)  # pytype: disable=bad-return-type  # jnp-type


def beta(u_bar: jax.Array, order: int = 3) -> jax.Array:
  """Computes the smoothness indicators in (10) of [1].

  Args:
    u_bar: jax.Array containing (u_{n-1}, u_n, u_{n+1}).
    order: Order of the interpolation of the ENO polynomials.

  Returns:
    The indicators of the local smoothness, β, of the function.
  """
  if order not in [3, 5]:
    raise ValueError(
        "Only 3rd and 5th order polynomials are supported, not ",
        f"order {order}.",
    )

  beta_array = jnp.array([0.0, 0.0])

  if order == 3:
    beta_0 = jnp.square(u_bar[1] - u_bar[0])
    beta_1 = jnp.square(u_bar[2] - u_bar[1])

    beta_array = jnp.array([beta_0, beta_1])

  elif order == 5:
    # from Eqs. (3.1), (3.2), and (3.3) in [2].
    beta_0 = 13.0 / 12.0 * (
        jnp.square(u_bar[0] - 2.0 * u_bar[1] + u_bar[2])
    ) + 0.25 * (jnp.square(u_bar[0] - 4.0 * u_bar[1] + 3 * u_bar[2]))
    beta_1 = 13.0 / 12.0 * (
        jnp.square(u_bar[1] - 2.0 * u_bar[2] + u_bar[3])
    ) + 0.25 * (jnp.square(u_bar[1] - u_bar[3]))
    beta_2 = 13.0 / 12.0 * (
        jnp.square(u_bar[2] - 2.0 * u_bar[3] + u_bar[4])
    ) + 0.25 * (jnp.square(3 * u_bar[2] - 4.0 * u_bar[3] + u_bar[4]))

    beta_array = jnp.array([beta_0, beta_1, beta_2])

  return beta_array


def omega_plus(
    u_bar: jax.Array,
    order: int = 3,
    p: int = 2,
    eps: jnp.float64 = 1e-15,
) -> jax.Array:
  r"""Computes the WENO weights in the interpolation.

  Args:
    u_bar: Array containing the cell averages of the approximation solution
      following (\bar{u}_{n-1}, \bar{u}_n, \bar{u}_{n+1}).
    order: Order of the interpolation of the ENO polynomials.
    p: Polynomial degree of the smoothness indicator.
    eps: Regularizer to avoid division by zero.

  Returns:
    The interpolation weights for the ENO polynomials.
  """
  # Unpacking the computation of β as described in Eq. (10) in [1].
  beta_w = beta(u_bar, order)

  # Extracting the upwind weights d₀ and d₁ as described in Eq. (8) in [1].
  _, d_plus = upwind_weights(order)

  # Computing α as described in Eq. (8) in [1].
  alpha = d_plus / jnp.power(beta_w + eps, p)

  # Computing ω as described in Eq. (8) in [1].
  alpha_sum = jnp.sum(alpha)
  omega = alpha / alpha_sum

  return omega


def interpolants_plus(u_bar: jax.Array, order: int = 3) -> jax.Array:
  r"""Computes the polynomial interpolants.

  We follow [2] to compute the interpolants, using the notation in [1].

  Args:
    u_bar: Array containing the cell averages of the approximation solution
      following (\bar{u}_{n-1}, \bar{u}_n, \bar{u}_{n+1}).
    order: Order of the underlying interpolation for smooth functions.

  Returns:
    The ENO polynomials evaluated at x_{n+1/2}.
  """
  if len(u_bar) != order:
    raise ValueError(
        f"Input size ({len(u_bar)}) and polynomial order ",
        f"({order}) do not match.",
    )

  eno_polynomials = jnp.array([0.0, 0.0])

  if order == 3:
    # u^0_{i+1/2} = 0.5( -u_{i-1} + 3 u_{i}).
    u_plus_0 = 0.5 * (-u_bar[0] + 3 * u_bar[1])
    # u^1_{i+1/2} = 0.5(u_{i} + u_{i+1}).
    u_plus_1 = 0.5 * (u_bar[1] + u_bar[2])

    eno_polynomials = jnp.array([u_plus_0, u_plus_1])

  elif order == 5:

    # u^0_{i+1/2} = ( -2 u_{i-2} - 7 u_{i-1} + 11 * u_{0}) / 6.
    u_plus_0 = (2 * u_bar[0] - 7 * u_bar[1] + 11 * u_bar[2]) / 6.0
    # u^1_{i+1/2} = ( - u_{i-1} + 5 u_{i} + 2 * u_{i+1}) / 6.
    u_plus_1 = (-u_bar[1] + 5 * u_bar[2] + 2 * u_bar[3]) / 6.0
    # u^1_{i+1/2} = ( 2 u_{i} + 5 u_{i+1} -1 u_{i+2}) / 6.
    u_plus_2 = (2 * u_bar[2] + 5 * u_bar[3] - u_bar[4]) / 6.0

    eno_polynomials = jnp.array([u_plus_0, u_plus_1, u_plus_2])

  return eno_polynomials


def interpolants_minus(u_bar: jax.Array, order: int = 3) -> jax.Array:
  r"""Computes the third order interpolants in Eq. (7) in [1].

  Args:
    u_bar: Array containing the cell averages of the approximation solution
      following (\bar{u}_{n-1}, \bar{u}_n, \bar{u}_{n+1}).
    order: Order of the underlying interpolation for smooth functions.

  Returns:
    The ENO polynomials evaluated at x_{n-1/2}.
  """

  if len(u_bar) != order:
    raise ValueError(
        f"Input size ({len(u_bar)}) and polynomial order ",
        f"({order}) do not match.",
    )

  eno_polynomials = jnp.array([0.0, 0.0])

  if order == 3:
    # u^1_{i-1/2} = 0.5(u_{i-1} + u_{i}).
    u_minus_0 = 0.5 * (u_bar[0] + u_bar[1])
    # u^0_{i-1/2} = 0.5( -u_{i+1} + 3 u_{i}).
    u_minus_1 = 0.5 * (3 * u_bar[1] - u_bar[2])

    eno_polynomials = jnp.array([u_minus_1, u_minus_0])

  elif order == 5:
    # u^1_{i-1/2} = ( 2 u_{i} + 5 u_{i+1} -1 u_{i+2}) / 6.
    u_minus_0 = (-u_bar[0] + 5 * u_bar[1] + 2 * u_bar[2]) / 6.0
    # u^1_{i-1/2} = ( - u_{i-1} + 5 u_{i} + 2 * u_{i+1}) / 6.
    u_minus_1 = (2 * u_bar[1] + 5 * u_bar[2] - u_bar[3]) / 6.0
    # u^0_{i-1/2} = ( -2 u_{i-2} - 7 u_{i-1} + 11 * u_{0}) / 6.
    u_minus_2 = (11 * u_bar[2] - 7 * u_bar[3] + 2 * u_bar[4]) / 6.0

    eno_polynomials = jnp.array([u_minus_2, u_minus_1, u_minus_0])

  return eno_polynomials


def weno_interpolation_plus(
    u_bar: jax.Array,
    omega_fun: Callable[[jax.Array, Optional[int]], jax.Array],
    order: int = 3,
) -> jnp.float64:
  r"""Interpolation to u_{i+1/2}.

  Args:
    u_bar: Array containing the cell averages of the approximation solution
      following (\bar{u}_{n-1}, \bar{u}_n, \bar{u}_{n+1}).
    omega_fun: Function that computes the weights from the average of u, u_bar.
    order: Order of the method.

  Returns:
    The value of u interpolated to x_{i+1/2}.
  """
  assert len(u_bar) == order, (
      "Input size and order do not match. They should be equal. ",
      f"Instead they are : input shape {len(u_bar)} and order {order}.",
  )

  # Computing ω.
  omega = omega_fun(u_bar, order)

  # Computing the interpolants (3rd order in  Eq. (7) in [1]).
  u_interp = interpolants_plus(u_bar, order)

  # Computing the interpolant following Eq. (6) in [1] using dot product.
  u_plus = jnp.dot(omega, u_interp)

  return u_plus


def weno_interpolation(
    u_bar: jax.Array,
    omega_fun: Callable[[jax.Array, Optional[int]], jax.Array],
    order: int = 3,
) -> jax.Array:
  r"""Interpolation to u_{i-1/2} and u_{i+1/2}.

  Args:
    u_bar: jax.Array containing the cell averages following
      \bar{u}_{n-1}, \bar{u}_n, \bar{u}_{n+1}).
    omega_fun: Function that computes the weights from the cell average of u,
      u_bar.
    order: Order of the method.

  Returns:
    The value of u interpolated to x_{i-1/2} and x_{i+1/2}.
  """
  if len(u_bar) != order:
    raise ValueError(
        f"Input size ({len(u_bar)}) and polynomial order ",
        f"({order}) do not match.",
    )

  # Computing ω.
  omega_p = omega_fun(u_bar, order)
  omega_m = omega_fun(u_bar[::-1], order)

  # Computing the interpolants (3rd order in  Eq. (7) in [1]).
  u_inter_p = interpolants_plus(u_bar, order)
  u_inter_m = interpolants_minus(u_bar, order)

  # Computing the interpolant following Eq. (6) in [1] using dot product.
  u_plus = jnp.dot(omega_p, u_inter_p)
  u_minus = jnp.dot(omega_m, u_inter_m)

  return jnp.array([u_minus, u_plus])


def _delta_layer(
    u_bar: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  r"""Helper function computing an unnormalized delta layer.

  Args:
    u_bar: Array containing the cell averages of the function to interpolate
    following (\bar{u}_{n-1}, \bar{u}_n, \bar{u}_{n+1}).

  Returns:
    A tuple containing the absolute value of the first and second order
    finite-differences of u_bar.
  """

  delta_1 = jnp.abs(u_bar[1] - u_bar[0])
  delta_2 = jnp.abs(u_bar[2] - u_bar[1])
  delta_3 = jnp.abs(u_bar[2] - u_bar[0])
  delta_4 = jnp.abs(u_bar[2] - 2 * u_bar[1] + u_bar[0])

  return (delta_1, delta_2, delta_3, delta_4)


def delta_layer(
    u_bar: jax.Array,
    global_norm: jnp.float64 | None = None,
    eps: jnp.float64 = 1e-15,
) -> jax.Array:
  r"""Implementation of Delta layer that outputs the features of the network.

  Implementation of Eqs. (14) and (15) in [1].

  Args:
    u_bar: Array containing the cell averages of the approximation solution
      following (\bar{u}_{n-1}, \bar{u}_n, \bar{u}_{n+1}).
    global_norm: Possible global normalization instead of using the input
      dependent normalization in Eq. (15) of [1].
    eps: Small number to avoid divisions by zero, following [1].

  Returns:
    Absolute value of the first and second order finite-differences of u_bar,
    normalized with the maximum absolute value between the forward and backward
    first order finite differences.
  """

  (delta_1, delta_2, delta_3, delta_4) = _delta_layer(u_bar)

  # Initializing a global normalization constant if provided.
  if not global_norm:
    re_norm = jnp.clip(jnp.maximum(delta_1, delta_2), min=eps)
  else:
    re_norm = global_norm

  return jnp.stack([delta_1, delta_2, delta_3, delta_4]) / re_norm


class FeaturesRationalLayer(nn.Module):
  """Fully rational layer for the input features.

  Attributes:
    dtype: Data type for the rational network.
    cutoff: Shift for the thresholding.
  """

  dtype: jnp.dtype = jnp.float64
  cutoff: Optional[jnp.float64] = None

  @nn.compact
  def __call__(self, u_bar: jax.Array) -> jax.Array:
    """Application of the of rational feature layer.

    Args:
      u_bar: jax.Array containing (u_{n-1}, u_n, u_{n+1}).

    Returns:
      Four normalized features (between 0 and 1) for the input to the network.
    """

    (delta_1, delta_2, delta_3, delta_4) = _delta_layer(u_bar)

    delta = jnp.stack([delta_1, delta_2, delta_3, delta_4])
    output = rational_networks.UnsharedRationalLayer(
        dtype=self.dtype, cutoff=self.cutoff
    )(delta)

    norm = jnp.linalg.norm(output)
    if self.cutoff:
      norm = hard_thresholding(
          norm, threshold_value=self.cutoff, cutoff=self.cutoff
      )

    return output / norm


class FeaturesRationalLayerDescentered(nn.Module):
  """Implementation of rational layer with descentered stencils.

  Attributes:
    dtype: Data type for the rational network.
    cutoff: Shift for the thresholding.
  """

  dtype: jnp.dtype = jnp.float64
  cutoff: Optional[jnp.float64] = None

  @nn.compact
  def __call__(self, u_bar: jax.Array) -> jax.Array:
    """Application of the of rational feature layer.

    Args:
      u_bar: jax.Array containing (u_{n-1}, u_n, u_{n+1}).

    Returns:
      Four normalized features (between 0 and 1) for the input to the network.
    """

    (delta_1, delta_2, delta_3, delta_4) = _delta_layer(u_bar)

    delta_5 = jnp.abs(-0.5 * u_bar[2] + 2 * u_bar[1] - 1.5 * u_bar[0])
    delta_6 = jnp.abs(-1.5 * u_bar[2] + 2 * u_bar[1] - 0.5 * u_bar[0])

    delta = jnp.stack([delta_1, delta_2, delta_3, delta_4, delta_5, delta_6])

    output = rational_networks.UnsharedRationalLayer(dtype=self.dtype)(delta)

    norm = jnp.linalg.norm(output)
    if self.cutoff:
      norm = hard_thresholding(
          norm, threshold_value=self.cutoff, cutoff=self.cutoff
      )

    return output / norm


def hard_thresholding(
    x: jnp.float64,
    threshold_value: jnp.float64,
    cutoff: jnp.float64 = 2e-4,
) -> jnp.float64:
  """Simple implementation of hard thresholding in Eq. (16) of [1].

  Args:
    x: Number to be thresholded.
    threshold_value: Value used if x<cutoff
    cutoff: Shift for the thresholding.

  Returns:
    The input, which we assume is a scalar, hard-thresholded by cutoff. Namely
    x if x > cutoff, `threshold_value` otherwise.
  """
  return jax.lax.cond(x < cutoff, lambda x: threshold_value, lambda x: x, x)


def eno_layer(omega: jax.Array, cutoff: jnp.float64 = 2e-4) -> jax.Array:
  """Implementation of the ENO_layer that thresholds and normalizes the weights.

  Args:
    omega: jax.Array with two elements containing the output of the network.
    cutoff: Cutoff for the hard thresholding.

  Returns:
    The hard thresholded and re-weighted version of omega.
  """
  omega_tilde = jax.vmap(hard_thresholding, in_axes=(0, None, None))(
      omega, 0.0, cutoff
  )
  norm_omega = jnp.sum(omega_tilde)
  omega_tilde = omega_tilde / norm_omega

  return omega_tilde


def gamma(u_bar: jax.Array, epsilon_gamma: jnp.float64 = 1e-15) -> jax.Array:
  """Computation of gamma in Eq. (22) of [1].

  Args:
    u_bar: jax.Array containing (u_{n-1}, u_n, u_{n+1}).
    epsilon_gamma: Regularized to avoid division by zero.

  Returns:
    Estimators of the smoothness of the function using the ratio between
    approximations of the second and first derivatives.
  """
  return jnp.abs(u_bar[0] - 2 * u_bar[1] + u_bar[2]) / (
      jnp.abs(u_bar[1] - u_bar[0])
      + jnp.abs(u_bar[2] - u_bar[1])
      + epsilon_gamma
  )


class OmegaNN(nn.Module):
  """Layer for computing the weights of the interpolants.

  Attributes:
    features: Number of neurons for the hidden layers.
    order: Order of the weno scheme. Default to 3.
    features_fun: Function that computes the input features to the MLP based on
      the input to the network. Options are the delta layer in [1], and features
      created using rational networks.
    act_fun: Activation function for the hidden layers. If act_fun is a
      Callable, meaning that the activation function is shared among all the
      neurons we just applied it. If act_fun is a string, it means that we have
      an activation function with trainable weights that are not shared across
      layers.
    act_fun_out: Activation function for the last (output) layer.
    dtype: Type of input/outputs and parameters.
    global_norm: If non-zero, it becomes a global normalization constant for the
      delta layers, instead of local normalization used by default.
    eno_layer_cutoff: Cutoff for the hard thresholding inside the ENO layer. The
      ENO layer should only be used during inference.
  """

  features: tuple[jnp.int64, ...]
  order: int = 3
  features_fun: Callable[[jax.Array], jax.Array] = functools.partial(
      delta_layer, global_norm=None, eps=1e-15
  )
  act_fun: Callable[[jax.Array], jax.Array] | str = nn.swish
  act_fun_out: Callable[[jax.Array], jax.Array] = nn.softmax
  dtype: jnp.dtype = jnp.float64
  global_norm: jnp.float64 | None = None
  eno_layer_cutoff: jnp.float64 = 2e-4

  @nn.compact
  def __call__(self, u_bar: jax.Array, test: bool = False) -> jax.Array:
    """Computation of the weights for the interpolation polynomials.

    Args:
      u_bar: the average of u a within the cells, [u_{i-1}, u_i, u_{i+1}].
      test: flag for change between training and testing. For the latter the ENO
        layer is active.

    Returns:
      The WENO_NN weights, which are the weights for interpolation.
    """
    delta = self.features_fun(u_bar)

    # Forcing the output to be consistent with the WENO order.
    features = self.features[:] + (self.order - 1,)

    for feats in features[:-1]:
      delta = nn.Dense(
          features=feats, param_dtype=self.dtype, dtype=self.dtype
      )(delta)

      # Apply the activation function.
      if isinstance(self.act_fun, str):
        if self.act_fun in _UNSHARED_ACTIVATION_FUNCTIONS:
          delta = _UNSHARED_ACTIVATION_FUNCTIONS[self.act_fun]()(delta)
        else:
          raise ValueError(
              f"Activation function {self.act_fun} not supported for WENO_NN."
          )
      else:
        delta = self.act_fun(delta)

    # Following [1], the last layer has a different activation function.
    omega_out = self.act_fun_out(
        nn.Dense(
            features=features[-1], param_dtype=self.dtype, dtype=self.dtype
        )(delta)
    )

    # Using the ENO layer during inference.
    if test:
      omega_out = eno_layer(omega_out, self.eno_layer_cutoff)

    return omega_out
