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

"""A library for the interpolation schemes."""

import enum
from typing import TypeAlias

import jax
import jax.numpy as jnp
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types


ScalarField: TypeAlias = types.ScalarField


class FluxLimiterType(enum.Enum):
  """Defines types of flux limiters."""

  VAN_LEER = 'van_leer'
  MUSCL = 'muscl'


def centered_node_to_face(
    v_node: ScalarField,
    axis: str,
    kernel_op: get_kernel_fn.ApplyKernelOp,
) -> ScalarField:
  """Performs centered 2nd-order interpolation from nodes to faces.

  * An array evaluated on nodes has index i <==> coordinate location x_i
  * An array evaluated on faces has index i <==> coordinate location x_{i-1/2}

  The axis is converted into appropriate dim based on data_axis_order.
  E.g., interpolating in dim 0:
    v_face[i, j, k] = 0.5 * (v_node[i, j, k] + v_node[i-1, j, k])

  Args:
    v_node: A 3D tensor, evaluated on nodes.
    axis: The axis along with the interpolation is performed.
    kernel_op: Kernel operation library.

  Returns:
    A 3D tensor interpolated from `v_node`, which is evaluated on faces in
    axis `axis`, and evaluated on nodes in other axes.
  """
  return 0.5 * kernel_op.apply_kernel_op(v_node, 'ks', axis)


def _get_weno_conv_kernel_op(
    k: int,
    grid_params: grid_parametrization.GridParametrization,
) -> get_kernel_fn.ApplyKernelConvOp:
  """Initializes a convolutional kernel operator with WENO related weights.

  Args:
    k: The order/stencil width of the WENO interpolation.
    grid_params: An instance of the GridParametrization class.

  Returns:
    A kernel of convolutional finite-difference operators.
  """
  # Coefficients for the interpolation and stencil selection.
  c = {
      2: {-1: [1.5, -0.5], 0: [0.5, 0.5], 1: [-0.5, 1.5]},
      3: {
          -1: [11.0 / 6.0, -7.0 / 6.0, 1.0 / 3.0],
          0: [1.0 / 3.0, 5.0 / 6.0, -1.0 / 6.0],
          1: [-1.0 / 6.0, 5.0 / 6.0, 1.0 / 3.0],
          2: [1.0 / 3.0, -7.0 / 6.0, 11.0 / 6.0],
      },
  }

  # Define the kernel operator with WENO customized weights.
  # Weights for the i + 1/2 face neg-interpolation. Values are saved at i.
  kernel_lib = {f'c{r}': (c[k][r], r) for r in range(k)}
  # Weights for the i - 1/2 face pos-interpolation. Values are saved at i.
  kernel_lib.update({f'cr{r}': (c[k][r - 1], r) for r in range(k)})
  # Weights for the smoothness measurement.
  if k == 2:  # WENO-3
    kernel_lib.update({
        'b0_0': ([1.0, -1.0], 0),
        'b1_0': ([1.0, -1.0], 1),
    })
  elif k == 3:  # WENO-5
    kernel_lib.update({
        'b0_0': ([1.0, -2.0, 1.0], 0),
        'b1_0': ([1.0, -2.0, 1.0], 1),
        'b2_0': ([1.0, -2.0, 1.0], 2),
        'b0_1': ([3.0, -4.0, 1.0], 0),
        'b1_1': ([1.0, 0.0, -1.0], 1),
        'b2_1': ([1.0, -4.0, 3.0], 2),
    })
  return get_kernel_fn.ApplyKernelConvOp(4, grid_params, kernel_lib)


def _get_weno_slice_kernel_op(
    k: int,
    grid_params: grid_parametrization.GridParametrization,
) -> get_kernel_fn.ApplyKernelSliceOp:
  """Initializes a slice kernel operator with WENO related weights.

  Args:
    k: The order/stencil width of the WENO interpolation.
    grid_params: An instance of the GridParametrization class.

  Returns:
    A kernel of slice finite-difference operators.
  """
  kernel_lib = {}
  if k == 2:
    # Weights for the i + 1/2 face neg-interpolation. Values are saved at i.
    kernel_lib.update({'c0': {'coeff': (0.5, 0.5), 'shift': (-1, 0)}})
    kernel_lib.update({'c1': {'coeff': (-0.5, 1.5), 'shift': (1, 0)}})
    # Weights for the i - 1/2 face pos-interpolation. Values are saved at i.
    kernel_lib.update({'cr0': {'coeff': (1.5, -0.5), 'shift': (0, -1)}})
    kernel_lib.update({'cr1': {'coeff': (0.5, 0.5), 'shift': (0, 1)}})
    # Weights for the smoothness measurement.
    kernel_lib.update({'b0_0': {'coeff': (1.0, -1.0), 'shift': (-1, 0)}})
    kernel_lib.update({'b1_0': {'coeff': (1.0, -1.0), 'shift': (0, 1)}})
  elif k == 3:
    # Weights for the i + 1/2 face neg-interpolation. Values are saved at i.
    kernel_lib.update({
        'c0': {
            'coeff': (1.0 / 3.0, 5.0 / 6.0, -1.0 / 6.0),
            'shift': (0, -1, -2),
        }
    })
    kernel_lib.update({
        'c1': {'coeff': (-1.0 / 6.0, 5.0 / 6.0, 1.0 / 3.0), 'shift': (1, 0, -1)}
    })
    kernel_lib.update({
        'c2': {'coeff': (1.0 / 3.0, -7.0 / 6.0, 11.0 / 6.0), 'shift': (2, 1, 0)}
    })
    # Weights for the i - 1/2 face pos-interpolation. Values are saved at i.
    kernel_lib.update({
        'cr0': {
            'coeff': (11.0 / 6.0, -7.0 / 6.0, 1.0 / 3.0),
            'shift': (0, -1, -2),
        }
    })
    kernel_lib.update({
        'cr1': {
            'coeff': (1.0 / 3.0, 5.0 / 6.0, -1.0 / 6.0),
            'shift': (1, 0, -1),
        }
    })
    kernel_lib.update({
        'cr2': {'coeff': (-1.0 / 6.0, 5.0 / 6.0, 1.0 / 3.0), 'shift': (2, 1, 0)}
    })
    # Weights for the smoothness measurement.
    kernel_lib.update(
        {'b0_0': {'coeff': (1.0, -2.0, 1.0), 'shift': (-2, -1, 0)}}
    )
    kernel_lib.update(
        {'b1_0': {'coeff': (1.0, -2.0, 1.0), 'shift': (-1, 0, 1)}}
    )
    kernel_lib.update({'b2_0': {'coeff': (1.0, -2.0, 1.0), 'shift': (0, 1, 2)}})
    kernel_lib.update(
        {'b0_1': {'coeff': [3.0, -4.0, 1.0], 'shift': [0, -1, -2]}}
    )
    kernel_lib.update(
        {'b1_1': {'coeff': [1.0, 0.0, -1.0], 'shift': [1, 0, -1]}}
    )
    kernel_lib.update({'b2_1': {'coeff': [1.0, -4.0, 3.0], 'shift': [2, 1, 0]}})
  return get_kernel_fn.ApplyKernelSliceOp(grid_params, kernel_lib)


def _calculate_weno_weights(
    v: ScalarField,
    kernel_op: get_kernel_fn.ApplyKernelOp,
    axis: str,
    k: int,
) -> tuple[list[ScalarField], list[ScalarField]]:
  """Calculates the weights for WENO interpolation from cell centered values.

  Args:
    v: A 3D tensor to which the interpolation is performed.
    kernel_op: A kernel of finite-difference operators.
    axis: The axis along with the interpolation is performed.
    k: The order/stencil width of the interpolation.

  Returns:
    A tuple of the weights for WENO interpolated values on the faces, with the
    first and second elements being the negative and positive weights at face i
    +
    1/2, respectively.
  """
  # A small constant that prevents division by zero when computing the weights
  # for stencil selection.
  eps = 1e-6

  # Linear coefficients for the interpolation using upwind
  d = {
      2: [2.0 / 3.0, 1.0 / 3.0],  # WENO-3
      3: [0.3, 0.6, 0.1],  # WENO-5
  }

  # Compute the smoothness measurement.
  if k == 2:  # WENO-3
    beta_fn = lambda f0: f0**2
    beta = [
        beta_fn(kernel_op.apply_kernel_op(v, f'b{r}_0', axis)) for r in range(k)
    ]
  elif k == 3:  # WENO-5
    beta_fn = lambda f0, f1: 13.0 / 12.0 * f0**2 + 0.25 * f1**2
    beta = [
        beta_fn(
            kernel_op.apply_kernel_op(v, f'b{r}_0', axis),
            kernel_op.apply_kernel_op(v, f'b{r}_1', axis),
        )
        for r in range(k)
    ]
  else:
    raise ValueError(f'k: {k} should be 2 or 3.')

  alpha_fn = lambda beta, dr: dr / (eps + beta) ** 2
  w_neg = [alpha_fn(beta[r], d[k][r]) for r in range(k)]
  w_pos = [alpha_fn(beta[r], d[k][k - 1 - r]) for r in range(k)]
  w_neg_sum = jnp.sum(jnp.stack(w_neg, axis=0), axis=0)
  w_pos_sum = jnp.sum(jnp.stack(w_pos, axis=0), axis=0)

  w_neg = [w / w_neg_sum for w in w_neg]
  w_pos = [w / w_pos_sum for w in w_pos]

  return w_neg, w_pos


def _reconstruct_weno_face_values(
    v: ScalarField,
    kernel_op: get_kernel_fn.ApplyKernelOp,
    axis: str,
    k: int,
) -> tuple[list[ScalarField], list[ScalarField]]:
  """Computes the reconstructed face values from cell centered values.

  Args:
    v: A 3D tensor to which the interpolation is performed.
    kernel_op: A kernel of convolutional finite-difference operators.
    axis: The axis along with the interpolation is performed.
    k: The order/stencil width of the interpolation.

  Returns:
    A tuple of the reconstructed face values for WENO interpolated values on the
    faces, with the first and second elements being the negative and positive
    weights at face i + 1/2, respectively.
  """
  # Compute the reconstructed values on faces.
  vr_neg = [kernel_op.apply_kernel_op(v, f'c{r}', axis) for r in range(k)]
  vr_pos = [kernel_op.apply_kernel_op(v, f'cr{r}', axis) for r in range(k)]
  return vr_neg, vr_pos


def _interpolate_with_weno_weights(
    v: ScalarField,
    w_neg: list[ScalarField],
    w_pos: list[ScalarField],
    vr_neg: list[ScalarField],
    vr_pos: list[ScalarField],
    axis: str,
    grid_params: grid_parametrization.GridParametrization,
) -> tuple[ScalarField, ScalarField]:
  """Performs the WENO interpolation from cell centers to faces.

  Args:
    v: A 3D tensor to which the interpolation is performed.
    w_neg: A sequence of FlowFieldVal with weights for negative side of WENO
      interpolation.
    w_pos: A sequence of FlowFieldVal with weights for negative side of WENO
      interpolation.
    vr_neg: A sequence of FlowFieldVal with reconstructed face values for
      negative side of WENO interpolation.
    vr_pos: A sequence of FlowFieldVal with reconstructed face values for
      positive side of WENO interpolation.
    axis: The axis along with the interpolation is performed.
    grid_params: An instance of the GridParametrization class.

  Returns:
    A tuple of the interpolated values on the faces, with the first and second
    elements being the negative and positive fluxes at face i + 1/2,
    respectively.
  """

  def _dot_prod(vr: list[ScalarField], w: list[ScalarField]) -> ScalarField:
    return jnp.sum(jnp.stack(vr, axis=0) * jnp.stack(w, axis=0), axis=0)

  # Compute the weighted interpolated face values.
  v_neg = _dot_prod(vr_neg, w_neg)
  v_pos = _dot_prod(vr_pos, w_pos)

  # Shift the negative face flux on the i - 1/2 face that stored at i - 1 to i.
  # With this shift, both the positive and negative face flux at i - 1/2 will
  # be stored at location i. Values on the lower end of the negative flux Tensor
  # will be set to be the same as v in the first cell along `dim`.
  axis_index = grid_params.get_axis_index(axis)
  v_neg = jnp.roll(v_neg, shift=1, axis=axis_index)
  v_slice = jax.lax.dynamic_slice_in_dim(
      v, start_index=0, slice_size=1, axis=axis_index
  )
  v_neg = jax.lax.dynamic_update_slice_in_dim(
      v_neg, v_slice, start_index=0, axis=axis_index
  )

  return v_neg, v_pos


def weno(
    v: ScalarField,
    axis: str,
    k: int,
    grid_params: grid_parametrization.GridParametrization,
    kernel_type: str,
) -> tuple[ScalarField, ScalarField]:
  """Performs the WENO interpolation from cell centers to faces.

  Args:
    v: A 3D tensor to which the interpolation is performed.
    axis: The axis along which the interpolation is performed.
    k: The order/stencil width of the interpolation.
    grid_params: An instance of the GridParametrization class.
    kernel_type: The type of kernel operator to use. Should be one of 'conv' or
      'slice'.

  Returns:
    A tuple of the interpolated values on the faces, with the first and second
    elements being the negative and positive fluxes at face i - 1/2,
    respectively.
  """
  if kernel_type == 'conv':
    kernel_op = _get_weno_conv_kernel_op(k, grid_params)
  elif kernel_type == 'slice':
    kernel_op = _get_weno_slice_kernel_op(k, grid_params)
  else:
    raise ValueError(
        f'kernel_type: {kernel_type} should be one of conv, slice.'
    )
  w_neg, w_pos = _calculate_weno_weights(v, kernel_op, axis, k)
  vr_neg, vr_pos = _reconstruct_weno_face_values(v, kernel_op, axis, k)
  v_neg, v_pos = _interpolate_with_weno_weights(
      v, w_neg, w_pos, vr_neg, vr_pos, axis, grid_params
  )
  return v_neg, v_pos


def flux_limiter(
    v: ScalarField,
    axis: str,
    limiter_type: FluxLimiterType,
    grid_params: grid_parametrization.GridParametrization,
    kernel_type: str,
) -> tuple[ScalarField, ScalarField]:
  """Performs interpolation using a limiter scheme.

  Computes the interpolation of the field variable to the faces using 2nd-order
  Upwind + a flux limiter.

  Args:
    v: A 3D tensor to which the interpolation is performed.
    axis: The axis along which the interpolation is performed.
    limiter_type: The type of flux limiter.
    grid_params: An instance of the GridParametrization class.
    kernel_type: The type of kernel operator to use. Should be one of 'conv' or
      'slice'.

  Returns:
    A tuple that represents the interpolated values of a variable onto faces
    that are normal to `axis`, with the first element being interpolated from
    the upwind/left-biased stencil and the second from the downwind/right-biased
    stencil.

  Raises:
    NotImplementedError if `limiter_type` is not one of the follwoing:
      'VAN_LEER', 'MUSCL'.
  """

  def van_leer(r: ScalarField) -> ScalarField:
    """Computes the Van Leer flux limiter."""
    van_leer_fn = lambda x: 2 * x / (1 + x)
    return jnp.where(r >= 0.0, van_leer_fn(r), 0.0)

  def muscl(r: ScalarField) -> ScalarField:
    """Computes the MUSCL flux limiter."""
    min_arg_1 = 2.0 * r
    min_arg_2 = 0.5 * (1.0 + r)
    min_arg = jnp.minimum(min_arg_1, min_arg_2)
    return jnp.maximum(0.0, jnp.minimum(2.0, min_arg))

  if limiter_type == FluxLimiterType.VAN_LEER:
    limiter = van_leer
  elif limiter_type == FluxLimiterType.MUSCL:
    limiter = muscl
  else:
    raise NotImplementedError(
        f'{limiter_type.name} is not supported. Available flux limiters are:'
        f' {list(FluxLimiterType)}'
    )
  if kernel_type == 'conv':
    kernel_op = get_kernel_fn.ApplyKernelConvOp(
        4, grid_params, {'shift_kernel': ([1.0, 0.0, 0.0], 1)}
    )
  elif kernel_type == 'slice':
    kernel_op = get_kernel_fn.ApplyKernelSliceOp(
        grid_params, {'shift_kernel': {'coeff': (1.0,), 'shift': (1,)}}
    )
  else:
    raise ValueError(
        f'kernel_type: {kernel_type} should be one of conv, slice.'
    )

  # Define functions
  v_shifted = kernel_op.apply_kernel_op(v, 'shift_kernel', axis)  # v[i - 1]
  diff1 = kernel_op.apply_kernel_op(v, 'kd+', axis)  # v[i + 1] - v[i]
  diff2 = kernel_op.apply_kernel_op(v, 'kd', axis)  # v[i] - v[i - 1]
  diff3 = kernel_op.apply_kernel_op(
      diff2, 'shift_kernel', axis
  )  # v[i - 1] - v[i - 2]

  def divide_no_nan(x: ScalarField, y: ScalarField) -> ScalarField:
    """Divides x by y, or returns 0 if y is 0."""
    return jnp.where(y == 0, 0, x / y)

  # Compute r = (state_D - state_U) / (state_U - state_UU).
  r_pos = divide_no_nan(diff2, diff3)
  r_neg = divide_no_nan(diff2, diff1)

  psi_pos = limiter(r_pos)
  psi_neg = limiter(r_neg)

  v_neg = v_shifted + 0.5 * psi_pos * diff3
  v_pos = v - 0.5 * psi_neg * diff1

  return v_neg, v_pos
