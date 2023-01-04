# Copyright 2022 The swirl_lm Authors.
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

import functools
from typing import Tuple, Dict, Sequence, Union

from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf


def _get_weno_kernel_op(
    k: int = 3,
)-> get_kernel_fn.ApplyKernelConvOp:
  """Initializes a convolutional kernel operator with WENO related weights.

  Args:
    k: The order/stencil width of the interpolation.

  Returns:
    A kernel of convolutional finite-difference operators.
  """
  # Coefficients for the interpolation and stencil selection.
  c = {
      3: {
          -1: [11.0 / 6.0, -7.0 / 6.0, 1.0 / 3.0,],
          0: [1.0 / 3.0, 5.0 / 6.0, -1.0 / 6.0],
          1: [-1.0 / 6.0, 5.0 / 6.0, 1.0 / 3.0],
          2: [1.0 / 3.0, -7.0 / 6.0, 11.0 / 6.0],
          }
  }

  # Define the kernel operator with WENO customized weights.
  # Weights for the i + 1/2 face interpolation. Values are saved at i.
  kernel_lib = {
      f'c{r}': (c[k][r], r) for r in range(k)
  }
  # Weights for the i - 1/2 face interpolation. Values are saved at i.
  kernel_lib.update({
      f'cr{r}': (c[k][r - 1], r) for r in range(k)
  })
  # Weights for the smoothness measurement.
  kernel_lib.update({
      'b0_0': ([1.0, -2.0, 1.0], 0),
      'b1_0': ([1.0, -2.0, 1.0], 1),
      'b2_0': ([1.0, -2.0, 1.0], 2),
      'b0_1': ([3.0, -4.0, 1.0], 0),
      'b1_1': ([1.0, 0.0, -1.0], 1),
      'b2_1': ([1.0, -4.0, 3.0], 2),
  })
  kernel_op = get_kernel_fn.ApplyKernelConvOp(4, kernel_lib)
  return kernel_op


def _calculate_weno_weights(
    v: types.FlowFieldVal,
    kernel_op: get_kernel_fn.ApplyKernelConvOp,
    dim: str,
    k: int = 3,
) -> Tuple[Dict[int, Union[Sequence[tf.Tensor], tf.Tensor]], Dict[int, Union[
    Sequence[tf.Tensor], tf.Tensor]]]:
  """Calculates the weights for WENO interpolation from cell centered values.

  Args:
    v: A 3D tensor to which the interpolation is performed.
    kernel_op: A kernel of convolutional finite-difference operators.
    dim: The dimension along with the interpolation is performed.
    k: The order/stencil width of the interpolation.

  Returns:
    A tuple of the weights for WENO interpolated values on the faces, with the
    first and second elements being the negative and postive weights at face i +
    1/2, respectively.
  """
  # A small constant that prevents division by zero when computing the weights
  # for stencil selection.
  eps = 1e-6

  # Linear coefficients for the interpolation using upwind
  d = {
      3: {0: 0.3, 1: 0.6, 2: 0.1},
  }

  kernel_fn = {
      'x':
          lambda u, name: kernel_op.apply_kernel_op_x(u, f'{name}x'),
      'y':
          lambda u, name: kernel_op.apply_kernel_op_y(u, f'{name}y'),
      'z':
          lambda u, name: kernel_op.apply_kernel_op_z(u, f'{name}z',  # pylint: disable=g-long-lambda
                                                      f'{name}zsh')
  }[dim]

  # Compute the smoothness measurement.
  beta_fn = lambda f0, f1: 13.0 / 12.0 * f0**2 + 0.25 * f1**2
  beta = {
      r: tf.nest.map_structure(beta_fn, kernel_fn(v, f'b{r}_0'),
                               kernel_fn(v, f'b{r}_1')) for r in range(k)
  }

  # Compute the WENO weights.
  w_neg = {}
  w_pos = {}
  w_neg_sum = tf.nest.map_structure(tf.zeros_like, beta[0])
  w_pos_sum = tf.nest.map_structure(tf.zeros_like, beta[0])

  alpha_fn = lambda beta, dr: dr / (eps + beta)**2

  for r in range(k):
    w_neg.update({
        r:
            tf.nest.map_structure(
                functools.partial(alpha_fn, dr=d[k][r]), beta[r])
    })
    w_pos.update({
        r:
            tf.nest.map_structure(
                functools.partial(alpha_fn, dr=d[k][k - 1 - r]), beta[r])
    })
    w_neg_sum = tf.nest.map_structure(tf.math.add, w_neg_sum, w_neg[r])
    w_pos_sum = tf.nest.map_structure(tf.math.add, w_pos_sum, w_pos[r])

  for r in range(k):
    w_neg[r] = tf.nest.map_structure(tf.math.divide_no_nan, w_neg[r], w_neg_sum)
    w_pos[r] = tf.nest.map_structure(tf.math.divide_no_nan, w_pos[r], w_pos_sum)

  return w_neg, w_pos


def _reconstruct_weno_face_values(
    v: types.FlowFieldVal,
    kernel_op: get_kernel_fn.ApplyKernelConvOp,
    dim: str,
    k: int = 3,
) -> Tuple[Dict[int, Union[Sequence[tf.Tensor], tf.Tensor]], Dict[int, Union[
    Sequence[tf.Tensor], tf.Tensor]]]:
  """Computes the reconstructed face values from cell centered values.

  Args:
    v: A 3D tensor to which the interpolation is performed.
    kernel_op: A kernel of convolutional finite-difference operators.
    dim: The dimension along with the interpolation is performed.
    k: The order/stencil width of the interpolation.

  Returns:
    A tuple of the reconstructed face values for WENO interpolated values on the
    faces, with the first and second elements being the negative and postive
    weights at face i + 1/2, respectively.
  """
  kernel_fn = {
      'x':
          lambda u, name: kernel_op.apply_kernel_op_x(u, f'{name}x'),
      'y':
          lambda u, name: kernel_op.apply_kernel_op_y(u, f'{name}y'),
      'z':
          lambda u, name: kernel_op.apply_kernel_op_z(u, f'{name}z',  # pylint: disable=g-long-lambda
                                                      f'{name}zsh')
  }[dim]

  # Compute the reconstructed values on faces.
  vr_neg = {}
  vr_pos = {}
  for r in range(k):
    vr_neg.update({r: kernel_fn(v, f'c{r}')})
    vr_pos.update({r: kernel_fn(v, f'cr{r}')})

  return vr_neg, vr_pos


def _interpolate_with_weno_weights(
    v: types.FlowFieldVal,
    w_neg: Dict[int, Union[Sequence[tf.Tensor], tf.Tensor]],
    w_pos: Dict[int, Union[Sequence[tf.Tensor], tf.Tensor]],
    vr_neg: Dict[int, Union[Sequence[tf.Tensor], tf.Tensor]],
    vr_pos: Dict[int, Union[Sequence[tf.Tensor], tf.Tensor]],
    dim: str,
    k: int = 3,
) -> Tuple[types.FlowFieldVal, types.FlowFieldVal]:
  """Performs the WENO interpolation from cell centers to faces.

  Args:
    v: A 3D tensor to which the interpolation is performed.
    w_neg: A dictionary of 1D tensors with weights for negative side of WENO
      interpolation.
    w_pos: A dictionary of 1D tensors with weights for negative side of WENO
      interpolation.
    vr_neg: A dictionary of 1D tensors with reconstructed face values for
      negative side of WENO interpolation.
    vr_pos: A dictionary of 1D tensors with reconstructed face values for
      positive side of WENO interpolation.
    dim: The dimension along with the interpolation is performed.
    k: The order/stencil width of the interpolation.

  Returns:
    A tuple of the interpolated values on the faces, with the first and second
    elements being the negative and postive fluxes at face i + 1/2,
    respectively.
  """

  # Compute the weighted interpolated face values.
  v_neg = tf.nest.map_structure(tf.zeros_like, v)
  v_pos = tf.nest.map_structure(tf.zeros_like, v)
  prod_sum_fn = lambda s, w, u: s + w * u
  for r in range(k):
    v_neg = tf.nest.map_structure(prod_sum_fn, v_neg, w_neg[r], vr_neg[r])
    v_pos = tf.nest.map_structure(prod_sum_fn, v_pos, w_pos[r], vr_pos[r])

  # Shift the positive face flux on the i - 1/2 face that stored at i to i - 1.
  # With this shift, both the positive and negative face flux at i + 1/2 will
  # be stored at location i. Values on the higher end of the postive flux Tensor
  # will be set to be the same as v in the last cell along `dim`.
  if dim == 'x':
    v_pos = tf.nest.map_structure(
        lambda u, v: tf.concat([u[1:, :], v[-1:, :]], 0), v_pos, v)
  elif dim == 'y':
    v_pos = tf.nest.map_structure(
        lambda u, v: tf.concat([u[:, 1:], v[:, -1:]], 1), v_pos, v)
  else:  # dim == 'z':
    if isinstance(v, tf.Tensor):
      v_pos = tf.concat([v_pos[1:, ...], v[-1:, ...]], 0)
    else:  # v and v_pos are lists.
      v_pos = v_pos[1:] + [v[-1]]

  return v_neg, v_pos


def weno(
    v: types.FlowFieldVal,
    dim: str,
    k: int = 3,
) -> Tuple[types.FlowFieldVal, types.FlowFieldVal]:
  """Performs the WENO interpolation from cell centers to faces.

  Args:
    v: A 3D tensor to which the interpolation is performed.
    dim: The dimension along with the interpolation is performed.
    k: The order/stencil width of the interpolation.

  Returns:
    A tuple of the interpolated values on the faces, with the first and second
    elements being the negative and postive fluxes at face i + 1/2,
    respectively.
  """
  kernel_op = _get_weno_kernel_op(k)
  w_neg, w_pos = _calculate_weno_weights(v, kernel_op, dim, k)
  vr_neg, vr_pos = _reconstruct_weno_face_values(v, kernel_op, dim, k)
  v_neg, v_pos = _interpolate_with_weno_weights(v, w_neg, w_pos, vr_neg, vr_pos,
                                                dim, k)

  return v_neg, v_pos
