# Copyright 2024 The swirl_lm Authors.
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

"""A library for filter operators."""

import functools
from typing import Callable, Optional
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal


def filter_op(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    f: FlowFieldVal,
    order: Optional[int] = 2,
) -> FlowFieldVal:
  """Performs filtering to a variable.

  Args:
    kernel_op: An ApplyKernelOp instance to use in computing the step update.
    f: The 3D variable to be filtered. Values in halos must be valid.
    order: The harmonic order of the filter.

  Returns:
    The filtered variable f, with values in the outermost layer being
    unfiltered.

  Raises:
    NotImplementedError: If order is not 2.
  """
  if order == 2:
    return filter_2(kernel_op, f)
  else:
    raise NotImplementedError(
        'Order {} filter is not supported. Available orders: 2.'.format(order))


def filter_2(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    f: FlowFieldVal,
    stencil: Optional[int] = 27,
) -> FlowFieldVal:
  r"""Performs filtering by adding second-order derivatives to a variable.

  For stencil = 7:
  \bar{f}_ijk = (1 - s)f_ijk + s/6(f_{i-1,j,k} + f_{i+1,j,k} +
      f_{i,j-1,k} + f_{i,j+1,k} + f_{i,j,k-1} + f_{i,j,k+1})
  For s = 0.5,
  \bar{f}_ijk = f_ijk + 1/12(\nabla^2_x f_ijk + \nabla^2_y f_ijk +
      \nabla^2_z f_ijk)
  For stencil = 27:
  \bar{f}_i = f_i + 1/4\nabla^2_x f_i
  \bar{f}_ij = \bar{f}_i + 1/4\nabla^2_y \bar{f}_i
  \bar{f}_ijk = \bar{f}_ij + 1/4\nabla^2_z \bar{f}_ij

  Reference:
  Haltiner, George J., and Roger Terry Williams. Numerical prediction and
  dynamic meteorology. No. 551.5 HAL. 1980, p. 392-397.

  Args:
    kernel_op: An ApplyKernelOp instance to use in computing the step update.
    f: The 3D variable to be filtered. Values in halos must be valid.
    stencil: The width of the stencil to be considered for filtering. Only 7 and
      27 are allowed.

  Returns:
    The filtered variable f, with values in the outermost layer being
    unfiltered.

  Raises:
    ValueError: If stencil is not 7 or 27.
  """
  filter_ops = (
      functools.partial(kernel_op.apply_kernel_op_x, name='kddx'),
      functools.partial(kernel_op.apply_kernel_op_y, name='kddy'),
      functools.partial(
          kernel_op.apply_kernel_op_z, name='kddz', shift='kddzsh'),
  )

  g = tf.nest.map_structure(tf.identity, f)
  if stencil == 7:
    for op in filter_ops:
      g = tf.nest.map_structure(
          lambda g_i, d2_f_i: g_i + 1.0 / 12.0 * d2_f_i, g, op(f))
  elif stencil == 27:
    for op in filter_ops:
      g = tf.nest.map_structure(
          lambda g_i, d2_g_i: g_i + 0.25 * d2_g_i, g, op(g))
  else:
    raise ValueError('Stencil width {} is not supported. Allowed stencil '
                     'widths are 7 and 27.'.format(stencil))

  # Handle the 3D tensor case.
  if isinstance(f, tf.Tensor):
    nz, nx, ny = f.shape  # pytype: disable=attribute-error
    mask = tf.pad(
        tf.constant(True, shape=(nz - 2, nx - 2, ny - 2)),
        paddings=[[1, 1], [1, 1], [1, 1]],
        mode='CONSTANT',
        constant_values=False)
    return tf.where(mask, g, f)

  # Handle the list-of-2D-tensors case.
  nx, ny = f[0].shape
  mask = tf.pad(
      tf.constant(True, shape=(nx - 2, ny - 2)),
      paddings=[[1, 1], [1, 1]],
      mode='CONSTANT',
      constant_values=False)

  g = tf.nest.map_structure(
      lambda g_i, f_i: tf.compat.v1.where(mask, g_i, f_i), g, f
  )
  g[0] = f[0]
  g[-1] = f[-1]

  return g


def global_box_filter_3d(state: FlowFieldVal,
                         halo_update_fn: Callable[[FlowFieldVal], FlowFieldVal],
                         filter_width: int, num_iter: int) -> FlowFieldVal:
  """Applies a balanced 3D Tophat filter to a 3D tensor.

  The following operation is performed by this function:
    u'_{lmn} =
      ∑_{p=l-N/2}^{l+N/2} ∑_{s=m-N/2}^{m+N/2} ∑_{t=n-N/2}^{n+N/2} 1/N³ uₚₛₜ
  Note that the filter is balanced, so only odd number is allowed as
  `filter_wdith`.

  Args:
    state: The 3D tensor field to be filtered.
    halo_update_fn: A function that is used to update the halos of `f`.
    filter_width: The full width of stencil of the filter in each direction.
    num_iter: The number of iterations that the filter is applied.

  Returns:
    The filtered `state`.

  Raises:
    ValueError if `filter_width` is even.
  """
  if filter_width % 2 == 0:
    raise ValueError(
        'Filter width has to be an odd number. {} is not allowed.'.format(
            filter_width))

  filter_coeffs = [1.0 / filter_width,] * filter_width
  offset = filter_width // 2
  kernel_dict = {'filter': (filter_coeffs, offset)}
  kernel_op = get_kernel_fn.ApplyKernelConvOp(8, kernel_dict)

  stop_condition = lambda i, f_filtered: i < num_iter

  def body(i, f_filtered):
    filtered_x = kernel_op.apply_kernel_op_x(f_filtered, 'filterx')
    filtered_y = kernel_op.apply_kernel_op_y(filtered_x, 'filtery')
    filtered_z = kernel_op.apply_kernel_op_z(filtered_y, 'filterz', 'filterzsh')
    return (i + 1, halo_update_fn(filtered_z))

  i0 = tf.constant(0)
  _, f_filtered = tf.while_loop(
      cond=stop_condition, body=body, loop_vars=(i0, state), back_prop=False)

  return f_filtered
