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

"""A library for filter operators."""

import functools
from typing import Callable, Optional
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import stretched_grid_util
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap


def filter_op(
    params: parameters_lib.SwirlLMParameters,
    f: FlowFieldVal,
    additional_states: FlowFieldMap,
    order: Optional[int] = 2,
) -> FlowFieldVal:
  """Performs filtering to a variable.

  Args:
    params: The solver configuration.
    f: The 3D variable to be filtered. Values in halos must be valid.
    additional_states: Mapping that contains the optional scale factors.
    order: The harmonic order of the filter.

  Returns:
    The filtered variable f, with values in the outermost layer being
    unfiltered.

  Raises:
    NotImplementedError: If order is not 2.
  """
  if order == 2:
    return filter_2(params, f, additional_states)
  else:
    raise NotImplementedError(
        'Order {} filter is not supported. Available orders: 2.'.format(order))


def filter_2(
    params: parameters_lib.SwirlLMParameters,
    f: FlowFieldVal,
    additional_states: FlowFieldMap,
    stencil: Optional[int] = 27,
) -> FlowFieldVal:
  r"""Performs filtering by adding second-order derivatives to a variable.

  Note that the `stencil` option applies only to the uniform mesh.
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
    params: The solver configuration.
    f: The 3D variable to be filtered. Values in halos must be valid.
    additional_states: Mapping that contains the optional scale factors.
    stencil: The width of the stencil to be considered for filtering. Only 7 and
      27 are allowed.

  Returns:
    The filtered variable f, with values in the outermost layer being
    unfiltered.

  Raises:
    ValueError: If stencil is not 7 or 27.
  """
  kernel_op = params.kernel_op
  kernel_op.add_kernel(
      {'shift_up': ([1.0, 0.0, 0.0], 1), 'shift_dn': ([0.0, 0.0, 1.0], 1)}
  )
  shift_up_ops = (
      functools.partial(kernel_op.apply_kernel_op_x, name='shift_upx'),
      functools.partial(kernel_op.apply_kernel_op_y, name='shift_upy'),
      functools.partial(
          kernel_op.apply_kernel_op_z, name='shift_upz', shift='shift_upzsh'
      ),
  )
  shift_dn_ops = (
      functools.partial(kernel_op.apply_kernel_op_x, name='shift_dnx'),
      functools.partial(kernel_op.apply_kernel_op_y, name='shift_dny'),
      functools.partial(
          kernel_op.apply_kernel_op_z, name='shift_dnz', shift='shift_dnzsh'
      ),
  )
  filter_ops = (
      functools.partial(kernel_op.apply_kernel_op_x, name='kddx'),
      functools.partial(kernel_op.apply_kernel_op_y, name='kddy'),
      functools.partial(
          kernel_op.apply_kernel_op_z, name='kddz', shift='kddzsh'
      ),
  )

  g = tf.nest.map_structure(tf.identity, f)
  for dim in range(3):
    if params.use_stretched_grid[dim]:
      # Approximate coefficients of the tophat filter with the Simpson's rule.
      # The tophat filter of a variable `f` is represented as [1]:
      # \bar{f} = 1 / (x_{i + 1} - x_{i - 1}) \int_{x_{i - 1}}^{x_{i + 1}} f dx.
      # Integrating it with the Simpson's rule results in [2]:
      # \bar{f} = 1 / 6[(2 - h_{i + 1/2} / h_{i - 1/2}) f_{i - 1} +
      #     (h_{i + 1/2} + h_{i - 1/2})^2 / (h_{i + 1/2} h_{i - 1/2}) +
      #     (2 - h_{i - 1/2} / h_{i + 1/2}) f_{i + 1}].
      # Reference:
      # 1. Lund, T. S. (1997). On the use of discrete filters for large eddy
      # simulation. Annual Research Briefs.
      # 2. Shklov, N. (1960). Simpson’s Rule for Unequally Spaced Ordinates.
      # The American Mathematical Monthly: The Official Journal of the
      # Mathematical Association of America, 67(10), 1022–1023.
      h_0 = additional_states[stretched_grid_util.h_key(dim)]
      h_1 = shift_dn_ops[dim](h_0)
      w_0 = tf.nest.map_structure(
          lambda h_0, h_1: (
              0.5 * (h_0 - h_1) - (h_0 - h_1) ** 2 / (6.0 * h_0) + h_1 / 3.0
          )
          / (h_0 + h_1),
          h_0,
          h_1,
      )
      w_1 = tf.nest.map_structure(
          lambda h_0, h_1: 2.0 / 3.0 + (h_0 - h_1) ** 2 / (6.0 * h_0 * h_1),
          h_0,
          h_1,
      )
      w_2 = tf.nest.map_structure(
          lambda h_0, h_1: (
              -0.5 * (h_0 - h_1) - (h_0 - h_1) ** 2 / (6.0 * h_1) + h_0 / 3.0
          )
          / (h_0 + h_1),
          h_0,
          h_1,
      )
      g = common_ops.map_structure_3d(
          lambda a, b, c, x, y, z: a * x + b * y + c * z,
          w_0,
          w_1,
          w_2,
          shift_up_ops[dim](g),
          g,
          shift_dn_ops[dim](g),
      )
    elif stencil == 7:
      g = tf.nest.map_structure(
          lambda g_i, d2_f_i: g_i + 1.0 / 12.0 * d2_f_i, g, filter_ops[dim](f)
      )
    elif stencil == 27:
      g = tf.nest.map_structure(
          lambda g_i, d2_g_i: g_i + 0.25 * d2_g_i, g, filter_ops[dim](g)
      )
    else:
      raise ValueError(
          'Stencil width {} is not supported. Allowed stencil '
          'widths are 7 and 27.'.format(stencil)
      )

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
