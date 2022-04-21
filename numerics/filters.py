"""A library for filter operators."""

import functools
from typing import List, Optional, Sequence
from swirl_lm.utility import get_kernel_fn
import tensorflow as tf


def filter_op(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    f: Sequence[tf.Tensor],
    order: Optional[int] = 2,
) -> List[tf.Tensor]:
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
    f: Sequence[tf.Tensor],
    stencil: Optional[int] = 27,
) -> List[tf.Tensor]:
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

  g = [tf.identity(f_i) for f_i in f]
  if stencil == 7:
    for op in filter_ops:
      g = [g_i + 1.0 / 12.0 * d2_f_i for g_i, d2_f_i in zip(g, op(f))]
  elif stencil == 27:
    for op in filter_ops:
      g = [g_i + 0.25 * d2_g_i for g_i, d2_g_i in zip(g, op(g))]
  else:
    raise ValueError('Stencil width {} is not supported. Allowed stencil '
                     'widths are 7 and 27.'.format(stencil))

  nx, ny = f[0].shape
  mask = tf.pad(
      tf.constant(True, shape=(nx - 2, ny - 2)),
      paddings=[[1, 1], [1, 1]],
      mode='CONSTANT',
      constant_values=False)

  g = [tf.compat.v1.where(mask, g_i, f_i) for g_i, f_i in zip(g, f)]
  g[0] = f[0]
  g[-1] = f[-1]

  return g
