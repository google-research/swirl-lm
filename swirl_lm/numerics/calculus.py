# Copyright 2023 The swirl_lm Authors.
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

"""Vector calculus operations."""

from typing import Sequence, Union

from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal


def _grad_impl(kernel_op: get_kernel_fn.ApplyKernelOp, state: FlowFieldVal,
               dim: int, grid_spacing: float) -> FlowFieldVal:
  """Computes gradient of `value` in `dim` with 2nd order central scheme."""
  if dim == 0:
    d_state = kernel_op.apply_kernel_op_x(state, 'kDx')
  elif dim == 1:
    d_state = kernel_op.apply_kernel_op_y(state, 'kDy')
  else:  # dim == 2
    d_state = kernel_op.apply_kernel_op_z(state, 'kDz', 'kDzsh')

  return tf.nest.map_structure(
      lambda d_state_i: d_state_i / (2.0 * grid_spacing), d_state)


def grad(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    field_vars: Union[FlowFieldVal, Sequence[FlowFieldVal]],
    delta: Sequence[float],
) -> Sequence[Sequence[FlowFieldVal]]:
  """Computes the gradient for all variables in `field_vars`.

  Args:
    kernel_op: Kernel operators that perform finite difference operations.
    field_vars: A list of 3D tensor variables, or a single 3D tensor variable.
    delta: The filter widths/grid spacing in three dimensions, which is a
      sequence of length 3.

  Returns:
    The gradients of all variables in `field_vars`. The first index of the
    returns structure indicates the variable sequence, and the second index
    is the direction of the gradient, which can only be 0, 1, and 2.

  Raises:
    TypeError if `field_vars` is not convertible to a rank 3 or 4 tensor.
  """
  n_dim = len(tf.convert_to_tensor(field_vars).shape)
  if n_dim == 3:
    return [_grad_impl(kernel_op, field_vars, j, delta[j]) for j in range(3)]
  elif n_dim == 4:
    return [[_grad_impl(kernel_op, field_var, j, delta[j])
             for j in range(3)]
            for field_var in field_vars]

  raise TypeError('Unknown type for `field_vars`.')


def divergence(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    field_var: Sequence[FlowFieldVal],
    delta: Sequence[float],
) -> FlowFieldVal:
  """Computes the divergence of `field_var`.

  Args:
    kernel_op: Kernel operators that perform finite difference operations.
    field_var: A sequence of 3D tensor variables. The length of the sequence has
      to be 3.
    delta: The grid spacing in three dimensions, which is a sequence of length
      3.

  Returns:
    The divergence of `field_var`.

  Raises:
    ValueError: If the length of either `field_var` or `delta` is not 3.
  """
  if len(field_var) != 3:
    raise ValueError(
        'The vector has to have exactly 3 components to '
        'compute the divergence. {} is given.'.format(len(field_var))
    )

  if len(delta) != 3:
    raise ValueError(
        'The length mesh size vector has to be 3. {} is given.'.format(
            len(delta)
        )
    )

  gradients = [
      _grad_impl(kernel_op, field_var[i], i, delta[i]) for i in range(3)
  ]
  return tf.nest.map_structure(
      lambda ddx, ddy, ddz: ddx + ddy + ddz, *gradients
  )


def laplacian(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    f: FlowFieldVal,
    nu: float,
    dx: float,
    dy: float,
    dz: float,
) -> FlowFieldVal:
  """Computes ν times the Laplacian of `f`.

  Args:
    kernel_op: An object holding a library of kernel operations.
    f: A list of `tf.Tensor` representing a 3D volume (as a list of 2D x-y
      slices) to which the operator is applied.
    nu: Scalar ν to multiply Laplacian by.
    dx: Mesh size in the x direction.
    dy: Mesh size in the y direction.
    dz: Mesh size in the z direction.

  Returns:
    The scaled Laplacian ν Δf.
  """
  ddx = [g / dx**2 for g in kernel_op.apply_kernel_op_x(f, 'kddx')]
  ddy = [g / dy**2 for g in kernel_op.apply_kernel_op_y(f, 'kddy')]
  ddz = [g / dz**2 for g in kernel_op.apply_kernel_op_z(f, 'kddz', 'kddzsh')]
  return [nu * (ddx_ + ddy_ + ddz_) for ddx_, ddy_, ddz_ in zip(ddx, ddy, ddz)]
