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

"""Vector calculus operations."""

from typing import Sequence, TypeAlias, Union

from swirl_lm.numerics import derivatives
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal: TypeAlias = types.FlowFieldVal
FlowFieldMap: TypeAlias = types.FlowFieldMap


def grad(
    deriv_lib: derivatives.Derivatives,
    field_vars: Union[FlowFieldVal, Sequence[FlowFieldVal]],
    additional_states: FlowFieldMap,
) -> Sequence[Sequence[FlowFieldVal]]:
  """Computes the gradient for all variables in `field_vars`.

  Args:
    deriv_lib: An instance of the derivatives library.
    field_vars: A list of 3D tensor variables, or a single 3D tensor variable.
    additional_states: Mapping that contains helper variables.

  Returns:
    The gradients of all variables in `field_vars`. The first index of the
    returns structure indicates the variable sequence, and the second index
    is the direction of the gradient, which can only be 0, 1, and 2.

  Raises:
    TypeError if `field_vars` is not convertible to a rank 3 or 4 tensor.
  """
  n_dim = len(tf.convert_to_tensor(field_vars).shape)
  if n_dim == 3:
    return [
        deriv_lib.deriv_centered(field_vars, dim, additional_states)
        for dim in (0, 1, 2)
    ]
  elif n_dim == 4:
    return [
        [
            deriv_lib.deriv_centered(field_var, dim, additional_states)
            for dim in (0, 1, 2)
        ]
        for field_var in field_vars
    ]
  raise TypeError('Unknown type for `field_vars`.')


def divergence(
    deriv_lib: derivatives.Derivatives,
    field_var: Sequence[FlowFieldVal],
    additional_states: FlowFieldMap,
) -> FlowFieldVal:
  """Computes the divergence of `field_var`.

  Args:
    deriv_lib: An instance of the derivatives library.
    field_var: A sequence of 3D tensor variables. The length of the sequence has
      to be 3.
    additional_states: Mapping that contains helper variables.

  Returns:
    The divergence of `field_var`.

  Raises:
    ValueError: If the length of `field_var` is not 3.
  """
  if len(field_var) != 3:
    raise ValueError(
        'The vector has to have exactly 3 components to '
        'compute the divergence. {} is given.'.format(len(field_var))
    )

  gradients = [
      deriv_lib.deriv_centered(field_var[dim], dim, additional_states)
      for dim in (0, 1, 2)
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
  ddx = tf.nest.map_structure(
      lambda g: g / dx**2, kernel_op.apply_kernel_op_x(f, 'kddx')
  )
  ddy = tf.nest.map_structure(
      lambda g: g / dy**2, kernel_op.apply_kernel_op_y(f, 'kddy')
  )
  ddz = tf.nest.map_structure(
      lambda g: g / dz**2, kernel_op.apply_kernel_op_z(f, 'kddz', 'kddzsh')
  )
  return tf.nest.map_structure(
      lambda ddx_, ddy_, ddz_: nu * (ddx_ + ddy_ + ddz_), ddx, ddy, ddz
  )
