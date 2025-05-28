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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Library for getting and applying a kernel operator.

NB: Care must be taken when using difference kernels and applying division to
interpret their results. In particular, the centered first-order differences,
owing to the space between grid points being twice as large as the forward and
backward first-order differences, should be used at callsites with a division
factor of two.

NB: Higher-order finite difference stencils provide a better approximation of
the derivatives, which allows us to use coarser grids in simulations.
However, when using higher-order stencils one should keep in mind the trade-off
between TPU utilization rate and halo exchange.


kSx: Centered one-step lookahead/lookback sum in x.
  s_x{i, j} = u_{i-1, j} + u_{i+1, j}
kSy: Centered one-step lookahead/lookback sum in y.
  s_y{i, j} = u_{i, j-1} + u_{i, j+1}
ksx: Backward one-step sum in x.
  s_x{i, j} = u_{i-1, j} + u_{i, j}
ksy: Backward one-step sum in y.
  s_y{i, j} = u_{i, j-1} + u_{i, j}
kDx: Centered first-order finite difference in x.
  u_x_{i,j} = u_{i+1, j} - u_{i-1, j}
kDy: Centered first-order finite difference in y.
  u_y_{i,j} = u_{i, j+1} - u_{i, j-1}
kD4x: Centered first-order finite difference in x (4th-order approximation).
  u_x_{i,j} = -u_{i+2, j} + 8 u_{i+1, j} - 8 u_{i-1, j} + u_{i-2, j}
kD4y: Centered first-order finite difference in y (4th-order approximation).
  u_y_{i,j} = -u_{i, j+2} + 8 u_{i, j+1} - 8 u_{i, j-1} + u_{i, j-2}
kdx: Backward first-order finite difference in x.
  u_x_{i, j} = u_{i, j} - u_{i-1, j}
kdy: Backward first-order finite difference in y.
  u_y_{i, j} = u_{i, j} - u_{i, j-1}
kdx+: Forward first-order finite difference in x.
  u_x_{i, j} = u_{i+1, j} - u_{i, j}
kdy+: Forward first-order finite difference in y.
  u_y_{i, j} = u_{i, j+1} - u_{i, j}
kddx: Centered second-order finite difference in x.
  u_xx_{i, j} = u_{i+1, j} - 2 * u_{i, j} + u_{i-1, j}
kddy: Centered second-order finite difference in y.
  u_yy_{i, j} = u_{i, j+1} - 2 * u_{i, j} + u_{i, j-1}
kdd8x: Centered second-order finite difference in x (8th-order approximation).
  u_xx_{i, j} = (- 9 * (u_{i+4, j} + u_{i-4, j})
                 + 128 * (u_{i+3, j} + u_{i-3, j})
                 - 1008 * (u_{i+2, j} + u_{i-2, j})
                 + 8064 * (u_{i+1, j} + u_{i-1, j})
                 - 14350 * u_{i, j}) / 5040
kdd8y: Centered second-order finite difference in y (8th-order approximation).
  u_yy_{i, j} = (- 9 * (u_{i, j+4} + u_{i, j-4})
                 + 128 * (u_{i, j+3} + u_{i, j-3})
                 - 1008 * (u_{i, j+2} + u_{i, j-2})
                 + 8064 * (u_{i, j+1} + u_{i, j+1})
                 - 14350 * u_{i, j}) / 5040
kdd16x: Centered second-order finite difference in x (16th-order approximation).
  u_xx_{i, j} = (- 735. * (u_{i+8, j} + u_{i-8, j})
                 + 15360 * (u_{i+7, j} + u_{i-7, j})
                 - 156800 * (u_{i+6, j} + u_{i-6, j})
                 + 1053696 * (u_{i+5, j} + u_{i-5, j})
                 - 5350800 * (u_{i+4, j} + u_{i-4, j})
                 + 22830080 * (u_{i+3, j} + u_{i-3, j})
                 - 94174080 * (u_{i+2, j} + u_{i-2, j})
                 + 538137600 * (u_{i+1, j} + u_{i-1, j})
                 - 9247086420 * u_{i, j}) / 302702400
kdd16y: Centered second-order finite difference in y (16th-order approximation).
  u_yy_{i, j} = (- 735. * (u_{i, j+8} + u_{i, j-8})
                 + 15360 * (u_{i, j+7} + u_{i, j-7})
                 - 156800 * (u_{i, j+6} + u_{i, j-6})
                 + 1053696 * (u_{i, j+5} + u_{i, j-5})
                 - 5350800 * (u_{i, j+4} + u_{i, j-4})
                 + 22830080 * (u_{i, j+3} + u_{i, j-3})
                 - 94174080 * (u_{i, j+2} + u_{i, j-2})
                 + 538137600 * (u_{i, j+1} + u_{i, j-1})
                 - 9247086420 * u_{i, j}) / 302702400
kf2x: Second-order flux reconstruction on the face of the mesh cell in x in the
  upwind condition, i.e. u_{i, j} > 0,
  f_{i, j} = -0.125 * u_{i-1, j} + 0.75 * u_{i, j} + 0.375 * u_{i+1, j}
  NB: The left face of node `i` is stored at the `i - 1` index in the tensor,
  and the right face of node `i` is stored at the `i` index. So in this context,
  we would be interested in:
   f_{i-1, j} =  -0.125 * u_{i-2, j} + 0.75 * u_{i-1, j} + 0.375 * u_{i, j},
  which is equivalent to the above expression.
kf2y: Second-order flux reconstruction on the face of the mesh cell in y in the
  upwind condition, i.e. u_{i, j} > 0,
  f_{i, j} = -0.125 * u_{i, j-1} + 0.75 * u_{i, j} + 0.375 * u_{i, j+1}
  NB: The left face of node `j` is stored at the `j - 1` index in the tensor,
  and the right face of node `j` is stored at the `j` index. So in this context,
  we would be interested in:
   f_{i, j-1} =  -0.125 * u_{i, j-2} + 0.75 * u_{i, j-1} + 0.375 * u_{i, j},
  which is equivalent to the above expression.
kf2x+: Second-order flux reconstruction on the face of the mesh cell in x in the
  downwind condition, i.e. u_{i, j} < 0,
  f_{i, j} = 0.375 * u_{i-1, j} + 0.75 * u_{i, j} - 0.125 * u_{i+1, j}
  NB: The left face of node `i` is stored at the `i` index in the tensor,
  and the right face of node `i` is stored at the `i + 1` index.
kf2y+: Second-order flux reconstruction on the face of the mesh cell in y in the
  downwind condition, i.e. u_{i, j} < 0,
  f_{i, j} = 0.375 * u_{i, j-1} + 0.75 * u_{i, j} - 0.125 * u_{i+1, j}
  NB: The left face of node `j` is stored at the `j` index in the tensor,
  and the right face of node `j` is stored at the `j + 1` index.
k3d1x+: Forward first order finite difference for the third order derivative.
  These coefficients are applied to a given field in the x direction.
  f_{i, j} = -u_{i-2, j} + 3 u_{i-1, j} - 3 u_{i, j} + u_{i+1, j}
k3d1y+: Forward first order finite difference for the third order derivative.
  These coefficients are applied to a given field in the y direction.
  f_{i, j} = -u_{i, j-2} + 3 u_{i, j-1} - 3 u_{i, j} + u_{i, j+1}
k4d2x: Central second order finite difference for the fourth order derivative.
  These coefficients are applied to a given field in the x direction.
  f_{i, j} = u_{i-2, j} - 4 u_{i-1, j} + 6 u_{i, j} - 4 u_{i+1, j} + u_{i+2, j}
k4d2y: Central second order finite difference for the fourth order derivative.
  These coefficients are applied to a given field in the y direction.
  f_{i, j} = u_{i, j-2} - 4 u_{i, j-1} + 6 u_{i, j} - 4 u_{i, j+1} + u_{i, j+2}
"""

import abc
from typing import Callable, Dict, Mapping, Optional, Sequence, Text, Tuple, Union

import numpy as np
import six
from swirl_lm.utility import common_ops
from swirl_lm.utility import types
import tensorflow as tf

# The general type of the matrix multiplication, convolution, and slice kernels.
KernelType = Union[np.ndarray, Sequence[float], Callable[[tf.Tensor],
                                                         tf.Tensor]]
# The general type of the kernel dictionary.
KernelDictType = Dict[Text, KernelType]
# The type of the input for customizing a kernel.
ExternalDictKernelType = Mapping[Text, Tuple[Sequence[float], int]]
# The operand type of a kernel.
FlowFieldVal = types.FlowFieldVal

_NP_DTYPE = types.NP_DTYPE
_TF_DTYPE = types.TF_DTYPE

COEFFS = {
    # Centered sum
    'centered_sum': [1., 0., 1.],
    # Backward sum
    'backward_sum': [1., 1.],
    # Centered finite difference of the first derivative
    'centered_difference_1': [-1., 0., 1.],
    # Centered finite difference of the first derivative with fourth order
    # approximation.
    'centered_difference_1_order_4': [1., -8., 0., 8., -1.],
    # Backward finite difference of the first derivative
    'backward_difference_1': [-1., 1.],
    # Forward finite difference of the first derivative
    'forward_difference_1': [-1., 1.],
    # Second-order approximation of the centered finite difference of
    # the second derivative
    'centered_difference_2_order_2': [1., -2., 1.],
    # 8th-order approximation of the centered finite difference of
    # the second derivative
    'centered_difference_2_order_8': [
        -9. / 5040., 128. / 5040., -1008. / 5040., 8064. / 5040.,
        -14350. / 5040., 8064. / 5040., -1008. / 5040., 128. / 5040.,
        -9. / 5040.
    ],
    # 16th-order approximation of the centered finite difference of
    # the second derivative
    'centered_difference_2_order_16': [
        -735. / 302702400., 15360. / 302702400., -156800. / 302702400.,
        1053696. / 302702400., -5350800. / 302702400., 22830080. / 302702400.,
        -94174080. / 302702400., 538137600. / 302702400.,
        -924708642. / 302702400., 538137600. / 302702400.,
        -94174080. / 302702400., 22830080. / 302702400., -5350800. / 302702400.,
        1053696. / 302702400., -156800. / 302702400., 15360. / 302702400.,
        -735. / 302702400.
    ],
    # 2nd-order approximation of the flux on the face of a cell following a
    # QUICK scheme
    'face_flux_quick': [-0.125, 0.75, 0.375],
    # First order forward difference of the third order derivative.
    'forward_difference_3_order_1': [-1., 3., -3., 1.],
    # Second order central difference of the fourth order derivative.
    'centered_difference_4_order_2': [1., -4., 6., -4., 1.],
}


def _validate_offset_and_stencil(offset, stencil):
  """Checks that offset is a valid index of the stencil.

  Args:
    offset: Index of an element of the stencil.
    stencil: List of stencil coefficients.

  Returns:
    A valid index of an element of the stencil.
  Raises:
    ValueError if offset is invalid.
  """
  if offset is None:
    offset = len(stencil) // 2

  if offset < 0 or offset >= len(stencil):
    raise ValueError('Offset must be positive and strictly less than '
                     'the length of the stencil, not %d.' % (offset,))
  return offset


def _make_banded_matrix(stencil, banded_matrix_size, offset=None):
  """Creates a banded matrix.

  The band diagonal elements are populated with the `stencil`.

  Args:
    stencil: List of coefficients in the diagonal band.
    banded_matrix_size: The integer size of the banded matrix.
    offset: Index of the element of the stencil at which to start, such that the
      top-first element of the banded matrix is `stencil[offset]`. Defaults to
      the middle element of the stencil.

  Returns:
    A banded matrix numpy array with shape
    (banded_matrix_size, banded_matrix_size) and type _NP_DTYPE.
  """
  offset = _validate_offset_and_stencil(offset, stencil)

  stencil = np.asarray(stencil, dtype=_NP_DTYPE)
  padding = np.zeros(banded_matrix_size - 1, dtype=_NP_DTYPE)
  padded_stencil = np.concatenate((padding, stencil, padding))
  strides = padded_stencil.strides[0]
  strided = np.lib.stride_tricks.as_strided
  return strided(
      padded_stencil[banded_matrix_size - 1 + offset:],
      shape=(banded_matrix_size, banded_matrix_size),
      strides=(-strides, strides))


def _make_backward_banded_matrix(
    stencil: Sequence[float],
    n: int,
    axis: Text,
    offset: Optional[int],
) -> np.ndarray:
  """Generates a banded matrix kernel with `stencil` as weights.

  The stencil will be biased backward if the length of stencil is even.

  Args:
    stencil: The weights of the banded matrix along a row.
    n: The size of the matrix.
    axis: The physical axis of that the matrix kernel will be applied to. Should
      be either 'x' or 'y'.
    offset: The index of the center of the stencil.

  Returns:
    A matrix A banded matrix numpy array with shape (n, n).
  """
  offset = _validate_offset_and_stencil(offset, stencil)
  if axis == 'y':
    stencil = stencil[::-1]
    if len(stencil) % 2 == 0:
      offset -= 1

  return _make_banded_matrix(stencil, n, offset)


def _make_convop_kernel(stencil, kernel_size, offset=None):
  """Creates a convolutional finite-difference operator.

  Args:
    stencil: List of coefficients in the convolution stencil.
    kernel_size: The integer size of the kernel (only square kernels are
      supported).
    offset: Index of the element of the stencil at which the operator is
      centered. Typically, for centered difference, this would be the middle
      element. For backward difference, this would be the last element of the
      stencil. For forward difference, this would be the first element of the
      stencil. Defaults to the middle element of the stencil.

  Returns:
    A convolutional finite difference operator.
  """
  offset = _validate_offset_and_stencil(offset, stencil)

  left_width = offset
  right_width = len(stencil) - offset - 1
  reversed_stencil = stencil[::-1]

  if left_width > 0:
    upper_triangle = np.concatenate([
        np.zeros([kernel_size - left_width, kernel_size], dtype=_NP_DTYPE),
        np.concatenate([
            _make_banded_matrix(reversed_stencil, left_width,
                                len(stencil) - 1),
            np.zeros([left_width, kernel_size - left_width], dtype=_NP_DTYPE)
        ],
                       axis=1)
    ])
  else:
    upper_triangle = np.zeros([kernel_size, kernel_size], dtype=_NP_DTYPE)

  if right_width > 0:
    lower_triangle = np.concatenate([
        np.concatenate([
            np.zeros([right_width, kernel_size - right_width], dtype=_NP_DTYPE),
            _make_banded_matrix(reversed_stencil, right_width, 0)
        ],
                       axis=1),
        np.zeros([kernel_size - right_width, kernel_size], dtype=_NP_DTYPE)
    ])
  else:
    lower_triangle = np.zeros([kernel_size, kernel_size], dtype=_NP_DTYPE)

  return np.stack([
      upper_triangle,
      _make_banded_matrix(reversed_stencil, kernel_size, right_width),
      lower_triangle
  ])


def _make_backward_convop_kernel(
    stencil: Sequence[float],
    n: int,
    axis: Text,
    offset: Optional[int],
) -> np.ndarray:
  """Generates a convolution kernel with `stencil` as weights.

  The stencil will be biased backward if the length of stencil is even. This
  function is used for customizing a convolution kernel function.

  Args:
    stencil: The weights of the banded matrix along a row.
    n: The size of the matrix.
    axis: The physical axis of that the matrix kernel will be applied to. Should
      be either 'x' or 'y'.
    offset: The index of the center of the stencil.

  Returns:
    A convolutional finite difference operator.
  """
  del axis

  return _make_convop_kernel(stencil, n, offset)


def _make_slice_kernel(u, stencil, axis, offset=None):
  """Creates a slice-based finite difference operator.

  Args:
    u: tf.Tensor on which to apply the operator.
    stencil: List of coefficients in the operator stencil.
    axis: The axis along which to apply the operator, must be either 'x' or 'y'.
    offset: Index of the element of the stencil at which the operator is
      centered. Typically, for centered difference, this would be the middle
      element. For backward difference, this would be the last element of the
      stencil. For forward difference, this would be the first element of the
      stencil. Defaults to the middle element of the stencil.

  Returns:
    A slice-based finite difference operator.
  Raises:
    ValueError if axis is not either 'x' or 'y'.
  """
  offset = _validate_offset_and_stencil(offset, stencil)

  kernel = stencil[offset] * u
  if axis == 'x':
    for i in range(offset):
      kernel += stencil[i] * tf.pad(
          u[:i - offset, :], paddings=[[offset - i, 0], [0, 0]])
    for i in range(offset + 1, len(stencil)):
      kernel += stencil[i] * tf.pad(
          u[i - offset:, :], paddings=[[0, i - offset], [0, 0]])
  elif axis == 'y':
    for i in range(offset):
      kernel += stencil[i] * tf.pad(
          u[:, :i - offset], paddings=[[0, 0], [offset - i, 0]])
    for i in range(offset + 1, len(stencil)):
      kernel += stencil[i] * tf.pad(
          u[:, i - offset:], paddings=[[0, 0], [0, i - offset]])
  else:
    raise ValueError("axis must be either 'x' or 'y', not %d." % (axis,))
  return kernel


def _make_backward_slice_kernel(
    stencil: Sequence[float],
    n: int,
    axis: Text,
    offset: Optional[int],
) -> Callable[[tf.Tensor], tf.Tensor]:
  """Generates a slice kernel with `stencil` as weights.

  The stencil will be biased backward if the length of stencil is even. This
  function is used for customizing a slice kernel function.

  Args:
    stencil: The weights of the banded matrix along a row.
    n: The size of the matrix.
    axis: The physical axis of that the matrix kernel will be applied to. Should
      be either 'x' or 'y'.
    offset: The index of the center of the stencil.

  Returns:
    A slice-based finite difference operator.
  """
  del n, offset

  def kernel_fn(u: tf.Tensor) -> tf.Tensor:
    """The kernel function that performs slicing operation."""
    return _make_slice_kernel(u, stencil, axis)

  return kernel_fn


# NB: Shared by all kernel ops.
def _z_kernel_dict(
    custom_kernel_dict: Optional[Mapping[Text, Sequence[Union[int,
                                                              float]]]] = None):
  """Defines the weights and offsets of kernels in the z dimension.

  Args:
    custom_kernel_dict: A dictionary that stores the weights of kernels and
      their offsets. The keys of the dictionary are the names of the kernel, the
      first argument in the tuple value is the weights of the kernel, and the
      second argument in the tuple value is the offset of the kernel.

  Returns:
    A dictionary of kernels that is used to perform the kernel operations. The
    returned dict contains both standard entries and customized ones.

  Raises:
    ValueError if any of the keys in `custom_kernel_dict` already exists in the
    standard kernel dict.
  """
  kernel_dict = {
      'kSz': [1., 1.],
      'kSzsh': [-1, 1],
      'ksz': [1., 1.],
      'kszsh': [-1, 0],
      'kDz': [-1., 1.],
      'kDzsh': [-1, 1],
      'kD4z': COEFFS['centered_difference_1_order_4'],
      'kD4zsh': [-2, -1, 0, 1, 2],
      'kdz': COEFFS['backward_difference_1'],
      'kdzsh': [-1, 0],
      'kdz+': COEFFS['forward_difference_1'],
      'kdz+sh': [0, 1],
      'kddz': COEFFS['centered_difference_2_order_2'],
      'kddzsh': [-1, 0, 1],
      'kdd8z': COEFFS['centered_difference_2_order_8'],
      'kdd8zsh': [-4, -3, -2, -1, 0, 1, 2, 3, 4],
      'kdd16z': COEFFS['centered_difference_2_order_16'],
      'kdd16zsh': [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
      'kf2z': COEFFS['face_flux_quick'],
      'kf2zsh': [-1, 0, 1],
      'kf2z+': COEFFS['face_flux_quick'][::-1],
      'kf2z+sh': [-1, 0, 1],
      'k3d1z+': COEFFS['forward_difference_3_order_1'],
      'k3d1z+sh': [-2, -1, 0, 1],
      'k4d2z': COEFFS['centered_difference_4_order_2'],
      'k4d2zsh': [-2, -1, 0, 1, 2],
  }
  if custom_kernel_dict is not None:
    _add_customized_kernel(kernel_dict, custom_kernel_dict, 'z')

  return kernel_dict


def _add_customized_kernel(
    kernel_dict: KernelDictType,
    custom_kernel_lib: Optional[ExternalDictKernelType],
    axis: Text,
    kernel_generation_fn: Optional[Callable[[Sequence[float], int, Text, int],
                                            KernelType]] = None,
    n: Optional[int] = None,
) -> KernelDictType:
  """Adds a customized kernel to a kernel dictionary."""
  if custom_kernel_lib is None:
    return kernel_dict

  if axis == 'z':
    for kernel_name_general, stencil in custom_kernel_lib.items():
      kernel_name = kernel_name_general + 'z'
      shift_name = kernel_name_general + 'zsh'

      shift = list(range(-stencil[1], len(stencil[0]) - stencil[1], 1))

      if kernel_name in kernel_dict or shift_name in kernel_dict:
        assert set((kernel_name, shift_name)).issubset(
            kernel_dict
        ), f'{kernel_name} and {shift_name} are not provided at the same time.'
        assert kernel_dict[kernel_name] == stencil[0], (
            f'Kernel {kernel_name} already defined with values'
            f' {kernel_dict[kernel_name]}. Redefining it with {stencil[0]} is'
            ' not allowed.'
        )
        assert kernel_dict[shift_name] == shift, (
            f'Kernel {shift_name} already defined with values'
            f' {kernel_dict[shift_name]}. Redefining it with {shift} is not'
            ' allowed.'
        )

        return kernel_dict

      kernel_dict.update({
          kernel_name: stencil[0],
          shift_name: shift,
      })
  else:
    for kernel_name, stencil in custom_kernel_lib.items():
      kernel_name_directional = kernel_name + axis

      if kernel_name_directional in kernel_dict:
        assert kernel_dict[kernel_name_directional] == stencil[0], (
            f'Kernel {kernel_name_directional} already defined with values'
            f' {kernel_dict[kernel_name_directional]}. Redefining it with'
            f' {stencil[0]} is not allowed.'
        )

        return kernel_dict

      assert kernel_generation_fn is not None, (
          '`kernel_generation_fn` is required to define operators in the x and'
          ' y direction, but None is provided.'
      )

      kernel_dict.update({
          kernel_name_directional:
              kernel_generation_fn(stencil[0], n, axis, stencil[1])
      })
  return kernel_dict


@six.add_metaclass(abc.ABCMeta)
class ApplyKernelOp(object):
  """An interface to be used in applying kernel operations."""

  def __init__(
      self,
      custom_kernel_dict: Optional[ExternalDictKernelType] = None) -> None:
    """Initializes the kernel dictionary in the z dimension.

    Args:
      custom_kernel_dict: A dictionary that stores the weights of kernels and
        their offsets. The keys of the dictionary are the names of the kernel,
        the first argument in the tuple value is the weights of the kernel, and
        the second argument in the tuple value is the offset of the kernel.
    """
    self._z_kernel_dict = _z_kernel_dict(custom_kernel_dict)

  @abc.abstractmethod
  def apply_kernel_op_x(self, tiles: FlowFieldVal,
                        name: Text) -> FlowFieldVal:
    """Applies a kernel op in x on a given FlowFieldVal input."""
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def apply_kernel_op_y(self, tiles: FlowFieldVal,
                        name: Text) -> FlowFieldVal:
    """Applies a kernel op in y on a given FlowFieldVal input."""
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def _add_kernel(self, custom_kernel_dict: ExternalDictKernelType):
    """Adds an customized kernel to the kernel library."""
    raise NotImplementedError('Calling an abstract method.')

  def add_kernel(self, custom_kernel_dict: ExternalDictKernelType):
    """Adds an customized kernel to the kernel library.

    Args:
      custom_kernel_dict: A dictionary that stores the weights of kernels and
        their offsets. The keys of the dictionary are the names of the kernel,
        the first argument in the tuple value is the weights of the kernel, and
        the second argument in the tuple value is the offset of the kernel.
    """
    _add_customized_kernel(self._z_kernel_dict, custom_kernel_dict, 'z')
    self._add_kernel(custom_kernel_dict)

  def apply_kernel_op_z(self,
                        tiles: FlowFieldVal,
                        name: Text,
                        shift: Optional[Text] = None) -> FlowFieldVal:
    """Applies a kernel op in z on a given FlowFieldVal input."""
    if (name not in self._z_kernel_dict or
        (shift and shift not in self._z_kernel_dict)):
      raise ValueError('Invalid kernel name requested.')
    return common_ops.apply_op_z(tiles, self._z_kernel_dict[name],
                                 self._z_kernel_dict[shift])  # pytype: disable=wrong-arg-types


def _mulop_kernel_dict(nx,
                       ny,
                       custom_kernel_dict: Optional[
                           ExternalDictKernelType] = None):
  """Defines the kernel for matrix multiplications.

  Args:
    nx: The number of grid points in the x direction.
    ny: The number of grid points in the y direction.
    custom_kernel_dict: A dictionary that stores the weights of kernels and
      their offsets. The keys of the dictionary are the names of the kernel, the
      first argument in the tuple value is the weights of the kernel, and the
      second argument in the tuple value is the offset of the kernel.

  Returns:
    The matrix kernel that is used in the apply kernel operation. The returned
    dict contains both standard entries and customized ones.

  Raises:
    ValueError if any of the keys in `custom_kernel_dict` already exists in the
    standard kernel dict.
  """
  kernel_dict = {
      'kSx':
          _make_banded_matrix(COEFFS['centered_sum'], nx),
      'kSy':
          _make_banded_matrix(COEFFS['centered_sum'], ny),
      'ksx':
          _make_banded_matrix(COEFFS['backward_sum'], nx),
      'ksy':
          _make_banded_matrix(COEFFS['backward_sum'], ny, offset=0),
      'kDx':
          _make_banded_matrix(COEFFS['centered_difference_1'], nx),
      'kDy':
          _make_banded_matrix(COEFFS['centered_difference_1'][::-1], ny),
      'kD4x':
          _make_banded_matrix(COEFFS['centered_difference_1_order_4'], nx),
      'kD4y':
          _make_banded_matrix(COEFFS['centered_difference_1_order_4'][::-1],
                              ny),
      'kdx':
          _make_banded_matrix(COEFFS['backward_difference_1'], nx, offset=1),
      'kdy':
          _make_banded_matrix(
              COEFFS['backward_difference_1'][::-1], ny, offset=0),
      'kdx+':
          _make_banded_matrix(COEFFS['forward_difference_1'], nx, offset=0),
      'kdy+':
          _make_banded_matrix(
              COEFFS['forward_difference_1'][::-1], ny, offset=1),
      'kddx':
          _make_banded_matrix(COEFFS['centered_difference_2_order_2'], nx),
      'kddy':
          _make_banded_matrix(COEFFS['centered_difference_2_order_2'], ny),
      'kdd8x':
          _make_banded_matrix(COEFFS['centered_difference_2_order_8'], nx),
      'kdd8y':
          _make_banded_matrix(COEFFS['centered_difference_2_order_8'], ny),
      'kdd16x':
          _make_banded_matrix(COEFFS['centered_difference_2_order_16'], nx),
      'kdd16y':
          _make_banded_matrix(COEFFS['centered_difference_2_order_16'], ny),
      'kf2x':
          _make_banded_matrix(COEFFS['face_flux_quick'], nx),
      'kf2y':
          _make_banded_matrix(COEFFS['face_flux_quick'][::-1], ny),
      'kf2x+':
          _make_banded_matrix(COEFFS['face_flux_quick'][::-1], nx),
      'kf2y+':
          _make_banded_matrix(COEFFS['face_flux_quick'], ny),
      'k3d1x+':
          _make_banded_matrix(COEFFS['forward_difference_3_order_1'], nx),
      'k3d1y+':
          _make_banded_matrix(
              COEFFS['forward_difference_3_order_1'][::-1], ny, offset=1),
      'k4d2x':
          _make_banded_matrix(COEFFS['centered_difference_4_order_2'], nx),
      'k4d2y':
          _make_banded_matrix(COEFFS['centered_difference_4_order_2'], ny),
  }

  if custom_kernel_dict:
    for axis, n in zip(('x', 'y'), (nx, ny)):
      _add_customized_kernel(
          kernel_dict,
          custom_kernel_dict,
          axis,
          kernel_generation_fn=_make_backward_banded_matrix,
          n=n)

  return kernel_dict


class ApplyKernelMulOp(ApplyKernelOp):
  """Applies a kernel op using matrix multiplication."""

  def __init__(self,
               nx: int,
               ny: int,
               custom_kernel_dict: Optional[ExternalDictKernelType] = None):
    """Initializes the matrix multiplication kernel operators.

    Args:
      nx: The number of grid points in the x direction.
      ny: The number of grid points in the y direction.
      custom_kernel_dict: A dictionary that stores the weights of kernels and
        their offsets. The keys of the dictionary are the names of the kernel,
        the first argument in the tuple value is the weights of the kernel, and
        the second argument in the tuple value is the offset of the kernel.
    """
    super(ApplyKernelMulOp, self).__init__(custom_kernel_dict)
    self._nx = nx
    self._ny = ny
    self._kernels = _mulop_kernel_dict(nx, ny, custom_kernel_dict)

  def _get_kernel(self, name: Text) -> tf.Tensor:
    if name not in self._kernels.keys():
      raise ValueError('Invalid kernel name requested.')
    return tf.constant(self._kernels[name], dtype=_TF_DTYPE)

  def _add_kernel(self, custom_kernel_dict: ExternalDictKernelType):
    """Adds an customized kernel to the kernel library."""
    for axis, n in zip(('x', 'y'), (self._nx, self._ny)):
      _add_customized_kernel(
          self._kernels,
          custom_kernel_dict,
          axis,
          kernel_generation_fn=_make_backward_banded_matrix,
          n=n)

  def apply_kernel_op_x(self, tiles: FlowFieldVal,
                        name: Text) -> FlowFieldVal:
    return common_ops.apply_op_x(tiles, self._get_kernel(name))  # pytype: disable=wrong-arg-types

  def apply_kernel_op_y(self, tiles: FlowFieldVal,
                        name: Text) -> FlowFieldVal:
    return common_ops.apply_op_y(tiles, self._get_kernel(name))  # pytype: disable=wrong-arg-types


def _convop_kernel_dict(
    n, custom_kernel_dict: Optional[ExternalDictKernelType] = None):
  """Generates a dictionary of convolutional finite-difference operators.

  Args:
    n: The integer size of the kernel (only square kernels are supported).
    custom_kernel_dict: A dictionary that stores the weights of kernels and
      their offsets. The keys of the dictionary are the names of the kernel, the
      first argument in the tuple value is the weights of the kernel, and the
      second argument in the tuple value is the offset of the kernel.

  Returns:
    A dictionary of convolutional finite difference operators. The returned dict
    contains both standard entries and customized ones.

  Raises:
    ValueError if any of the keys in `custom_kernel_dict` already exists in the
    standard kernel dict.
  """
  numpy_kernels = {
      'kSx': _make_convop_kernel(COEFFS['centered_sum'], n),
      'kSy': _make_convop_kernel(COEFFS['centered_sum'], n),
      'ksx': _make_convop_kernel(COEFFS['backward_sum'], n, offset=1),
      'ksy': _make_convop_kernel(COEFFS['backward_sum'], n, offset=1),
      'kDx': _make_convop_kernel(COEFFS['centered_difference_1'], n),
      'kDy': _make_convop_kernel(COEFFS['centered_difference_1'], n),
      'kdx': _make_convop_kernel(COEFFS['backward_difference_1'], n, offset=1),
      'kdy': _make_convop_kernel(COEFFS['backward_difference_1'], n, offset=1),
      'kdx+': _make_convop_kernel(COEFFS['forward_difference_1'], n, offset=0),
      'kdy+': _make_convop_kernel(COEFFS['forward_difference_1'], n, offset=0),
      'kddx': _make_convop_kernel(COEFFS['centered_difference_2_order_2'], n),
      'kddy': _make_convop_kernel(COEFFS['centered_difference_2_order_2'], n),
      'kf2x': _make_convop_kernel(COEFFS['face_flux_quick'], n),
      'kf2y': _make_convop_kernel(COEFFS['face_flux_quick'], n),
      'kf2x+': _make_convop_kernel(COEFFS['face_flux_quick'][::-1], n),
      'kf2y+': _make_convop_kernel(COEFFS['face_flux_quick'][::-1], n),
      'k3d1x+': _make_convop_kernel(COEFFS['forward_difference_3_order_1'], n),
      'k3d1y+': _make_convop_kernel(COEFFS['forward_difference_3_order_1'], n),
  }
  if n >= 4:
    numpy_kernels['kD4x'] = _make_convop_kernel(
        COEFFS['centered_difference_1_order_4'], n)
    numpy_kernels['kD4y'] = _make_convop_kernel(
        COEFFS['centered_difference_1_order_4'], n)
    numpy_kernels['kdd8x'] = _make_convop_kernel(
        COEFFS['centered_difference_2_order_8'], n)
    numpy_kernels['kdd8y'] = _make_convop_kernel(
        COEFFS['centered_difference_2_order_8'], n)
    numpy_kernels['k4d2x'] = _make_convop_kernel(
        COEFFS['centered_difference_4_order_2'], n)
    numpy_kernels['k4d2y'] = _make_convop_kernel(
        COEFFS['centered_difference_4_order_2'], n)
  if n >= 8:
    numpy_kernels['kdd16x'] = _make_convop_kernel(
        COEFFS['centered_difference_2_order_16'], n)
    numpy_kernels['kdd16y'] = _make_convop_kernel(
        COEFFS['centered_difference_2_order_16'], n)
  if custom_kernel_dict:
    for axis in ('x', 'y'):
      _add_customized_kernel(
          numpy_kernels,
          custom_kernel_dict,
          axis,
          kernel_generation_fn=_make_backward_convop_kernel,
          n=n)

  return {
      key: tf.constant(value, dtype=_TF_DTYPE)
      for key, value in numpy_kernels.items()
  }


class ApplyKernelConvOp(ApplyKernelOp):
  """Applies a kernel op using tf.conv1d."""

  def _get_kernel(self, name: Text) -> tf.Tensor:
    if name not in self._kernels.keys():
      raise ValueError('Invalid Kernel name requested.')
    return self._kernels[name]

  def __init__(
      self,
      kernel_size: int,
      custom_kernel_dict: Optional[ExternalDictKernelType] = None) -> None:
    """Initializes kernels of convolutional finite-difference operators.

    Args:
      kernel_size: The integer size of the kernel (only square kernels are
        supported).
      custom_kernel_dict: A dictionary that stores the weights of kernels and
        their offsets. The keys of the dictionary are the names of the kernel,
        the first argument in the tuple value is the weights of the kernel, and
        the second argument in the tuple value is the offset of the kernel.
    """
    super(ApplyKernelConvOp, self).__init__(custom_kernel_dict)
    self._kernel_size = kernel_size
    self._kernels = _convop_kernel_dict(kernel_size, custom_kernel_dict)

  def _add_kernel(self, custom_kernel_dict: ExternalDictKernelType):
    """Adds an customized kernel to the kernel library."""
    numpy_kernels = {}
    for axis in ('x', 'y'):
      _add_customized_kernel(
          numpy_kernels,
          custom_kernel_dict,
          axis,
          kernel_generation_fn=_make_backward_convop_kernel,
          n=self._kernel_size)
    self._kernels.update({
        key: tf.constant(value, dtype=_TF_DTYPE)
        for key, value in numpy_kernels.items()
    })

  def apply_kernel_op_x(self, tiles: FlowFieldVal,
                        name: Text) -> FlowFieldVal:
    dtype = (
        tiles.dtype if isinstance(tiles, tf.Tensor) else list(tiles)[0].dtype)
    return common_ops.apply_convolutional_op_x(
        tiles,
        tf.cast(self._get_kernel(name), dtype))

  def apply_kernel_op_y(self, tiles: FlowFieldVal,
                        name: Text) -> FlowFieldVal:
    dtype = (
        tiles.dtype if isinstance(tiles, tf.Tensor) else list(tiles)[0].dtype)
    return common_ops.apply_convolutional_op_y(
        tiles,
        tf.cast(self._get_kernel(name), dtype))


def _slice_kernel_dict(
    custom_kernel_dict: Optional[ExternalDictKernelType] = None
) -> Dict[Text, Callable[[tf.Tensor], tf.Tensor]]:
  """Generates a dictionary of slice-based finite-difference operators.

  Args:
    custom_kernel_dict: A dictionary that stores the weights of kernels and
      their offsets. The keys of the dictionary are the names of the kernel, the
      first argument in the tuple value is the weights of the kernel, and the
      second argument in the tuple value is the offset of the kernel.

  Returns:
    A dictionary of convolutional finite difference operators.

  Raises:
    ValueError if any of the keys in `custom_kernel_dict` already exists in the
    standard kernel dict.
  """

  def ksx(u):
    return tf.pad(
        u[1:, :], paddings=[[0, 1], [0, 0]]) + tf.pad(
            u[:-1, :], paddings=[[1, 0], [0, 0]])

  def ksy(u):
    return tf.pad(
        u[:, 1:], paddings=[[0, 0], [0, 1]]) + tf.pad(
            u[:, :-1], paddings=[[0, 0], [1, 0]])

  def ksx_back(u):
    return u + tf.pad(u[:-1, :], paddings=[[1, 0], [0, 0]])

  def ksy_back(u):
    return u + tf.pad(u[:, :-1], paddings=[[0, 0], [1, 0]])

  def k_dx(u):
    return tf.pad(
        u[1:, :], paddings=[[0, 1], [0, 0]]) - tf.pad(
            u[:-1, :], paddings=[[1, 0], [0, 0]])

  def k_dy(u):
    return tf.pad(
        u[:, 1:], paddings=[[0, 0], [0, 1]]) - tf.pad(
            u[:, :-1], paddings=[[0, 0], [1, 0]])

  def kd4x(u):
    return _make_slice_kernel(
        u, COEFFS['centered_difference_1_order_4'], axis='x')

  def kd4y(u):
    return _make_slice_kernel(
        u, COEFFS['centered_difference_1_order_4'], axis='y')

  def kdx(u):
    return _make_slice_kernel(
        u, COEFFS['backward_difference_1'], axis='x', offset=1)

  def kdy(u):
    return _make_slice_kernel(
        u, COEFFS['backward_difference_1'], axis='y', offset=1)

  def kdx_plus(u):
    return _make_slice_kernel(
        u, COEFFS['forward_difference_1'], axis='x', offset=0)

  def kdy_plus(u):
    return _make_slice_kernel(
        u, COEFFS['forward_difference_1'], axis='y', offset=0)

  def kddx(u):
    return _make_slice_kernel(
        u, COEFFS['centered_difference_2_order_2'], axis='x')

  def kddy(u):
    return _make_slice_kernel(
        u, COEFFS['centered_difference_2_order_2'], axis='y')

  def kdd8x(u):
    return _make_slice_kernel(
        u, COEFFS['centered_difference_2_order_8'], axis='x')

  def kdd8y(u):
    return _make_slice_kernel(
        u, COEFFS['centered_difference_2_order_8'], axis='y')

  def kdd16x(u):
    return _make_slice_kernel(
        u, COEFFS['centered_difference_2_order_16'], axis='x')

  def kdd16y(u):
    return _make_slice_kernel(
        u, COEFFS['centered_difference_2_order_16'], axis='y')

  def kf2x(u):
    return _make_slice_kernel(u, COEFFS['face_flux_quick'], axis='x')

  def kf2y(u):
    return _make_slice_kernel(u, COEFFS['face_flux_quick'], axis='y')

  def kf2x_plus(u):
    return _make_slice_kernel(u, COEFFS['face_flux_quick'][::-1], axis='x')

  def kf2y_plus(u):
    return _make_slice_kernel(u, COEFFS['face_flux_quick'][::-1], axis='y')

  def k3d1x_plus(u):
    return _make_slice_kernel(
        u, COEFFS['forward_difference_3_order_1'], axis='x')

  def k3d1y_plus(u):
    return _make_slice_kernel(
        u, COEFFS['forward_difference_3_order_1'], axis='y')

  def k4d2x(u):
    return _make_slice_kernel(
        u, COEFFS['centered_difference_4_order_2'], axis='x')

  def k4d2y(u):
    return _make_slice_kernel(
        u, COEFFS['centered_difference_4_order_2'], axis='y')

  kernel_dict = {
      'kSx': ksx,
      'kSy': ksy,
      'ksx': ksx_back,
      'ksy': ksy_back,
      'kDx': k_dx,
      'kDy': k_dy,
      'kD4x': kd4x,
      'kD4y': kd4y,
      'kdx': kdx,
      'kdy': kdy,
      'kdx+': kdx_plus,
      'kdy+': kdy_plus,
      'kddx': kddx,
      'kddy': kddy,
      'kdd8x': kdd8x,
      'kdd8y': kdd8y,
      'kdd16x': kdd16x,
      'kdd16y': kdd16y,
      'kf2x': kf2x,
      'kf2y': kf2y,
      'kf2x+': kf2x_plus,
      'kf2y+': kf2y_plus,
      'k3d1x+': k3d1x_plus,
      'k3d1y+': k3d1y_plus,
      'k4d2x': k4d2x,
      'k4d2y': k4d2y,
  }

  if custom_kernel_dict:
    for axis in ('x', 'y'):
      _add_customized_kernel(
          kernel_dict,
          custom_kernel_dict,
          axis,
          kernel_generation_fn=_make_backward_slice_kernel)

  return kernel_dict


class ApplyKernelSliceOp(ApplyKernelOp):
  """Applies a kernel op using naive slicing."""

  def _get_kernel(self, name: Text) -> Callable[[tf.Tensor], tf.Tensor]:
    if name not in self._kernels.keys():
      raise ValueError('Invalid Kernel name requested.')
    return self._kernels[name]

  def __init__(
      self,
      custom_kernel_dict: Optional[ExternalDictKernelType] = None) -> None:
    """Initializes kernels of slice-based finite-difference operators.

    Args:
      custom_kernel_dict: A dictionary that stores the weights of kernels and
        their offsets. The keys of the dictionary are the names of the kernel,
        the first argument in the tuple value is the weights of the kernel, and
        the second argument in the tuple value is the offset of the kernel.
    """
    super(ApplyKernelSliceOp, self).__init__(custom_kernel_dict)
    self._kernels = _slice_kernel_dict(custom_kernel_dict)

  def _add_kernel(self, custom_kernel_dict: ExternalDictKernelType):
    """Adds an customized kernel to the kernel library."""
    for axis in ('x', 'y'):
      _add_customized_kernel(
          self._kernels,
          custom_kernel_dict,
          axis,
          kernel_generation_fn=_make_backward_slice_kernel)

  def apply_kernel_op_x(self, tiles: FlowFieldVal,
                        name: Text) -> FlowFieldVal:
    return common_ops.apply_slice_op_x(tiles, self._get_kernel(name))  # pytype: disable=wrong-arg-types

  def apply_kernel_op_y(self, tiles: FlowFieldVal,
                        name: Text) -> FlowFieldVal:
    return common_ops.apply_slice_op_y(tiles, self._get_kernel(name))  # pytype: disable=wrong-arg-types
