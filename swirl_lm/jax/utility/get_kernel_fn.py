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
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import six
from swirl_lm.jax.utility import common_ops
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types


# Convolution kernels are arrays.
ConvKernelType = np.ndarray | jax.Array
# Slice kernels are dictionaries: {'coeff': tuple[float], 'shift': tuple[int]}
SliceKernelType = dict[str, tuple[Any, ...]]
# The types of the kernel dictionary.
ConvKernelDictType = dict[str, ConvKernelType]
SliceKernelDictType = dict[str, SliceKernelType]
# The type of the input for customizing a kernel.
ExternalDictKernelType = (
    dict[str, tuple[list[float], int]] | SliceKernelDictType
)
# The operand type of a kernel.
ScalarField = types.ScalarField
DType = jnp.float32

COEFFS = {
    # Centered sum
    'centered_sum': [1.0, 0.0, 1.0],
    # Backward sum
    'backward_sum': [1.0, 1.0],
    # Centered finite difference of the first derivative
    'centered_difference_1': [-1.0, 0.0, 1.0],
    # Centered finite difference of the first derivative with fourth order
    # approximation.
    'centered_difference_1_order_4': [1.0, -8.0, 0.0, 8.0, -1.0],
    # Backward finite difference of the first derivative
    'backward_difference_1': [-1.0, 1.0],
    # Forward finite difference of the first derivative
    'forward_difference_1': [-1.0, 1.0],
    # Second-order approximation of the centered finite difference of
    # the second derivative
    'centered_difference_2_order_2': [1.0, -2.0, 1.0],
    # 2nd-order approximation of the flux on the face of a cell following a
    # QUICK scheme
    'face_flux_quick': [-0.125, 0.75, 0.375],
    # First order forward difference of the third order derivative.
    'forward_difference_3_order_1': [-1.0, 3.0, -3.0, 1.0],
    # Second order central difference of the fourth order derivative.
    'centered_difference_4_order_2': [1.0, -4.0, 6.0, -4.0, 1.0],
}

SHIFTS = {
    # Note: shift=-1 corresponds to index [i+1].
    # Centered sum
    'centered_sum': [-1, 0, 1],
    # Backward sum
    'backward_sum': [1, 0],
    # Centered finite difference of the first derivative
    'centered_difference_1': [1, 0, -1],
    # Centered finite difference of the first derivative with fourth order
    # approximation.
    'centered_difference_1_order_4': [2, 1, 0, -1, -2],
    # Backward finite difference of the first derivative
    'backward_difference_1': [1, 0],
    # Forward finite difference of the first derivative
    'forward_difference_1': [0, -1],
    # Second-order approximation of the centered finite difference of
    # the second derivative
    'centered_difference_2_order_2': [-1, 0, 1],
    # 2nd-order approximation of the flux on the face of a cell following a
    # QUICK scheme
    'face_flux_quick': [1, 0, -1],
    # First order forward difference of the third order derivative.
    'forward_difference_3_order_1': [1, 0, -1, -2],
    # Second order central difference of the fourth order derivative.
    'centered_difference_4_order_2': [-2, -1, 0, 1, 2],
}


def _validate_offset_and_stencil(
    offset: int | None, stencil: list[float]
) -> int:
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
    raise ValueError(
        'Offset must be positive and strictly less than '
        'the length of the stencil, not %d.' % (offset,)
    )
  return offset


def _make_banded_matrix(
    stencil: list[float], banded_matrix_size: int, offset: int | None = None
) -> jax.Array:
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
    (banded_matrix_size, banded_matrix_size).
  """
  offset = _validate_offset_and_stencil(offset, stencil)
  stencil = jnp.array(stencil, dtype=DType)
  rows = jnp.arange(banded_matrix_size, dtype=jnp.int32)
  rows = jnp.tile(jnp.expand_dims(rows, axis=-1), (1, len(stencil)))
  cols = [jnp.arange(ir, ir + len(stencil)) for ir in range(banded_matrix_size)]
  cols = jnp.stack(cols, axis=0) + len(stencil)
  vals = jnp.tile(stencil, (banded_matrix_size, 1))
  matrix = jnp.zeros(
      (banded_matrix_size, banded_matrix_size + 2 * len(stencil))
  )
  matrix = matrix.at[rows, cols].set(vals, mode='drop')
  matrix = jnp.roll(matrix, -offset, axis=1)
  return jax.lax.dynamic_slice_in_dim(
      matrix, len(stencil), banded_matrix_size, axis=1
  )


def _make_backward_banded_matrix(
    stencil: list[float],
    banded_matrix_size: int,
    offset: int | None = None,
) -> jax.Array:
  """Generates a banded matrix kernel with `stencil` as weights.

  The stencil will be biased backward if the length of stencil is even.

  Args:
    stencil: The weights of the banded matrix along a row.
    banded_matrix_size: The size of the matrix.
    offset: The index of the center of the stencil.

  Returns:
    A matrix A banded matrix numpy array with shape (n, n).
  """
  offset = _validate_offset_and_stencil(offset, stencil)
  stencil = stencil[::-1]
  if len(stencil) % 2 == 0:
    offset -= 1
  return _make_banded_matrix(stencil, banded_matrix_size, offset)


def _make_convop_kernel(
    stencil: list[float], kernel_size: int, offset: int | None = None
) -> jax.Array:
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
    upper_triangle = jnp.concatenate([
        jnp.zeros([kernel_size - left_width, kernel_size]),
        jnp.concatenate(
            [
                _make_banded_matrix(
                    reversed_stencil, left_width, len(stencil) - 1
                ),
                jnp.zeros([left_width, kernel_size - left_width]),
            ],
            axis=1,
        ),
    ])
  else:
    upper_triangle = jnp.zeros([kernel_size, kernel_size])

  if right_width > 0:
    lower_triangle = jnp.concatenate([
        jnp.concatenate(
            [
                jnp.zeros([right_width, kernel_size - right_width]),
                _make_banded_matrix(reversed_stencil, right_width, 0),
            ],
            axis=1,
        ),
        jnp.zeros([kernel_size - right_width, kernel_size]),
    ])
  else:
    lower_triangle = jnp.zeros([kernel_size, kernel_size])

  return jnp.stack([
      upper_triangle,
      _make_banded_matrix(reversed_stencil, kernel_size, right_width),
      lower_triangle,
  ])


def _add_customized_conv_kernel(
    kernel_dict: ConvKernelDictType,
    custom_kernel_lib: ExternalDictKernelType | None,
    kernel_size: int,
    kernel_generation_fn: Callable[[list[float], int, int], ConvKernelType],
) -> ConvKernelDictType:
  """Adds a customized convolution kernel to a kernel dictionary."""
  if custom_kernel_lib is None:
    return kernel_dict

  for kernel_name, stencil in custom_kernel_lib.items():
    if kernel_name in kernel_dict:
      assert kernel_dict[kernel_name] == stencil[0], (
          f'Kernel {kernel_name} already defined with values'
          f' {kernel_dict[kernel_name]}. Redefining it with'
          f' {stencil[0]} is not allowed.'
      )

    assert kernel_generation_fn is not None, (
        '`kernel_generation_fn` is required to define operators, but None is'
        ' provided.'
    )

    kernel_dict.update(
        {kernel_name: kernel_generation_fn(stencil[0], kernel_size, stencil[1])}
    )
  return kernel_dict


@six.add_metaclass(abc.ABCMeta)
class ApplyKernelOp(object):
  """An interface to be used in applying kernel operations."""

  def __init__(
      self,
      custom_kernel_dict: ExternalDictKernelType | None = None,
  ) -> None:
    """Initializes the kernel dictionary in the z dimension.

    Args:
      custom_kernel_dict: A dictionary that stores the weights of kernels and
        their offsets. The keys of the dictionary are the names of the kernel,
        the first argument in the tuple value is the weights of the kernel, and
        the second argument in the tuple value is the offset of the kernel.
    """
    pass

  @abc.abstractmethod
  def apply_kernel_op(
      self, array: ScalarField, name: str, axis: str
  ) -> ScalarField:
    """Applies a kernel op along `axis` on a given `ScalarField` input."""
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def add_kernel(self, custom_kernel_dict: ExternalDictKernelType):
    """Adds a customized kernel to the kernel library."""
    raise NotImplementedError('Calling an abstract method.')


def _convop_kernel_dict(
    kernel_size: int,
    custom_kernel_dict: ExternalDictKernelType | None = None,
):
  """Generates a dictionary of convolutional finite-difference operators.

  Args:
    kernel_size: The integer size of the kernel (only square kernels are
      supported).
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
  jnp_kernels = {
      'kS': _make_convop_kernel(COEFFS['centered_sum'], kernel_size),
      'ks': _make_convop_kernel(COEFFS['backward_sum'], kernel_size, offset=1),
      'kD': _make_convop_kernel(COEFFS['centered_difference_1'], kernel_size),
      'kd': _make_convop_kernel(
          COEFFS['backward_difference_1'], kernel_size, offset=1
      ),
      'kd+': _make_convop_kernel(
          COEFFS['forward_difference_1'], kernel_size, offset=0
      ),
      'kdd': _make_convop_kernel(
          COEFFS['centered_difference_2_order_2'], kernel_size
      ),
      'kf2': _make_convop_kernel(COEFFS['face_flux_quick'], kernel_size),
      'kf2+': _make_convop_kernel(COEFFS['face_flux_quick'][::-1], kernel_size),
      'k3d1+': _make_convop_kernel(
          COEFFS['forward_difference_3_order_1'], kernel_size
      ),
  }
  if kernel_size >= 4:
    jnp_kernels['kD4'] = _make_convop_kernel(
        COEFFS['centered_difference_1_order_4'], kernel_size
    )
  if custom_kernel_dict:
    _add_customized_conv_kernel(
        jnp_kernels,
        custom_kernel_dict,
        kernel_size=kernel_size,
        kernel_generation_fn=_make_convop_kernel,
    )

  return jnp_kernels


class ApplyKernelConvOp(ApplyKernelOp):
  """Applies a kernel op using convolutional operators."""

  def _get_kernel(self, name: str) -> ConvKernelType:
    if name not in self._kernels.keys():
      raise ValueError(f'Invalid Kernel name: {name} is requested.')
    return self._kernels[name]

  def __init__(
      self,
      kernel_size: int,
      grid_params: grid_parametrization.GridParametrization,
      custom_kernel_dict: ExternalDictKernelType | None = None,
  ) -> None:
    """Initializes kernels of convolutional finite-difference operators.

    Args:
      kernel_size: The integer size of the kernel (only square kernels are
        supported).
      grid_params: The grid parametrization object.
      custom_kernel_dict: A dictionary that stores the weights of kernels and
        their offsets. The keys of the dictionary are the names of the kernel,
        the first argument in the tuple value is the weights of the kernel, and
        the second argument in the tuple value is the offset of the kernel.
    """
    super(ApplyKernelConvOp, self).__init__(custom_kernel_dict)
    self._kernel_size = kernel_size
    self._kernels = _convop_kernel_dict(kernel_size, custom_kernel_dict)
    self._grid_params = grid_params

  def add_kernel(self, custom_kernel_dict: ExternalDictKernelType):
    """Adds a customized kernel to the kernel library."""
    _add_customized_conv_kernel(
        self._kernels,
        custom_kernel_dict,
        kernel_size=self._kernel_size,
        kernel_generation_fn=_make_convop_kernel,
    )

  def apply_kernel_op(
      self, array: ScalarField, name: str, axis: str
  ) -> ScalarField:
    return common_ops.apply_convolutional_op(
        array,
        jnp.array(self._get_kernel(name), array.dtype),
        axis,
        self._grid_params,
    )


def _discard_zero_coefficients(
    coeff: list[float] | tuple[float, ...],
    shift: list[int] | tuple[int, ...],
    zero_tolerance: float = 1e-6,
) -> SliceKernelType:
  """Discards coefficients that are close to zero within a given tolerance.

  Args:
      coeff: List of coefficients.
      shift: List of corresponding shifts.
      zero_tolerance: Tolerance for considering a coefficient as zero.

  Returns:
      Dictionary with keys 'coeff' and 'shift' containing the modified lists.
  """
  if len(coeff) != len(shift):
    raise ValueError(
        f'coeff: {coeff} and shift: {shift} lists must have the same length.'
    )
  new_coeff = []
  new_shift = []
  for c, s in zip(coeff, shift):
    if abs(c) > zero_tolerance:
      new_coeff.append(c)
      new_shift.append(s)
  return {'coeff': tuple(new_coeff), 'shift': tuple(new_shift)}


def _add_customized_slice_kernel(
    kernel_dict: SliceKernelDictType,
    custom_kernel_lib: SliceKernelDictType | None,
) -> SliceKernelDictType:
  """Adds a customized slice kernel to a kernel dictionary."""
  if custom_kernel_lib is None:
    return kernel_dict

  for kernel_name, stencil in custom_kernel_lib.items():
    assert set(stencil.keys()) == {
        'coeff',
        'shift',
    }, (
        'The keys of `stencil` must be only "coeff" and "shift". Currently,'
        f' the keys are: {stencil.keys()}'
    )
    if kernel_name in kernel_dict:
      assert kernel_dict[kernel_name] == stencil, (
          f'Kernel {kernel_name} already defined with values'
          f' {kernel_dict[kernel_name]}. Redefining it with'
          f' {stencil} is not allowed.'
      )
    kernel_dict.update({
        kernel_name: _discard_zero_coefficients(
            stencil['coeff'], stencil['shift']
        )
    })
  return kernel_dict


def _slice_kernel_dict(
    custom_kernel_dict: SliceKernelDictType | None = None,
):
  """Generates a dictionary of slice-based finite-difference operators.

  Args:
    custom_kernel_dict: A dictionary that stores the weights of kernels and
      their offsets. The keys of the dictionary are the names of the kernel, the
      first argument in the tuple value is the weights of the kernel, and the
      second argument in the tuple value is the offset of the kernel.

  Returns:
    A dictionary of slice-based finite difference operators. The returned dict
    contains both standard entries and customized ones.

  Raises:
    ValueError if any of the keys in `custom_kernel_dict` already exists in the
    standard kernel dict.
  """

  jnp_kernels = {
      'kS': _discard_zero_coefficients(
          COEFFS['centered_sum'], SHIFTS['centered_sum']
      ),
      'ks': _discard_zero_coefficients(
          COEFFS['backward_sum'], SHIFTS['backward_sum']
      ),
      'kD': _discard_zero_coefficients(
          COEFFS['centered_difference_1'], SHIFTS['centered_difference_1']
      ),
      'kd': _discard_zero_coefficients(
          COEFFS['backward_difference_1'], SHIFTS['backward_difference_1']
      ),
      'kd+': _discard_zero_coefficients(
          COEFFS['forward_difference_1'], SHIFTS['forward_difference_1']
      ),
      'kdd': _discard_zero_coefficients(
          COEFFS['centered_difference_2_order_2'],
          SHIFTS['centered_difference_2_order_2'],
      ),
      'kf2': _discard_zero_coefficients(
          COEFFS['face_flux_quick'], SHIFTS['face_flux_quick']
      ),
      'kf2+': _discard_zero_coefficients(
          COEFFS['face_flux_quick'][::-1], SHIFTS['face_flux_quick'][::-1]
      ),
      'k3d1+': _discard_zero_coefficients(
          COEFFS['forward_difference_3_order_1'],
          SHIFTS['forward_difference_3_order_1'],
      ),
      'kD4': _discard_zero_coefficients(
          COEFFS['centered_difference_1_order_4'],
          SHIFTS['centered_difference_1_order_4'],
      ),
  }
  if custom_kernel_dict:
    _add_customized_slice_kernel(jnp_kernels, custom_kernel_dict)

  return jnp_kernels


class ApplyKernelSliceOp(ApplyKernelOp):
  """Applies a kernel op using slice operators."""

  def _get_kernel(self, name: str) -> SliceKernelType:
    if name not in self._kernels.keys():
      raise ValueError(f'Invalid Kernel name: {name} is requested.')
    return self._kernels[name]

  def __init__(
      self,
      grid_params: grid_parametrization.GridParametrization,
      custom_kernel_dict: SliceKernelDictType | None = None,
  ) -> None:
    """Initializes kernels of convolutional finite-difference operators.

    Args:
      grid_params: The grid parametrization object.
      custom_kernel_dict: A dictionary that stores the weights of kernels and
        their offsets. The keys of the dictionary are the names of the kernel,
        the first argument in the tuple value is the weights of the kernel, and
        the second argument in the tuple value is the offset of the kernel.
    """
    super(ApplyKernelSliceOp, self).__init__(custom_kernel_dict)
    self._kernels = _slice_kernel_dict(custom_kernel_dict)
    self._grid_params = grid_params

  def add_kernel(self, custom_kernel_dict: ExternalDictKernelType):
    """Adds a customized kernel to the kernel library."""
    _add_customized_slice_kernel(self._kernels, custom_kernel_dict)

  def apply_kernel_op(
      self, array: ScalarField, name: str, axis: str
  ) -> ScalarField:
    kernel = self._get_kernel(name)
    return common_ops.finite_diff_with_slice(
        array, kernel['coeff'], kernel['shift'], axis, self._grid_params
    )
