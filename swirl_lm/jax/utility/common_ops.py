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

"""Library for common operations."""

import enum
from typing import Any, Callable, Literal, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types


ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap
VectorField = types.VectorField


_ArrayEquivalent = jax.Array | list[int] | list[float] | np.ndarray


class NormType(enum.Enum):
  """The type of norm to be used to quantify the residual."""

  L1 = 0
  L2 = 1
  L_INF = 2


def array_scatter_1d_update(
    array: ScalarField,
    axis: str,
    index: int,
    updates: ScalarField | float,
    grid_params: grid_parametrization.GridParametrization,
) -> ScalarField:
  """Updates a plane in a 3D local part of an array.

  Args:
    array: The 3D array to be updated.
    axis: The axis normal to the plane to be updated. For example, if the update
      is for a x-y plane, `axis` should be set to `z`.
    index: The index of the plane to be updated in `axis`.
    updates: The new values to be assigned in the plane specified by `axis` and
      `index`. If `updates` is a 3D array with one dimension of size 1, the
      plane normal to `axis` at `index` is updated. If `updates` is a floating
      point scalar, the value of the plane will be set to this number.
    grid_params: The grid parametrization object.

  Returns:
    A 3D array with values updated at specified plane.

  Raises:
    ValueError: If the shape of `updates` is different from the plane to be
      updated.
  """
  axis_index = grid_params.get_axis_index(axis)
  if axis_index == 0:
    array = array.at[index, :, :].set(jnp.squeeze(updates))
  elif axis_index == 1:
    array = array.at[:, index, :].set(jnp.squeeze(updates))
  else:
    array = array.at[:, :, index].set(jnp.squeeze(updates))
  return array


def array_scatter_1d_update_global(
    array: ScalarField,
    mesh: jax.sharding.Mesh,
    axis: str,
    core_index: int,
    plane_index: int,
    updates: ScalarField | float,
    grid_params: grid_parametrization.GridParametrization,
) -> ScalarField:
  """Updates a plane at the given `core_index` for a 3D distributed array.

  Args:
    array: The (local portion of the global) 3D array to be updated.
    mesh: A jax Mesh object representing the device topology.
    axis: The axis normal to the plane to be updated. For example, if the update
      is for a x-y plane, `axis` should be set to `z`.
    core_index: The index of the core in `axis`, in which the plane will be
      updated. The 3D arrays at other core indices will remain unchanged.
    plane_index: The local index of the plane to be updated in `axis`.
    updates: The new (local portion of the global) values to be assigned in the
      plane specified by `axis` and `plane_index`. If this is not a `floating
      point` value, `updates` has to be also in a 3D array with one dimension of
      size 1, matching the plane to be updated. If `updates` is a floating point
      number, the value of the plane will be set to this number.
    grid_params: The grid parametrization object.

  Returns:
    A 3D array with values updated at specified plane, in the same format as
    the input `array`.

  Raises:
    ValueError: If the shape of `updates` is different from the plane to be
      updated.
  """
  array_updated = array_scatter_1d_update(
      array, axis, plane_index, updates, grid_params
  )
  axis_index = grid_params.get_axis_index(axis)
  return jnp.where(
      jax.lax.axis_index(mesh.axis_names[axis_index]) == core_index,
      array_updated,
      array,
  )


def generate_scalar_field(
    field_name: str,
    grid_params: grid_parametrization.GridParametrization,
) -> ScalarFieldMap:
  """Generates a scalar field with zeros.

  This function ensures that the array shape is according to the order of the
  axes. Hence, it is recommended to use this function instead of calling
  `jnp.zeros` directly.

  Args:
    field_name: The name of the field.
    grid_params: The grid parametrization object.

  Returns:
    A scalar field map with the specified field name and zeros as the value.
  """
  shape = grid_params.to_data_axis_order(
      grid_params.nx, grid_params.ny, grid_params.nz
  )
  return {field_name: jnp.zeros(shape)}


def convert_to_3d_array_and_tile(
    f_1d: jax.Array,
    axis: str,
    nx: int,
    ny: int,
    nz: int,
    grid_params: grid_parametrization.GridParametrization,
) -> ScalarField:
  """Converts 1D tensor `f_1d` to a tiled 3D tensor.

  Example: if data_axis_order is ('z', 'x', 'y') and axis is 'x', then the
  output will be of shape [nz, len(f_1d), ny].

  Args:
    f_1d: The 1D array to convert to 3D.
    axis: The axis along which `f_1d` is laid out.
    nx: Number of grid points per core in the x dimension.
    ny: Number of grid points per core in the y dimension.
    nz: Number of grid points per core in the z dimension.
    grid_params: The grid parametrization object.

  Returns:
    A 3D array corresponding to f_1d.
  """
  assert f_1d.ndim == 1, f'Expecting rank-1 array, got rank-{f_1d.ndim} array.'
  axis_index = grid_params.get_axis_index(axis)
  reps = list(grid_params.to_data_axis_order(nx, ny, nz))
  if reps[axis_index] != len(f_1d):
    raise ValueError(
        f'The length of `f_1d` ({len(f_1d)}) does not match the number of grid'
        f' points per core in the {axis} dimension ({reps[axis_index]}).'
    )
  reps[axis_index] = 1
  reps = tuple(reps)
  return jnp.tile(reshape_to_broadcastable(f_1d, axis, grid_params), reps=reps)


def reshape_to_broadcastable(
    f_1d: jax.Array,
    axis: str,
    grid_params: grid_parametrization.GridParametrization,
) -> ScalarField:
  """Reshapes a rank-1 tensor to a form broadcastable against 3D fields.

  Example: if data_axis_order is ('z', 'x', 'y') and axis is 'x', then the
  output will be of shape [1, len(f_1d), 1].

  Args:
    f_1d: A rank-1 tensor.
    axis: The axis of variation of the input tensor `f_1d`.
    grid_params: The grid parametrization object.

  Returns:
    The reshaped array that can be broadcast against a 3D field.
  """
  assert (
      f_1d.ndim == 1
  ), f'Expecting rank-1 tensor, got rank-{f_1d.ndim} tensor.'
  axis_index = grid_params.get_axis_index(axis)
  if axis_index == 0:
    return f_1d[:, jnp.newaxis, jnp.newaxis]
  elif axis_index == 1:
    return f_1d[jnp.newaxis, :, jnp.newaxis]
  else:
    return f_1d[jnp.newaxis, jnp.newaxis, :]


def get_slice_of_1d_array_for_core(
    v: jax.Array, core_id: int, core_n: int, n: int
) -> jax.Array:
  """Returns the slice corresponding to `core_id` from a 1D array `v`.

  Args:
    v: A 1D array of size `core_n * num_cores + 2 * halo_width`. This includes
      the halos on the 2 ends of the domain. However, the internal halos are not
      included.
    core_id: The logical coordinate for the dimension under consideration.
    core_n: The number of non-halo elements per replica for this dimension.
    n: The number of elements (including halos) per replica for this dimension.

  Returns:
    A 1D array representing the slice of `v` corresponding to `core_id`.
  """
  start = core_id * core_n
  end = start + n
  return v[start:end]


def get_device_count_along_axes(
    mesh: jax.sharding.Mesh, axis_names: tuple[str, ...] | str | None = None
) -> int:
  """Obtains the number of devices along the given axes.

  Args:
    mesh: A jax Mesh object representing the device topology.
    axis_names: A single string or a tuple of strings specifying the names of
      the axes along which the devices are counted. If it is None, all the axes
      are considered and an integer with total number of devices is returned.

  Returns:
    The number of devices along the given axes as an integer. If a subset of
    `mesh.axis_names` is specified, the result is an integer with the total
    number of devices along those `axis_names`.
  """
  axis_names = mesh.axis_names if axis_names is None else axis_names
  return jax.lax.psum(1, axis_names)


def finite_diff_with_slice(
    array: jax.Array,
    coeff: tuple[float, ...],
    shift: tuple[int, ...],
    axis: str,
    grid_params: grid_parametrization.GridParametrization,
) -> jax.Array:
  """Performs finite difference operation using slicing and zero padding.

  This function is similar to `apply_convolutional_op_*` but uses slicing
  instead of convolution.
  If `shift` is (-1, 0, 2), the slices along `axis` are defined as follows:
  [array[3:3+l], array[2:2+l], array[0:0+l]]
  where, l = [array.shape along `axis`] - 3.
  Each slice is multiplied by the corresponding coefficient in `coeff` and
  summed. The result is then zero padded to match the original shape of `array`.
  The operation can be performed along any specified `axis`.

  This function is logically equivalent to the following roll implementation:
  out = [
      c * np.roll(array, shift=s, axis=axis_index)
      for c, s in zip(coeff, shift)
  ]
  out = np.sum(np.stack(out, axis=0), axis=0)
  Followed by setting out[s_min:] and out[:s_max] along `axis` to 0. Please
  refer to unit test of this function for the roll implementation. We do not use
  roll implementation for performance reasons.

  Args:
    array: Array of any rank. This can be 1D, 2D, 3D etc.
    coeff: The 1D kernel to be applied.
    shift: The 1D shift to be applied.
    axis: The axis along which the convolution is to be applied.
    grid_params: The grid parametrization object.

  Returns:
    The finite difference stencil operated on array.
  """
  if array.ndim != 3:
    raise ValueError(
        f'`array` must be a 3D array but its shape is {array.shape}.'
    )
  if len(coeff) != len(shift):
    raise ValueError(
        '`coeff` and `shift` must have the same length, but len(coeff):'
        f' {len(coeff)} and len(shift): {len(shift)}.'
    )
  s_max = max(shift) if max(shift) > 0 else 0
  s_min = min(shift) if min(shift) < 0 else 0
  axis_index = grid_params.get_axis_index(axis)
  slice_size = array.shape[axis_index] - (abs(s_max) + abs(s_min))

  def _get_slice(s: int) -> jax.Array:
    return jax.lax.dynamic_slice_in_dim(
        array, s_max - s, slice_size, axis=axis_index
    )

  out = [c * _get_slice(s) for c, s in zip(coeff, shift)]
  out = jnp.sum(jnp.stack(out, axis=0), axis=0)
  npad = [(0, 0)] * array.ndim
  npad[axis_index] = (s_max, -s_min)
  return jnp.pad(out, npad, 'constant')


def apply_convolutional_op_axis_0(array: ScalarField, conv_op: jax.Array):
  r"""Apply convolutional op in axis=0 of the array.

  This function uses zero padding along axis=0 and performs convolution.
  Here, the conv_op (filter) is expected to be a 3D array. The method reshapes
  the array to [array.shape[0]//k[-1], k[-1], array.shape[1], array.shape[2]]
  where, k = conv_op.shape, applies 1D convolution in a blockwise manner, and
  reshapes the array back to its input.

  Given a transposed 2D array A \in R^{N X N} partitioned into blocks
  A = [A_1, A_2,..., A_k], A_i \in R^{N / k_s X k_s}, and a filter
  K = [K_1, K_2, K_3], K_i \in R^{k_s X k_s}, the result of the conv1d will be:
  A' = [\sum{i=1}^2 A_i K_{i+1}, \sum{i=1}^3 A_i K_i,...,
        \sum{i=k-1}^k A_i K_{i-(k-2)}].

  Note that the array reshape operation occurs in row-major order, and that the
  arrays formed thus are not the same as the blocks A_i above. Rather, the
  submatrices that are convolved are the row stripes. For row 1 and a kernel
  size of k_s, for example, each reshaped block is:
  [[A_{1,1},..., A_{1,k_s}]
   [A_{1,k_s+1},..., A_{1,2k_s}]
   ...
   [A_{1,k-k_s+1},..., A_{1,k_s}]].
  The kernel is "slid" down the matrix (as in standard 1d convolution). With a_i
  denoting rows of the blocks above, this results in:
  [[a_1 K_2 + a_2 K_3]
   [a_1 K_1 + a_2 K_2 + a_3 K_3]
   ...
   [a_{k/k_s - 1} K_1 + a_{k/k_s} K_2]].
  An application of reshape can show is equivalent to the initial
  formulation above.

  NB: For optimal efficiency on TPU, the channel size of the kernel/filter
  should be 8, and the input array dimensions should be multiples of 128.

  Args:
    array: 3D array.
    conv_op: 3D array with dimension: [spatial_width, kernel_size, kernel_size].

  Returns:
    3D array: The finite difference stencil operated on array.
  """
  k = conv_op.shape
  shape = array.shape

  # [shape[0], shape[1], shape[2]] reshaped to
  # [shape[0]//k[-1], k[-1], shape[1], shape[2]].
  array = jnp.reshape(array, (-1, k[-1], shape[1], shape[2]))

  conv_op = jnp.expand_dims(conv_op, axis=0)
  conv_output = jax.lax.conv_general_dilated(
      array,
      conv_op,
      window_strides=[1, 1],
      padding='SAME',
      dimension_numbers=('WCHN', 'HWIO', 'WCHN'),
  )

  # Reverse the previous reshape to get the original shape.
  return jnp.reshape(conv_output, shape)


def apply_convolutional_op_axis_1(
    array: ScalarField, conv_op: jax.Array
) -> ScalarField:
  """Apply convolutional op in axis=1 of the array.

  A detailed explanation can be found in the documentation for
  apply_convolutional_op_axis_0.

  Args:
    array: A single 3D array.
    conv_op: 3D array with dimension: [spatial_width, kernel_size, kernel_size]

  Returns:
    3D array: The finite difference stencil operated on array.
  """
  # Note that we transpose the array and call the convolution along axis=0.
  # This is done for performance reasons.
  # Following snippet is mathematically equivalent in order to perform
  # convolution in axis=1 direction directly:
  # shape = array.shape
  # # [shape[0], shape[1], shape[2]] reshaped to
  # # [shape[0], shape[1]//k[-1], k[-1], shape[2]].
  # array = jnp.reshape(array, (shape[0], -1, k[-1], shape[2]))
  # conv_op = jnp.expand_dims(conv_op, axis=0)
  # conv_output = jax.lax.conv_general_dilated(
  #     array,
  #     conv_op,
  #     window_strides=[1, 1],
  #     padding='SAME',
  #     dimension_numbers=('NWCH', 'HWIO', 'NWCH'),
  # )
  # # Reverse the previous reshape to get the original shape.
  # conv_output = jnp.reshape(conv_output, shape)

  array = jnp.transpose(array, (1, 0, 2))
  conv_output = apply_convolutional_op_axis_0(array, conv_op)
  return jnp.transpose(conv_output, (1, 0, 2))


def apply_convolutional_op_axis_2(
    array: ScalarField, conv_op: jax.Array
) -> ScalarField:
  """Apply convolutional op in axis=2.

  A detailed explanation can be found in the documentation for
  apply_convolutional_op_axis_0.

  Args:
    array: A single 3D array of size [nz, nx, ny].
    conv_op: 3D array with dimension: [spatial_width, kernel_size, kernel_size]

  Returns:
    3D array: The finite difference stencil operated on array.
  """
  # Note that we transpose the array and call the convolution along axis=0.
  # This is done for performance reasons.
  # Following snippet is mathematically equivalent in order to perform
  # convolution in Y direction directly:
  # shape = array.shape
  # # [shape[0], shape[1], shape[2]] reshaped to
  # # [shape[0], shape[1], shape[2]//k[-1], k[-1]].
  # array = jnp.reshape(array, (shape[0], shape[1], -1, k[-1]))
  # conv_op = jnp.expand_dims(conv_op, axis=0)
  # conv_output = jax.lax.conv_general_dilated(
  #     array,
  #     conv_op,
  #     window_strides=[1, 1],
  #     padding='SAME',
  #     dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
  # )
  # # Reverse the previous reshape to get the original shape.
  # conv_output = jnp.reshape(conv_output, shape)

  array = jnp.transpose(array, (2, 1, 0))
  conv_output = apply_convolutional_op_axis_0(array, conv_op)
  return jnp.transpose(conv_output, (2, 1, 0))


def apply_convolutional_op(
    array: ScalarField,
    conv_op: jax.Array,
    axis: str,
    grid_params: grid_parametrization.GridParametrization,
) -> ScalarField:
  """Applies the operation `conv_op` to `array` along `axis`.

  Args:
    array: A single 3D array.
    conv_op: 3D array with dimension: [spatial_width, kernel_size, kernel_size]
    axis: The axis along which the convolution is to be applied.
    grid_params: The grid parametrization object.

  Returns:
    The 3D array with the operation applied.
  """
  axis_index = grid_params.get_axis_index(axis)
  kernel_size = conv_op.shape
  if kernel_size[-1] != kernel_size[-2]:
    raise ValueError(
        f'Kernel must be squared-shaped but its shape is {kernel_size}.'
    )
  if array.ndim != 3:
    raise ValueError(
        f'`array` must be a 3D array but its shape is {array.shape}.'
    )
  if conv_op.ndim != 3:
    raise ValueError(
        f'Kernel must be a 3D array but its shape is {conv_op.shape}.'
    )
  if array.shape[axis_index] % kernel_size[-1] != 0:
    raise ValueError(
        f'Kernel size must divide array size along axis={axis} evenly but'
        f' array size is {array.shape[axis_index]} and kernel size is'
        f' {kernel_size[-1]}.'
    )
  if axis_index == 0:
    return apply_convolutional_op_axis_0(array, conv_op)
  elif axis_index == 1:
    return apply_convolutional_op_axis_1(array, conv_op)
  else:
    return apply_convolutional_op_axis_2(array, conv_op)


def global_dot(
    vec1: jax.Array,
    vec2: jax.Array,
    mesh: jax.sharding.Mesh,
) -> jax.Array:
  """Computes the dot product of two vectors distributed over mesh.axis_names.

  The vectors in the input argument are arrays representing 1-D, 2-D or 3D
  fields.

  Args:
    vec1: One of the vectors for the dot product.
    vec2: The other vector for the dot product.
    mesh: A jax Mesh object representing the device topology.

  Returns:
    The dot product of the two input vectors.
  """
  local_dot = jnp.vdot(vec1, vec2)
  return jax.lax.psum(local_dot, axis_name=mesh.axis_names)


def global_mean(
    array: jax.Array,
    mesh: jax.sharding.Mesh,
    halo_width_x: int,
    halo_width_y: int,
    halo_width_z: int,
    grid_params: grid_parametrization.GridParametrization,
    axis: Sequence[str] | str | None = None,
    partition_axis: Sequence[str] | str | None = None,
) -> jax.Array:
  """Computes the mean of the array in a distributed setting.

  If `axis` is None the result will be a scalar. Otherwise, the result will
  have the same structure as `f` and the reduction axes will be preserved. The
  halos are always removed from the output.

  Args:
    array: A single 3D array.
    mesh: A jax Mesh object representing the device topology.
    halo_width_x: The width of the (symmetric) halos for in x dimension.
    halo_width_y: The width of the (symmetric) halos for in y dimension.
    halo_width_z: The width of the (symmetric) halos for in z dimension.
    grid_params: The grid parametrization object.
    axis: The dimension to reduce. If None, all dimensions are reduced and the
      result is a scalar.
    partition_axis: The dimensions of the partitions to reduce. If None, it is
      assumed to be the same as `axis`.

  Returns:
    The reduced array. If `axis` is None, a scalar that is the global mean of
    `f` is returned. Otherwise, the reduced result along the `axis` will be
    returned, with the reduced dimensions kept, while the non-reduced dimensions
    will have the halos striped.
  """
  if partition_axis is None:
    partition_axis = axis
  if partition_axis is None:
    partition_axis = mesh.axis_names

  group_count = get_device_count_along_axes(mesh, partition_axis)
  array = strip_halos(
      array, halo_width_x, halo_width_y, halo_width_z, grid_params
  )

  def grid_size_local(array, axes):
    size = 1
    for axis in axes:
      size *= array.shape[axis]
    return size

  if axis is None:  # gets a scalar.
    local_sum = jnp.sum(array, list(range(3)), keepdims=False)
  else:
    axis_index = grid_params.get_axis_index(axis)
    local_sum = jnp.sum(array, axis_index, keepdims=True)

  global_sum = jax.lax.psum(local_sum, axis_name=partition_axis)
  # Divide by the dimension of the physical full grid along `axis`.
  if axis is None:
    axis = grid_params.data_axis_order
  count = group_count * grid_size_local(array, grid_params.get_axis_index(axis))
  return global_sum / count


def apply_global_operator(
    operand: jax.Array,
    operator: Callable[[jax.Array], jax.Array],
    axis_names: tuple[str, ...],
) -> jax.Array:
  """Applies `operator` to `operand` in a distributed setting.

  Args:
    operand: A subgrid of a array.
    operator: A JAX operation to be applied to the `operand`.
    axis_names: A tuple of strings specifying the names of the axes along which
      the `all_gather` operation is performed.

  Returns:
    An array that is the global value for operator(operand).
  """
  local_val = jnp.atleast_1d(operator(operand))
  global_vals = jax.lax.all_gather(local_val, axis_names)
  return jnp.atleast_1d(operator(global_vals))


def compute_norm(
    v: ScalarField,
    norm_types: set[NormType],
    mesh: jax.sharding.Mesh,
) -> dict[str, jax.Array]:
  """Computes various vector-norms for a array.

  Args:
    v: An array to compute norm.
    norm_types: The norm types to be used for computation.
    mesh: A jax Mesh object representing the device topology.

  Returns:
    A dict of norms, with the key being the enum name for norm_type and the
      value being the corresponding norm.

  Raises:
    NotImplementedError: If `norm_types` contains elements that are not one
      of 'L1', 'L2', and 'L_INF'.
    ValueError: If `norm_types` is empty.
  """
  if not norm_types:
    raise ValueError('Supplied `norm_types` is empty.')

  def as_key(norm_type: NormType) -> str:
    return norm_type.name

  reduction_op = {
      'L1': jnp.sum,
      'L2': jnp.sum,
      'L_INF': jnp.max,
  }
  element_wise_op = {
      'L1': jnp.abs,
      'L2': jnp.square,
      'L_INF': jnp.abs,
  }

  norms_by_type = {}
  for norm_type in norm_types:
    if as_key(norm_type) in norms_by_type:
      continue
    if norm_type not in [NormType.L1, NormType.L2, NormType.L_INF]:
      raise NotImplementedError(
          '{} is not a valid norm type.'.format(norm_type.name)
      )
    norms_by_type[as_key(norm_type)] = apply_global_operator(
        element_wise_op[as_key(norm_type)](v),
        reduction_op[as_key(norm_type)],
        mesh.axis_names,
    )
    if norm_type == NormType.L2:
      norms_by_type[as_key(norm_type)] = jnp.sqrt(
          norms_by_type[as_key(norm_type)]
      )
  return norms_by_type


def validate_velocity(velocity: VectorField) -> None:
  """Validates the components of the input field have the same shape."""
  if (
      velocity[0].shape != velocity[1].shape
      or velocity[0].shape != velocity[2].shape
  ):
    raise ValueError(
        'All three fields must have the same shape, but velocity[0].shape:'
        f' {velocity[0].shape}, velocity[1].shape: {velocity[1].shape},'
        f' velocity[2].shape: {velocity[2].shape}.'
    )


def get_spectral_index_grid(
    core_nx: int,
    core_ny: int,
    core_nz: int,
    mesh: jax.sharding.Mesh,
    grid_params: grid_parametrization.GridParametrization,
    dtype: jnp.dtype = jnp.int32,
    halo_width_x: int = 0,
    halo_width_y: int = 0,
    halo_width_z: int = 0,
    pad_value=0,
) -> Mapping[str, jax.Array]:
  """Generates 1D grid where the elements correspond to the spectral index.

  In a distributed spectral setting, following the convention of FFT, in 1D,
  if a 2n vector is used, the sepctral components are stored in the order of

  [0, dk, 2 * dk, ..., (n-1) * dk, n * dk, -(n-1) * dk, -(n-2) * dk, ... -1],

  and if a 2n+1 vector is used, the components are stored in the order:

  [0, dk, 2 * dk, ..., (n-1) * dk, n * dk, -n * dk, -(n-1) * dk, ... -1],

  It is useful to extract the indices:

  [0, 1, 2, ... (n-1), n, -(n-1), ..., -1] (for 2n case)
  [0, 1, 2, ... (n-1), n, -n, -(n-1), ..., -1] (for 2n+1 case).

  to be used for manipulations/computations in spectral domain. This function
  extracts these indices for each core in the distributed setting (so each
  core has the correct corresponding portion of the indices). It also
  generates the `conjugate index`, i.e. the index corresponding to the complex
  conjugate component (note in 2n case, n's conjugate will still be n. The
  results are 3 sets of grid, for x, y, and z. Each set includes the original
  and the conjugate.

  Args:
    core_nx: The number of x points on each core, excluding halos.
    core_ny: The number of y points on each core, excluding halos.
    core_nz: The number of z points on each core, excluding halos.
    mesh: A jax Mesh object representing the device topology.
    grid_params: The grid parametrization object.
    dtype: The data type of the output. Either `jnp.int64` or `jnp.int32`.
      Default is `jnp.int32`.
    halo_width_x: The width of the (symmetric) halos for in x dimension.
    halo_width_y: The width of the (symmetric) halos for in y dimension.
    halo_width_z: The width of the (symmetric) halos for in z dimension.
    pad_value: The value to pad in the halo region.

  Returns:
    A dictionary of text to array, containing the following:
      * `xx`, `yy`, `zz`: The 1D spectral index grid in FFT convention.
      * `xx_c`, `yy_c`, `zz_c`: The corresponding conjugate 1D spectral index
         grid.
  """
  core_n = grid_params.to_data_axis_order(core_nx, core_ny, core_nz)
  halos = grid_params.to_data_axis_order(
      halo_width_x, halo_width_y, halo_width_z
  )

  def get_grid(axis_index: int):
    c = jax.lax.axis_index(mesh.axis_names[axis_index])
    n = core_n[axis_index] * mesh.devices.shape[axis_index]
    gg = jnp.arange(core_n[axis_index], dtype=dtype) + c * core_n[axis_index]
    gg = jnp.where(gg > n // 2, gg - n, gg)
    if n % 2 == 0:
      gg_c = jnp.where(gg == n // 2, gg, -1 * gg)
    else:
      gg_c = -1 * gg
    gg = jnp.pad(
        gg,
        pad_width=[[halos[axis_index], halos[axis_index]]],
        constant_values=pad_value,
    )
    gg_c = jnp.pad(
        gg_c,
        pad_width=[[halos[axis_index], halos[axis_index]]],
        constant_values=pad_value,
    )

    # expand_dims is used so that after shard_map along 3 axes, the data is
    # stored along last axis. shard_map on 3-axis mesh does not work on 1D
    # arrays.
    return jnp.expand_dims(gg, axis=[0, 1, 2]), jnp.expand_dims(
        gg_c, axis=[0, 1, 2]
    )

  xx, xx_c = get_grid(grid_params.get_axis_index('x'))
  yy, yy_c = get_grid(grid_params.get_axis_index('y'))
  zz, zz_c = get_grid(grid_params.get_axis_index('z'))

  return {
      'xx': xx,
      'yy': yy,
      'zz': zz,
      'xx_c': xx_c,
      'yy_c': yy_c,
      'zz_c': zz_c,
  }


def global_cumsum(
    array: jax.Array,
    axis: str,
    mesh: jax.sharding.Mesh,
    grid_params: grid_parametrization.GridParametrization,
) -> tuple[jax.Array, jax.Array]:
  """Computes the global cumulative sum of an array along the given `axis`.

  Args:
    array: Input array.
    axis: The axis along which the cumsum is performed.
    mesh: A jax Mesh object representing the device topology.
    grid_params: The grid parametrization object.

  Returns:
    A 2-tuple of arrays that has the same size of `array`. The first array has a
    cumsum from the first layer to the current one; and the second one has a
    cumsum from the current layer to the last one.
  """
  axis_index = grid_params.get_axis_index(axis)

  def plane_index(idx: int) -> tuple[Any, ...]:
    """Generates the indices slice to get a plane from a 3D array at `idx`."""
    indices = [slice(0, None)] * 3
    indices[axis_index] = idx
    return tuple(indices)

  def cumsum(g: jax.Array) -> jax.Array:
    """Performs cumulative sum of a 3D array along `dim`."""
    # In the case of global reduce, the local array has an added dimension 0
    # for the all-to-all function. We need to transform it back to a 3D array
    # for cumulative sum along the correct dimension.
    if len(g.shape) == 4:
      if axis_index == 0:
        perm = (0, 1, 2)
      elif axis_index == 1:
        perm = (1, 0, 2)
      else:  # axis_index=2
        perm = (1, 2, 0)

      # Here we move the dimension of replicas to the cumsum dimension. For
      # example, if axis_index = 2 and originally the `array` had shape
      # [REPLICAS, shape[0], shape[1], 1], the squeeze will result in shape
      # [REPLICAS, shape[0], shape[1]] and the transpose will result in
      # [shape[0], shape[1], REPLICAS]. And then the last line of the
      # global_reduce will just run jnp.cumsum on this last array with axis=2.
      # So the final output of global_reduce will be
      # [shape[0], shape[1], REPLICAS] which contains the block-level sum
      # along `axis`.
      g = jnp.transpose(jnp.squeeze(g, axis=axis_index + 1), perm)

    return jnp.cumsum(g, axis=axis_index)

  iloc = jax.lax.axis_index(mesh.axis_names[axis_index])

  local_cumsum = cumsum(array)

  replica_cumsum = apply_global_operator(
      jnp.expand_dims(local_cumsum[plane_index(-1)], axis=axis_index),
      cumsum,
      (mesh.axis_names[axis_index],),
  )
  cumsum_from_0 = jnp.where(
      iloc == 0,
      local_cumsum,
      local_cumsum
      + jnp.expand_dims(replica_cumsum[plane_index(iloc - 1)], axis=axis_index),  # pytype: disable=wrong-arg-types  # lax-types
  )
  cumsum_to_end = (
      jnp.expand_dims(replica_cumsum[plane_index(-1)], axis=axis_index)
      - cumsum_from_0
  )
  return cumsum_from_0, cumsum_to_end


def integration_in_dim(
    array: jax.Array,
    mesh: jax.sharding.Mesh,
    h: float,
    axis: str,
    grid_params: grid_parametrization.GridParametrization,
) -> tuple[jax.Array, jax.Array]:
  """Computes the integration of `array` along the given `axis`.

  Integration with definite integrals using the trapezoidal rule is performed
  for `array` along the `axis`. The integration is performed for arrays
  distributed across multiple devices along the integration direction.

  Args:
    array: The field to be integrated represented as a 3D array. The integration
      includes all nodes. If there are halos, those nodes will also be included
      in the integration result.
    mesh: A jax Mesh object representing the device topology.
    h: The uniform grid spacing in the integration direction.
    axis: The axis along which the integration is performed.
    grid_params: The grid parametrization object.

  Returns:
    A 2-tuple of arrays that has the same size of `array`. In the first array,
    each layer normal to the `axis` dimension is the integrated value from the
    first layer to the current one; in the second one, each layer is the
    integrated value from the current layer to the last one.
  """
  axis_index = grid_params.get_axis_index(axis)

  def plane_index(idx: int) -> tuple[Any, ...]:
    """Generates the indices slice to get a plane from a 3D array at `idx`."""
    indices = [slice(0, None)] * 3
    indices[axis_index] = idx
    return tuple(indices)

  cumsum_from_0, cumsum_to_end = global_cumsum(array, axis, mesh, grid_params)

  # Subtract half of the sum of the starting and end points of the cumulative
  # sum to conform with the trapezoidal rule of integral.
  global_lim_low = jax.lax.all_gather(
      array[plane_index(0)], mesh.axis_names[axis_index]
  )[0, ...]
  global_lim_high = jax.lax.all_gather(
      array[plane_index(-1)], mesh.axis_names[axis_index]
  )[-1, ...]

  integral_from_0 = cumsum_from_0 - 0.5 * (
      jnp.expand_dims(global_lim_low, axis=axis_index) + array
  )
  integral_to_end = cumsum_to_end + 0.5 * (
      array - jnp.expand_dims(global_lim_high, axis=axis_index)
  )
  return h * integral_from_0, h * integral_to_end


def strip_halos(
    array: ScalarField,
    halo_width_x: int,
    halo_width_y: int,
    halo_width_z: int,
    grid_params: grid_parametrization.GridParametrization,
) -> ScalarField:
  """Removes the halos from the input field component.

  Args:
    array: A field component. This is expected to be a 3D array with the halos
      of the field included.
    halo_width_x: The width of the (symmetric) halos for in x dimension.
    halo_width_y: The width of the (symmetric) halos for in y dimension.
    halo_width_z: The width of the (symmetric) halos for in z dimension.
    grid_params: The grid parametrization object.

  Returns:
    The inner part of the field component with the halo region removed.
    Represented as the same format as the input: a 3D array with the halo region
    excluded.
  """
  halos = grid_params.to_data_axis_order(
      halo_width_x, halo_width_y, halo_width_z
  )
  size = (
      array.shape[0] - 2 * halos[0],
      array.shape[1] - 2 * halos[1],
      array.shape[2] - 2 * halos[2],
  )
  return jax.lax.dynamic_slice(array, halos, size)


def pad(
    array: ScalarField,
    pad_x: tuple[int, int],
    pad_y: tuple[int, int],
    pad_z: tuple[int, int],
    grid_params: grid_parametrization.GridParametrization,
    value: float = 0.0,
) -> ScalarField:
  """Pads the input field with a given value.

  Args:
    array: A field component. This is expected to be a 3D array.
    pad_x: The padding lengths for the x dimension in the format: (pad_x_low,
      pad_x_hi). For instance, (2, 3) means f will be padded with a width of 2
      on the lower face of the x dimension and with a width of 3 on the upper
      face of the x dimension.
    pad_y: The padding lengths for the y dimension in the format: (pad_y_low,
      pad_y_hi).
    pad_z: The padding lengths for the z dimension in the format: (pad_z_low,
      pad_z_hi).
    grid_params: The grid parametrization object.
    value: The constant value to be used for padding.

  Returns:
    The padded input field as a 3D array.
  """
  rotated_paddings = grid_params.to_data_axis_order(pad_x, pad_y, pad_z)
  return jnp.pad(array, rotated_paddings, constant_values=value)


def get_face(
    value: ScalarField,
    axis: str,
    face: Literal[0, 1],
    index: int,
    grid_params: grid_parametrization.GridParametrization,
) -> ScalarField:
  """Gets the face in `value` that is `index` number of points from boundary.

  This function extracts the requested plane from a 3D array.

  Args:
    value: 3D array representing the field.
    axis: The axis of the plane to slice.
    face: The face of the plane to slice, with 0 representing the lower face,
      and 1 representing the higher face.
    index: The number of points that is away from the boundary determined by
      `axis` and `face`.
    grid_params: The grid parametrization object.

  Returns:
    If face is 0, then the (index + 1)'th plane is returned; if face is 1, then
    the length - index'th plane is returned.
  """
  axis_index = grid_params.get_axis_index(axis)
  n = value.shape[axis_index]
  start_idx = [0, 0, 0]

  if face == 0:  # low
    start_idx[axis_index] = index
  elif face == 1:  # high
    start_idx[axis_index] = n - index - 1
  else:
    raise ValueError(f'`face` should be 0 or 1 but got {face}.')

  size = list(value.shape)
  size[axis_index] = 1
  return jnp.squeeze(jax.lax.dynamic_slice(value, start_idx, size))
