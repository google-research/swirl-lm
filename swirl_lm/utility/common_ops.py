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
import functools
from typing import Any, Callable, Dict, Iterable, Literal, Mapping, Optional, Sequence, Text, Tuple

import numpy as np
from swirl_lm.utility import types
import tensorflow as tf
import tensorflow.compat.v1 as tf1

FlowFieldVal = types.FlowFieldVal
Dot = Callable[[FlowFieldVal, FlowFieldVal], tf.Tensor]
LinearOp = Callable[[FlowFieldVal], FlowFieldVal]
TensorMap = types.TensorMap
MutableTensorMap = types.MutableTensorMap

_DTYPE = types.TF_DTYPE
_TensorEquivalent = tf.Tensor | list[int] | list[float] | np.ndarray


def _is_state_variable(variable: FlowFieldVal) -> bool:
  if isinstance(variable, tf.Tensor):
    return True
  if isinstance(variable, Sequence):
    return isinstance(variable[0], tf.Tensor)
  return False


class NormType(enum.Enum):
  """The type of norm to be used to quantify the residual."""
  # The L1 norm.
  L1 = 0
  # The L2 norm.
  L2 = 1
  # The L infinity norm.
  L_INF = 2


def get_shape(tensor: FlowFieldVal) -> Tuple[int, int, int]:
  """Gets the shape of a z-x-y oriented tensor following (nx, ny, nz) order."""
  if isinstance(tensor, tf.Tensor):
    nz, nx, ny = tensor.shape
  else:
    nz = len(tensor)
    nx, ny = tensor[0].shape
  return (nx, ny, nz)


def tensor_scatter_1d_update(
    tensor: FlowFieldVal,
    dim: int,
    index: int,
    updates: FlowFieldVal | float,
) -> FlowFieldVal:
  """Updates a plane in a 3D tensor (as a 3D or a list of 2D `tf.Tensor`).

  Args:
    tensor: The 3D tensor (represented as a list of 2D or a 3D `tf.Tensor`) to
      be updated. For the problem with physical/logical shape of [dim0, dim1,
      dim2], if the `tensor` is a list of 2D `tf.Tensor`, the length of the list
      represents the `dim2`, and the 2D `tf.Tensor` in the list have the shape
      of [dim0, dim1]. If it is a single 3D `tf.Tensor`, then the shape is
      [dim2, dim0, dim1].
    dim: The dimension normal to the plane to be updated. For example, if the
      update is for a x-y plane, `dim` will indicate `z` (i.e. dim=2).
    index: The index of the plane to be updated in `dim`.
    updates: The new values to be assigned in the plane specified by `dim` and
      `index`. If this is not a `floating point` value, its format must match
      the format of `tensor`: If `tensor` is a `Sequence[tf.Tensor]`, the shape
      of `updates` must be the same as the plane to be updated in
      `Sequence[tf.Tensor]` format, i.e. if the update is on a x-y plane, it
      will have to be a length-one list of 2D `tf.Tensor` of shape [nx, ny]; if
      the update is on a x-z plan, it must be a list with length of `nz` of 2D
      `tf.Tensor`, with shape of [nx, 1]. If `tensor` is represented in a single
      3D `tf.Tensor` format, then `updates` has to be also in a 3D `tf.Tensor`
      with one dimension of size 1, matching the plane to be updated. If
      `updates` is a floating point number, the value of the plane will be set
      to this number.

  Returns:
    A 3D tensor with values updated at specified plane, in the same format as
    the input `tensor`.

  Raises:
    ValueError: If the shape of `updates` is different from the plane to be
      updated.
  """
  # Handles the case where the input is a single tf.Tensor.
  if isinstance(tensor, tf.Tensor):
    input_shape = tensor.get_shape().as_list()
    # The order of the dimensions in the tensor is [2, 0, 1], this
    # shift gives the right index to access the corresponding dimension.
    shifted_dim = (dim + 1) % 3
    if isinstance(updates, float):
      update_shape = input_shape
      update_shape[shifted_dim] = 1
      updates = updates * tf.ones(update_shape, dtype=tensor.dtype)
    perm = [0, 1, 2]
    perm_inv = [0, 1, 2]
    for i in range(shifted_dim):
      perm = np.roll(perm, -1)
      perm_inv = np.roll(perm_inv, 1)

    return tf.transpose(
        tf.tensor_scatter_nd_update(tf.transpose(tensor, perm),
                                    [[index]],
                                    tf.transpose(updates, perm)),
        perm_inv
    )

  # Below handles the case where the input is a sequence of 2D tf.Tensor.
  nz = len(tensor)
  nx, ny = tensor[0].get_shape().as_list()
  target_dims = [nx, ny, nz]
  target_dims[dim] = 1

  if isinstance(updates, Sequence):
    nz_u = len(updates)
    nx_u, ny_u = updates[0].get_shape().as_list()
    update_dims = [nx_u, ny_u, nz_u]

    for i in range(3):
      if target_dims[i] != update_dims[i]:
        raise ValueError(
            f'Dimension {i} of update plane is {update_dims[i]}, which is '
            f'different from the tensor dimension to be updated '
            f'({target_dims[i]}).')

  def update_tensor(
      data: tf.Tensor,
      update_val: tf.Tensor | float,
  ) -> tf.Tensor:
    """Updates `data` at `index` in dimension `dim`."""
    if dim not in (0, 1):
      raise ValueError(
          f'Tensor slice update only applies for 2D tensors, but dim {dim} is '
          f'applied.')

    if isinstance(update_val, float):
      if dim == 0:
        update_shape = (ny,)
      else:  # dim == 1:
        update_shape = (nx,)
      update_val = update_val * tf.ones(update_shape, dtype=data.dtype)
    else:
      update_val = tf.squeeze(update_val)

    # Because slice updates with the `tensor_scatter_nd_update` function
    # applies to the outer dimension only, the input tensor needs to be
    # transposed when updates need to be applied to the inner dimension.
    if dim == 1:
      data = tf.transpose(data)

    data = tf.tensor_scatter_nd_update(data, [[tf.cast(index, dtype=tf.int32)]],
                                       update_val[tf.newaxis, ...])

    return tf.transpose(data) if dim == 1 else data

  tensor_updated = tf.nest.map_structure(tf.identity, tensor)
  if dim == 2:
    tensor_updated[index] = tf.identity(updates[0]) if isinstance(
        updates, Sequence) else updates * tf.ones_like(tensor[index])
  else:
    if isinstance(updates, float):
      updates = [
          updates,
      ] * nz
    tensor_updated = tf.nest.map_structure(update_tensor, tensor, updates)

  return tensor_updated


def tensor_scatter_1d_update_global(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    tensor: FlowFieldVal,
    dim: int,
    core_index: int,
    plane_index: int,
    updates: FlowFieldVal | float,
) -> FlowFieldVal:
  """Updates a plane in a 3D field.

  Note that the 3D field is represented as a `tf.Tensor` or list of `tf.Tensor`.
  Also, this is not an in-place update. A new tensor will be created.

  Args:
    replica_id: The index of the current TPU replica.
    replicas: A numpy array that maps grid coordinates to replica id numbers.
    tensor: The (local portion of the global) 3D tensor to be updated.
    dim: The dimension of the plane normal to. Note we follow the convention
      here [x, y, z] = [0, 1, 2].
    core_index: The index of the core in `dim`, in which the plane will be
      updated. The 3D tensor with other indices will remain unchanged.
    plane_index: The local index of the plane to be updated in `dim`.
    updates: The new (local portion of the global) values to be assigned in the
      plane specified by `dim` and `index`. If this is not a `floating point`
      value, its format must match the format of `tensor`: If `tensor` is a
      `Sequence[tf.Tensor]`, the shape of `updates` must be the same as the
      plane to be updated in `Sequence[tf.Tensor]` format, i.e. if the update is
      on a x-y plane, it will have to be a length-one list of 2D `tf.Tensor` of
      shape [nx, ny]; if the update is on a x-z plane, it must be a list with
      length of `nz` of 2D `tf.Tensor`, with shape of [nx, 1]. If `tensor` is
      represented in a single 3D `tf.Tensor` format, then `updates` has to be
      also in a 3D `tf.Tensor` with one dimension of size 1, matching the plane
      to be updated. If `updates` is a floating point number, the value of the
      plane will be set to this number.

  Returns:
    A 3D tensor with values updated at specified plane, in the same format as
    the input `tensor`.

  Raises:
    ValueError: If the shape of `updates` is different from the plane to be
      updated.
  """
  coordinates = get_core_coordinate(replicas, replica_id, dtype=tf.int32)

  tensor_updated = tensor_scatter_1d_update(
      tf.nest.map_structure(tf.identity, tensor), dim, plane_index, updates)

  return tf.cond(
      tf.equal(coordinates[dim], core_index),
      true_fn=lambda: tensor_updated,
      false_fn=lambda: tf.nest.map_structure(tf.identity, tensor))


def tf_cast(tensor: FlowFieldVal, dtype) -> FlowFieldVal:
  """Casts a sequence of tensors to desired dtype."""
  if dtype is None:
    return tensor

  return tf.nest.map_structure(functools.partial(tf.cast, dtype=dtype),
                               tensor)


def average(
    a: FlowFieldVal,
    b: FlowFieldVal,
) -> FlowFieldVal:
  return tf.nest.map_structure(lambda x, y: 0.5 * (x + y), a, b)


def get_tile_name(
    base_name: Text,
    tile_id: int,
) -> Text:
  return '%s_tile_%d' % (base_name, tile_id)


def gen_field(
    field_name: Text,
    nx: int,
    ny: int,
    nz: int,
    dtype: tf.dtypes.DType = _DTYPE,
) -> TensorMap:
  return {field_name: tf.zeros([nz, nx, ny], dtype)}


def _get_field_range(
    state: TensorMap,
    field_name: Text,
    nz_start: int,
    nz_end: int,
) -> tf.Tensor:
  return state[field_name][nz_start:nz_end, :, :]


def get_range_results(
    name: Text,
    start: int,
    end: int,
    inputs: Any,
    keyed_queue_elements: TensorMap,
    state: TensorMap,
    replicas: np.ndarray,
) -> tuple[tf.Tensor, TensorMap]:
  """Get the slices of a specific range from the sequence of Tensor slices."""
  _ = inputs
  _ = keyed_queue_elements
  _ = replicas
  return _get_field_range(state, name, start, end), state


def get_field(
    state: TensorMap,
    field_name: Text,
    nz: int,
) -> tf.Tensor:
  return tf.stack([state[get_tile_name(field_name, i)] for i in range(nz)])


def convert_to_3d_tensor_and_tile(
    f_1d: tf.Tensor,
    dim: Literal[0, 1, 2],
    nx: int,
    ny: int,
    nz: int,
) -> tf.Tensor:
  """Converts 1D tensor `f_1d` to a tiled 3D tensor.

  This function follows the SwirlLM convention that tensor dimensions are
  ordered (z, x, y).

  Args:
    f_1d: The 1D tensor to convert to 3D.
    dim: The dimension along which `f_1d` is laid out.
    nx: Number of grid points per core in the x dimension.
    ny: Number of grid points per core in the y dimension.
    nz: Number of grid points per core in the z dimension.

  Returns:
    A 3D tensor corresponding to f_1d.
  """
  if dim == 0:
    return tf.tile(f_1d[tf.newaxis, :, tf.newaxis], multiples=(nz, 1, ny))
  elif dim == 1:
    return tf.tile(f_1d[tf.newaxis, tf.newaxis, :], multiples=(nz, nx, 1))
  elif dim == 2:
    return tf.tile(f_1d[:, tf.newaxis, tf.newaxis], multiples=(1, nx, ny))
  else:
    raise ValueError(f'Unsupported dim: {dim}. `dim` must be 0, 1, or 2.')


def _reshape_to_broadcastable_3d_tensor(
    f_1d: tf.Tensor,
    dim: Literal[0, 1, 2],
) -> tf.Tensor:
  """Reshapes a rank-1 tensor to a form broadcastable against 3D fields.

  3D fields are assumed to be represented as 3D tensors.

  See docstring of `reshape_to_broadcastable` for additional detail.

  Args:
    f_1d: A rank-1 tensor.
    dim: The dimension of variation of the input tensor `f_1d`.

  Returns:
    A rank-3 tensor that can be broadcast directly against a 3D field.
  """
  assert (
      f_1d.ndim == 1
  ), f'Expecting rank-1 tensor, got rank-{f_1d.ndim} tensor.'
  if dim == 0:
    # Set shape of tensor to (1, nx, 1).
    return f_1d[tf.newaxis, :, tf.newaxis]
  elif dim == 1:
    # Set shape of tensor to (1, 1, ny).
    return f_1d[tf.newaxis, tf.newaxis, :]
  else:  # dim == 2
    # Set shape of tensor to (nz, 1, 1).
    return f_1d[:, tf.newaxis, tf.newaxis]


def _reshape_to_broadcastable_no_3d_tensor(
    f_1d: tf.Tensor, dim: Literal[0, 1, 2]) -> Sequence[tf.Tensor]:
  """Reshapes a rank-1 tensor to a form broadcastable against 3D fields.

  The returned tensor is in Swirl-LM 3D form, which is a list of 2D tensors. The
  list represents the z-dimension, and the dimensions of 2D tensors correspond
  to x and y dimensions.

  Note that the output is NOT suitable for passing to `tf.nest.map_structure`.
  Instead use `map_structure_3d` defined in this file. The reason
  `tf.nest.map_structure` doesn't work is that it expects all the input tensors
  to have the exact same length, but the output will be a list of length one if
  `f_1d` is along x or y.

  Args:
    f_1d: A rank-1 tensor.
    dim: The dimension of variation of the input tensor `f_1d`.

  Returns:
    A list containing rank-2 tensor(s), which can be passed to
    `map_structure_3d`.
  """
  return tf.unstack(_reshape_to_broadcastable_3d_tensor(f_1d, dim))


def reshape_to_broadcastable(
    f_1d: tf.Tensor,
    dim: Literal[0, 1, 2],
    use_3d_tf_tensor: bool,
) -> tf.Tensor:
  """Reshapes a rank-1 tensor to a form broadcastable against 3D fields.

  Here, `dim` is 0, 1, or 2, corresponding to dimension x, y, or z respectively.
  The rank-1 tensor `f_1d` will be reshaped such that it represents a 3D field
  whose values vary only along dimension `dim`. However, for memory efficiency,
  the number of elements do not change. The output can be used in operations
  with 3D fields, with broadcasting occurring.

  3D fields are stored with order (z, x, y). If `use_3d_tf_tensor == False`,
  then 3D fields are stored as lists of 2D tensors, and the output of this
  function changes accordingly.

  The number of elements of `f_1d` must be correct on input (this is NOT
  checked). That is, if `dim`==0, 1, or 2, then len(f_1d) must equal nx, ny, or
  nz, respectively, where `nx`, `ny`, `nz` are the corresponding sizes of 3D
  fields.

  Examples of how the output can be used with 3D fields via broadcasting:
    Suppose here that `q` is a 3D field (a 3D tensor or list of 2D tensors)
      fx = reshape_to_broadcastable(fx_1d, 0, True)
      fy = reshape_to_broadcastable(fy_1d, 1, True)
      fz = reshape_to_broadcastable(fz_1d, 2, True)

      # This is a valid operation.
      map_structure_3d(lambda q, fx, fy, fz: q * fx * fy * fz, q, fx, fy, fz)

  Args:
    f_1d: A rank-1 tensor.
    dim: The dimension of variation of the input tensor `f_1d`.
    use_3d_tf_tensor: Whether 3D fields are represented by 3D tensors or lists
      of 2D tensors.

  Returns:
    The reshaped tensor that can be broadcast against a 3D field.
  """
  if dim not in (0, 1, 2):
    raise ValueError(f'Unsupported dim: {dim}. `dim` must be 0, 1, or 2.')

  if use_3d_tf_tensor:
    return _reshape_to_broadcastable_3d_tensor(f_1d, dim)
  else:
    raise NotImplementedError(
        'use_3d_tf_tensor == False is no longer supported.'
    )


def get_local_slice_of_1d_array(
    v: tf.Tensor, core_id: int | tf.Tensor, core_n: int, n: int
) -> tf.Tensor:
  """Retrieves the values local to `core_id` from a global 1D array `v`.

  Args:
    v: A 1D tensor representing a global array.
    core_id: The logical coordinate for the dimension under consideration.
    core_n: The number of non-halo elements per replica for this dimension.
    n: The number of elements (including halos) per replica for this dimension.

  Returns:
    A 1D tensor representing the local slice of `v` in `core_id`.
  """
  start = core_id * core_n
  end = start + n
  return v[start : end]


def get_slice(
    replica_idx: int,
    num_replicas: int,
    preserve_outer_boundaries: bool,
    halo_width: int = 1,
) -> slice:
  """Returns a `Slice` to be used on a tensor tile.

  In particular, the `Slice` will conditionally remove the outermost indices
  of a given tensor in a given dimension.

  For example, consider a 1x3 computational grid with a compute dimension of
  128x128(x1) per core. In the case where boundaries correspond to halos of
  shape 128x1, they should be discarded when slicing for the "valid" values of
  each tensor tile, given that the boundaries of tile 1 correspond to the 126th
  column of tile 0 and the 2nd column of tile 2. That is, the stored
  representation:
    -------------------------
    |       |       |       |
    |128x128|128x128|128x128|
    |       |       |       |
    -------------------------
  contains valid values as follows:
    -------------------------
    |       |       |       |
    |128x127|128x126|128x127|
    |       |       |       |
    -------------------------

  Args:
    replica_idx: The replica index in the dimension for which the `Slice` is
      being determined.
    num_replicas: The number of replicas in given dimension for which the
      `Slice` is being determined.
    preserve_outer_boundaries: If false, exclude the exterior regardless of the
      position of the replica in the dimension under consideration. If true, the
      outermost-indexed slice for outer replicas (the first and last replicas in
      the dimension under consideration) will be included in the `Slice`.
    halo_width: Width of the halo. Defaults to 1.

  Returns:
    A `Slice` corresponding to the given input parameters.
  """

  def is_first_replica():
    return replica_idx == 0

  def is_last_replica():
    return replica_idx == num_replicas - 1

  if not preserve_outer_boundaries:
    return slice(halo_width, -halo_width)
  elif num_replicas == 1:
    return slice(None, None)
  elif is_first_replica():
    return slice(0, -halo_width)
  elif is_last_replica():
    return slice(halo_width, None)
  else:  # Interior replica.
    return slice(halo_width, -halo_width)


def group_replicas(
    replicas: np.ndarray | tf.Tensor,
    axis: Sequence[int] | int | None = None,
) -> np.ndarray | tf.Tensor:
  """Obtains all the replica groups along direction `axis` in the compute grid.

  The `axis` can either be a single dimension or a tuple of dimensions. If all
  3 dimensions are passed or if `axis` is None, a single group containing all
  the replicas is returned. If a single axis is provided, there will be one
  group for every 1D slice of replicas along the `axis` dimension. If 2 axes are
  provided, there will be one group for every 2D slice of replicas in the
  corresponding dimensions.

  Args:
    replicas: The mapping from the global coordinate of the core to
      `replica_id`.
    axis: The axis or axes to group the replicas by.

  Returns:
    A 2D numpy array or tf.Tensor for the group assignment. Each row corresponds
    to a group of replicas aligned in the `axis` dimension(s).
  """
  reshape = np.reshape if isinstance(replicas, np.ndarray) else tf.reshape
  transpose = (np.transpose if isinstance(replicas, np.ndarray)
               else tf.transpose)

  if axis is None:
    # Returns a single group with all the replica id's.
    return reshape(replicas, [1, -1])

  if isinstance(axis, int):
    axis = [axis]

  if len(axis) > 3:
    raise ValueError('Axis list should have at most 3 dimensions. Found %d.' %
                     len(axis))

  # Transpose `replicas` so the dimensions in `axis` occur last.
  remaining_axis = list(set([0, 1, 2]) - set(axis))
  transpose_axes = remaining_axis + axis
  transposed_replicas = transpose(replicas, transpose_axes)
  # Flatten replica slices.
  slice_size = np.prod([replicas.shape[dim] for dim in axis])
  return reshape(transposed_replicas, [-1, slice_size])


def prep_step_by_chunk_fn(
    field_name: Text,
    z_begin: int,
    z_end: int,
    inputs: Sequence[tf.Tensor],
    keyed_queue_elements: TensorMap,
    state: MutableTensorMap,
    replicas: np.ndarray,
) -> tuple[list[tf.Tensor], MutableTensorMap]:
  """Does an in-place update of replica states.

  Run once per-field and per-chunk. The necessary init values passed are in
  through replicated inputs.

  Args:
    field_name: The name of the field.
    z_begin: The initial z coordinate for the chunk.
    z_end: The final z coordinate for the chunk (not inclusive).
    inputs: The tensor values with which to update the states.
    keyed_queue_elements: The elements that are fed through the `InfeedQueue`.
    state: A mapping of `field_name`s to `tf.Tensor`s representing state.
    replicas: A mapping of replica grid coordinates to replica ids.

  Returns:
    Tuple of (dummy output, mapping from `field_name`s to updated `tf.Tensor`s
      representing state.
  """
  _, _ = replicas, keyed_queue_elements
  ind = tf.range(z_begin, z_end, dtype=tf.int32)
  state[field_name] = tf.tensor_scatter_nd_update(state[field_name],
                                                  ind[...,
                                                      tf.newaxis], inputs[1])

  outputs = [tf.constant(0)]
  return outputs, state


@tf.function
def apply_op_x(
    tile_list: FlowFieldVal,
    mulop: tf.Tensor,
) -> FlowFieldVal:
  """Apply op in x."""
  # Handles the case of a single 3D tf.Tensor.
  if isinstance(tile_list, tf.Tensor):
    return tf.einsum('lj,ijk->ilk', mulop, tile_list)

  # Below handles the case of list of 2D tf.Tensor.
  if mulop.shape.as_list()[0] != mulop.shape.as_list()[1]:
    raise RuntimeError(
        'apply_op_x requires a square mulop. mulop shape is {}.'.format(
            mulop.shape))
  kernel_size = mulop.shape.as_list()[0]

  def op_one_tile(tile):
    x_size = tile.shape.as_list()[0]
    if x_size % kernel_size != 0:
      raise RuntimeError(
          'apply_op_x needs the tensor dim 0 size to be '
          'divisible by mulop size {}. Tensor shape is {}.'.format(
              mulop.shape.as_list()[1], tile.shape))
    return tf.matmul(mulop, tile)

  result = tf.nest.map_structure(op_one_tile, tile_list)

  return result


@tf.function
def apply_op_y(
    tile_list: FlowFieldVal,
    mulop: tf.Tensor,
) -> FlowFieldVal:
  """Apply op in y."""
  # Handles the case of a single 3D tf.Tensor.
  if isinstance(tile_list, tf.Tensor):
    return tf.einsum('ijk,kl->ijl', tile_list, mulop)

  # Below handles the case of a list of 2D tf.Tensor.
  if mulop.shape.as_list()[0] != mulop.shape.as_list()[1]:
    raise RuntimeError(
        'apply_op_y requires a square mulop. mulop shape is {}.'.format(
            mulop.shape))
  kernel_size = mulop.shape.as_list()[0]

  def op_one_tile(tile):
    y_size = tile.shape.as_list()[1]
    if y_size % kernel_size != 0:
      raise RuntimeError(
          'apply_op_y needs the tensor dim 1 size to be '
          'divisible by mulop size {}. Tensor shape is {}.'.format(
              mulop.shape.as_list()[1], tile.shape))
    return tf.matmul(tile, mulop)

  result = tf.nest.map_structure(op_one_tile, tile_list)
  return result


def apply_op_z(
    tile_list: FlowFieldVal,
    z_op_list: Sequence[tf.Tensor],
    shift: Sequence[int] | None = None,
) -> FlowFieldVal:
  """Apply op in z."""
  if len(z_op_list) != len(shift):
    raise RuntimeError('apply_op_z requires z_op_list length ({}) be equal to '
                       'shift length ({}).'.format(len(z_op_list), len(shift)))

  # This handles the case when the input is a single 3D `tf.Tensor`.
  if isinstance(tile_list, tf.Tensor):
    z_size = tile_list.shape.as_list()[0]
    if z_size < len(z_op_list):
      raise RuntimeError(
          'apply_op_z requires the first dimension size ({}) be '
          'greater than or equal to z_op_list length ({}).'.format(
              z_size, len(z_op_list)))
    result = tf.zeros_like(tile_list)
    for s, op in zip(shift, z_op_list):
      paddings = [[max(0, -s), max(0, s)], [0, 0], [0, 0]]
      result += op * tf.pad(tile_list[max(0, s):z_size - max(0, -s), :, :],
                            paddings)
    return result

  # This handles the case when the input is a list of 2D `tf.Tensor`.
  z_len = len(tile_list)
  if z_len < len(z_op_list):
    raise RuntimeError('apply_op_z requires tile_list length ({}) be greater '
                       'than or equal to z_op_list length ({}).'.format(
                           z_len, len(z_op_list)))

  out_list = tf.nest.map_structure(tf.zeros_like, tile_list)
  for s, op in zip(shift, z_op_list):
    for i in range(len(tile_list)):
      if i + s >= 0 and i + s < z_len:
        out_list[i] += op * tile_list[i + s]

  return out_list


@tf.function
def apply_convolutional_op_x(
    tiles: FlowFieldVal,
    convop: tf.Operation,
) -> FlowFieldVal:
  r"""Apply convolutional op in x.

  Here, the convop (filter) is expected to be a 3D tensor, according to the
  requirements in:
  https://www.tensorflow.org/api_docs/python/tf/nn/conv1d
  For each tile in the list supplied, the method reshapes the tile, applies
  1D convolution in a blockwise manner, and reshapes the tile back to its
  input.

  Given a transposed 2D tensor A \in R^{N X N} partitioned into blocks
  A = [A_1, A_2,..., A_k], A_i \in R^{N / k_s X k_s}, and a filter
  K = [K_1, K_2, K_3], K_i \in R^{k_s X k_s}, the result of the conv1d will be:
  A' = [\sum{i=1}^2 A_i K_{i+1}, \sum{i=1}^3 A_i K_i,...,
        \sum{i=k-1}^k A_i K_{i-(k-2)}].

  Note that the tensor reshape operation occurs in row-major order, and that the
  tensors formed thus are not the same as the blocks A_i above. Rather, the
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
  should be 8, and the input tile dimensions should be multiples of 128.

  Args:
    tiles: List of length `nz` of (square) 2D `tf.Tensor` with size
      [spatial_length, spatial_length] or equivalently a single 3D `tf.Tensor`
      of size [nz, spatial_length, spatial_length].
    convop: 3D tensor with dimension: [spatial_width, kernel_size, kernel_size]

  Returns:
    List of convolved 2D tensors.
  """
  kernel_size = convop.shape.as_list()
  if kernel_size[-1] != kernel_size[-2]:
    raise ValueError('Kernel must be squared-shaped but its shape is '
                     f'{kernel_size}.')

  def do_convol_x(tile, x_size, perm, shape_in, shape_out):
    if x_size % kernel_size[-1] != 0:
      raise ValueError(
          'Kernel size must divide tensor size evenly but '
          f'tensor x size is {x_size} and kernel size is {kernel_size[-1]}.')
    reshaped_transposed_input = tf.reshape(
        tf.transpose(tile, perm=perm), shape_in)
    convolved_output = tf.nn.conv1d(
        reshaped_transposed_input, filters=convop, stride=1, padding='SAME')
    reshaped_output = tf.transpose(
        tf.reshape(convolved_output, shape_out), perm=perm)
    return reshaped_output

  # Handles the case when the input is a single 3D `tf.Tensor`.
  if isinstance(tiles, tf.Tensor):
    z_size, x_size, y_size = tiles.shape.as_list()
    return do_convol_x(
        tiles,
        x_size,
        perm=[0, 2, 1],
        shape_in=[z_size, y_size, -1, kernel_size[-1]],
        shape_out=[z_size, y_size, -1])

  # Below handles the case when the input tile is a list of 2D `tf.Tensor`.
  def op_one_tile(tile):
    x_size, y_size = tile.shape.as_list()
    return do_convol_x(
        tile,
        x_size,
        perm=[1, 0],
        shape_in=[y_size, -1, kernel_size[-1]],
        shape_out=[y_size, -1])
  result = tf.nest.map_structure(op_one_tile, tiles)
  return result


@tf.function
def apply_convolutional_op_y(
    tiles: FlowFieldVal,
    convop: tf.Operation,
) -> FlowFieldVal:
  """Apply convolutional op in y.

  A detailed explanation can be found in the documentation for
  apply_convolutional_op_x.

  Args:
    tiles: List of length `nz` of (square) 2D `tf.Tensor` with size
      [spatial_length, spatial_length] or equivalently a single 3D `tf.Tensor`
      of size [nz, spatial_length, spatial_length].
    convop: 3D tensor with dimension: [spatial_width, kernel_size, kernel_size]

  Returns:
    List of convolved 2D tensors.
  """
  kernel_size = convop.shape.as_list()
  if kernel_size[-1] != kernel_size[-2]:
    raise ValueError('Kernel must be squared-shaped but its shape is '
                     f'{kernel_size}.')

  def do_convol_y(tile, y_size, shape_in, shape_out):
    if y_size % kernel_size[-1] != 0:
      raise ValueError(
          'Kernel size must divide tensor size evenly but '
          f'tensor y size is {y_size} and kernel size is {kernel_size[-1]}.')
    reshaped_input = tf.reshape(tile, shape_in)
    convolved_output = tf.nn.conv1d(
        reshaped_input, filters=convop, stride=1, padding='SAME')
    reshaped_output = tf.reshape(convolved_output, shape_out)
    return reshaped_output

  # This handles the case where the input is a single 3D `tf.Tensor`.
  if isinstance(tiles, tf.Tensor):
    z_size, x_size, y_size = tiles.shape.as_list()
    return do_convol_y(
        tiles,
        y_size,
        shape_in=[z_size, x_size, -1, kernel_size[-1]],
        shape_out=[z_size, x_size, -1])

  # Below handles the case where the input is a liost of 2D `tf.Tensor`.
  def op_one_tile(tile):
    x_size, y_size = tile.shape.as_list()
    return do_convol_y(
        tile,
        y_size,
        shape_in=[x_size, -1, kernel_size[-1]],
        shape_out=[x_size, -1])
  result = tf.nest.map_structure(op_one_tile, tiles)
  return result


@tf.function
def _apply_slice_op(
    tiles: FlowFieldVal,
    op: Callable[[Iterable[tf.Tensor]], tf.Tensor],
) -> list[tf.Tensor]:
  """Helper to apply a slice op."""
  if isinstance(tiles, tf.Tensor):
    return tf.map_fn(op, tiles)
  else:
    return tf.nest.map_structure(op, tiles)


@tf.function
def apply_slice_op_x(
    tiles: FlowFieldVal,
    sliceop: Callable[[Iterable[tf.Tensor]], tf.Tensor],
) -> list[tf.Tensor]:
  """Apply slice op in x.

  Args:
    tiles: List of length `nz` of (square) 2D `tf.Tensor` with size
      [spatial_length, spatial_length] or equivalently a single 3D `tf.Tensor`
      of size [nz, spatial_length, spatial_length].
    sliceop: Function that applies a kernel to a 2D tensor and returns a 2D
      tensor of the same dimension.

  Returns:
    List of 2D tensors with op applied.
  """
  return _apply_slice_op(tiles, sliceop)


@tf.function
def apply_slice_op_y(
    tiles: FlowFieldVal,
    sliceop: Callable[[Iterable[tf.Tensor]], tf.Tensor],
) -> list[tf.Tensor]:
  """Apply slice op in y.

  Args:
    tiles: List of length `nz` of (square) 2D `tf.Tensor` with size
      [spatial_length, spatial_length] or equivalently a single 3D `tf.Tensor`
      of size [nz, spatial_length, spatial_length].
    sliceop: Function that applies a kernel to a 2D tensor and returns a 2D
      tensor of the same dimension.

  Returns:
    List of 2D tensors with op applied.
  """
  return _apply_slice_op(tiles, sliceop)


def split_state_in_z(
    state: TensorMap,
    state_keys: Iterable[Text],
    nz: int,
) -> MutableTensorMap:
  """Splits state in z, assuming that z is in the first dimension.

  Args:
    state: Dictionary mapping names to state variables.
    state_keys: A list of string keys (must be present in state dictionary).
    nz: Z-dimension length/size.

  Returns:
    State split in the z dimension.
  """
  out_dict = {}
  for state_key in state_keys:
    out_dict.update({
        get_tile_name(state_key, i): state[state_key][i, :, :]
        for i in range(nz)
    })
  return out_dict


def merge_state_in_z(
    state: TensorMap,
    state_keys: Iterable[Text],
    nz: int,
) -> MutableTensorMap:
  """Merges state in z, assuming that z is in the first dimension.

  Args:
    state: Dictionary mapping names to state variables.
    state_keys: A list of string keys (must be present in state dictionary).
    nz: Z-dimension length/size.

  Returns:
    State stacked in the z dimension.
  """
  out_dict = {}
  for state_key in state_keys:
    out_dict.update({
        state_key:
            tf.stack([state[get_tile_name(state_key, i)] for i in range(nz)],
                     axis=0)
    })
  return out_dict


def local_dot(
    vec1: FlowFieldVal,
    vec2: FlowFieldVal,
) -> tf.Tensor:
  """Computes the dot product of two local vectors.

  The vectors in the input argument can be `tf.Tensor` or list of `tf.Tensor`
  representing 1-D, 2-D or 3D fields.

  Args:
    vec1: One of the vectors for the dot product.
    vec2: The other vector for the dot product.

  Returns:
    The dot product of the two input vectors.
  """
  return tf.math.reduce_sum(vec1 * vec2)


def local_vdot(
    vec1: FlowFieldVal,
    vec2: FlowFieldVal,
) -> tf.Tensor:
  """Computes the dot product of two local complex tensors.

  Args:
    vec1: The first argument of the dot product, whose complex conjugate is
      taken before the calculation of the dot product.
    vec2: The second argument to the dot product.

  Returns:
    The dot product of the two input vectors.
  """
  return local_dot(tf.math.conj(vec1), vec2)


def global_dot(
    vec1: FlowFieldVal,
    vec2: FlowFieldVal,
    group_assignment: np.ndarray,
) -> tf.Tensor:
  """Computes the dot product of two distributed vectors.

  The vectors in the input argument can be `tf.Tensor` or list of `tf.Tensor`
  representing 1-D, 2-D or 3D fields.

  Args:
    vec1: One of the vectors for the dot product.
    vec2: The other vector for the dot product.
    group_assignment: A 2d int32 lists with shape [num_groups,
      num_replicas_per_group]

  Returns:
    The dot product of the two input vectors.
  """
  local_sum = local_dot(vec1, vec2)

  return tf1.tpu.cross_replica_sum(local_sum, group_assignment)


def global_mean(
    f: FlowFieldVal,
    replicas: np.ndarray,
    halos: Sequence[int] = (0, 0, 0),
    axis: Sequence[int] | int | None = None,
    partition_axis: Sequence[int] | int | None = None,
) -> FlowFieldVal:
  """Computes the mean of the tensor in a distributed setting.

  The 3D tensor in the input argument can be represented as a list of 2D
  `tf.Tensor` or a single 3D `tf.Tensor`. If it is a list of 2D `tf.Tensor`, the
  length of the list represents `nz` while the shape of the 2D `tf.Tensor` is
  [nx, ny]. If it is a single 3D `tf.Tensor`, its shape is [nz, nx, ny]. The
  `halos` is always represented as [halo_x, halo_y, halo_z].
  If `axis` is None the result will be a scalar. Otherwise, the result will
  have the same structure as `f` and the reduction axes will be preserved. The
  halos are always removed from the output.

  Args:
    f: The vectors for the mean computation. This is represented either as a
      length `nz` list of 2D `tf.Tensor` with shape of [nx, ny], or a single 3D
      `tf.Tensor` with shape of [nz, nx, ny].
    replicas: A 3D numpy array describing the grid of the TPU topology and
      mapping replica grid coordinates to replica ids, e.g. [[[0, 1], [2, 3]],
      [[4, 5], [6, 7]]] represents 8 cores partitioned into a 2 x 2 x 2 grid.
    halos: Region to exclude in the calculation. Represented as [halo_x, halo_y,
      halo_z].
    axis: The dimension to reduce. If None, all dimensions are reduced and the
      result is a scalar. Note the indexing of dimension is [x, y, z] = [0, 1,
      2].
    partition_axis: The dimensions of the partitions to reduce. If None, it is
      assumed to be the same as `axis`.

  Returns:
    The reduced tensor. If `axis` is None, a scalar that is the global mean of
    `f` is returned. Otherwise, the reduced result along the `axis` will be
    returned, with the reduced dimensions kept, while the non-reduced dimensions
    will have the halos striped, the returned result will be in a format
    consistent with the input (either a list of 2D `tf.Tensor` or a single 3D
    `tf.Tensor`.
  """
  if partition_axis is None:
    partition_axis = axis

  group_assignment = group_replicas(replicas, partition_axis)
  group_count = len(group_assignment[0])
  f = strip_halos(f, halos)

  def grid_size_local(f, axes):
    size = 1
    if isinstance(f, tf.Tensor):
      shape = f.shape.as_list()
      for axis in axes:
        size *= shape[(axis + 1) % 3]
      return size

    for axis in axes:
      shape = len(f) if axis == 2 else f[0].shape.as_list()[axis]
      size *= shape
    return size

  def reduce_local(x, axis, keep_dims=False):
    if isinstance(x, tf.Tensor):
      new_axis = [(ax + 1) % 3 for ax in axis]
      local_sum = tf.reduce_sum(x, new_axis, keep_dims)
      return local_sum

    dims = list(axis)
    if 2 in dims:
      x = [tf.math.add_n(x)]
      dims.remove(2)
    if dims:
      x = [tf.math.reduce_sum(x_i, axis=dims, keepdims=keep_dims) for x_i in x]
    return x

  denominator_type = f.dtype if isinstance(f, tf.Tensor) else f[0].dtype
  if axis is None:  # Returns a scalar.
    axis = list(range(3))
    local_sum_temp = reduce_local(f, axis, keep_dims=False)
    local_sum = local_sum_temp if isinstance(
        f, tf.Tensor) else local_sum_temp[0]
    global_sum = tf1.tpu.cross_replica_sum(local_sum, group_assignment)
    count = group_count * grid_size_local(f, axis)
    return global_sum / tf.cast(count, denominator_type)
  else:
    if isinstance(axis, int):
      axis = [axis]
    local_sum = reduce_local(f, axis, keep_dims=True)
    global_sum = tf1.tpu.cross_replica_sum(local_sum, group_assignment)

    # Divide by the dimension of the physical full grid along `axis`.
    count = group_count * grid_size_local(f, axis)
    mean = global_sum / tf.cast(count, denominator_type)

    # Recover the original format, since `cross_replica_sum` would implicitly
    # do a stacking if the original input is list of `tf.Tensor`.
    mean = tf.unstack(mean) if not isinstance(f, tf.Tensor) else mean
    return mean


def global_reduce(
    operand: tf.Tensor,
    operator: Callable[[tf.Tensor], tf.Tensor],
    group_assignment: np.ndarray,
) -> tf.Tensor:
  """Applies `operator` to `operand` in a distributed setting.

  Args:
    operand: A subgrid of a tensor.
    operator: A Tensorflow operation to be applied to the `operand`.
    group_assignment: A 2d int32 list with shape `[num_groups,
      num_replicas_per_group]`. It is assumed that the size of group is the same
      for all groups.

  Returns:
    A scalar that is the global value for operator(operand).
  """
  num_replicas = len(group_assignment[0])
  local_val = tf.repeat(tf.expand_dims(operator(operand), 0), num_replicas, 0)

  global_vals = tf.raw_ops.AllToAll(
      input=local_val,
      group_assignment=group_assignment,
      concat_dimension=0,
      split_dimension=0,
      split_count=num_replicas)
  return operator(global_vals)


def remove_global_mean(
    f: FlowFieldVal,
    replicas: np.ndarray,
    halo_width: int,
) -> FlowFieldVal:
  """Removes the mean (excluding halos) of the tensor in a distributed setting.

  Args:
    f: The vector to remove the mean.
    replicas: A 3D numpy array describing the grid of the TPU topology and
      mapping replica grid coordinates to replica ids, e.g. [[[0, 1], [2, 3]],
      [[4, 5], [6, 7]]] represents 8 cores partitioned into a 2 x 2 x 2 grid.
    halo_width: The width of the halo.

  Returns:
    The modified tensor with mean (excluding halos) component removed.
  """
  f_mean = global_mean(f, replicas, (halo_width,) * 3)

  return tf.nest.map_structure(lambda f_i: f_i - f_mean, f)


def compute_norm(
    v: FlowFieldVal,
    norm_types: Sequence[NormType],
    replicas: np.ndarray,
) -> Dict[Text, tf.Tensor]:
  """Computes various norms for a vector, as a tensor or a list of tensor.

  Args:
    v: The vector to compute norm, a tensor or a list of tensor.
    norm_types: The norm types to be used for computation.
    replicas: A numpy array that maps a replica's grid coordinate to its
      replica_id, e.g. replicas[0, 0, 0] = 0, replicas[0, 0, 1] = 1.

  Returns:
    A dict of norms, with the key being the enum name for norm_type and the
      value being the corresponding norm.

  Raises:
    NotImplementedError: If `norm_types` contains elements that are not one
      of 'L1', 'L2', and 'L_INF'.
    ValueError: If `norm_types` is empty.
  """
  if not isinstance(v, tf.Tensor):
    v = tf.stack(v)

  if not norm_types:
    raise ValueError('Supplied `norm_types` is empty.')

  num_replicas = np.prod(replicas.shape)
  group_assignment = np.array([range(num_replicas)], dtype=np.int32)

  def as_key(norm_type: NormType) -> Text:
    return norm_type.name

  typed_norms = {}
  for norm_type in norm_types:
    if as_key(norm_type) in typed_norms:
      continue

    if norm_type == NormType.L1:
      l1_norm_op = lambda u: tf.math.reduce_sum(tf.abs(u))
      norm = global_reduce(v, l1_norm_op, group_assignment)
    elif norm_type == NormType.L2:
      norm = tf.math.sqrt(
          global_reduce(v * v, tf.math.reduce_sum, group_assignment))
    elif norm_type == NormType.L_INF:
      l_inf_norm_op = lambda u: tf.math.reduce_max(tf.abs(u))
      norm = global_reduce(v, l_inf_norm_op, group_assignment)
    else:
      raise NotImplementedError('{} is not a valid norm type.'.format(
          norm_type.name))
    typed_norms[as_key(norm_type)] = norm

  return typed_norms


def get_core_coordinate(
    replicas: np.ndarray,
    replica_id: tf.Tensor,
    dtype: tf.dtypes.DType = tf.int32,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Get the coordinate for the core with `replica_id`.

  Args:
    replicas: The mapping from the global coordinate of the core to
      `replica_id`.
    replica_id: The replica id of the current core.
    dtype: The data type of the output. Either `tf.int64` or `tf.int32`. Default
      is `tf.int32`.

  Returns:
    A 1D tensor of length 3 that represents the coordinate of the core.
  """
  coordinate = tf1.where(
      tf.equal(tf.cast(replicas, dtype), tf.cast(replica_id, dtype)))

  # Using tf.math.reduce_mean to declare/clarify the shape so it is clear to
  # be a scalar. Otherwise, due to the use of `tf.where`, in some applications
  # the shape could not be inferred during the XLA compilation.
  x = tf.cast(tf.math.reduce_mean(coordinate[0, 0]), dtype=dtype)
  y = tf.cast(tf.math.reduce_mean(coordinate[0, 1]), dtype=dtype)
  z = tf.cast(tf.math.reduce_mean(coordinate[0, 2]), dtype=dtype)
  return x, y, z


def validate_fields(
    u: FlowFieldVal,
    v: FlowFieldVal,
    w: FlowFieldVal,
) -> None:
  """Validates the components of the input field have the same shape."""
  if ((isinstance(u, tf.Tensor) ^ isinstance(v, tf.Tensor)) or
      (isinstance(v, tf.Tensor) ^ isinstance(w, tf.Tensor))):
    raise ValueError('All inputs `u`, `v`, and `w` must all be in the same '
                     'format. They must all be lists of 2D `tf.Tensor` or '
                     'single 3D `tf.Tensor`.')
  u_nx, u_ny, u_nz = get_field_shape(u)
  v_nx, v_ny, v_nz = get_field_shape(v)
  w_nx, w_ny, w_nz = get_field_shape(w)
  if (u_nz != v_nz or u_nz != w_nz or (u_nx, u_ny) != (v_nx, v_ny) or
      (u_nx, u_ny) != (w_nx, w_ny)):
    raise ValueError('All fields musth have the same shape, but '
                     'number of x-y slices for `u`: %d, shape of slice: %s; '
                     'number of x-y slices for `v`: %d, shape of slice: %s; '
                     'number of x-y slices for `w`: %d, shape of slice: %s.' %
                     (u_nz, str((u_nx, u_ny)), v_nz, str(
                         (v_nx, v_ny)), w_nz, str((w_nx, w_nz))))


def get_field_shape(u: FlowFieldVal) -> tuple[int, int, int]:
  """Gets the 3D volume shape of the sequence of Tensor represents."""
  if isinstance(u, tf.Tensor):
    nz, nx, ny = u.shape.as_list()
  else:
    nz = len(u)
    nx = u[0].shape.as_list()[0]
    ny = u[0].shape.as_list()[1]
  return nx, ny, nz


def get_spectral_index_grid(
    core_nx: int,
    core_ny: int,
    core_nz: int,
    replicas: np.ndarray,
    replica_id: tf.Tensor,
    dtype: tf.dtypes.DType = tf.int32,
    halos: Sequence[int] = (0, 0, 0),
    pad_value=0,
) -> Mapping[Text, tf.Tensor]:
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
    replicas: The mapping from the core coordinate to the local replica id
      `replica_id`.
    replica_id: The id for the local core.
    dtype: The data type of the output. Either `tf.int64` or `tf.int32`. Default
      is `tf.int32`.
    halos: A sequence of int representing the halo in each dimension.
    pad_value: The value to pad in the halo region.

  Returns:
    A dictionary of text to tensor, containing the following:
      * `xx`, `yy`, `zz`: The 1D spectral index grid in FFT convention.
      * `xx_c`, `yy_c`, `zz_c`: The corresponding conjugate 1D spectral index
         grid.
  """
  coordinate = get_core_coordinate(replicas, replica_id, dtype)
  compute_shape = replicas.shape

  core_n = [core_nx, core_ny, core_nz]

  def get_grid(dim):
    c = coordinate[dim]
    n = core_n[dim] * compute_shape[dim]
    gg = tf.range(core_n[dim], dtype=dtype) + c * core_n[dim]
    gg = tf1.where(gg > n // 2, gg - n, gg)
    if n % 2 == 0:
      gg_c = tf1.where(tf.equal(gg, n // 2), gg, -1 * gg)
    else:
      gg_c = -1 * gg
    gg = tf.pad(
        gg, paddings=[[halos[dim], halos[dim]]], constant_values=pad_value)
    gg_c = tf.pad(
        gg_c, paddings=[[halos[dim], halos[dim]]], constant_values=pad_value)
    return gg, gg_c

  xx, xx_c = get_grid(0)
  yy, yy_c = get_grid(1)
  zz, zz_c = get_grid(2)

  return {
      'xx': xx,
      'yy': yy,
      'zz': zz,
      'xx_c': xx_c,
      'yy_c': yy_c,
      'zz_c': zz_c
  }


def get_tensor_shape(state: Sequence[tf.Tensor]) -> tuple[int, int, int]:
  """Retrieves the shape of a 3D tensor represented as a stack of 2D tf.Tensor.

  Args:
    state: A 3D tensor represented as a sequence of `tf.Tensor`. The length of
      the sequence is the shape in the z direction (a.k.a. dim 2), and the shape
      of the `tf.Tensor` corresponds to the size of the x-y plane.

  Returns:
    The shape of input 3D tensor represented as (nz, nx, ny).

  Raises:
    ValueError: If the `tf.Tensor` in the sequence is not 2D.
  """
  nz = len(state)
  if nz == 0:
    return (0, 0, 0)

  tensor_shape = state[0].shape
  if len(tensor_shape) != 2:
    raise ValueError('The tensor in the list has to be 2D. {} is given.'.format(
        len(tensor_shape)))

  return (nz, tensor_shape[0], tensor_shape[1])


def integration_in_dim(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    f: FlowFieldVal,
    h: float,
    dim: int,
) -> tuple[FlowFieldVal, FlowFieldVal]:
  """Computes the integration of `f` along the z dimension.

  Integration with definite integrals is performed for `f` along the `dim`
  dimension. The integration is performed for tensors distributed across
  multiple TPUs along the integration direction.

  Args:
    replica_id: The index of the current TPU replica.
    replicas: A 3D numpy array describing the grid of the TPU topology, e.g.
      [[[0, 1], [2, 3]], [[4, 5], [6, 7]]] represents 8 cores partitioned into a
      2 x 2 x 2 grid.
    f: The field to be integrated. Note that `f` is either represented as a list
      of 2D `tf.Tensor` (x-y slices) or a single 3D `tf.Tensor`. The integration
      includes all nodes. If there are halos, those nodes will also be included
      in the integration result. Note if it is a list of 2D `tf.Tensor`, the
      length of the list is `nz` and each slice has the shape [nx, ny]; if it is
      a single 3D `tf.Tensor`, the shape is [nz, nx, ny].
    h: The uniform grid spacing in the integration direction.
    dim: The dimension along which the integration is performed. The indices
      representing `dim` are in the convention of [x, y, z] = [0, 1, 2].

  Returns:
    Two decks of tensors that has the same size of `f`. In the first deck, each
    layer normal to the `dim` dimension is the integrated value from the first
    layer to the current one; in the second one, each layer is the integrated
    value from the current layer to the last one.
  """
  ix, iy, iz = get_core_coordinate(replicas, replica_id)
  group_assignment = group_replicas(replicas, dim)

  def plane_index(idx: int):
    """Generates the indices slice to get a plane from a 3D tensor at `idx`."""
    if dim == 0:
      indices = [slice(0, None), idx, slice(0, None)]
    elif dim == 1:
      indices = [slice(0, None), slice(0, None), idx]
    elif dim == 2:
      indices = [idx, slice(0, None), slice(0, None)]
    else:
      raise ValueError(
          'Integration dimension should be one of 0, 1, and 2. {} is given.'
          .format(dim))

    return indices

  if dim == 0:
    iloc = ix
    axis = 1
  elif dim == 1:
    iloc = iy
    axis = 2
  elif dim == 2:
    iloc = iz
    axis = 0
  else:
    raise ValueError(
        'Integration dimension should be one of 0, 1, and 2. {} is given.'
        .format(dim))

  def cumsum(g: tf.Tensor) -> tf.Tensor:
    """Performs cumulative sum of a 3D tensor along `dim`."""
    # In the case of global reduce, the local tensor has an added dimension 0
    # for the all-to-all function. We need to transform it back to a 3D tensor
    # for cumulative sum along the correct dimension.
    if len(g.shape) == 4:
      if dim == 0:
        perm = (1, 0, 2)
      elif dim == 1:
        perm = (1, 2, 0)
      elif dim == 2:
        perm = (0, 1, 2)

      # Here we move the dimension of replicas to the integration dimension. For
      # example, assuming dim = 1 (or y dimension) which implies axis = 2, if
      # originally the f tensor had shape [REPLICAS, dim_z, dim_x, 1], the
      # squeeze will result in shape [REPLICAS, dim_z, dim_x] and the transpose
      # will result in [dim_z, dim_x, REPLICAS]. And then the last line of the
      # global_reduce will just run tf.cumsum on this last tensor with axis=2.
      # So the final output of global_reduce will be [dim_z, dim_x, REPLICAS]
      # which contains the block-level integral along `axis`.
      g = tf.transpose(tf.squeeze(g, axis=axis + 1), perm)

    return tf.cumsum(g, axis=axis)

  f_stacked = f if isinstance(f, tf.Tensor) else tf.stack(f)
  local_cumsum = cumsum(f_stacked)
  # Because the last layer in `local_cumsum` is the sum of all layers in the
  # current TPU replica, the following operation provides block-level integrals
  # across all replicas.
  replica_cumsum = global_reduce(
      tf.expand_dims(local_cumsum[plane_index(-1)], axis=axis), cumsum,
      group_assignment)
  cumsum_from_0 = tf.cond(
      pred=tf.equal(iloc, 0),
      true_fn=lambda: local_cumsum,
      false_fn=lambda: local_cumsum + tf.expand_dims(  # pylint: disable=g-long-lambda
          replica_cumsum[plane_index(iloc - 1)],
          axis=axis))
  cumsum_to_end = tf.expand_dims(
      replica_cumsum[plane_index(-1)], axis=axis) - cumsum_from_0

  # Subtract half of the sum of the starting and end points of the cumulative
  # sum to conform with the trapazoidal rule of integral.
  num_replicas = len(group_assignment[0])
  local_lim_low = tf.repeat(
      tf.expand_dims(f_stacked[plane_index(0)], 0), num_replicas, 0)
  local_lim_high = tf.repeat(
      tf.expand_dims(f_stacked[plane_index(-1)], 0), num_replicas, 0)

  global_lim_low = tf.raw_ops.AllToAll(
      input=local_lim_low,
      group_assignment=group_assignment,
      concat_dimension=0,
      split_dimension=0,
      split_count=num_replicas)[0, ...]
  global_lim_high = tf.raw_ops.AllToAll(
      input=local_lim_high,
      group_assignment=group_assignment,
      concat_dimension=0,
      split_dimension=0,
      split_count=num_replicas)[-1, ...]

  integral_from_0 = cumsum_from_0 - 0.5 * (
      tf.expand_dims(global_lim_low, axis=axis) + f_stacked)
  integral_to_end = cumsum_to_end + 0.5 * (
      f_stacked - tf.expand_dims(global_lim_high, axis=axis))
  # pylint: disable=g-long-ternary
  integrated = ((h * integral_from_0,
                 h * integral_to_end) if isinstance(f, tf.Tensor) else
                (tf.unstack(h * integral_from_0),
                 tf.unstack(h * integral_to_end)))
  return integrated


def strip_halos(
    f: FlowFieldVal,
    halos: Sequence[int],
) -> FlowFieldVal:
  """Removes the halos from the input field component.

  Args:
    f: A field component. This is expected to be expressed in the form of a list
      of 2D `tf.Tensor` representing x-y slices, where each list element
      represents the slice of at a given z coordinate in ascending z order, or a
      single 3D `tf.Tensor` with the shape [nz, nx, ny]. The halos of the field
      are included.
    halos: The width of the (symmetric) halos for each dimension: for example
      [1, 2, 3] means the halos for `f` have width of 1, 2, 3 on both sides in
      x, y, z dimension respectively. This is in the form of [halo_x, halo_y,
      halo_z]

  Returns:
    The inner part of the field component with the halo region removed.
    Represented as the same format as the input: a list of 2D `tf.Tensor` or a
    single 3D `tf.Tensor` with the halo region excluded.
  """
  # Handles the case of single 3D tensor.
  if isinstance(f, tf.Tensor):
    nz, nx, ny = f.get_shape().as_list()
    return f[halos[2]:nz - halos[2], halos[0]:nx - halos[0],
             halos[1]:ny - halos[1]]

  # Handles the case of a list of 2D tensor.
  nx = f[0].get_shape().as_list()[0]
  ny = f[0].get_shape().as_list()[1]
  nz = len(f)
  return [
      f[i][halos[0]:nx - halos[0], halos[1]:ny - halos[1]]
      for i in range(halos[2], nz - halos[2])
  ]


def get_field_inner(
    u: FlowFieldVal,
    v: FlowFieldVal,
    w: FlowFieldVal,
    halos: Sequence[int],
) -> tuple[FlowFieldVal, FlowFieldVal, FlowFieldVal]:
  """Validates and removes the halos of the input field components.

  All three components can be represented as a list of 2D `tf.Tensor` or a
  single 3D `tf.Tensor` but they must all be in the same format.

  Args:
    u: The first component of the field/variable on the local core. This is
      expected to be expressed in the form of a length `nz` list of 2D
      `tf.Tensor` with shape of [nx, ny] or a single 3D `tf.Tensor` with shape
      of [nz, nx, ny]. The halos of the field are included.
    v: The second component of the field/variable on the local core. This is
      expected to be expressed in the form of a length `nz` list of 2D
      `tf.Tensor` with shape of [nx, ny] or a single 3D `tf.Tensor` with shape
      of [nz, nx, ny]. The halos of the field are included.
    w: The third component of the field/variable on the local core. This is
      expected to be expressed in the form of a length `nz` list of 2D
      `tf.Tensor` with shape of [nx, ny] or a single 3D `tf.Tensor` with shape
      of [nz, nx, ny]. The halos of the field are included.
    halos: The width of the (symmetric) halos for each dimension in the form of
      [halo_x, halo_y, halo_z], for example:[1, 2, 3] means the halos for `u`,
        `v`, and `w` have width of 1, 2, 3 on both sides in x, y, z dimension
        respectively. The halo region of the field will be removed.

  Returns:
    A tuple with three components each representing the inner part of a field
    component with the halo region removed. Each component is represented in the
    same format as they are passed in, either as a list of 2D `tf.Tensor` or a
    single 3D `tf.Tensor`.
  """
  validate_fields(u, v, w)
  u_inner = strip_halos(u, halos)
  v_inner = strip_halos(v, halos)
  w_inner = strip_halos(w, halos)
  return u_inner, v_inner, w_inner


def cross_replica_gather(x: tf.Tensor, num_replicas: int) -> list[tf.Tensor]:
  """Cross-replica gather of tensors.

  Args:
    x: The tensor to send to all the replicas.
    num_replicas: The total number of replicas.

  Returns:
    A list of the `x` gathered from all replicas in order of `replica_id`.
  """
  enlarged_shape = [num_replicas] + x.shape.as_list()
  group_assignment = [list(range(num_replicas))]
  broadcasted_tensor = tf.broadcast_to(
      tf.expand_dims(x, 0), shape=enlarged_shape)
  gathered = tf.raw_ops.AllToAll(
      input=broadcasted_tensor,
      group_assignment=group_assignment,
      concat_dimension=0,
      split_dimension=0,
      split_count=num_replicas,
      name='CrossReplicaGather',
  )
  return [gathered[i, ...] for i in range(num_replicas)]


def gather(x: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
  """Retrieves values in `x` located at `indices`.

  Args:
    x: A 3D `tf.Tensor` from which values are gathered.
    indices: A 2D `tf.Tensor` with shape (n_pts, 3), with the columns being the
      indices associated with the first to last dimension of `x`. Note that
      repeated indices entries are allowed.

  Returns:
    Values in `x` at locations specified by indices as a vector of length
    matching the first dimension of `indices`.  If `indices` is empty, return
    a vector with length 0.
  """
  i, j, k = [
      tf.cast(tf.one_hot(indices[:, l], depth=tf.shape(x)[l]), dtype=x.dtype)
      for l in range(3)
  ]

  return tf.einsum('qi,qj,qk,ijk->q', i, j, k, x)


def gather_from_mask(x: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
  """Retrieves values in `x` corresponding to locations of 1s in `mask`.

  Args:
    x: A 3D `tf.Tensor` from which values are gathered.
    mask: A 3D `tf.Tensor` of ones and zeros, with ones indicating the location
      where the values are gathered. Note that the shape of `x` and `mask` must
      be the same.

  Returns:
    A 1D vector storing values corresponding to locations of ones in `mask`. Its
    length equals the number of ones in `mask`. Note that the indices of ones
    in `mask` follows the order of the dimensions from first to last. If no ones
    are found in `mask`, return a vector with length 0.

  Raises:
    ValueError If the shapes of `x` and `mask` are different.
  """
  return gather(x, tf.where(tf.less(tf.abs(mask - 1.0), 1e-6)))


def scatter(
    x: tf.Tensor, indices: tf.Tensor, shape: tf.Tensor, dtype: tf.DType
) -> tf.Tensor:
  """Generates a 3D tensor with values in `x` specified at `indices`.

  Args:
    x: A 1D `tf.Tensor` to be scattered into a 3D tensor.
    indices: A 2D `tf.Tensor` with shape (n_pts, 3), with the columns being the
      indices associated with the dimensions of the output 3D tensor. The first
      dimension of `x` and `indices` must be the same. Note that repeated
      entries of indices is allowed, in which case the sum of these values will
      be scattered.
    shape: A 1D array of length 3 specifying the shape of the 3D tensor in the
      output.
    dtype: The data type of the returned 3D tensor.

  Returns:
    A 3D tensor with values in `x` scattered to locations specified by `indices`
    , with everywhere else being 0. If `indices` is empty, a 3D tensor with all
    zeros will be returned.
  """
  if tf.shape(indices)[0] == 0:
    return tf.zeros(shape, dtype=dtype)

  i, j, k = [
      tf.cast(tf.one_hot(indices[:, l], depth=shape[l]), dtype=x.dtype)
      for l in range(3)
  ]

  return tf.cast(tf.einsum('q,qi,qj,qk->ijk', x, i, j, k), dtype)


def scatter_to_mask(
    x: tf.Tensor, mask: tf.Tensor, dtype: tf.DType
) -> tf.Tensor:
  """Generates a 3D tensor with `x` scattered to locations of 1 in `mask`.

  Args:
    x: A 1D `tf.Tensor` to be scattered into a 3D tensor. The order of these
      points must follow the row-major order of the indices of ones in `mask`.
    mask: A 3D `tf.Tensor` of ones and zeros, with ones indicating the location
      where the values are scattered.
    dtype: The data type of the returned 3D tensor.

  Returns:
    A 3D tensor with values in `x` scattered to locations specified by ones in
    `mask`, with everywhere else being 0. If `mask` has all zeros, a 3D tensor
    with all zeros will be returned.
  """
  return scatter(
      x, tf.where(tf.less(tf.abs(mask - 1.0), 1e-6)), tf.shape(mask), dtype
  )


def pad(
    f: FlowFieldVal,
    paddings: Sequence[Sequence[int]],
    value: float = 0.0,
) -> FlowFieldVal:
  """Pads the input field with a given value.

  Args:
    f: A field component. This is expected to be expressed in the form of a
      length `nz` list of 2D `tf.Tensor` with shape of [nx, ny] or a single 3D
      `tf.Tensor` with shape of [nz, nx, ny].
    paddings: The padding lengths for each dimension in the format: [[pad_x_low,
      pad_x_hi], [pad_y_low, pad_y_hi], [pad_z_low, pad_z_hi]]. For instance,
      ((0, 0), (2, 0), (0, 3)) means f will be padded with a width of 2 on the
      lower face of the y dimension and with a width of 3 on the upper face of
      the z dimension.
    value: The constant value to be used for padding.

  Returns:
    The padded input field as a list of 2D tensors.
  """
  if isinstance(f, tf.Tensor):
    rotated_paddings = [paddings[2], paddings[0], paddings[1]]
    return tf.pad(f, rotated_paddings, constant_values=value)

  padded = [tf.pad(f_i, paddings[0:2], constant_values=value) for f_i in f]
  lower_pad = [value * tf.ones_like(padded[0])
              ] * paddings[2][0] if paddings[2][0] > 0 else []
  upper_pad = [value * tf.ones_like(padded[0])
              ] * paddings[2][1] if paddings[2][1] > 0 else []
  return lower_pad + list(padded) + upper_pad


def slice_field(
    f: FlowFieldVal,
    dim: int,
    start_idx: int,
    size: Optional[int] = None,
) -> FlowFieldVal:
  """Slices the input field along the given dimension.

  Args:
    f: A list of 2D `tf.Tensor` or a single 3D `tf.Tensor` representing a 3D
      field. If a list of 2D `tf.Tensor`, the length of the list is `nz` and
      each 2D `tf.Tensor` has the shape [nx, ny]. If a single 3D `tf.Tensor`,
      its shape is [nz, nx, ny].
    dim: The dimension of the plane to slice, should be one of 0, 1, and 2.
    start_idx: The index of the first point in the slice. If negative, it will
      be counted from the end of the field along `dim`.
    size: The optional length of the slice along `dim`. If not provided, the
      slice will run from `start_idx` to the end of the array.

  Returns:
    A slice having the same format as the input field but a dimension of `size`
    along `dim`.
  """
  shape = list(get_shape(f))
  start = [0, 0, 0]

  n = int(shape[dim])
  if start_idx < 0:
    start_idx = n + start_idx

  if size is None:
    size = n - start_idx

  start[dim] = start_idx
  shape[dim] = size

  if isinstance(f, tf.Tensor):
    # Handles the case of single 3D tensor.
    shape = tf.roll(shape, shift=1, axis=0)
    start = tf.roll(start, shift=1, axis=0)
    return tf.slice(f, start, shape)

  # Handles the case of list of 2D tensors.
  if dim in (0, 1):
    return tf.nest.map_structure(
        lambda x: tf.slice(x, start[:-1], shape[:-1]), f
    )
  elif dim == 2:  # Z
    return f[start_idx : start_idx + size]
  else:
    raise ValueError(
        f'`dim` has to be one of 0, 1, or 2. But {dim} is provided.'
    )


def get_face(
    value: FlowFieldVal,
    dim: int,
    face: int,
    index: int,
    scaling_factor: float = 1.0,
) -> FlowFieldVal:
  """Gets the face in `value` that is `index` number of points from boundary.

  This function extracts the requested plane from a 3D tensor.
  * If the 3D tensor is represented as a `Sequence[tf.Tensor]`. The returned
    value depends on `dim`:
    if `dim` is 0 (x plane): returns a list with length nz, each tensor in the
      list has size 1 x ny;
    if `dim` is 1 (y plane): returns a list with length nz, each tensor in the
      list has size nx x 1;
    if `dim` is 2 (z plane): returns a list with length 1, each tensor in the
      list has size nx x ny.
  * If the 3D tensor is a 3D `tf.Tensor`, the returned value is a one-element
    list [f], where f is a 3D `tf.Tensor` whose shape depends on `dim`:
      if `dim` is 0 (x plane): f has shape (nz, 1, ny);
      if `dim` is 1 (y plane): f has shape (nz, nx, 1);
      if `dim` is 2 (z plane): f has shape (1, nx, ny).

  Args:
    value: A list of 2D `tf.Tensor` or a single 3D `tf.Tensor` representing a 3D
      field. If a list of 2D `tf.Tensor`, the length of the list is `nz` and
      each 2D `tf.Tensor` has the shape [nx, ny]. If a single 3D `tf.Tensor`,
      its shape is [nz, nx, ny].
    dim: The dimension of the plane to slice, should be one of 0, 1, and 2.
    face: The face of the plane to slice, with 0 representing the lower face,
      and 1 representing the higher face.
    index: The number of points that is away from the boundary determined by
      `dim` and `face`.
    scaling_factor: A factor that is used to scale the target slice.

  Returns:
    If face is 0, then the (index + 1)'th plane is returned; if face is 1, then
    the length - index'th plane is returned. The returned slice will be
    multiplied by `scaling_factor`. If `value` is a 3D `tf.Tensor`, the returned
    value is a `list` with a single element consisting of a rank-3 tensor, where
    one of the shape dimensions is 1.
  """
  if face not in (0, 1):
    raise ValueError(
        f'`face` has to be one of 0 or 1. But {face} is provided.'
    )
  index = index if face == 0 else -index - 1
  bc_value = [
      tf.nest.map_structure(
          lambda x: scaling_factor * x, slice_field(value, dim, index, size=1)
      )
  ]
  if isinstance(value, Sequence) and dim == 2:
    bc_value = bc_value[0]

  return bc_value  # pytype: disable=bad-return-type


def meshgrid(xs: _TensorEquivalent, ys: _TensorEquivalent,
             zs: _TensorEquivalent) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Generates 3D meshgrid given grid points in xs, ys, zs.

  The meshgrid provided in Tensorflow has a large memory usage. We implement
  this version which has a smaller memory footprint. This is very useful for
  distributed grid initialization.

  Args:
    xs: A 1-D Tensor (or convertible to a 1-D tensor) representing x dimension
      grid points. It's the user's responsibility to ensure the values are
      sensible (ordered, correct spacing, etc.).
    ys: A 1-D Tensor (or convertible to a 1-D tensor) representing y dimension
      grid points. It's the user's responsibility to ensure the values are
      sensible (ordered, correct spacing, etc.).
    zs: A 1-D Tensor (or convertible to a 1-D tensor) representing z dimension
      grid points. It's the user's responsibility to ensure the values are
      sensible (ordered, correct spacing, etc.).

  Returns:
    A tuple of 3-D tensors (xx, yy, zz) representing the meshgrid corresponding
    to xs, ys, zs. The indexing follows the 'ij' convention.

  Raises:
    ValueError: If the input arguments are incorrect.
  """
  xs_ts = tf.cast(tf.convert_to_tensor(xs), _DTYPE)
  ys_ts = tf.cast(tf.convert_to_tensor(ys), _DTYPE)
  zs_ts = tf.cast(tf.convert_to_tensor(zs), _DTYPE)

  if xs_ts.shape.rank != 1 or ys_ts.shape.rank != 1 or zs_ts.shape.rank != 1:
    raise ValueError('Rank for inputs xs, ys, and zs must all be 1, but got '
                     '(%d, %d, %d) instead.' %
                     (xs_ts.shape.rank, ys_ts.shape.rank, zs_ts.shape.rank))

  xx = tf.einsum('ij,k->ijk', tf.einsum('i,j->ij', xs_ts, tf.ones_like(ys_ts)),
                 tf.ones_like(zs_ts))
  yy = tf.einsum('ij,k->ijk', tf.einsum('i,j->ij', tf.ones_like(xs_ts), ys_ts),
                 tf.ones_like(zs_ts))
  zz = tf.einsum('ij,k->ijk',
                 tf.einsum('i,j->ij', tf.ones_like(xs_ts), tf.ones_like(ys_ts)),
                 zs_ts)

  return xx, yy, zz


def _get_broadcast_z_size(tensors: Sequence[Sequence[tf.Tensor]]) -> int:
  """Returns the maximum length of the input tensors in the z dimension.

  Args:
    tensors: Tensors whose size in the z dimension is 1 or nz for some
      nz. If nz cannot be determined (i.e., there are non-length-1 tensors
      whose sizes are different), there will be an assertion error.

  Returns:
    The maximum size of the input tensors in the z dimension.
  """
  max_z = 1
  for tensor in tensors:
    assert isinstance(tensor, Sequence), (
        f'Expecting tensor to be of type Sequence, got {type(tensor)}.')
    if len(tensor) > 1:
      if max_z == 1:
        max_z = len(tensor)
      else:
        assert len(tensor) == max_z, (
            f'Got different lengths of tensors: {len(tensor)} and {max_z}')
  return max_z


def _broadcast_to_z_size(
    tensor: Sequence[tf.Tensor], z_size: int) -> Sequence[tf.Tensor]:
  """Broadcasts the input tensor to `z_size` if possible."""
  if len(tensor) != z_size:
    assert len(tensor) == 1, (
        f'Expecting length along z to be 1, but got {len(tensor)}.')
    return [tensor[0]] * z_size
  else:
    return tensor


def _is_sequence_of_list_of_2d_tensors(tensors: Sequence[FlowFieldVal]) -> bool:
  """Returns true if all input tensors are list-of-2D tensors."""
  all_3d = all(isinstance(tensor, tf.Tensor) for tensor in tensors)
  all_non_3d = all(not isinstance(tensor, tf.Tensor) for tensor in tensors)
  if all_3d:
    return False
  if all_non_3d:
    return True
  raise ValueError('Expecting tensors to be all 3D or all list-of-2D. Got '
                   f'{[type(tensor) for tensor in tensors]}.')


def map_structure_3d(f, *args):
  """Like tf.nest.map_structure but handles Swirl-LM list-of-2D tensors.

  Unlike tf.nest.map_structure, this function will attempt to broadcast along
  the z-dimension (i.e., the list dimension) for list-of-2D tensors.

  `args` can also be of plain tensor type, in which case they will be directly
  passed to `f`.

  Args:
    f: The function to apply to `args`.
    *args: List of arguments to pass to `f`.

  Returns:
    The result of applying `f` to `args`.
  """
  if not _is_sequence_of_list_of_2d_tensors(args):
    return f(*args)
  z_size = _get_broadcast_z_size(args)
  args = [_broadcast_to_z_size(arg, z_size) for arg in args]
  return tf.nest.map_structure(f, *args)
