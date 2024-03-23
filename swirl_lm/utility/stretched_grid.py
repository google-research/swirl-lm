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

"""Utilities for working with stretched grid."""

from collections.abc import Sequence
from typing import TypeAlias
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldMap: TypeAlias = types.FlowFieldMap
STRETCHED_GRID_KEY_PREFIX = 'stretched_grid'


def reshape_to_broadcastable(
    f_1d: tf.Tensor, dim: int, use_3d_tf_tensor: bool
) -> tf.Tensor | Sequence[tf.Tensor]:
  """Reshapes a rank-1 tensor to a form broadcastable against 3D fields.

  Here, `dim` is 0, 1, or 2, corresponding to dimension x, y, or z respectively.
  The rank-1 tensor `f_1d` will be reshaped such that it represents a 3D field
  whose values vary only along dimension `dim`. However, for memory efficiency,
  the number of elements do not change. The output can be used in operations
  with 3D fields in the most natural way possible, with broadcasting occurring.

  3D fields are stored with order (z, x, y). If `use_3d_tf_tensor == False`,
  then 3D fields are expected to be stored as lists of 2D tensors, and the
  output of this function changes accordingly.

  The number of elements of `f_1d` must be correct on input (this is NOT
  checked). That is, if `dim`==0, 1, or 2, then len(f_1d) must equal nx, ny, or
  nz, respectively, where `nx`, `ny`, `nz` are the corresponding sizes of 3D
  fields.

  Examples of how the output can be used with 3D fields via broadcasting:
    Suppose here that `q` is a 3D field (a 3D tensor or list of 2D tensors)

  If `use_3d_tf_tensor==True`:
      fx = reshape_to_broadcastable(fx_1d, 0, True)  # fx has shape (1, nx, 1)
      fy = reshape_to_broadcastable(fy_1d, 1, True)  # fy has shape (1, 1, ny)
      fz = reshape_to_broadcastable(fz_1d, 2, True)  # fz has shape (nz, 1, 1)
      q * fx * fy * fz  # This is a valid operation.

  If `use_3d_tf_tensor==False`, then for dim==0 or dim==1:
      fx = reshape_to_broadcastable(fx_1d, 0, False)  # fx has shape (nx, 1)
      fy = reshape_to_broadcastable(fy_1d, 1, False)  # fy has shape (1, ny)
      tf.nest.map_structure(lambda q: q * fx * fy, q)  # Valid operation.

  Arrays along z need a slightly different handling for
  `use_3d_tf_tensor==False`. For dim==2:
      fz = reshape_to_broadcastable(fz_1d, 2, False)  # fz is a list
      tf.nest.map_structure(tf.multiply, q, fz)  # Valid operation.

  Args:
    f_1d: A rank-1 tensor.
    dim: The dimension of variation of the input tensor `f_1d`.
    use_3d_tf_tensor: Whether 3D fields are represented by 3D tensors or lists
      of 2D tensors.

  Returns:
    A tensor (or list of tensors, in the case `dim==2` and
    `use_3d_tf_tensor==False`) that can be broadcast against a 3D field.
  """
  if dim == 0:
    if not use_3d_tf_tensor:
      # When using a list of 2D tensors, set shape of tensor to (nx, 1).
      return f_1d[:, tf.newaxis]
    else:
      # When using a 3D tensor, set shape of tensor to (1, nx, 1).
      return f_1d[tf.newaxis, :, tf.newaxis]
  elif dim == 1:
    if not use_3d_tf_tensor:
      # When using a list of 2D tensors, set shape of tensor to (1, ny).
      return f_1d[tf.newaxis, :]
    else:
      # When using a 3D tensor, set shape of tensor to (1, 1, ny).
      return f_1d[tf.newaxis, tf.newaxis, :]
  elif dim == 2:
    if not use_3d_tf_tensor:
      # When using a list of 2D tensors, set field to be a list (of length nz)
      # of rank-0 tensors.
      return tf.unstack(f_1d)
    else:
      # When using a 3D tensor, set shape of tensor to (nz, 1, 1).
      return f_1d[:, tf.newaxis, tf.newaxis]
  else:
    raise ValueError(f'Unsupported dim: {dim}. `dim` must be 0, 1, or 2.')


def get_helper_variables(
    additional_states: FlowFieldMap,
) -> FlowFieldMap:
  """Returns a dictionary with just the stretched grid helper variables."""
  return {
      key: additional_states[key]
      for key in additional_states
      if key.startswith(STRETCHED_GRID_KEY_PREFIX)
  }
