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

"""Utility library for performing Halo exchanges."""

import enum
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf



class BCType(enum.Enum):
  """A class defining the type of boundary conditions.

  NEUMANN: The Neumann boundary condition.
  DIRICHLET: The Dirichlet boundary condition.
  NO_TOUCH: Preserves the boundary at its current value. This condition could
    be used, for example, if boundary computation happens outside of the TPU
    computation loop on CPU. `NO_TOUCH` is useful in cases where the grid is
    staggered. Namely, if certain state variables are defined and computed only
    for interior grid points, their extremal values are already correct and do
    not need to be re-calculated/have a boundary condition imposed outside of
    the computation loop. For cases with width greater than 1, we hold all
    boundary planes constant for consistency, but such scenarios should not
    rise in practice.
  ADDITIVE: Like NEUMANN, but adds the supplied boundary values to the boundary
    plane itself, as opposed to the plane +/- 1. Helpful as in the
    case of NO_TOUCH when boundary computation happens outside of the TPU
    computation loop, but results must be imposed within the TPU loop.

  """
  NEUMANN = 1  # Neumann boundary condition.
  NEUMANN_2 = 5  # Neumann boundary condition estimated with 2nd order scheme.
  DIRICHLET = 2  # Dirichlet boundary condition.
  NO_TOUCH = 3  # Preserves the boundary at its current value.
  ADDITIVE = 4  # Adds the given values at the boundary.
  NONREFLECTING = 6  # Nonreflecting boundary condition.

_TensorOrArray = Union[tf.Tensor, np.ndarray]
_FaceBoundaryCondition = Tuple[BCType, Union[Sequence[
    Union[_TensorOrArray, Sequence[_TensorOrArray]]], float]]
_DimBoundaryConditions = Sequence[Optional[_FaceBoundaryCondition]]
BoundaryConditionsSpec = Sequence[Optional[_DimBoundaryConditions]]


class SideType(enum.Enum):
  """A class defining the type of axis side."""
  NONE = -1  # Not a side.
  LOW = 0  # The low side of an axis.
  HIGH = 1  # The high side of an axis.

homogeneous_bcs = lambda rank=3: [[(BCType.DIRICHLET, 0.)] * 2] * rank


def pad_in_dim(x: np.ndarray,
               low_pad: int,
               high_pad: int,
               value: float,
               axis: int) -> np.ndarray:
  padding = [(0, 0)] * x.ndim
  padding[axis] = (low_pad, high_pad)
  return np.pad(x, padding, mode="constant", constant_values=value)


def slice_in_dim(x: np.ndarray,
                 start: int,
                 end: Optional[int], axis: int) -> np.ndarray:
  slices = [slice(None)] * x.ndim
  slices[axis] = slice(start, end)
  return x[tuple(slices)]


def is_first_replica(replica_id: tf.Tensor,
                     replicas: np.ndarray,
                     replica_dim: int) -> tf.Tensor:
  first_replicas = slice_in_dim(replicas, start=0, end=1, axis=replica_dim)
  return tf.reduce_any(tf.equal(replica_id, first_replicas))


def is_last_replica(replica_id: tf.Tensor,
                    replicas: np.ndarray,
                    replica_dim: int) -> tf.Tensor:
  last_replicas = slice_in_dim(
      replicas, start=-1, end=None, axis=replica_dim)
  return tf.reduce_any(tf.equal(replica_id, last_replicas))


def apply_one_core_boundary_conditions_to_tensor_or_array(
    x: Union[tf.Tensor, np.ndarray],
    boundary_conditions: Optional[BoundaryConditionsSpec]
) -> Union[tf.Tensor, np.ndarray]:
  """Applies Neumann or Dirichlet boundary conditions to a tensor or array.

  The value in each boundary condition has to be broadcastable to the
  hyperplane it applies to.

  Args:
    x: The tensor or array to apply boundary conditions to.
    boundary_conditions: The boundary conditions to apply. If not supplied,
      `x` is returned unchanged.

  Returns:
    `x` with the boundary conditions applied.
  """
  if boundary_conditions is None:
    return x

  use_np = isinstance(x, np.ndarray)
  shape = x.shape if use_np else x.shape.as_list()
  rank = len(shape)

  concat, broadcast_to = ((np.concatenate, np.broadcast_to) if use_np
                          else (tf.concat, tf.broadcast_to))
  cast = lambda y: y.astype(x.dtype) if use_np else tf.cast(y, x.dtype)

  for axis, bc_per_axis in enumerate(boundary_conditions[:rank]):
    if bc_per_axis is None: continue
    for side, bc_per_side in enumerate(bc_per_axis):
      if bc_per_side is None: continue
      bc_type, bc_value = bc_per_side
      if bc_type == BCType.NO_TOUCH: continue
      bulk_slice = [slice(None)] * rank
      bulk_slice[axis] = (slice(1, None), slice(0, -1))[side]
      x_shaved = x[tuple(bulk_slice)]
      shape_with_axis_1d = list(shape)
      shape_with_axis_1d[axis] = 1
      sign = (-1, 1)[side]

      if bc_type == BCType.DIRICHLET:
        concat_plane = cast(broadcast_to(bc_value, shape_with_axis_1d))
      elif bc_type == BCType.NEUMANN:
        neumann_slice = [slice(None)] * rank
        neumann_slice[axis] = (slice(0, 1), slice(-1, None))[side]
        neumann_plane = x_shaved[tuple(neumann_slice)]
        concat_plane = (neumann_plane +
                        sign * cast(broadcast_to(bc_value, shape_with_axis_1d)))
      else:
        raise NotImplementedError("Only DIRICHLET, NEUMANN and NO_TOUCH "
                                  "boundary condition types are supported.")

      x = concat((x_shaved, concat_plane)[::sign], axis)

  return x


def apply_one_core_boundary_conditions(
    x: List[Union[tf.Tensor, np.ndarray]],
    boundary_conditions: Optional[BoundaryConditionsSpec]
) -> List[Union[tf.Tensor, np.ndarray]]:
  """Applies Neumann or Dirichlet boundary conditions to a field.

  The value in each boundary condition has to be broadcastable to the
  hyperplane it applies to.

  Args:
    x: The 3D field to apply boundary conditions to, given as a list of 2D
      arrays or tensors.
    boundary_conditions: The boundary conditions to apply. If not supplied,
      `x` is returned unchanged.

  Returns:
    `x` with the boundary conditions applied.
  """
  if boundary_conditions is None:
    return x

  use_np = isinstance(x[0], np.ndarray)
  shape_xy = x[0].shape if use_np else x[0].shape.as_list()

  concat, broadcast_to = ((np.concatenate, np.broadcast_to) if use_np
                          else (tf.concat, tf.broadcast_to))
  cast = lambda y: y.astype(x[0].dtype) if use_np else tf.cast(y, x[0].dtype)

  for axis, bc_per_axis in enumerate(boundary_conditions):
    if bc_per_axis is None: continue
    for side, bc_per_side in enumerate(bc_per_axis):
      if bc_per_side is None: continue
      bc_type, bc_value = bc_per_side
      if bc_type == BCType.NO_TOUCH: continue
      if axis < 2:
        bulk_slice = [slice(None), slice(None)]
        bulk_slice[axis] = (slice(1, None), slice(0, -1))[side]
        x_shaved = [x_2d[tuple(bulk_slice)] for x_2d in x]
        shape_2d_with_axis_1d = list(shape_xy)
        shape_2d_with_axis_1d[axis] = 1
        sign = (-1, 1)[side]

        if bc_type == BCType.DIRICHLET:
          concat_planes = [
              cast(broadcast_to(bc_value, shape_2d_with_axis_1d))] * len(x)
        elif bc_type == BCType.NEUMANN:
          neumann_slice = [slice(None)] * 2
          neumann_slice[axis] = (slice(0, 1), slice(-1, None))[side]
          neumann_planes = [x_2d[tuple(neumann_slice)] for x_2d in x_shaved]
          concat_planes = [
              plane + sign * cast(broadcast_to(bc_value, shape_2d_with_axis_1d))
              for plane in neumann_planes]
        else:
          raise NotImplementedError("Only DIRICHLET, NEUMANN and NO_TOUCH "
                                    "boundary condition types are supported.")

        x = [concat((x_shaved_2d, concat_plane)[::sign], axis)
             for x_shaved_2d, concat_plane in zip(x_shaved, concat_planes)]
      else:  # axis == 2.
        replacement_index = 0 if side == 0 else -1

        if bc_type == BCType.DIRICHLET:
          replacement_plane = cast(broadcast_to(bc_value, shape_xy))
        elif bc_type == BCType.NEUMANN:
          neumann_plane = x[1] if side == 0 else x[-2]
          sign = (-1, 1)[side]
          replacement_plane = (
              neumann_plane + cast(sign * broadcast_to(bc_value, shape_xy)))
        x[replacement_index] = replacement_plane

  return x
