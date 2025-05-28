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
from typing import TypeAlias

import jax
import jax.numpy as jnp


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


FaceBoundaryCondition: TypeAlias = tuple[BCType, list[jax.Array] | float]
DimBoundaryConditions: TypeAlias = tuple[
    FaceBoundaryCondition | None, FaceBoundaryCondition | None
]
BoundaryConditionsSpec: TypeAlias = tuple[DimBoundaryConditions | None, ...]


class SideType(enum.Enum):
  """A class defining the type of axis side."""

  NONE = -1  # Not a side.
  LOW = 0  # The low side of an axis.
  HIGH = 1  # The high side of an axis.


def pad_in_dim(
    x: jax.Array, low_pad: int, high_pad: int, value: float, dim: int
) -> jax.Array:
  """Pads a JAX array in a single dimension along `axis`.

  Args:
    x: The JAX array to pad.
    low_pad: The amount of padding to add to the low side of the dimension.
    high_pad: The amount of padding to add to the high side of the dimension.
    value: The value to use for padding.
    dim: The dimension to pad.

  Returns:
    The padded JAX array.
  """
  padding = [(0, 0)] * x.ndim
  padding[dim] = (low_pad, high_pad)
  return jnp.pad(x, padding, mode="constant", constant_values=value)


def apply_one_core_boundary_conditions(
    x: jax.Array,
    boundary_conditions: BoundaryConditionsSpec | None = None,
) -> jax.Array:
  """Applies Neumann or Dirichlet boundary conditions to a tensor or array.

  The value in each boundary condition has to be broadcastable to the
  hyperplane it applies to.

  Args:
    x: The tensor or array to apply boundary conditions to.
    boundary_conditions: The boundary conditions to apply. If not supplied, `x`
      is returned unchanged.

  Returns:
    `x` with the boundary conditions applied.
  """
  if boundary_conditions is None:
    return x

  rank = x.ndim

  for axis, bc_per_axis in enumerate(boundary_conditions[:rank]):
    if bc_per_axis is None:
      continue
    for side, bc_per_side in enumerate(bc_per_axis):
      if bc_per_side is None:
        continue
      bc_type, bc_value = bc_per_side
      if bc_type == BCType.NO_TOUCH:
        continue
      bulk_slice = [slice(None)] * rank
      bulk_slice[axis] = (slice(1, None), slice(0, -1))[side]
      x_shaved = x[tuple(bulk_slice)]
      shape_with_axis_1d = list(x.shape)
      shape_with_axis_1d[axis] = 1
      sign = (-1, 1)[side]

      if bc_type == BCType.DIRICHLET:
        concat_plane = jnp.broadcast_to(bc_value, shape_with_axis_1d)
      elif bc_type == BCType.NEUMANN:
        neumann_slice = [slice(None)] * rank
        neumann_slice[axis] = (slice(0, 1), slice(-1, None))[side]
        neumann_plane = x_shaved[tuple(neumann_slice)]
        concat_plane = neumann_plane + sign * jnp.broadcast_to(
            bc_value, shape_with_axis_1d
        )
      else:
        raise NotImplementedError(
            "Only DIRICHLET, NEUMANN and NO_TOUCH "
            "boundary condition types are supported."
        )

      x = jnp.concatenate((x_shaved, concat_plane)[::sign], axis)

  return x
