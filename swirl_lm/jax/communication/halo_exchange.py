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
"""Helper library for performing Halo exchanges."""

from typing import Literal

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from swirl_lm.jax.communication import halo_exchange_utils
from swirl_lm.jax.utility import common_ops
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types

FaceBoundaryCondition = halo_exchange_utils.FaceBoundaryCondition
BCType = halo_exchange_utils.BCType
BoundaryConditionsSpec = halo_exchange_utils.BoundaryConditionsSpec
ScalarField = types.ScalarField
SideType = halo_exchange_utils.SideType


def _get_homogeneous_neumann_bc_order2(
    array: ScalarField,
    axis: str,
    face: Literal[0, 1],
    plane: int,
    grid_params: grid_parametrization.GridParametrization,
) -> ScalarField:
  """Gets the 2D plane for homogeneous Neumann BC with second order difference.

  Args:
    array: A 3D tensor to which the homogeneous Neumann BC is applied.
    axis: The axis of the plane to slice.
    face: A binary indicator for the location of the boundary, with 0 and 1
      being the lower and higher ends of the domain, respectively.
    plane: The index of the plane to apply the BC. If `face` is 0, it counts
      from the lower end of the domain; if `face` is 1, it counts from the
      higher end of the domain.
    grid_params: The grid parametrization object.

  Returns:
    A 2D plane that corresponds to the homogeneous Neumann BC that is computed
    with the second-order one-sided finite difference scheme.
  """
  # Define the function that computes the value of plane so that the gradient at
  # this plane is 0.
  f_near = common_ops.get_face(array, axis, face, plane + 1, grid_params)
  f_far = common_ops.get_face(array, axis, face, plane + 2, grid_params)
  return 4.0 / 3.0 * f_near - f_far / 3.0


def _get_homogeneous_neumann_bc_order1(
    array: ScalarField,
    axis: str,
    face: Literal[0, 1],
    plane: int,
    grid_params: grid_parametrization.GridParametrization,
) -> ScalarField:
  """Gets the 2D plane for homogeneous Neumann BC with first order difference.

  Args:
    array: A 3D tensor to which the homogeneous Neumann BC is applied.
    axis: The axis of the plane to slice.
    face: A binary indicator for the location of the boundary, with 0 and 1
      being the lower and higher ends of the domain, respectively.
    plane: The index of the plane to apply the BC. If `face` is 0, it counts
      from the lower end of the domain; if `face` is 1, it counts from the
      higher end of the domain.
    grid_params: The grid parametrization object.

  Returns:
    A 2D plane that corresponds to the homogeneous Neumann BC that is computed
    with the first-order finite difference scheme.
  """
  return common_ops.get_face(array, axis, face, plane + 1, grid_params)


def _get_additive_bc(
    array: ScalarField,
    axis: str,
    face: Literal[0, 1],
    plane: int,
    grid_params: grid_parametrization.GridParametrization,
) -> ScalarField:
  """Gets the 2D plane for the additive BC.

  Args:
    array: A 3D tensor to which the homogeneous Neumann BC is applied.
    axis: The axis of the plane to slice.
    face: A binary indicator for the location of the boundary, with 0 and 1
      being the lower and higher ends of the domain, respectively.
    plane: The index of the plane to apply the BC. If `face` is 0, it counts
      from the lower end of the domain; if `face` is 1, it counts from the
      higher end of the domain.
    grid_params: The grid parametrization object.

  Returns:
    A 2D plane for the additive boundary condition.
  """
  return common_ops.get_face(array, axis, face, plane, grid_params)


def _do_exchange(
    mesh: Mesh,
    axis: str,
    low_halo_from_self: jax.Array,
    high_halo_from_self: jax.Array,
    periodic: bool,
) -> tuple[jax.Array, jax.Array]:
  """Performs halo exchange between devices along a single axis.

  Args:
    mesh: A jax Mesh object representing the device topology.
    axis: The axis along which to perform halo exchange.
    low_halo_from_self: The low halo values from the current device.
    high_halo_from_self: The high halo values from the current device.
    periodic: Whether the dimension is periodic.

  Returns:
    A tuple containing the low and high halo values for the current device.
  """
  num_cores = common_ops.get_device_count_along_axes(mesh, axis)
  # Special case for when `num_cores` is 1.
  # Note although in the periodic case the normal flow handles it correctly,
  # this carve-out handles it more efficiently.
  if num_cores == 1:
    if periodic:
      return high_halo_from_self, low_halo_from_self
    else:
      return (jnp.zeros_like(high_halo_from_self),) * 2

  high_halo_for_self = jax.lax.ppermute(
      low_halo_from_self,
      axis,
      perm=[(j, (j - 1) % num_cores) for j in range(num_cores)],
  )
  low_halo_for_self = jax.lax.ppermute(
      high_halo_from_self,
      axis,
      perm=[(j, (j + 1) % num_cores) for j in range(num_cores)],
  )
  return low_halo_for_self, high_halo_for_self


def _get_halo_from_bc(
    array: ScalarField,
    plane: int,
    bc: tuple[BCType, jax.Array | float],
    axis: str,
    side: SideType,
    grid_params: grid_parametrization.GridParametrization,
) -> jax.Array | float:
  """Returns a halo plane derived from the boundary condition.

  This should only be called if all the following are true:
    * the grid is not periodic
    * `bc` is not `None`
    * the replica is at the end of the dimension on the specified side in the
      computational shape

  Args:
    array: A 3D array representing the field.
    plane: The index of the plane to apply the BC. If `side` is `LOW`, it counts
      from the lower end of the domain; if `side` is `HIGH`, it counts from the
      higher end of the domain.
    bc: The boundary conditions specification of the form [type, value]. See
      `inplace_halo_exchange` for full details about boundary condition
      specifications.
    axis: The axis of the plane to get the boundary condition for.
    side: A SideType indicating whether the calculation is for the low or high
      side of the axis. Only relevant in the case of Neumann boundary
      conditions.
    grid_params: The grid parametrization object.

  Returns:
    The border which is derived from the provided boundary conditions.

  Raises:
    ValueError if parameters have incorrect values.
  """
  bc_type, bc_value = bc
  if isinstance(bc_value, jax.Array):
    bc_value = jnp.squeeze(bc_value)

  def neumann_value():
    buf = _get_homogeneous_neumann_bc_order1(
        array, axis, side.value, plane, grid_params
    )
    sign = -1.0 if side == SideType.LOW else 1.0
    return sign * bc_value + buf

  def neumann_value_order2():
    buf = _get_homogeneous_neumann_bc_order2(
        array, axis, side.value, plane, grid_params
    )
    sign = -1.0 if side == SideType.LOW else 1.0
    return sign * bc_value + buf

  def dirichlet_value():
    return bc_value

  def additive_value():
    buf = _get_additive_bc(array, axis, side.value, plane, grid_params)
    return bc_value + buf

  def no_touch_value():
    return _get_additive_bc(array, axis, side.value, plane, grid_params)

  if bc_type == BCType.NEUMANN:
    return neumann_value()
  elif bc_type == BCType.NEUMANN_2:
    return neumann_value_order2()
  elif bc_type in (BCType.DIRICHLET, BCType.NONREFLECTING):
    return dirichlet_value()
  elif bc_type == BCType.ADDITIVE:
    return additive_value()
  elif bc_type == BCType.NO_TOUCH:
    return no_touch_value()
  else:
    raise ValueError("Unknown boundary condition type: {}.".format(bc_type))


def _inplace_halo_exchange_1d(
    array: ScalarField,
    axis: str,
    mesh: Mesh,
    periodic: bool,
    bc_low: tuple[BCType, jax.Array | float] | None,
    bc_high: tuple[BCType, jax.Array | float] | None,
    halo_width: int,
    plane: int,
    grid_params: grid_parametrization.GridParametrization,
) -> ScalarField:
  """Performs halo exchange and assigns values to points in a boundary plane.

  This function exchanges and sets a single plane in the boundary or halo
  region. It needs to be called for each plane in the boundary or halo region,
  in order, from the innermost to outermost in order for Neumann boundary
  conditions to be correctly applied.

  Args:
    array: A 3D array representing the field.
    axis: Axis along which the halo exchange is to be performed.
    mesh: A jax Mesh object representing the device topology.
    periodic: Indicates whether the given dimension should be treated
      periodically.
    bc_low: The boundary condition for the low side of the axis. This is either
      `None` or of the form `(bc_type, bc_value)` where `bc_value` represents a
      single 2D plane and is either a 2D tensor or a float. See
      `inplace_halo_exchange` for more details about boundary condition
      specifications.
    bc_high: The boundary condition for the high side of the axis. See `bc_low`.
    halo_width: The halo width.
    plane: Which plane to exchange. This is the index relative to a set of
      low-side boundary planes. (The corresponding index for a set of high-side
      boundary planes is calculated.) This function must be called in order from
      `plane = width - 1` to `plane = 0` for Neumann boundary conditions to be
      correctly applied.
    grid_params: The grid parametrization object.

  Returns:
    `array` with the halo exchange and boundary conditions applied.

  Raises:
    ValueError if parameters are incorrect.
  """
  is_first = jax.lax.axis_index(axis) == 0
  is_last = (
      jax.lax.axis_index(axis)
      == common_ops.get_device_count_along_axes(mesh, axis) - 1
  )
  if not periodic:
    bc_low = bc_low if bc_low is not None else (BCType.DIRICHLET, 0.0)
    bc_high = bc_high if bc_high is not None else (BCType.DIRICHLET, 0.0)

  def maybe_replace_halo_from_boundary_conditions(side):
    """Maybe return 2D plane from boundary conditions rather than neighbor."""
    if side == SideType.LOW:
      is_first_or_last = is_first
      bc = bc_low
    elif side == SideType.HIGH:
      is_first_or_last = is_last
      bc = bc_high
    else:
      raise ValueError(f"Unsupported side type: {side.name}.")

    if periodic or not bc:
      return halo_from_neighbor[side.value]
    else:
      halo_from_bc = _get_halo_from_bc(
          array, plane, bc, axis, side, grid_params
      )
      if isinstance(halo_from_bc, float):
        halo_from_bc = jnp.array(
            jnp.broadcast_to(
                halo_from_bc, halo_from_neighbor[side.value].shape
            ),
            dtype=halo_from_neighbor[side.value].dtype,
        )
      else:
        halo_from_bc = jnp.array(
            halo_from_bc, dtype=halo_from_neighbor[side.value].dtype
        )
      return jax.lax.cond(
          pred=is_first_or_last,
          true_fun=lambda: halo_from_bc,
          false_fun=lambda: halo_from_neighbor[side.value],
      )

  plane_to_exchange = 2 * halo_width - plane - 1

  low_halo_from_self = common_ops.get_face(
      array, axis, 0, plane_to_exchange, grid_params
  )
  high_halo_from_self = common_ops.get_face(
      array, axis, 1, plane_to_exchange, grid_params
  )
  halo_from_neighbor = _do_exchange(
      mesh, axis, low_halo_from_self, high_halo_from_self, periodic
  )

  n = array.shape[grid_params.get_axis_index(axis)]

  result = common_ops.array_scatter_1d_update(
      array,
      axis,
      plane,
      maybe_replace_halo_from_boundary_conditions(SideType.LOW),
      grid_params,
  )
  result = common_ops.array_scatter_1d_update(
      result,
      axis,
      n - plane - 1,
      maybe_replace_halo_from_boundary_conditions(SideType.HIGH),
      grid_params,
  )
  return result


def inplace_halo_exchange(
    array: ScalarField,
    axes: tuple[str, ...],
    mesh: Mesh,
    grid_params: grid_parametrization.GridParametrization,
    periodic_dims: list[bool] | None = None,
    boundary_conditions: BoundaryConditionsSpec | None = None,
    halo_width: int = 1,
) -> ScalarField:
  """Performs a N-dimensional halo exchange.

  If boundary_conditions is not `None` it must have the form
  [ [(BCType for dim 0 lower bound, value for dim 0 lower bound),
  (BCType for dim 0 upper bound, value for dim 0 upper bound)],
  [(BCType for dim1 lower bound, value for dim 1 lower bound),
  (BCType for dim1 upper bound, value for dim 1 upper bound)], ... ]
  Note the innermost sequence can be `None`, in which case the corresponding
  boundary will be set to zero. Also, boundary conditions only apply when the
  corresponding dimension is not periodic. The value can be a float, or can be a
  sequence of planes of length `width`. An element of this sequence is a 2D
  array. For example, if the grid_params.data_axis_order=('z', 'x', 'y'), the
  shape of the 2D array is (nz, ny) for axis='x'. The order of planes in the
  sequence is from low to high along the dimension `axes`. This means for a low
  boundary the innermost plane is the last or `halo_width - 1` element in the
  sequence. For a high boundary the innermost plane is the 0th element. In the
  Neumann case the grid spacing is not taken into account, so the value
  specified by the constant or the 2D plane(s) should take into account the grid
  spacing as appropriate. Halo exchange / applying boundary conditions is done
  one plane at a time for performance reasons.

  Args:
    array: A 3D array representing the field.
    axes: Tuple of axes along which the halo exchange is to be performed. Each
      axis must be one of "x", "y", "z".
    mesh: A jax Mesh object representing the device topology.
    grid_params: The grid parametrization object.
    periodic_dims: If not `None`, must be a boolean vector with the same length
      as `axes`. Indicates whether the particular dimension uses periodic
      boundary conditions.
    boundary_conditions: The boundary conditions to apply. If `None`, the
      boundary will be set to 0. See more info about boundary conditions below.
    halo_width: The width of halo to exchange.

  Returns:
    `array` with the halo exchange and boundary conditions applied.
  """
  for axis in axes:
    if axis not in ("x", "y", "z"):
      raise ValueError(f"Unsupported axis: {axis}. It must be one of x, y, z.")
  periodic_dims = (
      periodic_dims if periodic_dims is not None else [False] * len(axes)
  )
  boundary_conditions = (
      boundary_conditions
      if boundary_conditions is not None
      else [[None, None]] * len(axes)
  )
  if len(axes) != len(periodic_dims):
    raise ValueError(
        f"The number of axes ({len(axes)}) must be equal to the number of"
        f" periodic dimensions ({len(periodic_dims)})."
    )
  if len(axes) != len(boundary_conditions):
    raise ValueError(
        f"The number of axes ({len(axes)}) must be equal to the number of"
        f" boundary conditions ({len(boundary_conditions)})."
    )

  def get_bc_val(bc_info, plane):
    """Updates the boundary condition information at a specific `plane`."""
    if bc_info is not None:
      # Create a mutable copy of the bc passed in.
      bc_info_plane = list(bc_info)
      # If the boundary condition is a list of planes select the relevant one.
      bc_info_plane[1] = (
          bc_info_plane[1]
          if isinstance(bc_info_plane[1], float)
          else bc_info_plane[1][plane]
      )
    else:
      bc_info_plane = None

    return bc_info_plane

  with jax.named_scope("HaloExchange"):
    for axis, periodic, bc in zip(axes, periodic_dims, boundary_conditions):
      bc_low, bc_high = bc if bc is not None else (None, None)
      _validate_boundary_condition(bc_low, array, axis, halo_width, grid_params)
      _validate_boundary_condition(
          bc_high, array, axis, halo_width, grid_params
      )

      # Apply halo exchange for each plane in `width`, one by one, from
      # innermost to outermost (`plane = width - 1` to `plane = 0`). The
      # `plane` index is relative to a low set of boundary planes ordered along
      # dim (that is, from outermost to innermost). Calling
      # `_inplace_halo_exchange_1d` with planes in this order is necessary in
      # order for Neumann boundary conditions to be applied correctly.
      for plane in range(halo_width - 1, -1, -1):
        # Select the relevant planes from the sequence of bc planes.
        bc_low_plane = get_bc_val(bc_low, plane)
        bc_high_plane = get_bc_val(bc_high, halo_width - plane - 1)

        array = _inplace_halo_exchange_1d(
            array,
            axis,
            mesh,
            bool(periodic),
            bc_low_plane,
            bc_high_plane,
            halo_width,
            plane,
            grid_params,
        )

    return array


def _validate_boundary_condition(
    bc: FaceBoundaryCondition,
    array: ScalarField,
    axis: str,
    width: int,
    grid_params: grid_parametrization.GridParametrization,
) -> None:
  """Checks the validity of the boundary condition.

  A boundary condition must be `None` or a sequence of length 2. If not `None`,
  the first element is a BCType and the second element is a sequence of length
  `width` describing an ordered set of 2D planes orthogonal to `dim`.

  Args:
    bc: A boundary condition for one side of one axis.
    array: A 3D array.
    axis: The axis of the plane to apply the boundary condition to.
    width: The width of the boundary (same as the width used in halo exchange).
    grid_params: The grid parametrization object.

  Returns:
    None.

  Raises:
    ValueError if the boundary condition is invalid.
  """
  if bc is None:
    return

  bc_type, bc_value = bc

  bc_types = list(BCType)
  if bc_type not in bc_types:
    raise ValueError(
        f"The first element of a boundary condition must be one of {bc_types}."
        f" Got {bc_type} (type {type(bc_type)})."
    )

  # bc_value must be a float or a sequence.
  if isinstance(bc_value, float):
    return

  if not isinstance(bc_value, list):
    raise ValueError(
        "The boundary condition must be specified as a float or a list. But"
        f" got type of {type(bc_value)}."
    )
  if len(bc_value) != width:
    raise ValueError(
        f"If bc is a list it should have length {width}. Found length"
        f" {len(bc_value)}."
    )

  if not isinstance(array, jax.Array):
    raise ValueError(
        f"Unsupported array type for halo exchange: {type(array)}."
    )
  if array.ndim != 3:
    raise ValueError(
        f"`array` must be a 3D array. Found array with ndim: {array.ndim}."
    )

  array_shape = list(array.shape)
  array_shape[grid_params.get_axis_index(axis)] = 1
  array_shape = tuple(array_shape)
  for bc_array in bc_value:
    if not isinstance(bc_array, jax.Array):
      raise ValueError(
          "The boundary condition must be a sequence of 2D `jax.Array` with"
          f" length being the halo width. Found list of {type(bc_array)}."
      )
    if bc_array.shape != array_shape:
      raise ValueError(
          "The boundary condition must be a float or a list of tensors of"
          f" shape {array_shape}. Found a tensor in the list with shape"
          f" {bc_array.shape}."
      )


def set_halos_to_zero(
    array: ScalarField,
    halo_width: int,
    grid_params: grid_parametrization.GridParametrization,
) -> ScalarField:
  """Sets values inside halos to zero.

  Sets halo values to zero for a 3D field in each device. It is needed in
  cases where the values in the halo shouldn't be used or are otherwise invalid.
  For example, when doing a dot product across devices, the halos store
  information that are already accounted for in another device. Computing the
  dot product without zeroing out the halos will cause some values in the two
  vectors to be counted twice.

  Args:
    array: A 3D array.
    halo_width: The width of the halo.
    grid_params: The grid parametrization object.

  Returns:
    A list of `tf.Tensor` corresponding to the 3D field with the interior
      unchanged and the halo values set to zero.

  Raises:
    ValueError if halo width is too large, and there are no interior points.
  """
  if halo_width <= 0:
    return array

  # Edge case: Invalid inputs if any dimension has no interior points at all.
  nx, ny, nz = grid_params.to_xyz_order(array.shape)

  if min((nx, ny, nz)) <= halo_width * 2:
    raise ValueError(
        f"Please double check sizes (nx, ny, nz) = ({nx}, {ny}, {nz}) <= 2 *"
        f" halo_width ({halo_width}), so there are no interior points."
    )
  return common_ops.pad(
      array[
          halo_width:-halo_width,
          halo_width:-halo_width,
          halo_width:-halo_width,
      ],
      (halo_width, halo_width),
      (halo_width, halo_width),
      (halo_width, halo_width),
      grid_params,
  )
