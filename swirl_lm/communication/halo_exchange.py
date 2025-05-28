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

from collections import abc
from typing import Optional, Sequence, Union

import numpy as np
from six.moves import range
from swirl_lm.communication import halo_exchange_utils
from swirl_lm.utility import common_ops
from swirl_lm.utility import types
import tensorflow as tf


BCType = halo_exchange_utils.BCType
BoundaryConditionsSpec = halo_exchange_utils.BoundaryConditionsSpec
FlowFieldVal = types.FlowFieldVal
SideType = halo_exchange_utils.SideType
_DTYPE = types.TF_DTYPE


def _a_x_plus_y(
    a: float,
    x: Union[float, FlowFieldVal],
    y: FlowFieldVal,
) -> FlowFieldVal:
  """Performs ax + y.

  Args:
    a: A constant coefficient.
    x: A constant or a 3D tensor represented as either a single `tf.Tensor` or
      a list of `tf.Tensor`.
    y: A 3D tensor represented as either a single `tf.Tensor` or a list of
      `tf.Tensor`

  Returns:
    A 3D tensor that has the same structure as `y` with value being
    `a` * `x` + `y`.
  """
  if isinstance(x, float):
    return tf.nest.map_structure(lambda y_i: a * x + y_i, y)
  else:
    return tf.nest.map_structure(lambda x_i, y_i: a * x_i + y_i, x, y)


@tf.function
def _get_homogeneous_neumann_bc_order2(
    f: types.FlowFieldVal,
    dim: int,
    face: int,
    plane: int,
) -> types.FlowFieldVal:
  """Gets the 2D plane for homogeneous Neumann BC with second order difference.

  Args:
    f: A 3D tensor to which the homogeneous Neumann BC is applied.
    dim: The dimension to apply the BC. Has to be one of 0, 1, or 2.
    face: A binary indicator for the location of the boundary, with 0 and 1
      being the lower and higher ends of the domain, respectively.
    plane: The index of the plane to apply the BC. If `face` is 0, it counts
      from the lower end of the domain; if `face` is 1, it counts from the
      higher end of the domain.

  Returns:
    A 2D plane that corresponds to the homogeneous Neumann BC that is computed
    with the second-order one-sided finite difference scheme.
  """
  # Define the function that computes the value of plane so that the gradient at
  # this plane is 0.
  bc_fn = lambda f_1, f_2: 4.0 / 3.0 * f_1 - f_2 / 3.0

  f_near = common_ops.get_face(f, dim, face, plane + 1)[0]
  f_far = common_ops.get_face(f, dim, face, plane + 2)[0]

  return tf.nest.map_structure(bc_fn, f_near, f_far)


@tf.function
def _get_homogeneous_neumann_bc_order1(
    f: types.FlowFieldVal,
    dim: int,
    face: int,
    plane: int,
) -> types.FlowFieldVal:
  """Gets the 2D plane for homogeneous Neumann BC with first order difference.

  Args:
    f: A 3D tensor to which the homogeneous Neumann BC is applied.
    dim: The dimension to apply the BC. Has to be one of 0, 1, or 2.
    face: A binary indicator for the location of the boundary, with 0 and 1
      being the lower and higher ends of the domain, respectively.
    plane: The index of the plane to apply the BC. If `face` is 0, it counts
      from the lower end of the domain; if `face` is 1, it counts from the
      higher end of the domain.

  Returns:
    A 2D plane that corresponds to the homogeneous Neumann BC that is computed
    with the first-order finite difference scheme.
  """
  return common_ops.get_face(f, dim, face, plane + 1)[0]


@tf.function
def _get_additive_bc(
    f: types.FlowFieldVal,
    dim: int,
    face: int,
    plane: int,
) -> types.FlowFieldVal:
  """Gets the 2D plane for the additive BC.

  Args:
    f: A 3D tensor to which the homogeneous Neumann BC is applied.
    dim: The dimension to apply the BC. Has to be one of 0, 1, or 2.
    face: A binary indicator for the location of the boundary, with 0 and 1
      being the lower and higher ends of the domain, respectively.
    plane: The index of the plane to apply the BC. If `face` is 0, it counts
      from the lower end of the domain; if `face` is 1, it counts from the
      higher end of the domain.

  Returns:
    A 2D plane for the additive boundary condition.
  """
  return common_ops.get_face(f, dim, face, plane)[0]


def _do_exchange(replicas, replica_dim, high_halo_for_predecessor,
                 low_halo_for_successor, periodic):
  """Does a halo exchange with predecessors/successors."""

  # This inner wrapper is used to avoid passing `replicas` directly, which
  # causes the function tracing to mistake it as an abstract tensor (when in
  # fact it is a numpy array) and causes breakage downstream.
  @tf.function
  def _inner_do_exchange(replica_dim, high_halo_for_predecessor,
                         low_halo_for_successor, periodic):
    # Special case for when number of replicas along `replica_dim` is 1.
    # Note although in the periodic case the normal flow handles it correctly,
    # this carve-out handles it more efficiently.
    if replicas.shape[replica_dim] == 1:
      if periodic:
        return low_halo_for_successor, high_halo_for_predecessor
      else:
        return [
            tf.nest.map_structure(tf.zeros_like, high_halo_for_predecessor),
        ] * 2

    # Compute the predecessor and successor replicas in `replica_dim`.
    if not periodic:
      padded_replicas = halo_exchange_utils.pad_in_dim(
          replicas, low_pad=1, high_pad=1, value=-1, axis=replica_dim)
    else:
      padded_replicas = np.concatenate(
          (halo_exchange_utils.slice_in_dim(replicas, -1, None, replica_dim),
           replicas, halo_exchange_utils.slice_in_dim(
               replicas, 0, 1, replica_dim)), replica_dim)
    predecessors = np.stack(
        (replicas,
         halo_exchange_utils.slice_in_dim(
             padded_replicas, start=0, end=-2, axis=replica_dim)),
        axis=-1)
    predecessors = [(a, b) for (a, b) in predecessors.reshape((-1, 2))
                    if b != -1]
    high_halo = tf.raw_ops.CollectivePermute(
        input=high_halo_for_predecessor,
        source_target_pairs=predecessors,
        name="high")

    successors = np.stack(
        (replicas,
         halo_exchange_utils.slice_in_dim(
             padded_replicas, start=2, end=None, axis=replica_dim)),
        axis=-1)
    successors = [(a, b) for (a, b) in successors.reshape((-1, 2)) if b != -1]
    low_halo = tf.raw_ops.CollectivePermute(
        input=low_halo_for_successor, source_target_pairs=successors,
        name="low")

    # Because the `CollectivePermute` function doesn't preserve the original
    # structure of the input tensor, we need to convert it back to its original
    # type here.
    if isinstance(high_halo_for_predecessor, Sequence):
      high_halo = tf.unstack(high_halo)

    if isinstance(low_halo_for_successor, Sequence):
      low_halo = tf.unstack(low_halo)

    return low_halo, high_halo

  return _inner_do_exchange(replica_dim, high_halo_for_predecessor,
                            low_halo_for_successor, periodic)


@tf.function
def _replace_halo(tensor, plane, bc, dim, side=None, low_side_padding=0):
  """Returns a halo plane derived from the boundary condition.

  This should only be called if all the following are true:
    * the grid is not periodic
    * `bc` is not `None`
    * the replica is at the end of the dimension on the specified side in the
      computational shape

  Args:
    tensor: A 3D tensor represented either as a tf.Tensor of shape (nz, nx, ny),
      or a list of length nz of tensors of shape (nx, ny), where nx, ny and nz
      are the number of points along the axes of a (sub)grid.
    plane: The index of the plane to apply the BC. If `side` is `LOW`, it counts
      from the lower end of the domain; if `side` is `HIGH`, it counts from the
      higher end of the domain.
    bc: The boundary conditions specification of the form [type, value]. See
      `inplace_halo_exchange` for full details about boundary condition
      specifications.
    dim: The dimension (aka axis), 0, 1 or 2 for x, y or z, respectively.
    side: A SideType indicating whether the calculation is for the low or high
      side of the axis. Only relevant in the case of Neumann boundary
      conditions.
    low_side_padding: The amount of padding on the lower side of the tensor
      along dim, i.e. left or top, where left and top refer to the 2d plane
      formed by dims 0 and 1. This is used only if `dim` is 0 or 1. This is used
      in the Saint-Venant simulation only.

  Returns:
    The border which is derived from the provided boundary conditions.

  Raises:
    ValueError if parameters have incorrect values.
  """
  if not isinstance(bc, abc.Sequence):
    raise ValueError("`bc` must be a sequence `(type, value)`.")

  if dim == 2 or side == SideType.HIGH:
    low_side_padding = 0

  # NB: Saint-Venant is the only sim with padding, and it uses halo width 1.
  # Using padding with `width > 1` has not been tested. Once this support is
  # tested, the below error should be removed.
  if low_side_padding > 0 and plane > 0:
    raise NotImplementedError("Padding with `width > 1` is not supported.")

  bc_type, bc_value = bc

  def neumann_value():
    buf = _get_homogeneous_neumann_bc_order1(tensor, dim, side.value,
                                             plane + low_side_padding)
    sign = -1.0 if side == SideType.LOW else 1.0
    return _a_x_plus_y(sign, bc_value, buf)

  def neumann_value_order2():
    buf = _get_homogeneous_neumann_bc_order2(tensor, dim, side.value,
                                             plane + low_side_padding)
    sign = -1.0 if side == SideType.LOW else 1.0
    return _a_x_plus_y(sign, bc_value, buf)

  def dirichlet_value():
    if isinstance(bc_value, float):
      buf = _get_homogeneous_neumann_bc_order1(tensor, dim, side.value, plane)
      return tf.nest.map_structure(lambda b: bc_value * tf.ones_like(b), buf)
    else:
      return bc_value

  def additive_value():
    buf = _get_additive_bc(tensor, dim, side.value, plane + low_side_padding)
    return _a_x_plus_y(1.0, bc_value, buf)

  def no_touch_value():
    return _get_additive_bc(tensor, dim, side.value, plane + low_side_padding)

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


def _inplace_halo_exchange_1d(tensor, dim, replica_id, replicas, replica_dim,
                              periodic, bc_low, bc_high, width, plane,
                              left_or_top_padding):
  """Performs halo exchange and assigns values to points in a boundary plane.

  This function exchanges and sets a single plane in the boundary or halo
  region. It needs to be called for each plane in the boundary or halo region,
  in order, from the innermost to outermost in order for Neumann boundary
  conditions to be correctly applied.

  Args:
    tensor: A 3D tensor represented either as a tf.Tensor of shape (nz, nx, ny),
      or a list of length nz of tensors of shape (nx, ny), where nx, ny and nz
      are the number of points along the axes of a (sub)grid.
    dim: The dimension of `z_list` in which halo exchange will be performed.
      Must be one of 0, 1 or 2 for x, y or z, respectively.
    replica_id: The replica id.
    replicas: A numpy array of replicas.
    replica_dim: The dimension of `replicas` along which the halo exchange is to
      be performed.
    periodic: Indicates whether the given dimension should be treated
      periodically.
    bc_low: The boundary condition for the low side of the axis. This is either
      `None` or of the form `(bc_type, bc_value)` where `bc_value` represents a
      single 2D plane and is either a 2D tensor of shape (nx, xy) or a sequence
      of length nz of tensors of shape (1, ny) or (nx, 1). See
      `inplace_halo_exchange` for more details about boundary condition
      specifications.
    bc_high: The boundary condition for the high side of the axis. See `bc_low`.
    width: The halo width.
    plane: Which plane to exchange. This is the index relative to a set of
      low-side boundary planes. (The corresponding index for a set of high-side
      boundary planes is calculated.) This function must be called in order from
      `plane = width - 1` to `plane = 0` for Neumann boundary conditions to be
      correctly applied.
    left_or_top_padding: The amount of left or top padding, where left and top
      refer to the 2d plane formed by dims 0 and 1. This is used only if `dim`
      is 0 or 1.

  Returns:
    The `z_list` with its `plane` boundary on the low side and corresponding
    plane on the high side in the `dim` dimension modified by the halos of its
    neighbors and/or boundary conditions.

  Raises:
    ValueError if parameters are incorrect.
  """
  assert dim in (0, 1, 2)

  tf.compat.v1.logging.debug(
      "dim: %d, replica_dim: %d, bc_low: %s, bc_high: %s", dim, replica_dim,
      bc_low, bc_high)

  is_first = halo_exchange_utils.is_first_replica(replica_id, replicas,
                                                  replica_dim)
  is_last = halo_exchange_utils.is_last_replica(replica_id, replicas,
                                                replica_dim)

  def maybe_replace_halo_from_boundary_conditions(side):
    """Maybe return 2D plane from boundary conditions rather than neighbor."""
    if side == SideType.LOW:
      pred = is_first
      bc = bc_low
    elif side == SideType.HIGH:
      pred = is_last
      bc = bc_high
    else:
      raise ValueError(f"Unsupported side type: {side.name}.")

    # `tf.cond` is potentially expensive as it evaluates the input of both
    # branches. The `if/else` statement can optimize performance by
    # eliminating an unnecessary `tf.cond` from the graph.
    if periodic or not bc:
      return halo_from_neighbor[side.value]
    else:
      halo_from_bc = _replace_halo(tensor, plane, bc, dim, side,
                                   left_or_top_padding)
      return tf.cond(
          pred=pred,
          true_fn=lambda: halo_from_bc,
          false_fn=lambda: halo_from_neighbor[side.value])

  plane_to_exchange = 2 * width - plane - 1

  low_halo_from_self = common_ops.get_face(tensor, dim, 0, plane_to_exchange)[0]
  high_halo_from_self = common_ops.get_face(tensor, dim, 1,
                                            plane_to_exchange)[0]
  halo_from_neighbor = _do_exchange(replicas, replica_dim, low_halo_from_self,
                                    high_halo_from_self, periodic)

  n = common_ops.get_shape(tensor)[dim]

  plane_padded = plane + left_or_top_padding
  result_list = tf.cond(
      pred=is_first,
      true_fn=lambda: common_ops.tensor_scatter_1d_update(
          tensor, dim, plane_padded,
          maybe_replace_halo_from_boundary_conditions(SideType.LOW)),
      false_fn=lambda: common_ops.tensor_scatter_1d_update(
          tensor, dim, plane,
          maybe_replace_halo_from_boundary_conditions(SideType.LOW)))
  result_list = common_ops.tensor_scatter_1d_update(
      result_list, dim, n - plane - 1,
      maybe_replace_halo_from_boundary_conditions(SideType.HIGH))

  return result_list


def inplace_halo_exchange(
    tensor: FlowFieldVal,
    dims: Sequence[int],
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    replica_dims: Sequence[int],
    periodic_dims: Optional[Sequence[bool]] = None,
    boundary_conditions: Optional[BoundaryConditionsSpec] = None,
    width: int = 1,
    left_padding: int = 0,
    top_padding: int = 0) -> FlowFieldVal:
  """Performs a N-dimensional halo exchange.

  Args:
    tensor: A 3D tensor represented either as a tf.Tensor of shape (nz, nx, ny),
      or a list of length nz of tensors of shape (nx, ny), where nx, ny and nz
      are the number of points along the axes of a (sub)grid.
    dims: The dimensions or axes along which halo exchange will be performed.
      This is a sequence containing some or all of 0, 1, 2 (corresponding to x,
      y, z).
    replica_id: The replica id.
    replicas: A numpy array of replicas.
    replica_dims: The dimensions of `replicas` along which halo exchange will be
      performed.
    periodic_dims: If not `None`, must be a boolean vector with the same length
      as replica_dims. Indicates whether the particular dimension uses periodic
      boundary conditions.
    boundary_conditions: The boundary conditions to apply. If `None`, the
      boundary will be set to 0. See more info about boundary conditions below.
    width: The width of halo to exchange.
    left_padding: The amount of left padding, referring the 2d plane formed by
      dims 0 and 1 (left is dim 1).
    top_padding: The amount of top padding, referring to the 2d plane formed by
      dims 0 and 1 (top is dim 0).  If boundary_conditions is not `None` it must
      have the form  [ [(BCType for dim 0 lower bound, value for dim 0 lower
      bound), (BCType for dim 0 upper bound, value for dim 0 upper bound)],
      [(BCType for dim1 lower bound, value for dim 1 lower bound), (BCType for
      dim1 upper bound, value for dim 1 upper bound)], ... ]  Note the innermost
      sequence can be `None`, in which case the corresponding boundary will be
      set to zero. Also, boundary conditions only apply when the corresponding
      dimension is not periodic. The value can be a float, or can be a sequence
      of planes of length `width`. An element of this sequence is a tensor if
      dim = 2 (z-axis) and a sequence if dim is 0 or 1. A z-axis boundary plane
      is specified by a 2D tensor of shape (nx, ny). A 2D x- or y-axis boundary
      plane is specified by a list of length nz of tensors of shape (1, ny) or
      (nx, 1), respectively. The order of planes in the sequence is from low to
      high along the dimension `dim`. This means for a low boundary the
      innermost plane is the last or `width - 1` element in the sequence. For a
      high boundary the innermost plane is the 0th element.  In the Neumann case
      the grid spacing is not taken into account, so the value specified by the
      constant or the 2D plane(s) should take into account the grid spacing as
      appropriate.  Halo exchange / applying boundary conditions is done one
      plane at a time for performance reasons. If width > 1 a single tensor of
      shape (e.g.) (nx, ny, width) could be exchanged. This was implemented but,
      when width = 1, was found to be slow compared to the original width = 1
      implementation. By doing halo exchange plane by plane the performance is
      the same as the original implementation in the most common and important
      width = 1 case.

  Returns:
    The incoming z_list modified to include the result of halo exchange and
      taking boundary conditions into account.
  """
  periodic_dims = periodic_dims or [None] * len(dims)
  boundary_conditions = (boundary_conditions if boundary_conditions is not None
                         else [[None, None]] * len(dims))

  assert len(dims) == len(replica_dims)
  assert len(dims) == len(periodic_dims)
  assert len(dims) == len(boundary_conditions)

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

  with tf.name_scope("HaloExchange"):
    for dim, replica_dim, periodic, bc in zip(
        dims, replica_dims, periodic_dims, boundary_conditions
    ):
      bc_low, bc_high = bc if bc is not None else (None, None)
      _validate_boundary_condition(bc_low, tensor, dim, width)
      _validate_boundary_condition(bc_high, tensor, dim, width)

      left_or_top_padding = (top_padding, left_padding, 0)[dim]

      # Apply halo exchange for each plane in `width`, one by one, from
      # innermost to outermost (`plane = width - 1` to `plane = 0`). The
      # `plane` index is relative to a low set of boundary planes ordered along
      # dim (that is, from outermost to innermost). Calling
      # `_inplace_halo_exchange_1d` with planes in this order is necessary in
      # order for Neumann boundary conditions to be applied correctly.
      for plane in range(width - 1, -1, -1):
        # Select the relevant planes from the sequence of bc planes.
        bc_low_plane = get_bc_val(bc_low, plane)
        bc_high_plane = get_bc_val(bc_high, width - plane - 1)

        tensor = _inplace_halo_exchange_1d(tensor, dim,
                                           replica_id, replicas, replica_dim,
                                           bool(periodic), bc_low_plane,
                                           bc_high_plane, width, plane,
                                           left_or_top_padding)

    return tensor


def _validate_boundary_condition(bc, tensor, dim, width):
  """Checks the validity of the boundary condition.

  A boundary condition must be `None` or a sequence of length 2. If not `None`,
  the first element is a BCType and the second element is a sequence of length
  `width` describing an ordered set of 2D planes orthogonal to `dim`.

  With nx, ny and nz being the number of grid points along the x, y and z axes
  of a (sub)grid:
  * If `tensor` is a sequence of `tf.Tensor`:
    - If `dim` is 2 (z-axis), each element is a tensor of shape (nx, ny);
    - If `dim` is 0 or 1 (x- or y-axis), each element is a sequence of length nz
    of tensors of shape (1, ny) or (nx, 1), respectively.
  * If `tensor` is a 3D `tf.Tensor`:
    - If `dim` is 2, each element is a `tf.Tensor` of shape (1, nx, ny);
    - If `dim` is 0, each element is a `tf.Tensor` of shape (nz, 1, ny);
    - If `dim` is 1, each element is a `tf.Tensor` of shape (nz, nx, 1);

  Args:
    bc: A boundary condition for one side of one axis.
    tensor: A 3D tensor represented either as a tf.Tensor of shape (nz, nx, ny),
      or a list of length nz of tensors of shape (nx, ny), where nx, ny and nz
      are the number of points along the axes of a (sub)grid.
    dim: The dimension or axis (0, 1 or 2 for x, y or z, respectively).
    width: The width of the boundary (same as the width used in halo exchange).

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
    raise ValueError("The first element of a boundary condition must be one of "
                     "{}. Got {} (type {}).".format(bc_types, bc_type,
                                                    type(bc_type)))

  # bc_value must be a float or a sequence.
  if isinstance(bc_value, float):
    return

  if not isinstance(bc_value, abc.Sequence):
    raise ValueError("The boundary condition must be specified as a float "
                     "or a list.")
  if len(bc_value) != width:
    raise ValueError("If bc is a list it should have length {}. Found "
                     "length {}.".format(width, len(bc_value)))

  if isinstance(tensor, tf.Tensor):
    tensor_shape = tensor.get_shape().as_list()
    axis = (dim + 1) % 3
    tensor_shape[axis] = 1

    for bc_tensor in bc_value:
      if not isinstance(bc_tensor, tf.Tensor):
        raise ValueError(
            f"If the 3D field is a `tf.Tensor`, the boundary condition must be "
            f"a sequence of 2D `tf.Tensor` with length being the halo width. "
            f"Found list of {type(bc_tensor)}."
        )
      if bc_tensor.shape != tensor_shape:
        raise ValueError(
            f"If the 3D tensor is a tf.Tensor, the boundary condition must be a"
            f"float or a list of tensors of shape {tensor_shape}. Found tensor "
            f"in list with shape {bc_tensor.shape}."
        )
  elif isinstance(tensor, Sequence):
    if dim == 2:
      # Each element of `bc_value` must be a tensor of shape `(nx, ny)`.
      for bc_tensor in bc_value:
        if not isinstance(bc_tensor, tf.Tensor):
          raise ValueError("If dim = 2 a boundary condition must be a float or "
                           "a list of tensors. Found list of {}.".format(
                               type(bc_tensor)))
        if bc_tensor.shape != tensor[0].shape:
          raise ValueError("If dim = 2 a boundary condition must be a float or "
                           "a list of tensors of shape {}. Found tensor in list"
                           "with shape {}.".format(tensor[0].shape,
                                                   bc_tensor.shape))
    else:  # dim in (0, 1)
      # Each element of bc_value must be a list of length nz of tensors of shape
      # (1, ny) or (nx, 1).
      for bc_z_list in bc_value:
        if not isinstance(bc_z_list, abc.Sequence):
          raise ValueError(
              "If dim is 0 or 1 a boundary condition must be a "
              "float or a list of list of tensors. Found list of {}.".format(
                  type(bc_z_list)))
        if len(bc_z_list) != len(tensor):
          raise ValueError("If dim is 0 or 1 each z-list should be length {}. "
                           "Found length {}.".format(
                               len(tensor), len(bc_z_list)))
        for bc_tensor in bc_z_list:
          if not isinstance(bc_tensor, tf.Tensor):
            raise ValueError(
                "If dim is 0 or 1 the boundary condition should be "
                "a list of list of tensor. Found list of list of {}.".format(
                    type(bc_tensor)))
          expected_shape = tensor[0].shape.as_list()
          expected_shape[dim] = 1
          if expected_shape != bc_tensor.shape.as_list():
            raise ValueError("For dim = {} the boundary condition should be a "
                             "float or a list of list of tensor of shape {}. "
                             "Found tensor with shape {}.".format(
                                 dim, expected_shape, bc_tensor.shape))
  else:
    raise ValueError(
        f"Unsupported 3D tensor type for halo exchange {type(tensor)}.")


def get_edge_of_3d_field(
    tensor: FlowFieldVal,
    dim: int,
    side: SideType,
    width: int = 1) -> Union[FlowFieldVal, Sequence[FlowFieldVal]]:
  """Helper function that returns an edge of a 3D field.

  Args:
    tensor: A 3D tensor represented either as a `tf.Tensor` of shape
      `(nz, nx, ny)`, or a list of length nz of tensors of shape (nx, ny), where
      `nx`, `ny` and `nz` are the number of points along the axes of a
      (sub)grid.
    dim: The dimension or axis (0, 1 or 2 for x, y or z).
    side: A SideType enum indicating the low or high side of the axis.
    width: The width of the edge.

  Returns:
    A list of length `width`. The elements of the list are 2D tensors of shape
      `(nx, ny)` if `dim = 2`, and are [list of length `nz` of 2D tensors of
      shape `(1, ny)` or `(nx, 1)`] if `dim = 0` or `1`, respectively.
  """
  wlist_of_zlist = []
  for plane in range(width):
    wlist_of_zlist.append(
        common_ops.get_face(tensor, dim, side.value, plane)[0])

  # Because the order of the generated list is from outer to inner, we need to
  # reverse the order on the higher end.
  if side == SideType.HIGH:
    wlist_of_zlist = wlist_of_zlist[::-1]

  return wlist_of_zlist


def clear_halos(x: FlowFieldVal, halo_width: int) -> FlowFieldVal:
  """Sets value inside halos to zero.

  Sets halo values to zero for a 3D field in each TPU core. It is needed in
  cases where the values in the halo shouldn't be used or are otherwise invalid.
  For example, when doing a dot product across TPUs, the halos store
  information that are already accounted for in another TPU. Computing the dot
  product without zeroing out the halos will cause some values in the two
  vectors to be counted twice.

  Args:
    x: A 3D tensor represented either as a tf.Tensor of shape `(nz, nx, ny)`, or
      a list of length `nz` of tensors of shape `(nx, ny)`, where `nx`, `ny` and
      `nz` are the number of points along the axes of a (sub)grid.
    halo_width: The width of the halo.

  Returns:
    A list of `tf.Tensor` corresponding to the 3D field with the interior
      unchanged and the halo values set to zero.

  Raises:
    ValueError if halo width is too large, and there are no interior points.
  """
  if (halo_width <= 0):
    return x

  # Edge case: Invalid inputs if any dimension has no interior points at all.
  nx, ny, nz = common_ops.get_shape(x)

  if min((nx, ny, nz)) <= halo_width * 2:
    raise ValueError(
        "Please double check sizes (nx, ny, nz) = ({}, {}, {}) <= 2 * "
        "halo_width ({}), so there are no interior points.".format(
            nx, ny, nz, halo_width))

  if isinstance(x, tf.Tensor):
    padding = tf.constant([[halo_width, halo_width]] * 3, dtype=tf.int32)
    return tf.pad(
        x[halo_width:-halo_width, halo_width:-halo_width,
          halo_width:-halo_width],
        paddings=padding,
        mode="CONSTANT")

  padding = tf.constant(
      [[halo_width, halo_width], [halo_width, halo_width]], dtype=tf.int32
  )
  x_new = [
      # pylint: disable=g-complex-comprehension
      tf.pad(
          x[i][halo_width:-halo_width, halo_width:-halo_width],
          paddings=padding,
          mode="CONSTANT") for i in range(halo_width, nz - halo_width)
      # pylint: enable=g-complex-comprehension
  ]
  x_new = [tf.zeros_like(x[0])] * halo_width + x_new + [tf.zeros_like(x[0])
                                                       ] * halo_width

  return x_new
