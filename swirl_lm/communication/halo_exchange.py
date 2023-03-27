# Copyright 2023 The swirl_lm Authors.
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


def _do_exchange(replicas, replica_dim, high_halo_for_predecessor,
                 low_halo_for_successor, periodic):
  """Does a halo exchange with predecessors/successors."""
  # Special case for single replica grid width along the replica_dim.
  # Note although in the periodic case the normal flow handles it correctly,
  # for performance efficiency, it is handled in this carve-out.
  if replicas.shape[replica_dim] == 1:
    if periodic:
      return high_halo_for_predecessor, low_halo_for_successor
    else:
      return [tf.zeros_like(high_halo_for_predecessor)] * 2

  # Compute the predecessor and successor replicas in `replica_dim`.
  if not periodic:
    padded_replicas = halo_exchange_utils.pad_in_dim(
        replicas, low_pad=1, high_pad=1, value=-1, axis=replica_dim)
  else:
    padded_replicas = np.concatenate(
        (halo_exchange_utils.slice_in_dim(replicas, -1, None, replica_dim),
         replicas, halo_exchange_utils.slice_in_dim(replicas, 0, 1,
                                                    replica_dim)), replica_dim)
  predecessors = np.stack(
      (replicas,
       halo_exchange_utils.slice_in_dim(
           padded_replicas, start=0, end=-2, axis=replica_dim)),
      axis=-1)
  predecessors = [(a, b) for (a, b) in predecessors.reshape((-1, 2)) if b != -1]
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
      input=low_halo_for_successor, source_target_pairs=successors, name="low")

  return high_halo, low_halo


def _replace_halo(plane, bc, dim, side=None):
  """Return a halo derived from boundary conditions.

  This should only be called if all the following are true:
    * the grid is not periodic
    * `bc` is not `None`
    * the replica is at the end of the dimension on the specified side in the
      computational shape

  Args:
    plane: A 2D tensor. The plane from the subgrid relevant in applying boundary
      conditions (and to get the shape for Dirichlet boundary conditions).
    bc: The boundary conditions specification of the form [type, value]. See
      `inplace_halo_exchange` for full details about boundary condition
      specifications.
    dim: The dimension (aka axis), 0, 1 or 2 for x, y or z, respectively.
    side: A SideType indicating whether the calculation is for the low or high
      side of the axis. Only relevant in the case of Neumann boundary
      conditions.

  Returns:
    The border which is derived from the provided boundary conditions.

  Raises:
    ValueError if parameters have incorrect values.
  """
  if not isinstance(bc, abc.Sequence):
    raise ValueError("`bc` must be a sequence `(type, value)`.")

  bc_type, bc_value = bc

  # bc_value could be a list of tensors of shape (1, ny) or (nx, 1). If so,
  # convert to a tensor of shape (nz, ny) or (nx, nz). nx, ny, nz are the number
  # of points along each axis of a (sub)grid. After this line, bc_value is
  # either a float or a 2D tensor.
  bc_value = (
      tf.concat(bc_value, dim) if isinstance(bc_value, list) else bc_value)

  def neumann_value():
    sign = -1.0 if side == SideType.LOW else 1.0
    return plane + sign * bc_value

  def dirichlet_value():
    if isinstance(bc_value, tf.Tensor):
      return bc_value
    return tf.ones_like(plane) * bc_value

  def additive_value():
    return plane + bc_value

  if bc_type in (BCType.NEUMANN, BCType.NEUMANN_2):
    return neumann_value()
  elif bc_type == BCType.DIRICHLET:
    return dirichlet_value()
  elif bc_type == BCType.ADDITIVE:
    return additive_value()
  else:
    raise ValueError("Unknown boundary condition type: {}.".format(bc_type))


def _sliced_tensor_fn(tensor, slices):
  return lambda: tensor[tuple(slices)]


def _halo_from_self_dim_0_1(z_list, dim, plane_to_exchange, is_first,
                            left_or_top_padding):
  """Returns halos from the z_list given the dimension and plane to exchange."""
  if dim not in [0, 1]:
    raise ValueError("dim not in [0, 1]: {}".format(dim))
  low_slices, low_slices_padded, high_slices = ([slice(None)] * 2,
                                                [slice(None)] * 2,
                                                [slice(None)] * 2)
  low_slices[dim] = slice(plane_to_exchange, plane_to_exchange + 1)
  low_slices_padded[dim] = slice(plane_to_exchange + left_or_top_padding,
                                 plane_to_exchange + left_or_top_padding + 1)
  shape = z_list[0].shape.as_list()[dim]
  high_slices[dim] = slice(shape - (plane_to_exchange + 1),
                           shape - plane_to_exchange)

  low_halo_from_self, high_halo_from_self = [], []
  for tensor in z_list:
    low_halo = tf.cond(
        pred=is_first,
        true_fn=_sliced_tensor_fn(tensor, low_slices_padded),
        false_fn=_sliced_tensor_fn(tensor, low_slices))
    low_halo_from_self.append(low_halo)
    high_halo_from_self.append(tensor[high_slices])
  # Convert to 2D tensor: a z-y or x-z plane.
  low_halo_from_self = _convert_zlist_to_2d_tensor(low_halo_from_self, dim)
  high_halo_from_self = _convert_zlist_to_2d_tensor(high_halo_from_self, dim)
  return low_halo_from_self, high_halo_from_self


def _plane_for_bc_dim_0_1(z_list, plane, dim, is_low, is_first,
                          left_or_top_padding):
  """Returns a plane to be used as a boundary condition."""
  if dim not in [0, 1]:
    raise ValueError("dim not in [0, 1]: {}".format(dim))
  slices, slices_padded = [slice(None)] * 2, [slice(None)] * 2
  shape = z_list[0].shape.as_list()[dim]
  slices[dim] = (
      slice(plane, plane + 1) if is_low else slice(shape - (plane + 1), shape -
                                                   plane))
  slices_padded[dim] = (
      slice(plane + left_or_top_padding, plane + left_or_top_padding +
            1) if is_low else slice(shape - (plane + 1), shape - plane))
  plane_for_bc = []
  for tensor in z_list:
    plane = tf.cond(
        pred=is_first,
        true_fn=_sliced_tensor_fn(tensor, slices_padded),
        false_fn=_sliced_tensor_fn(tensor, slices))
    plane_for_bc.append(plane)
  return _convert_zlist_to_2d_tensor(plane_for_bc, dim)


def _alias_inplace_update(x, plane, low):
  indices = [[plane]]
  return lambda: tf.tensor_scatter_nd_update(x, indices, [tf.squeeze(low),])


def _inplace_halo_exchange_1d(z_list, dim, replica_id, replicas, replica_dim,
                              periodic, bc_low, bc_high, width, plane,
                              left_or_top_padding):
  """Performs halo exchange and assigns values to points in a boundary plane.

  This function exchanges and sets a single plane in the boundary or halo
  region. It needs to be called for each plane in the boundary or halo region,
  in order, from the innermost to outermost in order for Neumann boundary
  conditions to be correctly applied.

  Args:
    z_list: A list of length nz of tensors of shape (nx, ny), where nx, ny and
      nz are the number of points along the axes of a (sub)grid.
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

    def low_from_bc():
      if bc_low[0] == BCType.NO_TOUCH:
        return low_plane_for_outermost_slice
      elif bc_low[0] == BCType.ADDITIVE:
        return _replace_halo(low_plane_for_outermost_slice, bc_low, dim,
                             SideType.LOW)
      elif bc_low[0] == BCType.NEUMANN_2:
        return _replace_halo(low_plane_for_neumann_2, bc_low, dim, SideType.LOW)
      else:  # bc_low[0] == BCType.NEUMANN
        return _replace_halo(low_plane_for_neumann, bc_low, dim, SideType.LOW)

    def high_from_bc():
      if bc_high[0] == BCType.NO_TOUCH:
        return high_plane_for_outermost_slice
      elif bc_high[0] == BCType.ADDITIVE:
        return _replace_halo(high_plane_for_outermost_slice, bc_high, dim,
                             SideType.HIGH)
      elif bc_high[0] == BCType.NEUMANN_2:
        return _replace_halo(high_plane_for_neumann_2, bc_high, dim,
                             SideType.HIGH)
      else:  # bc_low[0] == BCType.NEUMANN
        return _replace_halo(high_plane_for_neumann, bc_high, dim,
                             SideType.HIGH)

    if side == SideType.LOW:
      # `tf.cond` is potentially expensive as it evaluates the input of both
      # branches. The `if/else` statement can optimize performance by
      # eliminating an unnecessary `tf.cond` from the graph.
      if periodic or not bc_low:
        return low_halo_from_neighbor
      else:
        return tf.cond(
            pred=is_first,
            true_fn=low_from_bc,
            false_fn=lambda: low_halo_from_neighbor)
    else:  # side = HIGH
      if periodic or not bc_high:
        return high_halo_from_neighbor
      else:
        return tf.cond(
            pred=is_last,
            true_fn=high_from_bc,
            false_fn=lambda: high_halo_from_neighbor)

  low_plane_for_neumann_2 = tf.concat(
      _get_homogeneous_neumann_bc_order2(z_list, dim, 0, plane), dim)
  high_plane_for_neumann_2 = tf.concat(
      _get_homogeneous_neumann_bc_order2(z_list, dim, 1, plane), dim)

  plane_to_exchange = 2 * width - plane - 1
  if dim == 2:
    low_halo_from_self = z_list[plane_to_exchange]
    high_halo_from_self = z_list[-(plane_to_exchange + 1)]
    high_halo_from_neighbor, low_halo_from_neighbor = _do_exchange(
        replicas, replica_dim, low_halo_from_self, high_halo_from_self,
        periodic)
    low_plane_for_neumann = z_list[plane + 1]
    high_plane_for_neumann = z_list[-(plane + 2)]
    low_plane_for_outermost_slice = z_list[plane]
    low_edge = maybe_replace_halo_from_boundary_conditions(SideType.LOW)
    high_plane_for_outermost_slice = z_list[-(plane + 1)]
    high_edge = maybe_replace_halo_from_boundary_conditions(SideType.HIGH)
    grid = [low_edge] + z_list[plane + 1:-(plane + 1)] + [high_edge]
    if plane > 0:
      grid = z_list[:plane] + grid + z_list[-plane:]
    return grid

  # dim in (0, 1).
  low_halo_from_self, high_halo_from_self = _halo_from_self_dim_0_1(
      z_list, dim, plane_to_exchange, is_first, left_or_top_padding)
  high_halo_from_neighbor, low_halo_from_neighbor = _do_exchange(
      replicas, replica_dim, low_halo_from_self, high_halo_from_self, periodic)

  # If the plane being processed is the innermost (`plane = width - 1`), the
  # same plane is used in applying Neumann boundary conditions. `plane`
  # is an index relative to an ordered sequence of low-side boundary planes
  # where the 0th index is the outermost boundary plane.
  low_plane_for_neumann = low_halo_from_self
  high_plane_for_neumann = high_halo_from_self
  low_plane_for_outermost_slice, high_plane_for_outermost_slice = (
      _halo_from_self_dim_0_1(z_list, dim, plane_to_exchange - 1, is_first,
                              left_or_top_padding))
  if plane != width - 1:
    # NB: Saint-Venant is the only sim with padding, and it uses halo width 1,
    # so the following `_plane_for_bc_dim_0_1` calls are currently only made in
    # cases where `left_or_top_padding` is 0. Using padding with `width > 1` has
    # not been tested. Once this support is tested, the below error should be
    # removed.
    if left_or_top_padding:
      raise NotImplementedError("Padding with `width > 1` is not supported.")
    if bc_low and bc_low[0] == BCType.NEUMANN:
      low_plane_for_neumann = _plane_for_bc_dim_0_1(z_list, plane + 1, dim,
                                                    True, is_first,
                                                    left_or_top_padding)
    if bc_low and (bc_low[0] == BCType.NO_TOUCH or
                   bc_low[0] == BCType.ADDITIVE):
      low_plane_for_outermost_slice = _plane_for_bc_dim_0_1(
          z_list, plane, dim, True, is_first, left_or_top_padding)
    if bc_high and bc_high[0] == BCType.NEUMANN:
      high_plane_for_neumann = _plane_for_bc_dim_0_1(z_list, plane + 1, dim,
                                                     False, is_first,
                                                     left_or_top_padding)
    if bc_high and (bc_high[0] == BCType.NO_TOUCH or
                    bc_high[0] == BCType.ADDITIVE):
      high_plane_for_outermost_slice = _plane_for_bc_dim_0_1(
          z_list, plane, dim, False, is_first, left_or_top_padding)

  low_edge = maybe_replace_halo_from_boundary_conditions(SideType.LOW)
  high_edge = maybe_replace_halo_from_boundary_conditions(SideType.HIGH)

  high_edges = _convert_2d_tensor_to_zlist(high_edge, dim)
  low_edges = _convert_2d_tensor_to_zlist(low_edge, dim)
  result_list = []

  plane_padded = plane + left_or_top_padding
  for x, high, low in zip(z_list, high_edges, low_edges):
    if dim == 0:
      indices = [[x.shape.as_list()[0] - (plane + 1)]]
      x = tf.tensor_scatter_nd_update(
          tf.cond(
              pred=is_first,
              true_fn=_alias_inplace_update(x, plane_padded, low),
              false_fn=_alias_inplace_update(x, plane, low)),
          indices, [tf.squeeze(high),])
    else:
      indices = [[x.shape.as_list()[1] - (plane + 1)]]
      x = tf.transpose(
          tf.tensor_scatter_nd_update(
              tf.cond(
                  pred=is_first,
                  true_fn=_alias_inplace_update(
                      tf.transpose(x, perm=[1, 0]), plane_padded, low),
                  false_fn=_alias_inplace_update(
                      tf.transpose(x, perm=[1, 0]), plane, low)),
              indices, [tf.squeeze(high),]),
          perm=[1, 0])
    result_list.append(x)

  return result_list


def inplace_halo_exchange(
    z_list: FlowFieldVal,
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
    z_list: A list of length nz of tensors of shape (nx, ny), where nx, ny and
      nz are the number of points along the axes of a (sub)grid.
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
  boundary_conditions = boundary_conditions or [[None, None]] * len(dims)

  assert len(dims) == len(replica_dims)
  assert len(dims) == len(periodic_dims)
  assert len(dims) == len(boundary_conditions)

  with tf.name_scope("HaloExchange"):
    for (dim, replica_dim, periodic, bc) in zip(dims, replica_dims,
                                                periodic_dims,
                                                boundary_conditions):
      bc_low, bc_high = bc if bc else (None, None)
      _validate_boundary_condition(bc_low, z_list, dim, width)
      _validate_boundary_condition(bc_high, z_list, dim, width)

      left_or_top_padding = (top_padding, left_padding, 0)[dim]

      # Apply halo exchange for each plane in `width`, one by one, from
      # innermost to outermost (`plane = width - 1` to `plane = 0`). The
      # `plane` index is relative to a low set of boundary planes ordered along
      # dim (that is, from outermost to innermost). Calling
      # `_inplace_halo_exchange_1d` with planes in this order is necessary in
      # order for Neumann boundary conditions to be applied correctly.
      for plane in range(width - 1, -1, -1):
        # Select the relevant planes from the sequence of bc planes.
        if bc_low:
          # Create a mutable copy of the bc passed in.
          bc_low_plane = list(bc_low)
          # If the boundary condition is a list of planes select the relevant
          # one.
          bc_low_plane[1] = (
              bc_low_plane[1]
              if isinstance(bc_low_plane[1], float) else bc_low_plane[1][plane])
        else:
          bc_low_plane = None
        if bc_high:
          # Create a mutable copy of the bc passed in.
          bc_high_plane = list(bc_high)
          # If the boundary condition is a list of planes select the relevant
          # one.
          bc_high_plane[1] = (
              bc_high_plane[1] if isinstance(bc_high_plane[1], float) else
              bc_high_plane[1][width - plane - 1])
        else:
          bc_high_plane = None

        z_list = _inplace_halo_exchange_1d(z_list, dim,
                                           replica_id, replicas, replica_dim,
                                           bool(periodic), bc_low_plane,
                                           bc_high_plane, width, plane,
                                           left_or_top_padding)

    return z_list


def _validate_boundary_condition(bc, z_list, dim, width):
  """Checks the validity of the boundary condition.

  A boundary condition must be `None` or a sequence of length 2. If not `None`,
  the first element is a BCType and the second element is a sequence of length
  `width` describing an ordered set of 2D planes orthogonal to `dim`. If `dim`
  is 2 (z-axis) each element is a tensor of shape (nx, ny). If `dim` is 0 or 1
  (x- or y-axis), each element is a sequence of length nz of tensors of shape
  (1, ny) or (nx, 1), respectively. nx, ny and nz are the number of grid points
  along the x, y and z axes of a (sub)grid.

  Args:
    bc: A boundary condition for one side of one axis.
    z_list: A list of length nz of tensors of shape (nx, ny), where nx, ny and
      nz are the number of points along the axes of a (sub)grid.
    dim: The dimension or axis (0, 1 or 2 for x, y or z, respectively).
    width: The width of the boundary (same as the width used in halo exchange).

  Returns:
    None.

  Raises:
    ValueError if the boundary condition is invalid.
  """
  if not bc:
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

  if dim == 2:
    # Each element of bc_value must be a tensor of shape (nx, ny).
    for bc_tensor in bc_value:
      if not isinstance(bc_tensor, tf.Tensor):
        raise ValueError("If dim = 2 a boundary condition must be a float or a "
                         "list of tensors. Found list of {}.".format(
                             type(bc_tensor)))
      if bc_tensor.shape != z_list[0].shape:
        raise ValueError("If dim = 2 a boundary condition must be a float or a "
                         "list of tensors of shape {}. Found tensor in list "
                         "with shape {}.".format(z_list[0].shape,
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
      if len(bc_z_list) != len(z_list):
        raise ValueError("If dim is 0 or 1 each z-list should be length {}. "
                         "Found length {}.".format(len(z_list), len(bc_z_list)))
      for bc_tensor in bc_z_list:
        if not isinstance(bc_tensor, tf.Tensor):
          raise ValueError(
              "If dim is 0 or 1 the boundary condition should be "
              "a list of list of tensor. Found list of list of {}.".format(
                  type(bc_tensor)))
        expected_shape = z_list[0].shape.as_list()
        expected_shape[dim] = 1
        if expected_shape != bc_tensor.shape.as_list():
          raise ValueError("For dim = {} the boundary condition should be a "
                           "float or a list of list of tensor of shape {}. "
                           "Found tensor with shape {}.".format(
                               dim, expected_shape, bc_tensor.shape))


def get_edge_of_3d_field(
    z_list: FlowFieldVal,
    dim: int,
    side: SideType,
    width: int = 1) -> Union[FlowFieldVal, Sequence[FlowFieldVal]]:
  """Helper function that returns an edge of a 3D field.

  Args:
    z_list: A list of length nz of tensors of shape (nx, ny), where nx, ny and
      nz are the number of points along the axes of a (sub)grid.
    dim: The dimension or axis (0, 1 or 2 for x, y or z).
    side: A SideType enum indicating the low or high side of the axis.
    width: The width of the edge.

  Returns:
    A list of length `width`. The elements of the list are 2D tensors of shape
      (nx, ny) if dim = 2, and are [list of length nz of 2D tensors of shape
      (1, ny) or (nx, 1)] if dim = 0 or 1, respectively.
  """
  if dim == 2:
    return (z_list[:width] if side == SideType.LOW else z_list[-width:])

  x_or_y_slice = [slice(None)] * 2
  wlist_of_zlist = []
  for plane in range(width):
    slice_index = (plane if side == SideType.LOW else plane - width)
    slice_index_plus_1 = None if slice_index + 1 == 0 else slice_index + 1
    x_or_y_slice[dim] = slice(slice_index, slice_index_plus_1)
    zlist = [x[tuple(x_or_y_slice)] for x in z_list]
    wlist_of_zlist.append(zlist)
  return wlist_of_zlist


def _convert_zlist_to_2d_tensor(list_of_tensors, dim):
  """Concats a list of tensors along dimension dim.

  The usual use case is to convert a z-list of length nz of shape (1, ny) or
  (nx, 1) to a 2D tensor of shape (nz, ny) or (nx, nz).

  Args:
    list_of_tensors: A sequence of tensors.
    dim: The dimension to concat along.

  Returns:
    The tensor obtained after applying tf.concat.
  """
  return tf.concat(list_of_tensors, dim)


def _convert_2d_tensor_to_zlist(tensor, dim):
  """Converts a tensor to a list of tensors along dimension `dim`.

  Each tensor has length 1 along dimension `dim`. The usual use case is to
  convert a 2D tensor of shape (nz, ny) or (nx, nz) into a list of nz tensors of
  shape (1, ny) or (nx, 1).

  Args:
    tensor: A tensor.
    dim: The dimension to split along.

  Returns:
    The list of tensors obtained after applying tf.split.
  """
  nz = tensor.shape.as_list()[dim]
  return tf.split(tensor, nz, dim)


def clear_halos(x: FlowFieldVal, halo_width: int) -> FlowFieldVal:
  """Sets value inside halos to zero.

  Sets halo values to zero for a 3D field in each TPU core. It is needed in
  cases where the values in the halo shouldn't be used or are otherwise invalid.
  For example, when doing a dot product across TPUs, the halos store
  information that are already accounted for in another TPU. Computing the dot
  product without zeroing out the halos will cause some values in the two
  vectors to be counted twice.

  Args:
    x: A list of 2D `tf.Tensor`s representing a 3D field.
    halo_width: The width of the halo.

  Returns:
    A list of `tf.Tensor` corresponding to the 3D field with the interior
      unchanged and the halo values set to zero.

  Raises:
    ValueError if halo width is too large, and there are no interior points.
  """
  if (halo_width <= 0) or (not x):
    return x

  nz = len(x)

  # Edge case: Invalid inputs if any dimension has no interior points at all.
  nx, ny = x[0].shape
  if min((nx, ny, nz)) <= halo_width * 2:
    raise ValueError(
        "Please double check sizes (nx, ny, nz) = ({}, {}, {}) <= 2 * "
        "halo_width ({}), so there are no interior points.".format(
            nx, ny, nz, halo_width))

  padding = tf.constant(
      [[halo_width, halo_width], [halo_width, halo_width]],
      dtype=tf.int64 if x[0].dtype is tf.float64 else tf.int32)
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
