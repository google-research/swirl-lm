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

# Copyright 2023 Google LLC
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
"""Utility library for solving the radiative transfer equation (RTE)."""

import inspect
from typing import Callable, List, Sequence, Tuple

import numpy as np
from swirl_lm.communication import halo_exchange
from swirl_lm.utility import common_ops
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf


FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap


class RTEUtils:
  """A library for distributing radiative transfer computations on TPU's.

  Attributes:
    params: An instance of `GridParametrization` containing the grid dimensions
    and information about the TPU computational topology.
    num_cores: A 3-tuple containing the number of cores assigned to each
      dimension.
    grid_size: The local grid dimensions per core.
    halos: The number of halo points on each face of the grid.
  """

  def __init__(
      self,
      params: grid_parametrization.GridParametrization,
  ):
    self.params = params
    self.num_cores = (params.cx, params.cy, params.cz)
    self.grid_size = (params.nx, params.ny, params.nz)
    self.halos = params.halo_width

  def slice(
      self,
      f: types.FlowFieldVal,
      dim: int,
      idx: int,
      face: int,
  ) -> FlowFieldVal:
    """Slices a plane from `f` normal to `dim`."""
    face_slice = common_ops.get_face(f, dim, face, idx)
    if isinstance(f, tf.Tensor) or dim != 2:
      # Remove the outer list.
      return face_slice[0]
    return face_slice

  def _append(
      self,
      a: FlowFieldVal,
      b: FlowFieldVal,
      dim: int,
      forward: bool = True,
  ) -> FlowFieldVal:
    """Appends `a` to `b` along `axis` if `forward` and `b` to `a` otherwise."""
    if not forward:
      a, b = b, a
    a_is_tensor = isinstance(a, tf.Tensor)
    assert a_is_tensor == isinstance(b, tf.Tensor)

    if not a_is_tensor and dim == 2:  # Append to Python list.
      return a + b

    if a_is_tensor:
      # Handles the case of single 3D tensor. Shifts `dim` to conform with the
      # 2-0-1 3D tensor orientation.
      axis = (dim + 1) % 3
    else:
      axis = dim

    return tf.nest.map_structure(
        lambda x, y: tf.concat([x, y], axis=axis), a, b
    )

  def _pad(
      self,
      f: FlowFieldVal,
      low_n: int,
      high_n: int,
      dim: int
  ) -> FlowFieldVal:
    """Pads the field with zeros along the dimension `dim`."""
    paddings = [(0, 0)] * 3
    paddings[dim] = (low_n, high_n)
    return common_ops.pad(f, paddings)

  def _generate_adjacent_pair_assignments(
      self, replicas: np.ndarray, axis: int, forward: bool
  ) -> List[np.ndarray]:
    """Creates groups of source-target TPU devices along `axis`.

    The group assignments are used by `tf.raw.CollectivePermute` to exchange
    data between neighboring replicas along `axis`. There will be one such group
    for every interface of the topology. As an example consider a `replicas` of
    shape `[4, 2, 1]` where the replica id's are 0 through 7. Each replica is
    assigned to a unique 3-tuple `coordinate`. In this example, the mapping of
    coordinates to replica ids is `{(0, 0, 0): 0, (0, 1, 0): 1, (1, 0, 0): 2,
    (1, 1, 0): 3, (2, 0, 0): 4, (2, 1, 0): 5, (3, 0, 0): 6, (3, 1, 0): 7}` with
    each element following the form `(coordinate[0], coordinate[1],
    coordinate[2]): replica_id`. The group assignment along dimension 0 contains
    the replica ids with the same coordinate of dimensions 1 and 2 that neighbor
    each other along `axis` in the direction of increasing index if `forward` is
    set to `True`, and in the direction of decreasing index otherwise. In this
    example, if `axis` is 0 and `forward` is `True`, the assignments are:

    [[[0, 2], [1, 3]],
     [[[2, 4], [3, 5]],
     [[[4, 6], [5, 7]]].

    If `forward` is `False`, the pairs for each inteface are reversed:

    [[2, 0], [3, 1]],
     [[4, 2], [5, 3]],
     [[6, 4], [7, 5]]].

    Args:
      replicas: The mapping from the core coordinate to the local replica id.
      axis: The axis along which data will be propagated.
      forward: Whether the data propagation of `tf.raw_ops.CollectivePermute`
        will unravel in the direction of increasing index along `axis`.

    Returns:
      A list of groups of adjacent replica id pairs. There will be one such
      group for every interface of the computational topology along `axis`.
    """
    groups = common_ops.group_replicas(replicas, axis=axis)
    pair_groups = []
    depth = groups.shape[1]
    for i in range(depth - 1):
      pair_group = groups[:, i : i + 2]
      if not forward:
        # Reverse the order of the source-target pairs.
        pair_group = pair_group[:, ::-1]
      pair_groups.append(pair_group.tolist())
    return pair_groups

  def _local_recurrent_op(
      self,
      recurrent_fn: Callable[..., tf.Tensor],
      variables: FlowFieldMap,
      dim: int,
      n: int,
      forward: bool = True,
  ) -> Tuple[FlowFieldVal, FlowFieldVal]:
    """Computes a sequence of recurrent operations along a dimension.

    Each core performs the same operation on data local to it independently.
    Note that the initial input `x0` in `variables` is not included in the final
    output.

    Args:
      recurrent_fn: The local cumulative recurrent operation.
      variables: A dictionary containing the local fields that will be inputs to
        `recurrent_fn`. One of the entries must be `x0`, which should correspond
        to the boundary solution that initiates the recurrence. `x0` has the
        same structure as other fields in `variables` (i.e. either a 3-D
        tf.Tensor or a list of 2D `tf.Tensor`s) but has a size of 1 along the
        axis determined by `dim`.
      dim: The physical dimension along which the sequence of affine
        transformations will be applied.
      n: The number of layers in the final solution.
      forward: Whether the accumulation starts with the first layer of
        coefficients. If set to False, then the recurrence relation unravels
        from the last layer to the first as follows:
        x[i] = recurrent_fn(x[i + 1]).

    Returns:
      A tuple containing 1) A 3D variable with the cumulative output from the
      chain of recurrent transformations having the same structure and shape as
      any field in `variables` that is not `x0`. 2) the 2D output of the last
      recurrent transformation.
    """
    x = variables['x0']

    face = 0 if forward else 1

    for i in range(n):
      prev_idx = i - 1
      plane_args = {
          k: self.slice(v, dim, i, face)
          for k, v in variables.items()
          if k != 'x0'
      }
      plane_args['x0'] = (
          x if i == 0 else self.slice(x, dim, prev_idx, face)
      )
      arg_lst = [
          plane_args[k] for k in inspect.getfullargspec(recurrent_fn).args
      ]
      next_layer = tf.nest.map_structure(recurrent_fn, *arg_lst)
      x = next_layer if i == 0 else self._append(x, next_layer, dim, forward)

    last_local_layer = self.slice(x, dim, n - 1, face)

    return x, last_local_layer

  def _cumulative_recurrent_op_sequential(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      recurrent_fn: Callable[..., FlowFieldVal],
      variables: FlowFieldMap,
      dim: int,
      forward: bool = True,
  ) -> FlowFieldVal:
    """Computes a sequence of recurrent affine transformations globally.

    This particular implementation is sequential, so every layer of replicas
    along the accumulation axis needs to wait for the previous computational
    layer to complete before proceeding, which can be slow. On the other hand,
    this approach has a very small memory overhead, since the TPU communication
    only happens between pairs of adjacent computational layers through the
    `tf.raw_ops.CollectivePermute` operation.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      recurrent_fn: The local cumulative recurrent operation.
      variables: A dictionary containing the local fields that will be inputs to
        `recurrent_fn`. One of the entries must be `x0`, which should correspond
        to the boundary solution that initiates the recurrence.
      dim: The physical dimension along which the sequence of affine
        transformations will be applied.
      forward: Whether the accumulation starts with the first layer of
        coefficients. If set to False, then the recurrence relation unravels
        from the last layer to the first as follows:
        x[i] = w[i] * x[i + 1] + b[i].

    Returns:
      A 3D variable with the cumulative output from the chain of recurrent
      transformations having the same structure and shape as any field in
      `variables` that is not `x0`.
    """
    halos = [0] * 3
    halos[dim] = self.halos

    n = self.grid_size[dim] - 2 * self.halos

    # Remove halos along the axis.
    kwargs = {
        k: common_ops.strip_halos(v, halos)
        for k, v in variables.items()
        if k != 'x0'
    }
    kwargs['x0'] = variables['x0']

    def local_fn(x0: FlowFieldVal) -> Tuple[FlowFieldVal, FlowFieldVal]:
      """Generates the output of a cumulative operation and its last layer."""
      kwargs['x0'] = x0
      return self._local_recurrent_op(recurrent_fn, kwargs, dim, n, forward)

    def communicate_fn(x):
      return tf.raw_ops.CollectivePermute(
          input=x, source_target_pairs=pair_group
      )

    core_idx = common_ops.get_core_coordinate(replicas, replica_id)[dim]

    # Cumulative local output and its last layer. This result will only be valid
    # for the first level of the computational topology. All the subsequent
    # levels will need to wait for the output from the previous level before
    # evaluating their local function.
    x_cum, x_out = local_fn(x0=variables['x0'])

    pair_groups = self._generate_adjacent_pair_assignments(
        replicas, dim, forward
    )
    n_groups = len(pair_groups)
    interface_iter = range(n_groups) if forward else reversed(range(n_groups))

    # Sequentially evaluate a level of the computational topology and propagate
    # results to the next level.
    for i in interface_iter:
      pair_group = pair_groups[i]
      # Send / receive the last recurrent output layer.
      x_prev = tf.nest.map_structure(communicate_fn, x_out)
      # Index of the next set of cores receiving the data.
      recv_core_idx = i + 1 if forward else i
      x_cum, x_out = tf.cond(
          tf.equal(core_idx, recv_core_idx),
          true_fn=lambda: local_fn(x0=x_prev),  # pylint: disable=cell-var-from-loop
          false_fn=lambda: (x_cum, x_out))

    # Pad the result with halo layers.
    return self._pad(x_cum, self.halos, self.halos, dim)

  def _global_concatenate(
      self,
      operand: tf.Tensor,
      group_assignment: np.ndarray,
      dim: int,
  ) -> tf.Tensor:
    """Concatenates tensors across replicas.

    Args:
      operand: A subgrid of a tensor.
      group_assignment: A 2d int32 list with shape `[num_groups,
        num_replicas_per_group]`. It is assumed that the size of group is the
        same for all groups.
      dim: Dimension along which to concatenate tensors across replicas.

    Returns:
      A scalar that is the global value for operator(operand).
    """
    num_replicas = len(group_assignment[0])
    local_val = tf.repeat(operand, num_replicas, dim)

    return tf.raw_ops.AllToAll(
        input=local_val,
        group_assignment=group_assignment,
        concat_dimension=dim,
        split_dimension=dim,
        split_count=num_replicas)

  def cum_prod(
      self,
      f: FlowFieldVal,
      dim: int,
      forward: bool,
  ) -> FlowFieldVal:
    """Computes the cumulative product of a 3D variable along a direction.

    The cumulative product is inclusive, meaning the first plane of the output
    along `dim` matches the first plane of `f`.

    Args:
      f: A 3D variable.
      dim: The dimension along which to accumulate the product.
      forward: Whether the product should be computed forward (True) or in the
        reverse direction (False).

    Returns:
      A 3D variable with the same structure and shape as `f` having the
      cumulative product of the elements of `f` along `dim`, inclusive.
    """
    f_is_tensor = isinstance(f, tf.Tensor)
    reverse = not forward

    if not f_is_tensor and dim == 2:
      cum_prod = [tf.zeros_like(f[0])] * len(f)
      prod = tf.ones_like(f[0])
      idx_iter = range(len(f)) if forward else reversed(range(len(f)))
      for i in idx_iter:
        prod *= f[i]
        cum_prod[i] = prod
    else:
      if f_is_tensor:
        axis = (dim + 1) % 3
      else:
        axis = dim
      cum_prod = tf.nest.map_structure(
          lambda x: tf.math.cumprod(x, axis=axis, reverse=reverse), f
      )

    return cum_prod

  def cumulative_recurrent_affine_op_parallel(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      w: FlowFieldVal,
      b: FlowFieldVal,
      x0: FlowFieldVal,
      dim: int,
      forward: bool,
  ) -> FlowFieldVal:
    """Computes a sequence of recurrent affine transformations globally.

    Given 3D coefficients `w` and 3D bias terms `b`, this accumulates the
    outputs of a chain of pointwise affine transformations along the given
    dimension `dim` on the initial input plane `x0` following the recurrence
    relation:

    x[i] = w[i] * x[i - 1] + b[i]
    if `forward` is True.

    If `forward` is False, then the following recurrence relation is used:

    x[i] = w[i] * x[i + 1] + b[i].

    This accumulation happens globally across possibly multiple TPU cores,
    ignoring the halo layers along the dimension `dim`. The output is padded so
    its shape conforms to the shape of the input variables. The boundary value
    near the face where the recurrent operation is initiated is set to `x0`. The
    boundary value at the opposite face just repeats the values in the outermost
    fluid layers.

    This implementation computes a partial local output first and then combines
    this local output with data from all the other replicas via an
    `tf.raw_ops.AllToAll` operation. This reduces idle time significantly as
    each core is able to work independently until the global communication
    happens. Furthermore, the TPU communication happens directly
    between all pairs of cores in the same axis without going through
    intermediate nodes in the computational topology.

    Since this algorithm involves a local concatenation of 2D planes from all
    the TPU replicas along an axis, the HBM memory overhead can be potentially
    large given enough cores stacked along that axis. If the memory requirement
    becomes excessive, one should consider using the sequential implementation
    instead.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      w: The 3D variable that will be used as the coefficients of the affine
        transformation.
      b: The 3D variable that will be used as the additive (bias) terms of the
        affine transformation.
      x0: The 2D plane that is the input to the first affine transformation.
      dim: The physical dimension along which the sequence of affine
        transformations will be applied.
      forward: Whether the accumulation starts with the first layer of
        coefficients. If set to False, then the recurrence relation unravels
        from the last layer to the first as follows: x[i] = w[i] * x[i + 1] +
        b[i].

    Returns:
      A 3D variable having the same structure and shape as `w` holding the
      cumulative output of the sequence of affine transformations and having
      `x0` as the boundary value at the face that initiates the recurrence.
    """
    halos = [0] * 3
    halos[dim] = self.halos

    # Remove halos along the axis.
    w = common_ops.strip_halos(w, halos)
    b = common_ops.strip_halos(b, halos)

    n = self.grid_size[dim] - 2 * self.halos
    face = 0 if forward else 1

    field_is_tensor = isinstance(b, tf.Tensor)
    num_cores = self.num_cores[dim]

    core_idx = common_ops.get_core_coordinate(replicas, replica_id)[dim]
    first_core_idx = 0 if forward else num_cores - 1

    # Only allow the first layer of cores to use the initial source `x0`. Pass
    # zero otherwise.
    x0 = tf.cond(
        pred=tf.equal(core_idx, first_core_idx),
        true_fn=lambda: x0,
        false_fn=lambda: tf.nest.map_structure(tf.zeros_like, x0)
    )
    local_vars = {'w': w, 'b': b, 'x0': x0}
    affine_op = lambda w, b, x0: w * x0 + b
    local_cum_x, local_x_out = self._local_recurrent_op(
        affine_op, local_vars, dim, n, forward
    )

    local_w_cumprod = self.cum_prod(w, dim, forward)
    local_w_prod = self.slice(local_w_cumprod, dim, n - 1, face)

    group_assignment = common_ops.group_replicas(replicas, dim)

    concat_axis = dim
    if dim == 2 or field_is_tensor:
      concat_axis = (dim + 1) % 3
      # Handle the case of accumulation along the z-list.
      if not field_is_tensor:
        local_x_out = tf.stack(local_x_out)
        local_w_prod = tf.stack(local_w_prod)

    global_w = tf.nest.map_structure(
        lambda w: self._global_concatenate(w, group_assignment, concat_axis),
        local_w_prod,
    )
    global_b = tf.nest.map_structure(
        lambda x: self._global_concatenate(x, group_assignment, concat_axis),
        local_x_out,
    )

    global_x0 = tf.nest.map_structure(
        tf.zeros_like, self.slice(global_b, dim, 0, face)
    )

    # Run the cumulative recurrent affine operation again, this time on the
    # partial outputs of all the cores.
    global_vars = {
        'w': global_w,
        'b': global_b,
        'x0': global_x0,
    }
    global_x, _ = self._local_recurrent_op(
        affine_op, global_vars, dim, num_cores, forward
    )

    prev_core_idx = core_idx - 1 if forward else core_idx + 1
    prev_core_idx = tf.maximum(0, tf.minimum(prev_core_idx, num_cores - 1))

    prev_x = tf.cond(
        pred=tf.equal(core_idx, first_core_idx),
        true_fn=lambda: global_x0,  # Zero.
        false_fn=lambda: self.slice(global_x, dim, prev_core_idx, face=0),
    )

    if not field_is_tensor and dim == 2:
      # Repeat the solution plane of the previous core for all the local layers
      # to enable broadcast-like behavior with `tf.nest.map_structure` below.
      prev_x = [prev_x[0, ...]] * n

    def combine_local_and_global_fn():
      return tf.nest.map_structure(
          lambda x_i, cum_w_i, prev_core_x_i: x_i + prev_core_x_i * cum_w_i,
          local_cum_x,
          local_w_cumprod,
          prev_x,
      )

    val = self._pad(
        tf.cond(
            pred=tf.equal(core_idx, first_core_idx),
            true_fn=lambda: local_cum_x,
            false_fn=combine_local_and_global_fn,
        ), self.halos, self.halos, dim)

    # If the variables are 3D tensors, skip the halo exchange as it is not yet
    # supported for 3D tensors.
    if isinstance(w, tf.Tensor):
      return val

    # If the boundary condition is set along the list dimension, the halo
    # exchange expects the boundary plane represented as a 2D tensor. Otherwise,
    # if set along one of the tensor dimensions, a list of thin tensors of
    # dimension (1, ny) or (nx, 1) is expected.
    if dim == 2 and isinstance(x0, Sequence):
      x0 = x0[0]

    face = 0 if forward else 1
    bc = [[(halo_exchange.BCType.NEUMANN, 0.0)] * 2 for _ in range(3)]
    # Set the boundary plane that initiates the recurrent operation above as the
    # halo values.
    bc[dim][face] = (
        halo_exchange.BCType.DIRICHLET,
        [x0] * self.halos,
    )
    return halo_exchange.inplace_halo_exchange(
        val,
        (0, 1, 2),
        replica_id,
        replicas,
        (0, 1, 2),
        (False, False, False),
        boundary_conditions=bc,
        width=self.halos,
    )

  def cumulative_recurrent_op(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      recurrent_fn: Callable[..., FlowFieldVal],
      variables: FlowFieldMap,
      dim: int,
      forward: bool = True,
  ) -> FlowFieldVal:
    """Applies a recurrent operation globally along a specified dimension.

    This global operation is sequential and will process each layer along `dim`
    at a time in increase order if `forward` is `True` and in decreasing order
    otherwise. As a consequence, if there are multiple TPU cores assigned to
    dimension `dim` in the computational topology some cores will experience
    idleness as they wait for previous layers residing in other cores to be
    processed.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      recurrent_fn: The cumulative recurrent operation.
      variables: A dictionary containing the local fields that will be inputs to
        `recurrent_fn`. One of the entries must be `x0`, which should correspond
        to the boundary solution that initiates the recurrence.
      dim: The physical dimension along which the sequence of affine
        transformations will be applied.
      forward: Whether the accumulation starts with the first layer of
        coefficients. If set to False, then the recurrence relation unravels
        from the last layer to the first as follows:
        x[i] = recurrent_fn({'x0': x[i + 1], ...})

    Returns:
      A 3D variable with the cumulative output from the chain of recurrent
      transformations having the same structure and shape as any field in
      `variables` that is not `x0` and having `x0` as the boundary value at the
      face that initiates the recurrence.
    """
    val = self._cumulative_recurrent_op_sequential(
        replica_id, replicas, recurrent_fn, variables, dim, forward
    )

    # If the variables are 3D tensors, skip the halo exchange as it is not yet
    # supported for 3D tensors.
    x0 = variables['x0']
    if isinstance(x0, tf.Tensor):
      return val
    # If the boundary condition is set along the list dimension, the halo
    # exchange expects the boundary plane represented as a 2D tensor. Otherwise,
    # if set along one of the tensor dimensions, a list of thin tensors of
    # dimension (1, ny) or (nx, 1) is expected.
    elif dim == 2:
      x0 = x0[0]

    face = 0 if forward else 1
    bc = [[(halo_exchange.BCType.NEUMANN, 0.0)] * 2 for _ in range(3)]
    # Set the boundary plane that initiates the recurrent operation above as the
    # halo values.
    bc[dim][face] = (
        halo_exchange.BCType.DIRICHLET,
        [x0] * self.halos,
    )
    return halo_exchange.inplace_halo_exchange(
        val,
        (0, 1, 2),
        replica_id,
        replicas,
        (0, 1, 2),
        (False, False, False),
        boundary_conditions=bc,
        width=self.halos,
    )
