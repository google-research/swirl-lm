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

"""A library of the immersed boundary method."""

import itertools
from typing import Callable, Dict, Optional, Sequence, Text

from absl import logging
import numpy as np
from swirl_lm.base import initializer
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.boundary_condition import immersed_boundary_method_pb2
from swirl_lm.communication import halo_exchange
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap
InterpFnType = Callable[[FlowFieldVal], FlowFieldVal]
InitFn = initializer.ValueFunction
ThreeIntTuple = initializer.ThreeIntTuple


def _apply_3d_kernel(
    f: FlowFieldVal,
    kernel_op: get_kernel_fn.ApplyKernelOp,
    kernel_name_x: Text,
    kernel_name_y: Text,
    kernel_name_z: Text,
    kernel_name_zsh: Text,
) -> FlowFieldVal:
  """Applies the kernel to `f` in 3 dimensions and returns the summed result.

  NB: This function assumes that the input tensors already have the correct
  halos in place.

  Args:
    f: Variable to which to apply the kernel.
    kernel_op: An ApplyKernelOp instance to use in computing the update.
    kernel_name_x: Name of the kernel to use in the x direction.
    kernel_name_y: Name of the kernel to use in the y direction.
    kernel_name_z: Name of the kernel to use in the z direction.
    kernel_name_zsh: Name of the kernel to use for the shift in the z direction.

  Returns:
    Summed result of applying the kernel to `f` in 3 dimensions.
  """
  result_x = kernel_op.apply_kernel_op_x(f, kernel_name_x)
  result_y = kernel_op.apply_kernel_op_y(f, kernel_name_y)
  result_z = kernel_op.apply_kernel_op_z(f, kernel_name_z, kernel_name_zsh)
  return tf.nest.map_structure(
      lambda a, b, c: a + b + c, result_x, result_y, result_z
  )


def ib_info_map(
    ib_info: immersed_boundary_method_pb2.ImmersedBoundaryMethod,
) -> Dict[str, immersed_boundary_method_pb2.IBVariableInfo]:
  """Flattens the variable information in the IB config."""

  def flatten_ib_info(
      ib_vars: Sequence[immersed_boundary_method_pb2.IBVariableInfo],
  ) -> Dict[str, immersed_boundary_method_pb2.IBVariableInfo]:
    """Flattens a sequence of IB info."""
    out = {}
    for info in ib_vars:
      out[info.name] = info
    return out

  ib_type = ib_info.WhichOneof('type')
  if ib_type == 'cartesian_grid':
    return flatten_ib_info(ib_info.cartesian_grid.variables)
  elif ib_type == 'mac':
    return flatten_ib_info(ib_info.mac.variables)
  elif ib_type == 'sponge':
    return flatten_ib_info(ib_info.sponge.variables)
  elif ib_type == 'direct_forcing':
    return flatten_ib_info(ib_info.direct_forcing.variables)
  elif ib_type == 'direct_forcing_1d_interp':
    return flatten_ib_info(ib_info.direct_forcing_1d_interp.variables)
  elif ib_type == 'feedback_force_1d_interp':
    return flatten_ib_info(ib_info.feedback_force_1d_interp.variables)
  else:
    raise NotImplementedError(
        f'{ib_type} is not a valid IB type. Available options are:'
        ' "cartesian_grid", "mac", "sponge", "direct_forcing", '
        '"direct_forcing_1d_interp", "feedback_force_1d_interp".'
    )


def update_cartesian_grid_method_boundary_coefficients(
    boundary: FlowFieldVal,
    interior_mask: FlowFieldVal,
    kernel_op: get_kernel_fn.ApplyKernelOp,
    kernel_name_x: Text = 'kSx',
    kernel_name_y: Text = 'kSy',
    kernel_name_z: Text = 'kSz',
    kernel_name_zsh: Text = 'kSzsh',
) -> FlowFieldVal:
  """Updates the non-zero values of `boundary` with the correct coefficients.

  Helper function to update the non-zero values in the boundary layer mask
  with pre-computed coeffcients for the Cartesian grid method. The coefficients
  are set to 1./Nb, where Nb is the number of neighboring fluid cells. The
  kernel used should be the same as the one used when applying the Cartesian
  grid method. If Nb = 0, the coefficient is set to zero, in effect removing
  the grid point from the boundary layer since it has no neighboring fluid
  cells.

  NB: This function assumes that the input tensors already have the correct
  halos in place.

  Args:
    boundary: boundary layer mask for the Cartesian grid method before
      coefficient update. The grid points at the solid/fluid immersed boundary
      should be equal to one, while all other points are zero.
    interior_mask: Solid interior mask for the Cartesian grid method.
    kernel_op: An ApplyKernelOp instance to use in computing the update.
    kernel_name_x: Name of the kernel to use in the x direction. Defaults to
      central sum.
    kernel_name_y: Name of the kernel to use in the y direction. Defaults to
      central sum.
    kernel_name_z: Name of the kernel to use in the z direction. Defaults to
      central sum.
    kernel_name_zsh: Name of the kernel to use for the shift in the z direction.
      Defaults to central sum.

  Returns:
    `boundary` with pre-computed coefficients.
  """
  mask_sum = _apply_3d_kernel(interior_mask, kernel_op, kernel_name_x,
                              kernel_name_y, kernel_name_z, kernel_name_zsh)
  return tf.nest.map_structure(tf.math.divide_no_nan, boundary, mask_sum)


def get_fluid_solid_interface_z(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    ib_interior_mask: FlowFieldVal,
    halo_width: int,
) -> FlowFieldVal:
  """Generates a mask for the fluid layer in contact with the solid in z.

  Args:
    kernel_op: An object that holds the library of finite difference kernel
      operations.
    replica_id: The index of the TPU replica.
    replicas: A 3D array the stores the TPU topology.
    ib_interior_mask: A 3D tensor that holds a binary mask where fluid is 1 and
      solid is 0.
    halo_width: The size of the halo layer.

  Returns:
    A 3D tensor with the fluid layer contacting the solid (in the z direction)
    being 1, and everywhere else being 0.
  """
  # Get the fluid solid interface in the z direction. This operation will
  # generate a 3D tensor with the fluid being 2, the layers of fluid and soild
  # that contacts each other being 1, and the solid being 0.
  fluid_solid_mask = kernel_op.apply_kernel_op_z(ib_interior_mask, 'kSz',
                                                 'kSzsh')
  # Make only the solid-fluid interface 1, and everywhere else 0.
  fluid_solid_interface_z = tf.nest.map_structure(
      lambda mask_i: tf.compat.v1.where(
          tf.greater(mask_i, 1.0), tf.zeros_like(mask_i), mask_i
      ),
      fluid_solid_mask,
  )

  # Leave only the fluid layer as 1, and everywhere else 0.
  fluid_solid_mask = tf.nest.map_structure(
      tf.math.multiply, fluid_solid_interface_z, ib_interior_mask
  )

  # Assign 0 in the halos in the z direction assuming that there's no fluid
  # solid interface at the boundary of the domain.
  halo_dims = (2,)
  replica_dims = (2,)
  return halo_exchange.inplace_halo_exchange(
      fluid_solid_mask,
      halo_dims,
      replica_id,
      replicas,
      replica_dims,
      periodic_dims=None,
      boundary_conditions=[
          [(halo_exchange.BCType.DIRICHLET, 0.0),
           (halo_exchange.BCType.DIRICHLET, 0.0)],
      ],
      width=halo_width)


def interp_1d_coeff_init_fn(
    surface: tf.Tensor,
    dim: int,
    n_cores: initializer.ThreeIntTuple,
) -> InitFn:
  """Generates an `init_fn` for a tensor that interpolates `surface` along dim.

  Args:
    surface: A 2D tensor that defines a surface that cuts through `dim` as a
      function of the 2 coordinates perpendicular to `dim`. The size of it must
      equal to the size of the coordinates perpendicular to `dim` with halos
      excluded.
    dim: The dimension along which `surface` is interpolated onto.
    n_cores: The number of cores along the 3 dimensions.

  Returns:
    An `init_fn` that initializes the interpolation coefficients of surface in
    each core. Grid points that are not adjacent to `surface` are set to 0.
  """
  n_total = tf.shape(surface)
  kept_dims = [0, 1, 2]
  del kept_dims[dim]
  n_non_dim_cores = tf.gather(n_cores, kept_dims)

  # This implicitly checks that the length of `n_total` and `n_non_dim_cores`
  # are both 2, and expclicitly checks the divisibility. Note this assertion
  # works since this part is executed outside the TPU/not in the replica
  # context.
  tf.debugging.Assert(
      tf.math.reduce_all(tf.math.equal(
          tf.math.floormod(n_total, n_non_dim_cores), [0, 0])),
      [n_total, n_non_dim_cores], summarize=2)

  core_n = tf.math.floordiv(n_total, n_non_dim_cores)

  def init_fn(
      xx: tf.Tensor,
      yy: tf.Tensor,
      zz: tf.Tensor,
      lx: float,
      ly: float,
      lz: float,
      coord: initializer.ThreeIntTuple,
  ) -> tf.Tensor:
    """Computes the interpolation coefficients for surface."""
    del lx, ly, lz

    mesh_size_non_dim = tf.gather(tf.shape(xx), kept_dims)

    # Check if dimension of `surface` matches the mesh. Since assertion ops are
    # ignored on TPUs, we explicitly place the assertion ops on CPU using
    # outside_compilation() (by default this section will be on TPU since this
    # is executed under the replica context.
    def _check_mesh_size(a, b):
      op = tf.debugging.Assert(
          tf.math.reduce_all(tf.math.equal(a, b)),
          [a, b],
          summarize=2)
      with tf.control_dependencies([op]):
        return a, b

    local_core_n, _ = tf.compat.v1.tpu.outside_compilation(
        _check_mesh_size, *(core_n, mesh_size_non_dim)
    )
    # Get part of the surface that is local to the current core.
    coord = tf.gather(coord, kept_dims)
    surface_local = tf.expand_dims(
        tf.slice(surface, local_core_n * coord, local_core_n), dim)

    # Compute the interpolation weights. Because the point where the surface
    # cuts through the axis along `dim` has to fall between a mesh grid,
    # denoted by an interval [lo, hi] with a distance delta between lo and hi
    # being the grid spacing, the interpolation coefficient at each grid point
    # needs to be computed twice based on its 2 neighboring intervals. We
    # combine the 2 tensors after eliminating values on nodes that are not
    # adjacent to the surface to form the valid interpolation tensor.
    # In the case where the immersed boundary falls on a node, both tensors will
    # have a value 1 at that node. Here we consider the one computed with the
    # lower limit of the interval only.
    grid = (xx, yy, zz)[dim]
    delta = tf.experimental.numpy.diff(grid, axis=dim)
    paddings_lo = [[0, 0], [0, 0], [0, 0]]
    paddings_lo[dim] = [0, 1]
    surface_to_lo_coeff = tf.abs(
        (surface_local - grid) / tf.pad(delta, paddings_lo, 'SYMMETRIC')
    )
    paddings_hi = [[0, 0], [0, 0], [0, 0]]
    paddings_hi[dim] = [1, 0]
    surface_to_hi_coeff = tf.abs(
        (surface_local - grid) / tf.pad(delta, paddings_hi, 'SYMMETRIC')
    )
    interp_tensor_lo = tf.where(
        tf.logical_and(
            tf.less_equal(surface_to_lo_coeff, 1.0),
            tf.less_equal(zz, surface_local),
        ),
        1.0 - surface_to_lo_coeff,
        tf.zeros_like(surface_to_lo_coeff),
    )
    interp_tensor_hi = tf.where(
        tf.logical_and(
            tf.less(surface_to_hi_coeff, 1.0), tf.greater(zz, surface_local)
        ),
        1.0 - surface_to_hi_coeff,
        tf.zeros_like(surface_to_hi_coeff),
    )

    return interp_tensor_lo + interp_tensor_hi

  return init_fn


def get_fluid_solid_interface_value_z(
    replicas: np.ndarray,
    value: FlowFieldVal,
    ib_boundary_mask: FlowFieldVal,
) -> tf.Tensor:
  """Retrieves values from the fluid layer that contacts solid in z.

  It's assumed that there's only one fluid-solid interface in the z direction.

  Args:
    replicas: A 3D array the stores the TPU topology.
    value: A 3D tensor of flow field variable.
    ib_boundary_mask: A 3D tensor that holds a binary mask where fluid layer
      that contacts the solid is 1, and everywhere else is 0.

  Returns:
    A 2D tf.Tensor that contains values of the flow field variable in the fluid
    layer contacting the solid.
  """
  interface_val = tf.zeros_like(value[0])
  for i in range(len(value)):
    interface_val += value[i] * ib_boundary_mask[i]

  cx, cy, _ = replicas.shape
  group_assignment = np.array(
      [replicas[i, j, :] for i, j in itertools.product(range(cx), range(cy))])
  sum_op = lambda f: tf.math.reduce_sum(f, axis=0)

  return common_ops.global_reduce(
      tf.expand_dims(interface_val, 0), sum_op, group_assignment)


class ImmersedBoundaryMethod(object):
  """A library of the immersed boundary method."""

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    """Initializes the immersed boundary method library."""
    self._replica_dims = (0, 1, 2)
    self._halo_dims = (0, 1, 2)

    self._params = params
    assert (
        boundary_models := params.boundary_models
    ) is not None, '`boundary_models` must be set in the config.'
    self._ib_params = boundary_models.ib

  @property
  def type(self):
    """Provides the type of immersed boundary method for this instance."""
    return self._ib_params.WhichOneof('type')

  def update_states(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      boundary_conditions: Dict[Text, halo_exchange.BoundaryConditionsSpec],
  ) -> FlowFieldMap:
    """Updates `states` in the immersed boundary at each sub-iteration."""
    if self.type == 'cartesian_grid':
      return self._apply_cartesian_grid_method(kernel_op, replica_id, replicas,
                                               states, additional_states,
                                               boundary_conditions)
    elif self.type == 'mac':
      return self._apply_marker_and_cell_method(replica_id, replicas, states,
                                                additional_states,
                                                boundary_conditions)
    elif self.type in (
        'sponge',
        'direct_forcing',
        'feedback_force_1d_interp',
        'direct_forcing_1d_interp',
    ):
      return states
    else:
      raise ValueError(f'{self.type} is not a valid IB type.')

  def update_additional_states(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Updates `additional_states` at the beginning of each time step."""
    if self.type in (
        'cartesian_grid',
        'mac',
        'direct_forcing',
        'direct_forcing_1d_interp',
    ):
      return additional_states
    elif self.type == 'sponge':
      return self._apply_rayleigh_damping_method(kernel_op, replica_id,
                                                 replicas, states,
                                                 additional_states)
    elif self.type == 'feedback_force_1d_interp':
      return self._apply_feedback_force_1d_interp(
          kernel_op, states, additional_states
      )
    else:
      raise ValueError(f'{self.type} is not a valid IB type.')

  def update_forcing(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Updates `additional_states` during each subiteration."""
    del replica_id, replicas

    if self.type in (
        'cartesian_grid',
        'mac',
        'sponge',
        'feedback_force_1d_interp',
    ):
      return additional_states
    elif self.type == 'direct_forcing':
      return self._apply_direct_forcing_method(states, additional_states)
    elif self.type == 'direct_forcing_1d_interp':
      return self._apply_direct_forcing_1d_interp(
          kernel_op, states, additional_states
      )
    else:
      raise ValueError(f'{self.type} is not a valid IB type.')

  def generate_initial_states(
      self,
      coordinates: ThreeIntTuple,
      ib_flow_field_mask_fn: InitFn,
      ib_boundary_mask_fn: Optional[InitFn] = None,
  ) -> FlowFieldMap:
    """Generates initial states required by the IB model requested.

    Args:
      coordinates: A tuple that specifies the replica's grid coordinates in
        physical space.
      ib_flow_field_mask_fn: A function that produces a 3D tensor with values of
        ones and zeros, where 1 represents the flow field, and 0 represents the
        solid. The function takes 3 `tf.Tensor` representing the 3D mesh, and 3
        `float` representing the physical length of each dimension.
      ib_boundary_mask_fn: A function that produces a 3D tensor with values of
        ones and zeros, where 1 represents the solid-fluid interface, and 0
        represents elsewhere. The function takes 3 `tf.Tensor` representing the
        3D mesh, and 3 `float` representing the physical length of each
        dimension. The ib_boundary field is only used in the Cartesian grid
        method and the marker-and-cell method.

    Returns:
      A dictionary of required states by the selected IB method. The values of
      these states are set to zeros for the forcing terms.
    """

    def states_init(init_fn: InitFn) -> tf.Tensor:
      """Assigns value to a tensor with `init_fn`.

      Args:
        init_fn: A function that takes the local mesh_grid tensor for the core
          (in order x, y, z) and the global characteristic length floats (in
          order x, y, z) and returns a 3-D tensor representing the value for the
          local core (without including the margin/overlap between the cores).

      Returns:
        A 3D tensor with values assigned by `init_fn`.
      """
      return initializer.partial_mesh_for_core(
          self._params,
          coordinates,
          init_fn,
          pad_mode='SYMMETRIC',
          mesh_choice=initializer.MeshChoice.PARAMS,
      )

    def init_fn_zeros(xx: tf.Tensor, yy: tf.Tensor, zz: tf.Tensor, lx: float,
                      ly: float, lz: float, coord: ThreeIntTuple) -> tf.Tensor:
      """Creates a 3D tensor with value 0 that has the same size as `xx`."""
      del yy, zz, lx, ly, lz, coord
      return tf.zeros_like(xx, dtype=xx.dtype)

    output = {'ib_interior_mask': states_init(ib_flow_field_mask_fn)}
    if self.type == 'sponge':
      ib_boundary_included = False

      for variable in self._ib_params.sponge.variables:
        force_name = self.ib_force_name(variable.name)
        output.update({force_name: states_init(init_fn_zeros)})

        # Allocate a placeholder for the fluid-solid interface mask. True values
        # of this mask need to be initialized separately, e.g. in the preprocess
        # function.
        if (not ib_boundary_included and variable.bc
            == immersed_boundary_method_pb2.IBVariableInfo.NEUMANN_Z):
          ib_boundary_fn = (
              ib_boundary_mask_fn
              if ib_boundary_mask_fn is not None else init_fn_zeros)
          output.update({'ib_boundary': states_init(ib_boundary_fn)})
          ib_boundary_included = True
    elif self.type == 'feedback_force_1d_interp':
      output['ib_boundary'] = states_init(ib_boundary_mask_fn)

      for variable in self._ib_params.sponge.variables:
        force_name = self.ib_force_name(variable.name)
        output.update({force_name: states_init(init_fn_zeros)})
    elif self.type in ('cartesian_grid', 'mac'):
      output['ib_boundary'] = states_init(ib_boundary_mask_fn)

    return output

  def _exchange_halos(self, f, replica_id, replicas, boundary_conditions):
    """Performs halo change for the variable f."""
    return halo_exchange.inplace_halo_exchange(
        f,
        self._halo_dims,
        replica_id,
        replicas,
        self._replica_dims,
        periodic_dims=self._params.periodic_dims,
        boundary_conditions=boundary_conditions,
        width=self._params.halo_width)

  def _update_state_in_solid(
      self,
      f: FlowFieldVal,
      boundary_mask: FlowFieldVal,
      interior_mask: FlowFieldVal,
      sign: float,
      masked_value: float,
      interp_fn: InterpFnType,
  ) -> FlowFieldVal:
    """Updates states `f` inside the solid body and the solid-fluid interface.

    The values inside the solid are set to `masked_value`, and the values on the
    boundary are extrapolated from the neighboring fluid cells.

    NB: This function assumes that the input tensors already have the correct
    halos in place. It does not exchange the halos at the end of the update,
    since this is handled by the calling function `update_immersed_boundary`.

    Args:
      f: Variable to which to apply the Cartesian grid method.
      boundary_mask: A tensor where only the grid points at the solid/fluid
        immersed boundary are non-zero. Typically this is a 1-cell thick layer
        right outside of the solid interior mask. These grid points correspond
        to where the solid/fluid boundary cuts through the regular cells, and
        represent the fluid/solid transition. In order to save computation
        within the N-S solver loop, the values in the non-zero grid points of
        the `boundary_mask` are pre-computed coefficients. The pre-computed
        coefficients should be equal to 1./Nb, where Nb is the number of
        neighboring fluid cells using the same kernel as in this function.
      interior_mask: A tensor where all the grid points that correspond to cells
        located 100% inside the solid are set to zero, while all the others are
        set to one.
      sign: -1 for mirror flow (default), 1 for extrapolation, and 0 for not
        updating values in the solid with information infered from fluid.
      masked_value: Value to which to set the values that are inside the solid
        body mask.
      interp_fn: A function that infers the value at the fluid-solid interface
        from neiboring nodes.

    Returns:
      `f` with Cartesian grid method update.

    Raises:
      ValueError if `ib` is missing the required mask tensors for the method.
    """
    f = tf.nest.map_structure(
        lambda boundary_i, mask_i, f_boundary, f_i: tf.compat.v1.where(
            boundary_i > 0.0, sign * f_boundary, mask_i * f_i
        ),
        boundary_mask,
        interior_mask,
        interp_fn(f),
        f,
    )
    if masked_value != 0:

      def updated_mask(mask_i: tf.Tensor, boundary_i: tf.Tensor):
        """Generates a mask for points in the boundary and fluid."""
        return tf.math.logical_or(
            tf.greater(mask_i, 0.0), tf.greater(boundary_i, 0.0))

      f = tf.nest.map_structure(
          lambda f_i, mask_i, b_i: tf.compat.v1.where(
              updated_mask(mask_i, b_i), f_i, masked_value * tf.ones_like(f_i)
          ),
          f,
          interior_mask,
          boundary_mask,
      )

    return f

  def _apply_cartesian_grid_method(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      boundary_conditions: Dict[Text, halo_exchange.BoundaryConditionsSpec],
  ) -> FlowFieldMap:
    """Updates states inside the solid body and the solid-fluid interface."""

    def fluid_node_avg(val: FlowFieldVal) -> FlowFieldVal:
      """Computes the average of neiboring fluid nodes at the boundary."""
      # Mask out the values inside the solid.
      val = tf.nest.map_structure(
          tf.math.multiply, val, additional_states['ib_interior_mask']
      )
      # Compute the average over the neighboring fluid values.
      val_sum = _apply_3d_kernel(val, kernel_op, 'kSx', 'kSy', 'kSz', 'kSzsh')
      return tf.nest.map_structure(
          tf.math.multiply, additional_states['ib_boundary'], val_sum
      )

    states_new = {}
    states_new.update(states)
    for variable in self._ib_params.cartesian_grid.variables:
      if variable.name not in states.keys():
        # Not raising an error here to allow incomplete set of state variable
        # updates to save cost.
        continue
      sign = (-1.0 if variable.bc
              == immersed_boundary_method_pb2.IBVariableInfo.DIRICHLET else 1.0)
      state_with_ib = self._update_state_in_solid(
          states[variable.name],
          additional_states['ib_boundary'],
          additional_states['ib_interior_mask'],
          sign,
          variable.value,
          interp_fn=fluid_node_avg)
      states_new.update({
          variable.name:
              self._exchange_halos(state_with_ib, replica_id, replicas,
                                   boundary_conditions[variable.name])
      })

    return states_new

  def _apply_marker_and_cell_method(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      boundary_conditions: Dict[Text, halo_exchange.BoundaryConditionsSpec],
  ) -> FlowFieldMap:
    """Applies the marker-and-cell method."""

    def neumann_z(val: FlowFieldVal) -> FlowFieldVal:
      """Shifts `val` down by 1 index to mimic Neumann BC in z direction."""
      return tf.concat([val[1:], tf.zeros_like(val[:1], dtype=val[0].dtype)],
                       axis=0)

    states_new = {}
    states_new.update(states)
    for variable in self._ib_params.mac.variables:
      if variable.name not in states.keys():
        # Not raising an error here to allow incomplete set of state variable
        # updates to save cost.
        continue
      sign = (0.0 if variable.bc
              == immersed_boundary_method_pb2.IBVariableInfo.DIRICHLET else 1.0)
      state_with_ib = self._update_state_in_solid(
          states[variable.name],
          additional_states['ib_boundary'],
          additional_states['ib_interior_mask'],
          sign,
          variable.value,
          interp_fn=neumann_z)
      states_new.update({
          variable.name:
              self._exchange_halos(state_with_ib, replica_id, replicas,
                                   boundary_conditions[variable.name])
      })

    return states_new

  def ib_force_name(self, var_name: Text) -> Text:
    """Generates the name of the force term for `var_name` in the solid.

    The format of the force term is 'src_[var_name]'.

    Args:
      var_name: The name of variable to which the immersed boundary method is
        applied.

    Returns:
      The name of the force term corresponds to the input variable name.
    """
    return 'src_{}'.format(var_name)

  def ib_rhs_name(self, var_name: Text) -> Text:
    """Generates the name of the right hand side for `var_name` in the solid.

    The format of the force term is 'rhs_[var_name]'.

    Args:
      var_name: The name of variable to which the immersed boundary method is
        applied.

    Returns:
      The name of the right hand side corresponds to the input variable name.
    """
    return 'rhs_{}'.format(var_name)

  def _apply_rayleigh_damping_method(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Generates the Rayleigh Damping forcing term inside the solid."""
    del kernel_op, replica_id

    def update_sponge_force(
        value: tf.Tensor,
        target_value: tf.Tensor,
        original_force: tf.Tensor,
        damping_coeff: float,
        mask: tf.Tensor,
        override: bool,
    ) -> tf.Tensor:
      """Generates the sponge force for variable `var_name` within the solid."""
      a_max = np.power(damping_coeff * self._params.dt, -1)

      force = -a_max * (value - target_value) * (1.0 - mask)
      return force if override else original_force + force

    additional_states_new = {}
    additional_states_new.update(additional_states)
    for variable in self._ib_params.sponge.variables:
      if variable.name not in states.keys():
        logging.warn('%s is not a valid state. Available states are: %r',
                     variable.name, states.keys())
        continue

      force_name = self.ib_force_name(variable.name)
      if force_name not in additional_states.keys():
        raise ValueError(
            '{} needs to be initialized to use the Rayleigh '
            'damping approach of the immersed boundary method.'.format(
                force_name))

      if variable.bc == immersed_boundary_method_pb2.IBVariableInfo.NEUMANN_Z:
        target_value = get_fluid_solid_interface_value_z(
            replicas, states[variable.name], additional_states['ib_boundary'])
      else:
        target_value = variable.value

      damping_coeff = variable.damping_coeff if variable.HasField(
          'damping_coeff') else self._ib_params.sponge.damping_coeff

      additional_states_new.update({
          force_name: tf.nest.map_structure(
              lambda value, original_force, mask: update_sponge_force(
                  value,
                  target_value,  # pylint: disable=cell-var-from-loop
                  original_force,
                  damping_coeff,  # pylint: disable=cell-var-from-loop
                  mask,
                  variable.override,  # pylint: disable=cell-var-from-loop
              ),
              states[variable.name],
              additional_states[force_name],
              additional_states['ib_interior_mask'],
          )
      })

    return additional_states_new

  def _apply_direct_forcing_method(
      self,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Updates the equation right hand side with the direct forcing method.

    Reference:
    [1] Zhang, N., and Z. C. Zheng. 2007. “An Improved Direct-Forcing
       Immersed-Boundary Method for Finite Difference Applications.” Journal of
       Computational Physics 221 (1): 250–68.

    Args:
      states: Field variables to which the immersed boundary method are applied.
      additional_states: Helper states that are required to compute the new
        right hand side function. Must contain 'ib_interior_mask' and
        'rhs_[w+]', where 'w+' is the name of the state that this right hand
        side function belongs to.

    Returns:
      A dictionary of right hand side functions updated by the direct forcing
      immersed boundary method.

    Raises:
      ValueError: If 'ib_interior_mask' is not in `additional_states`, or no
      `rhs_[w+]` variable found for that variable.
    """
    if 'ib_interior_mask' not in additional_states.keys():
      raise ValueError('"ib_interor_mask" is not found in `additional_states`.')

    def update_rhs(
        value: FlowFieldVal,
        target_value: tf.Tensor | float,
        damping_coeff: float,
        rhs: FlowFieldVal,
        mask: FlowFieldVal,
    ) -> FlowFieldVal:
      """Updates the right hand side function with direct forcing."""
      coeff = np.power(damping_coeff * self._params.dt, -1)
      return tf.nest.map_structure(
          lambda v_i, r_i, m_i: r_i * m_i
          - coeff * (v_i - target_value) * (1.0 - m_i),
          value,
          rhs,
          mask,
      )

    var_dict = {
        variable.name: variable
        for variable in self._ib_params.direct_forcing.variables
    }

    rhs_updated = {}

    for key, value in states.items():
      rhs_name = 'rhs_{}'.format(key)
      if rhs_name not in additional_states.keys():
        raise ValueError('RHS for {} is not provided.'.format(key))

      if key not in var_dict.keys():
        logging.warn(
            'States information for  %s is not provided in the IB. Available '
            'states are: %r. Right hand side for %s is not updated and IB not '
            'applied.', key, var_dict.keys(), key)
        rhs_updated.update({rhs_name: additional_states[rhs_name]})
      else:
        damping_coeff = var_dict[key].damping_coeff if var_dict[key].HasField(
            'damping_coeff') else self._ib_params.direct_forcing.damping_coeff

        rhs_updated.update({
            rhs_name: update_rhs(
                value,
                var_dict[key].value,
                damping_coeff,
                additional_states[rhs_name],
                additional_states['ib_interior_mask'],
            )
        })

    return rhs_updated

  def _ib_1d_interp_force_fn(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      interp_weights: FlowFieldVal,
      dim: int,
  ) -> Callable[[FlowFieldVal, float, float], FlowFieldVal]:
    """Generates a function that computes the IB force with 1D interpolation.

    Args:
      kernel_op: An instance of the kernel operation library that performs
        numerical operations.
      interp_weights: A 3D tensor that stores the interpolation weights for the
        immersed boundary.
      dim: The dimension along which the interpolation is performed along.

    Returns:
      A function that takes a flow field variable, its target value on the IB,
      and a damping coefficient that scales the forcing term as input, and
      returns the force term due to the IB.
    """
    kernel_op.add_kernel({'shift_dn': ([0.0, 0.0, 1.0], 1)})

    sum_op = (
        lambda u: kernel_op.apply_kernel_op_x(u, 'ksx'),
        lambda u: kernel_op.apply_kernel_op_y(u, 'ksy'),
        lambda u: kernel_op.apply_kernel_op_z(u, 'ksz', 'kszsh'),
    )[dim]

    shift_op = (
        lambda u: kernel_op.apply_kernel_op_x(u, 'shift_dnx'),
        lambda u: kernel_op.apply_kernel_op_y(u, 'shift_dny'),
        lambda u: kernel_op.apply_kernel_op_z(u, 'shift_dnz', 'shift_dnzsh'),
    )[dim]

    def get_ib_force(
        u: FlowFieldVal, u_target: float, damping_coeff: float
    ) -> FlowFieldVal:
      """Computes the IB force term for variable `u`."""
      # Get the values on the 2 end points enclosing the immersed boundary, and
      # multiply them by the weights for interpolation.
      u_interp = tf.nest.map_structure(tf.math.multiply, u, interp_weights)

      # Interpolate values on the immersed boundary, and save it at the larger
      # mesh index. For example, if the immersed boundary that falls between
      # k - 1 and k, its value will be saved at k.
      u_ib = sum_op(u_interp)

      # Remove values that are not on the immersed boundary. Here we mask the
      # immersed boundary with non-zero interpolation weights with 1, and set 0
      # elsewhere. A 1-step sum of the mask following the same procedure will
      # make the value of the higher indexed grid point 2, and less than 2
      # everywhere else. Values that corresponds to indices with value 2 are
      # the actual interpolated value of `u` on the immersed boundary.
      ib_mask = tf.nest.map_structure(
          lambda v: tf.where(  # pylint: disable=g-long-lambda
              tf.greater(v, 0.0), tf.ones_like(v), tf.zeros_like(v)
          ),
          interp_weights,
      )
      ib_mask = sum_op(ib_mask)
      u_ib = tf.nest.map_structure(
          lambda v, m: tf.where(tf.greater(m, 1.5), v, tf.zeros_like(v)),
          u_ib,
          ib_mask,
      )

      # Compute the force required to drive the interpolated value on the
      # immersed boundary to the target value.
      beta = np.power(damping_coeff * self._params.dt, -1)
      f_ib = tf.nest.map_structure(lambda u: -beta * (u - u_target), u_ib)

      # Extrapolate the force back to the grid with the same weights. Because
      # the force term is saved at the index that corresponds to the higher end
      # of the interval only, we need to replicate it to the lower index.
      f_ib = tf.nest.map_structure(tf.math.add, f_ib, shift_op(f_ib))

      return tf.nest.map_structure(
          tf.math.multiply, f_ib, interp_weights
      )

    return get_ib_force

  def _apply_direct_forcing_1d_interp(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    R"""Updates the RHS with the direct forcing 1D interpolation method.

    Reference:
    [1] Zhang, N., and Z. C. Zheng. 2007. “An Improved Direct-Forcing
       Immersed-Boundary Method for Finite Difference Applications.” Journal of
       Computational Physics 221 (1): 250–68.

    Args:
      kernel_op: An instance of the kernel operation library that performs
        numerical operations.
      states: Field variables to which the immersed boundary method are applied.
      additional_states: Helper states that are required to compute the new
        right hand side function. Must contain 'ib_boundary' and 'rhs_\w+',
        where 'w+' is the name of the state that this right hand side function
        belongs to.

    Returns:
      A dictionary of right hand side functions updated by the direct forcing
      immersed boundary method.

    Raises:
      AssertionError: If 'ib_boundary' is not in
      `additional_states`, or no 'rhs_\w+' variable found for that variable.
    """
    assert 'ib_boundary' in additional_states, (
        '"ib_boundary" is required by the direct forcing IB method with 1D'
        ' interpolation, but is not found in `additional_states`.'
    )

    mask_internal_layer = tf.nest.map_structure(
        lambda w: tf.where(  # pylint: disable=g-long-lambda
            tf.greater(w, 0.0), tf.ones_like(w), tf.zeros_like(w)
        ),
        additional_states['ib_boundary'],
    )

    ib_force_fn = self._ib_1d_interp_force_fn(
        kernel_op,
        additional_states['ib_boundary'],
        self._ib_params.direct_forcing_1d_interp.dim,
    )

    def update_rhs(
        value: FlowFieldVal,
        target_value: float,
        rhs: FlowFieldVal,
    ) -> FlowFieldVal:
      """Updates the right hand side function with direct forcing."""
      ib_force = ib_force_fn(value, target_value, 1.0 / self._params.dt)

      return tf.nest.map_structure(
          lambda m, r, f: (1.0 - m) * r + m * f,
          mask_internal_layer,
          rhs,
          ib_force,
      )

    var_dict = {
        variable.name: variable
        for variable in self._ib_params.direct_forcing_1d_interp.variables
    }

    rhs_updated = {}

    for key, value in states.items():
      rhs_name = self.ib_rhs_name(key)
      assert rhs_name in additional_states, (
          f'RHS for {key} is required by the direct forcing IB method with 1D'
          ' interpolation, but is not provided.'
      )

      if key not in var_dict:
        # Use a warning here instead failing the function because we always
        # go through the IB step for all variables, but applying IB doesn't
        # have to be applied to all of them.
        logging.warn(
            'States information for  %s is not provided in the IB. Available '
            'states are: %r. Right hand side for %s is not updated and IB not '
            'applied.', key, var_dict.keys(), key)
        rhs_updated[rhs_name] = additional_states[rhs_name]
      else:
        rhs_updated[rhs_name] = update_rhs(
            value, var_dict[key].value, additional_states[rhs_name]
        )

    return rhs_updated

  def _apply_feedback_force_1d_interp(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    R"""Computes the feedback force due to IB with 1 dimensional interpolation.

    Here we make the following assumptions:
    1. The immersed boundary can be expressed as a function of the horizontal
       coordinates.
    2. The immersed boundary will be interpolated only along the selected axis,
       i.e., it only has values on the edges of each computational cell.
       Refined representation of the immersed boundary inside computational
       cells are not considered, e.g. a cave is not allowed.

    Reference:
    [1] Zhang, N., and Z. C. Zheng. 2007. “An Improved Direct-Forcing
       Immersed-Boundary Method for Finite Difference Applications.” Journal of
       Computational Physics 221 (1): 250–68.
    [2] Saiki, E. M., & Biringen, S. (1996). Numerical Simulation of a Cylinder
        in Uniform Flow: Application of a Virtual Boundary Method. Journal of
        Computational Physics, 123(2), 450–465.

    Args:
      kernel_op: An ApplyKernelOp instance to use in computing the update.
      states: Field variables to which the immersed boundary method are applied.
      additional_states: Helper states that are required to compute the new
        right hand side function. Must contain 'ib_boundary', which stores the
        interpolation weights above and below the immersed boundary.

    Returns:
      A dictionary of force terms with names `src_\w+` for variables that
      requires the IB closure.

    Raises:
      AssertionError: If 'ib_boundary' is not in `additional_states`.
    """
    assert (
        'ib_boundary' in additional_states
    ), '`ib_boundary` is required for to apply the feedback force method.'

    ib_force_fn = self._ib_1d_interp_force_fn(
        kernel_op,
        additional_states['ib_boundary'],
        self._ib_params.feedback_force_1d_interp.dim,
    )

    var_dict = {
        variable.name: variable
        for variable in self._ib_params.feedback_force_1d_interp.variables
    }

    ib_force = {}

    for key, value in states.items():
      if key not in var_dict.keys():
        continue

      damping_coeff = (
          var_dict[key].damping_coeff
          if var_dict[key].HasField('damping_coeff')
          else self._ib_params.feedback_force_1d_interp.damping_coeff
      )

      ib_force[f'src_{key}'] = ib_force_fn(
          value, var_dict[key].value, damping_coeff
      )

    return ib_force


def immersed_boundary_method_factory(
    params: parameters_lib.SwirlLMParameters,
) -> Optional[ImmersedBoundaryMethod]:
  """Constructs an `ImmersedBoudnaryMethod` object.

  Args:
    params: The configuration context of a simulation.

  Returns:
    An `ImmersedBoundaryMethod` object if immersed boundary method is requested
    in the config, otherwise returns `None`.
  """
  if params.boundary_models is None or not params.boundary_models.HasField(
      'ib'):
    return None

  return ImmersedBoundaryMethod(params)
