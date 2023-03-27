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

# coding=utf-8
"""Library of the convection scheme in the Navier-Stokes solver."""

from typing import Optional, Text, Tuple

import numpy as np
from swirl_lm.boundary_condition import boundary_condition_utils
from swirl_lm.equations import common
from swirl_lm.numerics import interpolation
from swirl_lm.numerics import numerics_pb2  # pylint: disable=line-too-long
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf

ConvectionScheme = numerics_pb2.ConvectionScheme
FlowFieldVal = types.FlowFieldVal


def first_order_upwinding(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    f: FlowFieldVal,
    f_plus: FlowFieldVal,
    velocity_in_dim: FlowFieldVal,
    grid_spacing: float,
    dim: int,
) -> FlowFieldVal:
  """Computes the first order derivative of a variable in the convection term.

  The derivative to be computed for a variable `f` in the convection term takes
  the form: `velocity_in_dim df / dx`. An upwinding approach is adopted here for
  numerical stability, where the backward difference is used if
  `velocity_in_dim` >= 0, otherwise the forward difference is used.

  Args:
    kernel_op: An object holding a library of kernel operations.
    f: A list of `tf.Tensor` to which the backward difference is applied. Each
      element in the `List` is an `x-y` plane (aka z-slice).
    f_plus: A list of `tf.Tensor` to which the forward difference is applied.
      Each element in the `List` is an `x-y` plane (aka z-slice).
    velocity_in_dim: A list of `tf.Tensor` holding the velocity in the direction
      where the derivative is computed. Each element in the `List` is an `x-y`
      plane (aka z-slice).
    grid_spacing: The mesh size in the direction where the derivative is
      computed.
    dim: The dimension of the derivative, with 0, 1, and 2 being x, y, and z,
      respectively.

  Returns:
    The upwinding first-order derivative of `f`, i.e. `df / dx`. Each element
    in the `List` is an `x-y` plane (aka z-slice).
  """
  grad_forward_fn = [
      lambda g: kernel_op.apply_kernel_op_x(g, 'kdx+'),
      lambda g: kernel_op.apply_kernel_op_y(g, 'kdy+'),
      lambda g: kernel_op.apply_kernel_op_z(g, 'kdz+', 'kdz+sh'),
  ]

  grad_backward_fn = [
      lambda g: kernel_op.apply_kernel_op_x(g, 'kdx'),
      lambda g: kernel_op.apply_kernel_op_y(g, 'kdy'),
      lambda g: kernel_op.apply_kernel_op_z(g, 'kdz', 'kdzsh'),
  ]

  df_dh_forward = [df / grid_spacing for df in  grad_forward_fn[dim](f_plus)]
  df_dh_backward = [df / grid_spacing for df in grad_backward_fn[dim](f)]

  return [
      tf.where(tf.less(velocity_in_dim_, 0), df_dh_forward_, df_dh_backward_)
      for velocity_in_dim_, df_dh_forward_, df_dh_backward_ in zip(
          velocity_in_dim, df_dh_forward, df_dh_backward)
  ]


def central2(kernel_op: get_kernel_fn.ApplyKernelOp, f: FlowFieldVal,
             grid_spacing: float, dim: int) -> FlowFieldVal:
  """Computes the first order derivative using second order centered difference.

  Args:
    kernel_op: An object holding a library of kernel operations.
    f: A list of `tf.Tensor` to which the operator is applied. Each element in
      the `List` is an `x-y` plane (aka z-slice).
    grid_spacing: The mesh size in the direction where the derivative is
      computed.
    dim: The dimension of the derivative, with 0, 1, and 2 being x, y, and z,
      respectively.

  Returns:
    The first-order derivative of `f`, i.e. `df / dx`. Each element in the
    `List` is an `x-y` plane (aka z-slice).
  """
  grad_fn = [
      lambda f: kernel_op.apply_kernel_op_x(f, 'kDx'),
      lambda f: kernel_op.apply_kernel_op_y(f, 'kDy'),
      lambda f: kernel_op.apply_kernel_op_z(f, 'kDz', 'kDzsh'),
  ]

  return [d_f / (2.0 * grid_spacing) for d_f in grad_fn[dim](f)]


def central4(kernel_op: get_kernel_fn.ApplyKernelOp, f: FlowFieldVal,
             grid_spacing: float, dim: int) -> FlowFieldVal:
  """Computes the first order derivative using fourth order centered difference.

  Args:
    kernel_op: An object holding a library of kernel operations.
    f: A list of `tf.Tensor` to which the operator is applied. Each element in
      the `List` is an `x-y` plane (aka z-slice).
    grid_spacing: The mesh size in the direction where the derivative is
      computed.
    dim: The dimension of the derivative, with 0, 1, and 2 being x, y, and z,
      respectively.

  Returns:
    The first-order derivative of `f`, i.e. `df / dx`. Each element in the
    `List` is an `x-y` plane (aka z-slice).
  """
  grad_fn = [
      lambda f: kernel_op.apply_kernel_op_x(f, 'kD4x'),
      lambda f: kernel_op.apply_kernel_op_y(f, 'kD4y'),
      lambda f: kernel_op.apply_kernel_op_z(f, 'kD4z', 'kD4zsh'),
  ]

  return [d_f / (12.0 * grid_spacing) for d_f in grad_fn[dim](f)]


def face_interpolation(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    state: FlowFieldVal,
    pressure: FlowFieldVal,
    dx: float,
    dt: float,
    dim: int,
    bc_types: Tuple[boundary_condition_utils.BoundaryType,
                    boundary_condition_utils.BoundaryType] = (
                        boundary_condition_utils.BoundaryType.UNKNOWN,
                        boundary_condition_utils.BoundaryType.UNKNOWN),
    varname: Optional[Text] = None,
    halo_width: Optional[int] = None,
    src: Optional[FlowFieldVal] = None,
    apply_correction: bool = True,
) -> FlowFieldVal:
  """Interpolates `state` from cells onto faces with the Rhie-Chow correction.

  The Rhie-Chow correction is enforced to remove the numerical fluctuation due
  to the 'checker-board' effect on collocated mesh while interpolating
  cell-centered values onto cell faces, the Rhie-Chow correction is
  enforced [1].

  If a source term is provided, it would be included as an additional correction
  term [2]. For an arbitrary source term s, the correction at face i - 1/2
  (taking the x direction as an example) is performed as:

    u_{i - 1/2} = 0.5 (u_{i - 1} + u_{i}) - 0.5 (s_{i - 1} + s_{i}) +
      0.5 (s_{i - 1} + 0.5 dx ((s_{i} - s_{i - 2}) / (2 dx)) +
           s_{i} - 0.5 dx ((s_{i + 1} - s_{i - 1}) / (2 dx)))
      = 0.5 (u_{i - 1} + u_{i}) +
        0.125 (-s_{i - 2} + s_{i - 1} + s_{i} - s_{i + 1})
  To be consistent with the pressure correction term in terms of storage, node i
  stores information on the i - 1/2 face in the kernel implementation, which is:

  f_{i, j} = -s_{i-2, j} + s_{i-1, j} + s_{i, j} - s_{i+1, j}.

  1. C. M. Rhie, W. L. Chow, Numerical Stud of the Turbulent Flow Past an
  Airfoil with Trailing Edge Separation, AIAA Journal, Vol. 21, No. 11, Nov
  1983.

  2. Zhang, Sijun, Xiang Zhao, and Sami Bayyuk. 2014. “Generalized Formulations
  for the Rhie–Chow Interpolation.” Journal of Computational Physics 258
  (February): 880–914.

  Args:
    kernel_op: An object holding a library of kernel operations.
    replica_id: The index of the current TPU replica.
    replicas: A numpy array that maps grid coordinates to replica id numbers.
    state: A list of `tf.Tensor` representing a 3D volume of velocity (for
      constant density) and momentum (for variable density).
    pressure: A list of `tf.Tensor` representing a 3D volume of pressure.
    dx: The grid spacing in dimension`dim`.
    dt: The time step size that is used in the simulation.
    dim: The dimension that is normal to the face where the interpolation is
      performed.
    bc_types: The type of the boundary conditions on the 2 ends along `dim`.
    varname: The name of the variable.
    halo_width: The number of points in the halo layer in the direction normal
      to a boundary plane.
    src: The source term needs to be included for the face-flux correction.
    apply_correction: An option for whether the Rhie-Chow correction is applied
      when interpolating velocity at faces.

  Returns:
    `state` Interpolated on the face normal to `dim`. Each point stores the face
    value to its left. Because of the width of the Rhie-Chow correction
    operator, values within 2 cells on the smaller index side and 1 cells on the
    larger index side are invalid.

  Raises:
    ValueError if `dim` is not one of 0, 1, and 2.
  """
  rc_weights = {'krc': ([-1.0, 1.0, 1.0, -1.0], 2)}

  if dim == 0:
    interp_type = ['ksx']
    correction_type = ['k3d1x+']
    rc_type = ['krcx']
    kernel_fn = kernel_op.apply_kernel_op_x
  elif dim == 1:
    interp_type = ['ksy']
    correction_type = ['k3d1y+']
    rc_type = ['krcy']
    kernel_fn = kernel_op.apply_kernel_op_y
  elif dim == 2:
    interp_type = ['ksz', 'kszsh']
    correction_type = ['k3d1z+', 'k3d1z+sh']
    rc_type = ['krcz', 'krczsh']
    kernel_fn = kernel_op.apply_kernel_op_z
  else:
    raise ValueError('`dim` has to be 0, 1, or 2. {} is provided.'.format(dim))

  interp = kernel_fn(state, *interp_type)
  correction = kernel_fn(pressure, *correction_type)

  state_face = [0.5 * interp_i for interp_i in interp]

  if apply_correction:
    state_face = [
        state_face_i - dt / 4.0 / dx * correction_i
        for state_face_i, correction_i in zip(state_face, correction)
    ]

    if src is not None:
      kernel_op.add_kernel(rc_weights)
      src_correction = kernel_fn(src, *rc_type)
      state_face = tf.nest.map_structure(lambda a, b: a + (dt / 8) * b,
                                         state_face, src_correction)

  # Set the flux to 0 on the face next to the first fluid layer when it is at a
  # wall, and the flux to be computed is a wall normal velocity component.
  if varname is not None and varname in (common.KEYS_VELOCITY[dim],
                                         common.KEYS_MOMENTUM[dim]):
    nz = len(state)
    nx, ny = state[0].get_shape().as_list()
    n_grid = (nx, ny, nz)[dim]
    n_core = replicas.shape[dim]

    for face in range(2):
      bc_type = bc_types[face]

      if bc_type not in (boundary_condition_utils.BoundaryType.SLIP_WALL,
                         boundary_condition_utils.BoundaryType.NON_SLIP_WALL,
                         boundary_condition_utils.BoundaryType.SHEAR_WALL):
        continue

      if halo_width is None:
        raise ValueError(
            f'`halo_width` must be provided to enforce wall boundary condition '
            f'for {varname}')

      plane_idx = halo_width if face == 0 else n_grid - halo_width
      core_idx = 0 if face == 0 else n_core - 1

      state_face = common_ops.tensor_scatter_1d_update_global(
          replica_id, replicas, tf.nest.map_structure(tf.identity, state_face),
          dim, core_idx, plane_idx, 0.0)

  return state_face  # pytype: disable=bad-return-type


def face_flux_quick(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    state: FlowFieldVal,
    rhou: FlowFieldVal,
    pressure: FlowFieldVal,
    dx: float,
    dt: float,
    dim: int,
    bc_types: Tuple[boundary_condition_utils.BoundaryType,
                    boundary_condition_utils.BoundaryType] = (
                        boundary_condition_utils.BoundaryType.UNKNOWN,
                        boundary_condition_utils.BoundaryType.UNKNOWN),
    varname: Optional[Text] = None,
    halo_width: Optional[int] = None,
    src: Optional[FlowFieldVal] = None,
    apply_correction: bool = True,
) -> FlowFieldVal:
  """Computes the face flux of `state` normal to `dim` with QUICK scheme.

  The QUICK scheme is retrieved from the following reference:
    Leonard, Brian P. "A stable and accurate convective modelling procedure
    based on quadratic upstream interpolation." Computer methods in applied
    mechanics and engineering 19.1 (1979): 59-98,
    https://drive.google.com/open?id=1ck2DCJq_7T1cxzf2MB-0ZmRKhcJEmorq.

  Args:
    replica_id: The index of the current TPU replica.
    replicas: A numpy array that maps grid coordinates to replica id numbers.
    state: A list of `tf.Tensor` representing a 3D volume of the variable for
      which the flux is computed.
    rhou: A list of `tf.Tensor` representing a 3D volume of momentum.
    pressure: A list of `tf.Tensor` representing a 3D volume of pressure.
    dx: The grid spacing in dimension`dim`.
    dt: The time step size that is used in the simulation.
    dim: The dimension that is normal to the face where the interpolation is
      performed.
    bc_types: The type of the boundary conditions on the 2 ends along `dim`.
    varname: The name of the variable.
    halo_width: The number of points in the halo layer in the direction normal
      to a boundary plane.
    src: The source term needs to be included for the face-flux correction.
    apply_correction: An option for whether the Rhie-Chow correction is applied
      when interpolating velocity at faces.

  Returns:
    The flux of `state` on faces that are normal to `dim`.

  Raises:
    ValueError if `dim` is not one of 0, 1, and 2.
  """
  kernel_op = get_kernel_fn.ApplyKernelConvOp(
      4, {'kf2-': ([-0.125, 0.75, 0.375], 2)})

  if dim == 0:
    quick_pos_type = ['kf2-x']
    quick_neg_type = ['kf2x+']
    kernel_fn = kernel_op.apply_kernel_op_x
  elif dim == 1:
    quick_pos_type = ['kf2-y']
    quick_neg_type = ['kf2y+']
    kernel_fn = kernel_op.apply_kernel_op_y
  elif dim == 2:
    quick_pos_type = ['kf2-z', 'kf2-zsh']
    quick_neg_type = ['kf2z+', 'kf2z+sh']
    kernel_fn = kernel_op.apply_kernel_op_z
  else:
    raise ValueError('`dim` has to be 0, 1, or 2. {} is provided.'.format(dim))

  state_pos = kernel_fn(state, *quick_pos_type)
  state_neg = kernel_fn(state, *quick_neg_type)

  rhou_face = face_interpolation(kernel_op, replica_id, replicas, rhou,
                                 pressure, dx, dt, dim, bc_types, varname,
                                 halo_width, src, apply_correction)

  return [
      0.5 * (rhou_i + tf.abs(rhou_i)) * s_pos_i + 0.5 *
      (rhou_i - tf.abs(rhou_i)) * s_neg_i
      for rhou_i, s_pos_i, s_neg_i in zip(rhou_face, state_pos, state_neg)
  ]


def convection_quick(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    state: FlowFieldVal,
    rhou: FlowFieldVal,
    pressure: FlowFieldVal,
    dx: float,
    dt: float,
    dim: int,
    bc_types: Tuple[boundary_condition_utils.BoundaryType,
                    boundary_condition_utils.BoundaryType] = (
                        boundary_condition_utils.BoundaryType.UNKNOWN,
                        boundary_condition_utils.BoundaryType.UNKNOWN),
    varname: Optional[Text] = None,
    halo_width: Optional[int] = None,
    src: Optional[FlowFieldVal] = None,
    apply_correction: bool = True,
) -> FlowFieldVal:
  """Computes the convection term for convservative variables with QUICK scheme.

  Args:
    kernel_op: An object holding a library of kernel operations.
    replica_id: The index of the current TPU replica.
    replicas: A numpy array that maps grid coordinates to replica id numbers.
    state: A list of `tf.Tensor` representing a 3D volume of the variable for
      which the convection term is computed.
    rhou: A list of `tf.Tensor` representing a 3D volume of momentum.
    pressure: A list of `tf.Tensor` representing a 3D volume of pressure.
    dx: The grid spacing.
    dt: The time step size that is used in the simulation.
    dim: The dimension that is normal to the face where the convection term is
      computed.
    bc_types: The type of the boundary conditions on the 2 ends along `dim`.
    varname: The name of the variable.
    halo_width: The number of points in the halo layer in the direction normal
      to a boundary plane.
    src: The source term needs to be corrected for the face-flux correction.
    apply_correction: An option for whether the Rhie-Chow correction is applied
      when interpolating velocity at faces.

  Returns:
    The convection term of `f`. Values within `halo_width` of 2 are invalid.

  Raises:
    ValueError if `dim` is not one of 0, 1, and 2.
  """
  if dim == 0:
    diff_op_type = ['kdx+']
    kernel_fn = kernel_op.apply_kernel_op_x
  elif dim == 1:
    diff_op_type = ['kdy+']
    kernel_fn = kernel_op.apply_kernel_op_y
  elif dim == 2:
    diff_op_type = ['kdz+', 'kdz+sh']
    kernel_fn = kernel_op.apply_kernel_op_z
  else:
    raise ValueError('`dim` has to be 0, 1, or 2. {} is provided.'.format(dim))

  flux = face_flux_quick(replica_id, replicas, state, rhou, pressure, dx, dt,
                         dim, bc_types, varname, halo_width, src,
                         apply_correction)

  return [d_flux / dx for d_flux in kernel_fn(flux, *diff_op_type)]


def face_flux_weno(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    state: FlowFieldVal,
    rhou: FlowFieldVal,
    pressure: FlowFieldVal,
    dim: int,
    bc_types: Tuple[boundary_condition_utils.BoundaryType,
                    boundary_condition_utils.BoundaryType] = (
                        boundary_condition_utils.BoundaryType.UNKNOWN,
                        boundary_condition_utils.BoundaryType.UNKNOWN),
    varname: Optional[Text] = None,
    halo_width: Optional[int] = None,
) -> FlowFieldVal:
  """Computes the face flux of `state` normal to `dim` with WENO scheme.

  Note:
  1. The Lax-Friedrich flux splitting is applied.
  2. Currently only the 5th order WENO scheme is supported, i.e. with k = 3.

  The WENO scheme is retrieved from the following reference:
  Shu, C.-W. (1998). Essentially non-oscillatory and weighted essentially
  non-oscillatory schemes for hyperbolic conservation laws. In Lecture Notes in
  Mathematics (pp. 325–432). https://doi.org/10.1007/bfb0096355

  Args:
    replica_id: The index of the current TPU replica.
    replicas: A numpy array that maps grid coordinates to replica id numbers.
    state: A list of `tf.Tensor` representing a 3D volume of the variable for
      which the flux is computed.
    rhou: A list of `tf.Tensor` representing a 3D volume of momentum.
    pressure: A list of `tf.Tensor` representing a 3D volume of pressure.
    dim: The dimension that is normal to the face where the interpolation is
      performed.
    bc_types: The type of the boundary conditions on the 2 ends along `dim`.
    varname: The name of the variable.
    halo_width: The number of points in the halo layer in the direction normal
      to a boundary plane.

  Returns:
    The flux of `state` on faces that are normal to `dim`.

  Raises:
    ValueError if `dim` is not one of 0, 1, and 2.
  """
  del replica_id, replicas, bc_types, halo_width
  dims = ('x', 'y', 'z')

  if varname is not None and varname in (common.KEYS_VELOCITY[dim],
                                         common.KEYS_MOMENTUM[dim]):
    flux = tf.nest.map_structure(lambda rhou_i, s_i, p_i: rhou_i * s_i + p_i,
                                 rhou, state, pressure)
  else:
    flux = tf.nest.map_structure(tf.math.multiply, rhou, state)

  flux_m = tf.nest.map_structure(lambda f, u, s: 0.5 * (f - tf.abs(u) * s),
                                 flux, rhou, state)
  flux_p = tf.nest.map_structure(lambda f, u, s: 0.5 * (f + tf.abs(u) * s),
                                 flux, rhou, state)

  # Interpolates the scalar value onto faces. Note that the `weno` functions
  # stores value on face i + 1/2 at i.
  f_neg, _ = interpolation.weno(flux_p, dims[dim])
  _, f_pos = interpolation.weno(flux_m, dims[dim])

  return tf.nest.map_structure(tf.math.add, f_neg, f_pos)


def convection_weno(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    state: FlowFieldVal,
    rhou: FlowFieldVal,
    pressure: FlowFieldVal,
    dx: float,
    dim: int,
    bc_types: Tuple[boundary_condition_utils.BoundaryType,
                    boundary_condition_utils.BoundaryType] = (
                        boundary_condition_utils.BoundaryType.UNKNOWN,
                        boundary_condition_utils.BoundaryType.UNKNOWN),
    varname: Optional[Text] = None,
    halo_width: Optional[int] = None,
) -> FlowFieldVal:
  """Computes the convection term for convservative variables with WENO scheme.

  Args:
    kernel_op: An object holding a library of kernel operations.
    replica_id: The index of the current TPU replica.
    replicas: A numpy array that maps grid coordinates to replica id numbers.
    state: A list of `tf.Tensor` representing a 3D volume of the variable for
      which the convection term is computed.
    rhou: A list of `tf.Tensor` representing a 3D volume of momentum.
    pressure: A list of `tf.Tensor` representing a 3D volume of pressure.
    dx: The grid spacing.
    dim: The dimension that is normal to the face where the convection term is
      computed.
    bc_types: The type of the boundary conditions on the 2 ends along `dim`.
    varname: The name of the variable.
    halo_width: The number of points in the halo layer in the direction normal
      to a boundary plane.

  Returns:
    The convection term of `f`. Values within `halo_width` of 3 are invalid.

  Raises:
    ValueError if `dim` is not one of 0, 1, and 2.
  """
  if dim == 0:
    diff_op_type = ['kdx']
    kernel_fn = kernel_op.apply_kernel_op_x
  elif dim == 1:
    diff_op_type = ['kdy']
    kernel_fn = kernel_op.apply_kernel_op_y
  elif dim == 2:
    diff_op_type = ['kdz', 'kdzsh']
    kernel_fn = kernel_op.apply_kernel_op_z
  else:
    raise ValueError('`dim` has to be 0, 1, or 2. {} is provided.'.format(dim))

  flux = face_flux_weno(replica_id, replicas, state, rhou, pressure, dim,
                        bc_types, varname, halo_width)

  return [d_flux / dx for d_flux in kernel_fn(flux, *diff_op_type)]


def convection_upwinding_1(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    state: FlowFieldVal,
    rhou: FlowFieldVal,
    pressure: FlowFieldVal,
    dx: float,
    dt: float,
    dim: int,
    bc_types: Tuple[boundary_condition_utils.BoundaryType,
                    boundary_condition_utils.BoundaryType] = (
                        boundary_condition_utils.BoundaryType.UNKNOWN,
                        boundary_condition_utils.BoundaryType.UNKNOWN),
    varname: Optional[Text] = None,
    halo_width: Optional[int] = None,
    src: Optional[FlowFieldVal] = None,
) -> FlowFieldVal:
  """Computes the convection term with first order upwinding scheme.

  Args:
    kernel_op: An object holding a library of kernel operations.
    replica_id: The index of the current TPU replica.
    replicas: A numpy array that maps grid coordinates to replica id numbers.
    state: A list of `tf.Tensor` representing a 3D volume of the variable for
      which the convection term is computed.
    rhou: A list of `tf.Tensor` representing a 3D volume of momentum.
    pressure: A list of `tf.Tensor` representing a 3D volume of pressure.
    dx: The grid spacing.
    dt: The time step size that is used in the simulation.
    dim: The dimension that is normal to the face where the convection term is
      computed
    bc_types: The type of the boundary conditions on the 2 ends along `dim`.
    varname: The name of the variable.
    halo_width: The number of points in the halo layer in the direction normal
      to a boundary plane.
    src: The source term needs to be included for the face-flux correction.

  Returns:
    The convection term of `f`. Values within `halo_width` of 1 are invalid.

  Raises:
    ValueError if `dim` is not one of 0, 1, and 2.
  """
  interp_fn = [
      lambda f: kernel_op.apply_kernel_op_x(f, 'ksx'),
      lambda f: kernel_op.apply_kernel_op_y(f, 'ksy'),
      lambda f: kernel_op.apply_kernel_op_z(f, 'ksz', 'kszsh'),
  ]

  grad_fn = [
      lambda f: kernel_op.apply_kernel_op_x(f, 'kdx+'),
      lambda f: kernel_op.apply_kernel_op_y(f, 'kdy+'),
      lambda f: kernel_op.apply_kernel_op_z(f, 'kdz+', 'kdz+sh'),
  ]

  rhou_face = face_interpolation(kernel_op, replica_id, replicas, rhou,
                                 pressure, dx, dt, dim, bc_types, varname,
                                 halo_width, src)
  sc_face = interp_fn[dim](state)

  rho_u_f = [rhou_i * state_i for rhou_i, state_i in zip(rhou_face, sc_face)]

  return [d_rho_u_f / dx for d_rho_u_f in grad_fn[dim](rho_u_f)]


def convection_central_2(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    state: FlowFieldVal,
    rhou: FlowFieldVal,
    pressure: FlowFieldVal,
    dx: float,
    dt: float,
    dim: int,
) -> FlowFieldVal:
  """Compute the convection term with the second order central scheme.

  Args:
    kernel_op: An object holding a library of kernel operations.
    state: A list of `tf.Tensor` representing a 3D volume of the variable for
      which the convection term is computed.
    rhou: A list of `tf.Tensor` representing a 3D volume of momentum.
    pressure: A list of `tf.Tensor` representing a 3D volume of pressure.
    dx: The grid spacing.
    dt: The time step size that is used in the simulation.
    dim: The dimension that is normal to the face where the convection term is
      computed

  Returns:
    The convection term of `f`. Values within `halo_width` of 1 are invalid.

  Raises:
    ValueError if `dim` is not one of 0, 1, and 2.
  """
  del pressure, dt

  flux = tf.nest.map_structure(tf.math.multiply, rhou, state)
  return central2(kernel_op, flux, dx, dim)


def convection_term(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    state: FlowFieldVal,
    rhou: FlowFieldVal,
    pressure: FlowFieldVal,
    dx: float,
    dt: float,
    dim: int,
    bc_types: Tuple[boundary_condition_utils.BoundaryType,
                    boundary_condition_utils.BoundaryType] = (
                        boundary_condition_utils.BoundaryType.UNKNOWN,
                        boundary_condition_utils.BoundaryType.UNKNOWN),
    varname: Optional[Text] = None,
    halo_width: Optional[int] = None,
    scheme: ConvectionScheme = ConvectionScheme.CONVECTION_SCHEME_QUICK,
    src: Optional[FlowFieldVal] = None,
    apply_correction: bool = True,
) -> FlowFieldVal:
  """Computes the convection term df/dx with selected scheme.

  Args:
    kernel_op: An object holding a library of kernel operations.
    replica_id: The index of the current TPU replica.
    replicas: A numpy array that maps grid coordinates to replica id numbers.
    state: A list of `tf.Tensor` representing a 3D volume of the variable for
      which the convection term is computed.
    rhou: A list of `tf.Tensor` representing a 3D volume of momentum.
    pressure: A list of `tf.Tensor` representing a 3D volume of pressure.
    dx: The grid spacing.
    dt: The time step size that is used in the simulation.
    dim: The dimension that is normal to the face where the convection term is
      computed
    bc_types: The type of the boundary conditions on the 2 ends along `dim`.
    varname: The name of the variable.
    halo_width: The number of points in the halo layer in the direction normal
      to a boundary plane.
    scheme: The numerical scheme to be used to compute the convection term.
    src: The source term needs to be corrected for the face-flux correction.
    apply_correction: An option for whether the Rhie-Chow correction is applied
      when interpolating velocity at faces.

  Returns:
    The convection term of `f`. Values within `halo_width` of 2 are invalid.

  Raises:
    NotImplementedError if scheme is not one of the following:
    CONVECTION_SCHEME_QUICK, CONVECTION_SCHEME_UPWIND_1,
    CONVECTION_SCHEME_CENTRAL_2.
  """
  if scheme == ConvectionScheme.CONVECTION_SCHEME_QUICK:
    return convection_quick(kernel_op, replica_id, replicas, state, rhou,
                            pressure, dx, dt, dim, bc_types, varname,
                            halo_width, src, apply_correction)
  elif scheme == ConvectionScheme.CONVECTION_SCHEME_WENO:
    return convection_weno(kernel_op, replica_id, replicas, state, rhou,
                           pressure, dx, dim, bc_types, varname, halo_width)
  elif scheme == ConvectionScheme.CONVECTION_SCHEME_UPWIND_1:
    return convection_upwinding_1(kernel_op, replica_id, replicas, state, rhou,
                                  pressure, dx, dt, dim, bc_types, varname,
                                  halo_width, src)
  elif scheme == ConvectionScheme.CONVECTION_SCHEME_CENTRAL_2:
    return convection_central_2(kernel_op, state, rhou, pressure, dx, dt, dim)
  else:
    raise NotImplementedError(
        '{} is not implemented. Available options are: '
        '{}, {}, {}, {}.'.format(
            numerics_pb2.ConvectionScheme.Name(scheme),
            numerics_pb2.ConvectionScheme.Name(
                ConvectionScheme.CONVECTION_SCHEME_QUICK),
            numerics_pb2.ConvectionScheme.Name(
                ConvectionScheme.CONVECTION_SCHEME_WENO),
            numerics_pb2.ConvectionScheme.Name(
                ConvectionScheme.CONVECTION_SCHEME_UPWIND_1),
            numerics_pb2.ConvectionScheme.Name(
                ConvectionScheme.CONVECTION_SCHEME_CENTRAL_2)))
