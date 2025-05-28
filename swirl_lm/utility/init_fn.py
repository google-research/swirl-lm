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

# coding=utf-8
"""A library of functions that generates initial conditions for simulations.

This library provides `init_fn` that are possibly shared by different
simulations. The returned value of these functions provides a dictionary whose
keys are the names variables to be initialized, and values are the `init_fn` of
that variable.
"""

from typing import List, Mapping, Optional, Text, Union

import numpy as np
import scipy.integrate as sp_integrate
import scipy.optimize as sp_optimize
from swirl_lm.base import initializer
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf

_NP_DTYPE = types.NP_DTYPE
_TF_DTYPE = types.TF_DTYPE

InitFn = initializer.ValueFunction
InitFnDict = Mapping[Text, InitFn]
ThreeIntTuple = initializer.ThreeIntTuple


def constant_init_fn(value: float) -> InitFn:
  """Initializes a field with const float value.

  Args:
    value: The value for the field.

  Returns:
    A function that generates a field with a const `value`.
  """

  def init_fn(xx: tf.Tensor, yy: tf.Tensor, zz: tf.Tensor, lx: float, ly: float,
              lz: float, coord: ThreeIntTuple) -> tf.Tensor:
    """Generates a field as a const `value`."""
    del xx, yy, lx, ly, lz, coord
    return value * tf.ones_like(zz, dtype=zz.dtype)

  return init_fn


def normal_distribution_init_fn(
    mean: Optional[float],
    std: float,
    mean_init_fn: Optional[InitFn] = None,
    eps: float = 1e-6,
    seed: Optional[int] = None,
) -> InitFn:
  """Initializes a float field with a normal distribution elementwise.

  Args:
    mean: The optional mean value for the field.
    std: The standard deviation for the field (non-negative).
    mean_init_fn: The optional mean initial value function for the field.
    eps: The minimum threshold for `std` (non-negative).
    seed: An optional int as a seed to the normal distribution.

  Returns:
    A function that generates a field with a normal distribution elementwise.

  Raises:
    ValueError: When neither or both of `mean` and `mean_init_fn` are `None`, or
      when `0 < eps <= std` is violated.
  """
  if (mean is None) == (mean_init_fn is None):
    raise ValueError(
        'Expecting one and only one of `mean` or `mean_init_fn` to be valid.')

  if eps <= 0 or std < eps:
    raise ValueError(
        '`std` and `eps` are expected to be positive, and `std >= eps`.')

  def init_fn(xx: tf.Tensor, yy: tf.Tensor, zz: tf.Tensor, lx: float, ly: float,
              lz: float, coord: ThreeIntTuple) -> tf.Tensor:
    """Generates a field with a normal distribution elementwise."""
    if mean is None:
      # Note that `mean_val` doesn't have to be homogeneous from `mean_init_fn`.
      mean_val = mean_init_fn(xx, yy, zz, lx, ly, lz, coord)
    else:
      mean_val = tf.ones_like(zz, dtype=zz.dtype) * mean

    return (
        tf.random.normal(
            tf.shape(mean_val), stddev=std, dtype=mean_val.dtype, seed=seed
        )
        + mean_val
    )

  return init_fn


def blasius_boundary_layer(
    u_inf: float,
    v_inf: float,
    nu: float,
    dx: float,
    dy: float,
    lz: float,
    nz: float,
    x: float,
    elevation: Optional[tf.Tensor] = None,
    apply_transition: bool = True,
    transition_fraction: float = 0.5,
    ) -> InitFnDict:
  """Generates `init_fn` for velocity with wall in the lower z dimension.

  The initial profile for the velocity is computed from the Blasius boundary
  layer formulation, which is:
    2f''' + f''f = 0,
    with:
      f(0) = 0,
      f'(0) = 0,
      f'(∞) = 1.
  This equation is solved with the shooting method.

  Args:
    u_inf: The free stream velocity in the x direction.
    v_inf: The free stream velocity in the y direction.
    nu: The kinematic viscosity.
    dx: The grid spacing in the x direction.
    dy: The grid spacing in the y direction.
    lz: The height of the domain.
    nz: The number of mesh points in the vertical dimension.
    x: The distance from which the boundary layer profile is computed.
    elevation: The map of elevation local to the current TPU core. If it's 2D,
      it's assumed that it provides the elevation information over the x-y
      plane; if it's 3D, it's assumed that the map is replicated over dimension
      2. If it's not provided, it's assumed that the elevation is 0.
    apply_transition: An indicator of whether the boundary layer profile is
      transitioned from surface normal to coordinates aligned. If `True`, the
      boundary layer profile will be transitioned at a fraction of the domain
      height; otherwise it'll stay as the proflie normal to the wall along the
      vertical direction.
    transition_fraction: The fraction in height of boundary layer that's
      considered as normal to the ground.

  Returns:
    A dictionary of `init_fn` for u, v, and w.
  """
  u_mag = np.sqrt(u_inf**2 + v_inf**2)

  # Prepare the elevation map with the correct format. Note that slice kernel
  # is used here to prevent issues from data size is not fully divisible by the
  # kernel size.
  kernel_op = get_kernel_fn.ApplyKernelSliceOp()
  theta = 0.0
  if elevation is None:
    elevation = 0.0
  else:
    if len(elevation.shape) == 3:
      elevation = tf.squeeze(elevation[..., 0])

    dh_dx_0 = kernel_op.apply_kernel_op_x(tf.expand_dims(elevation, 0),
                                          'kdx+') / dx
    dh_dx_c = kernel_op.apply_kernel_op_x(tf.expand_dims(elevation, 0),
                                          'kDx') / (2.0 * dx)
    dh_dx_e = kernel_op.apply_kernel_op_x(tf.expand_dims(elevation, 0),
                                          'kdx') / dx
    dh_dx = tf.concat(
        [dh_dx_0[0, 0:1, :], dh_dx_c[0, 1:-1, :], dh_dx_e[0, -2:-1, :]], axis=0)

    dh_dy_0 = kernel_op.apply_kernel_op_y(tf.expand_dims(elevation, 0),
                                          'kdy+') / dy
    dh_dy_c = kernel_op.apply_kernel_op_y(tf.expand_dims(elevation, 0),
                                          'kDy') / (2.0 * dy)
    dh_dy_e = kernel_op.apply_kernel_op_y(tf.expand_dims(elevation, 0),
                                          'kdy') / dy
    dh_dy = tf.concat(
        [dh_dy_0[0, :, 0:1], dh_dy_c[0, :, 1:-1], dh_dy_e[0, :, -2:-1]], axis=1)

    theta = tf.math.asin((u_inf * dh_dx + v_inf * dh_dy) / tf.math.sqrt(
        (1.0 + dh_dx**2 + dh_dy**2)) / u_mag)

    elevation = tf.expand_dims(elevation, 2)
    theta = tf.expand_dims(theta, 2)

  # Compute the grid spacing following the mesh definition in the simulation.
  dz = lz / (nz - 1)
  # Add an additional grid point in from to represent the ground. The actual
  # flow field starts from the second points.
  buf = dz * np.ones((nz + 1,), dtype=_NP_DTYPE)
  buf[0] = 0.0
  z = np.cumsum(buf)
  # Compute the dimensionless coordinates for the Blasius equation.
  eta = z * np.sqrt(u_mag / nu / x, dtype=_NP_DTYPE)

  # Compute the bin size that's used to convert z coordinates into indices.
  # The bin size is defined to be slightly smaller than the actual grid spacing
  # to prevent errors in the index due to floating point error.
  # With this definition, for z = k dz, the index is:
  # k' = floor(z / dz') = floor(k dz / (dz (1 - 1 / (2 (nz - 1)))) )
  #    = floor(k / (1 - 1 / (2 (nz - 1)))) = k + floor(1 / (2 (nz - 1)))
  # Note that the last term 1 / (2 (nz - 1)) < 1 for nz > 1, so k' = k.
  # Also note that this approach might still subject to error when nz is large,
  # and the margin between `bin_size` and `dz` is close to machine epsilon.
  bin_size = dz - dz / (nz - 1) / 2.0

  def rhs(y: List[float], t: float) -> List[float]:
    """The right hand side function of the linearized Blasius equation."""
    del t
    return [y[1], y[2], -0.5 * y[2] * y[0]]

  def objective(val):
    """The error in the high end boundary condition."""
    y = sp_integrate.odeint(rhs, [0.0, 0.0, val], eta)
    return 1.0 - y[-1][1]

  val = sp_optimize.newton(objective, 0.0)
  f = np.array(sp_integrate.odeint(rhs, [0.0, 0.0, val], eta), dtype=_NP_DTYPE)

  slope_correction = tf.ones_like(z, dtype=_TF_DTYPE)
  if apply_transition:
    z_transition = transition_fraction * lz
    slope_correction = tf.compat.v1.where(
        tf.less(z, z_transition),
        tf.math.cos(z / z_transition * np.pi / 2)**2,
        tf.zeros_like(z, dtype=_TF_DTYPE))

  def get_values(data: tf.Tensor, zz: tf.Tensor,) -> tf.Tensor:
    """Retrieves values from `f` at height zz."""
    indices = tf.cast(
        tf.maximum(tf.floor((zz - elevation) / bin_size) + 1, 0), tf.int32)
    return tf.gather(data, indices)

  def u_init_fn(
      xx: tf.Tensor,
      yy: tf.Tensor,
      zz: tf.Tensor,
      lx: float,
      ly: float,
      lz: float,
      coord: ThreeIntTuple,
  ) -> tf.Tensor:
    """Computes the value of the velocity component parallel to the wall."""
    del xx, yy, lx, ly, lz, coord
    u_n = u_inf * get_values(f[:, 1], zz)
    w_n = 0.5 * np.sqrt(nu * u_mag / x) * (
        get_values(eta, zz) * get_values(f[:, 1], zz) - get_values(f[:, 0], zz))
    corr = get_values(slope_correction, zz)
    return (u_n * tf.math.cos(theta) -
            w_n * tf.math.sin(theta)) * corr + u_n * (1.0 - corr)

  def v_init_fn(
      xx: tf.Tensor,
      yy: tf.Tensor,
      zz: tf.Tensor,
      lx: float,
      ly: float,
      lz: float,
      coord: ThreeIntTuple,
  ) -> tf.Tensor:
    """Computes the value of the velocity component parallel to the wall."""
    del xx, yy, lx, ly, lz, coord
    v_n = v_inf * get_values(f[:, 1], zz)
    w_n = 0.5 * np.sqrt(nu * u_mag / x) * (
        get_values(eta, zz) * get_values(f[:, 1], zz) - get_values(f[:, 0], zz))
    corr = get_values(slope_correction, zz)
    return (v_n * tf.math.cos(theta) -
            w_n * tf.math.sin(theta)) * corr + v_n * (1.0 - corr)

  def w_init_fn(
      xx: tf.Tensor,
      yy: tf.Tensor,
      zz: tf.Tensor,
      lx: float,
      ly: float,
      lz: float,
      coord: ThreeIntTuple,
  ) -> tf.Tensor:
    """Computes the value of the velocity component normal to the wall."""
    del xx, yy, lx, ly, lz, coord
    u_n = u_mag * get_values(f[:, 1], zz)
    w_n = 0.5 * np.sqrt(nu * u_mag / x) * (
        get_values(eta, zz) * get_values(f[:, 1], zz) - get_values(f[:, 0], zz))
    corr = get_values(slope_correction, zz)
    return (u_n * tf.math.sin(theta) +
            w_n * tf.math.cos(theta)) * corr + w_n * (1.0 - corr)

  return {'u': u_init_fn, 'v': v_init_fn, 'w': w_init_fn}


def logarithmic_boundary_layer(
    u_inf: float,
    v_inf: float,
    z_0: float,
    elevation: Union[tf.Tensor, float] = 0.0,
    ) -> InitFnDict:
  """Generates `init_fn` for velocity with wall in the lower z dimension.

  The initial profile for the velocity is computed from a logrithmic model,
  which takes the form:
    u = u_s / kappa * ln((z - d) / z_0),
  where u_s is the friction velocity, kappa is the von Karman constant (0.41),
  d is the zero plane displacement, and z_0 is the surface roughness.

  Note that this function assumes that flow field variables are oriented in the
  x-y-z order. The boundary layer profile is aligned with the z axis.

  Args:
    u_inf: The free stream velocity in the x direction.
    v_inf: The free stream velocity in the y direction.
    z_0: The surface roughness, in units of m.
    elevation: The map of elevation local to the current TPU core. If it's 2D,
      it's assumed that it provides the elevation information over the x-y
      plane; if it's 3D, only the first layer (index 0 of the dimension 2) is
      taken as the elevation map and all other information is ignored. If it's a
      floating point number, the height is a constant over the x-y plane. If
      it's not provided, it's assumed that the elevation is 0.

  Returns:
    A dictionary of `init_fn` for u, v, and w.
  """

  kappa = 0.41
  u_inf = tf.constant(u_inf, dtype=_TF_DTYPE)
  v_inf = tf.constant(v_inf, dtype=_TF_DTYPE)
  z_0 = tf.constant(z_0, dtype=_TF_DTYPE)
  zero = tf.constant(0.0, dtype=_TF_DTYPE)

  if isinstance(elevation, tf.Tensor):
    if len(elevation.shape) == 3:
      elevation = tf.squeeze(elevation[..., 0])
    elevation = elevation[..., tf.newaxis]
    elevation = tf.cast(elevation, dtype=_TF_DTYPE)
  else:
    elevation = tf.constant(elevation, dtype=_TF_DTYPE)

  def friction_velocity(u_ref: tf.Tensor, lz: float) -> tf.Tensor:
    """Computes the friction velocity based on the free stream velocity."""
    lz = tf.constant(lz, dtype=_TF_DTYPE)

    return u_ref * kappa / tf.math.log(
        tf.maximum(lz - elevation, zero) / z_0)

  def log_wind_profile(z: tf.Tensor, u_s: tf.Tensor, dz: tf.Tensor
                       ) -> tf.Tensor:
    """Generates the log-wind profile as a function of height."""

    # Round elevation down to its nearest integer multiple of dz to account for
    # the mesh resolution.
    elevation_corrected = (elevation // dz) * dz

    # Offset by dz so that the first halo layer is treated as the non-slip
    # ground.
    return tf.maximum(
        u_s / kappa *
        tf.math.log(tf.maximum(
            (z - elevation_corrected + dz) / z_0, zero)), zero)

  def u_init_fn(
      xx: tf.Tensor,
      yy: tf.Tensor,
      zz: tf.Tensor,
      lx: float,
      ly: float,
      lz: float,
      coord: ThreeIntTuple,
  ) -> tf.Tensor:
    """Computes the value of the velocity component parallel to the wall."""
    del xx, yy, lx, ly, coord

    u_s = friction_velocity(u_inf, lz)

    dz = zz[0, 0, 1] - zz[0, 0, 0]

    return log_wind_profile(zz, u_s, dz)

  def v_init_fn(
      xx: tf.Tensor,
      yy: tf.Tensor,
      zz: tf.Tensor,
      lx: float,
      ly: float,
      lz: float,
      coord: ThreeIntTuple,
  ) -> tf.Tensor:
    """Computes the value of the velocity component parallel to the wall."""
    del xx, yy, lx, ly, coord

    u_s = friction_velocity(v_inf, lz)

    dz = zz[0, 0, 1] - zz[0, 0, 0]

    return log_wind_profile(zz, u_s, dz)

  def w_init_fn(
      xx: tf.Tensor,
      yy: tf.Tensor,
      zz: tf.Tensor,
      lx: float,
      ly: float,
      lz: float,
      coord: ThreeIntTuple,
  ) -> tf.Tensor:
    """Computes the value of the velocity component normal to the wall."""
    del xx, yy, lx, ly, lz, coord

    return tf.zeros_like(zz)

  return {'u': u_init_fn, 'v': v_init_fn, 'w': w_init_fn}
