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

"""Library for computing basic analytics in a distributed setting."""

from typing import Sequence, Tuple

import numpy as np
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
from swirl_lm.utility.dist_fft import dist_tpu_fft
import tensorflow as tf


FlowFieldVal = types.FlowFieldVal

_CTYPE = types.TF_COMPLEX_DTYPE
_DTYPE = types.TF_DTYPE
VectorField = types.VectorField



def compute_global_tke(
    u: FlowFieldVal,
    v: FlowFieldVal,
    w: FlowFieldVal,
    halos: Sequence[int],
    replicas: np.ndarray,
) -> Tuple[FlowFieldVal, FlowFieldVal, FlowFieldVal, FlowFieldVal]:
  """Computes the turbulent kinetic enegy (TKE) from the field components.

  Given a vector field with three compoments:

               (u(x, y, z), v(x, y, z), w(x, y, z))

  this computes 1/2 <(u - <u>)^2 + (v - <v>)^2 + (w - <w>)^2)> over
  the entire domain in a distributed setting (across all cores). Note that with
  this formulation it is implied that all field components are homogeneous in
  all three dimensions. For cases where some of the dimensions are not
  homogeneous, the `background` value (DC term) can not be represented by a
  single <u>, <v>, <w> but would be varying along the dimension that is not
  homogeneous.

  Args:
    u: The first component of the field/variable on the local core. The halos of
      the field are included.
    v: The second component of the field/variable on the local core. The halos
      of the field are included.
    w: The third component of the field/variable on the local core. The halos of
      the field are included.
    halos: The width of the (symmetric) halos for each dimension: for example
      [1, 2, 3] means the halos for `u`, `v`, and `w` have width of 1, 2, 3 on
      both sides in x, y, z dimension respectively.
    replicas: A 3D numpy array representing the mapping from the core replica
      coordinate to the `replica_id`. The number of cores in each dimension is
      the number of splits of the global input for the transformation.

  Returns:
    A Tuple of scalar tensors: (TKE, u_mean, v_mean, w_mean) from the group the
    ore belongs to.
  """
  common_ops.validate_fields(u, v, w)
  u_mean = common_ops.global_mean(u, replicas, halos)
  v_mean = common_ops.global_mean(v, replicas, halos)
  w_mean = common_ops.global_mean(w, replicas, halos)
  local_tke = (u - u_mean)**2 + (v - v_mean)**2 + (w - w_mean)**2
  return (0.5 * common_ops.global_mean(local_tke, replicas, halos),
          u_mean, v_mean, w_mean)


def get_spectrum_num_bins(
    nx_global: int,
    ny_global: int,
    nz_global: int,
) -> int:
  """Gets the number of spectrum bins given the size of the grid.

  This function is used to generate the number of bins for spectrum consistently
  given the size of the entire grid. Useful for pre-allocating a tensor for
  storing the result from `global_energy_spectrum`.

  Args:
    nx_global: The total grid size in x dimension.
    ny_global: The total grid size in y dimension.
    nz_global: The total grid size in z dimension.

  Returns:
    An integer as the number of bins for the spectrum.
  """
  nx_limit = nx_global // 2 + 1
  ny_limit = ny_global // 2 + 1
  nz_limit = nz_global // 2 + 1
  return int(np.sqrt(nx_limit**2 + ny_limit**2 + nz_limit**2)) + 1


# Note: this currently uses tf.tensor_scatter_nd_add, which is extremely
# inefficient on TPU and will impact the performance. We will explore
# alternatives such as using einsum.
def global_energy_spectrum(
    l: float,
    u: FlowFieldVal,
    v: FlowFieldVal,
    w: FlowFieldVal,
    halos: Sequence[int],
    replicas: np.ndarray,
    replica_id: tf.Tensor,
    dtype: tf.DType = _DTYPE,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Calculates the energy spectrum given a three-component vector field.

  Note this function assumes the domain is a cube.

  Args:
    l: The length of the cubic domain.
    u: The first component of the field/variable on the local core. The halos of
      the field are included.
    v: The second component of the field/variable on the local core. The halos
      of the field are included.
    w: The third component of the field/variable on the local core. The halos of
      the field are included.
    halos: The width of the (symmetric) halos for each dimension: for example
      [1, 2, 3] means the halos for `u`, `v`, and `w` have width of 1, 2, 3 on
      both sides in x, y, z dimension respectively. The halo region of the field
      is first removed in the calculation of the spectrum.
    replicas: A 3D numpy array representing the mapping from the core replica
      coordinate to the `replica_id`. The number of cores in each dimension is
      the number of splits of the global input for the transformation.
    replica_id: A scalar integer tf.Tensor representing the `replica_id` of the
      current core.
    dtype: Datatype of the output. Default to tf.float32.

  Returns:
    A two-Tuple of (1D-Tensor, 1D-Tensor), representing (spectrum, wavenumber).
  """
  # Note that we are calling the FFT with halos and let FFT module to handle the
  # halos since the totla shape with halos in most cases are more conforming
  # with TPU HBM and this can prevent unnecessary huge padding.
  compute_shape = replicas.shape
  common_ops.validate_fields(u, v, w)
  nx_full, ny_full, nz_full = common_ops.get_field_shape(u)
  nx = nx_full - 2 * halos[0]
  ny = ny_full - 2 * halos[1]
  nz = nz_full - 2 * halos[2]
  nx_global = nx * compute_shape[0]
  ny_global = ny * compute_shape[1]
  nz_global = nz * compute_shape[2]
  dk = 2 * np.pi / l

  spectral_grid = common_ops.get_spectral_index_grid(
      nx, ny, nz, replicas, replica_id, halos=halos)

  def spectral_grid_square(dim):
    """Maps the grid to 3D and computes the square of the index values."""
    new_dims = [(-1, -1), (0, -1), (0, 0)]
    shapes = [[1, ny_full, nz_full], [nx_full, 1, nz_full],
              [nx_full, ny_full, 1]]
    grid_names = ['xx', 'yy', 'zz']
    return tf.square(
        tf.tile(
            tf.expand_dims(
                tf.expand_dims(
                    tf.cast(spectral_grid[grid_names[dim]], dtype),
                    new_dims[dim][0]), new_dims[dim][1]), shapes[dim]))

  update_indices = tf.cast(
      tf.math.floor(
          tf.math.sqrt(
              spectral_grid_square(0) + spectral_grid_square(1) +
              spectral_grid_square(2))), tf.int32)

  update_indices = tf.reshape(update_indices, [-1, 1])

  # Permute from (nz, nx, ny) to (nx, ny, nz)
  u_merged = tf.transpose(u, perm=(1, 2, 0))
  v_merged = tf.transpose(v, perm=(1, 2, 0))
  w_merged = tf.transpose(w, perm=(1, 2, 0))

  transformer = dist_tpu_fft.DistTPUFFT(replicas, replica_id)
  # Note all values in halo region will be zero.
  uk = tf.math.abs(transformer.transform_3d(tf.cast(u_merged, _CTYPE), halos))
  vk = tf.math.abs(transformer.transform_3d(tf.cast(v_merged, _CTYPE), halos))
  wk = tf.math.abs(transformer.transform_3d(tf.cast(w_merged, _CTYPE), halos))
  ek = tf.reshape(
      0.5 * (tf.square(uk) + tf.square(vk) + tf.square(wk)) /
      # Adjusting for the fact that (1) the Discrete FFT used is not a unitary
      # transform, (2) we want to get the spectrum for the TKE average over the
      # volume. So the adjustment for energy spectrum from (1) is
      # 1 / (nx * ny * ny) and from (2) is (dx * dy * dz) / (Lx * Ly * Lz)
      # so the total adjustment is 1 / (nx * ny * nz) ^ 2. Note the converted
      # spectrum represent the energy of each discrete bin of width of dk, not
      # the spectral density.
      ((tf.cast(nx_global, dtype) * tf.cast(ny_global, dtype) *
        tf.cast(nz_global, dtype))**2), [-1])

  group_assignment = np.array([range(replicas.size)])
  spec = tf.zeros([get_spectrum_num_bins(nx_global, ny_global, nz_global)])
  ks = dk * tf.range(spec.get_shape().as_list()[0], dtype=dtype)
  spec = tf.tensor_scatter_nd_add(spec, update_indices, ek)
  spec = tf.compat.v1.tpu.cross_replica_sum(
      spec, group_assignment=group_assignment)
  return spec, ks


def curl(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    u: FlowFieldVal,
    v: FlowFieldVal,
    w: FlowFieldVal,
    dx: float,
    dy: float,
    dz: float,
) -> Tuple[FlowFieldVal, FlowFieldVal, FlowFieldVal]:
  """Calculates ∇ × (u, v, w).

  This calculates the Curl of a three-component vector field. Note that the
  values in the width-one halo region in the result will not be correct.

  Args:
    kernel_op: An `get_kernel_fn.ApplyKernelOp` object that implements the
      finite difference operations.
    u: The first component of the field/variable on the local core.
    v: The second component of the field/variable on the local core.
    w: The third component of the field/variable on the local core.
    dx: The grid space in x dimension.
    dy: The grid space in y dimension.
    dz: The grid space in z dimension.

  Returns:
    A three-Tuple of FlowFieldVal representing the result of
    ∇ × (u, v, w).
  """

  dwdy = kernel_op.apply_kernel_op_y(w, 'kDy') / dy
  dvdz = kernel_op.apply_kernel_op_z(v, 'kDz', 'kDzsh') / dz
  curl_x = 0.5 * (dwdy - dvdz)

  dudz = kernel_op.apply_kernel_op_z(u, 'kDz', 'kDzsh') / dz
  dwdx = kernel_op.apply_kernel_op_x(w, 'kDx') / dx
  curl_y = 0.5 * (dudz - dwdx)

  dvdx = kernel_op.apply_kernel_op_x(v, 'kDx') / dx
  dudy = kernel_op.apply_kernel_op_y(u, 'kDy') / dy
  curl_z = 0.5 * (dvdx - dudy)

  return (curl_x, curl_y, curl_z)


def enstrophy(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    u: FlowFieldVal,
    v: FlowFieldVal,
    w: FlowFieldVal,
    dx: float,
    dy: float,
    dz: float,
    halos: Sequence[int],
    replicas: np.ndarray,
) -> FlowFieldVal:
  """Calculates Enstrophy, < |∇ × (u, v, w)|^2 >.

  The calculation will exclude the halo region on each core.

  Args:
    kernel_op: An `get_kernel_fn.ApplyKernelOp` object that implements the
      finite difference operations.
    u: The first component of the field/variable on the local core.
    v: The second component of the field/variable on the local core.
    w: The third component of the field/variable on the local core.
    dx: The grid space in x dimension.
    dy: The grid space in y dimension.
    dz: The grid space in z dimension.
    halos: The width of the (symmetric) halos for each dimension: for example
      [1, 2, 3] means the halos for `u`, `v`, and `w` have width of 1, 2, 3 on
      both sides in x, y, z dimension respectively.
    replicas: A 3D numpy array representing the mapping from the core replica
      coordinate to the `replica_id`. The number of cores in each dimension is
      the number of splits of the global input for the transformation.

  Returns:
    A scalar Tensor representing the Entropy from the group the core belongs to.
  """
  curl_x, curl_y, curl_z = curl(kernel_op, u, v, w, dx, dy, dz)
  return common_ops.global_mean(
      curl_x**2 + curl_y**2 + curl_z**2, replicas, halos
  )


def gradient(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    u: FlowFieldVal,
    dx: float,
    dy: float,
    dz: float,
) -> VectorField:
  """Calculates the gradient of a scalar field `u`: ∇ u.

  The values in the halo region of the output will not be correct.

  Args:
    kernel_op: An `get_kernel_fn.ApplyKernelOp` object that implements the
      finite difference operations.
    u: A scalar field/variable on the local core.
    dx: The grid space in x dimension.
    dy: The grid space in y dimension.
    dz: The grid space in z dimension.

  Returns:
    The gridient of input field `u`. The result is represented as a 3-Tuple with
    each element representing one component of the vector. Note the width one
    halo region will not have the correct values.
  """
  grad_x = 0.5 * kernel_op.apply_kernel_op_x(u, 'kDx') / dx
  grad_y = 0.5 * kernel_op.apply_kernel_op_y(u, 'kDy') / dy
  grad_z = 0.5 * kernel_op.apply_kernel_op_z(u, 'kDz', 'kDzsh') / dz
  return (grad_x, grad_y, grad_z)


def strain_rate(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    u: FlowFieldVal,
    v: FlowFieldVal,
    w: FlowFieldVal,
    dx: float,
    dy: float,
    dz: float,
) -> Tuple[VectorField, VectorField, VectorField]:
  """Calculates shear stree given the input field (u, v, w).

  Args:
    kernel_op: An `get_kernel_fn.ApplyKernelOp` object that implements the
      finite difference operations.
    u: The first velocity component of the field/variable on the local core.
    v: The second velocity component of the field/variable on the local core.
    w: The third velocity component of the field/variable on the local core.
    dx: The grid space in x dimension.
    dy: The grid space in y dimension.
    dz: The grid space in z dimension.

  Returns:
    The strain rate in the format of
        ((s11, s12, s13),
         (s21, s22, s23),
         (s31, s32, s33))
    Note that the value in the width-one halo region on each core will not be
    correct.
  """
  s1 = gradient(kernel_op, u, dx, dy, dz)
  s2 = gradient(kernel_op, v, dx, dy, dz)
  s3 = gradient(kernel_op, w, dx, dy, dz)
  s11 = s1[0]
  s12 = 0.5 * (s1[1] + s2[0])
  s13 = 0.5 * (s1[2] + s3[0])
  s21 = s12
  s22 = s2[1]
  s23 = 0.5 * (s2[2] + s3[1])
  s31 = s13
  s32 = s23
  s33 = s3[2]
  return ((s11, s12, s13), (s21, s22, s23), (s31, s32, s33))


def dissipation_rate(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    u: FlowFieldVal,
    v: FlowFieldVal,
    w: FlowFieldVal,
    dx: float,
    dy: float,
    dz: float,
    nu: float,
    halos: Sequence[int],
    replicas: np.ndarray,
) -> tf.Tensor:
  """Calculates dissipation given (u, v, w) and kinematic viscosity nu.

  The calculation will exclude the halo region on each core.

  Args:
    kernel_op: An `get_kernel_fn.ApplyKernelOp` object that implements the
      finite difference operations.
    u: The first component of the field/variable on the local core.
    v: The second component of the field/variable on the local core.
    w: The third component of the field/variable on the local core.
    dx: The grid space in x dimension.
    dy: The grid space in y dimension.
    dz: The grid space in z dimension.
    nu: The kinematic viscosity.
    halos: The width of the (symmetric) halos for each dimension: for example
      [1, 2, 3] means the halos for `u`, `v`, and `w` have width of 1, 2, 3 on
      both sides in x, y, z dimension respectively.
    replicas: A 3D numpy array representing the mapping from the core replica
      coordinate to the `replica_id`. The number of cores in each dimension is
      the number of splits of the global input for the transformation.

  Returns:
    The dissipation rate for the group the core belongs to.
  """
  s = strain_rate(kernel_op, u, v, w, dx, dy, dz)
  square_sum = tf.zeros_like(s[0][0])
  for i in range(3):
    for j in range(3):
      square_sum += tf.square(s[i][j])
  return 2.0 * nu * common_ops.global_mean(square_sum, replicas, halos)


def kolmogorov_scales(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    u: FlowFieldVal,
    v: FlowFieldVal,
    w: FlowFieldVal,
    dx: float,
    dy: float,
    dz: float,
    nu: float,
    halos: Sequence[int],
    group_assignment: np.ndarray,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Calculates Kolmogorov length, time and velocity scales.

  The calculation will exclude the halo region on each core.

  Args:
    kernel_op: An `get_kernel_fn.ApplyKernelOp` object that implements the
      finite difference operations.
    u: The first component of the field/variable on the local core.
    v: The second component of the field/variable on the local core.
    w: The third component of the field/variable on the local core.
    dx: The grid space in x dimension.
    dy: The grid space in y dimension.
    dz: The grid space in z dimension.
    nu: The kinematic viscosity.
    halos: The width of the (symmetric) halos for each dimension: for example
      [1, 2, 3] means the halos for `u`, `v`, and `w` have width of 1, 2, 3 on
      both sides in x, y, z dimension respectively.
    group_assignment: A 2D numpy array of the shape of [num_group,
      num_core_per_group] indicating the grouping for computation. For example,
      for a 4-core configuration, if `group_assignment` is [[0, 1, 2, 3]], the
      TKE computation will be done across all 4 cores and each core get exactly
      the same results. In another example, if `group_assignment` is [[0, 1],
      [2, 3]], then TKE is computed with cores 0 and 1 as one group, and cores 2
      and 3 as another group.

  Returns:
    The Kolmogorov scales in the 3-Tuple form: (length, time, velocity).
  """
  eps = dissipation_rate(kernel_op, u, v, w, dx, dy, dz, nu, halos,
                         group_assignment)
  eta = (nu**3 / eps)**0.25
  t_eta = tf.math.sqrt(nu / eps)
  u_eta = (nu * eps)**0.25
  return eta, t_eta, u_eta


def max_abs_velocity(
    u: FlowFieldVal,
    v: FlowFieldVal,
    w: FlowFieldVal,
    halos: Sequence[int],
    group_assignment: np.ndarray,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Calculates the max velocity magnitude and max for each velocity component.

  The calculation will exclude the halo region on each core.

  Args:
    u: The first component of the field/variable on the local core.
    v: The second component of the field/variable on the local core.
    w: The third component of the field/variable on the local core.
    halos: The width of the (symmetric) halos for each dimension: for example
      [1, 2, 3] means the halos for `u`, `v`, and `w` have width of 1, 2, 3 on
      both sides in x, y, z dimension respectively.
    group_assignment: A 2D numpy array of the shape of [num_group,
      num_core_per_group] indicating the grouping for computation. For example,
      for a 4-core configuration, if `group_assignment` is [[0, 1, 2, 3]], the
      TKE computation will be done across all 4 cores and each core get exactly
      the same results. In another example, if `group_assignment` is [[0, 1],
      [2, 3]], then TKE is computed with cores 0 and 1 as one group, and cores 2
      and 3 as another group.

  Returns:
    The max absolute velocity of individual components and the overall velocity
    magnitude.
  """
  common_ops.validate_fields(u, v, w)
  nx, ny, nz = common_ops.get_field_shape(u)
  slc = (
      slice(halos[2], nz - halos[2]),
      slice(halos[0], nx - halos[0]),
      slice(halos[1], ny - halos[1]),
  )
  abs_velocity = [tf.math.abs(v_j[slc]) for v_j in (u, v, w)]

  velocity_mag = tf.math.sqrt(abs_velocity[0]**2 + abs_velocity[1]**2 +
                              abs_velocity[2]**2)
  u_max = common_ops.global_reduce(abs_velocity[0], tf.math.reduce_max,
                                   group_assignment)
  v_max = common_ops.global_reduce(abs_velocity[1], tf.math.reduce_max,
                                   group_assignment)
  w_max = common_ops.global_reduce(abs_velocity[2], tf.math.reduce_max,
                                   group_assignment)
  mag_max = common_ops.global_reduce(velocity_mag, tf.math.reduce_max,
                                     group_assignment)
  return (u_max, v_max, w_max, mag_max)
