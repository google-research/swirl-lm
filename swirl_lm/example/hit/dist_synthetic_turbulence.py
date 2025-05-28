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

"""A library for distributedly generating synthetic turbulence."""

from typing import Tuple

import numpy as np
from swirl_lm.communication import halo_exchange
from swirl_lm.example.hit import analytics_util
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility.dist_fft import dist_tpu_fft
import tensorflow as tf


#  These are coefficients for modeled spectrum as described in `S. B. Pope,
#  Turbulent Flows Cambridge University Press, Cambridge, 2000, Section 6.5.3`.
ALPHA = 1.5
C_L = 6.78
C_ETA = 0.4
BETA = 5.2
P0 = 4.0

ThreeIntTuple = Tuple[int, int, int]
SevenTensorTuple = Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
                         tf.Tensor, tf.Tensor, tf.Tensor]

_R_TYPE = tf.float32
_C_TYPE = tf.complex64


class DistTurbulenceSynthesizer(object):
  """Distribued turbulence synthesizer."""

  def __init__(
      self,
      params: grid_parametrization.GridParametrization,
  ) -> None:
    self._params = params
    if params.lx != params.ly or params.lx != params.lz:
      raise ValueError(
          'DistTurbulenceSynthesizer only supports cube shaped geometry.'
          '`lx`, `ly` and `lz` for the grid must be equal.')
    self._l = params.lx
    self._core_nx = params.core_nx
    self._core_ny = params.core_ny
    self._core_nz = params.core_nz
    self._cx = params.cx
    self._cy = params.cy
    self._cz = params.cz
    self._nx = params.core_nx * params.cx
    self._ny = params.core_ny * params.cy
    self._nz = params.core_nz * params.cz
    self._halo_width_x = int((params.nx - params.core_nx) / 2)
    self._halo_width_y = int((params.ny - params.core_ny) / 2)
    self._halo_width_z = int((params.nz - params.core_nz) / 2)
    self._dx = params.dx
    self._dy = params.dy
    self._dz = params.dz

  def pope(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      le: float,
      ld: float,
      nu: float,
      random_seed: int = 0,
      rm_imaginary: bool = True,
  ) -> SevenTensorTuple:
    """Generates synthetic turbulence using the Pope model spectrum.

    This follows the model spectrum in `S. B. Pope, Turbulent Flows,
    Cambridge University Press, Cambridge, 2000, Section 6.5.3`.

    Args:
      kernel_op: An `ApplyKernelOp` object providing implementations for the
        difference operations.
      replica_id: The `replica_id` of the local core.
      replicas: The global mapping from the coordinate of the local core to its
        replica_id.
      le: The energetic length scale of the turbulent eddy.
      ld: The dissipative length scale of the turbulent eddy.
      nu: The kinematic viscosity.
      random_seed: The seed for generating the random phase.
      rm_imaginary: Whether to remove the imaginary part.

    Returns:
      A 7 component tuple of tensors, including 4 fields: u, v, w, p and 3
      scalars: tke_hat (turbulent kinematic energy), epsilon_hat (dissipation
      rate), and amp_hat (the amplitude). The shape returned for the 4 fields is
      before the transpose, i.e. [nx, ny, nz].
    """

    l = self._l
    core_nx = self._core_nx
    core_ny = self._core_ny
    core_nz = self._core_nz
    nx = self._nx
    ny = self._ny
    nz = self._nz
    halo_width_x = self._halo_width_x
    halo_width_y = self._halo_width_y
    halo_width_z = self._halo_width_z

    spectral_grids = common_ops.get_spectral_index_grid(
        core_nx,
        core_ny,
        core_nz,
        replicas,
        replica_id,
        halos=[halo_width_x, halo_width_y, halo_width_z])
    xx = spectral_grids['xx']
    yy = spectral_grids['yy']
    zz = spectral_grids['zz']
    xx_c = spectral_grids['xx_c']
    yy_c = spectral_grids['yy_c']
    zz_c = spectral_grids['zz_c']

    core_nx_full = core_nx + 2 * halo_width_x
    core_ny_full = core_ny + 2 * halo_width_y
    core_nz_full = core_nz + 2 * halo_width_z

    def get_mesh(
        xx,
        yy,
        zz,
    ):
      xx_mesh = tf.tile(
          tf.reshape(xx, [core_nx_full, 1, 1]), [1, core_ny_full, core_nz_full])
      yy_mesh = tf.tile(
          tf.reshape(yy, [1, core_ny_full, 1]), [core_nx_full, 1, core_nz_full])
      zz_mesh = tf.tile(
          tf.reshape(zz, [1, 1, core_nz_full]), [core_nx_full, core_ny_full, 1])
      return xx_mesh, yy_mesh, zz_mesh

    def gen_random_phase(
        xx,
        yy,
        zz,
        seed,
    ):

      def get_random_phase_plane(i):
        local_seed = tf.convert_to_tensor([seed, zz[i]], dtype=tf.int32)
        random_phase_pool = tf.random.stateless_uniform(
            [nx, ny], seed=local_seed, dtype=_R_TYPE) - 0.5

        # The padding is done so the overall shape still includes all the
        # halo, usually chosen to fit better with the HBM. The values in
        # these padded area (all in the high index region) will not be
        # referenced/used.
        random_phase_pool = tf.pad(
            random_phase_pool,
            paddings=[[0, 2 * self._cx * halo_width_x],
                      [0, 2 * self._cy * halo_width_y]])

        random_phase_plane = 2 * np.pi * tf.gather_nd(
            random_phase_pool,
            # Generating the indices to sample from the global random_phase
            # pool.
            tf.stack([
                tf.tile(tf.expand_dims(xx, 1), [1, core_ny_full]) +
                (nx - 1) // 2,
                tf.tile(tf.expand_dims(yy, 0), [core_nx_full, 1]) +
                (ny - 1) // 2
            ],
                     axis=2))
        return tf.expand_dims(random_phase_plane, 0)

      i0 = tf.constant(0, dtype=tf.int32)
      random_phase = tf.zeros([core_nz_full, core_nx_full, core_ny_full])
      random_phase = tf.tensor_scatter_nd_update(random_phase, [[i0]],
                                                 get_random_phase_plane(i0))
      get_random_phase_plane(i0)
      i0 = i0 + 1

      def body(
          i,
          random_phase,
      ):
        random_phase = tf.tensor_scatter_nd_update(random_phase, [[i]],
                                                   get_random_phase_plane(i))
        return i + 1, random_phase

      def cond(
          i,
          random_phase,
      ):
        del random_phase
        return i < core_nz_full

      _, random_phase = tf.while_loop(
          cond=cond, body=body, loop_vars=[i0, random_phase], back_prop=False)
      random_phase = tf.transpose(random_phase, perm=[1, 2, 0])
      return random_phase

    def get_spectrum(
        xx,
        yy,
        zz,
    ):
      psr = gen_random_phase(xx, yy, zz, random_seed + 7283)
      ps1 = gen_random_phase(xx, yy, zz, random_seed + 3019)
      ps2 = gen_random_phase(xx, yy, zz, random_seed + 1877)
      f_le = 1.0
      dk = 2.0 * np.pi / l
      eps = 1.0 / 1e6
      xx_mesh, yy_mesh, zz_mesh = get_mesh(xx, yy, zz)
      kx = tf.cast(xx_mesh, _R_TYPE) * dk
      ky = tf.cast(yy_mesh, _R_TYPE) * dk
      kz = tf.cast(zz_mesh, _R_TYPE) * dk
      kk = tf.math.sqrt(kx * kx + ky * ky + kz * kz)
      kk2 = tf.math.sqrt(kx * kx + ky * ky)

      # This follows the model spectrum in `S. B. Pope, Turbulent Flows,
      # Cambridge University Press, Cambridge, 2000, Section 6.5.3`.
      energy_spec = tf.math.sqrt(
          ALPHA * (nu**2.0) / (ld**(8.0 / 3.0)) * (kk**(-5.0 / 3.0)) *
          ((kk * le / f_le) /
           (((kk * le / f_le)**2 + C_L)**0.5))**(5.0 / 3.0 + P0) *
          tf.math.exp(-BETA * (((kk * ld)**4.0 + C_ETA**4.0)**0.25 - C_ETA)) /
          (4.0 * np.pi)) * (dk**0.5) / kk
      ak = tf.where(
          tf.less(eps, kk),
          tf.cast(energy_spec, _C_TYPE) *
          tf.math.exp(1j * tf.cast(ps1, _C_TYPE)) *
          tf.cast(tf.math.cos(psr), _C_TYPE), tf.zeros_like(kk, _C_TYPE))
      bk = tf.where(
          tf.less(eps, kk),
          tf.cast(energy_spec, _C_TYPE) *
          tf.math.exp(1j * tf.cast(ps2, _C_TYPE)) *
          tf.cast(tf.math.sin(psr), _C_TYPE), tf.zeros_like(kk, _C_TYPE))
      uk = tf.where(
          tf.less(kk2, eps), (ak + bk) / np.sqrt(2.0),
          (ak * tf.cast(kk * ky, _C_TYPE) +
           bk * tf.cast(kx * kz, _C_TYPE)) /
          tf.cast(kk * kk2, _C_TYPE))
      vk = tf.where(
          tf.less(kk2, eps), (bk - ak) / np.sqrt(2.0),
          (bk * tf.cast(ky * kz, _C_TYPE) -
           ak * tf.cast(kk * kx, _C_TYPE)) /
          tf.cast(kk * kk2, _C_TYPE))
      wk = tf.where(
          tf.less(kk2, eps), tf.zeros_like(kk, _C_TYPE),
          -bk * tf.cast(kk2 / kk, _C_TYPE))

      return uk, vk, wk

    uk, vk, wk = get_spectrum(xx, yy, zz)
    uk_c, vk_c, wk_c = get_spectrum(xx_c, yy_c, zz_c)

    xx_mesh, yy_mesh, zz_mesh = get_mesh(xx, yy, zz)
    xx_c_mesh, yy_c_mesh, zz_c_mesh = get_mesh(xx_c, yy_c, zz_c)

    def set_conjugate(
        uk,
        uk_c,
    ):
      uk_x_tmp = tf.where(
          tf.less(xx_mesh, xx_c_mesh), tf.math.conj(uk_c),
          tf.where(
              tf.greater(xx_mesh, xx_c_mesh), uk,
              0.5 * (uk + tf.math.conj(uk))))
      uk_xy_tmp = tf.where(
          tf.less(yy_mesh, yy_c_mesh), tf.math.conj(uk_c),
          tf.where(tf.greater(yy_mesh, yy_c_mesh), uk, uk_x_tmp))
      uk_out = tf.where(
          tf.less(zz_mesh, zz_c_mesh), tf.math.conj(uk_c),
          tf.where(tf.greater(zz_mesh, zz_c_mesh), uk, uk_xy_tmp))
      return uk_out

    uk_final = set_conjugate(uk, uk_c)
    vk_final = set_conjugate(vk, vk_c)
    wk_final = set_conjugate(wk, wk_c)

    transformer = dist_tpu_fft.DistTPUFFT(replicas, replica_id)

    halos = [self._halo_width_x, self._halo_width_y, self._halo_width_z]
    dk = 2.0 * np.pi / l

    # Initializes pressure.
    xx_mesh, yy_mesh, zz_mesh = get_mesh(xx, yy, zz)
    kx = tf.cast(xx_mesh, _R_TYPE) * dk
    ky = tf.cast(yy_mesh, _R_TYPE) * dk
    kz = tf.cast(zz_mesh, _R_TYPE) * dk
    kk_sq = kx * kx + ky * ky + kz * kz

    spec_sq = tf.math.real((uk_final * tf.math.conj(uk_final) +
                            vk_final * tf.math.conj(vk_final) +
                            wk_final * tf.math.conj(wk_final)))

    tke_hat = 0.5 * common_ops.global_mean(tf.unstack(spec_sq, axis=2),
                                           replicas, halos)
    epsilon_hat = (2.0 * nu *
                   common_ops.global_mean(
                       tf.unstack(kk_sq * spec_sq, axis=2),
                       replicas, halos) / nx / ny / nz)

    a_eta = nu ** (1.5) / epsilon_hat ** 0.5 / ld ** 2
    a_l = le * epsilon_hat / tke_hat ** 1.5
    amp = (a_eta * a_l) ** 0.5

    # Adjusting for the fact that (1) the Discrete FFT used is not a unitary
    # transform, (2) The transformed energy is the aggregation of a quantity
    # within a unit volume, (3) The transformed energy value is an average over
    # the physical vloume. So the correction for velocity form (1) is
    # (nx * ny * nz) ^ 0.5, from (2) is 1 / (dx * dy * dz) ^ 0.5 and from (3) is
    # (Lx * Ly * Lz) ^ 0.5. So the overall correction is (nx * ny * nz).
    normalization_adjust = (nx * ny * nz)
    u = normalization_adjust * transformer.transform_3d(
        uk_final, halo_widths=halos, inverse=True)
    v = normalization_adjust * transformer.transform_3d(
        vk_final, halo_widths=halos, inverse=True)
    w = normalization_adjust * transformer.transform_3d(
        wk_final, halo_widths=halos, inverse=True)

    def gradient(u):
      halo_dims = (0, 1, 2)
      replica_dims = (0, 1, 2)
      periodic_dims = (True, True, True)
      u = halo_exchange.inplace_halo_exchange(
          tf.transpose(u, perm=(2, 0, 1)),
          halo_dims,
          replica_id,
          replicas,
          replica_dims,
          periodic_dims,
          # There is an inonsistency in the API where halo_width for
          # current halo_exchange can take only one integer. In the current
          # case this is ok as they are all the same in all dimensions.
          width=self._halo_width_x
      )

      # Transpose back to [nx, ny, nz] shape. This is mostly for FFT.
      return [
          tf.transpose(component, perm=[1, 2, 0])
          for component in analytics_util.gradient(
              kernel_op, u, self._dx, self._dy, self._dz
          )
      ]

    grad_u = tf.cast(gradient(tf.math.real(u)), _C_TYPE)
    grad_v = tf.cast(gradient(tf.math.real(v)), _C_TYPE)
    grad_w = tf.cast(gradient(tf.math.real(w)), _C_TYPE)

    rhs = (grad_u[0] * grad_u[0] + grad_v[1] * grad_v[1] +
           grad_w[2] * grad_w[2] + 2.0 * grad_u[1] * grad_v[0] +
           2.0 * grad_u[2] * grad_w[0] + 2.0 * grad_v[2] * grad_w[1])
    rhs_k = transformer.transform_3d(rhs,
                                     halo_widths=halos, inverse=False)

    pk = tf.where(kk_sq > 1e-6,
                  tf.cast((1.0 / kk_sq), _C_TYPE) * rhs_k,
                  tf.zeros_like(rhs_k, _C_TYPE))
    # Note here we did a round-trip FFT, so no adjustment needed.
    p = transformer.transform_3d(pk, halo_widths=halos, inverse=True)
    if rm_imaginary:
      u = tf.math.real(u)
      v = tf.math.real(v)
      w = tf.math.real(w)
      p = tf.math.real(p)

    return u, v, w, p, tke_hat, epsilon_hat, amp
