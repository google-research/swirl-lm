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
R"""The radiative transfer equation solver.

References:
1. Shonk, Jonathan & Hogan, Robin. (2008). Tripleclouds: An Efficient Method for
   Representing Horizontal Cloud Inhomogeneity in 1D Radiation Schemes by Using
   Three Regions at Each Height. J. Climate. 21. 10.1175/2007JCLI1940.1.
2. Toon, Owen & McKay, C & Ackerman, T. & Santhanam, K.. (1989). Rapid
   calculation of radiative heating rates and photodissociation rates in
   Inhomogeneous multiple scattering atmospheres. Journal of Geophysical
   Research. 94. 10.1029/JD094iD13p16287.
"""

import math
import swirl_lm.physics.radiation.rte.rte_utils as utils
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf


FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap

# Secant of the longwave diffusivity angle per Fu et al. (1997).
_LW_DIFFUSIVE_FACTOR = 1.66
_EPSILON = 1e-7


class RTESolver:
  """A library for solving the two-stream radiative transfer equation."""

  def __init__(
      self,
      params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      g_dim: int,
  ):
    self.halos = params.halo_width
    self.g_dim = g_dim
    self.rte_utils = utils.RTEUtils(params)
    self._kernel_op = kernel_op
    self._kernel_op.add_kernel({'shift_up': ([1.0, 0.0, 0.0], 1)})
    self._kernel_op.add_kernel({'shift_dn': ([0.0, 0.0, 1.0], 1)})
    self._shift_up_fn = (
        lambda f: kernel_op.apply_kernel_op_x(f, 'shift_upx'),
        lambda f: kernel_op.apply_kernel_op_y(f, 'shift_upy'),
        lambda f: kernel_op.apply_kernel_op_z(f, 'shift_upz', 'shift_upzsh'),
    )[g_dim]
    self._shift_down_fn = (
        lambda f: kernel_op.apply_kernel_op_x(f, 'shift_dnx'),
        lambda f: kernel_op.apply_kernel_op_y(f, 'shift_dny'),
        lambda f: kernel_op.apply_kernel_op_z(f, 'shift_dnz', 'shift_dnzsh'),
    )[g_dim]

  def lw_combine_sources(
      self, planck_srcs: FlowFieldMap
  ) -> FlowFieldMap:
    """Combines the longwave source functions at each cell face.

    RRTMGP provides two source functions at each cell interface using the
    spectral mapping of each adjacent layer. These source functions are combined
    here via a geometric mean, and the result can be used for two-stream
    calculations.

    Args:
      planck_srcs: A dictionary containing the longwave Planck sources at the
        cell interfaces. The `level_planck_src_top` 3D variable contains the
        Planck source at the top cell face derived from the cell center's
        spectral mapping while the `level_planck_src_bottom` 3D variable
        contains the Planck source at the bottom cell face.

    Returns:
      A map of 3D variables for the combined Planck sources at the top face and
      the bottom cell face, respectively with the same keys as the ones in the
      input `planck_srcs`.
    """
    def geometric_mean(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
      return tf.math.sqrt(a * b)

    planck_src_top = planck_srcs['level_planck_src_top']
    planck_src_bottom = planck_srcs['level_planck_src_bottom']

    level_src_top = tf.nest.map_structure(
        geometric_mean, planck_src_top, self._shift_down_fn(planck_src_bottom)
    )
    level_src_bottom = self._shift_up_fn(level_src_top)

    return {
        'level_planck_src_top': level_src_top,
        'level_planck_src_bottom': level_src_bottom,
    }

  def lw_cell_source_and_properties(
      self,
      optical_depth: tf.Tensor,
      ssa: tf.Tensor,
      level_src_bottom: tf.Tensor,
      level_src_top: tf.Tensor,
      asymmetry_factor: tf.Tensor,
  ) -> FlowFieldMap:
    """Computes longwave two-stream reflectance, transmittance, and sources.

    The upwelling and downwelling Planck functions and the optical properties
    (transmission and reflectance) are calculated at the cell centers. Equations
    are developed in Meador and Weaver (1980) and Toon et al. (1989).

    Args:
      optical_depth: The pointwise local optical depth.
      ssa: The pointwise single-scattering albedo.
      level_src_bottom: The Planck source at the top cell face [W / m^2 / sr].
      level_src_top: The Planck source at the bottom cell face [W / m^2 / sr].
      asymmetry_factor: The pointwise asymmetry factor.

    Returns:
      A dictionary containing the following items:
      'reflectance': A 3D variable containing the pointwise reflectance.
      'transmittance': A 3D variable containing the pointwise transmittance.
      'src_up': A 3D variable containing the pointwise upwelling Planck source.
      'src_down': A 3D variable with the pointwise downwelling Planck source.
    """
    def gamma1_fn(ssa: tf.Tensor, asymmetry_factor: tf.Tensor) -> tf.Tensor:
      """The coefficient of the parallel irradiance in the 2-stream RTE."""
      return _LW_DIFFUSIVE_FACTOR * (1 - 0.5 * ssa * (1.0 + asymmetry_factor))

    def gamma2_fn(ssa: tf.Tensor, asymmetry_factor: tf.Tensor) -> tf.Tensor:
      """The coefficient of the antiparallel irradiance in the 2-stream RTE."""
      return _LW_DIFFUSIVE_FACTOR * 0.5 * ssa * (1.0 - asymmetry_factor)

    gamma1 = tf.nest.map_structure(gamma1_fn, ssa, asymmetry_factor)
    gamma2 = tf.nest.map_structure(gamma2_fn, ssa, asymmetry_factor)

    def k_fn(gamma1: tf.Tensor, gamma2: tf.Tensor) -> tf.Tensor:
      """Computes the k parameter used in the reflectance and trasmittance."""
      return tf.math.sqrt(
          tf.maximum((gamma1 + gamma2) * (gamma1 - gamma2), _EPSILON)
      )

    k = tf.nest.map_structure(k_fn, gamma1, gamma2)

    def denom_fn(tau: tf.Tensor, k: tf.Tensor, gamma1: tf.Tensor) -> tf.Tensor:
      """The shared denominator of the reflection and transmittance function."""
      # This is the denominator that appears in equations 25 and 26 of Meador
      # and Weaver (1980).
      # As in the original RRTMGP Fortran code, this expression has been
      # refactored to avoid rounding errors when k, gamma1 are of very different
      # magnitudes.
      return k * (1 + tf.math.exp(-2.0 * tau * k)) + gamma1 * (
          1 - tf.math.exp(-2.0 * tau * k)
      )

    denominator = tf.nest.map_structure(denom_fn, optical_depth, k, gamma1)

    def reflectance_fn(
        tau: tf.Tensor, k: tf.Tensor, denom: tf.Tensor, gamma2: tf.Tensor
    ) -> tf.Tensor:
      """The reflectance as a function of the optical depth, `k`, `gamma2`."""
      return gamma2 * (1.0 - tf.math.exp(-2.0 * tau * k)) / denom

    r_diff = tf.nest.map_structure(
        reflectance_fn, optical_depth, k, denominator, gamma2
    )

    # Transmittance function.
    def transmittance_fn(
        tau: tf.Tensor, k: tf.Tensor, denom: tf.Tensor
    ) -> tf.Tensor:
      return 1 / denom * 2.0 * k * tf.math.exp(-tau * k)

    t_diff = tf.nest.map_structure(
        transmittance_fn, optical_depth, k, denominator
    )

    # From Toon et al. (JGR 1989) Eqs 26-27, first-order coefficient of the
    # Taylor series expansion of the Planck function in terms of the optical
    # depth.
    def b_1_fn(src_bottom, src_top, gamma1, gamma2, tau) -> tf.Tensor:
      return (src_bottom - src_top) / (tau * (gamma1 + gamma2))

    b_1 = tf.nest.map_structure(
        b_1_fn, level_src_bottom, level_src_top, gamma1, gamma2, optical_depth
    )

    def add(a, b):
      return tf.nest.map_structure(tf.math.add, a, b)

    # Compute longwave source function for upward and downward emission at cell
    # interfaces using linear-in-tau assumption.
    c_up_top = add(b_1, level_src_top)
    c_up_bottom = add(b_1, level_src_bottom)
    neg_b_1n = tf.nest.map_structure(tf.math.negative, b_1)
    c_down_top = add(neg_b_1n, level_src_top)
    c_down_bottom = add(neg_b_1n, level_src_bottom)

    def cell_center_src_fn(
        downstream_out: tf.Tensor,
        downstream_in: tf.Tensor,
        upstream_in: tf.Tensor,
        refl: tf.Tensor,
        tran: tf.Tensor,
        tau: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the flux at the cell center consistent with face fluxes.

      The cell center source is the residual that remains when one subtracts
      from the downstream outward flux two contributions:
      1. the upstream inward flux that is transmitted through the cell and
      2. the downstream inward flux that is reflected off the cell.

      Args:
        downstream_out: Downstream outward flux.
        downstream_in: Downstream inward flux.
        upstream_in: Upstream inward flux.
        refl: The grid cell reflectance.
        tran: The grid cell transmittance.
        tau: The grid cell optical depth.

      Returns:
        The directional radiative source at the cell center consistent with the
        given face sources [W / m^2].
      """
      src = math.pi * (
          downstream_out - refl * downstream_in - tran * upstream_in
      )
      # Filter out sources where the optical depth is too small.
      return tf.where(tf.greater(tau, _EPSILON), src, tf.zeros_like(src))

    src_up = tf.nest.map_structure(
        cell_center_src_fn,
        c_up_top,
        c_down_top,
        c_up_bottom,
        r_diff,
        t_diff,
        optical_depth,
    )
    src_down = tf.nest.map_structure(
        cell_center_src_fn,
        c_down_bottom,
        c_up_bottom,
        c_down_top,
        r_diff,
        t_diff,
        optical_depth,
    )
    return {
        'transmission': t_diff,
        'reflection': r_diff,
        'src_up': src_up,
        'src_down': src_down,
    }
