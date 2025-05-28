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

Common symbols used in protected methods:
ssa: single-scattering albedo;
tau: optical depth;
g: asymmetry factor;
sw: shortwave;
lw: longwave;
gamma: exchange rate coefficient in the radiative transfer equation;
zenith: the zenith angle of collimated solar radiation.

References:
1. Shonk, Jonathan & Hogan, Robin. (2008). Tripleclouds: An Efficient Method for
   Representing Horizontal Cloud Inhomogeneity in 1D Radiation Schemes by Using
   Three Regions at Each Height. J. Climate. 21. 10.1175/2007JCLI1940.1.
2. Toon, Owen & McKay, C & Ackerman, T. & Santhanam, K.. (1989). Rapid
   calculation of radiative heating rates and photodissociation rates in
   Inhomogeneous multiple scattering atmospheres. Journal of Geophysical
   Research. 94. 10.1029/JD094iD13p16287.
3. Meador, W. E., and W. R. Weaver, 1980: Two-Stream Approximations to Radiative
   Transfer in Planetary Atmospheres: A Unified Description of Existing Methods
   and a New Improvement. J. Atmos. Sci., 37, 630–643.
4. Ukkonen, P. & Hogan, R. J. (2024) Twelve times faster yet accurate: A new
   state-of-the-art in radiation schemes via performance and spectral
   optimization. Journal of Advances in Modeling Earth Systems, 16(1). Portico.
"""

import math
from typing import Any, Callable, Dict, Optional

import numpy as np
import swirl_lm.physics.radiation.rte.rte_utils as utils
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_extension
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf


FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap
X0_KEY = utils.X0_KEY
PRIMARY_GRID_KEY = utils.PRIMARY_GRID_KEY
EXTENDED_GRID_KEY = utils.EXTENDED_GRID_KEY

# Secant of the longwave diffusivity angle per Fu et al. (1997).
_LW_DIFFUSIVE_FACTOR = 1.66
_EPSILON = 1e-6
# Minimum longwave optical depth required for nonzero source.
_MIN_TAU_FOR_LW_SRC = 1e-4
# Minimum value of the k parameter used in the transmittance.
_K_MIN = 1e-2
PRIMARY_GRID_KEY = utils.PRIMARY_GRID_KEY
EXTENDED_GRID_KEY = utils.EXTENDED_GRID_KEY


class MonochromaticTwoStreamSolver:
  """A library for solving the monochromatic two-stream radiative transfer."""

  def __init__(
      self,
      params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      g_dim: int,
      grid_extension_lib: Optional[grid_extension.GridExtension] = None,
  ):
    self.halos = params.halo_width
    self.g_dim = g_dim
    self.rte_utils = utils.RTEUtils(params, grid_extension_lib)
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

    planck_src_top = planck_srcs['planck_src_top']
    planck_src_bottom = planck_srcs['planck_src_bottom']

    combined_src_top = tf.nest.map_structure(
        geometric_mean, planck_src_top, self._shift_down_fn(planck_src_bottom)
    )
    combined_src_bottom = self._shift_up_fn(combined_src_top)

    return {
        'planck_src_top': combined_src_top,
        'planck_src_bottom': combined_src_bottom,
    }

  def _k_fn(self, gamma1: tf.Tensor, gamma2: tf.Tensor):
    """Computes the k parameter used in the transmittance."""
    k = tf.math.sqrt(
        tf.maximum((gamma1 + gamma2) * (gamma1 - gamma2), _EPSILON)
    )
    return tf.maximum(k, _K_MIN)

  def _rt_denominator_direct(
      self,
      gamma1: tf.Tensor,
      gamma2: tf.Tensor,
      tau: tf.Tensor,
      ssa: tf.Tensor,
      zenith: float,
  ):
    """"Shared denominator of direct reflectance and transmittance functions."""
    k = self._k_fn(gamma1, gamma2)
    denom = self._rt_denominator_diffuse(gamma1, gamma2, tau)
    k_mu_squared = tf.math.pow(k * tf.math.cos(zenith), 2)

    # Equation 14, multiplying top and bottom by exp(-k*tau) and rearranging to
    # avoid division by 0.
    return tf.where(
        tf.greater_equal(tf.abs(1.0 - k_mu_squared), _EPSILON),
        denom * (1.0 - k_mu_squared) / ssa,
        denom * _EPSILON / ssa,
    )

  def _direct_reflectance(
      self,
      gamma1: tf.Tensor,
      gamma2: tf.Tensor,
      gamma3: tf.Tensor,
      alpha2: tf.Tensor,
      tau: tf.Tensor,
      ssa: tf.Tensor,
      zenith: float,
  ):
    """Direct solar radiation reflectance (equation 14 of Meador and Weaver)."""
    k = self._k_fn(gamma1, gamma2)
    denom = self._rt_denominator_direct(gamma1, gamma2, tau, ssa, zenith)
    k_mu = k * tf.math.cos(zenith)

    # Transmittance of direct, unscattered beam.
    t0 = tf.math.exp(-tau / tf.math.cos(zenith))

    # Equation 14 of Meador and Weaver (1980), multiplying top and bottom by
    # exp(-k*tau) and rearranging to avoid division by 0.
    exp_minusktau = tf.math.exp(-k * tau)
    exp_minus2ktau = tf.math.exp(-2.0 * k * tau)
    return (
        (1.0 - k_mu) * (alpha2 + k * gamma3)
        - (1.0 + k_mu) * (alpha2 - k * gamma3) * exp_minus2ktau
        - 2.0 * (k * gamma3 - alpha2 * k_mu) * exp_minusktau * t0
    ) / denom

  def _direct_transmittance(
      self,
      gamma1: tf.Tensor,
      gamma2: tf.Tensor,
      gamma4: tf.Tensor,
      alpha1: tf.Tensor,
      tau: tf.Tensor,
      ssa: tf.Tensor,
      zenith: float,
  ):
    """Direct solar radiation transmittance (Meador and Weaver, equation 15)."""
    k = self._k_fn(gamma1, gamma2)
    denom = self._rt_denominator_direct(gamma1, gamma2, tau, ssa, zenith)
    k_mu = k * tf.math.cos(zenith)
    k_y4 = k * gamma4

    # Transmittance of direct, unscattered beam.
    t0 = tf.math.exp(-tau / tf.math.cos(zenith))

    exp_minusktau = tf.math.exp(-k * tau)
    exp_minus2ktau = tf.math.exp(-2.0 * k * tau)

    # Equation 15 (Meador and Weaver (1980)), refactored for numerical stability
    # by 1) multiplying top and bottom by exp(-k*tau), 2) multiplying through by
    # exp(-tau/mu0) to prefer underflow to overflow, and 3) omitting direct
    # transmittance.
    return (
        -(
            (1.0 + k_mu) * (alpha1 + k_y4) * t0
            - (1.0 - k_mu) * (alpha1 - k_y4) * exp_minus2ktau * t0
            - 2.0 * (k_y4 + alpha1 * k_mu) * exp_minusktau
        )
        / denom
    )

  def _rt_denominator_diffuse(
      self, gamma1: tf.Tensor, gamma2: tf.Tensor, tau: tf.Tensor
  ):
    """The shared denominator of the diffuse reflectance and transmittance."""

    # As in the original RRTMGP Fortran code, this expression has been
    # refactored to avoid rounding errors when k, gamma1 are of very different
    # magnitudes.
    k = self._k_fn(gamma1, gamma2)
    return k * (1 + tf.math.exp(-2.0 * tau * k)) + gamma1 * (
        1 - tf.math.exp(-2.0 * tau * k)
    )

  def _diffuse_reflectance(
      self, gamma1: tf.Tensor, gamma2: tf.Tensor, tau: tf.Tensor
  ):
    """The diffuse reflectance (equation 25 of Meador and Weaver (1980))."""
    k = self._k_fn(gamma1, gamma2)
    denom = self._rt_denominator_diffuse(gamma1, gamma2, tau)
    return gamma2 * (1.0 - tf.math.exp(-2.0 * tau * k)) / denom

  def _diffuse_transmittance(
      self, gamma1: tf.Tensor, gamma2: tf.Tensor, tau: tf.Tensor
  ):
    """The diffuse transmittance (equation 26 of Meador and Weaver (1980))."""
    k = self._k_fn(gamma1, gamma2)
    denom = self._rt_denominator_diffuse(gamma1, gamma2, tau)
    return 2.0 * k * tf.math.exp(-tau * k) / denom

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
      optical_depth: The pointwise optical depth.
      ssa: The pointwise single-scattering albedo.
      level_src_bottom: The Planck source at the bottom cell face
        [W / m^2 / sr].
      level_src_top: The Planck source at the top cell face [W / m^2 / sr].
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

    r_diff = tf.nest.map_structure(
        self._diffuse_reflectance, gamma1, gamma2, optical_depth,
    )

    t_diff = tf.nest.map_structure(
        self._diffuse_transmittance, gamma1, gamma2, optical_depth,
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
      return tf.where(
          tf.greater(tau, _MIN_TAU_FOR_LW_SRC), src, tf.zeros_like(src)
      )

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
        't_diff': t_diff,
        'r_diff': r_diff,
        'src_up': src_up,
        'src_down': src_down,
    }

  def sw_cell_properties(
      self,
      zenith: float,
      optical_depth: FlowFieldVal,
      ssa: FlowFieldVal,
      asymmetry_factor: FlowFieldVal,
  ) -> FlowFieldMap:
    """Computes shortwave reflectance and transmittance.

    Two-stream solutions to direct and diffuse reflectance and transmittance as
    a function of optical depth, single-scattering albedo, and asymmetry factor.
    Equations are developed in Meador and Weaver (1980).

    Args:
      zenith: The zenith angle of the shortwave collimated radiation.
      optical_depth: A 3D variable containing the pointwise optical depth.
      ssa: A 3D variable containing the pointwise single-scattering albedo.
      asymmetry_factor: A 3D variable containing the pointwise asymmetry factor.

    Returns:
      A dictionary containing the following items:
      't_diff': A 3D variable containing the diffuse transmittance.
      'r_diff': A 3D variable containing the diffuse reflectance.
      't_dir': A 3D variable containing the direct transmittance.
      'r_dir': A 3D variable containing the direct reflectance.
    """
    def gamma1_fn(ssa: tf.Tensor, g: tf.Tensor) -> tf.Tensor:
      return (8.0 - ssa * (5.0 + 3.0 * g)) * 0.25

    def gamma2_fn(ssa: tf.Tensor, g: tf.Tensor) -> tf.Tensor:
      return 3.0 * ssa * (1.0 - g) * 0.25

    def gamma3_fn(g: tf.Tensor) -> tf.Tensor:
      return (2.0 - 3.0 * tf.math.cos(zenith) * g) * 0.25

    def gamma4_fn(g: tf.Tensor) -> tf.Tensor:
      return 1.0 - gamma3_fn(g)

    def alpha1_fn(
        gamma1: tf.Tensor,
        gamma2: tf.Tensor,
        gamma3: tf.Tensor,
        gamma4: tf.Tensor,
    ) -> tf.Tensor:
      return gamma1 * gamma4 + gamma2 * gamma3

    def alpha2_fn(
        gamma1: tf.Tensor,
        gamma2: tf.Tensor,
        gamma3: tf.Tensor,
        gamma4: tf.Tensor,
    ) -> tf.Tensor:
      return gamma1 * gamma3 + gamma2 * gamma4

    # Exchange rate coefficients from Zdunkowski et al. (1980).
    gamma1 = tf.nest.map_structure(gamma1_fn, ssa, asymmetry_factor)
    gamma2 = tf.nest.map_structure(gamma2_fn, ssa, asymmetry_factor)
    gamma3 = tf.nest.map_structure(gamma3_fn, asymmetry_factor)
    gamma4 = tf.nest.map_structure(gamma4_fn, asymmetry_factor)
    alpha1 = tf.nest.map_structure(alpha1_fn, gamma1, gamma2, gamma3, gamma4)
    alpha2 = tf.nest.map_structure(alpha2_fn, gamma1, gamma2, gamma3, gamma4)

    # Diffuse reflectance and transmittance.
    r_diff = tf.nest.map_structure(
        self._diffuse_reflectance, gamma1, gamma2, optical_depth
    )
    t_diff = tf.nest.map_structure(
        self._diffuse_transmittance, gamma1, gamma2, optical_depth
    )

    # Direct reflectance and transmittance.
    def r_dir_fn(g1, g2, g3, a2, tau, ssa) -> tf.Tensor:
      return self._direct_reflectance(g1, g2, g3, a2, tau, ssa, zenith)

    def t_dir_fn(g1, g2, g4, a1, tau, ssa) -> tf.Tensor:
      return self._direct_transmittance(g1, g2, g4, a1, tau, ssa, zenith)

    r_dir_unconstrained = tf.nest.map_structure(
        r_dir_fn, gamma1, gamma2, gamma3, alpha2, optical_depth, ssa
    )
    t_dir_unconstrained = tf.nest.map_structure(
        t_dir_fn, gamma1, gamma2, gamma4, alpha1, optical_depth, ssa
    )

    # Constrain reflectance and transmittance to be positive and to not go above
    # physical limits by enforcing the constraint that the direct beam can
    # either be reflected, penetrate unscattetered to the bottom of the grid
    # cell, or penetrate through but be scattered on the way.

    def unscattered_beam_fn(g: tf.Tensor):
      """Transmittance of direct, unscattered beam."""
      return tf.math.exp(-g / tf.math.cos(zenith))

    # Direct transmittance.
    t0 = tf.nest.map_structure(unscattered_beam_fn, optical_depth)

    def constrain_r_dir(t0: tf.Tensor, r_dir: tf.Tensor):
      """Equation 9 of Hogan and Ukonnen (2024)."""
      return tf.clip_by_value(r_dir, 0.0, 1.0 - t0)

    def constrain_t_dir(t0: tf.Tensor, r_dir: tf.Tensor, t_dir: tf.Tensor):
      """Equation 10 of Hogan and Ukonnen (2024)."""
      return tf.clip_by_value(t_dir, 0.0, 1.0 - t0 - r_dir)

    r_dir = tf.nest.map_structure(
        constrain_r_dir, t0, r_dir_unconstrained
    )
    t_dir = tf.nest.map_structure(
        constrain_t_dir, t0, r_dir, t_dir_unconstrained
    )

    return {
        't_diff': t_diff,
        'r_diff': r_diff,
        't_dir': t_dir,
        'r_dir': r_dir,
    }

  def sw_cell_source(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      t_dir: FlowFieldVal,
      r_dir: FlowFieldVal,
      optical_depth: FlowFieldVal,
      toa_flux: FlowFieldVal,
      sfc_albedo_direct: FlowFieldVal,
      zenith: float,
      extended_grid_optical_props: Optional[FlowFieldMap] = None,
  ) -> Dict[str, FlowFieldMap]:
    """Computes monochromatic shortwave direct-beam flux and diffuse source.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      t_dir: A 3D variable for the direct-beam transmittance.
      r_dir: A 3D variable for the direct-beam reflectance.
      optical_depth: A 3D variable for the optical depth.
      toa_flux: The top of atmosphere incoming flux represented by a 2D plane.
      sfc_albedo_direct: The surface albedo with respect to direct radiation.
      zenith: The zenith solar angle.
      extended_grid_optical_props: An optional dictionary of optical properties
        for the extended grid above the simulation domain. It should contain the
        following keys: `t_dir, `r_dir`, and `optical_depth`.

    Returns:
      A dictionary containing a `FlowFieldMap` for the primary grid with key
      'primary' and possibly a `FlowFieldMap` for the extended grid with key
      'extended' if an extended grid is present.
      Each `FlowFieldMap` contains the following items:
      'src_up': A 3D variable for the cell center upward source.
      'src_down': A 3D variable for the cell center downward source.
      'flux_down_dir': A 3D variable for the solved downwelling direct-beam
        radiative flux at the bottom cell face.
      'sfc_src': A 2D variable for the shortwave source emanating from surface.
    """

    def t_noscat_fn(tau: tf.Tensor) -> tf.Tensor:
      """Transmittance of direct, unscattered beam."""
      return tf.math.exp(-tau / tf.math.cos(zenith))

    t_noscat = tf.nest.map_structure(t_noscat_fn, optical_depth)
    mu = tf.math.cos(zenith)

    # The vertical component of incident flux at the top boundary.
    flux_down_direct_bc = tf.nest.map_structure(
        lambda x: x * mu, toa_flux
    )

    # Global recurrent accumulation for the direct-beam downward flux at the
    # bottom cell face unraveling from the top of the atmosphere down to the
    # surface. The recurrence follows the simple relation:
    # flux_down_direct[i] = T_no_scatter[i] * flux_down_direct[i + 1]
    op = lambda w, x0: w * x0
    kwargs = {
        'w': t_noscat,
        X0_KEY: flux_down_direct_bc,
    }

    use_extended_grid = extended_grid_optical_props is not None
    kwargs_ext = None

    if use_extended_grid:
      t_noscat_ext = tf.nest.map_structure(
          t_noscat_fn, extended_grid_optical_props['optical_depth']
      )
      kwargs_ext = {
          'w': t_noscat_ext,
          'x0': flux_down_direct_bc,
      }

    flux_down_direct = self.rte_utils.cumulative_recurrent_op(
        replica_id,
        replicas,
        op,
        kwargs,
        dim=self.g_dim,
        forward=False,
        extended_grid_variables=kwargs_ext,
    )

    # Upward source from direct-beam reflection at the cell center.
    src_up = tf.nest.map_structure(
        lambda r, flux_down: r * flux_down,
        r_dir,
        self._shift_down_fn(flux_down_direct[PRIMARY_GRID_KEY]),
    )

    # Downward source from direct-beam transmittance at the cell center.
    src_down = tf.nest.map_structure(
        lambda t, flux_down: t * flux_down,
        t_dir,
        self._shift_down_fn(flux_down_direct[PRIMARY_GRID_KEY]),
    )

    # Direct-beam flux incident on the surface.
    flux_down_sfc = common_ops.slice_field(
        flux_down_direct[PRIMARY_GRID_KEY], self.g_dim, self.halos, size=1
    )
    core_idx = common_ops.get_core_coordinate(replicas, replica_id)[self.g_dim]

    def sfc_src_fn(flux_dn_0: tf.Tensor, sfc_albedo: tf.Tensor) -> tf.Tensor:
      return tf.cond(
          pred=tf.equal(core_idx, 0),
          true_fn=lambda: flux_dn_0 * sfc_albedo,
          false_fn=lambda: tf.zeros_like(flux_dn_0),
      )

    # The surface source is the direct-beam downard flux that is reflected from
    # the surface.
    sfc_src = tf.nest.map_structure(
        sfc_src_fn, flux_down_sfc, sfc_albedo_direct
    )

    srcs_primary = {
        'src_up': src_up,
        'src_down': src_down,
        'flux_down_dir': flux_down_direct[PRIMARY_GRID_KEY],
        'sfc_src': sfc_src,
    }

    if not use_extended_grid:
      return {PRIMARY_GRID_KEY: srcs_primary}

    # Upward source from direct-beam reflection for the extended grid.
    flux_down_direct_ext = flux_down_direct[EXTENDED_GRID_KEY]
    src_up_ext = tf.nest.map_structure(
        lambda r, flux_down: r * flux_down,
        extended_grid_optical_props['r_dir'],
        self._shift_down_fn(flux_down_direct_ext),
    )

    # Downward source from direct-beam transmittance for the extended grid.
    src_down_ext = tf.nest.map_structure(
        lambda t, flux_down: t * flux_down,
        extended_grid_optical_props['t_dir'],
        self._shift_down_fn(flux_down_direct_ext),
    )
    srcs_ext = {
        'src_up': src_up_ext,
        'src_down': src_down_ext,
        'flux_down_dir': flux_down_direct_ext,
        'sfc_src': sfc_src,
    }
    return {
        PRIMARY_GRID_KEY: srcs_primary,
        EXTENDED_GRID_KEY: srcs_ext,
    }

  def _solve_rte_2stream(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      t_diff: FlowFieldVal,
      r_diff: FlowFieldVal,
      src_up: FlowFieldVal,
      src_down: FlowFieldVal,
      top_flux_down: FlowFieldVal,
      sfc_emission: FlowFieldVal,
      sfc_reflectance: FlowFieldVal,
      extended_grid_optical_props: Optional[FlowFieldMap] = None,
  ) -> FlowFieldMap:
    r"""Solves the monochromatic two-stream radiative transfer equation.

    Given boundary conditions for the downward flux at the top of the atmosphere
    (`top_flux_down`) and the upward surface emission (`sfc_emission`), this
    computes the two-stream approximation of the upwelling and downwelling
    radiative fluxes at the cell faces based on the equations of Shonk and Hogan
    (2008), doi:10.1175/2007JCLI1940.1. All the computations here assume a
    single absorption interval (or `g` interval in RRTM nomenclature). This
    function needs to be applied to each `g` interval separately.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      t_diff: A 3D variable containing the cell center transmittance.
      r_diff: A 3D variable containing the cell center reflectance.
      src_up: A 3D variable containing the cell center upward emission.
      src_down: A 3D variable containing the cell center downward emission.
      top_flux_down: The downward component of the incoming flux at the top
        boundary of the atmosphere. This corresponds to the downward flux at the
        top face of the top fluid layer in the grid.
      sfc_emission: The upward surface emission. This corresponds to the bottom
        face of the bottom fluid layer in the grid.
      sfc_reflectance: The surface reflectance.
      extended_grid_optical_props: An optional dictionary of optical properties
        for the extended grid above the simulation domain. It should contain the
        following keys: `t_diff`, `r_diff`, `src_up`, and `src_down`.

    Returns:
      A dictionary containing fluxes at the bottom cell face:
      'flux_up' -> The upwelling radiative flux.
      'flux_down' -> The downwelling radiative flux.
      'flux_net' -> The net radiative flux.
    """
    use_extended_grid = extended_grid_optical_props is not None

    def global_recurrent_op(
        kwargs: FlowFieldMap,
        forward: bool,
        op: Callable[..., FlowFieldVal],
        extended_grid_vars: Optional[FlowFieldMap] = None,
    ) -> Dict[str, FlowFieldVal]:
      return self.rte_utils.cumulative_recurrent_op(
          replica_id,
          replicas,
          op,
          kwargs,
          dim=self.g_dim,
          forward=forward,
          extended_grid_variables=extended_grid_vars,
      )

    # Global recurrent accumulation for the albedo of the atmosphere below a
    # certain level, computed from the surface all the way to the top boundary.
    # The recurrence relation for albedo is taken from Shonk and Hogan Equation
    # 9.

    def albedo_op(r_diff: tf.Tensor, t_diff: tf.Tensor, x0: tf.Tensor):
      """Recurrent formula for albedo solution unraveling from the surface."""
      albedo_below = x0
      # Geometric series solution accounting for infinite reflection events.
      beta = tf.math.reciprocal(1.0 - r_diff * albedo_below)
      return r_diff + t_diff**2 * beta * albedo_below

    albedo_vars = {
        'r_diff': r_diff,
        't_diff': t_diff,
        X0_KEY: sfc_reflectance,
    }
    # Variables for the extended grid.
    albedo_vars_ext = None
    if use_extended_grid:
      albedo_vars_ext = {
          'r_diff': extended_grid_optical_props['r_diff'],
          't_diff': extended_grid_optical_props['t_diff'],
          'x0': sfc_reflectance,
      }
    albedo = global_recurrent_op(
        albedo_vars,
        forward=True,
        op=albedo_op,
        extended_grid_vars=albedo_vars_ext,
    )

    # Global recurrent accumulation for the aggregate upwelling source emission
    # computed from the surface all the way to the top of the atmosphere.
    # The coefficient and bias terms of the recurrence relation for emission are
    # taken from Shonk and Hogan Equation 11: the upward emission is a
    # combination of 1) the upward source from the grid cell center,
    # 2) aggregate emission from the atmosphere below, transmitted through the
    # cell, and 3) the downard source from the grid cell center that is
    # reflected from the atmosphere below and transmitted up through the layer.

    def upward_emission_op(src_up, src_down, t_diff, r_diff, albedo, x0):
      """Recurrent formula for upward emission starting from the surface."""
      emission_from_below = x0
      # Geometric series solution accounting for infinite reflection events.
      beta = tf.math.reciprocal(1.0 - r_diff * albedo)
      return src_up + t_diff * beta * (emission_from_below + src_down * albedo)

    emission_vars = {
        'src_up': src_up,
        'src_down': src_down,
        't_diff': t_diff,
        'r_diff': r_diff,
        'albedo': self._shift_up_fn(albedo[PRIMARY_GRID_KEY]),
        X0_KEY: sfc_emission,
    }
    # Variables for the extended grid.
    emission_vars_ext = None
    if use_extended_grid:
      emission_vars_ext = {
          'src_up': extended_grid_optical_props['src_up'],
          'src_down': extended_grid_optical_props['src_down'],
          't_diff': extended_grid_optical_props['t_diff'],
          'r_diff': extended_grid_optical_props['r_diff'],
          'albedo': self._shift_up_fn(albedo[EXTENDED_GRID_KEY]),
          'x0': sfc_emission,
      }
    emiss_up = global_recurrent_op(
        emission_vars,
        forward=True,
        op=upward_emission_op,
        extended_grid_vars=emission_vars_ext,
    )

    # Global recurrent accumulation for the downwelling radiative flux solution
    # at the bottom face, unravelling from the top of the atmosphere down to the
    # surface. The coefficient and bias terms are taken from Shonk and Hogan
    # Equation 13: the downward flux at the bottom face is a combination of
    # 1) the downard source emitted from the grid cell, 2) the downward flux
    # from the face above transmitted through the cell, and 3) the aggregate
    # upward emissions from the atmosphere below that are reflected from the
    # cell.

    def flux_down_op(emiss_up, src_down, t_diff, r_diff, albedo, x0):
      """Recurrent formula for downwelling flux initiating at top boundary."""
      flux_dn_from_above = x0
      # Geometric series solution accounting for infinite reflection events.
      beta = tf.math.reciprocal(1.0 - r_diff * albedo)
      return (t_diff * flux_dn_from_above + r_diff * emiss_up + src_down) * beta

    flux_down_vars = {
        'emiss_up': self._shift_up_fn(emiss_up[PRIMARY_GRID_KEY]),
        'src_down': src_down,
        't_diff': t_diff,
        'r_diff': r_diff,
        'albedo': self._shift_up_fn(albedo[PRIMARY_GRID_KEY]),
        X0_KEY: top_flux_down,
    }
    flux_down_vars_ext = None
    if use_extended_grid:
      flux_down_vars_ext = {
          'emiss_up': self._shift_up_fn(emiss_up[EXTENDED_GRID_KEY]),
          'src_down': extended_grid_optical_props['src_down'],
          't_diff': extended_grid_optical_props['t_diff'],
          'r_diff': extended_grid_optical_props['r_diff'],
          'albedo': self._shift_up_fn(albedo[EXTENDED_GRID_KEY]),
          'x0': top_flux_down,
      }
    flux_down = global_recurrent_op(
        flux_down_vars,
        forward=False,
        op=flux_down_op,
        extended_grid_vars=flux_down_vars_ext,
    )
    # After this, the extended grid can be forgotten.

    # The upwelling radiative flux at the bottom face can now be computed
    # directly from the cumulative upward emissions, the cumulative albedo of
    # the atmosphere below, and the downwelling radiative flux at the same face.
    flux_up = tf.nest.map_structure(
        lambda flux_dn, alb, emiss_up: flux_dn * alb + emiss_up,
        flux_down[PRIMARY_GRID_KEY],
        self._shift_up_fn(albedo[PRIMARY_GRID_KEY]),
        self._shift_up_fn(emiss_up[PRIMARY_GRID_KEY]),
    )

    fluxes = {
        'flux_up': flux_up,
        'flux_down': flux_down[PRIMARY_GRID_KEY],
    }

    if use_extended_grid:
      fluxes[f'{EXTENDED_GRID_KEY}_flux_up'] = tf.nest.map_structure(
          lambda flux_dn, alb, emiss_up: flux_dn * alb + emiss_up,
          flux_down[EXTENDED_GRID_KEY],
          self._shift_up_fn(albedo[EXTENDED_GRID_KEY]),
          self._shift_up_fn(emiss_up[EXTENDED_GRID_KEY]),
      )
      fluxes[f'{EXTENDED_GRID_KEY}_flux_down'] = flux_down[EXTENDED_GRID_KEY]
    return fluxes

  def lw_transport(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      t_diff: FlowFieldVal,
      r_diff: FlowFieldVal,
      src_up: FlowFieldVal,
      src_down: FlowFieldVal,
      top_flux_down: FlowFieldVal,
      sfc_src: FlowFieldVal,
      sfc_emissivity: FlowFieldVal,
      extended_grid_optical_props: Optional[Dict[str, Any]] = None,
  ) -> FlowFieldMap:
    """Computes the monochromatic longwave diffusive flux of the atmosphere.

    The upwelling and downwelling fluxes are computed from the equations of
    Shonk and Hogan (2008, doi:10.1175/2007JCLI1940.1) assuming a single
    reflection event. The net flux is also computed at every face. Note that the
    net flux is computed only at cell interfaces and does not correspond to the
    net flux into/out of the grid cell. For the overall grid cell net flux,
    one must take the difference of the net fluxes of the upper and bottom
    faces.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      t_diff: A 3D variable containing the cell center transmittance.
      r_diff: A 3D variable containing the cell center reflectance.
      src_up: A 3D variable containing the cell center Planck upward emission.
      src_down: A 3D variable containing the cell center Planck downward
        emission.
      top_flux_down: The downward flux at the top boundary of the atmosphere.
      sfc_src: The surface Planck source.
      sfc_emissivity: The surface emissivity.
      extended_grid_optical_props: An optional dictionary containing the optical
        properties of the extended grid. It should include `t_diff`, `r_diff`,
        `src_up`, and `src_down`.

    Returns:
      A dictionary containing fluxes at the bottom cell face [W/m^2]:
      'flux_up' -> The upwelling radiative flux.
      'flux_down' -> The downwelling radiative flux.
      'flux_net' -> The net radiative flux.
      If an extended grid is present, the following entries are also added:
      'extended_flux_up' -> The upwelling radiative flux in the extended grid.
      'extended_flux_down' -> The downwelling radiative flux in extended grid.
      'extended_flux_net' -> The net radiative flux in the extended grid.
    """
    # The source of diffuse radiation is the surface emission.
    sfc_emission = tf.nest.map_structure(
        lambda x, y: np.pi * x * y, sfc_emissivity, sfc_src
    )
    # The surface reflectance is just the complement of the surface emissivity.
    sfc_reflectance = tf.nest.map_structure(lambda x: 1.0 - x, sfc_emissivity)
    fluxes = {}
    fluxes.update(
        self._solve_rte_2stream(
            replica_id,
            replicas,
            t_diff,
            r_diff,
            src_up,
            src_down,
            top_flux_down,
            sfc_emission,
            sfc_reflectance,
            extended_grid_optical_props=extended_grid_optical_props,
        )
    )
    fluxes['flux_net'] = tf.nest.map_structure(
        tf.math.subtract, fluxes['flux_up'], fluxes['flux_down']
    )
    if extended_grid_optical_props is not None:
      fluxes[f'{EXTENDED_GRID_KEY}_flux_net'] = tf.nest.map_structure(
          tf.math.subtract,
          fluxes[f'{EXTENDED_GRID_KEY}_flux_up'],
          fluxes[f'{EXTENDED_GRID_KEY}_flux_down'],
      )
    return fluxes

  def sw_transport(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      t_diff: FlowFieldVal,
      r_diff: FlowFieldVal,
      src_up: FlowFieldVal,
      src_down: FlowFieldVal,
      sfc_src: FlowFieldVal,
      sfc_albedo: FlowFieldVal,
      flux_down_dir: FlowFieldVal,
      extended_grid_optical_props: Optional[Dict[str, Any]] = None,
  ) -> FlowFieldMap:
    """Computes the monochromatic shortwave fluxes in a layered atmosphere.

    The direct-beam downward flux `flux_down_dir` is added to the downwelling
    diffuse flux in the final solution.

    The upwelling and downwelling diffuse fluxes are computed from the equations
    of Shonk and Hogan (2008, doi:10.1175/2007JCLI1940.1) assuming a single
    reflection event. The net flux is also computed at every face. Note that the
    net flux is computed only at cell interfaces and does not correspond to the
    net flux into/out of the grid cell. For the overall grid cell net flux,
    one must take the difference of the net fluxes of the upper and bottom
    faces.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      t_diff: A 3D variable for the cell transmittance.
      r_diff: A 3D variable for the cell reflectance.
      src_up: A 3D variable for the cell center upward source.
      src_down: A 3D variable for the cell center downward source.
      sfc_src: A 2D variable for the direct-beam shortwave radiation reflected
        upward from the surface.
      sfc_albedo: The surface albedo.
      flux_down_dir: A 3D variable for the solved downwelling direct-beam
        radiative flux at the bottom cell face.
      extended_grid_optical_props: An optional dictionary containing the optical
        properties of the extended grid. It should include `t_diff`, `r_diff`,
        `src_up`, `src_down`, and `flux_down_dir`.

    Returns:
      A dictionary containing fluxes at the bottom cell face:
      'flux_up' -> The upwelling radiative flux.
      'flux_down' -> The downwelling radiative flux.
      'flux_net' -> The net radiative flux.
      If an extended grid is present, the following entries are also added:
      'extended_flux_up' -> The upwelling radiative flux in the extended grid.
      'extended_flux_down' -> The downwelling radiative flux in extended grid.
      'extended_flux_net' -> The net radiative flux in the extended grid.
    """
    fluxes = {}
    fluxes.update(
        self._solve_rte_2stream(
            replica_id,
            replicas,
            t_diff,
            r_diff,
            src_up,
            src_down,
            tf.nest.map_structure(tf.zeros_like, sfc_src),
            sfc_src,
            sfc_albedo,
            extended_grid_optical_props=extended_grid_optical_props,
        )
    )

    # Add the direct-beam contribution to the downwelling flux.
    fluxes['flux_down'] = tf.nest.map_structure(
        tf.math.add, fluxes['flux_down'], flux_down_dir
    )
    # The net flux computed at cell faces.
    fluxes['flux_net'] = tf.nest.map_structure(
        tf.math.subtract, fluxes['flux_up'], fluxes['flux_down']
    )

    # Handle the extended grid, if one is present.
    if extended_grid_optical_props is not None:
      # Add the direct-beam contribution to the downwelling flux.
      fluxes[f'{EXTENDED_GRID_KEY}_flux_down'] = tf.nest.map_structure(
          tf.math.add,
          fluxes[f'{EXTENDED_GRID_KEY}_flux_down'],
          extended_grid_optical_props['flux_down_dir'],
      )
      fluxes[f'{EXTENDED_GRID_KEY}_flux_net'] = tf.nest.map_structure(
          tf.math.subtract,
          fluxes[f'{EXTENDED_GRID_KEY}_flux_up'],
          fluxes[f'{EXTENDED_GRID_KEY}_flux_down'],
      )

    return fluxes
