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

"""A library of the Monin-Obukhov Similarity Theory.

This library is useful for simulating atmospheric boundary layers [1], in
applications such as cloud simulations [2]. Neumann boundary conditions are
enforced for variable u, v, and T (optional).

References:
1. Mahrt, Larry. 2014. “Stably Stratified Atmospheric Boundary Layers.” Annual
Review of Fluid Mechanics 46 (1): 23–45.
2. Stevens, Bjorn, Chin-Hoh Moeng, Andrew S. Ackerman, Christopher S.
Bretherton, Andreas Chlond, Stephan de Roode, James Edwards, et al. 2005.
“Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine
Stratocumulus.” Monthly Weather Review 133 (6): 1443–62.

The computation is mainly based on the references, and the only 2 exceptions are
both computational, to avoid `nan`s:
1. tf.math.divide_no_nan in divisions
2. Enforcing numbers to be non-negative when computing sqrt

The main idea for case #1 is the following:
1. When the Obukhov length is `0`, it indicates there's no friction, and it's
   safe to set the shear stress to `0`
2. When the velocity at the first fluid layer is `0`, it means the velocity is
   the same as a non-slip wall, and the shear stress at the wall is `0`
In both cases `tf.math.divide_no_nan` gives the desired output.
"""
import functools
from typing import Mapping, Optional, Sequence, Text, Tuple

import numpy as np
from swirl_lm.base import initializer
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import physical_variable_keys_manager
from swirl_lm.equations import common
from swirl_lm.numerics import root_finder
from swirl_lm.physics import constants
from swirl_lm.utility import common_ops
from swirl_lm.utility import debug_print
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf

# The type of a state variable.
FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap

# The von Karman constant.
_KAPPA = 0.4
# The stability correction for momentum.
_PHI_M = 0.0
# The threshold of the ratio between the height of the first fluid layer and the
# surface roughness.
_HEIGHT_TO_SURFACE_ROUGHNESS_RATIO_THRESHOLD = 1.1
# Name of key to use for constant exchange coefficient for momentum flux.
_MOMENTUM_FLUX_EXCHANGE_COEFF_KEY = 'momentum'


class MoninObukhovSimilarityTheory(object):
  """A library of the Monin-Obukhov Similarity Theory."""

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
      vertical_dim: int,
  ):
    """Initializes the library."""
    self.params = params
    self.nu = params.nu
    self.halo_width = params.halo_width

    # Store the height of the first fluid layer above the ground.
    if params.use_stretched_grid[vertical_dim]:
      # For a stretched grid, the first non-halo grid point coordinate value is
      # the height above the ground.
      self.height = params.global_xyz[vertical_dim][0]
    else:
      # Under a uniform grid assumption, because the wall is at the mid-point
      # face between the first fluid layer and the halo layers, the height of
      # the first fluid layer above the ground is half of the grid spacing.
      self.height = 0.5 * params.grid_spacings[vertical_dim]

    assert (
        boundary_models := params.boundary_models
    ) is not None, '`boundary_models` must be provided in `params`.'
    most_params = boundary_models.most
    self.z_0 = most_params.z_0
    self.z_t = most_params.z_t
    self.u_star = most_params.u_star
    self.t_0 = most_params.t_0
    self.t_s = most_params.t_s
    self.heat_flux = most_params.heat_flux
    self.beta_m = most_params.beta_m
    self.beta_h = most_params.beta_h
    self.gamma_m = most_params.gamma_m
    self.gamma_h = most_params.gamma_h
    self.alpha = most_params.alpha
    self._active_scalars = list(most_params.active_scalar)

    self.enable_theta_reg = most_params.enable_theta_reg
    self.theta_max = most_params.theta_max
    self.theta_min = most_params.theta_min
    self.surface_gustiness = most_params.surface_gustiness

    self.dbg = most_params.debug

    self.bc_manager = (
        physical_variable_keys_manager.BoundaryConditionKeysHelper())
    # Precompute the vertical axis and the horizontal dimensions and axes from
    # the vertical dimension.
    self.vertical_dim = vertical_dim
    self.horizontal_dims = [0, 1, 2]
    self.horizontal_dims.remove(vertical_dim)
    self.dim_to_v_key = (common.KEY_U, common.KEY_V, common.KEY_W)
    dim_to_axis = (1, 2, 0)
    self.vertical_axis = dim_to_axis[self.vertical_dim]
    self.horizontal_axes = [dim_to_axis[dim] for dim in self.horizontal_dims]

    self.sea_level_ref = {
        var.name: var.value for var in most_params.sea_level_ref
    }

    self.exchange_coeff = {
        var.name: var.value for var in most_params.exchange_coeff
    }

  def is_active_scalar(self, scalar_name: str) -> bool:
    """Checks if MOST is applied to a specific scalar."""
    return scalar_name in self._active_scalars

  def _stability_correction_function(
      self,
      zeta: FlowFieldVal,
      theta: FlowFieldVal,
  ) -> Tuple[FlowFieldVal, FlowFieldVal]:
    """Computes the stability correction function based on buoyancy condition.

    Args:
      zeta: The normalized height that is defined as z / L, where z is the
        symbolic representation of a vertical coordinate, and L is the Obukhov
        length.
      theta: The potential temperature in units of K. It will be used to compute
        the buoyancy and determine the stability of the boundary layer.

    Returns:
      The value of the stability function computed at a specific height.
    """
    b = tf.nest.map_structure(lambda t: t - self.t_s, theta)

    def stable(z: tf.Tensor, option: Text) -> Tuple[tf.Tensor, tf.Tensor]:
      """Computes the correction functions for a stable boundary layer.

      References"
      [1] Stoll, Rob, and Fernando Porté-Agel. 2009. “Surface Heterogeneity
          Effects on Regional-Scale Fluxes in Stable Boundary Layers: Surface
          Temperature Transitions.” Journal of the Atmospheric Sciences 66 (2):
          412–31.
      [2] Stoll, Rob, and Fernando Porté-Agel. 2008. “Large-Eddy Simulation of
          the Stable Atmospheric Boundary Layer Using Dynamic Models with
          Different Averaging Schemes.” Boundary-Layer Meteorology 126 (1):
          1–28.

      Args:
        z: The normalized vertical coordinates.
        option: The type of stability function to be returned. If it's 'M',
          the stability function for momentum will be returned; otherwise
          the stability function for energy will be returned.

      Returns:
        A tuple of state variables with the first and second element being the
        stability correction functions for the momemtum and energy,
        respectively.
      """
      return -self.beta_m * z if option == 'M' else -self.beta_h * z

    def neutral(z: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
      """Computes the correction functions for a neutral boundary layer.

      Reference:
      [1] Stoll, Rob, and Fernando Porté-Agel. 2006. “Dynamic Subgrid-Scale
          Models for Momentum and Scalar Fluxes in Large-Eddy Simulations of
          Neutrally Stratified Atmospheric Boundary Layers over Heterogeneous
          Terrain.” Water Resources Research 42 (1): 2121.

      Args:
        z: The normalized vertical coordinates.

      Returns:
        A tuple of state variables with the first and second element being the
        stability correction functions for the momemtum and energy,
        respectively.
      """
      return tf.zeros_like(z)

    def unstable(z: tf.Tensor, option: Text) -> Tuple[tf.Tensor, tf.Tensor]:
      """Computes the correction functions for a unstable boundary layer.

      References"
      [1] Stoll, Rob, and Fernando Porté-Agel. 2009. “Surface Heterogeneity
          Effects on Regional-Scale Fluxes in Stable Boundary Layers: Surface
          Temperature Transitions.” Journal of the Atmospheric Sciences 66 (2):
          412–31.

      Args:
        z: The normalized vertical coordinates.
        option: The type of stability function to be returned. If it's 'M',
          the stability function for momentum will be returned; otherwise
          the stability function for energy will be returned.

      Returns:
        A tuple of state variables with the first and second element being the
        stability correction functions for the momemtum and energy,
        respectively.
      """
      alpha = 1.0

      if option == 'M':
        x = tf.math.pow(tf.maximum(1.0 - self.gamma_m * z, 0.0), 0.25)

        psi = 2.0 * tf.math.log(0.5 * (1.0 + x)) + tf.math.log(
            0.5 * (1.0 + x**2)) - 2.0 * tf.math.atan(x) + 0.5 * np.pi
      else:
        y = tf.math.pow(tf.maximum(1.0 - self.gamma_h * z, 0.0), 0.5)
        psi = 2.0 * alpha * tf.math.log(0.5 * (1.0 + y))

      return psi

    def stability_fn(
        bi: tf.Tensor,
        zi: tf.Tensor,
        option: Text,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
      """Generates the correct stability function based on buoyancy."""
      return tf.where(
          tf.less(bi, 0.0), unstable(zi, option),
          tf.where(tf.greater(bi, 0.0), stable(zi, option), neutral(zi)))

    psi_m = tf.nest.map_structure(
        functools.partial(stability_fn, option='M'), b, zeta)
    psi_h = tf.nest.map_structure(
        functools.partial(stability_fn, option='H'), b, zeta)

    return psi_m, psi_h

  def _richardson_number(
      self,
      theta: FlowFieldVal,
      u1: FlowFieldVal,
      u2: FlowFieldVal,
      height: float,
  ) -> FlowFieldVal:
    """Computes the bulk Richardson number.

    Args:
      theta: The potential temperature (in units of K) at the first node obove
        ground.
      u1: The first component of the free stream velocity.
      u2: The second component of the free stream velocity.
      height: The height of the first grid point.

    Returns:
      The bulk Richardson number.
    """

    def richardson_number(
        t: tf.Tensor,
        u: tf.Tensor,
        v: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the Richardson number."""
      return constants.G * height * tf.math.divide_no_nan(t - self.t_s,
                                                          (u**2 + v**2) * t)

    return tf.nest.map_structure(richardson_number, theta, u1, u2)

  def _normalized_height(
      self,
      theta: FlowFieldVal,
      u1: FlowFieldVal,
      u2: FlowFieldVal,
      height: float,
  ) -> FlowFieldVal:
    """Computes the height normalized by the Obukhov length 𝜁 = z / L.

    Based on the definition of the Obukhov length, surface shear stress and heat
    flux [1], an equation for the bulk Richardson number is derived as follows:
    Rb = g z (θ(z) - θₛ) / [|u|²θ(z)]
       = 𝜁 [ln(z / z₀) - 𝚿ₕ(𝜁)] / [ln(z / z₀) - 𝚿ᴍ(𝜁)]²,
    where 𝚿ₕ(𝜁) and 𝚿ᴍ(𝜁) are the stability correction functions for energy and
    momentum, respectively. The form of these two functions are determined by
    the buoyancy of the flow. 𝜁 can be solved iteratively with this equation.

    Reference:
    [1] Stoll, Rob, and Fernando Porté-Agel. 2009. “Surface Heterogeneity
          Effects on Regional-Scale Fluxes in Stable Boundary Layers: Surface
          Temperature Transitions.” Journal of the Atmospheric Sciences 66 (2):
          412–31.

    Args:
      theta: The potential temperature (in units of K) at the first node obove
        ground.
      u1: The first component of the free stream velocity.
      u2: The second component of the free stream velocity.
      height: The height of the first grid point.

    Returns:
      The Oubkhov length normalized height.
    """
    ln_z_by_z0 = tf.math.log(height / self.z_0)
    r_b = self._richardson_number(theta, u1, u2, height)
    max_iter = 10

    def err_fn(
        r: tf.Tensor,
        z: tf.Tensor,
        p_h: tf.Tensor,
        p_m: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the error function for the iterative solve with tf.Tensor."""
      return r - z * (ln_z_by_z0 - p_h) / (ln_z_by_z0 - p_m)**2

    def rhs_fn(zeta: FlowFieldVal) -> FlowFieldVal:
      """Defines the right hand side function for the iterative solve."""
      psi_m, psi_h = self._stability_correction_function(zeta, theta)
      err = tf.nest.map_structure(err_fn, r_b, zeta, psi_h, psi_m)

      return err

    zeta_init = tf.nest.map_structure(tf.zeros_like, theta)

    return root_finder.newton_method(rhs_fn, zeta_init, max_iter)

  def _maybe_regularize_potential_temperature(
      self, theta: FlowFieldVal) -> FlowFieldVal:
    """Applies bounds to the potential temperature is requested.

    Args:
      theta: The potential temperature.

    Returns:
      The potential temperature bounded by the user specified limites. If
      `enable_theta_reg` is `false`, the input theta will be returned without
      modifications.
    """
    if self.enable_theta_reg:
      theta = tf.nest.map_structure(
          lambda t: tf.maximum(tf.minimum(t, self.theta_max), self.theta_min),
          theta)

    return theta

  def _surface_shear_stress(
      self,
      rho: tf.Tensor,
      u_j: tf.Tensor,
      u_mag: tf.Tensor,
      drag_coefficient: float,
  ) -> tf.Tensor:
    """Computes the surface shear stress for a given drag coefficient.

    Reference: CLIMA Atmosphere Model.

    Args:
      rho: The density at the first level above ground.
      u_j: The j-th component of the fluid velocity in the first level above
        ground.
      u_mag: The magnitude of the horizontal fluid velocity in the first level
        above ground.
      drag_coefficient: The drag coefficient.

    Returns:
      The surface shear stress for a given velocity component.
    """
    return -rho * drag_coefficient * u_j * u_mag

  def _surface_shear_stress_and_heat_flux(
      self,
      theta: FlowFieldVal,
      u1: FlowFieldVal,
      u2: FlowFieldVal,
      rho: FlowFieldVal,
      height: float,
  ) -> Tuple[FlowFieldVal, FlowFieldVal, FlowFieldVal]:
    """Computes the surface shear stress and heat flux.

    Reference:
    Stoll, Rob, and Fernando Porté-Agel. 2008. “Large-Eddy Simulation of the
    Stable Atmospheric Boundary Layer Using Dynamic Models with Different
    Averaging Schemes.” Boundary-Layer Meteorology 126 (1): 1–28.

    Args:
      theta: The potential temperature (in units of K) at the first node above
        ground.
      u1: The first component of the free stream velocity.
      u2: The second component of the free stream velocity.
      rho: The density at the first node above ground.
      height: The height of the first grid point.

    Returns:
      A 3 component tuple, with elements being (in order) the surface shear
      stress for velocity component u1 and u2, and the surface heat flux.
    """

    def surface_heat_flux(
        theta_i: tf.Tensor,
        u_s_i,
        phi: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the surface heat flux."""
      return (self.t_s - theta_i) * u_s_i * _KAPPA / (
          tf.math.log(height / self.z_0) - phi)

    def most_drag_coefficient(phi_m, rho):
      return _KAPPA**2 / (rho * (tf.math.log(height / self.z_0) - phi_m)**2)

    u_mag = tf.nest.map_structure(
        lambda u, v: tf.math.sqrt(self.surface_gustiness**2 + u**2 + v**2),
        u1,
        u2,
    )
    zeta = self._normalized_height(theta, u1, u2, height)
    phi_m, phi_h = self._stability_correction_function(zeta, theta)

    if _MOMENTUM_FLUX_EXCHANGE_COEFF_KEY in self.exchange_coeff:
      # Use a fixed drag coefficient for momentum flux, bypassing the more
      # complicated MOST theory.
      drag_coefficient = self.exchange_coeff[_MOMENTUM_FLUX_EXCHANGE_COEFF_KEY]
    else:
      # Compute the drag coefficient using the nonlinear MOST model.
      drag_coefficient = tf.nest.map_structure(
          most_drag_coefficient, phi_m, rho
      )

    tau_13 = tf.nest.map_structure(
        self._surface_shear_stress, rho, u1, u_mag, drag_coefficient
    )
    tau_23 = tf.nest.map_structure(
        self._surface_shear_stress, rho, u2, u_mag, drag_coefficient
    )

    u_s = tf.nest.map_structure(
        lambda t_13, t_23: tf.math.pow(t_13**2 + t_23**2, 0.25),
        tau_13,
        tau_23,
    )
    q_3 = tf.nest.map_structure(surface_heat_flux, theta, u_s, phi_h)

    if self.dbg:
      tau_13 = debug_print.log_mean_min_max(tau_13, message='tau_13')
      tau_23 = debug_print.log_mean_min_max(tau_23, message='tau_23')
      u_s = debug_print.log_mean_min_max(u_s, message='u_s')
      q_3 = debug_print.log_mean_min_max(q_3, message='q_3')

    return tau_13, tau_23, q_3

  def surface_shear_stress_and_heat_flux_update_fn(
      self,
      states: FlowFieldMap,
  ) -> Tuple[FlowFieldVal, FlowFieldVal, FlowFieldVal]:
    """Computes the wall shear stress and heat flux.

    Args:
      states: A keyed dictionary of states. Must include 'u', 'v', 'w', 'theta'.

    Returns:
      A 3 component tuple, with elements being (in order) the surface shear
      stress for velocity component u1 and u2, and the surface heat flux. Note
      that each component is a 2D slice of a 3D tensor.
    """
    # Get the velocity components that are tangential to the ground.
    velocity_keys = list(common.KEYS_VELOCITY)
    del velocity_keys[self.vertical_dim]

    # Get the slice of the first fluid layer above the ground for the ground
    # tangential velocity and potential temperature. Assume the ground is always
    # on the low-index end in a dimension.
    u1 = common_ops.get_face(states[velocity_keys[0]], self.vertical_dim, 0,
                             self.halo_width)[0]
    u2 = common_ops.get_face(states[velocity_keys[1]], self.vertical_dim, 0,
                             self.halo_width)[0]
    theta = self._maybe_regularize_potential_temperature(
        common_ops.get_face(states['theta'], self.vertical_dim, 0,
                            self.halo_width)[0])
    rho = common_ops.get_face(states['rho'], self.vertical_dim, 0,
                              self.halo_width)[0]

    return self._surface_shear_stress_and_heat_flux(
        theta, u1, u2, rho, self.height
    )

  def _exchange_coefficient(
      self,
      theta: FlowFieldVal,
      u1: FlowFieldVal,
      u2: FlowFieldVal,
      height: float,
      varname: Optional[str] = None,
  ) -> FlowFieldVal:
    """Computes the exchange coefficient for the energy equation.

    Reference:
    Schneider, T. (n.d.). CLIMA Atmosphere Model. Caltech.

    Args:
      theta: The potential temperature (in units of K) at the first node above
        ground.
      u1: The first component of the free stream velocity.
      u2: The second component of the free stream velocity.
      height: The height of the first grid point.
      varname: The name of the variable for which the exchange coefficient is
        computed. If not provided, assume this variable is a scalar instead of
        a velocity/momentum component.

    Returns:
      The exchange coefficient for the energy equation.
    """
    zeta = self._normalized_height(theta, u1, u2, height)
    phi_m, phi_h = self._stability_correction_function(zeta, theta)

    ln_z = tf.math.log(height / self.z_0)

    phi_val = (
        phi_m if varname in common.KEYS_MOMENTUM +
        common.KEYS_VELOCITY else phi_h)

    # The coefficient is set to 0 when ln(z_m / z_0) equals Psi_M or Psi_H,
    # which suggests a 0 surface flux.
    return tf.nest.map_structure(
        lambda p_m, p_h: tf.math.divide_no_nan(_KAPPA**2, (ln_z - p_h) *  # pylint: disable=g-long-lambda
                                               (ln_z - p_m)), phi_m, phi_val)

  def surface_flux_update_fn(
      self,
      states: FlowFieldMap,
      varname: Optional[str] = None,
  ) -> FlowFieldVal:
    """Computes the diffusive flux at the surface.

    Note that this function supports both scalar and velocity/momentum fluxes.

    Reference:
    Schneider, T. (n.d.). CLIMA Atmosphere Model. Caltech. (Eq. 5.7)

    Args:
      states: A keyed dictionary of states. Must include 'u', 'v', 'w', 'theta',
      'rho', 'phi'.
      varname: The name of the variable for which the scalar flux is computed.

    Returns:
      The flux of `phi` at the surface.
    """
    # Get the velocity components that are tangential to the ground.
    velocity_keys = list(common.KEYS_VELOCITY)
    del velocity_keys[self.vertical_dim]

    # Get the slice of the first fluid layer above the ground for the ground
    # tangential velocity and potential temperature. Assume the ground is always
    # on the low-index end in a dimension.
    u1 = common_ops.get_face(states[velocity_keys[0]], self.vertical_dim, 0,
                             self.halo_width)[0]
    u2 = common_ops.get_face(states[velocity_keys[1]], self.vertical_dim, 0,
                             self.halo_width)[0]
    theta = self._maybe_regularize_potential_temperature(
        common_ops.get_face(states['theta'], self.vertical_dim, 0,
                            self.halo_width)[0])
    rho = common_ops.get_face(states['rho'], self.vertical_dim, 0,
                              self.halo_width)[0]
    phi_zm = common_ops.get_face(states['phi'], self.vertical_dim, 0,
                                 self.halo_width)[0]

    # Use the user defined sea surface reference value in the configuration if
    # available, otherwise use values at the first halo layer as the sea surface
    # reference.
    phi_z0 = self.sea_level_ref.get(
        varname,
        common_ops.get_face(states['phi'], self.vertical_dim, 0,
                            self.halo_width - 1)[0])

    # Note that if an exchange coefficient is provided in the config, it will
    # override the MOST model (not computed).
    c_h = self.exchange_coeff.get(
        varname, self._exchange_coefficient(theta, u1, u2, self.height, varname)
    )

    def scalar_flux(
        rho_i: tf.Tensor,
        c_h_i: tf.Tensor,
        u1_i: tf.Tensor,
        u2_i: tf.Tensor,
        phi_zm_i: tf.Tensor,
        phi_z0_i: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the energy flux."""
      return (
          -rho_i
          * c_h_i
          * tf.math.sqrt(self.surface_gustiness**2 + u1_i**2 + u2_i**2)
          * (phi_zm_i - phi_z0_i)
      )

    if isinstance(phi_z0, Sequence) and isinstance(c_h, Sequence):
      sc_flux = tf.nest.map_structure(
          scalar_flux, rho, c_h, u1, u2, phi_zm, phi_z0
      )
    elif isinstance(c_h, Sequence):
      sc_flux = tf.nest.map_structure(
          functools.partial(scalar_flux, phi_z0_i=phi_z0),
          rho,
          c_h,
          u1,
          u2,
          phi_zm,
      )
    elif isinstance(phi_z0, Sequence):
      flux_fn = lambda rho_i, u1_i, u2_i, phi_zm_i, phi_z0_i: scalar_flux(  # pylint: disable=g-long-lambda
          rho_i, c_h, u1_i, u2_i, phi_zm_i, phi_z0_i
      )
      sc_flux = tf.nest.map_structure(flux_fn, rho, u1, u2, phi_zm, phi_z0)
    else:
      flux_fn = lambda rho_i, u1_i, u2_i, phi_zm_i: scalar_flux(  # pylint: disable=g-long-lambda
          rho_i, c_h, u1_i, u2_i, phi_zm_i, phi_z0
      )
      sc_flux = tf.nest.map_structure(flux_fn, rho, u1, u2, phi_zm)

    if self.dbg:
      sc_flux = debug_print.log_mean_min_max(sc_flux, message='scalar_flux')

    return sc_flux

  def neumann_bc_update_fn(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Computes the Neumann BC for all variables.

    Note: While this function is still used in geophysical_flow.py, it only sets
    values in halos, so that the gradient computed from these values is
    consistent with the shear stress/flux obtained from the MO similarity model.
    However, its functionality has been superseded by the preceding functions
    that set the diffusive fluxes directly. So in effect, this function is no
    longer necessary and could be removed and deprecated.

    Args:
      kernel_op: An object holding a library of kernel operations.
      states: A keyed dictionary of state variables.
      additional_states: A list of states that are needed by the update fn, but
        will not be updated by the main governing equations.

    Returns:
      An update function for `additional_states` that updates the boundary
      condition.
    """
    # Computes the boundary condition for all variables in states except for
    # those listed here.
    excluded_vars = ('p', 'rho', common.KEYS_VELOCITY[self.vertical_dim])

    helper_states = {
        'u': states['u'],
        'v': states['v'],
        'w': states['w'],
        'rho': states['rho'],
    }

    def get_potential_temperature(variables):
      """Retrieves potential temperature from a dictionary of variables."""
      if 'T' in variables:
        # Temperature is used interchangeably with potential temperature because
        # they are almost identical on the ground.
        return variables['T']
      elif 'theta' in variables:
        return variables['theta']
      elif 'theta_li' in variables:
        return variables['theta_li']
      else:
        return None

    theta = get_potential_temperature(states)
    if theta is None:
      theta = get_potential_temperature(additional_states)
    if theta is None:
      raise ValueError(
          'Potential temperature is required by the MOST model but is not '
          'provided.')
    helper_states.update({'theta': theta})

    # Get the turbulent viscosity and diffusivity.
    nu_t = additional_states.get('nu_t', 0.0)
    if self.params.sgs_model.WhichOneof('sgs_model_type') == 'smagorinsky':
      pr_t = self.params.sgs_model.smagorinsky.pr_t
    elif (self.params.sgs_model.WhichOneof('sgs_model_type') ==
          'smagorinsky_lilly'):
      pr_t = self.params.sgs_model.smagorinsky_lilly.pr_t
    elif self.params.sgs_model.WhichOneof('sgs_model_type') == 'vreman':
      pr_t = self.params.sgs_model.vreman.pr_t
    else:
      # The turbulent Prandtl number is set to 1 for other SGS models.
      pr_t = 1.0

    def tensor_op(op, a, b):
      """Applies operation `op` to `a` and `b`."""
      if isinstance(a, Sequence) and isinstance(b, Sequence):
        return tf.nest.map_structure(op, a, b)
      elif isinstance(a, Sequence):
        return tf.nest.map_structure(lambda a_i: op(a_i, b), a)
      elif isinstance(b, Sequence):
        return tf.nest.map_structure(lambda b_i: op(a, b_i), b)
      else:
        return op(a, b)

    d_t = tensor_op(tf.math.divide, nu_t, pr_t)
    nu_total = tensor_op(tf.math.add, self.params.nu, nu_t)

    sum_fn = (
        lambda f: kernel_op.apply_kernel_op_x(f, 'ksx'),
        lambda f: kernel_op.apply_kernel_op_y(f, 'ksy'),
        lambda f: kernel_op.apply_kernel_op_z(f, 'ksz', 'kszsh'),
    )[self.vertical_dim]
    interp_fn = lambda f: tf.nest.map_structure(lambda g: 0.5 * g, sum_fn(f))

    additional_states_new = dict(additional_states)
    for key, val in states.items():
      bc_key = self.bc_manager.generate_bc_key(key, self.vertical_dim, 0)
      if key in excluded_vars or bc_key not in additional_states:
        continue

      helper_states.update({'phi': val})
      flux = self.surface_flux_update_fn(helper_states, key)

      # Compute the gradient of the variable at the surface. Note that the
      # diffusive flux is computed as: flux = -\rho D \nabla\phi \delta_{ik},
      # where `k` indicates the vertical direction.
      if key in common.KEYS_VELOCITY:
        d_total = nu_total
      else:
        d_total = tensor_op(tf.math.add, self.params.diffusivity(key), d_t)
      rho_d = tensor_op(tf.math.multiply, states['rho'], d_total)
      rho_d_face = common_ops.get_face(
          interp_fn(rho_d), self.vertical_dim, 0, self.halo_width, -1.0)[0]
      grad_phi = tensor_op(tf.math.divide, flux, rho_d_face)

      # Assume the outer halo layers retains the same value.
      zeros = tf.nest.map_structure(tf.zeros_like, grad_phi)
      bc = []
      for _ in range(self.halo_width - 1):
        bc.append(zeros)
      bc.append(grad_phi)

      additional_states_new.update(
          {bc_key: tensor_op(tf.math.multiply, bc, self.height * 2.0)})

    return additional_states_new

  def _compute_obukhov_length(
      self,
      m: tf.Tensor,
      temperature: tf.Tensor,
      z_m: tf.Tensor,
  ) -> tf.Tensor:
    """Computes the Obukhov length.

    From Stoll and Porte-Agel [1],
      <tau_s> = -Cm <M(z_m)>^2,
      <q_s> = -Ch <M(z_m)>[T(z_m) - T_s],
      L = -u*^3 T_0 / (kappa g <q_s>),
    where Cm and Ch are the transfer coefficients that are functions of z_m / L.
    Based on these formulations, a quadratic equation can be derived by letting
    x = z_m / L, which takes the form a x^2 + b x + c = 0. The coefficients are:
      a = beta_m^2 + C / z_m * beta_h,
      b = 2 beta_m ln(z_m / z_0) + alpha C / z_m ln(z_m / z_t),
      c = ln(z_m / z_0)^2,
    where:
      C = (u^2 + v^2) / g * t_0 / (t - t_s).
    Note that the computation is all based on the equations, and the only
    exception is to avoid `nan`s with `tf.math.divide_no_nan`, and enforcing
    numbers to be non-negative when taking the square root. For example, for the
    former, while `g` and `t_0` are guaranteed to be non-zero, `(t - t_s)` in
    denominator could be `0` and needs special handling.

    Reference:
    1. Stoll, Rob, and Fernando Porté-Agel. 2009. “Surface Heterogeneity Effects
       on Regional-Scale Fluxes in Stable Boundary Layers: Surface Temperature
       Transitions.” Journal of the Atmospheric Sciences 66 (2): 412–31.

    Args:
      m: The mean velocity magnitude over the x-y plane at `z_m`.
      temperature: The mean temperature over the x-y plane at `z_m`.
      z_m: The height of the first grid point in the z direction.

    Returns:
      The Obukhov length.
    """
    param = tf.math.divide_no_nan(m**2 / constants.G * self.t_0,
                                  temperature - self.t_s)

    a = self.beta_m**2 + tf.math.divide_no_nan(param * self.beta_h, z_m)
    b = 2.0 * self.beta_m * tf.math.log(z_m / self.z_0) + tf.math.divide_no_nan(
        self.alpha * param * tf.math.log(z_m / self.z_t), z_m)
    c = tf.math.log(z_m / self.z_0)**2

    delta = tf.math.sqrt(tf.maximum(b**2 - 4.0 * a * c, 0.0))
    l_inv_1 = tf.math.divide_no_nan(-b - delta, 2.0 * a)
    l_inv_2 = tf.math.divide_no_nan(-b + delta, 2.0 * a)
    l_inv = tf.cond(
        pred=tf.less(a, 0.0), true_fn=lambda: l_inv_1, false_fn=lambda: l_inv_2)

    return tf.math.divide_no_nan(z_m, l_inv)

  def _compute_monin_obukhov_length_scale(self, u_star, temperature, heat_flux):
    """Computes the Monin-Obukhov length scale."""
    return tf.nest.map_structure(
        lambda u_star_i, t_i:
        tf.math.divide_no_nan(-u_star_i**3 * t_i,
                              _KAPPA * constants.G * heat_flux),
        u_star, temperature)

  def _compute_surface_heat(self, u_star):
    """Computes the surface heat -T*."""
    return tf.nest.map_structure(
        lambda u_star_i: tf.math.divide_no_nan(self.heat_flux, u_star_i), u_star
    )

  def _compute_shear_stresses(self, u, v, z, replicas):
    """Computes the shear stresses 𝛕₀₂ and 𝛕₁₂."""
    u_norm = tf.nest.map_structure(
        lambda u_i, v_i: tf.math.sqrt(u_i**2 + v_i**2), u, v)
    u_mean = tf.squeeze(common_ops.global_mean(u_norm, replicas))
    u_star = tf.math.divide_no_nan(u_mean * _KAPPA,
                                   tf.math.log(z / self.z_0) - _PHI_M)
    return (tf.nest.map_structure(
        lambda u_i: tf.math.divide_no_nan(-u_star**2 * u_i, u_mean), u),
            tf.nest.map_structure(
                lambda v_i: tf.math.divide_no_nan(-u_star**2 * v_i, u_mean), v))

  def _compute_friction_velocity(self, u, v, z, replicas):
    """Computes the friction velocity."""
    tau_vertical_0, tau_vertical_1 = self._compute_shear_stresses(
        u, v, z, replicas)
    return tf.nest.map_structure(
        lambda tau_0_i, tau_1_i:
        tf.math.pow(tau_0_i**2 + tau_1_i**2, 0.25),
        tau_vertical_0, tau_vertical_1)

  def _compute_nondimensional_gradient(self, u, v, temperature, z, replicas):
    """Computes the nondimensional gradient."""
    u_star = self._compute_friction_velocity(u, v, z, replicas)
    l = tf.nest.map_structure(
        lambda l_i:
        -l_i, self._compute_monin_obukhov_length_scale(
            u_star, temperature, self.heat_flux)
    )
    if self.heat_flux >= 0.0:
      return tf.nest.map_structure(
          lambda l_i: tf.math.pow(
              tf.maximum(1.0 - tf.math.divide_no_nan(15.0 * z, l_i), 0.0),
              -0.25), l)
    return tf.nest.map_structure(
        lambda l_i: 1.0 + tf.math.divide_no_nan(4.7 * z, l_i), l)

  def _compute_dimensional_gradient(self, f_star, phi, z):
    """Computes the dimensional gradient that is used for the Neumann BC."""
    return tf.nest.map_structure(
        lambda f_star_i, phi_i: tf.math.divide_no_nan(
            f_star_i * phi_i, _KAPPA * z), f_star, phi)

  def _check_additional_states_keys(
      self,
      additional_states: FlowFieldMap,
      update_bc_t: bool,
  ) -> None:
    """Checks if all required keys exist in `additional_states`.

    Args:
      additional_states: A list of states that are needed by the update fn, but
        will not be updated by the main governing equations.
      update_bc_t: An indicator of whether the temperature boundary condition
        will be updated.

    Raises:
      ValueError: If not all required keys are contained in `additional_states`.
    """
    velocity_keys = [self.dim_to_v_key[dim] for dim in self.horizontal_dims]
    required_bc_keys = set()
    for horizontal_v_key in velocity_keys:
      bc_v_key = self.bc_manager.generate_bc_key(horizontal_v_key,
                                                 self.vertical_dim, 0)
      required_bc_keys.add(bc_v_key)
    required_t_bc_key = self.bc_manager.generate_bc_key('T', self.vertical_dim,
                                                        0)
    if not required_bc_keys.issubset(additional_states.keys()):
      raise ValueError(
          'Required fields {} missing from `additional_states`.'.format(
              required_bc_keys))

    if update_bc_t and required_t_bc_key not in additional_states.keys():
      raise ValueError(
          '{} is not in `additional_states` but needs to be updated'.format(
              required_t_bc_key))

  def init_fn(
      self,
      config: grid_parametrization.GridParametrization,
      coordinates: initializer.ThreeIntTuple,
      update_bc_t: bool,
  ) -> Mapping[Text, tf.Tensor]:
    """Generates the required initial fields by the simulation.

    Args:
      config: An instance of `grid_parametrization.GridParametrization`.
      coordinates: A tuple that specifies the replica's grid coordinates in
        physical space.
      update_bc_t: An option of whether the Monin-Obukhov Similarity Theory is
        applied to temperature. If true, the temperature boundary condition
        will be included (e.g. 'bc_T_2_0' if the height dimension is along
        the z direction). Otherwise, only the horizontal velocity components'
        boundary conditions will be included (e.g. 'bc_u_2_0' and 'bc_v_2_0' if
        the height dimension is along the z direction).

    Returns:
      A dictionary of state variables that are required by the Monin-Obukhov
      Similarity Theory.
    """

    def states_init(initial_value_fn) -> tf.Tensor:
      """Assigns value to a tensor with `initial_value_fn`."""
      return initializer.partial_mesh_for_core(
          config,
          coordinates,
          initial_value_fn,
          pad_mode='SYMMETRIC',
          mesh_choice=initializer.MeshChoice.PARAMS,
      )
    # pylint: disable=g-long-lambda
    init_fn_zeros = lambda xx, yy, zz, lx, ly, lz, coord: tf.zeros_like(
        xx, dtype=xx.dtype)
    # pylint: enable=g-long-lambda

    output = {}
    velocity_keys = [self.dim_to_v_key[dim] for dim in self.horizontal_dims]
    for horizontal_v_key in velocity_keys:
      bc_v_key = self.bc_manager.generate_bc_key(horizontal_v_key,
                                                 self.vertical_dim, 0)
      output.update({bc_v_key: states_init(init_fn_zeros)})
    if update_bc_t:
      bc_t_key = self.bc_manager.generate_bc_key('T', self.vertical_dim, 0)
      output.update({bc_t_key: states_init(init_fn_zeros)})

    return output

  def _psi_m(self, z_m, l):
    """The stability correction for momentum."""
    return tf.math.divide_no_nan(-self.beta_m * z_m, l)

  def _psi_h(self, z_m, l):
    """The stability correction for heat."""
    return tf.math.divide_no_nan(-self.beta_h * z_m, l)

  def _c_m(self, z_m, l):
    """The stability corrected log-law for momentum."""
    return tf.math.divide_no_nan(_KAPPA**2, (tf.math.log(z_m / self.z_0) -
                                             self._psi_m(z_m, l))**2)

  def _c_h(self, z_m, l):
    """The stability corrected log-law for heat."""
    return tf.math.divide_no_nan(
        _KAPPA**2, (tf.math.log(z_m / self.z_0) - self._psi_m(z_m, l)) *
        (self.alpha * tf.math.log(z_m / self.z_t) - self._psi_h(z_m, l)))

  def _tau_s_average(self, z_m, m, l):
    """The average surface stress."""
    return -self._c_m(z_m, l) * m**2

  def _q_s_average(self, z_m, m, t_m, t_s, l):
    """The average surface heat flux."""
    return -self._c_h(z_m, l) * m * (t_m - t_s)

  def _get_slice(
      self,
      f: FlowFieldVal,
      idx: int,
  ) -> FlowFieldVal:
    """Returns a horizontal slice of `f` at level `idx`."""
    slices = common_ops.get_face(f, self.vertical_dim, 0, idx)
    if isinstance(f, tf.Tensor):
      return slices[0]
    else:
      return slices if self.vertical_dim == 2 else slices[0]

  def _expand_state(
      self, f: FlowFieldVal,
      params: grid_parametrization.GridParametrization) -> FlowFieldVal:
    """Expands the state variable along the vertical dimension."""
    if self.vertical_dim == 2:
      if isinstance(f, tf.Tensor):
        return tf.tile(f, [params.nz, 1, 1])
      else:
        return f * params.nz
    else:
      ns = [params.nx, params.ny]
      repeats = [1, 1]
      repeats[self.vertical_dim] = ns[self.vertical_dim]
      return tf.nest.map_structure(lambda f_i: tf.tile(f_i, repeats), f)

  def _get_horizontal_slices(
      self,
      states: FlowFieldMap,
      t: FlowFieldVal,
      params: grid_parametrization.GridParametrization,
      idx: int,
      strip_halos: bool = False,
  ):
    """Gets horizontal velocity components and temperature fields at `idx`."""
    halo = params.halo_width
    halos = [halo] * 3
    halos[self.vertical_dim] = 0
    dim_to_horizontal_velocity = {}
    for dim in self.horizontal_dims:
      v_key = self.dim_to_v_key[dim]
      horizontal_slice = self._get_slice(states[v_key], idx)
      dim_to_horizontal_velocity.update({dim: horizontal_slice})
    temperature = self._get_slice(t, idx)
    if strip_halos:
      dim_to_horizontal_velocity.update({
          dim: common_ops.strip_halos(f, halos)
          for dim, f in dim_to_horizontal_velocity.items()
      })
      temperature = common_ops.strip_halos(temperature, halos)
    return dim_to_horizontal_velocity, temperature

  def porte_agel_model_update_fn(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Computes the Neumann BC for u, v, T (optional) with Porte Agel's model.

    The wall shear stress is computed and applied as boundary conditions to the
    wall normal shear components of u and v (stream and spanwise velocity). The
    mean shear stress is computed based on the Monin-Obukhov similarity theory
    [1], which is applied locally by accounting for the velocity fluctuation in
    the first computation layer above the ground[2]. Note that field `T` must
    exist in either `states` or `additional_states`, but not both.

    Note that the computation is all based on the references, and the only
    exception is to avoid `nan`s with `tf.math.divide_no_nan`.

    References:
    [1] Stoll, Rob, and Fernando Porté-Agel. 2009. “Surface Heterogeneity
        Effects on Regional-Scale Fluxes in Stable Boundary Layers: Surface
        Temperature Transitions.” Journal of the Atmospheric Sciences 66 (2):
        412–31.
    [2] Porté-Agel, Fernando, Charles Meneveau, and Marc B. Parlange. 2000. “A
        Scale-Dependent Dynamic Model for Large-Eddy Simulation: Application to
        a Neutral Atmospheric Boundary Layer.” Journal of Fluid Mechanics 415
        (July): 261–84.

    Args:
      kernel_op: An object holding a library of kernel operations.
      replica_id: The id of the replica.
      replicas: The replicas. In particular, a numpy array that maps grid
        coordinates to replica id numbers.
      states: A keyed dictionary of states that will be updated. If `T` is in
        `states`, the boundary condition for `T` (a.k.a `bc_T_2_0`) will be
        updated.
      additional_states: A list of states that are needed by the update fn, but
        will not be updated by the main governing equations. If `T` is in
        `additional_states`, the boundary condition for `T` (a.k.a `bc_T_2_0`)
        will not be updated.
      params: An instance of `grid_parametrization.GridParametrization`.

    Returns:
      An update function for `additional_states` that updates the boundary
      condition.
    """
    del kernel_op, replica_id

    if 'T' in states.keys():
      update_bc_t = True
      state_t = states['T']
    elif 'T' in additional_states.keys():
      update_bc_t = False
      state_t = additional_states['T']
    else:
      raise ValueError('Field `T` is required to generate the Neumann boundary '
                       'condition with the Monin-Obukhov similarity theory, '
                       'but is not found.')

    dh = [params.dx, params.dy, params.dz]
    height_m = dh[self.vertical_dim]

    dim_to_horizontal_velocity, t = self._get_horizontal_slices(
        states, state_t, params, params.halo_width)
    horizontal_velocity_fields = list(dim_to_horizontal_velocity.values())

    nu_slice = self._get_slice(additional_states['nu_t'], params.halo_width)
    nu = tf.nest.map_structure(
        lambda nu_slice_i: nu_slice_i + self.nu, nu_slice)
    v_0_sq = tf.nest.map_structure(
        lambda v_i: v_i**2, horizontal_velocity_fields[0])
    v_1_sq = tf.nest.map_structure(
        lambda v_i: v_i**2, horizontal_velocity_fields[1])
    m = tf.nest.map_structure(
        lambda v_0_i, v_1_i: tf.math.sqrt(v_0_i + v_1_i), v_0_sq, v_1_sq)

    m_avg = tf.squeeze(
        common_ops.global_mean(m, replicas, axis=self.horizontal_dims)[0])
    t_avg = tf.squeeze(
        common_ops.global_mean(t, replicas, axis=self.horizontal_dims)[0])

    l = self._compute_obukhov_length(m_avg, t_avg, height_m)

    tau_s_avg = self._tau_s_average(height_m, m_avg, l)

    tau = {}
    for dim, v in dim_to_horizontal_velocity.items():
      tau.update(
          {dim: tf.nest.map_structure(
              lambda v_i: tf.math.divide_no_nan(-tau_s_avg * v_i, m_avg), v)})

    # Regularizes the change in velocity so that flow at the boundary is not
    # in the reverted direction.
    dv = {}
    for dim, u in dim_to_horizontal_velocity.items():
      dv.update({dim: tf.nest.map_structure(
          lambda u_i, tau_i, nu_i: tf.sign(u_i) * tf.minimum(
              tf.abs(tf.math.divide_no_nan(tau_i * height_m, nu_i)),
              tf.abs(u_i)), u, tau[dim], nu)
                 })

    additional_states_new = {}
    most_bc_keys = set()
    for dim in dim_to_horizontal_velocity:
      bc_key_v = self.bc_manager.generate_bc_key(self.dim_to_v_key[dim],
                                                 self.vertical_dim, 0)
      if bc_key_v in additional_states:
        most_bc_keys.add(bc_key_v)
        additional_states_new.update(
            {bc_key_v: self._expand_state(dv[dim], params)})
      bc_key_tau = (
          'bc_tau{horizontal_dim}{vertical_dim}_{vertical_dim}_0').format(
              horizontal_dim=dim, vertical_dim=self.vertical_dim)
      if bc_key_tau in additional_states:
        most_bc_keys.add(bc_key_tau)
        additional_states_new.update(
            {bc_key_tau: self._expand_state(tau[dim], params)})

    additional_states_new.update(
        {k: v for k, v in additional_states.items() if k not in most_bc_keys})

    if update_bc_t:
      q_s_avg = self._q_s_average(height_m, m_avg, t_avg, self.t_s, l)

      tau_t_vertical = tf.nest.map_structure(
          lambda m_i, t_i: -q_s_avg * tf.math.divide_no_nan(
              (m_i * (t_avg - self.t_s) + m_avg * (t_i - t_avg)),
              (m_avg * (t_avg - self.t_s))) * height_m, m, t)
      # Regularizes the temperature change so that the temperature at the
      # ground will not drop below the reference surface temperature.
      dt_max = t_avg - self.t_s
      dt = tf.nest.map_structure(
          lambda tau_t_vertical_i, nu_i: tf.sign(dt_max) * tf.minimum(
              tf.abs(tau_t_vertical_i * height_m / nu_i), tf.abs(dt_max)),
          tau_t_vertical, nu)

      bc_t_key = self.bc_manager.generate_bc_key('T', self.vertical_dim, 0)
      additional_states_new.update({bc_t_key: self._expand_state(dt, params)})
      bc_tau_t_key = 'bc_tauT{vertical_dim}_{vertical_dim}_0'.format(
          vertical_dim=self.vertical_dim)
      if bc_tau_t_key in additional_states:
        additional_states_new.update(
            {bc_tau_t_key: self._expand_state(tau_t_vertical, params)})

    return additional_states_new

  def moeng_model_update_fn(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Computes the Neumann BC for u, v, and T (optional) with Moeng's model.

    The boundary condition is updated for a wall following the Monin-Obukhov
    Similarity Theory, where `additional_states` with key `bc_u_2_0`,
    'bc_v_2_0', and 'bc_T_2_0 (optional) are updated and to be used as Neuamnn
    boundary conditions. Field `T` must exist in either `states` or
    `additional_states`, but not both.

    For F as a short-hand notation for u, v, and T:

    ∂F/∂z = φ(z/L)F* / κz. [2]
      where:
        κ: The von Karman constant (0.4),
        F*: for u, v, and T are computed as:
          u∗ = (𝛕₀₂² + 𝛕₁₂²)¹/⁴ is the friction velocity [1],
            where:
              𝛕ⱼ₂ = -[U(z)κ / {ln(z/z₀) − ΨM}]² uⱼ / U(z) [3],
                where:
                  U(z): <u>, the mean resolved horizontal velocity,
                  ΨM: The stability correction for momentum, in the case of
                      neutral stability, ΨM = 0
                  z₀:  The roughness length.
          T* = (w'T') / u∗ [2]
            where:
              w'T': The surface heat flux.
        φ(z/L) [1]:
          = [1 - (15 z/L)]⁻¹/⁴, for positive heat flux, w'T' (surface heating);
          = 1 + (4.7z/L), negative heat flux, w'T' (surface cooling).
            where:
              L ≡ u∗³ T/(κg (w'T')) [2] is the Monin-Obukhov length scale.
                where:
                  g: The acceleration of gravity,

    The following reference is use in this implementation:
    [1] Moeng, C.H., 1984. A large-eddy-simulation model for the study of
        planetary boundary-layer turbulence. Journal of the Atmospheric
        Sciences, 41(13), pp.2052-2062.
    [2] Mahrt, L., 2014. Stably stratified atmospheric boundary layers. Annual
        Review of Fluid Mechanics, 46, pp.23-45.
    [3] Porté-Agel, F., Meneveau, C. and Parlange, M.B., 2000. A scale-dependent
        dynamic model for large-eddy simulation: application to a neutral
        atmospheric boundary layer. Journal of Fluid Mechanics, 415, pp.261-284.

    Args:
      kernel_op: An object holding a library of kernel operations.
      replica_id: The id of the replica.
      replicas: The replicas. In particular, a numpy array that maps grid
        coordinates to replica id numbers.
      states: A keyed dictionary of states that will be updated. If `T` is in
        `states`, the boundary condition for `T` (a.k.a `bc_T_2_0`) will be
        updated.
      additional_states: A list of states that are needed by the update fn, but
        will not be updated by the main governing equations. If `T` is in
        `additional_states`, the boundary condition for `T` (a.k.a `bc_T_2_0`)
        will not be updated.
      params: An instance of `grid_parametrization.GridParametrization`.

    Returns:
      An update function for `additional_states` that updates the boundary
      condition.

    Raises:
      ValueError: If 'T' is not found in neither `states` nor
        `additional_states`.
    """
    del kernel_op, replica_id

    if 'T' in states.keys():
      update_bc_t = True
      t_full = states['T']
    elif 'T' in additional_states.keys():
      update_bc_t = False
      t_full = additional_states['T']
    else:
      raise ValueError('Field `T` is required to generate the Neumann boundary '
                       'condition with the Monin-Obukhov similarity theory, '
                       'but is not found.')

    self._check_additional_states_keys(additional_states, update_bc_t)

    dh = [params.dx, params.dy, params.dz]
    height = dh[self.vertical_dim]

    dim_to_horizontal_velocity, temperature = self._get_horizontal_slices(
        states, t_full, params, params.halo_width, strip_halos=True)
    horizontal_velocity_fields = list(dim_to_horizontal_velocity.values())

    phi = self._compute_nondimensional_gradient(horizontal_velocity_fields[0],
                                                horizontal_velocity_fields[1],
                                                temperature, height, replicas)
    u_star = self._compute_friction_velocity(horizontal_velocity_fields[0],
                                             horizontal_velocity_fields[1],
                                             height, replicas)

    paddings = [(params.halo_width, params.halo_width)] * 3
    paddings[self.vertical_dim] = (0, 0)
    dimensional_grad = self._compute_dimensional_gradient(u_star, phi, height)
    du = tf.nest.map_structure(
        lambda dg_i: dg_i * height, dimensional_grad)
    du = common_ops.pad(du, paddings, value=0.0)

    additional_states_new = {}
    most_bc_keys = set()
    for dim in dim_to_horizontal_velocity:
      bc_key_v = self.bc_manager.generate_bc_key(self.dim_to_v_key[dim],
                                                 self.vertical_dim, 0)
      if bc_key_v in additional_states:
        most_bc_keys.add(bc_key_v)
        additional_states_new.update({bc_key_v: self._expand_state(du, params)})

    for key, value in additional_states.items():
      if key not in most_bc_keys:
        additional_states_new.update({key: value})

    if update_bc_t:
      t_star = self._compute_surface_heat(u_star)
      dimensional_grad = self._compute_dimensional_gradient(t_star, phi, height)
      dt = tf.nest.map_structure(lambda dg_i: dg_i * height, dimensional_grad)
      dt = common_ops.pad(dt, paddings, value=0.0)
      bc_t_key = self.bc_manager.generate_bc_key('T', self.vertical_dim, 0)
      additional_states_new.update({bc_t_key: self._expand_state(dt, params)})

    return additional_states_new


def monin_obukhov_similarity_theory_factory(
    params: parameters_lib.SwirlLMParameters,
) -> MoninObukhovSimilarityTheory:
  """Generaets an object of `MoninObukhovSimilarityTheory`.

  Args:
    params: A object of the simulation parameter context. `boundary_models.most`
      and `nu` are used here.

  Returns:
    An instance of the `MoninObukhovSimilarityTheory` object.

  Raises:
    ValueError: If `most` is not defined in the parameter context.
    ValueError: If the gravity direction is absent.
    AssertionError: If the first fluid layer is below the tolerated surface
      roughness.
  """
  assert (
      boundary_models := params.boundary_models
  ) is not None, '`boundary_models` must be provided in `params`.'
  if not boundary_models.HasField('most'):
    raise ValueError(
        'Parameters for the Monin-Obukhov boundary layer model are not defined '
        'in the config.'
    )

  vertical_dim = params.g_dim
  if vertical_dim is None:
    raise ValueError(
        'Gravity must be defined to use the Monin-Obukhov boundary layer '
        'model.')

  # Get the height of the first fluid layer above the ground.
  if params.use_stretched_grid[vertical_dim]:
    # For a stretched grid, the first non-halo grid point coordinate value is
    # the height above the ground.
    height = params.global_xyz[vertical_dim][0]
  else:
    # Under a uniform grid assumption, because the wall is at the mid-point
    # face between the first fluid layer and the halo layers, the height of
    # the first fluid layer above the ground is half of the grid spacing.
    height = 0.5 * params.grid_spacings[vertical_dim]

  # If the height of the first fluid layer is close or below the surface
  # roughness, the wall is considered resolved, and a non-slip wall should be
  # used without the MOST model.
  z_0 = _HEIGHT_TO_SURFACE_ROUGHNESS_RATIO_THRESHOLD * boundary_models.most.z_0
  assert height > z_0, (
      f'The height of the first fluid layer ({height} m) is below the tolerated'
      f' surface roughness ({z_0} m). MOST model should be disabled and'
      ' replaced by a non-slip wall BC.'
  )

  return MoninObukhovSimilarityTheory(params, vertical_dim)
