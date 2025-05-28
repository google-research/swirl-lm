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

"""Utility functions that are commonly used in different equations."""

import functools
from typing import Callable, Dict, Literal, Optional, Text

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.boundary_condition import monin_obukhov_similarity_theory
from swirl_lm.numerics import calculus
from swirl_lm.numerics import derivatives
from swirl_lm.numerics import filters
from swirl_lm.numerics import interpolation
from swirl_lm.physics import constants
from swirl_lm.physics.thermodynamics import thermodynamics_pb2
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import stretched_grid_util
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap

# Parameters required by source terms due to subsidence velocity. Reference:
# Siebesma, A. Pier, A. Pier Siebesma, Christopher S. Bretherton, Andrew Brown,
# Andreas Chlond, Joan Cuxart, Peter G. Duynkerke, et al. 2003. “A Large Eddy
# Simulation Intercomparison Study of Shallow Cumulus Convection.” Journal of
# the Atmospheric Sciences.
_W_MAX = -0.65e-2
_Z_F1 = 1500.0
_Z_F5 = 2100.0
# Parameter required by the large-scale subsidence velocity, units 1/s.
# Reference:
# Stevens, Bjorn, Chin-Hoh Moeng, Andrew S. Ackerman, Christopher S. Bretherton,
# Andreas Chlond, Stephan de Roode, James Edwards, et al. 2005. “Evaluation of
# Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus.”
# Monthly Weather Review 133 (6): 1443–62.
_D = 3.75e-6


def shear_stress(
    deriv_lib: derivatives.Derivatives,
    mu: FlowFieldVal,
    u: FlowFieldVal,
    v: FlowFieldVal,
    w: FlowFieldVal,
    additional_states: FlowFieldMap,
    shear_bc_update_fn: Optional[Dict[Text, Callable[[FlowFieldVal],
                                                     FlowFieldVal]]] = None,
) -> FlowFieldMap:
  """Computes the viscous shear stress.

  The shear stress is computed as:
    τᵢⱼ = μ [∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ], i ≠ j
    τᵢⱼ = 2 μ [∂uᵢ/∂xⱼ - 1/3 ∂uₖ/∂xₖ δᵢⱼ], i = j
  Note that u, v, w are 3D tensors that are represented in the form of a list of
  2D x-y slices.

  Args:
    deriv_lib: An instance of the derivatives library.
    mu: Dynamic viscosity of the flow field.
    u: Velocity component in the x dimension, with updated boundary condition.
    v: Velocity component in the y dimension, with updated boundary condition.
    w: Velocity component in the z dimension, with updated boundary condition.
    additional_states: A dictionary container helper variables.
    shear_bc_update_fn: A dictionary of halo_exchange functions for the shear
      stress tensor.

  Returns:
    The 9 component stress stress tensor for each grid point. Values in the halo
    with width 1 is invalid.
  """
  du_dx = calculus.grad(deriv_lib, (u, v, w), additional_states)

  du_00 = du_dx[0][0]
  du_01 = du_dx[0][1]
  du_02 = du_dx[0][2]
  du_10 = du_dx[1][0]
  du_11 = du_dx[1][1]
  du_12 = du_dx[1][2]
  du_20 = du_dx[2][0]
  du_21 = du_dx[2][1]
  du_22 = du_dx[2][2]

  s00 = du_00
  s01 = tf.nest.map_structure(common_ops.average, du_01, du_10)
  s02 = tf.nest.map_structure(common_ops.average, du_02, du_20)
  s10 = s01
  s11 = du_11
  s12 = tf.nest.map_structure(common_ops.average, du_12, du_21)
  s20 = s02
  s21 = s12
  s22 = du_22

  div_u = tf.nest.map_structure(
      lambda x, y, z: x + y + z, du_00, du_11, du_22
  )

  tau_ij = lambda mu, s_ij: 2 * mu * s_ij
  tau_ii = lambda mu, s_ii, div_u: 2 * mu * (s_ii - div_u / 3)

  tau00 = tf.nest.map_structure(tau_ii, mu, s00, div_u)
  tau01 = tf.nest.map_structure(tau_ij, mu, s01)
  tau02 = tf.nest.map_structure(tau_ij, mu, s02)
  tau10 = tf.nest.map_structure(tau_ij, mu, s10)
  tau11 = tf.nest.map_structure(tau_ii, mu, s11, div_u)
  tau12 = tf.nest.map_structure(tau_ij, mu, s12)
  tau20 = tf.nest.map_structure(tau_ij, mu, s20)
  tau21 = tf.nest.map_structure(tau_ij, mu, s21)
  tau22 = tf.nest.map_structure(tau_ii, mu, s22, div_u)

  tau = {
      'xx': tau00,
      'xy': tau01,
      'xz': tau02,
      'yx': tau10,
      'yy': tau11,
      'yz': tau12,
      'zx': tau20,
      'zy': tau21,
      'zz': tau22,
  }

  if shear_bc_update_fn:
    for key, fn in shear_bc_update_fn.items():
      tau.update({key: fn(tau[key])})

  return tau

_ShearFluxFnArgTypes = [
    get_kernel_fn.ApplyKernelOp,
    derivatives.Derivatives,
    tf.Tensor,
    np.ndarray,
    FlowFieldVal,
    FlowFieldVal,
    FlowFieldVal,
    FlowFieldVal,
    FlowFieldVal,
    FlowFieldMap | None,
]


def shear_flux(
    params: parameters_lib.SwirlLMParameters,
) -> Callable[_ShearFluxFnArgTypes, FlowFieldMap]:
  """Generates a function that computes the shear fluxes at cell faces.

  Args:
    params: A object of the simulation parameter context. `boundary_models.most`
      is used here.

  Returns:
    A function that computes the 9 component shear stress tensor.
  """
  if (params.boundary_models is not None and
      params.boundary_models.HasField('most')):
    most = (
        monin_obukhov_similarity_theory.monin_obukhov_similarity_theory_factory(
            params))
  else:
    most = None

  def shear_flux_fn(
      kernel_op: get_kernel_fn.ApplyKernelOp,
      deriv_lib: derivatives.Derivatives,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      mu: FlowFieldVal,
      u: FlowFieldVal,
      v: FlowFieldVal,
      w: FlowFieldVal,
      rho: FlowFieldVal,
      helper_variables: FlowFieldMap,
  ) -> FlowFieldMap:
    """Computes the viscous shear stress on the cell faces.

    The shear stress is computed as:
      τᵢⱼ = μ [∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ], i ≠ j
      τᵢⱼ = 2 μ [∂uᵢ/∂xⱼ - 1/3 ∂uₖ/∂xₖ δᵢⱼ], i = j
    Note that u, v, w are 3D tensors that are represented in the form of a list
    of 2D x-y slices.

    Locations of the fluxes:
      tau00/tau_xx: x face, i - 1/2 stored at i;
      tau01/tau_xy: y face, j - 1/2 stored at j;
      tau02/tau_xz: z face, k - 1/2 stored at k;
      tau10/tau_yx: x face, i - 1/2 stored at i;
      tau11/tau_yy: y face, j - 1/2 stored at j;
      tau12/tau_yz: z face, k - 1/2 stored at k;
      tau20/tau_zx: x face, i - 1/2 stored at i;
      tau21/tau_zy: y face, j - 1/2 stored at j;
      tau22/tau_zz: z face, k - 1/2 stored at k.

    Args:
      kernel_op: An object holding a library of kernel operations.
      deriv_lib: An instance of the derivatives library.
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      mu: Dynamic viscosity of the flow field.
      u: Velocity component in the x dimension, with updated boundary condition.
      v: Velocity component in the y dimension, with updated boundary condition.
      w: Velocity component in the z dimension, with updated boundary condition.
      rho: Density of the flow field.
      helper_variables: A dictionary that stores variables that provides
        additional information for computing the diffusion term, e.g. the
        potential temperature for the Monin-Obukhov similarity theory.

    Returns:
      The 9 component stress tensor for each grid point. Values in the halo with
      width 1 are invalid.
    """
    interp = functools.partial(
        interpolation.centered_node_to_face, kernel_op=kernel_op
    )
    use_3d_tf_tensor = isinstance(u, tf.Tensor)

    def grad_interp(
        f: FlowFieldVal,
        deriv_dim: Literal[0, 1, 2],
        interp_dim: Literal[0, 1, 2],
        helper_variables: FlowFieldMap,
    ) -> FlowFieldVal:
      """Computes derivative of `f` in `deriv_dim` on faces in `interp_dim`."""
      deriv_f = deriv_lib.deriv_centered(f, deriv_dim, helper_variables)
      # Interpolate the computed derivative onto faces in `interp_dim`.
      deriv_f_face = interp(deriv_f, interp_dim)
      return deriv_f_face

    # Compute sᵢⱼ, evaluated on faces in dim j.
    s00 = deriv_lib.deriv_node_to_face(u, 0, helper_variables)
    # Note that `du/dy` is on j faces, but `dv/dx` is on i faces if we compute
    # it directly based on `v` at nodes. Therefore, to get S01 on j faces,
    # `dv/dx` (computed with a central-difference scheme) needs to be
    # interpolated onto j faces. Similar interpolations are applied to compute
    # other components in the strain rate tensor S.
    s01 = tf.nest.map_structure(
        lambda du_dy, dv_dx: 0.5 * (du_dy + dv_dx),
        deriv_lib.deriv_node_to_face(u, 1, helper_variables),
        grad_interp(v, 0, 1, helper_variables),
    )
    s02 = tf.nest.map_structure(
        lambda du_dz, dw_dx: 0.5 * (du_dz + dw_dx),
        deriv_lib.deriv_node_to_face(u, 2, helper_variables),
        grad_interp(w, 0, 2, helper_variables),
    )
    s10 = tf.nest.map_structure(
        lambda dv_dx, du_dy: 0.5 * (dv_dx + du_dy),
        deriv_lib.deriv_node_to_face(v, 0, helper_variables),
        grad_interp(u, 1, 0, helper_variables),
    )
    s11 = deriv_lib.deriv_node_to_face(v, 1, helper_variables)
    s12 = tf.nest.map_structure(
        lambda dv_dz, dw_dy: 0.5 * (dv_dz + dw_dy),
        deriv_lib.deriv_node_to_face(v, 2, helper_variables),
        grad_interp(w, 1, 2, helper_variables),
    )
    s20 = tf.nest.map_structure(
        lambda dw_dx, du_dz: 0.5 * (dw_dx + du_dz),
        deriv_lib.deriv_node_to_face(w, 0, helper_variables),
        grad_interp(u, 2, 0, helper_variables),
    )
    s21 = tf.nest.map_structure(
        lambda dw_dy, dv_dz: 0.5 * (dw_dy + dv_dz),
        deriv_lib.deriv_node_to_face(w, 1, helper_variables),
        grad_interp(v, 2, 1, helper_variables),
    )
    s22 = deriv_lib.deriv_node_to_face(w, 2, helper_variables)

    # Compute divergence of velocity, evaluated on faces in dim 0, 1, 2
    # respectively.
    div_velocity_face_0 = tf.nest.map_structure(
        lambda du_dx, dv_dy, dw_dz: du_dx + dv_dy + dw_dz,
        s00,
        grad_interp(v, 1, 0, helper_variables),
        grad_interp(w, 2, 0, helper_variables),
    )

    div_velocity_face_1 = tf.nest.map_structure(
        lambda du_dx, dv_dy, dw_dz: du_dx + dv_dy + dw_dz,
        grad_interp(u, 0, 1, helper_variables),
        s11,
        grad_interp(w, 2, 1, helper_variables),
    )

    div_velocity_face_2 = tf.nest.map_structure(
        lambda du_dx, dv_dy, dw_dz: du_dx + dv_dy + dw_dz,
        grad_interp(u, 0, 2, helper_variables),
        grad_interp(v, 1, 2, helper_variables),
        s22,
    )

    # Compute 𝜏ᵢⱼ, evaluated on faces in dim j.
    tau00 = tf.nest.map_structure(
        lambda mu_i, s00_i, div_u: 2.0 * mu_i * (s00_i - 1.0 / 3.0 * div_u),
        interp(mu, 0),
        s00,
        div_velocity_face_0,
    )
    tau01 = tf.nest.map_structure(
        lambda mu_i, s01_i: 2.0 * mu_i * s01_i, interp(mu, 1), s01
    )
    tau02 = tf.nest.map_structure(
        lambda mu_i, s02_i: 2.0 * mu_i * s02_i, interp(mu, 2), s02
    )
    tau10 = tf.nest.map_structure(
        lambda mu_i, s10_i: 2.0 * mu_i * s10_i, interp(mu, 0), s10
    )
    tau11 = tf.nest.map_structure(
        lambda mu_i, s11_i, div_u: 2.0 * mu_i * (s11_i - 1.0 / 3.0 * div_u),
        interp(mu, 1),
        s11,
        div_velocity_face_1,
    )
    tau12 = tf.nest.map_structure(
        lambda mu_i, s12_i: 2.0 * mu_i * s12_i, interp(mu, 2), s12
    )
    tau20 = tf.nest.map_structure(
        lambda mu_i, s20_i: 2.0 * mu_i * s20_i, interp(mu, 0), s20
    )
    tau21 = tf.nest.map_structure(
        lambda mu_i, s21_i: 2.0 * mu_i * s21_i, interp(mu, 1), s21
    )
    tau22 = tf.nest.map_structure(
        lambda mu_i, s22_i, div_u: 2.0 * mu_i * (s22_i - 1.0 / 3.0 * div_u),
        interp(mu, 2),
        s22,
        div_velocity_face_2,
    )

    # Add the closure from Monin-Obukhov similarity theory if requested.
    if most is not None:
      if 'theta' not in helper_variables:
        raise ValueError('`theta` is missing for the MOS model.')

      helper_vars = {
          'u': u,
          'v': v,
          'w': w,
          'theta': helper_variables['theta'],
          'rho': rho,
      }

      # Get the surface shear stress.
      tau_s1, tau_s2, _ = most.surface_shear_stress_and_heat_flux_update_fn(
          helper_vars)

      # The sign of the shear stresses need to be reversed to be consistent with
      # the diffusion scheme.
      tau_s1 = tf.nest.map_structure(lambda t: -t, tau_s1)
      tau_s2 = tf.nest.map_structure(lambda t: -t, tau_s2)

      if most.vertical_dim == 2 and not use_3d_tf_tensor:
        tau_s1 = [tau_s1]
        tau_s2 = [tau_s2]

      # Replace the shear stress at the ground surface with the MOS closure.
      core_index = 0
      plane_index = params.halo_width
      if most.vertical_dim == 0:
        # `tau_s1` corresponds to the first shear stress component for the v
        # velocity, and `tau_s2` corresponds to the first shear stress component
        # for the w velocity.
        tau10 = common_ops.tensor_scatter_1d_update_global(
            replica_id,
            replicas,
            tau10,
            most.vertical_dim,
            core_index,
            plane_index,
            tau_s1,
        )
        tau20 = common_ops.tensor_scatter_1d_update_global(
            replica_id,
            replicas,
            tau20,
            most.vertical_dim,
            core_index,
            plane_index,
            tau_s2,
        )
      elif most.vertical_dim == 1:
        # `tau_s1` corresponds to the second shear stress component for the u
        # velocity, and `tau_s2` corresponds to the second shear stress
        # component for the w velocity.
        tau01 = common_ops.tensor_scatter_1d_update_global(
            replica_id,
            replicas,
            tau01,
            most.vertical_dim,
            core_index,
            plane_index,
            tau_s1,
        )
        tau21 = common_ops.tensor_scatter_1d_update_global(
            replica_id,
            replicas,
            tau21,
            most.vertical_dim,
            core_index,
            plane_index,
            tau_s2,
        )
      elif most.vertical_dim == 2:
        # `tau_s1` corresponds to the third shear stress component for the u
        # velocity, and `tau_s2` corresponds to the third shear stress component
        # for the v velocity.
        tau02 = common_ops.tensor_scatter_1d_update_global(
            replica_id,
            replicas,
            tau02,
            most.vertical_dim,
            core_index,
            plane_index,
            tau_s1,
        )
        tau12 = common_ops.tensor_scatter_1d_update_global(
            replica_id,
            replicas,
            tau12,
            most.vertical_dim,
            core_index,
            plane_index,
            tau_s2,
        )
      else:
        raise ValueError('Unsupport dimension: {}'.format(most.vertical_dim))

    return {
        'xx': tau00,
        'xy': tau01,
        'xz': tau02,
        'yx': tau10,
        'yy': tau11,
        'yz': tau12,
        'zx': tau20,
        'zy': tau21,
        'zz': tau22,
    }

  return shear_flux_fn


def bound_viscosity(
    nu: float | FlowFieldVal,
    additional_states: FlowFieldMap,
    params: parameters_lib.SwirlLMParameters,
) -> float | FlowFieldVal:
  """Sets an upper bound to `nu` following the stability constraint."""
  if params.diff_stab_crit is None:
    return nu

  for dim in (0, 1, 2):
    if params.use_stretched_grid[dim]:
      h = additional_states[stretched_grid_util.h_face_key(dim)]
    else:
      h = tf.constant(params.grid_spacings[dim], dtype=types.TF_DTYPE)
    nu_max = params.diff_stab_crit * (h**2 / params.dt)
    nu = tf.math.minimum(nu, nu_max)

  return nu


def subsidence_velocity_stevens(zz: FlowFieldVal) -> FlowFieldVal:
  """Computes the subsidence velocity following the Stevens' [1] formulation.

  Reference:
  1. Stevens, Bjorn, Chin-Hoh Moeng, Andrew S. Ackerman,
     Christopher S. Bretherton, Andreas Chlond, Stephan de Roode, James Edwards,
     et al. 2005. “Evaluation of Large-Eddy Simulations via Observations of
     Nocturnal Marine Stratocumulus.” Monthly Weather Review 133 (6): 1443–62.

  Args:
    zz: The coordinates in the vertical direction.

  Returns:
    The subsidence velocity.
  """
  return tf.nest.map_structure(lambda z: -_D * z, zz)


def subsidence_velocity_siebesma(zz: FlowFieldVal) -> FlowFieldVal:
  """Computes the subsidence velocity following the Siebesma's [1] formulation.

  Reference:
  1. Siebesma, A. Pier, A. Pier Siebesma, Christopher S. Bretherton,
  Andrew Brown, Andreas Chlond, Joan Cuxart, Peter G. Duynkerke, et al. 2003.
  “A Large Eddy Simulation Intercomparison Study of Shallow Cumulus
  Convection.” Journal of the Atmospheric Sciences.

  Args:
    zz: The coordinates in the vertical direction.

  Returns:
    The subsidence velocity.
  """
  w = tf.nest.map_structure(
      lambda z: tf.compat.v1.where(
          tf.less_equal(z, _Z_F1),
          _W_MAX * z / _Z_F1,
          _W_MAX * (1.0 - (z - _Z_F1) / (_Z_F5 - _Z_F1)),
      ),
      zz,
  )
  return tf.nest.map_structure(
      lambda z, w_i: tf.compat.v1.where(
          tf.less_equal(z, _Z_F5), w_i, tf.zeros_like(w_i)
      ),
      zz,
      w,
  )


def source_by_subsidence_velocity(
    deriv_lib: derivatives.Derivatives,
    rho: FlowFieldVal,
    height: FlowFieldVal,
    field: FlowFieldVal,
    vertical_dim: int,
    additional_states: FlowFieldMap,
) -> FlowFieldVal:
  """Computes the source term for `field` due to subsidence velocity.

  Args:
    deriv_lib: An instance of the derivatives library.
    rho: The density of the flow field.
    height: The coordinates in the direction vertical to the ground.
    field: The quantity to which the source term is computed.
    vertical_dim: The vertical dimension that is aligned with gravity.
    additional_states: A dictionary that holds all helper variables.

  Returns:
    The source term for `field` due to the subsidence velocity.
  """
  df_dh = deriv_lib.deriv_centered(field, vertical_dim, additional_states)
  w = subsidence_velocity_stevens(height)
  return tf.nest.map_structure(
      lambda rho_i, w_i, df_dh_i: -rho_i * w_i * df_dh_i, rho, w, df_dh
  )


def buoyancy_source(
    rho: FlowFieldVal,
    rho_0: FlowFieldVal,
    params: parameters_lib.SwirlLMParameters,
    dim: int,
    additional_states: FlowFieldMap,
) -> FlowFieldVal:
  """Computes the gravitational force of the momentum equation.

  Args:
    rho: The density of the flow field.
    rho_0: The reference density of the environment.
    params: The simulation parameter context. `thermodynamics.solver_mode` is
      used here.
    dim: The spatial dimension that this source corresponds to.
    additional_states: Mapping that contains the optional scale factors.

  Returns:
    The source term of the momentum equation due to buoyancy.
  """
  def drho_fn(rho_i, rho_0_i):
    if params.solver_mode == thermodynamics_pb2.Thermodynamics.ANELASTIC:
      return (rho_i - rho_0_i) * rho_0_i / rho_i
    else:
      return rho_i - rho_0_i

  # Computes the gravitational force.
  drho = filters.filter_op(
      params,
      tf.nest.map_structure(drho_fn, rho, rho_0),
      additional_states,
      order=2)
  return tf.nest.map_structure(
      lambda drho_i: drho_i * params.gravity_direction[dim] * constants.G, drho)
