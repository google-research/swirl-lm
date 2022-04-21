"""Utility functions that are commonly used in different equations."""

from typing import Callable, Dict, List, Mapping, Optional, Sequence, Text, Union
import numpy as np
from swirl_lm.boundary_condition import monin_obukhov_similarity_theory
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
import tensorflow as tf
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_numerics

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

FlowFieldVar = Union[List[tf.Tensor], tf.Tensor]
FlowFieldMap = Mapping[Text, FlowFieldVar]


def shear_stress(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    mu: Sequence[tf.Tensor],
    dx: float,
    dy: float,
    dz: float,
    u: Sequence[tf.Tensor],
    v: Sequence[tf.Tensor],
    w: Sequence[tf.Tensor],
    shear_bc_update_fn: Optional[Dict[Text, Callable[[Sequence[tf.Tensor]],
                                                     List[tf.Tensor]]]] = None,
) -> FlowFieldMap:
  """Computes the viscous shear stress.

  The shear stress is computed as:
    τᵢⱼ = μ [∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ], i ≠ j
    τᵢⱼ = 2 μ [∂uᵢ/∂xⱼ - 1/3 ∂uₖ/∂xₖ δᵢⱼ], i = j
  Note that u, v, w are 3D tensors that are represented in the form of a list of
  2D x-y slices.

  Args:
    kernel_op: An object holding a library of kernel operations.
    mu: Dynamic viscosity of the flow field.
    dx: Grid spacing in the x dimension.
    dy: Grid spacing in the y dimension.
    dz: Grid spacing in the z dimension.
    u: Velocity component in the x dimension, with updated boundary condition.
    v: Velocity component in the y dimension, with updated boundary condition.
    w: Velocity component in the z dimension, with updated boundary condition.
    shear_bc_update_fn: A dictionary of halo_exchange functions for the shear
      stress tensor.

  Returns:
    The 9 component stress stress tensor for each grid point. Values in the halo
    with width 1 is invalid.
  """
  # TODO(b/150696474): Fold this into a utility library and call it when
  # gradient is computed.
  du_dx = incompressible_structured_mesh_numerics.grad(kernel_op, [u, v, w],
                                                       [dx, dy, dz])

  du_11 = du_dx[0][0]
  du_12 = du_dx[0][1]
  du_13 = du_dx[0][2]
  du_21 = du_dx[1][0]
  du_22 = du_dx[1][1]
  du_23 = du_dx[1][2]
  du_31 = du_dx[2][0]
  du_32 = du_dx[2][1]
  du_33 = du_dx[2][2]

  s11 = du_11
  s12 = [0.5 * (du_12_i + du_21_i) for du_12_i, du_21_i in zip(du_12, du_21)]
  s13 = [0.5 * (du_13_i + du_31_i) for du_13_i, du_31_i in zip(du_13, du_31)]
  s21 = s12
  s22 = du_22
  s23 = [0.5 * (du_23_i + du_32_i) for du_23_i, du_32_i in zip(du_23, du_32)]
  s31 = s13
  s32 = s23
  s33 = du_33

  du_kk = [
      du_11_i + du_22_i + du_33_i
      for du_11_i, du_22_i, du_33_i in zip(du_11, du_22, du_33)
  ]

  tau11 = [
      2.0 * mu_i * (s11_i - 1.0 / 3.0 * du_kk_i)
      for mu_i, s11_i, du_kk_i in zip(mu, s11, du_kk)
  ]
  tau12 = [2.0 * mu_i * s12_i for mu_i, s12_i in zip(mu, s12)]
  tau13 = [2.0 * mu_i * s13_i for mu_i, s13_i in zip(mu, s13)]
  tau21 = [2.0 * mu_i * s21_i for mu_i, s21_i in zip(mu, s21)]
  tau22 = [
      2.0 * mu_i * (s22_i - 1.0 / 3.0 * du_kk_i)
      for mu_i, s22_i, du_kk_i in zip(mu, s22, du_kk)
  ]
  tau23 = [2.0 * mu_i * s23_i for mu_i, s23_i in zip(mu, s23)]
  tau31 = [2.0 * mu_i * s31_i for mu_i, s31_i in zip(mu, s31)]
  tau32 = [2.0 * mu_i * s32_i for mu_i, s32_i in zip(mu, s32)]
  tau33 = [
      2.0 * mu_i * (s33_i - 1.0 / 3.0 * du_kk_i)
      for mu_i, s33_i, du_kk_i in zip(mu, s33, du_kk)
  ]

  tau_ij = {
      'xx': tau11,
      'xy': tau12,
      'xz': tau13,
      'yx': tau21,
      'yy': tau22,
      'yz': tau23,
      'zx': tau31,
      'zy': tau32,
      'zz': tau33,
  }

  if shear_bc_update_fn:
    for key, fn in shear_bc_update_fn.items():
      tau_ij.update({key: fn(tau_ij[key])})

  return tau_ij


def shear_flux(params: incompressible_structured_mesh_config
               .IncompressibleNavierStokesParameters) -> ...:
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
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      mu: Sequence[tf.Tensor],
      dx: float,
      dy: float,
      dz: float,
      u: Sequence[tf.Tensor],
      v: Sequence[tf.Tensor],
      w: Sequence[tf.Tensor],
      helper_variables: Optional[Dict[Text, Sequence[tf.Tensor]]] = None,
  ) -> FlowFieldMap:
    """Computes the viscous shear stress on the cell faces.

    The shear stress is computed as:
      τᵢⱼ = μ [∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ], i ≠ j
      τᵢⱼ = 2 μ [∂uᵢ/∂xⱼ - 1/3 ∂uₖ/∂xₖ δᵢⱼ], i = j
    Note that u, v, w are 3D tensors that are represented in the form of a list
    of 2D x-y slices.

    Locations of the fluxes:
      tau11/tau_xx: x face, i - 1/2 stored at i;
      tau12/tau_xy: y face, j - 1/2 stored at j;
      tau13/tau_xz: z face, k - 1/2 stored at k;
      tau21/tau_yx: x face, i - 1/2 stored at i;
      tau22/tau_yy: y face, j - 1/2 stored at j;
      tau23/tau_yz: z face, k - 1/2 stored at k;
      tau31/tau_zx: x face, i - 1/2 stored at i;
      tau32/tau_zy: y face, j - 1/2 stored at j;
      tau33/tau_zz: z face, k - 1/2 stored at k.

    Args:
      kernel_op: An object holding a library of kernel operations.
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      mu: Dynamic viscosity of the flow field.
      dx: Grid spacing in the x dimension.
      dy: Grid spacing in the y dimension.
      dz: Grid spacing in the z dimension.
      u: Velocity component in the x dimension, with updated boundary condition.
      v: Velocity component in the y dimension, with updated boundary condition.
      w: Velocity component in the z dimension, with updated boundary condition.
      helper_variables: A dictionarry that stores variables that provides
        additional information for computing the diffusion term, e.g. the
        potential temperature for the Monin-Obukhov similarity theory.

    Returns:
      The 9 component stress tensor for each grid point. Values in the halo with
      width 1 are invalid.
    """

    def interp(f: Sequence[tf.Tensor], dim: int) -> List[tf.Tensor]:
      """Interpolates `value` in `dim` onto faces (i - 1/2 stored at i)."""
      if dim == 0:
        df = kernel_op.apply_kernel_op_x(f, 'ksx')
      elif dim == 1:
        df = kernel_op.apply_kernel_op_y(f, 'ksy')
      else:  # dim == 2
        df = kernel_op.apply_kernel_op_z(f, 'ksz', 'kszsh')

      return [df_i / 2.0 for df_i in df]

    def grad_n(f: Sequence[tf.Tensor], dim: int, h: float) -> List[tf.Tensor]:
      """Computes gradient of `value` in `dim` on nodes."""
      if dim == 0:
        df = kernel_op.apply_kernel_op_x(f, 'kDx')
      elif dim == 1:
        df = kernel_op.apply_kernel_op_y(f, 'kDy')
      else:  # dim == 2
        df = kernel_op.apply_kernel_op_z(f, 'kDz', 'kDzsh')

      return [df_i / (2.0 * h) for df_i in df]

    def grad_f(f: Sequence[tf.Tensor], dim: int, h: float) -> List[tf.Tensor]:
      """Computes gradient of `value` in `dim` on faces."""
      if dim == 0:
        df = kernel_op.apply_kernel_op_x(f, 'kdx')
      elif dim == 1:
        df = kernel_op.apply_kernel_op_y(f, 'kdy')
      else:  # dim == 2
        df = kernel_op.apply_kernel_op_z(f, 'kdz', 'kdzsh')

      return [df_i / h for df_i in df]

    def grad_interp(f: Sequence[tf.Tensor], grad_dim: int, interp_dim: int,
                    h: float) -> List[tf.Tensor]:
      """Computes gradient of `value` in `grad_dim` on faces in `interp_dim`."""
      return interp(grad_n(f, grad_dim, h), interp_dim)

    s11 = grad_f(u, 0, dx)
    # Note that `du/dy` is on j faces, but `dv/dx` is on i faces if we compute
    # it directly based on `v` at nodes. Therefore, to get S12 on j faces,
    # `dv/dx` (computed with a central-difference scheme) needs to be
    # interpolated onto j faces. Similar interpolations are applied to compute
    # other components in the strain rate tensor S.
    s12 = [
        0.5 * (du_dy + dv_dx)
        for du_dy, dv_dx in zip(grad_f(u, 1, dy), grad_interp(v, 0, 1, dx))
    ]
    s13 = [
        0.5 * (du_dz + dw_dx)
        for du_dz, dw_dx in zip(grad_f(u, 2, dz), grad_interp(w, 0, 2, dx))
    ]
    s21 = [
        0.5 * (dv_dx + du_dy)
        for dv_dx, du_dy in zip(grad_f(v, 0, dx), grad_interp(u, 1, 0, dy))
    ]
    s22 = grad_f(v, 1, dy)
    s23 = [
        0.5 * (dv_dz + dw_dy)
        for dv_dz, dw_dy in zip(grad_f(v, 2, dz), grad_interp(w, 1, 2, dy))
    ]
    s31 = [
        0.5 * (dw_dx + du_dz)
        for dw_dx, du_dz in zip(grad_f(w, 0, dx), grad_interp(u, 2, 0, dz))
    ]
    s32 = [
        0.5 * (dw_dy + dv_dz)
        for dw_dy, dv_dz in zip(grad_f(w, 1, dy), grad_interp(v, 2, 1, dz))
    ]
    s33 = grad_f(w, 2, dz)

    du_kk_x = [
        du_dx + dv_dy + dw_dz for du_dx, dv_dy, dw_dz in zip(
            s11, grad_interp(v, 1, 0, dy), grad_interp(w, 2, 0, dz))
    ]

    du_kk_y = [
        du_dx + dv_dy + dw_dz for du_dx, dv_dy, dw_dz in zip(
            grad_interp(u, 0, 1, dx), s22, grad_interp(w, 2, 1, dz))
    ]

    du_kk_z = [
        du_dx + dv_dy + dw_dz for du_dx, dv_dy, dw_dz in zip(
            grad_interp(u, 0, 2, dx), grad_interp(v, 1, 2, dy), s33)
    ]

    tau11 = [
        2.0 * mu_i * (s11_i - 1.0 / 3.0 * du_kk_i)
        for mu_i, s11_i, du_kk_i in zip(interp(mu, 0), s11, du_kk_x)
    ]
    tau12 = [2.0 * mu_i * s12_i for mu_i, s12_i in zip(interp(mu, 1), s12)]
    tau13 = [2.0 * mu_i * s13_i for mu_i, s13_i in zip(interp(mu, 2), s13)]
    tau21 = [2.0 * mu_i * s21_i for mu_i, s21_i in zip(interp(mu, 0), s21)]
    tau22 = [
        2.0 * mu_i * (s22_i - 1.0 / 3.0 * du_kk_i)
        for mu_i, s22_i, du_kk_i in zip(interp(mu, 1), s22, du_kk_y)
    ]
    tau23 = [2.0 * mu_i * s23_i for mu_i, s23_i in zip(interp(mu, 2), s23)]
    tau31 = [2.0 * mu_i * s31_i for mu_i, s31_i in zip(interp(mu, 0), s31)]
    tau32 = [2.0 * mu_i * s32_i for mu_i, s32_i in zip(interp(mu, 1), s32)]
    tau33 = [
        2.0 * mu_i * (s33_i - 1.0 / 3.0 * du_kk_i)
        for mu_i, s33_i, du_kk_i in zip(interp(mu, 2), s33, du_kk_z)
    ]

    # Add the closure from Monin-Obukhov similarity theory if requested.
    if most is not None:
      if 'theta' not in helper_variables:
        raise ValueError('`theta` is missing for the MOS model.')

      helper_vars = {'u': u, 'v': v, 'w': w, 'theta': helper_variables['theta']}

      # Get the surface shear stress.
      tau_s1, tau_s2, _ = most.surface_shear_stress_and_heat_flux_update_fn(
          helper_vars)

      # The sign of the shear stresses need to be reversed to be consistent with
      # the diffusion scheme.
      tau_s1 = tf.nest.map_structure(lambda t: -t, tau_s1)
      tau_s2 = tf.nest.map_structure(lambda t: -t, tau_s2)

      if most.vertical_dim == 2:
        tau_s1 = [tau_s1]
        tau_s2 = [tau_s2]

      # Replace the shear stress at the ground surface with the MOS closure.
      core_index = 0
      plane_index = params.halo_width
      if most.vertical_dim == 0:
        # `tau_s1` corresponds to the first shear stress component for the v
        # velocity, and `tau_s2` corresponds to the first shear stress component
        # for the w velocity.
        tau21 = common_ops.tensor_scatter_1d_update_global(
            replica_id, replicas, tau21, most.vertical_dim, core_index,
            plane_index, tau_s1)
        tau31 = common_ops.tensor_scatter_1d_update_global(
            replica_id, replicas, tau31, most.vertical_dim, core_index,
            plane_index, tau_s2)
      elif most.vertical_dim == 1:
        # `tau_s1` corresponds to the second shear stress component for the u
        # velocity, and `tau_s2` corresponds to the second shear stress
        # component for the w velocity.
        tau12 = common_ops.tensor_scatter_1d_update_global(
            replica_id, replicas, tau12, most.vertical_dim, core_index,
            plane_index, tau_s1)
        tau32 = common_ops.tensor_scatter_1d_update_global(
            replica_id, replicas, tau32, most.vertical_dim, core_index,
            plane_index, tau_s2)
      else:  # most.vertical_dim == 2
        # `tau_s1` corresponds to the third shear stress component for the u
        # velocity, and `tau_s2` corresponds to the third shear stress component
        # for the v velocity.
        tau13 = common_ops.tensor_scatter_1d_update_global(
            replica_id, replicas, tau13, most.vertical_dim, core_index,
            plane_index, tau_s1)
        tau23 = common_ops.tensor_scatter_1d_update_global(
            replica_id, replicas, tau23, most.vertical_dim, core_index,
            plane_index, tau_s2)

    return {
        'xx': tau11,
        'xy': tau12,
        'xz': tau13,
        'yx': tau21,
        'yy': tau22,
        'yz': tau23,
        'zx': tau31,
        'zy': tau32,
        'zz': tau33,
    }

  return shear_flux_fn


def subsidence_velocity_stevens(zz: Sequence[tf.Tensor]) -> List[tf.Tensor]:
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
  return [-_D * z for z in zz]


def subsidence_velocity_siebesma(zz: Sequence[tf.Tensor]) -> List[tf.Tensor]:
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
  w = [
      tf.compat.v1.where(
          tf.less_equal(z, _Z_F1), _W_MAX * z / _Z_F1,
          _W_MAX * (1.0 - (z - _Z_F1) / (_Z_F5 - _Z_F1))) for z in zz
  ]
  return [
      tf.compat.v1.where(tf.less_equal(z, _Z_F5), w_i, tf.zeros_like(w_i))
      for z, w_i in zip(zz, w)
  ]


def source_by_subsidence_velocity(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    rho: Sequence[tf.Tensor],
    height: Sequence[tf.Tensor],
    h: float,
    field: Sequence[tf.Tensor],
    vertical_dim: int,
) -> List[tf.Tensor]:
  """Computes the source term for `field` due to subsidence velocity.

  Args:
    kernel_op: A library of finite difference operators.
    rho: The density of the flow field.
    height: The coordinates in the direction vertical to the ground.
    h: The grid spacing in the vertical direction discretization.
    field: The quantity to which the source term is computed.
    vertical_dim: The vertical dimension that is aligned with gravity.

  Returns:
    The source term for `field` due to the subsidence velocity.
  """
  if vertical_dim == 0:
    df = kernel_op.apply_kernel_op_x(field, 'kDx')
  elif vertical_dim == 1:
    df = kernel_op.apply_kernel_op_y(field, 'kDy')
  else:  # vertical_dim == 2
    df = kernel_op.apply_kernel_op_z(field, 'kDz', 'kDzsh')

  df_dh = [df_i / (2.0 * h) for df_i in df]
  w = subsidence_velocity_stevens(height)
  return [
      -rho_i * w_i * df_dh_i
      for rho_i, w_i, df_dh_i in zip(rho, w, df_dh)
  ]
