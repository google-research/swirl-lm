# Copyright 2026 The swirl_lm Authors.
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

# Copyright 2021 Google LLC
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
"""Library of the convection scheme in the Navier-Stokes solver."""


from typing import Callable, Optional, TypeAlias

import jax
import jax.numpy as jnp
from swirl_lm.jax.boundary_condition import boundary_condition_utils
from swirl_lm.jax.numerics import derivatives
from swirl_lm.jax.numerics import interpolation
from swirl_lm.jax.numerics import numerics_pb2
from swirl_lm.jax.utility import common_ops
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types

ConvectionScheme: TypeAlias = numerics_pb2.ConvectionScheme
NumericalFlux: TypeAlias = numerics_pb2.NumericalFlux
ScalarField: TypeAlias = types.ScalarField
ScalarFieldMap: TypeAlias = types.ScalarFieldMap

_AXES = ('x', 'y', 'z')
_KERNEL_TYPE = 'conv'

# Mapping from axis to dim index and velocity/momentum keys.
_AXIS_TO_DIM = {'x': 0, 'y': 1, 'z': 2}
_VELOCITY_KEYS = ('u', 'v', 'w')
_MOMENTUM_KEYS = ('rho_u', 'rho_v', 'rho_w')

_WALL_TYPES = (
    boundary_condition_utils.BoundaryType.SLIP_WALL,
    boundary_condition_utils.BoundaryType.NON_SLIP_WALL,
    boundary_condition_utils.BoundaryType.SHEAR_WALL,
)


def _zero_wall_normal_flux(
    state_face: ScalarField,
    axis: str,
    varname: str | None,
    bc_types: tuple[
        boundary_condition_utils.BoundaryType,
        boundary_condition_utils.BoundaryType,
    ],
    halo_width: int,
    grid_params: grid_parametrization.GridParametrization,
) -> ScalarField:
  """Zeroes wall-normal face flux at wall boundaries.

  For wall boundaries, the face flux of the wall-normal velocity/momentum
  component must be zero at the wall face to enforce impermeability.

  Args:
    state_face: The face-interpolated flux field.
    axis: The axis normal to the face ('x', 'y', or 'z').
    varname: The name of the variable being convected (e.g. 'u', 'rho_u').
    bc_types: The boundary type on the low and high faces.
    halo_width: The number of halo layers.
    grid_params: The grid parametrization object.

  Returns:
    The face flux with wall-normal components zeroed at wall boundaries.
  """
  dim = _AXIS_TO_DIM[axis]
  wall_normal_keys = (_VELOCITY_KEYS[dim], _MOMENTUM_KEYS[dim])

  if varname is None or varname not in wall_normal_keys:
    return state_face

  axis_idx = grid_params.get_axis_index(axis)
  n_grid = state_face.shape[axis_idx]  # pyrefly: ignore[bad-index]
  # Number of devices along this axis, known statically from the config.
  num_cores = grid_params.get_axis_entry(
      grid_params.cx, grid_params.cy, grid_params.cz, axis
  )

  for face in range(2):
    if bc_types[face] not in _WALL_TYPES:
      continue

    # The wall face is at the interface between the halo and the first fluid
    # layer. The face value at index `halo_width` (for face=0) or
    # `n_grid - halo_width` (for face=1) must be zeroed.
    plane_idx = halo_width if face == 0 else n_grid - halo_width

    if num_cores == 1:
      new_plane = 0.0
    else:
      # `state_face` is this device's subdomain, so `plane_idx` locates the
      # physical boundary only on the device that owns it. On every other
      # device it is an interior seam between subdomains, and zeroing the
      # flux there would block transport across the seam -- which shows up
      # as a persistent velocity jump at each device boundary. This mirrors
      # the `is_first`/`is_last` gating that
      # `halo_exchange.inplace_halo_exchange` applies for the same reason.
      owns_boundary = jax.lax.axis_index(axis) == (
          0 if face == 0 else num_cores - 1
      )
      current_plane = jax.lax.dynamic_slice_in_dim(
          state_face, plane_idx, 1, axis=axis_idx  # pyrefly: ignore[bad-argument-type]
      )
      new_plane = jnp.where(owns_boundary, 0.0, current_plane)

    state_face = common_ops.array_scatter_1d_update(
        state_face, axis, plane_idx, new_plane, grid_params
    )

  return state_face


def first_order_upwinding(
    deriv_lib: derivatives.Derivatives,
    f: ScalarField,
    velocity_in_dim: ScalarField,
    axis: str,
    additional_states: ScalarFieldMap,
    grid_params: grid_parametrization.GridParametrization,
) -> ScalarField:
  """Computes the first order derivative of a variable in the convection term.

  An upwinding approach is adopted: the backward difference is used if
  `velocity_in_dim` >= 0, otherwise the forward difference is used.

  Args:
    deriv_lib: An instance of the derivatives library.
    f: The 3D scalar field.
    velocity_in_dim: 3D field holding the velocity in the direction in which the
      derivative is computed.
    axis: The axis along which the derivative is taken ('x', 'y', or 'z').
    additional_states: A dictionary of additional states that are needed to
      compute the derivative, holding stretched grid scale factors.
    grid_params: The grid parametrization object.

  Returns:
    The upwinding first-order derivative of `f`, i.e. `df / dx`.
  """
  kernel_op = get_kernel_fn.ApplyKernelConvOp(
      grid_params.kernel_size, grid_params, {'shift': ([0.0, 0.0, 1.0], 1)}
  )

  dfdx_backward_deriv = deriv_lib.deriv_node_to_face(f, axis, additional_states)
  dfdx_forward_deriv = kernel_op.apply_kernel_op(
      dfdx_backward_deriv, 'shift', axis
  )

  return jnp.where(
      velocity_in_dim < 0,
      dfdx_forward_deriv,
      dfdx_backward_deriv,
  )


def central2(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    f: ScalarField,
    grid_spacing: float,
    axis: str,
) -> ScalarField:
  """Computes the first order derivative using second order centered difference.

  Args:
    kernel_op: An object holding a library of kernel operations.
    f: A 3D `jax.Array` to which the operator is applied.
    grid_spacing: The mesh size in the direction where the derivative is
      computed.
    axis: The axis of the derivative ('x', 'y', or 'z').

  Returns:
    The first-order derivative of `f`, i.e. `df / dx`.
  """
  return kernel_op.apply_kernel_op(f, 'kD', axis) / (2.0 * grid_spacing)


def central4(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    f: ScalarField,
    grid_spacing: float,
    axis: str,
) -> ScalarField:
  """Computes the first order derivative using fourth order centered difference.

  Args:
    kernel_op: An object holding a library of kernel operations.
    f: A 3D `jax.Array` to which the operator is applied.
    grid_spacing: The mesh size in the direction where the derivative is
      computed.
    axis: The axis of the derivative ('x', 'y', or 'z').

  Returns:
    The first-order derivative of `f`, i.e. `df / dx`.
  """
  return kernel_op.apply_kernel_op(f, 'kD4', axis) / (12.0 * grid_spacing)


def face_interpolation(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    state: ScalarField,
    pressure: ScalarField,
    dx: float,
    dt: float,
    axis: str,
    src: Optional[ScalarField] = None,
    apply_correction: bool = True,
    bc_types: (
        tuple[
            boundary_condition_utils.BoundaryType,
            boundary_condition_utils.BoundaryType,
        ]
        | None
    ) = None,
    varname: str | None = None,
    halo_width: int | None = None,
    grid_params: grid_parametrization.GridParametrization | None = None,
) -> ScalarField:
  """Interpolates `state` from cells onto faces with the Rhie-Chow correction.

  Args:
    kernel_op: An object holding a library of kernel operations.
    state: A 3D `jax.Array` representing velocity or momentum.
    pressure: A 3D `jax.Array` representing pressure.
    dx: The grid spacing in the given axis.
    dt: The time step size.
    axis: The axis normal to the face ('x', 'y', or 'z').
    src: The source term for face-flux correction.
    apply_correction: Whether to apply the Rhie-Chow correction.
    bc_types: Optional boundary types for (low, high) faces. When provided along
      with varname/halo_width/grid_params, the wall-normal face flux is zeroed
      at wall boundaries.
    varname: Name of the variable being interpolated (e.g. 'u', 'rho_u').
    halo_width: Number of halo layers.
    grid_params: Grid parametrization object.

  Returns:
    `state` interpolated on the face normal to `axis`.

  Raises:
    ValueError if `axis` is not one of 'x', 'y', or 'z'.
  """
  if axis not in _AXES:
    raise ValueError(f'`axis` must be one of {_AXES}. {axis} is provided.')

  interp = kernel_op.apply_kernel_op(state, 'ks', axis)
  correction = kernel_op.apply_kernel_op(pressure, 'k3d1+', axis)

  state_face = 0.5 * interp

  if apply_correction:
    state_face = state_face - dt / 4.0 / dx * correction

    if src is not None:
      rc_weights = {'krc': ([-1.0, 1.0, 1.0, -1.0], 2)}
      kernel_op.add_kernel(rc_weights)
      src_correction = kernel_op.apply_kernel_op(src, 'krc', axis)
      state_face = state_face + (dt / 8) * src_correction

  # Zero the face flux at wall boundaries for wall-normal velocity/momentum.
  if (
      bc_types is not None
      and halo_width is not None
      and grid_params is not None
  ):
    state_face = _zero_wall_normal_flux(
        state_face, axis, varname, bc_types, halo_width, grid_params
    )

  return state_face


def flux_upwinding(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    state: ScalarField,
    rhou: ScalarField,
    pressure: ScalarField,
    interp_fn: Callable[[ScalarField], tuple[ScalarField, ScalarField]],
    dx: float,
    dt: float,
    axis: str,
    bc_types: (
        tuple[
            boundary_condition_utils.BoundaryType,
            boundary_condition_utils.BoundaryType,
        ]
        | None
    ) = None,
    varname: str | None = None,
    halo_width: int | None = None,
    grid_params: grid_parametrization.GridParametrization | None = None,
    src: Optional[ScalarField] = None,
    apply_correction: bool = True,
) -> ScalarField:
  """Computes the upwinding numerical flux.

  Args:
    kernel_op: An object holding a library of kernel operations.
    state: A 3D `jax.Array` representing velocity or momentum.
    rhou: The momentum in the direction of the flux.
    pressure: A 3D `jax.Array` representing pressure.
    interp_fn: A function that interpolates `state` from nodes to faces.
    dx: The grid spacing in the given axis.
    dt: The time step size.
    axis: The axis normal to the face ('x', 'y', or 'z').
    bc_types: The boundary types on the low and high faces along `axis`.
    varname: The name of the variable (e.g. 'rho_u').
    halo_width: The number of halo layers.
    grid_params: The grid parametrization object.
    src: The source term for face-flux correction.
    apply_correction: Whether to apply the Rhie-Chow correction.

  Returns:
    The upwinding numerical flux.
  """
  rhou_face = face_interpolation(
      kernel_op,
      rhou,
      pressure,
      dx,
      dt,
      axis,
      src,
      apply_correction,
      bc_types=bc_types,
      varname=varname,
      halo_width=halo_width,
      grid_params=grid_params,
  )

  state_pos, state_neg = interp_fn(state)

  return (
      0.5 * (rhou_face + jnp.abs(rhou_face)) * state_pos
      + 0.5 * (rhou_face - jnp.abs(rhou_face)) * state_neg
  )


def flux_roe(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    state: ScalarField,
    rhou: ScalarField,
    pressure: ScalarField,
    interp_fn: Callable[[ScalarField], tuple[ScalarField, ScalarField]],
    dx: float,
    dt: float,
    axis: str,
    bc_types: (
        tuple[
            boundary_condition_utils.BoundaryType,
            boundary_condition_utils.BoundaryType,
        ]
        | None
    ) = None,
    varname: str | None = None,
    halo_width: int | None = None,
    grid_params: grid_parametrization.GridParametrization | None = None,
    src: Optional[ScalarField] = None,
    apply_correction: bool = True,
) -> ScalarField:
  """Computes the Roe flux.

  Args:
    kernel_op: An object holding a library of kernel operations.
    state: A 3D `jax.Array` representing velocity or momentum.
    rhou: The momentum in the direction of the flux.
    pressure: A 3D `jax.Array` representing pressure.
    interp_fn: A function that interpolates `state` from nodes to faces.
    dx: The grid spacing in the given axis.
    dt: The time step size.
    axis: The axis normal to the face ('x', 'y', or 'z').
    bc_types: The boundary types on the low and high faces along `axis`.
    varname: The name of the variable (e.g. 'rho_u').
    halo_width: The number of halo layers.
    grid_params: The grid parametrization object.
    src: The source term for face-flux correction.
    apply_correction: Whether to apply the Rhie-Chow correction.

  Returns:
    The Roe numerical flux.

  Raises:
    NotImplementedError: If Rhie-Chow correction is enabled.
  """
  del pressure, dx, dt, src, bc_types, varname, halo_width, grid_params

  if apply_correction:
    raise NotImplementedError(
        'The Rhie-Chow correction is not implemented with the Roe flux.'
    )

  flux = rhou * state

  diff_state = kernel_op.apply_kernel_op(state, 'kd', axis)
  roe_speed = jnp.where(
      diff_state != 0,
      kernel_op.apply_kernel_op(flux, 'kd', axis) / diff_state,
      0.0,
  )

  f_neg, f_pos = interp_fn(flux)

  return jnp.where(roe_speed >= 0.0, f_neg, f_pos)


def flux_lf(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    state: ScalarField,
    rhou: ScalarField,
    pressure: ScalarField,
    interp_fn: Callable[[ScalarField], tuple[ScalarField, ScalarField]],
    dx: float,
    dt: float,
    axis: str,
    bc_types: (
        tuple[
            boundary_condition_utils.BoundaryType,
            boundary_condition_utils.BoundaryType,
        ]
        | None
    ) = None,
    varname: str | None = None,
    halo_width: int | None = None,
    grid_params: grid_parametrization.GridParametrization | None = None,
    src: Optional[ScalarField] = None,
    apply_correction: bool = True,
) -> ScalarField:
  """Computes the Lax-Friedrichs numerical flux.

  The Lax-Friedrichs flux splits the flux into left- and right-going waves
  using the global maximum wave speed (max |rhou|) as the dissipation
  coefficient.

  Args:
    kernel_op: An object holding a library of kernel operations.
    state: A 3D `jax.Array` representing velocity or momentum.
    rhou: The momentum in the direction of the flux.
    pressure: A 3D `jax.Array` representing pressure.
    interp_fn: A function that interpolates `state` from nodes to faces.
    dx: The grid spacing in the given axis.
    dt: The time step size.
    axis: The axis normal to the face ('x', 'y', or 'z').
    bc_types: The boundary types on the low and high faces along `axis`.
    varname: The name of the variable (e.g. 'rho_u').
    halo_width: The number of halo layers.
    grid_params: The grid parametrization object.
    src: The source term for face-flux correction.
    apply_correction: Whether to apply the Rhie-Chow correction.

  Returns:
    The Lax-Friedrichs numerical flux.

  Raises:
    NotImplementedError: If Rhie-Chow correction is enabled.
  """
  del kernel_op, pressure, dx, dt, axis, src
  del bc_types, varname, halo_width, grid_params

  if apply_correction:
    raise NotImplementedError(
        'The Rhie-Chow correction is not implemented with the Lax-Friedrichs '
        'flux.'
    )

  flux = rhou * state

  # Global maximum of |rhou| for numerical dissipation.
  rhou_max = jnp.max(jnp.abs(rhou))

  flux_m = 0.5 * (flux - rhou_max * state)
  flux_p = 0.5 * (flux + rhou_max * state)

  f_neg, _ = interp_fn(flux_p)
  _, f_pos = interp_fn(flux_m)

  return f_neg + f_pos


def face_interp_fn_first_order_upwind(
    axis: str,
    grid_params: grid_parametrization.GridParametrization,
) -> Callable[[ScalarField], tuple[ScalarField, ScalarField]]:
  """Generates a function that returns face values for the upwind scheme.

  Args:
    axis: The axis normal to the face ('x', 'y', or 'z').
    grid_params: The grid parametrization object.

  Returns:
    A function that returns upwind values of a variable on faces.
  """
  kernel_op = get_kernel_fn.ApplyKernelConvOp(
      grid_params.kernel_size,
      grid_params,
      {
          'shift': ([1.0, 0.0, 0.0], 1),
      },
  )

  def first_order_upwind_fn(
      state: ScalarField,
  ) -> tuple[ScalarField, ScalarField]:
    """Computes the face flux of `state` normal to `axis` with upwind scheme."""
    s_pos = kernel_op.apply_kernel_op(state, 'shift', axis)
    s_neg = state
    return s_pos, s_neg

  return first_order_upwind_fn


def face_interp_fn_quick(
    axis: str,
    grid_params: grid_parametrization.GridParametrization,
) -> Callable[[ScalarField], tuple[ScalarField, ScalarField]]:
  """Generates a function that performs interpolation with the QUICK scheme.

  Args:
    axis: The axis normal to the face ('x', 'y', or 'z').
    grid_params: The grid parametrization object.

  Returns:
    A function that interpolates values of a variable onto faces.
  """
  kernel_op = get_kernel_fn.ApplyKernelConvOp(
      grid_params.kernel_size, grid_params, {'kf2-': ([-0.125, 0.75, 0.375], 2)}
  )

  def quick_fn(state: ScalarField) -> tuple[ScalarField, ScalarField]:
    """Computes the face flux of `state` normal to `axis` with QUICK scheme."""
    s_pos = kernel_op.apply_kernel_op(state, 'kf2-', axis)
    s_neg = kernel_op.apply_kernel_op(state, 'kf2+', axis)
    return s_pos, s_neg

  return quick_fn


def face_interp_fn_flux_limiter(
    axis: str,
    interp_scheme: ConvectionScheme,
    grid_params: grid_parametrization.GridParametrization,
) -> Callable[[ScalarField], tuple[ScalarField, ScalarField]]:
  """Generates a function that performs interpolation using a limiter scheme.

  Args:
    axis: The axis normal to the face ('x', 'y', or 'z').
    interp_scheme: The scheme for interpolation.
    grid_params: The grid parametrization object.

  Returns:
    A function that interpolates values of a variable onto faces.
  """
  if interp_scheme == numerics_pb2.CONVECTION_SCHEME_FLUX_LIMITER_VAN_LEER:
    limiter_type = interpolation.FluxLimiterType.VAN_LEER
  elif interp_scheme == numerics_pb2.CONVECTION_SCHEME_FLUX_LIMITER_MUSCL:
    limiter_type = interpolation.FluxLimiterType.MUSCL
  else:
    raise NotImplementedError(
        f'{interp_scheme} is not supported. Available options are:'
        f' {numerics_pb2.CONVECTION_SCHEME_FLUX_LIMITER_VAN_LEER},'
        f' {numerics_pb2.CONVECTION_SCHEME_FLUX_LIMITER_MUSCL}.'
    )

  def interp_fn(state: ScalarField) -> tuple[ScalarField, ScalarField]:
    """Computes the face value of `state` with limiter scheme."""
    return interpolation.flux_limiter(
        state, axis, limiter_type, grid_params, _KERNEL_TYPE
    )

  return interp_fn


def face_interp_fn_weno(
    axis: str,
    grid_params: grid_parametrization.GridParametrization,
    order: int = 3,
) -> Callable[[ScalarField], tuple[ScalarField, ScalarField]]:
  """Generates a function that performs interpolation with the WENO scheme.

  Args:
    axis: The axis normal to the face ('x', 'y', or 'z').
    grid_params: The grid parametrization object.
    order: The order/stencil width of the interpolation.

  Returns:
    A function that interpolates values of a variable onto faces.
  """

  def weno_fn(f: ScalarField) -> tuple[ScalarField, ScalarField]:
    """Computes the face value of `f` normal to `axis` with WENO scheme."""
    return interpolation.weno(f, axis, order, grid_params, _KERNEL_TYPE)

  return weno_fn


def convection_from_flux(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    deriv_lib: derivatives.Derivatives,
    interp_scheme: ConvectionScheme,
    flux_scheme: NumericalFlux,
    state: ScalarField,
    rhou: ScalarField,
    pressure: ScalarField,
    dx: float,
    dt: float,
    axis: str,
    helper_variables: ScalarFieldMap,
    grid_params: grid_parametrization.GridParametrization,
    bc_types: (
        tuple[
            boundary_condition_utils.BoundaryType,
            boundary_condition_utils.BoundaryType,
        ]
        | None
    ) = None,
    varname: str | None = None,
    halo_width: int | None = None,
    src: Optional[ScalarField] = None,
    apply_correction: bool = True,
) -> ScalarField:
  """Computes the convection term for conservative variables.

  Args:
    kernel_op: An object holding a library of kernel operations.
    deriv_lib: An instance of the derivatives library.
    interp_scheme: The scheme for interpolation.
    flux_scheme: The scheme for computing the numerical flux.
    state: A 3D `jax.Array` representing the variable for which the convection
      term is computed.
    rhou: A 3D `jax.Array` representing momentum.
    pressure: A 3D `jax.Array` representing pressure.
    dx: The grid spacing.
    dt: The time step size.
    axis: The axis normal to the face ('x', 'y', or 'z').
    helper_variables: Dictionary that holds all helper variables.
    grid_params: The grid parametrization object.
    bc_types: The boundary types on the low and high faces along `axis`.
    varname: The name of the variable (e.g. 'rho_u').
    halo_width: The number of halo layers.
    src: The source term for the face-flux correction.
    apply_correction: Whether to apply the Rhie-Chow correction.

  Returns:
    The convection term of `f`.
  """
  # Re-creating kernel_op for performance (matching TF implementation pattern).
  del kernel_op
  kernel_op = get_kernel_fn.ApplyKernelConvOp(
      grid_params.kernel_size, grid_params
  )
  deriv_lib = deriv_lib.create_copy_with_custom_kernel_op(kernel_op)

  if flux_scheme == numerics_pb2.NUMERICAL_FLUX_UPWINDING:
    flux_fn = flux_upwinding
  elif flux_scheme == numerics_pb2.NUMERICAL_FLUX_ROE:
    flux_fn = flux_roe
  elif flux_scheme == numerics_pb2.NUMERICAL_FLUX_LF:
    flux_fn = flux_lf
  else:
    raise NotImplementedError(
        'Unknown numerical flux'
        f' {NumericalFlux.Name(flux_scheme)}. Available options'
        ' are:'
        f' {NumericalFlux.Name(numerics_pb2.NUMERICAL_FLUX_UPWINDING)},'
        f' {NumericalFlux.Name(numerics_pb2.NUMERICAL_FLUX_ROE)},'
        f' {NumericalFlux.Name(numerics_pb2.NUMERICAL_FLUX_LF)}.'
    )

  if interp_scheme == numerics_pb2.CONVECTION_SCHEME_UPWIND_1:
    interp_fn = face_interp_fn_first_order_upwind(axis, grid_params)
  elif interp_scheme == numerics_pb2.CONVECTION_SCHEME_QUICK:
    interp_fn = face_interp_fn_quick(axis, grid_params)
  elif interp_scheme in (
      numerics_pb2.CONVECTION_SCHEME_FLUX_LIMITER_VAN_LEER,
      numerics_pb2.CONVECTION_SCHEME_FLUX_LIMITER_MUSCL,
  ):
    interp_fn = face_interp_fn_flux_limiter(axis, interp_scheme, grid_params)
  elif interp_scheme == numerics_pb2.CONVECTION_SCHEME_WENO_3:
    interp_fn = face_interp_fn_weno(axis, grid_params, order=2)
  elif interp_scheme == numerics_pb2.CONVECTION_SCHEME_WENO_5:
    interp_fn = face_interp_fn_weno(axis, grid_params, order=3)
  else:
    raise ValueError(
        'Unknown convection scheme'
        f' {ConvectionScheme.Name(interp_scheme)}. Available'
        ' options are:'
        f' {ConvectionScheme.Name(numerics_pb2.CONVECTION_SCHEME_QUICK)},'
        f' {ConvectionScheme.Name(numerics_pb2.CONVECTION_SCHEME_WENO_3)},'
        f' {ConvectionScheme.Name(numerics_pb2.CONVECTION_SCHEME_WENO_5)}.'
    )

  flux = flux_fn(
      kernel_op,
      state,
      rhou,
      pressure,
      interp_fn,
      dx,
      dt,
      axis,
      bc_types=bc_types,
      varname=varname,
      halo_width=halo_width,
      grid_params=grid_params,
      src=src,
      apply_correction=apply_correction,
  )

  # Compute convection term, e.g., d/dx (flux_x) etc., evaluated on nodes
  return deriv_lib.deriv_face_to_node(flux, axis, helper_variables)


def convection_central_2(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    state: ScalarField,
    rhou: ScalarField,
    pressure: ScalarField,
    dx: float,
    dt: float,
    axis: str,
) -> ScalarField:
  """Compute the convection term with the second order central scheme.

  Args:
    kernel_op: An object holding a library of kernel operations.
    state: A 3D `jax.Array` for which the convection term is computed.
    rhou: A 3D `jax.Array` representing momentum.
    pressure: A 3D `jax.Array` representing pressure.
    dx: The grid spacing.
    dt: The time step size.
    axis: The axis normal to the face ('x', 'y', or 'z').

  Returns:
    The convection term of `f`.
  """
  del pressure, dt

  flux = rhou * state
  return central2(kernel_op, flux, dx, axis)


def convection_term(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    deriv_lib: derivatives.Derivatives,
    state: ScalarField,
    rhou: ScalarField,
    pressure: ScalarField,
    dx: float,
    dt: float,
    axis: str,
    helper_variables: ScalarFieldMap,
    grid_params: grid_parametrization.GridParametrization,
    scheme: ConvectionScheme = ConvectionScheme.CONVECTION_SCHEME_QUICK,
    flux_scheme: NumericalFlux = NumericalFlux.NUMERICAL_FLUX_UPWINDING,
    bc_types: (
        tuple[
            boundary_condition_utils.BoundaryType,
            boundary_condition_utils.BoundaryType,
        ]
        | None
    ) = None,
    varname: str | None = None,
    halo_width: int | None = None,
    src: Optional[ScalarField] = None,
    apply_correction: bool = True,
) -> ScalarField:
  """Computes the convection term df/dx with selected scheme.

  Args:
    kernel_op: An object holding a library of kernel operations.
    deriv_lib: An instance of the derivatives library.
    state: A 3D `jax.Array` for which the convection term is computed.
    rhou: A 3D `jax.Array` representing momentum.
    pressure: A 3D `jax.Array` representing pressure.
    dx: The grid spacing.
    dt: The time step size.
    axis: The axis normal to the face ('x', 'y', or 'z').
    helper_variables: Dictionary that holds all helper variables.
    grid_params: The grid parametrization object.
    scheme: The numerical scheme to use for the convection term.
    flux_scheme: The scheme for computing the numerical flux.
    bc_types: The boundary types on the low and high faces along `axis`.
    varname: The name of the variable (e.g. 'rho_u').
    halo_width: The number of halo layers.
    src: The source term for the face-flux correction.
    apply_correction: Whether to apply the Rhie-Chow correction.

  Returns:
    The convection term of `f`.
  """
  if scheme in (
      ConvectionScheme.CONVECTION_SCHEME_UPWIND_1,
      ConvectionScheme.CONVECTION_SCHEME_QUICK,
      ConvectionScheme.CONVECTION_SCHEME_FLUX_LIMITER_VAN_LEER,
      ConvectionScheme.CONVECTION_SCHEME_FLUX_LIMITER_MUSCL,
      ConvectionScheme.CONVECTION_SCHEME_WENO_3,
      ConvectionScheme.CONVECTION_SCHEME_WENO_5,
  ):
    return convection_from_flux(
        kernel_op,
        deriv_lib,
        scheme,
        flux_scheme,
        state,
        rhou,
        pressure,
        dx,
        dt,
        axis,
        helper_variables,
        grid_params,
        bc_types=bc_types,
        varname=varname,
        halo_width=halo_width,
        src=src,
        apply_correction=apply_correction,
    )
  elif scheme == ConvectionScheme.CONVECTION_SCHEME_CENTRAL_2:
    return convection_central_2(kernel_op, state, rhou, pressure, dx, dt, axis)
  else:
    raise NotImplementedError(
        f'{numerics_pb2.ConvectionScheme.Name(scheme)} is '
        'not implemented. Available options are: '
        + ', '.join(
            ConvectionScheme.Name(s)
            for s in [
                ConvectionScheme.CONVECTION_SCHEME_UPWIND_1,
                ConvectionScheme.CONVECTION_SCHEME_QUICK,
                ConvectionScheme.CONVECTION_SCHEME_FLUX_LIMITER_VAN_LEER,
                ConvectionScheme.CONVECTION_SCHEME_FLUX_LIMITER_MUSCL,
                ConvectionScheme.CONVECTION_SCHEME_WENO_3,
                ConvectionScheme.CONVECTION_SCHEME_WENO_5,
                ConvectionScheme.CONVECTION_SCHEME_CENTRAL_2,
            ]
        )
    )
