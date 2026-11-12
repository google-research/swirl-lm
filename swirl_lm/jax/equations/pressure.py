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
"""A library for solving the pressure equation.

This is the JAX port of `swirl_lm.equations.pressure`. It solves the pressure
Poisson equation using the predictor-corrector approach.

For TGV and basic incompressible flows:
  - All periodic BCs -> homogeneous Neumann dp BCs (standard for periodic).
  - Constant density -> drho_dt = 0.
  - Uses JAX Poisson solvers (CG, Fast Diagonalization).

Geophysical extensions (thermodynamics, boundary models, monitoring) will be
added as separate methods when those physics modules are ported.
"""


from collections.abc import Callable
import functools

from google.protobuf import text_format
import jax
import jax.numpy as jnp
from swirl_lm.equations import pressure_pb2
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.base import physical_variable_keys_manager
from swirl_lm.jax.boundary_condition import boundary_condition_utils
from swirl_lm.jax.communication import halo_exchange
from swirl_lm.jax.communication import halo_exchange_utils
from swirl_lm.jax.equations import utils as eq_utils
from swirl_lm.jax.linalg import base_poisson_solver
from swirl_lm.jax.linalg import poisson_solver
from swirl_lm.jax.linalg import poisson_solver_pb2
from swirl_lm.jax.numerics import filters
from swirl_lm.jax.numerics import interpolation
from swirl_lm.jax.physics.thermodynamics import manager as thermodynamics_manager
from swirl_lm.jax.utility import common_ops
from swirl_lm.jax.utility import types
from swirl_lm.physics.thermodynamics import thermodynamics_pb2

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

# All axes for halo exchange.
_ALL_AXES = ('x', 'y', 'z')

# Default Poisson solver config (CG, used when no config is provided).
_DEFAULT_SOLVER_PARAMS = (
    R'conjugate_gradient {  '
    R'  max_iterations: 100  '
    R'  halo_width: 2  '
    R'}'
)

# Number of drho filter applications.
_DEFAULT_NUM_D_RHO_FILTER = 3

# The immersed boundary's fluid indicator, one in fluid and zero in solid.
# @TODO: Looked up by name rather than imported since the method lives in a
# module only some branches carry (not yet on main).
# `mass_source` is handled the same way below.
_INTERIOR_MASK_KEY = 'ib_interior_mask'

# Neumann BC spec for all 3 dimensions (used for drho filtering).
_NEUMANN_BC_ALL: halo_exchange_utils.BoundaryConditionsSpec = (
    (
        (halo_exchange_utils.BCType.NEUMANN, 0.0),
        (halo_exchange_utils.BCType.NEUMANN, 0.0),
    ),
    (
        (halo_exchange_utils.BCType.NEUMANN, 0.0),
        (halo_exchange_utils.BCType.NEUMANN, 0.0),
    ),
    (
        (halo_exchange_utils.BCType.NEUMANN, 0.0),
        (halo_exchange_utils.BCType.NEUMANN, 0.0),
    ),
)


def _get_first_last_grid_spacing(
    params: parameters_lib.SwirlLMParameters,
    dim: int,
) -> tuple[float, float]:
  """Returns the first and last grid spacing in `dim`, allowing nonuniform grid.

  Used for wall-like Neumann boundary conditions where the spacing at the
  boundary may differ from the interior (stretched grids).

  Args:
    params: The simulation parameters.
    dim: The physical dimension index (0=x, 1=y, 2=z), matching the convention
      used by `g_dim` and TF's `_get_first_last_grid_spacing_for_wall_bc`.

  Returns:
    A tuple (first_spacing, last_spacing).
  """
  # Convert physical xyz index to data-axis index. grid_spacings,
  # use_stretched_grid, and global_xyz_with_halos are all in data_axis_order.
  axis_name = ('x', 'y', 'z')[dim]
  data_dim = params.grid_params.get_axis_index(axis_name)
  if params.use_stretched_grid[data_dim]:  # pyrefly: ignore[bad-index]
    halo_width = params.halo_width
    coord = params.grid_params.global_xyz_with_halos[data_dim]  # pyrefly: ignore[bad-index]
    first_spacing = float(coord[halo_width] - coord[halo_width - 1])
    last_spacing = float(coord[-halo_width] - coord[-(halo_width + 1)])
  else:
    spacing = params.grid_spacings[data_dim]  # pyrefly: ignore[bad-index]
    first_spacing = last_spacing = spacing
  return first_spacing, last_spacing


class Pressure:
  """A library for solving the pressure equation."""

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
      thermodynamics: thermodynamics_manager.ThermodynamicsManager,
      solver_option: poisson_solver_pb2.PoissonSolver | None = None,
      num_d_rho_filter: int = _DEFAULT_NUM_D_RHO_FILTER,
  ):
    """Initializes the pressure library.

    Args:
      params: The simulation parameters.
      thermodynamics: The thermodynamics manager for computing reference density
        in flow-dependent pressure BCs.
      solver_option: The Poisson solver configuration proto. If None, uses a
        default CG solver.
      num_d_rho_filter: Number of density difference filter applications.
    """
    self._params = params
    self._kernel_op = params.kernel_op
    self._deriv_lib = params.deriv_lib
    self._grid_params = params.grid_params
    self._thermodynamics = thermodynamics

    if solver_option is not None:
      self._solver_option = solver_option
    else:
      self._solver_option = text_format.Parse(
          _DEFAULT_SOLVER_PARAMS, poisson_solver_pb2.PoissonSolver()
      )

    self._bc = dict(params.bc)

    self._solver = poisson_solver.poisson_solver_factory(
        self._grid_params,
        self._kernel_op,
        self._solver_option,
        use_stretched_grid=tuple(params.use_stretched_grid),
    )
    self._num_d_rho_filter = num_d_rho_filter

    # Pre-register filter kernels used by drho filtering so that
    # jax.lax.fori_loop in the step can call filter_op without triggering
    # kernel re-registration during tracing.
    self._kernel_op.add_kernel(
        {'shift_up': ([1.0, 0.0, 0.0], 1), 'shift_dn': ([0.0, 0.0, 1.0], 1)}
    )

    self._src_manager = physical_variable_keys_manager.SourceKeysHelper()
    self._source: dict[str, ScalarField | None] = {}

    # Set the initial flow-dependent pressure BCs.
    self._update_pressure_bc()

    # Hand the solver the conditions `dp` is posed with, from the same call
    # `_dp_halo_update_fn` makes, so the two cannot drift. Multigrid reads
    # these to build the exchanges for its coarse levels, which run at their
    # own shapes and so cannot reuse the closure passed to `solve`.
    #
    # After `_update_pressure_bc` for the same reason the reprojection
    # decision below is: `_build_dp_bc` reads `self._bc['p']`, which that call
    # is what populates.
    self._solver.boundary_conditions = self._build_dp_bc()

    # Conjugate gradient reprojects onto the range of the operator by removing
    # the mean, which is right for an all-Neumann problem, whose null space is
    # the constant, and wrong for one with a Dirichlet face, where the mean is
    # part of the solution. Determine from the boundary conditions whether to
    # subtract mean, unless explicitly configured otherwise.
    #
    # Decision needs to be made after `_update_pressure_bc`, which turns
    # `pressure_outlet` into the Dirichlet face that makes the operator
    # non-singular. Deciding before it reads the boundary conditions as
    # configured rather than as they end up (i.e., flow-dependent), and every
    # case that pins its outlet would reproject when it must not. The solver
    # holds the option message by reference and reads `reprojection` when it
    # solves, so settling it here still reaches it.
    if self._solver_option.HasField('conjugate_gradient'):
      cg_option = self._solver_option.conjugate_gradient
      if not cg_option.HasField('reprojection'):
        cg_option.reprojection = not self._has_dirichlet_pressure_bc()

  def _update_pressure_bc(self) -> None:
    """Updates the boundary condition of pressure based on the flow type.

    The pressure BC is derived from the type of each boundary face:
    - PERIODIC: No BC (handled by halo exchange periodicity).
    - INFLOW: Neumann (dp/dn = 0, 2nd order).
    - OUTFLOW: Dirichlet (p = 0) if pressure_outlet enabled, else Neumann.
    - WALL (non-slip, slip, shear): Neumann (dp/dn = 0).

    The diffusion contribution to the wall-normal momentum equation is zero at
    the wall (see detailed analysis in TF pressure module), so the wall
    pressure BC simplifies to a homogeneous Neumann condition.
    """
    bc_p: list[list | None] = [[None, None], [None, None], [None, None]]  # pylint: disable=g-bare-generic
    periodic = list(self._grid_params.to_xyz_order(self._params.periodic_dims))  # pyrefly: ignore[bad-argument-type]

    wall_types = (
        boundary_condition_utils.BoundaryType.SLIP_WALL,
        boundary_condition_utils.BoundaryType.NON_SLIP_WALL,
        boundary_condition_utils.BoundaryType.SHEAR_WALL,
    )

    for dim in range(3):
      for face in range(2):
        bt = self._params.bc_type[dim][face]

        if (
            bt == boundary_condition_utils.BoundaryType.PERIODIC
            or periodic[dim]
        ):
          bc_p[dim][face] = None  # pyrefly: ignore[unsupported-operation]

        elif bt == boundary_condition_utils.BoundaryType.INFLOW:
          bc_p[dim][face] = (  # pyrefly: ignore[unsupported-operation]
              halo_exchange_utils.BCType.NEUMANN_2,
              0.0,
          )

        elif bt == boundary_condition_utils.BoundaryType.OUTFLOW:
          # Use Dirichlet if pressure_outlet is configured, else Neumann.
          if (
              self._params.pressure is not None
              and self._params.pressure.HasField('pressure_outlet')
              and self._params.pressure.pressure_outlet
          ):
            bc_p[dim][face] = (  # pyrefly: ignore[unsupported-operation]
                halo_exchange_utils.BCType.DIRICHLET,
                0.0,
            )
          else:
            bc_p[dim][face] = (  # pyrefly: ignore[unsupported-operation]
                halo_exchange_utils.BCType.NEUMANN_2,
                0.0,
            )

        elif bt in wall_types:
          # Homogeneous Neumann BC for pressure at walls.
          bc_p[dim][face] = (  # pyrefly: ignore[unsupported-operation]
              halo_exchange_utils.BCType.NEUMANN,
              0.0,
          )

        else:
          bc_p[dim][face] = (  # pyrefly: ignore[unsupported-operation]
              halo_exchange_utils.BCType.NEUMANN,
              0.0,
          )

    self._bc['p'] = bc_p  # pyrefly: ignore[unsupported-operation]

  def _pressure_bc_balanced_vertical(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      face: int,
  ) -> tuple[halo_exchange_utils.BCType, list[ScalarField]]:
    """Sets buoyancy-balanced vertical pressure BC.

    Avoids spurious forcing at vertical boundaries by balancing the pressure
    gradient with the buoyancy force. For the lower face (face=0), extrapolates
    buoyancy to the wall and uses Neumann BC. For the upper face (face=1), uses
    Dirichlet BC that enforces p_N = p_{N-2} + 2*dz*b_{N-1}.

    Args:
      states: Flow field variables (must include 'rho_thermal', 'p').
      additional_states: Helper variables (must include 'zz' for rho_ref).
      face: Which face: 0 (bottom) or 1 (top).

    Returns:
      A tuple of (BCType, bc_planes) for the halo exchange.
    """
    g_dim = self._params.g_dim
    assert g_dim is not None
    halo_width = self._params.halo_width
    # g_dim is a physical xyz index (0=x, 1=y, 2=z), not a data-axis index.
    g_axis = ('x', 'y', 'z')[g_dim]

    # Compute buoyancy source.
    zz = additional_states.get('zz', None)
    rho_0 = self._thermodynamics.rho_ref(zz, additional_states)  # pyrefly: ignore[bad-argument-type]
    b_raw = eq_utils.buoyancy_source(
        states['rho_thermal'],
        rho_0,
        self._params,
        g_dim,
        additional_states,
    )

    # For ANELASTIC mode, divide by reference density.
    if self._params.solver_mode == thermodynamics_pb2.Thermodynamics.ANELASTIC:
      b = b_raw / rho_0
    else:
      b = b_raw

    b_first = common_ops.get_face(
        b, g_axis, face, halo_width, self._grid_params  # pyrefly: ignore[bad-argument-type]
    )
    b_second = common_ops.get_face(
        b, g_axis, face, halo_width + 1, self._grid_params  # pyrefly: ignore[bad-argument-type]
    )

    if face == 0:
      # Extrapolate buoyancy to wall: b_{-1/2} = (3*b_0 - b_1) / 2.
      b_wall = (3.0 * b_first - b_second) / 2.0
      # Multiply by the grid spacing at the wall, which may differ from
      # the interior spacing on stretched grids.
      first_spacing, _ = _get_first_last_grid_spacing(self._params, g_dim)
      dz = first_spacing
      bc_value = dz * b_wall
      return (
          halo_exchange_utils.BCType.NEUMANN,
          [jnp.zeros_like(bc_value)] * (halo_width - 1) + [bc_value],
      )
    else:
      # Upper face: p_N = p_{N-2} + 2*dz*b_{N-1}.
      p_second = common_ops.get_face(
          states['p'], g_axis, face, halo_width + 1, self._grid_params  # pyrefly: ignore[bad-argument-type]
      )
      # Use last grid spacing for stretched grids.
      data_dim = self._grid_params.get_axis_index(g_axis)
      if self._params.use_stretched_grid[data_dim]:  # pyrefly: ignore[bad-index]
        coord = self._grid_params.global_xyz_with_halos[data_dim]  # pyrefly: ignore[bad-index]
        if halo_width > 1:
          dz_last = (
              float(coord[-(halo_width - 1)] - coord[-(halo_width + 1)]) / 2.0
          )
        else:
          dz_last = float(coord[-1] - coord[-2])
      else:
        dz_last = self._params.grid_spacings[data_dim]  # pyrefly: ignore[bad-index]
      bc_value = p_second + 2.0 * dz_last * b_first
      return (
          halo_exchange_utils.BCType.DIRICHLET,
          [bc_value] * halo_width,
      )

  def _pressure_bc_approximate_vertical(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      face: int,
  ) -> tuple[halo_exchange_utils.BCType, list[ScalarField]]:
    """Sets approximate buoyancy-balanced vertical pressure BC.

    Original implementation where pressure and buoyancy at the wall only
    approximately balance. Uses Neumann BC on both faces with buoyancy
    averaged between the first interior node and the halo-adjacent node.

    Args:
      states: Flow field variables (must include 'rho_thermal', 'p').
      additional_states: Helper variables (must include 'zz' for rho_ref).
      face: Which face: 0 (bottom) or 1 (top).

    Returns:
      A tuple of (BCType, bc_planes) for the halo exchange.
    """
    g_dim = self._params.g_dim
    assert g_dim is not None
    halo_width = self._params.halo_width
    g_axis = ('x', 'y', 'z')[g_dim]

    zz = additional_states.get('zz', None)
    rho_0 = self._thermodynamics.rho_ref(zz, additional_states)  # pyrefly: ignore[bad-argument-type]
    b = eq_utils.buoyancy_source(
        states['rho_thermal'],
        rho_0,
        self._params,
        g_dim,
        additional_states,
    )

    # Average buoyancy between first interior and halo-adjacent nodes.
    b_first = common_ops.get_face(
        b, g_axis, face, halo_width, self._grid_params  # pyrefly: ignore[bad-argument-type]
    )
    b_second = common_ops.get_face(
        b, g_axis, face, halo_width - 1, self._grid_params  # pyrefly: ignore[bad-argument-type]
    )
    bc_value = 0.5 * (b_first + b_second)

    # Multiply by grid spacing for NEUMANN BC.
    first_spacing, last_spacing = _get_first_last_grid_spacing(
        self._params, g_dim
    )
    dz = first_spacing if face == 0 else last_spacing
    bc_value = dz * bc_value

    if face == 0:
      return (
          halo_exchange_utils.BCType.NEUMANN,
          [jnp.zeros_like(bc_value)] * (halo_width - 1) + [bc_value],
      )
    else:
      return (
          halo_exchange_utils.BCType.NEUMANN,
          [bc_value] + [jnp.zeros_like(bc_value)] * (halo_width - 1),
      )

  def update_pressure_bc_by_flow(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> None:
    """Updates pressure BCs dynamically based on the flow field.

    When the gravity dimension has wall boundaries, the pressure BC at those
    faces is set to balance with buoyancy. The treatment depends on the
    ``vertical_bc_treatment`` proto setting:
    - APPROXIMATE (default): Uses averaged buoyancy with Neumann BC on both
      faces.
    - PRESSURE_BUOYANCY_BALANCING: Uses extrapolated buoyancy with Neumann
      on bottom and Dirichlet on top.

    Args:
      states: Flow field variables.
      additional_states: Helper variables.
    """
    g_dim = self._params.g_dim
    if g_dim is None:
      return

    wall_types = (
        boundary_condition_utils.BoundaryType.SLIP_WALL,
        boundary_condition_utils.BoundaryType.NON_SLIP_WALL,
        boundary_condition_utils.BoundaryType.SHEAR_WALL,
    )

    # Determine vertical BC treatment from proto config.
    vertical_bc = pressure_pb2.Pressure.APPROXIMATE
    if self._params.pressure is not None:
      vertical_bc = self._params.pressure.vertical_bc_treatment

    for face_idx in (0, 1):
      bt = self._params.bc_type[g_dim][face_idx]
      if bt in wall_types:
        if vertical_bc == pressure_pb2.Pressure.PRESSURE_BUOYANCY_BALANCING:
          bc = self._pressure_bc_balanced_vertical(
              states, additional_states, face_idx
          )
        else:
          bc = self._pressure_bc_approximate_vertical(
              states, additional_states, face_idx
          )
        self._bc['p'][g_dim][face_idx] = bc  # pyrefly: ignore[unsupported-operation]

  def update_pressure_halos(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      mesh: jax.sharding.Mesh,
  ) -> dict[str, ScalarField]:
    """Updates pressure halos with flow-dependent BCs.

    Should be called after the pressure solve when buoyancy-balanced
    vertical BCs are needed. This re-derives the pressure BC from the
    current flow field and re-applies halo exchange.

    Args:
      states: Flow field variables (must include 'p', 'rho_thermal').
      additional_states: Helper variables.
      mesh: JAX device mesh.

    Returns:
      A dictionary with the updated 'p' field.
    """
    update_by_flow = (
        self._params.pressure is not None
        and self._params.pressure.update_p_bc_by_flow
    )
    if update_by_flow:
      self.update_pressure_bc_by_flow(states, additional_states)

    halo_width = self._params.halo_width
    periodic = list(self._grid_params.to_xyz_order(self._params.periodic_dims))  # pyrefly: ignore[bad-argument-type]
    p_updated = halo_exchange.inplace_halo_exchange(
        states['p'],
        _ALL_AXES,
        mesh,
        self._grid_params,
        periodic,
        self._bc.get('p'),
        halo_width=halo_width,
    )
    return {'p': p_updated}

  def _compute_divergence_term(
      self,
      rho_u: ScalarField,
      rho_v: ScalarField,
      rho_w: ScalarField,
      dt: float,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes nabla dot (rho u) / dt for the RHS of the Poisson equation."""
    axes = self._grid_params.data_axis_order
    div = jnp.zeros_like(rho_u)
    momentum = {'x': rho_u, 'y': rho_v, 'z': rho_w}
    for axis in axes:
      div = div + self._deriv_lib.deriv_centered(
          momentum[axis], axis, additional_states
      )
    return div / dt

  def _numerical_consistency_correction(
      self,
      dp: ScalarField,
      rho: ScalarField,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes the numerical consistency term.

    Computes the difference between the 2h-stencil and h-stencil Laplacians
    to correct for the inconsistency between the pressure gradient and velocity
    divergence approximations.

    Args:
      dp: The pressure correction.
      rho: The reference density (used for anelastic mode).
      additional_states: Helper variables for stretched grid support.

    Returns:
      The numerical consistency correction term.
    """
    anelastic = (
        self._params.solver_mode == thermodynamics_pb2.Thermodynamics.ANELASTIC
    )
    axes = self._grid_params.data_axis_order

    # 1h-stencil: node -> face -> node.
    derivs_1h: list[ScalarField] = []
    for axis in axes:
      inner = self._deriv_lib.deriv_node_to_face(dp, axis, additional_states)
      if anelastic:
        rho_face = interpolation.centered_node_to_face(
            rho, axis, self._kernel_op
        )
        inner = rho_face * inner
      outer = self._deriv_lib.deriv_face_to_node(inner, axis, additional_states)
      derivs_1h.append(outer)

    # 2h-stencil: centered -> centered.
    derivs_2h: list[ScalarField] = []
    for axis in axes:
      inner = self._deriv_lib.deriv_centered(dp, axis, additional_states)
      if anelastic:
        inner = rho * inner
      outer = self._deriv_lib.deriv_centered(inner, axis, additional_states)
      derivs_2h.append(outer)

    div_2h = derivs_2h[0] + derivs_2h[1] + derivs_2h[2]
    div_1h = derivs_1h[0] + derivs_1h[1] + derivs_1h[2]
    return div_2h - div_1h

  def _rhie_chow_numerical_consistency_correction(
      self,
      p: ScalarField,
      grid_spacings: tuple[float, ...],
  ) -> ScalarField:
    """Computes the Rhie-Chow correction for the pressure Poisson equation.

    When the Rhie-Chow correction is enabled for face-flux interpolation,
    the corresponding numerical consistency term in the Poisson RHS uses
    4th-order pressure derivatives: sum_i d^4p / dx_i^4 / (4 * dx_i^2).

    Args:
      p: The pressure field.
      grid_spacings: Grid spacings (dx, dy, dz).

    Returns:
      The Rhie-Chow numerical consistency correction.
    """
    axes = self._grid_params.data_axis_order

    correction = jnp.zeros_like(p)
    for i, axis in enumerate(axes):
      # 4th-order centered derivative: k4d2 kernel applies d^4/dx^4.
      d4p = self._kernel_op.apply_kernel_op(p, 'k4d2', axis)
      correction = correction + d4p / (4.0 * grid_spacings[i] ** 2)

    return correction

  def _has_dirichlet_pressure_bc(self) -> bool:
    """Whether any non-periodic face carries a Dirichlet condition on `p`.

    Determines whether the Poisson operator is singular. With every
    non-periodic face set to Neumann, the constant is in its null space, so the
    solution is only determined up to a constant. One Dirichlet face pins it.

    Returns:
      True if at least one non-periodic face has a Dirichlet condition on `p`.
    """
    periodic = list(self._grid_params.to_xyz_order(self._params.periodic_dims))  # pyrefly: ignore[bad-argument-type]
    bc_p = self._bc.get('p')
    if bc_p is None:
      return False
    return any(
        not periodic[dim]
        and bc_p[dim] is not None  # pyrefly: ignore[unsupported-operation]
        and any(
            bc_p[dim][face] is not None  # pyrefly: ignore[unsupported-operation]
            and bc_p[dim][face][0] == halo_exchange_utils.BCType.DIRICHLET  # pyrefly: ignore[unsupported-operation]
            for face in range(2)
        )
        for dim in range(3)
    )

  def _build_dp_bc(
      self,
  ) -> halo_exchange_utils.BoundaryConditionsSpec:
    """Builds the boundary conditions for the pressure correction."""
    bc_dp_list: list[halo_exchange_utils.DimBoundaryConditions | None] = []
    periodic = list(self._grid_params.to_xyz_order(self._params.periodic_dims))  # pyrefly: ignore[bad-argument-type]
    for dim in range(3):
      if periodic[dim]:
        bc_dp_list.append(None)
      else:
        # Check if pressure has Dirichlet BC -- if so, dp should be 0 there.
        lo: halo_exchange_utils.FaceBoundaryCondition | None = (
            halo_exchange_utils.BCType.NEUMANN,
            0.0,
        )
        hi: halo_exchange_utils.FaceBoundaryCondition | None = (
            halo_exchange_utils.BCType.NEUMANN,
            0.0,
        )
        if self._bc.get('p') is not None and self._bc['p'] is not None:
          p_bc = self._bc['p']
          if (
              p_bc[dim] is not None
              and p_bc[dim][0] is not None  # pyrefly: ignore[unsupported-operation]
              and p_bc[dim][0][0] == halo_exchange_utils.BCType.DIRICHLET  # pyrefly: ignore[unsupported-operation]
          ):
            lo = (halo_exchange_utils.BCType.DIRICHLET, 0.0)
          if (
              p_bc[dim] is not None
              and p_bc[dim][1] is not None  # pyrefly: ignore[unsupported-operation]
              and p_bc[dim][1][0] == halo_exchange_utils.BCType.DIRICHLET  # pyrefly: ignore[unsupported-operation]
          ):
            hi = (halo_exchange_utils.BCType.DIRICHLET, 0.0)
        bc_dp_list.append((lo, hi))
    return tuple(bc_dp_list)

  def _max_iterations(self) -> int:
    """The iteration limit the configured solver was given.

    Multigrid uses `num_iterations`, counting V-cycles rather than
    iterations.

    Returns:
      The limit, or 0 if the selected solver does not declare one.
    """
    selected = self._solver_option.WhichOneof('solver')
    if selected is None:
      return 0
    option = getattr(self._solver_option, selected)
    for field in ('max_iterations', 'num_iterations'):
      if getattr(option, field, 0):
        return getattr(option, field)
    return 0

  def _report_poisson_residual(
      self,
      dp: ScalarField,
      b: ScalarField,
      mesh: jax.sharding.Mesh,
      halo_update_fn: Callable[[ScalarField], ScalarField],
      singular: bool,
      iterations: jax.Array | None = None,
  ) -> None:
    """Prints the relative L2 residual of the pressure solve.

    Reports `||A dp - b|| / ||b||` rather than the absolute residual, so the
    number is comparable across cases and across grid refinements. The norms
    are taken over the interior: `compute_residual` zeroes the halos of the
    residual, and `b` is zeroed here to match, so each cell counts once even
    though neighbouring replicas hold copies of each other's edge cells.

    Called only when `params.log_poisson_residual` is set. The line is emitted
    by every replica, carrying the same globally reduced value, because an
    `axis_index` predicate around a debug callback is not supported under SPMD
    partitioning.

    Args:
      dp: The pressure correction the solver returned.
      b: The right hand side it was given.
      mesh: The device mesh.
      halo_update_fn: The boundary conditions `dp` was solved against. Passing
        this for cases whose boundary conditions are not all Neumann.
      singular: Whether the problem is all-Neumann, in which case the mean is
        removed from `b` before measuring, since a singular operator cannot
        resolve a constant shift.
      iterations: How many iterations the solve took, when it reports that.
    """
    halo_width = self._grid_params.halo_width
    norms, _ = self._solver.compute_residual(
        mesh,
        dp,
        b,
        norm_types=(common_ops.NormType.L2,),
        halo_update_fn=halo_update_fn,
        remove_mean_from_rhs=singular,
    )
    b_interior = halo_exchange.set_halos_to_zero(
        b, halo_width, self._grid_params
    )
    b_norm = common_ops.compute_norm(
        b_interior, {common_ops.NormType.L2}, mesh
    )['L2']
    # `compute_norm` returns shape-(1,) arrays; a format spec needs a scalar.
    residual_norm = jnp.squeeze(norms['L2'])
    rhs_norm = jnp.squeeze(b_norm)
    relative = residual_norm / jnp.where(rhs_norm > 0.0, rhs_norm, 1.0)
    if iterations is None:
      jax.debug.print(
          'Poisson residual: relative={relative:.3e} absolute={absolute:.3e} '
          'rhs={rhs:.3e}',
          relative=relative,
          absolute=residual_norm,
          rhs=rhs_norm,
      )
    else:
      limit = self._max_iterations()
      jax.debug.print(
          'Poisson residual: relative={relative:.3e} absolute={absolute:.3e} '
          'rhs={rhs:.3e} iterations={iterations}/{limit}'
          ' at_limit={at_limit}',
          relative=relative,
          absolute=residual_norm,
          rhs=rhs_norm,
          iterations=iterations,
          limit=limit,
          at_limit=iterations >= limit,
      )

  def _report_continuity_residual(
      self,
      states: ScalarFieldMap,
      states_0: ScalarFieldMap,
      dp: ScalarField,
      divergence_term: ScalarField,
      physical_rhs: ScalarField,
      mesh: jax.sharding.Mesh,
      additional_states: ScalarFieldMap,
  ) -> None:
    """Prints how far the corrected momentum is from satisfying continuity.

    The Poisson residual tells how well the linear system was solved. It does
    not tell whether the correction removed any divergence.
    The solver inverts the compact `kdd` Laplacian while
    `velocity.correction_step` reaches the divergence through
    `deriv_centered` twice, and the two differ most on the odd-even mode. This
    replicates the correction and measures what it achieved.

    The target may not always be zero divergence. At low Mach number continuity
    reads `div(rho u) = -drho/dt + src_rho`. With combustion, the
    corrected momentum should carry a large divergence, and comparing
    `||div(rho u)||` before and after would be misleading.
    Reported instead is the residual of that equation,
    `||div(rho u)/dt - rhs_phys||`, where `rhs_phys` is the right hand side
    without the numerical consistency term, measured before and after the
    correction. For a constant-density case with no mass source `rhs_phys`
    is the divergence term, so the two formulations coincide.

    **Restricted to the fluid** by masking with the ib_interior_mask. Intended
    to report continuity in the fluid domain only.

    Called only when `params.log_poisson_residual` is set. The correction is
    recomputed rather than shared with `correction_step`, so this must stay in
    step with `velocity.correction_step` if that changes.

    Args:
      states: Flow field variables from the latest prediction.
      states_0: Flow field variables from the previous time step.
      dp: The pressure correction the solver returned.
      divergence_term: `div(rho u) / dt` as the right hand side used it.
      physical_rhs: The right hand side without the numerical consistency
        term, which is what continuity asks `div(rho u) / dt` to equal.
      mesh: The device mesh.
      additional_states: Helper variables, for the stretched-grid factors.
    """
    dt = self._params.dt
    halo_width = self._grid_params.halo_width
    anelastic = (
        self._params.solver_mode == thermodynamics_pb2.Thermodynamics.ANELASTIC
    )
    rho_mid = (
        states['rho']
        if anelastic
        else 0.5 * (states['rho'] + states_0['rho'])
    )

    corrected = {}
    for key, axis in zip(('rho_u', 'rho_v', 'rho_w'), ('x', 'y', 'z')):
      grad_dp = self._deriv_lib.deriv_centered(dp, axis, additional_states)
      if anelastic:
        corrected[key] = states[key] - dt * rho_mid * grad_dp
      else:
        corrected[key] = states[key] - dt * grad_dp

    divergence_after = self._compute_divergence_term(
        corrected['rho_u'],
        corrected['rho_v'],
        corrected['rho_w'],
        dt,
        additional_states,
    )

    # One in fluid, zero in solid; absent when the case has no body, in which
    # case every cell counts and the numbers are what they always were.
    fluid = additional_states.get(_INTERIOR_MASK_KEY)

    def norm(field: ScalarField) -> jax.Array:
      if fluid is not None:
        field = field * fluid
      interior = halo_exchange.set_halos_to_zero(
          field, halo_width, self._grid_params
      )
      return jnp.squeeze(
          common_ops.compute_norm(interior, {common_ops.NormType.L2}, mesh)[
              'L2'
          ]
      )

    # Continuity requires `div(rho u)/dt = rhs_phys` once the correction is
    # applied, so with `dp = 0` the residual is `rhs_phys` itself, and after
    # the correction it is `rhs_phys` less what the correction removed. For a
    # constant-density case with no mass source `rhs_phys` is the divergence
    # term and these reduce to the divergence before and after.
    before = norm(physical_rhs)
    after = norm(physical_rhs - (divergence_term - divergence_after))
    ratio = after / jnp.where(before > 0.0, before, 1.0)
    if fluid is None:
      jax.debug.print(
          'Continuity: before={before:.3e} after={after:.3e} ratio={ratio:.3e}',
          before=before,
          after=after,
          ratio=ratio,
      )
    else:
      # The fraction counted, so the line says what it measured rather than
      # leaving a reader to infer it from which case produced the log.
      counted = jnp.squeeze(
          common_ops.global_mean(
              fluid,
              mesh,
              halo_width,
              halo_width,
              halo_width,
              self._grid_params,
          )
      )
      jax.debug.print(
          'Continuity: before={before:.3e} after={after:.3e} '
          'ratio={ratio:.3e} fluid={fluid:.3f}',
          before=before,
          after=after,
          ratio=ratio,
          fluid=counted,
      )

  def _dp_halo_update_fn(
      self,
      dp: ScalarField,
      mesh: jax.sharding.Mesh,
  ) -> ScalarField:
    """Updates halos for the pressure correction with homogeneous Neumann BC."""
    return halo_exchange.inplace_halo_exchange(
        dp,
        _ALL_AXES,
        mesh,
        self._grid_params,
        list(self._grid_params.to_xyz_order(self._params.periodic_dims)),  # pyrefly: ignore[bad-argument-type]
        self._build_dp_bc(),
        halo_width=self._params.halo_width,
    )

  def prestep(
      self,
      additional_states: ScalarFieldMap,
  ) -> None:
    """Updates additional information required for pressure step.

    This function is called before the beginning of each time step. It updates
    the mass source term on the right hand side of the Poisson equation. These
    information will be held within this helper object.

    Args:
      additional_states: A dictionary that holds constants that will be used in
        the simulation, e.g. boundary conditions, forcing terms.
    """
    # Parse additional states to extract external source/forcing terms.
    self._source.update(
        self._src_manager.update_helper_variable_from_additional_states(
            additional_states
        )
    )

  def step(
      self,
      states: ScalarFieldMap,
      states_0: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      mesh: jax.sharding.Mesh,
  ) -> dict[str, ScalarField]:
    """Updates the pressure and its correction for the current subiteration.

    Args:
      states: A dictionary holding flow field variables from the latest
        prediction. Must contain 'rho_u', 'rho_v', 'rho_w', 'p', 'dp', 'rho',
        and optionally 'drho'.
      states_0: A dictionary holding flow field variables from the previous time
        step.
      additional_states: A dictionary holding constants and helper variables.
      mesh: A jax Mesh object representing the device topology.

    Returns:
      A dictionary with the updated pressure ('p') and pressure correction
      ('dp').
    """
    dt = self._params.dt
    inv_dt = 1.0 / dt
    halo_width = self._params.halo_width
    periodic = list(self._grid_params.to_xyz_order(self._params.periodic_dims))  # pyrefly: ignore[bad-argument-type]

    # === Compute drho_dt ===
    if self._params.solver_mode == thermodynamics_pb2.Thermodynamics.ANELASTIC:
      drho_dt = jnp.zeros_like(states['rho'])
    else:
      # Low Mach mode.
      if 'drho' in states:
        drho_0 = jnp.where(
            jnp.abs(states['drho']) < 1e-8 * jnp.abs(states_0['rho']),
            0.0,
            states['drho'],
        )

        # Apply drho filtering using jax.lax.fori_loop.
        # Filter kernels are pre-registered in __init__ so add_kernel()
        # is a no-op inside the traced loop body.
        def filter_body(_: int, drho_i: ScalarField) -> ScalarField:
          filtered = filters.filter_op(
              self._kernel_op,
              self._grid_params,
              drho_i,
              additional_states,
              order=2,
          )
          # Halo exchange after filter (homogeneous Neumann).
          return halo_exchange.inplace_halo_exchange(
              filtered,
              _ALL_AXES,
              mesh,
              self._grid_params,
              periodic,
              _NEUMANN_BC_ALL,
              halo_width=halo_width,
          )

        drho = jax.lax.fori_loop(0, self._num_d_rho_filter, filter_body, drho_0)
        drho_dt = drho * inv_dt
      else:
        # Constant density case (e.g., TGV): no density change.
        drho_dt = jnp.zeros_like(states['p'])

    # === Build RHS of Poisson equation ===
    # b = div(rho*u)/dt + drho_dt/dt + numerical_consistency
    divergence_term = self._compute_divergence_term(
        states['rho_u'],
        states['rho_v'],
        states['rho_w'],
        dt,
        additional_states,
    )

    if self._params.enable_rhie_chow_correction:
      numerical_consistency = self._rhie_chow_numerical_consistency_correction(
          states['p'], self._params.grid_spacings
      )
    else:
      numerical_consistency = self._numerical_consistency_correction(
          states['dp'], states['rho'], additional_states
      )

    # Source term (mass source). Supports both JAX convention ('mass_source')
    # and TF convention ('src_rho'), as well as source_update_fn for 'rho'.
    src_rho = additional_states.get(
        'mass_source',
        additional_states.get('src_rho', jnp.zeros_like(states['p'])),
    )
    rho_src_fn = self._params.source_update_fn('rho')
    if rho_src_fn is not None:
      src_rho = src_rho + rho_src_fn(states, additional_states)  # pyrefly: ignore[unsupported-operation]

    b = (
        (divergence_term + numerical_consistency)
        + drho_dt * inv_dt
        - src_rho * inv_dt
    )

    # === Solve Poisson equation ===
    dp0 = jnp.zeros_like(b)

    halo_update_fn = functools.partial(self._dp_halo_update_fn, mesh=mesh)

    poisson_solution = self._solver.solve(
        rhs=b,
        p0=dp0,
        mesh=mesh,
        halo_update_fn=halo_update_fn,
        additional_states=dict(additional_states),
    )

    dp: ScalarField = jnp.asarray(poisson_solution[base_poisson_solver.X])

    # Remove global mean for all-Neumann problems (no Dirichlet BC).
    has_dirichlet_bc = self._has_dirichlet_pressure_bc()

    if not has_dirichlet_bc:
      dp = dp - common_ops.global_mean(
          dp, mesh, halo_width, halo_width, halo_width, self._grid_params
      )

    if self._params.log_poisson_residual:
      self._report_poisson_residual(
          dp,
          b,
          mesh,
          halo_update_fn,
          singular=not has_dirichlet_bc,
          iterations=poisson_solution.get(base_poisson_solver.ITERATIONS),  # pyrefly: ignore[bad-argument-type]
      )
      self._report_continuity_residual(
          states,
          states_0,
          dp,
          divergence_term,
          b - numerical_consistency,
          mesh,
          additional_states,
      )

    return {
        'p': states['p'] + dp,
        'dp': dp,
    }
