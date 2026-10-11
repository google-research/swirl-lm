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
"""Flow past a circular cylinder, the immersed boundary validation case.

This is the case Ghias, Mittal and Dong use to validate their ghost-cell
method (J. Comput. Phys. 225 (2007) 528-553, section 3.3), and it is here for
the same reason: it has published answers, and it exercises a curved boundary
that no height-field description could represent.

Usage mirrors the other examples, except that the case parameters come from a
`CylinderCase` proto and the solver configuration is derived from them:

  ```python
  simulation = cylinder.Cylinder(case)
  params = simulation.params
  state = driver.run_simulation(
      params=params,
      init_fn=simulation.initialization,
      num_steps=20000,
  )
  print(simulation.recirculation_length(state))
  ```

Deriving the solver configuration rather than writing it by hand is
deliberate. The Reynolds number, the viscosity, the grid spacing, the time
step and the immersed boundary geometry all have to agree, and the usual way
one of these runs goes wrong is that the Reynolds number is changed and the
viscosity is not. Here `nu = u_inf * d / Re` by construction.

What to expect:

  Re = 40   steady, symmetric, closed recirculation bubble of length
            L/d ~ 2.2 measured from the rear of the cylinder.
  Re = 100  periodic shedding, Strouhal number ~ 0.165 to 0.17.

Drag is not computed: that needs the surface force integration of the paper's
Eq. 32, which is not implemented. `recirculation_length` and
`strouhal_number` are both derived from the velocity field alone.
"""

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from google.protobuf import text_format
from swirl_lm.base import parameters_pb2
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.boundary_condition import immersed_boundary_ghost_point
from swirl_lm.jax.examples.cylinder import cylinder_pb2
from swirl_lm.jax.utility import types


ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

VELOCITY = ('u', 'v', 'w')

# `convection.py` and `interpolation.py` build their own kernel operator with
# this width regardless of the configuration, so every per-core extent has to
# be divisible by it as well as by the configured `kernel_size`.
_CONVECTION_KERNEL_SIZE = 4


class Cylinder:
  """Flow past a circular cylinder, with a ghost-point immersed boundary."""

  def __init__(self, case: cylinder_pb2.CylinderCase | None = None):
    """Builds the case and its solver configuration.

    Args:
      case: The case parameters. Defaults are used for anything unset.

    Raises:
      ValueError: If the case is not self-consistent, for example a grid that
        does not divide evenly across the requested devices.
    """
    self.case = case or cylinder_pb2.CylinderCase()
    self._validate()

    diameter = self.case.diameter
    self.diameter = diameter
    self.u_inf = self.case.inflow_speed
    self.nu = self.u_inf * diameter / self.case.reynolds_number

    # Physical extents. The cylinder sits `length_upstream` diameters from
    # the inflow so the wake has room to develop downstream.
    self.lx = (self.case.length_upstream + self.case.length_downstream) * diameter
    self.ly = 2.0 * self.case.half_width * diameter
    self.lz = self.case.span * diameter
    self.center = (
        self.case.length_upstream * diameter,
        0.5 * self.ly,
        0.5 * self.lz,
    )

    cells = self.case.cells_per_diameter
    self.fx = int(round(self.lx / diameter * cells))
    self.fy = int(round(self.ly / diameter * cells))
    self.fz = self.case.spanwise_cells
    self.dx = self.lx / self.fx

    # The CFL is set from the free-stream speed, but the flow accelerates to
    # roughly twice that around the cylinder, so the number the solver
    # reports comes out about twice `cfl`. Measured on a coarse Re = 40 run,
    # `cfl = 0.5` reported 1.04.
    self.dt = (
        self.case.dt
        if self.case.HasField('dt')
        else self.case.cfl * self.dx / self.u_inf
    )

    self.params = self._build_params()
    self.gp = self.params.grid_params

    # The promise this class makes is that the Reynolds number and the
    # viscosity agree. Check it on the parsed configuration rather than
    # trusting the template, so a later edit to either cannot break it
    # quietly. The tolerance is float32 round-trip through the proto.
    expected_re = self.u_inf * diameter / self.params.nu
    if not math.isclose(
        expected_re, self.case.reynolds_number, rel_tol=1e-5
    ):
      raise ValueError(
          f'The generated configuration has nu = {self.params.nu:g}, which '
          f'is Re = {expected_re:g}, not the requested '
          f'{self.case.reynolds_number:g}.'
      )

    # The immersed boundary. The geometry comes from the generated config, so
    # the solver and the diagnostics below describe the same body.
    self.geometry_fn = immersed_boundary_ghost_point.geometry_from_proto(
        self.params.boundary_models.ib.ghost_point.geometry
    )
    self._ghost_points: Any = None

  # -- Configuration --------------------------------------------------------

  def _validate(self) -> None:
    """Checks the case parameters before anything is built from them."""
    case = self.case
    if case.reynolds_number <= 0.0:
      raise ValueError(
          f'`reynolds_number` must be positive, got {case.reynolds_number}.'
      )
    if case.diameter <= 0.0 or case.inflow_speed <= 0.0:
      raise ValueError('`diameter` and `inflow_speed` must be positive.')
    if case.cells_per_diameter < 8:
      raise ValueError(
          f'`cells_per_diameter` is {case.cells_per_diameter}; below about 8 '
          'the cylinder is not a circle on this grid and the result is '
          'meaningless. Twenty is a reasonable minimum at Re = 100.'
      )
    if case.halo_width < 2:
      raise ValueError(
          f'`halo_width` is {case.halo_width}; the QUICK convection scheme '
          'needs 2, and the ghost-point stencil may need more.'
      )

  def _grid_size(self, interior: int, cores: int, axis: str) -> int:
    """Per-core grid size including halos, with the divisibility checks.

    Args:
      interior: Interior points along this axis, over the whole domain.
      cores: Devices along this axis.
      axis: Axis name, for the error messages.

    Returns:
      The per-core extent including halos.

    Raises:
      ValueError: If the domain does not split evenly, or if the resulting
        extent is not divisible by a kernel width in use.
    """
    if interior % cores:
      raise ValueError(
          f'The {axis} direction has {interior} interior points, which does '
          f'not divide evenly across {cores} cores. Adjust '
          '`cells_per_diameter` or the domain extent.'
      )
    extent = interior // cores + 2 * self.case.halo_width

    # A per-core extent that is not divisible by a kernel width sends that
    # kernel down a slice fallback which disagrees with the convolution path
    # for some stencils -- `kf2+` by a whole cell, and `shift_up`/`shift_dn`
    # swapped. Nothing warns. See MEMORY.md, kernel-operator defects.
    #
    # Two widths are in play and only one of them is configurable: the case's
    # `kernel_size`, and the 4 that `convection.py` and `interpolation.py`
    # hard-code when they build their own operator.
    for width, source in (
        (self.case.kernel_size, '`kernel_size`'),
        (_CONVECTION_KERNEL_SIZE, 'the width convection.py hard-codes'),
    ):
      if extent % width:
        remedy = (
            f'raise `spanwise_cells` to a multiple of {width * cores}'
            if axis == 'z'
            else 'adjust `cells_per_diameter` or the domain extent'
        )
        raise ValueError(
            f'The {axis} direction has {extent} points per core, which is '
            f'not divisible by {width} ({source}). That silently selects a '
            'finite-difference path that disagrees with the usual one for '
            f'some stencils. To split this axis across {cores} cores, '
            f'{remedy}, or change `kernel_size`.'
        )
    return extent

  def config_text(self) -> str:
    """The solver configuration this case expands into, as a text proto.

    Returned rather than only used internally so a run can record exactly
    what it solved, and so the derivation can be checked without reading
    Python.

    Returns:
      A `SwirlLMParameters` text proto.
    """
    case = self.case
    inflow = {'u': self.u_inf, 'v': 0.0, 'w': 0.0}

    def velocity_bc(name):
      # Inlet and both lateral faces carry the free stream; only the outlet is
      # zero gradient.
      #
      # The lateral faces matter more than they look. Leaving them zero
      # gradient makes `find_bc_type.is_outflow` classify them as outlets --
      # all three components Neumann is its entire test -- so the sides of the
      # domain become outlets, nothing constrains transverse momentum, and the
      # wake drifts off the centreline and leaves through the side. That is
      # what the first Re = 100 run did. Ghias et al. section 2.4 (p. 533)
      # specify the free stream here: "If the flow being simulated is
      # unconfined then at the transverse boundaries we specify the free
      # stream velocity and density."
      faces = []
      for dim, location in ((0, 0), (1, 0), (1, 1)):
        faces.append(
            f'  boundary_info {{ dim: {dim} location: {location} '
            f'type: BC_TYPE_DIRICHLET value: {inflow[name]} }}'
        )
      faces.append(
          '  boundary_info { dim: 0 location: 1 '
          'type: BC_TYPE_NEUMANN value: 0.0 }'
      )
      body = '\n'.join(faces)
      return f'boundary_conditions {{\n  name: "{name}"\n{body}\n}}'

    pressure_faces = '\n'.join(
        f'  boundary_info {{ dim: {d} location: {l} '
        'type: BC_TYPE_NEUMANN value: 0.0 }'
        for d in (0, 1)
        for l in (0, 1)
    )
    pressure_bc = (
        'boundary_conditions {\n  name: "p"\n' + pressure_faces + '\n}'
    )
    variables = '\n'.join(
        f'      variables {{ name: "{name}" value: 0.0 bc: DIRICHLET }}'
        for name in VELOCITY
    )
    nx = self._grid_size(self.fx, case.cores_x, 'x')
    ny = self._grid_size(self.fy, case.cores_y, 'y')
    nz = self._grid_size(self.fz, case.cores_z, 'z')

    return f"""\
# Generated by swirl_lm.jax.examples.cylinder.Cylinder from a CylinderCase.
# Re = {case.reynolds_number:g}, so nu = u_inf * d / Re = {self.nu:.6g}.
solver_procedure: VARIABLE_DENSITY
convection_scheme: CONVECTION_SCHEME_QUICK
time_integration_scheme: TIME_SCHEME_CN_EXPLICIT_ITERATION
grid_params {{
  computation_shape {{
    dim_0: {case.cores_x} dim_1: {case.cores_y} dim_2: {case.cores_z}
  }}
  grid_size {{ dim_0: {nx} dim_1: {ny} dim_2: {nz} }}
  length {{ dim_0: {self.lx:g} dim_1: {self.ly:g} dim_2: {self.lz:g} }}
  halo_width: {case.halo_width}
  dt: {self.dt:.8g}
  kernel_size: {case.kernel_size}
  # The span is periodic, so a thin slab reproduces the two-dimensional flow
  # the published numbers describe. Streamwise and transverse are not.
  periodic {{ dim_0: false dim_1: false dim_2: true }}
}}
pressure {{
  solver {{
    jacobi {{
      max_iterations: {case.pressure_iterations}
      halo_width: {case.halo_width}
      omega: 0.67
    }}
  }}
  # Pins the pressure level at the outlet. Without it every non-periodic face
  # gets a homogeneous Neumann condition, the Poisson problem is singular, and
  # the unconverged residual acts as a spurious global pressure gradient. Only
  # correct because the lateral faces above are Dirichlet: while they were
  # zero gradient they classified as OUTFLOW, and this would have pinned p = 0
  # along both sides of the domain and driven flow through them.
  pressure_outlet: true
  # The density is constant, so `drho` is identically zero and filtering it is
  # pure waste -- at run time, and at compile time where XLA constant-folds
  # the stencil over the whole grid. Verified bit-identical. See MEMORY.md.
  num_d_rho_filter: 0
}}
thermodynamics {{ constant_density {{}} }}
density: 1.0
kinematic_viscosity: {self.nu:.8g}
num_sub_iterations: {case.num_sub_iterations}
use_sgs: false
# One at the wake probe cell and zero elsewhere, so the transverse velocity
# there can be reduced out of the sharded field with a sum.
additional_state_keys: "probe_mask"
enable_rhie_chow_correction: false
boundary_models {{
  ib {{
    ghost_point {{
{variables}
      geometry {{
        cylinder {{
          base {{ x: {self.center[0]:g} y: {self.center[1]:g} z: 0.0 }}
          axis {{ z: 1.0 }}
          radius: {0.5 * self.diameter:g}
        }}
      }}
      num_sweeps: {case.num_sweeps}
      write_interior_mask: true
      write_ghost_mask: true
      write_signed_distance: true
    }}
  }}
}}
{velocity_bc('u')}
{velocity_bc('v')}
{velocity_bc('w')}
{pressure_bc}
"""

  def _build_params(self) -> parameters_lib.SwirlLMParameters:
    """Expands the case into a full solver configuration."""
    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(self.config_text(), config)
    return parameters_lib.SwirlLMParameters(config)

  # -- Initialization -------------------------------------------------------

  def ghost_points(self, mesh) -> Any:
    """Builds the ghost-point geometry for `mesh`, once.

    Args:
      mesh: The device mesh.

    Returns:
      The `GhostPointGeometry` for this case.
    """
    if self._ghost_points is None:
      config = self.params.boundary_models.ib.ghost_point
      self._ghost_points = immersed_boundary_ghost_point.GhostPointGeometry(
          self.geometry_fn,
          self.gp,
          mesh,
          write_interior_mask=config.write_interior_mask,
          write_ghost_mask=config.write_ghost_mask,
          write_signed_distance=config.write_signed_distance,
      )
      self.params.additional_state_partition_specs.update(
          self._ghost_points.partition_specs()
      )
    return self._ghost_points

  def initialization(
      self,
      params: parameters_lib.SwirlLMParameters,
      logical_coordinates: tuple[int, int, int],
  ) -> dict[str, ScalarField]:
    """Uniform flow everywhere, plus the immersed boundary geometry.

    Starting from a uniform stream rather than from rest lets the wake
    develop from the impulsive start, which is the usual way these cases are
    run and reaches the periodic state sooner.

    Args:
      params: Simulation parameters.
      logical_coordinates: `(cx_i, cy_i, cz_i)` position of this device.

    Returns:
      Dict of device-local initial states.
    """
    grid = params.grid_params
    shape = (grid.nx, grid.ny, grid.nz)
    ones = jnp.ones(shape, jnp.float32)
    zeros = jnp.zeros(shape, jnp.float32)

    states = {
        'rho': ones,
        'u': self.u_inf * ones,
        'v': zeros,
        'w': zeros,
        'p': zeros,
    }
    geometry = self.ghost_points(_mesh_for(params))
    states.update(geometry.initial_states(logical_coordinates))
    states['probe_mask'] = self.probe_mask(params, logical_coordinates)
    # Start the interior of the body at rest rather than at the free-stream
    # speed, so the first step does not have to undo a uniform flow inside a
    # solid. The ghost cells are overwritten by the method immediately.
    if immersed_boundary_ghost_point.INTERIOR_MASK_KEY in states:
      fluid = states[immersed_boundary_ghost_point.INTERIOR_MASK_KEY]
      states['u'] = states['u'] * fluid
    return states

  def probe_mask(
      self,
      params: parameters_lib.SwirlLMParameters,
      logical_coordinates: tuple[int, int, int],
  ) -> ScalarField:
    """One at the wake probe cell on the replica that owns it, zero elsewhere.

    A mask rather than an index because the probe has to be read from inside
    `shard_map`, where each device holds a different subdomain and no device
    knows whether the probe is its own. Summing the masked field and reducing
    across devices sidesteps that entirely.

    Args:
      params: Simulation parameters.
      logical_coordinates: `(cx_i, cy_i, cz_i)` position of this device.

    Returns:
      A device-local 3D field.
    """
    grid = params.grid_params
    halo = grid.halo_width
    mask = np.zeros((grid.nx, grid.ny, grid.nz), np.float32)

    # Global interior index of the probe, then this replica's slice of it.
    global_index = self.probe_index()
    core = grid.to_data_axis_order(grid.core_nx, grid.core_ny, grid.core_nz)
    replica = grid.to_data_axis_order(*logical_coordinates)
    probe = grid.to_data_axis_order(*global_index)

    local = []
    for axis in range(3):
      offset = probe[axis] - replica[axis] * core[axis]
      if not 0 <= offset < core[axis]:
        return jnp.asarray(mask)      # the probe is on another replica
      local.append(offset + halo)
    mask[local[0], local[1], local[2]] = 1.0
    return jnp.asarray(mask)

  def probe_fn(self, mesh):
    """A diagnostic for `driver.run_simulation` sampling `v` at the probe.

    Args:
      mesh: The device mesh, for the cross-device reduction.

    Returns:
      A function of the state returning the transverse velocity at the probe.
    """
    axis_names = tuple(mesh.axis_names)

    def sample(state: ScalarFieldMap) -> jax.Array:
      # Exactly one cell of one replica is unmasked, so the sum over all of
      # them is that cell's value.
      return jax.lax.psum(
          jnp.sum(state['v'] * state['probe_mask']), axis_names
      )

    return sample

  # -- Diagnostics ----------------------------------------------------------

  def _interior(self, field: ScalarField) -> np.ndarray:
    """Strips halos and the duplicated device seams from a global field."""
    from swirl_lm.jax.utility import global_layout  # pylint: disable=g-import-not-at-top

    return global_layout.strip_halos_global(np.asarray(field), self.gp)

  def recirculation_length(self, state: ScalarFieldMap) -> float:
    """Length of the closed wake bubble, in diameters.

    Measured along the wake axis from the rear of the cylinder to the point
    where the streamwise velocity changes sign. Defined only while the wake
    is steady, which it is below about Re = 47; above that the bubble sheds
    and the number oscillates.

    Args:
      state: The final simulation state.

    Returns:
      `L/d`, or `0.0` if the flow never reverses, which means either that the
      Reynolds number is too low for a bubble or that the run has not
      developed.
    """
    u = self._interior(state['u'])
    axis = self.gp.get_axis_index('x')
    u = np.moveaxis(u, (axis, self.gp.get_axis_index('y')), (0, 1))
    j = u.shape[1] // 2                       # the wake axis
    line = u[:, j, u.shape[2] // 2]
    x = np.arange(line.size) * self.dx + 0.5 * self.dx

    rear = self.center[0] + 0.5 * self.diameter
    downstream = x > rear
    reversed_flow = np.where(downstream & (line < 0.0))[0]
    if reversed_flow.size == 0:
      return 0.0
    # The bubble closes at the first reattachment past the reversed region.
    after = np.where(downstream & (line > 0.0) & (x > x[reversed_flow[0]]))[0]
    if after.size == 0:
      return 0.0
    return float((x[after[0]] - rear) / self.diameter)

  def probe_index(self) -> tuple[int, int, int]:
    """Grid index of the wake probe used for the shedding frequency."""
    x = self.center[0] + self.case.probe_downstream * self.diameter
    y = self.center[1] + self.case.probe_offset * self.diameter
    return (
        int(round(x / self.dx - 0.5)),
        int(round(y / (self.ly / self.fy) - 0.5)),
        self.fz // 2,
    )

  def strouhal_number(self, probe_signal, dt: float | None = None) -> float:
    """Shedding frequency from a transverse-velocity record, non-dimensional.

    Args:
      probe_signal: Transverse velocity sampled once per step at the wake
        probe. Use the second half of a run, after the transient.
      dt: The sampling interval. Defaults to the case time step.

    Returns:
      `St = f * d / u_inf`, or `0.0` if the record shows no clear peak.
    """
    signal = np.asarray(probe_signal, np.float64)
    if signal.size < 16:
      return 0.0
    signal = signal - signal.mean()
    if not np.any(np.abs(signal) > 0.0):
      return 0.0
    step = dt if dt is not None else self.dt
    spectrum = np.abs(np.fft.rfft(signal))
    # Skip the zero-frequency bin, which carries any residual drift.
    peak = int(np.argmax(spectrum[1:])) + 1

    # Interpolate the peak parabolically across its two neighbours. Without
    # this the answer is quantised to the bin width, and the bin width is set
    # by how long the record is: a 32-period record gives bins 0.005 wide in
    # Strouhal number, which is the same size as the difference between the
    # published values being compared against. The fit costs nothing and
    # removes that quantisation.
    offset = 0.0
    if 0 < peak < spectrum.size - 1:
      left, middle, right = spectrum[peak - 1], spectrum[peak], spectrum[peak + 1]
      denominator = left - 2.0 * middle + right
      if abs(denominator) > 1e-30:
        offset = 0.5 * (left - right) / denominator
        # A well-formed peak sits within half a bin of its sample; anything
        # further means the spectrum is not peaked and the fit is meaningless.
        offset = float(np.clip(offset, -0.5, 0.5))

    frequency = (peak + offset) / (signal.size * step)
    return float(frequency * self.diameter / self.u_inf)

  def summary(self) -> str:
    """A one-line description of the resolved case, for logging."""
    return (
        f'cylinder: Re = {self.case.reynolds_number:g}, d = '
        f'{self.diameter:g} m, u_inf = {self.u_inf:g} m/s, nu = {self.nu:.4g} '
        f'm^2/s; domain {self.lx:g} x {self.ly:g} x {self.lz:g} m on '
        f'{self.fx} x {self.fy} x {self.fz} points '
        f'({self.case.cells_per_diameter}/d); dt = {self.dt:.4g} s on '
        f'{self.case.cores_x}x{self.case.cores_y}x{self.case.cores_z} cores'
    )


def _mesh_for(params: parameters_lib.SwirlLMParameters):
  """The device mesh for `params`, imported lazily to avoid a cycle."""
  from swirl_lm.jax.core import driver as driver_lib  # pylint: disable=g-import-not-at-top

  return driver_lib.create_mesh(params)
