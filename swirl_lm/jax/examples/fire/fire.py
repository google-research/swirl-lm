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
"""A simulation of a grassland fire over terrain (JAX port).

This mirrors the TensorFlow `swirl_lm.example.fire.fire.Fire` class: the
simulation owns its models (thermodynamics, combustion, inflow, sponge,
immersed boundary) and exposes the four hooks the driver calls, so that
nothing depends on module-level state.

  ```python
  simulation = fire.Fire(params)
  params.source_update_fn_lib = simulation.source_term_update_fn()
  params.additional_states_update_fn = simulation.additional_states_update_fn
  params.preprocessing_states_update_fn = simulation.pre_simulation_update_fn

  state = driver.run_simulation(
      params=params,
      init_fn=simulation.initialization,
      num_steps=200,
  )
  ```

Note on source terms: `additional_states_update_fn` *accumulates* the
immersed-boundary and sponge forcings onto `src_*`, exactly as the TF version
does. That is only correct because the driver clears those source terms at the
start of every step, before this function is called; see
`driver._update_additional_states`.
"""

import collections
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.boundary_condition import immersed_boundary_method
from swirl_lm.jax.boundary_condition import rayleigh_damping_layer
from swirl_lm.jax.boundary_condition import synthetic_turbulent_inflow
from swirl_lm.jax.physics.combustion import turbulent_kinetic_energy
from swirl_lm.jax.physics.combustion import wood
from swirl_lm.jax.physics.thermodynamics import manager as thermodynamics_manager
from swirl_lm.jax.utility import types


ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

# The threshold below which the fuel is treated as depleted.
EPSILON = 1e-6

# Names of the velocity components.
VELOCITY = ('u', 'v', 'w')

# A line-shaped ignition kernel, at an angle to the x axis.
SlantLine = collections.namedtuple(
    'SlantLine', ['center_x', 'center_y', 'length', 'thickness', 'angle']
)


class Fire:
  """A library for the simulation of a fire spreading over terrain."""

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
      fuel_bed_height: float = 4.0,
      fuel_load: float = 0.6,
      moisture_content: float = 0.142,
      c_d: float = 0.01,
      a_v: float = 8.0,
      ignition_line: Optional[SlantLine] = None,
      ignition_temperature: float = 600.0,
      u_rms: float = 2.0,
      v_rms: float = 2.0,
      w_rms: float = 2.0,
      z_0: float = 0.15,
      terrain_type: str = 'gaussian',
      inflow_dim: int = 0,
      inflow_face: int = 0,
      inflow_seed: int = 42,
      inflow_length_scale_cells: float = 10.0,
  ):
    """Initializes the simulation setup and all of its models.

    Args:
      params: Simulation parameters.
      fuel_bed_height: Height of the fuel bed above the terrain (m). A value of
        zero disables the fuel and, with it, the fire.
      fuel_load: Fuel load (kg/m^2).
      moisture_content: Fuel moisture content, as a fraction of the fuel
        density.
      c_d: Drag coefficient of the vegetation.
      a_v: Frontal area per unit volume of the vegetation (1/m).
      ignition_line: Geometry of the line ignition kernel. Defaults to a line
        across the domain at x = 20 m.
      ignition_temperature: Temperature of the ignition kernel (K).
      u_rms: Root mean square of the inflow velocity fluctuation in x (m/s).
      v_rms: Root mean square of the inflow velocity fluctuation in y (m/s).
      w_rms: Root mean square of the inflow velocity fluctuation in z (m/s).
      z_0: Aerodynamic roughness length of the surface (m).
      terrain_type: Shape of the terrain represented by the immersed boundary.
      inflow_dim: Dimension along which the turbulent inflow is injected.
      inflow_face: Face of `inflow_dim` where the inflow is injected.
      inflow_seed: Seed for the random fields of the synthetic inflow.
      inflow_length_scale_cells: Turbulence integral length scale of the
        synthetic inflow, in grid cells. This sets the digital filter's width,
        and the inflow requires at least twice this many cells per core along
        each in-plane direction, so it has to be reduced on small or heavily
        decomposed grids.
    """
    self.config = params
    self.gp = params.grid_params

    # Fire and fuel properties.
    self.fuel_bed_height = fuel_bed_height
    self.fuel_load = fuel_load
    self.moisture_content = moisture_content
    self.fuel_density = fuel_load / 1.5
    self.moisture_density = moisture_content * self.fuel_density
    self.c_d = c_d
    self.a_v = a_v

    # Ignition.
    self.ignition_line = ignition_line or SlantLine(
        center_x=20.0,
        center_y=100.0,
        length=60.0,
        thickness=8.0,
        angle=90.0,
    )
    self.ignition_temperature = ignition_temperature

    # Flow field properties. The inflow and initial conditions are taken from
    # the boundary conditions in the config, matching the TF version's
    # `fire_utils`.
    self.u_rms = u_rms
    self.v_rms = v_rms
    self.w_rms = w_rms
    self.z_0 = z_0
    self.u_init = self.config.bc['u'][0][0][1]
    self.v_init = self.config.bc['v'][0][0][1]
    self.t_init = self.config.bc['theta'][0][0][1]
    self.y_o_init = self.config.bc['Y_O'][0][0][1]

    self.terrain_type = terrain_type

    # Thermodynamics.
    self.thermodynamics = thermodynamics_manager.ThermodynamicsManager(
        self.config
    )

    # Combustion. Without a fuel bed there is nothing to burn.
    self.include_fuel = self.fuel_bed_height > 0.0
    self.include_fire = self.include_fuel and self.config.combustion is not None
    self.combustion = (
        wood.wood_combustion_factory(self.config) if self.include_fire else None
    )

    # Synthetic turbulent inflow.
    cells = inflow_length_scale_cells
    length_scales = (
        cells * self.gp.dx,
        cells * self.gp.dy,
        cells * self.gp.dz,
    )
    delta = (self.gp.dx, self.gp.dy, self.gp.dz)
    mesh_size = (self.gp.core_nx, self.gp.core_ny, self.gp.core_nz)
    self.inflow_dim = inflow_dim
    self.inflow_face = inflow_face
    self.inflow_seed = inflow_seed
    self.inflow = synthetic_turbulent_inflow.SyntheticTurbulentInflow(
        length_scales, delta, mesh_size, inflow_dim, inflow_face
    )
    # A real key is required: without one the per-step noise slice is
    # drawn from a constant `jax.random.PRNGKey(0)`, so after ~2*n_pad
    # steps the inflow turbulence is frozen in time. The update function
    # folds `step_id` into this key on every call.
    self.inflow_update_fn = self.inflow.generate_inflow_update_fn(
        key=jax.random.PRNGKey(inflow_seed)
    )
    # The `mean_*` and `rms_*` planes are built per replica and differ between
    # devices covering different parts of the inflow plane, so the driver has
    # to be told how they map onto the mesh. Without this the driver cannot
    # distribute them and raises.
    self.config.additional_state_partition_specs.update(
        self.inflow.partition_specs()
    )

    # The Rayleigh damping layer.
    self.sponge = (
        rayleigh_damping_layer.RayleighDampingLayer(self.config.sponge)
        if self.config.sponge
        else None
    )

    # The immersed boundary method for the terrain representation.
    self.ib = (
        immersed_boundary_method.ImmersedBoundaryMethod(self.config)
        if self.config.boundary_models is not None
        and self.config.boundary_models.HasField('ib')
        else None
    )

  # -- Grid helpers ---------------------------------------------------------

  def _partial_mesh_for_core(
      self,
      params: parameters_lib.SwirlLMParameters,
      logical_coordinates: tuple[int, int, int],
      value_fn: Callable[..., ScalarField],
      pad_mode: str | None = 'SYMMETRIC',
  ) -> ScalarField:
    """Creates a 3D field on one device by evaluating `value_fn` on a meshgrid.

    The meshgrid covers the core region of the device (without halos). The
    result is padded with `pad_mode` to include halo layers. This replaces TF's
    `initializer.partial_mesh_for_core`.

    Args:
      params: Simulation parameters.
      logical_coordinates: (cx_i, cy_i, cz_i) position of this device.
      value_fn: Function(xx, yy, zz, lx, ly, lz) -> 3D array on the core.
      pad_mode: Padding mode for halos. 'CONSTANT': Pad with zeros.
        'SYMMETRIC': Mirror-reflect at edges. 'REFLECT': Reflect without
        repeating edge. 'PHYSICAL': Use physically correct coordinate values in
        halos from `global_xyz_with_halos`. This is essential for coordinate
        fields like `zz` where halo values must represent real physical
        positions (e.g. for hydrostatic pressure computation at device
        boundaries).

    Returns:
      A 3D array of shape matching data_axis_order with halos.

    Raises:
      ValueError: If `pad_mode` is not recognized.
    """
    gp = params.grid_params
    hw = gp.halo_width

    # Core dimensions per device.
    core_nx = gp.core_nx
    core_ny = gp.core_ny
    core_nz = gp.core_nz

    # Global lengths.
    lx = gp.lx
    ly = gp.ly
    lz = gp.lz

    # This device's position in the computational grid.
    gx, gy, gz = logical_coordinates

    if pad_mode is not None and pad_mode.upper() == 'PHYSICAL':
      # Use global coordinate arrays that include physically correct halo
      # values. `global_xyz_with_halos` is in data_axis_order, so map device
      # coordinates accordingly.
      core_ns = gp.to_data_axis_order(core_nx, core_ny, core_nz)
      ns = gp.to_data_axis_order(gp.nx, gp.ny, gp.nz)
      device_ids = gp.to_data_axis_order(gx, gy, gz)

      slices_1d = []
      for dim in range(3):
        coord = gp.global_xyz_with_halos[dim]
        start = device_ids[dim] * core_ns[dim]
        end = start + ns[dim]
        slices_1d.append(jnp.array(coord[start:end]))

      # Build meshgrid in data_axis_order directly.
      grids = jnp.meshgrid(*slices_1d, indexing='ij')

      # Convert data_axis_order grids back to xyz for value_fn.
      xx, yy, zz = gp.to_xyz_order(grids)

      # The result is already in data_axis_order (from the meshgrid).
      return value_fn(xx, yy, zz, lx, ly, lz)

    # Build 1D coordinate arrays for the core region of this device.
    dx = lx / (gp.cx * core_nx)
    dy = ly / (gp.cy * core_ny)
    dz = lz / (gp.cz * core_nz)

    x_start = gx * core_nx * dx
    y_start = gy * core_ny * dy
    z_start = gz * core_nz * dz

    x_1d = jnp.linspace(x_start, x_start + (core_nx - 1) * dx, core_nx)
    y_1d = jnp.linspace(y_start, y_start + (core_ny - 1) * dy, core_ny)
    z_1d = jnp.linspace(z_start, z_start + (core_nz - 1) * dz, core_nz)

    xx, yy, zz = jnp.meshgrid(x_1d, y_1d, z_1d, indexing='ij')

    # Evaluate the value function on the core mesh.
    core_values = value_fn(xx, yy, zz, lx, ly, lz)

    # Pad with halos.
    if pad_mode is None or hw == 0:
      padded = core_values
    elif pad_mode.upper() == 'CONSTANT':
      padded = jnp.pad(core_values, hw, mode='constant', constant_values=0.0)
    elif pad_mode.upper() == 'SYMMETRIC':
      padded = jnp.pad(core_values, hw, mode='symmetric')
    elif pad_mode.upper() == 'REFLECT':
      padded = jnp.pad(core_values, hw, mode='reflect')
    else:
      raise ValueError(f'Unknown pad_mode: {pad_mode}')

    # Transpose from xyz indexing to data_axis_order. `meshgrid` with
    # indexing='ij' produces (x, y, z) order.
    return jnp.transpose(padded, gp.get_axis_index(('x', 'y', 'z')))

  def _constant_field(
      self,
      params: parameters_lib.SwirlLMParameters,
      logical_coordinates: tuple[int, int, int],
      value: float,
  ) -> ScalarField:
    """Creates a constant 3D field on one device."""
    return self._partial_mesh_for_core(
        params,
        logical_coordinates,
        lambda xx, yy, zz, lx, ly, lz: value * jnp.ones_like(xx),
    )

  def _logarithmic_boundary_layer_init(
      self,
      params: parameters_lib.SwirlLMParameters,
      logical_coordinates: tuple[int, int, int],
  ) -> dict[str, ScalarField]:
    """Initializes velocity with a logarithmic boundary layer profile."""
    lz = params.grid_params.lz

    def log_profile(u_inf):
      def value_fn(xx, yy, zz, lx, ly, lz_):
        del xx, yy, lx, ly, lz_
        z_safe = jnp.maximum(zz, self.z_0)
        return u_inf * jnp.log(z_safe / self.z_0) / jnp.log(lz / self.z_0)

      return value_fn

    return {
        'u': self._partial_mesh_for_core(
            params, logical_coordinates, log_profile(self.u_init)
        ),
        'v': self._partial_mesh_for_core(
            params, logical_coordinates, log_profile(self.v_init)
        ),
        'w': self._partial_mesh_for_core(
            params,
            logical_coordinates,
            lambda xx, yy, zz, lx, ly, lz_: jnp.zeros_like(xx),
        ),
    }

  def _get_terrain_init_fn(self):
    """Generates functions that initialize the IB to represent the terrain.

    Returns:
      A tuple of (terrain height, flow field mask, IB boundary mask) functions.

    Raises:
      ValueError: If `self.terrain_type` is not recognized.
    """

    def gaussian(xx, yy, zz, lx, ly, lz):
      """Generates a bump with the Gaussian profile."""
      del yy, zz, ly

      h_ratio = 0.1
      w_ratio = 0.1

      return (
          h_ratio
          * lz
          * jnp.exp(-((xx - 0.5 * lx) ** 2) / (2.0 * (w_ratio * lx) ** 2))
      )

    if self.terrain_type == 'gaussian':
      terrain_height_fn = gaussian
    else:
      raise ValueError(
          f'Unrecognized terrain type {self.terrain_type}. Available options: '
          '"gaussian".'
      )

    def flow_field_mask_fn(xx, yy, zz, lx, ly, lz):
      """Sets values to 1 in the flow field and 0 in the solid."""
      h = terrain_height_fn(xx, yy, zz, lx, ly, lz)
      return jnp.where(zz > h, jnp.ones_like(zz), jnp.zeros_like(zz))

    def ib_boundary_mask_fn(xx, yy, zz, lx, ly, lz):
      """Sets values to 1 at the fluid-solid interface and 0 elsewhere."""
      h = terrain_height_fn(xx, yy, zz, lx, ly, lz)
      dz = lz / (self.gp.cz * self.gp.core_nz)
      return jnp.where(
          jnp.logical_and(zz <= h, zz > h - dz),
          jnp.ones_like(zz),
          jnp.zeros_like(zz),
      )

    return terrain_height_fn, flow_field_mask_fn, ib_boundary_mask_fn

  # -- Initialization -------------------------------------------------------

  def initialization(
      self,
      params: parameters_lib.SwirlLMParameters,
      logical_coordinates: tuple[int, int, int],
  ) -> dict[str, ScalarField]:
    """Initializes the states of a grassland fire simulation.

    This conforms to the JAX driver's `InitFn` signature.

    Args:
      params: Simulation parameters.
      logical_coordinates: `(cx_i, cy_i, cz_i)` position of this device.

    Returns:
      Dict of device-local initial states.
    """
    gp = params.grid_params
    hw = gp.halo_width

    # Axis indices for data_axis_order-aware slicing.
    ax_x = gp.get_axis_index('x')
    ax_y = gp.get_axis_index('y')
    ax_z = gp.get_axis_index('z')

    def pmfc(value_fn, pad_mode='SYMMETRIC'):
      return self._partial_mesh_for_core(
          params, logical_coordinates, value_fn, pad_mode
      )

    def constant(value):
      return self._constant_field(params, logical_coordinates, value)

    # Initializes the terrain if the IB is used.
    if self.ib is not None:
      terrain_height_fn, flow_field_mask_fn, ib_boundary_mask_fn = (
          self._get_terrain_init_fn()
      )
    else:
      terrain_height_fn = lambda xx, yy, zz, lx, ly, lz: jnp.zeros_like(zz)

    def rho_f_init_fn(xx, yy, zz, lx, ly, lz):
      """Generates the initial `rho_f` field."""
      h = terrain_height_fn(xx, yy, zz, lx, ly, lz)
      return jnp.where(
          jnp.logical_and(zz <= h + self.fuel_bed_height, zz >= h),
          self.fuel_density * jnp.ones_like(zz),
          jnp.zeros_like(zz),
      )

    def rho_m_init_fn(xx, yy, zz, lx, ly, lz):
      """Generates the initial moisture `rho_m` field."""
      h = terrain_height_fn(xx, yy, zz, lx, ly, lz)
      return jnp.where(
          jnp.logical_and(zz <= h + self.fuel_bed_height, zz >= h),
          self.moisture_density * jnp.ones_like(zz),
          jnp.zeros_like(zz),
      )

    def ignition_kernel_init_fn(xx, yy, zz, lx, ly, lz):
      """Generates a line ignition kernel."""
      h = terrain_height_fn(xx, yy, zz, lx, ly, lz)

      rad = np.radians(self.ignition_line.angle)
      cos = np.cos(rad)
      sin = np.sin(rad)

      x_new = cos * (xx - self.ignition_line.center_x) + sin * (
          yy - self.ignition_line.center_y
      )
      y_new = -sin * (xx - self.ignition_line.center_x) + cos * (
          yy - self.ignition_line.center_y
      )

      location_x = jnp.logical_and(
          x_new <= self.ignition_line.length / 2.0,
          x_new >= -self.ignition_line.length / 2.0,
      )
      location_y = jnp.logical_and(
          y_new <= self.ignition_line.thickness / 2.0,
          y_new >= -self.ignition_line.thickness / 2.0,
      )
      location_xy = jnp.logical_and(location_x, location_y)

      location_z = jnp.logical_and(zz <= h + self.fuel_bed_height, zz >= h)
      location = jnp.logical_and(location_xy, location_z)

      return jnp.where(location, jnp.ones_like(zz), jnp.zeros_like(zz))

    output = {}

    # Initializes the terrain if the IB is used.
    if self.ib is not None:
      ib_interior_mask = pmfc(flow_field_mask_fn)
      ib_boundary_mask = pmfc(ib_boundary_mask_fn)
      output.update(
          self.ib.generate_initial_states(
              ib_interior_mask.shape, ib_interior_mask, ib_boundary_mask
          )
      )
      # Only some IB variants emit `ib_boundary` themselves -- a `sponge` IB
      # does so only if one of its variables uses a NEUMANN_Z condition -- so
      # supply it here when the config declares it, for the benefit of other
      # modules. This mirrors the TF version.
      if (
          'ib_boundary' in self.config.additional_state_keys
          and 'ib_boundary' not in output
      ):
        output['ib_boundary'] = ib_boundary_mask

    # Initializes the flow field states.
    uvw = self._logarithmic_boundary_layer_init(params, logical_coordinates)
    output.update({
        'rho': constant(self.config.rho),
        'u': uvw['u'],
        'v': uvw['v'],
        'w': uvw['w'],
        'p': constant(0.0),
        'theta': constant(self.t_init),
        'Y_O': constant(self.y_o_init),
        # Turbulent viscosity for the SGS model.
        'nu_t': constant(0.0),
        # Geopotential height. Uses 'PHYSICAL' pad mode so that halo values
        # contain correct physical z-coordinates (essential for hydrostatic
        # pressure computation at device boundaries).
        'zz': pmfc(lambda xx, yy, zz, lx, ly, lz: zz, 'PHYSICAL'),
    })

    # Variables required by the fire model.
    if self.include_fuel:
      output.update({
          'rho_f': pmfc(rho_f_init_fn, 'CONSTANT'),
          'rho_f_init': pmfc(rho_f_init_fn, 'CONSTANT'),
          'rho_m': pmfc(rho_m_init_fn, 'CONSTANT'),
          'phi_w': constant(0.0),
      })

    if self.include_fire:
      output.update({
          'T_s': constant(self.t_init),
          'src_rho': constant(0.0),
          'src_theta': constant(0.0),
          'src_Y_O': constant(0.0),
          'tke': constant(0.0),
          'ignition_kernel': pmfc(ignition_kernel_init_fn),
      })

    if 'ib_interior_mask' in output:
      output.update({
          vel: output[vel] * output['ib_interior_mask'] for vel in VELOCITY
      })
      if 'ignition_kernel' in output:
        output['ignition_kernel'] = (
            output['ignition_kernel'] * output['ib_interior_mask']
        )

    # Generates variables required by the synthetic turbulent inflow. The
    # replica's position enters the seed so that neighbouring subdomains get
    # independent turbulence; with a single seed every device would draw the
    # identical field, which is not the intent of a synthetic inflow.
    replica_seed = self.inflow_seed + int(
        np.ravel_multi_index(
            logical_coordinates, (max(gp.cx, 1), max(gp.cy, 1), max(gp.cz, 1))
        )
    )
    r = self.inflow.generate_random_fields(jax.random.PRNGKey(replica_seed))

    def cut_yz_plane(f):
      """Retrieves the y-z plane at the inflow face from `f` without halos.

      The inflow generator expects 2D planes of shape (m[0], m[1]) where
      m = mesh_size[inflow_plane] = (core_ny, core_nz) for inflow_dim=0.
      With data_axis_order=('x','y','z'), squeezing the x-dim from a 3D field
      gives (core_ny, core_nz) directly -- no transpose needed. TF's
      `cut_yz_plane` transposed because data_axis_order=('z','x','y') produced
      (core_nz, core_ny) after squeezing x.

      Args:
        f: A 3D field with halos.

      Returns:
        The 2D y-z plane at the inflow face.
      """
      start = [0, 0, 0]
      start[ax_x] = hw
      start[ax_y] = hw
      start[ax_z] = hw
      sizes = list(f.shape)
      sizes[ax_x] = 1
      sizes[ax_y] = gp.core_ny
      sizes[ax_z] = gp.core_nz
      return jnp.squeeze(jax.lax.dynamic_slice(f, start, sizes))

    def slice_inflow_bc(f):
      """Slices the first `hw + 1` planes along the inflow axis."""
      slices = [slice(None)] * 3
      slices[ax_x] = slice(None, hw + 1)
      return f[tuple(slices)]

    mask = cut_yz_plane(jnp.ones_like(output['u']))
    output.update({
        'bc_u_0_0': slice_inflow_bc(output['u']),
        'bc_v_0_0': slice_inflow_bc(output['v']),
        'bc_w_0_0': slice_inflow_bc(output['w']),
        'mean_u_0_0': cut_yz_plane(output['u']),
        'mean_v_0_0': cut_yz_plane(output['v']),
        'mean_w_0_0': cut_yz_plane(output['w']),
        'rms_u_0_0': self.u_rms * mask,
        'rms_v_0_0': self.v_rms * mask,
        'rms_w_0_0': self.w_rms * mask,
        'rand_u_0_0': r[0],
        'rand_v_0_0': r[1],
        'rand_w_0_0': r[2],
    })

    # Generates variables required by the Rayleigh damping layer.
    if self.sponge is not None:
      sponge_info = next(iter(self.config.sponge))
      sponge_init_fn = rayleigh_damping_layer.klemp_lilly_relaxation_coeff_fn(
          list(sponge_info.orientation), x0=0.0, y0=0.0, z0=0.0
      )
      output.update({'sponge_beta': pmfc(sponge_init_fn)})

      for variable in sponge_info.variable_info:
        # Add the target state if requested.
        if variable.WhichOneof('target') == 'target_state_name':
          output.update({f'{variable.name}_init': output[variable.name]})

        # The source term from Rayleigh damping for the velocity is computed by
        # an externally defined source term function.
        if variable.name in VELOCITY:
          continue

        output.update({f'src_{variable.name}': constant(0.0)})

    # Source terms for the velocity.
    for vel in VELOCITY:
      output.setdefault(f'src_{vel}', constant(0.0))

    return output

  # -- Driver hooks ---------------------------------------------------------

  def _ignite(
      self,
      ignition_kernel: ScalarField,
      temperature: ScalarField,
  ) -> ScalarField:
    """Sets a high temperature region at the specified location."""
    return (
        temperature
        + (self.ignition_temperature - temperature) * ignition_kernel
    )

  def pre_simulation_update_fn(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> dict[str, ScalarField]:
    """Ignites the fire.

    This is invoked once, at the step specified by `preprocess_step_id`.

    Args:
      states: Essential (prognostic) states.
      additional_states: Additional helper states.

    Returns:
      A dict of updated states, with an elevated temperature where the
      ignition kernel overlaps the fuel, a density consistent with that
      temperature, and a velocity rescaled to conserve momentum.
    """
    output = dict(states)
    output.update(additional_states)

    if not self.include_fire:
      return output

    temperature = self._ignite(
        additional_states['ignition_kernel'], states['theta']
    )
    t_s = self._ignite(
        additional_states['ignition_kernel'], additional_states['T_s']
    )

    output.update({
        'theta': jnp.where(
            additional_states['rho_f'] > 0.0, temperature, states['theta']
        ),
        'T_s': jnp.where(
            additional_states['rho_f'] > 0.0, t_s, additional_states['T_s']
        ),
    })

    rho = self.thermodynamics.update_thermal_density(
        {'theta': output['theta'], 'Y_O': states['Y_O']}, {}
    )
    output.update({'rho': rho})

    # Rescale the velocity so that momentum stays the same as before ignition.
    output.update(
        {vel: states['rho'] * states[vel] / rho for vel in VELOCITY}
    )

    return output

  def additional_states_update_fn(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      step_id: Any = None,
  ) -> dict[str, ScalarField]:
    """Updates `additional_states` for the fire physics and the forcing terms.

    The immersed-boundary and sponge forcings are *added* to `src_*`, which
    relies on the driver having cleared those source terms at the start of the
    step; see `driver._update_additional_states`. Without that, the forcings
    would accumulate from step to step without bound.

    Args:
      states: Essential (prognostic) states.
      additional_states: Additional helper states, with the source terms
        already cleared for this step.
      step_id: The current step index. Folded into the synthetic inflow's
        random key so that each step draws fresh turbulence.

    Returns:
      The updated `additional_states`.
    """
    states_updated = dict(states)
    updated = dict(additional_states)

    if self.include_fire:
      # Update the turbulent kinetic energy for the combustion source term.
      tke_update_fn = turbulent_kinetic_energy.tke_update_fn_manager(
          self.config.combustion.wood.tke
      )
      updated.update(
          tke_update_fn(
              states=states_updated,
              additional_states=updated,
              dx=self.gp.dx,
              dy=self.gp.dy,
              dz=self.gp.dz,
              halo_width=self.gp.halo_width,
          )
      )

      # Update the solid states and the reaction source terms for the
      # potential temperature and the oxygen mass fraction.
      updated.update(
          self.combustion.dry_wood_update_fn(
              rho_f_init=additional_states.get('rho_f_init')
          )(
              replica_id=jnp.int32(0),
              replicas=np.array([[[0]]]),
              states=states_updated,
              additional_states=updated,
          )
      )

    # Update the inflow.
    updated.update(
        self.inflow_update_fn(
            states=states_updated,
            additional_states=updated,
            params=self.config,
            step_id=step_id,
        )
    )

    # Add the forcing term from the immersed boundary method.
    if self.ib is not None:
      ib_vars = ('u', 'v', 'w', 'theta', 'Y_O')
      ib_additional_states = {
          f'src_{var}': jnp.zeros_like(states[var]) for var in ib_vars
      }
      ib_additional_states.update({
          helper: additional_states[helper]
          for helper in ('ib_interior_mask', 'ib_boundary')
          if helper in additional_states
      })
      ib_additional_states = self.ib.update_additional_states(
          states=states,
          additional_states=ib_additional_states,
      )
      for var in ib_vars:
        src = f'src_{var}'
        if src in updated:
          updated[src] = updated[src] + ib_additional_states[src]

    # Add the Rayleigh damping forcing terms.
    if self.sponge is not None:
      sponge_scalars = ('theta', 'Y_O')
      sponge_states = {
          var: states_updated[var]
          for var in sponge_scalars
          if var in states_updated
      }
      sponge_additional_states = {
          'sponge_beta': additional_states['sponge_beta'],
      }
      sponge_additional_states.update({
          f'src_{var}': jnp.zeros_like(val)
          for var, val in sponge_states.items()
      })
      for var in sponge_states:
        target = f'{var}_init'
        if target in additional_states:
          sponge_additional_states[target] = additional_states[target]

      sponge_additional_states = self.sponge.additional_states_update_fn(
          states=sponge_states,
          additional_states=sponge_additional_states,
          dt=self.config.dt,
      )
      for var in sponge_states:
        src = f'src_{var}'
        if src in updated:
          updated[src] = updated[src] + sponge_additional_states[src]

    return updated

  # -- Source terms ---------------------------------------------------------

  def _vegetation_drag_update_fn(self):
    """Generates an update function for the drag forces due to vegetation."""

    def update_fn(states, additional_states):
      """Updates 'src_u', 'src_v', 'src_w'."""
      u_mag = jnp.sqrt(states['u'] ** 2 + states['v'] ** 2 + states['w'] ** 2)

      def drag_force(u):
        """Computes the drag force for the specified velocity."""
        drag = -3.0 * self.c_d * states['rho'] * self.a_v * u_mag * u / 8.0
        return jnp.where(
            additional_states['rho_f'] < EPSILON, jnp.zeros_like(drag), drag
        )

      updated = dict(additional_states)
      for varname in VELOCITY:
        updated[f'src_{varname}'] = drag_force(states[varname])
      return updated

    return update_fn

  def source_term_update_fn(self):
    """Generates a library of functions that update the requested source terms.

    Returns:
      A dict keyed by variable name, whose values are functions of
      `(states, additional_states)` returning the corresponding `src_*` term.
    """

    def src_uvw_fn(states, additional_states):
      """Updates the source terms for u, v, and w."""
      src = {f'src_{var}': jnp.zeros_like(states[var]) for var in VELOCITY}

      # The canopy drag force.
      if self.include_fuel:
        helper_states = {'rho_f': additional_states['rho_f']}
        helper_states.update({f'src_{var}': None for var in VELOCITY})
        drag = self._vegetation_drag_update_fn()(states, helper_states)
        src.update({key: src[key] + drag[key] for key in src})

      # The Rayleigh damping force.
      if self.sponge is not None:
        sponge_states = {var: states[var] for var in VELOCITY}
        sponge_states['rho'] = states['rho']
        helper_states = {'sponge_beta': additional_states['sponge_beta']}
        helper_states.update(
            {f'src_{var}': jnp.zeros_like(states[var]) for var in VELOCITY}
        )
        helper_states.update({
            f'{var}_init': additional_states[f'{var}_init']
            for var in VELOCITY
            if f'{var}_init' in additional_states
        })
        sponge_force = self.sponge.additional_states_update_fn(
            states=sponge_states,
            additional_states=helper_states,
            dt=self.config.dt,
        )
        src.update({
            key: src[key] + states['rho'] * sponge_force[key] for key in src
        })

      return src

    def src_fn(varname):
      def fn(states, additional_states):
        return {
            f'src_{varname}': src_uvw_fn(states, additional_states)[
                f'src_{varname}'
            ]
        }

      return fn

    return {var: src_fn(var) for var in VELOCITY}
