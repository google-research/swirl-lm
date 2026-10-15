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
"""Integration test: Taylor-Green Vortex on the JAX solver.

This test verifies the full simulation pipeline end-to-end:
  - Config parsing from textproto.
  - SwirlLMParameters construction.
  - TGV initialization.
  - Predictor-corrector time stepping (simulation.step).
  - Basic physical validation: kinetic energy decays monotonically.

The simulation step must run inside `shard_map` because the halo exchange
uses `jax.lax.axis_index`, which requires a named-axis context.
The step function is JIT-compiled for performance.
"""


import logging as pylog
import os
import tempfile
import unittest

from absl.testing import absltest
from google.protobuf import text_format
import jax
from jax.experimental import shard_map
import jax.experimental.mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
import numpy as np

from swirl_lm.base import parameters_pb2
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.boundary_condition import boundary_condition_utils
from swirl_lm.jax.core import driver as driver_lib
from swirl_lm.jax.core import simulation as simulation_lib
from swirl_lm.jax.equations import common
from swirl_lm.jax.io import visualization as visualization_lib
from swirl_lm.jax.physics.turbulence import sgs_model as sgs_model_lib
from swirl_lm.jax.utility import global_layout
from swirl_lm.jax.utility import stretched_grid
from swirl_lm.jax.utility import stretched_grid_util

jax.config.update('jax_enable_x64', False)

# TGV config textproto — minimal config sized for fast testing.
# Uses KERNEL_OP_SLICE to avoid kernel_size divisibility constraints, and
# a small 8x8x8 grid with few Jacobi iterations for speed.
_TGV_CONFIG = """
solver_procedure: VARIABLE_DENSITY
convection_scheme: CONVECTION_SCHEME_QUICK
time_integration_scheme: TIME_SCHEME_CN_EXPLICIT_ITERATION
grid_params {
  computation_shape { dim_0: 1 dim_1: 1 dim_2: 1 }
  grid_size { dim_0: 8 dim_1: 8 dim_2: 8 }
  length { dim_0: 6.2831853 dim_1: 6.2831853 dim_2: 6.2831853 }
  halo_width: 2
  dt: 1e-3
  kernel_size: 12
  periodic { dim_0: true dim_1: true dim_2: true }
}
pressure {
  solver {
    jacobi {
      max_iterations: 5 halo_width: 2 omega: 0.67
    }
  }
}
thermodynamics {
  constant_density {}
}
density: 1.0
kinematic_viscosity: 6.25e-4
num_sub_iterations: 3
use_sgs: false
enable_rhie_chow_correction: false
"""

_U_MAG = 1.0
_RHO = 1.0

_STATE_KEYS = (
    common.KEY_RHO,
    common.KEY_U,
    common.KEY_V,
    common.KEY_W,
    common.KEY_P,
)


def _make_params() -> parameters_lib.SwirlLMParameters:
  """Creates SwirlLMParameters from inline textproto."""
  config = parameters_pb2.SwirlLMParameters()
  text_format.Parse(_TGV_CONFIG, config)
  return parameters_lib.SwirlLMParameters(config)


def _tgv_init(
    params: parameters_lib.SwirlLMParameters,
) -> dict[str, jax.Array]:
  """Initializes the Taylor-Green Vortex flow field.

  u =  U sin(x) cos(y) cos(z)
  v = -U cos(x) sin(y) cos(z)
  w = 0
  p = (rho U^2 / 16) * (cos(2z) + 2) * (cos(2x) + cos(2y))

  Args:
    params: Simulation parameters.

  Returns:
    Initial states dict with rho, u, v, w, p.
  """
  gp = params.grid_params
  nx = gp.nx + 2 * gp.halo_width
  ny = gp.ny + 2 * gp.halo_width
  nz = gp.nz + 2 * gp.halo_width

  dx = gp.to_xyz_order(gp.grid_spacings)[0]
  dy = gp.to_xyz_order(gp.grid_spacings)[1]
  dz = gp.to_xyz_order(gp.grid_spacings)[2]

  x_1d = jnp.arange(nx) * dx - gp.halo_width * dx
  y_1d = jnp.arange(ny) * dy - gp.halo_width * dy
  z_1d = jnp.arange(nz) * dz - gp.halo_width * dz

  xx, yy, zz = jnp.meshgrid(x_1d, y_1d, z_1d, indexing='ij')

  u = _U_MAG * jnp.sin(xx) * jnp.cos(yy) * jnp.cos(zz)
  v = -_U_MAG * jnp.cos(xx) * jnp.sin(yy) * jnp.cos(zz)
  w = jnp.zeros_like(xx)
  p = (
      _RHO
      * _U_MAG**2
      / 16.0
      * ((jnp.cos(2.0 * zz) + 2.0) * (jnp.cos(2.0 * xx) + jnp.cos(2.0 * yy)))
  )
  rho = _RHO * jnp.ones_like(xx)

  return {
      common.KEY_RHO: rho,
      common.KEY_U: u,
      common.KEY_V: v,
      common.KEY_W: w,
      common.KEY_P: p,
  }


def _kinetic_energy(states: dict[str, jax.Array], hw: int = 2) -> float:
  """Computes mean kinetic energy over the interior (excluding halos)."""
  s = slice(hw, -hw) if hw > 0 else slice(None)
  u = states[common.KEY_U][s, s, s]
  v = states[common.KEY_V][s, s, s]
  w = states[common.KEY_W][s, s, s]
  rho = states[common.KEY_RHO][s, s, s]
  ke = 0.5 * rho * (u**2 + v**2 + w**2)
  return float(jnp.mean(ke))


def _make_step_fn(
    sim: simulation_lib.Simulation,
    mesh: Mesh,
):
  """Creates a JIT-compiled, shard_map-wrapped step function.

  The simulation step calls `jax.lax.axis_index` inside halo exchange,
  which requires a named-axis context provided by `shard_map`.
  JIT compilation is applied on top for performance.

  Args:
    sim: The simulation object.
    mesh: JAX mesh for device topology.

  Returns:
    A JIT-compiled function that takes and returns state dicts.
  """
  spec = P(*mesh.axis_names)
  in_specs = {k: spec for k in _STATE_KEYS}
  out_specs = {k: spec for k in _STATE_KEYS}

  def step_fn(states):
    return sim.step(states, {}, mesh)

  return jax.jit(
      shard_map.shard_map(
          step_fn,
          mesh=mesh,
          in_specs=(in_specs,),
          out_specs=out_specs,
          check_rep=False,
      )
  )


class TgvSimulationTest(absltest.TestCase):
  """Integration tests for TGV on the JAX solver."""

  def _setup(self):
    """Creates params, mesh, init states, and JIT-compiled step function."""
    params = _make_params()
    devices = jax.experimental.mesh_utils.create_device_mesh(
        (1, 1, 1), devices=jax.local_devices()[:1]
    )
    mesh = Mesh(devices, axis_names=params.grid_params.data_axis_order)
    states = _tgv_init(params)
    sim = simulation_lib.Simulation(params)
    step_fn = _make_step_fn(sim, mesh)
    return states, step_fn

  def test_simulation_runs_without_error(self):
    """Verifies the simulation can run a few steps without crashing."""
    states, step_fn = self._setup()

    # Run 2 steps (first step includes JIT compilation time).
    for _ in range(2):
      states = step_fn(states)

    # Verify output has the expected keys.
    for key in _STATE_KEYS:
      self.assertIn(key, states)

  def test_kinetic_energy_decays(self):
    """Verifies kinetic energy decays for viscous TGV flow."""
    states, step_fn = self._setup()
    ke_initial = _kinetic_energy(states)

    # Run 3 steps.
    for _ in range(3):
      states = step_fn(states)

    ke_final = _kinetic_energy(states)

    # Kinetic energy must decrease for viscous flow.
    self.assertGreater(ke_initial, 0.0)
    self.assertLess(
        ke_final,
        ke_initial,
        f'KE should decay: initial={ke_initial}, final={ke_final}',
    )

  def test_density_stays_constant(self):
    """For constant-density TGV, rho should not change."""
    states, step_fn = self._setup()

    for _ in range(2):
      states = step_fn(states)

    # Density should remain constant.
    np.testing.assert_allclose(
        states[common.KEY_RHO],
        _RHO * jnp.ones_like(states[common.KEY_RHO]),
        atol=1e-10,
    )


def _make_periodic_stretched_coord_file(
    n_points: int,
    domain_size: float,
    stretch_amplitude: float = 0.1,
) -> tuple[str, np.ndarray]:
  """Creates a temp file with mildly stretched periodic coordinates.

  Generates a sinusoidally perturbed uniform grid: x_i = i*dx + A*sin(2*pi*i/N).
  The final point (= first point + domain_size) is appended for the periodic
  convention expected by grid_parametrization.

  Args:
    n_points: Number of interior grid points.
    domain_size: Domain length.
    stretch_amplitude: Amplitude of sinusoidal perturbation.

  Returns:
    Tuple of (filepath, coordinate_array_without_final_point).
  """
  dx = domain_size / n_points
  indices = np.arange(n_points)
  coords = indices * dx + stretch_amplitude * np.sin(
      2.0 * np.pi * indices / n_points
  )
  # Append final point for periodic convention.
  coords_with_end = np.append(coords, domain_size)

  tmpdir = tempfile.mkdtemp()
  filepath = os.path.join(tmpdir, 'stretched_coords.txt')
  np.savetxt(filepath, coords_with_end, fmt='%.15f')
  return filepath, coords


# Stretched grid TGV config template. The dim_z path is filled at runtime.
_TGV_STRETCHED_CONFIG_TEMPLATE = """\
solver_procedure: VARIABLE_DENSITY
convection_scheme: CONVECTION_SCHEME_QUICK
time_integration_scheme: TIME_SCHEME_CN_EXPLICIT_ITERATION
grid_params {{
  computation_shape {{ dim_0: 1 dim_1: 1 dim_2: 1 }}
  grid_size {{ dim_0: 8 dim_1: 8 dim_2: 8 }}
  length {{ dim_0: 6.2831853 dim_1: 6.2831853 dim_2: 0.0 }}
  halo_width: 2
  dt: 1e-3
  kernel_size: 8
  periodic {{ dim_0: true dim_1: true dim_2: true }}
  stretched_grid_files {{ dim_2 {{ path: "{coord_file}" }} }}
}}
pressure {{
  solver {{
    jacobi {{
      max_iterations: 5 halo_width: 2 omega: 0.67
    }}
  }}
}}
thermodynamics {{
  constant_density {{}}
}}
density: 1.0
kinematic_viscosity: 6.25e-4
num_sub_iterations: 3
use_sgs: false
enable_rhie_chow_correction: false
"""


class TgvStretchedGridTest(absltest.TestCase):
  """Integration tests for TGV with a stretched grid in one dimension."""

  def _setup(self):
    """Creates params with stretched z-grid, mesh, states, and step fn."""
    domain_z = 2.0 * np.pi
    coord_file, _ = _make_periodic_stretched_coord_file(
        n_points=4, domain_size=domain_z, stretch_amplitude=0.05
    )

    config_text = _TGV_STRETCHED_CONFIG_TEMPLATE.format(coord_file=coord_file)
    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(config_text, config)
    params = parameters_lib.SwirlLMParameters(config)

    devices = jax.experimental.mesh_utils.create_device_mesh(
        (1, 1, 1), devices=jax.local_devices()[:1]
    )
    mesh = Mesh(devices, axis_names=params.grid_params.data_axis_order)

    # Initialize flow field using gp.nx (= grid_size from proto), which
    # already includes halos: core_n + 2*halo_width. This matches the size of
    # the stretched grid scale factors from
    # local_stretched_grid_vars_from_global_xyz.
    gp = params.grid_params
    nx, ny, _ = gp.nx, gp.ny, gp.nz
    dx = gp.to_xyz_order(gp.grid_spacings)[0]
    dy = gp.to_xyz_order(gp.grid_spacings)[1]

    # For the stretched z-dimension, use global_xyz_with_halos coordinates.
    z_coords = gp.global_xyz_with_halos[gp.get_axis_index('z')]

    x_1d = jnp.arange(nx) * dx - gp.halo_width * dx
    y_1d = jnp.arange(ny) * dy - gp.halo_width * dy
    z_1d = z_coords

    xx, yy, zz = jnp.meshgrid(x_1d, y_1d, z_1d, indexing='ij')

    states = {
        common.KEY_RHO: _RHO * jnp.ones_like(xx),
        common.KEY_U: _U_MAG * jnp.sin(xx) * jnp.cos(yy) * jnp.cos(zz),
        common.KEY_V: -_U_MAG * jnp.cos(xx) * jnp.sin(yy) * jnp.cos(zz),
        common.KEY_W: jnp.zeros_like(xx),
        common.KEY_P: (
            _RHO
            * _U_MAG**2
            / 16.0
            * (
                (jnp.cos(2.0 * zz) + 2.0)
                * (jnp.cos(2.0 * xx) + jnp.cos(2.0 * yy))
            )
        ),
    }

    # Compute stretched grid scale factors.
    sg_vars = stretched_grid.local_stretched_grid_vars_from_global_xyz(
        params, logical_coordinates=(0, 0, 0)
    )

    sim = simulation_lib.Simulation(params)

    # Determine all state keys for shard_map specs.
    all_state_keys = list(_STATE_KEYS)
    additional_keys = list(sg_vars.keys())
    all_keys = all_state_keys + additional_keys

    spec = P(*mesh.axis_names)
    in_specs = {k: spec for k in all_keys}
    out_specs = {k: spec for k in all_keys}

    # Separate essential and additional states for the step call.
    def step_fn(combined_states):
      ess = {k: combined_states[k] for k in _STATE_KEYS}
      add = {k: combined_states[k] for k in additional_keys}
      return sim.step(ess, add, mesh)

    jit_step = jax.jit(
        shard_map.shard_map(
            step_fn,
            mesh=mesh,
            in_specs=(in_specs,),
            out_specs=out_specs,
            check_rep=False,
        )
    )

    # Merge states and stretched grid vars for combined input.
    combined = dict(states)
    combined.update(sg_vars)

    return combined, jit_step, all_state_keys, additional_keys

  def test_stretched_grid_init_produces_valid_states(self):
    """Verifies stretched grid initialization produces valid scale factors."""
    combined, _, _, additional_keys = self._setup()

    # Stretched grid should produce h and h_face keys for the z dimension.
    z_dim = 2  # z is the stretched dimension
    h_key = stretched_grid_util.h_key(z_dim)
    h_face_key = stretched_grid_util.h_face_key(z_dim)
    self.assertIn(h_key, additional_keys)
    self.assertIn(h_face_key, additional_keys)

    # Scale factors should be positive (physical grid spacing > 0).
    h = combined[h_key]
    h_face = combined[h_face_key]
    self.assertTrue(jnp.all(h > 0), f'h has non-positive values: {h}')
    self.assertTrue(
        jnp.all(h_face > 0), f'h_face has non-positive values: {h_face}'
    )

    # Scale factors should be broadcastable to 3D field shape.
    rho = combined[common.KEY_RHO]
    # h has shape (1, 1, nz) for dim=2, should broadcast to (nx, ny, nz).
    result = rho * h  # Should not raise.
    self.assertEqual(result.shape, rho.shape)

  def test_stretched_grid_simulation_runs(self):
    """Verifies simulation with stretched grid runs without error."""
    combined, jit_step, _, additional_keys = self._setup()

    for _ in range(2):
      result = jit_step(combined)
      # Carry over additional_states (unchanged by step).
      combined = dict(result)
      for k in additional_keys:
        if k not in combined:
          combined[k] = self._setup()[0][k]

    for key in _STATE_KEYS:
      self.assertIn(key, result)

  def test_stretched_grid_kinetic_energy_decays(self):
    """Verifies KE decays for stretched-grid TGV."""
    combined, jit_step, _, additional_keys = self._setup()
    ke_initial = _kinetic_energy(combined)

    for _ in range(3):
      result = jit_step(combined)
      combined = dict(result)
      for k in additional_keys:
        if k not in combined:
          combined[k] = self._setup()[0][k]

    ke_final = _kinetic_energy(result)

    self.assertGreater(ke_initial, 0.0)
    self.assertLess(
        ke_final,
        ke_initial,
        f'KE should decay: initial={ke_initial}, final={ke_final}',
    )


# ---- Driver-level TGV tests (using run_simulation) ---- #

# Config for driver tests: single device, periodic TGV.
_DRIVER_TGV_CONFIG_111 = """\
solver_procedure: VARIABLE_DENSITY
convection_scheme: CONVECTION_SCHEME_QUICK
time_integration_scheme: TIME_SCHEME_CN_EXPLICIT_ITERATION
grid_params {
  computation_shape { dim_0: 1 dim_1: 1 dim_2: 1 }
  grid_size { dim_0: 32 dim_1: 32 dim_2: 32 }
  length { dim_0: 6.2831853 dim_1: 6.2831853 dim_2: 6.2831853 }
  halo_width: 2
  dt: 1e-3
  kernel_size: 8
  periodic { dim_0: true dim_1: true dim_2: true }
}
pressure {
  solver {
    jacobi {
      max_iterations: 5 halo_width: 2 omega: 0.67
    }
  }
}
thermodynamics {
  constant_density {}
}
density: 1.0
kinematic_viscosity: 6.25e-4
num_sub_iterations: 3
use_sgs: false
enable_rhie_chow_correction: false
"""

# Config for driver tests: 4 devices (2x2x1), periodic TGV.
_DRIVER_TGV_CONFIG_221 = """\
solver_procedure: VARIABLE_DENSITY
convection_scheme: CONVECTION_SCHEME_QUICK
time_integration_scheme: TIME_SCHEME_CN_EXPLICIT_ITERATION
grid_params {
  computation_shape { dim_0: 2 dim_1: 2 dim_2: 1 }
  grid_size { dim_0: 16 dim_1: 16 dim_2: 32 }
  length { dim_0: 6.2831853 dim_1: 6.2831853 dim_2: 6.2831853 }
  halo_width: 2
  dt: 1e-3
  kernel_size: 8
  periodic { dim_0: true dim_1: true dim_2: true }
}
pressure {
  solver {
    jacobi {
      max_iterations: 5 halo_width: 2 omega: 0.67
    }
  }
}
thermodynamics {
  constant_density {}
}
density: 1.0
kinematic_viscosity: 6.25e-4
num_sub_iterations: 3
use_sgs: false
enable_rhie_chow_correction: false
"""


def _tgv_driver_init_fn(
    params: parameters_lib.SwirlLMParameters,
    logical_coordinates: tuple[int, int, int],
) -> dict[str, jax.Array]:
  """TGV init function matching the driver's InitFn signature.

  Generates device-local TGV initial conditions based on the device's
  position in the computational grid (logical_coordinates).

  Args:
    params: Simulation parameters.
    logical_coordinates: (cx_i, cy_i, cz_i) position of this device.

  Returns:
    Device-local initial states dict with rho, u, v, w, p.
  """
  gp = params.grid_params
  hw = gp.halo_width

  # Core points per device in each dimension.
  core_nx = gp.nx - 2 * hw
  core_ny = gp.ny - 2 * hw
  core_nz = gp.nz - 2 * hw

  # Total points per device (core + halos).
  local_nx = core_nx + 2 * hw
  local_ny = core_ny + 2 * hw
  local_nz = core_nz + 2 * hw

  # Grid spacing (uniform).
  dx, dy, dz = gp.to_xyz_order(gp.grid_spacings)

  # Global coordinate offsets for this device.
  cx_i, cy_i, cz_i = logical_coordinates
  x_offset = cx_i * core_nx * dx
  y_offset = cy_i * core_ny * dy
  z_offset = cz_i * core_nz * dz

  # Local coordinates including halos.
  x_1d = jnp.arange(local_nx) * dx - hw * dx + x_offset
  y_1d = jnp.arange(local_ny) * dy - hw * dy + y_offset
  z_1d = jnp.arange(local_nz) * dz - hw * dz + z_offset

  xx, yy, zz = jnp.meshgrid(x_1d, y_1d, z_1d, indexing='ij')

  u = _U_MAG * jnp.sin(xx) * jnp.cos(yy) * jnp.cos(zz)
  v = -_U_MAG * jnp.cos(xx) * jnp.sin(yy) * jnp.cos(zz)
  w = jnp.zeros_like(xx)
  p = (
      _RHO
      * _U_MAG**2
      / 16.0
      * ((jnp.cos(2.0 * zz) + 2.0) * (jnp.cos(2.0 * xx) + jnp.cos(2.0 * yy)))
  )
  rho = _RHO * jnp.ones_like(xx)

  return {
      common.KEY_RHO: rho,
      common.KEY_U: u,
      common.KEY_V: v,
      common.KEY_W: w,
      common.KEY_P: p,
  }


def _remove_halos(params, states, varnames):
  """Converts global sharded fields into contiguous interior arrays.

  Delegates to `global_layout.strip_halos_global`, which also handles a
  non-xyz `data_axis_order`; the previous inline version indexed `gp.cx`/
  `gp.nx` against array axes 0/1/2 and was only correct for `'xyz'`.

  Args:
    params: The simulation parameters carrying the grid parametrization.
    states: Mapping from variable name to the global sharded array.
    varnames: The variable names to convert; other entries are dropped.

  Returns:
    Mapping from each selected variable name to its halo-free interior array.
  """
  return {
      key: global_layout.strip_halos_global(np.asarray(v), params.grid_params)
      for key, v in states.items()
      if key in varnames
  }


def _make_driver_params() -> parameters_lib.SwirlLMParameters:
  """Builds the single-device driver-level TGV parameters."""
  config = parameters_pb2.SwirlLMParameters()
  text_format.Parse(_DRIVER_TGV_CONFIG_111, config)
  return parameters_lib.SwirlLMParameters(config)


class TgvDriverSingleDeviceTest(absltest.TestCase):
  """Tests run_simulation end-to-end on a single device."""

  def _make_params(self):
    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(_DRIVER_TGV_CONFIG_111, config)
    return parameters_lib.SwirlLMParameters(config)

  def test_run_simulation_completes(self):
    """Verifies run_simulation runs to completion on 1 device."""
    params = self._make_params()
    state = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=100,
        num_steps_per_cycle=3,
    )

    # Verify output has expected keys.
    for key in _STATE_KEYS:
      self.assertIn(key, state)

    # Verify no NaN/Inf.
    for key in _STATE_KEYS:
      self.assertTrue(
          jnp.all(jnp.isfinite(state[key])),
          f'Non-finite values in {key}',
      )

    state_no_halos = _remove_halos(params, state, ('u', 'v', 'w', 'p'))
    write_dir = os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')
    _ = visualization_lib.plot_overview(
        state_no_halos, save_path=f'{write_dir}/tgv_1x1x1.png'
    )

  def test_run_simulation_kinetic_energy_decays(self):
    """Verifies KE decays when run through the driver."""
    params = self._make_params()

    # Run 1 step to get reference KE.
    state_1 = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=1,
    )
    ke_1 = _kinetic_energy(state_1)

    # Run 5 steps.
    state_5 = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=5,
    )
    ke_5 = _kinetic_energy(state_5)

    self.assertGreater(ke_1, 0.0)
    self.assertLess(
        ke_5,
        ke_1,
        f'KE should decay: 1-step={ke_1}, 5-step={ke_5}',
    )

  def test_run_simulation_time_accumulates(self):
    """Verifies TIME_VARNAME is incremented correctly."""
    params = self._make_params()
    num_steps = 5
    state = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=num_steps,
    )

    expected_time = num_steps * params.dt
    actual_time = float(state[driver_lib.TIME_VARNAME])
    np.testing.assert_allclose(actual_time, expected_time, rtol=1e-6)


class TgvDriverMultiDeviceTest(absltest.TestCase):
  """Tests run_simulation end-to-end on 4 devices (1x1x4)."""

  def _make_params(self):
    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(_DRIVER_TGV_CONFIG_221, config)
    return parameters_lib.SwirlLMParameters(config)

  def test_run_simulation_multi_device_completes(self):
    """Verifies run_simulation with 4 devices runs to completion."""
    params = self._make_params()
    state = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=100,
        num_steps_per_cycle=3,
    )

    # Verify output has expected keys.
    for key in _STATE_KEYS:
      self.assertIn(key, state)

    # Verify no NaN/Inf.
    for key in _STATE_KEYS:
      self.assertTrue(
          jnp.all(jnp.isfinite(state[key])),
          f'Non-finite values in {key}',
      )

    state_no_halos = _remove_halos(params, state, ('u', 'v', 'w', 'p'))
    write_dir = os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')
    _ = visualization_lib.plot_overview(
        state_no_halos, save_path=f'{write_dir}/tgv_2x2x1.png'
    )

  def test_run_simulation_multi_device_ke_decays(self):
    """Verifies KE decays on multi-device TGV."""
    params = self._make_params()

    state_1 = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=1,
    )
    state_no_halos_1 = _remove_halos(params, state_1, _STATE_KEYS)
    ke_1 = _kinetic_energy(state_no_halos_1, hw=0)

    state_5 = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=5,
    )
    state_no_halos_5 = _remove_halos(params, state_5, _STATE_KEYS)
    ke_5 = _kinetic_energy(state_no_halos_5, hw=0)

    self.assertGreater(ke_1, 0.0)
    self.assertLess(
        ke_5,
        ke_1,
        f'KE should decay: 1-step={ke_1}, 5-step={ke_5}',
    )

  def test_run_simulation_multi_device_time_accumulates(self):
    """Verifies TIME_VARNAME accumulates correctly on multi-device."""
    params = self._make_params()
    num_steps = 5
    state = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=num_steps,
    )

    expected_time = num_steps * params.dt
    actual_time = float(state[driver_lib.TIME_VARNAME])
    np.testing.assert_allclose(actual_time, expected_time, rtol=1e-6)

  def test_run_simulation_with_boundary_condition_additional_states(self):
    """Verifies run_simulation accepts boundary condition additional_states."""
    params = self._make_params()
    gp = params.grid_params
    plane_dims = [d for d in range(3) if d != 0]
    full_sizes = (gp.fx, gp.fy, gp.fz)
    plane_2d = jnp.ones(
        (full_sizes[plane_dims[0]], full_sizes[plane_dims[1]]),
        dtype=jnp.float32,
    )
    bc_arr = boundary_condition_utils.boundary_plane_to_bc(
        plane_2d,
        dim=0,
        halo_width=params.halo_width,
        grid_params=gp,
    )
    additional_states = {
        'bc_u_0_0': bc_arr,
    }
    state = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=2,
        additional_states=additional_states,
    )
    self.assertIn('bc_u_0_0', state)
    self.assertEqual(state['bc_u_0_0'].shape, bc_arr.shape)


class _CycleCompileCounter(pylog.Handler):
  """Counts XLA compilations of the jitted simulation cycle.

  This scrapes a JAX-internal log message, which is the only handle on the
  compilation count from outside `run_simulation`: the jitted callable is a
  local there, so neither `_cache_size()` nor an aval comparison via
  `jax.eval_shape` is reachable. If JAX ever changes the message, the count
  falls to zero and the assertions below fail loudly rather than passing
  vacuously.
  """

  def __init__(self):
    super().__init__()
    self.count = 0

  def emit(self, record):
    # Depending on the JAX version the module is named `_shard_map_cycle_body`
    # or `jit(_shard_map_cycle_body)`; accept both.
    message = record.getMessage()
    if message.startswith('Compiling ') and '_shard_map_cycle_body' in message:
      self.count += 1


class CycleCompilationTest(absltest.TestCase):
  """The simulation cycle must be compiled exactly once per run.

  JAX keys its jit cache on `weak_type` as well as shape and dtype, so a
  weakly typed state array entering the cycle and coming back strongly typed
  costs a second trace and compilation of the entire cycle. `init_fn`s
  routinely produce weak arrays -- `jnp.arange(n) * dx` with a Python-float
  `dx` is enough -- so `driver._strengthen_dtypes` canonicalizes the state
  both on the way in and on the scan carry.
  """

  def _count_compiles(self, **run_kwargs):
    params = _make_driver_params()
    # A warm cache from an earlier test in this process would otherwise make
    # the count zero and hide a regression.
    jax.clear_caches()
    counter = _CycleCompileCounter()
    logger = pylog.getLogger('jax')
    previous_level = logger.level
    logger.setLevel(pylog.DEBUG)
    logger.addHandler(counter)
    try:
      with jax.log_compiles(True):
        state = driver_lib.run_simulation(
            params=params,
            init_fn=_tgv_driver_init_fn,
            num_steps=2,
            num_steps_per_cycle=1,
            **run_kwargs,
        )
    finally:
      logger.removeHandler(counter)
      logger.setLevel(previous_level)
    return counter.count, state

  def test_compiled_once_with_weakly_typed_init_fn(self):
    """`_tgv_driver_init_fn` returns weakly typed fields; one compile only."""
    count, state = self._count_compiles()

    self.assertEqual(count, 1)
    weak = [k for k, v in state.items() if getattr(v, 'weak_type', False)]
    self.assertEmpty(weak)

  def test_compiled_once_with_weakly_typed_update_fn(self):
    """A user update fn returning a weak value must not retrigger a compile.

    Step 3 of the cycle (`_update_additional_states`) is the only step not
    wrapped in `lax.cond`, which would unify the types itself, and `lax.scan`
    does not promote a weakly typed carry. `jnp.where(x > 0.0, 1.0, 0.0)` --
    the idiomatic indicator field -- returns a weak array even from strong
    inputs, so without the guard on the carry this run compiles twice.
    """
    params = _make_driver_params()
    gp = params.grid_params
    plane = jnp.ones((gp.fy, gp.fz), dtype=jnp.float32)
    bc_arr = boundary_condition_utils.boundary_plane_to_bc(
        plane, dim=0, halo_width=params.halo_width, grid_params=gp
    )

    def weakly_typed_update_fn(states, additional_states, step_id=None):
      del states, step_id
      updated = dict(additional_states)
      updated['bc_u_0_0'] = jnp.where(
          additional_states['bc_u_0_0'] > 0.0, 1.0, 0.0
      )
      return updated

    count, _ = self._count_compiles(
        additional_states={'bc_u_0_0': bc_arr},
    )
    self.assertEqual(count, 1)

    # Same run, now with the weak-producing update function.
    params.additional_states_update_fn = weakly_typed_update_fn
    jax.clear_caches()
    counter = _CycleCompileCounter()
    logger = pylog.getLogger('jax')
    previous_level = logger.level
    logger.setLevel(pylog.DEBUG)
    logger.addHandler(counter)
    try:
      with jax.log_compiles(True):
        state = driver_lib.run_simulation(
            params=params,
            init_fn=_tgv_driver_init_fn,
            num_steps=2,
            num_steps_per_cycle=1,
            additional_states={'bc_u_0_0': bc_arr},
        )
    finally:
      logger.removeHandler(counter)
      logger.setLevel(previous_level)

    self.assertEqual(counter.count, 1)
    self.assertFalse(getattr(state['bc_u_0_0'], 'weak_type', False))


class UpdateAdditionalStatesTest(absltest.TestCase):
  """`_update_additional_states` must not discard its own earlier steps.

  The function clears `src_*` source terms and refreshes the nonreflecting
  boundary condition states before handing control to a user-supplied
  `additional_states_update_fn`. Both of those have to reach the user function
  and survive its return, which they did not while its result replaced the
  accumulated dict instead of being merged into it.
  """

  def _run(self, additional_states, update_fn):
    params = _make_driver_params()
    params.additional_states_update_fn = update_fn
    return driver_lib._update_additional_states(
        {'u': jnp.zeros((4, 4, 4), jnp.float32)},
        additional_states,
        jnp.int32(0),
        params,
        mesh=None,
    )

  def test_source_terms_are_cleared_before_the_user_fn_runs(self):
    seen = {}

    def update_fn(states, additional_states, step_id=None):
      del states, step_id
      seen.update(additional_states)
      updated = dict(additional_states)
      # The idiomatic pattern: accumulate this step's contribution onto the
      # cleared slot.
      updated['src_theta'] = updated['src_theta'] + 1.0
      return updated

    stale = jnp.full((4, 4, 4), 5.0, jnp.float32)
    out = self._run({'src_theta': stale}, update_fn)

    # The user function must have seen zeros, not last step's 5.0 ...
    np.testing.assert_array_equal(
        np.asarray(seen['src_theta']), np.zeros((4, 4, 4), np.float32)
    )
    # ... so its accumulation starts from zero every step.
    np.testing.assert_array_equal(
        np.asarray(out['src_theta']), np.ones((4, 4, 4), np.float32)
    )

  def test_keys_the_user_fn_omits_are_preserved(self):
    def update_fn(states, additional_states, step_id=None):
      del states, additional_states, step_id
      # A function that reports only what it touched.
      return {'other': jnp.full((4, 4, 4), 2.0, jnp.float32)}

    out = self._run(
        {
            'src_theta': jnp.full((4, 4, 4), 5.0, jnp.float32),
            'other': jnp.zeros((4, 4, 4), jnp.float32),
        },
        update_fn,
    )

    # The key set must be preserved: dropping one changes the scan carry
    # structure and fails deep inside the cycle.
    self.assertEqual(set(out), {'src_theta', 'other'})
    np.testing.assert_array_equal(
        np.asarray(out['src_theta']), np.zeros((4, 4, 4), np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(out['other']), np.full((4, 4, 4), 2.0, np.float32)
    )

  def test_user_fn_cannot_mutate_the_state_being_assembled(self):
    def update_fn(states, additional_states, step_id=None):
      del states, step_id
      # A function that mutates its argument in place and returns nothing of
      # its own must not corrupt the caller's dict.
      additional_states['src_theta'] = jnp.full((4, 4, 4), 9.0, jnp.float32)
      return {}

    out = self._run(
        {'src_theta': jnp.full((4, 4, 4), 5.0, jnp.float32)}, update_fn
    )
    np.testing.assert_array_equal(
        np.asarray(out['src_theta']), np.zeros((4, 4, 4), np.float32)
    )


# The TGV driver config with the static Smagorinsky model switched on and
# `nu_t` tracked as an additional state, which is how a real LES config asks
# the driver to report the turbulent viscosity.
_DRIVER_TGV_CONFIG_SGS = _DRIVER_TGV_CONFIG_111.replace(
    'use_sgs: false',
    'use_sgs: true\n'
    'sgs_model { smagorinsky { c_s: 0.18 pr_t: 1.0 } }\n'
    'additional_state_keys: "nu_t"\n',
)


class SgsStatePersistenceTest(absltest.TestCase):
  """The solver's own additional states must survive the step.

  `sim.step` owns a few additional states -- `nu_t`, `drho`, the buoyancy
  diagnostics -- and returns the updated value for them. The driver has to
  pull those back into `add` before writing `add` into the step's output,
  or the stale pre-step value is put back and the computed one is lost.

  That is not hypothetical: it kept `nu_t` identically zero for entire runs
  despite `use_sgs: true`, and with it `tke`, which the combustion model
  derives from the stored `nu_t` as `(nu_t / (c_k * delta))**2`. The momentum
  equation was unaffected because it recomputes `nu_t` internally each step
  and never reads the stored copy, so the defect was invisible in the
  velocity field and showed up only in the reported diagnostics and in the
  models that consume them.
  """

  def _params(self):
    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(_DRIVER_TGV_CONFIG_SGS, config)
    return parameters_lib.SwirlLMParameters(config)

  def _init_fn(self, params, logical_coordinates):
    states = _tgv_driver_init_fn(params, logical_coordinates)
    states['nu_t'] = jnp.zeros_like(states['u'])
    return states

  def test_sgs_model_produces_nonzero_viscosity(self):
    # Guards the premise: if the model itself returned zero, the driver test
    # below would pass for the wrong reason.
    params = self._params()
    model = sgs_model_lib.SgsModel(params)
    init = self._init_fn(params, (0, 0, 0))
    nu_t = np.asarray(
        model.turbulent_viscosity(
            [init['u'], init['v'], init['w']], {}
        )
    )
    self.assertGreater(float(np.abs(nu_t).max()), 0.0)

  def test_nu_t_is_reported_after_a_step(self):
    params = self._params()
    state = driver_lib.run_simulation(
        params=params,
        init_fn=self._init_fn,
        num_steps=3,
        num_steps_per_cycle=1,
    )
    nu_t = np.asarray(state['nu_t'])
    self.assertTrue(np.all(np.isfinite(nu_t)))
    self.assertGreater(
        float(np.abs(nu_t).max()),
        0.0,
        'nu_t came back identically zero, so the driver discarded the value '
        'the solver computed.',
    )


class StrengthenDtypesTest(absltest.TestCase):

  def test_strips_weak_type_and_preserves_everything_else(self):
    weak = jnp.full((4, 4, 4), 1.25)
    self.assertTrue(weak.weak_type)

    strong = driver_lib._strengthen_dtypes(weak)

    self.assertFalse(strong.weak_type)
    self.assertEqual(strong.dtype, weak.dtype)
    np.testing.assert_array_equal(np.asarray(strong), np.asarray(weak))

  def test_already_strong_leaves_are_untouched(self):
    strong = jnp.zeros((4, 4, 4), jnp.float32)
    self.assertIs(driver_lib._strengthen_dtypes(strong), strong)

  def test_maps_over_a_state_dict_including_scalars_and_masks(self):
    state = {
        'u': jnp.full((2, 2, 2), 0.5),
        'simulation_time': jnp.float32(0.0),
        'mask': jnp.where(jnp.zeros((2, 2)) > 0.0, True, False),
        'count': jnp.full((2,), 3),
    }
    out = driver_lib._strengthen_dtypes(state)

    self.assertEqual(set(out), set(state))
    for key, value in out.items():
      self.assertFalse(
          getattr(value, 'weak_type', False), f'{key} is still weakly typed'
      )
      self.assertEqual(value.dtype, state[key].dtype)


if __name__ == '__main__':
  absltest.main()
