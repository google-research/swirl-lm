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
"""Run a small Taylor-Green Vortex case with the JAX Swirl-LM solver.

This is a standalone version of
`TgvSimulationTest.test_simulation_runs_without_error`. It constructs the
configuration and initial state in memory, runs `Simulation.step` under a
single-device `shard_map`, and checks that the final state is finite.

Run this script from the root of a Swirl-LM checkout, for example:

  python run_tgv_jax.py --num-steps 10

On MacOS, only CPU is supported! Run with:

  JAX_PLATFORMS=cpu python run_tgv_jax.py --num-steps 10

"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from google.protobuf import text_format
import jax
from jax.experimental import shard_map
import jax.experimental.mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from swirl_lm.base import parameters_pb2
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.core import simulation as simulation_lib
from swirl_lm.jax.equations import common


jax.config.update("jax_enable_x64", False)


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
      max_iterations: 5
      halo_width: 2
      omega: 0.67
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
  """Builds the small TGV configuration used by the integration test."""
  config = parameters_pb2.SwirlLMParameters()
  text_format.Parse(_TGV_CONFIG, config)
  return parameters_lib.SwirlLMParameters(config)


def _initial_state(
    params: parameters_lib.SwirlLMParameters,
) -> dict[str, jax.Array]:
  """Creates the periodic Taylor-Green Vortex initial condition."""
  grid = params.grid_params
  halo_width = grid.halo_width
  nx = grid.nx + 2 * halo_width
  ny = grid.ny + 2 * halo_width
  nz = grid.nz + 2 * halo_width

  dx, dy, dz = grid.to_xyz_order(grid.grid_spacings)
  x = jnp.arange(nx) * dx - halo_width * dx
  y = jnp.arange(ny) * dy - halo_width * dy
  z = jnp.arange(nz) * dz - halo_width * dz
  xx, yy, zz = jnp.meshgrid(x, y, z, indexing="ij")

  u = _U_MAG * jnp.sin(xx) * jnp.cos(yy) * jnp.cos(zz)
  v = -_U_MAG * jnp.cos(xx) * jnp.sin(yy) * jnp.cos(zz)
  w = jnp.zeros_like(xx)
  pressure = (
      _RHO
      * _U_MAG**2
      / 16.0
      * (jnp.cos(2.0 * zz) + 2.0)
      * (jnp.cos(2.0 * xx) + jnp.cos(2.0 * yy))
  )

  return {
      common.KEY_RHO: _RHO * jnp.ones_like(xx),
      common.KEY_U: u,
      common.KEY_V: v,
      common.KEY_W: w,
      common.KEY_P: pressure,
  }


def _make_mesh(params: parameters_lib.SwirlLMParameters) -> Mesh:
  """Creates the named single-device mesh required by halo exchange."""
  devices = jax.experimental.mesh_utils.create_device_mesh(
      (1, 1, 1), devices=jax.local_devices()[:1]
  )
  return Mesh(devices, axis_names=params.grid_params.data_axis_order)


def _make_step_function(
    simulation: simulation_lib.Simulation,
    mesh: Mesh,
):
  """Wraps `Simulation.step` in the required shard-map axis context."""
  spec = P(*mesh.axis_names)
  state_specs = {key: spec for key in _STATE_KEYS}

  def step(states):
    return simulation.step(states, {}, mesh)

  return jax.jit(
      shard_map.shard_map(
          step,
          mesh=mesh,
          in_specs=(state_specs,),
          out_specs=state_specs,
          check_rep=False,
      )
  )


def _block_until_ready(states: dict[str, jax.Array]) -> None:
  """Waits for asynchronous JAX work to finish."""
  jax.tree_util.tree_map(lambda value: value.block_until_ready(), states)


def _kinetic_energy(
    states: dict[str, jax.Array], halo_width: int
) -> float:
  """Returns mean kinetic energy over the non-halo cells."""
  interior = (
      slice(halo_width, -halo_width)
      if halo_width
      else slice(None)
  )
  region = (interior, interior, interior)
  rho = states[common.KEY_RHO][region]
  u = states[common.KEY_U][region]
  v = states[common.KEY_V][region]
  w = states[common.KEY_W][region]
  return float(jnp.mean(0.5 * rho * (u**2 + v**2 + w**2)))


def _validate_final_state(states: dict[str, jax.Array]) -> None:
  """Raises an exception if required variables are absent or non-finite."""
  missing = [key for key in _STATE_KEYS if key not in states]
  if missing:
    raise RuntimeError(f"Simulation output is missing state variables: {missing}")

  non_finite = [
      key for key in _STATE_KEYS if not bool(jnp.all(jnp.isfinite(states[key])))
  ]
  if non_finite:
    raise FloatingPointError(
        f"Simulation produced NaN or Inf in variables: {non_finite}"
    )


def run_tgv(
    num_steps: int,
    report_every: int = 1,
) -> dict[str, jax.Array]:
  """Runs the single-device TGV simulation and returns its final state.

  Args:
    num_steps: Number of calls to `Simulation.step`.
    report_every: Print kinetic energy every N steps. Zero disables periodic
      reports; the initial and final energies are still printed.

  Returns:
    Final Swirl-LM state dictionary.

  Raises:
    ValueError: If either argument is outside its supported range.
    RuntimeError: If an expected state variable is absent.
    FloatingPointError: If the final state contains NaN or infinity.
  """
  if num_steps < 1:
    raise ValueError("num_steps must be at least 1")
  if report_every < 0:
    raise ValueError("report_every cannot be negative")

  params = _make_params()
  mesh = _make_mesh(params)
  states = _initial_state(params)
  simulation = simulation_lib.Simulation(params)
  step_function = _make_step_function(simulation, mesh)

  initial_energy = _kinetic_energy(states, params.grid_params.halo_width)
  print(f"JAX backend: {jax.default_backend()}")
  print(f"Device: {jax.local_devices()[0]}")
  print(f"Steps: {num_steps}; dt: {params.dt:g}")
  print(f"step 0/{num_steps}: kinetic_energy={initial_energy:.9e}")

  start_time = time.perf_counter()
  for step_number in range(1, num_steps + 1):
    states = step_function(states)
    should_report = (
        step_number == num_steps
        or (report_every > 0 and step_number % report_every == 0)
    )
    if should_report:
      _block_until_ready(states)
      energy = _kinetic_energy(states, params.grid_params.halo_width)
      print(
          f"step {step_number}/{num_steps}: "
          f"kinetic_energy={energy:.9e}"
      )

  _block_until_ready(states)
  elapsed = time.perf_counter() - start_time
  _validate_final_state(states)

  final_energy = _kinetic_energy(states, params.grid_params.halo_width)
  print(
      f"Completed {num_steps} steps in {elapsed:.3f} s "
      "(includes first-call JIT compilation)."
  )
  print(f"Kinetic-energy change: {final_energy - initial_energy:+.9e}")
  print("Final state is finite and contains all required variables.")
  return states


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run a small single-device Taylor-Green Vortex in JAX."
  )
  parser.add_argument(
      "--num-steps",
      type=int,
      default=2,
      help="number of simulation steps to run (default: 2)",
  )
  parser.add_argument(
      "--report-every",
      type=int,
      default=1,
      help="report kinetic energy every N steps; 0 means final only",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  run_tgv(args.num_steps, args.report_every)


if __name__ == "__main__":
  main()
