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
"""Driver for the JAX Navier-Stokes solver.

Provides utilities to parse configs, run the simulation loop with optional
checkpointing, restart from a previous checkpoint, and visualization hooks.

The main solver loop structure:

  solver_loop  (Python cycle loop, handles IO/checkpointing)
    └─ _one_cycle  (JIT-compiled, runs `num_steps` time steps)
         └─ per-step:
              1. preprocess (conditional on step_id)
              2. _update_additional_states
              3. model.step
              4. postprocess (conditional on step_id)
              5. TIME_VARNAME accumulation

Multi-device support:
  The simulation uses SPMD execution via `shard_map`. The 3D computational
  domain is partitioned across devices according to `params.cx/cy/cz`
  (matching TF's `computation_shape`). A JAX `Mesh` with axis names
  matching `data_axis_order` is constructed from `cx * cy * cz` devices.

  The step body runs inside `shard_map`, providing named-axis context for:
    - `jax.lax.ppermute` (halo exchange between neighboring devices)
    - `jax.lax.axis_index` (device boundary detection)
    - `jax.lax.psum` (global reductions in linear solvers)

  State arrays are distributed across devices using `NamedSharding`.
  Each device holds its local partition of size `(nx, ny, nz)`.

Usage example (TGV with checkpointing):
  ```python
  from swirl_lm.jax.core import driver

  config_proto = driver.load_config('tgv_3d.textpb')
  params = parameters_lib.SwirlLMParameters(config_proto)

  def tgv_init_fn(params):
    ...
    return {'rho': rho, 'u': u, 'v': v, 'w': w, 'p': p}

  states = driver.run_simulation(
      params=params,
      init_fn=tgv_init_fn,
      num_steps=1000,
      mesh=mesh,
      checkpoint_dir='/tmp/tgv_run',
      checkpoint_interval=100,
  )
  ```

Usage example (restart from a previous checkpoint):
  ```python
  # Restart from step 500 and run 500 more steps.
  states = driver.run_simulation(
      params=params,
      init_fn=tgv_init_fn,
      num_steps=500,
      mesh=mesh,
      restart_from='/tmp/tgv_run/step_000500.zarr',
      checkpoint_dir='/tmp/tgv_run_continued',
      checkpoint_interval=100,
  )
  ```
"""


from collections.abc import Callable, Mapping, Sequence
import inspect
import os
import time
from typing import Any

from absl import logging
from google.protobuf import text_format
import jax
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
from jax.experimental import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
from jax.sharding import NamedSharding  # pylint: disable=g-importing-member
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
import numpy as np
from swirl_lm.base import parameters_pb2
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.boundary_condition import boundary_condition_utils
from swirl_lm.jax.boundary_condition import nonreflecting_boundary
from swirl_lm.jax.core import simulation as simulation_lib
from swirl_lm.jax.io import checkpoint as checkpoint_lib
from swirl_lm.jax.physics.lpt import lpt
from swirl_lm.jax.utility import global_layout
from swirl_lm.jax.utility import grid_parametrization as gp_lib
from swirl_lm.jax.utility import stretched_grid
from swirl_lm.jax.utility import stretched_grid_util
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

TIME_VARNAME = 'simulation_time'

# Type for the user-supplied initialization function.
# It receives params and a logical_coordinate tuple (cx_i, cy_i, cz_i) for the
# device, and returns the device-local initial states dict.
# This matches TF's `init_fn(replica_id, coordinates)` pattern.
InitFn = Callable[
    [parameters_lib.SwirlLMParameters, tuple[int, int, int]],
    dict[str, ScalarField],
]


def load_config(filepath: str) -> parameters_pb2.SwirlLMParameters:
  """Loads a SwirlLMParameters proto from a textproto file.

  Args:
    filepath: Path to the textproto configuration file.

  Returns:
    The parsed SwirlLMParameters proto.
  """
  with open(filepath, 'r') as f:
    config_text = f.read()
  config = parameters_pb2.SwirlLMParameters()
  text_format.Parse(config_text, config)
  return config


def create_mesh(
    params: parameters_lib.SwirlLMParameters,
) -> Mesh:
  """Creates a JAX Mesh from the simulation parameters.

  The mesh shape is `(cx, cy, cz)` reordered to match `data_axis_order` from
  `params.grid_params`, so that mesh axis 'x' has size cx, 'y' has size cy,
  and 'z' has size cz. For example, with data_axis_order=('z', 'x', 'y'),
  the mesh shape is (cz, cx, cy).

  Args:
    params: The simulation parameters with cx, cy, cz and grid_params.

  Returns:
    A JAX Mesh suitable for shard_map execution.
  """
  gp = params.grid_params
  # Reorder (cx, cy, cz) from (x, y, z) order to data_axis_order so that
  # each mesh axis name matches its partition count.
  mesh_shape = gp.to_data_axis_order(gp.cx, gp.cy, gp.cz)
  num_devices = gp.cx * gp.cy * gp.cz
  devices = mesh_utils.create_device_mesh(
      mesh_shape, devices=jax.devices()[:num_devices]
  )
  return Mesh(devices, axis_names=gp.data_axis_order)


def _get_state_keys(
    params: parameters_lib.SwirlLMParameters,
) -> tuple[list[str], list[str], list[str]]:
  """Returns essential, additional, and helper var state keys.

  Mirrors TF `_get_state_keys`. Separates all simulation state keys into
  three categories:
    - essential_keys: prognostic flow field variables.
    - additional_keys: boundary conditions, forcing terms, TIME_VARNAME, etc.
    - helper_var_keys: Poisson solver helper vars, monitor analytics, etc.

  Args:
    params: The simulation parameters.

  Returns:
    A tuple of (essential_keys, additional_keys, helper_var_keys).
  """
  essential_keys = ['u', 'v', 'w', 'p'] + list(params.transport_scalars_names)
  if params.solver_procedure == parameters_lib.SolverProcedure.VARIABLE_DENSITY:
    essential_keys += ['rho']

  additional_keys = list(
      params.additional_state_keys if params.additional_state_keys else []
  )
  helper_var_keys = list(
      params.helper_var_keys if params.helper_var_keys else []
  )

  additional_keys.append(TIME_VARNAME)

  # Add LPT keys if configured.
  additional_keys += lpt.required_keys(params.lpt)

  # Add stretched grid keys. In JAX all fields are 3D arrays, so scale
  # factors go into helper_var_keys and 3D coordinates into additional_keys.
  coordinate_keys_3d = ('xx', 'yy', 'zz')
  use_stretched_grid_xyz = params.grid_params.to_xyz_order((
      params.use_stretched_grid[0],
      params.use_stretched_grid[1],
      params.use_stretched_grid[2],
  ))
  for dim in range(3):
    if use_stretched_grid_xyz[dim]:
      additional_keys.append(coordinate_keys_3d[dim])
      helper_var_keys.append(stretched_grid_util.h_key(dim))
      helper_var_keys.append(stretched_grid_util.h_face_key(dim))

  # Check for duplicates.
  all_keys = essential_keys + additional_keys + helper_var_keys
  if len(set(essential_keys)) + len(set(additional_keys)) + len(
      set(helper_var_keys)
  ) != len(set(all_keys)):
    raise ValueError(
        'Duplicated keys detected between the three types of states: '
        f'essential states: {essential_keys}, additional states: '
        f'{additional_keys}, and helper vars: {helper_var_keys}'
    )

  return essential_keys, additional_keys, helper_var_keys


def _logical_coordinates(
    params: parameters_lib.SwirlLMParameters,
) -> list[tuple[int, int, int]]:
  """Generates logical coordinates for all replicas.

  Mirrors TF's `tpu_util.grid_coordinates(computation_shape)`. Each replica
  gets a `(cx_i, cy_i, cz_i)` coordinate identifying its position in the
  3D computational grid.

  The iteration order must match `_distribute_states`, which uses
  `np.unravel_index` over `computation_shape` (data_axis_order). We iterate
  in C-order over data_axis_order dimensions and convert each position
  back to physical `(ix, iy, iz)` coordinates.

  Args:
    params: The simulation parameters.

  Returns:
    A list of `(cx_i, cy_i, cz_i)` tuples, one per replica, ordered by
    replica id (C-order over data_axis_order dimensions).
  """
  gp = params.grid_params
  # computation_shape in data_axis_order, matching _distribute_states.
  c_data = gp.to_data_axis_order(gp.cx, gp.cy, gp.cz)
  coords = []
  for i0 in range(c_data[0]):
    for i1 in range(c_data[1]):
      for i2 in range(c_data[2]):
        # Convert from data_axis_order position to physical (ix, iy, iz).
        xyz = gp.to_xyz_order((i0, i1, i2))
        coords.append(xyz)
  return coords


def _init_fn(
    params: parameters_lib.SwirlLMParameters,
    logical_coordinates: tuple[int, int, int],
    customized_init_fn: InitFn | None = None,
) -> dict[str, ScalarField]:
  """Generates the initial state for a single device.

  Mirrors TF `_init_fn` which produces a per-replica init function.
  Each device calls this with its own `logical_coordinates` to get
  its device-local state partition.

  Args:
    params: The simulation parameters.
    logical_coordinates: `(cx_i, cy_i, cz_i)` position of this device in the
      computational grid.
    customized_init_fn: Optional user-provided init function. It receives
      `(params, logical_coordinates)` and returns a dict of device-local state
      arrays.

  Returns:
    A dict of initial states for this device.
  """
  states: dict[str, ScalarField] = {}

  # Initialize stretched grid variables (scale factors and 3D coordinates).
  if any(params.use_stretched_grid):
    stretched_vars = stretched_grid.local_stretched_grid_vars_from_global_xyz(
        params, logical_coordinates=logical_coordinates
    )
    states.update(stretched_vars)

  # Initialize TIME_VARNAME.
  states[TIME_VARNAME] = jnp.float64(0.0)

  # Add LPT variables if configured.
  if params.lpt is not None:
    states.update(lpt.init_fn(params))

  # Apply user-defined init_fn last to allow overrides.
  if customized_init_fn is not None:
    states.update(customized_init_fn(params, logical_coordinates))

  # Add nonreflecting BC states.
  states.update(nonreflecting_boundary.nonreflecting_bc_state_init_fn(params))

  return states


def _global_shape_for_spec(
    local_shape: Sequence[int],
    spec: P,
    mesh: Mesh,
) -> tuple[int, ...]:
  """Expands a per-replica shape into the global shape implied by `spec`.

  An axis named in the spec is distributed, so the global extent is the local
  extent times the size of the mesh axes it is mapped onto. An axis the spec
  leaves as None is held whole by every device.

  Args:
    local_shape: The shape of one replica's array.
    spec: The partition spec for that array.
    mesh: The device mesh.

  Returns:
    The shape of the assembled global array.
  """
  global_shape = []
  for axis, extent in enumerate(local_shape):
    names = spec[axis] if axis < len(spec) else None
    if names is None:
      global_shape.append(extent)
      continue
    if isinstance(names, str):
      names = (names,)
    for name in names:
      extent *= mesh.shape[name]
    global_shape.append(extent)
  return tuple(global_shape)


def _assemble_sharded_array(
    key: str,
    per_replica_states: list[dict[str, ScalarField]],
    spec: P,
    mesh: Mesh,
    computation_shape: tuple[int, ...],
) -> jax.Array:
  """Places each replica's array on the device that owns its shard.

  With a partition spec in hand, the replica index alone determines where a
  tile belongs, so nothing has to be concatenated on the host. This is the
  same property TF relies on, where `distribute_values` only ever needs to
  know which device a value belongs to.

  Devices that end up holding the same shard -- which happens for every mesh
  axis the spec leaves out -- must have been given identical arrays. That is
  checked rather than assumed: silently keeping one of them is exactly how a
  genuinely per-replica array gets replaced by replica 0's copy.

  Args:
    key: The state name, used only for error messages.
    per_replica_states: One state dict per replica, ordered by replica id.
    spec: The partition spec mapping the array's axes onto the mesh.
    mesh: The device mesh.
    computation_shape: The mesh shape, in data axis order.

  Returns:
    The assembled global array with a `NamedSharding` built from `spec`.

  Raises:
    ValueError: If `key` is missing from some replica, or if replicas sharing
      a shard disagree.
  """
  missing = [i for i, s in enumerate(per_replica_states) if key not in s]
  if missing:
    raise ValueError(
        f'State {key!r} was produced by some replicas but not by {missing}. '
        'Every replica must initialize the same set of states.'
    )

  tiles = [s[key] for s in per_replica_states]
  global_shape = _global_shape_for_spec(np.shape(tiles[0]), spec, mesh)
  sharding = NamedSharding(mesh, spec)

  # Replica index -> device, matching the C-order walk over
  # `computation_shape` that `_logical_coordinates` uses.
  replica_of_device = {}
  for replica in range(len(tiles)):
    position = np.unravel_index(replica, computation_shape)
    replica_of_device[mesh.devices[position].id] = replica

  index_map = sharding.addressable_devices_indices_map(global_shape)

  # Group replicas by the shard they would write, and require agreement
  # within each group. For a fully distributed spec every group holds one
  # replica and this costs nothing; for a replicated spec it is the check
  # that the replicas really are interchangeable.
  by_shard: dict[str, list[int]] = {}
  for device, index in index_map.items():
    by_shard.setdefault(str(index), []).append(replica_of_device[device.id])

  for replicas in by_shard.values():
    reference = np.asarray(tiles[replicas[0]])
    for replica in replicas[1:]:
      other = np.asarray(tiles[replica])
      if not np.array_equal(reference, other):
        raise ValueError(
            f'State {key!r} differs between replicas {replicas[0]} and '
            f'{replica} (by up to '
            f'{float(np.max(np.abs(reference - other))):.6g}), but the '
            f'partition spec {spec} maps both onto the same piece of the '
            'global array, so one of them would be silently discarded. '
            'Declare a spec for this state in '
            '`params.additional_state_partition_specs` that distributes the '
            'axes along which the replicas differ.'
        )

  arrays = [
      jax.device_put(tiles[replica_of_device[device.id]], device)
      for device in index_map
  ]
  return jax.make_array_from_single_device_arrays(
      global_shape, sharding, arrays
  )


def _distribute_states(
    per_replica_states: list[dict[str, ScalarField]],
    mesh: Mesh,
    grid_params: gp_lib.GridParametrization | None = None,
    params: parameters_lib.SwirlLMParameters | None = None,
) -> dict[str, jax.Array]:
  """Assembles per-replica states into globally-sharded JAX arrays.

  Each replica produces its local state partition via `_init_fn`; `shard_map`
  consumes one global array per state, so each state's axes have to be mapped
  onto the mesh. That mapping comes from, in order of precedence: a caller
  supplied distributor, the boundary condition convention, a declared
  partition spec, the default for a 3D field, or -- failing all of those --
  replication, which is only valid if every replica agrees and is verified
  during assembly.

  Args:
    per_replica_states: List of state dicts, one per device, ordered by replica
      id.
    mesh: The JAX mesh defining device layout.
    grid_params: Optional grid parametrization.
    params: Optional SwirlLMParameters containing custom partition specs and
      distributors.

  Returns:
    A single state dict of globally-sharded JAX arrays.
  """
  all_keys = list(dict.fromkeys([k for s in per_replica_states for k in s]))  # pylint: disable=g-complex-comprehension
  computation_shape = tuple(mesh.shape[name] for name in mesh.axis_names)

  global_state: dict[str, jax.Array] = {}

  for key in all_keys:
    sample = next((s[key] for s in per_replica_states if key in s), None)
    if sample is None:
      continue

    # A caller-supplied distributor takes precedence over everything.
    if params is not None and key in params.additional_state_distributors:
      global_state[key] = params.additional_state_distributors[key](
          key, per_replica_states, mesh, grid_params
      )
      continue

    # Boundary condition states are not a plain tile placement: only the
    # replicas on the target face contribute, so they keep their own
    # assembly.
    if boundary_condition_utils.is_bc_key(key):
      global_state[key] = boundary_condition_utils.distribute_bc_state(
          key, per_replica_states, mesh, grid_params
      )
      continue

    if params is not None and key in params.additional_state_partition_specs:
      spec = params.additional_state_partition_specs[key]
    elif np.ndim(sample) == 3:
      # A 3D field is one local subdomain per replica.
      spec = P(*mesh.axis_names)
    else:
      # Nothing declares how this state maps onto the mesh, so the only
      # possibility left is that it is the same on every replica.
      spec = P()

    global_state[key] = _assemble_sharded_array(
        key, per_replica_states, spec, mesh, computation_shape
    )

  return global_state


def _shard_global_additional_states(
    additional_states: Mapping[str, Any],
    mesh: Mesh,
    params: parameters_lib.SwirlLMParameters,
) -> dict[str, jax.Array]:
  """Places pre-existing, globally assembled caller arrays onto devices.

  This function places pre-existing, globally assembled caller arrays onto
  devices, whereas `_distribute_states` stitches together per-core subdomain
  tiles.

  Args:
    additional_states: User-provided dictionary of additional state variables.
    mesh: The device mesh.
    params: Simulation parameters.

  Returns:
    A dictionary of sharded device arrays.
  """
  sharding_3d = NamedSharding(mesh, P(*mesh.axis_names))
  sharding_replicated = NamedSharding(mesh, P())
  sharded_states: dict[str, jax.Array] = {}
  for key, val in additional_states.items():
    if boundary_condition_utils.is_bc_key(key):
      spec = boundary_condition_utils.get_bc_partition_spec(
          key, mesh, params.grid_params
      )
      sharded_states[key] = jax.device_put(
          jnp.asarray(val), NamedSharding(mesh, spec)
      )
    elif key in params.additional_state_partition_specs:
      spec = params.additional_state_partition_specs[key]
      sharded_states[key] = jax.device_put(
          jnp.asarray(val), NamedSharding(mesh, spec)
      )
    elif np.ndim(val) == 0:
      sharded_states[key] = jax.device_put(
          jnp.asarray(val), sharding_replicated
      )
    elif np.ndim(val) == 3:
      sharded_states[key] = jax.device_put(jnp.asarray(val), sharding_3d)
    else:
      sharded_states[key] = jax.device_put(
          jnp.asarray(val), sharding_replicated
      )
  return sharded_states


def _stateless_update_if_present(
    mapping: dict[str, Any], updates: dict[str, Any]
) -> dict[str, Any]:
  """Returns a copy of `mapping` with only existing keys updated."""
  result = mapping.copy()
  result.update({key: val for key, val in updates.items() if key in result})
  return result


def _strengthen_dtypes(tree: Any) -> Any:
  """Strips `weak_type` from every array leaf of `tree`.

  JAX includes `weak_type` in the jit cache key, so a weakly typed input
  followed by a strongly typed one retraces and recompiles the whole cycle.
  State arrays come out of a typical `init_fn` weakly typed, because
  `jnp.arange(n) * dx` with a Python-float `dx` promotes an integer array
  against a Python scalar and yields a weak float; the cycle then returns
  those same fields strongly typed, having passed through `lax` collectives.
  Canonicalizing to strong types makes the cycle's signature stable, so it is
  compiled once.

  `lax.convert_element_type` preserves the sharding, the commitment and the
  values exactly, and is skipped for leaves that are already strong, for
  which it would return the same buffer anyway. It is also valid on tracers,
  so this can be applied to the scan carry inside the cycle, where it lowers
  to nothing.

  Args:
    tree: Any pytree; non-array leaves are returned untouched.

  Returns:
    The tree with every weakly typed array leaf replaced by its strongly
    typed equivalent.
  """

  def strengthen(leaf: Any) -> Any:
    if getattr(leaf, 'weak_type', False):
      return jax.lax.convert_element_type(leaf, leaf.dtype)
    return leaf

  return jax.tree.map(strengthen, tree)


def _update_additional_states(
    essential_states: ScalarFieldMap,
    additional_states: dict[str, ScalarField],
    step_id: jax.Array,
    params: parameters_lib.SwirlLMParameters,
    mesh: Mesh,
) -> dict[str, ScalarField]:
  """Updates additional_states at each time step.

  Mirrors TF `_update_additional_states`. Performs:
    1. Clear `src_*` source terms from the previous step.
    2. Update nonreflecting BC states.
    3. Update LPT states.
    4. Call user-defined `additional_states_update_fn`.

  Args:
    essential_states: The essential (prognostic) states.
    additional_states: The additional states to update.
    step_id: The current step index.
    params: The simulation parameters.
    mesh: JAX mesh for device topology.

  Returns:
    Updated additional_states dict.
  """
  del mesh

  updated = dict(additional_states)

  # 1. Clear source terms from the previous step.
  for varname in updated:
    if varname.startswith('src_'):
      updated[varname] = jnp.zeros_like(updated[varname])

  # 2. Update nonreflecting BC states.
  updated.update(
      nonreflecting_boundary.nonreflecting_bc_state_update_fn(
          params, essential_states, additional_states, step_id  # pyrefly: ignore[bad-argument-type]
      )
  )

  # 3. Update LPT states.
  # NOTE: LPT.step() requires (replica_id, replicas, states,
  # additional_states, step_id) which need multi-device context via
  # shard_map. When running inside shard_map, replica_id and replicas
  # are available from jax.lax.axis_index. This integration is deferred
  # until LPT is fully ported.
  # TODO(wqing): Integrate LPT step with shard_map context.

  # 4. Call user-defined additional_states_update_fn.
  if params.additional_states_update_fn is not None:
    fn = params.additional_states_update_fn
    try:
      sig = inspect.signature(fn)
      accepts_params = 'params' in sig.parameters or any(
          p.kind == inspect.Parameter.VAR_KEYWORD
          for p in sig.parameters.values()
      )
    except (ValueError, TypeError):
      accepts_params = False

    # `updated` is passed in rather than the original `additional_states`, and
    # the result is merged rather than substituted. Both matter: the user
    # function has to see the cleared source terms and the fresh nonreflecting
    # boundary condition states, and rebinding `updated` here would discard
    # steps 1 and 2 outright. The TF original gets away with the same
    # substitution because there `updated` is an alias of `additional_states`
    # and those steps mutate it in place; this port copies the dict, which
    # breaks that implicit hand-off.
    kwargs: dict[str, Any] = dict(
        states=essential_states,
        # A copy, so that a function which mutates its argument cannot reach
        # back into the state being assembled here.
        additional_states=dict(updated),
        step_id=step_id,
    )
    if accepts_params:
      kwargs['params'] = params
    updated.update(fn(**kwargs))

  return updated


def _process_at_step_id(
    process_fn: Callable[..., ScalarFieldMap],
    essential_states: dict[str, ScalarField],
    additional_states: dict[str, ScalarField],
    step_id: jax.Array,
    process_step_id: int,
    is_periodic: bool,
) -> tuple[dict[str, ScalarField], dict[str, ScalarField]]:
  """Executes `process_fn` conditionally depending on `step_id`.

  Mirrors TF `_process_at_step_id`.

  Args:
    process_fn: Function accepting `states` and `additional_states` kwargs,
      returning updated states.
    essential_states: The essential (prognostic) states.
    additional_states: The additional states.
    step_id: The current step id.
    process_step_id: Step id at which to trigger execution.
    is_periodic: If True, trigger whenever step_id % process_step_id == 0.

  Returns:
    Updated (essential_states, additional_states).
  """
  should_process = (
      (step_id % process_step_id == 0)
      if is_periodic
      else step_id == process_step_id
  )

  def _do_process(ess, add):
    updated_states = dict(process_fn(states=ess, additional_states=add))
    return (
        _stateless_update_if_present(ess, updated_states),
        _stateless_update_if_present(add, updated_states),
    )

  def _skip_process(ess, add):
    return ess, add

  essential_states, additional_states = jax.lax.cond(
      should_process,
      _do_process,
      _skip_process,
      essential_states,
      additional_states,
  )

  return essential_states, additional_states


def _restore_from_checkpoint(
    state: dict[str, jax.Array],
    restart_from: str,
    params: parameters_lib.SwirlLMParameters,
    essential_keys: Sequence[str],
    restart_fields: Sequence[str] | None = None,
) -> dict[str, object]:
  """Overrides `state` in place with the contents of a zarr checkpoint.

  A checkpoint holds the unique grid points, while a simulation state holds
  the sharded layout in which every replica carries its own halos. Each
  restored variable is therefore expanded back to that layout, using the
  axis classification recorded at save time, and then placed with the
  sharding of the array it replaces so that the state stays structurally
  identical to a non-restarted run and the compiled cycle is not retraced.

  Args:
    state: The initialized state, modified in place.
    restart_from: Path to the zarr checkpoint.
    params: Simulation parameters.
    essential_keys: The prognostic variable names, used for the default
      whitelist.
    restart_fields: Variables to restore. If None, the prognostic state, the
      accumulated simulation time and the boundary condition states are
      restored, and everything else in the checkpoint is skipped. Derived and
      grid-derived variables (`nu_t`, the coordinate fields, stretched-grid
      scale factors) are recomputed from the restart configuration, and
      restoring them from a checkpoint written against a different grid would
      corrupt them without any shape mismatch to catch it.

  Returns:
    The checkpoint metadata.

  Raises:
    ValueError: If a restored variable does not match the shape and dtype of
      the array it replaces.
  """
  ckpt_states, metadata = checkpoint_lib.load(restart_from)
  # Every process reads the store independently, so make sure they are all
  # done before anything downstream assumes a consistent view.
  if jax.process_count() > 1:
    multihost_utils.sync_global_devices('swirl_lm_checkpoint_loaded')

  if restart_fields is None:
    restart_fields = (
        list(essential_keys)
        + [TIME_VARNAME]
        + [k for k in ckpt_states if boundary_condition_utils.is_bc_key(k)]
    )
  wanted = set(restart_fields)

  variable_attrs = metadata.get('variable_attrs', {})
  restored = []
  skipped = {}
  for key in sorted(ckpt_states):
    if key not in state:
      skipped[key] = 'not in the initialized state'
      continue
    if key not in wanted:
      skipped[key] = 'not in restart_fields'
      continue

    target = state[key]
    value = np.asarray(ckpt_states[key])

    if value.ndim:
      # The outer halos of a non-periodic axis have no neighbour to come
      # from, so reuse the initialized values, which are consistent with the
      # boundary conditions. In a multi-process job the initialized array
      # cannot be gathered, and the first boundary condition application
      # corrects them instead.
      existing = (
          np.asarray(target)
          if getattr(target, 'is_fully_addressable', True)
          else None
      )
      value = global_layout.inflate_to_global(
          value,
          params.grid_params,
          target_shape=target.shape,
          stripped_axes=variable_attrs.get(key, {}).get(
              checkpoint_lib.STRIPPED_AXES_ATTR
          ),
          existing=existing,
      )

    if value.shape != target.shape or value.dtype != target.dtype:
      raise ValueError(
          f'Restored {key!r} has shape {value.shape} and dtype '
          f'{value.dtype}, but the initialized state has {target.shape} and '
          f'{target.dtype}. Check that the restart configuration matches the '
          'one the checkpoint was written with.'
      )

    if value.ndim == 0:
      state[key] = jax.device_put(value, target.sharding)
    else:
      # Only the addressable shards are requested, so this is also correct
      # when the state spans several processes.
      state[key] = jax.make_array_from_callback(
          target.shape, target.sharding, lambda index, v=value: v[index]
      )
    restored.append(key)

  logging.info(
      'Restarted from checkpoint %s (step %s, sim_time %s). Restored: %s',
      restart_from,
      metadata.get('step', '?'),
      metadata.get('sim_time', '?'),
      restored,
  )
  for key, reason in skipped.items():
    logging.info('Checkpoint variable %r not restored: %s', key, reason)

  return metadata


def _save_checkpoint(
    states: dict[str, ScalarField],
    grid_params: gp_lib.GridParametrization,
    step: int,
    checkpoint_dir: str,
) -> None:
  """Saves a checkpoint to the given directory.

  Args:
    states: Flow field variables.
    grid_params: Grid parametrization for coordinate metadata.
    step: Current simulation step number.
    checkpoint_dir: Directory for zarr checkpoints.
  """
  ckpt_path = os.path.join(checkpoint_dir, f'step_{step:06d}.zarr')
  # Prefer the accumulated time carried in the state over `step * dt`, which
  # is only equal to it while `dt` never changes.
  sim_time = (
      float(states[TIME_VARNAME])
      if TIME_VARNAME in states
      else step * grid_params.dt
  )
  checkpoint_lib.save(
      states,
      grid_params,
      step=step,
      path=ckpt_path,
      sim_time=sim_time,
  )
  logging.info('Checkpoint saved: %s', ckpt_path)


def run_simulation(
    params: parameters_lib.SwirlLMParameters,
    init_fn: InitFn,
    num_steps: int,
    mesh: Mesh | None = None,
    additional_states: ScalarFieldMap | None = None,
    restart_from: str | None = None,
    restart_fields: Sequence[str] | None = None,
    num_steps_per_cycle: int = 1,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 0,
) -> dict[str, ScalarField]:
  """Runs the simulation for the given number of steps.

  Mirrors TF `solver` + `solver_loop`. Structure:
    - Create JAX mesh from params.cx/cy/cz (if not provided).
    - Initialize per-device states via `_init_fn` with logical_coordinates.
    - Distribute states across devices using NamedSharding.
    - Run cycles, each calling `_one_cycle` (JIT + shard_map compiled).
    - IO/checkpointing happens between cycles on host.

  The step body runs inside `shard_map` to provide named-axis context for
  collective ops (ppermute, axis_index, psum). This mirrors TF's
  `strategy.run(step_fn, args=(init_state,))`.

  Args:
    params: The simulation parameters.
    init_fn: A callable `(params, logical_coordinates) -> dict[str,
      ScalarField]` that returns per-device initial states. The
      `logical_coordinates` is a `(cx_i, cy_i, cz_i)` tuple identifying the
      device's position in the computational grid, matching TF's
      `init_fn(replica_id, coordinates)` pattern.
    num_steps: Total number of time steps to run.
    mesh: Optional JAX mesh for device topology. If None, it is created
      automatically from `params.cx/cy/cz` via `create_mesh(params)`.
    additional_states: Optional helper variables (boundary conditions, stretched
      grid scale factors, etc.). Empty dict if None.
    restart_from: Path to a zarr checkpoint to restart from. Variables present
      in the checkpoint override those from `init_fn`.
    restart_fields: Variables to restore when restarting. If None, the
      prognostic state, the accumulated simulation time and the boundary
      condition states are restored; derived and grid-derived variables are
      recomputed from this configuration instead.
    num_steps_per_cycle: Number of time steps per cycle. Higher values reduce
      Python loop overhead but delay IO. Default is 1.
    checkpoint_dir: Directory for zarr checkpoints. If None, no checkpointing.
    checkpoint_interval: Save a checkpoint every N cycles. 0 = no periodic
      saves.

  Returns:
    The final states dict (globally-sharded arrays).

  Raises:
    RuntimeError: If the simulation diverges (non-finite values in velocity).
  """
  # Create mesh from params if not provided.
  if mesh is None:
    mesh = create_mesh(params)

  # Initialize per-device states.
  coords = _logical_coordinates(params)
  per_replica_states = []
  for lc in coords:
    device_state = _init_fn(params, lc, customized_init_fn=init_fn)
    per_replica_states.append(device_state)

  # Distribute across devices.
  state = _distribute_states(
      per_replica_states, mesh, params.grid_params, params=params
  )

  # Merge in user-provided additional_states.
  if additional_states is not None:
    state.update(
        _shard_global_additional_states(additional_states, mesh, params)
    )

  # Create simulation model.
  sim = simulation_lib.Simulation(params)
  gp = params.grid_params

  # Get key categorization.
  essential_keys, additional_keys, helper_var_keys = _get_state_keys(params)

  # Log key overlap info.
  state_keys = set(state.keys())
  declared_keys = (
      set(essential_keys) | set(additional_keys) | set(helper_var_keys)
  )
  logging.info(
      'Keys in state but not in params: %s',
      sorted(state_keys - declared_keys),
  )
  logging.info(
      'Keys in params but not in state: %s',
      sorted(declared_keys - state_keys),
  )

  # Override with checkpoint data if restarting. This runs before the
  # shard_map specs are derived from `state`, so the restored arrays are what
  # those specs describe.
  if restart_from is not None:
    _restore_from_checkpoint(
        state, restart_from, params, essential_keys, restart_fields
    )

  # Canonicalize the dtypes of everything that enters the cycle. This is the
  # only place that sees the merged state, so one call covers `init_fn` via
  # `_distribute_states`, caller-supplied `additional_states` via
  # `_shard_global_additional_states`, and the restart override. It must cover
  # every leaf, including the 0D `TIME_VARNAME` and the boundary condition and
  # helper variables: a single weakly typed leaf left behind costs the same
  # extra compilation as all of them. For a weak leaf this is a real device
  # copy, paid once here rather than as a second compilation of the cycle.
  state = _strengthen_dtypes(state)

  # Build the JIT-compiled cycle function with shard_map.
  # The cycle function runs `num_steps_per_cycle` time steps. Inside each step:
  #   1. Split state -> essential + additional
  #   2. Preprocess (conditional on step_id)
  #   3. _update_additional_states
  #   4. sim.step
  #   5. Refresh the additional states from the solver's output
  #   6. Postprocess (conditional on step_id)
  #   7. TIME_VARNAME += dt
  #   8. Merge back into flat state
  #
  # This matches the TF `_one_cycle` structure. The step body runs inside
  # shard_map so that collective ops (ppermute, axis_index, psum) have
  # named-axis context, matching TF's `strategy.run(step_fn, ...)`.

  # Build shard_map specs: rank-3 fields get P(*axis_names).
  # Boundary condition and helper states get their model-specific PartitionSpec.
  # All other ranks (scalars, 1D arrays) are replicated.
  all_state_keys = sorted(state.keys())
  spec_3d = P(*mesh.axis_names)
  spec_replicated = P()
  in_specs = {}
  out_specs = {}
  for key in all_state_keys:
    if boundary_condition_utils.is_bc_key(key):
      spec = boundary_condition_utils.get_bc_partition_spec(
          key, mesh, params.grid_params
      )
      in_specs[key] = spec
      out_specs[key] = spec
    elif key in params.additional_state_partition_specs:
      spec = params.additional_state_partition_specs[key]
      in_specs[key] = spec
      out_specs[key] = spec
    elif np.ndim(state[key]) == 3:
      in_specs[key] = spec_3d
      out_specs[key] = spec_3d
    else:
      in_specs[key] = spec_replicated
      out_specs[key] = spec_replicated

  def _shard_map_cycle_body(
      state: dict[str, ScalarField],
      init_step_id: jax.Array,
  ) -> dict[str, ScalarField]:
    """Step body that runs inside shard_map (per-device)."""

    def step_body(
        carry: tuple[dict[str, ScalarField], jax.Array],
        _: None,
    ) -> tuple[tuple[dict[str, ScalarField], jax.Array], None]:
      state_carry, step_id = carry

      # Split state into essential and additional.
      ess = {k: state_carry[k] for k in essential_keys if k in state_carry}
      add = {k: v for k, v in state_carry.items() if k not in ess}

      # 1. Preprocess.
      if params.apply_preprocess and params.preprocessing_states_update_fn:
        ess, add = _process_at_step_id(
            process_fn=params.preprocessing_states_update_fn,
            essential_states=ess,
            additional_states=add,
            step_id=step_id,
            process_step_id=params.preprocess_step_id,
            is_periodic=params.preprocess_periodic,
        )

      # 2. Update additional states.
      add = _update_additional_states(ess, add, step_id, params, mesh)

      # 3. Run one simulation step.
      updated_state = sim.step(ess, add, mesh)

      # 4. Pull the solver's results back into `add` before anything else
      # reads or writes it. `sim.step` owns some of the additional states --
      # `nu_t`, `drho`, the buoyancy diagnostics -- and returns the updated
      # value for those while returning the unchanged input for the rest, so
      # this refresh is what makes `add` current. Without it the stale
      # pre-step `add` is written back over the solver's output in step 6
      # below, which is how `nu_t` stayed identically zero for a whole run
      # despite `use_sgs: true`, and with it `tke`, which is derived from the
      # stored `nu_t` by the combustion model.
      add = _stateless_update_if_present(add, updated_state)

      # 5. Postprocess.
      if params.apply_postprocess and params.postprocessing_states_update_fn:
        # Split updated_state back into essential/additional.
        ess = _stateless_update_if_present(ess, updated_state)

        ess, add = _process_at_step_id(
            process_fn=params.postprocessing_states_update_fn,
            essential_states=ess,
            additional_states=add,
            step_id=step_id,
            process_step_id=params.postprocess_step_id,
            is_periodic=params.postprocess_periodic,
        )

        # Merge back.
        updated_state = _stateless_update_if_present(updated_state, ess)

      # 6. Merge the additional states back into updated_state. `add` is
      # current as of step 4, so this no longer discards the solver's output.
      updated_state.update(add)

      # 7. Accumulate simulation time.
      updated_state[TIME_VARNAME] = state_carry[TIME_VARNAME] + jnp.float64(
          params.dt
      )

      # 8. Merge updates back into state (pass-through keys like helper vars).
      # The carry is re-canonicalized because step 3 above is the only step
      # not wrapped in `lax.cond` (which would unify the types itself), so a
      # user `additional_states_update_fn` can hand back a weakly typed value
      # -- `jnp.where(x > 0.0, 1.0, 0.0)` does exactly that -- and `lax.scan`
      # does not promote a weak carry. That would flip the cycle's output
      # aval and recompile it. This lowers to nothing when the types already
      # match.
      new_state = _strengthen_dtypes(
          _stateless_update_if_present(state_carry, updated_state)
      )

      return (new_state, step_id + 1), None

    init_carry = (state, init_step_id)
    (final_state, _), _ = jax.lax.scan(
        step_body, init_carry, None, length=num_steps_per_cycle
    )
    return final_state

  # Build the JIT-compiled shard_map function.
  # init_step_id is a scalar, replicated across all devices.
  _jit_one_cycle = jax.jit(
      shard_map.shard_map(
          _shard_map_cycle_body,
          mesh=mesh,
          in_specs=(in_specs, spec_replicated),
          out_specs=out_specs,
          check_rep=False,
      )
  )

  # Compute cycle counts. TF uses exact `num_steps * num_cycles` — no remainder.
  num_cycles = num_steps // max(num_steps_per_cycle, 1)

  logging.info(
      'Simulation: %d total steps, %d steps/cycle, %d cycles, mesh=%s.',
      num_steps,
      num_steps_per_cycle,
      num_cycles,
      dict(mesh.shape),
  )

  # Main simulation loop (Python cycle loop).
  steps_completed = 0

  for cycle_idx in range(num_cycles):
    init_step_id = jnp.int32(steps_completed)

    # Time the compute portion of each cycle.
    compute_start = time.time()
    state = _jit_one_cycle(state, init_step_id)
    # Block until computation completes (for accurate timing).
    jax.block_until_ready(state)
    compute_time = time.time() - compute_start

    steps_completed += num_steps_per_cycle

    # Check for divergence (NaN/Inf in velocity fields).
    for vel_key in ('u', 'v', 'w'):
      if vel_key in state and not jnp.all(jnp.isfinite(state[vel_key])):
        if checkpoint_dir:
          _save_checkpoint(state, gp, steps_completed, checkpoint_dir)
          logging.error(
              'Non-finite values detected in %s at step %d. Checkpoint saved.',
              vel_key,
              steps_completed,
          )
        raise RuntimeError(
            f'Simulation diverged: non-finite values in {vel_key} at step'
            f' {steps_completed}.'
        )

    # Per-cycle logging with timing info.
    io_start = time.time()
    if num_steps_per_cycle > 0:
      gs = params.grid_spacings
      cfl_field = (
          jnp.abs(state['u']) / gs[0]
          + jnp.abs(state['v']) / gs[1]
          + jnp.abs(state['w']) / gs[2]
      )
      cfl = float(params.dt * jnp.max(cfl_field))
      logging.info(
          'Cycle %d: step %d / %d  CFL=%.4f  compute_time=%.2fs',
          cycle_idx,
          steps_completed,
          num_steps,
          cfl,
          compute_time,
      )

    # Checkpointing.
    if (
        checkpoint_dir
        and checkpoint_interval > 0
        and (cycle_idx + 1) % checkpoint_interval == 0
    ):
      _save_checkpoint(state, gp, steps_completed, checkpoint_dir)

    io_time = time.time() - io_start
    logging.info(
        'Cycle %d: io_time=%.2fs',
        cycle_idx,
        io_time,
    )

  # Save final checkpoint. `steps_completed` rather than `num_steps`: the
  # cycle loop runs `num_steps // num_steps_per_cycle` cycles, so the two
  # differ whenever that division has a remainder, and `num_steps` would
  # label the file with steps that were never taken.
  if checkpoint_dir:
    _save_checkpoint(state, gp, steps_completed, checkpoint_dir)

  logging.info('Simulation completed after %d steps.', steps_completed)
  return state
