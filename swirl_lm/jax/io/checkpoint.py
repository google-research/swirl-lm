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
"""Checkpoint utilities: save/load simulation state as xarray/zarr.

This module provides a clean interface for persisting simulation states.
Each state dict `{name: jax.Array}` is stored as an `xarray.Dataset` in
zarr format, with named dimensions and physical coordinates from the grid.

Key features:
  - Named dimensions (`x`, `y`, `z`) with physical coordinate values.
  - Metadata (step, time, grid config) stored in zarr attrs.
  - Halo-aware: saves only interior points by default.
  - Selective field loading via xarray lazy access.
  - Cloud-compatible (zarr works with local, GCS, etc.).

Usage:
  ```python
  from swirl_lm.jax.io import checkpoint

  # Save.
  checkpoint.save(states, grid_params, step=100, path='/tmp/ckpt_100.zarr')

  # Load.
  states, metadata = checkpoint.load('/tmp/ckpt_100.zarr')
  ```
"""


from collections.abc import Sequence

import jax
from jax.experimental import multihost_utils
import numpy as np
from swirl_lm.jax.utility import global_layout
from swirl_lm.jax.utility import grid_parametrization as gp_lib
import xarray as xr


def _interior_coords(
    grid_params: gp_lib.GridParametrization,
    exclude_halos: bool = True,
) -> dict[str, np.ndarray]:
  """Builds physical coordinate arrays in data axis order.

  Note: `global_xyz` already excludes halo points, while
  `global_xyz_with_halos` includes them.

  Args:
    grid_params: Grid parametrization with global coordinate info.
    exclude_halos: If True, return interior coordinates only. If False, return
      coordinates including the halo points, so that fields saved with their
      halos still align with the coordinates.

  Returns:
    Dict mapping dimension name ('x', 'y', 'z') to 1D coordinate arrays.
  """
  global_coords = (
      grid_params.global_xyz
      if exclude_halos
      else grid_params.global_xyz_with_halos
  )
  coords = {}
  for i, axis_name in enumerate(grid_params.data_axis_order):
    coords[axis_name] = np.asarray(global_coords[i])
  return coords


def _data_axis_dims(
    grid_params: gp_lib.GridParametrization,
) -> tuple[str, ...]:
  """Returns dimension names in data axis order."""
  return tuple(grid_params.data_axis_order)


def _to_numpy(arr: jax.Array) -> np.ndarray:
  """Materializes a possibly-sharded state array on the host.

  States are global arrays sharded over the device mesh. Within a single
  process all shards are addressable and `np.asarray` gathers them. In a
  multi-process job (multi-host TPU pod or multi-node GPU) 3D fields span
  non-addressable devices, so an explicit cross-process gather is needed.
  This is a collective: every process must call it for every variable.

  Args:
    arr: A state array, sharded or not.

  Returns:
    The full (global) value as a numpy array.
  """
  if jax.process_count() > 1 and not getattr(arr, 'is_fully_addressable', True):
    return np.asarray(multihost_utils.process_allgather(arr, tiled=True))
  return np.asarray(arr)


# Attribute names recording, per variable, how it was laid out on the way out,
# so that restart never has to infer it. Inference by length is genuinely
# ambiguous: a boundary condition state's thin normal axis has extent
# `halo_width + 1`, which equals the global interior length whenever that axis
# has one core and `grid_size == 3 * halo_width + 1`.
GLOBAL_SHAPE_ATTR = 'swirl_lm_global_shape'
STRIPPED_AXES_ATTR = 'swirl_lm_stripped_axes'

# Bumped when the on-disk layout changes in a way readers must notice.
FORMAT_VERSION = 1


def _as_data_array(
    name: str,
    arr: np.ndarray,
    dims: tuple[str, ...],
    coords: dict[str, np.ndarray],
    grid_params: gp_lib.GridParametrization,
    exclude_halos: bool,
) -> xr.DataArray:
  """Wraps a single state array as a `DataArray`.

  Two things make this more than a `DataArray` constructor call.

  First, a global 3D field is the concatenation of the per-replica arrays,
  each carrying its own halos, so its extent along an axis is
  `num_cores * (core_n + 2 * halo_width)` and the halo slabs at the interior
  seams are duplicates. Those are discarded here, leaving the unique points,
  which is what aligns with the grid coordinates.

  Second, not every entry of a state dict is a full field: `simulation_time`
  is a 0D scalar, stretched-grid scale factors are broadcastable (e.g.
  `(n, 1, 1)` per replica), and boundary condition states are thin planes.
  Axes that do not carry the sharded layout are passed through, and any axis
  whose length does not match a grid coordinate is given a shared,
  size-suffixed dimension name so that equally shaped variables still share
  dimensions and stay comparable.

  Args:
    name: Variable name, used only for messages.
    arr: The variable's data, in the global sharded layout.
    dims: Grid dimension names in data axis order.
    coords: Physical coordinates keyed by the names in `dims`.
    grid_params: Grid parametrization describing the sharded layout.
    exclude_halos: If True, keep the interior only. If False, additionally
      keep the halos at the outer domain boundaries. Either way the
      duplicated interior seams are discarded.

  Returns:
    A `DataArray` carrying grid coordinates on every axis of grid extent, and
    the layout attributes needed to restore the sharded array.
  """
  del name  # Reserved for messages; none are raised here.
  # Scalars (e.g. `simulation_time`) are stored as dimensionless variables.
  if arr.ndim == 0:
    return xr.DataArray(arr)

  global_shape = list(arr.shape)
  stripped_axes = global_layout.stripped_axes_for(arr, grid_params)
  arr = global_layout.strip_halos_global(
      arr, grid_params, include_halos=not exclude_halos
  )

  # Anything that is not 3D cannot be mapped onto the grid axes; keep it on
  # shared, size-suffixed dimensions so that it still round-trips.
  if arr.ndim != 3:
    out_dims = tuple(f'dim{i}_{n}' for i, n in enumerate(arr.shape))
    used_coords = {}
  else:
    out_dims = tuple(
        dim_name if arr.shape[axis] == len(coords[dim_name])
        else f'{dim_name}_{arr.shape[axis]}'
        for axis, dim_name in enumerate(dims)
    )
    used_coords = {d: coords[d] for d in out_dims if d in coords}

  data_array = xr.DataArray(arr, dims=out_dims, coords=used_coords)
  data_array.attrs[GLOBAL_SHAPE_ATTR] = global_shape
  data_array.attrs[STRIPPED_AXES_ATTR] = list(stripped_axes)
  return data_array


def save(
    states: dict[str, jax.Array],
    grid_params: gp_lib.GridParametrization,
    step: int,
    path: str,
    sim_time: float = 0.0,
    exclude_halos: bool = True,
    fields: Sequence[str] | None = None,
) -> None:
  """Saves simulation state as an xarray Dataset in zarr format.

  Args:
    states: Flow field variables, e.g. {'rho': array, 'u': array, ...}.
    grid_params: Grid parametrization for coordinate metadata.
    step: Current simulation step number.
    path: Output path for the zarr store (directory).
    sim_time: Current simulation time.
    exclude_halos: If True, keep the interior only. If False, additionally
      keep the halos at the outer domain boundaries. Either way the halo
      slabs duplicated at the interior device seams are discarded, and axes
      that do not carry the sharded layout (scalars, thin or broadcastable
      variables) are preserved intact.
    fields: Optional list of field names to save. If None, saves all.
  """
  hw = grid_params.halo_width
  coords = _interior_coords(grid_params, exclude_halos)
  dims = _data_axis_dims(grid_params)

  # `_to_numpy` is a collective under multi-process, so every process must
  # walk the same keys in the same order, even though only one writes.
  keys = sorted(fields if fields is not None else states.keys())
  data_vars = {}
  for name in keys:
    data_vars[name] = _as_data_array(
        name, _to_numpy(states[name]), dims, coords, grid_params,
        exclude_halos,
    )

  # In a multi-process job every process holds the same gathered data; only
  # one of them may write to the store.
  if jax.process_index() == 0:
    ds = xr.Dataset(data_vars)
    ds.attrs['format_version'] = FORMAT_VERSION
    ds.attrs['step'] = step
    ds.attrs['sim_time'] = sim_time
    ds.attrs['halo_width'] = hw
    ds.attrs['exclude_halos'] = int(exclude_halos)
    ds.attrs['data_axis_order'] = ''.join(grid_params.data_axis_order)
    # Per-core grid size, including halos.
    ds.attrs['nx'] = grid_params.nx
    ds.attrs['ny'] = grid_params.ny
    ds.attrs['nz'] = grid_params.nz
    # Number of cores per axis, and the resulting global interior size.
    ds.attrs['cx'] = grid_params.cx
    ds.attrs['cy'] = grid_params.cy
    ds.attrs['cz'] = grid_params.cz
    ds.attrs['fx'] = grid_params.fx
    ds.attrs['fy'] = grid_params.fy
    ds.attrs['fz'] = grid_params.fz
    ds.attrs['lx'] = grid_params.lx
    ds.attrs['ly'] = grid_params.ly
    ds.attrs['lz'] = grid_params.lz
    ds.attrs['dt'] = grid_params.dt

    # TODO(karlt): one chunk per variable is written, which is ~4 GB per field
    # at 1024^3. Expose a `chunks` argument.
    ds.to_zarr(path, mode='w')

  # The writer must be done before any process reads the store back.
  if jax.process_count() > 1:
    multihost_utils.sync_global_devices('swirl_lm_checkpoint_saved')


def load(
    path: str,
    fields: Sequence[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
  """Loads simulation state from a zarr checkpoint.

  Host numpy arrays are returned rather than JAX arrays, deliberately. The
  values still have to be converted back to the sharded layout before they
  can enter a simulation state, which is host-side work; and building JAX
  arrays here would both put a full global field on the default device and
  silently downcast float64 when x64 is disabled. See
  `driver.run_simulation`'s restart path, which places them with the
  sharding of the array each one replaces.

  Args:
    path: Path to the zarr store.
    fields: Optional list of field names to load. If None, loads all.

  Returns:
    A tuple of (states, metadata) where states maps names to numpy arrays,
    each carrying the per-variable layout attributes, and metadata contains
    step, sim_time, and grid info from the store attributes.
  """
  ds = xr.open_zarr(path)
  keys = fields if fields is not None else list(ds.data_vars)
  states = {name: np.asarray(ds[name].values) for name in keys}

  metadata = dict(ds.attrs)
  # Carry the per-variable layout attributes alongside the store's own, so
  # that the caller can restore the sharded layout without guessing.
  metadata['variable_attrs'] = {
      name: dict(ds[name].attrs) for name in keys
  }
  ds.close()
  return states, metadata


def load_as_dataset(path: str) -> xr.Dataset:
  """Loads a checkpoint as an xarray Dataset for interactive analysis.

  This is useful for Colab/notebook workflows where you want to use
  xarray's built-in plotting and analysis tools directly.

  Args:
    path: Path to the zarr store.

  Returns:
    The xarray Dataset with all variables, coordinates, and metadata.
  """
  return xr.open_zarr(path)
