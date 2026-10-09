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
"""Visualization utilities for Swirl-LM simulation results.

Provides functions for plotting 2D slices of 3D flow fields. Works with
both in-memory JAX arrays (during a simulation) and xarray Datasets
(loaded from zarr checkpoints).

All plotting functions return matplotlib Figure objects so they can be
displayed inline in Colab or saved to disk.

Usage:
  ```python
  from swirl_lm.jax.io import visualization as viz

  # From live states dict.
  fig = viz.plot_slice(states, 'u', axis='z', index=8, grid_params=gp)
  fig.savefig('u_slice.png')

  # From a zarr checkpoint.
  ds = checkpoint.load_as_dataset('/tmp/ckpt_100.zarr')
  fig = viz.plot_dataset_slice(ds, 'u', axis='z', index=4)

  # Velocity magnitude slice.
  fig = viz.plot_velocity_magnitude(states, axis='z', index=8, grid_params=gp)

  # Multi-panel overview of all velocity components.
  fig = viz.plot_overview(states, grid_params=gp, axis='z', index=8)
  ```
"""


from collections.abc import Sequence

from etils import epath
import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from swirl_lm.jax.utility import global_layout
from swirl_lm.jax.utility import grid_parametrization as gp_lib
import xarray as xr

# Use a non-interactive backend for headless environments.
matplotlib.use('Agg')

# Default colormap for flow quantities.
_DEFAULT_CMAP = 'RdBu_r'
_MAGNITUDE_CMAP = 'viridis'


def _to_interior(
    arr: np.ndarray,
    grid_params: gp_lib.GridParametrization | None,
) -> np.ndarray:
  """Converts a global sharded field to its contiguous interior points.

  A global field is the concatenation of the per-replica arrays, each with
  its own halos, so simply dropping `halo_width` points from each end would
  leave the duplicated halo slabs at every interior device seam in place and
  misreport the extent. `strip_halos_global` discards them, giving an array
  whose axes line up with `global_xyz`.

  Args:
    arr: 3D array in data axis order, or any array when `grid_params` is
      None.
    grid_params: Grid parametrization, or None to leave `arr` untouched, for
      callers that pass arrays which are already interior-only.

  Returns:
    The interior array. Axes that do not carry the sharded layout, including
    those of an already-stripped array, are passed through.
  """
  if grid_params is None:
    return arr
  return global_layout.strip_halos_global(arr, grid_params)


def _get_slice(
    arr: np.ndarray,
    axis: str,
    index: int,
    data_axis_order: tuple[str, ...],
) -> np.ndarray:
  """Extracts a 2D slice from a 3D array.

  Args:
    arr: 3D numpy array in data_axis_order, interior only.
    axis: The axis to slice along ('x', 'y', or 'z').
    index: The index along that axis.
    data_axis_order: Tuple like ('x', 'y', 'z') specifying array dim order.

  Returns:
    2D numpy array.
  """
  axis_idx = list(data_axis_order).index(axis)
  return np.take(arr, index, axis=axis_idx)


def _get_plane_axes(
    axis: str,
    data_axis_order: tuple[str, ...],
) -> tuple[str, str]:
  """Returns the two axes of the plane perpendicular to `axis`."""
  remaining = [a for a in data_axis_order if a != axis]
  return remaining[0], remaining[1]


def _get_coords(
    grid_params: gp_lib.GridParametrization,
    axis: str,
) -> tuple[np.ndarray, np.ndarray, str, str]:
  """Returns coordinate arrays and axis labels for the slicing plane.

  Note: `global_xyz` already excludes halo points.

  Args:
    grid_params: Grid parametrization.
    axis: The axis being sliced ('x', 'y', or 'z').

  Returns:
    (coord_0, coord_1, label_0, label_1) for the two in-plane axes.
  """
  dao = grid_params.data_axis_order
  ax0, ax1 = _get_plane_axes(axis, dao)

  idx0 = list(dao).index(ax0)
  idx1 = list(dao).index(ax1)

  c0 = np.asarray(grid_params.global_xyz[idx0])
  c1 = np.asarray(grid_params.global_xyz[idx1])

  return c0, c1, ax0, ax1


def plot_slice(
    states: dict[str, jax.Array],
    field_name: str,
    axis: str = 'z',
    index: int | None = None,
    grid_params: gp_lib.GridParametrization | None = None,
    cmap: str = _DEFAULT_CMAP,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
  """Plots a 2D slice of a 3D field.

  Args:
    states: Dict of flow field arrays.
    field_name: Key into states dict (e.g. 'u', 'v', 'p').
    axis: Axis to slice along ('x', 'y', 'z').
    index: Slice index (interior-relative). Defaults to midplane.
    grid_params: If provided, uses physical coordinates on axes.
    cmap: Matplotlib colormap name.
    title: Plot title. Defaults to `field_name` slice info.
    figsize: Figure size in inches.

  Returns:
    Matplotlib Figure object.
  """
  # Strip first, so that the index is interpreted against the interior.
  arr = _to_interior(np.asarray(states[field_name]), grid_params)

  if grid_params is not None:
    dao = tuple(grid_params.data_axis_order)
  else:
    dao = ('x', 'y', 'z')

  # Default to midplane.
  axis_idx = list(dao).index(axis)
  if index is None:
    index = arr.shape[axis_idx] // 2

  slice_2d = _get_slice(arr, axis, index, dao)

  fig, ax = plt.subplots(1, 1, figsize=figsize)

  if grid_params is not None:
    c0, c1, label0, label1 = _get_coords(grid_params, axis)
    extent = [c0[0], c0[-1], c1[0], c1[-1]]
    im = ax.imshow(
        slice_2d.T,
        origin='lower',
        cmap=cmap,
        aspect='equal',
        extent=extent,
    )
    ax.set_xlabel(label0)
    ax.set_ylabel(label1)
  else:
    im = ax.imshow(slice_2d.T, origin='lower', cmap=cmap, aspect='equal')

  fig.colorbar(im, ax=ax)

  if title is None:
    title = f'{field_name} (slice {axis}={index})'
  ax.set_title(title)
  fig.tight_layout()
  return fig


def plot_dataset_slice(
    ds: xr.Dataset,
    field_name: str,
    axis: str = 'z',
    index: int | None = None,
    cmap: str = _DEFAULT_CMAP,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
  """Plots a 2D slice from an xarray Dataset (loaded from zarr checkpoint).

  Args:
    ds: xarray Dataset with named dimensions and coordinates.
    field_name: Variable name in the dataset.
    axis: Axis to slice along ('x', 'y', 'z').
    index: Slice index. Defaults to midplane.
    cmap: Matplotlib colormap name.
    figsize: Figure size in inches.

  Returns:
    Matplotlib Figure object.
  """
  da = ds[field_name]
  n = da.sizes[axis]
  if index is None:
    index = n // 2

  slice_2d = da.isel({axis: index})

  fig, ax = plt.subplots(1, 1, figsize=figsize)
  slice_2d.plot(ax=ax, cmap=cmap)
  step = ds.attrs.get('step', '?')
  ax.set_title(f'{field_name} ({axis}={index}, step={step})')
  fig.tight_layout()
  return fig


def plot_velocity_magnitude(
    states: dict[str, jax.Array],
    axis: str = 'z',
    index: int | None = None,
    grid_params: gp_lib.GridParametrization | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
  """Plots the velocity magnitude |u| = sqrt(u^2 + v^2 + w^2) on a 2D slice.

  Args:
    states: Dict with 'u', 'v', 'w' fields.
    axis: Axis to slice along.
    index: Slice index (interior-relative). Defaults to midplane.
    grid_params: If provided, uses physical coordinates.
    figsize: Figure size.

  Returns:
    Matplotlib Figure object.
  """
  u = np.asarray(states['u'])
  v = np.asarray(states['v'])
  w = np.asarray(states['w'])
  vmag = np.sqrt(u**2 + v**2 + w**2)
  vmag_states = {'|u|': vmag}
  return plot_slice(
      vmag_states,
      '|u|',
      axis=axis,
      index=index,
      grid_params=grid_params,
      cmap=_MAGNITUDE_CMAP,
      title=f'Velocity magnitude (slice {axis}={index})',
      figsize=figsize,
  )


def plot_overview(
    states: dict[str, jax.Array],
    grid_params: gp_lib.GridParametrization | None = None,
    axis: str = 'z',
    index: int | None = None,
    fields: Sequence[str] = ('u', 'v', 'w', 'p'),
    figsize: tuple[float, float] = (16, 12),
    save_path: str | None = None,
) -> plt.Figure:
  """Plots a multi-panel overview of selected fields on a 2D slice.

  Creates a 2x2 (or Nx1 for fewer fields) grid of subplots showing
  slices of the specified fields.

  Args:
    states: Dict of flow field arrays.
    grid_params: If provided, uses physical coordinates.
    axis: Axis to slice along.
    index: Slice index. Defaults to midplane.
    fields: Field names to plot.
    figsize: Figure size.
    save_path: Optional path to save the figure to. If provided, the figure will
      be saved to this path.

  Returns:
    Matplotlib Figure object.
  """
  n = len(fields)
  ncols = min(n, 2)
  nrows = (n + ncols - 1) // ncols

  if grid_params is not None:
    dao = tuple(grid_params.data_axis_order)
  else:
    dao = ('x', 'y', 'z')

  fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

  for i, fname in enumerate(fields):
    row, col = divmod(i, ncols)
    ax_plot = axes[row, col]

    arr = _to_interior(np.asarray(states[fname]), grid_params)
    axis_idx = list(dao).index(axis)
    idx = index if index is not None else arr.shape[axis_idx] // 2

    slice_2d = _get_slice(arr, axis, idx, dao)

    if grid_params is not None:
      c0, c1, label0, label1 = _get_coords(grid_params, axis)
      extent = [c0[0], c0[-1], c1[0], c1[-1]]
      im = ax_plot.imshow(
          slice_2d.T,
          origin='lower',
          cmap=_DEFAULT_CMAP,
          aspect='equal',
          extent=extent,
      )
      ax_plot.set_xlabel(label0)
      ax_plot.set_ylabel(label1)
    else:
      im = ax_plot.imshow(
          slice_2d.T, origin='lower', cmap=_DEFAULT_CMAP, aspect='equal'
      )
    fig.colorbar(im, ax=ax_plot)
    ax_plot.set_title(f'{fname} ({axis}={idx})')

  # Hide unused axes.
  for i in range(n, nrows * ncols):
    row, col = divmod(i, ncols)
    axes[row, col].set_visible(False)

  fig.tight_layout()

  if save_path is not None:
    plot_path = epath.Path(save_path)
    with plot_path.open('wb') as f:
      plt.savefig(f, format='png')

  return fig
