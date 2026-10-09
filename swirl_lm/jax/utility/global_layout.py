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
"""Host-side conversions between the sharded and the contiguous grid layout.

A global 3D field is the concatenation of the per-replica arrays, each of
which carries `halo_width` halo cells on both sides so that halo exchange can
run as a neighbour collective inside `shard_map`. Its extent along an axis is
therefore

    c * n  ==  num_cores * (core_n + 2 * halo_width)

and the halo slabs straddling every interior seam are duplicates of the
neighbour's interior cells. That layout is right for computation and wrong for
storage and plotting, where a single contiguous array of unique points is
wanted, matching `grid_params.global_xyz` (length `f == c * core_n`) or
`grid_params.global_xyz_with_halos` (length `f + 2 * halo_width`).

This module provides that conversion and its exact inverse:

  - `strip_halos_global`: sharded layout -> contiguous unique points.
  - `inflate_to_global`: contiguous unique points -> sharded layout.

Both operate on host NumPy arrays and never touch a device mesh.
"""

from collections.abc import Sequence

import numpy as np
from swirl_lm.jax.utility import grid_parametrization as gp_lib


def _axis_sizes(
    grid_params: gp_lib.GridParametrization,
    axis_index: int,
) -> tuple[int, int, int, bool]:
  """Returns (num_cores, local_n, core_n, is_periodic) for an array axis.

  Args:
    grid_params: The grid parametrization.
    axis_index: The array axis, i.e. an index into `data_axis_order`.

  Returns:
    The number of cores along the axis, the per-core extent including halos,
    the per-core interior extent, and whether the axis is periodic.
  """
  axis = grid_params.data_axis_order[axis_index]
  num_cores = grid_params.get_axis_entry(
      grid_params.cx, grid_params.cy, grid_params.cz, axis
  )
  local_n = grid_params.get_axis_entry(
      grid_params.nx, grid_params.ny, grid_params.nz, axis
  )
  core_n = grid_params.get_axis_entry(
      grid_params.core_nx, grid_params.core_ny, grid_params.core_nz, axis
  )
  return num_cores, local_n, core_n, grid_params.periodic_dims[axis_index]


def stripped_axes_for(
    arr: np.ndarray,
    grid_params: gp_lib.GridParametrization,
) -> tuple[int, ...]:
  """Returns the axes of `arr` that carry the sharded, halo-duplicated layout.

  An axis qualifies only when its extent is exactly `num_cores * local_n`.
  Axes that do not match are passed through untouched by
  `strip_halos_global`; this is what makes it safe to hand an array that has
  already been stripped, or one whose axis is unrelated to the grid, such as
  the thin normal axis of a boundary condition state or the broadcast axes of
  a stretched-grid scale factor.

  Args:
    arr: The array to classify.
    grid_params: The grid parametrization.

  Returns:
    The qualifying array axes, in increasing order.
  """
  if arr.ndim != 3:
    return ()

  axes = []
  for axis_index in range(3):
    num_cores, local_n, _, _ = _axis_sizes(grid_params, axis_index)
    if arr.shape[axis_index] == num_cores * local_n:
      axes.append(axis_index)
  return tuple(axes)


def strip_halos_global(
    arr: np.ndarray,
    grid_params: gp_lib.GridParametrization,
    include_halos: bool = False,
) -> np.ndarray:
  """Converts a global sharded array to contiguous, unique grid points.

  Args:
    arr: A host array in the global sharded layout. Axes whose extent is not
      `num_cores * local_n` are passed through untouched.
    grid_params: The grid parametrization.
    include_halos: If False, keep the interior only, so each converted axis
      ends up the length of `global_xyz`. If True, additionally keep the halos
      at the two outer domain boundaries, matching `global_xyz_with_halos`.
      Either way the duplicated interior seams are discarded.

  Returns:
    The converted array, with the same dtype and rank as `arr`.
  """
  for axis_index in stripped_axes_for(arr, grid_params):
    indices = grid_params.global_axis_indices(axis_index, include_halos)
    arr = np.take(arr, indices, axis=axis_index)
  return arr


def inflate_to_global(
    arr: np.ndarray,
    grid_params: gp_lib.GridParametrization,
    target_shape: Sequence[int],
    stripped_axes: Sequence[int] | None = None,
    existing: np.ndarray | None = None,
) -> np.ndarray:
  """Converts contiguous grid points back to the global sharded layout.

  This is the exact inverse of `strip_halos_global`: the interior of every
  replica is restored from `arr`, and each replica's halos are refilled from
  the neighbouring interior, so that the interior seams are consistent by
  construction. That matters because an interior halo holds neighbour data,
  not boundary condition data, so carrying over halo values from a differently
  initialized array would leave the result internally inconsistent.

  Args:
    arr: A host array of unique grid points, as written by
      `strip_halos_global`. Each axis must be either already the target
      length, or the interior length `f`, or `f + 2 * halo_width`.
    grid_params: The grid parametrization.
    target_shape: The shape of the global sharded array to produce.
    stripped_axes: The axes that were converted on the way out, as recorded by
      `stripped_axes_for`. If None, the axes are inferred by matching lengths,
      which is ambiguous for thin axes and should be avoided when the
      information is available.
    existing: Optional array of `target_shape` supplying the halo values at
      the outer domain boundaries of non-periodic axes, where the neighbour
      data does not exist and the initialized values are boundary-condition
      consistent. Ignored where `arr` already carries those halos.

  Returns:
    An array of `target_shape` and the dtype of `arr`.

  Raises:
    ValueError: If an axis of `arr` has a length that cannot be expanded to
      the corresponding entry of `target_shape`.
  """
  target_shape = tuple(target_shape)
  if arr.ndim != len(target_shape):
    raise ValueError(
        f'Cannot inflate an array of rank {arr.ndim} to shape {target_shape}.'
    )

  if stripped_axes is None:
    stripped_axes = stripped_axes_for(
        np.empty(target_shape, dtype=bool), grid_params
    )
  stripped_axes = tuple(stripped_axes)

  hw = grid_params.halo_width
  # Axes whose outer halos were reconstructed rather than read from `arr`.
  padded_axes = []

  for axis_index in range(arr.ndim):
    target = target_shape[axis_index]
    length = arr.shape[axis_index]
    if length == target:
      continue
    if axis_index not in stripped_axes:
      raise ValueError(
          f'Axis {axis_index} has length {length} but the target is {target}, '
          'and the axis was not recorded as converted on the way out.'
      )

    num_cores, local_n, core_n, periodic = _axis_sizes(
        grid_params, axis_index
    )
    if target != num_cores * local_n:
      raise ValueError(
          f'Axis {axis_index}: target length {target} does not match the '
          f'sharded layout {num_cores} * {local_n}.'
      )

    interior = num_cores * core_n
    if length == interior:
      # Rebuild the outer halos so that every replica window is in range. On a
      # periodic axis wrapping reproduces the halo exchange exactly; otherwise
      # edge values stand in until `existing` or the first exchange corrects
      # them.
      pad = [(0, 0)] * arr.ndim
      pad[axis_index] = (hw, hw)
      arr = np.pad(arr, pad, mode='wrap' if periodic else 'edge')
      if not periodic:
        padded_axes.append(axis_index)
    elif length != interior + 2 * hw:
      raise ValueError(
          f'Axis {axis_index} has length {length}, expected the interior '
          f'({interior}), the interior with halos ({interior + 2 * hw}), or '
          f'the sharded layout ({target}).'
      )

    # Re-tile: each replica takes a window of `local_n` starting at its
    # interior offset, so consecutive windows overlap by `2 * halo_width`.
    tiles = [
        np.take(
            arr,
            np.arange(i_core * core_n, i_core * core_n + local_n),
            axis=axis_index,
        )
        for i_core in range(num_cores)
    ]
    arr = np.concatenate(tiles, axis=axis_index)

  if arr.shape != target_shape:
    raise ValueError(
        f'Inflated to {arr.shape}, expected {target_shape}.'
    )

  if existing is not None and padded_axes:
    if existing.shape != target_shape:
      raise ValueError(
          f'`existing` has shape {existing.shape}, expected {target_shape}.'
      )
    arr = np.array(arr, copy=True)
    for axis_index in padded_axes:
      for index in (slice(0, hw), slice(target_shape[axis_index] - hw, None)):
        window = [slice(None)] * arr.ndim
        window[axis_index] = index
        arr[tuple(window)] = existing[tuple(window)]

  return arr
