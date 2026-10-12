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
"""Ghost-point geometry for the sharp-interface immersed boundary method.

Implements the precomputation for the ghost-cell method of Ghias, Mittal and
Dong, "A sharp interface immersed boundary method for compressible viscous
flows", J. Comput. Phys. 225 (2007) 528-553, section 2.5.

A *ghost cell* is a solid cell with at least one fluid neighbour in its
27-point neighbourhood. For each one the method constructs

  - the **body intercept** BI, the normal projection of the ghost cell onto
    the surface;
  - the **image point** IP = GC + m*(BI - GC), with m = 2 placing BI exactly
    midway between GC and IP;

interpolates the solution trilinearly to IP, and extrapolates back along the
normal to get the ghost value,

    phi_GC = zeta * phi_IP + Gamma.

Everything above depends only on the geometry and the grid, so this module
computes it **once** and returns plain index and weight arrays. The per-step
work is then a gather, a weighted sum and a scatter, with no geometry
evaluation and no automatic differentiation inside the time loop.

Three departures from a literal reading of the paper, each deliberate:

**The probe length adapts.** The paper keeps m = 2 everywhere and widens the
inter-process overlap region instead, noting (p. 540) that for non-isotropic
grids "the size of this interpolation stencil can vary over a large range".
Measured here, a cylinder needs 3 halo cells and a Gaussian hill 4, but a hill
on a grid refined 4x in the wall-normal direction needs 19 -- and
`halo_width` is a single scalar shared by every axis, so that is not a
configuration anyone can run. Instead m is reduced, per ghost cell, only where
the m = 2 stencil would leave the local array. The extrapolation stays exact
for a linear profile at any m > 1, so this degrades smoothly rather than
silently; see `_extrapolation_coefficients`. The number of capped cells and the
smallest m are reported so a badly matched grid is visible rather than quietly
less accurate.

**Case 2 is closed by solving the implicit relation, not by editing a matrix
row.** When the interpolation stencil contains the ghost cell itself the
equation becomes implicit in `phi_GC`. The paper replaces the corresponding
row of the Vandermonde matrix with the boundary condition (Eqs. 23-24). Here
the scalar equation is simply solved,

    phi_GC = (zeta * sum_{i != self} alpha_i u_i + Gamma) / (1 - zeta * w_self)

which is exact, needs no matrix surgery, and -- unlike the row replacement --
takes the same form for Neumann as for Dirichlet.

**Indices come from a coordinate search, not from division by a spacing.** The
enclosing cell of an image point is found with `np.searchsorted` on the actual
coordinate array, so stretched grids work without a special case. On a uniform
grid the result is identical to `ip // h`.
"""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.utility import signed_distance as sdf_lib

# The eight corners of the cell enclosing an image point, as (0, 1) offsets
# along each of the three data axes. The ordering fixes the column order of
# `corner_index` and `alpha` and must match `_trilinear_weights`.
_CORNERS = tuple(
    (i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)
)

# The image point must lie far enough beyond the body intercept to keep the
# extrapolation well conditioned. The Dirichlet slope is `zeta = -1/(m-1)`, so
# the scheme amplifies any error in the interpolated value by `1/(m-1)`:
# m = 1.05 would magnify float32 round-off twentyfold. At m = 1.5 the factor
# is 2, which costs nothing in practice because with `halo_width >= 2` the
# stencil almost always fits and the probe stays at 2 anyway. A cell that
# cannot reach this takes the nearest-fluid fallback.
_MIN_PROBE = 1.5

# `1 - zeta * w_self` closer to zero than this means the implicit solve for a
# self-referencing stencil is ill-conditioned; the cell is dropped.
_MIN_DENOM = 1e-3

# Distances below this are treated as zero.
_EPS = 1e-12


@dataclasses.dataclass(frozen=True)
class GhostPoints:
  """Precomputed ghost-point geometry for one replica's local subdomain.

  All index arrays refer to the *flattened* local array, halos included, so
  applying the method is `u.ravel()[corner_index]` followed by a scatter to
  `gc_index`.

  Attributes:
    gc_index: `(n,)` flat index of each ghost cell.
    corner_index: `(n, 8)` flat indices of the cells enclosing each image
      point, ordered as `_CORNERS`.
    alpha: `(n, 8)` trilinear interpolation weights. The column belonging to
      the ghost cell itself, if any, is zeroed and its weight moved to
      `self_weight`.
    self_weight: `(n,)` the interpolation weight that fell on the ghost cell
      itself, zero for the common case where it did not.
    zeta_dirichlet: `(n,)` the extrapolation slope `-1/(m-1)` for a Dirichlet
      condition, which is `-1` wherever the probe was not capped.
    separation: `(n,)` the distance from the ghost cell to the point its
      value is interpolated from, `m*|phi|` normally and the distance to the
      nearest fluid cell for a fallback cell. This is the lever arm a Neumann
      condition multiplies.
    normal: `(n, 3)` the unit surface normal at the body intercept, pointing
      into the fluid.
    distance: `(n,)` `|phi|` at the ghost cell.
    probe: `(n,)` the multiplier `m` actually used.
    signed_distance: The signed distance at every cell of this replica's
      local array, halos included. Computed anyway while identifying ghost
      cells, and kept because the masks and the field itself are useful
      post-processing output.
    fallback: `(n,)` `True` where the normal construction could not be used
      and the nearest-fluid approximation was substituted. Those cells are
      locally first-order rather than second-order.
    shape: the local array shape the flat indices refer to.
    num_capped: how many cells had their probe shortened below 2.
    num_fallback: how many cells could not use the normal construction and
      fell back to the nearest fluid cell. A large count means the grid is
      too coarse for the geometry near those cells.
    min_probe: the smallest `m` used, `2.0` when nothing was capped.
    max_reach_cells: per data axis, the largest image-point offset in cells.
      This is what would set the required `halo_width` if the probe were not
      capped, and is reported so the cost of capping is visible.
  """

  gc_index: np.ndarray
  corner_index: np.ndarray
  alpha: np.ndarray
  self_weight: np.ndarray
  zeta_dirichlet: np.ndarray
  separation: np.ndarray
  normal: np.ndarray
  distance: np.ndarray
  probe: np.ndarray
  signed_distance: np.ndarray
  fallback: np.ndarray
  shape: tuple[int, int, int]
  num_capped: int
  num_fallback: int
  min_probe: float
  max_reach_cells: tuple[float, float, float]

  @property
  def interior_mask(self) -> np.ndarray:
    """1 in the fluid and 0 in the solid, as a float field for output."""
    return (self.signed_distance > 0.0).astype(np.float32)

  @property
  def ghost_mask(self) -> np.ndarray:
    """1 at each ghost cell and 0 elsewhere, as a float field for output."""
    mask = np.zeros(self.shape, np.float32)
    if self.num_ghost_points:
      mask.reshape(-1)[self.gc_index] = 1.0
    return mask

  @property
  def num_ghost_points(self) -> int:
    """How many ghost cells are represented."""
    return int(self.gc_index.shape[0])

  def summary(self) -> str:
    """A one-line description for logging at initialization."""
    reach = ', '.join(f'{r:.2f}' for r in self.max_reach_cells)
    msg = (
        f'ghost points: {self.num_ghost_points}; '
        f'image-point reach (cells, per data axis): {reach}'
    )
    # Always reported, not only when something was capped. `m = 2` puts the
    # image point exactly twice the wall distance out, which is the method's
    # ideal; anything less was shortened to stay inside the array. Printing
    # it unconditionally is the difference between knowing the probes are
    # healthy and merely not being told that they are not.
    msg += f'; min m = {self.min_probe:.2f}'
    if self.num_capped:
      pct = 100.0 * self.num_capped / max(self.num_ghost_points, 1)
      msg += f' (capped on {self.num_capped}, {pct:.1f}%)'
    if self.num_fallback:
      pct = 100.0 * self.num_fallback / max(self.num_ghost_points, 1)
      msg += f'; {self.num_fallback} ({pct:.1f}%) on the nearest-fluid fallback'
    return msg


def interior_mask(
    phi: sdf_lib.SignedDistanceFn,
    coords: Sequence[np.ndarray],
    dims: Sequence[int] = (0, 1, 2),
) -> np.ndarray:
  """The fluid mask: `True` outside the body.

  Args:
    phi: The geometry, a signed distance function of a physical `(x, y, z)`
      point. Positive in the fluid.
    coords: Cell-centre coordinates along each data axis, in data-array index
      order.
    dims: The physical dimension (0 for x, 1 for y, 2 for z) held by each data
      axis. The identity for `data_axis_order == ('x', 'y', 'z')`.

  Returns:
    A boolean array of shape `(len(coords[0]), len(coords[1]),
    len(coords[2]))`, `True` where the cell centre is in the fluid.
  """
  return _signed_distance_field(phi, coords, dims) > 0.0


def _signed_distance_field(
    phi: sdf_lib.SignedDistanceFn,
    coords: Sequence[np.ndarray],
    dims: Sequence[int],
) -> np.ndarray:
  """Evaluates `phi` at every cell centre of the local grid."""
  _validate_axes(coords, dims)
  mesh = np.meshgrid(*[np.asarray(c, np.float64) for c in coords],
                     indexing='ij')
  # Reorder the data axes into physical (x, y, z) before calling `phi`.
  physical = [None, None, None]
  for data_axis, dim in enumerate(dims):
    physical[dim] = mesh[data_axis]
  pts = np.stack(physical, axis=-1).astype(np.float32)
  return np.asarray(sdf_lib.evaluate(phi, pts), np.float64)


def _validate_axes(coords: Sequence[np.ndarray], dims: Sequence[int]) -> None:
  """Checks the coordinate arrays and the data-to-physical axis mapping."""
  if len(coords) != 3:
    raise ValueError(f'Expected 3 coordinate arrays, got {len(coords)}.')
  if sorted(dims) != [0, 1, 2]:
    raise ValueError(
        f'`dims` must be a permutation of (0, 1, 2), got {tuple(dims)}.'
    )
  for axis, c in enumerate(coords):
    c = np.asarray(c)
    if c.ndim != 1 or c.size < 2:
      raise ValueError(
          f'Coordinate array for data axis {axis} must be 1D with at least 2 '
          f'entries, got shape {c.shape}.'
      )
    if not np.all(np.diff(c) > 0.0):
      raise ValueError(
          f'Coordinate array for data axis {axis} must be strictly '
          'increasing.'
      )


def ghost_cell_mask(fluid: np.ndarray, halo_width: int) -> np.ndarray:
  """Solid cells with a fluid neighbour, excluding the halo layers.

  Ghias et al. section 2.5 take the 27-point neighbourhood, so a solid cell
  touching fluid only at a corner still counts. The dilation deliberately does
  **not** wrap: a wrapping neighbourhood would mark cells against a physical
  boundary as ghost cells because the opposite face of the domain is fluid,
  and their image points then land arbitrarily far away.

  Ghost cells in the halo layers are excluded, because their image points
  could fall outside the local array. Those cells are interior to the
  neighbouring replica, which computes them, so a halo exchange after the
  update fills them in.

  Args:
    fluid: Boolean array, `True` in the fluid.
    halo_width: Number of halo layers on each side of each axis.

  Returns:
    A boolean array, `True` at each ghost cell.
  """
  if halo_width < 0:
    raise ValueError(f'`halo_width` must be non-negative, got {halo_width}.')

  solid = ~fluid
  neighbour_fluid = np.zeros_like(fluid)
  for di in (-1, 0, 1):
    for dj in (-1, 0, 1):
      for dk in (-1, 0, 1):
        if di == dj == dk == 0:
          continue
        offset = (di, dj, dk)
        src = tuple(
            slice(max(0, -d), fluid.shape[a] - max(0, d))
            for a, d in enumerate(offset)
        )
        dst = tuple(
            slice(max(0, d), fluid.shape[a] - max(0, -d))
            for a, d in enumerate(offset)
        )
        neighbour_fluid[dst] |= fluid[src]

  mask = solid & neighbour_fluid
  if halo_width > 0:
    interior = np.zeros_like(mask)
    core = tuple(slice(halo_width, -halo_width) for _ in range(3))
    interior[core] = True
    mask &= interior
  return mask


def _extrapolation_coefficients(
    probe: np.ndarray, distance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """The Dirichlet slope and the intercept-to-image distance, for any probe.

  Along the surface normal, with `s` measured from the body intercept and the
  profile taken as linear, the ghost cell is at `s = -d` and the image point
  at `s = (m-1)*d`, where `d = |phi|`. Matching the line through `(BI, Phi)`
  and `(IP, phi_IP)` at `s = -d` gives

      phi_GC = Phi - (phi_IP - Phi)/(m-1)
             = zeta * phi_IP + Gamma,
      zeta   = -1/(m-1),   Gamma = Phi * (1 - zeta).

  At `m = 2` this is the paper's `zeta = -1`, `Gamma = 2*Phi`. For a Neumann
  condition the slope is fixed at `+1` and the offset is `Psi` times the
  intercept-to-image distance `(m-1)*d`, which is why that distance is
  returned alongside.

  Args:
    probe: The multiplier `m` for each ghost cell, strictly greater than 1.
    distance: `|phi|` at each ghost cell.

  Returns:
    A tuple of the Dirichlet slope `zeta` and the distance from the body
    intercept to the image point.
  """
  reach = probe - 1.0
  return -1.0 / reach, reach * distance


def _max_probe_within(
    gc: np.ndarray,
    to_surface: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
  """The largest probe multiplier keeping the image point inside the grid.

  `IP(m) = GC + m * (BI - GC)` is linear in `m`, so for each axis the bound is
  a single division, and the binding constraint is the smallest over axes.

  Args:
    gc: `(n, 3)` ghost-cell coordinates, in data-axis order.
    to_surface: `(n, 3)` the vector `BI - GC`, in data-axis order.
    lo: `(3,)` the lowest usable coordinate on each data axis.
    hi: `(3,)` the highest usable coordinate on each data axis.

  Returns:
    `(n,)` the largest permissible `m`, capped at 2.
  """
  limit = np.full(gc.shape[0], 2.0)
  for axis in range(3):
    step = to_surface[:, axis]
    moving = np.abs(step) > _EPS
    # Where the step is positive the upper bound binds, and vice versa.
    target = np.where(step > 0.0, hi[axis], lo[axis])
    with np.errstate(divide='ignore', invalid='ignore'):
      bound = (target - gc[:, axis]) / step
    limit = np.minimum(limit, np.where(moving, bound, np.inf))
  return np.minimum(limit, 2.0)


def build(
    phi: sdf_lib.SignedDistanceFn,
    coords: Sequence[np.ndarray],
    halo_width: int,
    dims: Sequence[int] = (0, 1, 2),
    newton_iterations: int = 3,
) -> GhostPoints:
  """Builds the ghost-point geometry for one replica's local subdomain.

  Args:
    phi: The geometry, a signed distance function of a physical `(x, y, z)`
      point, positive in the fluid.
    coords: Cell-centre coordinates along each data axis, halos included, in
      data-array index order. Need not be uniformly spaced.
    halo_width: Number of halo layers on each side of each axis.
    dims: The physical dimension held by each data axis.
    newton_iterations: Iterations used to project onto the surface. One is
      exact for a true distance function; three covers curved level sets such
      as a height field.

  Returns:
    The precomputed geometry.
  """
  _validate_axes(coords, dims)
  axes = [np.asarray(c, np.float64) for c in coords]
  shape = tuple(a.size for a in axes)

  value = _signed_distance_field(phi, axes, dims)
  mask = ghost_cell_mask(value > 0.0, halo_width)
  ijk = np.argwhere(mask)

  if ijk.shape[0] == 0:
    return _empty(shape, value.astype(np.float32))

  # Ghost-cell coordinates, in data-axis order and in physical order.
  gc_data = np.stack([axes[a][ijk[:, a]] for a in range(3)], axis=-1)
  gc_phys = np.empty_like(gc_data)
  for data_axis, dim in enumerate(dims):
    gc_phys[:, dim] = gc_data[:, data_axis]

  bi_phys = np.asarray(
      sdf_lib.project_to_surface(
          phi, gc_phys.astype(np.float32), iterations=newton_iterations
      ),
      np.float64,
  )
  bi_data = np.stack([bi_phys[:, dims[a]] for a in range(3)], axis=-1)

  to_surface = bi_data - gc_data
  distance = np.linalg.norm(to_surface, axis=-1)

  # The surface normal, pointing into the fluid. Taken from the geometry
  # rather than from `BI - GC`, which is degenerate when the ghost cell
  # already lies on the surface.
  normal_phys = np.asarray(
      sdf_lib.normal(phi, gc_phys.astype(np.float32)), np.float64
  )

  lo = np.array([a[0] for a in axes])
  hi = np.array([a[-1] for a in axes])
  probe = _max_probe_within(gc_data, to_surface, lo, hi)

  # Report what the uncapped stencil would have required, before capping.
  spacing = np.array([np.min(np.diff(a)) for a in axes])
  reach_cells = np.abs(2.0 * to_surface) / spacing
  max_reach = tuple(float(r) for r in reach_cells.max(axis=0))

  # A ghost cell whose probe cannot reach `_MIN_PROBE` -- too close to the
  # edge of the local array -- cannot use the normal construction, because
  # `zeta = -1/(m-1)` blows up as `m -> 1`. It must not simply be discarded:
  # a discarded ghost cell is still a stencil corner for its neighbours, and
  # would feed them whatever stale value happened to be sitting in the solid.
  # Instead it falls back to the nearest fluid cell, giving `phi_GC = Phi`
  # for Dirichlet and a one-sided gradient for Neumann. That is locally
  # first-order, which is what Gibou et al. resort to in this limit (Ghias
  # p. 538), but it is defined everywhere.
  fallback = probe < _MIN_PROBE
  probe = np.maximum(probe, _MIN_PROBE)

  ip = gc_data + probe[:, None] * to_surface
  # Fallback cells get their weights overwritten below, but the image point
  # still has to be inside the array for `_stencil` to produce valid indices.
  for axis in range(3):
    ip[:, axis] = np.clip(ip[:, axis], axes[axis][0], axes[axis][-1])
  corner_index, alpha = _stencil(ip, axes, shape)

  gc_index = np.ravel_multi_index(
      (ijk[:, 0], ijk[:, 1], ijk[:, 2]), shape
  ).astype(np.int32)

  # Ghias case 2: the stencil contains the ghost cell itself, making the
  # relation implicit. Move that weight aside so `apply` can divide it out.
  is_self = corner_index == gc_index[:, None]
  self_weight = (alpha * is_self).sum(axis=1)
  alpha = np.where(is_self, 0.0, alpha)

  zeta_dirichlet, _ = _extrapolation_coefficients(probe, distance)
  separation = probe * distance

  # The implicit solve divides by `1 - zeta * w_self`. For Dirichlet `zeta`
  # is negative so that always exceeds 1, but for Neumann `zeta = +1` and a
  # self-weight near 1 is ill-conditioned. Those cells take the fallback too.
  fallback |= np.abs(1.0 - self_weight) < _MIN_DENOM

  if np.any(fallback):
    nb_index, nb_distance = _nearest_fluid(
        ijk[fallback], value > 0.0, axes, shape
    )
    corner_index[fallback] = nb_index[:, None]
    alpha[fallback] = 0.0
    alpha[fallback, 0] = 1.0
    self_weight[fallback] = 0.0
    zeta_dirichlet[fallback] = 0.0     # phi_GC = Gamma = Phi
    separation[fallback] = nb_distance

  num_capped = int((probe < 2.0 - 1e-6).sum())
  return GhostPoints(
      gc_index=gc_index,
      corner_index=corner_index.astype(np.int32),
      alpha=alpha.astype(np.float32),
      self_weight=self_weight.astype(np.float32),
      zeta_dirichlet=zeta_dirichlet.astype(np.float32),
      separation=separation.astype(np.float32),
      normal=normal_phys.astype(np.float32),
      distance=distance.astype(np.float32),
      probe=probe.astype(np.float32),
      signed_distance=value.astype(np.float32),
      fallback=fallback,
      shape=shape,
      num_capped=num_capped,
      num_fallback=int(fallback.sum()),
      min_probe=float(probe.min()) if probe.size else 2.0,
      max_reach_cells=max_reach,
  )


def _stencil(
    ip: np.ndarray, axes: Sequence[np.ndarray], shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
  """Locates the cell enclosing each image point and its trilinear weights.

  Uses `np.searchsorted` on the coordinate arrays rather than dividing by a
  spacing, so a stretched grid needs no special case. On a uniform grid the
  result is identical.

  Args:
    ip: `(n, 3)` image-point coordinates in data-axis order.
    axes: Cell-centre coordinates along each data axis.
    shape: The local array shape.

  Returns:
    A tuple of `(n, 8)` flat corner indices and `(n, 8)` weights, ordered as
    `_CORNERS`.
  """
  base = np.empty((ip.shape[0], 3), np.int64)
  frac = np.empty((ip.shape[0], 3), np.float64)
  for axis in range(3):
    coord = axes[axis]
    # `searchsorted` gives the insertion point; one less is the lower corner.
    idx = np.searchsorted(coord, ip[:, axis]) - 1
    idx = np.clip(idx, 0, shape[axis] - 2)
    width = coord[idx + 1] - coord[idx]
    base[:, axis] = idx
    frac[:, axis] = (ip[:, axis] - coord[idx]) / width

  corner_index = np.empty((ip.shape[0], 8), np.int64)
  alpha = np.empty((ip.shape[0], 8), np.float64)
  for col, (di, dj, dk) in enumerate(_CORNERS):
    corner_index[:, col] = np.ravel_multi_index(
        (base[:, 0] + di, base[:, 1] + dj, base[:, 2] + dk), shape
    )
    alpha[:, col] = (
        np.where(di, frac[:, 0], 1.0 - frac[:, 0])
        * np.where(dj, frac[:, 1], 1.0 - frac[:, 1])
        * np.where(dk, frac[:, 2], 1.0 - frac[:, 2])
    )
  return corner_index, alpha


def _nearest_fluid(
    ijk: np.ndarray,
    fluid: np.ndarray,
    axes: Sequence[np.ndarray],
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
  """The closest fluid cell in the 27-point neighbourhood of each cell.

  Only used for the fallback path. Every ghost cell has a fluid neighbour by
  definition, so the search always succeeds.

  Args:
    ijk: `(n, 3)` indices of the cells to search around.
    fluid: Boolean array, `True` in the fluid.
    axes: Cell-centre coordinates along each data axis.
    shape: The local array shape.

  Returns:
    A tuple of `(n,)` flat indices and `(n,)` distances.

  Raises:
    ValueError: If any cell has no fluid neighbour, which would mean it was
      not a ghost cell.
  """
  best_index = np.full(ijk.shape[0], -1, np.int64)
  best_distance = np.full(ijk.shape[0], np.inf)
  here = np.stack([axes[a][ijk[:, a]] for a in range(3)], axis=-1)

  for di in (-1, 0, 1):
    for dj in (-1, 0, 1):
      for dk in (-1, 0, 1):
        if di == dj == dk == 0:
          continue
        nb = ijk + np.array([di, dj, dk])
        inside = np.all((nb >= 0) & (nb < np.array(shape)), axis=1)
        safe = np.clip(nb, 0, np.array(shape) - 1)
        is_fluid = inside & fluid[safe[:, 0], safe[:, 1], safe[:, 2]]
        there = np.stack([axes[a][safe[:, a]] for a in range(3)], axis=-1)
        dist = np.linalg.norm(there - here, axis=-1)
        flat = np.ravel_multi_index(
            (safe[:, 0], safe[:, 1], safe[:, 2]), shape
        )
        better = is_fluid & (dist < best_distance)
        best_index = np.where(better, flat, best_index)
        best_distance = np.where(better, dist, best_distance)

  if np.any(best_index < 0):
    raise ValueError(
        'A ghost cell has no fluid neighbour, which contradicts how ghost '
        'cells are identified. This is a bug in `ghost_cell_mask`.'
    )
  return best_index, best_distance


def _empty(
    shape: tuple[int, int, int],
    signed_distance: np.ndarray,
    max_reach_cells: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> GhostPoints:
  """The geometry for a replica whose subdomain contains no ghost cells."""
  return GhostPoints(
      gc_index=np.zeros((0,), np.int32),
      corner_index=np.zeros((0, 8), np.int32),
      alpha=np.zeros((0, 8), np.float32),
      self_weight=np.zeros((0,), np.float32),
      zeta_dirichlet=np.zeros((0,), np.float32),
      separation=np.zeros((0,), np.float32),
      normal=np.zeros((0, 3), np.float32),
      distance=np.zeros((0,), np.float32),
      probe=np.zeros((0,), np.float32),
      signed_distance=signed_distance,
      fallback=np.zeros((0,), bool),
      shape=shape,
      num_capped=0,
      num_fallback=0,
      min_probe=2.0,
      max_reach_cells=max_reach_cells,
  )


# -- Applying the boundary condition -----------------------------------------


def dirichlet_coefficients(
    geom: GhostPoints, value: float | jax.Array
) -> tuple[jax.Array, jax.Array]:
  """The `(zeta, Gamma)` pair imposing `phi = value` on the surface.

  Args:
    geom: The precomputed geometry.
    value: The surface value, either a scalar or one value per ghost cell.

  Returns:
    The slope and offset of `phi_GC = zeta * phi_IP + Gamma`.
  """
  zeta = jnp.asarray(geom.zeta_dirichlet)
  return zeta, jnp.asarray(value) * (1.0 - zeta)


def neumann_coefficients(
    geom: GhostPoints, flux: float | jax.Array
) -> tuple[jax.Array, jax.Array]:
  """The `(zeta, Gamma)` pair imposing `dphi/dn = flux` on the surface.

  The normal derivative is taken along the outward normal, into the fluid, so
  a positive `flux` means the variable increases away from the body. A zero
  flux is the common adiabatic or zero-gradient case.

  Args:
    geom: The precomputed geometry.
    flux: The surface normal derivative, scalar or one value per ghost cell.

  Returns:
    The slope and offset of `phi_GC = zeta * phi_IP + Gamma`.
  """
  # Along the normal the profile is `u(s) = u(BI) + flux * s`, so stepping
  # from the image point at `s = (m-1)*d` back to the ghost cell at `s = -d`
  # crosses `m*d`, which is exactly `|IP - GC|`. Writing it as `2*d` would be
  # right only at `m = 2` and wrong wherever the probe was capped.
  separation = jnp.asarray(geom.separation)
  ones = jnp.ones_like(separation)
  return ones, -jnp.asarray(flux) * separation


def apply(
    u: jax.Array,
    geom: GhostPoints,
    zeta: jax.Array,
    gamma: jax.Array,
    num_sweeps: int = 1,
) -> jax.Array:
  """Writes the ghost-cell values into `u`.

  The interpolation stencil of one ghost cell may contain another (Ghias case
  3), so the values are coupled. The paper resolves this inside its
  Gauss-Seidel solve; here the update is simply repeated `num_sweeps` times,
  which is the explicit analogue.

  Where the stencil contains the ghost cell being solved for (case 2), the
  relation is implicit and is solved directly:

      phi_GC = (zeta * sum_i alpha_i u_i + Gamma) / (1 - zeta * w_self)

  Args:
    u: The local field, shape `geom.shape`.
    geom: The precomputed geometry.
    zeta: Per-ghost-cell extrapolation slope.
    gamma: Per-ghost-cell extrapolation offset.
    num_sweeps: How many times to repeat the update.

  Returns:
    `u` with the ghost cells overwritten.
  """
  if u.shape != geom.shape:
    raise ValueError(
        f'`u` has shape {u.shape} but the geometry was built for '
        f'{geom.shape}.'
    )
  if num_sweeps < 1:
    raise ValueError(f'`num_sweeps` must be at least 1, got {num_sweeps}.')
  if geom.num_ghost_points == 0:
    return u

  corner = jnp.asarray(geom.corner_index)
  alpha = jnp.asarray(geom.alpha)
  gc = jnp.asarray(geom.gc_index)
  denom = 1.0 - zeta * jnp.asarray(geom.self_weight)

  flat = u.reshape(-1)
  for _ in range(num_sweeps):
    phi_ip = jnp.einsum('gk,gk->g', alpha, flat[corner])
    flat = flat.at[gc].set((zeta * phi_ip + gamma) / denom)
  return flat.reshape(u.shape)


# -- Distributing the geometry across replicas -------------------------------


class ShardedGhostPoints(NamedTuple):
  """Per-replica ghost-point arrays, padded and stacked for `shard_map`.

  Every array carries three leading axes indexed by the replica's position in
  the device mesh, in `data_axis_order`, followed by a ghost-point axis padded
  to the largest count over all replicas. Under a
  `P(*mesh.axis_names, None, ...)` spec each device therefore receives its own
  geometry with the leading axes of size one, which `squeeze_local` removes.

  The padding entries point at flat index 0. That cell is in the halo, and
  `ghost_cell_mask` never marks a halo cell, so writing it back unchanged is a
  genuine no-op and cannot race with a real update. This relies on
  `halo_width >= 1`, which `build_for_mesh` checks.

  Attributes:
    gc_index: `(c0, c1, c2, n)` flat index of each ghost cell.
    corner_index: `(c0, c1, c2, n, 8)` flat indices of the stencil corners.
    alpha: `(c0, c1, c2, n, 8)` interpolation weights.
    self_weight: `(c0, c1, c2, n)` weight falling on the ghost cell itself.
    zeta_dirichlet: `(c0, c1, c2, n)` Dirichlet extrapolation slope.
    separation: `(c0, c1, c2, n)` ghost-cell-to-image-point distance.
    valid: `(c0, c1, c2, n)` `False` for padding entries.
  """

  gc_index: jax.Array
  corner_index: jax.Array
  alpha: jax.Array
  self_weight: jax.Array
  zeta_dirichlet: jax.Array
  separation: jax.Array
  valid: jax.Array


def partition_specs(mesh: jax.sharding.Mesh) -> ShardedGhostPoints:
  """The `PartitionSpec` for each field of `ShardedGhostPoints`.

  The geometry is genuinely different on every replica, so it must be
  partitioned over the mesh rather than replicated. This mirrors
  `synthetic_turbulent_inflow.partition_specs`, which exists because a
  per-replica state with no declared spec silently became replica 0's copy on
  every device.

  Args:
    mesh: The device mesh.

  Returns:
    A `ShardedGhostPoints` holding specs instead of arrays.
  """
  names = tuple(mesh.axis_names)
  flat = jax.sharding.PartitionSpec(*names, None)
  with_corners = jax.sharding.PartitionSpec(*names, None, None)
  return ShardedGhostPoints(
      gc_index=flat,
      corner_index=with_corners,
      alpha=with_corners,
      self_weight=flat,
      zeta_dirichlet=flat,
      separation=flat,
      valid=flat,
  )


def squeeze_local(sharded: ShardedGhostPoints) -> ShardedGhostPoints:
  """Drops the three replica axes, for use inside `shard_map`.

  Args:
    sharded: The geometry as seen by one device, with leading axes of size 1.

  Returns:
    The same arrays without the replica axes.
  """
  return jax.tree.map(lambda a: a[0, 0, 0], sharded)


def local_coordinates(
    grid_params, replica: Sequence[int]
) -> list[np.ndarray]:
  """The cell-centre coordinates of one replica's subdomain, halos included.

  The host-side counterpart of
  `GridParametrization.grid_local_with_coord`, which needs a named-axis
  context and so cannot run during initialization.

  Args:
    grid_params: The grid parametrization.
    replica: The replica's position along each data axis.

  Returns:
    One coordinate array per data axis, in data-array index order.
  """
  halo = grid_params.halo_width
  core = grid_params.to_data_axis_order(
      grid_params.core_nx, grid_params.core_ny, grid_params.core_nz
  )
  coords = []
  for axis in range(3):
    full = np.asarray(grid_params.global_xyz_with_halos[axis])
    start = replica[axis] * core[axis]
    coords.append(full[start : start + core[axis] + 2 * halo])
  return coords


def build_for_mesh(
    phi: sdf_lib.SignedDistanceFn,
    grid_params,
    mesh: jax.sharding.Mesh,
    newton_iterations: int = 3,
) -> tuple[ShardedGhostPoints, list[GhostPoints]]:
  """Builds the geometry for every replica and places it on the mesh.

  The direct route, for tests and for callers that drive `shard_map`
  themselves. A simulation normally uses `GhostPointGeometry` instead, which
  routes the same arrays through `additional_states`.

  Args:
    phi: The geometry, positive in the fluid.
    grid_params: The grid parametrization.
    mesh: The device mesh.
    newton_iterations: Newton steps for the surface projection.

  Returns:
    A tuple of the placed arrays and the per-replica geometries.
  """
  packed, per_replica = _build_packed(
      phi, grid_params, mesh, newton_iterations
  )
  # `_build_packed` keeps everything float32 so the driver sees one dtype;
  # here the arrays are used directly, so restore the natural types.
  typed = packed._replace(
      gc_index=packed.gc_index.astype(np.int32),
      corner_index=packed.corner_index.astype(np.int32),
      valid=packed.valid > 0.5,
  )
  specs = partition_specs(mesh)
  placed = jax.tree.map(
      lambda a, s: jax.device_put(a, jax.sharding.NamedSharding(mesh, s)),
      typed,
      specs,
      is_leaf=lambda x: isinstance(x, jax.sharding.PartitionSpec),
  )
  return placed, per_replica


def apply_local(
    u: jax.Array,
    geom: ShardedGhostPoints,
    zeta: jax.Array,
    gamma: jax.Array,
    num_sweeps: int = 2,
    exchange_fn: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
  """Writes the ghost-cell values into one replica's field.

  The `shard_map` counterpart of `apply`. Padding entries address flat index
  0, a halo cell, and are masked back to their original value, so they are
  inert.

  Args:
    u: The local field.
    geom: This replica's geometry, already squeezed.
    zeta: Per-ghost-cell extrapolation slope.
    gamma: Per-ghost-cell extrapolation offset.
    num_sweeps: How many times to repeat the update, for the case where one
      ghost cell's stencil contains another.
    exchange_fn: An optional halo exchange applied between sweeps.

      Ghost cells in the halo are excluded from this replica's geometry and
      are updated by the neighbour that owns them, so after a sweep this
      replica's halo copies are stale.

      Within this function that rarely matters: the update is Jacobi, so one
      sweep reads the same input everywhere, and a ghost cell's image point
      always points away from its own body, so reaching across a seam into
      *another* ghost cell needs a fluid gap narrower than the stencil. A
      sphere, a plane and a one-cell slab straddling a seam all reproduce the
      single-device answer at one, two and four sweeps without it.

      **That is not the same as saying no exchange is needed.** The caller
      must exchange after this returns, because the cells written here are
      ones a neighbour holds copies of, and the next operator to read those
      copies gets a pre-boundary-condition value. Measured on a cylinder
      split so the seam cuts the body, omitting that exchange moved the
      solution by 7%. `velocity.prediction_step` and
      `scalars.prediction_step` do it; anything else calling this must too.

  Returns:
    `u` with the ghost cells overwritten.
  """
  if num_sweeps < 1:
    raise ValueError(f'`num_sweeps` must be at least 1, got {num_sweeps}.')

  gc = geom.gc_index
  denom = 1.0 - zeta * geom.self_weight
  result = u
  for sweep in range(num_sweeps):
    if sweep and exchange_fn is not None:
      result = exchange_fn(result)
    flat = result.reshape(-1)
    phi_ip = jnp.einsum('gk,gk->g', geom.alpha, flat[geom.corner_index])
    updated = (zeta * phi_ip + gamma) / denom
    result = flat.at[gc].set(
        jnp.where(geom.valid, updated, flat[gc])
    ).reshape(u.shape)
  return result


# -- Building the geometry from configuration --------------------------------


def _vec3(message) -> tuple[float, float, float]:
  """Reads a `Vec3` proto into a tuple."""
  return (message.x, message.y, message.z)


def geometry_from_proto(
    messages: Sequence[Any],
    height_fields: dict[str, Callable[[Any, Any], Any]] | None = None,
) -> sdf_lib.SignedDistanceFn:
  """Builds a signed distance function from a list of `IBGeometry` messages.

  The bodies are combined by union, so listing several describes a scene --
  a city block of boxes and cylinders as readily as a single obstacle.

  Bodies of the same type are gathered into one batched primitive rather than
  one callable each. That matters at scale: a callable per body makes the
  traced graph grow with the body count, and compile time with it (measured:
  200 bodies 2.1 s, 1000 bodies 80.6 s), whereas the batched form compiles
  5000 bodies in 0.41 s. The graph here therefore grows with the number of
  shape *types* present, not the number of bodies.

  Args:
    messages: `IBGeometry` protos.
    height_fields: Elevation functions by name, supplying the `profile` of a
      `height_field` geometry. An example owns these because terrain is
      case-specific. These cannot be batched, since each is an arbitrary
      Python function, but a scene has at most a few.

  Returns:
    The combined geometry, positive in the fluid.

  Raises:
    ValueError: If the list is empty, a shape is unset, or a named height
      field profile was not supplied.
  """
  if not messages:
    raise ValueError(
        'The ghost-point method needs at least one `geometry` entry.'
    )

  spheres: list[tuple] = []
  boxes: list[tuple] = []
  cylinders: list[tuple] = []
  terrain: list[sdf_lib.SignedDistanceFn] = []

  for message in messages:
    kind = message.WhichOneof('shape')
    if kind == 'sphere':
      spheres.append((_vec3(message.sphere.center), message.sphere.radius))
    elif kind == 'box':
      boxes.append((
          _vec3(message.box.center),
          _vec3(message.box.half_extents),
          _vec3(message.box.rotation_deg),
          message.box.HasField('rotation_deg'),
      ))
    elif kind == 'cylinder':
      cylinders.append((
          _vec3(message.cylinder.base),
          _vec3(message.cylinder.axis),
          message.cylinder.radius,
          message.cylinder.height
          if message.cylinder.HasField('height')
          else np.inf,
      ))
    elif kind == 'height_field':
      name = message.height_field.profile
      if not height_fields or name not in height_fields:
        available = sorted(height_fields) if height_fields else []
        raise ValueError(
            f'The height field profile {name!r} was not supplied. '
            f'Available: {available}.'
        )
      terrain.append(sdf_lib.height_field(height_fields[name]))
    else:
      raise ValueError(
          'Every `IBGeometry` must set one of sphere, box, cylinder or '
          'height_field.'
      )

  groups: list[sdf_lib.SignedDistanceFn] = []
  if spheres:
    centers, radii = zip(*spheres)
    groups.append(sdf_lib.batched_spheres(np.array(centers), np.array(radii)))
  if boxes:
    centers, halves, rotations, rotated = zip(*boxes)
    groups.append(
        sdf_lib.batched_boxes(
            np.array(centers),
            np.array(halves),
            # Skip the rotation entirely when no box asked for one, rather
            # than multiplying every query point by an identity.
            np.array(rotations) if any(rotated) else None,
        )
    )
  if cylinders:
    bases, axes, radii, heights = zip(*cylinders)
    groups.append(
        sdf_lib.batched_cylinders(
            np.array(bases), np.array(axes), np.array(radii),
            np.array(heights),
        )
    )
  groups.extend(terrain)

  return groups[0] if len(groups) == 1 else sdf_lib.union(*groups)


def coefficients_for(
    geom: ShardedGhostPoints | GhostPoints,
    bc_type: str,
    value: float,
) -> tuple[jax.Array, jax.Array]:
  """The `(zeta, Gamma)` pair for a named boundary-condition type.

  Args:
    geom: The geometry.
    bc_type: Either `'dirichlet'` or `'neumann'`.
    value: The surface value for Dirichlet, or the normal derivative for
      Neumann.

  Returns:
    The extrapolation slope and offset.

  Raises:
    ValueError: If `bc_type` is not recognized.
  """
  if bc_type == 'dirichlet':
    return dirichlet_coefficients(geom, value)
  if bc_type == 'neumann':
    return neumann_coefficients(geom, value)
  raise ValueError(
      f"`bc_type` must be 'dirichlet' or 'neumann', got {bc_type!r}."
  )


def forcing(
    u: jax.Array,
    geom: ShardedGhostPoints,
    zeta: jax.Array,
    gamma: jax.Array,
    damping_coeff: float,
    num_sweeps: int = 2,
) -> jax.Array:
  """The relaxation force driving `u` towards the ghost values.

  The alternative to assigning the ghost value outright, matching the
  forcing-based methods already in this package. The boundary is then only
  approximately satisfied and the result depends on `damping_coeff`, so
  `DIRECT` is the default; this exists so the two can be compared on the same
  case.

  Args:
    u: The local field.
    geom: This replica's geometry, already squeezed.
    zeta: Per-ghost-cell extrapolation slope.
    gamma: Per-ghost-cell extrapolation offset.
    damping_coeff: The relaxation rate.
    num_sweeps: Sweeps used to evaluate the target ghost values.

  Returns:
    A field that is `-damping_coeff * (u - u_ghost)` at the ghost cells and
    zero everywhere else.
  """
  target = apply_local(u, geom, zeta, gamma, num_sweeps=num_sweeps)
  return -damping_coeff * (u - target)


# -- Transport through `additional_states` -----------------------------------
#
# The packed geometry rides `additional_states` like any other helper
# variable. Measured, putting it in the scan carry rather than closing over it
# costs nothing -- within noise at both 1.0 M and 8.4 M cells -- because XLA
# hoists the loop-invariant operands by itself, so the simpler plumbing wins.
#
# A per-replica array of shape `(1, 1, 1, n, ...)` with a
# `P(*mesh.axis_names, None, ...)` spec assembles into `(c0, c1, c2, n, ...)`
# with each replica's tile intact, and arrives back inside `shard_map` at its
# original shape. That needs no change to the driver.

# Solver internals, excluded from checkpoints by `is_geometry_key`.
GEOMETRY_PREFIX = 'ib_geom_'

# Post-processing output. These keep the names the other immersed boundary
# methods already use, so existing tooling keeps working.
INTERIOR_MASK_KEY = 'ib_interior_mask'
GHOST_MASK_KEY = 'ib_boundary'
SIGNED_DISTANCE_KEY = 'ib_phi'

_FLAT_FIELDS = (
    'gc_index', 'self_weight', 'zeta_dirichlet', 'separation', 'valid'
)
_CORNER_FIELDS = ('corner_index', 'alpha')


def is_geometry_key(key: str) -> bool:
  """Whether `key` holds solver internals that should not be checkpointed.

  The index and weight arrays are five-dimensional and meaningless outside
  the solver. The helper *fields* -- the masks and the signed distance -- are
  ordinary 3D states and deliberately are not covered by this.

  Args:
    key: A state name.

  Returns:
    True if the key should be left out of checkpoints.
  """
  return key.startswith(GEOMETRY_PREFIX)


# How far into the solid the fluid's own stencils reach. The convection
# scheme is blended next to the body, but `DIFFUSION_SCHEME_CENTRAL_5` is a
# five-point stencil and nothing blends it, so a fluid cell adjacent to the
# surface reads two cells deep. Anything held fixed within that reach is a
# discontinuity the fluid can see.
_STENCIL_REACH = 2


def deep_interior_mask(
    interior_mask: jax.Array,
    geom: ShardedGhostPoints,
    reach: int = _STENCIL_REACH,
) -> jax.Array:
  """The solid cells no fluid cell can see: deeper than any stencil reaches.

  The ghost-point construction sets one layer of solid cells. Everything
  deeper is outside the solution, and a caller may want to hold it at the
  boundary value rather than let the momentum equation integrate it. But
  "deeper" has to mean deeper than the fluid's own stencils reach, not
  merely below the ghost layer.

  Getting that wrong is not a subtle error. Clamping at one layer puts a
  hard zero two cells from the surface, right inside the five-point
  diffusion stencil of the first fluid cell, next to a correctly mirrored
  ghost value one cell away. The resulting kink injects momentum every step:
  measured on flat ground under a 5 m/s wind, the velocity was clean for 25
  steps, five times too large by 100 and non-finite by 200, while the same
  case with no clamping at all held the inflow value exactly.

  Args:
    interior_mask: This replica's fluid mask, one in the fluid.
    geom: This replica's geometry, already squeezed. Unused except to keep
      the ghost layer out of the result even where `reach` is zero.
    reach: How far a fluid cell's stencil reaches into the solid.

  Returns:
    A boolean field, `True` in the solid cells nothing reads.
  """
  fluid = interior_mask > 0.5
  # Grow the fluid by `reach` cells. Six-neighbour growth repeated `reach`
  # times covers every cell within `reach` steps along the axes, which is
  # what a dimension-by-dimension stencil reads.
  # Shifted with an edge-replicating pad rather than rolled: a roll wraps
  # the far face onto the near one, which in this codebase has already
  # produced one phantom result.
  seen = fluid
  for _ in range(max(reach, 0)):
    padded = jnp.pad(seen, 1, mode='edge')
    grown = seen
    for axis in range(3):
      low, high = [slice(1, -1)] * 3, [slice(1, -1)] * 3
      low[axis], high[axis] = slice(0, -2), slice(2, None)
      grown = grown | padded[tuple(low)] | padded[tuple(high)]
    seen = grown
  solid = (~seen).reshape(-1)
  # Padding entries address flat index 0, a halo cell that `ghost_cell_mask`
  # never marks, so clearing it costs nothing: halo values are replaced by
  # the exchange or the boundary condition before anything reads them.
  solid = solid.at[geom.gc_index].set(False)
  return solid.reshape(interior_mask.shape)


def from_additional_states(
    additional_states: dict[str, jax.Array],
) -> ShardedGhostPoints:
  """Recovers this replica's geometry from `additional_states`.

  For use inside `shard_map`, where each array still carries the three
  replica axes of size one.

  Args:
    additional_states: The helper variables.

  Returns:
    The geometry, with the replica axes removed.

  Raises:
    KeyError: If the geometry was not registered, which usually means the
      example did not add `GhostPointGeometry.initial_states` to its
      `init_fn`.
  """
  missing = [
      f'{GEOMETRY_PREFIX}{name}'
      for name in _FLAT_FIELDS + _CORNER_FIELDS
      if f'{GEOMETRY_PREFIX}{name}' not in additional_states
  ]
  if missing:
    raise KeyError(
        f'The ghost-point geometry is missing from `additional_states`: '
        f'{missing}. The example must include '
        '`GhostPointGeometry.initial_states(...)` in its `init_fn` and '
        'register `partition_specs()`.'
    )
  get = lambda name: additional_states[f'{GEOMETRY_PREFIX}{name}'][0, 0, 0]
  return ShardedGhostPoints(
      gc_index=get('gc_index').astype(jnp.int32),
      corner_index=get('corner_index').astype(jnp.int32),
      alpha=get('alpha'),
      self_weight=get('self_weight'),
      zeta_dirichlet=get('zeta_dirichlet'),
      separation=get('separation'),
      # Carried as a float so every geometry array has the same dtype
      # handling through the driver; restored to a mask here.
      valid=get('valid') > 0.5,
  )


class GhostPointGeometry:
  """Builds the geometry once and serves each replica its initial states.

  An example constructs one of these, registers `partition_specs()` with
  `params.additional_state_partition_specs`, and calls `initial_states` from
  its `init_fn`. The build runs on the host during initialization, so neither
  the signed distance function nor its gradient ever enters the compiled step.
  """

  def __init__(
      self,
      phi: sdf_lib.SignedDistanceFn,
      grid_params,
      mesh: jax.sharding.Mesh,
      write_interior_mask: bool = True,
      write_ghost_mask: bool = True,
      write_signed_distance: bool = False,
      newton_iterations: int = 3,
  ):
    """Builds the geometry for every replica.

    Args:
      phi: The geometry, positive in the fluid.
      grid_params: The grid parametrization.
      mesh: The device mesh.
      write_interior_mask: Emit `ib_interior_mask` as an output field.
      write_ghost_mask: Emit `ib_boundary` as an output field.
      write_signed_distance: Emit `ib_phi` as an output field.
      newton_iterations: Newton steps for the surface projection.
    """
    self._mesh = mesh
    self._counts = tuple(mesh.shape[name] for name in mesh.axis_names)
    self._grid_params = grid_params
    self._write = {
        INTERIOR_MASK_KEY: write_interior_mask,
        GHOST_MASK_KEY: write_ghost_mask,
        SIGNED_DISTANCE_KEY: write_signed_distance,
    }
    self._packed, self.per_replica = _build_packed(
        phi, grid_params, mesh, newton_iterations
    )

  @property
  def num_ghost_points(self) -> int:
    """Ghost cells over all replicas."""
    return sum(g.num_ghost_points for g in self.per_replica)

  def summary(self) -> str:
    """A description of the whole mesh, for logging at initialization."""
    worst = max(self.per_replica, key=lambda g: g.num_capped)
    capped = sum(g.num_capped for g in self.per_replica)
    fallback = sum(g.num_fallback for g in self.per_replica)
    reach = ', '.join(
        f'{max(g.max_reach_cells[a] for g in self.per_replica):.2f}'
        for a in range(3)
    )
    text = (
        f'ghost points: {self.num_ghost_points} over '
        f'{len(self.per_replica)} replica(s); image-point reach '
        f'(cells, per data axis): {reach}'
    )
    text += f'; min m = {min(g.min_probe for g in self.per_replica):.2f}'
    if capped:
      text += f' (capped on {capped})'
    if fallback:
      text += f'; {fallback} on the nearest-fluid fallback'
    del worst
    return text

  def partition_specs(self) -> dict[str, jax.sharding.PartitionSpec]:
    """How each geometry array maps onto the device mesh.

    Register these in `params.additional_state_partition_specs`. Without
    them the driver can only replicate a non-3D state, which would hand every
    device replica 0's geometry.

    Returns:
      A dict from state name to `PartitionSpec`.
    """
    names = tuple(self._mesh.axis_names)
    flat = jax.sharding.PartitionSpec(*names, None)
    corners = jax.sharding.PartitionSpec(*names, None, None)
    specs = {f'{GEOMETRY_PREFIX}{n}': flat for n in _FLAT_FIELDS}
    specs.update({f'{GEOMETRY_PREFIX}{n}': corners for n in _CORNER_FIELDS})
    return specs

  def initial_states(
      self, logical_coordinates: Sequence[int]
  ) -> dict[str, np.ndarray]:
    """This replica's geometry arrays and any enabled output fields.

    Args:
      logical_coordinates: The replica's position as `(cx_i, cy_i, cz_i)`, in
        physical order, as the driver supplies it to an `init_fn`.

    Returns:
      A dict to merge into the states returned by the example's `init_fn`.
    """
    i0, i1, i2 = self._grid_params.to_data_axis_order(*logical_coordinates)
    states: dict[str, np.ndarray] = {}
    for name in _FLAT_FIELDS + _CORNER_FIELDS:
      tile = getattr(self._packed, name)[i0, i1, i2]
      states[f'{GEOMETRY_PREFIX}{name}'] = tile[None, None, None, ...]

    flat_index = int(np.ravel_multi_index((i0, i1, i2), self._counts))
    geom = self.per_replica[flat_index]
    if self._write[INTERIOR_MASK_KEY]:
      states[INTERIOR_MASK_KEY] = geom.interior_mask
    if self._write[GHOST_MASK_KEY]:
      states[GHOST_MASK_KEY] = geom.ghost_mask
    if self._write[SIGNED_DISTANCE_KEY]:
      states[SIGNED_DISTANCE_KEY] = geom.signed_distance
    return states


def _build_packed(
    phi: sdf_lib.SignedDistanceFn,
    grid_params,
    mesh: jax.sharding.Mesh,
    newton_iterations: int,
) -> tuple[ShardedGhostPoints, list[GhostPoints]]:
  """Builds every replica's geometry and pads it to a common length."""
  if grid_params.halo_width < 1:
    raise ValueError(
        'The ghost-point method needs `halo_width >= 1`: ghost cells in the '
        'halo are excluded and filled by the halo exchange instead, and the '
        'padding entries rely on flat index 0 being a halo cell.'
    )

  counts = tuple(mesh.shape[name] for name in mesh.axis_names)
  dims = [grid_params.get_axis_index(a) for a in ('x', 'y', 'z')]
  axis_to_dim = [0, 0, 0]
  for physical, data_axis in enumerate(dims):
    axis_to_dim[data_axis] = physical

  per_replica = []
  for i0 in range(counts[0]):
    for i1 in range(counts[1]):
      for i2 in range(counts[2]):
        per_replica.append(
            build(
                phi,
                local_coordinates(grid_params, (i0, i1, i2)),
                halo_width=grid_params.halo_width,
                dims=axis_to_dim,
                newton_iterations=newton_iterations,
            )
        )

  n_max = max(1, max(g.num_ghost_points for g in per_replica))
  flat_shape = counts + (n_max,)
  corner_shape = flat_shape + (8,)

  packed = ShardedGhostPoints(
      gc_index=np.zeros(flat_shape, np.float32),
      corner_index=np.zeros(corner_shape, np.float32),
      alpha=np.zeros(corner_shape, np.float32),
      self_weight=np.zeros(flat_shape, np.float32),
      zeta_dirichlet=np.zeros(flat_shape, np.float32),
      separation=np.ones(flat_shape, np.float32),
      valid=np.zeros(flat_shape, np.float32),
  )
  for flat_pos, geom in enumerate(per_replica):
    i0, i1, i2 = np.unravel_index(flat_pos, counts)
    n = geom.num_ghost_points
    if n == 0:
      continue
    packed.gc_index[i0, i1, i2, :n] = geom.gc_index
    packed.corner_index[i0, i1, i2, :n] = geom.corner_index
    packed.alpha[i0, i1, i2, :n] = geom.alpha
    packed.self_weight[i0, i1, i2, :n] = geom.self_weight
    packed.zeta_dirichlet[i0, i1, i2, :n] = geom.zeta_dirichlet
    packed.separation[i0, i1, i2, :n] = geom.separation
    packed.valid[i0, i1, i2, :n] = 1.0
  return packed, per_replica


def required_keys(params) -> list[str]:
  """State keys the ghost-point method needs, if it is configured.

  Returns an empty list for every other immersed boundary type and when no
  immersed boundary is configured at all, so the driver can call this
  unconditionally. This follows `lpt.required_keys`: a model declares the
  state it needs rather than every configuration having to list the solver's
  internal arrays by hand.

  Args:
    params: The simulation parameters.

  Returns:
    The additional state keys to allocate.
  """
  models = getattr(params, 'boundary_models', None)
  if models is None or not models.HasField('ib'):
    return []
  if models.ib.WhichOneof('type') != 'ghost_point':
    return []

  config = models.ib.ghost_point
  keys = [
      f'{GEOMETRY_PREFIX}{name}'
      for name in _FLAT_FIELDS + _CORNER_FIELDS
  ]
  if config.write_interior_mask:
    keys.append(INTERIOR_MASK_KEY)
  if config.write_ghost_mask:
    keys.append(GHOST_MASK_KEY)
  if config.write_signed_distance:
    keys.append(SIGNED_DISTANCE_KEY)
  return keys
