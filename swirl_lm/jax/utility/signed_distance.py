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
"""Signed distance functions for immersed boundary geometry.

A geometry is a callable `phi(xyz) -> scalar` taking a point as an array of
shape `(3,)` and returning the signed distance to the surface, with the sign
convention

    phi > 0  in the fluid,
    phi < 0  in the solid,
    phi == 0 on the surface,

and `|phi|` equal to the Euclidean distance to the nearest surface point. Every
primitive here is an *exact* distance function, for which `|grad phi| == 1`
everywhere the gradient is defined.

This representation is what lets the immersed boundary work for arbitrary
shapes. The previous TensorFlow implementation described the body as a height
field `z = h(x, y)` and obtained the wall-normal projection from a gradient
derived by hand for each profile, which confines it to terrain: a cylinder or a
cube is not a function of `(x, y)`. Here the surface normal comes from `jax.grad`
(see `normal`), so adding a shape means writing its distance function and
nothing else.

Shapes compose. `union`, `intersection` and `difference` are the usual
constructive-solid-geometry operators, and `translate`/`rotate` move a shape
without changing its definition. `union` is how a scene of several bodies is
built, and it is exact outside the union, which is the region the immersed
boundary cares about.

A level set that is not a true distance function is still usable --
`height_field` is the important example, since `z - h(x, y)` measures the
vertical gap rather than the shortest distance. `project_to_surface` performs
Newton iterations rather than a single step for exactly this reason, and
`normal` normalizes the gradient rather than assuming it is already a unit
vector.

Reference for the primitive distance functions: Quilez, "Distance functions",
https://iquilezles.org/articles/distfunctions/.
"""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

# A geometry: maps a point of shape (3,) to its signed distance.
SignedDistanceFn = Callable[[jax.Array], jax.Array]

# Distances below this are treated as zero when deciding whether a point sits
# on the surface, and guard the division in `normal`.
_EPS = 1e-12


def _safe_norm(v: jax.Array) -> jax.Array:
  """Euclidean norm whose gradient is defined at the origin.

  `jnp.linalg.norm` differentiates to NaN at zero, and zero is not an edge
  case here: `box` takes the norm of `max(q, 0)`, which is exactly the zero
  vector at every point inside the box, and `cylinder` takes the norm of the
  radial offset, which vanishes on the axis. Masking only the value is not
  enough, because the NaN is produced inside the `sqrt` and would survive the
  select; the square has to be masked before the `sqrt` as well.

  The gradient at the origin is zero, which is the right value for the callers
  here -- the term being differentiated is identically zero in a neighbourhood
  of those points, and the distance there is set by the other branch.

  Args:
    v: A vector.

  Returns:
    `|v|`, with a zero gradient where `v` is zero.
  """
  sq = jnp.sum(v * v)
  is_zero = sq < _EPS
  return jnp.where(is_zero, 0.0, jnp.sqrt(jnp.where(is_zero, 1.0, sq)))


def _as_point(xyz: Sequence[float]) -> jnp.ndarray:
  """Validates and converts a 3-vector argument."""
  arr = jnp.asarray(xyz, dtype=jnp.float32)
  if arr.shape != (3,):
    raise ValueError(f'Expected a 3-vector, got shape {arr.shape}.')
  return arr


# -- Primitives --------------------------------------------------------------


def sphere(center: Sequence[float], radius: float) -> SignedDistanceFn:
  """A sphere, or a 2D circle when the grid is flat along one axis.

  Args:
    center: The centre of the sphere, as (x, y, z).
    radius: The radius. Must be positive.

  Returns:
    The signed distance function of the fluid outside the sphere.
  """
  if radius <= 0.0:
    raise ValueError(f'`radius` must be positive, got {radius}.')
  c = _as_point(center)

  def phi(xyz: jax.Array) -> jax.Array:
    return _safe_norm(xyz - c) - radius

  return phi


def box(
    center: Sequence[float],
    half_extents: Sequence[float],
    rotation_deg: Sequence[float] | None = None,
) -> SignedDistanceFn:
  """A rectangular prism, which is a cube when the half extents are equal.

  Exact both outside and inside: the outside branch measures the distance to
  the nearest face, edge or corner, and the inside branch the distance to the
  nearest face.

  Args:
    center: The centre of the box, as (x, y, z).
    half_extents: Half the side length along each axis, as (x, y, z), before
      rotation. All must be positive.
    rotation_deg: Optional intrinsic rotation about the x, y and z axes in
      degrees, applied about `center`.

  Returns:
    The signed distance function of the fluid outside the box.
  """
  c = _as_point(center)
  h = _as_point(half_extents)
  if bool(jnp.any(h <= 0.0)):
    raise ValueError(f'`half_extents` must all be positive, got {half_extents}.')

  rot = None if rotation_deg is None else _rotation_matrix(rotation_deg)

  def phi(xyz: jax.Array) -> jax.Array:
    p = xyz - c
    if rot is not None:
      # Rotate the query point into the box's frame, i.e. by the inverse.
      p = rot.T @ p
    q = jnp.abs(p) - h
    outside = _safe_norm(jnp.maximum(q, 0.0))
    inside = jnp.minimum(jnp.max(q), 0.0)
    return outside + inside

  return phi


def cylinder(
    base: Sequence[float],
    axis: Sequence[float],
    radius: float,
    height: float | None = None,
) -> SignedDistanceFn:
  """A circular cylinder about an arbitrary axis.

  The axis is not restricted to a grid direction, which is the point: the
  height-field representation this replaces could only ever describe a surface
  that is single-valued in z.

  Args:
    base: A point on the axis. With a finite `height` this is the centre of the
      base cap; for an infinite cylinder it is any axis point.
    axis: The direction of the axis, as (x, y, z). Need not be normalized.
    radius: The radius. Must be positive.
    height: The length measured from `base` along `axis`. `None` gives an
      infinite cylinder, which is the right choice for a spanwise cylinder in a
      domain that is periodic along the axis.

  Returns:
    The signed distance function of the fluid outside the cylinder.
  """
  if radius <= 0.0:
    raise ValueError(f'`radius` must be positive, got {radius}.')
  if height is not None and height <= 0.0:
    raise ValueError(f'`height` must be positive when given, got {height}.')

  b = _as_point(base)
  a = _as_point(axis)
  norm = jnp.linalg.norm(a)
  if float(norm) < _EPS:
    raise ValueError(f'`axis` must have non-zero length, got {axis}.')
  a = a / norm

  def phi(xyz: jax.Array) -> jax.Array:
    d = xyz - b
    # Split the offset into along-axis and radial parts.
    along = jnp.dot(d, a)
    radial = _safe_norm(d - along * a) - radius

    if height is None:
      return radial

    # Distance outside the two end caps, negative when between them.
    cap = jnp.abs(along - 0.5 * height) - 0.5 * height
    outside = _safe_norm(jnp.maximum(jnp.stack([radial, cap]), 0.0))
    inside = jnp.minimum(jnp.maximum(radial, cap), 0.0)
    return outside + inside

  return phi


def plane(point: Sequence[float], normal_vec: Sequence[float]) -> SignedDistanceFn:
  """A half-space, with the fluid on the side the normal points toward.

  Args:
    point: Any point on the plane.
    normal_vec: The plane normal, pointing into the fluid. Need not be
      normalized.

  Returns:
    The signed distance function of the fluid half-space.
  """
  p = _as_point(point)
  n = _as_point(normal_vec)
  norm = jnp.linalg.norm(n)
  if float(norm) < _EPS:
    raise ValueError(f'`normal_vec` must have non-zero length, got {normal_vec}.')
  n = n / norm

  def phi(xyz: jax.Array) -> jax.Array:
    return jnp.dot(xyz - p, n)

  return phi


def height_field(
    h_fn: Callable[[jax.Array, jax.Array], jax.Array],
) -> SignedDistanceFn:
  """Terrain given by an elevation `z = h(x, y)`, with the fluid above.

  This reproduces the geometry the TensorFlow implementation supported, as one
  shape among the others rather than as a built-in assumption.

  Note the result is a level set but **not** an exact distance function: it
  returns the vertical gap `z - h(x, y)`, which overestimates the true distance
  on sloped terrain by `1 / cos(slope)`. `normal` and `project_to_surface` both
  handle that correctly -- the first normalizes the gradient, and the second
  iterates -- but `|phi|` should not be read as a distance for steep terrain.

  Args:
    h_fn: The elevation as a function of `(x, y)`, returning a scalar.

  Returns:
    The level set function of the fluid above the terrain.
  """

  def phi(xyz: jax.Array) -> jax.Array:
    return xyz[2] - h_fn(xyz[0], xyz[1])

  return phi


# -- Composition -------------------------------------------------------------


def union(*shapes: SignedDistanceFn) -> SignedDistanceFn:
  """The fluid outside every shape, i.e. the solids are merged.

  Exact outside the union, which is where the immersed boundary evaluates it.
  Inside, `min` underestimates the distance near a seam between two merged
  solids, which does not affect the ghost-point construction because ghost
  cells lie within one cell of the surface.

  Args:
    *shapes: The shapes to merge. At least one.

  Returns:
    The combined signed distance function.
  """
  if not shapes:
    raise ValueError('`union` needs at least one shape.')

  def phi(xyz: jax.Array) -> jax.Array:
    return jnp.min(jnp.stack([s(xyz) for s in shapes]))

  return phi


def intersection(*shapes: SignedDistanceFn) -> SignedDistanceFn:
  """The fluid outside the region common to every shape.

  Args:
    *shapes: The shapes to intersect. At least one.

  Returns:
    The combined signed distance function.
  """
  if not shapes:
    raise ValueError('`intersection` needs at least one shape.')

  def phi(xyz: jax.Array) -> jax.Array:
    return jnp.max(jnp.stack([s(xyz) for s in shapes]))

  return phi


def difference(
    shape: SignedDistanceFn, removed: SignedDistanceFn
) -> SignedDistanceFn:
  """`shape` with the solid of `removed` carved out of it.

  Args:
    shape: The shape to cut.
    removed: The shape whose solid region becomes fluid.

  Returns:
    The combined signed distance function.
  """

  def phi(xyz: jax.Array) -> jax.Array:
    return jnp.maximum(shape(xyz), -removed(xyz))

  return phi


def translate(
    shape: SignedDistanceFn, offset: Sequence[float]
) -> SignedDistanceFn:
  """Moves a shape by `offset`, preserving exactness."""
  d = _as_point(offset)

  def phi(xyz: jax.Array) -> jax.Array:
    return shape(xyz - d)

  return phi


def rotate(
    shape: SignedDistanceFn,
    rotation_deg: Sequence[float],
    center: Sequence[float] | None = None,
) -> SignedDistanceFn:
  """Rotates a shape about `center`, preserving exactness.

  Args:
    shape: The shape to rotate.
    rotation_deg: Intrinsic rotation about the x, y and z axes, in degrees.
    center: The centre of rotation. Defaults to the origin.

  Returns:
    The rotated signed distance function.
  """
  rot = _rotation_matrix(rotation_deg)
  c = jnp.zeros(3, jnp.float32) if center is None else _as_point(center)

  def phi(xyz: jax.Array) -> jax.Array:
    # Rotating the shape forward is rotating the query point backward.
    return shape(rot.T @ (xyz - c) + c)

  return phi


def _rotation_matrix(rotation_deg: Sequence[float]) -> jnp.ndarray:
  """Builds an intrinsic x-then-y-then-z rotation matrix from degrees."""
  a = _as_point(rotation_deg) * jnp.pi / 180.0
  cx, cy, cz = jnp.cos(a[0]), jnp.cos(a[1]), jnp.cos(a[2])
  sx, sy, sz = jnp.sin(a[0]), jnp.sin(a[1]), jnp.sin(a[2])
  rx = jnp.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
  ry = jnp.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
  rz = jnp.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
  return rz @ ry @ rx


# -- Queries -----------------------------------------------------------------


def evaluate(phi: SignedDistanceFn, points: jax.Array) -> jax.Array:
  """Evaluates `phi` at many points.

  Args:
    phi: The geometry.
    points: Query points of shape `(..., 3)`.

  Returns:
    The signed distance at each point, of shape `(...)`.
  """
  flat = points.reshape(-1, 3)
  return jax.vmap(phi)(flat).reshape(points.shape[:-1])


def normal(phi: SignedDistanceFn, points: jax.Array) -> jax.Array:
  """The unit surface normal at each point, pointing into the fluid.

  Obtained by automatic differentiation, so it is correct for any composed
  geometry without a hand-derived gradient. The gradient is normalized rather
  than assumed to be a unit vector, so inexact level sets such as
  `height_field` give the right direction.

  Args:
    phi: The geometry.
    points: Query points of shape `(..., 3)`.

  Returns:
    Unit normals of shape `(..., 3)`.
  """
  flat = points.reshape(-1, 3)
  g = jax.vmap(jax.grad(phi))(flat)
  n = g / jnp.maximum(jnp.linalg.norm(g, axis=-1, keepdims=True), _EPS)
  return n.reshape(points.shape)


def project_to_surface(
    phi: SignedDistanceFn, points: jax.Array, iterations: int = 3
) -> jax.Array:
  """Projects points onto the surface along the normal.

  Newton's method on `phi`, which for an exact distance function converges in a
  single step because `|grad phi| == 1`. The extra iterations matter for
  inexact level sets such as `height_field`, where one step lands short on
  sloped terrain.

  This is the body-intercept calculation of Ghias et al. (2007) §2.5. Unlike
  the TensorFlow implementation it needs no per-shape derivative: the gradient
  comes from `jax.grad`.

  Args:
    phi: The geometry.
    points: Query points of shape `(..., 3)`.
    iterations: Number of Newton steps. Three is ample for the level sets used
      here; one suffices for an exact distance function.

  Returns:
    The projected points, of shape `(..., 3)`.
  """
  if iterations < 1:
    raise ValueError(f'`iterations` must be at least 1, got {iterations}.')

  shape = points.shape
  x = points.reshape(-1, 3)
  grad_phi = jax.vmap(jax.value_and_grad(phi))

  def step(x, _):
    value, g = grad_phi(x)
    denom = jnp.maximum(jnp.sum(g * g, axis=-1, keepdims=True), _EPS)
    return x - value[:, None] * g / denom, None

  x, _ = jax.lax.scan(step, x, None, length=iterations)
  return x.reshape(shape)


def grid_points(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> jnp.ndarray:
  """Builds the `(nx, ny, nz, 3)` point array for a tensor-product grid.

  Args:
    x: Cell-centre coordinates along x.
    y: Cell-centre coordinates along y.
    z: Cell-centre coordinates along z.

  Returns:
    The coordinates of every grid point, of shape `(nx, ny, nz, 3)`.
  """
  xx, yy, zz = jnp.meshgrid(
      jnp.asarray(x), jnp.asarray(y), jnp.asarray(z), indexing='ij'
  )
  return jnp.stack([xx, yy, zz], axis=-1)


# -- Batched primitives ------------------------------------------------------
#
# `union` takes one Python callable per body, so every body adds operations to
# the traced graph and the compile time grows superlinearly with the body
# count: measured, 200 bodies cost 2.1 s to compile, 500 cost 13.9 s and 1000
# cost 80.6 s. That is fine for a cylinder or a hill and unusable for a city
# block of buildings.
#
# The constructors below take the parameters of many bodies of one type as
# arrays and reduce over a leading body axis, so the graph stays the same size
# whatever the count -- 5000 bodies compile in 0.41 s. The flops are still
# proportional to bodies times grid points, which is a cost at initialization
# only; if that becomes the limit the next step is to cull bodies by bounding
# box per subdomain, not to change the method.
#
# Mixed scenes group by type and `union` the handful of groups, so the graph
# grows with the number of *types*, which is three.


def _safe_norm_batched(v: jax.Array) -> jax.Array:
  """`_safe_norm` over the last axis of a stack of vectors.

  Same masking, for the same reason: the norm of `max(q, 0)` is exactly zero
  at every point inside a box, and `jnp.linalg.norm` differentiates to NaN
  there.

  Args:
    v: An array whose last axis holds vector components.

  Returns:
    The norm along the last axis, with a zero gradient where the vector is
    zero.
  """
  sq = jnp.sum(v * v, axis=-1)
  is_zero = sq < _EPS
  return jnp.where(is_zero, 0.0, jnp.sqrt(jnp.where(is_zero, 1.0, sq)))


def _as_points(values, name: str, width: int = 3) -> jnp.ndarray:
  """Validates an `(n, width)` array of per-body parameters."""
  arr = jnp.asarray(values, dtype=jnp.float32)
  if arr.ndim != 2 or arr.shape[1] != width:
    raise ValueError(
        f'`{name}` must have shape (n, {width}), got {tuple(arr.shape)}.'
    )
  if arr.shape[0] == 0:
    raise ValueError(f'`{name}` is empty; there is nothing to build.')
  return arr


def _as_scalars(values, name: str, count: int) -> jnp.ndarray:
  """Validates an `(n,)` array of per-body parameters."""
  arr = jnp.atleast_1d(jnp.asarray(values, dtype=jnp.float32))
  if arr.shape != (count,):
    raise ValueError(
        f'`{name}` must have shape ({count},), got {tuple(arr.shape)}.'
    )
  return arr


def _rotation_matrices(angles_deg) -> jnp.ndarray:
  """Intrinsic x-then-y-then-z rotation matrices for many bodies at once.

  Built with whole-array arithmetic rather than a loop over bodies, so the
  cost of constructing a large scene is not itself proportional to the body
  count.

  Args:
    angles_deg: `(n, 3)` rotations about x, y and z, in degrees.

  Returns:
    `(n, 3, 3)` rotation matrices.
  """
  a = np.asarray(angles_deg, np.float64) * np.pi / 180.0
  cx, cy, cz = np.cos(a[:, 0]), np.cos(a[:, 1]), np.cos(a[:, 2])
  sx, sy, sz = np.sin(a[:, 0]), np.sin(a[:, 1]), np.sin(a[:, 2])
  zero, one = np.zeros_like(cx), np.ones_like(cx)
  rx = np.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], -1)
  ry = np.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], -1)
  rz = np.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], -1)
  shape = (-1, 3, 3)
  return jnp.asarray(
      rz.reshape(shape) @ ry.reshape(shape) @ rx.reshape(shape), jnp.float32
  )


def batched_spheres(centers, radii) -> SignedDistanceFn:
  """Many spheres, as the fluid outside their union.

  Args:
    centers: `(n, 3)` sphere centres.
    radii: `(n,)` radii, all positive.

  Returns:
    The signed distance function of the fluid outside every sphere.
  """
  c = _as_points(centers, 'centers')
  r = _as_scalars(radii, 'radii', c.shape[0])
  if bool(jnp.any(r <= 0.0)):
    raise ValueError('All `radii` must be positive.')

  def phi(xyz: jax.Array) -> jax.Array:
    return jnp.min(_safe_norm_batched(xyz[None, :] - c) - r)

  return phi


def batched_boxes(
    centers, half_extents, rotations_deg=None
) -> SignedDistanceFn:
  """Many rectangular prisms, as the fluid outside their union.

  Args:
    centers: `(n, 3)` box centres.
    half_extents: `(n, 3)` half side lengths before rotation, all positive.
    rotations_deg: Optional `(n, 3)` intrinsic rotations about x, y and z in
      degrees, applied about each centre.

  Returns:
    The signed distance function of the fluid outside every box.
  """
  c = _as_points(centers, 'centers')
  h = _as_points(half_extents, 'half_extents')
  if h.shape != c.shape:
    raise ValueError(
        f'`half_extents` {tuple(h.shape)} must match `centers` '
        f'{tuple(c.shape)}.'
    )
  if bool(jnp.any(h <= 0.0)):
    raise ValueError('All `half_extents` must be positive.')
  rot = None
  if rotations_deg is not None:
    _as_points(rotations_deg, 'rotations_deg')
    rot = _rotation_matrices(rotations_deg)
    if rot.shape[0] != c.shape[0]:
      raise ValueError('`rotations_deg` must have one entry per box.')

  def phi(xyz: jax.Array) -> jax.Array:
    p = xyz[None, :] - c                                   # (n, 3)
    if rot is not None:
      # Rotate the query point into each box's frame, i.e. by the inverse.
      p = jnp.einsum('nji,nj->ni', rot, p)
    q = jnp.abs(p) - h
    outside = _safe_norm_batched(jnp.maximum(q, 0.0))
    inside = jnp.minimum(jnp.max(q, axis=-1), 0.0)
    return jnp.min(outside + inside)

  return phi


def batched_cylinders(
    bases, axes, radii, heights=None
) -> SignedDistanceFn:
  """Many circular cylinders, as the fluid outside their union.

  Args:
    bases: `(n, 3)` a point on each axis; the base-cap centre when a height is
      given.
    axes: `(n, 3)` axis directions, each non-zero. Need not be normalized and
      need not be grid directions.
    radii: `(n,)` radii, all positive.
    heights: Optional `(n,)` lengths along each axis. Use a non-positive value
      or `inf` to mark an individual cylinder as infinite. `None` makes every
      cylinder infinite.

  Returns:
    The signed distance function of the fluid outside every cylinder.
  """
  b = _as_points(bases, 'bases')
  a = _as_points(axes, 'axes')
  n_bodies = b.shape[0]
  if a.shape[0] != n_bodies:
    raise ValueError('`axes` must have one entry per cylinder.')
  r = _as_scalars(radii, 'radii', n_bodies)
  if bool(jnp.any(r <= 0.0)):
    raise ValueError('All `radii` must be positive.')

  lengths = jnp.linalg.norm(a, axis=-1, keepdims=True)
  if bool(jnp.any(lengths < _EPS)):
    raise ValueError('Every entry of `axes` must have non-zero length.')
  a = a / lengths

  if heights is None:
    finite = jnp.zeros((n_bodies,), bool)
    half_h = jnp.ones((n_bodies,), jnp.float32)
  else:
    h = _as_scalars(heights, 'heights', n_bodies)
    finite = jnp.isfinite(h) & (h > 0.0)
    # The unused branch still has to evaluate, so keep its height finite or
    # the arithmetic produces NaNs that survive the select.
    half_h = 0.5 * jnp.where(finite, h, 1.0)

  def phi(xyz: jax.Array) -> jax.Array:
    d = xyz[None, :] - b
    along = jnp.sum(d * a, axis=-1)                        # (n,)
    radial = _safe_norm_batched(d - along[:, None] * a) - r

    cap = jnp.abs(along - half_h) - half_h
    capped = _safe_norm_batched(
        jnp.maximum(jnp.stack([radial, cap], axis=-1), 0.0)
    ) + jnp.minimum(jnp.maximum(radial, cap), 0.0)
    return jnp.min(jnp.where(finite, capped, radial))

  return phi
