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

# Copyright 2022 Google LLC
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
"""Tests for the multigrid Poisson solver.

The tests are organized in three groups:

* The building blocks (transfers, Galerkin operator, smoother, coarsening plan
  and validation), tested in isolation with stub boundary exchanges.
* The constructor and the hierarchy.
* `solve` end to end under `shard_map`, against an exact discrete solution
  computed independently with NumPy.
"""


import functools
import itertools
import os
from typing import Any, NamedTuple
import unittest

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
import jax
from jax import sharding
from jax.experimental import shard_map
import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.communication import halo_exchange
from swirl_lm.jax.linalg import base_poisson_solver
from swirl_lm.jax.linalg import multigrid_solver
from swirl_lm.jax.linalg import poisson_solver
from swirl_lm.jax.linalg import poisson_solver_pb2
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

_X = base_poisson_solver.X
_RESIDUAL = base_poisson_solver.RESIDUAL_L2_NORM
_ITERATIONS = base_poisson_solver.ITERATIONS

_NEUMANN = (halo_exchange.BCType.NEUMANN, 0.0)
_DIRICHLET = (halo_exchange.BCType.DIRICHLET, 0.0)

# ============================================================================
# Helpers.
# ============================================================================


def _stub_exchange(kind: str):
  """A boundary exchange for a single device, without collectives.

  Mirrors what `_make_exchange` returns on one device for the given condition
  applied on every face.

  Args:
    kind: One of 'dirichlet', 'neumann' or 'periodic'.

  Returns:
    A function `(u, axis) -> (beyond_low, beyond_high)`.
  """

  def exchange(u, axis):
    first = u[multigrid_solver._first(axis)]
    last = u[multigrid_solver._last(u, axis)]
    if kind == 'dirichlet':
      return jnp.zeros_like(first), jnp.zeros_like(last)
    if kind == 'neumann':
      return first, last
    if kind == 'periodic':
      return last, first
    raise ValueError(f'Unknown kind: {kind}')

  return exchange


def _stub_laplacian(spacings, exchange):
  """A 7-point Laplacian whose boundary rows are set by `exchange`."""

  def operator(u):
    result = jnp.zeros_like(u)
    for axis, h in enumerate(spacings):
      beyond_low, beyond_high = exchange(u, axis)
      result += (
          multigrid_solver._shift_up(u, axis, beyond_high)
          - 2.0 * u
          + multigrid_solver._shift_down(u, axis, beyond_low)
      ) / h**2
    return result

  return operator


def _dense(fn, in_shape) -> np.ndarray:
  """The matrix of the linear map `fn` acting on arrays of `in_shape`."""
  size = int(np.prod(in_shape))
  columns = jax.vmap(lambda e: fn(e.reshape(in_shape)).ravel())(
      jnp.eye(size, dtype=jnp.float32)
  )
  return np.asarray(columns).T


def _make_grid_params(
    interior_xyz=(16, 16, 16),
    cores_xyz=(1, 1, 1),
    halo=1,
    lengths_xyz=(1.0, 1.0, 1.0),
    periodic_xyz=(False, False, False),
    data_axis_order='zxy',
) -> grid_parametrization.GridParametrization:
  """A grid parametrization from the *global* interior extent per axis."""
  for n, c in zip(interior_xyz, cores_xyz):
    assert n % c == 0, (interior_xyz, cores_xyz)
  grid_size = [n // c + 2 * halo for n, c in zip(interior_xyz, cores_xyz)]
  periodic = [str(p).lower() for p in periodic_xyz]
  proto_text = f"""
    computation_shape {{
      dim_x: {cores_xyz[0]} dim_y: {cores_xyz[1]} dim_z: {cores_xyz[2]}
    }}
    grid_size {{
      dim_x: {grid_size[0]} dim_y: {grid_size[1]} dim_z: {grid_size[2]}
    }}
    length {{
      dim_x: {lengths_xyz[0]} dim_y: {lengths_xyz[1]} dim_z: {lengths_xyz[2]}
    }}
    periodic {{
      dim_x: {periodic[0]} dim_y: {periodic[1]} dim_z: {periodic[2]}
    }}
    halo_width: {halo}
    dt: 0.001
    data_axis_order: "{data_axis_order}"
  """
  proto = grid_parametrization.grid_parametrization_pb2.GridParametrization()
  text_format.Parse(proto_text, proto)
  return grid_parametrization.GridParametrization(proto)


def _make_solver_option(
    num_iterations: int = 5,
    n_smooth: int = 2,
    weight: float = 0.6667,
    n_coarse: int = 1,
    coarsest_xyz: tuple[int, int, int] | None = None,
) -> poisson_solver_pb2.PoissonSolver:
  """A multigrid solver option proto."""
  coarsest = ''
  if coarsest_xyz is not None:
    coarsest = (
        'coarsest_subgrid_shape {'
        f' dim_x: {coarsest_xyz[0]} dim_y: {coarsest_xyz[1]}'
        f' dim_z: {coarsest_xyz[2]} }}'
    )
  proto_text = f"""
    multigrid {{
      num_iterations: {num_iterations}
      n_smooth: {n_smooth}
      weight: {weight}
      n_coarse: {n_coarse}
      {coarsest}
    }}
  """
  proto = poisson_solver_pb2.PoissonSolver()
  text_format.Parse(proto_text, proto)
  return proto


def _bc_spec(kinds_xyz):
  """`BoundaryConditionsSpec` from per-axis kinds such as 'DN' or 'P'.

  Args:
    kinds_xyz: Per axis in x, y, z order: 'P' for periodic, otherwise two
      characters, each 'D' (homogeneous Dirichlet) or 'N' (homogeneous Neumann),
      for the low and the high face.

  Returns:
    The spec, in x, y, z order as `Pressure._build_dp_bc` returns it.
  """
  face = {'D': _DIRICHLET, 'N': _NEUMANN}
  return tuple(
      None if kind == 'P' else (face[kind[0]], face[kind[1]])
      for kind in kinds_xyz
  )


class _Setup(NamedTuple):
  """Everything one problem needs."""

  grid_params: grid_parametrization.GridParametrization
  mesh: sharding.Mesh
  bc_xyz: tuple[str, str, str]
  solver: multigrid_solver.MultigridSolver
  halo_update_fn: Any
  cores: tuple[int, ...]  # In data axis order.


def _make_setup(
    interior_xyz=(16, 16, 16),
    bc_xyz=('NN', 'NN', 'NN'),
    cores_xyz=(1, 1, 1),
    halo=1,
    lengths_xyz=(1.0, 1.0, 1.0),
    data_axis_order='zxy',
    kernel='slice',
    set_boundary_conditions=True,
    **option_kwargs,
) -> _Setup:
  """Builds the grid, mesh, solver and halo update for one problem."""
  periodic_xyz = tuple(kind == 'P' for kind in bc_xyz)
  gp = _make_grid_params(
      interior_xyz, cores_xyz, halo, lengths_xyz, periodic_xyz, data_axis_order
  )
  cores = tuple(gp.to_data_axis_order(*cores_xyz))
  devices = np.array(jax.devices()[: int(np.prod(cores))]).reshape(cores)
  mesh = sharding.Mesh(devices, tuple(gp.data_axis_order))
  if kernel == 'slice':
    kernel_op = get_kernel_fn.ApplyKernelSliceOp(gp)
  else:
    kernel_op = get_kernel_fn.ApplyKernelConvOp(4, gp)
  solver = multigrid_solver.MultigridSolver(
      gp, kernel_op, _make_solver_option(**option_kwargs)
  )
  bc = _bc_spec(bc_xyz)
  if set_boundary_conditions:
    solver.boundary_conditions = bc
  # The same composition `Pressure._dp_halo_update_fn` uses.
  halo_update_fn = functools.partial(
      halo_exchange.inplace_halo_exchange,
      axes=('x', 'y', 'z'),
      mesh=mesh,
      grid_params=gp,
      periodic_dims=list(periodic_xyz),
      boundary_conditions=bc,
      halo_width=halo,
  )
  return _Setup(gp, mesh, tuple(bc_xyz), solver, halo_update_fn, cores)


def _with_halos(interior: np.ndarray, cores, halo: int) -> np.ndarray:
  """Lays a global interior out as per-core blocks, each with zero halos."""
  local = [n // c for n, c in zip(interior.shape, cores)]
  full = np.zeros(
      [c * (n + 2 * halo) for c, n in zip(cores, local)], interior.dtype
  )
  for index in itertools.product(*(range(c) for c in cores)):
    src = tuple(slice(i * n, (i + 1) * n) for i, n in zip(index, local))
    dst = tuple(
        slice(i * (n + 2 * halo) + halo, i * (n + 2 * halo) + halo + n)
        for i, n in zip(index, local)
    )
    full[dst] = interior[src]
  return full


def _blocks(full: np.ndarray, cores) -> dict[tuple[int, ...], np.ndarray]:
  """Splits a global array into its per-core blocks, halos included."""
  local = [n // c for n, c in zip(full.shape, cores)]
  return {
      index: full[
          tuple(slice(i * n, (i + 1) * n) for i, n in zip(index, local))
      ]
      for index in itertools.product(*(range(c) for c in cores))
  }


def _strip_halos(full: np.ndarray, cores, halo: int) -> np.ndarray:
  """The inverse of `_with_halos`."""
  local = [n // c - 2 * halo for n, c in zip(full.shape, cores)]
  interior = np.zeros([c * n for c, n in zip(cores, local)], full.dtype)
  inner = (slice(halo, -halo),) * 3
  for index, block in _blocks(full, cores).items():
    dst = tuple(slice(i * n, (i + 1) * n) for i, n in zip(index, local))
    interior[dst] = block[inner]
  return interior


def _reference_matrices(setup: _Setup) -> list[np.ndarray]:
  """The 1D second-difference matrices, one per array axis.

  The fine operator is `halo_update_fn` followed by the 3-point `kdd` kernel,
  so each axis contributes a tridiagonal matrix whose boundary rows follow the
  halo convention: Neumann copies the edge cell into the halo, Dirichlet sets
  the halo to zero, periodic wraps.

  Args:
    setup: The problem.

  Returns:
    Matrices in data axis order, in float64.
  """
  gp = setup.grid_params
  extents = gp.to_data_axis_order(
      gp.core_nx * gp.cx, gp.core_ny * gp.cy, gp.core_nz * gp.cz
  )
  spacings = gp.to_data_axis_order(gp.dx, gp.dy, gp.dz)
  kinds = gp.to_data_axis_order(*setup.bc_xyz)
  matrices = []
  for n, h, kind in zip(extents, spacings, kinds):
    t = -2.0 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)
    if kind == 'P':
      t[0, -1] += 1.0
      t[-1, 0] += 1.0
    else:
      if kind[0] == 'N':
        t[0, 0] += 1.0
      if kind[1] == 'N':
        t[-1, -1] += 1.0
    matrices.append(t / h**2)
  return matrices


def _apply_reference(u: np.ndarray, matrices) -> np.ndarray:
  u = np.asarray(u, np.float64)
  return (
      np.einsum('ia,ajk->ijk', matrices[0], u)
      + np.einsum('ja,iak->ijk', matrices[1], u)
      + np.einsum('ka,ija->ijk', matrices[2], u)
  )


def _solve_reference(b: np.ndarray, matrices) -> np.ndarray:
  """The exact discrete solution, by diagonalising each axis.

  For a singular operator (no Dirichlet face) this is the minimum-norm
  solution: the null-space component is dropped.

  Args:
    b: The right hand side, interior only.
    matrices: From `_reference_matrices`.

  Returns:
    The solution in float64.
  """
  eig = [np.linalg.eigh(m) for m in matrices]
  (l0, v0), (l1, v1), (l2, v2) = eig
  b_hat = np.einsum('ai,bj,ck,abc->ijk', v0, v1, v2, np.asarray(b, np.float64))
  denominator = l0[:, None, None] + l1[None, :, None] + l2[None, None, :]
  scale = np.max(np.abs(denominator))
  singular = np.abs(denominator) < 1e-10 * scale
  x_hat = np.where(singular, 0.0, b_hat / np.where(singular, 1.0, denominator))
  return np.einsum('ia,jb,kc,abc->ijk', v0, v1, v2, x_hat)


def _smooth_reference(
    b: np.ndarray, matrices, weight: float, sweeps: int
) -> np.ndarray:
  """Weighted Jacobi sweeps from a zero guess.

  Every row is divided by the diagonal of an interior row, as in the solver,
  which probes one cell with `_interior_diagonal` and uses it everywhere.

  Args:
    b: The right hand side, interior only.
    matrices: From `_reference_matrices`.
    weight: The damping factor.
    sweeps: How many sweeps.

  Returns:
    The iterate in float64.
  """
  diagonal = sum(m[1, 1] for m in matrices)
  x = np.zeros(b.shape)
  for _ in range(sweeps):
    x += weight * (b - _apply_reference(x, matrices)) / diagonal
  return x


def _is_singular(bc_xyz) -> bool:
  return not any('D' in kind for kind in bc_xyz)


def _rhs(setup: _Setup, seed: int = 0, smooth: bool = False) -> np.ndarray:
  """A right hand side, interior only, in data axis order."""
  gp = setup.grid_params
  shape = gp.to_data_axis_order(
      gp.core_nx * gp.cx, gp.core_ny * gp.cy, gp.core_nz * gp.cz
  )
  if smooth:
    grids = np.meshgrid(
        *[(np.arange(n) + 0.5) / n for n in shape], indexing='ij'
    )
    b = np.prod([np.cos(np.pi * g) + 0.3 * g for g in grids], axis=0)
  else:
    b = np.random.default_rng(seed).standard_normal(shape)
  if _is_singular(setup.bc_xyz):
    b -= b.mean()
  return b.astype(np.float32)


def _solve(
    setup: _Setup,
    rhs: np.ndarray,
    p0: np.ndarray | None = None,
    use_default_halo_update: bool = False,
):
  """Runs `solve` under `jit` and `shard_map`.

  Args:
    setup: The problem.
    rhs: The right hand side, global interior in data axis order.
    p0: The initial guess, likewise. Zero if `None`.
    use_default_halo_update: Pass `halo_update_fn=None` to the solver.

  Returns:
    `(x, residual, result)`: the global interior of the solution, the reported
    residual norm, and the raw result dict with the halos still in place.
  """
  halo = setup.grid_params.halo_width
  spec = sharding.PartitionSpec(*setup.grid_params.data_axis_order)
  fn = functools.partial(
      setup.solver.solve,
      mesh=setup.mesh,
      halo_update_fn=None if use_default_halo_update else setup.halo_update_fn,
  )
  fn = shard_map.shard_map(
      fn,
      mesh=setup.mesh,
      in_specs=(spec, spec),
      out_specs={
          _X: spec,
          _RESIDUAL: sharding.PartitionSpec(),
          _ITERATIONS: sharding.PartitionSpec(),
      },
      check_rep=False,
  )
  if p0 is None:
    p0 = np.zeros_like(rhs)
  result = jax.jit(fn)(
      jnp.asarray(_with_halos(rhs, setup.cores, halo)),
      jnp.asarray(_with_halos(p0.astype(np.float32), setup.cores, halo)),
  )
  x = _strip_halos(np.asarray(result[_X]), setup.cores, halo)
  return x, float(result[_RESIDUAL]), result


def _relative_error(actual, expected, mean_free: bool = False) -> float:
  actual = np.asarray(actual, np.float64)
  expected = np.asarray(expected, np.float64)
  if mean_free:
    actual = actual - actual.mean()
    expected = expected - expected.mean()
  return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def _to_xyz(u: np.ndarray, data_axis_order: str) -> np.ndarray:
  return np.transpose(u, [data_axis_order.index(a) for a in 'xyz'])


def _from_xyz(u: np.ndarray, data_axis_order: str) -> np.ndarray:
  return np.transpose(u, ['xyz'.index(a) for a in data_axis_order])


# ============================================================================
# Building blocks.
# ============================================================================


class IndexingAndShiftTest(parameterized.TestCase):

  def test_first_and_last_select_edge_slices(self):
    u = jnp.arange(24.0).reshape(2, 3, 4)
    for axis in range(3):
      np.testing.assert_array_equal(
          u[multigrid_solver._first(axis)], jnp.take(u, 0, axis=axis)
      )
      np.testing.assert_array_equal(
          u[multigrid_solver._last(u, axis)], jnp.take(u, -1, axis=axis)
      )

  @parameterized.parameters(0, 1, 2)
  def test_shifts_take_the_supplied_plane_at_the_end(self, axis):
    u = jnp.arange(60.0).reshape(3, 4, 5)
    plane_shape = list(u.shape)
    del plane_shape[axis]
    plane = jnp.full(plane_shape, -1.0)

    up = multigrid_solver._shift_up(u, axis, plane)
    down = multigrid_solver._shift_down(u, axis, plane)

    n = u.shape[axis]
    np.testing.assert_array_equal(
        jnp.take(up, jnp.arange(n - 1), axis=axis),
        jnp.take(u, jnp.arange(1, n), axis=axis),
    )
    np.testing.assert_array_equal(jnp.take(up, n - 1, axis=axis), plane)
    np.testing.assert_array_equal(
        jnp.take(down, jnp.arange(1, n), axis=axis),
        jnp.take(u, jnp.arange(n - 1), axis=axis),
    )
    np.testing.assert_array_equal(jnp.take(down, 0, axis=axis), plane)

  @parameterized.parameters(0, 1, 2)
  def test_interleave_alternates_inputs(self, axis):
    even = jnp.arange(24.0).reshape(2, 3, 4)
    odd = -even - 1.0

    woven = multigrid_solver._interleave(even, odd, axis)

    n = even.shape[axis]
    self.assertEqual(woven.shape[axis], 2 * n)
    np.testing.assert_array_equal(
        jnp.take(woven, jnp.arange(0, 2 * n, 2), axis=axis), even
    )
    np.testing.assert_array_equal(
        jnp.take(woven, jnp.arange(1, 2 * n, 2), axis=axis), odd
    )


class TransferTest(parameterized.TestCase):

  def test_prolongation_interpolates_linearly_along_one_axis(self):
    coarse = jnp.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1, 1)

    fine = multigrid_solver._prolongate_axis(
        coarse, 0, _stub_exchange('dirichlet')
    )

    # Each fine cell takes 3/4 from its own coarse cell and 1/4 from the
    # neighbour it leans towards; zero beyond the ends.
    expected = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 3.0]
    np.testing.assert_allclose(fine.ravel(), expected, rtol=1e-6)

  @parameterized.parameters('neumann', 'periodic')
  def test_prolongation_preserves_constants(self, kind):
    coarse = jnp.full((4, 3, 2), 2.5)

    fine = multigrid_solver._prolongate(coarse, _stub_exchange(kind), (0, 1, 2))

    self.assertEqual(fine.shape, (8, 6, 4))
    np.testing.assert_allclose(fine, 2.5, rtol=1e-6)

  def test_restriction_of_a_constant_sums_the_weights(self):
    fine = jnp.ones((8, 8, 8))

    coarse = multigrid_solver._restrict(fine, _stub_exchange('neumann'), (0, 1))

    # Each coarsened axis weighs 3/4 + 3/4 + 1/4 + 1/4 = 2.
    self.assertEqual(coarse.shape, (4, 4, 8))
    np.testing.assert_allclose(coarse, 4.0, rtol=1e-6)

  @parameterized.parameters(
      *itertools.product(
          ('dirichlet', 'neumann'),
          ((0,), (1,), (2,), (0, 2), (0, 1, 2)),
      )
  )
  def test_restriction_is_the_transpose_of_prolongation(self, kind, axes):
    exchange = _stub_exchange(kind)
    fine_shape = (8, 6, 4)
    coarse_shape = multigrid_solver._coarsened(fine_shape, axes)

    p = _dense(
        lambda c: multigrid_solver._prolongate(c, exchange, axes), coarse_shape
    )
    r = _dense(
        lambda f: multigrid_solver._restrict(f, exchange, axes), fine_shape
    )

    np.testing.assert_allclose(r, p.T, atol=1e-6)

  # TODO(wqing): Known defect. `_restrict_axis` takes the plane past the high
  # face of `lower` from `upper`'s exchange (and vice versa). That is the
  # transpose only where the outside cell is a reflection or zero, i.e. a
  # Neumann or Dirichlet face. Where it is a copy of another cell -- a periodic
  # face or an inter-device seam -- the transpose needs `lower`'s own
  # neighbour. Remove `expectedFailure` once fixed.
  @unittest.expectedFailure
  def test_restriction_is_the_transpose_of_prolongation_periodic(self):
    exchange = _stub_exchange('periodic')
    fine_shape = (8, 6, 4)
    for axes in ((0,), (1,), (2,), (0, 1, 2)):
      coarse_shape = multigrid_solver._coarsened(fine_shape, axes)
      p = _dense(
          lambda c, axes=axes: multigrid_solver._prolongate(c, exchange, axes),
          coarse_shape,
      )
      r = _dense(
          lambda f, axes=axes: multigrid_solver._restrict(f, exchange, axes),
          fine_shape,
      )
      np.testing.assert_allclose(r, p.T, atol=1e-6)

  def test_restriction_does_not_depend_on_axis_order(self):
    fine = jax.random.normal(jax.random.PRNGKey(0), (8, 8, 8))
    exchange = _stub_exchange('neumann')

    forward = multigrid_solver._restrict(fine, exchange, (0, 1, 2))
    backward = multigrid_solver._restrict(fine, exchange, (2, 1, 0))

    np.testing.assert_allclose(forward, backward, rtol=1e-5, atol=1e-5)


class GalerkinOperatorTest(parameterized.TestCase):

  @parameterized.parameters(
      *itertools.product(('dirichlet', 'neumann'), ((0,), (1, 2), (0, 1, 2)))
  )
  def test_coarse_operator_is_the_galerkin_product(self, kind, axes):
    exchange = _stub_exchange(kind)
    fine_shape = (8, 8, 4)
    coarse_shape = multigrid_solver._coarsened(fine_shape, axes)
    fine_operator = _stub_laplacian((1.0, 0.5, 2.0), exchange)

    a = _dense(fine_operator, fine_shape)
    p = _dense(
        lambda c: multigrid_solver._prolongate(c, exchange, axes), coarse_shape
    )
    a_coarse = _dense(
        functools.partial(
            multigrid_solver._coarsen_operator,
            fine_operator=fine_operator,
            exchange=exchange,
            axes=axes,
        ),
        coarse_shape,
    )

    np.testing.assert_allclose(a_coarse, p.T @ a @ p, rtol=1e-5, atol=1e-4)

  def test_interior_diagonal_of_the_laplacian(self):
    spacings = (1.0, 0.5, 2.0)
    operator = _stub_laplacian(spacings, _stub_exchange('dirichlet'))

    diagonal = multigrid_solver._interior_diagonal(operator, (8, 8, 8))

    expected = -2.0 * sum(h**-2 for h in spacings)
    np.testing.assert_allclose(diagonal, expected, rtol=1e-6)

  def test_interior_diagonal_of_a_galerkin_operator(self):
    exchange = _stub_exchange('neumann')
    fine_shape = (16, 16, 16)
    axes = (0, 1, 2)
    coarse_shape = multigrid_solver._coarsened(fine_shape, axes)
    operator = functools.partial(
        multigrid_solver._coarsen_operator,
        fine_operator=_stub_laplacian((1.0, 1.0, 1.0), exchange),
        exchange=exchange,
        axes=axes,
    )

    diagonal = multigrid_solver._interior_diagonal(operator, coarse_shape)

    a_coarse = _dense(operator, coarse_shape)
    centre = np.ravel_multi_index(
        tuple(n // 2 for n in coarse_shape), coarse_shape
    )
    np.testing.assert_allclose(diagonal, a_coarse[centre, centre], rtol=1e-5)

  # On the smallest block the hierarchy may produce, only the two middle cells
  # along each axis are clear of the boundary rows, so the two tests below fail
  # if the probe is moved one cell off centre. They use Neumann faces because
  # those are the rows whose diagonal differs from the interior one.

  def test_interior_diagonal_of_the_laplacian_on_the_smallest_block(self):
    spacings = (1.0, 0.5, 2.0)
    operator = _stub_laplacian(spacings, _stub_exchange('neumann'))
    shape = (multigrid_solver._MIN_PER_CORE_EXTENT,) * 3

    diagonal = multigrid_solver._interior_diagonal(operator, shape)

    # A boundary row would give half of this along the axis it lies on.
    expected = -2.0 * sum(h**-2 for h in spacings)
    np.testing.assert_allclose(diagonal, expected, rtol=1e-6)

  def test_interior_diagonal_of_a_galerkin_operator_on_the_smallest_block(self):
    exchange = _stub_exchange('neumann')
    coarse_shape = (multigrid_solver._MIN_PER_CORE_EXTENT,) * 3
    operator = functools.partial(
        multigrid_solver._coarsen_operator,
        fine_operator=_stub_laplacian((1.0, 0.5, 2.0), exchange),
        exchange=exchange,
        axes=(0, 1, 2),
    )

    diagonal = multigrid_solver._interior_diagonal(operator, coarse_shape)

    a_coarse = _dense(operator, coarse_shape)
    interior = a_coarse.diagonal().reshape(coarse_shape)[1:-1, 1:-1, 1:-1]
    np.testing.assert_allclose(interior, diagonal, rtol=1e-5)

  @parameterized.parameters('neumann', 'periodic')
  def test_interior_diagonal_on_a_two_cell_axis(self, kind):
    # A quasi-two-dimensional grid may carry a span of two cells, which is
    # never halved. With Neumann faces the probe has no interior row to land
    # on along that axis, since both cells touch a face. The entry it reads is
    # still the right one, because both rows along that axis are the same
    # boundary row.
    shape = (8, 8, 2)
    operator = _stub_laplacian((1.0, 0.5, 2.0), _stub_exchange(kind))

    diagonal = multigrid_solver._interior_diagonal(operator, shape)

    # The rows clear of the faces of the two long axes.
    interior = _dense(operator, shape).diagonal().reshape(shape)[1:-1, 1:-1, :]
    np.testing.assert_allclose(interior, diagonal, rtol=1e-6)


class SmootherTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.operator = _stub_laplacian(
        (1.0, 1.0, 1.0), _stub_exchange('dirichlet')
    )
    self.diagonal = jnp.float32(-6.0)
    key_u, key_b = jax.random.split(jax.random.PRNGKey(1))
    self.u = jax.random.normal(key_u, (8, 8, 8))
    self.b = jax.random.normal(key_b, (8, 8, 8))

  def test_zero_sweeps_is_the_identity(self):
    smoothed = multigrid_solver._smooth(
        self.u, self.b, self.operator, self.diagonal, 0.7, 0
    )
    np.testing.assert_array_equal(smoothed, self.u)

  def test_one_sweep_is_weighted_jacobi(self):
    smoothed = multigrid_solver._smooth(
        self.u, self.b, self.operator, self.diagonal, 0.7, 1
    )
    expected = self.u + 0.7 * (self.b - self.operator(self.u)) / self.diagonal
    np.testing.assert_allclose(smoothed, expected, rtol=1e-5, atol=1e-5)

  def test_sweeps_damp_the_most_oscillatory_error(self):
    index = jnp.indices((8, 8, 8)).sum(axis=0)
    checkerboard = jnp.where(index % 2 == 0, 1.0, -1.0)
    zeros = jnp.zeros_like(checkerboard)

    # Error equation: with b = 0 the iterate is the error.
    smoothed = multigrid_solver._smooth(
        checkerboard, zeros, self.operator, self.diagonal, 2.0 / 3.0, 3
    )

    self.assertLess(
        float(jnp.linalg.norm(smoothed)),
        0.1 * float(jnp.linalg.norm(checkerboard)),
    )


class CoarseningPlanTest(parameterized.TestCase):

  def test_coarsened_halves_only_the_given_axes(self):
    self.assertEqual(
        multigrid_solver._coarsened((16, 8, 6), (0, 2)), (8, 8, 3)
    )
    self.assertEqual(multigrid_solver._coarsened((16, 8, 6), ()), (16, 8, 6))

  @parameterized.named_parameters(
      dict(
          testcase_name='all_axes',
          shape=(16, 16, 16),
          cores=(1, 1, 1),
          coarsest=(8, 8, 8),
          expected=(0, 1, 2),
      ),
      # 17 // 2 is 8, which clears both floors, so parity alone excludes it.
      dict(
          testcase_name='odd_extent',
          shape=(16, 17, 16),
          cores=(1, 1, 1),
          coarsest=(8, 8, 8),
          expected=(0, 2),
      ),
      dict(
          testcase_name='odd_extent_on_several_cores',
          shape=(9, 16, 16),
          cores=(2, 1, 1),
          coarsest=(8, 8, 8),
          expected=(1, 2),
      ),
      dict(
          testcase_name='global_floor',
          shape=(16, 8, 16),
          cores=(1, 1, 1),
          coarsest=(8, 8, 8),
          expected=(0, 2),
      ),
      dict(
          testcase_name='per_core_floor',
          shape=(6, 16, 16),
          cores=(4, 1, 1),
          coarsest=(8, 8, 8),
          expected=(1, 2),
      ),
      dict(
          testcase_name='global_extent_counts_every_core',
          shape=(8, 8, 8),
          cores=(2, 1, 1),
          coarsest=(8, 8, 8),
          expected=(0,),
      ),
      dict(
          testcase_name='custom_coarsest',
          shape=(32, 32, 32),
          cores=(1, 1, 1),
          coarsest=(8, 32, 16),
          expected=(0, 2),
      ),
      dict(
          testcase_name='nothing',
          shape=(8, 8, 8),
          cores=(1, 1, 1),
          coarsest=(8, 8, 8),
          expected=(),
      ),
  )
  def test_coarsenable_axes(self, shape, cores, coarsest, expected):
    self.assertEqual(
        multigrid_solver._coarsenable_axes(shape, cores, coarsest), expected
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='full_coarsening',
          shape=(32, 32, 32),
          cores=(1, 1, 1),
          expected=[(0, 1, 2), (0, 1, 2)],
      ),
      dict(
          testcase_name='per_axis',
          shape=(64, 16, 8),
          cores=(1, 1, 1),
          expected=[(0, 1), (0,), (0,)],
      ),
      dict(
          testcase_name='depth_is_capped',
          shape=(512, 512, 512),
          cores=(1, 1, 1),
          expected=[(0, 1, 2)] * multigrid_solver._MAX_COARSENINGS,
      ),
      dict(
          testcase_name='partitioned',
          shape=(16, 16, 16),
          cores=(2, 1, 1),
          expected=[(0, 1, 2), (0,)],
      ),
      dict(
          testcase_name='nothing_to_coarsen',
          shape=(8, 8, 8),
          cores=(1, 1, 1),
          expected=[],
      ),
      # No extent is a power of two. Each axis is halved until it would drop
      # below the floor (24 -> 12, 40 -> 20 -> 10) or turns odd
      # (72 -> 36 -> 18 -> 9), and is carried at that extent from then on.
      dict(
          testcase_name='not_a_power_of_two',
          shape=(24, 40, 72),
          cores=(1, 1, 1),
          expected=[(0, 1, 2), (1, 2), (2,)],
      ),
      # 17 would halve to 8 and clear both floors, so only its parity keeps it
      # out of the plan.
      dict(
          testcase_name='odd_axis_is_never_halved',
          shape=(32, 17, 32),
          cores=(1, 1, 1),
          expected=[(0, 2), (0, 2)],
      ),
      # A thin span is carried at its full extent while the long axes are
      # halved: two cells would halve to one and four to two, both below the
      # per-core floor.
      dict(
          testcase_name='quasi_two_dimensional',
          shape=(2, 64, 64),
          cores=(1, 1, 1),
          expected=[(1, 2), (1, 2), (1, 2)],
      ),
      dict(
          testcase_name='quasi_one_dimensional',
          shape=(4, 4, 64),
          cores=(1, 1, 1),
          expected=[(2,), (2,), (2,)],
      ),
  )
  def test_coarsening_plan(self, shape, cores, expected):
    self.assertEqual(
        multigrid_solver._coarsening_plan(shape, cores, (8, 8, 8)), expected
    )


class ValidationTest(parameterized.TestCase):

  def test_coarsest_shape_of_eight_is_accepted(self):
    multigrid_solver._validate_coarsest_shape((8, 8, 8))
    multigrid_solver._validate_coarsest_shape((8, 16, 32))

  def test_coarsest_shape_below_eight_is_rejected(self):
    with self.assertRaisesRegex(ValueError, 'at least 8'):
      multigrid_solver._validate_coarsest_shape((8, 4, 8))

  def test_coarsenable_grid_is_accepted(self):
    multigrid_solver._validate_coarsenable(
        _make_grid_params((16, 16, 16)), (8, 8, 8)
    )

  def test_grid_at_the_floor_is_accepted(self):
    multigrid_solver._validate_coarsenable(
        _make_grid_params((8, 6, 8)), (8, 8, 8)
    )

  @parameterized.named_parameters(
      ('quasi_two_dimensional', (32, 32, 2)),
      ('quasi_one_dimensional', (64, 2, 2)),
  )
  def test_thin_grid_is_accepted(self, interior_xyz):
    # One axis that can be halved is enough, so a thin span is no reason to
    # refuse the grid.
    multigrid_solver._validate_coarsenable(
        _make_grid_params(interior_xyz), (8, 8, 8)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='odd',
          interior_xyz=(15, 15, 15),
          cores_xyz=(1, 1, 1),
          message='which is odd',
      ),
      # Halves to 8, which clears both floors: refused for its parity alone.
      dict(
          testcase_name='odd_above_both_floors',
          interior_xyz=(17, 17, 17),
          cores_xyz=(1, 1, 1),
          message='dim_0 is 17, which is odd',
      ),
      dict(
          testcase_name='per_core_too_small',
          interior_xyz=(24, 24, 24),
          cores_xyz=(4, 4, 4),
          message='below the floor of 4',
      ),
      dict(
          testcase_name='global_too_small',
          interior_xyz=(14, 14, 14),
          cores_xyz=(1, 1, 1),
          message='below the coarsest extent 8',
      ),
  )
  def test_uncoarsenable_grid_is_rejected(
      self, interior_xyz, cores_xyz, message
  ):
    with self.assertRaisesRegex(ValueError, message):
      multigrid_solver._validate_coarsenable(
          _make_grid_params(interior_xyz, cores_xyz), (8, 8, 8)
      )


# ============================================================================
# Constructor and hierarchy.
# ============================================================================


class ConstructorTest(absltest.TestCase):

  def _solver(self, gp, option=None, use_stretched_grid=(False,) * 3):
    return multigrid_solver.MultigridSolver(
        gp,
        get_kernel_fn.ApplyKernelSliceOp(gp),
        option or _make_solver_option(),
        use_stretched_grid,
    )

  def test_reads_the_configuration(self):
    solver = self._solver(
        _make_grid_params(),
        _make_solver_option(
            num_iterations=7, n_smooth=3, weight=0.5, n_coarse=2
        ),
    )
    self.assertEqual(solver._num_iterations, 7)
    self.assertEqual(solver._n_smooth, 3)
    self.assertAlmostEqual(solver._weight, 0.5)
    self.assertEqual(solver._n_coarse, 2)
    self.assertEqual(solver.boundary_conditions, [])

  def test_coarsest_shape_and_cores_are_in_data_axis_order(self):
    gp = _make_grid_params(
        interior_xyz=(64, 32, 16), cores_xyz=(4, 2, 1), data_axis_order='zxy'
    )
    solver = self._solver(gp, _make_solver_option(coarsest_xyz=(16, 8, 32)))
    self.assertEqual(solver._coarsest_shape, (32, 16, 8))
    self.assertEqual(solver._cores, (1, 4, 2))

  def test_default_coarsest_shape(self):
    solver = self._solver(_make_grid_params())
    self.assertEqual(solver._coarsest_shape, (8, 8, 8))

  def test_rejects_stretched_grid(self):
    with self.assertRaisesRegex(NotImplementedError, 'stretched'):
      self._solver(_make_grid_params(), use_stretched_grid=(False, True, False))

  def test_rejects_small_coarsest_subgrid_shape(self):
    with self.assertRaisesRegex(ValueError, 'coarsest subgrid'):
      self._solver(
          _make_grid_params(), _make_solver_option(coarsest_xyz=(8, 4, 8))
      )

  def test_rejects_uncoarsenable_grid(self):
    with self.assertRaisesRegex(ValueError, 'cannot halve a single axis'):
      self._solver(_make_grid_params((14, 14, 14)))

  def test_accepts_grid_at_the_floor(self):
    self._solver(_make_grid_params((8, 8, 8)))

  def test_factory_creates_a_multigrid_solver(self):
    gp = _make_grid_params()
    solver = poisson_solver.poisson_solver_factory(
        gp, get_kernel_fn.ApplyKernelSliceOp(gp), _make_solver_option()
    )
    self.assertIsInstance(solver, multigrid_solver.MultigridSolver)

  def test_factory_forwards_stretched_grid(self):
    gp = _make_grid_params()
    with self.assertRaises(NotImplementedError):
      poisson_solver.poisson_solver_factory(
          gp,
          get_kernel_fn.ApplyKernelSliceOp(gp),
          _make_solver_option(),
          (True, False, False),
      )


class HierarchyTest(absltest.TestCase):

  def test_levels_follow_the_coarsening_plan(self):
    # Data axis order zxy: the array is (z, x, y) = (8, 32, 16).
    gp = _make_grid_params(interior_xyz=(32, 16, 8))
    solver = multigrid_solver.MultigridSolver(
        gp, get_kernel_fn.ApplyKernelSliceOp(gp), _make_solver_option()
    )
    exchange = _stub_exchange('dirichlet')
    solver._exchange = exchange
    spacings = (1.0, 0.5, 2.0)

    levels = solver._hierarchy(
        (8, 32, 16), _stub_laplacian(spacings, exchange)
    )

    self.assertEqual(
        [level.shape for level in levels], [(8, 32, 16), (8, 16, 8), (8, 8, 8)]
    )
    self.assertEqual(
        [level.coarsen_axes for level in levels], [(1, 2), (1,), ()]
    )
    np.testing.assert_allclose(
        levels[0].diagonal, -2.0 * sum(h**-2 for h in spacings), rtol=1e-6
    )
    for level in levels:
      self.assertLess(float(level.diagonal), 0.0)
      self.assertEqual(
          level.operator(jnp.zeros(level.shape)).shape, level.shape
      )


# ============================================================================
# The boundary exchange used by the transfers.
# ============================================================================


class ExchangeTest(parameterized.TestCase):

  def _planes(self, setup: _Setup, u: np.ndarray, axis: int):
    """Runs `_make_exchange` on the global interior `u`, per device."""
    exchange = multigrid_solver._make_exchange(
        setup.mesh, setup.grid_params, _bc_spec(setup.bc_xyz)
    )
    spec = sharding.PartitionSpec(*setup.grid_params.data_axis_order)

    def fn(block):
      low, high = exchange(block, axis)
      return jnp.stack([low, high])[None, None, None]

    out = shard_map.shard_map(
        fn,
        mesh=setup.mesh,
        in_specs=(spec,),
        out_specs=sharding.PartitionSpec(*setup.grid_params.data_axis_order),
        check_rep=False,
    )(jnp.asarray(u))
    return np.asarray(out)  # [c0, c1, c2, 2, ...plane]

  @parameterized.parameters(
      *itertools.product((0, 1, 2), ('DD', 'NN', 'DN', 'P'))
  )
  def test_single_device_planes_follow_the_boundary_condition(self, axis, kind):
    # zxy: array axis 0 is z, 1 is x, 2 is y.
    name = 'zxy'[axis]
    bc_xyz = ['NN', 'NN', 'NN']
    bc_xyz['xyz'.index(name)] = kind
    setup = _make_setup(interior_xyz=(16, 16, 16), bc_xyz=tuple(bc_xyz))
    u = np.random.default_rng(0).standard_normal((16, 16, 16)).astype(
        np.float32
    )

    planes = self._planes(setup, u, axis)[0, 0, 0]

    first = np.take(u, 0, axis=axis)
    last = np.take(u, -1, axis=axis)
    if kind == 'P':
      expected_low, expected_high = last, first
    else:
      expected_low = first if kind[0] == 'N' else np.zeros_like(first)
      expected_high = last if kind[1] == 'N' else np.zeros_like(last)
    np.testing.assert_allclose(planes[0], expected_low, rtol=1e-6)
    np.testing.assert_allclose(planes[1], expected_high, rtol=1e-6)

  @parameterized.parameters('DN', 'P')
  def test_planes_at_inter_device_seams_come_from_the_neighbour(self, kind):
    # Two devices along x, which is array axis 1 in zxy.
    setup = _make_setup(
        interior_xyz=(16, 16, 16),
        cores_xyz=(2, 1, 1),
        bc_xyz=(kind, 'NN', 'NN'),
    )
    u = np.random.default_rng(1).standard_normal((16, 16, 16)).astype(
        np.float32
    )

    planes = self._planes(setup, u, axis=1)

    left, right = u[:, :8, :], u[:, 8:, :]
    # The seam.
    np.testing.assert_allclose(planes[0, 0, 0, 1], right[:, 0, :], rtol=1e-6)
    np.testing.assert_allclose(planes[0, 1, 0, 0], left[:, -1, :], rtol=1e-6)
    # The faces of the domain.
    if kind == 'P':
      np.testing.assert_allclose(planes[0, 0, 0, 0], right[:, -1, :])
      np.testing.assert_allclose(planes[0, 1, 0, 1], left[:, 0, :])
    else:
      np.testing.assert_allclose(planes[0, 0, 0, 0], 0.0)
      np.testing.assert_allclose(planes[0, 1, 0, 1], right[:, -1, :])


# ============================================================================
# `solve`.
# ============================================================================

# Boundary condition configurations, in x, y, z order.
_BC_CASES = dict(
    dirichlet=('DD', 'DD', 'DD'),
    neumann=('NN', 'NN', 'NN'),
    outlet=('ND', 'NN', 'NN'),
    periodic=('P', 'P', 'DD'),
    periodic_neumann=('P', 'NN', 'NN'),
)

# Quasi-two-dimensional grids, in x, y, z order: x and y are resolved and z is
# a thin span that is never halved, so every level coarsens x and y alone.
# These take two coarsenings; see
# `test_quasi_two_dimensional_grid_with_one_coarsening_converges` for one.
_QUASI_2D_PROBLEMS = (
    dict(interior_xyz=(32, 32, 4), bc_xyz=('DD', 'DD', 'P')),
    dict(interior_xyz=(32, 32, 2), bc_xyz=('DD', 'DD', 'P')),
    dict(interior_xyz=(32, 32, 4), bc_xyz=('DD', 'DD', 'NN')),
    # Not a power of two either: x goes 24 -> 12 and y goes 40 -> 20 -> 10.
    dict(interior_xyz=(24, 40, 4), bc_xyz=('DD', 'DD', 'P')),
)

# Quasi-one-dimensional grids: only x is resolved, and y and z are thin
# periodic spans. As in `SolveTest._SINGLE_AXIS_PROBLEM` every level coarsens x
# alone, but the spans here are below the floor rather than at it.
_QUASI_1D_PROBLEMS = dict(
    four_cell_spans=dict(interior_xyz=(64, 4, 4), bc_xyz=('DD', 'P', 'P')),
    two_cell_spans=dict(interior_xyz=(64, 2, 2), bc_xyz=('DD', 'P', 'P')),
    # x goes 48 -> 24 -> 12.
    not_a_power_of_two=dict(interior_xyz=(48, 4, 4), bc_xyz=('DD', 'P', 'P')),
)


class SolveTest(parameterized.TestCase):

  def test_requires_boundary_conditions(self):
    setup = _make_setup(set_boundary_conditions=False)
    with self.assertRaisesRegex(ValueError, 'boundary_conditions'):
      _solve(setup, _rhs(setup))

  def test_zero_rhs_gives_zero_solution_and_the_expected_outputs(self):
    setup = _make_setup(bc_xyz=_BC_CASES['outlet'], num_iterations=3)
    rhs = np.zeros((16, 16, 16), np.float32)

    x, residual, result = _solve(setup, rhs)

    np.testing.assert_array_equal(x, 0.0)
    self.assertEqual(residual, 0.0)
    self.assertEqual(result[_X].shape, (18, 18, 18))
    self.assertEqual(result[_ITERATIONS].dtype, jnp.int32)
    self.assertEqual(int(result[_ITERATIONS]), 3)
    self.assertEqual(result[_RESIDUAL].shape, ())

  @parameterized.named_parameters(
      *[
          dict(testcase_name=f'{name}_halo{halo}', bc_xyz=bc_xyz, halo=halo)
          for (name, bc_xyz), halo in itertools.product(
              _BC_CASES.items(), (1, 2)
          )
      ],
      dict(
          testcase_name='conv_kernel',
          bc_xyz=_BC_CASES['outlet'],
          halo=2,
          kernel='conv',
      ),
  )
  def test_converges_to_the_discrete_solution(
      self, bc_xyz, halo, kernel='slice'
  ):
    setup = _make_setup(
        bc_xyz=bc_xyz, halo=halo, kernel=kernel, num_iterations=30
    )
    matrices = _reference_matrices(setup)
    rhs = _rhs(setup)

    x, _, _ = _solve(setup, rhs)

    expected = _solve_reference(rhs, matrices)
    # The tolerance is the float32 floor, not the convergence: one Dirichlet
    # face leaves a poorly conditioned smooth mode, and the outlet case
    # settles at 2.2e-3 however many more cycles it is given.
    self.assertLess(
        _relative_error(x, expected, mean_free=_is_singular(bc_xyz)), 5e-3
    )

  @parameterized.named_parameters(
      ('dirichlet', _BC_CASES['dirichlet']),
      ('neumann', _BC_CASES['neumann']),
      ('outlet', _BC_CASES['outlet']),
  )
  def test_residual_decreases_with_every_cycle(self, bc_xyz):
    residuals = []
    for num_iterations in (1, 2, 3, 4, 6):
      setup = _make_setup(
          interior_xyz=(32, 32, 32),
          bc_xyz=bc_xyz,
          num_iterations=num_iterations,
      )
      residuals.append(_solve(setup, _rhs(setup))[1])

    initial = np.linalg.norm(_rhs(setup).astype(np.float64))
    self.assertLess(residuals[0], initial)
    for before, after in zip(residuals, residuals[1:]):
      self.assertLess(after, before)

  @parameterized.named_parameters(
      ('dirichlet', _BC_CASES['dirichlet']),
      ('outlet', _BC_CASES['outlet']),
      ('periodic', _BC_CASES['periodic']),
  )
  def test_reported_residual_is_that_of_the_returned_solution(self, bc_xyz):
    setup = _make_setup(bc_xyz=bc_xyz, num_iterations=1)
    rhs = _rhs(setup)

    x, residual, _ = _solve(setup, rhs)

    expected = np.linalg.norm(
        rhs.astype(np.float64) - _apply_reference(x, _reference_matrices(setup))
    )
    np.testing.assert_allclose(residual, expected, rtol=1e-3)

  @parameterized.named_parameters(
      dict(testcase_name='one_device', cores_xyz=(1, 1, 1)),
      dict(testcase_name='eight_devices', cores_xyz=(2, 2, 2)),
  )
  def test_returned_halos_are_filled(self, cores_xyz):
    # Dirichlet on x, Neumann on y, periodic on z.
    setup = _make_setup(
        bc_xyz=('DD', 'NN', 'P'), cores_xyz=cores_xyz, num_iterations=1
    )

    _, _, result = _solve(setup, _rhs(setup))

    expected = np.asarray(
        jax.jit(
            shard_map.shard_map(
                setup.halo_update_fn,
                mesh=setup.mesh,
                in_specs=sharding.PartitionSpec(*'zxy'),
                out_specs=sharding.PartitionSpec(*'zxy'),
                check_rep=False,
            )
        )(result[_X])
    )
    np.testing.assert_allclose(np.asarray(result[_X]), expected, rtol=1e-6)
    # Not trivially: the halos must hold something other than zero.
    halo_mask = np.ones(result[_X].shape, bool)
    for block in _blocks(halo_mask, setup.cores).values():
      block[1:-1, 1:-1, 1:-1] = False
    self.assertGreater(np.abs(np.asarray(result[_X])[halo_mask]).max(), 0.0)

  def test_initial_guess_is_used(self):
    setup = _make_setup(bc_xyz=_BC_CASES['outlet'], num_iterations=1)
    rhs = _rhs(setup)
    exact = _solve_reference(rhs, _reference_matrices(setup))

    _, residual_from_zero, _ = _solve(setup, rhs)
    x, residual_from_exact, _ = _solve(setup, rhs, p0=exact)

    self.assertLess(residual_from_exact, 1e-2 * residual_from_zero)
    self.assertLess(_relative_error(x, exact), 1e-4)

  def test_default_halo_update_is_homogeneous_neumann(self):
    setup = _make_setup(bc_xyz=_BC_CASES['neumann'], num_iterations=2)
    rhs = _rhs(setup)

    explicit, _, _ = _solve(setup, rhs)
    default, _, _ = _solve(setup, rhs, use_default_halo_update=True)

    np.testing.assert_allclose(default, explicit, rtol=1e-6, atol=1e-6)

  def test_solution_is_linear_in_the_rhs(self):
    setup = _make_setup(bc_xyz=_BC_CASES['outlet'], num_iterations=2)
    b1, b2 = _rhs(setup, seed=1), _rhs(setup, seed=2)

    x1, _, _ = _solve(setup, b1)
    x2, _, _ = _solve(setup, b2)
    x12, _, _ = _solve(setup, 2.0 * b1 - b2)

    np.testing.assert_allclose(
        x12, 2.0 * x1 - x2, atol=1e-5 * np.abs(x1).max()
    )

  def _asymmetry(self, setup: _Setup) -> float:
    """`|<M b1, b2> - <b1, M b2>|`, relative, for one cycle `M` from zero."""
    b1, b2 = _rhs(setup, seed=3), _rhs(setup, seed=4)
    m_b1, _, _ = _solve(setup, b1)
    m_b2, _, _ = _solve(setup, b2)
    lhs = float(np.vdot(m_b1.astype(np.float64), b2))
    rhs = float(np.vdot(b1.astype(np.float64), m_b2))
    return abs(lhs - rhs) / float(np.linalg.norm(m_b1) * np.linalg.norm(b2))

  @parameterized.named_parameters(
      ('dirichlet', _BC_CASES['dirichlet']),
      ('neumann', _BC_CASES['neumann']),
      ('outlet', _BC_CASES['outlet']),
  )
  def test_one_cycle_from_zero_is_a_symmetric_operator(self, bc_xyz):
    # Symmetry is what lets the cycle precondition conjugate gradient, and it
    # rests on the restriction being the exact transpose of the prolongation.
    setup = _make_setup(bc_xyz=bc_xyz, num_iterations=1)
    self.assertLess(self._asymmetry(setup), 1e-5)

  # TODO(wqing): Known defect; see
  # `test_restriction_is_the_transpose_of_prolongation_periodic`. Measured
  # asymmetry: 6.8e-4 periodic, 5.7e-3 on eight devices.
  @unittest.expectedFailure
  def test_one_cycle_is_symmetric_with_periodic_faces_and_seams(self):
    for bc_xyz, cores_xyz in (
        (_BC_CASES['periodic'], (1, 1, 1)),
        (_BC_CASES['neumann'], (2, 2, 2)),
    ):
      setup = _make_setup(
          bc_xyz=bc_xyz, cores_xyz=cores_xyz, num_iterations=1
      )
      self.assertLess(self._asymmetry(setup), 1e-5)

  # Global interior (32, 16, 16) in x, y, z with mixed conditions so that both
  # faces and seams are exercised on every axis. The spacing is kept (nearly)
  # isotropic so that these tests isolate the seams from the smoother
  # instability of `test_anisotropic_grid_converges`.
  _PARTITION_PROBLEM = dict(
      interior_xyz=(32, 16, 16),
      bc_xyz=('ND', 'P', 'DN'),
      lengths_xyz=(2.0, 1.0, 1.0),
  )
  _PARTITIONS = (
      ((2, 1, 1), None),
      ((1, 2, 1), None),
      ((1, 1, 2), None),
      ((2, 2, 2), None),
      # Four cores leave x at eight per core, which the per-core floor of four
      # stops after one halving; a coarser global floor keeps the plan.
      ((4, 1, 1), (16, 8, 8)),
  )

  def test_partitions_under_test_keep_the_coarsening_plan(self):
    # Precondition for the partition tests: a different plan is a different
    # algorithm, so its result is expected to differ.
    for cores_xyz, coarsest_xyz in self._PARTITIONS:
      plans = []
      for cores in ((1, 1, 1), cores_xyz):
        setup = _make_setup(
            cores_xyz=cores,
            coarsest_xyz=coarsest_xyz,
            **self._PARTITION_PROBLEM,
        )
        solver = setup.solver
        local = tuple(n // c for n, c in zip((16, 32, 16), solver._cores))
        plans.append(
            multigrid_solver._coarsening_plan(
                local, solver._cores, solver._coarsest_shape
            )
        )
      self.assertEqual(plans[0], plans[1], cores_xyz)

  @parameterized.named_parameters(
      ('two_devices', (2, 1, 1)),
      ('eight_devices', (2, 2, 2)),
  )
  def test_partitioned_solve_converges_to_the_discrete_solution(
      self, cores_xyz
  ):
    setup = _make_setup(
        cores_xyz=cores_xyz, num_iterations=30, **self._PARTITION_PROBLEM
    )
    rhs = _rhs(setup)

    x, _, _ = _solve(setup, rhs)

    expected = _solve_reference(rhs, _reference_matrices(setup))
    self.assertLess(_relative_error(x, expected), 5e-3)

  # TODO(wqing): Known defect. Weighted Jacobi with the default weight is
  # unstable on the Galerkin operators of an anisotropic grid: with spacing
  # halved along one axis, lambda_max(D^-1 A) reaches 2.8 on the first coarse
  # level and 3.6 on the second, so `weight * lambda_max` exceeds 2 and every
  # sweep amplifies the most oscillatory mode. The error after 30 cycles is
  # 3e14. Isotropic full coarsening peaks at 2.1, which is why the cubic cases
  # converge. Remove `expectedFailure` once the smoother weight is bounded by
  # the operator's spectrum (e.g. Chebyshev or an estimated lambda_max).
  @unittest.expectedFailure
  def test_anisotropic_grid_converges(self):
    setup = _make_setup(
        interior_xyz=(32, 16, 16),
        bc_xyz=('ND', 'NN', 'DN'),
        lengths_xyz=(2.0, 1.0, 0.5),
        num_iterations=30,
    )
    rhs = _rhs(setup)

    x, _, _ = _solve(setup, rhs)

    expected = _solve_reference(rhs, _reference_matrices(setup))
    self.assertLess(_relative_error(x, expected), 5e-3)

  # TODO(wqing): Known defect; see
  # `test_restriction_is_the_transpose_of_prolongation_periodic`. At a seam the
  # restriction is not the transpose of the prolongation, so the partitioned
  # cycle is a different (and non-symmetric) operator from the single-device
  # one, and their results differ well beyond roundoff.
  @unittest.expectedFailure
  def test_result_does_not_depend_on_the_partition(self):
    for cores_xyz, coarsest_xyz in self._PARTITIONS:
      single = _make_setup(
          num_iterations=3, coarsest_xyz=coarsest_xyz, **self._PARTITION_PROBLEM
      )
      partitioned = _make_setup(
          cores_xyz=cores_xyz,
          num_iterations=3,
          coarsest_xyz=coarsest_xyz,
          **self._PARTITION_PROBLEM,
      )
      rhs = _rhs(single)

      x_single, residual_single, _ = _solve(single, rhs)
      x_partitioned, residual_partitioned, _ = _solve(partitioned, rhs)

      self.assertLess(_relative_error(x_partitioned, x_single), 1e-5)
      np.testing.assert_allclose(
          residual_partitioned, residual_single, rtol=1e-4
      )

  def test_result_does_not_depend_on_the_data_axis_order(self):
    rhs_xyz = (
        np.random.default_rng(5).standard_normal((32, 16, 8)).astype(np.float32)
    )
    solutions = {}
    for order in ('zxy', 'xyz', 'yzx'):
      setup = _make_setup(
          interior_xyz=(32, 16, 8),
          bc_xyz=('DN', 'NN', 'P'),
          lengths_xyz=(2.0, 1.0, 0.5),
          data_axis_order=order,
          num_iterations=4,
      )
      x, _, _ = _solve(setup, _from_xyz(rhs_xyz, order))
      solutions[order] = _to_xyz(x, order)

    self.assertLess(_relative_error(solutions['xyz'], solutions['zxy']), 1e-5)
    self.assertLess(_relative_error(solutions['yzx'], solutions['zxy']), 1e-5)

  def test_w_cycle_reduces_the_residual_at_least_as_much(self):
    v = _make_setup(interior_xyz=(32, 32, 32), num_iterations=3, n_coarse=1)
    w = _make_setup(interior_xyz=(32, 32, 32), num_iterations=3, n_coarse=2)
    rhs = _rhs(v)

    _, residual_v, _ = _solve(v, rhs)
    _, residual_w, _ = _solve(w, rhs)

    self.assertLessEqual(residual_w, residual_v)

  # Only x can be halved: y and z are already at the floor of eight. Three
  # halvings take x to a spacing seven times that of y and z, so the coarsest
  # level is strongly anisotropic, and the smoothest error along x is what is
  # left there.
  _SINGLE_AXIS_PROBLEM = dict(
      interior_xyz=(64, 8, 8),
      bc_xyz=('DD', 'NN', 'NN'),
      lengths_xyz=(8.0, 1.0, 1.0),
  )

  def test_single_axis_coarsening_reduces_the_residual(self):
    residuals = []
    for num_iterations in (10, 30):
      setup = _make_setup(
          num_iterations=num_iterations, **self._SINGLE_AXIS_PROBLEM
      )
      rhs = _rhs(setup)
      residuals.append(_solve(setup, rhs)[1])

    logging.info('Single-axis residuals after 10 and 30 cycles: %s', residuals)
    self.assertLess(residuals[0], np.linalg.norm(rhs))
    self.assertLess(residuals[1], residuals[0])

  # TODO(wqing): Known defect. Each x-only level has lambda_max(D^-1 A) of 3.2
  # to 3.3, so weighted Jacobi at the default weight amplifies the most
  # oscillatory mode (see `test_anisotropic_grid_converges`); the coarsest
  # level is also smoothed rather than solved. After 30 cycles the error is
  # still 0.59 of the solution. Remove `expectedFailure` once both are fixed.
  @unittest.expectedFailure
  def test_single_axis_coarsening_converges(self):
    setup = _make_setup(num_iterations=30, **self._SINGLE_AXIS_PROBLEM)
    rhs = _rhs(setup)

    x, _, _ = _solve(setup, rhs)

    self.assertLess(
        _relative_error(x, _solve_reference(rhs, _reference_matrices(setup))),
        5e-3,
    )

  def test_grid_at_the_floor_still_reduces_the_residual(self):
    # Nothing to coarsen: the cycle degenerates to its smoother.
    setup = _make_setup(
        interior_xyz=(8, 8, 8), bc_xyz=_BC_CASES['dirichlet'], num_iterations=2
    )
    rhs = _rhs(setup)

    _, residual, _ = _solve(setup, rhs)

    self.assertLess(residual, np.linalg.norm(rhs))

  def test_grid_at_the_floor_smooths_four_times_harder(self):
    # Nothing to coarsen, so one cycle is weighted Jacobi alone: `n_smooth`
    # sweeps of pre-smoothing, then `4 * n_smooth` in place of the coarse-grid
    # correction. Three sweeps per round rather than the default two, so that
    # 3 * 4 coincides with neither 3 + 4 nor a power of either.
    n_smooth = 3
    weight = 0.6667
    setup = _make_setup(
        interior_xyz=(8, 8, 8),
        bc_xyz=_BC_CASES['dirichlet'],
        num_iterations=1,
        n_smooth=n_smooth,
        weight=weight,
    )
    rhs = _rhs(setup)

    x, _, _ = _solve(setup, rhs)

    expected = _smooth_reference(
        rhs, _reference_matrices(setup), weight, sweeps=n_smooth + 4 * n_smooth
    )
    self.assertLess(_relative_error(x, expected), 1e-4)

  def test_incompatible_rhs_on_a_singular_problem(self):
    # All Neumann with a rhs whose mean is not zero: no solution exists. The
    # part of the iterate outside the null space must still converge to the
    # least-squares solution, and nothing may blow up.
    setup = _make_setup(bc_xyz=_BC_CASES['neumann'], num_iterations=30)
    rhs = _rhs(setup) + np.float32(0.5)

    x, residual, _ = _solve(setup, rhs)

    self.assertTrue(np.all(np.isfinite(x)))
    self.assertTrue(np.isfinite(residual))
    expected = _solve_reference(rhs, _reference_matrices(setup))
    self.assertLess(_relative_error(x, expected, mean_free=True), 2e-3)

  @parameterized.named_parameters(
      dict(testcase_name='16', n=16, threshold=0.25),
      dict(testcase_name='32', n=32, threshold=0.25),
  )
  def test_convergence_factor(self, n, threshold):
    # A ceiling on the measured residual reduction per cycle, to catch a
    # regression in convergence that still converges. Measured: 0.205 at 16^3
    # and 0.212 at 32^3.
    cycles = 6
    setup = _make_setup(
        interior_xyz=(n, n, n),
        bc_xyz=_BC_CASES['dirichlet'],
        num_iterations=cycles,
    )
    rhs = _rhs(setup)

    _, residual, _ = _solve(setup, rhs)

    factor = (residual / np.linalg.norm(rhs)) ** (1.0 / cycles)
    logging.info('Convergence factor at %d^3: %f', n, factor)
    self.assertLess(factor, threshold)

  @parameterized.named_parameters(
      # One halving leaves x at 9, which is odd: coarsest grid 9 x 8 x 8.
      dict(testcase_name='ends_odd', interior_xyz=(18, 16, 16)),
      # One halving leaves x at 15: coarsest grid 15 x 8 x 8.
      dict(testcase_name='ends_odd_and_wide', interior_xyz=(30, 16, 16)),
      # x stops at 12 while y and z are halved once more: 12 x 8 x 8.
      dict(testcase_name='axes_stop_at_different_levels',
           interior_xyz=(24, 32, 32)),
  )
  def test_grid_that_is_not_a_power_of_two_converges(self, interior_xyz):
    # An axis that is not a power of two is not padded: it stops being halved
    # once it is odd or would drop below the floor, and the other axes go on
    # without it. The spacing is uniform so that this measures the coarsening
    # and not the anisotropy of `test_anisotropic_grid_converges`. Measured
    # factors: 0.205, 0.211 and 0.208, against 0.205 at 16^3 and 0.212 at
    # 32^3; measured errors: 9.1e-5, 1.1e-4 and 8.2e-5.
    cycles = 6
    setup = _make_setup(
        interior_xyz=interior_xyz,
        lengths_xyz=tuple(n / 16.0 for n in interior_xyz),
        bc_xyz=_BC_CASES['dirichlet'],
        num_iterations=cycles,
    )
    rhs = _rhs(setup)

    x, residual, _ = _solve(setup, rhs)

    factor = (residual / np.linalg.norm(rhs)) ** (1.0 / cycles)
    error = _relative_error(
        x, _solve_reference(rhs, _reference_matrices(setup))
    )
    logging.info(
        'Convergence factor at %s: %f, error %e', interior_xyz, factor, error
    )
    self.assertLess(factor, 0.25)
    self.assertLess(error, 1e-3)

  def _solve_with_uniform_spacing(self, interior_xyz, bc_xyz, **option_kwargs):
    """Solves a problem that has a Dirichlet face, on a uniform grid.

    The spacing is the same along every axis, so that a thin grid measures the
    coarsening and not the anisotropy of `test_anisotropic_grid_converges`.

    Args:
      interior_xyz: The global interior extent per axis.
      bc_xyz: The boundary conditions, as `_bc_spec` takes them.
      **option_kwargs: Forwarded to `_make_solver_option`.

    Returns:
      `(residual, error, rhs_norm)`: the reported residual norm, the error
      relative to the exact discrete solution, and the norm of the right hand
      side.
    """
    setup = _make_setup(
        interior_xyz=interior_xyz,
        # `GridParametrization` spaces n nodes `length / (n - 1)` apart.
        lengths_xyz=tuple((n - 1) / 16.0 for n in interior_xyz),
        bc_xyz=bc_xyz,
        **option_kwargs,
    )
    rhs = _rhs(setup)

    x, residual, _ = _solve(setup, rhs)

    error = _relative_error(
        x, _solve_reference(rhs, _reference_matrices(setup))
    )
    logging.info(
        'Uniform grid %s, %s, %s: residual %e, error %e',
        interior_xyz,
        bc_xyz,
        option_kwargs,
        residual,
        error,
    )
    return residual, error, float(np.linalg.norm(rhs))

  @parameterized.named_parameters(
      dict(
          testcase_name='four_cell_span',
          interior_xyz=(16, 16, 4),
          bc_xyz=('DD', 'DD', 'P'),
      ),
      dict(
          testcase_name='two_cell_span',
          interior_xyz=(16, 16, 2),
          bc_xyz=('DD', 'DD', 'P'),
      ),
      # The span along another array axis: x is axis 1 in zxy, z is axis 0.
      dict(
          testcase_name='span_along_x',
          interior_xyz=(4, 16, 16),
          bc_xyz=('P', 'DD', 'DD'),
      ),
  )
  def test_quasi_two_dimensional_grid_with_one_coarsening_converges(
      self, interior_xyz, bc_xyz
  ):
    # The thin span is never halved, so the one coarsening halves the two long
    # axes alone. Measured errors after 30 cycles: 4.0e-6 with a span of four
    # cells, 2.9e-6 with a span of two and 3.5e-6 with the span along x.
    _, error, _ = self._solve_with_uniform_spacing(
        interior_xyz, bc_xyz, num_iterations=30
    )

    self.assertLess(error, 1e-4)

  # TODO(wqing): Known defect. Once two levels halve the long axes alone, the
  # cycle diverges at the default weight: after 6 cycles the residual is `inf`
  # on the periodic spans, 2.4e12 on the Neumann span and 6.1e11 on
  # 24 x 40 x 4, and after 30 cycles every one of them is NaN. This is
  # consistent with the smoother instability noted in `_smooth`: lowering the
  # weight to 0.4 restores convergence (the test below) and 0.5 does not.
  # Remove `expectedFailure` once the smoother weight is bounded by the
  # operator's spectrum.
  @unittest.expectedFailure
  def test_quasi_two_dimensional_grid_converges(self):
    for problem in _QUASI_2D_PROBLEMS:
      _, error, _ = self._solve_with_uniform_spacing(
          num_iterations=30, **problem
      )
      self.assertLess(error, 5e-3)

  def test_quasi_two_dimensional_grid_converges_with_a_lower_weight(self):
    # The same hierarchy as the first problem of
    # `test_quasi_two_dimensional_grid_converges` with nothing changed but the
    # smoother weight, which pins that defect on the smoother rather than on
    # the transfers or the coarse operators of a thin grid. Measured error
    # after 30 cycles: 2.9e-4.
    _, error, _ = self._solve_with_uniform_spacing(
        num_iterations=30, weight=0.4, **_QUASI_2D_PROBLEMS[0]
    )

    self.assertLess(error, 5e-3)

  @parameterized.named_parameters(
      *[
          dict(testcase_name=name, problem=problem)
          for name, problem in _QUASI_1D_PROBLEMS.items()
      ]
  )
  def test_quasi_one_dimensional_grid_reduces_the_residual(self, problem):
    # Measured residuals after 6 and 30 cycles: 1.7 and 0.96 on 64 x 4 x 4,
    # 1.0 and 0.20 on 64 x 2 x 2, and 0.65 and 0.26 on 48 x 4 x 4.
    residuals = []
    for num_iterations in (6, 30):
      residual, _, rhs_norm = self._solve_with_uniform_spacing(
          num_iterations=num_iterations, **problem
      )
      residuals.append(residual)

    self.assertLess(residuals[0], rhs_norm)
    self.assertLess(residuals[1], residuals[0])

  # TODO(wqing): Known defect; see `test_single_axis_coarsening_converges`.
  # The residual falls but the solution is not reached: after 30 cycles the
  # error is still 0.67, 0.56 and 0.56 of the solution, and a lower weight
  # does not help (0.74 at 0.5 and 0.83 at 0.3 on 64 x 4 x 4). Remove
  # `expectedFailure` once that defect is fixed.
  @unittest.expectedFailure
  def test_quasi_one_dimensional_grid_converges(self):
    for problem in _QUASI_1D_PROBLEMS.values():
      _, error, _ = self._solve_with_uniform_spacing(
          num_iterations=30, **problem
      )
      self.assertLess(error, 5e-3)


if __name__ == '__main__':
  absltest.main()
