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

# Copyright 2026 Google LLC
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

import os

from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
import jax
from jax.experimental import shard_map
import jax.numpy as jnp
import numpy as np
from swirl_lm.base import parameters_pb2
from swirl_lm.boundary_condition import immersed_boundary_method_pb2
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.boundary_condition import immersed_boundary_ghost_point as ghost_point
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import signed_distance

# Expose several host devices so that multi-replica meshes can be built. This
# has to be set before JAX initializes its backend.
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

_GridParametrization = grid_parametrization.GridParametrization
_P = jax.sharding.PartitionSpec

# Most tests use a horizontal wall in a (6, 6, 8) array with a halo width of 1
# and unit grid spacing: solid in the layers `k <= 2`, fluid above. Its ghost
# cells are the topmost solid layer `k = 2` restricted to the interior
# `1 <= i, j <= 4`. These are their flat indices, `(i * 6 + j) * 8 + k`.
_WALL_SHAPE = (6, 6, 8)
_WALL_GHOST_INDICES = [
    58, 66, 74, 82,
    106, 114, 122, 130,
    154, 162, 170, 178,
    202, 210, 218, 226,
]  # pyformat: disable

# `u = 2 k + 1` along the wall-normal axis. The ghost layer `k = 2` holds 5.
_LINEAR_PROFILE = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]

# The state keys of the packed geometry, in the order they are declared.
_GEOMETRY_KEYS = [
    'ib_geom_gc_index',
    'ib_geom_self_weight',
    'ib_geom_bi_weight',
    'ib_geom_zeta_dirichlet',
    'ib_geom_separation',
    'ib_geom_distance',
    'ib_geom_valid',
    'ib_geom_corner_index',
    'ib_geom_alpha',
]


def _unit_grid(shape) -> list[np.ndarray]:
  """Cell-centre coordinates with unit spacing starting at 0, per axis."""
  return [np.arange(n, dtype=np.float64) for n in shape]


def _wall(height: float) -> signed_distance.SignedDistanceFn:
  """A horizontal wall: solid below `z = height`, fluid above."""
  return signed_distance.plane((0.0, 0.0, height), (0.0, 0.0, 1.0))


def _varying_along_last_axis(profile, shape) -> np.ndarray:
  """An array of `shape` holding `profile` along the last axis everywhere."""
  return np.broadcast_to(np.asarray(profile, np.float32), shape).copy()


def _with_corrupted_wall_ghost_cells(field: np.ndarray) -> jax.Array:
  """Overwrites the ghost cells of the horizontal wall with a bogus value."""
  corrupted = field.copy()
  corrupted[1:5, 1:5, 2] = 99.0
  return jnp.asarray(corrupted)


def _make_mesh(shape, axis_names=('x', 'y', 'z')) -> jax.sharding.Mesh:
  devices = np.array(jax.devices()[: int(np.prod(shape))]).reshape(shape)
  return jax.sharding.Mesh(devices, axis_names)


def _make_grid_params(
    num_cores_x: int = 1, data_axis_order: str = 'xyz'
) -> _GridParametrization:
  """A grid whose per-core arrays are 6 x 6 x 8 in (x, y, z), halo width 1.

  The spacing is 1. Halos included, the cell centres run from -1 to
  `4 * num_cores_x` along x, from -1 to 4 along y and from -1 to 6 along z, so
  a local index `k` along z sits at `z = k - 1`.

  Args:
    num_cores_x: The number of cores along x.
    data_axis_order: The order of the array axes.

  Returns:
    The grid parametrization.
  """
  return _GridParametrization.create_from_grid_lengths_and_etc(
      grid_lengths={'x': 4.0 * num_cores_x - 1.0, 'y': 3.0, 'z': 5.0},
      computation_shape={'x': num_cores_x, 'y': 1, 'z': 1},
      subgrid_shape={'x': 6, 'y': 6, 'z': 8},
      halo_width=1,
      data_axis_order=data_axis_order,
  )


def _single_source_geometry(
    gc_index, source_index, valid=None
) -> ghost_point.ShardedGhostPoints:
  """A local geometry whose ghost cells each read one cell with full weight."""
  n = len(gc_index)
  one_hot = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  return ghost_point.ShardedGhostPoints(
      gc_index=jnp.array(gc_index, jnp.int32),
      corner_index=jnp.array([[s] * 8 for s in source_index], jnp.int32),
      alpha=jnp.array([one_hot] * n, jnp.float32),
      self_weight=jnp.zeros(n),
      bi_weight=jnp.zeros(n),
      zeta_dirichlet=-jnp.ones(n),
      separation=jnp.ones(n),
      distance=jnp.full(n, 0.5),
      valid=jnp.ones(n, bool) if valid is None else jnp.array(valid),
  )


def _make_params(
    boundary_models_textpb: str = '',
) -> parameters_lib.SwirlLMParameters:
  config = text_format.Parse(
      f"""
      grid_params {{
        computation_shape {{ dim_0: 1 dim_1: 1 dim_2: 1 }}
        length {{ dim_0: 1.0 dim_1: 1.0 dim_2: 1.0 }}
        grid_size {{ dim_0: 8 dim_1: 8 dim_2: 8 }}
        dt: 0.001
      }}
      {boundary_models_textpb}
      """,
      parameters_pb2.SwirlLMParameters(),
  )
  return parameters_lib.SwirlLMParameters(config)


def _parse_geometry(textpb: str) -> immersed_boundary_method_pb2.IBGeometry:
  return text_format.Parse(textpb, immersed_boundary_method_pb2.IBGeometry())


class GetInteriorMaskTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='surface_between_cells', height=2.5),
      # A cell centre exactly on the surface is not in the fluid.
      dict(testcase_name='surface_through_cell_centres', height=2.0),
  )
  def test_is_true_only_strictly_inside_the_fluid(self, height):
    mask = ghost_point.get_interior_mask(_wall(height), _unit_grid((3, 4, 6)))

    expected = np.zeros((3, 4, 6), bool)
    expected[:, :, 3:] = True
    np.testing.assert_array_equal(mask, expected)

  def test_dims_maps_data_axes_to_physical_dimensions(self):
    # The array axes are (z, x, y), so the wall cuts the first axis.
    z, x, y = _unit_grid((6, 3, 4))

    mask = ghost_point.get_interior_mask(_wall(2.5), [z, x, y], dims=(2, 0, 1))

    expected = np.zeros((6, 3, 4), bool)
    expected[3:] = True
    np.testing.assert_array_equal(mask, expected)

  def test_uses_actual_coordinates_of_stretched_grid(self):
    x, y, _ = _unit_grid((3, 4, 5))
    z = np.array([0.0, 1.0, 2.0, 4.0, 8.0])

    mask = ghost_point.get_interior_mask(_wall(5.0), [x, y, z])

    expected = np.zeros((3, 4, 5), bool)
    expected[:, :, 4:] = True
    np.testing.assert_array_equal(mask, expected)

  def test_wrong_number_of_coordinate_arrays_raises(self):
    with self.assertRaisesRegex(ValueError, 'Expected 3 coordinate arrays'):
      ghost_point.get_interior_mask(_wall(2.5), _unit_grid((3, 4)))

  def test_dims_that_is_not_a_permutation_raises(self):
    with self.assertRaisesRegex(ValueError, '`dims` must be a permutation'):
      ghost_point.get_interior_mask(
          _wall(2.5), _unit_grid((3, 4, 6)), dims=(0, 0, 1)
      )

  def test_coordinate_array_with_single_entry_raises(self):
    with self.assertRaisesRegex(ValueError, 'at least 2 entries'):
      ghost_point.get_interior_mask(_wall(2.5), _unit_grid((3, 1, 6)))

  def test_coordinate_array_that_is_not_increasing_raises(self):
    x, y, z = _unit_grid((3, 4, 6))

    with self.assertRaisesRegex(ValueError, 'must be strictly increasing'):
      ghost_point.get_interior_mask(_wall(2.5), [x, y[::-1], z])


class GhostCellMaskTest(parameterized.TestCase):

  def test_marks_solid_cell_with_fluid_neighbours(self):
    fluid = np.ones((5, 5, 5), bool)
    fluid[2, 2, 2] = False

    mask = ghost_point.ghost_cell_mask(fluid, halo_width=1)

    expected = np.zeros((5, 5, 5), bool)
    expected[2, 2, 2] = True
    np.testing.assert_array_equal(mask, expected)

  def test_solid_cell_touching_fluid_only_at_a_corner_counts(self):
    fluid = np.zeros((4, 4, 4), bool)
    fluid[1, 1, 1] = True

    mask = ghost_point.ghost_cell_mask(fluid, halo_width=0)

    # The whole 27-point neighbourhood of the one fluid cell, except itself.
    expected = np.zeros((4, 4, 4), bool)
    expected[0:3, 0:3, 0:3] = True
    expected[1, 1, 1] = False
    np.testing.assert_array_equal(mask, expected)

  def test_neighbourhood_does_not_wrap_around_the_array(self):
    fluid = np.zeros((4, 4, 4), bool)
    fluid[0] = True

    mask = ghost_point.ghost_cell_mask(fluid, halo_width=0)

    # Only the layer next to the fluid; the layer at the far face is not.
    expected = np.zeros((4, 4, 4), bool)
    expected[1] = True
    np.testing.assert_array_equal(mask, expected)

  @parameterized.named_parameters(
      dict(testcase_name='no_halo', halo_width=0, kept=slice(0, 6)),
      dict(testcase_name='halo_of_1', halo_width=1, kept=slice(1, 5)),
      dict(testcase_name='halo_of_2', halo_width=2, kept=slice(2, 4)),
  )
  def test_excludes_ghost_cells_in_halo_layers(self, halo_width, kept):
    fluid = np.zeros((6, 6, 6), bool)
    fluid[:, :, 3:] = True

    mask = ghost_point.ghost_cell_mask(fluid, halo_width)

    expected = np.zeros((6, 6, 6), bool)
    expected[kept, kept, 2] = True
    np.testing.assert_array_equal(mask, expected)

  def test_ghost_layer_inside_the_halo_gives_empty_mask(self):
    fluid = np.zeros((6, 6, 6), bool)
    fluid[:, :, 1:] = True

    mask = ghost_point.ghost_cell_mask(fluid, halo_width=1)

    np.testing.assert_array_equal(mask, np.zeros((6, 6, 6), bool))

  def test_negative_halo_width_raises(self):
    with self.assertRaisesRegex(ValueError, '`halo_width` must be non-neg'):
      ghost_point.ghost_cell_mask(np.ones((4, 4, 4), bool), halo_width=-1)


class BuildWallTest(absltest.TestCase):

  def test_ghost_cells_are_top_solid_layer_inside_the_halos(self):
    geom = ghost_point.build(_wall(2.5), _unit_grid(_WALL_SHAPE), halo_width=1)

    self.assertEqual(geom.num_ghost_points, 16)
    self.assertEqual(geom.shape, _WALL_SHAPE)
    np.testing.assert_array_equal(geom.gc_index, _WALL_GHOST_INDICES)

  def test_image_point_mirrors_ghost_cell_across_the_surface(self):
    geom = ghost_point.build(_wall(2.5), _unit_grid(_WALL_SHAPE), halo_width=1)

    # The surface is half a cell above the ghost cell, so the image point is a
    # full cell above it.
    np.testing.assert_allclose(geom.distance, np.full(16, 0.5), atol=1e-6)
    np.testing.assert_allclose(geom.probe, np.full(16, 2.0), atol=1e-6)
    np.testing.assert_allclose(geom.separation, np.full(16, 1.0), atol=1e-6)
    np.testing.assert_allclose(
        geom.zeta_dirichlet, np.full(16, -1.0), atol=1e-6
    )
    np.testing.assert_allclose(
        geom.normal, np.tile([0.0, 0.0, 1.0], (16, 1)), atol=1e-6
    )

  def test_image_point_on_a_node_puts_full_weight_on_that_node(self):
    geom = ghost_point.build(_wall(2.5), _unit_grid(_WALL_SHAPE), halo_width=1)

    # The image point coincides with the fluid cell directly above the ghost
    # cell, which is the last of the eight stencil corners.
    np.testing.assert_allclose(
        geom.alpha,
        np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (16, 1)),
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        geom.corner_index[:, 7], np.array(_WALL_GHOST_INDICES) + 1
    )
    np.testing.assert_allclose(geom.self_weight, np.zeros(16), atol=1e-6)
    np.testing.assert_allclose(geom.bi_weight, np.zeros(16), atol=1e-6)

  def test_stencil_that_fits_reports_no_capping_and_no_fallback(self):
    geom = ghost_point.build(_wall(2.5), _unit_grid(_WALL_SHAPE), halo_width=1)

    self.assertEqual(geom.num_capped, 0)
    self.assertEqual(geom.num_fallback, 0)
    self.assertEqual(geom.min_probe, 2.0)
    np.testing.assert_array_equal(geom.fallback, np.zeros(16, bool))
    np.testing.assert_allclose(geom.max_reach_cells, (0.0, 0.0, 1.0), atol=1e-6)

  def test_output_fields_describe_the_local_array(self):
    geom = ghost_point.build(_wall(2.5), _unit_grid(_WALL_SHAPE), halo_width=1)

    expected_interior = np.zeros(_WALL_SHAPE, np.float32)
    expected_interior[:, :, 3:] = 1.0
    expected_ghost = np.zeros(_WALL_SHAPE, np.float32)
    expected_ghost[1:5, 1:5, 2] = 1.0
    np.testing.assert_array_equal(geom.interior_mask, expected_interior)
    np.testing.assert_array_equal(geom.ghost_mask, expected_ghost)
    np.testing.assert_allclose(
        geom.signed_distance,
        _varying_along_last_axis(
            [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5], _WALL_SHAPE
        ),
        atol=1e-6,
    )

  def test_summary_reports_count_and_smallest_probe(self):
    geom = ghost_point.build(_wall(2.5), _unit_grid(_WALL_SHAPE), halo_width=1)

    summary = geom.summary()

    self.assertIn('ghost points: 16', summary)
    self.assertIn('min m = 2.00', summary)
    self.assertNotIn('capped', summary)
    self.assertNotIn('fallback', summary)

  def test_weight_on_the_ghost_cell_itself_is_set_aside(self):
    # A quarter cell below the surface, the image point is midway between the
    # ghost cell and the fluid cell above it.
    geom = ghost_point.build(_wall(2.25), _unit_grid(_WALL_SHAPE), halo_width=1)

    np.testing.assert_allclose(geom.self_weight, np.full(16, 0.5), atol=1e-6)
    np.testing.assert_allclose(
        geom.alpha,
        np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], (16, 1)),
        atol=1e-6,
    )

  def test_stencil_follows_coordinates_of_stretched_grid(self):
    x, y, _ = _unit_grid(_WALL_SHAPE)
    z = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])

    geom = ghost_point.build(_wall(2.75), [x, y, z], halo_width=1)

    # The image point at z = 3.5 is three quarters of the way from the ghost
    # cell at z = 2 to the fluid cell at z = 4.
    np.testing.assert_array_equal(geom.gc_index, _WALL_GHOST_INDICES)
    np.testing.assert_allclose(geom.self_weight, np.full(16, 0.25), atol=1e-6)
    np.testing.assert_allclose(geom.alpha[:, 7], np.full(16, 0.75), atol=1e-6)
    np.testing.assert_allclose(geom.separation, np.full(16, 1.5), atol=1e-6)
    np.testing.assert_allclose(geom.max_reach_cells, (0.0, 0.0, 1.5), atol=1e-6)

  def test_probe_is_shortened_to_keep_image_point_inside_the_array(self):
    # With only one fluid layer above a wall at 2.625, mirroring the ghost
    # cell would put the image point at z = 3.25, beyond the last cell at 3.
    geom = ghost_point.build(_wall(2.625), _unit_grid((6, 6, 4)), halo_width=1)

    np.testing.assert_allclose(geom.probe, np.full(16, 1.6), atol=1e-6)
    np.testing.assert_allclose(
        geom.zeta_dirichlet, np.full(16, -1.0 / 0.6), atol=1e-5
    )
    np.testing.assert_allclose(geom.separation, np.full(16, 1.0), atol=1e-6)
    self.assertEqual(geom.num_capped, 16)
    self.assertEqual(geom.num_fallback, 0)
    self.assertAlmostEqual(geom.min_probe, 1.6, places=6)
    self.assertIn('capped on 16', geom.summary())

  def test_probe_too_short_to_extrapolate_falls_back_to_nearest_fluid(self):
    # The image point of a wall at 2.875 would have to be pulled in to 1.14
    # times the wall distance, where the extrapolation is ill-conditioned.
    geom = ghost_point.build(_wall(2.875), _unit_grid((6, 6, 4)), halo_width=1)

    np.testing.assert_array_equal(geom.fallback, np.ones(16, bool))
    self.assertEqual(geom.num_fallback, 16)
    # Every stencil corner is the fluid cell directly above the ghost cell.
    np.testing.assert_array_equal(
        geom.corner_index, np.tile(geom.gc_index[:, None] + 1, (1, 8))
    )
    np.testing.assert_allclose(
        geom.alpha,
        np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (16, 1)),
        atol=1e-6,
    )
    np.testing.assert_allclose(geom.zeta_dirichlet, np.zeros(16), atol=1e-6)
    np.testing.assert_allclose(geom.separation, np.full(16, 1.0), atol=1e-6)
    self.assertIn('16 (100.0%) on the nearest-fluid fallback', geom.summary())

  def test_ghost_cell_on_the_surface_falls_back_to_nearest_fluid(self):
    # The image point coincides with the ghost cell, so the whole stencil
    # weight is on the unknown itself.
    geom = ghost_point.build(_wall(2.0), _unit_grid(_WALL_SHAPE), halo_width=1)

    np.testing.assert_array_equal(geom.gc_index, _WALL_GHOST_INDICES)
    np.testing.assert_allclose(geom.distance, np.zeros(16), atol=1e-6)
    np.testing.assert_array_equal(geom.fallback, np.ones(16, bool))
    np.testing.assert_allclose(geom.self_weight, np.zeros(16), atol=1e-6)
    np.testing.assert_allclose(geom.zeta_dirichlet, np.zeros(16), atol=1e-6)

  def test_dims_maps_data_axes_to_physical_dimensions(self):
    # The array axes are (z, x, y), so the wall cuts the first axis and the
    # flat index is `(k * 6 + i) * 6 + j`.
    z, x, y = _unit_grid((8, 6, 6))

    geom = ghost_point.build(
        _wall(2.5), [z, x, y], halo_width=1, dims=(2, 0, 1)
    )

    expected_ghost_indices = [
        79, 80, 81, 82,
        85, 86, 87, 88,
        91, 92, 93, 94,
        97, 98, 99, 100,
    ]  # pyformat: disable
    np.testing.assert_array_equal(geom.gc_index, expected_ghost_indices)
    # The fluid cell above is one step along the first axis, 36 cells on.
    np.testing.assert_array_equal(
        geom.corner_index[:, 7], np.array(expected_ghost_indices) + 36
    )
    # The normal stays in physical (x, y, z) order; the reach is per data axis.
    np.testing.assert_allclose(
        geom.normal, np.tile([0.0, 0.0, 1.0], (16, 1)), atol=1e-6
    )
    np.testing.assert_allclose(geom.max_reach_cells, (1.0, 0.0, 0.0), atol=1e-6)

  def test_domain_without_solid_has_no_ghost_points(self):
    geom = ghost_point.build(
        _wall(-10.0), _unit_grid(_WALL_SHAPE), halo_width=1
    )

    self.assertEqual(geom.num_ghost_points, 0)
    self.assertEqual(geom.corner_index.shape, (0, 8))
    self.assertEqual(geom.alpha.shape, (0, 8))
    self.assertEqual(geom.min_probe, 2.0)
    np.testing.assert_array_equal(
        geom.ghost_mask, np.zeros(_WALL_SHAPE, np.float32)
    )
    np.testing.assert_array_equal(
        geom.interior_mask, np.ones(_WALL_SHAPE, np.float32)
    )

  def test_coordinate_array_that_is_not_increasing_raises(self):
    x, y, z = _unit_grid(_WALL_SHAPE)

    with self.assertRaisesRegex(ValueError, 'must be strictly increasing'):
      ghost_point.build(_wall(2.5), [x, y, z[::-1]], halo_width=1)


class BuildObliqueWallTest(absltest.TestCase):
  """A wall rising with x, through `z = 0.75 x + 2`, with the fluid above.

  Its unit normal is (-0.6, 0, 0.8). The ghost cell at (1, 1, 2) is 0.6 from
  the surface, so its image point is at (0.28, 1, 2.96), inside the cell with
  corners `i in {0, 1}`, `k in {2, 3}`. Of those, (0, 1, 2) is a solid cell in
  the halo, which is neither fluid nor a ghost cell, so nothing ever writes it.
  """

  # The flat index of (1, 1, 2) in a (6, 4, 8) array.
  _GHOST_INDEX = 42

  def _build(self) -> ghost_point.GhostPoints:
    return ghost_point.build(
        signed_distance.plane((0.0, 0.0, 2.0), (-3.0, 0.0, 4.0)),
        _unit_grid((6, 4, 8)),
        halo_width=1,
    )

  def test_weight_on_cell_nothing_writes_moves_to_the_body_intercept(self):
    geom = self._build()

    (row,) = np.flatnonzero(geom.gc_index == self._GHOST_INDEX)
    np.testing.assert_allclose(geom.distance[row], 0.6, atol=1e-5)
    # 0.72 * 0.04 from the halo solid cell (0, 1, 2).
    np.testing.assert_allclose(geom.bi_weight[row], 0.0288, atol=1e-5)
    # 0.28 * 0.04 from the ghost cell itself.
    np.testing.assert_allclose(geom.self_weight[row], 0.0112, atol=1e-5)
    # 0.72 * 0.96 and 0.28 * 0.96 from the fluid cells (0, 1, 3) and (1, 1, 3).
    np.testing.assert_allclose(
        geom.alpha[row],
        [0.0, 0.0, 0.0, 0.6912, 0.0, 0.0, 0.0, 0.2688],
        atol=1e-5,
    )

  def test_no_weight_is_left_on_a_cell_nothing_writes(self):
    geom = self._build()

    written = (geom.signed_distance > 0.0).reshape(-1)
    written[geom.gc_index] = True
    np.testing.assert_array_equal(geom.alpha[~written[geom.corner_index]], 0.0)

  def test_weights_still_partition_unity(self):
    geom = self._build()

    np.testing.assert_allclose(
        geom.alpha.sum(axis=1) + geom.self_weight + geom.bi_weight,
        np.ones(geom.num_ghost_points),
        atol=1e-6,
    )

  def test_dirichlet_reproduces_a_constant_field(self):
    geom = self._build()
    u = jnp.full((6, 4, 8), 7.0)

    zeta, gamma = ghost_point.dirichlet_coefficients(geom, 7.0)
    result = ghost_point.apply(u, geom, zeta, gamma)

    np.testing.assert_allclose(result, np.full((6, 4, 8), 7.0), atol=1e-5)

  def test_summary_reports_stencils_anchored_on_the_body_intercept(self):
    geom = self._build()

    self.assertIn(
        'degenerate stencils anchored on the body intercept', geom.summary()
    )


class BuildSphereTest(absltest.TestCase):
  """A sphere of radius 3.2 in the middle of a 16^3 grid of unit spacing.

  The body is convex and well away from the halos, so every stencil is made of
  fluid and ghost cells only and the scheme is exact for a linear field.
  """

  _GRADIENT = np.array([0.5, -0.25, 2.0])

  def _build(self) -> ghost_point.GhostPoints:
    return ghost_point.build(
        signed_distance.sphere((7.5, 7.5, 7.5), 3.2),
        _unit_grid((16, 16, 16)),
        halo_width=2,
    )

  def _linear_field(self, points: np.ndarray) -> np.ndarray:
    """`1 + 0.5 x - 0.25 y + 2 z` at points of shape `(..., 3)`."""
    return 1.0 + points @ self._GRADIENT

  def _grid_field(self) -> np.ndarray:
    points = np.stack(
        np.meshgrid(*_unit_grid((16, 16, 16)), indexing='ij'), axis=-1
    )
    return self._linear_field(points).astype(np.float32)

  def _body_intercepts(self, geom: ghost_point.GhostPoints) -> np.ndarray:
    # With unit spacing from 0 the coordinates of a cell are its indices.
    ghost_cells = np.stack(
        np.unravel_index(geom.gc_index, geom.shape), axis=-1
    ).astype(np.float64)
    return ghost_cells + geom.distance[:, None] * geom.normal

  def test_ghost_cells_lie_in_the_solid_at_the_recorded_distance(self):
    geom = self._build()

    phi_at_ghost_cells = geom.signed_distance.reshape(-1)[geom.gc_index]
    self.assertGreater(geom.num_ghost_points, 0)
    np.testing.assert_array_less(phi_at_ghost_cells, 0.0)
    np.testing.assert_allclose(geom.distance, -phi_at_ghost_cells, atol=1e-5)

  def test_normals_are_unit_vectors(self):
    geom = self._build()

    np.testing.assert_allclose(
        np.linalg.norm(geom.normal, axis=1),
        np.ones(geom.num_ghost_points),
        atol=1e-6,
    )

  def test_stencils_need_no_capping_fallback_or_body_intercept(self):
    geom = self._build()

    self.assertEqual(geom.num_capped, 0)
    self.assertEqual(geom.num_fallback, 0)
    np.testing.assert_array_equal(
        geom.bi_weight, np.zeros(geom.num_ghost_points, np.float32)
    )

  def test_weights_partition_unity(self):
    geom = self._build()

    np.testing.assert_allclose(
        geom.alpha.sum(axis=1) + geom.self_weight,
        np.ones(geom.num_ghost_points),
        atol=1e-6,
    )

  def test_dirichlet_is_exact_for_a_linear_field(self):
    geom = self._build()
    field = self._grid_field()
    surface_values = self._linear_field(self._body_intercepts(geom))

    zeta, gamma = ghost_point.dirichlet_coefficients(geom, surface_values)
    result = ghost_point.apply(jnp.asarray(field), geom, zeta, gamma)

    # The linear field already satisfies the boundary condition, so writing
    # the ghost cells has to leave it as it is.
    np.testing.assert_allclose(result, field, atol=2e-3)

  def test_neumann_is_exact_for_a_linear_field(self):
    geom = self._build()
    field = self._grid_field()
    surface_fluxes = geom.normal @ self._GRADIENT

    zeta, gamma = ghost_point.neumann_coefficients(geom, surface_fluxes)
    result = ghost_point.apply(jnp.asarray(field), geom, zeta, gamma)

    np.testing.assert_allclose(result, field, atol=2e-3)


class CoefficientsTest(parameterized.TestCase):

  def test_dirichlet_mirrors_about_the_surface_value(self):
    geom = ghost_point.build(_wall(2.5), _unit_grid(_WALL_SHAPE), halo_width=1)

    zeta, gamma = ghost_point.dirichlet_coefficients(geom, 3.0)

    np.testing.assert_allclose(zeta, np.full(16, -1.0), atol=1e-6)
    np.testing.assert_allclose(gamma, np.full(16, 6.0), atol=1e-6)

  def test_dirichlet_accepts_one_value_per_ghost_cell(self):
    geom = ghost_point.build(_wall(2.5), _unit_grid(_WALL_SHAPE), halo_width=1)

    _, gamma = ghost_point.dirichlet_coefficients(geom, jnp.arange(16.0))

    np.testing.assert_allclose(gamma, 2.0 * np.arange(16.0), atol=1e-6)

  def test_dirichlet_divides_out_weight_on_the_ghost_cell_itself(self):
    # Half the interpolation weight is on the ghost cell, so the implicit
    # relation is divided by 1.5.
    geom = ghost_point.build(_wall(2.25), _unit_grid(_WALL_SHAPE), halo_width=1)

    zeta, gamma = ghost_point.dirichlet_coefficients(geom, 3.0)

    np.testing.assert_allclose(zeta, np.full(16, -1.0 / 1.5), atol=1e-6)
    np.testing.assert_allclose(gamma, np.full(16, 4.0), atol=1e-6)

  def test_neumann_steps_back_from_image_point_by_flux_times_separation(self):
    geom = ghost_point.build(_wall(2.5), _unit_grid(_WALL_SHAPE), halo_width=1)

    zeta, gamma = ghost_point.neumann_coefficients(geom, 2.0)

    np.testing.assert_allclose(zeta, np.full(16, 1.0), atol=1e-6)
    np.testing.assert_allclose(gamma, np.full(16, -2.0), atol=1e-6)

  def test_neumann_divides_out_weight_on_the_ghost_cell_itself(self):
    # The image point is half a cell from the ghost cell, and half the
    # interpolation weight is on the ghost cell.
    geom = ghost_point.build(_wall(2.25), _unit_grid(_WALL_SHAPE), halo_width=1)

    zeta, gamma = ghost_point.neumann_coefficients(geom, 2.0)

    np.testing.assert_allclose(zeta, np.full(16, 2.0), atol=1e-6)
    np.testing.assert_allclose(gamma, np.full(16, -2.0), atol=1e-6)

  @parameterized.named_parameters(
      dict(
          testcase_name='dirichlet',
          bc_type='dirichlet',
          value=3.0,
          expected_zeta=-1.0,
          expected_gamma=6.0,
      ),
      dict(
          testcase_name='neumann',
          bc_type='neumann',
          value=2.0,
          expected_zeta=1.0,
          expected_gamma=-2.0,
      ),
  )
  def test_coefficients_for_selects_condition_by_name(
      self, bc_type, value, expected_zeta, expected_gamma
  ):
    geom = ghost_point.build(_wall(2.5), _unit_grid(_WALL_SHAPE), halo_width=1)

    zeta, gamma = ghost_point.coefficients_for(geom, bc_type, value)

    np.testing.assert_allclose(zeta, np.full(16, expected_zeta), atol=1e-6)
    np.testing.assert_allclose(gamma, np.full(16, expected_gamma), atol=1e-6)

  def test_coefficients_for_unknown_condition_raises(self):
    geom = ghost_point.build(_wall(2.5), _unit_grid(_WALL_SHAPE), halo_width=1)

    with self.assertRaisesRegex(ValueError, "'dirichlet' or 'neumann'"):
      ghost_point.coefficients_for(geom, 'robin', 1.0)


class ApplyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='image_point_on_node', height=2.5, surface_value=6.0),
      dict(
          testcase_name='stencil_contains_ghost_cell',
          height=2.25,
          surface_value=5.5,
      ),
      # The ghost cell is on the surface, so it takes the surface value.
      dict(
          testcase_name='ghost_cell_on_surface', height=2.0, surface_value=5.0
      ),
  )
  def test_dirichlet_restores_linear_profile_in_ghost_cells(
      self, height, surface_value
  ):
    geom = ghost_point.build(
        _wall(height), _unit_grid(_WALL_SHAPE), halo_width=1
    )
    linear = _varying_along_last_axis(_LINEAR_PROFILE, _WALL_SHAPE)
    u = _with_corrupted_wall_ghost_cells(linear)

    zeta, gamma = ghost_point.dirichlet_coefficients(geom, surface_value)
    result = ghost_point.apply(u, geom, zeta, gamma)

    np.testing.assert_allclose(result, linear, atol=1e-5)

  @parameterized.named_parameters(
      dict(testcase_name='image_point_on_node', height=2.5),
      dict(testcase_name='stencil_contains_ghost_cell', height=2.25),
  )
  def test_neumann_restores_linear_profile_in_ghost_cells(self, height):
    geom = ghost_point.build(
        _wall(height), _unit_grid(_WALL_SHAPE), halo_width=1
    )
    linear = _varying_along_last_axis(_LINEAR_PROFILE, _WALL_SHAPE)
    u = _with_corrupted_wall_ghost_cells(linear)

    # The profile `2 k + 1` rises by 2 per cell away from the wall.
    zeta, gamma = ghost_point.neumann_coefficients(geom, 2.0)
    result = ghost_point.apply(u, geom, zeta, gamma)

    np.testing.assert_allclose(result, linear, atol=1e-5)

  def test_shortened_probe_is_still_exact_for_linear_profile(self):
    geom = ghost_point.build(_wall(2.625), _unit_grid((6, 6, 4)), halo_width=1)
    linear = _varying_along_last_axis([1.0, 3.0, 5.0, 7.0], (6, 6, 4))
    u = _with_corrupted_wall_ghost_cells(linear)

    zeta, gamma = ghost_point.dirichlet_coefficients(geom, 6.25)
    result = ghost_point.apply(u, geom, zeta, gamma)

    np.testing.assert_allclose(result, linear, atol=1e-5)

  def test_dirichlet_fallback_assigns_the_surface_value(self):
    geom = ghost_point.build(_wall(2.875), _unit_grid((6, 6, 4)), halo_width=1)
    linear = _varying_along_last_axis([1.0, 3.0, 5.0, 7.0], (6, 6, 4))
    u = _with_corrupted_wall_ghost_cells(linear)

    zeta, gamma = ghost_point.dirichlet_coefficients(geom, 6.75)
    result = ghost_point.apply(u, geom, zeta, gamma)

    # First order: the ghost cells hold the surface value instead of the 5
    # that would continue the linear profile.
    expected = linear.copy()
    expected[1:5, 1:5, 2] = 6.75
    np.testing.assert_allclose(result, expected, atol=1e-5)

  def test_neumann_fallback_takes_one_sided_gradient_from_nearest_fluid(self):
    geom = ghost_point.build(_wall(2.875), _unit_grid((6, 6, 4)), halo_width=1)
    linear = _varying_along_last_axis([1.0, 3.0, 5.0, 7.0], (6, 6, 4))
    u = _with_corrupted_wall_ghost_cells(linear)

    zeta, gamma = ghost_point.neumann_coefficients(geom, 2.0)
    result = ghost_point.apply(u, geom, zeta, gamma)

    np.testing.assert_allclose(result, linear, atol=1e-5)

  def test_geometry_without_ghost_points_leaves_field_unchanged(self):
    geom = ghost_point.build(
        _wall(-10.0), _unit_grid(_WALL_SHAPE), halo_width=1
    )
    u = jnp.asarray(_varying_along_last_axis(_LINEAR_PROFILE, _WALL_SHAPE))

    result = ghost_point.apply(u, geom, jnp.zeros(0), jnp.zeros(0))

    np.testing.assert_array_equal(result, u)

  def test_field_of_wrong_shape_raises(self):
    geom = ghost_point.build(_wall(2.5), _unit_grid(_WALL_SHAPE), halo_width=1)
    zeta, gamma = ghost_point.dirichlet_coefficients(geom, 0.0)

    with self.assertRaisesRegex(ValueError, 'the geometry was built for'):
      ghost_point.apply(jnp.zeros((6, 6, 7)), geom, zeta, gamma)

  def test_fewer_than_one_sweep_raises(self):
    geom = ghost_point.build(_wall(2.5), _unit_grid(_WALL_SHAPE), halo_width=1)
    zeta, gamma = ghost_point.dirichlet_coefficients(geom, 0.0)

    with self.assertRaisesRegex(ValueError, '`num_sweeps` must be at least 1'):
      ghost_point.apply(jnp.zeros(_WALL_SHAPE), geom, zeta, gamma, num_sweeps=0)


class ApplyLocalTest(parameterized.TestCase):

  def test_padding_entries_leave_their_cell_untouched(self):
    u = jnp.arange(27.0).reshape(3, 3, 3)
    # The second entry is padding: it addresses cell 0 and is marked invalid.
    geom = _single_source_geometry(
        gc_index=[13, 0], source_index=[14, 0], valid=[True, False]
    )

    result = ghost_point.apply_local(
        u,
        geom,
        zeta=jnp.array([-1.0, 5.0]),
        gamma=jnp.array([4.0, 7.0]),
        num_sweeps=1,
    )

    # Cell 13 becomes `-1 * 14 + 4`; cell 0 would have become 7.
    expected = np.arange(27.0)
    expected[13] = -10.0
    np.testing.assert_allclose(result, expected.reshape(3, 3, 3), atol=1e-6)

  @parameterized.named_parameters(
      dict(testcase_name='one_sweep', num_sweeps=1, expected=[0.0, 0.0, 5.0]),
      dict(testcase_name='two_sweeps', num_sweeps=2, expected=[0.0, 5.0, 5.0]),
  )
  def test_each_sweep_reads_the_values_of_the_previous_one(
      self, num_sweeps, expected
  ):
    # Cell 1 copies ghost cell 2, which copies the fluid cell 3.
    u = jnp.array([[[0.0, 0.0, 0.0, 5.0]]])
    geom = _single_source_geometry(gc_index=[1, 2], source_index=[2, 3])

    result = ghost_point.apply_local(
        u, geom, zeta=jnp.ones(2), gamma=jnp.zeros(2), num_sweeps=num_sweeps
    )

    np.testing.assert_allclose(result[0, 0, :3], expected, atol=1e-6)

  def test_exchange_fn_runs_between_sweeps(self):
    u = jnp.array([[[0.0, 0.0, 0.0, 5.0]]])
    geom = _single_source_geometry(gc_index=[2], source_index=[3])
    fields_exchanged = []

    def exchange(field):
      fields_exchanged.append(field)
      return field.at[0, 0, 3].set(9.0)

    result = ghost_point.apply_local(
        u,
        geom,
        zeta=jnp.ones(1),
        gamma=jnp.zeros(1),
        num_sweeps=2,
        exchange_fn=exchange,
    )

    # The first sweep copies 5; the second copies the exchanged 9.
    self.assertLen(fields_exchanged, 1)
    np.testing.assert_allclose(fields_exchanged[0], [[[0.0, 0.0, 5.0, 5.0]]])
    np.testing.assert_allclose(result, [[[0.0, 0.0, 9.0, 9.0]]], atol=1e-6)

  def test_fewer_than_one_sweep_raises(self):
    geom = _single_source_geometry(gc_index=[2], source_index=[3])

    with self.assertRaisesRegex(ValueError, '`num_sweeps` must be at least 1'):
      ghost_point.apply_local(
          jnp.zeros((1, 1, 4)), geom, jnp.ones(1), jnp.zeros(1), num_sweeps=0
      )


class ForcingTest(absltest.TestCase):

  def test_relaxes_ghost_cells_towards_their_target_value(self):
    u = jnp.arange(27.0).reshape(3, 3, 3)
    geom = _single_source_geometry(gc_index=[13], source_index=[14])

    force = ghost_point.forcing(
        u,
        geom,
        zeta=jnp.array([-1.0]),
        gamma=jnp.array([4.0]),
        damping_coeff=2.0,
    )

    # The target for cell 13 is `-1 * 14 + 4 = -10`, so the force is
    # `-2 * (13 - (-10))`. Everywhere else the field is already on target.
    expected = np.zeros(27)
    expected[13] = -46.0
    np.testing.assert_allclose(force, expected.reshape(3, 3, 3), atol=1e-5)


class DeepInteriorMaskTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # With no reach, only the ghost layer `k = 3` is kept out.
      dict(testcase_name='no_reach', reach=0, num_deep_layers=3),
      dict(testcase_name='reach_of_1', reach=1, num_deep_layers=3),
      dict(testcase_name='reach_of_2', reach=2, num_deep_layers=2),
  )
  def test_marks_solid_cells_beyond_the_reach_of_fluid_stencils(
      self, reach, num_deep_layers
  ):
    # Fluid in the layers `k >= 4` of a (2, 2, 6) array, whose ghost cells are
    # the layer `k = 3`, at flat indices `(i * 2 + j) * 6 + 3`.
    interior_mask = np.zeros((2, 2, 6), np.float32)
    interior_mask[:, :, 4:] = 1.0
    geom = _single_source_geometry(
        gc_index=[3, 9, 15, 21], source_index=[4, 10, 16, 22]
    )

    deep = ghost_point.deep_interior_mask(
        jnp.asarray(interior_mask), geom, reach=reach
    )

    expected = np.zeros((2, 2, 6), bool)
    expected[:, :, :num_deep_layers] = True
    np.testing.assert_array_equal(deep, expected)


class IsGeometryKeyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='packed_weights', key='ib_geom_alpha', expected=True),
      dict(
          testcase_name='packed_indices', key='ib_geom_gc_index', expected=True
      ),
      dict(testcase_name='fluid_mask', key='ib_interior_mask', expected=False),
      dict(testcase_name='signed_distance', key='ib_phi', expected=False),
      dict(testcase_name='velocity', key='u', expected=False),
  )
  def test_only_packed_geometry_is_solver_internal(self, key, expected):
    self.assertEqual(ghost_point.is_geometry_key(key), expected)


class FromAdditionalStatesTest(absltest.TestCase):

  def _make_states(self) -> dict[str, jax.Array]:
    """Two ghost points, the second of them padding, as one replica sees it."""
    per_ghost_point = {
        'ib_geom_gc_index': [5.0, 0.0],
        'ib_geom_self_weight': [0.25, 0.0],
        'ib_geom_bi_weight': [0.125, 0.0],
        'ib_geom_zeta_dirichlet': [-1.0, 0.0],
        'ib_geom_separation': [1.5, 1.0],
        'ib_geom_distance': [0.75, 1.0],
        'ib_geom_valid': [1.0, 0.0],
    }
    states = {
        key: jnp.array(value, jnp.float32).reshape(1, 1, 1, 2)
        for key, value in per_ghost_point.items()
    }
    states['ib_geom_corner_index'] = jnp.full((1, 1, 1, 2, 8), 6.0)
    states['ib_geom_alpha'] = jnp.full((1, 1, 1, 2, 8), 0.125)
    return states

  def test_drops_replica_axes_and_restores_index_and_mask_types(self):
    geom = ghost_point.from_additional_states(self._make_states())

    np.testing.assert_array_equal(geom.gc_index, np.array([5, 0], np.int32))
    self.assertEqual(geom.gc_index.dtype, jnp.int32)
    np.testing.assert_array_equal(
        geom.corner_index, np.full((2, 8), 6, np.int32)
    )
    self.assertEqual(geom.corner_index.dtype, jnp.int32)
    np.testing.assert_array_equal(geom.valid, np.array([True, False]))
    self.assertEqual(geom.valid.dtype, jnp.bool_)

  def test_passes_weights_through_unchanged(self):
    geom = ghost_point.from_additional_states(self._make_states())

    np.testing.assert_array_equal(geom.alpha, np.full((2, 8), 0.125))
    np.testing.assert_array_equal(geom.self_weight, [0.25, 0.0])
    np.testing.assert_array_equal(geom.bi_weight, [0.125, 0.0])
    np.testing.assert_array_equal(geom.zeta_dirichlet, [-1.0, 0.0])
    np.testing.assert_array_equal(geom.separation, [1.5, 1.0])
    np.testing.assert_array_equal(geom.distance, [0.75, 1.0])

  def test_missing_geometry_raises_with_the_missing_keys(self):
    states = self._make_states()
    del states['ib_geom_alpha']

    with self.assertRaisesRegex(KeyError, 'ib_geom_alpha'):
      ghost_point.from_additional_states(states)


class LocalCoordinatesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='first_replica',
          replica=(0, 0, 0),
          expected_x=[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
      ),
      # Neighbouring subdomains overlap by twice the halo width.
      dict(
          testcase_name='second_replica',
          replica=(1, 0, 0),
          expected_x=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
      ),
  )
  def test_returns_cell_centres_of_one_subdomain_with_halos(
      self, replica, expected_x
  ):
    grid_params = _make_grid_params(num_cores_x=2)

    x, y, z = ghost_point.local_coordinates(grid_params, replica)

    np.testing.assert_allclose(x, expected_x, atol=1e-6)
    np.testing.assert_allclose(y, [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], atol=1e-6)
    np.testing.assert_allclose(
        z, [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], atol=1e-6
    )


class PartitionSpecsTest(absltest.TestCase):

  def test_geometry_is_partitioned_over_every_mesh_axis(self):
    mesh = _make_mesh((2, 1, 1))

    specs = ghost_point.partition_specs(mesh)

    per_ghost_point = _P('x', 'y', 'z', None)
    per_corner = _P('x', 'y', 'z', None, None)
    self.assertEqual(
        specs,
        ghost_point.ShardedGhostPoints(
            gc_index=per_ghost_point,
            corner_index=per_corner,
            alpha=per_corner,
            self_weight=per_ghost_point,
            bi_weight=per_ghost_point,
            zeta_dirichlet=per_ghost_point,
            separation=per_ghost_point,
            distance=per_ghost_point,
            valid=per_ghost_point,
        ),
    )


class SqueezeLocalTest(absltest.TestCase):

  def test_drops_the_three_replica_axes(self):
    geom = _single_source_geometry(gc_index=[13, 7], source_index=[14, 8])
    as_seen_by_one_device = jax.tree.map(lambda a: a[None, None, None], geom)

    squeezed = ghost_point.squeeze_local(as_seen_by_one_device)

    np.testing.assert_array_equal(squeezed.gc_index, [13, 7])
    self.assertEqual(squeezed.corner_index.shape, (2, 8))
    self.assertEqual(squeezed.alpha.shape, (2, 8))
    self.assertEqual(squeezed.valid.shape, (2,))


class BuildForMeshTest(absltest.TestCase):

  def test_every_replica_gets_the_geometry_of_its_own_subdomain(self):
    # In local index space the wall at z = 1.5 sits between k = 2 and k = 3,
    # exactly as the wall at 2.5 does on the unit grid.
    grid_params = _make_grid_params(num_cores_x=2)
    mesh = _make_mesh((2, 1, 1))

    sharded, per_replica = ghost_point.build_for_mesh(
        _wall(1.5), grid_params, mesh
    )

    self.assertLen(per_replica, 2)
    self.assertEqual(sharded.gc_index.dtype, jnp.int32)
    self.assertEqual(sharded.corner_index.dtype, jnp.int32)
    self.assertEqual(sharded.corner_index.shape, (2, 1, 1, 16, 8))
    self.assertEqual(sharded.alpha.shape, (2, 1, 1, 16, 8))
    np.testing.assert_array_equal(
        sharded.gc_index,
        np.array([_WALL_GHOST_INDICES, _WALL_GHOST_INDICES]).reshape(
            2, 1, 1, 16
        ),
    )
    np.testing.assert_array_equal(sharded.valid, np.ones((2, 1, 1, 16), bool))

  def test_replica_with_fewer_ghost_points_is_padded_with_invalid_entries(self):
    # The sphere lies wholly inside the subdomain of the first replica.
    grid_params = _make_grid_params(num_cores_x=2)
    mesh = _make_mesh((2, 1, 1))
    sphere = signed_distance.sphere((1.0, 1.5, 2.5), 1.2)

    sharded, per_replica = ghost_point.build_for_mesh(sphere, grid_params, mesh)

    num_in_first = per_replica[0].num_ghost_points
    self.assertGreater(num_in_first, 0)
    self.assertEqual(per_replica[1].num_ghost_points, 0)
    self.assertEqual(sharded.gc_index.shape, (2, 1, 1, num_in_first))
    np.testing.assert_array_equal(
        sharded.valid[0, 0, 0], np.ones(num_in_first, bool)
    )
    np.testing.assert_array_equal(
        sharded.valid[1, 0, 0], np.zeros(num_in_first, bool)
    )
    # Padding addresses flat index 0, a halo cell.
    np.testing.assert_array_equal(
        sharded.gc_index[1, 0, 0], np.zeros(num_in_first, np.int32)
    )

  def test_shard_map_applies_the_condition_on_every_replica(self):
    grid_params = _make_grid_params(num_cores_x=2)
    mesh = _make_mesh((2, 1, 1))
    sharded, _ = ghost_point.build_for_mesh(_wall(1.5), grid_params, mesh)
    linear = _varying_along_last_axis(_LINEAR_PROFILE, _WALL_SHAPE)
    corrupted = _with_corrupted_wall_ghost_cells(linear)
    u = jnp.concatenate([corrupted, corrupted], axis=0)

    def apply_surface_value(field, geometry):
      geom = ghost_point.squeeze_local(geometry)
      zeta, gamma = ghost_point.coefficients_for(geom, 'dirichlet', 6.0)
      return ghost_point.apply_local(field, geom, zeta, gamma)

    result = shard_map.shard_map(
        apply_surface_value,
        mesh=mesh,
        in_specs=(_P('x', 'y', 'z'), ghost_point.partition_specs(mesh)),
        out_specs=_P('x', 'y', 'z'),
    )(u, sharded)

    np.testing.assert_allclose(
        result, np.concatenate([linear, linear], axis=0), atol=1e-5
    )


class GhostPointGeometryTest(absltest.TestCase):

  def test_partition_specs_cover_every_packed_array(self):
    geometry = ghost_point.GhostPointGeometry(
        _wall(1.5), _make_grid_params(num_cores_x=2), _make_mesh((2, 1, 1))
    )

    specs = geometry.partition_specs()

    per_ghost_point = _P('x', 'y', 'z', None)
    per_corner = _P('x', 'y', 'z', None, None)
    self.assertEqual(
        specs,
        {
            'ib_geom_gc_index': per_ghost_point,
            'ib_geom_self_weight': per_ghost_point,
            'ib_geom_bi_weight': per_ghost_point,
            'ib_geom_zeta_dirichlet': per_ghost_point,
            'ib_geom_separation': per_ghost_point,
            'ib_geom_distance': per_ghost_point,
            'ib_geom_valid': per_ghost_point,
            'ib_geom_corner_index': per_corner,
            'ib_geom_alpha': per_corner,
        },
    )

  def test_initial_states_hold_one_replica_tile_of_the_packed_geometry(self):
    geometry = ghost_point.GhostPointGeometry(
        _wall(1.5), _make_grid_params(num_cores_x=2), _make_mesh((2, 1, 1))
    )

    states = geometry.initial_states((1, 0, 0))

    # Everything, indices and mask included, is carried as float32.
    np.testing.assert_array_equal(
        states['ib_geom_gc_index'],
        np.array(_WALL_GHOST_INDICES, np.float32).reshape(1, 1, 1, 16),
    )
    np.testing.assert_array_equal(
        states['ib_geom_valid'], np.ones((1, 1, 1, 16), np.float32)
    )
    self.assertEqual(states['ib_geom_corner_index'].shape, (1, 1, 1, 16, 8))
    self.assertEqual(states['ib_geom_alpha'].shape, (1, 1, 1, 16, 8))

  def test_initial_states_include_mask_output_fields_by_default(self):
    geometry = ghost_point.GhostPointGeometry(
        _wall(1.5), _make_grid_params(num_cores_x=2), _make_mesh((2, 1, 1))
    )

    states = geometry.initial_states((0, 0, 0))

    self.assertCountEqual(
        states.keys(), _GEOMETRY_KEYS + ['ib_interior_mask', 'ib_boundary']
    )
    expected_interior = np.zeros(_WALL_SHAPE, np.float32)
    expected_interior[:, :, 3:] = 1.0
    expected_ghost = np.zeros(_WALL_SHAPE, np.float32)
    expected_ghost[1:5, 1:5, 2] = 1.0
    np.testing.assert_array_equal(states['ib_interior_mask'], expected_interior)
    np.testing.assert_array_equal(states['ib_boundary'], expected_ghost)

  def test_output_fields_follow_the_write_flags(self):
    geometry = ghost_point.GhostPointGeometry(
        _wall(1.5),
        _make_grid_params(num_cores_x=2),
        _make_mesh((2, 1, 1)),
        write_interior_mask=False,
        write_ghost_mask=False,
        write_signed_distance=True,
    )

    states = geometry.initial_states((0, 0, 0))

    self.assertCountEqual(states.keys(), _GEOMETRY_KEYS + ['ib_phi'])
    # A local index `k` along z sits at `z = k - 1`.
    np.testing.assert_allclose(
        states['ib_phi'],
        _varying_along_last_axis(
            [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5], _WALL_SHAPE
        ),
        atol=1e-6,
    )

  def test_states_round_trip_into_a_geometry_that_applies_the_condition(self):
    geometry = ghost_point.GhostPointGeometry(
        _wall(1.5), _make_grid_params(num_cores_x=2), _make_mesh((2, 1, 1))
    )
    linear = _varying_along_last_axis(_LINEAR_PROFILE, _WALL_SHAPE)
    u = _with_corrupted_wall_ghost_cells(linear)

    states = {
        key: jnp.asarray(value)
        for key, value in geometry.initial_states((1, 0, 0)).items()
    }
    geom = ghost_point.from_additional_states(states)
    zeta, gamma = ghost_point.coefficients_for(geom, 'dirichlet', 6.0)
    result = ghost_point.apply_local(u, geom, zeta, gamma)

    np.testing.assert_allclose(result, linear, atol=1e-5)

  def test_counts_ghost_points_over_all_replicas(self):
    geometry = ghost_point.GhostPointGeometry(
        _wall(1.5), _make_grid_params(num_cores_x=2), _make_mesh((2, 1, 1))
    )

    self.assertEqual(geometry.num_ghost_points, 32)
    self.assertIn('ghost points: 32 over 2 replica(s)', geometry.summary())

  def test_logical_coordinates_are_mapped_to_data_axis_order(self):
    # The array axes are (z, x, y), and the sphere lies wholly inside the
    # subdomain of the second replica along x.
    geometry = ghost_point.GhostPointGeometry(
        signed_distance.sphere((6.0, 1.5, 2.5), 1.2),
        _make_grid_params(num_cores_x=2, data_axis_order='zxy'),
        _make_mesh((1, 2, 1), axis_names=('z', 'x', 'y')),
    )

    first = geometry.initial_states((0, 0, 0))
    second = geometry.initial_states((1, 0, 0))

    self.assertEqual(second['ib_boundary'].shape, (8, 6, 6))
    np.testing.assert_array_equal(
        first['ib_boundary'], np.zeros((8, 6, 6), np.float32)
    )
    self.assertGreater(second['ib_boundary'].sum(), 0.0)
    # The cell at local index (3, 3, 2) is at z = 2 and y = 1 on both replicas,
    # and at x = 2 on the first but x = 6, inside the sphere, on the second.
    self.assertEqual(first['ib_interior_mask'][3, 3, 2], 1.0)
    self.assertEqual(second['ib_interior_mask'][3, 3, 2], 0.0)


class GeometryFromProtoTest(parameterized.TestCase):

  def test_sphere(self):
    phi = ghost_point.geometry_from_proto(
        [_parse_geometry('sphere { center { x: 1 y: 2 z: 3 } radius: 2 }')]
    )

    np.testing.assert_allclose(phi(jnp.array([1.0, 2.0, 6.0])), 1.0, atol=1e-6)

  @parameterized.named_parameters(
      dict(testcase_name='unrotated', rotation='', expected=1.0),
      dict(
          testcase_name='rotated',
          rotation='rotation_deg { z: 90 }',
          expected=2.0,
      ),
  )
  def test_box_is_rotated_only_when_a_rotation_is_given(
      self, rotation, expected
  ):
    box = _parse_geometry(
        f'box {{ half_extents {{ x: 2 y: 1 z: 1 }} {rotation} }}'
    )

    phi = ghost_point.geometry_from_proto([box])

    np.testing.assert_allclose(
        phi(jnp.array([3.0, 0.0, 0.0])), expected, atol=1e-5
    )

  @parameterized.named_parameters(
      # Without a height the cylinder is infinite along its axis.
      dict(testcase_name='infinite', height='', expected=-1.0),
      dict(testcase_name='finite', height='height: 2', expected=1.0),
  )
  def test_cylinder_is_capped_only_when_a_height_is_given(
      self, height, expected
  ):
    cylinder = _parse_geometry(
        f'cylinder {{ axis {{ z: 1 }} radius: 1 {height} }}'
    )

    phi = ghost_point.geometry_from_proto([cylinder])

    np.testing.assert_allclose(
        phi(jnp.array([0.0, 0.0, 3.0])), expected, atol=1e-6
    )

  def test_height_field_uses_the_named_profile(self):
    phi = ghost_point.geometry_from_proto(
        [_parse_geometry('height_field { profile: "ramp" }')],
        height_fields={'ramp': lambda x, y: 0.5 * x},
    )

    np.testing.assert_allclose(phi(jnp.array([2.0, 0.0, 3.0])), 2.0, atol=1e-6)

  @parameterized.named_parameters(
      dict(testcase_name='near_sphere', point=(3.0, 0.0, 0.0), expected=2.0),
      dict(
          testcase_name='near_second_sphere',
          point=(0.0, 0.0, 13.0),
          expected=1.0,
      ),
      dict(testcase_name='near_box', point=(8.0, 0.0, 0.0), expected=1.0),
      dict(testcase_name='near_cylinder', point=(0.0, 7.0, 50.0), expected=2.0),
      dict(testcase_name='near_terrain', point=(4.0, 4.0, -18.0), expected=2.0),
  )
  def test_bodies_of_every_type_are_combined_by_union(self, point, expected):
    phi = ghost_point.geometry_from_proto(
        [
            _parse_geometry('sphere { radius: 1 }'),
            _parse_geometry('sphere { center { z: 10 } radius: 2 }'),
            _parse_geometry(
                'box { center { x: 10 } half_extents { x: 1 y: 1 z: 1 } }'
            ),
            _parse_geometry(
                'cylinder { base { y: 10 } axis { z: 1 } radius: 1 }'
            ),
            _parse_geometry('height_field { profile: "floor" }'),
        ],
        height_fields={'floor': lambda x, y: -20.0},
    )

    np.testing.assert_allclose(phi(jnp.array(point)), expected, atol=1e-5)

  def test_no_geometry_raises(self):
    with self.assertRaisesRegex(ValueError, 'at least one `geometry` entry'):
      ghost_point.geometry_from_proto([])

  def test_geometry_without_a_shape_raises(self):
    with self.assertRaisesRegex(ValueError, 'must set one of sphere, box'):
      ghost_point.geometry_from_proto(
          [immersed_boundary_method_pb2.IBGeometry()]
      )

  def test_height_field_profile_that_was_not_supplied_raises(self):
    with self.assertRaisesRegex(ValueError, "'ramp' was not supplied"):
      ghost_point.geometry_from_proto(
          [_parse_geometry('height_field { profile: "ramp" }')],
          height_fields={'hill': lambda x, y: 0.0},
      )


class RequiredKeysTest(absltest.TestCase):

  def test_no_immersed_boundary_needs_no_keys(self):
    params = _make_params()

    self.assertEmpty(ghost_point.required_keys(params))

  def test_other_immersed_boundary_method_needs_no_keys(self):
    params = _make_params("""
        boundary_models {
          ib {
            direct_forcing {
              damping_coeff: 0.1
              variables { name: "u" value: 0.0 }
            }
          }
        }
    """)

    self.assertEmpty(ghost_point.required_keys(params))

  def test_ghost_point_needs_packed_geometry_and_default_output_fields(self):
    params = _make_params("""
        boundary_models {
          ib { ghost_point { variables { name: "u" } } }
        }
    """)

    self.assertEqual(
        ghost_point.required_keys(params),
        _GEOMETRY_KEYS + ['ib_interior_mask', 'ib_boundary'],
    )

  def test_output_fields_follow_the_write_flags(self):
    params = _make_params("""
        boundary_models {
          ib {
            ghost_point {
              variables { name: "u" }
              write_interior_mask: false
              write_ghost_mask: false
              write_signed_distance: true
            }
          }
        }
    """)

    self.assertEqual(
        ghost_point.required_keys(params), _GEOMETRY_KEYS + ['ib_phi']
    )


if __name__ == '__main__':
  absltest.main()
