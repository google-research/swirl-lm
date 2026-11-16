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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.utility import signed_distance

_SQRT2 = float(np.sqrt(2.0))
_SQRT3 = float(np.sqrt(3.0))


class SphereTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='in_fluid', point=(3.0, 0.0, 0.0), expected=2.0),
      dict(testcase_name='on_surface', point=(0.0, 1.0, 0.0), expected=0.0),
      dict(testcase_name='in_solid', point=(0.0, 0.0, 0.5), expected=-0.5),
      dict(testcase_name='at_center', point=(0.0, 0.0, 0.0), expected=-1.0),
  )
  def test_unit_sphere_distance(self, point, expected):
    phi = signed_distance.sphere(center=(0.0, 0.0, 0.0), radius=1.0)

    distance = phi(jnp.array(point))

    np.testing.assert_allclose(distance, expected, atol=1e-6)

  def test_distance_is_measured_from_center(self):
    phi = signed_distance.sphere(center=(1.0, 2.0, 3.0), radius=2.0)

    distance = phi(jnp.array([1.0, 2.0, 6.0]))

    np.testing.assert_allclose(distance, 1.0, atol=1e-6)

  def test_distance_is_unchanged_under_jit(self):
    phi = signed_distance.sphere(center=(0.0, 0.0, 0.0), radius=1.0)

    distance = jax.jit(phi)(jnp.array([3.0, 0.0, 0.0]))

    np.testing.assert_allclose(distance, 2.0, atol=1e-6)

  @parameterized.parameters(0.0, -1.0)
  def test_non_positive_radius_raises(self, radius):
    with self.assertRaisesRegex(ValueError, '`radius` must be positive'):
      signed_distance.sphere(center=(0.0, 0.0, 0.0), radius=radius)

  def test_center_that_is_not_a_3_vector_raises(self):
    with self.assertRaisesRegex(ValueError, 'Expected a 3-vector'):
      signed_distance.sphere(center=(0.0, 0.0), radius=1.0)


class BoxTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='beyond_face', point=(3.0, 0.0, 0.0), expected=2.0),
      dict(testcase_name='beyond_edge', point=(2.0, 3.0, 0.0), expected=_SQRT2),
      dict(
          testcase_name='beyond_corner',
          point=(2.0, 3.0, 4.0),
          expected=_SQRT3,
      ),
      dict(testcase_name='on_face', point=(1.0, 0.0, 0.0), expected=0.0),
      dict(testcase_name='at_center', point=(0.0, 0.0, 0.0), expected=-1.0),
      dict(
          testcase_name='inside_nearest_to_y_face',
          point=(0.0, 1.5, 0.0),
          expected=-0.5,
      ),
  )
  def test_axis_aligned_box_distance(self, point, expected):
    phi = signed_distance.box(
        center=(0.0, 0.0, 0.0), half_extents=(1.0, 2.0, 3.0)
    )

    distance = phi(jnp.array(point))

    np.testing.assert_allclose(distance, expected, atol=1e-6)

  @parameterized.named_parameters(
      dict(
          testcase_name='along_rotated_long_side',
          point=(0.0, 3.0, 0.0),
          expected=1.0,
      ),
      dict(
          testcase_name='along_rotated_short_side',
          point=(3.0, 0.0, 0.0),
          expected=2.0,
      ),
  )
  def test_rotation_about_z_turns_long_side_onto_y(self, point, expected):
    phi = signed_distance.box(
        center=(0.0, 0.0, 0.0),
        half_extents=(2.0, 1.0, 1.0),
        rotation_deg=(0.0, 0.0, 90.0),
    )

    distance = phi(jnp.array(point))

    np.testing.assert_allclose(distance, expected, atol=1e-5)

  def test_non_positive_half_extent_raises(self):
    with self.assertRaisesRegex(ValueError, '`half_extents` must all be'):
      signed_distance.box(center=(0.0, 0.0, 0.0), half_extents=(1.0, 0.0, 1.0))


class CylinderTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='in_fluid', point=(3.0, 0.0, 5.0), expected=2.0),
      dict(testcase_name='on_axis', point=(0.0, 0.0, 7.0), expected=-1.0),
      dict(
          testcase_name='on_axis_below_base',
          point=(0.0, 0.0, -50.0),
          expected=-1.0,
      ),
  )
  def test_infinite_cylinder_ignores_position_along_axis(self, point, expected):
    phi = signed_distance.cylinder(
        base=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=1.0
    )

    distance = phi(jnp.array(point))

    np.testing.assert_allclose(distance, expected, atol=1e-6)

  @parameterized.named_parameters(
      dict(testcase_name='above_top_cap', point=(0.0, 0.0, 3.0), expected=1.0),
      dict(
          testcase_name='below_base_cap', point=(0.0, 0.0, -1.0), expected=1.0
      ),
      dict(testcase_name='beside_wall', point=(3.0, 0.0, 1.0), expected=2.0),
      dict(testcase_name='beyond_rim', point=(2.0, 0.0, 3.0), expected=_SQRT2),
      dict(testcase_name='at_middle', point=(0.0, 0.0, 1.0), expected=-1.0),
      dict(
          testcase_name='inside_nearest_to_cap',
          point=(0.0, 0.0, 1.75),
          expected=-0.25,
      ),
  )
  def test_finite_cylinder_distance(self, point, expected):
    phi = signed_distance.cylinder(
        base=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=1.0, height=2.0
    )

    distance = phi(jnp.array(point))

    np.testing.assert_allclose(distance, expected, atol=1e-6)

  @parameterized.named_parameters(
      dict(
          testcase_name='off_axis',
          point=(1.0, -1.0, 0.0),
          expected=_SQRT2 - 1.0,
      ),
      dict(testcase_name='on_axis', point=(5.0, 5.0, 0.0), expected=-1.0),
  )
  def test_axis_need_not_be_normalized_or_grid_aligned(self, point, expected):
    phi = signed_distance.cylinder(
        base=(0.0, 0.0, 0.0), axis=(3.0, 3.0, 0.0), radius=1.0
    )

    distance = phi(jnp.array(point))

    np.testing.assert_allclose(distance, expected, atol=1e-5)

  def test_non_positive_radius_raises(self):
    with self.assertRaisesRegex(ValueError, '`radius` must be positive'):
      signed_distance.cylinder(
          base=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=0.0
      )

  def test_non_positive_height_raises(self):
    with self.assertRaisesRegex(ValueError, '`height` must be positive'):
      signed_distance.cylinder(
          base=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=1.0, height=0.0
      )

  def test_zero_axis_raises(self):
    with self.assertRaisesRegex(ValueError, '`axis` must have non-zero'):
      signed_distance.cylinder(
          base=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 0.0), radius=1.0
      )


class PlaneTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='fluid_side', point=(5.0, 5.0, 3.0), expected=2.0),
      dict(testcase_name='on_plane', point=(-4.0, 9.0, 1.0), expected=0.0),
      dict(testcase_name='solid_side', point=(0.0, 0.0, 0.0), expected=-1.0),
  )
  def test_fluid_is_on_the_side_the_normal_points_to(self, point, expected):
    # The normal is deliberately not a unit vector.
    phi = signed_distance.plane(
        point=(0.0, 0.0, 1.0), normal_vec=(0.0, 0.0, 2.0)
    )

    distance = phi(jnp.array(point))

    np.testing.assert_allclose(distance, expected, atol=1e-6)

  def test_zero_normal_raises(self):
    with self.assertRaisesRegex(ValueError, '`normal_vec` must have non-zero'):
      signed_distance.plane(point=(0.0, 0.0, 0.0), normal_vec=(0.0, 0.0, 0.0))


class HeightFieldTest(absltest.TestCase):

  def test_returns_vertical_gap_above_terrain(self):
    phi = signed_distance.height_field(lambda x, y: 0.5 * x + y)

    gap = phi(jnp.array([2.0, 1.0, 5.0]))

    np.testing.assert_allclose(gap, 3.0, atol=1e-6)


class CompositionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='inside_first_only',
          point=(0.0, 0.0, 0.0),
          expected=-2.0,
      ),
      dict(testcase_name='inside_both', point=(1.5, 0.0, 0.0), expected=-0.5),
      dict(testcase_name='outside_both', point=(6.0, 0.0, 0.0), expected=1.0),
  )
  def test_union_merges_solids(self, point, expected):
    phi = signed_distance.union(
        signed_distance.sphere((0.0, 0.0, 0.0), 2.0),
        signed_distance.sphere((3.0, 0.0, 0.0), 2.0),
    )

    distance = phi(jnp.array(point))

    np.testing.assert_allclose(distance, expected, atol=1e-6)

  @parameterized.named_parameters(
      dict(
          testcase_name='inside_first_only', point=(0.0, 0.0, 0.0), expected=1.0
      ),
      dict(testcase_name='inside_both', point=(1.5, 0.0, 0.0), expected=-0.5),
      dict(testcase_name='outside_both', point=(6.0, 0.0, 0.0), expected=4.0),
  )
  def test_intersection_keeps_only_the_common_solid(self, point, expected):
    phi = signed_distance.intersection(
        signed_distance.sphere((0.0, 0.0, 0.0), 2.0),
        signed_distance.sphere((3.0, 0.0, 0.0), 2.0),
    )

    distance = phi(jnp.array(point))

    np.testing.assert_allclose(distance, expected, atol=1e-6)

  @parameterized.named_parameters(
      dict(
          testcase_name='in_carved_hollow', point=(0.0, 0.0, 0.0), expected=1.0
      ),
      dict(
          testcase_name='in_remaining_shell',
          point=(1.5, 0.0, 0.0),
          expected=-0.5,
      ),
      dict(testcase_name='outside', point=(3.0, 0.0, 0.0), expected=1.0),
  )
  def test_difference_turns_removed_solid_into_fluid(self, point, expected):
    phi = signed_distance.difference(
        signed_distance.sphere((0.0, 0.0, 0.0), 2.0),
        removed=signed_distance.sphere((0.0, 0.0, 0.0), 1.0),
    )

    distance = phi(jnp.array(point))

    np.testing.assert_allclose(distance, expected, atol=1e-6)

  def test_union_without_shapes_raises(self):
    with self.assertRaisesRegex(ValueError, '`union` needs at least one'):
      signed_distance.union()

  def test_intersection_without_shapes_raises(self):
    with self.assertRaisesRegex(
        ValueError, '`intersection` needs at least one'
    ):
      signed_distance.intersection()

  def test_translate_moves_shape_by_offset(self):
    phi = signed_distance.translate(
        signed_distance.sphere((0.0, 0.0, 0.0), 1.0), offset=(2.0, 0.0, 0.0)
    )

    distance = phi(jnp.array([4.0, 0.0, 0.0]))

    np.testing.assert_allclose(distance, 1.0, atol=1e-6)

  def test_rotate_about_origin_carries_shape_around_it(self):
    # A box on the +x axis, turned a quarter turn about z, ends up on +y.
    box = signed_distance.box(
        center=(5.0, 0.0, 0.0), half_extents=(2.0, 1.0, 1.0)
    )
    phi = signed_distance.rotate(box, rotation_deg=(0.0, 0.0, 90.0))

    distance = phi(jnp.array([0.0, 8.0, 0.0]))

    np.testing.assert_allclose(distance, 1.0, atol=1e-5)

  def test_rotate_about_center_turns_shape_in_place(self):
    box = signed_distance.box(
        center=(5.0, 0.0, 0.0), half_extents=(2.0, 1.0, 1.0)
    )
    phi = signed_distance.rotate(
        box, rotation_deg=(0.0, 0.0, 90.0), center=(5.0, 0.0, 0.0)
    )

    distance = phi(jnp.array([5.0, 3.0, 0.0]))

    np.testing.assert_allclose(distance, 1.0, atol=1e-5)


class EvaluateTest(absltest.TestCase):

  def test_evaluates_every_point_and_drops_the_coordinate_axis(self):
    phi = signed_distance.sphere((0.0, 0.0, 0.0), 1.0)
    points = jnp.array([
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        [[0.0, 0.0, 4.0], [0.0, -5.0, 0.0], [-6.0, 0.0, 0.0]],
    ])

    values = signed_distance.evaluate(phi, points)

    np.testing.assert_allclose(
        values, [[-1.0, 1.0, 2.0], [3.0, 4.0, 5.0]], atol=1e-6
    )

  def test_blocked_evaluation_covers_a_final_short_block(self):
    phi = signed_distance.sphere((0.0, 0.0, 0.0), 1.0)
    points = jnp.stack(
        [jnp.arange(10.0), jnp.zeros(10), jnp.zeros(10)], axis=-1
    )

    # Ten points in blocks of four leaves a last block of two.
    values = signed_distance.evaluate(phi, points, block=4)

    np.testing.assert_allclose(
        values,
        [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        atol=1e-6,
    )


class NormalTest(absltest.TestCase):

  def test_sphere_normal_points_radially_into_fluid(self):
    phi = signed_distance.sphere((0.0, 0.0, 0.0), 1.0)
    points = jnp.array([[2.0, 0.0, 0.0], [0.0, -3.0, 0.0], [0.0, 0.0, 0.5]])

    normals = signed_distance.normal(phi, points)

    np.testing.assert_allclose(
        normals,
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-6,
    )

  def test_inexact_level_set_still_gives_unit_normal(self):
    # `z - x` has a gradient of length sqrt(2), so it has to be normalized.
    phi = signed_distance.height_field(lambda x, y: x)

    normals = signed_distance.normal(phi, jnp.array([[0.0, 0.0, 1.0]]))

    np.testing.assert_allclose(
        normals, [[-1.0 / _SQRT2, 0.0, 1.0 / _SQRT2]], atol=1e-6
    )

  def test_preserves_leading_shape_of_points(self):
    phi = signed_distance.plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    points = jnp.zeros((2, 4, 3))

    normals = signed_distance.normal(phi, points)

    np.testing.assert_allclose(
        normals, np.broadcast_to([0.0, 0.0, 1.0], (2, 4, 3)), atol=1e-6
    )

  def test_box_normal_inside_points_to_nearest_face(self):
    # Inside a box the outside branch is the norm of a zero vector, whose naive
    # gradient is NaN.
    phi = signed_distance.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    normals = signed_distance.normal(phi, jnp.array([[0.5, 0.0, 0.0]]))

    np.testing.assert_allclose(normals, [[1.0, 0.0, 0.0]], atol=1e-6)

  def test_finite_cylinder_normal_on_axis_points_to_nearest_cap(self):
    # On the axis the radial offset is a zero vector, whose naive gradient is
    # NaN.
    phi = signed_distance.cylinder(
        base=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=1.0, height=2.0
    )

    normals = signed_distance.normal(phi, jnp.array([[0.0, 0.0, 0.25]]))

    np.testing.assert_allclose(normals, [[0.0, 0.0, -1.0]], atol=1e-6)

  def test_sphere_center_gives_zero_vector_rather_than_nan(self):
    phi = signed_distance.sphere((0.0, 0.0, 0.0), 1.0)

    normals = signed_distance.normal(phi, jnp.array([[0.0, 0.0, 0.0]]))

    np.testing.assert_array_equal(normals, [[0.0, 0.0, 0.0]])


class ProjectToSurfaceTest(absltest.TestCase):

  def test_exact_distance_function_projects_in_one_step(self):
    phi = signed_distance.sphere((0.0, 0.0, 0.0), 1.0)
    points = jnp.array([[3.0, 0.0, 0.0], [0.0, 0.5, 0.0]])

    projected = signed_distance.project_to_surface(phi, points, iterations=1)

    np.testing.assert_allclose(
        projected, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], atol=1e-6
    )

  def test_sloped_plane_level_set_projects_along_the_normal(self):
    # Not a distance function: the vertical gap overestimates the distance.
    phi = signed_distance.height_field(lambda x, y: 0.5 * x)

    projected = signed_distance.project_to_surface(
        phi, jnp.array([[0.0, 0.0, 1.0]])
    )

    np.testing.assert_allclose(projected, [[0.4, 0.0, 0.2]], atol=1e-6)

  def test_single_step_lands_short_on_curved_level_set(self):
    phi = signed_distance.height_field(lambda x, y: x**2)

    projected = signed_distance.project_to_surface(
        phi, jnp.array([[1.0, 0.0, 3.0]]), iterations=1
    )

    np.testing.assert_allclose(projected, [[1.8, 0.0, 2.6]], atol=1e-5)

  def test_default_iterations_reach_curved_level_set(self):
    phi = signed_distance.height_field(lambda x, y: x**2)

    projected = signed_distance.project_to_surface(
        phi, jnp.array([[1.0, 0.0, 3.0]])
    )

    np.testing.assert_allclose(phi(projected[0]), 0.0, atol=1e-3)

  def test_preserves_leading_shape_of_points(self):
    phi = signed_distance.plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    points = jnp.ones((2, 4, 3))

    projected = signed_distance.project_to_surface(phi, points)

    np.testing.assert_allclose(
        projected, np.broadcast_to([1.0, 1.0, 0.0], (2, 4, 3)), atol=1e-6
    )

  def test_fewer_than_one_iteration_raises(self):
    phi = signed_distance.sphere((0.0, 0.0, 0.0), 1.0)

    with self.assertRaisesRegex(ValueError, '`iterations` must be at least 1'):
      signed_distance.project_to_surface(phi, jnp.zeros((1, 3)), iterations=0)


class GridPointsTest(absltest.TestCase):

  def test_builds_tensor_product_of_axis_coordinates(self):
    points = signed_distance.grid_points(
        x=np.array([0.0, 1.0]),
        y=np.array([10.0, 20.0, 30.0]),
        z=np.array([5.0]),
    )

    np.testing.assert_array_equal(
        points,
        [
            [[[0.0, 10.0, 5.0]], [[0.0, 20.0, 5.0]], [[0.0, 30.0, 5.0]]],
            [[[1.0, 10.0, 5.0]], [[1.0, 20.0, 5.0]], [[1.0, 30.0, 5.0]]],
        ],
    )


class BatchedSpheresTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='nearest_first', point=(4.0, 0.0, 0.0), expected=3.0),
      dict(testcase_name='inside_second', point=(9.0, 0.0, 0.0), expected=-1.0),
      dict(
          testcase_name='nearest_second', point=(13.0, 0.0, 0.0), expected=1.0
      ),
  )
  def test_distance_to_nearest_sphere(self, point, expected):
    phi = signed_distance.batched_spheres(
        centers=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], radii=[1.0, 2.0]
    )

    distance = phi(jnp.array(point))

    np.testing.assert_allclose(distance, expected, atol=1e-6)

  def test_centers_that_are_not_n_by_3_raise(self):
    with self.assertRaisesRegex(ValueError, r'`centers` must have shape'):
      signed_distance.batched_spheres(centers=[0.0, 0.0, 0.0], radii=[1.0])

  def test_empty_centers_raise(self):
    with self.assertRaisesRegex(ValueError, '`centers` is empty'):
      signed_distance.batched_spheres(centers=np.zeros((0, 3)), radii=[])

  def test_radii_count_mismatch_raises(self):
    with self.assertRaisesRegex(ValueError, '`radii` must have shape'):
      signed_distance.batched_spheres(
          centers=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], radii=[1.0]
      )

  def test_non_positive_radius_raises(self):
    with self.assertRaisesRegex(ValueError, '`radii` must be positive'):
      signed_distance.batched_spheres(
          centers=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], radii=[1.0, 0.0]
      )


class BatchedBoxesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='between', point=(5.0, 0.0, 0.0), expected=3.0),
      dict(testcase_name='inside_first', point=(0.5, 0.0, 0.0), expected=-0.5),
      dict(testcase_name='beside_second', point=(10.0, 4.0, 0.0), expected=3.0),
  )
  def test_distance_to_nearest_axis_aligned_box(self, point, expected):
    phi = signed_distance.batched_boxes(
        centers=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        half_extents=[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
    )

    distance = phi(jnp.array(point))

    np.testing.assert_allclose(distance, expected, atol=1e-6)

  @parameterized.named_parameters(
      dict(testcase_name='between', point=(5.0, 0.0, 0.0), expected=4.0),
      dict(testcase_name='beside_second', point=(10.0, 4.0, 0.0), expected=2.0),
  )
  def test_each_box_is_rotated_by_its_own_angles(self, point, expected):
    # Only the second box is turned, which puts its long side along y.
    phi = signed_distance.batched_boxes(
        centers=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        half_extents=[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
        rotations_deg=[[0.0, 0.0, 0.0], [0.0, 0.0, 90.0]],
    )

    distance = phi(jnp.array(point))

    np.testing.assert_allclose(distance, expected, atol=1e-5)

  def test_half_extents_count_mismatch_raises(self):
    with self.assertRaisesRegex(ValueError, 'must match `centers`'):
      signed_distance.batched_boxes(
          centers=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
          half_extents=[[1.0, 1.0, 1.0]],
      )

  def test_non_positive_half_extent_raises(self):
    with self.assertRaisesRegex(ValueError, '`half_extents` must be positive'):
      signed_distance.batched_boxes(
          centers=[[0.0, 0.0, 0.0]], half_extents=[[1.0, -1.0, 1.0]]
      )

  def test_rotations_count_mismatch_raises(self):
    with self.assertRaisesRegex(ValueError, 'one entry per box'):
      signed_distance.batched_boxes(
          centers=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
          half_extents=[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
          rotations_deg=[[0.0, 0.0, 90.0]],
      )


class BatchedCylindersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='above_cap_of_finite',
          point=(0.0, 0.0, 5.0),
          expected=3.0,
      ),
      dict(
          testcase_name='far_along_axis_of_infinite',
          point=(10.0, 0.0, 100.0),
          expected=-1.0,
      ),
      dict(
          testcase_name='beside_infinite', point=(13.0, 0.0, 50.0), expected=2.0
      ),
  )
  def test_infinite_height_marks_individual_cylinder_as_infinite(
      self, point, expected
  ):
    phi = signed_distance.batched_cylinders(
        bases=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        axes=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        radii=[1.0, 1.0],
        heights=[2.0, np.inf],
    )

    distance = phi(jnp.array(point))

    np.testing.assert_allclose(distance, expected, atol=1e-5)

  @parameterized.named_parameters(
      dict(testcase_name='omitted', heights=None),
      dict(testcase_name='non_positive', heights=[-1.0, 0.0]),
  )
  def test_cylinders_without_usable_height_are_infinite(self, heights):
    phi = signed_distance.batched_cylinders(
        bases=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        axes=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        radii=[1.0, 1.0],
        heights=heights,
    )

    distance = phi(jnp.array([0.0, 0.0, 5.0]))

    np.testing.assert_allclose(distance, -1.0, atol=1e-6)

  def test_axes_count_mismatch_raises(self):
    with self.assertRaisesRegex(ValueError, 'one entry per cylinder'):
      signed_distance.batched_cylinders(
          bases=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
          axes=[[0.0, 0.0, 1.0]],
          radii=[1.0, 1.0],
      )

  def test_zero_axis_raises(self):
    with self.assertRaisesRegex(ValueError, 'must have non-zero length'):
      signed_distance.batched_cylinders(
          bases=[[0.0, 0.0, 0.0]], axes=[[0.0, 0.0, 0.0]], radii=[1.0]
      )

  def test_non_positive_radius_raises(self):
    with self.assertRaisesRegex(ValueError, '`radii` must be positive'):
      signed_distance.batched_cylinders(
          bases=[[0.0, 0.0, 0.0]], axes=[[0.0, 0.0, 1.0]], radii=[0.0]
      )


if __name__ == '__main__':
  absltest.main()
