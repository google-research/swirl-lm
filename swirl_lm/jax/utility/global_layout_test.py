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
import numpy as np
from swirl_lm.jax.utility import global_layout
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import grid_parametrization_pb2

_GridParametrization = grid_parametrization.GridParametrization

# The tests share one layout: two cores along the first array axis and one
# along the others, with a halo width of 1. Per core the array is (6, 5, 4), of
# which (4, 3, 2) is interior, so that
#
#   - the global sharded array is (12, 5, 4);
#   - the unique interior points are (8, 3, 2);
#   - the unique points with the outer halos kept are (10, 5, 4).
#
# Along the first axis the sharded layout is
#
#   index:  0  1  2  3  4  5 |  6  7  8  9 10 11
#   role:   h  i  i  i  i  h |  h  i  i  i  i  h
#
# where the halos at 5 and 6 duplicate the interior cells at 7 and 4.
_SHARDED_SHAPE = (12, 5, 4)
_INTERIOR_SHAPE = (8, 3, 2)
_WITH_HALOS_SHAPE = (10, 5, 4)


def _make_grid_params(periodic_x: bool = False) -> _GridParametrization:
  """Builds the layout above with `x` as the two-core axis."""
  proto = grid_parametrization_pb2.GridParametrization(halo_width=1)
  proto.computation_shape.dim_x = 2
  proto.computation_shape.dim_y = 1
  proto.computation_shape.dim_z = 1
  proto.grid_size.dim_x = 6
  proto.grid_size.dim_y = 5
  proto.grid_size.dim_z = 4
  proto.length.dim_x = 1.0
  proto.length.dim_y = 1.0
  proto.length.dim_z = 1.0
  proto.periodic.dim_x = periodic_x
  return _GridParametrization(proto)


def _varying_along_first_axis(profile, shape) -> np.ndarray:
  """An array of `shape` holding `profile` along axis 0 at every (j, k)."""
  return np.broadcast_to(
      np.asarray(profile, np.float32)[:, None, None], shape
  ).copy()


class StrippedAxesForTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='sharded', shape=(12, 5, 4), expected=(0, 1, 2)),
      dict(testcase_name='already_stripped', shape=(8, 3, 2), expected=()),
      dict(testcase_name='thin_middle_axis', shape=(12, 1, 4), expected=(0, 2)),
      dict(testcase_name='not_rank_3', shape=(12, 5), expected=()),
  )
  def test_selects_axes_whose_extent_is_the_sharded_layout(
      self, shape, expected
  ):
    grid_params = _make_grid_params()

    axes = global_layout.stripped_axes_for(np.zeros(shape), grid_params)

    self.assertEqual(axes, expected)


class StripHalosGlobalTest(absltest.TestCase):

  def test_keeps_only_the_interior_of_every_core(self):
    grid_params = _make_grid_params()
    sharded = np.arange(240).reshape(_SHARDED_SHAPE)

    stripped = global_layout.strip_halos_global(sharded, grid_params)

    np.testing.assert_array_equal(
        stripped, sharded[np.ix_([1, 2, 3, 4, 7, 8, 9, 10], [1, 2, 3], [1, 2])]
    )

  def test_include_halos_keeps_outer_halos_but_drops_seam_duplicates(self):
    grid_params = _make_grid_params()
    sharded = np.arange(240).reshape(_SHARDED_SHAPE)

    stripped = global_layout.strip_halos_global(
        sharded, grid_params, include_halos=True
    )

    np.testing.assert_array_equal(
        stripped,
        sharded[
            np.ix_(
                [0, 1, 2, 3, 4, 7, 8, 9, 10, 11], [0, 1, 2, 3, 4], [0, 1, 2, 3]
            )
        ],
    )

  def test_passes_through_axis_unrelated_to_the_grid(self):
    grid_params = _make_grid_params()
    thin = np.arange(48).reshape(12, 1, 4)

    stripped = global_layout.strip_halos_global(thin, grid_params)

    np.testing.assert_array_equal(
        stripped, thin[np.ix_([1, 2, 3, 4, 7, 8, 9, 10], [0], [1, 2])]
    )

  def test_already_stripped_array_is_returned_unchanged(self):
    grid_params = _make_grid_params()
    stripped = np.arange(48).reshape(_INTERIOR_SHAPE)

    stripped_again = global_layout.strip_halos_global(stripped, grid_params)

    np.testing.assert_array_equal(stripped_again, stripped)

  def test_preserves_dtype(self):
    grid_params = _make_grid_params()
    sharded = np.zeros(_SHARDED_SHAPE, np.int16)

    stripped = global_layout.strip_halos_global(sharded, grid_params)

    self.assertEqual(stripped.dtype, np.int16)

  def test_array_axes_follow_data_axis_order(self):
    # With 'zxy' the first array axis is z, so the two cores along z are what
    # produce the 12-long axis.
    grid_params = _GridParametrization.create_from_grid_lengths_and_etc(
        grid_lengths={'x': 1.0, 'y': 1.0, 'z': 1.0},
        computation_shape={'x': 1, 'y': 1, 'z': 2},
        subgrid_shape={'x': 5, 'y': 4, 'z': 6},
        halo_width=1,
        data_axis_order='zxy',
    )
    sharded = np.arange(240).reshape(_SHARDED_SHAPE)

    stripped = global_layout.strip_halos_global(sharded, grid_params)

    np.testing.assert_array_equal(
        stripped, sharded[np.ix_([1, 2, 3, 4, 7, 8, 9, 10], [1, 2, 3], [1, 2])]
    )


class InflateToGlobalTest(absltest.TestCase):

  def test_restores_sharded_layout_from_points_with_outer_halos(self):
    grid_params = _make_grid_params()
    unique = np.arange(200, dtype=np.float32).reshape(_WITH_HALOS_SHAPE)

    inflated = global_layout.inflate_to_global(
        unique, grid_params, target_shape=_SHARDED_SHAPE
    )

    # Each core is a window of 6 starting at its interior offset, so the two
    # windows overlap by twice the halo width.
    np.testing.assert_array_equal(
        inflated, np.concatenate([unique[0:6], unique[4:10]], axis=0)
    )

  def test_is_the_inverse_of_strip_halos_global(self):
    grid_params = _make_grid_params()
    unique = np.arange(200, dtype=np.float32).reshape(_WITH_HALOS_SHAPE)
    sharded = np.concatenate([unique[0:6], unique[4:10]], axis=0)

    stripped = global_layout.strip_halos_global(
        sharded, grid_params, include_halos=True
    )
    inflated = global_layout.inflate_to_global(
        stripped, grid_params, target_shape=_SHARDED_SHAPE
    )

    np.testing.assert_array_equal(stripped, unique)
    np.testing.assert_array_equal(inflated, sharded)

  def test_interior_seam_halos_are_refilled_from_the_neighbour(self):
    grid_params = _make_grid_params()
    interior = _varying_along_first_axis(
        [10, 11, 12, 13, 14, 15, 16, 17], _INTERIOR_SHAPE
    )

    inflated = global_layout.inflate_to_global(
        interior, grid_params, target_shape=_SHARDED_SHAPE
    )

    # Without halos in the input, the outer halos of a non-periodic axis
    # repeat the edge value.
    np.testing.assert_array_equal(
        inflated,
        _varying_along_first_axis(
            [10, 10, 11, 12, 13, 14, 13, 14, 15, 16, 17, 17], _SHARDED_SHAPE
        ),
    )

  def test_outer_halos_of_periodic_axis_wrap_around(self):
    grid_params = _make_grid_params(periodic_x=True)
    interior = _varying_along_first_axis(
        [10, 11, 12, 13, 14, 15, 16, 17], _INTERIOR_SHAPE
    )

    inflated = global_layout.inflate_to_global(
        interior, grid_params, target_shape=_SHARDED_SHAPE
    )

    np.testing.assert_array_equal(
        inflated,
        _varying_along_first_axis(
            [17, 10, 11, 12, 13, 14, 13, 14, 15, 16, 17, 10], _SHARDED_SHAPE
        ),
    )

  def test_existing_supplies_outer_halos_of_non_periodic_axes(self):
    grid_params = _make_grid_params()
    interior = _varying_along_first_axis(
        [10, 11, 12, 13, 14, 15, 16, 17], _INTERIOR_SHAPE
    )
    existing = np.full(_SHARDED_SHAPE, -1.0, np.float32)

    inflated = global_layout.inflate_to_global(
        interior, grid_params, target_shape=_SHARDED_SHAPE, existing=existing
    )

    # The outer halos of all three axes come from `existing`; the seam halos
    # at 5 and 6 still come from the neighbouring interior.
    expected = np.full(_SHARDED_SHAPE, -1.0, np.float32)
    expected[1:11, 1:4, 1:3] = _varying_along_first_axis(
        [10, 11, 12, 13, 14, 13, 14, 15, 16, 17], (10, 3, 2)
    )
    np.testing.assert_array_equal(inflated, expected)

  def test_existing_is_ignored_where_input_carries_outer_halos(self):
    grid_params = _make_grid_params()
    unique = _varying_along_first_axis(
        [9, 10, 11, 12, 13, 14, 15, 16, 17, 18], _WITH_HALOS_SHAPE
    )
    existing = np.full(_SHARDED_SHAPE, -1.0, np.float32)

    inflated = global_layout.inflate_to_global(
        unique, grid_params, target_shape=_SHARDED_SHAPE, existing=existing
    )

    np.testing.assert_array_equal(
        inflated,
        _varying_along_first_axis(
            [9, 10, 11, 12, 13, 14, 13, 14, 15, 16, 17, 18], _SHARDED_SHAPE
        ),
    )

  def test_existing_is_ignored_on_periodic_axis(self):
    grid_params = _make_grid_params(periodic_x=True)
    # Interior only along the periodic first axis; the other two axes already
    # carry their outer halos.
    unique = _varying_along_first_axis(
        [10, 11, 12, 13, 14, 15, 16, 17], (8, 5, 4)
    )
    existing = np.full(_SHARDED_SHAPE, -1.0, np.float32)

    inflated = global_layout.inflate_to_global(
        unique, grid_params, target_shape=_SHARDED_SHAPE, existing=existing
    )

    np.testing.assert_array_equal(
        inflated,
        _varying_along_first_axis(
            [17, 10, 11, 12, 13, 14, 13, 14, 15, 16, 17, 10], _SHARDED_SHAPE
        ),
    )

  def test_axis_already_at_target_length_is_left_alone(self):
    grid_params = _make_grid_params()
    interior = _varying_along_first_axis(
        [10, 11, 12, 13, 14, 15, 16, 17], (8, 1, 2)
    )

    inflated = global_layout.inflate_to_global(
        interior, grid_params, target_shape=(12, 1, 4)
    )

    np.testing.assert_array_equal(
        inflated,
        _varying_along_first_axis(
            [10, 10, 11, 12, 13, 14, 13, 14, 15, 16, 17, 17], (12, 1, 4)
        ),
    )

  def test_preserves_dtype(self):
    grid_params = _make_grid_params()
    interior = np.zeros(_INTERIOR_SHAPE, np.int16)

    inflated = global_layout.inflate_to_global(
        interior, grid_params, target_shape=_SHARDED_SHAPE
    )

    self.assertEqual(inflated.dtype, np.int16)

  def test_rank_mismatch_raises(self):
    grid_params = _make_grid_params()

    with self.assertRaisesRegex(ValueError, 'Cannot inflate an array of rank'):
      global_layout.inflate_to_global(
          np.zeros((8, 3)), grid_params, target_shape=_SHARDED_SHAPE
      )

  def test_axis_not_recorded_as_stripped_raises(self):
    grid_params = _make_grid_params()

    with self.assertRaisesRegex(ValueError, 'not recorded as converted'):
      global_layout.inflate_to_global(
          np.zeros(_INTERIOR_SHAPE),
          grid_params,
          target_shape=_SHARDED_SHAPE,
          stripped_axes=(1, 2),
      )

  def test_length_that_is_neither_interior_nor_with_halos_raises(self):
    grid_params = _make_grid_params()

    with self.assertRaisesRegex(ValueError, 'Axis 0 has length 9, expected'):
      global_layout.inflate_to_global(
          np.zeros((9, 5, 4)), grid_params, target_shape=_SHARDED_SHAPE
      )

  def test_target_that_is_not_the_sharded_layout_raises(self):
    grid_params = _make_grid_params()

    with self.assertRaisesRegex(
        ValueError, 'does not match the sharded layout'
    ):
      global_layout.inflate_to_global(
          np.zeros((8, 5, 4)),
          grid_params,
          target_shape=(13, 5, 4),
          stripped_axes=(0,),
      )

  def test_existing_of_wrong_shape_raises(self):
    grid_params = _make_grid_params()

    with self.assertRaisesRegex(ValueError, '`existing` has shape'):
      global_layout.inflate_to_global(
          np.zeros(_INTERIOR_SHAPE),
          grid_params,
          target_shape=_SHARDED_SHAPE,
          existing=np.zeros((12, 5, 5)),
      )


if __name__ == '__main__':
  absltest.main()
