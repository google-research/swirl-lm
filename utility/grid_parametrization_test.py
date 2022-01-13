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

"""Tests for grid_parametrization."""

import itertools

from absl import flags
from absl.testing import parameterized
import numpy as np
from swirl_lm.utility import grid_parametrization
import tensorflow as tf

FLAGS = flags.FLAGS


class GridParametrizationTest(tf.test.TestCase, parameterized.TestCase):

  NX_NY_NZ = [
      (1, 3, 5),
      (5, 1, 3),
      (3, 5, 1),
  ]

  HALO_WIDTH = [1, 2]

  FLAGS_VALUES = [
      element[0] + (element[1],) for element in itertools.product(
          NX_NY_NZ, HALO_WIDTH)
  ]
  EXPECTED_VALUES = [
      (None, 1, 3),
      (None, None, 1),
      (3, None, 1),
      (1, None, None),
      (1, 3, None),
      (None, 1, None)
  ]

  RUN_CONDITIONS = [
      flags_values + expected_values for flags_values, expected_values in
      zip(FLAGS_VALUES, EXPECTED_VALUES)
  ]

  @parameterized.parameters(RUN_CONDITIONS)
  def testCoreNxNyNzProperties(
      self, nx, ny, nz, halo_width, expected_core_nx, expected_core_ny,
      expected_core_nz):
    FLAGS.cx = 1
    FLAGS.cy = 1
    FLAGS.cz = 1
    FLAGS.nx = nx
    FLAGS.ny = ny
    FLAGS.nz = nz
    FLAGS.halo_width = halo_width
    params = grid_parametrization.GridParametrization()
    self.assertEqual(expected_core_nx, params.core_nx)
    self.assertEqual(expected_core_ny, params.core_ny)
    self.assertEqual(expected_core_nz, params.core_nz)
    self.assertEqual(1, params.num_replicas)

  def testCreateFromFlags(self):
    FLAGS.cx = FLAGS.cy = FLAGS.cz = 2
    params = grid_parametrization.GridParametrization.create_from_flags()
    self.assertEqual(8, params.num_replicas)

  @parameterized.parameters(*zip(HALO_WIDTH))
  def testFullGridCreatedCorrectly(self, halo_width):
    """Checks if full grids in dim 0, 1, and 2 are created correctly."""
    FLAGS.cx = 2
    FLAGS.cy = 2
    FLAGS.cz = 2
    FLAGS.nx = 16
    FLAGS.ny = 32
    FLAGS.nz = 64
    FLAGS.lx = 1.0
    FLAGS.ly = 2.0
    FLAGS.lz = 4.0
    FLAGS.halo_width = halo_width
    FLAGS.num_boundary_points = 0
    params = grid_parametrization.GridParametrization()

    nx = 2 * (16 - 2 * halo_width)
    ny = 2 * (32 - 2 * halo_width)
    nz = 2 * (64 - 2 * halo_width)
    expected_x = np.linspace(0, 1.0, nx)
    expected_y = np.linspace(0, 2.0, ny)
    expected_z = np.linspace(0, 4.0, nz)

    with self.subTest(name='GridInDim0'):
      with self.session():
        x = params.x.numpy()
      self.assertEqual(nx, params.fx)
      self.assertEqual(expected_x[1] - expected_x[0], params.dx)
      self.assertAllClose(expected_x, x)

    with self.subTest(name='GridInDim0'):
      with self.session():
        y = params.y.numpy()
      self.assertEqual(ny, params.fy)
      self.assertEqual(expected_y[1] - expected_y[0], params.dy)
      self.assertAllClose(expected_y, y)

    with self.subTest(name='GridInDim0'):
      with self.session():
        z = params.z.numpy()
      self.assertEqual(nz, params.fz)
      self.assertEqual(expected_z[1] - expected_z[0], params.dz)
      self.assertAllClose(expected_z, z)

  def testCreateFromGridLengthsAndEtc(self):
    grid_lengths = (2, 3, 4)
    computation_shape = (5, 6, 7)
    subgrid_shape = (8, 9, 10)
    halo_width = 3

    actual = (grid_parametrization.GridParametrization.
              create_from_grid_lengths_and_etc(
                  grid_lengths, computation_shape, subgrid_shape, halo_width))

    self.assertEqual(actual.lx, grid_lengths[0])
    self.assertEqual(actual.ly, grid_lengths[1])
    self.assertEqual(actual.lz, grid_lengths[2])

    self.assertEqual(actual.cx, computation_shape[0])
    self.assertEqual(actual.cy, computation_shape[1])
    self.assertEqual(actual.cz, computation_shape[2])

    self.assertEqual(actual.nx, subgrid_shape[0])
    self.assertEqual(actual.ny, subgrid_shape[1])
    self.assertEqual(actual.nz, subgrid_shape[2])

    self.assertEqual(actual.halo_width, halo_width)

  def testCreateFromGridLengthsAndEtcWithDefaults(self):
    grid_lengths = (2, 3, 4)
    computation_shape = (5, 6, 7)

    actual = (grid_parametrization.GridParametrization.
              create_from_grid_lengths_and_etc_with_defaults(
                  grid_lengths, computation_shape))

    self.assertEqual(actual.lx, grid_lengths[0])
    self.assertEqual(actual.ly, grid_lengths[1])
    self.assertEqual(actual.lz, grid_lengths[2])

    self.assertEqual(actual.cx, computation_shape[0])
    self.assertEqual(actual.cy, computation_shape[1])
    self.assertEqual(actual.cz, computation_shape[2])

    # Default values from flags.
    self.assertEqual(actual.nx, 4)
    self.assertEqual(actual.ny, 4)
    self.assertEqual(actual.nz, 4)
    self.assertEqual(actual.halo_width, 1)

  def testCreateFromGridLengthsAndEtcWithDefaults_CheckGridSpacings(self):
    lx, ly, lz = 0.22, 0.33, 0.44
    grid_lengths = (lx, ly, lz)

    actual = (grid_parametrization.GridParametrization.
              create_from_grid_lengths_and_etc_with_defaults(grid_lengths))

    self.assertAlmostEqual(actual.lx, grid_lengths[0])
    self.assertAlmostEqual(actual.ly, grid_lengths[1])
    self.assertAlmostEqual(actual.lz, grid_lengths[2])

    # If `computation_shape` and `subgrid_spacings` have default values,
    # `grid_lengths` are grid spacings.
    self.assertAlmostEqual(actual.dx, lx)
    self.assertAlmostEqual(actual.dy, ly)
    self.assertAlmostEqual(actual.dz, lz)


if __name__ == '__main__':
  tf.test.main()
