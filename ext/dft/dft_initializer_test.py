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

"""Tests for dft_initializer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from swirl_lm.ext.dft import dft_initializer
from swirl_lm.ext.dft.dft_initializer import PartitionDimension
from swirl_lm.utility import grid_parametrization
import tensorflow.compat.v1 as tf

from google3.testing.pybase import parameterized

_NUM_PTS_PER_DIM_PER_CORE = (
    (4, 4, 4),
    (8, 4, 4),
    (4, 8, 4),
    (8, 8, 4),
    (1, 4, 4),
    (4, 1, 4),
    (4, 8, 8),
    (8, 8, 8),
)

_NUM_CORES_PER_DIM = (
    (4, 2, 1),
    (2, 4, 1),
    (4, 4, 1),
    (2, 2, 1),
    (2, 2, 4),
    (4, 2, 2),
    (2, 2, 2),
    (4, 2, 4),
)

_PARTITION_DIMENSION = (
    PartitionDimension.DIM0,
    PartitionDimension.DIM0,
    PartitionDimension.DIM1,
    PartitionDimension.DIM1,
    PartitionDimension.DIM2,
    PartitionDimension.DIM2,
    PartitionDimension.DIM0,
    PartitionDimension.DIM1,
)


def _gen_value_for_vandermonde(mat_dim0, mat_dim1, num_pts):
  """Generates values of a Vandermonde matrix based on given grids.

  Grids are supplied in the form of `np.ndarray`s.

  Args:
    mat_dim0: An `np.ndarray` representing the coordinates of the grids
      along dimension 0.
    mat_dim1: An `np.ndarray` representing the coordinates of the grids
      along dimension 1.
    num_pts: The total number of sampling points in the calculation of
      Fourier transform.
  Returns:
    An `np.ndarray` representing the Vandermonde matrix.
  """
  coeff = -2.0 * dft_initializer._PI * dft_initializer._J / num_pts
  return np.exp(coeff * np.multiply(mat_dim0, mat_dim1))


class DFTInitializerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(DFTInitializerTest, self).setUp()
    self.params = grid_parametrization.GridParametrization()

  def set_params(self, n=(4, 4, 4), c=(2, 2, 1)):
    self.params.nx = n[0]
    self.params.ny = n[1]
    self.params.nz = n[2]
    self.params.cx = c[0]
    self.params.cy = c[1]
    self.params.cz = c[2]

  @parameterized.parameters(zip(_NUM_PTS_PER_DIM_PER_CORE,
                                _NUM_CORES_PER_DIM, _PARTITION_DIMENSION))
  def testGenVandermondeSpatiotemporalPartition(
      self, n, c, partition_dimension):
    self.set_params(n, c)

    if partition_dimension == PartitionDimension.DIM0:
      num_pts = self.params.cy * self.params.ny
    elif partition_dimension == PartitionDimension.DIM1:
      num_pts = self.params.cx * self.params.nx
    elif partition_dimension == PartitionDimension.DIM2:
      num_pts = self.params.cz * self.params.nz

    vec_full = np.linspace(0.0, np.float32(num_pts - 1), num_pts)
    mat_dim0_full, mat_dim1_full = np.meshgrid(vec_full,
                                               vec_full,
                                               indexing='ij')
    expected_full_value = _gen_value_for_vandermonde(
        mat_dim0_full, mat_dim1_full, num_pts)

    def get_expected_partial_value(expected_full_value, coordinate, n0, n1, n2,
                                   partition_dimension):
      g0 = coordinate[0]
      g1 = coordinate[1]
      g2 = coordinate[2]

      if partition_dimension == PartitionDimension.DIM0:
        return expected_full_value[g1 * n1:(g1 + 1) * n1, :]
      elif partition_dimension == PartitionDimension.DIM1:
        return expected_full_value[:, g0 * n0:(g0 + 1) * n0]
      elif partition_dimension == PartitionDimension.DIM2:
        return expected_full_value[g2 * n2:(g2 + 1) * n2, :]

    with self.session() as sess:
      for i in range(self.params.cx):
        for j in range(self.params.cy):
          for k in range(self.params.cz):
            with self.subTest(
                name='vandermonde_matrix_core ({}, {}, {})'.format(i, j, k)):
              actual_partial_value = sess.run(
                  dft_initializer.gen_vandermonde_mat(
                      self.params, [i, j, k], partition_dimension,
                      dft_initializer.PartitionDomain.SPATIO_TEMPORAL))
              expected_partial_value = get_expected_partial_value(
                  expected_full_value, [i, j, k], self.params.nx,
                  self.params.ny, self.params.nz, partition_dimension)
              self.assertAllClose(np.abs(actual_partial_value),
                                  np.abs(expected_partial_value),
                                  msg='coordinate {}, {}, {}'.format(i, j, k))

  @parameterized.parameters(zip(_NUM_PTS_PER_DIM_PER_CORE,
                                _NUM_CORES_PER_DIM, _PARTITION_DIMENSION))
  def testGenVandermondeSpectralPartition(
      self, n, c, partition_dimension):
    self.set_params(n, c)

    if partition_dimension == PartitionDimension.DIM0:
      num_pts = self.params.cx * self.params.nx
    elif partition_dimension == PartitionDimension.DIM1:
      num_pts = self.params.cy * self.params.ny
    if partition_dimension == PartitionDimension.DIM2:
      num_pts = self.params.cz * self.params.nz

    vec_full = np.linspace(0.0, np.float32(num_pts - 1), num_pts)
    mat_dim0_full, mat_dim1_full = np.meshgrid(vec_full,
                                               vec_full,
                                               indexing='ij')
    expected_full_value = _gen_value_for_vandermonde(
        mat_dim0_full, mat_dim1_full, num_pts)

    def get_expected_partial_value(expected_full_value, coordinate, n0, n1, n2,
                                   partition_dimension):
      g0 = coordinate[0]
      g1 = coordinate[1]
      g2 = coordinate[2]

      if partition_dimension == PartitionDimension.DIM0:
        return expected_full_value[g0 * n0:(g0 + 1) * n0, :]
      elif partition_dimension == PartitionDimension.DIM1:
        return expected_full_value[g1 * n1:(g1 + 1) * n1, :]
      elif partition_dimension == PartitionDimension.DIM2:
        return expected_full_value[g2 * n2:(g2 + 1) * n2, :]

    with self.session() as sess:
      for i in range(self.params.cx):
        for j in range(self.params.cy):
          for k in range(self.params.cz):
            with self.subTest(
                name='vandermonde_matrix_core ({}, {}, {})'.format(i, j, k)):
              actual_partial_value = sess.run(
                  dft_initializer.gen_vandermonde_mat(
                      self.params, [i, j, k], partition_dimension,
                      dft_initializer.PartitionDomain.SPECTRAL))
              expected_partial_value = get_expected_partial_value(
                  expected_full_value, [i, j, k], self.params.nx,
                  self.params.ny, self.params.nz, partition_dimension)
              self.assertAllClose(np.abs(actual_partial_value),
                                  np.abs(expected_partial_value),
                                  msg='coordinate {}, {}, {}'.format(i, j, k))

if __name__ == '__main__':
  tf.test.main()
