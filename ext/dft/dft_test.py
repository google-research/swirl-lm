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

"""Tests for DFT computation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from swirl_lm.ext.dft import dft
from swirl_lm.ext.dft import dft_initializer
from swirl_lm.utility import grid_parametrization
import tensorflow.compat.v1 as tf

from google3.research.simulation.tensorflow.fluid.framework import util
from google3.testing.pybase import parameterized
from google3.third_party.tensorflow.contrib.tpu.python.tpu import topology as tpu_topology
from google3.third_party.tensorflow.contrib.tpu.python.tpu import tpu


def _concatenate_2d(replica_outputs, group_assignment_dim0):
  """Joins the outputs from individual replicas into one single 2D `np.ndarray`.

  Args:
    replica_outputs: `List[tf.Tensor]` containing the computation results
      of DFT.
    group_assignment_dim0: A `List` of replica_groups with each replica_group
      corresponding to a `List` of replica ids.

  Returns:
    A 2D `np.ndarray` containing the concatenated computation results across all
      replicas.
  """
  a = []
  for replicas_per_group in group_assignment_dim0:
    b = []
    for replica_id in replicas_per_group:
      b.append(np.asarray(replica_outputs[replica_id][0]))
    a.append(np.concatenate(b, axis=0))
  return np.concatenate(a, axis=1)


def _concatenate_3d(replica_outputs, num_cores_dim0,
                    num_cores_dim1, group_assignment_dim2):
  """Joins the outputs from individual replicas into one single 3D `np.ndarray`.

  Args:
    replica_outputs: `List[tf.Tensor]` containing the computation results
      of DFT.
    num_cores_dim0: An `int` representing the number of cores along
      dimension 0.
    num_cores_dim1: An `int` representing the number of cores along
      dimension 1.
    group_assignment_dim2: A `List` of replica_groups along dimension 2.

  Returns:
    A 3D `np.ndarray` containing the concatenated computation results across all
      replicas.
  """

  concatenated_results_dim2 = []
  for replicas_per_group in group_assignment_dim2:
    b = []
    for replica_id in replicas_per_group:
      b.append(np.asarray(replica_outputs[replica_id][0]))
    concatenated_results_dim2.append([np.concatenate(b, axis=2)])

  num_replicas = num_cores_dim0 * num_cores_dim1
  replica_ids = np.arange(num_replicas).reshape((num_cores_dim0,
                                                 num_cores_dim1))

  group_assignment_dim0 = []
  for x in np.transpose(replica_ids):
    if len(x) < num_cores_dim1:
      group_assignment_dim0.append(x.tolist())
    else:
      for y in (np.transpose(x.reshape((-1, num_cores_dim1)))).tolist():
        group_assignment_dim0.append(y)

  return _concatenate_2d(concatenated_results_dim2, group_assignment_dim0)


def _rectangle(mm, nn, aa, bb):
  """Generates a rectangular signal.

  Args:
    mm: An `int` representing the length of input/output along dimension 0.
    nn: An `int` representing the length of input/output along dimension 1.
    aa: An `int` representing the length of the rectangle of ones
      along dimension 0.
    bb: An `int` representing the length of the rectangle of ones
      along dimension 1.

  Returns:
    A 2D `np.ndarray` representing a rectangular signal of shape [mm, nn] with
    ones in the central region of shape [aa, bb] and zeros elsewhere.
  """
  center_dim0 = mm // 2
  center_dim1 = nn // 2
  mat_rectangle = np.zeros((mm, nn))
  mat_rectangle[(center_dim0 - aa // 2):(center_dim0 + aa // 2),
                (center_dim1 - bb // 2):(center_dim1 + bb // 2)] = 1
  return mat_rectangle


class DftTest(tf.test.TestCase, parameterized.TestCase):

  # The number of points along dimension 2 is not used by Dft2d
  NUM_PTS_PER_DIM_PER_CORE_DFT2D = [
      (8, 16, 4),
      (16, 8, 4),
      (16, 16, 4),
  ]

  # The number of cores along dimension 2 must be 1 in Dft2d
  NUM_CORES_PER_DIM_DFT2D = [
      (1, 2, 1),
      (2, 1, 1),
  ]

  NUM_PTS_PER_DIM_PER_CORE_DFT3D = [
      (4, 4, 8),
  ]
  NUM_CORES_PER_DIM_DFT3D = [
      (2, 1, 1),
      (1, 2, 1),
      (1, 1, 2),
  ]

  def setUp(self):
    super(DftTest, self).setUp()
    self.params = grid_parametrization.GridParametrization()

  def set_params(self, n=(4, 4, 4), c=(2, 2, 1)):
    self.params.nx = n[0]
    self.params.ny = n[1]
    self.params.nz = n[2]
    self.params.cx = c[0]
    self.params.cy = c[1]
    self.params.cz = c[2]

  def testGenGroupAssignment2D(self):
    computation_shape = [3, 2, 1]
    actual_group_assignment = dft.gen_group_assignment(computation_shape,
                                                       dft.Dimension.DIM0)
    expected_group_assignment = [[0, 2, 4], [1, 3, 5]]
    self.assertEqual(actual_group_assignment, expected_group_assignment)
    actual_group_assignment = dft.gen_group_assignment(computation_shape,
                                                       dft.Dimension.DIM1)
    expected_group_assignment = [[0, 1], [2, 3], [4, 5]]
    self.assertEqual(actual_group_assignment, expected_group_assignment)

  def testGenGroupAssignment3D(self):
    computation_shape = [2, 2, 2]
    expected_group_assignment = [[0, 4], [2, 6], [1, 5], [3, 7]]
    actual_group_assignment = dft.gen_group_assignment(computation_shape,
                                                       dft.Dimension.DIM0)
    self.assertEqual(actual_group_assignment, expected_group_assignment)

    expected_group_assignment = [[0, 2], [4, 6], [1, 3], [5, 7]]
    actual_group_assignment = dft.gen_group_assignment(computation_shape,
                                                       dft.Dimension.DIM1)
    self.assertEqual(actual_group_assignment, expected_group_assignment)

    expected_group_assignment = [[0, 1], [2, 3], [4, 5], [6, 7]]
    actual_group_assignment = dft.gen_group_assignment(computation_shape,
                                                       dft.Dimension.DIM2)
    self.assertEqual(actual_group_assignment, expected_group_assignment)

  def testGenSourceTargetPairsCubeShape(self):
    """Tests the generation of source-target pairs with a cubic shape.

    The `group_assignment` is a list of `replica_group`s along one specific
    dimension of the 3D mesh and the soruce-target paris are created within
    each `group_assignment`. In the test, the computation shape is chosen as
    [3, 3, 3] such that
    the group assignment along dim0 is [[0, 9, 18], [3, 12, 21], [6, 15, 24],
                                        [1, 10, 19], [4, 13, 22], [7, 16, 25],
                                        [2, 11, 20], [5, 14, 23], [8, 17, 26]],
    the group assignment along dim1 is [[0, 3, 6], [9, 12, 15], [18, 21, 24],
                                        [1, 4, 7], [10, 13, 16], [19, 22, 25],
                                        [2, 5, 8], [11, 14, 17], [20, 23, 26]],
    and the group assignment along dim2 is [[0, 1, 2], [3, 4, 5], [6, 7, 8],
                                            [9, 10, 11], [12, 13, 14],
                                            [15, 16, 17],
                                            [18, 19, 20], [21, 22, 23],
                                            [24, 25, 26]].
    """
    computation_shape = [3, 3, 3]

    expected_source_target_pairs_dim0 = [(9, 0), (18, 9), (0, 18),
                                         (12, 3), (21, 12), (3, 21),
                                         (15, 6), (24, 15), (6, 24),
                                         (10, 1), (19, 10), (1, 19),
                                         (13, 4), (22, 13), (4, 22),
                                         (16, 7), (25, 16), (7, 25),
                                         (11, 2), (20, 11), (2, 20),
                                         (14, 5), (23, 14), (5, 23),
                                         (17, 8), (26, 17), (8, 26)]

    expected_source_target_pairs_dim1 = [(3, 0), (6, 3), (0, 6),
                                         (12, 9), (15, 12), (9, 15),
                                         (21, 18), (24, 21), (18, 24),
                                         (4, 1), (7, 4), (1, 7),
                                         (13, 10), (16, 13), (10, 16),
                                         (22, 19), (25, 22), (19, 25),
                                         (5, 2), (8, 5), (2, 8),
                                         (14, 11), (17, 14), (11, 17),
                                         (23, 20), (26, 23), (20, 26)]

    expected_source_target_pairs_dim2 = [(1, 0), (2, 1), (0, 2),
                                         (4, 3), (5, 4), (3, 5),
                                         (7, 6), (8, 7), (6, 8),
                                         (10, 9), (11, 10), (9, 11),
                                         (13, 12), (14, 13), (12, 14),
                                         (16, 15), (17, 16), (15, 17),
                                         (19, 18), (20, 19), (18, 20),
                                         (22, 21), (23, 22), (21, 23),
                                         (25, 24), (26, 25), (24, 26)]

    actual_source_target_pairs_dim0 = dft.gen_source_target_pairs(
        computation_shape, dft.Dimension.DIM0)
    self.assertEqual(actual_source_target_pairs_dim0,
                     expected_source_target_pairs_dim0,
                     'Source-target pairs along dimension 0.')

    actual_source_target_pairs_dim1 = dft.gen_source_target_pairs(
        computation_shape, dft.Dimension.DIM1)
    self.assertEqual(actual_source_target_pairs_dim1,
                     expected_source_target_pairs_dim1,
                     'Source-target pairs along dimension 1.')

    actual_source_target_pairs_dim2 = dft.gen_source_target_pairs(
        computation_shape, dft.Dimension.DIM2)
    self.assertEqual(actual_source_target_pairs_dim2,
                     expected_source_target_pairs_dim2,
                     'Source-target pairs along dimension 2.')

  def testGenSourceTargetPairsNonCubeShape(self):
    """Tests the generation of source-target pairs with non-cube shape."""
    computation_shape = [2, 3, 4]
    expected_source_target_pairs_dim0 = [(12, 0), (0, 12), (16, 4), (4, 16),
                                         (20, 8), (8, 20), (13, 1), (1, 13),
                                         (17, 5), (5, 17), (21, 9), (9, 21),
                                         (14, 2), (2, 14), (18, 6), (6, 18),
                                         (22, 10), (10, 22), (15, 3), (3, 15),
                                         (19, 7), (7, 19), (23, 11), (11, 23)]

    expected_source_target_pairs_dim1 = [(4, 0), (8, 4), (0, 8), (16, 12),
                                         (20, 16), (12, 20), (5, 1), (9, 5),
                                         (1, 9), (17, 13), (21, 17), (13, 21),
                                         (6, 2), (10, 6), (2, 10), (18, 14),
                                         (22, 18), (14, 22), (7, 3), (11, 7),
                                         (3, 11), (19, 15), (23, 19), (15, 23)]

    expected_source_target_pairs_dim2 = [(1, 0), (2, 1), (3, 2), (0, 3),
                                         (5, 4), (6, 5), (7, 6), (4, 7),
                                         (9, 8), (10, 9), (11, 10), (8, 11),
                                         (13, 12), (14, 13), (15, 14), (12, 15),
                                         (17, 16), (18, 17), (19, 18), (16, 19),
                                         (21, 20), (22, 21), (23, 22), (20, 23)]

    actual_source_target_pairs_dim0 = dft.gen_source_target_pairs(
        computation_shape, dft.Dimension.DIM0)
    self.assertEqual(actual_source_target_pairs_dim0,
                     expected_source_target_pairs_dim0,
                     'Source-target pairs along dimension 0.')

    actual_source_target_pairs_dim1 = dft.gen_source_target_pairs(
        computation_shape, dft.Dimension.DIM1)
    self.assertEqual(actual_source_target_pairs_dim1,
                     expected_source_target_pairs_dim1,
                     'Source-target pairs along dimension 1.')

    actual_source_target_pairs_dim2 = dft.gen_source_target_pairs(
        computation_shape, dft.Dimension.DIM2)
    self.assertEqual(actual_source_target_pairs_dim2,
                     expected_source_target_pairs_dim2,
                     'Source-target pairs along dimension 2.')

  @parameterized.parameters(*itertools.product(NUM_PTS_PER_DIM_PER_CORE_DFT2D,
                                               NUM_CORES_PER_DIM_DFT2D))
  def testDft2d(self,
                num_pts_per_dim_per_core,
                num_cores_per_dim):
    self.set_params(num_pts_per_dim_per_core,
                    num_cores_per_dim)
    computation_shape = np.array(num_cores_per_dim)
    num_replicas = np.prod(computation_shape)
    mm = num_pts_per_dim_per_core[0] * computation_shape[0]
    nn = num_pts_per_dim_per_core[1] * computation_shape[1]
    input_signal = _rectangle(mm, nn, aa=mm // 4, bb=nn // 4)
    group_assignment_dim0 = dft.gen_group_assignment(computation_shape,
                                                     dft.Dimension.DIM0)

    def device_fn(*args):
      """Creates 2D DFT computation that is passed to TPU replicas.

      Args:
        *args: args[0] and args[1] are 2D `Tensor`s of `tf.complex64`
          representing the Vandermonde matrices that pre- and post- multiplies
          the input `Tensor`, respectively; and args[2] is the 2D `Tensor`
          representing the input signal.

      Returns:
        A 2D `Tensor` of `tf.complex64`.
      """
      a = args[2]
      vm = args[0]
      vn = args[1]
      return dft.dft_2d(a, vm, vn, computation_shape)

    with self.session() as sess:
      topology = tpu_topology.Topology(sess.run(tpu.initialize_system()))

      (device_assignment,
       compute_core_assignment) = util.tpu_device_assignment(computation_shape,
                                                             topology)
      replica_inputs = []
      for replica_id in range(num_replicas):
        coordinates = compute_core_assignment[replica_id, :]
        inputs = []
        inputs.append(dft_initializer.gen_vandermonde_mat(
            self.params,
            [coordinates[0], coordinates[1], 0],
            dft_initializer.PartitionDimension.DIM1,
            dft_initializer.PartitionDomain.SPATIO_TEMPORAL))
        inputs.append(dft_initializer.gen_vandermonde_mat(
            self.params,
            [coordinates[0], coordinates[1], 0],
            dft_initializer.PartitionDimension.DIM0,
            dft_initializer.PartitionDomain.SPATIO_TEMPORAL))
        inputs.append(tf.convert_to_tensor(input_signal[
            coordinates[0] * self.params.nx:
            (coordinates[0] + 1) * self.params.nx,
            coordinates[1] * self.params.ny:
            (coordinates[1] + 1) * self.params.ny], dtype=tf.complex64))
        replica_inputs.append(inputs)

      tpu_step = tpu.replicate(device_fn,
                               inputs=replica_inputs,
                               device_assignment=device_assignment)

      replica_outputs = sess.run(tpu_step)
      actual_results = _concatenate_2d(replica_outputs, group_assignment_dim0)
      expected_results = np.fft.fft2(input_signal)

      self.assertAllClose(a=actual_results.real,
                          b=expected_results.real,
                          rtol=1e-04,
                          atol=1e-04)
      self.assertAllClose(a=actual_results.imag,
                          b=expected_results.imag,
                          rtol=1e-04,
                          atol=1e-04)

  @parameterized.parameters(*itertools.product(NUM_PTS_PER_DIM_PER_CORE_DFT3D,
                                               NUM_CORES_PER_DIM_DFT3D))
  def testDft3d(self,
                num_pts_per_dim_per_core,
                num_cores_per_dim):
    self.set_params(num_pts_per_dim_per_core,
                    num_cores_per_dim)
    computation_shape = np.array(num_cores_per_dim)
    num_replicas = np.prod(computation_shape)
    mm = num_pts_per_dim_per_core[0] * computation_shape[0]
    nn = num_pts_per_dim_per_core[1] * computation_shape[1]
    ss = num_pts_per_dim_per_core[2] * computation_shape[2]

    input_signal = (np.random.normal(0, 0.1, mm * nn * ss)).reshape((mm,
                                                                     nn,
                                                                     ss))
    group_assignment_dim2 = dft.gen_group_assignment(computation_shape,
                                                     dft.Dimension.DIM2)

    def device_fn(*args):
      """Creates 3D DFT computation that is passed to TPU replicas.

      Args:
        *args: args[0] and args[1] are 2D `Tensor`s of `tf.complex64`
          representing the Vandermonde matrices that pre- and post- multiplies
          the input `Tensor` in 2D DFT computation, respectively; args[2] is
          the 3D `Tensor` representing the input signal; and args[3] is the 2D
          `Tensor` of `tf.complex64` representintg the Vandermonde matrix for
          the DFT computation along the third dimension.

      Returns:
        A 3D `Tensor` of `tf.complex64`.
      """
      a = args[2]
      vm = args[0]
      vn = args[1]
      vs = args[3]
      return dft.dft_3d(a, vm, vn, vs, computation_shape)

    with self.session() as sess:
      topology = tpu_topology.Topology(sess.run(tpu.initialize_system()))

      (device_assignment,
       compute_core_assignment) = util.tpu_device_assignment(computation_shape,
                                                             topology)
      replica_inputs = []
      for replica_id in range(num_replicas):
        coordinates = compute_core_assignment[replica_id, :]
        inputs = []

        inputs.append(dft_initializer.gen_vandermonde_mat(
            self.params,
            [coordinates[0], coordinates[1], 0],
            dft_initializer.PartitionDimension.DIM1,
            dft_initializer.PartitionDomain.SPATIO_TEMPORAL))
        inputs.append(dft_initializer.gen_vandermonde_mat(
            self.params,
            [coordinates[0], coordinates[1], 0],
            dft_initializer.PartitionDimension.DIM0,
            dft_initializer.PartitionDomain.SPATIO_TEMPORAL))
        inputs.append(tf.convert_to_tensor(input_signal[
            coordinates[0] * self.params.nx:
            (coordinates[0] + 1) * self.params.nx,
            coordinates[1] * self.params.ny:
            (coordinates[1] + 1) * self.params.ny,
            coordinates[2] * self.params.nz:
            (coordinates[2] + 1) * self.params.nz], dtype=tf.complex64))
        inputs.append(dft_initializer.gen_vandermonde_mat(
            self.params, coordinates, dft_initializer.PartitionDimension.DIM2,
            dft_initializer.PartitionDomain.SPATIO_TEMPORAL))
        replica_inputs.append(inputs)

      tpu_step = tpu.replicate(device_fn,
                               inputs=replica_inputs,
                               device_assignment=device_assignment)

      replica_outputs = sess.run(tpu_step)
      actual_results = _concatenate_3d(replica_outputs, self.params.cx,
                                       self.params.cy, group_assignment_dim2)
      expected_results = np.fft.fftn(input_signal)

      self.assertAllClose(a=actual_results.real,
                          b=expected_results.real,
                          rtol=1e-04,
                          atol=1e-04)
      self.assertAllClose(a=actual_results.imag,
                          b=expected_results.imag,
                          rtol=1e-04,
                          atol=1e-04)

  @parameterized.parameters(*itertools.product(NUM_PTS_PER_DIM_PER_CORE_DFT3D,
                                               NUM_CORES_PER_DIM_DFT3D))
  def testDft3dSliceCrossReplicaSum(self,
                                    num_pts_per_dim_per_core,
                                    num_cores_per_dim):
    self.set_params(num_pts_per_dim_per_core,
                    num_cores_per_dim)
    computation_shape = np.array(num_cores_per_dim)
    num_replicas = np.prod(computation_shape)
    mm = num_pts_per_dim_per_core[0] * computation_shape[0]
    nn = num_pts_per_dim_per_core[1] * computation_shape[1]
    ss = num_pts_per_dim_per_core[2] * computation_shape[2]

    input_signal = (np.random.normal(0, 0.1, mm * nn * ss)).reshape((mm,
                                                                     nn,
                                                                     ss))
    group_assignment_dim2 = dft.gen_group_assignment(computation_shape,
                                                     dft.Dimension.DIM2)

    def device_fn(*args):
      """Creates 3D DFT batch-mode computation that is passed to TPU replicas.

      Args:
        *args: args[0] and args[1] are 2D `Tensor`s of `tf.complex64`
          representing the Vandermonde matrices that pre- and post- multiplies
          the input `Tensor` in 2D DFT computation, respectively; args[2] is
          the 3D `Tensor` representing the input signal; args[3] is the 2D
          `Tensor` of `tf.complex64` representintg the Vandermonde matrix for
          the DFT computation along the third dimension; and args[4] contains
          the indices of the TPU core in the logical mesh [i, j, k].
      Returns:
        A 3D `Tensor` of `tf.complex64`.
      """
      a = args[2]
      vm = args[0]
      vn = args[1]
      vs = args[3]
      core_indices = args[4]
      return dft.dft_3d_slice_cross_replica_sum(
          a, vm, vn, vs, computation_shape, core_indices)

    with self.session() as sess:
      topology = tpu_topology.Topology(sess.run(tpu.initialize_system()))

      (device_assignment,
       compute_core_assignment) = util.tpu_device_assignment(computation_shape,
                                                             topology)
      replica_inputs = []
      for replica_id in range(num_replicas):
        coordinates = compute_core_assignment[replica_id, :]
        inputs = []

        inputs.append(dft_initializer.gen_vandermonde_mat(
            self.params,
            [coordinates[0], coordinates[1], 0],
            dft_initializer.PartitionDimension.DIM1,
            dft_initializer.PartitionDomain.SPATIO_TEMPORAL))
        inputs.append(dft_initializer.gen_vandermonde_mat(
            self.params,
            [coordinates[0], coordinates[1], 0],
            dft_initializer.PartitionDimension.DIM0,
            dft_initializer.PartitionDomain.SPATIO_TEMPORAL))
        inputs.append(tf.convert_to_tensor(input_signal[
            coordinates[0] * self.params.nx:
            (coordinates[0] + 1) * self.params.nx,
            coordinates[1] * self.params.ny:
            (coordinates[1] + 1) * self.params.ny,
            coordinates[2] * self.params.nz:
            (coordinates[2] + 1) * self.params.nz], dtype=tf.complex64))
        inputs.append(dft_initializer.gen_vandermonde_mat(
            self.params, coordinates, dft_initializer.PartitionDimension.DIM2,
            dft_initializer.PartitionDomain.SPATIO_TEMPORAL))
        inputs.append((tf.constant(coordinates[0], tf.int32),
                       tf.constant(coordinates[1], tf.int32),
                       tf.constant(coordinates[2], tf.int32)))
        replica_inputs.append(inputs)

      tpu_step = tpu.replicate(device_fn,
                               inputs=replica_inputs,
                               device_assignment=device_assignment)

      replica_outputs = sess.run(tpu_step)
      actual_results = _concatenate_3d(replica_outputs, self.params.cx,
                                       self.params.cy, group_assignment_dim2)
      expected_results = np.fft.fftn(input_signal)

      self.assertAllClose(a=actual_results.real,
                          b=expected_results.real,
                          rtol=1e-04,
                          atol=1e-04)
      self.assertAllClose(a=actual_results.imag,
                          b=expected_results.imag,
                          rtol=1e-04,
                          atol=1e-04)

  def testGenSourceTargetPairsForTpuCollectivePermute(self):
    """Tests the generation of source-target pairs.

    The `group_assignment` is a list of `replica_group`s along one specific
    dimension of the 3D mesh and the soruce-target paris are created within
    each `group_assignment`. In the test, the computation shape is chosen as
    [3, 3, 3] such that
    the group assignment along dim0 is [[0, 9, 18], [3, 12, 21], [6, 15, 24],
                                        [1, 10, 19], [4, 13, 22], [7, 16, 25],
                                        [2, 11, 20], [5, 14, 23], [8, 17, 26]],
    the group assignment along dim1 is [[0, 3, 6], [9, 12, 15], [18, 21, 24],
                                        [1, 4, 7], [10, 13, 16], [19, 22, 25],
                                        [2, 5, 8], [11, 14, 17], [20, 23, 26]],
    and the group assignment along dim2 is [[0, 1, 2], [3, 4, 5], [6, 7, 8],
                                            [9, 10, 11], [12, 13, 14],
                                            [15, 16, 17],
                                            [18, 19, 20], [21, 22, 23],
                                            [24, 25, 26]].
    """
    computation_shape = [3, 3, 3]

    expected_source_target_pairs_dim0 = [(9, 0), (18, 9), (0, 18),
                                         (12, 3), (21, 12), (3, 21),
                                         (15, 6), (24, 15), (6, 24),
                                         (10, 1), (19, 10), (1, 19),
                                         (13, 4), (22, 13), (4, 22),
                                         (16, 7), (25, 16), (7, 25),
                                         (11, 2), (20, 11), (2, 20),
                                         (14, 5), (23, 14), (5, 23),
                                         (17, 8), (26, 17), (8, 26)]

    expected_source_target_pairs_dim1 = [(3, 0), (6, 3), (0, 6),
                                         (12, 9), (15, 12), (9, 15),
                                         (21, 18), (24, 21), (18, 24),
                                         (4, 1), (7, 4), (1, 7),
                                         (13, 10), (16, 13), (10, 16),
                                         (22, 19), (25, 22), (19, 25),
                                         (5, 2), (8, 5), (2, 8),
                                         (14, 11), (17, 14), (11, 17),
                                         (23, 20), (26, 23), (20, 26)]

    expected_source_target_pairs_dim2 = [(1, 0), (2, 1), (0, 2),
                                         (4, 3), (5, 4), (3, 5),
                                         (7, 6), (8, 7), (6, 8),
                                         (10, 9), (11, 10), (9, 11),
                                         (13, 12), (14, 13), (12, 14),
                                         (16, 15), (17, 16), (15, 17),
                                         (19, 18), (20, 19), (18, 20),
                                         (22, 21), (23, 22), (21, 23),
                                         (25, 24), (26, 25), (24, 26)]

    actual_source_target_pairs_dim0 = dft.gen_source_target_pairs(
        computation_shape, dft.Dimension.DIM0)
    self.assertEqual(actual_source_target_pairs_dim0,
                     expected_source_target_pairs_dim0)

    actual_source_target_pairs_dim1 = dft.gen_source_target_pairs(
        computation_shape, dft.Dimension.DIM1)
    self.assertEqual(actual_source_target_pairs_dim1,
                     expected_source_target_pairs_dim1)

    actual_source_target_pairs_dim2 = dft.gen_source_target_pairs(
        computation_shape, dft.Dimension.DIM2)
    self.assertEqual(actual_source_target_pairs_dim2,
                     expected_source_target_pairs_dim2)

  @parameterized.parameters(*itertools.product(NUM_PTS_PER_DIM_PER_CORE_DFT3D,
                                               NUM_CORES_PER_DIM_DFT3D))
  def testDft3dOneShuffle(self,
                          num_pts_per_dim_per_core,
                          num_cores_per_dim):
    self.set_params(num_pts_per_dim_per_core,
                    num_cores_per_dim)
    computation_shape = np.array(num_cores_per_dim)
    num_replicas = np.prod(computation_shape)
    mm = num_pts_per_dim_per_core[0] * computation_shape[0]
    nn = num_pts_per_dim_per_core[1] * computation_shape[1]
    ss = num_pts_per_dim_per_core[2] * computation_shape[2]

    input_signal = (np.random.normal(0, 0.1, mm * nn * ss)).reshape((mm,
                                                                     nn,
                                                                     ss))
    group_assignment_dim2 = dft.gen_group_assignment(computation_shape,
                                                     dft.Dimension.DIM2)

    def device_fn(*args):
      """Creates 3D DFT batch-mode computation that is passed to TPU replicas.

      Args:
        *args: args[0] and args[1] are 2D `Tensor`s of `tf.complex64`
          representing the Vandermonde matrices that pre- and post- multiplies
          the input `Tensor` in 2D DFT computation, respectively; args[2] is
          the 3D `Tensor` representing the input signal; args[3] is the 2D
          `Tensor` of `tf.complex64` representintg the Vandermonde matrix for
          the DFT computation along the third dimension; and args[4] contains
          the indices of the TPU core in the logical mesh [i, j, k].
      Returns:
        A 3D `Tensor` of `tf.complex64`.
      """
      a = args[2]
      vm = args[0]
      vn = args[1]
      vs = args[3]
      core_indices = args[4]
      return dft.dft_3d_slice_one_shuffle(a, vm, vn, vs, computation_shape,
                                          core_indices)

    with self.session() as sess:
      topology = tpu_topology.Topology(sess.run(tpu.initialize_system()))

      (device_assignment,
       compute_core_assignment) = util.tpu_device_assignment(computation_shape,
                                                             topology)
      replica_inputs = []
      for replica_id in range(num_replicas):
        coordinates = compute_core_assignment[replica_id, :]
        inputs = []

        inputs.append(dft_initializer.gen_vandermonde_mat(
            self.params,
            [coordinates[0], coordinates[1], 0],
            dft_initializer.PartitionDimension.DIM0,
            dft_initializer.PartitionDomain.SPECTRAL))
        inputs.append(dft_initializer.gen_vandermonde_mat(
            self.params,
            [coordinates[0], coordinates[1], 0],
            dft_initializer.PartitionDimension.DIM1,
            dft_initializer.PartitionDomain.SPECTRAL))
        inputs.append(tf.convert_to_tensor(input_signal[
            coordinates[0] * self.params.nx:
            (coordinates[0] + 1) * self.params.nx,
            coordinates[1] * self.params.ny:
            (coordinates[1] + 1) * self.params.ny,
            coordinates[2] * self.params.nz:
            (coordinates[2] + 1) * self.params.nz], dtype=tf.complex64))
        inputs.append(dft_initializer.gen_vandermonde_mat(
            self.params, coordinates,
            dft_initializer.PartitionDimension.DIM2,
            dft_initializer.PartitionDomain.SPECTRAL))
        inputs.append((tf.constant(coordinates[0], tf.int32),
                       tf.constant(coordinates[1], tf.int32),
                       tf.constant(coordinates[2], tf.int32)))
        replica_inputs.append(inputs)

      tpu_step = tpu.replicate(device_fn,
                               inputs=replica_inputs,
                               device_assignment=device_assignment)

      replica_outputs = sess.run(tpu_step)
      actual_results = _concatenate_3d(replica_outputs, self.params.cx,
                                       self.params.cy, group_assignment_dim2)
      expected_results = np.fft.fftn(input_signal)

      self.assertAllClose(a=actual_results.real,
                          b=expected_results.real,
                          rtol=1e-04,
                          atol=1e-04)
      self.assertAllClose(a=actual_results.imag,
                          b=expected_results.imag,
                          rtol=1e-04,
                          atol=1e-04)


if __name__ == '__main__':
  tf.test.main()
