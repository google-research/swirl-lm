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
"""Tests for get_kernel_fn.

To update the test data, run on blaze with the `update_test_data` flag:
blaze test -c opt <path_to_test> --test_arg=--update_test_data
then go to the "Output Files" tab in sponge to download the new test data.
"""
# pylint: disable=bad-whitespace

import itertools
import os

from absl import flags
import numpy as np
from swirl_lm.utility import get_kernel_fn
# For grid related flags.
from swirl_lm.utility import grid_parametrization  # pylint: disable=unused-import
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.pyglib import gfile
from google3.pyglib import resources
from google3.testing.pybase import parameterized


flags.DEFINE_bool('update_testdata', False, 'If true, update test data.')

FLAGS = flags.FLAGS

_TESTDATA_DIR = 'google3/third_party/py/swirl_lm/utility/testdata'


@test_util.run_all_in_graph_and_eager_modes
class ApplyKernelOpTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(*itertools.product((
      'kS{}',
      'ks{}',
      'kD{}',
      'kD4{}',
      'kd{}',
      'kd{}+',
      'kdd{}',
      'kdd8{}',
      'kdd16{}',
      'kf2{}',
      'kf2{}+',
      'k3d1{}+',
      'k4d2{}',
  ), (2, 3, 4, 10)))
  def testmulop_kernel_dict_symmetry(self, template_name, kernel_size):
    """Tests `_mulop_kernel_dict`'s symmetry, y should be x transpose."""
    kernel_dict = (
        get_kernel_fn.ApplyKernelMulOp(kernel_size, kernel_size)._kernels)

    name_x = template_name.format('x')
    name_y = template_name.format('y')

    self.assertAllEqual(kernel_dict[name_x], np.transpose(kernel_dict[name_y]))


@test_util.run_all_in_graph_and_eager_modes
class GetKernelFnTest(tf.test.TestCase, parameterized.TestCase):

  # TODO(yusef): Reconsider systematic ways for selecting test data.
  KERNEL_SIZES = [2, 4, 8, 8, 8]
  KERNEL_OP = [
      lambda ks, _, __, kd=None: get_kernel_fn.ApplyKernelConvOp(ks, kd),
      lambda ks, _, __, kd=None: get_kernel_fn.ApplyKernelConvOp(ks, kd),
      lambda ks, _, __, kd=None: get_kernel_fn.ApplyKernelConvOp(ks, kd),
      lambda _, nx, ny, kd=None: get_kernel_fn.ApplyKernelMulOp(nx, ny, kd),
      lambda _, __, ___, kd=None: get_kernel_fn.ApplyKernelSliceOp(kd)
  ]

  def setUp(self):
    super(GetKernelFnTest, self).setUp()
    FLAGS.nx = 16
    FLAGS.ny = 8
    FLAGS.nz = 1
    self.tile = tf.constant([[ 1,  2,  3,  4,  5,   6,   7,   8],
                             [11, 12, 13, 14, 15,  16,  17,  18],
                             [25, 24, 23, 22, 21,  20,  19,  18],
                             [90, 92, 94, 96, 98, 100, 102, 104],
                             [ 0,  3,  6,  9, 12,  15,  18,  21],
                             [ 2,  4,  6,  8, 12,  14,  16,  18],
                             [10, 20, 30, 40, 50,  60,  70,  80],
                             [ 5,  5,  5,  5,  5,   5,   5,   5],
                             [31, 32, 33, 34, 35,  36,  37,  38],
                             [44, 43, 42, 41, 39,  38,  37,  36],
                             [ 0,  0,  0,  0,  0,   0,   0,   0],
                             [ 1,  1,  2,  2,  3,   3,   4,   4],
                             [ 7,  7,  7,  8,  8,   8,   9,   9],
                             [ 5,  5,  5,  5,  6,   6,   6,   6],
                             [ 0,  1,  2,  1,  0,  -1,  -2,  -3],
                             [ 9,  9,  9,  9,  9,   9,   9,   9]],
                            tf.float32)

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpDx(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    kdx = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_x([self.tile], 'kDx'))
    self.assertLen(kdx, 1)
    expected = [[11, 12, 13, 14, 15, 16, 17, 18],
                [24, 22, 20, 18, 16, 14, 12, 10],
                [79, 80, 81, 82, 83, 84, 85, 86],
                [-25, -21, -17, -13, -9, -5, -1, 3],
                [-88, -88, -88, -88, -86, -86, -86, -86],
                [10, 17, 24, 31, 38, 45, 52, 59],
                [3, 1, -1, -3, -7, -9, -11, -13],
                [21, 12, 3, -6, -15, -24, -33, -42],
                [39, 38, 37, 36, 34, 33, 32, 31],
                [-31, -32, -33, -34, -35, -36, -37, -38],
                [-43, -42, -40, -39, -36, -35, -33, -32],
                [7, 7, 7, 8, 8, 8, 9, 9], [4, 4, 3, 3, 3, 3, 2, 2],
                [-7, -6, -5, -7, -8, -9, -11, -12], [4, 4, 4, 4, 3, 3, 3, 3],
                [0, -1, -2, -1, 0, 1, 2, 3]]
    self.assertAllEqual(expected, kdx[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpdx(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    kdx = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_x([self.tile], 'kdx'))
    self.assertLen(kdx, 1)
    expected = [[1, 2, 3, 4, 5, 6, 7, 8], [10, 10, 10, 10, 10, 10, 10, 10],
                [14, 12, 10, 8, 6, 4, 2, 0], [65, 68, 71, 74, 77, 80, 83, 86],
                [-90, -89, -88, -87, -86, -85, -84, -83],
                [2, 1, 0, -1, 0, -1, -2, -3], [8, 16, 24, 32, 38, 46, 54, 62],
                [-5, -15, -25, -35, -45, -55, -65, -75],
                [26, 27, 28, 29, 30, 31, 32, 33], [13, 11, 9, 7, 4, 2, 0, -2],
                [-44, -43, -42, -41, -39, -38, -37, -36],
                [1, 1, 2, 2, 3, 3, 4, 4], [6, 6, 5, 6, 5, 5, 5, 5],
                [-2, -2, -2, -3, -2, -2, -3, -3],
                [-5, -4, -3, -4, -6, -7, -8, -9], [9, 8, 7, 8, 9, 10, 11, 12]]
    self.assertAllEqual(expected, kdx[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpDxPlus(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    kdx = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_x([self.tile], 'kdx+'))
    self.assertLen(kdx, 1)
    expected = [[10, 10, 10, 10, 10, 10, 10, 10], [14, 12, 10, 8, 6, 4, 2, 0],
                [65, 68, 71, 74, 77, 80, 83, 86],
                [-90, -89, -88, -87, -86, -85, -84, -83],
                [2, 1, 0, -1, 0, -1, -2, -3], [8, 16, 24, 32, 38, 46, 54, 62],
                [-5, -15, -25, -35, -45, -55, -65, -75],
                [26, 27, 28, 29, 30, 31, 32, 33], [13, 11, 9, 7, 4, 2, 0, -2],
                [-44, -43, -42, -41, -39, -38, -37, -36],
                [1, 1, 2, 2, 3, 3, 4, 4], [6, 6, 5, 6, 5, 5, 5, 5],
                [-2, -2, -2, -3, -2, -2, -3, -3],
                [-5, -4, -3, -4, -6, -7, -8, -9], [9, 8, 7, 8, 9, 10, 11, 12],
                [-9, -9, -9, -9, -9, -9, -9, -9]]
    self.assertAllEqual(expected, kdx[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpDy(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    kdy = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_y([self.tile], 'kDy'))
    self.assertLen(kdy, 1)
    expetced = [[2, 2, 2, 2, 2, 2, 2, -7], [12, 2, 2, 2, 2, 2, 2, -17],
                [24, -2, -2, -2, -2, -2, -2, -19], [92, 4, 4, 4, 4, 4, 4, -102],
                [3, 6, 6, 6, 6, 6, 6, -18], [4, 4, 4, 6, 6, 4, 4, -16],
                [20, 20, 20, 20, 20, 20, 20, -70], [5, 0, 0, 0, 0, 0, 0, -5],
                [32, 2, 2, 2, 2, 2, 2, -37], [43, -2, -2, -3, -3, -2, -2, -37],
                [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, -4],
                [7, 0, 1, 1, 0, 1, 1, -9], [5, 0, 0, 1, 1, 0, 0, -6],
                [1, 2, 0, -2, -2, -2, -2, 2], [9, 0, 0, 0, 0, 0, 0, -9]]
    self.assertAllEqual(expetced, kdy[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpdy(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    kdy = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_y([self.tile], 'kdy'))
    self.assertLen(kdy, 1)
    expected = [[1, 1, 1, 1, 1, 1, 1, 1], [11, 1, 1, 1, 1, 1, 1, 1],
                [25, -1, -1, -1, -1, -1, -1, -1], [90, 2, 2, 2, 2, 2, 2, 2],
                [0, 3, 3, 3, 3, 3, 3, 3], [2, 2, 2, 2, 4, 2, 2, 2],
                [10, 10, 10, 10, 10, 10, 10, 10], [5, 0, 0, 0, 0, 0, 0, 0],
                [31, 1, 1, 1, 1, 1, 1, 1], [44, -1, -1, -1, -2, -1, -1, -1],
                [0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0],
                [7, 0, 0, 1, 0, 0, 1, 0], [5, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, -1, -1, -1, -1, -1], [9, 0, 0, 0, 0, 0, 0, 0]]
    self.assertAllEqual(expected, kdy[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpDyPlus(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    kdy = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_y([self.tile], 'kdy+'))
    self.assertLen(kdy, 1)
    expected = [[1, 1, 1, 1, 1, 1, 1, -8], [1, 1, 1, 1, 1, 1, 1, -18],
                [-1, -1, -1, -1, -1, -1, -1, -18], [2, 2, 2, 2, 2, 2, 2, -104],
                [3, 3, 3, 3, 3, 3, 3, -21], [2, 2, 2, 4, 2, 2, 2, -18],
                [10, 10, 10, 10, 10, 10, 10, -80], [0, 0, 0, 0, 0, 0, 0, -5],
                [1, 1, 1, 1, 1, 1, 1, -38], [-1, -1, -1, -2, -1, -1, -1, -36],
                [0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0, -4],
                [0, 0, 1, 0, 0, 1, 0, -9], [0, 0, 0, 1, 0, 0, 0, -6],
                [1, 1, -1, -1, -1, -1, -1, 3], [0, 0, 0, 0, 0, 0, 0, -9]]
    self.assertAllEqual(expected, kdy[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpddx(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    kddx = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_x([self.tile], 'kddx'))
    self.assertLen(kddx, 1)
    expected = [[9, 8, 7, 6, 5, 4, 3, 2], [4, 2, 0, -2, -4, -6, -8, -10],
                [51, 56, 61, 66, 71, 76, 81, 86],
                [-155, -157, -159, -161, -163, -165, -167, -169],
                [92, 90, 88, 86, 86, 84, 82, 80],
                [6, 15, 24, 33, 38, 47, 56, 65],
                [-13, -31, -49, -67, -83, -101, -119, -137],
                [31, 42, 53, 64, 75, 86, 97, 108],
                [-13, -16, -19, -22, -26, -29, -32, -35],
                [-57, -54, -51, -48, -43, -40, -37, -34],
                [45, 44, 44, 43, 42, 41, 41, 40], [5, 5, 3, 4, 2, 2, 1, 1],
                [-8, -8, -7, -9, -7, -7, -8, -8],
                [-3, -2, -1, -1, -4, -5, -5, -6],
                [14, 12, 10, 12, 15, 17, 19, 21],
                [-18, -17, -16, -17, -18, -19, -20, -21]]
    self.assertAllEqual(expected, kddx[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpddy(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    kddy = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_y([self.tile], 'kddy'))
    self.assertLen(kddy, 1)
    expected = [[0, 0, 0, 0, 0, 0, 0, -9], [-10, 0, 0, 0, 0, 0, 0, -19],
                [-26, 0, 0, 0, 0, 0, 0, -17], [-88, 0, 0, 0, 0, 0, 0, -106],
                [3, 0, 0, 0, 0, 0, 0, -24], [0, 0, 0, 2, -2, 0, 0, -20],
                [0, 0, 0, 0, 0, 0, 0, -90], [-5, 0, 0, 0, 0, 0, 0, -5],
                [-30, 0, 0, 0, 0, 0, 0, -39], [-45, 0, 0, -1, 1, 0, 0, -35],
                [0, 0, 0, 0, 0, 0, 0, 0], [-1, 1, -1, 1, -1, 1, -1, -4],
                [-7, 0, 1, -1, 0, 1, -1, -9], [-5, 0, 0, 1, -1, 0, 0, -6],
                [1, 0, -2, 0, 0, 0, 0, 4], [-9, 0, 0, 0, 0, 0, 0, -9]]
    self.assertAllEqual(expected, kddy[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpSx(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    ksx = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_x([self.tile], 'kSx'))
    self.assertLen(ksx, 1)
    expected = [[11, 12, 13, 14, 15, 16, 17, 18],
                [26, 26, 26, 26, 26, 26, 26, 26],
                [101, 104, 107, 110, 113, 116, 119, 122],
                [25, 27, 29, 31, 33, 35, 37, 39],
                [92, 96, 100, 104, 110, 114, 118, 122],
                [10, 23, 36, 49, 62, 75, 88, 101],
                [7, 9, 11, 13, 17, 19, 21, 23],
                [41, 52, 63, 74, 85, 96, 107, 118],
                [49, 48, 47, 46, 44, 43, 42, 41],
                [31, 32, 33, 34, 35, 36, 37, 38],
                [45, 44, 44, 43, 42, 41, 41, 40], [7, 7, 7, 8, 8, 8, 9, 9],
                [6, 6, 7, 7, 9, 9, 10, 10], [7, 8, 9, 9, 8, 7, 7, 6],
                [14, 14, 14, 14, 15, 15, 15, 15], [0, 1, 2, 1, 0, -1, -2, -3]]
    self.assertAllEqual(expected, ksx[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpSy(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    ksy = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_y([self.tile], 'kSy'))
    self.assertLen(ksy, 1)
    expected = [[2, 4, 6, 8, 10, 12, 14, 7], [12, 24, 26, 28, 30, 32, 34, 17],
                [24, 48, 46, 44, 42, 40, 38, 19],
                [92, 184, 188, 192, 196, 200, 204, 102],
                [3, 6, 12, 18, 24, 30, 36, 18], [4, 8, 12, 18, 22, 28, 32, 16],
                [20, 40, 60, 80, 100, 120, 140, 70],
                [5, 10, 10, 10, 10, 10, 10, 5],
                [32, 64, 66, 68, 70, 72, 74, 37],
                [43, 86, 84, 81, 79, 76, 74, 37], [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 3, 3, 5, 5, 7, 7, 4], [7, 14, 15, 15, 16, 17, 17, 9],
                [5, 10, 10, 11, 11, 12, 12, 6], [1, 2, 2, 2, 0, -2, -4, -2],
                [9, 18, 18, 18, 18, 18, 18, 9]]
    self.assertAllEqual(expected, ksy[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpsx(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    ksx = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_x([self.tile], 'ksx'))
    self.assertLen(ksx, 1)
    expected = [[1., 2., 3., 4., 5., 6., 7., 8.],
                [12., 14., 16., 18., 20., 22., 24., 26.],
                [36., 36., 36., 36., 36., 36., 36., 36.],
                [115., 116., 117., 118., 119., 120., 121., 122.],
                [90., 95., 100., 105., 110., 115., 120., 125.],
                [2., 7., 12., 17., 24., 29., 34., 39.],
                [12., 24., 36., 48., 62., 74., 86., 98.],
                [15., 25., 35., 45., 55., 65., 75., 85.],
                [36., 37., 38., 39., 40., 41., 42., 43.],
                [75., 75., 75., 75., 74., 74., 74., 74.],
                [44., 43., 42., 41., 39., 38., 37., 36.],
                [1., 1., 2., 2., 3., 3., 4., 4.],
                [8., 8., 9., 10., 11., 11., 13., 13.],
                [12., 12., 12., 13., 14., 14., 15., 15.],
                [5., 6., 7., 6., 6., 5., 4., 3.],
                [9., 10., 11., 10., 9., 8., 7., 6.]]
    self.assertAllEqual(expected, ksx[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpsy(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    ksy = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_y([self.tile], 'ksy'))
    self.assertLen(ksy, 1)
    expected = [[1., 3., 5., 7., 9., 11., 13., 15.],
                [11., 23., 25., 27., 29., 31., 33., 35.],
                [25., 49., 47., 45., 43., 41., 39., 37.],
                [90., 182., 186., 190., 194., 198., 202., 206.],
                [0., 3., 9., 15., 21., 27., 33., 39.],
                [2., 6., 10., 14., 20., 26., 30., 34.],
                [10., 30., 50., 70., 90., 110., 130., 150.],
                [5., 10., 10., 10., 10., 10., 10., 10.],
                [31., 63., 65., 67., 69., 71., 73., 75.],
                [44., 87., 85., 83., 80., 77., 75., 73.],
                [0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 2., 3., 4., 5., 6., 7., 8.],
                [7., 14., 14., 15., 16., 16., 17., 18.],
                [5., 10., 10., 10., 11., 12., 12., 12.],
                [0., 1., 3., 3., 1., -1., -3., -5.],
                [9., 18., 18., 18., 18., 18., 18., 18.]]
    self.assertAllEqual(expected, ksy[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpf2x(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    kf2x = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_x([self.tile], 'kf2x'))
    self.assertLen(kf2x, 1)
    expected = [[4.875, 6., 7.125, 8.25, 9.375, 10.5, 11.625, 12.75],
                [17.5, 17.75, 18., 18.25, 18.5, 18.75, 19., 19.25],
                [51.125, 51., 50.875, 50.75, 50.625, 50.5, 50.375, 50.25],
                [
                    64.375, 67.125, 69.875, 72.625, 75.375, 78.125, 80.875,
                    83.625
                ], [-10.5, -7.75, -5., -2.25, 1.25, 4., 6.75, 9.5],
                [5.25, 10.125, 15., 19.875, 26.25, 31.125, 36., 40.875],
                [9.125, 16.375, 23.625, 30.875, 37.875, 45.125, 52.375, 59.625],
                [14.125, 13.25, 12.375, 11.5, 10.625, 9.75, 8.875, 8.],
                [39.125, 39.5, 39.875, 40.25, 40.25, 40.625, 41., 41.375],
                [29.125, 28.25, 27.375, 26.5, 24.875, 24., 23.125, 22.25],
                [-5.125, -5., -4.5, -4.375, -3.75, -3.625, -3.125, -3.],
                [3.375, 3.375, 4.125, 4.5, 5.25, 5.25, 6.375, 6.375],
                [7., 7., 6.875, 7.625, 7.875, 7.875, 8.5, 8.5],
                [2.875, 3.25, 3.625, 3.125, 3.5, 3.125, 2.625, 2.25],
                [2.75, 3.5, 4.25, 3.5, 2.625, 1.875, 1.125, 0.375],
                [6.75, 6.625, 6.5, 6.625, 6.75, 6.875, 7., 7.125]]
    self.assertAllEqual(expected, kf2x[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpf2y(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    kf2y = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_y([self.tile], 'kf2y'))
    self.assertLen(kf2y, 1)
    expected = [[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 5.125],
                [12.75, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 11.375],
                [27.75, 23.5, 22.5, 21.5, 20.5, 19.5, 18.5, 11.125],
                [102., 93., 95., 97., 99., 101., 103., 65.25],
                [1.125, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 13.5],
                [3., 5., 7., 9.75, 13.25, 15., 17., 11.5],
                [15., 25., 35., 45., 55., 65., 75., 51.25],
                [5.625, 5., 5., 5., 5., 5., 5., 3.125],
                [35.25, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 23.875],
                [49.125, 42.5, 41.5, 40.125, 38.375, 37.5, 36.5, 22.375],
                [0., 0., 0., 0., 0., 0., 0., 0.],
                [1.125, 1.375, 2.125, 2.375, 3.125, 3.375, 4.125, 2.5],
                [7.875, 7., 7.375, 8.125, 8., 8.375, 9.125, 5.625],
                [5.625, 5., 5., 5.375, 6.125, 6., 6., 3.75],
                [0.375, 1.5, 1.75, 0.5, -0.5, -1.5, -2.5, -2.],
                [10.125, 9., 9., 9., 9., 9., 9., 5.625]]
    self.assertAllEqual(expected, kf2y[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpf2xPlus(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    kf2x_plus = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_x([self.tile], 'kf2x+'))
    self.assertLen(kf2x_plus, 1)
    expected = [[-0.625, 0., 0.625, 1.25, 1.875, 2.5, 3.125, 3.75],
                [5.5, 6.75, 8., 9.25, 10.5, 11.75, 13., 14.25],
                [11.625, 11., 10.375, 9.75, 9.125, 8.5, 7.875, 7.25],
                [
                    76.875, 77.625, 78.375, 79.125, 79.875, 80.625, 81.375,
                    82.125
                ], [33.5, 36.25, 39., 41.75, 44.25, 47., 49.75, 52.5],
                [0.25, 1.625, 3., 4.375, 7.25, 8.625, 10., 11.375],
                [7.625, 15.875, 24.125, 32.375, 41.375, 49.625, 57.875, 66.125],
                [3.625, 7.25, 10.875, 14.5, 18.125, 21.75, 25.375, 29.],
                [19.625, 20.5, 21.375, 22.25, 23.25, 24.125, 25., 25.875],
                [44.625, 44.25, 43.875, 43.5, 42.375, 42., 41.625, 41.25],
                [16.375, 16., 15.5, 15.125, 14.25, 13.875, 13.375, 13.],
                [-0.125, -0.125, 0.625, 0.5, 1.25, 1.25, 1.875, 1.875],
                [5., 5., 5.375, 6.125, 6.375, 6.375, 7.5, 7.5],
                [6.375, 6.25, 6.125, 6.625, 7.5, 7.625, 8.125, 8.25],
                [0.75, 1.5, 2.25, 1.5, 1.125, 0.375, -0.375, -1.125],
                [6.75, 7.125, 7.5, 7.125, 6.75, 6.375, 6., 5.625]]

    self.assertAllEqual(expected, kf2x_plus[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpf2yPlus(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    kf2y_plus = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_y([self.tile], 'kf2y+'))
    self.assertLen(kf2y_plus, 1)
    expected = [[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.625],
                [6.75, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 19.875],
                [15.75, 24.5, 23.5, 22.5, 21.5, 20.5, 19.5, 20.625],
                [56., 91., 93., 95., 97., 99., 101., 116.25],
                [-0.375, 1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 22.5],
                [1., 3., 5., 6.75, 10.25, 13., 15., 19.5],
                [5., 15., 25., 35., 45., 55., 65., 86.25],
                [3.125, 5., 5., 5., 5., 5., 5., 5.625],
                [19.25, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 42.375],
                [27.625, 43.5, 42.5, 41.625, 39.875, 38.5, 37.5, 40.875],
                [0., 0., 0., 0., 0., 0., 0., 0.],
                [0.625, 0.875, 1.625, 1.875, 2.625, 2.875, 3.625, 4.5],
                [4.375, 7., 6.875, 7.625, 8., 7.875, 8.625, 10.125],
                [3.125, 5., 5., 4.875, 5.625, 6., 6., 6.75],
                [-0.125, 0.5, 1.75, 1.5, 0.5, -0.5, -1.5, -3.],
                [5.625, 9., 9., 9., 9., 9., 9., 10.125]]
    self.assertAllEqual(expected, kf2y_plus[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpd4x(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    k3d1x_plus = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_x([self.tile], 'k3d1x+'))
    self.assertLen(k3d1x_plus, 1)
    expected = [[8., 6., 4., 2., -0., -2., -4., -6.],
                [-5., -6., -7., -8., -9., -10., -11., -12.],
                [47., 54., 61., 68., 75., 82., 89., 96.],
                [-206., -213., -220., -227., -234., -241., -248., -255.],
                [247., 247., 247., 247., 249., 249., 249., 249.],
                [-86., -75., -64., -53., -48., -37., -26., -15.],
                [-19., -46., -73., -100., -121., -148., -175., -202.],
                [44., 73., 102., 131., 158., 187., 216., 245.],
                [-44., -58., -72., -86., -101., -115., -129., -143.],
                [-44., -38., -32., -26., -17., -11., -5., 1.],
                [102., 98., 95., 91., 85., 81., 78., 74.],
                [-40., -39., -41., -39., -40., -39., -40., -39.],
                [-13., -13., -10., -13., -9., -9., -9., -9.],
                [5., 6., 6., 8., 3., 2., 3., 2.],
                [17., 14., 11., 13., 19., 22., 24., 27.],
                [-32., -29., -26., -29., -33., -36., -39., -42.]]
    self.assertAllEqual(expected, k3d1x_plus[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpd4y(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    k3d1y_plus = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx,
                  FLAGS.ny).apply_kernel_op_y([self.tile], 'k3d1y+'))
    self.assertLen(k3d1y_plus, 1)
    expected = [[-1., -0., -0., -0., -0., -0., -0., -9.],
                [-21., 10., -0., -0., -0., -0., -0., -19.],
                [-51., 26., -0., -0., -0., -0., -0., -17.],
                [-178., 88., -0., -0., -0., -0., -0., -106.],
                [3., -3., -0., -0., -0., -0., -0., -24.],
                [-2., -0., -0., 2., -4., 2., -0., -20.],
                [-10., -0., -0., -0., -0., -0., -0., -90.],
                [-10., 5., -0., -0., -0., -0., -0., -5.],
                [-61., 30., -0., -0., -0., -0., -0., -39.],
                [-89., 45., -0., -1., 2., -1., -0., -35.],
                [-0., -0., -0., -0., -0., -0., -0., -0.],
                [-2., 2., -2., 2., -2., 2., -2., -3.],
                [-14., 7., 1., -2., 1., 1., -2., -8.],
                [-10., 5., -0., 1., -2., 1., -0., -6.],
                [1., -1., -2., 2., -0., -0., -0., 4.],
                [-18., 9., -0., -0., -0., -0., -0., -9.]]
    self.assertAllEqual(expected, k3d1y_plus[0])

  @parameterized.parameters(
      [get_kernel_fn.ApplyKernelConvOp(2),
       get_kernel_fn.ApplyKernelMulOp(2, 2),
       get_kernel_fn.ApplyKernelSliceOp()])
  def testKernelZ(self, kernel_op):
    tile_0 = tf.constant([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [2, 3, 4],
                          [5, 6, 7]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30],
                          [40, 50, 60],
                          [70, 80, 90],
                          [20, 30, 40],
                          [50, 60, 70]], tf.float32)
    tile_2 = tf.constant([[11, 22, 33],
                          [44, 55, 66],
                          [77, 88, 99],
                          [22, 33, 44],
                          [55, 66, 77]], tf.float32)
    tile_3 = tf.constant([[10, 40, 70],
                          [20, 50, 80],
                          [30, 60, 90],
                          [40, 70,  0],
                          [50, 80, 20]], tf.float32)
    tiles = [tile_0, tile_1, tile_2, tile_3]

    k_sz = self.evaluate(kernel_op.apply_kernel_op_z(tiles, 'kSz', 'kSzsh'))
    self.assertLen(k_sz, 4)
    self.assertAllEqual([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 6, 7]],
                        k_sz[0])
    self.assertAllEqual(
        [[12, 24, 36], [48, 60, 72], [84, 96, 108], [24, 36, 48], [60, 72, 84]],
        k_sz[1])
    self.assertAllEqual([[20, 60, 100], [60, 100, 140], [100, 140, 180],
                         [60, 100, 40], [100, 140, 90]], k_sz[2])
    self.assertAllEqual(
        [[10, 40, 70], [20, 50, 80], [30, 60, 90], [40, 70, 0], [50, 80, 20]],
        k_sz[3])

    k_sz_back = self.evaluate(
        kernel_op.apply_kernel_op_z(tiles, 'ksz', 'kszsh'))
    self.assertLen(k_sz_back, 4)
    self.assertAllEqual([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 6, 7]],
                        k_sz_back[0])
    self.assertAllEqual([[11., 22., 33.], [44., 55., 66.], [77., 88., 99.],
                         [22., 33., 44.], [55., 66., 77.]], k_sz_back[1])
    self.assertAllEqual([[21., 42., 63.], [84., 105., 126.], [147., 168., 189.],
                         [42., 63., 84.], [105., 126., 147.]], k_sz_back[2])
    self.assertAllEqual(
        [[21., 62., 103.], [64., 105., 146.], [107., 148., 189.],
         [62., 103., 44.], [105., 146., 97.]], k_sz_back[3])

    k_dz = self.evaluate(kernel_op.apply_kernel_op_z(tiles, 'kDz', 'kDzsh'))
    self.assertLen(k_dz, 4)
    self.assertAllEqual([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 6, 7]],
                        k_dz[0])
    self.assertAllEqual(
        [[10, 20, 30], [40, 50, 60], [70, 80, 90], [20, 30, 40], [50, 60, 70]],
        k_dz[1])
    self.assertAllEqual(
        [[0, 20, 40], [-20, 0, 20], [-40, -20, 0], [20, 40, -40], [0, 20, -50]],
        k_dz[2])
    self.assertAllEqual(
        [[10, 40, 70], [20, 50, 80], [30, 60, 90], [40, 70, 0], [50, 80, 20]],
        k_dz[3])
    kdz = self.evaluate(kernel_op.apply_kernel_op_z(tiles, 'kdz+', 'kdz+sh'))
    self.assertLen(kdz, 4)
    self.assertAllEqual(
        [[9, 18, 27], [36, 45, 54], [63, 72, 81], [18, 27, 36], [45, 54, 63]],
        kdz[0])
    self.assertAllEqual([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 6, 7]],
                        kdz[1])
    self.assertAllEqual([[-1, 18, 37], [-24, -5, 14], [-47, -28, -9],
                         [18, 37, -44], [-5, 14, -57]], kdz[2])
    self.assertAllEqual(
        [[10, 40, 70], [20, 50, 80], [30, 60, 90], [40, 70, 0], [50, 80, 20]],
        kdz[3])
    kdz = self.evaluate(kernel_op.apply_kernel_op_z(tiles, 'kdz', 'kdzsh'))
    self.assertLen(kdz, 4)
    self.assertAllEqual([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 6, 7]],
                        kdz[0])
    self.assertAllEqual(
        [[9, 18, 27], [36, 45, 54], [63, 72, 81], [18, 27, 36], [45, 54, 63]],
        kdz[1])
    self.assertAllEqual([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 6, 7]],
                        kdz[2])
    self.assertAllEqual([[-1, 18, 37], [-24, -5, 14], [-47, -28, -9],
                         [18, 37, -44], [-5, 14, -57]], kdz[3])
    kddz = self.evaluate(kernel_op.apply_kernel_op_z(tiles, 'kddz', 'kddzsh'))
    self.assertLen(kddz, 4)
    self.assertAllEqual([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 6, 7]],
                        kddz[0])
    self.assertAllEqual([[-8, -16, -24], [-32, -40, -48], [-56, -64, -72],
                         [-16, -24, -32], [-40, -48, -56]], kddz[1])
    self.assertAllEqual([[-2, 16, 34], [-28, -10, 8], [-54, -36, -18],
                         [16, 34, -48], [-10, 8, -64]], kddz[2])
    self.assertAllEqual(
        [[10, 40, 70], [20, 50, 80], [30, 60, 90], [40, 70, 0], [50, 80, 20]],
        kddz[3])
    kf2z = self.evaluate(kernel_op.apply_kernel_op_z(tiles, 'kf2z', 'kf2zsh'))
    self.assertLen(kf2z, 4)
    self.assertAllEqual(tile_0, kf2z[0])
    self.assertAllEqual(
        [[11.5, 23., 34.5], [46., 57.5, 69.], [80.5, 92., 103.5],
         [23., 34.5, 46.], [57.5, 69., 80.5]], kf2z[1])
    self.assertAllEqual(
        [[10.75, 29., 47.25], [35.5, 53.75, 72.], [60.25, 78.5, 96.75],
         [29., 47.25, 28.], [53.75, 72., 56.5]], kf2z[2])
    self.assertAllEqual(tile_3, kf2z[3])

    kf2z_plus = self.evaluate(
        kernel_op.apply_kernel_op_z(tiles, 'kf2z+', 'kf2z+sh'))
    self.assertLen(kf2z_plus, 4)
    self.assertAllEqual(tile_0, kf2z_plus[0])
    self.assertAllEqual([[6.5, 13., 19.5], [26., 32.5, 39.], [45.5, 52., 58.5],
                         [13., 19.5, 26.], [32.5, 39., 45.5]], kf2z_plus[1])
    self.assertAllEqual(
        [[10.75, 19., 27.25], [45.5, 53.75, 62.], [80.25, 88.5, 96.75],
         [19., 27.25, 48.], [53.75, 62., 81.5]], kf2z_plus[2])
    self.assertAllEqual(tile_3, kf2z_plus[3])

    kd4z_plus = self.evaluate(
        kernel_op.apply_kernel_op_z(tiles, 'k3d1z+', 'k3d1z+sh'))
    self.assertLen(kd4z_plus, 4)
    self.assertAllEqual(tile_0, kd4z_plus[0])
    self.assertAllEqual(tile_1, kd4z_plus[1])
    self.assertAllEqual([[6., 32., 58.], [4., 30., 56.], [2., 28., 54.],
                         [32., 58., -16.], [30., 56., -8.]], kd4z_plus[2])
    self.assertAllEqual(tile_3, kd4z_plus[3])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testExternallyDefinedKernelMatchesPrescribedKernels(
      self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    external_kernel_dict = {'kDExt': ([-1.0, 0.0, 1.0], 1)}
    kernel =  kernel_op(kernel_size, FLAGS.nx, FLAGS.ny, external_kernel_dict)

    tile_0 = tf.constant([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [2, 3, 4],
                          [5, 6, 7]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30],
                          [40, 50, 60],
                          [70, 80, 90],
                          [20, 30, 40],
                          [50, 60, 70]], tf.float32)
    tile_2 = tf.constant([[11, 22, 33],
                          [44, 55, 66],
                          [77, 88, 99],
                          [22, 33, 44],
                          [55, 66, 77]], tf.float32)
    tile_3 = tf.constant([[10, 40, 70],
                          [20, 50, 80],
                          [30, 60, 90],
                          [40, 70,  0],
                          [50, 80, 20]], tf.float32)
    tiles = [tile_0, tile_1, tile_2, tile_3]

    with self.subTest(name='CentralDifferenceInX'):
      kd = self.evaluate(kernel.apply_kernel_op_x([self.tile], 'kDx'))
      kd_ext = self.evaluate(kernel.apply_kernel_op_x([self.tile], 'kDExtx'))
      self.assertLen(kd_ext, 1)
      self.assertAllEqual(kd[0], kd_ext[0])

    with self.subTest(name='CentralDifferenceInY'):
      kd = self.evaluate(kernel.apply_kernel_op_y([self.tile], 'kDy'))
      kd_ext = self.evaluate(kernel.apply_kernel_op_y([self.tile], 'kDExty'))
      self.assertLen(kd_ext, 1)
      self.assertAllEqual(kd[0], kd_ext[0])

    with self.subTest(name='CentralDifferenceInZ'):
      kd = self.evaluate(kernel.apply_kernel_op_z(tiles, 'kDz', 'kDzsh'))
      kd_ext = self.evaluate(
          kernel.apply_kernel_op_z(tiles, 'kDExtz', 'kDExtzsh'))
      self.assertLen(kd_ext, 4)
      for i in range(4):
        self.assertAllEqual(kd[i], kd_ext[i])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testExternallyDefinedBackwardKernelMatchesPrescribedKernels(
      self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    external_kernel_dict = {'k3d1Ext': ([-1.0, 3.0, -3.0, 1.0], 2)}
    kernel =  kernel_op(kernel_size, FLAGS.nx, FLAGS.ny, external_kernel_dict)

    tiles = [
        self.tile, -1.0 * self.tile, 2.0 * self.tile, -3.0 * self.tile,
        4.0 * self.tile, -5.0 * self.tile
    ]

    with self.subTest(name='ThirdOrderDifferenceInX'):
      kd = self.evaluate(kernel.apply_kernel_op_x([self.tile], 'k3d1x+'))
      kd_ext = self.evaluate(kernel.apply_kernel_op_x([self.tile], 'k3d1Extx'))
      self.assertLen(kd_ext, 1)
      self.assertAllEqual(kd[0], kd_ext[0])

    with self.subTest(name='ThirdOrderDifferenceInY'):
      kd = self.evaluate(kernel.apply_kernel_op_y([self.tile], 'k3d1y+'))
      kd_ext = self.evaluate(kernel.apply_kernel_op_y([self.tile], 'k3d1Exty'))
      self.assertLen(kd_ext, 1)
      self.assertAllEqual(kd[0], kd_ext[0])

    with self.subTest(name='ThirdOrderDifferenceInZ'):
      kd = self.evaluate(kernel.apply_kernel_op_z(tiles, 'k3d1z+', 'k3d1z+sh'))
      kd_ext = self.evaluate(
          kernel.apply_kernel_op_z(tiles, 'k3d1Extz', 'k3d1Extzsh'))
      self.assertLen(kd_ext, 6)
      for i in range(6):
        self.assertAllEqual(kd[i], kd_ext[i])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testExternallyAddedBackwardKernelMatchesPrescribedKernels(
      self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    external_kernel_dict = {'k3d1Ext': ([-1.0, 3.0, -3.0, 1.0], 2)}
    kernel =  kernel_op(kernel_size, FLAGS.nx, FLAGS.ny)
    kernel.add_kernel(external_kernel_dict)

    tiles = [
        self.tile, -1.0 * self.tile, 2.0 * self.tile, -3.0 * self.tile,
        4.0 * self.tile, -5.0 * self.tile
    ]

    with self.subTest(name='ThirdOrderDifferenceInX'):
      kd = self.evaluate(kernel.apply_kernel_op_x([self.tile], 'k3d1x+'))
      kd_ext = self.evaluate(kernel.apply_kernel_op_x([self.tile], 'k3d1Extx'))
      self.assertLen(kd_ext, 1)
      self.assertAllEqual(kd[0], kd_ext[0])

    with self.subTest(name='ThirdOrderDifferenceInY'):
      kd = self.evaluate(kernel.apply_kernel_op_y([self.tile], 'k3d1y+'))
      kd_ext = self.evaluate(kernel.apply_kernel_op_y([self.tile], 'k3d1Exty'))
      self.assertLen(kd_ext, 1)
      self.assertAllEqual(kd[0], kd_ext[0])

    with self.subTest(name='ThirdOrderDifferenceInZ'):
      kd = self.evaluate(kernel.apply_kernel_op_z(tiles, 'k3d1z+', 'k3d1z+sh'))
      kd_ext = self.evaluate(
          kernel.apply_kernel_op_z(tiles, 'k3d1Extz', 'k3d1Extzsh'))
      self.assertLen(kd_ext, 6)
      for i in range(6):
        self.assertAllEqual(kd[i], kd_ext[i])


@test_util.run_all_in_graph_and_eager_modes
class GetBigKernelFnTest(tf.test.TestCase, parameterized.TestCase):

  KERNEL_SIZES = [8, 16, 16, 16]
  KERNEL_OP = [lambda ks, _, __: get_kernel_fn.ApplyKernelConvOp(ks),
               lambda ks, _, __: get_kernel_fn.ApplyKernelConvOp(ks),
               lambda _, nx, ny: get_kernel_fn.ApplyKernelMulOp(nx, ny),
               lambda _, __, ___: get_kernel_fn.ApplyKernelSliceOp()]

  def setUp(self):
    super(GetBigKernelFnTest, self).setUp()
    FLAGS.nx = 16
    FLAGS.ny = 16
    FLAGS.nz = 1
    big_tile_fname = os.path.join(_TESTDATA_DIR, 'big_tile.txt')
    with gfile.GFile(resources.GetResourceFilename(big_tile_fname)) as f:
      self.big_tile_np = np.loadtxt(f, dtype=np.float32)

    self._testdata_dir = (
        os.getenv('TEST_UNDECLARED_OUTPUTS_DIR') if FLAGS.update_testdata
        else _TESTDATA_DIR)

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpD4x(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    big_tile = tf.constant(self.big_tile_np)
    kd4x = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx, FLAGS.ny).apply_kernel_op_x([big_tile],
                                                                     'kD4x'))

    expected_fname = os.path.join(self._testdata_dir,
                                  'big_tile_convop_kD4x.txt')
    with gfile.GFile(resources.GetResourceFilename(expected_fname)) as f:
      expected = np.loadtxt(f, dtype=np.float32)

    self.assertLen(kd4x, 1)
    self.assertAllEqual(expected, kd4x[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpD4y(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    big_tile = tf.constant(self.big_tile_np)
    kd4y = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx, FLAGS.ny).apply_kernel_op_y([big_tile],
                                                                     'kD4y'))

    expected_fname = os.path.join(self._testdata_dir,
                                  'big_tile_convop_kD4y.txt')
    with gfile.GFile(resources.GetResourceFilename(expected_fname)) as f:
      expected = np.loadtxt(f, dtype=np.float32)

    self.assertLen(kd4y, 1)
    self.assertAllEqual(expected, kd4y[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpdd8x(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    big_tile = tf.constant(self.big_tile_np)
    kdd8x = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx, FLAGS.ny).apply_kernel_op_x([big_tile],
                                                                     'kdd8x'))
    self.assertLen(kdd8x, 1)
    expected_fname = os.path.join(self._testdata_dir,
                                  'big_tile_convop_kdd8x.txt')
    if FLAGS.update_testdata:
      with gfile.GFile(expected_fname, 'w') as f:
        np.savetxt(f, kdd8x[0])
      return
    with gfile.GFile(resources.GetResourceFilename(expected_fname)) as f:
      expected = np.loadtxt(f, dtype=np.float32)
    self.assertAllClose(expected, kdd8x[0], atol=2e-6)

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpdd8y(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    big_tile = tf.constant(self.big_tile_np)
    kdd8y = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx, FLAGS.ny).apply_kernel_op_y([big_tile],
                                                                     'kdd8y'))
    self.assertLen(kdd8y, 1)
    expected_fname = os.path.join(self._testdata_dir,
                                  'big_tile_convop_kdd8y.txt')
    if FLAGS.update_testdata:
      with gfile.GFile(expected_fname, 'w') as f:
        np.savetxt(f, kdd8y[0])
      return
    with gfile.GFile(resources.GetResourceFilename(expected_fname)) as f:
      expected = np.loadtxt(f, dtype=np.float32)
    self.assertAllClose(expected, kdd8y[0], atol=2e-6)

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpdd16x(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    big_tile = tf.constant(self.big_tile_np)
    kdd16x = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx, FLAGS.ny).apply_kernel_op_x([big_tile],
                                                                     'kdd16x'))
    self.assertLen(kdd16x, 1)
    expected_fname = os.path.join(self._testdata_dir,
                                  'big_tile_convop_kdd16x.txt')
    if FLAGS.update_testdata:
      with gfile.GFile(expected_fname, 'w') as f:
        np.savetxt(f, kdd16x[0])
      return
    with gfile.GFile(resources.GetResourceFilename(expected_fname)) as f:
      expected = np.loadtxt(f, dtype=np.float32)
    self.assertAllClose(expected, kdd16x[0], atol=2e-5)

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOpdd16y(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    big_tile = tf.constant(self.big_tile_np)
    kdd16y = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx, FLAGS.ny).apply_kernel_op_y([big_tile],
                                                                     'kdd16y'))
    self.assertLen(kdd16y, 1)
    expected_fname = os.path.join(self._testdata_dir,
                                  'big_tile_convop_kdd16y.txt')
    if FLAGS.update_testdata:
      with gfile.GFile(expected_fname, 'w') as f:
        np.savetxt(f, kdd16y[0])
      return
    with gfile.GFile(resources.GetResourceFilename(expected_fname)) as f:
      expected = np.loadtxt(f, dtype=np.float32)
    self.assertAllClose(expected, kdd16y[0], atol=2e-5)

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOp4d2x(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    big_tile = tf.constant(self.big_tile_np)
    k4d2x = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx, FLAGS.ny).apply_kernel_op_x([big_tile],
                                                                     'k4d2x'))

    expected_fname = os.path.join(self._testdata_dir,
                                  'big_tile_convop_k4d2x.txt')
    with gfile.GFile(resources.GetResourceFilename(expected_fname)) as f:
      expected = np.loadtxt(f, dtype=np.float32)

    self.assertLen(k4d2x, 1)
    self.assertAllEqual(expected, k4d2x[0])

  @parameterized.parameters(*zip(KERNEL_SIZES, KERNEL_OP))
  def testKernelConvOp4d2y(self, kernel_size, kernel_op):
    FLAGS.kernel_size = kernel_size
    big_tile = tf.constant(self.big_tile_np)
    k4d2y = self.evaluate(
        kernel_op(kernel_size, FLAGS.nx, FLAGS.ny).apply_kernel_op_y([big_tile],
                                                                     'k4d2y'))

    expected_fname = os.path.join(self._testdata_dir,
                                  'big_tile_convop_k4d2y.txt')
    with gfile.GFile(resources.GetResourceFilename(expected_fname)) as f:
      expected = np.loadtxt(f, dtype=np.float32)

    self.assertLen(k4d2y, 1)
    self.assertAllEqual(expected, k4d2y[0])

  @parameterized.parameters(
      [get_kernel_fn.ApplyKernelConvOp(8),
       get_kernel_fn.ApplyKernelMulOp(16, 16),
       get_kernel_fn.ApplyKernelSliceOp()])
  def testKernelZ(self, kernel_op):
    tile_0 = tf.constant([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [2, 3, 4],
                          [5, 6, 7]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30],
                          [40, 50, 60],
                          [70, 80, 90],
                          [20, 30, 40],
                          [50, 60, 70]], tf.float32)
    tile_2 = tf.constant([[15, 20, 25],
                          [10, 15, 20],
                          [5,  10, 15],
                          [0,   5, 10],
                          [-5,  0,  5]], tf.float32)
    tile_3 = tf.constant([[10, 40, 70],
                          [20, 50, 80],
                          [30, 60, 90],
                          [40, 70,  0],
                          [50, 80, 20]], tf.float32)
    tile_4 = tf.constant([[2, 2, 2],
                          [2, 2, 2],
                          [2, 2, 2],
                          [2, 2, 2],
                          [2, 2, 2]], tf.float32)
    tile_5 = tf.constant([[25, 24, 23],
                          [22, 21, 20],
                          [19, 18, 17],
                          [16, 15, 14],
                          [13, 12, 11]], tf.float32)
    tiles = [tile_0, tile_1, tile_2, tile_3, tile_4, tile_5] * 3

    kdd8z = self.evaluate(
        kernel_op.apply_kernel_op_z(tiles, 'kdd8z', 'kdd8zsh'))
    self.assertLen(kdd8z, 18)
    if FLAGS.update_testdata:
      for i in range(len(tiles)):
        expected_fname = os.path.join(self._testdata_dir,
                                      'big_tile_convop_kdd8z_%d.txt' % i)
        with gfile.GFile(expected_fname, 'w') as f:
          np.savetxt(f, kdd8z[i])
    else:
      expected = []
      for i in range(len(tiles)):
        expected_fname = os.path.join(self._testdata_dir,
                                      'big_tile_convop_kdd8z_%d.txt' % i)
        with gfile.GFile(resources.GetResourceFilename(expected_fname)) as f:
          expected.append(np.loadtxt(f, dtype=np.float32))

      self.assertAllEqual(expected[0], kdd8z[0])
      self.assertAllEqual(expected[1], kdd8z[1])
      self.assertAllEqual(expected[2], kdd8z[2])
      self.assertAllEqual(expected[3], kdd8z[3])
      self.assertAllClose(expected[4], kdd8z[4], atol=2e-6)
      self.assertAllClose(expected[5], kdd8z[5], atol=2e-6)
      self.assertAllClose(expected[6], kdd8z[6], atol=2e-6)
      self.assertAllClose(expected[7], kdd8z[7], atol=2e-6)
      self.assertAllClose(expected[8], kdd8z[8], atol=2e-6)
      self.assertAllClose(expected[9], kdd8z[9], atol=2e-6)
      self.assertAllClose(expected[10], kdd8z[10], atol=2e-6)
      self.assertAllClose(expected[11], kdd8z[11], atol=2e-6)
      self.assertAllClose(expected[12], kdd8z[12], atol=2e-6)
      self.assertAllClose(expected[13], kdd8z[13], atol=2e-6)
      self.assertAllEqual(expected[14], kdd8z[14])
      self.assertAllEqual(expected[15], kdd8z[15])
      self.assertAllEqual(expected[16], kdd8z[16])
      self.assertAllEqual(expected[17], kdd8z[17])

      kdd16z = self.evaluate(
          kernel_op.apply_kernel_op_z(tiles, 'kdd16z', 'kdd16zsh'))
      self.assertLen(kdd16z, 18)
      if FLAGS.update_testdata:
        for i in range(len(tiles)):
          expected_fname = os.path.join(self._testdata_dir,
                                        'big_tile_convop_kdd16z_%d.txt'% i)
          with gfile.GFile(expected_fname, 'w') as f:
            np.savetxt(f, kdd16z[i])
      else:
        expected = []
        for i in range(len(tiles)):
          expected_fname = os.path.join(self._testdata_dir,
                                        'big_tile_convop_kdd16z_%d.txt'% i)
          with gfile.GFile(resources.GetResourceFilename(expected_fname)) as f:
            expected.append(np.loadtxt(f, dtype=np.float32))

        self.assertAllEqual(expected[0], kdd16z[0])
        self.assertAllEqual(expected[1], kdd16z[1])
        self.assertAllEqual(expected[2], kdd16z[2])
        self.assertAllEqual(expected[3], kdd16z[3])
        self.assertAllEqual(expected[4], kdd16z[4])
        self.assertAllEqual(expected[5], kdd16z[5])
        self.assertAllEqual(expected[6], kdd16z[6])
        self.assertAllEqual(expected[7], kdd16z[7])
        self.assertAllClose(expected[8], kdd16z[8], atol=2e-6)
        self.assertAllClose(expected[9], kdd16z[9], atol=2e-6)
        self.assertAllEqual(expected[10], kdd16z[10])
        self.assertAllEqual(expected[11], kdd16z[11])
        self.assertAllEqual(expected[12], kdd16z[12])
        self.assertAllEqual(expected[13], kdd16z[13])
        self.assertAllEqual(expected[14], kdd16z[14])
        self.assertAllEqual(expected[15], kdd16z[15])
        self.assertAllEqual(expected[16], kdd16z[16])
        self.assertAllEqual(expected[17], kdd16z[17])

      # Tests the second order finite difference for the fourth order derivative
      # in the z direction.
      k4d2z = self.evaluate(
          kernel_op.apply_kernel_op_z(tiles, 'k4d2z', 'k4d2zsh'))

      expected = [
          np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 6, 7]],
                   dtype=np.float32),
          np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90], [20, 30, 40],
                    [50, 60, 70]],
                   dtype=np.float32),
          np.array([[13., -116., -245.], [-174., -303., -432.],
                    [-361., -490., -619.], [-236., -365., -94.],
                    [-423., -552., -321.]],
                   dtype=np.float32),
          np.array([[27., 196., 365.], [134., 303., 472.], [241., 410., 579.],
                    [268., 437., 6.], [375., 544., 173.]],
                   dtype=np.float32),
          np.array([[-112., -222., -332.], [-142., -252., -362.],
                    [-172., -282., -392.], [-210., -320., -30.],
                    [-240., -350., -100.]],
                   dtype=np.float32),
          np.array([[158., 188., 218.], [168., 198., 228.], [178., 208., 238.],
                    [140., 170., 100.], [150., 180., 120.]],
                   dtype=np.float32),
          np.array([[-117., -142., -167.], [-212., -237., -262.],
                    [-307., -332., -357.], [-130., -155., -180.],
                    [-225., -250., -275.]],
                   dtype=np.float32),
          np.array([[31., 96., 161.], [226., 291., 356.], [421., 486., 551.],
                    [168., 233., 198.], [363., 428., 403.]],
                   dtype=np.float32),
          np.array([[13., -116., -245.], [-174., -303., -432.],
                    [-361., -490., -619.], [-236., -365., -94.],
                    [-423., -552., -321.]],
                   dtype=np.float32),
          np.array([[27., 196., 365.], [134., 303., 472.], [241., 410., 579.],
                    [268., 437., 6.], [375., 544., 173.]],
                   dtype=np.float32),
          np.array([[-112., -222., -332.], [-142., -252., -362.],
                    [-172., -282., -392.], [-210., -320., -30.],
                    [-240., -350., -100.]],
                   dtype=np.float32),
          np.array([[158., 188., 218.], [168., 198., 228.], [178., 208., 238.],
                    [140., 170., 100.], [150., 180., 120.]],
                   dtype=np.float32),
          np.array([[-117., -142., -167.], [-212., -237., -262.],
                    [-307., -332., -357.], [-130., -155., -180.],
                    [-225., -250., -275.]],
                   dtype=np.float32),
          np.array([[31., 96., 161.], [226., 291., 356.], [421., 486., 551.],
                    [168., 233., 198.], [363., 428., 403.]],
                   dtype=np.float32),
          np.array([[13., -116., -245.], [-174., -303., -432.],
                    [-361., -490., -619.], [-236., -365., -94.],
                    [-423., -552., -321.]],
                   dtype=np.float32),
          np.array([[27., 196., 365.], [134., 303., 472.], [241., 410., 579.],
                    [268., 437., 6.], [375., 544., 173.]],
                   dtype=np.float32),
          np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                   dtype=np.float32),
          np.array([[25, 24, 23], [22, 21, 20], [19, 18, 17], [16, 15, 14],
                    [13, 12, 11]],
                   dtype=np.float32),
      ]

      self.assertLen(k4d2z, 18)
      for i in range(18):
        self.assertAllEqual(expected[i], k4d2z[i])

      # Tests the fourth order finite difference for the first order derivative
      # in the z direction.
      kd4z = self.evaluate(kernel_op.apply_kernel_op_z(tiles, 'kD4z', 'kD4zsh'))

      expected = [
          np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 6, 7]],
                   dtype=np.float32),
          np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90], [20, 30, 40],
                    [50, 60, 70]],
                   dtype=np.float32),
          np.array([[-1., 160., 321.], [-158., 3., 164.], [-315., -154., 7.],
                    [160., 321., -318.], [3., 164., -395.]],
                   dtype=np.float32),
          np.array([[-119., -148., -177.], [-46., -75., -104.],
                    [27., -2., -31.], [20., -9., -38.], [93., 64., 35.]],
                   dtype=np.float32),
          np.array(
              [[134., -110., -354.], [22., -222., -466.], [-90., -334., -578.],
               [-194., -438., 118.], [-306., -550., -74.]],
              dtype=np.float32),
          np.array([[-8., 20., 48.], [-4., 24., 52.], [0., 28., 56.],
                    [20., 48., -24.], [24., 52., -10.]],
                   dtype=np.float32),
          np.array([[-133., -50., 33.], [136., 219., 302.], [405., 488., 571.],
                    [34., 117., 200.], [303., 386., 469.]],
                   dtype=np.float32),
          np.array([[127., 128., 129.], [50., 51., 52.], [-27., -26., -25.],
                    [-40., -39., 62.], [-117., -116., -25.]],
                   dtype=np.float32),
          np.array([[-1., 160., 321.], [-158., 3., 164.], [-315., -154., 7.],
                    [160., 321., -318.], [3., 164., -395.]],
                   dtype=np.float32),
          np.array([[-119., -148., -177.], [-46., -75., -104.],
                    [27., -2., -31.], [20., -9., -38.], [93., 64., 35.]],
                   dtype=np.float32),
          np.array(
              [[134., -110., -354.], [22., -222., -466.], [-90., -334., -578.],
               [-194., -438., 118.], [-306., -550., -74.]],
              dtype=np.float32),
          np.array([[-8., 20., 48.], [-4., 24., 52.], [0., 28., 56.],
                    [20., 48., -24.], [24., 52., -10.]],
                   dtype=np.float32),
          np.array([[-133., -50., 33.], [136., 219., 302.], [405., 488., 571.],
                    [34., 117., 200.], [303., 386., 469.]],
                   dtype=np.float32),
          np.array([[127., 128., 129.], [50., 51., 52.], [-27., -26., -25.],
                    [-40., -39., 62.], [-117., -116., -25.]],
                   dtype=np.float32),
          np.array([[-1., 160., 321.], [-158., 3., 164.], [-315., -154., 7.],
                    [160., 321., -318.], [3., 164., -395.]],
                   dtype=np.float32),
          np.array([[-119., -148., -177.], [-46., -75., -104.],
                    [27., -2., -31.], [20., -9., -38.], [93., 64., 35.]],
                   dtype=np.float32),
          np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                   dtype=np.float32),
          np.array([[25, 24, 23], [22, 21, 20], [19, 18, 17], [16, 15, 14],
                    [13, 12, 11]],
                   dtype=np.float32),
      ]

      self.assertLen(kd4z, 18)
      for i in range(18):
        self.assertAllEqual(expected[i], kd4z[i])


if __name__ == '__main__':
  tf.test.main()
