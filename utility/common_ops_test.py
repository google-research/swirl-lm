# pylint: disable=bad-whitespace
"""Tests for common_ops."""

import functools
import itertools

import numpy as np
from swirl_lm.utility import common_ops
from swirl_lm.utility import grid_parametrization
import tensorflow as tf
from google3.research.simulation.tensorflow.fluid.framework import initializer
from google3.research.simulation.tensorflow.fluid.framework import util
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized

_NormType = common_ops.NormType


class CommonOpsTest(tf.test.TestCase, parameterized.TestCase):

  def testGetTileName(self):
    self.assertEqual(common_ops.get_tile_name('u', 5), 'u_tile_5')

  def testGenField(self):
    out = self.evaluate(common_ops.gen_field('u', 2, 2, 3))
    self.assertLen(out, 1)
    self.assertAllEqual(out['u'], np.zeros((3, 2, 2)))

  def testGetFieldRange(self):
    state = {'u': tf.zeros([3, 3, 2], dtype=tf.float32)}
    out = self.evaluate(
        common_ops.get_range_results('u', 0, 3, None, None, state, None))
    self.assertLen(out, 2)
    self.assertAllEqual(out[0], np.zeros((3, 3, 2)))

  def testGetField(self):
    state = {'u_tile_0': 0}
    self.assertEqual([0], common_ops.get_field(state, 'u', 1))

  def testPrepStepByChunkFn(self):
    input_state = {'u': tf.zeros([3, 3, 2], dtype=tf.float32)}
    inputs = tf.ones([2, 3, 3, 2], dtype=tf.float32)
    outputs, state = self.evaluate(
        common_ops.prep_step_by_chunk_fn('u', 0, 3, inputs, None, input_state,
                                         None))
    self.assertEqual([0], outputs)
    self.assertAllEqual(state['u'], np.ones((3, 3, 2)))

  @parameterized.named_parameters(
      ('TensorUpdateIn0', 0, 2,
       np.reshape(np.arange(30, dtype=np.float32), (6, 1, 5))),
      ('TensorUpdateIn1', 1, 3,
       np.reshape(np.arange(24, dtype=np.float32), (6, 4, 1))),
      ('TensorUpdateIn2', 2, 4,
       np.reshape(np.arange(20, dtype=np.float32), (1, 4, 5))),
      ('ConstantUpdateIn0', 0, 3, 6.0),
      ('ConstantUpdateIn1', 1, 2, 8.0),
      ('ConstantUpdateIn2', 2, 1, 36.0),
  )
  def testTensorScatter1DUpdatesProvidesCorrectPlaneUpdates(
      self, dim, index, updates):
    """Checks if a plane in a 3D tensor is updated correctly."""
    nx = 4
    ny = 5
    nz = 6

    updates_tf = tf.unstack(tf.convert_to_tensor(
        updates,
        dtype=tf.float32)) if not isinstance(updates, float) else updates

    tensor = tf.unstack(
        tf.convert_to_tensor(np.zeros((nz, nx, ny)), dtype=tf.float32))

    result = self.evaluate(
        common_ops.tensor_scatter_1d_update(tensor, dim, index, updates_tf))

    with self.subTest(name='UpdatedTensorIsCorrect'):
      expected = np.zeros((nz, nx, ny), dtype=np.float32)
      updates = np.squeeze(updates)
      if dim == 0:
        expected[:, index, :] = updates
      elif dim == 1:
        expected[..., index] = updates
      else:  # dim == 2
        expected[index, ...] = updates

      self.assertAllClose(expected, result)

    with self.subTest(name='OriginalTensorUnchanged'):
      expected = np.zeros((nz, nx, ny), dtype=np.float32)
      self.assertAllEqual(expected, self.evaluate(tensor))

  _REPLICAS = (
      np.array([[[0, 1]]]),
      np.array([[[0], [1]]]),
      np.array([[[0]], [[1]]]),
  )
  _DIM = (0, 1, 2)
  _CORE_INDEX = (0, 1)

  @parameterized.parameters(*itertools.product(_REPLICAS, _DIM, _CORE_INDEX))
  def testTensorScatter1DUpdatesGlobalProvidesCorrectPlaneUpdates(
      self, replicas, dim, core_index):
    """Checks if the wall normal velocity is 0 at lower wall."""
    nx = 4
    ny = 5
    nz = 6

    tensor = tf.unstack(
        tf.convert_to_tensor(np.zeros((nz, nx, ny)), dtype=tf.float32))

    plane_index = 2
    updates = 6.0

    def device_fn(replica_id):
      """The face interpolation function wrapper for TPU."""
      return common_ops.tensor_scatter_1d_update_global(replica_id, replicas,
                                                        tensor, dim, core_index,
                                                        plane_index, updates)

    inputs = [[tf.constant(0)], [tf.constant(1)]]
    device_inputs = [list(x) for x in zip(*inputs)]

    computation_shape = np.array(replicas.shape)
    runner = TpuRunner(computation_shape=computation_shape)
    output = runner.run(device_fn, *device_inputs)
    output_0 = np.array(output[0])
    output_1 = np.array(output[1])

    axis = np.where(np.roll(computation_shape, 1) == 2)[0]
    output_all = np.concatenate([output_0, output_1], axis=int(axis))

    expected = np.zeros((nz * computation_shape[2], nx * computation_shape[0],
                         ny * computation_shape[1]),
                        dtype=np.float32)
    if core_index < computation_shape[dim]:
      if dim == 0:
        expected[:, core_index * 4 + 2, :] = 6.0
      elif dim == 1:
        expected[..., core_index * 5 + 2] = 6.0
      else:
        expected[core_index * 6 + 2, ...] = 6.0

    self.assertAllEqual(expected, output_all)

  def testApplyMulopX(self):
    tile_0 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]], tf.float32)
    tile_2 = tf.constant([[11, 22, 33], [44, 55, 66], [77, 88, 99]], tf.float32)
    tile_3 = tf.constant([[10, 40, 70], [20, 50, 80], [30, 60, 90]], tf.float32)

    out = common_ops.apply_op_x([tile_0, tile_1, tile_2, tile_3],
                                tf.constant([[1, -1, 0], [0, 1, -1], [0, 0, 1]],
                                            tf.float32))
    self.assertLen(out, 4)
    self.assertAllEqual(
        out[0], np.array([[-3, -3, -3], [-3, -3, -3], [7, 8, 9]], np.float32))
    self.assertAllEqual(
        out[1],
        np.array([[-30, -30, -30], [-30, -30, -30], [70, 80, 90]], np.float32))
    self.assertAllEqual(
        out[2],
        np.array([[-33, -33, -33], [-33, -33, -33], [77, 88, 99]], np.float32))
    self.assertAllEqual(
        out[3],
        np.array([[-10, -10, -10], [-10, -10, -10], [30, 60, 90]], np.float32))

  def testExceptionApplyMulopXWithNonSquareMulop(self):
    tile_0 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]], tf.float32)
    with self.assertRaises(RuntimeError) as err:
      self.evaluate(
          common_ops.apply_op_x([tile_0, tile_1],
                                tf.constant([[1, -1, 0], [0, 1, 1]],
                                            tf.float32)))
    self.assertEqual(
        str(err.exception), 'apply_op_x requires a square mulop. '
        'mulop shape is (2, 3).')

  def testExceptionApplyMulopXWithWrongMulopShape(self):
    tile_0 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]], tf.float32)
    with self.assertRaises(RuntimeError) as err:
      self.evaluate(
          common_ops.apply_op_x([tile_0, tile_1],
                                tf.constant([[1, -1], [0, 1]], tf.float32)))
    self.assertEqual(
        str(err.exception), 'apply_op_x needs the tensor dim 0 '
        'size to be divisible by mulop size 2. Tensor shape is '
        '(3, 3).')

  def testApplyMulopY(self):
    tile_0 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]], tf.float32)
    tile_2 = tf.constant([[11, 22, 33], [44, 55, 66], [77, 88, 99]], tf.float32)
    tile_3 = tf.constant([[10, 40, 70], [20, 50, 80], [30, 60, 90]], tf.float32)

    out = self.evaluate(
        common_ops.apply_op_y([tile_0, tile_1, tile_2, tile_3],
                              tf.constant([[1, -1, 0], [0, 1, -1], [0, 0, 1]],
                                          tf.float32)))
    self.assertLen(out, 4)
    self.assertAllEqual(out[0],
                        np.array([[1, 1, 1], [4, 1, 1], [7, 1, 1]], np.float32))
    self.assertAllEqual(
        out[1], np.array([[10, 10, 10], [40, 10, 10], [70, 10, 10]],
                         np.float32))
    self.assertAllEqual(
        out[2], np.array([[11, 11, 11], [44, 11, 11], [77, 11, 11]],
                         np.float32))
    self.assertAllEqual(
        out[3], np.array([[10, 30, 30], [20, 30, 30], [30, 30, 30]],
                         np.float32))

  def testExceptionApplyMulopYWithNonSquareMulop(self):
    tile_0 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]], tf.float32)
    with self.assertRaises(RuntimeError) as err:
      self.evaluate(
          common_ops.apply_op_y([tile_0, tile_1],
                                tf.constant([[1, -1, 0], [0, 1, 0]],
                                            tf.float32)))
    self.assertEqual(
        str(err.exception), 'apply_op_y requires a square mulop. '
        'mulop shape is (2, 3).')

  def testExceptionApplyMulopYWithWrongMulopShape(self):
    tile_0 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]], tf.float32)
    with self.assertRaises(RuntimeError) as err:
      self.evaluate(
          common_ops.apply_op_y([tile_0, tile_1],
                                tf.constant([[1, -1], [0, 1]], tf.float32)))
    self.assertEqual(
        str(err.exception), 'apply_op_y needs the tensor dim 1 '
        'size to be divisible by mulop size 2. Tensor shape is '
        '(3, 3).')

  def testApplyMulopZ(self):
    tile_0 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]], tf.float32)
    tile_2 = tf.constant([[11, 22, 33], [44, 55, 66], [77, 88, 99]], tf.float32)
    tile_3 = tf.constant([[10, 40, 70], [20, 50, 80], [30, 60, 90]], tf.float32)
    out = self.evaluate(
        common_ops.apply_op_z([tile_0, tile_1, tile_2, tile_3], [-1.0, 1.0],
                              [-1, 1]))
    self.assertLen(out, 4)
    self.assertAllEqual(out[0],
                        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.float32))
    self.assertAllEqual(
        out[1], np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                         np.float32))
    self.assertAllEqual(
        out[2], np.array([[0, 20, 40], [-20, 0, 20], [-40, -20, 0]],
                         np.float32))
    self.assertAllEqual(
        out[3], np.array([[10, 40, 70], [20, 50, 80], [30, 60, 90]],
                         np.float32))

  def testExceptionApplyMulopZWrongTileListLength(self):
    tile_0 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32)
    with self.assertRaises(RuntimeError) as err:
      self.evaluate(common_ops.apply_op_z([tile_0], [-1.0, 1.0], [-1, 1]))
    self.assertEqual(
        str(err.exception), 'apply_op_z requires tile_list '
        'length (1) be greater than or equal to z_op_list '
        'length (2).')

  def testExceptionApplyMulopZWrongZOpListLength(self):
    tile_0 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]], tf.float32)
    with self.assertRaises(RuntimeError) as err:
      self.evaluate(
          common_ops.apply_op_z([tile_0, tile_1], [-1.0, 1.0], [-1, 1, 0]))
    self.assertEqual(
        str(err.exception), 'apply_op_z requires z_op_list length '
        '(2) be equal to shift length (3).')

  def testApplyMulopZ2(self):
    tile_0 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]], tf.float32)
    tile_2 = tf.constant([[11, 22, 33], [44, 55, 66], [77, 88, 99]], tf.float32)
    tile_3 = tf.constant([[10, 40, 70], [20, 50, 80], [30, 60, 90]], tf.float32)
    out = self.evaluate(
        common_ops.apply_op_z([tile_0, tile_1, tile_2, tile_3], [-1.0, 1.0],
                              [-1, 0]))
    self.assertLen(out, 4)
    self.assertAllEqual(out[0],
                        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.float32))
    self.assertAllEqual(
        out[1], np.array([[9, 18, 27], [36, 45, 54], [63, 72, 81]], np.float32))
    self.assertAllEqual(out[2],
                        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.float32))
    self.assertAllEqual(
        out[3],
        np.array([[-1, 18, 37], [-24, -5, 14], [-47, -28, -9]], np.float32))

  def testApplyConvolutionalOpX(self):
    tile_0 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]], tf.float32)
    tile_2 = tf.constant([[11, 22, 33], [44, 55, 66], [77, 88, 99]], tf.float32)
    tile_3 = tf.constant([[10, 40, 70], [20, 50, 80], [30, 60, 90]], tf.float32)
    # Apply the backward first order finite difference in x:
    # u_x_{i, j} = u_{i, j} - u_{i-1, j}.
    kernel = tf.stack([
        tf.convert_to_tensor(
            np.eye(3, 3) - np.eye(3, 3, k=1), dtype=tf.float32)
    ])

    out = self.evaluate(
        common_ops.apply_convolutional_op_x([tile_0, tile_1, tile_2, tile_3],
                                            kernel))
    self.assertLen(out, 4)
    self.assertAllEqual(out[0],
                        np.array([[1, 2, 3], [3, 3, 3], [3, 3, 3]], np.float32))
    self.assertAllEqual(
        out[1], np.array([[10, 20, 30], [30, 30, 30], [30, 30, 30]],
                         np.float32))
    self.assertAllEqual(
        out[2], np.array([[11, 22, 33], [33, 33, 33], [33, 33, 33]],
                         np.float32))
    self.assertAllEqual(
        out[3], np.array([[10, 40, 70], [10, 10, 10], [10, 10, 10]],
                         np.float32))

  def testApplyConvolutionalOpY(self):
    tile_0 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]], tf.float32)
    tile_2 = tf.constant([[11, 22, 33], [44, 55, 66], [77, 88, 99]], tf.float32)
    tile_3 = tf.constant([[10, 40, 70], [20, 50, 80], [30, 60, 90]], tf.float32)
    # Apply the backward first order finite difference in y:
    # u_y_{i,j} = u_{i,j} - u_{i,j-1}.
    kernel = tf.stack([
        tf.convert_to_tensor(
            np.eye(3, 3) - np.eye(3, 3, k=1), dtype=tf.float32)
    ])

    out = self.evaluate(
        common_ops.apply_convolutional_op_y([tile_0, tile_1, tile_2, tile_3],
                                            kernel))
    self.assertLen(out, 4)
    self.assertAllEqual(out[0],
                        np.array([[1, 1, 1], [4, 1, 1], [7, 1, 1]], np.float32))
    self.assertAllEqual(
        out[1], np.array([[10, 10, 10], [40, 10, 10], [70, 10, 10]],
                         np.float32))
    self.assertAllEqual(
        out[2], np.array([[11, 11, 11], [44, 11, 11], [77, 11, 11]],
                         np.float32))
    self.assertAllEqual(
        out[3], np.array([[10, 30, 30], [20, 30, 30], [30, 30, 30]],
                         np.float32))

  def testApplySliceOpX(self):
    tile_0 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]], tf.float32)
    tile_2 = tf.constant([[11, 22, 33], [44, 55, 66], [77, 88, 99]], tf.float32)
    tile_3 = tf.constant([[10, 40, 70], [20, 50, 80], [30, 60, 90]], tf.float32)
    # Apply the backward first order finite difference in x:
    # u_x_{i, j} = u_{i, j} - u_{i-1, j}.
    kernel = lambda u: u - tf.pad(u[:-1, :], paddings=[[1, 0], [0, 0]])

    out = self.evaluate(
        common_ops.apply_slice_op_x([tile_0, tile_1, tile_2, tile_3], kernel))
    self.assertLen(out, 4)
    self.assertAllEqual(out[0],
                        np.array([[1, 2, 3], [3, 3, 3], [3, 3, 3]], np.float32))
    self.assertAllEqual(
        out[1], np.array([[10, 20, 30], [30, 30, 30], [30, 30, 30]],
                         np.float32))
    self.assertAllEqual(
        out[2], np.array([[11, 22, 33], [33, 33, 33], [33, 33, 33]],
                         np.float32))
    self.assertAllEqual(
        out[3], np.array([[10, 40, 70], [10, 10, 10], [10, 10, 10]],
                         np.float32))

  def testApplySliceOpY(self):
    tile_0 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]], tf.float32)
    tile_2 = tf.constant([[11, 22, 33], [44, 55, 66], [77, 88, 99]], tf.float32)
    tile_3 = tf.constant([[10, 40, 70], [20, 50, 80], [30, 60, 90]], tf.float32)
    # Apply the backward first order finite difference in y:
    # u_y_{i,j} = u_{i,j} - u_{i,j-1}.
    kernel = lambda u: u - tf.pad(u[:, :-1], paddings=[[0, 0], [1, 0]])

    out = self.evaluate(
        common_ops.apply_slice_op_y([tile_0, tile_1, tile_2, tile_3], kernel))
    self.assertLen(out, 4)
    self.assertAllEqual(out[0],
                        np.array([[1, 1, 1], [4, 1, 1], [7, 1, 1]], np.float32))
    self.assertAllEqual(
        out[1], np.array([[10, 10, 10], [40, 10, 10], [70, 10, 10]],
                         np.float32))
    self.assertAllEqual(
        out[2], np.array([[11, 11, 11], [44, 11, 11], [77, 11, 11]],
                         np.float32))
    self.assertAllEqual(
        out[3], np.array([[10, 30, 30], [20, 30, 30], [30, 30, 30]],
                         np.float32))

  def testSplitStateInZ(self):
    tile_0 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]], tf.float32)
    tile_2 = tf.constant([[11, 22, 33], [44, 55, 66], [77, 88, 99]], tf.float32)
    tile_3 = tf.constant([[10, 40, 70], [20, 50, 80], [30, 60, 90]], tf.float32)
    tiles = {'a': tf.stack([tile_0, tile_1]), 'b': tf.stack([tile_2, tile_3])}

    out = self.evaluate(common_ops.split_state_in_z(tiles, ['a', 'b'], 2))
    self.assertLen(out, 4)
    self.assertAllEqual(out['a_tile_0'],
                        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.float32))
    self.assertAllEqual(
        out['a_tile_1'],
        np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], np.float32))
    self.assertAllEqual(
        out['b_tile_0'],
        np.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]], np.float32))
    self.assertAllEqual(
        out['b_tile_1'],
        np.array([[10, 40, 70], [20, 50, 80], [30, 60, 90]], np.float32))

  def testMergeStateInZ(self):
    tile_0 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32)
    tile_1 = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]], tf.float32)
    tile_2 = tf.constant([[11, 22, 33], [44, 55, 66], [77, 88, 99]], tf.float32)
    tile_3 = tf.constant([[10, 40, 70], [20, 50, 80], [30, 60, 90]], tf.float32)
    tiles = {
        'a_tile_0': tile_0,
        'a_tile_1': tile_1,
        'b_tile_0': tile_2,
        'b_tile_1': tile_3
    }

    out = self.evaluate(common_ops.merge_state_in_z(tiles, ['a', 'b'], 2))
    self.assertLen(out, 2)
    self.assertAllEqual(
        out['a'],
        np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                  [[10, 20, 30], [40, 50, 60], [70, 80, 90]]], np.float32))
    self.assertAllEqual(
        out['b'],
        np.array([[[11, 22, 33], [44, 55, 66], [77, 88, 99]],
                  [[10, 40, 70], [20, 50, 80], [30, 60, 90]]], np.float32))

  @parameterized.named_parameters(
      # -
      ('Case10', common_ops.subtract, 1, 2, tf.float32, np.float32, -1),
      ('Case11', common_ops.subtract, 2, 1, tf.float64, np.float64, +1),
      # *
      ('Case20', common_ops.multiply, -1, 2, tf.float32, np.float32, -2),
      ('Case21', common_ops.multiply, -2, -1, tf.float64, np.float64, 2),
      # /
      ('Case30', common_ops.divide, 1, 0, tf.float32, np.float32, np.inf),
      ('Case31', common_ops.divide, 0, -1, tf.float64, np.float64, 0),
      ('Case32', common_ops.divide, -2, -1, tf.float32, np.float32, 2),
      ('Case33', common_ops.divide, -1, 2, tf.float64, np.float64, -0.5),
      # /'
      ('Case40', common_ops.divide_no_nan, 1, 0, tf.float32, np.float32, 0),
      ('Case41', common_ops.divide_no_nan, 0, -1, tf.float64, np.float64, 0),
      ('Case42', common_ops.divide_no_nan, -2, -1, tf.float32, np.float32, 2),
      ('Case43', common_ops.divide_no_nan, -1, 2, tf.float64, np.float64, -0.5),
      # +'
      ('Case50', functools.partial(
          common_ops.scaled_sum, scale=0.5), 1, 2, tf.float32, np.float32, 1.5),
      ('Case51', functools.partial(common_ops.scaled_sum, scale=-0.5), -1, 2,
       tf.float64, np.float64, -0.5),
      ('Case52', functools.partial(common_ops.scaled_sum,
                                   scale=2), -2, 1, tf.float32, np.float32, -2),
      # +': average.
      ('Case53', common_ops.average, 1, 2, tf.float32, np.float32, 1.5),
      ('Case54', common_ops.average, -3, 2, tf.float64, np.float64, -0.5),
      # +''
      ('Case60',
       functools.partial(
           common_ops.linear_combination, scale_lhs=0.3,
           scale_rhs=0.5), 1, 2, tf.float32, np.float32, 1.3),
      ('Case61',
       functools.partial(
           common_ops.linear_combination, scale_lhs=-0.3,
           scale_rhs=-0.5), -1, 2, tf.float64, np.float64, -0.7),
  )
  def testOperators(self, fn, v1, v2, tf_dtype, expected_np_dtype,
                    expected_result):
    """Tests operators: +, -, *, / for 3D tensors, as a list of 2D tensors."""
    replicas = np.array([[[0]], [[1]]])
    computation_shape = replicas.shape
    num_replicas = np.prod(computation_shape)

    nx = 4
    ny = 2
    nz = 3
    vec1 = [tf.ones(shape=(nx, ny), dtype=tf_dtype) * v1 for _ in range(nz)]
    vec2 = [tf.ones(shape=(nx, ny), dtype=tf_dtype) * v2 for _ in range(nz)]

    runner = TpuRunner(replicas)
    res = runner.run(fn, [vec1, vec1], [vec2, vec2])

    self.assertLen(res, num_replicas)
    for i in range(num_replicas):
      res_i = res[i]
      self.assertLen(res_i, nz)

      for res_i_j in res_i:
        self.assertEqual(res_i_j.dtype, expected_np_dtype)
        self.assertAllClose(expected_result * np.ones((nx, ny)), res_i_j)

  @parameterized.named_parameters(('LhsFloatRhsList', 1.0, 2.0, True, False),
                                  ('LhsListRhsFloat', 1.0, 2.0, False, True),
                                  ('BothFloat', 1.0, 2.0, True, True))
  def testLinearCombinationProvidesCorrectResultsWithMixedInputTypes(
      self, v1, v2, v1_float, v2_float):
    """Checks if the linear combination function is correct with mixed input."""
    nx = 4
    ny = 2
    nz = 3
    vec1 = v1 if v1_float else [
        tf.ones(shape=(nx, ny), dtype=common_ops._DTYPE) * v1,
    ] * nz
    vec2 = v2 if v2_float else [
        tf.ones(shape=(nx, ny), dtype=common_ops._DTYPE) * v2,
    ] * nz

    test_fn = functools.partial(
        common_ops.linear_combination, scale_lhs=0.3, scale_rhs=-0.5)

    if v1_float and v2_float:
      self.assertEqual(-0.7, test_fn(v1, v2))
    else:
      res = self.evaluate(test_fn(vec1, vec2))
      expected = -0.7 * np.ones((nz, nx, ny), dtype=np.float32)
      self.assertAllEqual(expected, np.array(res))

  @parameterized.named_parameters(
      ('Case00', tf.float32, np.float32),
      ('Case01', tf.float64, np.float64),
  )
  def testGlobalDotInConjugateGradientSolverProduceCorrectVectorProduct(
      self, tf_dtype, np_dtype):
    replicas = np.array([[[0]], [[1]]])
    computation_shape = replicas.shape
    num_replicas = np.prod(computation_shape)
    group_assignment = np.array([range(num_replicas)], dtype=np.int32)

    tf_dot = functools.partial(
        common_ops.global_dot, group_assignment=group_assignment)

    nx = 8
    ny = 8
    nz = 6
    vec1 = [tf.ones(shape=(nx, ny), dtype=tf_dtype) for _ in range(nz)]
    vec2 = [tf.ones(shape=(nx, ny), dtype=tf_dtype) for _ in range(nz)]

    runner = TpuRunner(replicas=replicas)
    dot = runner.run(tf_dot, [vec1, vec1], [vec2, vec2])

    self.assertLen(dot, num_replicas)
    for i in range(num_replicas):
      dot_i = dot[i]

      self.assertIsInstance(dot_i, np_dtype)
      self.assertEqual(
          np.prod([nx, ny, nz * num_replicas]), dot_i,
          'Dot product in the {}th replica is incorrect!'.format(i))

  def testLocalVdotOneDimension(self):
    a = 1.2 + 3.2j
    b = 4.8 - 0.6j

    expected_results = np.vdot(a, b)
    actual_results = self.evaluate(
        common_ops.local_vdot(
            tf.convert_to_tensor(a, dtype=tf.complex64),
            tf.convert_to_tensor(b, dtype=tf.complex64)))

    self.assertAllClose(
        expected_results.real,
        actual_results.real,
        rtol=1e-05,
        atol=1e-05,
        msg='Real part failed.')

    self.assertAllClose(
        expected_results.imag,
        actual_results.imag,
        rtol=1e-05,
        atol=1e-05,
        msg='Imaginary part failed.')

  def testLocalVdotTwoDimension(self):
    dim0 = 3
    dim1 = 4
    a = -np.random.rand(dim0, dim1) + 1j * np.random.rand(dim0, dim1)
    b = np.random.rand(dim0, dim1) - 1j * np.random.rand(dim0, dim1)

    expected_results = np.vdot(a, b)
    actual_results = self.evaluate(
        common_ops.local_vdot([tf.convert_to_tensor(a, dtype=tf.complex64)],
                              [tf.convert_to_tensor(b, dtype=tf.complex64)]))
    self.assertAllClose(
        expected_results.real,
        actual_results.real,
        rtol=1e-05,
        atol=1e-05,
        msg='Real part failed.')

    self.assertAllClose(
        expected_results.imag,
        actual_results.imag,
        rtol=1e-05,
        atol=1e-05,
        msg='Imaginary part failed.')

  def testLocalVdotThreeDimension(self):
    dim0 = 5
    dim1 = 4
    dim2 = 6
    a = (-np.random.rand(dim0, dim1, dim2) +
         1j * np.random.rand(dim0, dim1, dim2))
    b = (
        np.random.rand(dim0, dim1, dim2) -
        1j * np.random.rand(dim0, dim1, dim2))

    expected_results = np.vdot(a, b)
    actual_results = self.evaluate(
        common_ops.local_vdot([tf.convert_to_tensor(a, dtype=tf.complex64)],
                              tf.convert_to_tensor(b, dtype=tf.complex64)))
    self.assertAllClose(
        expected_results.real,
        actual_results.real,
        rtol=1e-05,
        atol=1e-05,
        msg='Real part failed.')

    self.assertAllClose(
        expected_results.imag,
        actual_results.imag,
        rtol=1e-05,
        atol=1e-05,
        msg='Imaginary part failed.')

  _L_1_NORM = 36
  _L_2_NORM = 6
  _L_INF_NORM = 1

  _NORM_TYPES_AND_VALUES = (
      #
      # Single norm type.
      #
      ((_NormType.L1,), {
          'L1': _L_1_NORM,
      }),
      ((_NormType.L2,), {
          'L2': _L_2_NORM,
      }),
      ((_NormType.L_INF,), {
          'L_INF': _L_INF_NORM,
      }),
      #
      # Multiple norm types.
      #
      ((_NormType.L1, _NormType.L2), {
          'L1': _L_1_NORM,
          'L2': _L_2_NORM,
      }),
      ((_NormType.L1, _NormType.L_INF), {
          'L1': _L_1_NORM,
          'L_INF': _L_INF_NORM,
      }),
      ((_NormType.L1, _NormType.L2, _NormType.L_INF), {
          'L1': _L_1_NORM,
          'L2': _L_2_NORM,
          'L_INF': _L_INF_NORM,
      }),

      # Reverse ordering.
      ((_NormType.L2, _NormType.L1), {
          'L1': _L_1_NORM,
          'L2': _L_2_NORM,
      }),
      # With duplicates.
      ((_NormType.L1, _NormType.L2, _NormType.L1), {
          'L1': _L_1_NORM,
          'L2': _L_2_NORM,
      }),
  )

  @parameterized.parameters(*itertools.product(_NORM_TYPES_AND_VALUES,
                                               (True, False), (
                                                   (tf.float32, np.float32),
                                                   (tf.float64, np.float64),
                                               )))
  def testComputeNorm(self, norm_type_and_values, as_list, dtypes):
    replicas = np.array([[[0]], [[1]]])
    num_replicas = np.prod(replicas.shape)

    tf_compute_norm = functools.partial(
        common_ops.compute_norm,
        norm_types=norm_type_and_values[0],
        replicas=replicas)

    nx = 3
    ny = 3
    nz = 2
    if as_list:
      vec = [tf.ones(shape=(nx, ny), dtype=dtypes[0]) for _ in range(nz)]
    else:
      vec = tf.ones(shape=(nx, ny, nz), dtype=dtypes[0])

    runner = TpuRunner(replicas)
    typed_norms = runner.run(tf_compute_norm, [vec, vec])

    self.assertLen(typed_norms, num_replicas)
    expected_typed_norms = norm_type_and_values[1]
    for i in range(num_replicas):
      typed_norms_i = typed_norms[i]

      self.assertCountEqual(expected_typed_norms.keys(), typed_norms_i.keys())
      for norm_type_str in expected_typed_norms:
        typed_norm_i = typed_norms_i[norm_type_str]

        self.assertIsInstance(typed_norm_i, dtypes[1])
        self.assertAlmostEqual(
            expected_typed_norms[norm_type_str], typed_norm_i, places=6)

  def testCrossReplicaGather(self):
    num_replicas = 8
    tensors = [tf.constant(i) for i in range(num_replicas)]
    fn = lambda x: common_ops.cross_replica_gather(x, num_replicas)

    replicas = list(range(num_replicas))
    runner = TpuRunner(replicas)
    all_gathered = runner.run(fn, tensors)
    for replica in range(num_replicas):
      gathered = all_gathered[replica]
      # The original TF1 code permuted the components to 0, 1, 2, 3, 6, 7, 4, 5
      for i in range(num_replicas):
        with self.subTest(f'replica {replica}; tensor {i}'):
          self.assertAllEqual(i, gathered[i])


class ParameterizedCommonOpsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('_HaloWidth1', 1),
      ('_HaloWidth2', 2),
      ('_HaloWidth3', 3),
  )
  def testGetSlice(
      self,
      halo_width,
  ):
    self.assertEqual(
        slice(halo_width, -halo_width),
        common_ops.get_slice(0, 1, False, halo_width))
    self.assertEqual(
        slice(halo_width, -halo_width),
        common_ops.get_slice(0, 1, False, halo_width))
    self.assertEqual(
        slice(None, None), common_ops.get_slice(0, 1, True, halo_width))
    self.assertEqual(
        slice(halo_width, -halo_width),
        common_ops.get_slice(0, 3, False, halo_width))
    self.assertEqual(
        slice(0, -halo_width), common_ops.get_slice(0, 3, True, halo_width))
    self.assertEqual(
        slice(halo_width, -halo_width),
        common_ops.get_slice(1, 3, False, halo_width))
    self.assertEqual(
        slice(halo_width, -halo_width),
        common_ops.get_slice(1, 3, True, halo_width))
    self.assertEqual(
        slice(halo_width, -halo_width),
        common_ops.get_slice(2, 3, False, halo_width))
    self.assertEqual(
        slice(halo_width, None), common_ops.get_slice(2, 3, True, halo_width))

  _REPLICAS = [
      np.array([[[0], [1]]]),
      np.array([[[0, 1]]]),
      np.array([[[0]], [[1]]]),
      np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
      np.array([[[0, 3], [5, 7]], [[4, 1], [6, 2]]]),
      np.array([[[0, 3, 5, 7]], [[4, 1, 6, 2]]]),
      np.array([[[0], [4], [6], [7]], [[2], [3], [1], [5]]]),
      np.array([[[0], [2], [4], [7], [1], [3], [6], [5]]])
  ]

  _AXES = [
      0,
      1,
      2,
      [0, 1],
      [0, 2],
      [1, 2],
      [2, 1],
      [2, 0],
      [1, 0],
      [0, 1, 2],
  ]

  @parameterized.parameters(*zip(_REPLICAS))
  def testGroupReplicasAlongAxisReturnsVariableGroupAssignments(self, replicas):
    # Checks that group_replicas returns the correct group assignments for
    # arbitrary computation shapes and axis arguments.
    def sort(assignments):
      # Sorts the elements in each group and sorts the groups.
      return sorted(np.sort(assignments, axis=-1).tolist(), key=lambda x: x[0])

    cx, cy, cz = replicas.shape
    with self.subTest('axis=0'):
      expected = np.array([
          replicas[:, i, j] for i, j in itertools.product(range(cy), range(cz))
      ])
      actual = common_ops.group_replicas(replicas, axis=0)
      self.assertAllEqual(sort(expected), sort(actual))

    with self.subTest('axis=1'):
      expected = np.array([
          replicas[i, :, j] for i, j in itertools.product(range(cx), range(cz))
      ])
      actual = common_ops.group_replicas(replicas, axis=1)
      self.assertAllEqual(sort(expected), sort(actual))

    with self.subTest('axis=2'):
      expected = np.array([
          replicas[i, j, :] for i, j in itertools.product(range(cx), range(cy))
      ])
      actual = common_ops.group_replicas(replicas, axis=2)
      self.assertAllEqual(sort(expected), sort(actual))

    with self.subTest('axis=[0, 1]'):
      expected = np.array([replicas[:, :, i].reshape([-1]) for i in range(cz)])
      actual = common_ops.group_replicas(replicas, axis=[0, 1])
      self.assertAllEqual(sort(expected), sort(actual))

    with self.subTest('axis=[0, 2]'):
      expected = np.array([replicas[:, i, :].reshape([-1]) for i in range(cy)])
      actual = common_ops.group_replicas(replicas, axis=[0, 2])
      self.assertAllEqual(sort(expected), sort(actual))

    with self.subTest('axis=[1, 2]'):
      expected = np.array([replicas[i, :, :].reshape([-1]) for i in range(cx)])
      actual = common_ops.group_replicas(replicas, axis=[1, 2])
      self.assertAllEqual(sort(expected), sort(actual))

    with self.subTest('axis=[0, 1, 2]'):
      expected = replicas.reshape([1, -1])
      actual = common_ops.group_replicas(replicas, axis=[0, 1, 2])
      self.assertAllEqual(sort(expected), sort(actual))

    with self.subTest('axis=None'):
      expected = replicas.reshape([1, -1])
      actual = common_ops.group_replicas(replicas)
      self.assertAllEqual(sort(expected), sort(actual))

  @parameterized.parameters(*itertools.product(_REPLICAS, (
      (tf.float32, np.float32),
      (tf.float64, np.float64),
  )))
  def testGlobalMeanProduceCorrectResult(self, replicas, dtypes):
    computation_shape = np.array(replicas.shape)
    num_replicas = np.prod(computation_shape)
    grid_lengths = (1, 1, 1)
    nx = 8
    ny = 8
    nz = 6
    halo_width = 2

    def generate_init_field(xx, yy, zz, lx, ly, lz, coord):
      """Generates the initial field as a step function in each dimension."""
      del coord
      return 8. * tf.where(
          tf.greater(xx, lx / 2.), tf.ones_like(xx),
          tf.zeros_like(xx)) * tf.where(
              tf.greater(yy, ly / 2.), tf.ones_like(yy),
              tf.zeros_like(yy)) * tf.where(
                  tf.greater(zz, lz / 2.), tf.ones_like(zz), tf.zeros_like(zz))

    def init_fn(coordinates):
      params = (
          grid_parametrization.GridParametrization
          .create_from_grid_lengths_and_etc(grid_lengths, computation_shape,
                                            (nx, ny, nz), halo_width))
      return initializer.partial_mesh_for_core(params, coordinates,
                                               generate_init_field)

    def step_fn(state, replicas, replica_id):
      """Computes the global mean of `tensor_field`."""
      del replica_id
      return common_ops.global_mean(
          common_ops.tf_cast(tf.unstack(state), dtypes[0]), replicas,
          [halo_width, halo_width, halo_width])

    coordinates = util.grid_coordinates(computation_shape)
    runner = TpuRunner(computation_shape=computation_shape)
    mean = runner.run_with_replica_args(
        step_fn, [init_fn(coordinates[i]) for i in range(num_replicas)])

    self.assertLen(mean, num_replicas)
    for i in range(num_replicas):
      mean_value = mean[i]
      self.assertIsInstance(mean_value, dtypes[1])
      self.assertEqual(1., mean_value,
                       'Mean in the {}th replica is incorrect!'.format(i))

  @parameterized.parameters(
      (np.array([[[0]], [[1]]]), 0),
      (np.array([[[0], [1]]]), 1),
      (np.array([[[0, 1]]]), 2),
      (np.array([[[0], [1], [2], [3], [4], [5], [6], [7]]]), 1),
      (np.array([[[0], [1], [2], [3]], [[4], [5], [6], [7]]]), [0, 1]),
      (np.array([[[0, 3, 5, 7]], [[4, 1, 6, 2]]]), [0, 2]),
      (np.array([[[0, 1], [2, 3]]]), [1, 2]),
      (np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), [0, 1, 2]),
  )
  def testGlobalMeanInAxis(
      self,
      replicas,
      axis,
  ):
    """Tests that the distributed mean computation matches the full grid mean."""
    computation_shape = np.array(replicas.shape)
    num_replicas = np.prod(computation_shape)
    nx = 8
    ny = 8
    nz = 6
    halo_width = 2
    lx = 1.
    ly = 1.
    lz = 1.

    def generate_init_field(xx, yy, zz, lx, ly, lz, coord):
      """Generates the initial field as a step function in each dimension."""
      del coord
      return 8. * tf.where(
          tf.greater(xx, lx / 2.), tf.ones_like(xx),
          tf.zeros_like(xx)) * tf.where(
              tf.greater(yy, ly / 2.), tf.ones_like(yy),
              tf.zeros_like(yy)) * tf.where(
                  tf.greater(zz, lz / 2.), tf.ones_like(zz), tf.zeros_like(zz))

    def init_fn(coordinates):
      """Initializes the tensor field to feed into the fluid framework."""
      params = (
          grid_parametrization.GridParametrization
          .create_from_grid_lengths_and_etc((lx, ly, lz), computation_shape,
                                            (nx, ny, nz), halo_width))
      return initializer.partial_mesh_for_core(params, coordinates,
                                               generate_init_field)

    def step_fn(state, replicas, replica_id):
      """Computes the global axis mean of `tensor_field`."""
      del replica_id
      return common_ops.global_mean(
          tf.unstack(state),
          replicas, [halo_width, halo_width, halo_width],
          axis=axis)

    # Generates physical full grid.
    global_nx = (nx - 2 * halo_width) * computation_shape[0]
    global_ny = (ny - 2 * halo_width) * computation_shape[1]
    global_nz = (nz - 2 * halo_width) * computation_shape[2]
    x = np.linspace(0, lx, global_nx)
    y = np.linspace(0, ly, global_ny)
    z = np.linspace(0, lz, global_nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    global_grid = generate_init_field(xx, yy, zz, lx, ly, lz, None)

    # Computes the mean of the physical full grid along the axis.
    reduced_physical_grid = tf.reduce_mean(
        global_grid, axis=axis, keepdims=True)
    # Convert to z-list of tensor slices
    reduced_physical_grid = tf.unstack(reduced_physical_grid, axis=2)

    coordinates = util.grid_coordinates(computation_shape)
    runner = TpuRunner(computation_shape=computation_shape)
    mean = runner.run_with_replica_args(
        step_fn, [init_fn(coordinates[i]) for i in range(num_replicas)])

    self.assertLen(mean, num_replicas)
    for i in range(num_replicas):
      for expected, actual in zip(reduced_physical_grid, mean[i]):
        self.assertAllEqual(expected, actual,
                            f'Mean in the {i}th replica is incorrect!')

  @parameterized.parameters(*zip(_REPLICAS))
  def testGlobalReduce(
      self,
      replicas,
  ):
    """Checks `global_reduce` produces the correct global maximum as a scalar."""
    computation_shape = np.array(replicas.shape)
    num_replicas = np.prod(computation_shape)
    group_assignment = np.array(
        [np.reshape(replicas_i, -1) for replicas_i in replicas])

    device_fn = functools.partial(
        common_ops.global_reduce,
        operator=tf.math.reduce_max,
        group_assignment=group_assignment)

    runner = TpuRunner(computation_shape=computation_shape)
    output = runner.run(
        device_fn,
        [i * tf.ones((8, 8), dtype=tf.float32) for i in range(num_replicas)])

    for i in range(num_replicas):
      expected_max = None
      for g in range(group_assignment.shape[0]):
        if i in group_assignment[g]:
          expected_max = np.max(group_assignment[g])
      assert expected_max is not None
      self.assertEqual(output[i], expected_max)

  @parameterized.parameters(*zip(_REPLICAS))
  def testGetCoreCoordinate(
      self,
      replicas,
  ):
    computation_shape = np.array(replicas.shape)
    num_replicas = np.prod(computation_shape)
    inputs = [[i] for i in range(num_replicas)]

    def computation(replica_id):
      return common_ops.get_core_coordinate(replicas, replica_id)

    runner = TpuRunner(replicas)
    results = runner.run(computation, inputs)

    for i in range(num_replicas):
      self.assertAllEqual(results[i], np.squeeze(np.where(replicas == i)))

  _SPECTRAL_GRID_REPLICAS = [
      np.array([[[0, 3], [5, 7]], [[4, 1], [6, 2]]]),
      np.array([[[0, 3, 5, 7], [4, 1, 6, 2]]]),
      np.array([[[0], [2], [4], [7], [1], [3], [6], [5]]]),
      np.array([[[0, 3, 5, 7], [4, 1, 6, 2]]])
  ]

  _CORE_NXYZ = [[2, 3, 3], [3, 5, 3], [3, 2, 5], [3, 5, 3]]

  _HALOS = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 1]]
  _EXPECTED_GRID = [
      [
          {
              'xx': [0, 1],
              'yy': [0, 1, 2],
              'zz': [0, 1, 2],
              'xx_c': [0, -1],
              'yy_c': [0, -1, -2],
              'zz_c': [0, -1, -2],
          },
          {
              'xx': [2, -1],
              'yy': [0, 1, 2],
              'zz': [3, -2, -1],
              'xx_c': [2, 1],
              'yy_c': [0, -1, -2],
              'zz_c': [3, 2, 1],
          },
          {
              'xx': [2, -1],
              'yy': [3, -2, -1],
              'zz': [3, -2, -1],
              'xx_c': [2, 1],
              'yy_c': [3, 2, 1],
              'zz_c': [3, 2, 1],
          },
          {
              'xx': [0, 1],
              'yy': [0, 1, 2],
              'zz': [3, -2, -1],
              'xx_c': [0, -1],
              'yy_c': [0, -1, -2],
              'zz_c': [3, 2, 1],
          },
          {
              'xx': [2, -1],
              'yy': [0, 1, 2],
              'zz': [0, 1, 2],
              'xx_c': [2, 1],
              'yy_c': [0, -1, -2],
              'zz_c': [0, -1, -2],
          },
          {
              'xx': [0, 1],
              'yy': [3, -2, -1],
              'zz': [0, 1, 2],
              'xx_c': [0, -1],
              'yy_c': [3, 2, 1],
              'zz_c': [0, -1, -2],
          },
          {
              'xx': [2, -1],
              'yy': [3, -2, -1],
              'zz': [0, 1, 2],
              'xx_c': [2, 1],
              'yy_c': [3, 2, 1],
              'zz_c': [0, -1, -2],
          },
          {
              'xx': [0, 1],
              'yy': [3, -2, -1],
              'zz': [3, -2, -1],
              'xx_c': [0, -1],
              'yy_c': [3, 2, 1],
              'zz_c': [3, 2, 1],
          },
      ],
      [
          {
              'xx': [0, 1, -1],
              'yy': [0, 1, 2, 3, 4],
              'zz': [0, 1, 2],
              'xx_c': [0, -1, 1],
              'yy_c': [0, -1, -2, -3, -4],
              'zz_c': [0, -1, -2],
          },
          {
              'xx': [0, 1, -1],
              'yy': [5, -4, -3, -2, -1],
              'zz': [3, 4, 5],
              'xx_c': [0, -1, 1],
              'yy_c': [5, 4, 3, 2, 1],
              'zz_c': [-3, -4, -5],
          },
          {
              'xx': [0, 1, -1],
              'yy': [5, -4, -3, -2, -1],
              'zz': [-3, -2, -1],
              'xx_c': [0, -1, 1],
              'yy_c': [5, 4, 3, 2, 1],
              'zz_c': [3, 2, 1],
          },
          {
              'xx': [0, 1, -1],
              'yy': [0, 1, 2, 3, 4],
              'zz': [3, 4, 5],
              'xx_c': [0, -1, 1],
              'yy_c': [0, -1, -2, -3, -4],
              'zz_c': [-3, -4, -5],
          },
          {
              'xx': [0, 1, -1],
              'yy': [5, -4, -3, -2, -1],
              'zz': [0, 1, 2],
              'xx_c': [0, -1, 1],
              'yy_c': [5, 4, 3, 2, 1],
              'zz_c': [0, -1, -2],
          },
          {
              'xx': [0, 1, -1],
              'yy': [0, 1, 2, 3, 4],
              'zz': [6, -5, -4],
              'xx_c': [0, -1, 1],
              'yy_c': [0, -1, -2, -3, -4],
              'zz_c': [6, 5, 4],
          },
          {
              'xx': [0, 1, -1],
              'yy': [5, -4, -3, -2, -1],
              'zz': [6, -5, -4],
              'xx_c': [0, -1, 1],
              'yy_c': [5, 4, 3, 2, 1],
              'zz_c': [6, 5, 4],
          },
          {
              'xx': [0, 1, -1],
              'yy': [0, 1, 2, 3, 4],
              'zz': [-3, -2, -1],
              'xx_c': [0, -1, 1],
              'yy_c': [0, -1, -2, -3, -4],
              'zz_c': [3, 2, 1],
          },
      ],
      [
          {
              'xx': [0, 1, -1],
              'yy': [0, 1],
              'zz': [0, 1, 2, -2, -1],
              'xx_c': [0, -1, 1],
              'yy_c': [0, -1],
              'zz_c': [0, -1, -2, 2, 1],
          },
          {
              'xx': [0, 1, -1],
              'yy': [8, -7],
              'zz': [0, 1, 2, -2, -1],
              'xx_c': [0, -1, 1],
              'yy_c': [8, 7],
              'zz_c': [0, -1, -2, 2, 1],
          },
          {
              'xx': [0, 1, -1],
              'yy': [2, 3],
              'zz': [0, 1, 2, -2, -1],
              'xx_c': [0, -1, 1],
              'yy_c': [-2, -3],
              'zz_c': [0, -1, -2, 2, 1],
          },
          {
              'xx': [0, 1, -1],
              'yy': [-6, -5],
              'zz': [0, 1, 2, -2, -1],
              'xx_c': [0, -1, 1],
              'yy_c': [6, 5],
              'zz_c': [0, -1, -2, 2, 1],
          },
          {
              'xx': [0, 1, -1],
              'yy': [4, 5],
              'zz': [0, 1, 2, -2, -1],
              'xx_c': [0, -1, 1],
              'yy_c': [-4, -5],
              'zz_c': [0, -1, -2, 2, 1],
          },
          {
              'xx': [0, 1, -1],
              'yy': [-2, -1],
              'zz': [0, 1, 2, -2, -1],
              'xx_c': [0, -1, 1],
              'yy_c': [2, 1],
              'zz_c': [0, -1, -2, 2, 1],
          },
          {
              'xx': [0, 1, -1],
              'yy': [-4, -3],
              'zz': [0, 1, 2, -2, -1],
              'xx_c': [0, -1, 1],
              'yy_c': [4, 3],
              'zz_c': [0, -1, -2, 2, 1],
          },
          {
              'xx': [0, 1, -1],
              'yy': [6, 7],
              'zz': [0, 1, 2, -2, -1],
              'xx_c': [0, -1, 1],
              'yy_c': [-6, -7],
              'zz_c': [0, -1, -2, 2, 1],
          },
      ],
      [
          {
              'xx': [99, 0, 1, -1, 99],
              'yy': [99, 99, 0, 1, 2, 3, 4, 99, 99],
              'zz': [99, 0, 1, 2, 99],
              'xx_c': [99, 0, -1, 1, 99],
              'yy_c': [99, 99, 0, -1, -2, -3, -4, 99, 99],
              'zz_c': [99, 0, -1, -2, 99],
          },
          {
              'xx': [99, 0, 1, -1, 99],
              'yy': [99, 99, 5, -4, -3, -2, -1, 99, 99],
              'zz': [99, 3, 4, 5, 99],
              'xx_c': [99, 0, -1, 1, 99],
              'yy_c': [99, 99, 5, 4, 3, 2, 1, 99, 99],
              'zz_c': [99, -3, -4, -5, 99],
          },
          {
              'xx': [99, 0, 1, -1, 99],
              'yy': [99, 99, 5, -4, -3, -2, -1, 99, 99],
              'zz': [99, -3, -2, -1, 99],
              'xx_c': [99, 0, -1, 1, 99],
              'yy_c': [99, 99, 5, 4, 3, 2, 1, 99, 99],
              'zz_c': [99, 3, 2, 1, 99],
          },
          {
              'xx': [99, 0, 1, -1, 99],
              'yy': [99, 99, 0, 1, 2, 3, 4, 99, 99],
              'zz': [99, 3, 4, 5, 99],
              'xx_c': [99, 0, -1, 1, 99],
              'yy_c': [99, 99, 0, -1, -2, -3, -4, 99, 99],
              'zz_c': [99, -3, -4, -5, 99],
          },
          {
              'xx': [99, 0, 1, -1, 99],
              'yy': [99, 99, 5, -4, -3, -2, -1, 99, 99],
              'zz': [99, 0, 1, 2, 99],
              'xx_c': [99, 0, -1, 1, 99],
              'yy_c': [99, 99, 5, 4, 3, 2, 1, 99, 99],
              'zz_c': [99, 0, -1, -2, 99],
          },
          {
              'xx': [99, 0, 1, -1, 99],
              'yy': [99, 99, 0, 1, 2, 3, 4, 99, 99],
              'zz': [99, 6, -5, -4, 99],
              'xx_c': [99, 0, -1, 1, 99],
              'yy_c': [99, 99, 0, -1, -2, -3, -4, 99, 99],
              'zz_c': [99, 6, 5, 4, 99],
          },
          {
              'xx': [99, 0, 1, -1, 99],
              'yy': [99, 99, 5, -4, -3, -2, -1, 99, 99],
              'zz': [99, 6, -5, -4, 99],
              'xx_c': [99, 0, -1, 1, 99],
              'yy_c': [99, 99, 5, 4, 3, 2, 1, 99, 99],
              'zz_c': [99, 6, 5, 4, 99],
          },
          {
              'xx': [99, 0, 1, -1, 99],
              'yy': [99, 99, 0, 1, 2, 3, 4, 99, 99],
              'zz': [99, -3, -2, -1, 99],
              'xx_c': [99, 0, -1, 1, 99],
              'yy_c': [99, 99, 0, -1, -2, -3, -4, 99, 99],
              'zz_c': [99, 3, 2, 1, 99],
          },
      ],
  ]

  @parameterized.parameters(*zip(_SPECTRAL_GRID_REPLICAS, _CORE_NXYZ, _HALOS,
                                 _EXPECTED_GRID))
  def testGetSpectralIndexGrid(
      self,
      replicas,
      core_nxyz,
      halos,
      expected_grid,
  ):
    num_replicas = np.prod(replicas.shape)
    grid = functools.partial(
        common_ops.get_spectral_index_grid,
        core_nxyz[0],
        core_nxyz[1],
        core_nxyz[2],
        halos=halos,
        pad_value=99)

    runner = TpuRunner(replicas)
    result = runner.run_with_replica_args(grid)

    for i in range(num_replicas):
      for k in {'xx', 'yy', 'zz', 'xx_c', 'yy_c', 'zz_c'}:
        self.assertAllEqual(expected_grid[i][k], result[i][k])

  _DIMS = (0, 1, 2)

  @parameterized.parameters(*itertools.product(_REPLICAS, _DIMS))
  def testIntegrationInDim(
      self,
      replicas,
      dim,
  ):
    """Checks if integrating a constant function results in linear integral."""
    computation_shape = np.array(replicas.shape)
    ones = tf.unstack(
        tf.ones(
            (
                int(32 / computation_shape[2]),
                int(32 / computation_shape[0]),
                int(32 / computation_shape[1]),
            ),
            dtype=tf.float32,
        ))
    num_replicas = np.product(computation_shape)
    runner = TpuRunner(replicas)
    output = runner.run_with_replica_args(
        functools.partial(common_ops.integration_in_dim, h=0.2, dim=dim),
        f=[ones] * num_replicas)

    buf_z_0 = []
    buf_z_1 = []
    for k in range(computation_shape[2]):
      buf_y_0 = []
      buf_y_1 = []
      for j in range(computation_shape[1]):
        buf_x_0 = []
        buf_x_1 = []
        for i in range(computation_shape[0]):
          buf_x_0.append(output[replicas[i, j, k]][0])
          buf_x_1.append(output[replicas[i, j, k]][1])
        buf_y_0.append(np.concatenate(buf_x_0, axis=1))
        buf_y_1.append(np.concatenate(buf_x_1, axis=1))
      buf_z_0.append(np.concatenate(buf_y_0, axis=2))
      buf_z_1.append(np.concatenate(buf_y_1, axis=2))
    integral_from_0 = np.concatenate(buf_z_0, axis=0)
    integral_to_end = np.concatenate(buf_z_1, axis=0)

    def expand_dims(a):
      """Expands a 1D array to 3D in the last 2 dimensions."""
      if dim == 0:
        return a[np.newaxis, :, np.newaxis]
      elif dim == 1:
        return a[np.newaxis, np.newaxis, :]
      elif dim == 2:
        return a[:, np.newaxis, np.newaxis]

    if dim == 0:
      tile_shape = (32, 1, 32)
    elif dim == 1:
      tile_shape = (32, 32, 1)
    elif dim == 2:
      tile_shape = (1, 32, 32)

    with self.subTest(name='IntegrationFrom0'):
      expected = np.tile(expand_dims(0.2 * np.linspace(0, 31, 32)), tile_shape)
      self.assertAllClose(expected, integral_from_0)

    with self.subTest(name='IntegrationToEnd'):
      expected = np.tile(expand_dims(0.2 * np.arange(31, -1, -1)), tile_shape)
      self.assertAllClose(expected, integral_to_end)

  def testGetTensorShapeProducesCorrectShapeOfEmptyTensor(self):
    """Checks if the shape of an empty tensor is correct."""
    state = []

    shape = common_ops.get_tensor_shape(state)

    expected = (0, 0, 0)
    self.assertAllEqual(expected, shape)

  def testGetTensorShapeProducesCorrectShapeOfAListOf2DTensors(self):
    """Checks if the shape of a sequence of `tf.Tensor` is correct."""
    state = [
        tf.ones((3, 4), dtype=tf.float32),
        tf.zeros((3, 4), dtype=tf.float32),
    ]

    shape = common_ops.get_tensor_shape(state)

    expected = (2, 3, 4)
    self.assertAllEqual(expected, shape)

  def testGetTensorShapeRaisesValueErrorForNon3DTensor(self):
    """Checks if incorrect tensor shape raises `ValueError`."""
    with self.subTest(name='2DTensor'):
      state = [
          tf.ones((3,), dtype=tf.float32),
          tf.zeros((3,), dtype=tf.float32),
      ]

      with self.assertRaisesRegex(ValueError,
                                  'The tensor in the list has to be 2D'):
        _ = common_ops.get_tensor_shape(state)

    with self.subTest(name='4DTensor'):
      state = [
          tf.ones((3, 4, 5), dtype=tf.float32),
          tf.zeros((3, 4, 5), dtype=tf.float32),
      ]

      with self.assertRaisesRegex(ValueError,
                                  'The tensor in the list has to be 2D'):
        _ = common_ops.get_tensor_shape(state)

  def testPadCorrectlyPadsInputField(self):
    f = [tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32),
         tf.constant([[10, 11, 12], [13, 14, 15], [16, 17, 18]], tf.float32)]

    out = self.evaluate(common_ops.pad(f, [[1, 2], [2, 1], [2, 3]], value=0.0))
    self.assertLen(out, 7)
    self.assertAllEqual(out[0], tf.zeros([6,6], dtype=tf.float32))
    self.assertAllEqual(out[1], tf.zeros([6,6], dtype=tf.float32))
    self.assertAllEqual(
        out[2],
        np.array([[0, 0, 0, 0, 0, 0], [0, 0, 1, 2, 3, 0], [0, 0, 4, 5, 6, 0],
                  [0, 0, 7, 8, 9, 0], [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]], np.float32))
    self.assertAllEqual(
        out[3],
        np.array([[0, 0, 0, 0, 0, 0], [0, 0, 10, 11, 12, 0],
                  [0, 0, 13, 14, 15, 0], [0, 0, 16, 17, 18, 0],
                  [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], np.float32))
    self.assertAllEqual(out[4], tf.zeros([6,6], dtype=tf.float32))
    self.assertAllEqual(out[5], tf.zeros([6,6], dtype=tf.float32))
    self.assertAllEqual(out[5], tf.zeros([6,6], dtype=tf.float32))


class CrossReplicaGatherTest(tf.test.TestCase, parameterized.TestCase):

  @staticmethod
  def step_fn(state, replicas, replica_id):
    del replica_id  # Not used.
    num_replicas = np.prod(replicas.shape)
    subgrids = common_ops.cross_replica_gather(state, num_replicas)
    combined_grid = util.combine_subgrids(subgrids, replicas)
    return combined_grid

  def distribute_gather_combine(self, computation_shape, x):
    nx, ny, nz = [(f - 2) // c + 2 for f, c in zip(x.shape, computation_shape)]
    grid_lengths = (1, 1, 1)

    params = (
        grid_parametrization.GridParametrization
        .create_from_grid_lengths_and_etc(
            grid_lengths, computation_shape, (nx, ny, nz), halo_width=1))
    num_replicas = np.prod(computation_shape)
    coordinates = util.grid_coordinates(computation_shape)

    runner = TpuRunner(computation_shape=computation_shape)
    return runner.run_with_replica_args(self.step_fn, [
        initializer.subgrid_of_3d_grid_from_params(x, params, coordinates[i])
        for i in range(num_replicas)
    ])

  @parameterized.parameters(*itertools.product(
      range(1, 3), range(1, 3), range(1, 3)))
  def testCrossReplicaGather(self, cx, cy, cz):
    if cx * cy * cz == 1:
      return
    computation_shape = cx, cy, cz
    fs = 6, 6, 6
    x = np.random.rand(*fs).astype(np.float32)
    x_gathered = self.distribute_gather_combine(computation_shape, x)

    # Check that every replica has the full grid.
    for i in range(np.prod(computation_shape)):
      np.testing.assert_allclose(x, x_gathered[i])


if __name__ == '__main__':
  tf.test.main()
