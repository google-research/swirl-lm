"""Tests for convcetion."""

import itertools
import numpy as np
from swirl_lm.boundary_condition import boundary_condition_utils
from swirl_lm.numerics import convection
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized


class IncompressibleStructuredMeshNumericsTest(tf.test.TestCase,
                                               parameterized.TestCase):

  MESH_SIZES = [1.0, 1.0, 1.0]
  DIRECTIONS = [0, 1, 2]

  def setUp(self):
    super(IncompressibleStructuredMeshNumericsTest, self).setUp()

    self.f = [
        tf.constant([[2, 3, 5, 6], [3, 4, 6, 7], [5, 6, 8, 9], [6, 7, 9, 10]],
                    tf.float32),
        tf.constant([[3, 4, 6, 7], [4, 5, 7, 8], [6, 7, 9, 10], [7, 8, 10, 11]],
                    tf.float32),
        tf.constant(
            [[5, 6, 8, 9], [6, 7, 9, 10], [8, 9, 11, 12], [9, 10, 12, 13]],
            tf.float32),
        tf.constant(
            [[6, 7, 9, 10], [7, 8, 10, 11], [9, 10, 12, 13], [10, 11, 13, 14]],
            tf.float32)
    ]

  @parameterized.parameters(*zip(MESH_SIZES, DIRECTIONS))
  @test_util.run_in_graph_and_eager_modes
  def testPositiveVelocityGivesBackwardDifference(self, mesh_size, direction):
    u = [
        tf.constant(1.0, shape=self.f[i].shape.as_list())
        for i in range(len(self.f))
    ]

    dfdh = self.evaluate(
        convection.first_order_upwinding(
            get_kernel_fn.ApplyKernelConvOp(4), self.f, self.f, u, mesh_size,
            direction))

    if direction == 0:
      tile_1 = np.array(
          [[2, 3, 5, 6], [1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1]],
          dtype=np.float32)
      tile_2 = np.array(
          [[3, 4, 6, 7], [1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1]],
          dtype=np.float32)
      tile_3 = np.array(
          [[5, 6, 8, 9], [1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1]],
          dtype=np.float32)
      tile_4 = np.array(
          [[6, 7, 9, 10], [1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1]],
          dtype=np.float32)
    elif direction == 1:
      tile_1 = np.array(
          [[2, 1, 2, 1], [3, 1, 2, 1], [5, 1, 2, 1], [6, 1, 2, 1]],
          dtype=np.float32)
      tile_2 = np.array(
          [[3, 1, 2, 1], [4, 1, 2, 1], [6, 1, 2, 1], [7, 1, 2, 1]],
          dtype=np.float32)
      tile_3 = np.array(
          [[5, 1, 2, 1], [6, 1, 2, 1], [8, 1, 2, 1], [9, 1, 2, 1]],
          dtype=np.float32)
      tile_4 = np.array(
          [[6, 1, 2, 1], [7, 1, 2, 1], [9, 1, 2, 1], [10, 1, 2, 1]],
          dtype=np.float32)
    elif direction == 2:
      tile_1 = np.array(
          [[2, 3, 5, 6], [3, 4, 6, 7], [5, 6, 8, 9], [6, 7, 9, 10]],
          dtype=np.float32)
      tile_2 = np.array(
          [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
          dtype=np.float32)
      tile_3 = np.array(
          [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]],
          dtype=np.float32)
      tile_4 = np.array(
          [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
          dtype=np.float32)

    expected = [tile_1, tile_2, tile_3, tile_4]

    self.assertLen(dfdh, len(self.f))
    for i in range(len(self.f)):
      self.assertAllEqual(expected[i], dfdh[i])

  @parameterized.parameters(*zip(MESH_SIZES, DIRECTIONS))
  @test_util.run_in_graph_and_eager_modes
  def testNegativeVelocityGivesForwardDifference(self, mesh_size, direction):
    u = [
        tf.constant(-1.0, shape=self.f[i].shape.as_list())
        for i in range(len(self.f))
    ]

    dfdh = self.evaluate(
        convection.first_order_upwinding(
            get_kernel_fn.ApplyKernelConvOp(4), self.f, self.f, u, mesh_size,
            direction))

    if direction == 0:
      tile_1 = np.array(
          [[1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1], [-6, -7, -9, -10]],
          dtype=np.float32)
      tile_2 = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1],
                         [-7, -8, -10, -11]],
                        dtype=np.float32)
      tile_3 = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1],
                         [-9, -10, -12, -13]],
                        dtype=np.float32)
      tile_4 = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1],
                         [-10, -11, -13, -14]],
                        dtype=np.float32)
    elif direction == 1:
      tile_1 = np.array(
          [[1, 2, 1, -6], [1, 2, 1, -7], [1, 2, 1, -9], [1, 2, 1, -10]],
          dtype=np.float32)
      tile_2 = np.array([[1, 2, 1, -7], [1, 2, 1, -8], [1, 2, 1, -10],
                         [1, 2, 1, -11]],
                        dtype=np.float32)
      tile_3 = np.array([[1, 2, 1, -9], [1, 2, 1, -10], [1, 2, 1, -12],
                         [1, 2, 1, -13]],
                        dtype=np.float32)
      tile_4 = np.array([[1, 2, 1, -10], [1, 2, 1, -11], [1, 2, 1, -13],
                         [1, 2, 1, -14]],
                        dtype=np.float32)
    elif direction == 2:
      tile_1 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
                         [1, 1, 1, 1]],
                        dtype=np.float32)
      tile_2 = np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2],
                         [2, 2, 2, 2]],
                        dtype=np.float32)
      tile_3 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
                         [1, 1, 1, 1]],
                        dtype=np.float32)
      tile_4 = np.array([[6, 7, 9, 10], [7, 8, 10, 11],
                         [9, 10, 12, 13], [10, 11, 13, 14]],
                        dtype=np.float32)

    expected = [tile_1, tile_2, tile_3, tile_4]

    self.assertLen(dfdh, len(self.f))
    for i in range(len(self.f)):
      self.assertAllEqual(expected[i], dfdh[i])

  @parameterized.parameters(*zip(MESH_SIZES, DIRECTIONS))
  @test_util.run_in_graph_and_eager_modes
  def testMixedVelocityGivesUpwindingDifference(self, mesh_size, direction):
    u = [
        tf.constant(-1.0, shape=self.f[0].shape.as_list()),
        tf.constant(1.0, shape=self.f[1].shape.as_list()),
        tf.constant(-1.0, shape=self.f[2].shape.as_list()),
        tf.constant(1.0, shape=self.f[3].shape.as_list())
    ]

    dfdh = self.evaluate(
        convection.first_order_upwinding(
            get_kernel_fn.ApplyKernelConvOp(4), self.f, self.f, u, mesh_size,
            direction))

    if direction == 0:
      tile_1 = np.array(
          [[1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1], [-6, -7, -9, -10]],
          dtype=np.float32)
      tile_2 = np.array(
          [[3, 4, 6, 7], [1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1]],
          dtype=np.float32)
      tile_3 = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1],
                         [-9, -10, -12, -13]],
                        dtype=np.float32)
      tile_4 = np.array(
          [[6, 7, 9, 10], [1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1]],
          dtype=np.float32)
    elif direction == 1:
      tile_1 = np.array(
          [[1, 2, 1, -6], [1, 2, 1, -7], [1, 2, 1, -9], [1, 2, 1, -10]],
          dtype=np.float32)
      tile_2 = np.array(
          [[3, 1, 2, 1], [4, 1, 2, 1], [6, 1, 2, 1], [7, 1, 2, 1]],
          dtype=np.float32)
      tile_3 = np.array([[1, 2, 1, -9], [1, 2, 1, -10], [1, 2, 1, -12],
                         [1, 2, 1, -13]],
                        dtype=np.float32)
      tile_4 = np.array(
          [[6, 1, 2, 1], [7, 1, 2, 1], [9, 1, 2, 1], [10, 1, 2, 1]],
          dtype=np.float32)
    elif direction == 2:
      tile_1 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
                         [1, 1, 1, 1]],
                        dtype=np.float32)
      tile_2 = np.array(
          [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
          dtype=np.float32)
      tile_3 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
                         [1, 1, 1, 1]],
                        dtype=np.float32)
      tile_4 = np.array(
          [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
          dtype=np.float32)

    expected = [tile_1, tile_2, tile_3, tile_4]

    self.assertLen(dfdh, len(self.f))
    for i in range(len(self.f)):
      self.assertAllEqual(expected[i], dfdh[i])

  @parameterized.parameters(DIRECTIONS)
  @test_util.run_in_graph_and_eager_modes
  def testFaceInterpolationProducesCorrectValuesOnTheFace(self, dim):
    """Tests face interpolation at (4, 4, 4) is correct."""
    state = tf.unstack(
        tf.constant(np.reshape(np.arange(512), (8, 8, 8)), dtype=tf.float32))
    pressure = tf.unstack(
        tf.constant(np.reshape(np.arange(512), (8, 8, 8)), dtype=tf.float32))
    dx = 1.0
    dt = 4.0

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])

    f_face = convection.face_interpolation(
        get_kernel_fn.ApplyKernelConvOp(4), replica_id, replicas, state,
        pressure, dx, dt, dim)

    f_face_val = self.evaluate(f_face)

    if dim == 0:
      self.assertEqual(f_face_val[4][4, 4], 288.0)
    elif dim == 1:
      self.assertEqual(f_face_val[4][4, 4], 291.5)
    else:
      self.assertEqual(f_face_val[4][4, 4], 260.0)

  @parameterized.parameters(DIRECTIONS)
  @test_util.run_in_graph_and_eager_modes
  def testFaceInterpolationWithSourceProducesCorrectValuesOnTheFace(self, dim):
    """Tests face interpolation at (4, 4, 4) is correct."""
    state = tf.unstack(
        tf.constant(np.reshape(np.arange(512), (8, 8, 8)), dtype=tf.float32))
    pressure = tf.unstack(
        tf.constant(np.reshape(np.arange(512), (8, 8, 8)), dtype=tf.float32))
    src = np.reshape(np.arange(512), (8, 8, 8))
    src[2:6, 4, 4] = [6, 8, 25, -7]
    src[4, 2:6, 4] = [36, 64, 25, 49]
    src[4, 4, 2:6] = [-9, 16, 25, 81]
    src = tf.unstack(tf.convert_to_tensor(src, dtype=tf.float32))
    dx = 1.0
    dt = 4.0

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])

    f_face = convection.face_interpolation(
        get_kernel_fn.ApplyKernelConvOp(4), replica_id, replicas, state,
        pressure, dx, dt, dim, src=src)

    f_face_val = self.evaluate(f_face)

    if dim == 0:
      self.assertEqual(f_face_val[4][4, 4], 290.0)
    elif dim == 1:
      self.assertEqual(f_face_val[4][4, 4], 276.0)
    else:
      self.assertEqual(f_face_val[4][4, 4], 277.0)

  _REPLICAS = (
      np.array([[[0, 1]]]),
      np.array([[[0], [1]]]),
      np.array([[[0]], [[1]]]),
  )
  _DIM = (0, 1, 2)
  _VARNAME = ('u', 'v', 'w', 'rho_u', 'rho_v', 'rho_w')
  _LOWER_WALL_BC_TYPES = (
      (boundary_condition_utils.BoundaryType.SLIP_WALL,
       boundary_condition_utils.BoundaryType.UNKNOWN),
      (boundary_condition_utils.BoundaryType.NON_SLIP_WALL,
       boundary_condition_utils.BoundaryType.UNKNOWN),
      (boundary_condition_utils.BoundaryType.SHEAR_WALL,
       boundary_condition_utils.BoundaryType.UNKNOWN),
  )

  @parameterized.parameters(*itertools.product(_REPLICAS, _DIM, _VARNAME,
                                               _LOWER_WALL_BC_TYPES))
  def testFaceInterpolationProvidesZerosForWallNormalVelocityAtLowerWalls(
      self, replicas, dim, varname, bc_types):
    """Checks if the wall normal velocity is 0 at lower wall."""
    state = tf.unstack(tf.random.uniform((8, 8, 8), dtype=tf.float32, seed=29))
    pressure = tf.unstack(
        tf.random.uniform((8, 8, 8), dtype=tf.float32, seed=6))
    src = tf.unstack(tf.random.uniform((8, 8, 8), dtype=tf.float32, seed=8))

    dx = 1.0
    dt = 4.0
    halo_width = 2

    def device_fn(replica_id):
      """The face interpolation function wrapper for TPU."""
      return convection.face_interpolation(
          get_kernel_fn.ApplyKernelConvOp(4), replica_id, replicas, state,
          pressure, dx, dt, dim, bc_types, varname, halo_width, src)

    inputs = [[tf.constant(0)], [tf.constant(1)]]
    device_inputs = [list(x) for x in zip(*inputs)]

    computation_shape = np.array(replicas.shape)
    runner = TpuRunner(computation_shape=computation_shape)
    output = runner.run(device_fn, *device_inputs)
    output_0 = np.array(output[0])
    output_1 = np.array(output[1])

    zeros = np.zeros((8, 8), dtype=np.float32)
    if varname in ('u', 'rho_u'):
      if dim == 0:
        self.assertAllEqual(zeros, np.squeeze(output_0[:, 2, :]))
        if computation_shape[0] == 1:
          self.assertAllEqual(zeros, np.squeeze(output_1[:, 2, :]))
        else:
          self.assertNotAllEqual(zeros, np.squeeze(output_1[:, 2, :]))
      else:
        self.assertNotAllEqual(zeros, np.squeeze(output_0[:, 2, :]))
        self.assertNotAllEqual(zeros, np.squeeze(output_1[:, 2, :]))
    elif varname in ('v', 'rho_v'):
      if dim == 1:
        self.assertAllEqual(zeros, np.squeeze(output_0[:, :, 2]))
        if computation_shape[1] == 1:
          self.assertAllEqual(zeros, np.squeeze(output_1[:, :, 2]))
        else:
          self.assertNotAllEqual(zeros, np.squeeze(output_1[:, :, 2]))
      else:
        self.assertNotAllEqual(zeros, np.squeeze(output_0[:, :, 2]))
        self.assertNotAllEqual(zeros, np.squeeze(output_1[:, :, 2]))
    elif varname in ('w', 'rho_w'):
      if dim == 2:
        self.assertAllEqual(zeros, np.squeeze(output_0[2, :, :]))
        if computation_shape[2] == 1:
          self.assertAllEqual(zeros, np.squeeze(output_1[2, :, :]))
        else:
          self.assertNotAllEqual(zeros, np.squeeze(output_1[2, :, :]))
      else:
        self.assertNotAllEqual(zeros, np.squeeze(output_0[2, :, :]))
        self.assertNotAllEqual(zeros, np.squeeze(output_1[2, :, :]))

  _HIGHER_WALL_BC_TYPES = (
      (boundary_condition_utils.BoundaryType.UNKNOWN,
       boundary_condition_utils.BoundaryType.SLIP_WALL),
      (boundary_condition_utils.BoundaryType.UNKNOWN,
       boundary_condition_utils.BoundaryType.NON_SLIP_WALL),
      (boundary_condition_utils.BoundaryType.UNKNOWN,
       boundary_condition_utils.BoundaryType.SHEAR_WALL),
  )

  @parameterized.parameters(*itertools.product(_REPLICAS, _DIM, _VARNAME,
                                               _HIGHER_WALL_BC_TYPES))
  def testFaceInterpolationProvidesZerosForWallNormalVelocityAtHigherWalls(
      self, replicas, dim, varname, bc_types):
    """Checks if the wall normal velocity is 0 at higher wall."""
    state = tf.unstack(tf.random.uniform((8, 8, 8), dtype=tf.float32, seed=29))
    pressure = tf.unstack(
        tf.random.uniform((8, 8, 8), dtype=tf.float32, seed=6))
    src = tf.unstack(tf.random.uniform((8, 8, 8), dtype=tf.float32, seed=8))

    dx = 1.0
    dt = 4.0
    halo_width = 2

    def device_fn(replica_id):
      """The face interpolation function wrapper for TPU."""
      return convection.face_interpolation(
          get_kernel_fn.ApplyKernelConvOp(4), replica_id, replicas, state,
          pressure, dx, dt, dim, bc_types, varname, halo_width, src)

    inputs = [[tf.constant(0)], [tf.constant(1)]]
    device_inputs = [list(x) for x in zip(*inputs)]

    computation_shape = np.array(replicas.shape)
    runner = TpuRunner(computation_shape=computation_shape)
    output = runner.run(device_fn, *device_inputs)
    output_0 = np.array(output[0])
    output_1 = np.array(output[1])

    zeros = np.zeros((8, 8), dtype=np.float32)
    if varname in ('u', 'rho_u'):
      if dim == 0:
        self.assertAllEqual(zeros, np.squeeze(output_1[:, -3, :]))
        if computation_shape[0] == 1:
          self.assertAllEqual(zeros, np.squeeze(output_0[:, -3, :]))
        else:
          self.assertNotAllEqual(zeros, np.squeeze(output_0[:, -3, :]))
      else:
        self.assertNotAllEqual(zeros, np.squeeze(output_0[:, -3, :]))
        self.assertNotAllEqual(zeros, np.squeeze(output_1[:, -3, :]))
    elif varname in ('v', 'rho_v'):
      if dim == 1:
        self.assertAllEqual(zeros, np.squeeze(output_1[:, :, -3]))
        if computation_shape[1] == 1:
          self.assertAllEqual(zeros, np.squeeze(output_0[:, :, -3]))
        else:
          self.assertNotAllEqual(zeros, np.squeeze(output_0[:, :, -3]))
      else:
        self.assertNotAllEqual(zeros, np.squeeze(output_0[:, :, -3]))
        self.assertNotAllEqual(zeros, np.squeeze(output_1[:, :, -3]))
    elif varname in ('w', 'rho_w'):
      if dim == 2:
        self.assertAllEqual(zeros, np.squeeze(output_1[-3, :, :]))
        if computation_shape[2] == 1:
          self.assertAllEqual(zeros, np.squeeze(output_0[-3, :, :]))
        else:
          self.assertNotAllEqual(zeros, np.squeeze(output_0[-3, :, :]))
      else:
        self.assertNotAllEqual(zeros, np.squeeze(output_0[-3, :, :]))
        self.assertNotAllEqual(zeros, np.squeeze(output_1[-3, :, :]))

  _BOTH_WALL_BC_TYPES = (
      (boundary_condition_utils.BoundaryType.NON_SLIP_WALL,
       boundary_condition_utils.BoundaryType.SLIP_WALL),
      (boundary_condition_utils.BoundaryType.SHEAR_WALL,
       boundary_condition_utils.BoundaryType.NON_SLIP_WALL),
      (boundary_condition_utils.BoundaryType.SLIP_WALL,
       boundary_condition_utils.BoundaryType.SHEAR_WALL),
  )

  @parameterized.parameters(*itertools.product(_REPLICAS, _DIM, _VARNAME,
                                               _BOTH_WALL_BC_TYPES))
  def testFaceInterpolationProvidesZerosForWallNormalVelocityAtBothWalls(
      self, replicas, dim, varname, bc_types):
    """Checks if the wall normal velocity is 0 at both walls."""
    state = tf.unstack(tf.random.uniform((8, 8, 8), dtype=tf.float32, seed=29))
    pressure = tf.unstack(
        tf.random.uniform((8, 8, 8), dtype=tf.float32, seed=6))
    src = tf.unstack(tf.random.uniform((8, 8, 8), dtype=tf.float32, seed=8))

    dx = 1.0
    dt = 4.0
    halo_width = 2

    def device_fn(replica_id):
      """The face interpolation function wrapper for TPU."""
      return convection.face_interpolation(
          get_kernel_fn.ApplyKernelConvOp(4), replica_id, replicas, state,
          pressure, dx, dt, dim, bc_types, varname, halo_width, src)

    inputs = [[tf.constant(0)], [tf.constant(1)]]
    device_inputs = [list(x) for x in zip(*inputs)]

    computation_shape = np.array(replicas.shape)
    runner = TpuRunner(computation_shape=computation_shape)
    output = runner.run(device_fn, *device_inputs)
    output_0 = np.array(output[0])
    output_1 = np.array(output[1])

    zeros = np.zeros((8, 8), dtype=np.float32)
    if varname in ('u', 'rho_u'):
      if dim == 0:
        self.assertAllEqual(zeros, np.squeeze(output_0[:, 2, :]))
        self.assertAllEqual(zeros, np.squeeze(output_1[:, -3, :]))
        if computation_shape[0] == 1:
          self.assertAllEqual(zeros, np.squeeze(output_0[:, -3, :]))
          self.assertAllEqual(zeros, np.squeeze(output_1[:, 2, :]))
        else:
          self.assertNotAllEqual(zeros, np.squeeze(output_0[:, -3, :]))
          self.assertNotAllEqual(zeros, np.squeeze(output_1[:, 2, :]))
      else:
        self.assertNotAllEqual(zeros, np.squeeze(output_0[:, 2, :]))
        self.assertNotAllEqual(zeros, np.squeeze(output_1[:, 2, :]))
        self.assertNotAllEqual(zeros, np.squeeze(output_0[:, -3, :]))
        self.assertNotAllEqual(zeros, np.squeeze(output_1[:, -3, :]))
    elif varname in ('v', 'rho_v'):
      if dim == 1:
        self.assertAllEqual(zeros, np.squeeze(output_0[:, :, 2]))
        self.assertAllEqual(zeros, np.squeeze(output_1[:, :, -3]))
        if computation_shape[1] == 1:
          self.assertAllEqual(zeros, np.squeeze(output_0[:, :, -3]))
          self.assertAllEqual(zeros, np.squeeze(output_1[:, :, 2]))
        else:
          self.assertNotAllEqual(zeros, np.squeeze(output_0[:, :, -3]))
          self.assertNotAllEqual(zeros, np.squeeze(output_1[:, :, 2]))
      else:
        self.assertNotAllEqual(zeros, np.squeeze(output_0[:, :, 2]))
        self.assertNotAllEqual(zeros, np.squeeze(output_1[:, :, 2]))
        self.assertNotAllEqual(zeros, np.squeeze(output_0[:, :, -3]))
        self.assertNotAllEqual(zeros, np.squeeze(output_1[:, :, -3]))
    elif varname in ('w', 'rho_w'):
      if dim == 2:
        self.assertAllEqual(zeros, np.squeeze(output_0[2, :, :]))
        self.assertAllEqual(zeros, np.squeeze(output_1[-3, :, :]))
        if computation_shape[2] == 1:
          self.assertAllEqual(zeros, np.squeeze(output_0[-3, :, :]))
          self.assertAllEqual(zeros, np.squeeze(output_1[2, :, :]))
        else:
          self.assertNotAllEqual(zeros, np.squeeze(output_0[-3, :, :]))
          self.assertNotAllEqual(zeros, np.squeeze(output_1[2, :, :]))
      else:
        self.assertNotAllEqual(zeros, np.squeeze(output_0[2, :, :]))
        self.assertNotAllEqual(zeros, np.squeeze(output_1[2, :, :]))
        self.assertNotAllEqual(zeros, np.squeeze(output_0[-3, :, :]))
        self.assertNotAllEqual(zeros, np.squeeze(output_1[-3, :, :]))

  @parameterized.parameters(DIRECTIONS)
  @test_util.run_in_graph_and_eager_modes
  def testFaceFluxQuickProducesCorrectValues(self, dim):
    """Tests face flux at (4, 4, 4) is correct."""
    state = tf.unstack(
        tf.constant(np.reshape(np.arange(512), (8, 8, 8)), dtype=tf.float32))
    rhou = tf.unstack(
        tf.constant(np.reshape(np.arange(512), (8, 8, 8)), dtype=tf.float32))
    pressure = tf.unstack(
        tf.constant(np.reshape(np.arange(512), (8, 8, 8)), dtype=tf.float32))
    dx = 1.0
    dt = 4.0

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])

    f_face = convection.face_flux_quick(replica_id, replicas, state, rhou,
                                        pressure, dx, dt, dim)

    f_face_val = self.evaluate(f_face)

    if dim == 0:
      self.assertEqual(f_face_val[4][4, 4], 82944.0)
    elif dim == 1:
      self.assertEqual(f_face_val[4][4, 4], 84972.25)
    else:
      self.assertEqual(f_face_val[4][4, 4], 67600.0)

  @parameterized.parameters(DIRECTIONS)
  @test_util.run_in_graph_and_eager_modes
  def testConvectionQuickProducesCorrectionConvectionTerm(self, dim):
    """Tests convection term at (4, 4, 4) is correct."""
    state = tf.unstack(
        tf.constant(np.reshape(np.arange(512), (8, 8, 8)), dtype=tf.float32))
    pressure = tf.unstack(
        tf.constant(np.reshape(np.arange(512), (8, 8, 8)), dtype=tf.float32))
    rhou = tf.unstack(
        tf.constant(np.reshape(np.arange(512), (8, 8, 8)), dtype=tf.float32))
    dx = 1.0
    dt = 4.0

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])

    conv_terms = convection.convection_quick(
        get_kernel_fn.ApplyKernelConvOp(4), replica_id, replicas, state, rhou,
        pressure, dx, dt, dim)

    convection_val = self.evaluate(conv_terms)

    if dim == 0:
      self.assertEqual(convection_val[4][4, 4], 4672.0)
    elif dim == 1:
      self.assertEqual(convection_val[4][4, 4], 584.0)
    else:
      self.assertEqual(convection_val[4][4, 4], 37376.0)

  @parameterized.parameters(*zip(MESH_SIZES, DIRECTIONS))
  @test_util.run_in_graph_and_eager_modes
  def testCentralDifferenceGivesCorrectFirstOrderDerivative(
      self, mesh_size, direction):
    dfdh = self.evaluate(
        convection.central2(
            get_kernel_fn.ApplyKernelConvOp(4), self.f, mesh_size, direction))

    if direction == 0:
      tile_1 = 0.5 * np.array(
          [[3, 4, 6, 7], [3, 3, 3, 3], [3, 3, 3, 3], [-5, -6, -8, -9]],
          dtype=np.float32)
      tile_2 = 0.5 * np.array(
          [[4, 5, 7, 8], [3, 3, 3, 3], [3, 3, 3, 3], [-6, -7, -9, -10]],
          dtype=np.float32)
      tile_3 = 0.5 * np.array(
          [[6, 7, 9, 10], [3, 3, 3, 3], [3, 3, 3, 3], [-8, -9, -11, -12]],
          dtype=np.float32)
      tile_4 = 0.5 * np.array(
          [[7, 8, 10, 11], [3, 3, 3, 3], [3, 3, 3, 3], [-9, -10, -12, -13]],
          dtype=np.float32)
    elif direction == 1:
      tile_1 = 0.5 * np.array(
          [[3, 3, 3, -5], [4, 3, 3, -6], [6, 3, 3, -8], [7, 3, 3, -9]],
          dtype=np.float32)
      tile_2 = 0.5 * np.array(
          [[4, 3, 3, -6], [5, 3, 3, -7], [7, 3, 3, -9], [8, 3, 3, -10]],
          dtype=np.float32)
      tile_3 = 0.5 * np.array(
          [[6, 3, 3, -8], [7, 3, 3, -9], [9, 3, 3, -11], [10, 3, 3, -12]],
          dtype=np.float32)
      tile_4 = 0.5 * np.array(
          [[7, 3, 3, -9], [8, 3, 3, -10], [10, 3, 3, -12], [11, 3, 3, -13]],
          dtype=np.float32)
    elif direction == 2:
      tile_1 = 0.5 * np.array(
          [[2, 3, 5, 6], [3, 4, 6, 7], [5, 6, 8, 9], [6, 7, 9, 10]], np.float32)
      tile_2 = 1.5 * np.ones((4, 4), dtype=np.float32)
      tile_3 = 1.5 * np.ones((4, 4), dtype=np.float32)
      tile_4 = 0.5 * np.array([[6, 7, 9, 10], [7, 8, 10, 11], [9, 10, 12, 13],
                               [10, 11, 13, 14]], np.float32)

    expected = [tile_1, tile_2, tile_3, tile_4]

    self.assertLen(dfdh, len(self.f))
    for i in range(len(self.f)):
      self.assertAllEqual(expected[i], dfdh[i])


if __name__ == '__main__':
  tf.test.main()
