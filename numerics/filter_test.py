"""Tests for filter."""

import functools
import itertools
import numpy as np
from swirl_lm.communication import halo_exchange
from swirl_lm.numerics import filters
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized


class FilterTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Initializes common tensors."""
    super(FilterTest, self).setUp()

    nx = 64
    ny = 32
    nz = 8
    kernel_size = 8
    self.kernel_op = get_kernel_fn.ApplyKernelConvOp(kernel_size)

    # Generate the mesh.
    x = np.linspace(0, 2.0 * np.pi, nx)
    y = np.linspace(0, 2.0 * np.pi, ny)
    z = np.linspace(0, 2.0 * np.pi, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Prepare the input.
    f = np.cos(xx) * np.sin(yy) * np.cos(zz)
    self.f = np.transpose(f, (2, 0, 1))
    self.tf_f = tf.unstack(tf.convert_to_tensor(self.f))

  @test_util.run_in_graph_and_eager_modes
  def testFilter2WithStencil27ProvidesCorrectSecondOrderFiltering(self):
    """Checks if the second order filtering is performed correctly."""
    res = self.evaluate(
        filters.filter_2(self.kernel_op, self.tf_f, 27))

    expected = np.copy(self.f)
    fx = self.f[1:-1, ...] + 0.25 * (
        self.f[:-2, ...] - 2.0 * self.f[1:-1, ...] + self.f[2:, ...])
    fy = fx[:, 1:-1, ...] + 0.25 * (
        fx[:, :-2, :] - 2.0 * fx[:, 1:-1, :] + fx[:, 2:, :])
    fz = fy[..., 1:-1] + 0.25 * (
        fy[..., :-2] - 2.0 * fy[..., 1:-1] + fy[..., 2:])
    expected[1:-1, 1:-1, 1:-1] = fz

    self.assertAllClose(expected, res)

  @test_util.run_in_graph_and_eager_modes
  def testFilter2WithStencil7ProvidesCorrectSecondOrderFiltering(self):
    """Checks if the second order filtering is performed correctly."""
    res = self.evaluate(
        filters.filter_2(self.kernel_op, self.tf_f, 7))

    expected = np.copy(self.f)
    expected[1:-1, 1:-1, 1:-1] = 0.5 * self.f[1:-1, 1:-1, 1:-1] + 1.0 / 12.0 * (
        self.f[:-2, 1:-1, 1:-1] + self.f[2:, 1:-1, 1:-1] +
        self.f[1:-1, :-2, 1:-1] + self.f[1:-1, 2:, 1:-1] +
        self.f[1:-1, 1:-1, :-2] + self.f[1:-1, 1:-1, 2:])

    self.assertAllClose(expected, res)

  _REPLICAS = [
      np.array([[[0], [1]]]), np.array([[[0, 1]]]), np.array([[[0]], [[1]]])
  ]
  _NUM_ITER = [0, 1, 3]
  _FILTER_WIDTH = [3, 5]

  @parameterized.parameters(
      *itertools.product(_REPLICAS, _NUM_ITER, _FILTER_WIDTH))
  def testGlobalFilter3DPreservesConstantField(self, replicas, num_iter,
                                               filter_width):
    """Tests filtering a constant field gives the same constant."""
    computation_shape = np.array(replicas.shape)
    num_replicas = np.prod(computation_shape)

    def halo_update_fn(state, replica_id):
      """Applies periodic boundary condition to `f`."""
      return halo_exchange.inplace_halo_exchange(
          state, [0, 1, 2],
          replica_id,
          replicas, [0, 1, 2], [True, True, True],
          boundary_conditions=None,
          width=filter_width // 2)

    inputs = []
    for _ in range(num_replicas):
      args = []

      state = [tf.ones((8, 8), dtype=tf.float32)] * 8
      args.append(state)

      inputs.append(args)

    filter_fn = functools.partial(
        filters.global_box_filter_3d,
        halo_update_fn=functools.partial(
            halo_update_fn, replica_id=tf.constant(0)),
        filter_width=filter_width,
        num_iter=num_iter)

    # transpose inputs.
    device_inputs = [list(x) for x in zip(*inputs)]
    computation_shape = replicas.shape
    runner = TpuRunner(computation_shape=computation_shape)
    output = runner.run(filter_fn, *device_inputs)

    with self.subTest(name='NumberOfReplicaOutputsIsTwo'):
      self.assertLen(output, 2)

    with self.subTest(name='OutputInReplicaOneIsOnes'):
      self.assertAllEqual(np.stack(output[0]), np.ones((8, 8, 8)))

    with self.subTest(name='OutputInReplicaTwoIsOnes'):
      self.assertAllEqual(np.stack(output[1]), np.ones((8, 8, 8)))

  def testGlobalFilter3DProducesCorrectResultWithFilterWidthThree(self):
    """Tests filter with width 3 produces correct result."""
    replicas = np.array([[[0], [1]]])
    computation_shape = np.array(replicas.shape)
    filter_width = 3

    def halo_update_fn(state):
      """Applies periodic boundary condition to `f`."""
      replica_id = tf.compat.v1.where(
          tf.math.reduce_sum(state) >= 18.0, tf.constant(0), tf.constant(1))
      return halo_exchange.inplace_halo_exchange(
          state, [0, 1, 2],
          replica_id,
          replicas, [0, 1, 2], [False, False, False],
          boundary_conditions=[[(halo_exchange.BCType.DIRICHLET, 0.0)] * 2] * 3,
          width=filter_width // 2)

    buf0 = np.zeros((8, 8, 8), dtype=np.float32)
    buf1 = np.zeros((8, 8, 8), dtype=np.float32)
    buf0[4, 6, 4] = 27.0
    buf1[4, 0, 4] = 27.0
    inputs = [[tf.unstack(tf.convert_to_tensor(buf0))],
              [tf.unstack(tf.convert_to_tensor(buf1))]]

    filter_fn = functools.partial(
        filters.global_box_filter_3d,
        halo_update_fn=halo_update_fn,
        filter_width=filter_width,
        num_iter=1)

    # transpose inputs.
    device_inputs = [list(x) for x in zip(*inputs)]
    computation_shape = replicas.shape
    runner = TpuRunner(computation_shape=computation_shape)
    output = runner.run(filter_fn, *device_inputs)

    with self.subTest(name='NumberOfReplicaOutputsIsTwo'):
      self.assertLen(output, 2)

    with self.subTest(name='OutputInReplicaOneIsCorrect'):
      print(output[0])
      expected = np.zeros((8, 8, 8), dtype=np.float32)
      expected[3:6, 5:-1, 3:6] = 1.0
      self.assertAllEqual(np.stack(output[0]), expected)

    with self.subTest(name='OutputInReplicaTwoIsCorrect'):
      print(output[1])
      expected = np.zeros((8, 8, 8), dtype=np.float32)
      expected[3:6, 1:2, 3:6] = 1.0
      self.assertAllEqual(np.stack(output[1]), expected)

  def testGlobalFilter3DRaisesValueErrorForEvenFilterWidth(self):
    """Tests filter with width 4 raises `ValueError`."""
    replicas = np.array([[[0], [1]]])
    computation_shape = np.array(replicas.shape)
    filter_width = 4

    dummy_halo_update_fn = lambda state: state

    buf0 = np.zeros((8, 8, 8), dtype=np.float32)
    buf1 = np.zeros((8, 8, 8), dtype=np.float32)
    inputs = [[tf.unstack(tf.convert_to_tensor(buf0))],
              [tf.unstack(tf.convert_to_tensor(buf1))]]

    filter_fn = functools.partial(
        filters.global_box_filter_3d,
        halo_update_fn=dummy_halo_update_fn,
        filter_width=filter_width,
        num_iter=1)

    # transpose inputs.
    device_inputs = [list(x) for x in zip(*inputs)]
    computation_shape = replicas.shape
    with self.assertRaisesRegex(ValueError,
                                'Filter width has to be an odd number.'):
      runner = TpuRunner(computation_shape=computation_shape)
      runner.run(filter_fn, *device_inputs)


if __name__ == '__main__':
  tf.test.main()
