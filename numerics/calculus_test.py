"""Tests for calculus."""

import itertools
import numpy as np
from swirl_lm.numerics import calculus
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf
from google3.testing.pybase import parameterized


class CalculusTest(tf.test.TestCase, parameterized.TestCase):

  _DIM = [0, 1, 2]
  _DX = [2.0, 3.0, 1.0]

  @parameterized.parameters(*zip(_DIM, _DX))
  @test_util.run_in_graph_and_eager_modes
  def testGradProducesTheCorrectGradient(self, dim, dx):
    """Test the gradient at [6, 6, 6] is 1."""
    u = np.zeros((8, 8, 8), dtype=np.float32)
    u[5, 6, 6] = 1.0
    u[7, 6, 6] = 3.0
    u[6, 5, 6] = 2.0
    u[6, 7, 6] = 6.0
    u[6, 6, 5] = -4.0
    u[6, 6, 7] = 2.0
    u = tf.unstack(tf.convert_to_tensor(u))

    grad_u = calculus._grad_impl(get_kernel_fn.ApplyKernelConvOp(4), u, dim, dx)
    grad_u_val = self.evaluate(grad_u)

    self.assertEqual(grad_u_val[6][6, 6], 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testGradForAllProducesTheCorrectGradientVectors(self):
    """Test gradients at [6, 6, 6] for `u` and `v` is 1 in all directions."""
    u = np.zeros((8, 8, 8), dtype=np.float32)
    u[5, 6, 6] = 1.0
    u[7, 6, 6] = 3.0
    u[6, 5, 6] = 2.0
    u[6, 7, 6] = 6.0
    u[6, 6, 5] = -4.0
    u[6, 6, 7] = 2.0
    u = tf.unstack(tf.convert_to_tensor(u))

    v = np.zeros((8, 8, 8), dtype=np.float32)
    v[5, 6, 6] = 2.0
    v[7, 6, 6] = 4.0
    v[6, 5, 6] = 3.0
    v[6, 7, 6] = 7.0
    v[6, 6, 5] = -3.0
    v[6, 6, 7] = 3.0
    v = tf.unstack(tf.convert_to_tensor(v))

    grads = calculus.grad(get_kernel_fn.ApplyKernelConvOp(4), (u, v), self._DX)
    grad_vals = self.evaluate(grads)

    for i, j in itertools.product(range(2), range(3)):
      self.assertEqual(grad_vals[i][j][6][6, 6], 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testDivergenceAccuracyForVectorOfLengthThree(self):
    """Checks if the divergence of a length 3 vector is computed correctly."""
    u = np.zeros((8, 8, 8), dtype=np.float32)
    u[6, 5, 6] = 0.0
    u[6, 7, 6] = 6.0
    u = tf.unstack(tf.convert_to_tensor(u))

    v = np.zeros((8, 8, 8), dtype=np.float32)
    v[6, 6, 5] = -3.0
    v[6, 6, 7] = 3.0
    v = tf.unstack(tf.convert_to_tensor(v))

    w = np.zeros((8, 8, 8), dtype=np.float32)
    w[5, 6, 6] = 1.0
    w[7, 6, 6] = 5.0
    w = tf.unstack(tf.convert_to_tensor(w))

    div = calculus.divergence(
        get_kernel_fn.ApplyKernelConvOp(4), (u, v, w), self._DX)

    div_val = self.evaluate(div)

    self.assertAllClose(4.5, div_val[6][6, 6])

  @test_util.run_in_graph_and_eager_modes
  def testLaplacianOutputsCorrectTensor(self):
    nu = 1e-4
    dx = 0.1
    dy = 0.2
    dz = 0.5

    f = [
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

    # d^2f / dx^2:
    ddx = np.array(
        [[[-1, -2, -4, -5], [1, 1, 1, 1], [-1, -1, -1, -1], [-7, -8, -10, -11]],
         [[-2, -3, -5, -6], [1, 1, 1, 1], [-1, -1, -1, -1], [-8, -9, -11, -12]],
         [[-4, -5, -7, -8], [1, 1, 1, 1], [-1, -1, -1, -1],
          [-10, -11, -13, -14]],
         [[-5, -6, -8, -9], [1, 1, 1, 1], [-1, -1, -1, -1],
          [-11, -12, -14, -15]]],
        dtype=np.float32)
    # d^2f / dy^2:
    ddy = np.array(
        [[[-1, 1, -1, -7], [-2, 1, -1, -8], [-4, 1, -1, -10], [-5, 1, -1, -11]],
         [[-2, 1, -1, -8], [-3, 1, -1, -9], [-5, 1, -1, -11], [-6, 1, -1, -12]],
         [[-4, 1, -1, -10], [-5, 1, -1, -11], [-7, 1, -1, -13],
          [-8, 1, -1, -14]],
         [[-5, 1, -1, -11], [-6, 1, -1, -12], [-8, 1, -1, -14],
          [-9, 1, -1, -15]]],
        dtype=np.float32)
    # d^2f / dz^2:
    ddz = np.array(
        [[[2, 3, 5, 6], [3, 4, 6, 7], [5, 6, 8, 9], [6, 7, 9, 10]],
         [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
         [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
          [-1, -1, -1, -1]],
         [[6, 7, 9, 10], [7, 8, 10, 11], [9, 10, 12, 13], [10, 11, 13, 14]]],
        dtype=np.float32)

    expected = nu * (ddx / dx**2 + ddy / dy**2 + ddz / dz**2)

    diffusion = self.evaluate(
        calculus.laplacian(
            get_kernel_fn.ApplyKernelConvOp(4), f, nu, dx, dy, dz))

    self.assertLen(diffusion, len(f))
    for i in range(len(f)):
      self.assertAllEqual(expected[i], diffusion[i])


if __name__ == '__main__':
  tf.test.main()
