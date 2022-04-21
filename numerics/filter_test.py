"""Tests for filter."""

import numpy as np
from swirl_lm.numerics import filters
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf


@test_util.run_all_in_graph_and_eager_modes
class FilterTest(tf.test.TestCase):

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


if __name__ == '__main__':
  tf.test.main()
