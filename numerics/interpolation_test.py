"""Tests for interpolation."""

from absl.testing import parameterized
import numpy as np
from swirl_lm.numerics import interpolation
import tensorflow as tf


DIMS = ('x', 'y', 'z')


class InterpolationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(*DIMS)
  def test_weno5_performs_correct_reconstruction(self, dim):
    """Checks if the 5th order WENO scheme reconstructs correct face values."""
    # Construct a sine function and align it along the dimension of test.
    nx = 16
    dx = 2 * np.pi / (nx - 5)
    x = dx * np.arange(nx, dtype=np.float32) - 3 * dx
    y = np.sin(x)

    if dim == 'x':
      y = np.tile(y[np.newaxis, :, np.newaxis], (nx, 1, nx))
    elif dim == 'y':
      y = np.tile(y[np.newaxis, np.newaxis, :], (nx, nx, 1))
    else:  # dim == 'z':
      y = np.tile(y[:, np.newaxis, np.newaxis], (1, nx, nx))

    v = tf.unstack(tf.convert_to_tensor(y))

    # Compute the reconstructed face fluxes.
    v_neg, v_pos = self.evaluate(interpolation.weno(v, dim, 3))

    # Check the values inside the domain only, i.e. excluding halo layers of
    # width 3 on both ends.
    if dim == 'x':
      v_neg_res = np.squeeze(np.stack(v_neg)[nx // 2, 3:-3, nx // 2])
      v_pos_res = np.squeeze(np.stack(v_pos)[nx // 2, 3:-3, nx // 2])
    elif dim == 'y':
      v_neg_res = np.squeeze(np.stack(v_neg)[nx // 2, nx // 2, 3:-3])
      v_pos_res = np.squeeze(np.stack(v_pos)[nx // 2, nx // 2, 3:-3])
    else:  # if dim == 'z':
      v_neg_res = np.squeeze(np.stack(v_neg)[3:-3, nx // 2, nx // 2])
      v_pos_res = np.squeeze(np.stack(v_pos)[3:-3, nx // 2, nx // 2])

    print(f'v_neg = {v_neg_res}')
    print(f'v_pos = {v_pos_res}')

    with self.subTest(name='v-'):
      expected = [
          0.27891243, 0.76181704, 1.0051062, 0.92013156, 0.54974455, 0.00615088,
          -0.5419955, -0.9206312, -1.0050877, -0.7646685
      ]
      self.assertAllClose(expected, v_neg_res)

    with self.subTest(name='v+'):
      expected = [
          0.29005462, 0.76466835, 1.0050876, 0.92063123, 0.5419959, -0.00615061,
          -0.5497443, -0.9201313, -1.0051063, -0.76181716
      ]
      self.assertAllClose(expected, v_pos_res)


if __name__ == '__main__':
  tf.test.main()
