"""Tests for halo_exchange_utils.py."""

import numpy as np
from swirl_lm.communication import halo_exchange_utils
import tensorflow as tf

from google3.testing.pybase import parameterized


class HaloExchangeUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.x = np.array([[[1., 2, 3], [4, 5, 6], [7, 8, 9]],
                       [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
                       [[21, 22, 23], [24, 25, 26], [27, 28, 29]]])

    no_touch = halo_exchange_utils.BCType.NO_TOUCH
    dirichlet = halo_exchange_utils.BCType.DIRICHLET
    neumann = halo_exchange_utils.BCType.NEUMANN
    self.boundary_conditions = [[(no_touch, 1.), (dirichlet, 2.)],
                                [(neumann, 3.), (no_touch, 4.)],
                                [(dirichlet, 5.), (neumann, 6.)]]

    # Expected after dim 0:
    #   [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    #    [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
    #    [[2, 2, 2], [2, 2, 2], [2, 2, 2]]]

    # Expected after dim 1:
    #   [[[4 - 3, 5 - 3, 6 - 3], [4, 5, 6], [7, 8, 9]],
    #    [[14 - 3, 15 - 3, 16 - 3], [14, 15, 16], [17, 18, 19]],
    #    [[2 - 3, 2 - 3, 2 - 3], [2, 2, 2], [2, 2, 2]]]

    self.expected = [[[5., 2, 2 + 6], [5, 5, 5 + 6], [5, 8, 8 + 6]],
                     [[5, 12, 12 + 6], [5, 15, 15 + 6], [5, 18, 18 + 6]],
                     [[5, -1, -1 + 6], [5, 2, 2 + 6], [5, 2, 2 + 6]]]

  def testApplyOneCoreBoundaryConditionsToTensorOrArray(self):
    actual = (halo_exchange_utils.
              apply_one_core_boundary_conditions_to_tensor_or_array(
                  self.x, self.boundary_conditions))
    self.assertAllEqual(self.expected, actual)

  def testApplyOneCoreBoundaryConditions(self):
    tiles = [self.x[:, :, i] for i in range(self.x.shape[-1])]
    actual = halo_exchange_utils.apply_one_core_boundary_conditions(
        tiles, self.boundary_conditions)
    actual = np.stack(actual, axis=-1)
    self.assertAllEqual(self.expected, actual)


class ParameterizedHaloExchangeUtilsTest(tf.test.TestCase,
                                         parameterized.TestCase):

  @parameterized.named_parameters(
      ('_1D_d', 1, (), False, halo_exchange_utils.BCType.DIRICHLET),
      ('_2D_d', 2, (), True, halo_exchange_utils.BCType.DIRICHLET),
      ('_3D_d', 3, (), False, halo_exchange_utils.BCType.DIRICHLET),
      ('_2D_skip_0_d', 2, (0,), False, halo_exchange_utils.BCType.DIRICHLET),
      ('_3D_skip_0_d', 3, (0,), True, halo_exchange_utils.BCType.DIRICHLET),
      ('_3D_skip_0_1_d', 3, (0, 1,), False,
       halo_exchange_utils.BCType.DIRICHLET),
      ('_3D_skip_2_d', 3, (2,), True, halo_exchange_utils.BCType.DIRICHLET),
      ('_1D_n', 1, (), False, halo_exchange_utils.BCType.NEUMANN),
      ('_2D_n', 2, (), True, halo_exchange_utils.BCType.NEUMANN),
      ('_3D_n', 3, (), False, halo_exchange_utils.BCType.NEUMANN),
      ('_2D_skip_0_n', 2, (0,), False, halo_exchange_utils.BCType.NEUMANN),
      ('_3D_skip_0_n', 3, (0,), True, halo_exchange_utils.BCType.NEUMANN),
      ('_3D_skip_0_1_n', 3, (0, 1,), False, halo_exchange_utils.BCType.NEUMANN),
      ('_3D_skip_2_n', 3, (2,), True, halo_exchange_utils.BCType.NEUMANN),
  )
  def testApplyOneCoreBoundaryConditionsToTensorOrArray_Parameterized(
      self, rank, skip_axes, tf_test, bc_type):
    shape = (12, 15, 17)[:rank]
    x = np.random.rand(*shape).astype(np.float32)
    boundary_conditions = [[(bc_type, 1.), (bc_type, 2.)],
                           [(bc_type, 3.), (bc_type, 4.)],
                           [(bc_type, 5.), (bc_type, 6.)]][:rank]
    for skip_axis in skip_axes:
      boundary_conditions[skip_axis] = None

    x_in = tf.convert_to_tensor(x) if tf_test else x
    actual = (halo_exchange_utils.
              apply_one_core_boundary_conditions_to_tensor_or_array(
                  x_in, boundary_conditions))
    if tf_test:
      actual = actual.numpy()

    # Check interior is unchanged.
    inner_slices = (slice(1, -1),) * rank
    self.assertAllEqual(actual[inner_slices], x[inner_slices])

    # Check all sides are as expected.
    axes_touched = []
    for axis in range(rank - 1, -1, -1):
      for side in range(2):
        with self.subTest(
            f'side={side}, axis={axis}, axis skipped={axis in skip_axes}'):
          outer_slice = [slice(None)] * rank
          outer_slice[axis] = slice(0, 1) if side == 0 else slice(-1, None)
          actual_plane = actual[outer_slice]

          if axis in skip_axes:
            # Unchanged.
            expected_plane = x[outer_slice]
          elif bc_type == halo_exchange_utils.BCType.DIRICHLET:
            # Dirichlet.
            value = axis * 2 + 1 + side
            expected_plane = value * np.ones_like(actual_plane)
            axes_touched.append(axis)
          else:
            # Neumann.
            neumann_slice = [slice(None)] * rank
            neumann_slice[axis] = slice(1, 2) if side == 0 else slice(-2, -1)
            neumann_plane = actual[neumann_slice]
            sign = -1 if side == 0 else 1
            value = axis * 2 + 1 + side
            expected_plane = neumann_plane + sign * value
            axes_touched.append(axis)

          maybe_ignore_ends = [slice(None)] * rank
          # Check current axis. (Assume some axis is not skipped.)
          if axis in skip_axes:
            maybe_ignore_ends[axis] = slice(1, -1)
          # Check axes after current one.
          for a in range(axis + 1, rank):
            if a in axes_touched:
              maybe_ignore_ends[a] = slice(1, -1)
          maybe_ignore_ends = tuple(maybe_ignore_ends)

          self.assertAllClose(actual_plane[maybe_ignore_ends],
                              expected_plane[maybe_ignore_ends], atol=1e-15)


if __name__ == '__main__':
  tf.test.main()
