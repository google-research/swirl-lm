"""Tests for igniter."""

import numpy as np
from swirl_lm.physics.combustion import igniter
import tensorflow as tf


class IgniterTest(tf.test.TestCase):

  def setUp(self):
    """Initializes the ignition kernel library object."""
    super(IgniterTest, self).setUp()

    ignition_speed = 8.0
    ignition_start_point = (40.0, 100.0, 0.0)
    ignition_duration = 5.0
    start_step_id = 20000
    igniter_radius = 12.0
    dt = 0.01

    self.igniter = igniter.Igniter(ignition_speed, ignition_start_point,
                                   ignition_duration, start_step_id,
                                   igniter_radius, dt)

    # Generate a mesh.
    self.nx = 9
    self.ny = 17
    self.nz = 11
    self.lx = 80.0
    self.ly = 160.0
    self.lz = 5.0
    x = np.linspace(0.0, self.lx, self.nx)
    y = np.linspace(0.0, self.ly, self.ny)
    z = np.linspace(0.0, self.lz, self.nz)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    self.xx = tf.convert_to_tensor(xx, dtype=tf.float32)
    self.yy = tf.convert_to_tensor(yy, dtype=tf.float32)
    self.zz = tf.convert_to_tensor(zz, dtype=tf.float32)

  def ignition_kernel_fn(self, xx, yy, zz, lx, ly, lz, coord):
    """Generates the overall ignition kernel used in this test."""
    del yy, zz, lx, ly, lz, coord
    nx, ny, nz = xx.get_shape().as_list()
    ignition_kernel = np.zeros((nx, ny, nz), dtype=np.float32)
    ignition_kernel[3:5, 6:16, :5] = 1.0
    return ignition_kernel

  def testIgnitionKernelCreatedCorrectly(self):
    """Checks if the ignition tensors are created correctly."""

    coord = (0, 0, 0)
    ignition_schedule = self.igniter.ignition_schedule_init_fn(
        self.ignition_kernel_fn)(self.xx, self.yy, self.zz, self.lx, self.ly,
                                 self.lz, coord)

    ignition_schedule = tf.unstack(ignition_schedule)

    with self.subTest(name='BeforeIgnitionGeneratesZeros'):
      expected = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
      self.assertAllClose(
          expected,
          np.stack(self.igniter.ignition_kernel(19999, ignition_schedule)))

    with self.subTest(name='DuringIgnitionGeneratesCorrectKernel'):
      expected = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
      expected[3, 7:10, :5] = 1.0
      expected[3, 11:14, :5] = 1.0
      expected[4, 7:9, :5] = 1.0
      expected[4, 12:14, :5] = 1.0

      ignition_kernel = np.stack(self.evaluate(
          self.igniter.ignition_kernel(20300, ignition_schedule)))

      self.assertAllClose(expected, ignition_kernel)

    with self.subTest(name='AfterIgnitionGeneratesZeros'):
      expected = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
      self.assertAllClose(
          expected,
          np.stack(self.igniter.ignition_kernel(20501, ignition_schedule)))


if __name__ == '__main__':
  tf.test.main()
