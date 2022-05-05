"""Tests for time_integration."""

import numpy as np
from swirl_lm.numerics import time_integration
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf
from google3.testing.pybase import parameterized

_DTYPE = tf.float32


class IncompressibleStructuredMeshNumericsTest(tf.test.TestCase,
                                               parameterized.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testRK3OutputsCorrectTimeIntegrationResults(self):

    # pylint: disable=g-doc-args, g-doc-return-or-yield
    def const_change(u, v, w):
      """A right hand side function for du / dt = f(u) with f(u) = c."""
      nz = len(u)

      rhs_u = [
          tf.constant(1 + i, shape=u[0].shape.as_list(), dtype=_DTYPE)
          for i in range(nz)
      ]
      rhs_v = [
          tf.constant(-1 - i, shape=v[0].shape.as_list(), dtype=_DTYPE)
          for i in range(nz)
      ]
      rhs_w = [
          tf.constant(-1**(i % 2), shape=w[0].shape.as_list(), dtype=_DTYPE)
          for i in range(nz)
      ]

      return (rhs_u, rhs_v, rhs_w)

    # pylint: enable=g-doc-args, g-doc-return-or-yield

    # Starting from a quiescent ambient field, with a (4, 4, 4) mesh.
    dt = 1e-2
    nx = ny = nz = 4
    u = [tf.constant(0, shape=(nx, ny), dtype=_DTYPE) for i in range(nz)]
    v = [tf.constant(0, shape=(nx, ny), dtype=_DTYPE) for i in range(nz)]
    w = [tf.constant(0, shape=(nx, ny), dtype=_DTYPE) for i in range(nz)]

    expected_u, expected_v, expected_w = const_change(u, v, w)
    expected_u = [dt * tile for tile in expected_u]
    expected_v = [dt * tile for tile in expected_v]
    expected_w = [dt * tile for tile in expected_w]

    ut, vt, wt = self.evaluate(
        time_integration.time_advancement_explicit(
            const_change, dt,
            time_integration.TimeIntegrationScheme.TIME_SCHEME_RK3, (u, v, w),
            (u, v, w)))

    self.assertLen(ut, 4)
    self.assertLen(vt, 4)
    self.assertLen(wt, 4)

    for i in range(nz):
      self.assertAllClose(expected_u[i], ut[i])
      self.assertAllClose(expected_v[i], vt[i])
      self.assertAllClose(expected_w[i], wt[i])

  @test_util.run_in_graph_and_eager_modes
  def testCrankNicolsonExplicitIterationOutputsCorrectTimeIntegrationResults(
      self):

    def const_change(u, v, w):
      """A right hand side function for du / dt = f(u) with f(u) = u."""
      return (u, v, w)

    # Starting from a quiescent ambient field, with a (4, 4, 4) mesh.
    dt = 1e-2
    nx = ny = nz = 4
    u0 = [tf.constant(0, shape=(nx, ny), dtype=_DTYPE) for _ in range(nz)]
    v0 = [tf.constant(0, shape=(nx, ny), dtype=_DTYPE) for _ in range(nz)]
    w0 = [tf.constant(0, shape=(nx, ny), dtype=_DTYPE) for _ in range(nz)]
    un = [tf.constant(2, shape=(nx, ny), dtype=_DTYPE) for _ in range(nz)]
    vn = [tf.constant(4, shape=(nx, ny), dtype=_DTYPE) for _ in range(nz)]
    wn = [tf.constant(6, shape=(nx, ny), dtype=_DTYPE) for _ in range(nz)]

    expected_u = [dt * np.ones((nx, ny), dtype=np.float32) for _ in range(nz)]
    expected_v = [
        dt * 2. * np.ones((nx, ny), dtype=np.float32) for _ in range(nz)
    ]
    expected_w = [
        dt * 3. * np.ones((nx, ny), dtype=np.float32) for _ in range(nz)
    ]

    ut, vt, wt = self.evaluate(
        time_integration.time_advancement_explicit(
            const_change, dt, time_integration.TimeIntegrationScheme
            .TIME_SCHEME_CN_EXPLICIT_ITERATION, (u0, v0, w0), (un, vn, wn)))

    self.assertLen(ut, 4)
    self.assertLen(vt, 4)
    self.assertLen(wt, 4)

    for i in range(nz):
      self.assertAllClose(expected_u[i], ut[i])
      self.assertAllClose(expected_v[i], vt[i])
      self.assertAllClose(expected_w[i], wt[i])


if __name__ == '__main__':
  tf.test.main()
