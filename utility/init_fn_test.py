"""Tests for google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.utilities.init_fn."""

import numpy as np
from swirl_lm.utility import init_fn
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf
from google3.testing.pybase import parameterized

_M = 3
_N = 2


def _mean_init_fn(xx: tf.Tensor, yy: tf.Tensor, zz: tf.Tensor, lx: float,
                  ly: float, lz: float,
                  coord: init_fn.ThreeIntTuple) -> tf.Tensor:
  """Optional to `normal_distribution_init_fn` with a non-homogeneous mean."""
  del xx, yy, lx, ly, lz, coord

  t = np.array([
      [10, 0, 0],
      [0, -20, 0],
      [0, 0, 0],
  ])

  return tf.convert_to_tensor(t, dtype=zz.dtype)


@test_util.run_all_in_graph_and_eager_modes
class InitFnTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      # 2D
      ('Case2D00', 0.0, np.zeros((_M, _N)), np.zeros(
          (_M, _N)), np.ones((_M, _N), dtype=np.float32), 1.0, 2.0, 3.0,
       np.zeros((_M, _N), dtype=np.float32)),
      ('Case2D01', 1.0, np.zeros((_M, _M)), np.zeros(
          (_M, _M)), np.zeros((_M, _M)), 1.0, 2.0, 3.0, np.ones((_M, _M))),
      ('Case2D02', 2.2, np.zeros((_M, _M)), np.zeros(
          (_M, _M)), np.zeros(
              (_N, _N), dtype=np.float64), 3.0, 2.0, 1.0, np.ones(
                  (_N, _N)) * 2.2),
      # 3D
      ('Case3D00', 0.0, np.zeros((_M, _N, _N)), np.ones(
          (_M, _N, _N)), np.ones((_M, _N, _N), dtype=np.float32), 3.0, 2.0, 1.0,
       np.zeros((_M, _N, _N), dtype=np.float32)),
      ('Case3D01', 1.0, np.zeros((_M, _M, _M)), np.zeros(
          (_M, _M, _M)), np.zeros(
              (_M, _M, _M)), 3.0, 2.0, 1.0, np.ones((_M, _M, _M))),
      ('Case3D02', 3.3, np.zeros((_M, _M, _M)), np.zeros(
          (_M, _M, _M)), np.zeros(
              (_N, _N, _N), dtype=np.float64), 1.0, 2.0, 3.0,
       np.ones((_N, _N, _N), dtype=np.float64) * 3.3),
  )
  def testConstantInitFn(self, value, xx, yy, zz, lx, ly, lz, expected_field):
    """Generates a field with a const value with `constant_init_fn`."""
    field = init_fn.constant_init_fn(value)(tf.convert_to_tensor(xx),
                                            tf.convert_to_tensor(yy),
                                            tf.convert_to_tensor(zz), lx, ly,
                                            lz, None)
    result = self.evaluate(field)

    np.testing.assert_allclose(result, expected_field, atol=1e-6)
    self.assertEqual(result.dtype, expected_field.dtype)

  @parameterized.named_parameters(
      # 2D: 2 different seeds
      ('Case2D00', 0.0, 1, None, 1, np.zeros((_M, _M)), np.zeros(
          (_M, _M)), np.ones((_M, _M), dtype=np.float32), 1.0, 2.0, 3.0,
       np.array([
           [-0.811318, 1.484599, 0.065329],
           [-2.442704, 0.099248, 0.591224],
           [0.592823, -2.12293, -0.722897],
       ],
                dtype=np.float32)),
      ('Case2D01', 0.0, 1, None, 9, np.zeros((_M, _M)), np.zeros(
          (_M, _M)), np.ones((_M, _M), dtype=np.float32), 1.0, 2.0, 3.0,
       np.array([
           [1.155073, -0.491057, 0.444481],
           [-2.438359, -1.996019, -0.484241],
           [-0.491944, 0.527831, 1.043522],
       ],
                dtype=np.float32)),
      # Different from `Case2D01`: `mean_init_fn` instead of `mean`
      ('Case2D02', None, 1, _mean_init_fn, 9, np.zeros(
          (_M, _M)), np.zeros((_M, _M)), np.ones(
              (_M, _M), dtype=np.float32), 1.0, 2.0, 3.0,
       np.array([
           [11.155073, -0.491057, 0.444481],
           [-2.438359, -21.996019, -0.484241],
           [-0.491944, 0.527831, 1.043522],
       ],
                dtype=np.float32)),
      # Different from `Case2D01`: `dtype` in inputs.
      ('Case2D03', 0.0, 1, None, 9, np.zeros((_M, _M)), np.zeros(
          (_M, _M)), np.ones((_M, _M), dtype=np.float64), 1.0, 2.0, 3.0,
       np.array([
           [0.68641, -0.650979, 0.211482],
           [0.122365, -1.548266, -1.924003],
           [0.894135, -0.578994, -0.197934],
       ])),
      # 3D.
      ('Case3D00', 1.0, 1, None, 1, np.zeros(
          (_M, _N, _N)), np.zeros(
              (_M, _N, _N)), np.ones(
                  (_M, _N, _N), dtype=np.float32), 3.0, 2.0, 1.0,
       np.array([
           [
               [0.18868178, 2.4845986],
               [1.0653293, -1.4427042],
           ],
           [
               [1.0992484, 1.5912243],
               [1.592823, -1.1229296],
           ],
           [
               [0.27710277, 0.94372964],
               [1.6435448, 0.73567593],
           ],
       ],
                dtype=np.float32)),
  )
  def testNormalDistributionInitFn(self, mean, std, mean_init_fn, seed, xx, yy,
                                   zz, lx, ly, lz, expected_field):
    """Generates a field with normal distribution elementwise."""
    xx = tf.convert_to_tensor(xx)
    yy = tf.convert_to_tensor(yy)
    zz = tf.convert_to_tensor(zz)

    field = init_fn.normal_distribution_init_fn(
        mean, std, mean_init_fn, seed=seed)(xx, yy, zz, lx, ly, lz, None)

    result = self.evaluate(field)

    eps = 1e-6
    np.testing.assert_allclose(result, expected_field, atol=eps)
    self.assertEqual(result.dtype, expected_field.dtype)

    # Second call is idempotent when running Eagerly.
    if tf.executing_eagerly():
      result = self.evaluate(field)
      np.testing.assert_allclose(result, expected_field, atol=eps)
      self.assertEqual(result.dtype, expected_field.dtype)

  @parameterized.named_parameters(
      ('Case00', None, 1.0, None, 1e-6,
       'Expecting one and only one of `mean` or `mean_init_fn` to be valid.'),
      ('Case01', 1.0, 1.0, _mean_init_fn, 1e-6,
       'Expecting one and only one of `mean` or `mean_init_fn` to be valid.'),
      ('Case02', 1.0, 1.0, None, -1e-6,
       '`std` and `eps` are expected to be positive, and `std >= eps`.'),
      ('Case03', 1.0, -1., None, 1e-6,
       '`std` and `eps` are expected to be positive, and `std >= eps`.'),
      ('Case04', 1.0, 1e-8, None, 1e-6,
       '`std` and `eps` are expected to be positive, and `std >= eps`.'),
  )
  def testNormalDistributionInitFnFailure(self, mean, std, mean_init_fn, eps,
                                          expected_error_msg):
    """Failure cases to generate a field with normal distribution."""
    with self.assertRaisesRegex(ValueError, expected_error_msg):
      init_fn.normal_distribution_init_fn(mean, std, mean_init_fn, eps=eps)

  def run_blasius_boundary_layer_test(self,
                                      lx,
                                      ly,
                                      lz,
                                      elevation,
                                      apply_transition=False):
    """Generates velocity fields from `blasius_boundary_layer`."""
    u_inf = 8.0
    v_inf = 6.0
    nu = 1e-5
    nz = 4
    x = 1.0

    def expand_dims(a, axis):
      """Expands dimsion of `a` is specified axis."""
      for dim in axis:
        a = np.expand_dims(a, dim)
      return a

    xx = tf.tile(
        expand_dims(np.linspace(0, lx, 4, dtype=np.float32), (1, 2)), [1, 4, 4])
    yy = tf.tile(
        expand_dims(np.linspace(0, ly, 4, dtype=np.float32), (0, 2)), [4, 1, 4])
    zz = tf.tile(
        expand_dims(np.linspace(0, lz, nz, dtype=np.float32), (0, 1)),
        [4, 4, 1])

    dx = lx / 3.0
    dy = ly / 3.0
    if elevation is None:
      init_fn_dict = init_fn.blasius_boundary_layer(
          u_inf,
          v_inf,
          nu,
          dx,
          dy,
          lz,
          nz,
          x,
          apply_transition=apply_transition)
    else:
      init_fn_dict = init_fn.blasius_boundary_layer(
          u_inf,
          v_inf,
          nu,
          dx,
          dy,
          lz,
          nz,
          x,
          tf.convert_to_tensor(elevation),
          apply_transition=apply_transition)
    self.assertCountEqual(['u', 'v', 'w'], init_fn_dict.keys())

    u = self.evaluate(init_fn_dict['u'](xx, yy, zz, lx, ly, lz, None))
    v = self.evaluate(init_fn_dict['v'](xx, yy, zz, lx, ly, lz, None))
    w = self.evaluate(init_fn_dict['w'](xx, yy, zz, lx, ly, lz, None))

    return u, v, w

  @parameterized.named_parameters(
      ('Case00', None),
      ('Case01', 5e-3 * np.ones((4, 4, 4), dtype=np.float32)),
      ('Case02', 5e-3 * np.ones((4, 4), dtype=np.float32)),
  )
  def testBlasiusBoundaryLayerProducesCorrectProfiles(self, elevation):
    """Checks if the Blasius boundary layer profiles are computed correctly."""
    lx = 1.0
    ly = 1.0
    lz = 1e-2
    u, v, w = self.run_blasius_boundary_layer_test(lx, ly, lz, elevation)

    u_ref = [0.0, 7.14937485, 7.99844219, 7.99999998, 8.0]
    v_ref = [0.0, 5.36203113, 5.99883164, 5.99999999, 6.0]
    w_ref = [0.0, 0.00645903, 0.0085971, 0.00860394, 0.00860394]

    with self.subTest(name='UProfileIsCorrect'):
      if elevation is None:
        buf = u_ref[1:]
      else:
        buf = np.zeros((4,), dtype=np.float32)
        buf[2:] = u_ref[1:-2]
      expected = np.tile(buf, (4, 4, 1))
      self.assertAllClose(expected, u)

    with self.subTest(name='VProfileIsCorrect'):
      if elevation is None:
        buf = v_ref[1:]
      else:
        buf = np.zeros((4,), dtype=np.float32)
        buf[2:] = v_ref[1:-2]
      expected = np.tile(buf, (4, 4, 1))
      self.assertAllClose(expected, v)

    with self.subTest(name='WProfileIsCorrect'):
      if elevation is None:
        buf = w_ref[1:]
      else:
        buf = np.zeros((4,), dtype=np.float32)
        buf[2:] = w_ref[1:-2]
      expected = np.tile(buf, (4, 4, 1))
      self.assertAllClose(expected, w)

  def testBlasiusBoundaryLayerProfilesCorrectProfilesOnASlope(self):
    """Checks if the Blasius boundary layer profiles are computed correctly."""
    lx = 6.0
    ly = 6.0
    lz = 12.0
    slope = 45.0
    x = np.linspace(0.0, lx, 4, dtype=np.float32)
    elevation = np.tile(1.0 + x * np.tan(slope * np.pi / 180.0),
                        [4, 1]).transpose()

    u, v, w = self.run_blasius_boundary_layer_test(lx, ly, lz, elevation, True)

    with self.subTest(name='UProfileIsCorrect'):
      expected = np.zeros((4, 4, 4), dtype=np.float32)
      expected[:2, :, 1] = 7.648026
      expected[:2, :, 2:] = 8.0
      expected[2:, :, 2] = 7.648026
      expected[2:, :, 3] = 8.0
      self.assertAllClose(expected, u)

    with self.subTest(name='VProfileIsCorrect'):
      expected = np.zeros((4, 4, 4), dtype=np.float32)
      expected[:2, :, 1] = 5.735715
      expected[:2, :, 2:] = 6.0
      expected[2:, :, 2] = 5.735715
      expected[2:, :, 3] = 6.0
      self.assertAllClose(expected, v)

    with self.subTest(name='WProfileIsCorrect'):
      expected = np.zeros((4, 4, 4), dtype=np.float32)
      expected[:2, :, 1] = 1.42244
      expected[:2, :, 2:] = 0.00860352
      expected[2:, :, 2] = 1.42244
      expected[2:, :, 3] = 0.00860352
      self.assertAllClose(expected, w)

  def run_logarithmic_boundary_layer_test(self, lx, ly, lz, elevation):
    """Generates velocity fields from `blasius_boundary_layer`."""
    u_inf = 8.0
    v_inf = 6.0
    z_0 = 0.15
    nz = 4

    def expand_dims(a, axis):
      """Expands dimsion of `a` is specified axis."""
      for dim in axis:
        a = np.expand_dims(a, dim)
      return a

    xx = tf.tile(
        expand_dims(np.linspace(0, lx, 4, dtype=np.float32), (1, 2)), [1, 4, 4])
    yy = tf.tile(
        expand_dims(np.linspace(0, ly, 4, dtype=np.float32), (0, 2)), [4, 1, 4])
    zz = tf.tile(
        expand_dims(np.linspace(0, lz, nz, dtype=np.float32), (0, 1)),
        [4, 4, 1])

    if elevation is None:
      init_fn_dict = init_fn.logarithmic_boundary_layer(
          u_inf,
          v_inf,
          z_0)
    else:
      init_fn_dict = init_fn.logarithmic_boundary_layer(
          u_inf,
          v_inf,
          z_0,
          tf.convert_to_tensor(elevation))
    self.assertCountEqual(['u', 'v', 'w'], init_fn_dict.keys())

    u = self.evaluate(init_fn_dict['u'](xx, yy, zz, lx, ly, lz, None))
    v = self.evaluate(init_fn_dict['v'](xx, yy, zz, lx, ly, lz, None))
    w = self.evaluate(init_fn_dict['w'](xx, yy, zz, lx, ly, lz, None))

    return u, v, w

  @parameterized.named_parameters(
      ('NoElevation', None),
      ('ConstElevation3D', 40 * np.ones((4, 4, 4), dtype=np.float32)),
      ('ConstElevation2D', 40 * np.ones((4, 4), dtype=np.float32)),
  )
  def testLogarithmicBoundaryLayerProducesCorrectProfiles(self, elevation):
    """Checks if the Blasius boundary layer profiles are computed correctly."""
    lx = 1.0
    ly = 1.0
    lz = 100.0
    u, v, w = self.run_logarithmic_boundary_layer_test(lx, ly, lz, elevation)

    with self.subTest(name='UProfileIsCorrect'):
      if elevation is None:
        buf = [6.648338, 7.5011415, 8.0, 8.353946]
      else:
        buf = [0.0, 7.21516796, 8.14068082, 8.68207113]
      expected = np.tile(buf, (4, 4, 1))
      self.assertAllClose(expected, u)

    with self.subTest(name='VProfileIsCorrect'):
      if elevation is None:
        buf = [4.98625353, 5.62585634, 6.0, 6.26545915]
      else:
        buf = [0.0, 5.41137597, 6.10551061, 6.51155335]
      expected = np.tile(buf, (4, 4, 1))
      self.assertAllClose(expected, v)

    with self.subTest(name='WProfileIsCorrect'):
      expected = np.zeros((4, 4, 4), dtype=np.float32)
      self.assertAllClose(expected, w)

  def testLogarithmicBoundaryLayerProfilesCorrectProfilesOnASlope(self):
    """Checks if the Blasius boundary layer profiles are computed correctly."""
    lx = 120.0
    ly = 120.0
    lz = 300.0
    slope = 45.0
    x = np.linspace(0.0, lx, 4, dtype=np.float32)
    elevation = np.tile(x * np.tan(slope * np.pi / 180.0), [4, 1]).transpose()

    u, v, w = self.run_logarithmic_boundary_layer_test(lx, ly, lz, elevation)

    with self.subTest(name='UProfileIsCorrect'):
      expected = [
          np.tile([[6.84370332, 7.5732453, 8.0, 8.30278728]], (4, 1)),
          np.tile([[6.9750208, 7.71856128, 8.15350459, 8.46210177]], (4, 1)),
          np.tile([[7.13484058, 7.89541793, 8.34032716, 8.65599529]], (4, 1)),
          np.tile([[0.0, 7.33677823, 8.11888223, 8.57638374]], (4, 1)),
      ]
      self.assertAllClose(expected, u)

    with self.subTest(name='VProfileIsCorrect'):
      expected = [
          np.tile([[5.13277749, 5.67993397, 6.0, 6.22709046]], (4, 1)),
          np.tile([[5.2312656, 5.78892096, 6.11512844, 6.34657633]], (4, 1)),
          np.tile([[5.35113044, 5.92156345, 6.25524537, 6.49199646]], (4, 1)),
          np.tile([[0.0, 5.50258367, 6.08916167, 6.4322878]], (4, 1)),
      ]
      self.assertAllClose(expected, v)

    with self.subTest(name='WProfileIsCorrect'):
      expected = np.zeros((4, 4, 4), dtype=np.float32)
      self.assertAllClose(expected, w)


if __name__ == '__main__':
  tf.test.main()
