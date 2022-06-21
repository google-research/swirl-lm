"""Tests for google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.numerics.root_finder."""

import itertools
import os

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import parameters_pb2
from swirl_lm.numerics import root_finder
from swirl_lm.physics.thermodynamics import water
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.pyglib import gfile
from google3.pyglib import resources
from google3.testing.pybase import parameterized

_MAX_ITERATIONS = 10
_TESTDATA_DIR = 'google3/third_party/py/swirl_lm/numerics/testdata'

_ARGS = (
    #
    # tf.float32
    #
    #   - Analytical jacobi function.
    ('Case00', None, None, True, tf.float32, np.float32),
    ('Case01', 1e-6, None, True, tf.float32, np.float32),
    ('Case02', None, 1e-6, True, tf.float32, np.float32),
    ('Case03', 1e-8, 1e-6, True, tf.float32, np.float32),
    #   - Numerical jacobi function.
    ('Case04', None, None, False, tf.float32, np.float32),
    ('Case05', 1e-6, None, False, tf.float32, np.float32),
    ('Case06', None, 1e-6, False, tf.float32, np.float32),
    ('Case07', 1e-8, 1e-6, False, tf.float32, np.float32),
    #
    # tf.float64
    #
    #   - Analytical jacobi function.
    ('Case10', None, None, True, tf.float64, np.float64),
    ('Case11', 1e-6, None, True, tf.float64, np.float64),
    ('Case12', None, 1e-6, True, tf.float64, np.float64),
    ('Case13', 1e-8, 1e-6, True, tf.float64, np.float64),
    #   - Numerical jacobi function.
    ('Case14', None, None, False, tf.float64, np.float64),
    ('Case15', 1e-6, None, False, tf.float64, np.float64),
    ('Case16', None, 1e-6, False, tf.float64, np.float64),
    ('Case17', 1e-8, 1e-6, False, tf.float64, np.float64),
)


@test_util.run_all_in_graph_and_eager_modes
class RootFinderTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(*_ARGS)
  def testNewtonMethodMultiDimFindsCorrectRoots(
      self, value_tolerance, position_tolerance, with_jacobian_fn, tf_dtype,
      expected_np_dtype):
    """Checks if the Newton solver finds the roots."""

    f = lambda x: [[x_i**2.0 - 1.0 for x_i in x]]
    j = lambda x: [[[2.0 * x_i for x_i in x]]]

    x0 = [[tf.convert_to_tensor([-0.8, 0.8], dtype=tf_dtype)]]

    root = self.evaluate(
        root_finder.newton_method_multi_dim(
            f,
            x0,
            _MAX_ITERATIONS,
            value_tolerance=value_tolerance,
            position_tolerance=position_tolerance,
            analytical_jacobian_fn=j if with_jacobian_fn else None))
    self.assertLen(root, 1)
    self.assertLen(root[0], 1)

    solution = root[0][0]
    self.assertEqual(solution.dtype, expected_np_dtype)
    self.assertAllClose(np.array([-1.0, 1.0]), solution)

  @parameterized.named_parameters(*_ARGS)
  def testNewtonMethodMultiDimFindsCorrectRootsWithZeroElementInInitialGuess(
      self, value_tolerance, position_tolerance, with_jacobian_fn, tf_dtype,
      expected_np_dtype):
    """Checks the Newton solver finds root with 0 element in initial guess."""
    # pylint: disable=g-long-lambda
    f = lambda x: [[x_i**2.0 - 3.0 * x_i + 2.0 for x_i in x]]
    # pylint: enable=g-long-lambda
    j = lambda x: [[[2.0 * x_i - 3.0 for x_i in x]]]

    x0 = [[
        tf.convert_to_tensor([0.0, 3.0], dtype=tf_dtype),
    ]]

    root = self.evaluate(
        root_finder.newton_method_multi_dim(
            f,
            x0,
            _MAX_ITERATIONS,
            value_tolerance=value_tolerance,
            position_tolerance=position_tolerance,
            analytical_jacobian_fn=j if with_jacobian_fn else None))

    self.assertLen(root, 1)
    self.assertLen(root[0], 1)

    solution = root[0][0]
    self.assertEqual(solution.dtype, expected_np_dtype)
    self.assertAllClose(np.array([1.0, 2.0]), solution)

  @parameterized.named_parameters(*_ARGS)
  def testNewtonMethodMultiDimFindsCorrectRootsWithZeroElementInSolution(
      self, value_tolerance, position_tolerance, with_jacobian_fn, tf_dtype,
      expected_np_dtype):
    """Checks the Newton solver finds root with 0 element in solution."""
    f = lambda x: [[x_i**2.0 - 3.0 * x_i for x_i in x]]
    j = lambda x: [[[2.0 * x_i - 3.0 for x_i in x]]]

    x0 = [[
        tf.convert_to_tensor([0.5, 3.5], dtype=tf_dtype),
    ]]

    root = self.evaluate(
        root_finder.newton_method_multi_dim(
            f,
            x0,
            _MAX_ITERATIONS,
            value_tolerance=value_tolerance,
            position_tolerance=position_tolerance,
            analytical_jacobian_fn=j if with_jacobian_fn else None))

    self.assertLen(root, 1)
    self.assertLen(root[0], 1)

    solution = root[0][0]
    self.assertEqual(solution.dtype, expected_np_dtype)
    self.assertAllClose(np.array([0.0, 3.0]), solution)

  _REPLICAS = np.array([[[0]]], dtype=np.int32)
  _SOLUTION = (-0.10983, 0.10983)

  @parameterized.parameters(*(
      # Disable cache.
      list(
          itertools.product(
              (
                  # Note the solution with numerical Jacobi could be different
                  # from the exact Jacobian only with `tf.float32`, and the
                  # former has more oscillations.
                  #
                  # - First few iterations.
                  ('Iter00', 0, ((-1., 1.), (-1., 1.))),
                  ('Iter01', 1, ((-1., 1.), (-1., 1.))),
                  ('Iter02', 2, ((1., -1.), (0.997288, -0.997288))),
                  ('Iter03', 3, ((-1., 1.), (-1.006838, 1.006838))),
                  ('Iter04', 4, ((1., -1.), (0.987803, -0.987803))),
                  ('Iter05', 5, ((-1., 1.), (-1.026149, 1.026149))),
                  # - More iterations.
                  ('Iter21', 21, ((-1., 1.), (2.299398, -2.299398))),
                  ('Iter22', 22, ((1., -1.), (0.500269, -0.500269))),
              ),
              (False, True),
              (
                  (tf.float32, np.float32),
                  (tf.float64, np.float64),
              ),
              (None,))) + [
                  # Disable cache: Show where the best solution is from.
                  (('Iter12Float32NumericalNoCache', 12,
                    ((-1.846011, 1.846011),)), False,
                   (tf.float32, np.float32), None),
                  (('Iter13Float32NumericalNoCache', 13, (_SOLUTION,)), False,
                   (tf.float32, np.float32), None),
                  (('Iter14Float32NumericalNoCache', 14,
                    ((13.474586, -13.474586),)), False,
                   (tf.float32, np.float32), None),
                  # Enable cache.
                  (('Iter14Float32Numerical', 14, (_SOLUTION,)), False,
                   (tf.float32, np.float32), _REPLICAS),
                  (('Iter22Float32Numerical', 22, (_SOLUTION,)), False,
                   (tf.float32, np.float32), _REPLICAS),
                  (('Iter22Float32Analytical', 22, ((1., -1.),)), True,
                   (tf.float32, np.float32), _REPLICAS),
                  (('Iter22Float64Numerical', 22, ((-1, 1.),)), False,
                   (tf.float64, np.float64), _REPLICAS),
                  (('Iter22Float64Analytical', 22, ((1., -1.),)), True,
                   (tf.float64, np.float64), _REPLICAS),
              ]))
  def testNewtonMethodMultiDimNoSolution(
      self, args, with_jacobian_fn, dtypes, replicas):
    """Checks the Newton solver when there is no solution."""
    _, max_iter, expected_solutions = args
    tf_dtype, expected_np_dtype = dtypes

    f = lambda x: [[x_i**2.0 + 3.0 for x_i in x]]
    j = lambda x: [[[2.0 * x_i for x_i in x]]]

    x0 = [[
        tf.convert_to_tensor([-1, 1], dtype=tf_dtype),
    ]]

    root = self.evaluate(
        root_finder.newton_method_multi_dim(
            f,
            x0,
            max_iter,
            analytical_jacobian_fn=j if with_jacobian_fn else None,
            replicas=replicas))

    self.assertLen(root, 1)
    self.assertLen(root[0], 1)

    solution = root[0][0]
    tf.compat.v1.logging.info('(max_iter, solution) = (%02d, %s).', max_iter,
                              solution)
    self.assertEqual(solution.dtype, expected_np_dtype)
    self.assertAllClose(
        np.array(expected_solutions[min(
            0 if (with_jacobian_fn or tf_dtype == tf.float64) else 1,
            len(expected_solutions) - 1)]), solution)

  @parameterized.named_parameters(
      # Numerical Jacobian.
      ('Case00', None, None, False, tf.float32, np.float32),
      ('Case01', 1e-6, None, False, tf.float32, np.float32),
      ('Case02', 1e-8, 1e-6, False, tf.float64, np.float64),
      # Analytical Jacobian.
      ('Case10', None, None, True, tf.float32, np.float32),
      ('Case11', 1e-6, None, True, tf.float32, np.float32),
      ('Case12', 1e-8, 1e-6, True, tf.float64, np.float64),
  )
  def testNewtonMethodMultiDimFindsCorrectRoots2D(
      self, value_tolerance, position_tolerance, with_jacobian_fn, tf_dtype,
      expected_np_dtype):
    """Checks if the Newton solver finds the roots."""

    def f(x, y):
      return [
          [x_i**2 + y_i**2 - 1 for x_i, y_i in zip(x, y)],
          [x_i + y_i - 1 for x_i, y_i in zip(x, y)],
      ]

    def j(x, y):
      return [
          [
              [x_i * 2 for x_i in x],
              [y_i * 2 for y_i in y],
          ],
          [
              [tf.ones_like(x_i) for x_i in x],
              [tf.ones_like(y_i) for y_i in y],
          ],
      ]

    x0 = [
        [tf.convert_to_tensor([-0.2, 0.2, 0.8, 1.2], dtype=tf_dtype)],
        [tf.convert_to_tensor([1.5, 0.8, 0.2, -0.5], dtype=tf_dtype)],
    ]

    root = self.evaluate(
        root_finder.newton_method_multi_dim(
            f,
            x0,
            _MAX_ITERATIONS,
            value_tolerance=value_tolerance,
            position_tolerance=position_tolerance,
            analytical_jacobian_fn=j if with_jacobian_fn else None))

    self.assertLen(root, 2)

    x, y = root
    self.assertLen(x, 1)
    self.assertLen(y, 1)

    self.assertEqual(x[0].dtype, expected_np_dtype)
    self.assertEqual(y[0].dtype, expected_np_dtype)

    self.assertAllClose(np.array([0, 0, 1, 1]), x[0])
    self.assertAllClose(np.array([1, 1, 0, 0]), y[0])

  @parameterized.named_parameters(
      # Numerical Jacobian.
      ('Case00', None, None, False, tf.float32, np.float32),
      ('Case01', 1e-6, None, False, tf.float32, np.float32),
      ('Case02', 1e-8, 1e-6, False, tf.float64, np.float64),
      # Analytical Jacobian.
      ('Case10', None, None, True, tf.float32, np.float32),
      ('Case11', 1e-6, None, True, tf.float32, np.float32),
      ('Case12', 1e-8, 1e-6, True, tf.float64, np.float64),
  )
  def testNewtonMethodMultiDimFindsCorrectRoots3D(
      self, value_tolerance, position_tolerance, with_jacobian_fn, tf_dtype,
      expected_np_dtype):
    """Checks if the Newton solver finds the roots."""

    def f(x, y, z):
      return [
          [x_i**2 + y_i**2 + z_i**2 - 26 for x_i, y_i, z_i in zip(x, y, z)],
          [x_i + y_i - 7 for x_i, y_i in zip(x, y)],
          [z_i**2 - 1 for z_i in z],
      ]

    def j(x, y, z):
      return [
          [
              [x_i * 2 for x_i in x],
              [y_i * 2 for y_i in y],
              [z_i * 2 for z_i in z],
          ],
          [
              [tf.ones_like(x_i) for x_i in x],
              [tf.ones_like(y_i) for y_i in y],
              [tf.zeros_like(z_i) for z_i in z],
          ],
          [
              [tf.zeros_like(x_i) for x_i in x],
              [tf.zeros_like(y_i) for y_i in y],
              [z_i * 2 for z_i in z],
          ],
      ]

    x0 = [
        [tf.convert_to_tensor([+3.2, 1.2, -0.8, 1.2], dtype=tf_dtype)],
        [tf.convert_to_tensor([3.5, -0.8, 0.2, -0.5], dtype=tf_dtype)],
        [tf.convert_to_tensor([-1.5, -0.8, 0.2, -0.5], dtype=tf_dtype)],
    ]

    root = self.evaluate(
        root_finder.newton_method_multi_dim(
            f,
            x0,
            _MAX_ITERATIONS,
            value_tolerance=value_tolerance,
            position_tolerance=position_tolerance,
            analytical_jacobian_fn=j if with_jacobian_fn else None))

    self.assertLen(root, 3)

    x, y, z = root
    self.assertLen(x, 1)
    self.assertLen(y, 1)
    self.assertLen(z, 1)

    self.assertEqual(x[0].dtype, expected_np_dtype)
    self.assertEqual(y[0].dtype, expected_np_dtype)
    self.assertEqual(z[0].dtype, expected_np_dtype)

    self.assertAllClose(np.array([3, 4, 3, 4]), x[0])
    self.assertAllClose(np.array([4, 3, 4, 3]), y[0])
    self.assertAllClose(np.array([-1, -1, 1, -1]), z[0])

  def testNewtonMethodMultiDimFindsCorrectTemperatureInWaterThermodynamics(
      self):
    """Checks if temperature in water thermodynamics is computed correctly."""
    with gfile.Open(
        resources.GetResourceFilename(
            os.path.join(_TESTDATA_DIR, 'config.textpb'))) as f:
      config = text_format.Parse(f.read(), parameters_pb2.SwirlLMParameters())

    params = parameters_lib.SwirlLMParameters(config)

    model = water.Water(params)

    t_1 = [
        tf.constant(290.0, dtype=tf.float32),
        tf.constant(306.0, dtype=tf.float32),
    ]
    rho = [
        tf.constant(1.22, dtype=tf.float32),
        tf.constant(1.10, dtype=tf.float32),
    ]
    q_t = [
        tf.constant(1e-2, dtype=tf.float32),
        tf.constant(5e-3, dtype=tf.float32),
    ]
    e_int = [
        tf.constant(3.6e4, dtype=tf.float32),
        tf.constant(3.2e4, dtype=tf.float32),
    ]

    t_sat = model.saturation_temperature('e_int', t_1, e_int, rho, q_t)
    e_int_sat = model.saturation_internal_energy(t_sat, rho, q_t)

    res_t = self.evaluate(t_sat)
    res_e = self.evaluate(e_int_sat)

    with self.subTest(name='SaturationTemperatureIsCorrect'):
      # Expected values are obtained from a off-line benchmark code. The
      # relative error is specified in the model config, and the absolute error
      # is derived based on the magnitude of the solution, which is O(1e-3).
      expected = [293.28455, 302.62656]
      with self.assertRaisesRegex(
          AssertionError, 'Not equal to tolerance rtol=1e-08, atol=1e-08'):
        self.assertAllClose(
            expected,
            res_t,
            rtol=model._f_temperature_atol_and_rtol,
            atol=model._f_temperature_atol_and_rtol)
      self.assertAllClose(
          expected, res_t, rtol=model._f_temperature_atol_and_rtol, atol=1e-5)

    with self.subTest(name='SaturationInternalEnergyEqualsInput'):
      expected = [3.6e4, 3.2e4]
      self.assertAllClose(expected, res_e, rtol=1e-6, atol=1e-1)

  def testNewtonMethodFindsCorrectRootsWithAnalyticalJacobian(self):
    """Checks if the Newton solver finds the root with analytical Jacobian."""
    f = lambda x: [x_i**2.0 - 1.0 for x_i in x]
    df = lambda x: [2.0 * x_i for x_i in x]
    x0 = [tf.convert_to_tensor([-0.8, 0.8], dtype=tf.float32),]
    max_iter = 10

    expected = np.array([-1.0, 1.0])

    with self.subTest(name='WithoutTolerance'):
      root = self.evaluate(
          root_finder.newton_method(
              f, x0, max_iter, analytical_jacobian_fn=df))

      self.assertAllClose(expected, root[0])

    with self.subTest(name='WithTolerance'):
      root = self.evaluate(
          root_finder.newton_method(
              f,
              x0,
              max_iter,
              analytical_jacobian_fn=df,
              value_tolerance=1e-6))

      self.assertAllClose(expected, root[0])

  def testNewtonMethodFindsCorrectRootsWithNumericalJacobian(self):
    """Checks if the Newton solver finds the root with numerical Jacobian."""
    f = lambda x: [x_i**2.0 - 1.0 for x_i in x]
    x0 = [tf.convert_to_tensor([-0.8, 0.8], dtype=tf.float32),]
    max_iter = 10

    root = self.evaluate(root_finder.newton_method(f, x0, max_iter))

    expected = np.array([-1.0, 1.0])

    self.assertAllClose(expected, root[0])

  @parameterized.named_parameters(
      ('InputTensor', 'TENSOR'), ('InputListTensor', 'LIST_TENSOR'))
  def testNewtonMethodFindsCorrectRootsWithInitialGuessZero(self, option):
    """Checks if the Newton solver finds roots with initial guess being 0."""
    if option == 'LIST_TENSOR':
      f = lambda x: [x_i**2 - 3.0 * x_i + 2.0 for x_i in x]
      df = lambda x: [2.0 * x_i - 3.0 for x_i in x]
      x0 = [tf.convert_to_tensor([0.0, 3.0], dtype=tf.float32),]
    else:  # option == 'TENSOR'
      f = lambda x: x**2 - 3.0 * x + 2.0
      df = lambda x: 2.0 * x - 3.0
      x0 = tf.convert_to_tensor([0.0, 3.0], dtype=tf.float32)

    max_iter = 10

    expected = np.array([1.0, 2.0])

    with self.subTest(name='WithAnalyticalJacobian'):

      root = self.evaluate(
          root_finder.newton_method(
              f, x0, max_iter, analytical_jacobian_fn=df))

      self.assertAllClose(expected, np.squeeze(root))

    with self.subTest(name='WithNumericalJacobian'):
      root = self.evaluate(root_finder.newton_method(f, x0, max_iter))

      self.assertAllClose(expected, np.squeeze(root))

  def testNewtonMethodFindsCorrectTemperatureInWaterThermodynamics(self):
    """Checks if temperature in water thermodynamics is computed correctly."""
    with gfile.Open(
        resources.GetResourceFilename(
            os.path.join(_TESTDATA_DIR, 'config.textpb'))) as f:
      config = text_format.Parse(f.read(), parameters_pb2.SwirlLMParameters())

    params = parameters_lib.SwirlLMParameters(config)

    model = water.Water(params)

    t_1 = [
        tf.constant(290.0, dtype=tf.float32),
        tf.constant(306.0, dtype=tf.float32),
    ]
    rho = [
        tf.constant(1.22, dtype=tf.float32),
        tf.constant(1.10, dtype=tf.float32),
    ]
    q_t = [
        tf.constant(1e-2, dtype=tf.float32),
        tf.constant(5e-3, dtype=tf.float32),
    ]
    e_int = [
        tf.constant(3.6e4, dtype=tf.float32),
        tf.constant(3.2e4, dtype=tf.float32),
    ]

    t_sat = model.saturation_temperature('e_int', t_1, e_int, rho, q_t)
    e_int_sat = model.saturation_internal_energy(t_sat, rho, q_t)

    res_t = self.evaluate(t_sat)
    res_e = self.evaluate(e_int_sat)

    with self.subTest(name='SaturationTemperatureIsCorrect'):
      # Expected values are obtained from a off-line benchmark code. The
      # relative error is specified in the model config, and the absolute error
      # is derived based on the magnitude of the solution, which is O(1e3).
      expected = [293.28455, 302.62656]
      self.assertAllClose(
          expected, res_t, rtol=model._temperature_atol_and_rtol, atol=1e-5)

    with self.subTest(name='SaturationInternalEnergyEqualsInput'):
      expected = [3.6e4, 3.2e4]
      self.assertAllClose(expected, res_e, rtol=1e-6, atol=1e-1)

if __name__ == '__main__':
  tf.test.main()
