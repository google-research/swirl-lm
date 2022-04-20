# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multigrid-related test infrastructure used in multiple modules."""
import functools
import itertools
from typing import Callable, Sequence, Text, Union

import numpy as np
from swirl_lm.linalg import multigrid_3d_utils
from swirl_lm.linalg import multigrid_utils
from swirl_lm.utility import types
import tensorflow as tf
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner

TensorOrArray = multigrid_utils.TensorOrArray
TilesOrTensorOrArray = Union[multigrid_3d_utils.Tiles, TensorOrArray]
Solver = Callable[[TilesOrTensorOrArray, TilesOrTensorOrArray],
                  TilesOrTensorOrArray]

_NP_DTYPE = types.NP_DTYPE


class ConvergenceTest(tf.test.TestCase):
  """Tests convergence of Jacobi and Multigrid Poisson solvers."""

  _AXES = (0, 1, 2)
  _JACOBI_DIFFS_NORMS_5X5X5 = (
      (1,
       (1.944e-01, 1.481e-01, 1.127e-01, 7.600e-02, 5.718e-02),
       (6.874e+01, 4.203e+01, 2.823e+01, 1.954e+01, 1.369e+01)),
      (3,
       (1.127e-01, 4.087e-02, 1.432e-02, 5.349e-03, 1.790e-03),
       (2.823e+01, 9.637e+00, 3.397e+00, 1.201e+00, 4.244e-01)),
      (6,
       (4.087e-02, 5.349e-03, 6.711e-04, 8.392e-05, 1.049e-05),
       (9.637e+00, 1.201e+00, 1.501e-01, 1.876e-02, 2.345e-03)),
  )  # pyformat: disable
  JACOBI_DIFFS_NORMS_5X5X5 = itertools.product(_AXES, _JACOBI_DIFFS_NORMS_5X5X5)

  _JACOBI_DIFFS_NORMS_WEIGHT_2_3_5X5X5 = (
      (1,
       (2.407e-01, 1.893e-01, 1.548e-01, 1.261e-01, 1.024e-01),
       (8.822e+01, 5.767e+01, 4.275e+01, 3.300e+01, 2.591e+01)),
      (3,
       (1.548e-01, 8.289e-02, 4.349e-02, 2.283e-02, 1.222e-02),
       (4.275e+01, 2.053e+01, 1.048e+01, 5.430e+00, 2.826e+00)),
      (6,
       (8.289e-02, 2.283e-02, 6.451e-03, 1.770e-03, 4.820e-04),
       (2.053e+01, 5.430e+00, 1.472e+00, 3.997e-01, 1.086e-01)),
  )  # pyformat: disable
  JACOBI_DIFFS_NORMS_WEIGHT_2_3_5X5X5 = itertools.product(
      _AXES, _JACOBI_DIFFS_NORMS_WEIGHT_2_3_5X5X5)

  _MG_DIFFS_NORMS_5X5X5 = (
      (1, 1,
       (9.572e-02, 3.727e-02, 1.517e-02, 6.249e-03, 2.571e-03),
       (4.430e+01, 1.703e+01, 6.899e+00, 2.811e+00, 1.147e+00)),
      (1, 3,
       (4.139e-02, 6.248e-03, 9.794e-04, 1.541e-04, 2.366e-05),
       (1.219e+01, 1.715e+00, 2.534e-01, 3.785e-02, 5.685e-03)),
      (3, 1,
       (3.005e-02, 4.746e-03, 9.022e-04, 1.738e-04, 3.266e-05),
       (1.546e+01, 2.392e+00, 3.891e-01, 6.545e-02, 1.125e-02)),
      (3, 3,
       (3.550e-03, 4.852e-05, 7.153e-07, 0.000e+00, 0.000e+00),
       (1.061e+00, 1.353e-02, 1.834e-04, 0.000e+00, 0.000e+00)),
  )  # pyformat: disable
  MG_DIFFS_NORMS_5X5X5 = itertools.product(_AXES, _MG_DIFFS_NORMS_5X5X5)

  _MG_DIFFS_NORMS_5X10X13 = (
      (1, 1,
       (1.129e+00, 4.970e-01, 2.404e-01, 1.233e-01, 7.532e-02),
       (4.097e+03, 1.869e+03, 1.052e+03, 6.715e+02, 4.621e+02)),
      (1, 3,
       (3.442e-01, 8.342e-02, 3.930e-02, 1.986e-02, 1.019e-02),
       (1.298e+03, 3.991e+02, 1.656e+02, 7.731e+01, 3.812e+01)),
      (1, 6,
       (1.217e-01, 3.233e-02, 8.987e-03, 2.732e-03, 8.895e-04),
       (5.260e+02, 1.029e+02, 2.628e+01, 7.578e+00, 2.306e+00)),
      (3, 1,
       (4.097e-01, 9.687e-02, 5.048e-02, 3.266e-02, 2.100e-02),
       (1.697e+03, 6.142e+02, 3.082e+02, 1.735e+02, 1.023e+02)),
      (3, 3,
       (6.064e-02, 1.588e-02, 4.388e-03, 1.315e-03, 4.086e-04),
       (3.539e+02, 6.950e+01, 1.714e+01, 4.681e+00, 1.338e+00)),
      (6, 3,
       (2.177e-02, 2.343e-03, 3.003e-04, 3.719e-05, 4.470e-06),
       (1.002e+02, 8.786e+00, 9.672e-01, 1.116e-01, 1.321e-02)),
      (6, 6,
       (3.142e-03, 5.293e-05, 8.345e-07, 0.000e+00, 0.000e+00),
       (1.063e+01, 1.436e-01, 3.600e-03, 0.000e+00, 0.000e+00)),
  )  # pyformat: disable
  MG_DIFFS_NORMS_5X10X13 = itertools.product(_AXES, _MG_DIFFS_NORMS_5X10X13)

  _MG_DIFFS_NORMS_5X10X13_COARSEST_4X4X4 = (
      (1, 1,
       (1.081e+00, 4.209e-01, 1.784e-01, 8.071e-02, 3.830e-02),
       (3.934e+03, 1.604e+03, 7.637e+02, 4.036e+02, 2.326e+02)),
      (1, 3,
       (2.920e-01, 4.544e-02, 1.382e-02, 4.869e-03, 1.913e-03),
       (1.102e+03, 2.323e+02, 6.300e+01, 2.243e+01, 9.744e+00)),
      (1, 6,
       (8.725e-02, 1.537e-02, 2.849e-03, 5.552e-04, 1.158e-04),
       (4.066e+02, 4.960e+01, 7.556e+00, 1.361e+00, 2.810e-01)),
      (3, 1,
       (3.370e-01, 5.619e-02, 2.544e-02, 1.427e-02, 8.616e-03),
       (1.371e+03, 3.435e+02, 1.325e+02, 6.703e+01, 3.807e+01)),
      (3, 3,
       (2.910e-02, 5.269e-03, 1.299e-03, 3.581e-04, 1.035e-04),
       (1.641e+02, 2.160e+01, 4.971e+00, 1.273e+00, 3.398e-01)),
      (6, 3,
       (7.631e-03, 6.743e-04, 7.591e-05, 8.583e-06, 3.695e-06),
       (3.216e+01, 2.495e+00, 2.448e-01, 2.516e-02, 3.505e-03)),
      (6, 6,
       (5.841e-04, 7.749e-06, 1.311e-06, 1.311e-06, 1.311e-06),
       (2.212e+00, 2.228e-02, 7.635e-04, 1.526e-04, 1.526e-04)),
  )  # pyformat: disable
  MG_DIFFS_NORMS_5X10X13_COARSEST_4X4X4 = itertools.product(
      _AXES, _MG_DIFFS_NORMS_5X10X13_COARSEST_4X4X4)

  def _run_convergence_test(self,
                            solver: Solver,
                            name: Text,
                            starting: TensorOrArray,
                            expected: TensorOrArray,
                            expected_diffs: Sequence[float],
                            expected_norms: Sequence[float],
                            using_tiles: bool = False):
    """Solver convergence test."""
    shape = expected.shape
    if isinstance(expected, tf.Tensor):
      shape = shape.as_list()

    if using_tiles:
      nz = shape[2]
      starting = [tf.convert_to_tensor(starting[:, :, i]) for i in range(nz)]
      b = [tf.zeros_like(starting[0]) for _ in range(nz)]
      fn = functools.partial(solver, b=b)
    else:
      starting = tf.convert_to_tensor(starting)
      b = tf.zeros_like(expected)
      fn = functools.partial(solver, b=b)

    def values_as_str(name, xs):
      s = f'{name} = ('
      for i, x in enumerate(xs):
        s += f'{x:.3e}'
        if i != len(xs) - 1:
          s += ', '
      s += ')'
      return s

    actual_diffs = []
    actual_norms = []
    actual = starting
    runner = TpuRunner(computation_shape=(1,))
    for _ in range(len(expected_diffs)):
      actual = runner.run(fn, [actual])[0]
      array = np.stack(actual, axis=-1) if using_tiles else actual
      actual_diffs.append(np.max(np.abs(array - expected) / expected))
      if using_tiles:
        actual_norms.append(multigrid_3d_utils.poisson_residual_norm(actual, b))
      else:
        actual_norms.append(multigrid_utils.poisson_residual_norm(actual, b))

    # TODO(dmpierce): Revert atol and rtol to small numbers once the changes
    # in cl/398515026 are applied to the "list of 2D tensors" code.
    # see original values at rcl=394519072
    with self.subTest(name + ' diffs'):
      self.assertAllClose(
          expected_diffs,
          actual_diffs,
          rtol=0.25,
          atol=0.19,
          msg=values_as_str('actual diffs', actual_diffs))
    with self.subTest(name + ' norms'):
      self.assertAllClose(
          expected_norms,
          actual_norms,
          rtol=0.3,
          atol=0.19,
          msg=values_as_str('actual norms', actual_norms))

  def _starting_and_expected(self, transpose_axis: int,
                             starting_xy: np.ndarray):
    expected_xys = []
    for i in range(1, starting_xy.shape[0] + 1):
      expected_xys.append(np.ones_like(starting_xy[0, :]) * i)
    expected_xy = np.stack(expected_xys)

    # Trivial 3rd dimension (constant).
    nz = 5

    starting_tiles = ([np.copy(expected_xy)] +
                      [np.copy(starting_xy) for _ in range(nz - 2)] +
                      [np.copy(expected_xy)])
    starting = np.stack(starting_tiles, axis=0)

    expected_tiles = [expected_xy for _ in range(nz)]
    expected = np.stack(expected_tiles, axis=0)

    axes = tuple(range(transpose_axis, 3)) + tuple(range(0, transpose_axis))
    return np.transpose(starting, axes), np.transpose(expected, axes)

  def starting_and_expected_5x5x5(self, transpose_axis: int):
    starting_xy = np.array([[1, 1, 1, 1, 1], [2, 1.5, 2.5, 1.5, 2],
                            [3, 2.5, 2, 2.5, 3], [4, 3.5, 3, 3.5, 4],
                            [5, 5, 5, 5, 5]]).astype(_NP_DTYPE)

    return self._starting_and_expected(transpose_axis, starting_xy)

  def starting_and_expected_5x10x13(self, transpose_axis: int):
    starting_xy = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [2, 6, 9, 7, 2, 8, 5, 3, 4, 2],
                            [3, 9, 7, 2, 8, 5, 3, 4, 6, 3],
                            [4, 7, 2, 8, 5, 3, 4, 9, 6, 4],
                            [5, 2, 8, 5, 3, 4, 9, 6, 7, 5],
                            [6, 8, 5, 3, 4, 2, 6, 9, 7, 6],
                            [7, 5, 3, 4, 2, 6, 9, 7, 8, 7],
                            [8, 5, 3, 4, 6, 9, 7, 2, 8, 8],
                            [9, 4, 3, 6, 9, 7, 2, 8, 5, 9],
                            [10, 4, 6, 9, 7, 2, 8, 5, 3, 10],
                            [11, 4, 9, 7, 2, 8, 5, 3, 4, 11],
                            [12, 4, 9, 7, 2, 4, 5, 3, 4, 12],
                            [13, 13, 13, 13, 13, 13, 13, 13, 13,
                             13]]).astype(_NP_DTYPE)

    return self._starting_and_expected(transpose_axis, starting_xy)
