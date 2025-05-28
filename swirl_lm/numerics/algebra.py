# Copyright 2025 The swirl_lm Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A library for common linear algebra operations."""

from typing import List, Sequence, Tuple

from swirl_lm.utility import types
import tensorflow as tf


FlowFieldVal = types.FlowFieldVal
Fields = Sequence[FlowFieldVal]
OutputFields = List[FlowFieldVal]
# A matrix of Fields.
FieldMatrix = Sequence[Sequence[FlowFieldVal]]


def _validate_matrix_shape(matrix: FieldMatrix, shape: Tuple[int, int]):
  """Validates field matrix shape is the same as expected."""
  if len(matrix) != shape[0]:
    raise ValueError(
        'Invalid matrix row len = %d: not consistent with expected shape: %s.' %
        (len(matrix), shape))

  for m in matrix:
    if len(m) != shape[1]:
      raise ValueError(
          'Invalid matrix col len = %d: not consistent with expected shape: %s.'
          % (len(m), shape))


def det_2x2(matrix: FieldMatrix) -> FlowFieldVal:
  r"""Computes determinant for 2x2 matrices elementwise in 3D tensors.

  M is a matrix of 3D tensors, with rows being the matrix rows, and columns
  being matrix columns, i.e. M \equiv [[a, b], [c, d]] corresponds to:
  M = (a, b)
      (c, d)
  where {`a`, `b`, `c`, `d`} are all 3D tensors.

  Note that for each cube in the 3D space, it has a 2x2 matrix on M, and all
  cubes are completely decoupled and independent.

  det(M) = a * d - b * c

  Args:
    matrix: The 2x2 matrix to get elementwise (for each cube) determinant.

  Returns:
    The elementwise determinant of the 2x2 matrix.
  """
  _validate_matrix_shape(matrix, (2, 2))

  det = lambda a, b, c, d: a * d - b * c

  a, b = matrix[0]
  c, d = matrix[1]

  return tf.nest.map_structure(det, a, b, c, d)


def det_3x3(matrix: FieldMatrix) -> FlowFieldVal:
  r"""Computes determinant for 3x3 matrices elementwise in 3D tensors.

  M is a matrix of 3D tensors:
  M = (a, b, c)
      (d, e, f)
      (g, h, i)

  det(M) = + a * det([[e, f], [h, i]])
           - b * det([[d, f], [g, i]])
           + c * det([[d, e], [g, h]])

  Args:
    matrix: The 3x3 matrix to get elementwise (for each cube) determinant.

  Returns:
    The elementwise determinant of the 3x3 matrix.
  """
  _validate_matrix_shape(matrix, (3, 3))

  det2 = lambda a, b, c, d: a * d - b * c

  def det3(a, b, c, d, e, f, g, h, i):
    return a * det2(e, f, h, i) - b * det2(d, f, g, i) + c * det2(d, e, g, h)

  a, b, c = matrix[0]
  d, e, f = matrix[1]
  g, h, i = matrix[2]

  return tf.nest.map_structure(det3, a, b, c, d, e, f, g, h, i)


def solve_2x2(matrix: FieldMatrix,
              rhs: Fields) -> OutputFields:
  """Solves a linear system of 2x2 for each cube in 3D tensors.

  M * x = rhs elementwise for each cube, and one can refer to `det_2x2` for `M`
  structure:
    (a, b) * (x0)   (e)
    (c, d)   (x1) = (f)
  When the 2-by-2 matrices are singular for some cubes, the solution is set to
  be trivial (with `tf.math.divide_no_nan`).

  Args:
    matrix: The 2x2 matrix as a linear operator elementwise for each cube.
    rhs: The vector on the right hand side.

  Returns:
    The solution `x`, so that M * x = rhs holds elementwise for each cube.
  """
  _validate_matrix_shape(matrix, (2, 2))

  a, b = matrix[0]
  c, d = matrix[1]
  e, f = rhs

  inv_factor = det_2x2(matrix)

  return [
      tf.nest.map_structure(tf.math.divide_no_nan, det_2x2([
          [e, b],
          [f, d],
      ]), inv_factor),
      tf.nest.map_structure(tf.math.divide_no_nan, det_2x2([
          [a, e],
          [c, f],
      ]), inv_factor),
  ]


def solve_3x3(matrix: FieldMatrix,
              rhs: Fields) -> OutputFields:
  """Solves a linear system of 3x3 for each cube in 3D tensors.

  M * x = rhs elementwise for each cube, and one can refer to `det_2x2` for `M`
  structure:
  M * x = rhs:
    (a, b, c)    (x0)   (j)
    (d, e, f)  * (x1) = (k)
    (g, h, i)    (x2)   (l)
  When the 3-by-3 matrices are singular for some cubes, the solution is set to
  be trivial (with `tf.math.divide_no_nan`).

  Args:
    matrix: The 3x3 matrix as a linear operator elementwise for each cube.
    rhs: The vector on the right hand side.

  Returns:
    The solution `x`, so that M * x = rhs holds elementwise for each cube.
  """
  _validate_matrix_shape(matrix, (3, 3))

  a, b, c = matrix[0]
  d, e, f = matrix[1]
  g, h, i = matrix[2]
  j, k, l = rhs

  inv_factor = det_3x3(matrix)

  return [
      tf.nest.map_structure(tf.math.divide_no_nan,
                            det_3x3([
                                [j, b, c],
                                [k, e, f],
                                [l, h, i],
                            ]), inv_factor),
      tf.nest.map_structure(tf.math.divide_no_nan,
                            det_3x3([
                                [a, j, c],
                                [d, k, f],
                                [g, l, i],
                            ]), inv_factor),
      tf.nest.map_structure(tf.math.divide_no_nan,
                            det_3x3([
                                [a, b, j],
                                [d, e, k],
                                [g, h, l],
                            ]), inv_factor),
  ]
