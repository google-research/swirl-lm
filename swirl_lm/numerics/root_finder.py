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

"""A library for root finding methods."""

import collections
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from swirl_lm.numerics import algebra
from swirl_lm.utility import common_ops
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap

# A list of 3D tensors, which is a list of 2D tensors.
Fields = Sequence[FlowFieldVal]
OutputFields = List[FlowFieldVal]

# A list of Fields, which shows up in the Jacobian function.
FieldMatrix = Sequence[Sequence[FlowFieldVal]]

# A small number to be used as perturbation to the solution.
_EPS = 1e-4

_NewtonState = collections.namedtuple(
    'NewtonState', ('x', 'x0', 'f', 'best_residual', 'best_x'))


def newton_method_multi_dim(
    objective_fn: Callable[[Fields], OutputFields],
    initial_position: Fields,
    max_iterations: int,
    value_tolerance: Optional[float] = None,
    position_tolerance: Optional[float] = None,
    analytical_jacobian_fn: Optional[Callable[[Fields], FieldMatrix]] = None,
    replicas: Optional[np.ndarray] = None,
) -> OutputFields:
  """Finds the root of `objective_fn` with the Newton method.

  Args:
    objective_fn: The objective function that seeks for root.
    initial_position: The initial guess of the solution.
    max_iterations: The maximum number of iterations allowed in the Newton
      solver before the result converges. Considering the efficiency on TPU, the
      solver runs for a fixed number of iterations that is specified by this
      number.
    value_tolerance: An optional quantity that specifies the maximum absolute
      error allowed in the function evaluation. This provides an additional
      constraint on the termination of the Newton iterations.
    position_tolerance: An optional quantity that specifies the maximum relative
      error allowed in the solution. This provides an additional constraint on
      the termination of the Newton iterations.
    analytical_jacobian_fn: An optional function that computes the gradient of
      `objective_fn`. If it's absent, the method reduces to the secant method,
      in which case this function will be estimated using the finite difference
      gradient. The perturbation is set to the negative power of 2 that is
      closest to the floating point precision of the initial guess' data type.
      In cases where the solution is zero, a small number specified by `_EPS` is
      used as the perturbation.
    replicas: The mapping from the global coordinate of the core to
      `replica_id`, useful if one needs to cache and desire the best solution.

  Returns:
    The solution of `objective_fn`.

  Raises:
    ValueError: If any tolerance is negative.
  """
  if max_iterations <= 0:
    return list(initial_position)

  dimensions = len(initial_position)

  def valid_tol(tol):
    """Tolerance should be either None or non-negative."""
    return tol is None or tol >= 0

  if not (valid_tol(value_tolerance) and valid_tol(position_tolerance)):
    raise ValueError(('Tolerance should be non-negative: (value_tolerance, '
                      'position_tolerance) = (%s, %s) < 0.') %
                     (str(value_tolerance), str(position_tolerance)))

  if initial_position[0][0].dtype == tf.float64:
    eps = np.finfo(np.float64(1.0)).resolution
  else:
    # Default perturbation is computed based on single precision.
    eps = np.finfo(np.float32(1.0)).resolution

  # eps       raw     processed
  # float32   1e-06   1.53e-05
  # float64   1e-15   1.42e-14
  eps = np.power(2.0, np.ceil(np.log(10.0 * eps) / np.log(2.0)))

  def numerical_jacobian_fn(x: Fields) -> FieldMatrix:
    """The Jacobian estimated with the finite difference method."""

    def cp(x: Fields) -> OutputFields:
      return [x_i[:] for x_i in x]

    def numerical_jacobian_fn_1d(x: Fields, dim: int) -> OutputFields:
      """Computes the Jacobian along the given dimension."""
      x1 = cp(x)
      x2 = cp(x)

      # Apply perturbation to `x` only along the given dimension.
      dx_dim = tf.nest.map_structure(
          lambda x_dim_i: tf.maximum(eps * tf.abs(x_dim_i), _EPS), x[dim])

      x1[dim] = tf.nest.map_structure(
          lambda x_dim_i, dx_dim_i: x_dim_i - dx_dim_i * 0.5, x[dim], dx_dim)
      x2[dim] = tf.nest.map_structure(
          lambda x_dim_i, dx_dim_i: x_dim_i + dx_dim_i * 0.5, x[dim], dx_dim)

      f1 = objective_fn(*x1)
      f2 = objective_fn(*x2)

      return [
          tf.nest.map_structure(lambda a, b, c: tf.math.divide_no_nan(a - b, c),
                                f2_i, f1_i, dx_dim)
          for f1_i, f2_i in zip(f1, f2)
      ]

    if dimensions == 1:
      return [numerical_jacobian_fn_1d(x, 0)]
    elif dimensions == 2:
      #  (a, b)
      #  (c, d)
      a, c = numerical_jacobian_fn_1d(x, 0)
      b, d = numerical_jacobian_fn_1d(x, 1)
      return [[a, b], [c, d]]
    elif dimensions == 3:
      #  (a, b, c)
      #  (d, e, f)
      #  (g, h, i)
      a, d, g = numerical_jacobian_fn_1d(x, 0)
      b, e, h = numerical_jacobian_fn_1d(x, 1)
      c, f, i = numerical_jacobian_fn_1d(x, 2)
      return [[a, b, c], [d, e, f], [g, h, i]]
    else:
      raise ValueError(
          'Not implemented for Newton method with number of variables = %d 3.' %
          dimensions)

  default_residual = tf.constant(-1., dtype=initial_position[0][0].dtype)

  def body(i: tf.Tensor,
           states: _NewtonState) -> Tuple[tf.Tensor, _NewtonState]:
    """The main function for one Newton iteration."""
    x = states.x
    f = objective_fn(*x)

    if analytical_jacobian_fn is None:
      df = numerical_jacobian_fn(x)
    else:
      df = analytical_jacobian_fn(*x)


    if dimensions == 1:
      dx = [tf.nest.map_structure(tf.math.divide_no_nan, f[0], df[0][0])]
    elif dimensions == 2:
      dx = algebra.solve_2x2(df, f)
    elif dimensions == 3:
      dx = algebra.solve_3x3(df, f)
    else:
      raise ValueError(
          'Not implemented for Newton method with number of variables = %d 3.' %
          dimensions)

    x1 = tf.nest.map_structure(tf.math.subtract, x, dx)

    if replicas is None:
      best_residual, best_x = default_residual, x
    else:
      norm_type = common_ops.NormType.L1
      residuals = [
          common_ops.compute_norm(f_i, (norm_type,), replicas)[norm_type.name]
          for f_i in f
      ]
      residual = common_ops.compute_norm(tf.stack(residuals), (norm_type,),
                                         replicas)[norm_type.name]

      best_residual, best_x = tf.cond(
          pred=tf.math.logical_or(
              tf.less_equal(i, 0), residual <= states.best_residual),
          # New best.
          true_fn=lambda: (residual, x),
          # Cached best.
          false_fn=lambda: (states.best_residual, states.best_x))

    return (
        i + 1,
        _NewtonState(
            x=x1,
            x0=x,
            f=f,
            # Cache best residual & result.
            best_residual=best_residual,
            best_x=best_x))

  def cond(i: tf.Tensor, states: _NewtonState) -> bool:
    """The stop condition of Newton iterations."""
    cond_value_not_converge = True
    if value_tolerance is not None:
      cond_value_not_converge = tf.reduce_any(
          [[tf.greater(tf.abs(f_i), value_tolerance)
            for f_i_j in f_i]
           for f_i in states.f])

    cond_position_not_converge = True
    if position_tolerance is not None:
      # pylint: disable=g-complex-comprehension
      cond_position_not_converge = tf.reduce_any([[
          tf.greater(
              tf.abs(x0_i_j - x_i_j), position_tolerance *
              (1.0 + tf.abs(x_i_j))) for x0_i_j, x_i_j in zip(x0_i, x_i)
      ] for x0_i, x_i in zip(states.x0, states.x)])
      # pylint: enable=g-complex-comprehension

    cond_max_iter = tf.less(i, max_iterations)
    return tf.math.reduce_all(
        input_tensor=(cond_max_iter, cond_value_not_converge,
                      cond_position_not_converge))

  i0 = tf.constant(0)
  states_0 = _NewtonState(
      x=initial_position,
      x0=[[1.0 + 2.0 * tf.abs(x)
           for x in initial_position_i]
          for initial_position_i in initial_position],
      f=objective_fn(*initial_position),  # pytype: disable=wrong-arg-types
      # Cache best residual & result.
      best_residual=default_residual,
      best_x=initial_position)
  _, sol = tf.while_loop(
      cond=cond, body=body, loop_vars=(i0, states_0), back_prop=False)

  return sol.best_x


def newton_method(
    objective_fn: Callable[[FlowFieldVal], FlowFieldVal],
    initial_position: FlowFieldVal,
    max_iterations: int,
    value_tolerance: Optional[float] = None,
    position_tolerance: Optional[float] = None,
    analytical_jacobian_fn: Optional[Callable[[FlowFieldVal],
                                              FlowFieldVal]] = None,
) -> FlowFieldVal:
  """Finds the root of `objective_fn` with the Newton method.

  Args:
    objective_fn: The objective function that seeks for root.
    initial_position: The initial guess of the solution.
    max_iterations: The maximum number of iterations allowed in the Newton
      solver before the result converges. Considering the efficiency on TPU, the
      solver runs for a fixed number of iterations that is specified by this
      number.
    value_tolerance: An optional quantity that specifies the maximum error
      allowed in the function evaluation. This provides an additional constraint
      on the termination of the Newton iterations.
    position_tolerance: An optional quantity that specifies the maximum absolute
      error allowed in the solution. This provides an additional constraint
      on the termination of the Newton iterations.
    analytical_jacobian_fn: An optional function that computes the gradient of
      `objective_fn`. If it's absent, the method reduces to the secant method,
      in which case this function will be estimated using the finite difference
      gradient. The perturbation is set to the negative power of 2 that is
      closest to the floating point precision of the initial guess' data type.
      In cases where the solution is zero, a small number specified by
      `_EPS` is used as the perturbation.

  Returns:
    The solution of `objective_fn`.
  """
  dtype = tf.nest.flatten(initial_position)[0].dtype
  eps = np.finfo(dtype.as_numpy_dtype).resolution
  eps = np.power(2.0, np.ceil(np.log(10.0 * eps) / np.log(2.0)))

  def numerical_jacobian_fn(x: FlowFieldVal) -> FlowFieldVal:
    """The Jacobian estimated with the finite difference method."""
    dx = tf.nest.map_structure(lambda x_i: eps * tf.abs(x_i), x)
    dx = tf.nest.map_structure(
        lambda dx_i: tf.where(  # pylint: disable=g-long-lambda
            tf.equal(dx_i, 0.0), _EPS * tf.ones_like(dx_i), dx_i), dx)
    x1 = tf.nest.map_structure(lambda x_i, dx_i: x_i - dx_i / 2.0, x, dx)
    x2 = tf.nest.map_structure(lambda x_i, dx_i: x_i + dx_i / 2.0, x, dx)
    return tf.nest.map_structure(lambda f1, f0, dx_i: (f1 - f0) / dx_i,
                                 objective_fn(x2), objective_fn(x1), dx)

  jacobian_fn = (
      numerical_jacobian_fn
      if analytical_jacobian_fn is None else analytical_jacobian_fn)

  def body(i: tf.Tensor,
           states: FlowFieldMap) -> Tuple[tf.Tensor, FlowFieldMap]:
    """The main function for one Newton iteration."""
    x = states['x']
    f = objective_fn(x)
    df = jacobian_fn(x)
    h = tf.nest.map_structure(tf.math.divide_no_nan, f, df)
    x1 = tf.nest.map_structure(tf.math.subtract, x, h)
    return (i + 1, {'x': x1, 'x0': x, 'f': f})

  def cond(i: tf.Tensor, states: FlowFieldMap) -> bool:
    """The stop condition of Newton iterations."""
    cond_value_not_converge = True
    if value_tolerance is not None:
      cond_value_not_converge = tf.reduce_any(tf.nest.map_structure(
          lambda f_i: tf.greater(tf.abs(f_i), value_tolerance), states['f']))

    cond_position_not_converge = True
    if position_tolerance is not None:
      cond_position_not_converge = tf.reduce_any(
          tf.nest.map_structure(
              lambda x0_i, x_i: tf.greater(  # pylint: disable=g-long-lambda
                  tf.abs(x0_i - x_i), position_tolerance * (1.0 + tf.abs(x_i))),
              states['x0'], states['x']))

    cond_max_iter = tf.less(i, max_iterations)
    return tf.math.reduce_all(
        [cond_max_iter, cond_value_not_converge, cond_position_not_converge])

  i0 = tf.constant(0)
  states_0 = {
      'x':
          initial_position,
      'x0':
          tf.nest.map_structure(lambda x: 1.0 + 2.0 * tf.abs(x),
                                initial_position),
      'f':
          objective_fn(initial_position),
  }
  _, sol = tf.while_loop(
      cond,
      body,
      loop_vars=(i0, states_0),
      back_prop=False)

  return sol['x']
