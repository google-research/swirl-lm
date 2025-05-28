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

"""Time integration / time stepping."""

from typing import List, Sequence

from swirl_lm.numerics import numerics_pb2
from swirl_lm.utility import common_ops
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal

TimeIntegrationScheme = numerics_pb2.TimeIntegrationScheme

# Coefficients in the 3rd order Runge Kutta time integration scheme.
# For du / dt = f(u), from u_{i} to u_{i+1}, the following steps are applied:
# u_1 = u_{i} + dt * f(u_{i})
# u_2 = c11 * u_{i} + c12 * (u_1 + dt * f(u_1))
# u_{i+1} = c21 * u_{i} + c22 * (u_2 + dt * f(u_2))
_RK3_COEFFS = {'c11': 0.75, 'c12': 0.25, 'c21': 1.0 / 3.0, 'c22': 2.0 / 3.0}


def _rk3(rhs, dt: float, var: Sequence[FlowFieldVal]) -> List[FlowFieldVal]:
  """Computes the time integration using the 3rd order Runge-Kutta method.

  The time integration of dvar / dt = rhs(var_0, var_n) is computed.

  Args:
    rhs: The function that takes a sequence of variables and computes the update
      of these variables at the current time step.
    dt: The size of the time step.
    var: A random number of variables as lists of `tf.Tensor` representing the
      initial condition of the 3D fields (as a list of 2D x-y slices).

  Returns:
    The variable fields in the next time step.
  """
  # The first RK step.
  rhs_1 = rhs(*var)

  var_1 = [
      tf.nest.map_structure(
          lambda var_i_j, rhs_1_i_j: var_i_j + dt * rhs_1_i_j, var_i, rhs_1_i
      )
      for var_i, rhs_1_i in zip(var, rhs_1)
  ]

  # The second RK step.
  rhs_2 = rhs(*var_1)

  var_2 = []
  for i in range(len(var)):
    var_2.append(
        tf.nest.map_structure(
            lambda var_i_j, var_1_i_j, rhs_2_i_j: _RK3_COEFFS['c11'] * var_i_j
            + _RK3_COEFFS['c12'] * (var_1_i_j + dt * rhs_2_i_j),
            var[i],
            var_1[i],
            rhs_2[i],
        )
    )

  # The third RK step.
  rhs_3 = rhs(*var_2)

  var_3 = []
  for i in range(len(var)):
    var_3.append(
        tf.nest.map_structure(
            lambda var_i_j, var_2_i_j, rhs_3_i_j: _RK3_COEFFS['c21'] * var_i_j
            + _RK3_COEFFS['c22'] * (var_2_i_j + dt * rhs_3_i_j),
            var[i],
            var_2[i],
            rhs_3[i],
        )
    )

  return var_3


def _crank_nicolson_explicit_subiteration(
    rhs, dt: float, var_0: Sequence[FlowFieldVal],
    var_n: Sequence[FlowFieldVal]) -> List[FlowFieldVal]:
  """Computes the time integration with the semi-implicit Crank-Nicolson method.

  The time integration of dvar / dt = rhs(var_0, var_n) is computed.

  Args:
    rhs: The function that takes a sequence of variables and computes the update
      of these variables at the current time step.
    dt: The size of the time step.
    var_0: A sequence of lists of `tf.Tensor`s representing the initial
      condition of the 3D fields (as a list of 2D x-y slices).
    var_n: A random number of variables as lists of `tf.Tensor` representing the
      3D fields (as a list of 2D x-y slices), which is a guess of the velocity
      at the next time step. It is used in semi-implicit schemes.

  Returns:
    The variable fields in the next time step.
  """
  var_m = []
  for i in range(len(var_0)):
    var_m.append(tf.nest.map_structure(
        common_ops.average, var_0[i], var_n[i]))

  rhs_m = rhs(*var_m)
  if len(var_0) == 1:
    rhs_m = (rhs_m,)

  var_next = []
  for i in range(len(var_0)):
    var_next.append(tf.nest.map_structure(
        lambda var_0_i_j, rhs_m_i_j: var_0_i_j + dt * rhs_m_i_j,
        var_0[i], rhs_m[i]))

  return var_next


def time_advancement_explicit(
    rhs, dt: float, scheme: TimeIntegrationScheme,
    var_0: Sequence[FlowFieldVal],
    var_n: Sequence[FlowFieldVal]) -> List[FlowFieldVal]:
  """Computes the time integration using the selected explicit scheme.

  The time integration of dvar / dt = rhs(var_0, var_n) is computed.

  Args:
    rhs: The function that takes a sequence of variables and computes the update
      of these variables at the current time step.
    dt: The size of the time step.
    scheme: The scheme to be used for the time integration.
    var_0: A random number of variables as lists of `tf.Tensor` representing the
      initial condition of the 3D fields (as a list of 2D x-y slices).
    var_n: A random number of variables as lists of `tf.Tensor` representing the
      3D fields (as a list of 2D x-y slices), which is a guess of the velocity
      at the next time step. It is used in semi-implicit schemes.

  Returns:
    The variable fields in the next time step.
  """
  if scheme == (TimeIntegrationScheme.TIME_SCHEME_RK3):
    # In RK3, the right hand side terms are computed based on the starting
    # field values.
    return _rk3(rhs, dt, var_0)
  if scheme == (TimeIntegrationScheme.TIME_SCHEME_CN_EXPLICIT_ITERATION):
    return _crank_nicolson_explicit_subiteration(rhs, dt, var_0, var_n)
  else:
    raise NotImplementedError('Scheme {} is not implemented yet.'.format(
        TimeIntegrationScheme.Name(scheme)))
