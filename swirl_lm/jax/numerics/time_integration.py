# Copyright 2026 The swirl_lm Authors.
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

# Copyright 2021 Google LLC
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
"""Time integration / time stepping."""


from typing import Sequence

from swirl_lm.jax.numerics import numerics_pb2 as numerics_pb2_jax
from swirl_lm.jax.utility import types
from swirl_lm.numerics import numerics_pb2

ScalarField = types.ScalarField

# Coefficients in the 3rd order Runge Kutta time integration scheme.
# For du / dt = f(u), from u_{i} to u_{i+1}, the following steps are applied:
# u_1 = u_{i} + dt * f(u_{i})
# u_2 = c11 * u_{i} + c12 * (u_1 + dt * f(u_1))
# u_{i+1} = c21 * u_{i} + c22 * (u_2 + dt * f(u_2))
_RK3_COEFFS = {'c11': 0.75, 'c12': 0.25, 'c21': 1.0 / 3.0, 'c22': 2.0 / 3.0}


def _rk3(rhs, dt: float, var: Sequence[ScalarField]) -> list[ScalarField]:
  """Computes the time integration using the 3rd order Runge-Kutta method.

  The time integration of dvar / dt = rhs(var_0, var_n) is computed.

  Args:
    rhs: The function that takes a sequence of variables and computes the update
      of these variables at the current time step.
    dt: The size of the time step.
    var: A sequence of 3D `jax.Array` representing the initial condition of the
      3D fields.

  Returns:
    The variable fields in the next time step.
  """
  # The first RK step.
  rhs_1 = rhs(*var)

  var_1 = [var_i + dt * rhs_1_i for var_i, rhs_1_i in zip(var, rhs_1)]

  # The second RK step.
  rhs_2 = rhs(*var_1)

  var_2 = [
      _RK3_COEFFS['c11'] * var[i]
      + _RK3_COEFFS['c12'] * (var_1[i] + dt * rhs_2[i])
      for i in range(len(var))
  ]

  # The third RK step.
  rhs_3 = rhs(*var_2)

  var_3 = [
      _RK3_COEFFS['c21'] * var[i]
      + _RK3_COEFFS['c22'] * (var_2[i] + dt * rhs_3[i])
      for i in range(len(var))
  ]

  return var_3


def _crank_nicolson_explicit_subiteration(
    rhs, dt: float, var_0: Sequence[ScalarField], var_n: Sequence[ScalarField]
) -> list[ScalarField]:
  """Computes the time integration with the semi-implicit Crank-Nicolson method.

  The time integration of dvar / dt = rhs(var_0, var_n) is computed.

  Args:
    rhs: The function that takes a sequence of variables and computes the update
      of these variables at the current time step.
    dt: The size of the time step.
    var_0: A sequence of 3D `jax.Array` representing the initial condition of
      the 3D fields.
    var_n: A sequence of 3D `jax.Array` representing a guess of the variable at
      the next time step. It is used in semi-implicit schemes.

  Returns:
    The variable fields in the next time step.
  """
  var_m = [0.5 * (var_0[i] + var_n[i]) for i in range(len(var_0))]

  rhs_m = rhs(*var_m)
  if len(var_0) == 1:
    rhs_m = (rhs_m,)

  var_next = [var_0[i] + dt * rhs_m[i] for i in range(len(var_0))]

  return var_next


def time_advancement_explicit(
    rhs,
    dt: float,
    scheme: (
        numerics_pb2_jax.TimeIntegrationScheme.ValueType
        | numerics_pb2.TimeIntegrationScheme.ValueType
    ),
    var_0: Sequence[ScalarField],
    var_n: Sequence[ScalarField],
) -> list[ScalarField]:
  """Computes the time integration using the selected explicit scheme.

  The time integration of dvar / dt = rhs(var_0, var_n) is computed.

  Args:
    rhs: The function that takes a sequence of variables and computes the update
      of these variables at the current time step.
    dt: The size of the time step.
    scheme: The scheme to be used for the time integration.
    var_0: A sequence of 3D `jax.Array` representing the initial condition of
      the 3D fields.
    var_n: A sequence of 3D `jax.Array` representing a guess of the variable at
      the next time step. It is used in semi-implicit schemes.

  Returns:
    The variable fields in the next time step.
  """
  if scheme == (
      numerics_pb2_jax.TimeIntegrationScheme.TIME_SCHEME_RK3
      | numerics_pb2.TimeIntegrationScheme.TIME_SCHEME_RK3
  ):
    # In RK3, the right hand side terms are computed based on the starting
    # field values.
    return _rk3(rhs, dt, var_0)
  if scheme == (
      numerics_pb2_jax.TimeIntegrationScheme.TIME_SCHEME_CN_EXPLICIT_ITERATION
      | numerics_pb2.TimeIntegrationScheme.TIME_SCHEME_CN_EXPLICIT_ITERATION
  ):
    return _crank_nicolson_explicit_subiteration(rhs, dt, var_0, var_n)
  else:
    raise NotImplementedError(
        'Scheme {} is not implemented yet.'.format(scheme)
    )
