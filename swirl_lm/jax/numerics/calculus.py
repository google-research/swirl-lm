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

"""Vector calculus operations."""

from typing import TypeAlias

from swirl_lm.jax.numerics import derivatives
from swirl_lm.jax.utility import types

ScalarField: TypeAlias = types.ScalarField
ScalarFieldMap: TypeAlias = types.ScalarFieldMap


def grad(
    deriv_lib: derivatives.Derivatives,
    field_vars: ScalarField | list[ScalarField],
    additional_states: ScalarFieldMap,
) -> list[ScalarField] | list[list[ScalarField]]:
  """Computes the gradient for all variables in `field_vars`.

  Args:
    deriv_lib: An instance of the derivatives library.
    field_vars: A list of 3D tensor variables, or a single 3D tensor variable.
    additional_states: Mapping that contains helper variables.

  Returns:
    The gradients of all variables in `field_vars`. If `field_vars` is a list,
    the first index of the returned structure indicates the variable sequence,
    and the second index is the direction of the gradient, which can only be 0,
    1, and 2. If `field_vars` is a single tensor, the returned structure is a
    list of length 3, where the first index is the direction of the gradient,
    which can only be 0, 1, and 2. The direction of the gradient is the same as
    the grid_params.data_axis_order, i.e., if
    grid_params.data_axis_order=('z','x','y'), then the order of derivatives is
    [df_dz, df_dx, df_dy].

  Raises:
    TypeError if `field_vars` is not `ScalarField` or `list[ScalarField]`.
  """
  if isinstance(field_vars, ScalarField):
    return [
        deriv_lib.deriv_centered(
            field_vars,
            deriv_lib.grid_params.data_axis_order[dim],
            additional_states,
        )
        for dim in (0, 1, 2)
    ]
  elif isinstance(field_vars, list) or isinstance(field_vars, tuple):
    return [
        [
            deriv_lib.deriv_centered(
                field_var,
                deriv_lib.grid_params.data_axis_order[dim],
                additional_states,
            )
            for dim in (0, 1, 2)
        ]
        for field_var in field_vars
    ]
  raise TypeError(f'Unknown type for `field_vars`: {type(field_vars)}.')


def divergence(
    deriv_lib: derivatives.Derivatives,
    u: ScalarField,
    v: ScalarField,
    w: ScalarField,
    additional_states: ScalarFieldMap,
) -> ScalarField:
  """Computes the divergence of `field_var`.

  Args:
    deriv_lib: An instance of the derivatives library.
    u: X-component of the vector field.
    v: Y-component of the vector field.
    w: Z-component of the vector field.
    additional_states: Mapping that contains helper variables.

  Returns:
    The divergence of `field_var`.

  Raises:
    ValueError: If the length of `field_var` is not 3.
  """
  du_dx = deriv_lib.deriv_centered(u, 'x', additional_states)
  dv_dy = deriv_lib.deriv_centered(v, 'y', additional_states)
  dw_dz = deriv_lib.deriv_centered(w, 'z', additional_states)
  return du_dx + dv_dy + dw_dz


def laplacian(
    deriv_lib: derivatives.Derivatives,
    f: ScalarField,
    additional_states: ScalarFieldMap,
) -> ScalarField:
  """Computes the Laplacian of `f`.

  Args:
    deriv_lib: An instance of the derivatives library.
    f: The field to compute the Laplacian of.
    additional_states: Mapping that contains helper variables.

  Returns:
    The Laplacian Δf.
  """
  ddx = deriv_lib.deriv_2_node(f, 'x', additional_states)
  ddy = deriv_lib.deriv_2_node(f, 'y', additional_states)
  ddz = deriv_lib.deriv_2_node(f, 'z', additional_states)
  return ddx + ddy + ddz
