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

"""Library for computing derivatives.

Provides functionality for computing derivatives on 3D fields. The aim is to
abstract away the underlying algebraic kernel operations on raw arrays.
"""

from typing import TypeAlias

from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import stretched_grid_util
from swirl_lm.jax.utility import types
from typing_extensions import Self

ScalarField: TypeAlias = types.ScalarField
ScalarFieldMap: TypeAlias = types.ScalarFieldMap


class Derivatives:
  """Library for computing derivatives.

  Currently, the derivatives implemented are in the context of a second-order
  finite-difference code, where variable values are located at nodes on a
  colocated mesh, and fluxes are located at faces.  The computational mesh must
  be uniform in the coordinates.

  The library allows for 1D coordinate transforms in each dimension, i.e.,
  allows for x=x(q⁰), y=y(q¹), z=z(q²).  If a coordinate transform
  ("stretched grid") is used, the scale factors are h₀=dx/dq⁰, h₁=dy/dq¹, and
  h₂=dz/dq².


  The library computes ∂f/∂xⱼ, where x=x₀, y=x₁, z=x₂.  Note that even when
  using transformed coordinates, for which ∂f/∂xⱼ = 1/hⱼ ∂f/∂qʲ (no sum over j),
  this library still computes and returns ∂f/∂xⱼ rather than ∂f/∂qʲ.

  Implementation details:
  The convention in SwirlLM is that faces to the *left* of a node are given the
  same index. See the diagram below, where `.` represents a node and `|`
  represents a face. For simplicity in this documentation, the diagram and
  descriptions use 1 dimension, but the functionality works on 3D fields.

  Index & coord loc.         i-1        i        i+1
                         |    .    |    .    |    .    |
  Index                 i-1        i        i+1
  Coord. loc.           i-3/2     i-1/2     i+1/2

  * For an array f evaluated on nodes: index i <==> coordinate location x_i
  * For an array f_face evaluated at faces: index i <==> coordinate location
    x_{i-1/2}.

  * Note, the value of f_face at the boundary (e.g., index 0 on a boundary
    replica) is not necessarily meaningful, because its location would be
    outside of the domain. This won't pollute computations, because field values
    at these points will be updated by the halo exchange / boundary conditions.

  Example: Given f on nodes, compute ∂f/∂x on faces.
    * Given f on nodes, the derivative at i-1/2 is stored at index i, and is
    given by df_dx_face[i] = (f[i] - f[i-1]) / dx. Note a backward sum is used.
    If a coordinate transform x(q) is involved, we divide by (h_face[i] * dq)
    rather than by dx.
    * Use method deriv_node_to_face()

  Example: Given f on faces, compute ∂f/∂x on nodes.
    * Given f on faces, the derivative at i is stored at index i, and is given
    by df_dx[i] = (f_face[i+1] - f_face[i]) / dx. Note a forward sum is used.
    If a coordinate transform x(q) is involved, we divide by (h[i] * dq) rather
    than by dx.
    * Use method deriv_face_to_node()

  Example: Given f on nodes, compute ∂f/∂x on nodes.
    * Given f on nodes, use a centered stencil to compute ∂f/∂x on nodes.  Then
    df_dx[i] = (f[i+1] - f[i-1]) / (2 * dx). If a coordinate transform x(q) is
    involved, we divide by (h[i] * dq) rather than by dx.
    * Use method deriv_centered()

  This library supports representing a 3D field as either a 3D tensor or as a
  list of 2D tensors.

  Some of the code complexity stems from wanting to minimize the memory
  consumption. This means keeping the scale factors as 1D arrays, and not
  creating them as 3D field unnecessarily. As a result, different logic is
  required when the code represents fields as either lists of 2D tensors vs. 3D
  tensors.

  Additionally, some of the code complexity stems from maintaining the case of
  non-stretched grids as a separate case. It is possible to eliminate this
  branching if a non-stretched grid just used a scale factor set to an array of
  ones, but that approach is not used here both for clarity and for potential
  efficiency gain.

  Attributes:
    kernel_op: Kernel op library.
    grid_params: The grid parametrization object.
  """

  def __init__(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      grid_params: grid_parametrization.GridParametrization,
  ):
    """Instantiates a derivatives library.

    A stretched grid is used in dimension j only if
    grid_params.use_stretched_grid[j] is True.

    If grid_params.use_stretched_grid[j] == True, the scale factors must be
    provided when calling the derivative methods below. They must be provided as
    (key, value) pairs in the `additional_states` mapping, with keys
    `stretched_grid_h{j}` and `stretched_grid_h{j}_face`.

    The scale factors themselves must be 1D tensors that are already in the
    correct shape for broadcasting to the 3D fields (as either a 3D tensor or
    list of 2D tensors). E.g., the scale factor in the y dimension must have
    shape (1, 1, ny) for 3D tensor or (1, ny) for list of 2D tensors, where ny
    is the number of grid points (including halos) in that dimension on one
    replica.

    The `grid_spacings` input holds the coordinate grid spacing for each
    dimension.  E.g., for dim 0,
      * If using a uniform Cartesian grid in dim 0 (i.e., not a stretched
      grid), then the grid_spacing is just the physical dx.
      * If using a stretched grid in dim 0, then the grid spacing is the
      distance between nodes in the transformed coordinate q⁰ (the mesh is
      assumed to be uniform in the new coordinate).

    Args:
      kernel_op: Kernel op library.
      grid_params: The grid parametrization object.
    """
    self._kernel_op = kernel_op
    self.grid_params = grid_params

  def _backward_difference(self, array: ScalarField, axis: str) -> ScalarField:
    """Computes the backward difference along the given axis."""
    return self._kernel_op.apply_kernel_op(array, 'kd', axis)

  def _forward_difference(self, array: ScalarField, axis: str) -> ScalarField:
    """Computes the forward difference along the given axis."""
    return self._kernel_op.apply_kernel_op(array, 'kd+', axis)

  def _centered_difference(self, array: ScalarField, axis: str) -> ScalarField:
    """Computes the centered difference along the given axis."""
    return self._kernel_op.apply_kernel_op(array, 'kD', axis)

  def deriv_node_to_face(
      self,
      array_node: ScalarField,
      axis: str,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Given the value of a field f at nodes, returns the derivative at faces.

    Compute the partial derivative ∂f/∂{axis}. The resulting field is evaluated
    on faces in dimension j, and at nodes in the other dimensions.
    The derivative is 2nd-order-accurate.

    This function is often needed when computing diffusive fluxes from nodal
    values.

    Args:
      array_node: A 3D field given at nodes.
      axis: The axis along which to compute the derivative.
      additional_states: Mapping that contains the optional scale factors.

    Returns:
      The derivative of array_node along axis, evaluated at faces `axis` and
      nodes in the other dimensions.
    """
    df_dim_face = self._backward_difference(array_node, axis)
    axis_index = self.grid_params.get_axis_index(axis)
    if self.grid_params.use_stretched_grid[axis_index]:
      h_face_key = stretched_grid_util.h_face_key(axis_index)
      h_face = additional_states[h_face_key]
      return df_dim_face / (h_face * self.grid_params.grid_spacings[axis_index])
    else:
      return df_dim_face / self.grid_params.grid_spacings[axis_index]

  def deriv_face_to_node(
      self,
      array_face: ScalarField,
      axis: str,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Given the value of a field at faces, return the derivative at nodes.

    Compute the partial derivative ∂f/∂{axis}. The resulting field is evaluated
    on nodes. The derivative is 2nd-order-accurate.

    This function is often needed when computing divergences of fluxes that are
    evaluated at faces.

    Args:
      array_face: A 3D field given at faces in `axis` and nodes in the other
        dimensions.
      axis: The axis along which to compute the derivative.
      additional_states: Mapping that contains the optional scale factors.

    Returns:
      The derivative of array_face along axis, evaluated at nodes.
    """
    df_dim = self._forward_difference(array_face, axis)
    axis_index = self.grid_params.get_axis_index(axis)
    if self.grid_params.use_stretched_grid[axis_index]:
      h_key = stretched_grid_util.h_key(axis_index)
      h = additional_states[h_key]
      return df_dim / (h * self.grid_params.grid_spacings[axis_index])
    else:
      return df_dim / self.grid_params.grid_spacings[axis_index]

  def deriv_centered(
      self,
      array_node: ScalarField,
      axis: str,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Given the value of a field f at nodes, return the derivative at nodes.

    Compute the partial derivative ∂f/∂xⱼ, where j = `dim`. The resulting field
    is evaluated on nodes. The derivative is 2nd-order-accurate.

    Args:
      array_node: A 3D field f, evaluated at nodes.
      axis: The axis along which to compute the derivative.
      additional_states: Mapping that contains the optional scale factors.

    Returns:
      The derivative of f along dim, evaluated at nodes.
    """
    df_dim = self._centered_difference(array_node, axis)
    axis_index = self.grid_params.get_axis_index(axis)
    if self.grid_params.use_stretched_grid[axis_index]:
      h_key = stretched_grid_util.h_key(axis_index)
      h = additional_states[h_key]
      return df_dim / (h * 2 * self.grid_params.grid_spacings[axis_index])
    else:
      return df_dim / (2 * self.grid_params.grid_spacings[axis_index])

  def deriv_2_node(
      self,
      array_node: ScalarField,
      axis: str,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Given array on nodes, computes second order derivative of it on nodes.

    Compute the partial derivative ∂²f/∂{axis}². The resulting field is
    evaluated on nodes. The derivative is 2nd-order-accurate.

    Args:
      array_node: A 3D field evaluated at nodes.
      axis: The axis along which to compute the derivative.
      additional_states: Mapping that contains the optional scale factors.

    Returns:
      The derivative of f along dim, evaluated at nodes.
    """
    df_dh_face = self.deriv_node_to_face(array_node, axis, additional_states)
    return self.deriv_face_to_node(df_dh_face, axis, additional_states)

  def create_copy_with_custom_kernel_op(
      self, custom_kernel_op: get_kernel_fn.ApplyKernelOp
  ) -> Self:
    """Creates a copy of derivative lib, but using a custom kernel op."""
    return Derivatives(custom_kernel_op, self.grid_params)
