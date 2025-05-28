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
# coding=utf-8
"""A library of solvers for the Poisson equation.

Methods presented here are used to solve the Poisson equation, i.e.
  ∇²p = b,
in a distributed setting.
"""

import abc
from typing import Callable

import jax
from jax import sharding
import six
from swirl_lm.jax.communication import halo_exchange
from swirl_lm.jax.linalg import poisson_solver_pb2
from swirl_lm.jax.utility import common_ops
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types

PoissonSolverSolution = dict[str, float | jax.Array]
ScalarField = types.ScalarField

X = 'x'
RESIDUAL_L2_NORM = 'residual_l2_norm'
COMPONENT_WISE_DISTANCE = 'component_wise_distance_from_rhs'
ITERATIONS = 'iterations'

# The variable name to be used for the coefficient of the Poisson problem. If a
# variable with this name is found in `additional_states`, the variable
# coefficient Poisson equation 𝛁·(h 𝛁(w)) = f will be solved instead of
# 𝛁²w = f, with h being the variable coefficient.
VARIABLE_COEFF = 'poisson_variable_coeff'

NormType = common_ops.NormType

_HaloUpdateFn = Callable[[ScalarField], ScalarField]


def _halo_update_homogeneous_neumann(
    mesh: sharding.Mesh,
    grid_params: grid_parametrization.GridParametrization,
) -> _HaloUpdateFn:
  """Updates the halo following the homogeneous Neumann boundary condition.

  Args:
    mesh: A jax Mesh object representing the device topology.
    grid_params: The grid parametrization object.

  Returns:
    A function that updates the halos of the input 3D tensor.
  """
  bc_p = [
      ((halo_exchange.BCType.NEUMANN, 0.0), (halo_exchange.BCType.NEUMANN, 0.0))
  ] * 3

  def halo_update_fn(p: ScalarField) -> ScalarField:
    """Updates the halos with homogeneous Neumann boundary condition."""
    return halo_exchange.inplace_halo_exchange(
        p,
        axes=('x', 'y', 'z'),
        mesh=mesh,
        grid_params=grid_params,
        periodic_dims=[False, False, False],
        boundary_conditions=tuple(bc_p),
        halo_width=grid_params.halo_width,
    )

  return halo_update_fn


@six.add_metaclass(abc.ABCMeta)
class PoissonSolver(object):
  """A library of utilities for solving the Poisson equation."""

  def __init__(
      self,
      grid_params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
  ):
    """Switches the Poisson solver following the option provided by caller.

    Args:
      grid_params: The grid parametrization.
      kernel_op: An object holding a library of kernel operations.
      solver_option: The option of the selected solver to be used to solve the
        Poisson equation.

    Raises:
      ValueError: If the Poisson solver field is not found.
    """
    self._grid_params = grid_params
    self._kernel_op = kernel_op
    self._solver_option = solver_option
    self.boundary_conditions = []

  def _laplacian_terms(
      self,
      f: ScalarField,
      halo_update: _HaloUpdateFn | None = None,
      axes: list[str] | None = None,
  ) -> dict[str, ScalarField]:
    """Computes the Laplacian terms of `f`, in the `axes` directions.

    Args:
      f: The input field.
      halo_update: A function that updates the halo of the input.
      axes: The axes to compute the Laplacian terms. The list should contain
        ['x', 'y', 'z'] or a subset of them.

    Returns:
      A dict of arrays that represent the Laplacian terms of `f` in `axes`.
    """
    f = halo_update(f) if halo_update else f

    if axes is None:
      axes = ['x', 'y', 'z']
    else:
      axes = list(set(axes))

    terms = {}
    if 'x' in axes:
      terms['x'] = (
          self._kernel_op.apply_kernel_op(f, 'kdd', 'x')
          * self._grid_params.dx**-2
      )
    if 'y' in axes:
      terms['y'] = (
          self._kernel_op.apply_kernel_op(f, 'kdd', 'y')
          * self._grid_params.dy**-2
      )
    if 'z' in axes:
      terms['z'] = (
          self._kernel_op.apply_kernel_op(f, 'kdd', 'z')
          * self._grid_params.dz**-2
      )
    return terms

  def _laplacian(
      self,
      f: ScalarField,
      halo_update: _HaloUpdateFn | None = None,
  ) -> ScalarField:
    """Computes the Laplacian of `f`."""
    laplacian_terms = self._laplacian_terms(f, halo_update=halo_update)
    return laplacian_terms['x'] + laplacian_terms['y'] + laplacian_terms['z']

  def compute_residual(
      self,
      mesh: sharding.Mesh,
      f: ScalarField,
      rhs: ScalarField,
      norm_types: tuple[NormType, ...] = (NormType.L_INF,),
      halo_update_fn: _HaloUpdateFn | None = None,
      remove_mean_from_rhs: bool = False,
  ) -> tuple[dict[str, jax.Array], ScalarField]:
    """Computes the residual of the Poisson equation.

    The residual is defined as
      r = Ax - b,
    where x is the solution vector, b is the right hand side vector, and A is
    the Laplacian operator.

    Args:
      mesh: A jax Mesh object representing the device topology.
      f: The solution to the Poisson equation given `rhs`.
      rhs: A 3D array that represents the right hand side tensor `b` in the
        Poisson equation.
      norm_types: The types of norm to be used to compute the residual. Default
        option is the infinity norm only.
      halo_update_fn: A function that updates the halo of the input.
      remove_mean_from_rhs: Whether to remove mean from rhs, as the operator
        could be singular, and potentially unable to resolve a constant shift.
        Could be useful for homogeneous Neumann boundary conditions, when
        computing residuals. Also in conjugate gradient solver, if reprojection
        is turned on, it removes mean from both the guess vector and residual
        vector, effectively and implicitly removing mean from rhs as well.

    Returns:
      A tuple consisting of norms of the residual and the pointwise residual.
      The first represents the residual with norm types specified from the
      input. The second is the raw point-wise residual at every point in the 3D
      volume, with the shape of the local grid including halos; note the values
      in the halos should be completely ignored as we do not guarantee any
      specific values to occupy these halos.
    """

    halo_update = (
        halo_update_fn
        if halo_update_fn
        else _halo_update_homogeneous_neumann(mesh, self._grid_params)
    )
    f = halo_update(f)

    if remove_mean_from_rhs:
      hw = self._grid_params.halo_width
      rhs = rhs - common_ops.global_mean(
          rhs, mesh, hw, hw, hw, self._grid_params
      )

    laplacian_f = self._laplacian(f)
    res = halo_exchange.set_halos_to_zero(
        laplacian_f - rhs, self._grid_params.halo_width, self._grid_params
    )

    return common_ops.compute_norm(res, set(norm_types), mesh), res

  def solve(
      self,
      rhs: ScalarField,
      p0: ScalarField,
      mesh: sharding.Mesh,
      halo_update_fn: _HaloUpdateFn | None = None,
      additional_states: dict[str, jax.Array] | None = None,
  ) -> PoissonSolverSolution:
    """Solves the Poisson equation following the option provided by caller.

    Args:
      rhs: A 3D array that represents the right hand side tensor `b` in the
        Poisson equation.
      p0: A 3D array that provides initial guess to the Poisson equation.
      mesh: A jax Mesh object representing the device topology.
      halo_update_fn: A function that updates the halo of the input.
      additional_states: Additional static fields needed in the computation.

    Returns:
      A dict with potentially the following elements:
        1. A 3D tensor of the same shape as `rhs` that stores the solution to
           the Poisson equation.
        2. L2 norm for the residual vector
        3. Number of iterations for the computation
      where the last 2 elements might be -1 when it doesn't apply.
    """
    raise NotImplementedError('`PoissonSolver` is an abstract class.')

  def residual(
      self,
      p: ScalarField,
      rhs: ScalarField,
      mesh: sharding.Mesh,
      additional_states: dict[str, jax.Array] | None = None,
  ) -> jax.Array:
    """Computes the residual (LHS - RHS) of the Poisson equation.

    Args:
      p: The approximate solution to the Poisson equation.
      rhs: The right hand side of the Poisson equation.
      mesh: A jax Mesh object representing the device topology.
      additional_states: Additional fields needed in the computation.

    Returns:
      The residual (LHS - RHS) of the Poisson equation.
    """
    raise NotImplementedError('`PoissonSolver` is an abstract class.')
