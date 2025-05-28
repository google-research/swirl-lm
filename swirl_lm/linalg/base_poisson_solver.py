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
from typing import Callable, Dict, Mapping, Optional, Sequence, Text, Tuple, Union

import numpy as np
import six
from swirl_lm.communication import halo_exchange
from swirl_lm.linalg import poisson_solver_pb2
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf

PoissonSolverSolution = Dict[Text, Union[float, tf.Tensor, Sequence[tf.Tensor]]]
FlowFieldVal = types.FlowFieldVal

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

_TF_DTYPE = types.TF_DTYPE
_HaloUpdateFn = Callable[[FlowFieldVal], FlowFieldVal]


def _halo_update_homogeneous_neumann(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    halo_width: int,
) -> _HaloUpdateFn:
  """Updates the halo following the homogeneous Neumann boundary condition.

  Args:
    replica_id: The index of the current TPU core.
    replicas: A numpy array that maps a replica's grid coordinate to its
      replica_id, e.g. replicas[0, 0, 0] = 0, replicas[0, 0, 1] = 1.
    halo_width: The width of the halo layer.

  Returns:
    A function that updates the halos of the input 3D tensor.
  """
  bc_p = [[(halo_exchange.BCType.NEUMANN, 0.),
           (halo_exchange.BCType.NEUMANN, 0.)]] * 3

  def halo_update_fn(p: FlowFieldVal) -> FlowFieldVal:
    """Updates the halos with homogeneous Neumann boundary condition."""
    return halo_exchange.inplace_halo_exchange(
        p,
        dims=(0, 1, 2),
        replica_id=replica_id,
        replicas=replicas,
        replica_dims=(0, 1, 2),
        periodic_dims=(False, False, False),
        boundary_conditions=bc_p,
        width=halo_width)

  return halo_update_fn


@six.add_metaclass(abc.ABCMeta)
class PoissonSolver(object):
  """A library of utilities for solving the Poisson equation."""

  def __init__(
      self,
      params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
  ):
    """Switches the Poisson solver following the option provided by caller.

    Args:
      params: The grid parametrization.
      kernel_op: An object holding a library of kernel operations.
      solver_option: The option of the selected solver to be used to solve the
        Poisson equation.

    Raises:
      ValueError: If the Poisson solver field is not found.
    """
    self._params = params
    self._kernel_op = kernel_op
    self._solver_option = solver_option
    self.boundary_conditions = []

  def _laplacian_terms(
      self,
      f: FlowFieldVal,
      halo_update: Optional[_HaloUpdateFn] = None,
      indices: Optional[Sequence[int]] = None,
  ) -> Tuple[FlowFieldVal]:
    """Computes the Laplacian terms of `f`, in 3 directions respectively."""
    f = halo_update(f) if halo_update else f

    if indices is None:
      indices = [0, 1, 2]
    else:
      indices = sorted(list(set(indices)))

    terms = []
    if 0 in indices:
      terms.append(
          tf.nest.map_structure(
              lambda x_i: x_i * self._params.dx**-2,
              self._kernel_op.apply_kernel_op_x(f, 'kddx'))
      )
    if 1 in indices:
      terms.append(
          tf.nest.map_structure(
              lambda y_i: y_i * self._params.dy**-2,
              self._kernel_op.apply_kernel_op_y(f, 'kddy'))
      )
    if 2 in indices:
      terms.append(
          tf.nest.map_structure(
              lambda z_i: z_i * self._params.dz**-2,
              self._kernel_op.apply_kernel_op_z(f, 'kddz', 'kddzsh'))
      )
    return tuple(terms)

  def _laplacian(
      self,
      f: FlowFieldVal,
      halo_update: Optional[_HaloUpdateFn] = None,
  ) -> FlowFieldVal:
    """Computes the Laplacian of `f`."""
    laplacian_terms = self._laplacian_terms(f, halo_update=halo_update)
    return tf.nest.map_structure(
        lambda ddx_, ddy_, ddz_: ddx_ + ddy_ + ddz_,
        laplacian_terms[0], laplacian_terms[1], laplacian_terms[2])

  def compute_residual(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      f: FlowFieldVal,
      rhs: FlowFieldVal,
      norm_types: Sequence[NormType] = (NormType.L_INF,),
      halo_width: int = 2,
      halo_update_fn: Optional[_HaloUpdateFn] = None,
      remove_mean_from_rhs: bool = False,
      internal_dtype=None,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Computes the residual of the Poisson equation.

    The residual is defined as
      r = Ax - b,
    where x is the solution vector, b is the right hand side vector, and A is
    the Laplacian operator.

    Args:
      replica_id: The ID of the replica.
      replicas: A numpy array that maps a replica's grid coordinate to its
        replica_id, e.g. replicas[0, 0, 0] = 0, replicas[0, 0, 1] = 1.
      f: The solution to the Poisson equation given `rhs`.
      rhs: A 3D field stored in a list of `tf.Tensor` that represents the right
        hand side tensor `b` in the Poisson equation.
      norm_types: The types of norm to be used to compute the residual. Default
        option is the infinity norm only.
      halo_width: The width of the halo layer. Default setting to 2 for second
        order scheme in the Navier-Stokes solver.
      halo_update_fn: A function that updates the halo of the input.
      remove_mean_from_rhs: Whether to remove mean from rhs, as the operator
        could be singular, and potentially unable to resolve a constant shift.
        Could be useful for homogeneous Neumann boundary conditions, when
        computing residuals. Also in conjugate gradient solver, if reprojection
        is turned on, it removes mean from both the guess vector and residual
        vector, effectively and implicitly removing mean from rhs as well.
      internal_dtype: Which tf.dtype to use {tf.float32, tf.float64}.
        `tf.float32` is the default, mainly to be backward compatible, but
        `tf.float64` is strongly recommended to get accurate evaluation of `L_p`
        norms.

    Returns:
      A tuple of (`tf.Tensor, `tf.Tensor`). The first represents the
      residual with norm types specified from the input. The second is the raw
      point-wise residual at every point in the 3D volume, with the shape of
      [nz, nx, ny] where `nx`, `ny` and `nz` are the sizes of the local grid
      including halos; note the values in the halos should be completely ignored
      as we do not guarantee any specific values to occupy these halos.
    """
    # Using `rhs` instead of `f` here, as `f` might be a different type,
    # depending on the precision for the Poisson solver.
    input_dtype = rhs[0].dtype

    f = common_ops.tf_cast(f, internal_dtype)
    rhs = common_ops.tf_cast(rhs, internal_dtype)

    halo_update = (
        halo_update_fn if halo_update_fn else _halo_update_homogeneous_neumann(
            replica_id, replicas, halo_width))
    f = halo_update(f)

    if remove_mean_from_rhs:
      rhs = common_ops.remove_global_mean(rhs, replicas, halo_width)

    laplacian_f = self._laplacian(f)
    res = tf.stack(
        halo_exchange.clear_halos(
            tf.nest.map_structure(
                lambda lap_f, rhs_i: lap_f - rhs_i, laplacian_f, rhs),
            halo_width))

    typed_norms = common_ops.compute_norm(res, norm_types, replicas)
    norms = [typed_norms[norm_type.name] for norm_type in norm_types]

    return tf.cast(tf.stack(norms), input_dtype), tf.cast(res, input_dtype)

  def solve(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      rhs: FlowFieldVal,
      p0: FlowFieldVal,
      halo_update_fn: Optional[_HaloUpdateFn] = None,
      internal_dtype: Optional[tf.dtypes.DType] = None,
      additional_states: Optional[Mapping[Text, tf.Tensor]] = None,
  ) -> PoissonSolverSolution:
    """Solves the Poisson equation following the option provided by caller.

    Args:
      replica_id: The ID of the replica.
      replicas: A numpy array that maps a replica's grid coordinate to its
        replica_id, e.g. replicas[0, 0, 0] = 0, replicas[0, 0, 1] = 1.
      rhs: A 3D field stored in a list of `tf.Tensor` that represents the right
        hand side tensor `b` in the Poisson equation.
      p0: A 3D field stored in a list of `tf.Tensor` that provides initial guess
        to the Poisson equation.
      halo_update_fn: A function that updates the halo of the input.
      internal_dtype: Which tf.dtype to use {tf.float32, tf.float64}.
        `tf.float32` is the default, mainly to be backward compatible, but
        `tf.float64` is recommended to avoid numerical error accumulation and
        get accurate evaluation of `L_p` norms.
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
      p: tf.Tensor,
      rhs: tf.Tensor,
      additional_states: dict[str, tf.Tensor] | None = None,
  ) -> tf.Tensor:
    """Computes the residual (LHS - RHS) of the Poisson equation.

    Args:
      p: The approximate solution to the Poisson equation.
      rhs: The right hand side of the Poisson equation.
      additional_states: Additional fields needed in the computation.

    Returns:
      The residual (LHS - RHS) of the Poisson equation.
    """
    raise NotImplementedError('`PoissonSolver` is an abstract class.')
