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
# Lint as: python3
"""A library of solvers for the Poisson equation.

Methods presented here are used to solve the Poisson equation, i.e.
  ∇²p = b,
in a distributed setting.
"""

from typing import Callable, List, Mapping, Optional, Sequence, Text

import numpy as np
from swirl_lm.communication import halo_exchange
from swirl_lm.linalg import base_poisson_solver
from swirl_lm.linalg import poisson_solver_pb2
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf

_TF_DTYPE = types.TF_DTYPE

_HaloUpdateFn = Callable[[Sequence[tf.Tensor]], List[tf.Tensor]]
_PoissonSolverSolution = base_poisson_solver.PoissonSolverSolution

X = base_poisson_solver.X
RESIDUAL_L2_NORM = base_poisson_solver.RESIDUAL_L2_NORM
ITERATIONS = base_poisson_solver.ITERATIONS


NormType = common_ops.NormType


def halo_update_for_compatibility_fn(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    computation_shape: Sequence[int],
    rhs_mean: tf.Tensor,
    halo_width: int,
    dx: float,
    dy: float,
    dz: float,
) -> _HaloUpdateFn:
  """Updates the halo following the divergence theorem.

  Args:
    replica_id: The index of the current TPU core.
    replicas: A numpy array that maps a replica's grid coordinate to its
      replica_id, e.g. replicas[0, 0, 0] = 0, replicas[0, 0, 1] = 1.
    computation_shape: The dimension of the TPU topology.
    rhs_mean: The mean of the right hand side of the Poisson equation.
    halo_width: The width of the halo layer.
    dx: The grid spacing dimension 0.
    dy: The grid spacing dimension 1.
    dz: The grid spacing dimension 2.

  Returns:
    A function that updates the halos of the input 3D tensor.
  """

  def halo_update_fn(p: Sequence[tf.Tensor]) -> List[tf.Tensor]:
    """Updates the halo following the divergence theorem."""
    nz = len(p)
    nx, ny = p[0].get_shape().as_list()
    lx = dx * (nx - 2 * halo_width) * computation_shape[0]
    ly = dy * (ny - 2 * halo_width) * computation_shape[1]
    lz = dz * (nz - 2 * halo_width) * computation_shape[2]

    dtype = p[0].dtype
    new_rhs_mean = tf.cast(rhs_mean, dtype)

    bc_p_x_low = [-new_rhs_mean / 6. * dx * lx * tf.ones(
        (1, ny), dtype=dtype)] * nz
    bc_p_x_high = [new_rhs_mean / 6. * dx * lx * tf.ones(
        (1, ny), dtype=dtype)] * nz
    bc_p_y_low = [-new_rhs_mean / 6. * dy * ly * tf.ones(
        (nx, 1), dtype=dtype)] * nz
    bc_p_y_high = [new_rhs_mean / 6. * dy * ly * tf.ones(
        (nx, 1), dtype=dtype)] * nz
    bc_p_z = new_rhs_mean / 6. * dz * lz * tf.ones((nx, ny), dtype=dtype)

    bc_p = [
        [
            (halo_exchange.BCType.NEUMANN, [
                bc_p_x_low,
            ] * halo_width),
            (halo_exchange.BCType.NEUMANN, [
                bc_p_x_high,
            ] * halo_width),
        ],
        [
            (halo_exchange.BCType.NEUMANN, [
                bc_p_y_low,
            ] * halo_width),
            (halo_exchange.BCType.NEUMANN, [
                bc_p_y_high,
            ] * halo_width),
        ],
        [
            (halo_exchange.BCType.NEUMANN, [
                -bc_p_z,
            ] * halo_width),
            (halo_exchange.BCType.NEUMANN, [
                bc_p_z,
            ] * halo_width),
        ],
    ]
    return halo_exchange.inplace_halo_exchange(
        [p_i for p_i in p],
        dims=(0, 1, 2),
        replica_id=replica_id,
        replicas=replicas,
        replica_dims=(0, 1, 2),
        periodic_dims=(False, False, False),
        boundary_conditions=bc_p,
        width=halo_width)

  return halo_update_fn


class JacobiSolver(base_poisson_solver.PoissonSolver):
  """A library of solving the Poisson equation with weighted Jacobi method."""

  def __init__(
      self,
      params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
  ):
    """Initializes the Jacobi solver for the Poisson equation."""
    super().__init__(params, kernel_op, solver_option)

    self._omega = solver_option.jacobi.omega
    self._num_iters = solver_option.jacobi.max_iterations
    self._halo_width = solver_option.jacobi.halo_width

    self._factor_b = 0.5 / (params.dx**-2 + params.dy**-2 + params.dz**-2)
    # TODO(b/190727604): Note that `_factor_x` defined in this way works, while
    # saving the inverse will break the golden test.
    self._factor_x = params.dx**2 / self._factor_b
    self._factor_y = params.dy**2 / self._factor_b
    self._factor_z = params.dz**2 / self._factor_b

  def solve(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      rhs: Sequence[tf.Tensor],
      p0: Sequence[tf.Tensor],
      halo_update_fn: Optional[_HaloUpdateFn] = None,
      internal_dtype: Optional[tf.dtypes.DType] = None,
      additional_states: Optional[Mapping[Text, tf.Tensor]] = None,
  ) -> _PoissonSolverSolution:
    """Solves the Poisson equation with the Jacobi iterative method.

    Args:
      replica_id: The ID of the replica.
      replicas: A numpy array that maps a replica's grid coordinate to its
        replica_id, e.g. replicas[0, 0, 0] = 0, replicas[0, 0, 1] = 2.
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
      A dict with the following elements:
        1. A 3D tensor of the same shape as `rhs` that stores the solution to
           the Poisson equation.
        2. Number of iterations for the computation
    """
    del additional_states, internal_dtype

    def p_next(p):
      """Computes the pressure correction for the next sub-iteration."""
      # Compute the right hand side function for interior points.
      p = halo_update(p)

      p_terms = zip(
          self._kernel_op.apply_kernel_op_x(p, 'kSx'),
          self._kernel_op.apply_kernel_op_y(p, 'kSy'),
          self._kernel_op.apply_kernel_op_z(p, 'kSz', 'kSzsh'), rhs)

      p_interior = [(fx / self._factor_x + fy / self._factor_y +
                     fz / self._factor_z - self._factor_b * fb)
                    for fx, fy, fz, fb in p_terms]

      return [
          self._omega * p_interior_i + (1.0 - self._omega) * p_i
          for p_interior_i, p_i in zip(p_interior, p)
      ]

    rhs_mean = common_ops.global_mean(rhs, replicas, (self._halo_width,) * 3)

    halo_update_for_compatibility = halo_update_for_compatibility_fn(
        replica_id, replicas, replicas.shape, rhs_mean, self._halo_width,
        self._params.dx, self._params.dy, self._params.dz)

    halo_update = (
        halo_update_fn if halo_update_fn else halo_update_for_compatibility)

    i0 = tf.constant(0)
    stop_condition = lambda i, p: i < self._num_iters
    body = lambda i, p: (i + 1, p_next(p))

    iterations, p = tf.while_loop(
        cond=stop_condition, body=body, loop_vars=(i0, p0), back_prop=False)

    return {
        X: halo_update(p),
        ITERATIONS: iterations,
    }
