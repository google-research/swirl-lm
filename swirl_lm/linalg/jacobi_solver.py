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

"""A library to provide SwirlLM an interface to the Jacobi solvers.

This library adapts the interface of the Jacobi solvers to what is needed for
the pressure correction Poisson equation in SwirlLM.

Each variant of the Poisson equation has its own interface class.
"""

from typing import Callable, Sequence, TypeAlias, cast

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.communication import halo_exchange
from swirl_lm.linalg import base_poisson_solver
from swirl_lm.linalg import jacobi_solver_impl
from swirl_lm.linalg import poisson_solver_pb2
from swirl_lm.physics.thermodynamics import thermodynamics_pb2
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import stretched_grid_util
from swirl_lm.utility import types
import tensorflow as tf
from typing_extensions import override


FlowFieldVal: TypeAlias = types.FlowFieldVal
FlowFieldMap: TypeAlias = types.FlowFieldMap
_HaloUpdateFn: TypeAlias = Callable[[FlowFieldVal], FlowFieldVal]
PoissonSolverSolution: TypeAlias = base_poisson_solver.PoissonSolverSolution

VARIABLE_COEFF = base_poisson_solver.VARIABLE_COEFF


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

  def halo_update_fn(p: FlowFieldVal) -> FlowFieldVal:
    """Updates the halo following the divergence theorem."""
    nz = len(p)
    nx, ny = p[0].get_shape().as_list()
    lx = dx * (nx - 2 * halo_width) * computation_shape[0]
    ly = dy * (ny - 2 * halo_width) * computation_shape[1]
    lz = dz * (nz - 2 * halo_width) * computation_shape[2]

    dtype = p[0].dtype
    new_rhs_mean = tf.cast(rhs_mean, dtype)

    bc_p_x = new_rhs_mean / 6.0 * dx * lx * tf.ones((nz, 1, ny), dtype=dtype)
    bc_p_y = new_rhs_mean / 6.0 * dy * ly * tf.ones((nz, nx, 1), dtype=dtype)
    bc_p_z = new_rhs_mean / 6.0 * dz * lz * tf.ones((1, nx, ny), dtype=dtype)

    bc_p = [
        [
            (
                halo_exchange.BCType.NEUMANN,
                [
                    -bc_p_x,
                ]
                * halo_width,
            ),
            (
                halo_exchange.BCType.NEUMANN,
                [
                    bc_p_x,
                ]
                * halo_width,
            ),
        ],
        [
            (
                halo_exchange.BCType.NEUMANN,
                [
                    -bc_p_y,
                ]
                * halo_width,
            ),
            (
                halo_exchange.BCType.NEUMANN,
                [
                    bc_p_y,
                ]
                * halo_width,
            ),
        ],
        [
            (
                halo_exchange.BCType.NEUMANN,
                [
                    -bc_p_z,
                ]
                * halo_width,
            ),
            (
                halo_exchange.BCType.NEUMANN,
                [
                    bc_p_z,
                ]
                * halo_width,
            ),
        ],
    ]
    return halo_exchange.inplace_halo_exchange(
        p,
        dims=(0, 1, 2),
        replica_id=replica_id,
        replicas=replicas,
        replica_dims=(0, 1, 2),
        periodic_dims=(False, False, False),
        boundary_conditions=bc_p,
        width=halo_width,
    )

  return halo_update_fn


class PlainPoissonForPressure(base_poisson_solver.PoissonSolver):
  """Interface for SwirlLM to the Plain Poisson Jacobi Solver.

  Solve the Poisson equation that arises for the pressure correction in the
  Navier-Stokes formulation used in SwirlLM.  This class acts to adapt the
  interface of the Poisson Solver to that expected by SwirlLM.
  """

  def __init__(
      self,
      params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
  ):
    super().__init__(params, kernel_op, solver_option)
    self._halo_width = solver_option.jacobi.halo_width
    self._grid_spacings = params.grid_spacings
    self._plain_poisson_jacobi_solver = jacobi_solver_impl.PlainPoisson(
        self._grid_spacings, kernel_op, solver_option
    )

  @override
  def solve(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      rhs: FlowFieldVal,
      p0: FlowFieldVal,
      halo_update_fn: _HaloUpdateFn | None = None,
      internal_dtype: tf.dtypes.DType | None = None,
      additional_states: FlowFieldMap | None = None,
  ) -> PoissonSolverSolution:
    """Solves the Poisson equation with the Jacobi iterative method.

    Args:
      replica_id: The ID of the replica.
      replicas: A numpy array that maps a replica's grid coordinate to its
        replica_id, e.g. replicas[0, 0, 0] = 0, replicas[0, 0, 1] = 1.
      rhs: A 3D field stored in a list of `tf.Tensor` that represents the right
        hand side tensor in the Poisson equation.
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
    del internal_dtype
    rhs_mean = common_ops.global_mean(rhs, replicas, (self._halo_width,) * 3)
    halo_update_for_compatibility = halo_update_for_compatibility_fn(
        replica_id,
        replicas,
        replicas.shape,
        rhs_mean,
        self._halo_width,
        self._grid_spacings[0],
        self._grid_spacings[1],
        self._grid_spacings[2],
    )

    halo_update = (
        halo_update_fn if halo_update_fn else halo_update_for_compatibility
    )

    return self._plain_poisson_jacobi_solver.solve(rhs, p0, halo_update)


class VariableCoefficientForPressure(base_poisson_solver.PoissonSolver):
  """Interface for SwirlLM to the Variable Coefficient Jacobi Solver.

  Solve the Poisson equation that arises for the pressure correction in the
  Navier-Stokes formulation used in SwirlLM.  This class acts to adapt the
  interface of the Poisson Solver to that expected by SwirlLM.
  """

  def __init__(
      self,
      params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
  ):
    super().__init__(params, kernel_op, solver_option)
    self._halo_width = solver_option.jacobi.halo_width
    self._grid_spacings = params.grid_spacings
    self._variable_coefficient_jacobi_solver = (
        jacobi_solver_impl.VariableCoefficient(
            self._grid_spacings, kernel_op, solver_option
        )
    )

  @override
  def solve(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      rhs: FlowFieldVal,
      p0: FlowFieldVal,
      halo_update_fn: _HaloUpdateFn | None = None,
      internal_dtype: tf.dtypes.DType | None = None,
      additional_states: FlowFieldMap | None = None,
  ) -> PoissonSolverSolution:
    """Solves the Poisson equation with the Jacobi iterative method.

    Args:
      replica_id: The ID of the replica.
      replicas: A numpy array that maps a replica's grid coordinate to its
        replica_id, e.g. replicas[0, 0, 0] = 0, replicas[0, 0, 1] = 1.
      rhs: A 3D field stored in a list of `tf.Tensor` that represents the right
        hand side tensor in the Poisson equation.
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
    del internal_dtype
    assert (
        additional_states is not None
    ), '`additional_states` must be provided.'
    assert (
        VARIABLE_COEFF in additional_states
    ), f'`{VARIABLE_COEFF}` must be a key in `additional_states`.'

    rhs_mean = common_ops.global_mean(rhs, replicas, (self._halo_width,) * 3)

    halo_update_for_compatibility = halo_update_for_compatibility_fn(
        replica_id,
        replicas,
        replicas.shape,
        rhs_mean,
        self._halo_width,
        self._grid_spacings[0],
        self._grid_spacings[1],
        self._grid_spacings[2],
    )

    halo_update = (
        halo_update_fn if halo_update_fn else halo_update_for_compatibility
    )

    w = additional_states[VARIABLE_COEFF]
    return self._variable_coefficient_jacobi_solver.solve(
        w, rhs, p0, halo_update
    )


class ThreeWeightForPressure(base_poisson_solver.PoissonSolver):
  """Interface for SwirlLM to the Three-Weight Jacobi Solver.

  Solves the Poisson equation that arises for the pressure correction in the
  Navier-Stokes formulation used in SwirlLM, particularly for when stretched
  grids are used. This class acts to adapt the interface of the Poisson Solver
  to that expected by SwirlLM.
  """

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
  ):
    super().__init__(params, kernel_op, solver_option)
    # Tell the type checker the more precise type of `params` here compared to
    # the base class.
    self._params = cast(parameters_lib.SwirlLMParameters, self._params)

    self._halo_width = solver_option.jacobi.halo_width
    self._grid_spacings = params.grid_spacings
    self._three_weight_jacobi_solver = jacobi_solver_impl.ThreeWeight(
        self._grid_spacings, kernel_op, solver_option
    )

  @staticmethod
  def _generate_weights_and_modified_rhs(
      rhs: FlowFieldVal,
      additional_states: FlowFieldMap,
      use_3d_tf_tensor: bool,
      use_stretched_grid: tuple[bool, bool, bool],
      grid_dims: tuple[int, int, int],
      solver_mode: thermodynamics_pb2.Thermodynamics.SolverMode,
  ) -> tuple[FlowFieldVal, FlowFieldVal, FlowFieldVal, FlowFieldVal]:
    """Generates the 3 weighting coefficients and the modified RHS.

    Forms the weighting coefficients that arise in translating the Poisson
    equation ∇²p = rhs (Low-Mach) or ∇・ρ∇p = rhs (anelastic) into
    coordinate-dependent form, including stretched-grid scale factors.  The RHS
    is also modified when stretched grid is used.

    The stretched grid scale-factor arrays are stored in `additional_states`.
    The weight coefficients `w0`, `w1`, `w2` are materialized here into 3D
    fields because that is what the implementation of the 3-weight Jacobi solver
    takes as input.

    For low-mach:

        w0 = h1 * h2 / h0
        w1 = h0 * h2 / h1
        w2 = h0 * h1 / h2
        rhs *= h0 * h1 * h2

    For anelastic: start with the coefficients in the low-mach case, and then
    each weighting coefficient is multiplied by the reference density.

        w0 *= rho_0
        w1 *= rho_0
        w2 *= rho_0


    Args:
      rhs: The RHS of the pressure equation
      additional_states: Additional static fields needed in the computation.
      use_3d_tf_tensor: Whether to use 3D tf.Tensor or 1D tf.Tensor.
      use_stretched_grid: A tuple of booleans indicating whether to use a
        stretched grid in each dimension.
      grid_dims: A tuple of integers indicating the grid dimensions per replica.
      solver_mode: The solver mode.

    Returns:
      A tuple of 4 elements: The 3 weighting coefficients w0, w1, w2, and the
      modified rhs.
    """
    h = []
    for dim in (0, 1, 2):
      if use_stretched_grid[dim]:
        h_key = stretched_grid_util.h_key(dim)
        h.append(additional_states[h_key])
      else:
        n = grid_dims[dim]
        ones_1d = tf.ones(n)
        h.append(
            common_ops.reshape_to_broadcastable(
                ones_1d, dim, use_3d_tf_tensor
            )
        )

    w0 = common_ops.map_structure_3d(lambda h0, h1, h2: h1 * h2 / h0, *h)
    w1 = common_ops.map_structure_3d(lambda h0, h1, h2: h0 * h2 / h1, *h)
    w2 = common_ops.map_structure_3d(lambda h0, h1, h2: h0 * h1 / h2, *h)

    # Update the rhs for stretched-grid factors
    rhs = common_ops.map_structure_3d(
        lambda rhs, h0, h1, h2: rhs * h0 * h1 * h2,
        rhs,
        *h,
    )

    # For the anelastic case, all the weighting coefficients are multiplied by
    # the reference density.
    multiply = lambda a, b: tf.nest.map_structure(tf.math.multiply, a, b)
    if solver_mode == thermodynamics_pb2.Thermodynamics.ANELASTIC:
      rho_0 = additional_states[VARIABLE_COEFF]
      w0 = multiply(w0, rho_0)
      w1 = multiply(w1, rho_0)
      w2 = multiply(w2, rho_0)

    return w0, w1, w2, rhs

  @override
  def solve(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      rhs: FlowFieldVal,
      p0: FlowFieldVal,
      halo_update_fn: _HaloUpdateFn | None = None,
      internal_dtype: tf.dtypes.DType | None = None,
      additional_states: FlowFieldMap | None = None,
  ) -> PoissonSolverSolution:
    """Solves the 3-weight Poisson equation with the Jacobi iterative method.

    Args:
      replica_id: The ID of the replica.
      replicas: A numpy array that maps a replica's grid coordinate to its
        replica_id, e.g. replicas[0, 0, 0] = 0, replicas[0, 0, 1] = 1.
      rhs: A 3D field that represents the right hand side in the Poisson
        equation.
      p0: A 3D field that provides the initial guess to the Poisson equation.
      halo_update_fn: A function that updates the halo of the input.
      internal_dtype: Deprecated, do not use.
      additional_states: Additional static fields needed in the computation.

    Returns:
      A dict with the following elements:
        1. A 3D field of the same shape as `rhs` that stores the solution to
           the Poisson equation.
        2. Number of iterations for the computation
    """
    del internal_dtype
    assert (
        additional_states is not None
    ), '`additional_states` must be provided.'

    w0, w1, w2, rhs = self._generate_weights_and_modified_rhs(
        rhs,
        additional_states,
        self._params.use_3d_tf_tensor,
        self._params.use_stretched_grid,
        (self._params.nx, self._params.ny, self._params.nz),
        self._params.solver_mode,
    )

    rhs_mean = common_ops.global_mean(rhs, replicas, (self._halo_width,) * 3)

    halo_update_for_compatibility = halo_update_for_compatibility_fn(
        replica_id,
        replicas,
        replicas.shape,
        rhs_mean,
        self._halo_width,
        self._grid_spacings[0],
        self._grid_spacings[1],
        self._grid_spacings[2],
    )

    halo_update = (
        halo_update_fn if halo_update_fn else halo_update_for_compatibility
    )

    return self._three_weight_jacobi_solver.solve(
        w0, w1, w2, rhs, p0, halo_update
    )

  @override
  def residual(
      self,
      p: tf.Tensor,
      rhs: tf.Tensor,
      additional_states: FlowFieldMap | None = None,
  ) -> tf.Tensor:
    """Computes the residual (LHS - RHS) of the Poisson equation.

    Given approximate solution `p` to the Poisson equation, compute the
    residual.  The approximate solution `p` might be obtained, for example, from
    a finite number of Jacobi iterations that are not necessarily fully
    converged.

    Args:
      p: The approximate solution to the Poisson equation.
      rhs: The right hand side of the Poisson equation.
      additional_states: Additional fields needed in the computation.

    Returns:
      The residual (LHS - RHS) of the Poisson equation.
    """
    assert additional_states is not None, '`additional_states` must be given.'
    w0, w1, w2, rhs = self._generate_weights_and_modified_rhs(
        rhs,
        additional_states,
        self._params.use_3d_tf_tensor,
        self._params.use_stretched_grid,
        (self._params.nx, self._params.ny, self._params.nz),
        self._params.solver_mode,
    )
    return self._three_weight_jacobi_solver.residual(p, w0, w1, w2, rhs)


def jacobi_solver_factory(
    params: parameters_lib.SwirlLMParameters,
    kernel_op: get_kernel_fn.ApplyKernelOp,
    solver_option: poisson_solver_pb2.PoissonSolver,
) -> base_poisson_solver.PoissonSolver:
  """Creates a Jacobi solver for pressure correction equation."""
  if any(params.use_stretched_grid):
    return ThreeWeightForPressure(params, kernel_op, solver_option)
  elif params.solver_mode == thermodynamics_pb2.Thermodynamics.LOW_MACH:
    return PlainPoissonForPressure(params, kernel_op, solver_option)
  elif params.solver_mode == thermodynamics_pb2.Thermodynamics.ANELASTIC:
    return VariableCoefficientForPressure(params, kernel_op, solver_option)
  else:
    raise ValueError(f'Unknown solver mode {params.solver_mode}.')
