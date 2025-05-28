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

import functools
from typing import Any, Callable, List, Mapping, Optional, Sequence, Text

import numpy as np
import scipy as sp
import scipy.sparse  # pylint: disable=unused-import
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.communication import halo_exchange
from swirl_lm.linalg import base_poisson_solver
from swirl_lm.linalg import conjugate_gradient_solver
from swirl_lm.linalg import fast_diagonalization_solver
from swirl_lm.linalg import jacobi_solver
from swirl_lm.linalg import multigrid_3d
from swirl_lm.linalg import multigrid_utils
from swirl_lm.linalg import poisson_solver_pb2
from swirl_lm.numerics import analytics
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import grid_parametrization_pb2
from swirl_lm.utility import types
import tensorflow as tf

PoissonSolver = base_poisson_solver.PoissonSolver

_TF_DTYPE = types.TF_DTYPE

_HaloUpdateFn = Callable[[types.FlowFieldVal], types.FlowFieldVal]
_PoissonSolverSolution = base_poisson_solver.PoissonSolverSolution

X = base_poisson_solver.X
RESIDUAL_L2_NORM = base_poisson_solver.RESIDUAL_L2_NORM
ITERATIONS = base_poisson_solver.ITERATIONS
VARIABLE_COEFF = base_poisson_solver.VARIABLE_COEFF
FlowFieldVal = types.FlowFieldVal

NormType = common_ops.NormType


def _make_laplacian_matrix(
    n: int,
    grid_spacing: float,
    bc_type: Sequence[Any],
) -> np.ndarray:
  """Generates a Laplacian operator matrix in 1 dimension with a specific BC.

  Args:
    n: The size of the matrix.
    grid_spacing: The size of the mesh.
    bc_type: A sequence with length 2 specifying the types of boundary
      conditions at the lower and higher faces, respectively.

  Returns:
    The Laplacian matrix with user specified boundary condition.
  """
  a = sp.sparse.diags(
      [-2. * np.ones(n), np.ones(n - 1),
       np.ones(n - 1)], [0, -1, 1]).toarray()

  if np.any(np.array(bc_type) == grid_parametrization_pb2.BC_TYPE_PERIODIC):
    if not np.all(
        np.array(bc_type) == grid_parametrization_pb2.BC_TYPE_PERIODIC):
      raise ValueError(
          'Periodic boundary condition is ambiguous for Laplacian matrix '
          'construction: ({}, {}).'
          .format(
              grid_parametrization_pb2.BoundaryConditionType.Name(bc_type[0]),
              grid_parametrization_pb2.BoundaryConditionType.Name(bc_type[1])))
    a[0, -1] = 1.0
    a[-1, 0] = 1.0

  # Setting the weights for the boundary nodes to -1 to enforce the
  # homogeneous Neumann boundary condition between the first halo node and the
  # first fluid node. Taking the lower boundary for example. Applying the
  # Laplacian operator discretely gives: (p_{-1} - 2 p_{0} + p_{1}) / h^2,
  # where location -1 is the first halo layer, and 0 is the first fluid layer.
  # The homogeneous Neumann BC implies that p_{-1} = p_{0}. With this, the
  # Laplacian operator at the boundary becomes (-p_{0} + p_{1}) / h^2.
  # Note for now we handle 2nd order estimate Neumann in the same way.
  if bc_type[0] in (grid_parametrization_pb2.BC_TYPE_NEUMANN,
                    grid_parametrization_pb2.BC_TYPE_NEUMANN_2):
    a[0, 0] = -1.0

  # Following the same logic above, on the high-index side, the Neumann BC
  # is (-p_{n-1} + p_{n}) / h^2. Note that the index -1 here refers to the last
  # element in an array in the python syntax, which is different from the "-1"
  # in the comment above.
  if bc_type[1] in (grid_parametrization_pb2.BC_TYPE_NEUMANN,
                    grid_parametrization_pb2.BC_TYPE_NEUMANN_2):
    a[-1, -1] = -1.0

  a /= grid_spacing**2
  return a.astype(types.NP_DTYPE)


class FastDiagonalization(PoissonSolver):
  """A library of solving the Poisson equation with fast diagonalization."""

  def __init__(
      self,
      params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
  ):
    """Initializes the Fast Diagonlization solver for the Poisson equation."""
    super().__init__(params, kernel_op, solver_option)

    # The fast diagonalization solver stacks the list of 2D tensors to form a
    # 3D tensor. Therefore the dimensions of the 3D tensor is (z, x, y)
    # following the convention of the TPU simulation framework.
    self._grid_spacing = (params.dz, params.dx, params.dy)

    # The indicator of boundary conditions in all dimensions of a 3D tensor.
    bc_l = solver_option.fast_diagonalization.boundary_condition_low
    bc_h = solver_option.fast_diagonalization.boundary_condition_high
    self._bc = ((bc_l.dim_2, bc_h.dim_2), (bc_l.dim_0, bc_h.dim_0),
                (bc_l.dim_1, bc_h.dim_1))

    # Remove halos from the grid so that the Laplacian operator is only applied
    # to the interior points.
    self._halo_width = solver_option.fast_diagonalization.halo_width

    n_local = [params.nz, params.nx, params.ny]
    self._n_local = [n_local_i - 2 * self._halo_width for n_local_i in n_local]

  def solve(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      rhs: FlowFieldVal,
      p0: FlowFieldVal,
      halo_update_fn: Optional[_HaloUpdateFn] = None,
      internal_dtype: Optional[tf.dtypes.DType] = None,
      additional_states: Optional[Mapping[Text, tf.Tensor]] = None,
  ) -> _PoissonSolverSolution:
    """Solves the Poisson equation solver with the fast diagonalization method.

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
      A dict with the following elements:
        1. A 3D tensor of the same shape as `rhs` that stores the solution to
           the Poisson equation.
    """
    del p0, halo_update_fn, additional_states

    # The first dimension has to be the z direction.
    computation_shape = np.roll(np.array(replicas.shape), 1)

    n_global = [
        c_i * n_local_i
        for c_i, n_local_i in zip(computation_shape, self._n_local)
    ]

    a = []
    for dim in range(3):
      a.append(
          _make_laplacian_matrix(n_global[dim], self._grid_spacing[dim],
                                 self._bc[dim]))

    solver = fast_diagonalization_solver.fast_diagonalization_solver(
        a, self._n_local, replica_id, np.transpose(replicas, (2, 0, 1)),
        self._solver_option.fast_diagonalization.cutoff)

    rhs_interior = common_ops.strip_halos(rhs, (self._halo_width,) * 3)

    if isinstance(rhs_interior, tf.Tensor):
      p_interior = solver(rhs_interior)
      paddings = tf.constant([[self._halo_width, self._halo_width]] * 3)
      p = tf.pad(p_interior, paddings=paddings, mode='CONSTANT')
    else:
      p_interior = tf.unstack(solver(tf.stack(rhs_interior)))
      paddings = tf.constant([
          [self._halo_width, self._halo_width],
          [self._halo_width, self._halo_width],
      ])
      p = [
          tf.pad(p_i, paddings=paddings, mode='CONSTANT') for p_i in p_interior
      ]
      p = (
          [tf.zeros_like(rhs[0])] * self._halo_width
          + p
          + [tf.zeros_like(rhs[0])] * self._halo_width
      )

    # Enforce Neumann boundary condition for the Poisson system.
    bc_p_rhs = [[(halo_exchange.BCType.NEUMANN, 0.),
                 (halo_exchange.BCType.NEUMANN, 0.)]] * 3

    return {
        X:
            halo_exchange.inplace_halo_exchange(
                p,
                dims=(0, 1, 2),
                replica_id=replica_id,
                replicas=replicas,
                replica_dims=(0, 1, 2),
                periodic_dims=(False, False, False),
                boundary_conditions=bc_p_rhs,
                width=self._halo_width),
    }


class ConjugateGradient(PoissonSolver):
  """A library for solving the Poisson equation with the CG method."""

  def __init__(
      self,
      params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
  ):
    """Initializes the Conjugate Gradient solver for the Poisson equation."""
    super().__init__(params, kernel_op, solver_option)

    cg = solver_option.conjugate_gradient

    self._max_iters = cg.max_iterations
    self._tol = cg.atol
    self._l2_norm_reduction = cg.l2_norm_reduction
    # pylint:disable=g-long-ternary
    self._component_wise_convergence = (
        cg.component_wise_convergence if
        (cg.HasField('component_wise_convergence') and
         (cg.component_wise_convergence.atol > 0 or
          cg.component_wise_convergence.rtol > 0)) else None)
    # pylint:enable=g-long-ternary
    self._halo_width = cg.halo_width
    self._reprojection = cg.reprojection

  def solve(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      rhs: FlowFieldVal,
      p0: FlowFieldVal,
      halo_update_fn: Optional[_HaloUpdateFn] = None,
      internal_dtype: Optional[tf.dtypes.DType] = None,
      additional_states: Optional[Mapping[Text, tf.Tensor]] = None,
  ) -> _PoissonSolverSolution:
    """Solves the Poisson equation with the conjugate gradient method.

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
      A dict with the following elements:
        1. A 3D tensor of the same shape as `rhs` that stores the solution to
           the Poisson equation.
        2. L2 norm for the residual vector, -1 as an indicator of invalid value
        3. Number of iterations for the computation
    """
    del additional_states

    computation_shape = replicas.shape
    num_replicas = np.prod(computation_shape)
    group_assignment = np.array([range(num_replicas)], dtype=np.int32)

    def dot(vec1: FlowFieldVal, vec2: FlowFieldVal) -> tf.Tensor:
      """Performs the dot product between `vec1` and `vec2` excluding halos."""
      vec1 = halo_exchange.clear_halos(vec1, self._halo_width)
      vec2 = halo_exchange.clear_halos(vec2, self._halo_width)
      return common_ops.global_dot(vec1, vec2, group_assignment)

    rhs_mean = common_ops.global_mean(rhs, replicas, (self._halo_width,) * 3)
    halo_update_for_compatibility = (
        jacobi_solver.halo_update_for_compatibility_fn(
            replica_id, replicas, computation_shape, rhs_mean, self._halo_width,
            self._params.dx, self._params.dy, self._params.dz))
    halo_update = (
        halo_update_fn if halo_update_fn else halo_update_for_compatibility)

    laplacian = functools.partial(self._laplacian, halo_update=halo_update)

    if self._component_wise_convergence is None:
      component_wise_distance_fn = None
    else:
      component_wise_distance_fn = functools.partial(
          analytics.pair_distance_with_tol,
          rhs=common_ops.tf_cast(rhs, internal_dtype),
          atol=self._component_wise_convergence.atol,
          rtol=self._component_wise_convergence.rtol,
          replicas=replicas,
          halo_width=self._halo_width,
          symmetric=self._component_wise_convergence.symmetric)

    if self._reprojection:
      reprojection_fn = functools.partial(
          common_ops.remove_global_mean,
          replicas=replicas,
          halo_width=self._halo_width)
    else:
      reprojection_fn = None

    cg = self._solver_option.conjugate_gradient
    if (cg.HasField('preconditioner') and
        cg.preconditioner.HasField('band_preconditioner')):

      def preconditioner_fn(nested_rhs: FlowFieldVal):
        """Precondtioner for CG, an approximation for Laplacian inverse."""
        nested_rhs = halo_update(nested_rhs)

        # Prepare ConvOps.
        band_config = cg.preconditioner.band_preconditioner
        filter_coeffs = list(band_config.coefficients)
        offset = len(filter_coeffs) // 2
        kernel_dict = {'filter': (filter_coeffs, offset)}
        kernel_op = get_kernel_fn.ApplyKernelConvOp(8, kernel_dict)

        def band_matrix_terms(
            f: FlowFieldVal,
            indices: Optional[Sequence[int]] = None,
        ):
          r"""Applies cutomized kernel with the band matrix as preconditioner.

          Note that this is NOT from existing algorithms, which is designed for
          CPUs or GPUs, as our use case is on TPUs, and the candidate matrices
          (or a function of them) have the constraint that:
          0. As a preconditioner, it is required to be positive or negative
             (semi) definite.
          1. It has to be sparse, e.g. with halo width of 2, one can construct a
             matrix up to 5 bands.
          2. It's row/ step and TPU topology invariant, so that it could be
             implemented as a ConvOp in a straightforward way.

          Given the original Laplacian operator in x direction:
          A_x = (
            -1,  1,  0,  0,  0 \cdots
             1, -2,  1,  0,  0 \cdots
             0,  1, -2,  1,  0 \cdots
                               \vdots
                               \cdots 1, -2,  1,  0,
                               \cdots 0,  1, -2,  1,
                               \cdots 0,  0,  1, -1,
          )
          Or in its compact form:
              A_x = (\cdots, 0, 1, -2, 1, 0, \cdots)

          One can compute the optimal (\alpha, \beta, \gamma) by enforing zero
          elements for x * A_x, in the non-zero elements coordinates as x.
              x \approx A_x^{-1}
              x \equiv (\cdots, \gamma, \beta, \alpha, \beta, \gamma, \cdots)

          In the end, the final preconditioner can be shown to be:
              M^{-1} = x (1 - O + O^2 - O^3 + \cdots)
          where `x` is the dominating direction, i.e. dx < {dy, dz} and
              O \equiv (A_y + A_z) x

          Args:
            f: The vector to apply with M^{-1}.
            indices: Which directions (`x`, `y`, `z`) to compute.

          Returns:
            `M^{-1} f` applied in the given set of directions (indices).
          """
          if indices is None:
            indices = [0, 1, 2]
          else:
            indices = sorted(list(set(indices)))

          terms = []
          if 0 in indices:
            terms.append(self._params.dx**2 *
                         kernel_op.apply_kernel_op_x(tf.stack(f), 'filterx'))

          if 1 in indices:
            terms.append(self._params.dy**2 *
                         kernel_op.apply_kernel_op_y(tf.stack(f), 'filtery'))

          if 2 in indices:
            terms.append(self._params.dz**2 *
                         kernel_op.apply_kernel_op_z(
                             tf.stack(f), 'filterz', 'filterzsh'))

          return tuple(terms)

        values = (self._params.dx, self._params.dy, self._params.dz)
        if band_config.HasField('dominating_direction'):
          dominating_index = band_config.dominating_direction
        else:
          dominating_index = np.argmin(values)

        # Apply ConvOps.
        def get_term(f):
          """Apply preconditioner with the operator `O`."""
          # Compares with the second miminal grid spacing.
          if (band_config.HasField('dominating_grid_spacing_gap') and
              values[dominating_index] >=
              sorted(values)[1] + band_config.dominating_grid_spacing_gap):
            raise ValueError(
                'To enable preconditioning with Taylor expansion, please make '
                'sure one direction dominates over the other, i.e. there is '
                'only one minimal in (dx, dy, dz) = (%g, %g, %g) with gap %g.' %
                (self._params.dx, self._params.dy, self._params.dz,
                 band_config.dominating_grid_spacing_gap))

          # Step 01: x (If x is the dominating direction)
          rhs = band_matrix_terms(
              halo_update(f), indices=(dominating_index,))[0]

          # Step 02: (A_y + A_z) x
          indices = [i for i in range(3) if i != dominating_index]
          partial_laplacian_rhs = self._laplacian_terms(
              rhs, halo_update=None, indices=indices)

          return tf.nest.map_structure(tf.math.add, *partial_laplacian_rhs)  # pytype: disable=bad-unpacking

        # Optional taylor expansion, up to order 6 now.
        max_taylor_order = 6
        taylor_order = band_config.taylor_expansion_order
        terms_to_o_n = [tf.zeros_like(nested_rhs)] * max_taylor_order

        for count in range(max_taylor_order):
          if taylor_order < count:
            break

          terms_to_o_n[count] = get_term(nested_rhs if count ==
                                         0 else halo_update(terms_to_o_n[count -
                                                                         1]))

        # Case 1: Taylor expansion.
        if taylor_order >= 1:
          # pylint: disable=g-complex-comprehension
          # Add values with smaller abs values first.
          term = tf.nest.map_structure(
              lambda nested_rhs_i, o1_i, o2_i, o3_i, o4_i, o5_i, o6_i: (
                  ((((o6_i - o5_i) + o4_i) - o3_i) + o2_i) - o1_i
              )
              + nested_rhs_i,
              *([nested_rhs] + terms_to_o_n),
          )
          # pylint: enable=g-complex-comprehension
          return band_matrix_terms(halo_update(term), [dominating_index])[0]

        # Case 2: Falls back to the one single term version.
        if band_config.symmetric:
          # pylint: disable=unbalanced-tuple-unpacking
          rhs_x, rhs_y, rhs_z = band_matrix_terms(nested_rhs)
          # pylint: enable=unbalanced-tuple-unpacking
          return tf.nest.map_structure(
              lambda a, b, c: a + b + c, rhs_x, rhs_y, rhs_z
          )

        return band_matrix_terms(nested_rhs, [dominating_index])[0]

      preconditioner = preconditioner_fn
    else:
      preconditioner = None

    cg_solution = conjugate_gradient_solver.conjugate_gradient_solver(
        laplacian,
        dot,
        rhs,
        self._max_iters,
        self._tol,
        p0,
        l2_norm_reduction=self._l2_norm_reduction,
        component_wise_distance_fn=component_wise_distance_fn,
        reprojection=reprojection_fn,
        preconditioner=preconditioner,
        internal_dtype=internal_dtype)  # pytype: disable=wrong-arg-types

    cg_solution[X] = halo_update(tf.stack(cg_solution[X]))

    return cg_solution


class Multigrid(PoissonSolver):
  """A class for solving the Poisson equation with multigrid."""

  def __init__(
      self,
      params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
  ):
    """Initializes the multigrid solver for the Poisson equation."""
    super().__init__(params, kernel_op, solver_option)

    if params.halo_width < 1:
      raise ValueError('Multgrid solver only supports halo width >= 1.')

    self.extra_halo_width = params.halo_width - 1
    mg_params = self._solver_option.multigrid

    for bc in (mg_params.boundary_condition.dim_0,
               mg_params.boundary_condition.dim_1,
               mg_params.boundary_condition.dim_2):
      if bc == grid_parametrization_pb2.BC_TYPE_DIRICHLET:
        self.boundary_conditions.append(
            [(halo_exchange.BCType.DIRICHLET, 0.)] * 2)
      elif bc in (grid_parametrization_pb2.BC_TYPE_NEUMANN,
                  grid_parametrization_pb2.BC_TYPE_NEUMANN_2):
        self.boundary_conditions.append(
            [(halo_exchange.BCType.NEUMANN, 0.)] * 2)
      else:
        raise ValueError(f'Boundary condition type {bc} is not supported.')

    coarsest_subgrid_shape = (
        mg_params.coarsest_subgrid_shape.dim_0,
        mg_params.coarsest_subgrid_shape.dim_1,
        mg_params.coarsest_subgrid_shape.dim_2,
    )
    self._mg_cycle_step_fn = multigrid_3d.poisson_mg_cycle_step_fn(
        params,
        coarsest_subgrid_shape,
        mg_params.n_coarse,
        mg_params.n_smooth,
        mg_params.weight,
        self.boundary_conditions,
        mg_params.use_a_inv,
        mg_params.num_iterations)

  def solve(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      rhs: FlowFieldVal,
      p: FlowFieldVal,
      halo_update_fn: Optional[_HaloUpdateFn] = None,
      internal_dtype: Optional[tf.dtypes.DType] = None,
      additional_states: Optional[Mapping[Text, tf.Tensor]] = None,
  ) -> _PoissonSolverSolution:
    """Solves the Poisson equation with multigrid.

    Args:
      replica_id: The ID of the replica.
      replicas: A numpy array that maps a replica's grid coordinate to its
        replica_id, e.g. replicas[0, 0, 0] = 0, replicas[0, 0, 1] = 1.
      rhs: A 3D field stored in a list of `tf.Tensor` that represents the right
        hand side tensor `b` in the Poisson equation.
      p: The 3D candidate solution stored in a list of `tf.Tensor`.
      halo_update_fn: If provided, it is used to perform halo exchange before
        returning the results so that the final halos will be valid.
      internal_dtype: Not implemented in multigrid.
      additional_states: Additional static fields, prolongation and restriction
        matrices, needed in the computation.

    Returns:
      A dict with the following elements:
        'x': A 3D tensor of the same shape as `rhs` that stores the updated
           candidate solution to the Poisson equation.
    """
    assert additional_states is not None, (
        'Prolongation and restriction matrices are required from '
        '`additional_states` by the multigrid solver, but it is not provided.'
    )

    coordinates = tf.stack(common_ops.get_core_coordinate(replicas, replica_id))

    ps = multigrid_utils.remove_prefix_in_dict(additional_states, 'ps')
    rs = multigrid_utils.remove_prefix_in_dict(additional_states, 'rs')

    prs = multigrid_utils.convert_ps_rs_dict_to_tuple(ps, rs)
    mg_cycle = self._mg_cycle_step_fn(prs, replica_id, replicas, coordinates)

    # TODO(b/262934073): This is a quick workaround, striping down the halo to
    # be just 1 as the current multigrid solver assumes it to have one layer
    # of halos. The correct way is to allow the multigrid solver to support
    # arbitrary halo sizes.
    if self.extra_halo_width > 0:
      p = common_ops.strip_halos(p, [self.extra_halo_width,] * 3)
      rhs = common_ops.strip_halos(rhs, [self.extra_halo_width,] * 3)
    x = mg_cycle(p, rhs)

    if self.extra_halo_width > 0:
      if isinstance(x, tf.Tensor):
        paddings = [[self.extra_halo_width,] * 2,] * 3
        x = tf.pad(x, paddings, 'CONSTANT')
      else:  # x is a list of 2D tensor.
        paddings = [[self.extra_halo_width,] * 2,] * 2
        x = tf.nest.map_structure(
            lambda x_i: tf.pad(x_i, paddings, 'CONSTANT'), x)
        pad_layers = [tf.zeros_like(x[0]),] * self.extra_halo_width
        x = pad_layers + x + pad_layers
    if halo_update_fn is not None:
      x = halo_update_fn(x)
    return {
        X: x,
    }


def validate_cg_config(solver_option: poisson_solver_pb2.PoissonSolver) -> None:
  """Validate conjugate gradient configs' preconditioner."""
  cg = solver_option.conjugate_gradient
  if not cg.HasField('preconditioner'):
    return

  precond = cg.preconditioner
  # Case 1: Band preconditioner.
  if precond.HasField('band_preconditioner'):
    if (cg.halo_width != precond.band_preconditioner.halo_width or
        len(precond.band_preconditioner.coefficients) % 2 == 0 or
        cg.halo_width < len(precond.band_preconditioner.coefficients) // 2 or
        sum([abs(c) for c in precond.band_preconditioner.coefficients]) < 1e-6):
      raise ValueError(
          ('`halo_width` mismatch for CG and its band preconditioner: %d != '
           '%d, coefficients length is not odd as expected: len = %d, '
           'halo_width < len // 2, or coefficients abs sum < 1e-6.') %
          (cg.halo_width, precond.band_preconditioner.halo_width,
           len(precond.band_preconditioner.coefficients)))
    else:
      return


def poisson_solver_factory(
    params: parameters_lib.SwirlLMParameters,
    kernel_op: get_kernel_fn.ApplyKernelOp,
    solver_option: poisson_solver_pb2.PoissonSolver,
) -> PoissonSolver:
  """Constructs an object handler for the Poisson solver.

  Args:
    params: The SwirlLM parameters.
    kernel_op: An object holding a library of kernel operations.
    solver_option: The option of the selected solver to be used to solve the
      Poisson equation.

  Returns:
    A `PoissonSolver` object handler.

  Raises:
    ValueError: If the Poisson solver field is not found.
  """
  if solver_option.HasField('jacobi'):
    return jacobi_solver.jacobi_solver_factory(params, kernel_op, solver_option)
  elif solver_option.HasField('fast_diagonalization'):
    return FastDiagonalization(params, kernel_op, solver_option)
  elif solver_option.HasField('conjugate_gradient'):
    validate_cg_config(solver_option)
    return ConjugateGradient(params, kernel_op, solver_option)
  elif solver_option.HasField('multigrid'):
    return Multigrid(params, kernel_op, solver_option)

  raise ValueError('Unknown Poisson solver option {}.'.format(solver_option))


def poisson_solver_helper_variable_keys(
    params: parameters_lib.SwirlLMParameters,
) -> List[str]:
  """Determines the required helper variable keys for the Poisson solver.

  Args:
    params: An instance of the `SwirlLMParameters` specifying a simulation
      configuration.

  Returns:
    A sequence of strings representing helper variables that are required by
    the selected type of Poisson solver.
  """
  if params.pressure is None:
    return []

  solver_type = params.pressure.solver.WhichOneof('solver')
  if solver_type == 'multigrid':
    return multigrid_utils.get_multigrid_helper_var_keys(params)
  else:
    return []


def poisson_solver_helper_variable_init_fn(
    params: parameters_lib.SwirlLMParameters,
) -> Optional[types.InitFn]:
  """Generates a `InitFn` for helper variables required by the Poisson solver.

  Args:
    params: An instance of the `SwirlLMParameters` specifying a simulation
      configuration.

  Returns:
    A function that initializes the helper variables that are required by the
    Poisson solver.
  """
  if params.pressure is None:
    return None

  solver_type = params.pressure.solver.WhichOneof('solver')
  if solver_type == 'multigrid':
    return multigrid_utils.get_multigrid_helper_var_init_fn(params)
  else:
    return None
