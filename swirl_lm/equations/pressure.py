# Copyright 2023 The swirl_lm Authors.
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

# coding=utf-8
"""A library for solving the pressure equation.

   Currently following monitors are also supported:

     MONITOR_pressure_convergence_l-1: This records the L1 norm of the residual
       at the end of the step.
     MONITOR_pressure_convergence_l-2: This records the L1 norm of the residual
       at the end of the step.
     MONITOR_pressure_convergence_l-inf: This records the L-inf norm of the
       residual at the end of the step.

     MONITOR_pressure_convergence_solver-l-2: L2 norm from the Poisson
       solver's last iteration.
     MONITOR_pressure_convergence_solver-iterations: Number of iterations
       executed from the Poisson solver.

     MONITOR_pressure_raw_b: This records the raw RHS of the Poisson equation at
       the end of the step.
     MONITOR_pressure_raw_b-term-div: The contribution to `b` from divergence.
     MONITOR_pressure_raw_b-term-drho-dt: The contribution to `b` from rho
       change.
     MONITOR_pressure_raw_b-term-source-rho: The contribution to `b` from source
       of rho.

     MONITOR_pressure_raw_convergence: This records the
       residual vector of the Poisson equation by definition, `r = b - A * x`.
     MONITOR_pressure_raw_dp: This records the raw solution (dp) of the Poisson
       equation at the end of the step.
     MONITOR_pressure_raw_p: This records the raw p vector.

     MONITOR_pressure_raw_p-rho: The raw input rho vector to pressure module.

     MONITOR_pressure_raw_p-rho-u: The raw momentum vector input to pressure
       module.
     MONITOR_pressure_raw_p-rho-v:
     MONITOR_pressure_raw_p-rho-w:

     MONITOR_pressure_raw_p-u: The raw velocity vector input to pressure module.
     MONITOR_pressure_raw_p-v:
     MONITOR_pressure_raw_p-w:

     MONITOR_pressure_scalar_b-l-1: L1 norm of the `b` vector.
     MONITOR_pressure_scalar_b-l-2:
     MONITOR_pressure_scalar_b-l-inf:

     MONITOR_pressure_scalar_b-term-div-l-1: L1 norm of `b-term-div`.
     MONITOR_pressure_scalar_b-term-div-l-2:
     MONITOR_pressure_scalar_b-term-div-l-inf:

     MONITOR_pressure_scalar_b-term-drho-dt-l-1: L1 norm of `b-term-drho-dt`.
     MONITOR_pressure_scalar_b-term-drho-dt-l-2:
     MONITOR_pressure_scalar_b-term-drho-dt-l-inf:

     MONITOR_pressure_scalar_b-term-source-rho-l-1: L1 norm of
       `b-term-source-rho`.
     MONITOR_pressure_scalar_b-term-source-rho-l-2:
     MONITOR_pressure_scalar_b-term-source-rho-l-inf:

     MONITOR_pressure_scalar_convergence: This records the
       residual vector of the Poisson equation by definition, `r = b - A * x`.

     MONITOR_pressure_scalar_dp-l-1: L1 norm of the `dp` vector.
     MONITOR_pressure_scalar_dp-l-2:
     MONITOR_pressure_scalar_dp-l-inf:

     MONITOR_pressure_scalar_p-l-1: L1 norm of the `p` vector.
     MONITOR_pressure_scalar_p-l-2:
     MONITOR_pressure_scalar_p-l-inf:

     MONITOR_pressure_scalar_p-rho-l-1: L1 norm of input rho vector.
     MONITOR_pressure_scalar_p-rho-l-2:
     MONITOR_pressure_scalar_p-rho-l-inf:

     MONITOR_pressure_scalar_p-rho-u-l-1: L1 norm of input momentum vector.
     MONITOR_pressure_scalar_p-rho-u-l-2:
     MONITOR_pressure_scalar_p-rho-u-l-inf:

     MONITOR_pressure_scalar_p-rho-v-l-1: L1 norm of input momentum vector.
     MONITOR_pressure_scalar_p-rho-v-l-2:
     MONITOR_pressure_scalar_p-rho-v-l-inf:

     MONITOR_pressure_scalar_p-rho-w-l-1: L1 norm of input momentum vector.
     MONITOR_pressure_scalar_p-rho-w-l-2:
     MONITOR_pressure_scalar_p-rho-w-l-inf:

     MONITOR_pressure_scalar_p-u-l-1: L1 norm of input u vector.
     MONITOR_pressure_scalar_p-u-l-2:
     MONITOR_pressure_scalar_p-u-l-inf:

     MONITOR_pressure_scalar_p-v-l-1: L1 norm of input v vector.
     MONITOR_pressure_scalar_p-v-l-2:
     MONITOR_pressure_scalar_p-v-l-inf:

     MONITOR_pressure_scalar_p-w-l-1: L1 norm of input w vector.
     MONITOR_pressure_scalar_p-w-l-2:
     MONITOR_pressure_scalar_p-w-l-inf:

     MONITOR_pressure_subiter-scalar_convergence_l-1: This records the L1 norm
       of the residual at the end of each subiteration in the step.
     MONITOR_pressure_subiter-scalar_convergence_l-2: This records the L2 norm
       of the residual at the end of each subiteration in the step.
     MONITOR_pressure_subiter-scalar_convergence_l-inf: This records the L-inf
       norm of the residual at the end of each subiteration in the step.

  These can be activated by adding the corresponding keys in the
  `helper_var_keys` part of the config file.
"""

import collections
import functools
from typing import Any, Dict, Mapping, Text, Tuple

from absl import logging
import attr
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import physical_variable_keys_manager
from swirl_lm.boundary_condition import boundary_condition_utils
from swirl_lm.communication import halo_exchange
from swirl_lm.equations import pressure_pb2
from swirl_lm.equations import utils as eq_utils
from swirl_lm.linalg import poisson_solver
from swirl_lm.numerics import filters
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.physics.thermodynamics import thermodynamics_pb2
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import monitor
from swirl_lm.utility import types
import tensorflow as tf

from google.protobuf import text_format

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap

_G_THRESHOLD = 1e-6

# Poisson solver's dtype is the same as other parts by default (being `None` or
# types.TF_DTYPE), but it doesn't have to. Useful if one needs higher precision
# for the Poisson solver only.
_POISSON_SOLVER_INTERNAL_DTYPE = None
_TF_DTYPE = types.TF_DTYPE

_NormType = common_ops.NormType


def _monitor_key(
    statistic_type: Text,
    metric_name: Text,
) -> Text:
  return monitor.MONITOR_KEY_TEMPLATE.format(
      module='pressure', statistic_type=statistic_type, metric_name=metric_name)


# 1. Computed from `compute_residual` out of solver's iterations, by definition.
_MONITOR_PRESSURE_RAW_CLEAN_CONVERGENCE_RESIDUAL_VECTOR = _monitor_key(
    'raw', 'convergence')

# 2. Computed from `solve()` from the solver's last iteration, subject to
#    numerical error accumulation, especially with `float32`.
_MONITOR_PRESSURE_RAW_SOLVER_CONVERGENCE_L2_NORM = _monitor_key(
    'convergence', 'solver-l-2')
_MONITOR_PRESSURE_RAW_SOLVER_CONVERGENCE_NUM_ITERATIONS = _monitor_key(
    'convergence', 'solver-iterations')

_MONITOR_PRESSURE_RAW_CONVERGENCE_KEYS = (
    # By definition from `compute_residual`.
    _MONITOR_PRESSURE_RAW_CLEAN_CONVERGENCE_RESIDUAL_VECTOR,

    # From solver's last iteration.
    _MONITOR_PRESSURE_RAW_SOLVER_CONVERGENCE_L2_NORM,
    _MONITOR_PRESSURE_RAW_SOLVER_CONVERGENCE_NUM_ITERATIONS,
)

_DEFAULT_PRESSURE_PARAMS = (R'solver {  '
                            R'  jacobi{  '
                            R'    max_iterations: 10  '
                            R'    halo_width: 2  '
                            R'    omega: 0.67  '
                            R'  }  '
                            R'}  '
                            R'num_d_rho_filter: 3  ')

_FIELD_MAP = {
    'rho': 'p-rho',
    # Velocity.
    'u': 'p-u',
    'v': 'p-v',
    'w': 'p-w',
    # Momentum.
    'rho_u': 'p-rho-u',
    'rho_v': 'p-rho-v',
    'rho_w': 'p-rho-w',
    # Pressure.
    'p': 'p',
}

_B_TERM_DIV = 'b-term-div'
_B_TERM_DRHO_DT = 'b-term-drho-dt'
_B_TERM_SOURCE_RHO = 'b-term-source-rho'


class DensityInfo(object):
  """The base class for density information in the pressure solver."""


@attr.s
class ConstantDensityInfo(DensityInfo):
  """Density information used in the pressure solver for constant density."""
  rho = attr.ib()


@attr.s
class VariableDensityInfo(DensityInfo):
  """Density information used in the pressure solver for variable density."""
  drho_dt = attr.ib()


def _supported_convergence_norms() -> Dict[Text, _NormType]:
  """Creates a dict containing all supported convergence norms."""
  return {
      _monitor_key('convergence', 'l-1'): _NormType.L1,
      _monitor_key('convergence', 'l-2'): _NormType.L2,
      _monitor_key('convergence', 'l-inf'): _NormType.L_INF,
      _monitor_key('subiter-scalar', 'convergence_l-1'): _NormType.L1,
      _monitor_key('subiter-scalar', 'convergence_l-2'): _NormType.L2,
      _monitor_key('subiter-scalar', 'convergence_l-inf'): _NormType.L_INF,
  }


def _gen_monitor_data(
    monitor_lib: monitor.Monitor,
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    states: FlowFieldMap,
    input_monitor_params: Mapping[Text, Any],
    halo_width: int,
) -> monitor.MonitorDataType:
  """Generates monitoring data.

  Args:
    monitor_lib: The monitor object.
    replica_id: The replica id.
    replicas: The mapping of the replicas.
    states: A dictionary that holds flow field variables from the latest
      prediction. Must have 'rho_u', 'rho_v', 'rho_w', 'u', 'v', 'w', 'p', and
      'rho' in it.
    input_monitor_params: A dict contains all needed objects and values to
      generate the monitor values.
    halo_width: A int representing the halo_width for all dimensions. The values
      inside the halos will be excluded from the monitor data calculation.

  Returns:
    A dictionary representing the mapping of the monitor key to the monitor
    values.
  """
  monitor_vars = {}

  monitor_params = dict(input_monitor_params)
  for raw_key, new_key in _FIELD_MAP.items():
    if raw_key in states:
      monitor_params.update({
          new_key: states[raw_key],
      })

  for key in [
      'b',
      'dp',
      _B_TERM_DIV,
      _B_TERM_DRHO_DT,
      _B_TERM_SOURCE_RHO,
  ] + sorted(_FIELD_MAP.values()):
    if key not in monitor_params:
      continue

    cleared_v = common_ops.strip_halos(monitor_params[key], [halo_width] * 3)
    stacked_v = tf.pad(
        cleared_v if isinstance(cleared_v, tf.Tensor) else tf.stack(cleared_v),
        [[halo_width, halo_width]] * 3)

    # Vector.
    monitor_key = _monitor_key('raw', key)
    if monitor_lib.check_key(monitor_key):
      monitor_vars.update({monitor_key: tf.cast(stacked_v, _TF_DTYPE)})

    # Scalar: norms.
    norm_types = (
        _NormType.L1,
        _NormType.L2,
        _NormType.L_INF,
    )

    def _key(
        key,
        norm_type,
    ):
      return _monitor_key('scalar', '{}-{}'.format(key, norm_type))

    monitor_keys = (
        _key(key, 'l-1'),
        _key(key, 'l-2'),
        _key(key, 'l-inf'),
    )
    if not monitor_lib.check_key(monitor_keys):
      continue

    typed_norms = common_ops.compute_norm(stacked_v, norm_types, replicas)
    for monitor_key, norm_type in zip(monitor_keys, norm_types):
      if monitor_lib.check_key(monitor_key):
        monitor_vars.update(
            {monitor_key: tf.cast(typed_norms[norm_type.name], _TF_DTYPE)})

  b = monitor_params['b']
  dp = monitor_params['dp']
  solver = monitor_params['solver']
  subiter = monitor_params['subiter']
  dp_exchange_halos = monitor_params['dp_exchange_halos']
  halo_width = monitor_params['halo_width']

  supported_convergence_norms = _supported_convergence_norms()
  requested_norms = collections.defaultdict(list)
  for norm_metric, norm_type in supported_convergence_norms.items():
    if monitor_lib.check_key(norm_metric):
      requested_norms[norm_type].append(norm_metric)

  # In case raw convergence is requested but not any norm type has been
  # requested, we use NormType.L2 as a stand-in.
  if not requested_norms and monitor_lib.check_key(
      _MONITOR_PRESSURE_RAW_CONVERGENCE_KEYS):
    requested_norms[_NormType.L2].append(None)

  # If requested_norms is empty, there is no need to get the residual.
  if not requested_norms:
    return monitor_vars

  norm_types = requested_norms.keys()
  norms, r_raw = solver.compute_residual(  # pytype: disable=attribute-error
      replica_id,
      replicas,
      dp,
      b,
      norm_types,
      halo_width,
      dp_exchange_halos,
      internal_dtype=_POISSON_SOLVER_INTERNAL_DTYPE)

  for norm_type, norm in zip(norm_types, tf.unstack(norms)):
    for norm_metric in requested_norms[norm_type]:
      # Check whether this is from the 'stand-in' for just the raw residual.
      if norm_metric is None:
        continue

      if (monitor_lib.statistic_type(norm_metric) ==
          monitor.StatisticType.SUBITER_SCALAR):
        if subiter is None:
          logging.error('Missing subiter counter, monitor key: `%s` ignored.',
                        norm_metric)
        else:
          monitor_vars.update({
              norm_metric:
                  tf.tensor_scatter_nd_update(monitor_lib.data[norm_metric],
                                              [[subiter]], [norm])
          })
      else:
        monitor_vars.update({norm_metric: norm})

  for k, v in {
      # From `compute_residual` after solver iterations if any.
      _MONITOR_PRESSURE_RAW_CLEAN_CONVERGENCE_RESIDUAL_VECTOR:
          r_raw,

      # From solver's last iteration: Optional.
      _MONITOR_PRESSURE_RAW_SOLVER_CONVERGENCE_L2_NORM:
          monitor_params.get(poisson_solver.RESIDUAL_L2_NORM),
      _MONITOR_PRESSURE_RAW_SOLVER_CONVERGENCE_NUM_ITERATIONS:
          monitor_params.get(poisson_solver.ITERATIONS),
  }.items():
    if monitor_lib.check_key(k) and v is not None:
      if k == _MONITOR_PRESSURE_RAW_CLEAN_CONVERGENCE_RESIDUAL_VECTOR:
        vv = tf.stack(v)
      elif k == _MONITOR_PRESSURE_RAW_SOLVER_CONVERGENCE_NUM_ITERATIONS:
        # Num of iterations is an `int`, converting to float{32, 64} for export.
        vv = tf.cast(v, _TF_DTYPE)
      else:
        vv = v
      monitor_vars.update({k: vv})

  return monitor_vars  # pytype: disable=bad-return-type


class Pressure(object):
  """A library for solving the pressure equation."""

  def __init__(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      params: parameters_lib.SwirlLMParameters,
      thermodynamics: thermodynamics_manager.ThermodynamicsManager,
      monitor_lib: monitor.Monitor,
  ):
    """Initializes the pressure library."""
    self._kernel_op = kernel_op
    self._params = params
    self._thermodynamics = thermodynamics
    self.monitor = monitor_lib

    self._pressure_params = (
        params.pressure if params.pressure is not None else text_format.Parse(
            _DEFAULT_PRESSURE_PARAMS, pressure_pb2.Pressure()))

    self._solver = poisson_solver.poisson_solver_factory(
        params, self._kernel_op, self._pressure_params.solver)

    self._gravity_vec = params.gravity_direction or (0, 0, 0)

    # Find the direction of gravity. Only vector along a particular dimension is
    # considered currently.
    self.g_dim = None
    for i in range(3):
      if np.abs(np.abs(self._gravity_vec[i]) - 1.0) < _G_THRESHOLD:
        self.g_dim = i
        break

    self._halo_dims = (0, 1, 2)
    self._replica_dims = (0, 1, 2)

    self._bc = params.bc
    self._n_filter = self._pressure_params.num_d_rho_filter

    self._source = {'rho': None}

    self._src_manager = (physical_variable_keys_manager.SourceKeysHelper())

  def _exchange_halos(
      self,
      f,
      bc_f,
      replica_id,
      replicas,
  ):
    """Performs halo exchange for the variable f."""
    return halo_exchange.inplace_halo_exchange(
        f,
        self._halo_dims,
        replica_id,
        replicas,
        self._replica_dims,
        self._params.periodic_dims,
        bc_f,
        width=self._params.halo_width)

  def _pressure_corrector_update(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      rho_info: DensityInfo,
      subiter: tf.Tensor = None,
  ) -> Tuple[FlowFieldVal, monitor.MonitorDataType]:  # pytype: disable=annotation-type-mismatch
    """Updates the pressure correction.

    This method follows the approach introduced in:

    Charles D. Pierce and Parviz Moin, Progress-variable approach for large-eddy
    simulation of turbulent combustion. California, USA: Stanford University,
    2001.

    A reference to the thesis can be found at:
    https://drive.google.com/corp/drive/u/0/folders/18iBZ6ltE_526HncSXOhPU-1lsSG1piAG

    The second-order approximation used to compute the Laplacian of the pressure
    introduces an inconsistency between the pressure gradient approximation and
    the velocity divergence approximation. To account for this inconsistency, we
    introduce an explicit correction to the right-hand side.

    Args:
      replica_id: The ID number of the replica.
      replicas: A numpy array that maps a replica's grid coordinate to its
        replica_id, e.g. replicas[0, 0, 0] = 0, replicas[0, 0, 1] = 1.
      states: A dictionary that holds flow field variables from the latest
        prediction.
      additional_states: A dictionary that holds helper variables required by
        the Poisson solver.
      rho_info: The density information required the pressure solver. For
        constant density, `rho_info` is an instance of `ConstantDensityInfo`
        which contains the value of the density as a float. For variable
        density, `rho_info` is the rate of change of density, as a 3D tensor.
      subiter: A scalar Tensor of the integer type that represents the
        subiteration count. Default to `None`, and when it is not `None` and
        corresponding monitor variable `MONITOR_pressure_subiter_convergence` is
        specified in the config, the norm of the pressure residual will be
        stored separately for each subiteration.

    Returns:
      A Tuple with two elements: The first one is the pressure correction for
        the next time step. The second element is a dictionary for the monitor
        related metrics.

    Raises:
      ValueError if `rho_info` is not one of `ConstantDensityInfo` or
        `VariableDensityInfo`.
    """
    dx = self._params.dx
    dy = self._params.dy
    dz = self._params.dz
    dt = self._params.dt
    inv_dt = 1.0 / dt
    halo_width = self._params.halo_width

    exchange_halos = functools.partial(
        self._exchange_halos, replica_id=replica_id, replicas=replicas)

    # pylint: enable=line-too-long

    def build_rhs():
      """Builds the right hand side function of the pressure Poisson equation.

      The right hand side expression of Poisson equations is:
        lap(p) = rho / dt * (du/dx + dv/dy + dw/dz)

      Returns:
        A list of tf.Tensor with the right hand side values for the pressure
        Poisson equation of the present time step.
      """

      def div(
          coeff_rho,
          momentum_x,
          momentum_y,
          momentum_z,
      ):
        """Computes the divergence of the velocity field."""
        # Compute the fourth order derivative of the pressure for the face
        # velocity correction.
        p_corr = (
            states['p']
            if self._params.enable_rhie_chow_correction else states['dp'])
        d4p_dx4 = self._kernel_op.apply_kernel_op_x(p_corr, 'k4d2x')
        d4p_dy4 = self._kernel_op.apply_kernel_op_y(p_corr, 'k4d2y')
        d4p_dz4 = self._kernel_op.apply_kernel_op_z(p_corr, 'k4d2z',
                                                    'k4d2zsh')

        # Compute velocity gradient based on interpolated values on cell faces.
        coeff_x = dt / (4. * coeff_rho * dx**2)
        du = self._kernel_op.apply_kernel_op_x(momentum_x, 'kDx')
        du_dx = [
            du_i / (2. * dx) + coeff_x * d4p_dx4_i
            for du_i, d4p_dx4_i in zip(du, d4p_dx4)
        ]

        coeff_y = dt / (4. * coeff_rho * dy**2)
        dv = self._kernel_op.apply_kernel_op_y(momentum_y, 'kDy')
        dv_dy = [
            dv_i / (2. * dy) + coeff_y * d4p_dy4_i
            for dv_i, d4p_dy4_i in zip(dv, d4p_dy4)
        ]

        coeff_z = dt / (4. * coeff_rho * dz**2)
        dw = self._kernel_op.apply_kernel_op_z(momentum_z, 'kDz', 'kDzsh')
        dw_dz = [
            dw_i / (2. * dz) + coeff_z * d4p_dz4_i
            for dw_i, d4p_dz4_i in zip(dw, d4p_dz4)
        ]

        return [
            du_dx_i + dv_dy_i + dw_dz_i
            for du_dx_i, dv_dy_i, dw_dz_i in zip(du_dx, dv_dy, dw_dz)
        ]

      def add_factor(
          v,
          factor,
      ):
        return [factor * v_i for v_i in v]

      b_terms = {
          _B_TERM_SOURCE_RHO: add_factor(src_rho, inv_dt),
      }
      if isinstance(rho_info, ConstantDensityInfo):
        b_terms.update({
            _B_TERM_DIV:
                add_factor(
                    div(rho_info.rho, states['u'], states['v'], states['w']),
                    inv_dt * rho_info.rho),
            _B_TERM_DRHO_DT: [
                tf.zeros_like(src_rho_i) for src_rho_i in src_rho
            ],
        })

      elif isinstance(rho_info, VariableDensityInfo):
        b_terms.update({
            _B_TERM_DIV:
                add_factor(
                    div(1.0, states['rho_u'], states['rho_v'], states['rho_w']),
                    inv_dt),
            _B_TERM_DRHO_DT:
                add_factor(rho_info.drho_dt, inv_dt),
        })

      else:
        raise ValueError(
            '`rho_info` has to be either `ConstantDensityInfo` or '
            f'`VariableDensityInfo`, but {rho_info} is provided.'
        )

      # pylint: disable=g-complex-comprehension
      return [(div_i + drho_dt_i - src_rho_i)
              for div_i, drho_dt_i, src_rho_i in zip(
                  b_terms[_B_TERM_DIV],
                  b_terms[_B_TERM_DRHO_DT],
                  b_terms[_B_TERM_SOURCE_RHO],
              )], b_terms
      # pylint: enable=g-complex-comprehension

    def dp_exchange_halos(dpr,):
      """Updates halos and applies the homogeneoues boundary condition."""
      bc_dp = [
          [
              (halo_exchange.BCType.NEUMANN, 0.0),
          ] * 2,
      ] * 3
      for dim in range(3):
        if self._params.periodic_dims[dim]:
          bc_dp[dim] = [
              None,
          ] * 2
          continue

        for face in range(2):
          # If a boundary of the pressure is set to be a fixed value, no
          # modifications should be made to that boundary, hence the pressure
          # correction should be 0.
          if self._bc['p'][dim][face][0] == halo_exchange.BCType.DIRICHLET:
            bc_dp[dim][face] = (halo_exchange.BCType.DIRICHLET, 0.0)

      return exchange_halos(dpr, bc_dp)

    src_rho = self._source['rho'] if self._source['rho'] is not None else [
        tf.zeros_like(p_i, dtype=p_i.dtype) for p_i in states['p']
    ]

    # Mean removal is only applied when no Dirichlet boundary conditions are
    # specified, in which case there will be infinite number of solutions.
    has_dirichlet_bc = any(
        not self._params.periodic_dims[dim] and  # pylint: disable=g-complex-comprehension
        self._bc['p'][dim][face][0] == halo_exchange.BCType.DIRICHLET
        for dim in range(3) for face in range(2))
    mean_removal = not has_dirichlet_bc

    b, monitor_params = build_rhs()

    dp0 = [tf.zeros_like(b_i) for b_i in b]

    helper_vars = dict(additional_states)
    if (self._thermodynamics.solver_mode ==
        thermodynamics_pb2.Thermodynamics.ANELASTIC):
      helper_vars[poisson_solver.VARIABLE_COEFF] = states['rho']

    # Note that the solution that is denoted as `dp` from the Poisson solver has
    # different meanings under different modes of thermodynamics. In the low
    # Mach number model, `dp` is the pressure correction; in the anelastic mode,
    # it is the product of the specific volume and the pressure correction.
    poisson_solution = self._solver.solve(
        replica_id,
        replicas,
        b,
        dp0,
        dp_exchange_halos,
        internal_dtype=_POISSON_SOLVER_INTERNAL_DTYPE,
        additional_states=helper_vars)
    dp = poisson_solution[poisson_solver.X]

    dp = common_ops.remove_global_mean(
        common_ops.tf_cast(dp, _TF_DTYPE), replicas,
        halo_width) if mean_removal else common_ops.tf_cast(dp, _TF_DTYPE)

    if (self._thermodynamics.solver_mode ==
        thermodynamics_pb2.Thermodynamics.ANELASTIC):
      dp = tf.nest.map_structure(tf.math.multiply, dp, states['rho'])

    # Updates monitor after solving the Poisson equation.
    monitor_params.update({
        'b': b,
        'dp': dp,
        'solver': self._solver,
        'subiter': subiter,
        'dp_exchange_halos': dp_exchange_halos,
        'halo_width': halo_width,
    })
    monitor_params.update(poisson_solution)

    monitor_vars = _gen_monitor_data(self.monitor, replica_id, replicas, states,
                                     monitor_params, halo_width)

    return (dp, monitor_vars)

  def _update_pressure_bc(
      self,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ):
    """Updates the boundary condition of pressure based on the flow field."""
    bc_p = [[None, None], [None, None], [None, None]]

    velocity_keys = ['u', 'v', 'w']
    grid_spacing = (self._params.dx, self._params.dy, self._params.dz)

    def grad_per_dim(f, dim):
      """Computes the diffusion term in a specific dimension."""
      grad_ops = (
          lambda f: self._kernel_op.apply_kernel_op_x(f, 'kDx'),
          lambda f: self._kernel_op.apply_kernel_op_y(f, 'kDy'),
          lambda f: self._kernel_op.apply_kernel_op_z(f, 'kDz', 'kDzsh'),
      )
      return tf.nest.map_structure(
          lambda grad: grad / (2.0 * grid_spacing[dim]), grad_ops[dim](f))

    def ddh_per_dim(f, dim):
      """Computes the second order derivative of `f` along `dim`."""
      diff_ops = [
          lambda f: self._kernel_op.apply_kernel_op_x(f, 'kddx'),
          lambda f: self._kernel_op.apply_kernel_op_y(f, 'kddy'),
          lambda f: self._kernel_op.apply_kernel_op_z(f, 'kddz', 'kddzsh'),
      ]
      return tf.nest.map_structure(lambda diff: diff / grid_spacing[dim]**2,
                                   diff_ops[dim](f))

    # The diffusion term for the 3 velocity component can be expressed in vector
    # form as:
    # 𝛁·𝛕 = 𝜇 𝛁²u + 1/3𝜇 𝛁(𝛁·u).
    # We rearange terms in the wall-oriented coordinates (n is for the direction
    # normal to the wall, and t is for directions parallel/tangent to the wall).
    # Because the wall normal velocity component uₙ is 0 at the wall, 𝜕uₙ/𝜕t = 0
    # the equation above can be expressed as:
    # 𝛁·𝛕ₙ = 4/3 𝜇 𝜕²uₙ/𝜕n² + 1/3𝜇 𝜕/𝜕n (𝜕uₜ/𝜕t),
    # where n is for the direction normal to the wall, and t is for directions
    # parallel/tangent to the wall.
    # In additional, we assume that there's no turbulence at the wall, therefore
    # 𝜇 is the molecular viscosity.
    def diff_fn(
        mu_i: tf.Tensor,
        ddu_n_i: tf.Tensor,
        ddu_t_i: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the diffusion term at walls."""
      return mu_i * (4.0 / 3.0 * ddu_n_i + 1.0 / 3.0 * ddu_t_i)

    mu = tf.nest.map_structure(lambda rho_i: self._params.nu * rho_i,
                               states['rho'])
    ddu_n = [ddh_per_dim(states[velocity_keys[i]], i) for i in range(3)]
    du_dx = [grad_per_dim(states[velocity_keys[i]], i) for i in range(3)]
    du_t = (
        # The x component.
        tf.nest.map_structure(tf.math.add, du_dx[1], du_dx[2]),
        # The y component.
        tf.nest.map_structure(tf.math.add, du_dx[0], du_dx[2]),
        # The z component.
        tf.nest.map_structure(tf.math.add, du_dx[0], du_dx[1]),
    )
    ddu_t = [grad_per_dim(du_t[i], i) for i in range(3)]

    diff = [
        tf.nest.map_structure(diff_fn, mu, ddu_n[i], ddu_t[i]) for i in range(3)
    ]

    # Updates the pressure boundary condition based on the simulation setup.
    for i in range(3):
      for j in range(2):
        if (self._params.bc_type[i][j] ==
            boundary_condition_utils.BoundaryType.PERIODIC):
          bc_p[i][j] = None

        elif (self._params.bc_type[i][j] ==
              boundary_condition_utils.BoundaryType.INFLOW):
          bc_p[i][j] = (halo_exchange.BCType.NEUMANN_2, 0.0)

        elif (self._params.bc_type[i][j]
              == boundary_condition_utils.BoundaryType.OUTFLOW):
          if self._pressure_params.pressure_outlet:
            # Enforce a pressure outlet boundary condition on demand.
            bc_p[i][j] = (halo_exchange.BCType.DIRICHLET, 0.0)
          else:
            bc_p[i][j] = (halo_exchange.BCType.NEUMANN_2, 0.0)

        elif self._params.bc_type[i][j] in (
            boundary_condition_utils.BoundaryType.SLIP_WALL,
            boundary_condition_utils.BoundaryType.NON_SLIP_WALL,
            boundary_condition_utils.BoundaryType.SHEAR_WALL):

          bc_value = common_ops.get_face(diff[i], i, j,
                                         self._params.halo_width - 1,
                                         grid_spacing[i])[0]
          if i == self.g_dim:
            # Ensures the pressure balances with the buoyancy at the first fluid
            # layer by assigning values to the pressure in halos adjacent to the
            # fluid domain.
            rho_0 = self._thermodynamics.rho_ref(
                additional_states.get('zz', None), additional_states
            )
            b = eq_utils.buoyancy_source(self._kernel_op, states['rho_thermal'],
                                         rho_0, self._params, i)

            bc_value = tf.nest.map_structure(
                tf.math.add, bc_value,
                common_ops.get_face(b, i, j, self._params.halo_width - 1,
                                    grid_spacing[i])[0])

          # The boundary condition for pressure is applied at the interface
          # between the boundary and fluid only. Assuming everything is
          # homogeneous behind the halo layer that's closest to the fluid, a
          # homogeneous Neumann BC is applied to all other layers for pressure.
          zeros = [tf.nest.map_structure(tf.zeros_like, bc_value)] * (
              self._params.halo_width - 1)

          bc_planes = zeros + [bc_value] if j == 0 else [bc_value] + zeros

          bc_p[i][j] = (halo_exchange.BCType.NEUMANN_2, bc_planes)
        else:
          raise ValueError('{} is not defined for pressure boundary.'.format(
              self._params.bc_type[i][j]))

    self._bc['p'] = bc_p

  def update_pressure_halos(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Updates halos for p.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      states: A dictionary that holds flow field variables from the latest
        prediction. Must have 'rho_u', 'rho_v', 'rho_w', 'u', 'v', 'w', 'p', and
        'rho', 'rho_thermal' in it.
      additional_states: A dictionary that holds helper variables.

    Returns:
      A dictionary of 'p' with halos updated.

    Raises:
      ValueError If the boundary type of one of the faces is unknown.
    """
    exchange_halos = functools.partial(
        self._exchange_halos, replica_id=replica_id, replicas=replicas)

    if self._pressure_params.update_p_bc_by_flow:
      self._update_pressure_bc(states, additional_states)

    return {'p': exchange_halos(states['p'], self._bc['p'])}

  def prestep(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      additional_states: FlowFieldMap,
  ) -> None:
    """Updates additional information required for pressure step.

    This function is called before the beginning of each time step. It updates
    the mass source term on the right hand side of the Poisson equation. These
    information will be hold within this helper object.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      additional_states: A dictionary that holds constants that will be used in
        the simulation, e.g. boundary conditions, forcing terms.
    """
    del replica_id, replicas
    # Parse additional states to extract external source/forcing terms.
    self._source.update(
        self._src_manager.update_helper_variable_from_additional_states(
            additional_states))

  def step(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      states_0: FlowFieldMap,
      additional_states: FlowFieldMap,
      subiter: tf.Tensor = None,
  ) -> FlowFieldMap:  # pytype: disable=annotation-type-mismatch
    """Updates the pressure and its correction for the current subiteration.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      states: A dictionary that holds flow field variables from the latest
        prediction.
      states_0: A dictionary that holds flow field variables from the previous
        time step.
      additional_states: A dictionary that holds constants that will be used in
        the simulation, e.g. boundary conditions, forcing terms.
      subiter: A scalar Tensor of the integer type that represents the
        subiteration count. Default to `None`, and when it is not `None` the
        pressure residual will be stored separately.

    Returns:
      A dictionary with the updated pressure and pressure corrector.
    """
    drho_dt = tf.nest.map_structure(tf.zeros_like, states['rho'])
    if (self._thermodynamics.solver_mode ==
        thermodynamics_pb2.Thermodynamics.LOW_MACH):
      exchange_halos = functools.partial(
          self._exchange_halos,
          bc_f=[[(halo_exchange.BCType.NEUMANN, 0.0)] * 2] * 3,
          replica_id=replica_id,
          replicas=replicas)

      # pylint: disable=g-complex-comprehension
      drho_0 = [
          tf.compat.v1.where(
              tf.abs(drho_i) <
              self._pressure_params.d_rho_rtol * tf.abs(rho0_i),
              tf.zeros_like(drho_i), drho_i)
          for drho_i, rho0_i in zip(states['drho'], states_0['rho'])
      ]
      # pylint: enable=g-complex-comprehension

      drho_filter_cond = lambda i, drho_i: i < self._n_filter

      def drho_filter_fn(
          i,
          drho_i,
      ):
        """The body function for drho filtering."""
        return i + 1, exchange_halos(
            filters.filter_op(self._kernel_op, drho_i, order=2))

      i0 = tf.constant(0)
      _, drho = tf.while_loop(
          cond=drho_filter_cond,
          body=drho_filter_fn,
          loop_vars=(i0, drho_0),
          back_prop=False)

      drho_dt = [drho_i / self._params.dt for drho_i in drho]
    elif (self._thermodynamics.solver_mode ==
          thermodynamics_pb2.Thermodynamics.ANELASTIC):
      drho_dt = tf.nest.map_structure(tf.zeros_like, states['rho'])

    rho_info = VariableDensityInfo(drho_dt)

    dp, monitor_vars = self._pressure_corrector_update(
        replica_id, replicas, states, additional_states, rho_info, subiter
    )

    states_updated = {
        'p': tf.nest.map_structure(lambda p_, dp_: p_ + dp_, states['p'], dp),
        'dp': dp
    }
    states_updated.update(monitor_vars)
    return states_updated
