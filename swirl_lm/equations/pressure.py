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

# coding=utf-8
"""A library for solving the pressure equation.

   Currently following monitors are also supported:

     MONITOR_pressure_convergence_l-1: This records the L1 norm of the residual
       at the end of the step.
     MONITOR_pressure_convergence_l-2: This records the L2 norm of the residual
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
from typing import Any, Dict, Literal, Mapping, Optional, Text, Tuple, TypeAlias

from absl import logging
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import physical_variable_keys_manager
from swirl_lm.boundary_condition import boundary_condition_utils
from swirl_lm.communication import halo_exchange
from swirl_lm.equations import pressure_pb2
from swirl_lm.equations import utils as eq_utils
from swirl_lm.linalg import poisson_solver
from swirl_lm.numerics import filters
from swirl_lm.numerics import interpolation
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.physics.thermodynamics import thermodynamics_pb2
from swirl_lm.utility import common_ops
from swirl_lm.utility import debug_output
from swirl_lm.utility import debug_print
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import monitor
from swirl_lm.utility import types
import tensorflow as tf

from google.protobuf import text_format


FlowFieldVal: TypeAlias = types.FlowFieldVal
FlowFieldMap: TypeAlias = types.FlowFieldMap
BoundaryType: TypeAlias = boundary_condition_utils.BoundaryType
VerticalBCTreatment: TypeAlias = pressure_pb2.Pressure.VerticalBCTreatment
_NormType: TypeAlias = common_ops.NormType

_G_THRESHOLD = 1e-6

# Poisson solver's dtype is the same as other parts by default (being `None` or
# types.TF_DTYPE), but it doesn't have to. Useful if one needs higher precision
# for the Poisson solver only.
_POISSON_SOLVER_INTERNAL_DTYPE = None
_TF_DTYPE = types.TF_DTYPE

_DEBUG_PRINT_LOG_LEVEL = debug_print.LogLevel.INFO


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


def _get_first_last_grid_spacing_for_wall_bc(
    params: parameters_lib.SwirlLMParameters, dim: Literal[0, 1, 2]
) -> tuple[float, float]:
  """Returns the first and last grid spacing in `dim`, allowing nonuniform grid.

  Used for a wall-like, Neumann boundary condition, not a periodic BC.

  Args:
    params: An instance of SwirlLMParameters specifying the configuration.
    dim: The dimension under consideration.

  Returns:
    The first and last grid spacing.
  """
  if params.use_stretched_grid[dim]:
    halo_width = params.halo_width
    coord = params.global_xyz_with_halos[dim]
    first_grid_spacing = coord[halo_width] - coord[halo_width - 1]
    last_grid_spacing = coord[-halo_width] - coord[-(halo_width + 1)]
  else:
    first_grid_spacing = last_grid_spacing = params.grid_spacings[dim]
  return first_grid_spacing, last_grid_spacing


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
    self._deriv_lib = params.deriv_lib
    self._thermodynamics = thermodynamics
    self.monitor = monitor_lib

    if (pressure_params := params.pressure) is not None:
      self._pressure_params = pressure_params
    else:
      self._pressure_params = text_format.Parse(
          _DEFAULT_PRESSURE_PARAMS, pressure_pb2.Pressure()
      )

    self._solver = poisson_solver.poisson_solver_factory(
        params, self._kernel_op, self._pressure_params.solver)

    self._gravity_vec = params.gravity_direction or (0, 0, 0)
    # Find the direction of gravity. Only vector along a particular dimension is
    # considered currently.
    self.g_dim = params.g_dim

    self._halo_dims = (0, 1, 2)
    self._replica_dims = (0, 1, 2)

    self._bc = params.bc
    self._n_filter = self._pressure_params.num_d_rho_filter

    self._source = {'rho': None}

    self._src_manager = physical_variable_keys_manager.SourceKeysHelper()

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

  def _compute_divergence_term(
      self,
      rho_u: FlowFieldVal,
      rho_v: FlowFieldVal,
      rho_w: FlowFieldVal,
      dt: float,
      additional_states: FlowFieldMap,
  ) -> FlowFieldVal:
    """Computes ∇·(ρu)/dt, needed for the RHS of the Poisson equation."""
    d_rho_u_dx = self._deriv_lib.deriv_centered(rho_u, 0, additional_states)
    d_rho_v_dy = self._deriv_lib.deriv_centered(rho_v, 1, additional_states)
    d_rho_w_dz = self._deriv_lib.deriv_centered(rho_w, 2, additional_states)
    return tf.nest.map_structure(
        lambda t0, t1, t2: (t0 + t1 + t2) / dt,
        d_rho_u_dx,
        d_rho_v_dy,
        d_rho_w_dz,
    )

  def _rhie_chow_numerical_consistency_correction(
      self, p: FlowFieldVal, grid_spacings: tuple[float, float, float]
  ) -> FlowFieldVal:
    """Computes the Rhie-Chow correction to the pressure correction equation."""
    if any(self._params.use_stretched_grid):
      raise NotImplementedError(
          'Stretched grids are not yet supported for the Rhie-Chow correction.'
      )

    def multiply_by_scalar(f: FlowFieldVal, scalar: float):
      return tf.nest.map_structure(lambda f: f * scalar, f)

    d4p_dx4 = self._kernel_op.apply_kernel_op_x(p, 'k4d2x')
    d4p_dy4 = self._kernel_op.apply_kernel_op_y(p, 'k4d2y')
    d4p_dz4 = self._kernel_op.apply_kernel_op_z(p, 'k4d2z', 'k4d2zsh')

    px = multiply_by_scalar(d4p_dx4, 1 / (4 * grid_spacings[0] ** 2))
    py = multiply_by_scalar(d4p_dy4, 1 / (4 * grid_spacings[1] ** 2))
    pz = multiply_by_scalar(d4p_dz4, 1 / (4 * grid_spacings[2] ** 2))
    return tf.nest.map_structure(lambda t1, t2, t3: t1 + t2 + t3, px, py, pz)

  def _numerical_consistency_correction_pressure_only(
      self,
      dp: FlowFieldVal,
      rho: FlowFieldVal,
      additional_states: FlowFieldMap,
  ) -> FlowFieldVal:
    """Computes numerical consistency term (non-Rhie-Chow type).

    Computes the numerical consistency term using the strategy described in the
    appendix to Chammas et al., Accelerating Large-Eddy Simulations of Clouds
    With Tensor Processing Units, Journal of Advances in Modeling Earth
    Systems (2023).

    This function is compatible with both Low Mach and Anelastic modes. This
    function is not called when the Rhie-Chow correction is enabled.

    Args:
      dp: The pressure correction.
      rho: The reference density, used for the anelastic mode.
      additional_states: A dictionary that holds helper variables, used here for
        optional stretched grid variables.

    Returns:
      The numerical consistency term.
    """
    anelastic = (
        self._thermodynamics.solver_mode
        == thermodynamics_pb2.Thermodynamics.ANELASTIC
    )
    multiply = lambda a, b: tf.nest.map_structure(tf.multiply, a, b)

    # Compute the one-grid-spacing derivative terms.
    derivs_1h = []
    for dim in (0, 1, 2):
      inner_deriv = self._deriv_lib.deriv_node_to_face(
          dp, dim, additional_states
      )
      if anelastic:
        rho_face = interpolation.centered_node_to_face(
            rho, dim, self._kernel_op
        )
        inner_deriv = multiply(rho_face, inner_deriv)
      outer_deriv = self._deriv_lib.deriv_face_to_node(
          inner_deriv, dim, additional_states
      )
      derivs_1h.append(outer_deriv)

    # Compute the two-grid-spacing derivative terms.
    derivs_2h = []
    for dim in (0, 1, 2):
      inner_deriv = self._deriv_lib.deriv_centered(dp, dim, additional_states)
      if anelastic:
        inner_deriv = multiply(rho, inner_deriv)
      outer_deriv = self._deriv_lib.deriv_centered(
          inner_deriv, dim, additional_states
      )
      derivs_2h.append(outer_deriv)

    # Compute and return the correction term.
    sum_terms = lambda t1, t2, t3: t1 + t2 + t3
    div_2h = tf.nest.map_structure(sum_terms, *derivs_2h)
    div_h = tf.nest.map_structure(sum_terms, *derivs_1h)
    return tf.nest.map_structure(tf.math.subtract, div_2h, div_h)

  def _pressure_corrector_update(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      drho_dt: FlowFieldVal,
      subiter: Optional[tf.Tensor] = None,
      step_id: Optional[tf.Tensor] = None,
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
      drho_dt: Rate of change of density, as a 3D field. Used for the pressure
        solver in Low Mach solver mode. For anelastic mode, this field consists
        of zeros.
      subiter: A scalar Tensor of the integer type that represents the
        subiteration count. Default to `None`, and when it is not `None` and
        corresponding monitor variable `MONITOR_pressure_subiter_convergence` is
        specified in the config, the norm of the pressure residual will be
        stored separately for each subiteration.
      step_id: A `tf.Tensor` denoting the current step id. Default to `None`. It
        is used in logging of pressure solver residual when it is not `None`.

    Returns:
      A Tuple with two elements: The first one is the pressure correction for
        the next time step. The second element is a dictionary for the monitor
        related metrics.
    """
    dt = self._params.dt
    inv_dt = 1.0 / dt
    halo_width = self._params.halo_width

    exchange_halos = functools.partial(
        self._exchange_halos, replica_id=replica_id, replicas=replicas)

    # pylint: enable=line-too-long

    def build_rhs():
      """Builds the right hand side function of the pressure Poisson equation.

      The right hand side expression of Poisson equations is:
          ∇·(ρu)/dt + (∂ρ/∂t)/dt + (src term) + (numerical consistency term)

      Returns:
        A list of tf.Tensor with the right hand side values for the pressure
        Poisson equation of the present time step.
      """
      def multiply_by_scalar(f: FlowFieldVal, scalar: float):
        return tf.nest.map_structure(lambda f: f * scalar, f)
      b_terms = {_B_TERM_SOURCE_RHO: multiply_by_scalar(src_rho, inv_dt)}
      divergence_term = self._compute_divergence_term(
          states['rho_u'],
          states['rho_v'],
          states['rho_w'],
          dt,
          additional_states,
      )

      if self._params.enable_rhie_chow_correction:
        numerical_consistency_correction = (
            self._rhie_chow_numerical_consistency_correction(
                states['p'], self._params.grid_spacings
            )
        )
      else:
        numerical_consistency_correction = (
            self._numerical_consistency_correction_pressure_only(
                states['dp'], states['rho'], additional_states
            )
        )
      b_terms[_B_TERM_DIV] = tf.nest.map_structure(
          tf.add, divergence_term, numerical_consistency_correction
      )
      b_terms[_B_TERM_DRHO_DT] = multiply_by_scalar(drho_dt, inv_dt)

      return (
          tf.nest.map_structure(
              lambda div_i, drho_dt_i, src_rho_i: (
                  div_i + drho_dt_i - src_rho_i
              ),
              b_terms[_B_TERM_DIV],
              b_terms[_B_TERM_DRHO_DT],
              b_terms[_B_TERM_SOURCE_RHO],
          ),
          b_terms,
      )

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

    src_rho = additional_states['mass_source']
    if self._source['rho'] is not None:
      src_rho = tf.nest.map_structure(tf.math.add, self._source['rho'], src_rho)

    # Mean removal is only applied when no Dirichlet boundary conditions are
    # specified, in which case there will be infinite number of solutions.
    has_dirichlet_bc = any(
        not self._params.periodic_dims[dim] and  # pylint: disable=g-complex-comprehension
        self._bc['p'][dim][face][0] == halo_exchange.BCType.DIRICHLET
        for dim in range(3) for face in range(2))
    mean_removal = not has_dirichlet_bc

    b, monitor_params = build_rhs()

    dp0 = tf.nest.map_structure(tf.zeros_like, b)

    helper_vars = dict(additional_states)
    if (self._thermodynamics.solver_mode ==
        thermodynamics_pb2.Thermodynamics.ANELASTIC):
      helper_vars[poisson_solver.VARIABLE_COEFF] = states['rho']

    # Note that the solution that is denoted as `dp` from the Poisson solver has
    # different meanings under different modes of thermodynamics. In the low
    # Mach number model, `dp` is the pressure correction; in the anelastic mode,
    # it is the product of the reference specific volume and the pressure
    # correction.
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

    if debug_output.is_debug_enabled('debug_pressure_residual'):
      residual = self._solver.residual(dp, b, helper_vars)
      debug_output.dump_value('debug_pressure_residual', residual)

    # Debug print is guarded behind the flag so the default (not enabled)
    # is a no-op where the computational graph is not changed.
    if debug_print.log_enabled(_DEBUG_PRINT_LOG_LEVEL):
      norm_types = (_NormType.L1, _NormType.L2, _NormType.L_INF)
      norms, residual_raw = self._solver.compute_residual(
          replica_id=replica_id,
          replicas=replicas,
          f=dp,
          rhs=b,
          norm_types=norm_types,
          halo_width=halo_width,
          halo_update_fn=dp_exchange_halos,
          remove_mean_from_rhs=False,
          internal_dtype=_POISSON_SOLVER_INTERNAL_DTYPE,
      )
      debug_print.log_mean_min_max(
          common_ops.strip_halos(residual_raw, [halo_width] * 3),
          step_id,
          replica_id=replica_id,
          message='Pressure solver local residual: ',
          log_level=_DEBUG_PRINT_LOG_LEVEL,
      )
      for i, norm_type in enumerate(norm_types):
        debug_print.log_mean_min_max(
            norms[i],
            step_id,
            replica_id=replica_id,
            message=f'Pressure solver global residual {norm_type} norm: ',
            log_level=_DEBUG_PRINT_LOG_LEVEL,
        )

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
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ):
    """Updates the boundary condition of pressure based on the flow field.

    Note that the boundary condition for pressure is derived from the flow field
    information if the `update_p_bc_by_flow` flag is set to `true`.

    Also treatment for inhomogeneous shear stress on the wall is currently
    unsupported.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      states: A dictionary that holds flow field variables from the latest
        prediction.
      additional_states: A dictionary that holds helper variables.

    Returns:
      A dictionary that specifies the boundary condition of pressure.
    """
    del replica_id, replicas

    bc_p = [[None, None], [None, None], [None, None]]

    # Note that the diffusion term does not contribute to the pressure boundary
    # condition at a wall based on the following analysis.
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
    # Applying this formulation on the wall surface, which is defined at the
    # mid-point between the first halo layer and the first fluid layer, we see
    # that the second term 1/3𝜇 𝜕/𝜕n (𝜕uₜ/𝜕t) = 0 for both non-slip and
    # free-slip walls. Specifically, for non-slip wall, 𝜕uₜ/𝜕t = 0 because
    # uₜ = 0; for free slip wall, 𝜕uₜ/𝜕n = 0 following its definition. Note
    # that for a shear wall, this term is also 0 if the shear stress is
    # homogeneously distributed on the wall. However, for inhomogeneously
    # distributed shear stress, e.g. with wall models like the Monin-Obukhov
    # similarity theory, the contribution due to this term needs to be
    # considered.
    # Additionally, because an linear extrapolation is performed for the wall
    # normal velocity component across the wall, the second order derivative
    # of it is 0, so that the first term in the shear stress formulation is 0.

    # Updates the pressure boundary condition based on the simulation setup.
    for dim in (0, 1, 2):
      for face in (0, 1):
        if self._params.bc_type[dim][face] == BoundaryType.PERIODIC:
          bc_p[dim][face] = None

        elif self._params.bc_type[dim][face] == BoundaryType.INFLOW:
          bc_p[dim][face] = (halo_exchange.BCType.NEUMANN_2, 0.0)

        elif self._params.bc_type[dim][face] == BoundaryType.OUTFLOW:
          if self._pressure_params.pressure_outlet:
            # Enforce a pressure outlet boundary condition on demand.
            bc_p[dim][face] = (halo_exchange.BCType.DIRICHLET, 0.0)
          else:
            bc_p[dim][face] = (halo_exchange.BCType.NEUMANN_2, 0.0)

        elif self._params.bc_type[dim][face] in (
            BoundaryType.SLIP_WALL,
            BoundaryType.NON_SLIP_WALL,
            BoundaryType.SHEAR_WALL,
        ):
          first_last_grid_spacing = _get_first_last_grid_spacing_for_wall_bc(
              self._params, dim
          )

          if dim == self.g_dim:
            vertical_bc = self._pressure_params.vertical_bc_treatment
            if vertical_bc == VerticalBCTreatment.PRESSURE_BUOYANCY_BALANCING:
              bc_p[dim][face] = self._pressure_bc_balanced_vertical(
                  states, additional_states, face
              )
            elif vertical_bc == VerticalBCTreatment.APPROXIMATE:
              bc_p[dim][face] = self._pressure_bc_approximate_vertical(
                  states, additional_states, face
              )
            else:
              raise ValueError(
                  'Unknown vertical BC treatment:'
                  f' {self._pressure_params.vertical_bc_treatment}'
              )
          else:
            bc_value = common_ops.get_face(
                tf.nest.map_structure(tf.zeros_like, states['p']),
                dim,
                face,
                self._params.halo_width - 1,
                first_last_grid_spacing[face],
            )[0]

            # The boundary condition for pressure is applied at the interface
            # between the boundary and fluid only. Assuming everything is
            # homogeneous behind the halo layer that's closest to the fluid, a
            # homogeneous Neumann BC is applied to all other layers for
            # pressure.
            zeros = [tf.nest.map_structure(tf.zeros_like, bc_value)] * (
                self._params.halo_width - 1
            )
            bc_planes = zeros + [bc_value] if face == 0 else [bc_value] + zeros
            bc_p[dim][face] = (halo_exchange.BCType.NEUMANN, bc_planes)
        else:
          raise ValueError(
              '{} is not defined for pressure boundary.'.format(
                  self._params.bc_type[dim][face]
              )
          )

    self._bc['p'] = bc_p

  def _pressure_bc_approximate_vertical(
      self,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      face: Literal[0, 1],
  ) -> tuple[halo_exchange.BCType, list[tf.Tensor]]:
    """Sets the vertical pressure boundary condition, using approximate balance.

    This function is the original implementation of the vertical boundary
    condition for pressure. The pressure and buoyancy are approximately
    balanced, which can sometimes lead to spurious forcing of the fluid at the
    boundaries. For new simulations, prefer the balanced vertical boundary
    condition instead; this function is retained (for now) but will eventually
    be removed.

    See `update_pressure_bc` for more details.

    Args:
      states: A dictionary that holds flow field variables from the latest
        prediction.
      additional_states: A dictionary that holds helper variables.
      face: Which face to get BC for: bottom (`face==0`) or top (`face==1`).

    Returns:
      A dictionary that specifies the boundary condition of pressure.
    """
    dim = self.g_dim
    first_last_grid_spacing = _get_first_last_grid_spacing_for_wall_bc(
        self._params, dim
    )
    # Default treatment for vertical boundary condition.
    # Ensures the pressure balances with the buoyancy at the first fluid
    # layer by assigning values to the pressure in halos adjacent to the
    # fluid domain. Note that 'zz' in the `additional_states` here refers to the
    # vertical coordinates instead of the z coordinates.
    rho_0 = self._thermodynamics.rho_ref(
        additional_states.get('zz', None), additional_states
    )
    b = eq_utils.buoyancy_source(
        states['rho_thermal'],
        rho_0,
        self._params,
        dim,
        additional_states,
    )
    bc_value = tf.nest.map_structure(
        common_ops.average,
        common_ops.get_face(
            b,
            dim,
            face,
            self._params.halo_width,
            first_last_grid_spacing[face],
        )[0],
        common_ops.get_face(
            b,
            dim,
            face,
            self._params.halo_width - 1,
            first_last_grid_spacing[face],
        )[0],
    )

    # The boundary condition for pressure is applied at the interface
    # between the boundary and fluid only. Assuming everything is
    # homogeneous behind the halo layer that's closest to the fluid, a
    # homogeneous Neumann BC is applied to all other layers for pressure.
    zeros = [tf.nest.map_structure(tf.zeros_like, bc_value)] * (
        self._params.halo_width - 1
    )
    bc_planes = zeros + [bc_value] if face == 0 else [bc_value] + zeros
    bc_p_dim_face = (halo_exchange.BCType.NEUMANN, bc_planes)
    return bc_p_dim_face

  def _pressure_bc_balanced_vertical(
      self,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      face: Literal[0, 1],
  ) -> tuple[halo_exchange.BCType, list[tf.Tensor]]:
    """Updates the boundary condition of pressure based on the flow field.

    Similar to `_update_pressure_bc`, but this function handles the vertical
    boundary condition in an alternate way, which avoids spurious forcing of the
    fluid and should be more stable in the presence of buoyancy.  See comments
    in that function for more detail.

    Args:
      states: A dictionary that holds flow field variables from the latest
        prediction.
      additional_states: A dictionary that holds helper variables.
      face: Which face to get BC for: bottom (`face==0`) or top (`face==1`).

    Returns:
      A dictionary that specifies the boundary condition of pressure.
    """
    dim = self.g_dim
    first_last_grid_spacing = _get_first_last_grid_spacing_for_wall_bc(
        self._params, dim
    )

    # Ensures the pressure balances with the buoyancy at the first fluid layer
    # by assigning values to the pressure in halos adjacent to the fluid domain.
    rho_0 = self._thermodynamics.rho_ref(
        additional_states.get('zz', None), additional_states
    )
    bbar = eq_utils.buoyancy_source(
        states['rho_thermal'],
        rho_0,
        self._params,
        dim,
        additional_states,
    )
    if (
        self._thermodynamics.solver_mode
        == thermodynamics_pb2.Thermodynamics.ANELASTIC
    ):
      b = tf.nest.map_structure(tf.divide, bbar, rho_0)
    else:
      b = bbar
    b_first_interior = common_ops.get_face(
        b, dim, face, self._params.halo_width
    )[0]
    b_second_interior = common_ops.get_face(
        b, dim, face, self._params.halo_width + 1
    )[0]
    if face == 0:
      # If the wall is at index -1/2, use the values of b on the first
      # two interior nodes to extrapolate to the wall, using formula
      #   b_{-1/2} = (3*b_0 - b_1) / 2
      b_wall = tf.nest.map_structure(
          lambda b0, b1: (3 * b0 - b1) / 2,
          b_first_interior,
          b_second_interior,
      )
      # Multiply by the grid spacing at the wall, which is required when using
      # Neumann boundary conditions. E.g., for the lower face, use grid spacing
      # z_0 - z_{-1}.
      bc_value = first_last_grid_spacing[face] * b_wall
      # The boundary condition for pressure is applied at the interface
      # between the boundary and fluid only. Assuming everything is
      # homogeneous behind the halo layer that's closest to the fluid, a
      # homogeneous Neumann BC is applied to all other layers for
      # pressure.
      zeros = [tf.nest.map_structure(tf.zeros_like, bc_value)] * (
          self._params.halo_width - 1
      )
      bc_planes = zeros + [bc_value]
      bc_p_dim_face = (halo_exchange.BCType.NEUMANN, bc_planes)
    else:  # face == 1.
      # N is first halo node, N-1 is last interior, N-2 is second last
      # interior. Set p in the first halo node using:
      #   p_N = p_{N-2} + 2 h_{N-1} b_{N-1}
      p_second_interior = common_ops.get_face(
          states['p'], dim, face, self._params.halo_width + 1
      )[0]
      if self._params.use_stretched_grid[dim]:
        halo_width = self._params.halo_width
        coord = self._params.global_xyz_with_halos[dim]
        dz_last = (
            coord[-(halo_width - 1)] - coord[-(halo_width + 1)]
        ) / 2
      else:
        dz_last = self._params.grid_spacings[dim]

      def compute_bc_value(p, dz, b):
        return tf.nest.map_structure(
            lambda p_, b_: p_ + 2 * dz * b_, p, b
        )

      bc_value = compute_bc_value(
          p_second_interior, dz_last, b_first_interior
      )
      bc_planes = [bc_value] * self._params.halo_width
      bc_p_dim_face = (halo_exchange.BCType.DIRICHLET, bc_planes)
    return bc_p_dim_face

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
      self._update_pressure_bc(replica_id, replicas, states, additional_states)

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
      subiter: Optional[tf.Tensor] = None,
      step_id: Optional[tf.Tensor] = None,
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
      step_id: A `tf.Tensor` denoting the current step id. Default to `None`.
        It is used in logging of pressure solver residual when it is not `None`.

    Returns:
      A dictionary with the updated pressure and pressure corrector.
    """

    # Determine drho_dt based on solver mode.
    if (
        self._thermodynamics.solver_mode
        == thermodynamics_pb2.Thermodynamics.ANELASTIC
    ):
      drho_dt = tf.nest.map_structure(tf.zeros_like, states['rho'])
    elif (
        self._thermodynamics.solver_mode
        == thermodynamics_pb2.Thermodynamics.LOW_MACH
    ):
      exchange_halos = functools.partial(
          self._exchange_halos,
          bc_f=[[(halo_exchange.BCType.NEUMANN, 0.0)] * 2] * 3,
          replica_id=replica_id,
          replicas=replicas)

      # pylint: disable=g-long-lambda
      drho_0 = tf.nest.map_structure(
          lambda drho_i, rho0_i: tf.where(
              tf.abs(drho_i)
              < self._pressure_params.d_rho_rtol * tf.abs(rho0_i),
              tf.zeros_like(drho_i),
              drho_i,
          ),
          states['drho'],
          states_0['rho'],
      )
      # pylint: enable=g-long-lambda

      drho_filter_cond = lambda i, drho_i: i < self._n_filter

      def drho_filter_fn(
          i,
          drho_i,
      ):
        """The body function for drho filtering."""
        return i + 1, exchange_halos(
            filters.filter_op(
                self._params, drho_i, additional_states, order=2
            )
        )

      i0 = tf.constant(0)
      _, drho = tf.nest.map_structure(
          tf.stop_gradient,
          tf.while_loop(
              cond=drho_filter_cond,
              body=drho_filter_fn,
              loop_vars=(i0, drho_0),
          ),
      )
      drho_dt = tf.nest.map_structure(
          lambda drho_i: drho_i / self._params.dt, drho
      )
    else:
      raise ValueError(
          f'Unknown solver mode: {self._thermodynamics.solver_mode}'
      )

    dp, monitor_vars = self._pressure_corrector_update(
        replica_id,
        replicas,
        states,
        additional_states,
        drho_dt,
        subiter,
        step_id,
    )

    states_updated = {
        'p': tf.nest.map_structure(lambda p_, dp_: p_ + dp_, states['p'], dp),
        'dp': dp,
    }
    states_updated.update(monitor_vars)
    return states_updated
