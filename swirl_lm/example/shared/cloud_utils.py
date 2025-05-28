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

"""A library of utility functions for cloud simulations.

Functions in this library include the computation of source terms in the total
energy equation and the humidity equation, which are required in cloud
simulations.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from absl import flags
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.communication import halo_exchange
from swirl_lm.equations import common
from swirl_lm.equations import utils
from swirl_lm.physics.thermodynamics import thermodynamics_pb2
from swirl_lm.physics.thermodynamics import water
from swirl_lm.utility import common_ops
from swirl_lm.utility import composite_types
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import stretched_grid_util
from swirl_lm.utility import types
import tensorflow as tf

_DTYPE = types.TF_DTYPE
_CLOUD_SIM_INITIAL_CLOUD_TOP = flags.DEFINE_float(
    'cloud_sim_initial_cloud_top',
    840.0,
    'The initial height of the cloud, in units of m.',
    allow_override=True)

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap
VectorField = types.VectorField
SourceFn = Callable[
    [
        get_kernel_fn.ApplyKernelOp,
        tf.Tensor,
        np.ndarray,
        FlowFieldMap,
        FlowFieldMap,
        grid_parametrization.GridParametrization,
    ],
    FlowFieldMap,
]

# The rotation rate of the Earth, in units of rad/s.
_OMEGA = 7.2721e-5

# The step size used to solve temperature iteratively from liquid potential
# temperature and humidity.
_STEP_SIZE = 0.3


def _slice_in_dim(
    f: tf.Tensor,
    start_index: int,
    slice_len: int,
    dim: int,
) -> tf.Tensor:
  """Slices the field along a given dimension.

  Args:
    f: The local field to be sliced.
    start_index: The starting vertical index.
    slice_len: The length of the vertical slice.
    dim: The dimension along which the field will be sliced.

  Returns:
    The slice of the field `f` along `dim` starting at the index
    `start_index` and spanning `slice_len` grid points.
  """
  slice_in_dim = slice(start_index, start_index + slice_len)
  shifted_dim = (dim + 1) % 3
  slices = [slice(None), slice(None), slice(None)]
  slices[shifted_dim] = slice_in_dim
  return f[slices]


def _cumsum(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    f: tf.Tensor,
    f_0: tf.Tensor,
    dim: int,
) -> tf.Tensor:
  """Performs cumulative sum for `f` along `dim`.

  Args:
    replica_id: The index of the current core.
    replicas: A 3D array of the topology of the partition.
    f: The 3D tensor to be integrated.
    f_0: The starting value of integration.
    dim: The dimension along which the integration is performed.

  Returns:
    The cumulative sum of `f` along `dim` globally.
  """
  core_i = common_ops.get_core_coordinate(replicas, replica_id)[dim]
  group_assignment = common_ops.group_replicas(replicas, dim)

  f_0 = tf.cond(
      pred=tf.equal(core_i, 0),
      true_fn=lambda: f_0,
      false_fn=lambda: tf.zeros_like(f_0),
  )
  shifted_dim = (dim + 1) % 3
  cumsum_local = tf.math.cumsum(f, axis=shifted_dim) + f_0
  n = tf.cast(cumsum_local.shape[shifted_dim], tf.int32)
  sum_local = _slice_in_dim(cumsum_local, n-1, 1, dim)

  def global_reduce_fn(x):
    return common_ops.global_reduce(
        x[tf.newaxis, ...], tf.math.cumsum, group_assignment)

  cumsum_global = global_reduce_fn(sum_local)
  cumsum_prev = tf.cond(
      pred=tf.equal(core_i, 0),
      true_fn=lambda: tf.zeros_like(sum_local),
      false_fn=lambda: tf.squeeze(cumsum_global[core_i - 1, ...], axis=0),
  )
  return cumsum_local + cumsum_prev


def compute_buoyancy_balanced_hydrodynamic_pressure(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    rho: FlowFieldVal,
    rho_0: FlowFieldVal,
    g_dim: int,
    params: parameters_lib.SwirlLMParameters,
    additional_states: FlowFieldMap
) -> FlowFieldVal:
  """Computes p so that dp/dz = (rho - rho_0) g with central difference.

  From:
  p^2 - p^0 = -2h (rho^1 - rho_0^1) g,
  p^4 - p^2 = -2h (rho^3 - rho_0^3) g,
  ...
  We have:
  p^(2n) - p^0 = -2h g sum_{k = 1}^{k = n} (rho^{2k - 1} - rho_0^{2k - 1}).

  Similarly,
  p^{2n + 1} - p^1 = -2h g sum_{k = 1}^{k = n} (rho^{2k} - rho_0^{2k}).

  Args:
    kernel_op: An object holding a library of kernel operations.
    replica_id: The index of the current core.
    replicas: A 3D array of the topology of the partition.
    rho: The density of the flow field.
    rho_0: The density of the background reference.
    g_dim: The dimension of the vertical direction.
    params: SwirlLMParameters that contains the configuration for the
      simulation.
    additional_states: Dictionary of helper variables, used for stretched grid
      arrays.

  Returns:
    The hydrodynamic pressure that balances the buoyancy term numerically.
  """
  use_3d_tf_tensor = isinstance(rho, tf.Tensor)
  if not use_3d_tf_tensor:
    # Transform the required stretched grid arrays from list-of-2D to 3D format.
    stretched_grid_states = stretched_grid_util.get_helper_variables(
        additional_states
    )
    additional_states_3d = {
        key: tf.stack(value) for key, value in stretched_grid_states.items()
    }
    p_3d = compute_buoyancy_balanced_hydrodynamic_pressure(
        kernel_op,
        replica_id,
        replicas,
        tf.stack(rho),
        tf.stack(rho_0),
        g_dim,
        params,
        additional_states_3d,
    )
    return tf.unstack(p_3d)

  del kernel_op
  b = utils.buoyancy_source(rho, rho_0, params, g_dim, additional_states)

  # In ANELASTIC mode, the gradient to be balanced is for the buoyancy
  # normalized by the reference density.
  if params.solver_mode == thermodynamics_pb2.Thermodynamics.ANELASTIC:
    b = b / rho_0

  core_n = (params.core_nx, params.core_ny, params.core_nz)[g_dim]
  cores = (params.cx, params.cy, params.cz)[g_dim]
  # Total number of grid points along the vertical direction.
  global_n = cores * core_n
  halo_width = params.halo_width

  def pad_in_vertical(t: types.FlowFieldVal, low_n: int, high_n: int):
    """Pads the field with zeros along the vertical direction."""
    paddings = [(0, 0)] * 3
    paddings[g_dim] = (low_n, high_n)
    return common_ops.pad(t, paddings)

  def binary_mask_like(
      f: types.FlowFieldVal,
      even: bool,
  ) -> FlowFieldVal:
    """Creates a mask like `f` with 1's at alternating vertical indices."""
    core_i = common_ops.get_core_coordinate(replicas, replica_id)[g_dim]
    offset = 1 if even else 0
    sequence = tf.slice(
        tf.linspace(offset, global_n - 1 + offset, global_n),
        [core_i * core_n],
        [core_n],
    )
    binary_sequence = tf.cast(tf.math.mod(sequence, 2), dtype=_DTYPE)
    shifted_dim = (g_dim + 1) % 3
    broadcast_shape = [1, 1, 1]
    broadcast_shape[shifted_dim] = core_n
    broadcastable_mask = tf.reshape(binary_sequence, broadcast_shape)
    ones = tf.ones_like(f)
    return ones * broadcastable_mask

  # Compute the buoyancy multiplied by twice the grid spacing.
  if params.use_stretched_grid[g_dim]:
    h = additional_states[stretched_grid_util.h_key(g_dim)]
  else:
    h = params.grid_spacings[g_dim]
  b_dz = 2 * b * h

  # Remove the halos along the vertical direction.
  b = _slice_in_dim(b, halo_width, core_n, g_dim)
  b_dz = _slice_in_dim(b_dz, halo_width, core_n, g_dim)

  # Split the buoyancy term into odd and even indices. The first internal fluid
  # node is assumed to have index 0.
  even_mask = binary_mask_like(b_dz, even=True)
  odd_mask = binary_mask_like(b_dz, even=False)

  b_dz_even = even_mask * b_dz
  b_dz_odd = odd_mask * b_dz

  # Prepend a 0 to the buoyancy terms to make them align with the corresponding
  # pressure levels.
  b_dz_even = pad_in_vertical(b_dz_even, 1, 0)
  b_dz_odd = pad_in_vertical(b_dz_odd, 1, 0)
  # Integrate the buoyancy tensors to get the pressure.
  zeros = tf.zeros_like(_slice_in_dim(b_dz_odd, 0, 1, g_dim))
  p_even = _cumsum(replica_id, replicas, b_dz_odd, zeros, g_dim)
  p_even = even_mask * _slice_in_dim(p_even, 0, core_n, g_dim)

  # Shifting the starting point of integration of the odd part so that the
  # overall profile is smooth. That is, the p_odd and p_even sequences are
  # decoupled, so we choose the starting point of p_odd so that the p profile
  # is smooth.
  # We can choose a smooth p by using trapezoidal integration from z0 to z1 on
  # the continuous buoyancy equilibrium equation, obtaining:
  #   p1 = p0 + hf[1] * (b0 + b1) / 2, where hf[1] = z1 - z0
  p_0 = _slice_in_dim(p_even, 0, 1, g_dim)
  b0 = _slice_in_dim(b, 0, 1, g_dim)
  b1 = _slice_in_dim(b, 1, 1, g_dim)
  b_dz_0 = _slice_in_dim(b_dz_even, 1, 1, g_dim)

  if params.use_stretched_grid[g_dim]:
    h_face = additional_states[stretched_grid_util.h_face_key(g_dim)]
    hf1 = tf.squeeze(h_face)[halo_width + 1]  # z1 - z0.
    delta_p = hf1 * (b0 + b1) / 2
  else:
    delta_p = params.grid_spacings[g_dim] * (b0 + b1) / 2

  # The desired value for p_1 is p_0 + delta_p. But because of the way the
  # cumsum function behaves, b_dz_0 is added, so we need to subtract it here.
  # Detailed explanation:
  #   b_dz_even:
  #     [0, g0, 0, g2, 0, g4, 0, ...]
  #   cumsum with f_0 = 0 is:
  #     [0, g0, g0, g0 + g2, g0 + g2, g0 + g2 + g4, ...]
  #   cumsum with nonzero f_0 is
  #     [f_0, f_0 + g0, f_0 + g0, f_0 + g0 + g2, f_0 + g0 + g2, ...]
  #   Apply odd mask:
  #     [0, f_0 + g0, 0, f_0 + g0 + g2, 0, f_0 + g0 + g2 + g4, 0, ...]
  #   Take f_0 = p_1 - g0:
  #     [0, p1, 0, p1 + g2, ...]
  p_1 = p_0 + delta_p
  f_0 = p_1 - b_dz_0

  p_odd = _cumsum(replica_id, replicas, b_dz_even, f_0, g_dim)
  p_odd = odd_mask * _slice_in_dim(p_odd, 0, core_n, g_dim)

  # Combine the 2 tensors into one and update values in the halos. Note that
  # only the layer of halo that is closest to the interior domain matters, which
  # is used to compute the pressure gradient at the first fluid point.
  p = p_even + p_odd
  # The lower boundary condition corresponding to the innermost halo (p_l) must
  # satisfy p[1] - p_l = b_dz[0]
  p_l = _slice_in_dim(p, 1, 1, g_dim) - _slice_in_dim(b_dz, 0, 1, g_dim)
  # The upper boundary condition (p_h) must satisfy p_h - p[n-2] = bd_z[n-1],
  # where n-1 is the index of the last internal fluid layer.
  p_h = _slice_in_dim(p, core_n - 2, 1, g_dim) + _slice_in_dim(
      b_dz, core_n - 1, 1, g_dim
  )

  p = pad_in_vertical(p, halo_width, halo_width)
  bc = [[(halo_exchange.BCType.NEUMANN, 0.0)] * 2] * 3
  bc[g_dim] = [
      (
          halo_exchange.BCType.DIRICHLET,
          [p_l] * halo_width,
      ),
      (
          halo_exchange.BCType.DIRICHLET,
          [p_h] * halo_width,
      ),
  ]
  p = halo_exchange.inplace_halo_exchange(
      p,
      (0, 1, 2),
      replica_id,
      replicas,
      (0, 1, 2),
      params.periodic_dims,
      boundary_conditions=bc,
      width=halo_width,
  )

  # In LOW_MACH mode, the output represents the hydrodynamic pressure p.  In
  # ANELASTIC mode, the output represents p / rho_ref.
  return p


def cross_product(
    f: Sequence[float],
    g: VectorField,
    components: Sequence[int],
) -> Dict[int, List[float]]:
  """Calculates f × g, where f is a constant vector and g a vector field."""
  # TODO(b/222068509): Make this more generic and move to common_ops.
  fx, fy, fz = f
  gx, gy, gz = g
  result = {}
  if 0 in components:
    cross_x = tf.nest.map_structure(
        lambda gy_k, gz_k: fy * gz_k - fz * gy_k, gy, gz
    )
    result.update({0: cross_x})
  if 1 in components:
    cross_y = tf.nest.map_structure(
        lambda gx_k, gz_k: fz * gx_k - fx * gz_k, gx, gz
    )
    result.update({1: cross_y})
  if 2 in components:
    cross_z = tf.nest.map_structure(
        lambda gx_k, gy_k: fx * gy_k - fy * gx_k, gx, gy
    )
    result.update({2: cross_z})
  return result


def coriolis_force(
    phi: float,
    u_g: Dict[str, float],
    vertical_dim: int,
    ambient_from_states: bool = False,
) -> composite_types.StatesUpdateFn:
  """Generates an update function for the Coriolis force term.

  The Coriolis coefficient f is computed as:
    f = 2Ω sin φ,
  where:
    Ω = 7.2921 × 10⁻⁵ rad/s is rotation rate of the Earth, and
    φ is the latitude.
  Let fg be the vector with f in the vertical dimension and 0 elsewhere.
  The Coriolis force term is computed as:
    F = -fg ⨯ (u - Ug)
  where u and Ug are the velocity and geostrophic wind vectors, respectively.

  Args:
    phi: The latitude (in radian).
    u_g: A dict containing the geostrophic winds keyed by the velocity
      component they correspond to.
    vertical_dim: The dimension of the vertical direction.
    ambient_from_states: An indicator of whether to use ambient states from
      `additional_states` instead of `u_g` to compute the Coriolis force. The
      ambient states has to be named as '{varname}_ambient'.

  Returns:
    An update function for the `additional_states` with keys `src_u` and
    `src_v`
    that stores the Coriolis force term.
  """
  f = 2.0 * _OMEGA * np.sin(phi)
  coeff = [0, 0, 0]
  coeff[vertical_dim] = f
  ordered_velocity_keys = (common.KEY_U, common.KEY_V, common.KEY_W)

  def subtract_scalar(t, sc):
    return tf.nest.map_structure(lambda t_i: t_i - sc, t)

  def get_force_update_fn(
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Computes the Coriolis force and store in `src_u` and `src_v`."""
    del kernel_op, replica_id, replicas, params

    additional_states_new = {}
    deltas = []
    for k in ordered_velocity_keys:
      ambient_name = f'{k}_ambient'
      if not ambient_from_states or ambient_name not in additional_states:
        deltas.append(subtract_scalar(states[k], u_g[k]))
      else:
        deltas.append(
            tf.nest.map_structure(
                tf.math.subtract, states[k], additional_states[ambient_name]
            )
        )
    deltas = tuple(deltas)
    horizontal_dims = [0, 1, 2]
    del horizontal_dims[vertical_dim]

    # Compute the horizontal components of the cross product.
    velocity_srcs = cross_product(coeff, deltas, horizontal_dims)
    dim_to_src = {
        k: tf.nest.map_structure(lambda src_i: -src_i, src) for k, src in
        velocity_srcs.items()
    }
    for dim in horizontal_dims:
      src_key = 'src_{}'.format(ordered_velocity_keys[dim])
      if src_key in additional_states:
        additional_states_new.update({src_key: dim_to_src[dim]})

    additional_states_new.update({
        k: v
        for k, v in additional_states.items()
        if k not in additional_states_new
    })
    return additional_states_new

  return get_force_update_fn


def add_source_fns(source_fns: Sequence[SourceFn]) -> SourceFn:
  """Generates a function that computes the sum of all source functions.

  Args:
    source_fns: A sequence of source functions. The source functions can contain
      source terms for multiple variables.

  Returns:
    The sum of sources computed with all source function in `source_fns`.
  """

  def source_fn(
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Computes the sum of all source functions in `source_fns`."""
    combined_source_fns = {}
    for source_fn in source_fns:
      source = source_fn(
          kernel_op, replica_id, replicas, states, additional_states, params
      )
      for key, val in source.items():
        if key in combined_source_fns:
          combined_source_fns[key] = tf.nest.map_structure(
              tf.math.add, combined_source_fns[key], val
          )
        else:
          combined_source_fns[key] = val

    return combined_source_fns

  return source_fn


class CloudUtils(object):
  """A library of utilities for cloud simulations."""

  def __init__(
      self,
      config: parameters_lib.SwirlLMParameters,
      initial_cloud_top: float,
      vertical_dim: int,
      temperature_max_iter: Optional[int] = 100,
      temperature_tol: Optional[float] = 1e-4,
  ):
    """Initializes parameters required in a cloud simulation."""
    self.config = config
    self.thermodynamics = water.Water(self.config)
    self._zi = initial_cloud_top
    self._t_max_iter = temperature_max_iter
    self._t_tol = temperature_tol
    self._p_inv = 1.0 / self.config.p_thermal
    self._vertical_dim = vertical_dim

  @property
  def zi(self) -> float:
    """The initial cloud top height."""
    return self._zi

  def temperature_update_fn(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Computes the temeprature based on total energy and humidity.

    Args:
      kernel_op: An object holding a library of kernel operations.
      replica_id: The id of the replica.
      replicas: The replicas. In particular, a numpy array that maps grid
        coordinates to replica id numbers.
      states: A keyed dictionary of states that will be updated.
      additional_states: A list of states that are needed by the update fn, but
        will not be updated by the main governing equations.
      params: An instance of `grid_parametrization.GridParametrization`.

    Returns:
      `additional_states` with updated temperature.
    """
    del kernel_op, replica_id, replicas, params

    temperatures = self.thermodynamics.update_temperatures(
        states, additional_states)

    updated_additional_states = {}
    updated_additional_states.update(additional_states)
    if 'T' in additional_states.keys():
      updated_additional_states.update({'T': temperatures['T']})

    if 'theta_li' in additional_states.keys():
      updated_additional_states.update({'theta_li': temperatures['theta_li']})

    if 'theta_v' in additional_states.keys():
      updated_additional_states.update({'theta_v': temperatures['theta_v']})

    if 'theta' in additional_states.keys():
      updated_additional_states.update({'theta': temperatures['theta']})

    return updated_additional_states  # pytype: disable=bad-return-type

  def temperature(
      self,
      theta: FlowFieldVal,
      q_t: FlowFieldVal,
      zz: FlowFieldVal,
      theta_type: str = water.PotentialTemperature.THETA_LI.value,
      helper_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldMap:
    """Solves for temperature iteratively given `theta` and `q_t`.

    Args:
      theta: The potential temperature of type `theta_type`, in units of K.
      q_t: The total specific humidity, in units of kg/kg.
      zz: The vertical coordinates, in units of m.
      theta_type: The type of the potential temperature.
      helper_states: Helper variables that might be required to determine
        thermodynamic states.

    Returns:
      A dictionary that contains temperature, the mass specific gas constant of
      the mixture, and the specific heat of the mixture.
    """
    # Variables in the loop are:
    #   - 'q_v': The water vapor mass fraction, in units of kg/kg.
    #   - 'q_c': The condensed phase water mass fraction, in units of kg/kg.
    #   - 'res': The change in potential temperature in the new iteration.
    states_0 = {
        'q_v': tf.nest.map_structure(tf.zeros_like, zz),
        'q_c': q_t,
        'res': q_t,
    }
    q_i = tf.nest.map_structure(tf.zeros_like, zz)
    p = self.thermodynamics.p_ref(zz, helper_states)

    def body(i: tf.Tensor,
             states: FlowFieldMap) -> Tuple[tf.Tensor, FlowFieldMap]:
      """Solves the potential temperature iteratively."""
      r_m = self.thermodynamics.r_mix(q_t, states['q_c'])

      # Because there's no ice phase in the current simulation setup, the liquid
      # humidity is equivalent to condensed phase humidity.
      q_l = states['q_c']

      t = self.thermodynamics.potential_temperature_to_temperature(
          theta_type, theta, q_t, q_l, q_i, zz, helper_states)
      rho = tf.nest.map_structure(lambda p_i, r_m_i, t_i: p_i / r_m_i / t_i, p,
                                  r_m, t)
      q_v_s = self.thermodynamics.saturation_q_vapor(t, rho, q_l, states['q_c'])
      q_v = tf.nest.map_structure(tf.minimum, q_t, q_v_s)
      q_c = tf.nest.map_structure(tf.math.subtract, q_t, q_v)

      q_c = tf.nest.map_structure(
          lambda q_c_new, q_c_old: q_c_old + _STEP_SIZE * (q_c_new - q_c_old),
          q_c, states['q_c'])

      res = tf.nest.map_structure(tf.math.subtract, q_c, states['q_c'])

      return i + 1, {'q_v': q_v, 'q_c': q_c, 'res': res}

    def cond(i: tf.Tensor, states: FlowFieldMap) -> tf.Tensor:
      """The continue condition of the temperature iteration."""
      return tf.math.logical_and(
          tf.less(i, self._t_max_iter),
          tf.math.reduce_any(
              tf.nest.map_structure(
                  lambda res: tf.greater(tf.math.abs(res), self._t_tol),
                  states['res'],
              )
          ),
      )

    i0 = tf.constant(0)
    _, states = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(i0, states_0),
        back_prop=False,
    )

    return {
        'temperature':
            self.thermodynamics.potential_temperature_to_temperature(
                theta_type, theta, q_t, states['q_c'], q_i, zz, helper_states),
        'r_m':
            self.thermodynamics.r_mix(q_t, states['q_c']),
        'cp_m':
            self.thermodynamics.cp_m(q_t, states['q_c'], q_i),
    }


def cloud_utils_factory(
    config: Optional[parameters_lib.SwirlLMParameters] = None,
    initial_cloud_top: Optional[float] = None,
    vertical_dim: int = 2,
) -> CloudUtils:
  """Generates a handler of `CloudUtils`."""
  # pylint: disable=g-long-ternary
  config = config if config is not None else (
      parameters_lib.params_from_config_file_flag())
  # pylint: enable=g-long-ternary
  initial_cloud_top = (
      initial_cloud_top
      if initial_cloud_top is not None else _CLOUD_SIM_INITIAL_CLOUD_TOP.value)
  return CloudUtils(config, initial_cloud_top, vertical_dim)
