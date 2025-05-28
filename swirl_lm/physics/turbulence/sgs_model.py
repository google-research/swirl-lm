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
"""A library for the sub-grid scale models in large-eddy simulations.

Note that SGS models are fine tuned for 3D problems. It is not advisable to use
these models for quasi-2D problems:
Reference: Awad E, Toorman E, Lacor C. Large eddy simulations for quasi-2D
turbulence in shallow flows: A comparison between different subgrid scale
models. Journal of Marine Systems. 2009 Jun 1;77(4):511-28.

Length scale is defined as sqrt(dx**2 + dy**2 + dz**2) or (dx*dy*dz)**(1/3)
based on the specific model. Hence, for quasi-2D problem along the X-Y plane,
the dz is arbitrary. In case any SGS model is used for such configurations, lz
and nz have to carefully set so that an arbitrary value of dz does not pollute
the turbulent viscosity.
"""

import functools
import itertools
from typing import Optional, Sequence, TypeAlias

from absl import logging
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import parameters_pb2
from swirl_lm.numerics import calculus
from swirl_lm.numerics import filters
from swirl_lm.physics import constants
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf


FlowFieldVal: TypeAlias = types.FlowFieldVal
FlowFieldMap: TypeAlias = types.FlowFieldMap
DeltaFormula: TypeAlias = (
    parameters_pb2.SubGridScaleModel.SmagorinskyModel.DeltaFormula)

# The originally proposed value for the Smagorinsky constant (dimensionless)
# with turbulence being isotropic and homogeneous.
_CS_CLASSICAL = 0.18


def _test_filter(value: FlowFieldVal) -> FlowFieldVal:
  """Filter `value` with twice the filter width `delta`.

  Args:
    value: The 3D tensor needs to be filtered.

  Returns:
    Value filtered by a box filter with stencil width 3.
  """
  # A dummpy halo update function used by the `global_box_filter_3d`. Doing
  # this is equivalent to applying a local box filter.
  dummy_halo_update = lambda value: value

  return filters.global_box_filter_3d(
      value, dummy_halo_update, filter_width=3, num_iter=1
  )


def _dot(u_i: FlowFieldVal, u_j: FlowFieldVal) -> FlowFieldVal:
  """Computes the dot product between two 3D tensors.

  Args:
    u_i: The first 3D tensor.
    u_j: The second 3D tenosr.

  Returns:
    The dot product of `u_i` and `u_j`.
  """
  return tf.nest.map_structure(tf.math.multiply, u_i, u_j)


def _einsum_ij(
    u: Sequence[Sequence[FlowFieldVal]], v: Sequence[Sequence[FlowFieldVal]]
) -> Sequence[Sequence[FlowFieldVal]]:
  """Performs Einstein sum u_i,j v_i,j.

  Note that operations are performed elementwise here. A possible improvement to
  consider is converting everything into 5-D tensors and use `tf.einsum`.

  Args:
    u: A structure of 3D Tensors with u[i][j] being the 3D tensor of u_i,j.
    v: A structure of 3D Tensors with v[i][j] being the 3D tensor of v_i,j.

  Returns:
    The Einstein sum of u and v.

  Raises:
    ValueError if the first and second dimension of `u` and `v` mismatch.
  """
  n1 = len(u)
  n2 = len(u[0])
  if n1 != len(v) or n2 != len(v[0]):
    raise ValueError(
        'Dimension mismatch: u is {} x {}, v is {} x {}'.format(
            n1, n2, len(v), len(v[0])
        )
    )

  res = tf.nest.map_structure(tf.zeros_like, u[0][0])
  for i, j in itertools.product(range(n1), range(n2)):
    res = tf.nest.map_structure(
        lambda res_ij, u_ij, v_ij: res_ij + u_ij * v_ij, res, u[i][j], v[i][j]
    )
  return res


def _strain_rate_magnitude(
    strain_rate: Sequence[Sequence[FlowFieldVal]],
) -> FlowFieldVal:
  """Computes the magnitude of the strain rate tensor."""
  strain_rate_prod = tf.nest.map_structure(tf.zeros_like, strain_rate[0][0])

  for i in range(len(strain_rate)):
    for j in range(3):
      strain_rate_prod = tf.nest.map_structure(
          lambda s_l, strain_rate_ij: s_l + 2.0 * tf.square(strain_rate_ij),
          strain_rate_prod,
          strain_rate[i][j],
      )

  return tf.nest.map_structure(tf.math.sqrt, strain_rate_prod)


def _strain_rate_tensor(
    du_dx: Sequence[Sequence[FlowFieldVal]],
) -> Sequence[Sequence[FlowFieldVal]]:
  """Computes the strain rate tensor based on velocity gradients.

  The strain rate is defined as Sₖₗ = 0.5 * (𝜕uₖ/𝜕xₗ + 𝜕uₗ/𝜕xₖ) - 1/3 𝛁u.

  Args:
    du_dx: The velocity gradient tensor. The first index indicates the velocity
      component, and the second index indicates the direction.

  Returns:
    The strain rate tensor.
  """
  sij = [
      [common_ops.average(du_dx[i][j], du_dx[j][i]) for j in range(3)]
      for i in range(3)
  ]
  div = tf.nest.map_structure(
      lambda du_11, du_22, du_33: du_11 + du_22 + du_33,
      du_dx[0][0],
      du_dx[1][1],
      du_dx[2][2],
  )

  def remove_divergence(value: FlowFieldVal) -> FlowFieldVal:
    """Remove 1/3 of the divergence from `value`."""
    return tf.nest.map_structure(
        lambda value_i, div_i: value_i - div_i / 3.0, value, div
    )

  return [
      [sij[i][j] if i != j else remove_divergence(sij[i][j]) for j in range(3)]
      for i in range(3)
  ]


def _germano_averaging(
    value: FlowFieldVal,
    periodic_dims: Sequence[bool],
    replicas: np.ndarray,
) -> FlowFieldVal:
  """Computes the Germano averaging across all periodic directions."""
  cx, cy, cz = replicas.shape
  nx, ny, nz = common_ops.get_shape(value)

  count = 1.0

  def sum_in_dim(val: tf.Tensor, group_assignment: np.ndarray) -> tf.Tensor:
    """Computes the global sum in one dimension."""
    local_sum_op = functools.partial(tf.math.reduce_sum, axis=0)
    return common_ops.global_reduce(
        tf.expand_dims(val, 0), local_sum_op, group_assignment
    )

  def repeat(val: tf.Tensor, axis: int, reps: int) -> tf.Tensor:
    """Repeats `val` along `axis` `reps` time."""
    return tf.repeat(tf.expand_dims(val, axis), reps, axis)

  if isinstance(value, tf.Tensor):
    # Dimensions that correspond to the physical axes of a 3D tensor in
    # Swirl-LM, i.e. physical dimension 0 corresponds to the dimension index 1
    # in a 3D tensor.
    axes = [1, 2, 0]
    for dim in range(3):
      if not periodic_dims[dim]:
        replicas = np.transpose(replicas, (1, 2, 0))
        continue
      n = (nx, ny, nz)[dim]

      # Here we always shift the dimension for communication to the first one.
      c0, c1, c2 = replicas.shape
      group_assignment = np.array([
          replicas[:, i, j] for i, j in itertools.product(range(c1), range(c2))
      ])
      replicas = np.transpose(replicas, (1, 2, 0))

      local_sum = tf.math.reduce_sum(value, axis=axes[dim])
      value = repeat(sum_in_dim(local_sum, group_assignment), axes[dim], n)
      count *= float(n * c0)

  else:
    if periodic_dims[2]:
      group_assignment_z = np.array([
          replicas[i, j, :] for i, j in itertools.product(range(cx), range(cy))
      ])
      local_sum = tf.zeros_like(value[0], dtype=value[0].dtype)
      for value_i in value:
        local_sum += value_i
      value_z_sum = sum_in_dim(local_sum, group_assignment_z)
      value = [
          value_z_sum,
      ] * nz

      count *= float(nz * cz)

    if periodic_dims[1]:
      group_assignment_y = np.array([
          replicas[i, :, k] for i, k in itertools.product(range(cx), range(cz))
      ])
      local_sum = tf.nest.map_structure(
          lambda value_i: tf.math.reduce_sum(value_i, axis=1), value
      )
      value = tf.nest.map_structure(
          lambda sum_i: repeat(sum_in_dim(sum_i, group_assignment_y), 1, ny),
          local_sum,
      )
      count *= float(ny * cy)

    if periodic_dims[0]:
      group_assignment_x = np.array([
          replicas[:, j, k] for j, k in itertools.product(range(cy), range(cz))
      ])
      local_sum = tf.nest.map_structure(
          lambda value_i: tf.math.reduce_sum(value_i, axis=0), value
      )
      value = tf.nest.map_structure(
          lambda sum_i: repeat(sum_in_dim(sum_i, group_assignment_x), 0, nx),
          local_sum,
      )
      count *= float(nx * cx)

  return tf.nest.map_structure(lambda value_i: value_i / count, value)


class SgsModel(object):
  """A library of sub-grid scale (SGS) models in large-eddy simulations."""

  def __init__(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      swirllm_params: parameters_lib.SwirlLMParameters,
  ):
    """Initializes the sub-grid scale model library.

    Args:
      kernel_op: An object holding a library of kernel operations.
      swirllm_params: An instance of SwirlLMParameters specifying the config.
    """
    self._kernel_op = kernel_op
    self._swirllm_params = swirllm_params
    self._params = swirllm_params.sgs_model
    self._deriv_lib = swirllm_params.deriv_lib

    self._nu_t_max = (
        self._params.nu_t_max if self._params.HasField('nu_t_max') else None
    )
    self._diff_t_max = (
        self._params.diff_t_max if self._params.HasField('diff_t_max') else None
    )

    if not self._params or not self._params.HasField('sgs_model_type'):
      logging.warning(
          'SGS is used but no model is specified. Turbulent viscosity and '
          'diffusivity are computed using the Smagorinsky model with default '
          'constants (C_s = 0.18, Pr_t = 0.3).'
      )
    else:
      logging.info('SGS model: %r', self._params)

  def turbulent_diffusivity(
      self,
      field_vars: Sequence[FlowFieldVal],
      additional_states: FlowFieldMap,
      velocity: Optional[Sequence[FlowFieldVal]] = None,
      replicas: Optional[np.ndarray] = None,
  ) -> FlowFieldVal:
    """Computes the turbulent diffusivity for `field_vars`.

    Args:
      field_vars: Scalars based on which the model is computed. The length of
        `field_vars` can be an arbitrary number. The SGS term is computed based
        on the magnitude of the gradient vector.
      additional_states: A map of variables that might be needed by the SGS
        model requested.
      velocity: A list of 3D tensors representing the 3 velocity components.
        Required in the dynamic Smagorinsky model.
      replicas: The topology of the TPU cores.

    Returns:
      The turbulent diffusivity.

    Raises:
      ValueError: If dynamic Smagorinsky model is used but no velocity is
        provided.
      ValueError: If dynamic Smagorinsky model is used but no replicas is
        provided.
      ValueError: If the requested SGS model type is not implemented.
    """
    if not self._params or not self._params.HasField('sgs_model_type'):
      return self.smagorinsky(field_vars, additional_states,
                              DeltaFormula.DIAGONAL)

    if self._params.WhichOneof('sgs_model_type') == 'smagorinsky':
      use_pr_t = self._params.smagorinsky.use_pr_t
      coeff = np.sqrt(self._params.smagorinsky.pr_t) if use_pr_t else 1.0

      if 'c_s' in additional_states:
        c_s = tf.nest.map_structure(
            lambda c_s_i: c_s_i / coeff,
            additional_states['c_s'],
        )
      else:
        c_s = tf.nest.map_structure(
            lambda var: self._params.smagorinsky.c_s
            / coeff
            * tf.ones_like(var),
            field_vars[0],
        )
      smagorinsky_vars = velocity if use_pr_t else field_vars
      diff_t = self.smagorinsky(
          smagorinsky_vars, additional_states,
          self._params.smagorinsky.delta_formula, c_s)
    elif self._params.WhichOneof('sgs_model_type') == 'dynamic_smagorinsky':
      if not velocity:
        raise ValueError(
            'Velocity field is required for the dynamic Smagorinsky model'
        )
      if replicas is None:
        raise ValueError(
            'TPU topology replicas needs to be specified for the '
            'dynamic Smagorinsky model.'
        )
      periodic_dims = [
          self._params.dynamic_smagorinsky.periodic_x,
          self._params.dynamic_smagorinsky.periodic_y,
          self._params.dynamic_smagorinsky.periodic_z,
      ]
      diff_t = self.dynamic_smagorinsky(
          periodic_dims,
          replicas,
          velocity,
          additional_states,
          field_vars[0],
      )
    elif self._params.WhichOneof('sgs_model_type') == 'smagorinsky_lilly':
      if 'theta_v' not in additional_states:
        raise ValueError(
            '`theta_v` is required in the `additional_states` for the '
            'Smagorinsky-Lilly model.'
        )

      use_pr_t = self._params.smagorinsky_lilly.use_pr_t
      coeff = np.sqrt(self._params.smagorinsky_lilly.pr_t) if use_pr_t else 1.0
      c_s = self._params.smagorinsky_lilly.c_s / coeff

      scalar = velocity if use_pr_t else field_vars

      diff_t = self.smagorinsky_lilly(
          scalar,
          velocity,
          additional_states['theta_v'],
          c_s,
          self._params.smagorinsky_lilly.pr_t,
          additional_states,
      )
    elif self._params.WhichOneof('sgs_model_type') == 'vreman':
      diff_t = tf.nest.map_structure(
          lambda nu_t: nu_t / self._params.vreman.pr_t,
          self.vreman(velocity, self._params.vreman.c_s, additional_states),
      )
    else:
      raise ValueError(
          'Unsupported sub-grid scale model {}'.format(
              self._params.WhichOneof('sgs_model_type')
          )
      )

    diff_t = tf.nest.map_structure(
        lambda d: tf.math.maximum(d, self._params.diff_t_min), diff_t
    )
    return diff_t if self._diff_t_max is None else tf.nest.map_structure(
        lambda d: tf.math.minimum(d, self._diff_t_max), diff_t
    )

  def turbulent_viscosity(
      self,
      field_vars: Sequence[FlowFieldVal],
      additional_states: FlowFieldMap,
      replicas: Optional[np.ndarray] = None,
  ) -> FlowFieldVal:
    """Computes the turbulent viscosity for `field_vars`.

    Args:
      field_vars: A length-three vector representing the three velocity
        components of the flow field.
      additional_states: A map of variables that might be needed by the SGS
        model requested.
      replicas: The topology of the TPU cores.

    Returns:
      The turbulent viscosity.

    Raises:
      ValueError: If dynamic Smagorinsky model is used but no replicas is
        provided.
      ValueError: If the requested SGS model type is not implemented.
    """
    if not self._params or not self._params.HasField('sgs_model_type'):
      return self.smagorinsky(field_vars, additional_states,
                              DeltaFormula.DIAGONAL)

    if self._params.WhichOneof('sgs_model_type') == 'smagorinsky':
      if 'c_s' in additional_states:
        c_s = additional_states['c_s']
      else:
        c_s = tf.nest.map_structure(
            lambda var: self._params.smagorinsky.c_s * tf.ones_like(var),
            field_vars[0],
        )
      nu_t = self.smagorinsky(field_vars, additional_states,
                              self._params.smagorinsky.delta_formula, c_s)
    elif self._params.WhichOneof('sgs_model_type') == 'dynamic_smagorinsky':
      if replicas is None:
        raise ValueError(
            'TPU topology replicas needs to be specified for the '
            'dynamic Smagorinsky model.'
        )
      periodic_dims = [
          self._params.dynamic_smagorinsky.periodic_x,
          self._params.dynamic_smagorinsky.periodic_y,
          self._params.dynamic_smagorinsky.periodic_z,
      ]
      nu_t = self.dynamic_smagorinsky(
          periodic_dims, replicas, field_vars, additional_states
      )
    elif self._params.WhichOneof('sgs_model_type') == 'smagorinsky_lilly':
      if 'theta_v' not in additional_states:
        raise ValueError(
            '`theta_v` is required in the `additional_states` for the '
            'Smagorinsky-Lilly model.'
        )
      nu_t = self.smagorinsky_lilly(
          field_vars,
          field_vars,
          additional_states['theta_v'],
          self._params.smagorinsky_lilly.c_s,
          self._params.smagorinsky_lilly.pr_t,
          additional_states,
      )
    elif self._params.WhichOneof('sgs_model_type') == 'vreman':
      nu_t = self.vreman(field_vars, self._params.vreman.c_s, additional_states)
    else:
      raise ValueError(
          'Unsupported sub-grid scale model {}'.format(
              self._params.WhichOneof('sgs_model_type')
          )
      )

    nu_t = tf.nest.map_structure(
        lambda n: tf.math.maximum(n, self._params.nu_t_min), nu_t
    )
    return nu_t if self._nu_t_max is None else tf.nest.map_structure(
        lambda n: tf.math.minimum(n, self._nu_t_max), nu_t
    )

  def smagorinsky(
      self,
      field_vars: Sequence[FlowFieldVal],
      additional_states: FlowFieldMap,
      delta_formula: DeltaFormula,
      c_s_in: Optional[FlowFieldVal] = None,
  ) -> FlowFieldVal:
    """Computes the turbulent viscosity from the Smagorinsky model [1].

    The turbulent viscosity is computed as:
      νₜ = (cₛ Δ)²√(2 Sₖₗ Sₖₗ),
    where Sₖₗ is the strain rate tensor.

    [1] Smagorinsky, J., 1963. General circulation experiments with the
        primitive equations: I. The basic experiment. Monthly weather review,
         91(3), pp.99-164.

    Args:
      field_vars: Variables based on which the model is computed. The length of
        `field_vars` can be an arbitrary number. If the length of `field_vars`
        is three, it is assumed that `field_vars` is the velocity. If
        `field_vars` is treated as velocity, then the SGS term is computed based
        on strain rate Sₖₗ, which is a 3x3 tensor; otherwise `field_vars` will
        be treated as individual scalars, and the SGS term is computed based on
        the magnitude of the gradient vector.
      additional_states: A map of variables that might be needed by the SGS
        model requested.
      delta_formula: Which formulation to use for delta.
      c_s_in: The Smagorinsky constant.

    Returns:
      The turbulent viscosity.
    """
    if c_s_in is None:
      c_s = tf.nest.map_structure(
          lambda var: _CS_CLASSICAL * tf.ones_like(var),
          field_vars[0],
      )
    else:
      c_s = c_s_in

    nvar = len(field_vars)

    du_dx = calculus.grad(self._deriv_lib, field_vars, additional_states)

    # Treat `field_vars` as the velocity vector if the length of it is 3, where
    # strain rate is used instead of gradients.
    if nvar == 3:
      s_ij = _strain_rate_tensor(du_dx)
    else:
      s_ij = du_dx

    strain_rate_magnitude = _strain_rate_magnitude(s_ij)

    use_3d_tf_tensor = isinstance(field_vars[0], tf.Tensor)
    dx_dy_dz = tuple(
        self._swirllm_params.physical_grid_spacing(
            dim, use_3d_tf_tensor, additional_states
        )
        for dim in (0, 1, 2)
    )

    match delta_formula:
      case DeltaFormula.DIAGONAL:
        delta_square = common_ops.map_structure_3d(
            lambda a, b, c: a**2 + b**2 + c**2, *dx_dy_dz
        )
      case DeltaFormula.GEOMETRIC_MEAN:
        delta_square = common_ops.map_structure_3d(
            lambda a, b, c: (a * b * c) ** (2 / 3), *dx_dy_dz
        )
      case _:
        raise ValueError(
            f'Unhandled DeltaFormua enum {DeltaFormula.Name(delta_formula)}')

    return tf.nest.map_structure(
        lambda c_s_i, delta_square_, s_l: c_s_i**2 * delta_square_ * s_l,
        c_s,
        delta_square,
        strain_rate_magnitude,
    )

  def dynamic_smagorinsky(
      self,
      periodic_dims: Sequence[bool],
      replicas: np.ndarray,
      velocity: Sequence[FlowFieldVal],
      additional_states: FlowFieldMap,
      scalar: Optional[FlowFieldVal] = None,
  ) -> FlowFieldVal:
    """Computes the turbulent viscosity using the dynamic Smagorinsky model.

    The Smagorinsky constant is determined from the Germano dynamic procedure.

    Args:
      periodic_dims: A boolean list of length 3, with each element indicates if
        that dimension is periodic.
      replicas: The topology of the TPU cores.
      velocity: A 3 component list of 3D tensors representing the three velocity
        components.
      additional_states: A map of helper variables.
      scalar: A 3D tensor representing a transported scalar field. If `None`,
        turbulent viscosity is computed (SGS model for momentum); otherwise the
        SGS model is computed for the scalar.

    Returns:
      The turbulent viscosity if `scalar` is `None`, otherwise the diffusivity.
    """
    if any(self._swirllm_params.use_stretched_grid):
      raise NotImplementedError(
          'Stretched grid is not supported with dynamic smagorinsky sgs model.'
      )
    delta = self._swirllm_params.grid_spacings
    delta_square = np.sum(np.array(delta) ** 2)

    s_ij = _strain_rate_tensor(
        calculus.grad(self._deriv_lib, velocity, additional_states)
    )
    s = _strain_rate_magnitude(s_ij)

    velocity_filtered = [_test_filter(velocity_i) for velocity_i in velocity]
    s_ij_filtered = [
        [_test_filter(s_ij[i][j]) for j in range(3)] for i in range(3)
    ]
    s_filtered = _strain_rate_magnitude(s_ij_filtered)

    def lm_mm_momentum():
      """Compute the L_ij M_ij, M_ij M_ij in the dynamic model for momentum."""

      def resolved_shear_stress(i, j):
        """Computes the resolved shear stress L_ij."""
        t_ij = _test_filter(_dot(velocity[i], velocity[j]))
        tau_ij = _dot(velocity_filtered[i], velocity_filtered[j])
        return tf.nest.map_structure(tf.math.subtract, t_ij, tau_ij)

      def anisotropic_shear_stress(i, j):
        """Computes the anisotropic shear stress M_ij."""
        ss_filtered = _test_filter(_dot(s, s_ij[i][j]))
        ss_prod = _dot(s_filtered, s_ij_filtered[i][j])
        return tf.nest.map_structure(
            lambda ss_filtered_l, ss_prod_l: 2.0
            * delta_square
            * (ss_filtered_l - 4.0 * ss_prod_l),
            ss_filtered,
            ss_prod,
        )

      l_ij = [[resolved_shear_stress(i, j) for j in range(3)] for i in range(3)]
      m_ij = [
          [anisotropic_shear_stress(i, j) for j in range(3)] for i in range(3)
      ]

      return _einsum_ij(l_ij, m_ij), _einsum_ij(m_ij, m_ij)

    def lm_mm_scalar():
      """Compute the L_i M_i, M_i M_i in the dynamic model for scalar."""
      scalar_filtered = _test_filter(scalar)
      g_i = calculus.grad(self._deriv_lib, (scalar,), additional_states)[0]
      g_i_filtered = [_test_filter(g_i_l) for g_i_l in g_i]

      def resolved_scalar_stress(i):
        """Computes the resolved scalar stress L_ij."""
        t_i = _test_filter(_dot(velocity[i], scalar))
        tau_i = _dot(velocity_filtered[i], scalar_filtered)
        return tf.nest.map_structure(tf.math.subtract, t_i, tau_i)

      def anisotropic_scalar_stress(i):
        """Computes the anisotropic scalar stress M_ij."""
        ss_filtered = _test_filter(_dot(s, g_i[i]))
        ss_prod = _dot(s_filtered, g_i_filtered[i])
        return tf.nest.map_structure(
            lambda ss_filtered_l, ss_prod_l: 2.0
            * delta_square
            * (ss_filtered_l - 4.0 * ss_prod_l),
            ss_filtered,
            ss_prod,
        )

      l_ij = [[resolved_scalar_stress(i) for i in range(3)]]
      m_ij = [[anisotropic_scalar_stress(i) for i in range(3)]]

      return _einsum_ij(l_ij, m_ij), _einsum_ij(m_ij, m_ij)

    lm, mm = lm_mm_momentum() if scalar is None else lm_mm_scalar()

    # The 2 halos are invalid values and should not be included in the
    # average.
    def germano_avg_exclude_halos(m):
      """Computes the Germano average without using halos."""
      # This is the width consistent with the hard-coded filter width and the
      # gradient calculation. If flexible halo widths support is needed, the
      # best approach will be rewrite the class to incorporate all needed
      # information of the grid config.
      halo_width = 2
      m_inner = common_ops.strip_halos(m, (halo_width, halo_width, halo_width))  # pytype: disable=wrong-arg-types  # always-use-return-annotations
      m_avg_inner = _germano_averaging(m_inner, periodic_dims, replicas)
      if self._swirllm_params.use_3d_tf_tensor:
        m_avg_full = tf.pad(
            m_avg_inner, [[halo_width] * 2] * 3, constant_values=1.0
        )
      else:
        z_pad = [
            # These are in the halos and can be set to any values. Although they
            # might enter (temporarily) into the point-wise time advancement
            # step, the contribution will not accumulate as the values in the
            # halos will be reset/replaced every step. Here we choose 0 for
            # simplicity.
            tf.zeros_like(m_avg_inner[0]),
        ] * halo_width
        m_avg_inner = z_pad + m_avg_inner + z_pad
        m_avg_full = tf.nest.map_structure(
            lambda inner: tf.pad(  # pylint: disable=g-long-lambda
                inner,
                [[halo_width, halo_width], [halo_width, halo_width]],
                constant_values=1.0,
            ),
            m_avg_inner,
        )
      return m_avg_full

    lm_avg = germano_avg_exclude_halos(lm)
    mm_avg = germano_avg_exclude_halos(mm)

    c_s_square = tf.nest.map_structure(
        lambda lm_l, mm_l: tf.maximum(
            tf.math.divide_no_nan(lm_l, mm_l),
            tf.zeros_like(lm_l, dtype=lm_l.dtype),
        ),
        lm_avg,
        mm_avg,
    )

    return tf.nest.map_structure(
        lambda c_s_square_l, s_l: c_s_square_l * delta_square * s_l,
        c_s_square,
        s,
    )

  def smagorinsky_lilly(
      self,
      field_vars: Sequence[FlowFieldVal],
      velocity: Sequence[FlowFieldVal],
      temperature: FlowFieldVal,
      c_s: float,
      pr_t: float,
      additional_states: FlowFieldMap,
  ) -> FlowFieldVal:
    """Computes the turbulent viscosity from the Smagorinsky-Lilly model.

    Reference:
    Lilly, D. K. 1962. “On the Numerical Simulation of Buoyant Convection.”
    Tell’Us 14 (2): 148–72.

    Args:
      field_vars: Variables based on which the model is computed. The length of
        `field_vars` can be an arbitrary number. If the length of `field_vars`
        is three, it is assumed that `field_vars` is the velocity. If
        `field_vars` is treated as velocity, then the SGS term is computed based
        on strain rate Sₖₗ, which is a 3x3 tensor; otherwise `field_vars` will
        be treated as individual scalars, and the SGS term is computed based on
        the magnitude of the gradient vector.
      velocity: A list of 3D tensors representing the 3 velocity components.
        Required in the dynamic Smagorinsky model.
      temperature: A 3D tensor of the temperature field.
      c_s: The Smagorinsky constant.
      pr_t: The turbulent Prandtl number.
      additional_states: A map of helper variables.

    Returns:
      The turbulent viscosity/diffusivity.
    """

    def richardson_number():
      """Computes the Richardson number."""
      g_dim = 2  # Currently assuming the vertical dimension is dim 2.
      dt_dz = self._deriv_lib.deriv_centered(
          temperature, g_dim, additional_states
      )
      buoyancy_freq_square = tf.nest.map_structure(
          lambda t_i, dt_dz_i: constants.G / t_i * dt_dz_i, temperature, dt_dz
      )
      return tf.nest.map_structure(
          lambda n_square, s: tf.math.divide_no_nan(n_square, s**2),
          buoyancy_freq_square,
          strain_rate_magnitude,
      )

    def f_b():
      """Computes the stratification correction coefficient."""
      ri = richardson_number()
      return tf.nest.map_structure(
          lambda ri_i: tf.compat.v1.where(  # pylint: disable=g-long-lambda
              tf.less_equal(ri_i, 0.0),
              tf.ones_like(ri_i),
              tf.math.pow(tf.maximum(0.0, 1.0 - ri_i / pr_t), 0.25),
          ),
          ri,
      )

    nvar = len(field_vars)

    # Compute the magnitude of the gradient of `field_var`. Treat `field_vars`
    # as the velocity vector if the length of it is 3, where strain rate is used
    # instead of gradients.
    df_dx = calculus.grad(self._deriv_lib, field_vars, additional_states)
    if nvar == 3:
      df_ij = _strain_rate_tensor(df_dx)
    else:
      df_ij = df_dx
    df_magnitude = _strain_rate_magnitude(df_ij)

    # Compute the magnitude of strain rate. This variable is used to compute
    # the Lilly efficiency factor `f_b`.
    du_dx = calculus.grad(self._deriv_lib, velocity, additional_states)
    s_ij = _strain_rate_tensor(du_dx)
    strain_rate_magnitude = _strain_rate_magnitude(s_ij)

    use_3d_tf_tensor = isinstance(velocity[0], tf.Tensor)
    dx_dy_dz = tuple(
        self._swirllm_params.physical_grid_spacing(
            dim, use_3d_tf_tensor, additional_states
        )
        for dim in (0, 1, 2)
    )
    delta_updated = common_ops.map_structure_3d(
        lambda dx, dy, dz, fb: (dx * dy * dz) ** (1 / 3) * fb, *dx_dy_dz, f_b()
    )

    return tf.nest.map_structure(
        lambda delta_l, s_l: (c_s * delta_l) ** 2 * s_l,
        delta_updated,
        df_magnitude,
    )

  def vreman(
      self,
      velocity: Sequence[FlowFieldVal],
      c_s: float,
      additional_states: FlowFieldMap,
  ) -> FlowFieldVal:
    """Computes the turbulent viscosity from the Vreman model.

    Reference:
    Vreman, A. W. 2004. “An Eddy-Viscosity Subgrid-Scale Model for Turbulent
    Shear Flow: Algebraic Theory and Applications.” Physics of Fluids
    16 (10): 3670–81.

    Args:
      velocity: A list of list of 2D xy tensors representing the 3 3D velocity
        components.
      c_s: The Smagorinsky constant.
      additional_states: A map of helper variables.

    Returns:
      The turbulent viscosity.
    """
    if any(self._swirllm_params.use_stretched_grid):
      raise NotImplementedError(
          'Stretched grid is not supported with Vreman sgs model.'
      )
    delta = self._swirllm_params.grid_spacings
    alpha = calculus.grad(self._deriv_lib, velocity, additional_states)

    def beta(i: int, j: int) -> FlowFieldVal:
      """Computes the beta coefficient in the Vreman SGS model."""
      beta_ij = [
          tf.nest.map_structure(
              lambda a_mi, a_mj: delta[m] ** 2 * a_mi * a_mj,  # pylint: disable=cell-var-from-loop
              alpha[i][m],
              alpha[j][m],
          )
          for m in range(3)
      ]
      return tf.nest.map_structure(
          lambda a, b, c: a + b + c, beta_ij[0], beta_ij[1], beta_ij[2]
      )

    beta_terms = (
        beta(0, 0),
        beta(1, 1),
        beta(2, 2),
        beta(0, 1),
        beta(0, 2),
        beta(1, 2),
    )
    b_beta = tf.nest.map_structure(
        lambda b_11, b_22, b_33, b_12, b_13, b_23: b_11 * b_22
        - b_12**2
        + b_11 * b_33
        - b_13**2
        + b_22 * b_33
        - b_23**2,
        *beta_terms
    )

    alpha_terms = (
        alpha[0][0],
        alpha[0][1],
        alpha[0][2],
        alpha[1][0],
        alpha[1][1],
        alpha[1][2],
        alpha[2][0],
        alpha[2][1],
        alpha[2][2],
    )
    alpha_sq = tf.nest.map_structure(
        lambda a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32, a_33: a_11**2
        + a_12**2
        + a_13**2
        + a_21**2
        + a_22**2
        + a_23**2
        + a_31**2
        + a_32**2
        + a_33**2,
        *alpha_terms
    )

    c = 2.5 * c_s**2

    return tf.nest.map_structure(
        lambda b, a_sq: c
        * tf.math.sqrt(tf.maximum(tf.math.divide_no_nan(b, a_sq), 0.0)),
        b_beta,
        alpha_sq,
    )
