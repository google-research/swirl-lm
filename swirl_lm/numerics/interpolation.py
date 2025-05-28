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

"""A library for the interpolation schemes."""

import enum
import functools
import itertools
from typing import Sequence, Tuple, TypeAlias

from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal: TypeAlias = types.FlowFieldVal


class FluxLimiterType(enum.Enum):
  """Defines types of flux limiters."""

  VAN_LEER = 'van_leer'
  MUSCL = 'muscl'


def centered_node_to_face(
    v_node: FlowFieldVal,
    dim: int,
    kernel_op: get_kernel_fn.ApplyKernelOp,
) -> FlowFieldVal:
  """Performs centered 2nd-order interpolation from nodes to faces.

  * An array evaluated on nodes has index i <==> coordinate location x_i
  * An array evaluated on faces has index i <==> coordinate location x_{i-1/2}

  E.g., interpolating in dim 0:
    v_face[i, j, k] = 0.5 * (v_node[i, j, k] + v_node[i-1, j, k])

  Args:
    v_node: A 3D tensor, evaluated on nodes.
    dim: The dimension along with the interpolation is performed.
    kernel_op: Kernel operation library.

  Returns:
    A 3D tensor interpolated from `v_node`, which is evaluted on faces in
    dimension `dim`, and evaluated on nodes in other dimensions.
  """
  if dim == 0:
    sum_backward_v = kernel_op.apply_kernel_op_x(v_node, 'ksx')
  elif dim == 1:
    sum_backward_v = kernel_op.apply_kernel_op_y(v_node, 'ksy')
  elif dim == 2:
    sum_backward_v = kernel_op.apply_kernel_op_z(v_node, 'ksz', 'kszsh')
  else:
    raise ValueError(f'Unsupported dim: {dim}.  `dim` must be 0, 1, or 2.')
  v_face = tf.nest.map_structure(lambda x: 0.5 * x, sum_backward_v)
  return v_face


def _get_weno_kernel_op(
    k: int = 3,
) -> get_kernel_fn.ApplyKernelConvOp:
  """Initializes a convolutional kernel operator with WENO related weights.

  Args:
    k: The order/stencil width of the interpolation.

  Returns:
    A kernel of convolutional finite-difference operators.
  """
  # Coefficients for the interpolation and stencil selection.
  c = {
      2: {-1: [1.5, -0.5], 0: [0.5, 0.5], 1: [-0.5, 1.5]},
      3: {
          -1: [11.0 / 6.0, -7.0 / 6.0, 1.0 / 3.0],
          0: [1.0 / 3.0, 5.0 / 6.0, -1.0 / 6.0],
          1: [-1.0 / 6.0, 5.0 / 6.0, 1.0 / 3.0],
          2: [1.0 / 3.0, -7.0 / 6.0, 11.0 / 6.0],
      },
  }

  # Define the kernel operator with WENO customized weights.
  # Weights for the i + 1/2 face interpolation. Values are saved at i.
  kernel_lib = {f'c{r}': (c[k][r], r) for r in range(k)}
  # Weights for the i - 1/2 face interpolation. Values are saved at i.
  kernel_lib.update({f'cr{r}': (c[k][r - 1], r) for r in range(k)})
  # Weights for the smoothness measurement.
  if k == 2:  # WENO-3
    kernel_lib.update({
        'b0_0': ([1.0, -1.0], 0),
        'b1_0': ([1.0, -1.0], 1),
    })
  elif k == 3:  # WENO-5
    kernel_lib.update({
        'b0_0': ([1.0, -2.0, 1.0], 0),
        'b1_0': ([1.0, -2.0, 1.0], 1),
        'b2_0': ([1.0, -2.0, 1.0], 2),
        'b0_1': ([3.0, -4.0, 1.0], 0),
        'b1_1': ([1.0, 0.0, -1.0], 1),
        'b2_1': ([1.0, -4.0, 3.0], 2),
    })

  kernel_op = get_kernel_fn.ApplyKernelConvOp(4, kernel_lib)
  return kernel_op


def _calculate_weno_weights(
    v: types.FlowFieldVal,
    kernel_op: get_kernel_fn.ApplyKernelConvOp,
    dim: str,
    k: int = 3,
) -> Tuple[Sequence[types.FlowFieldVal], Sequence[types.FlowFieldVal]]:
  """Calculates the weights for WENO interpolation from cell centered values.

  Args:
    v: A 3D tensor to which the interpolation is performed.
    kernel_op: A kernel of convolutional finite-difference operators.
    dim: The dimension along with the interpolation is performed.
    k: The order/stencil width of the interpolation.

  Returns:
    A tuple of the weights for WENO interpolated values on the faces, with the
    first and second elements being the negative and postive weights at face i +
    1/2, respectively.
  """
  # A small constant that prevents division by zero when computing the weights
  # for stencil selection.
  eps = 1e-6

  # Linear coefficients for the interpolation using upwind
  d = {
      2: [2.0 / 3.0, 1.0 / 3.0],  # WENO-3
      3: [0.3, 0.6, 0.1],  # WENO-5
  }

  kernel_fn = {
      'x': lambda u, name: kernel_op.apply_kernel_op_x(u, f'{name}x'),
      'y': lambda u, name: kernel_op.apply_kernel_op_y(u, f'{name}y'),
      'z': lambda u, name: kernel_op.apply_kernel_op_z(
          u, f'{name}z', f'{name}zsh'  # pylint: disable=g-long-lambda
      ),
  }[dim]

  # Compute the smoothness measurement.
  if k == 2:  # WENO-3
    beta_fn = lambda f0: f0**2
    beta = [
        tf.nest.map_structure(beta_fn, kernel_fn(v, f'b{r}_0'))
        for r in range(k)
    ]
  elif k == 3:  # WENO-5
    beta_fn = lambda f0, f1: 13.0 / 12.0 * f0**2 + 0.25 * f1**2
    beta = [
        tf.nest.map_structure(
            beta_fn, kernel_fn(v, f'b{r}_0'), kernel_fn(v, f'b{r}_1')
        )
        for r in range(k)
    ]

  # Compute the WENO weights.
  w_neg_sum = tf.nest.map_structure(tf.zeros_like, beta[0])
  w_pos_sum = tf.nest.map_structure(tf.zeros_like, beta[0])

  alpha_fn = lambda beta, dr: dr / (eps + beta) ** 2
  w_neg = [
      tf.nest.map_structure(functools.partial(alpha_fn, dr=d[k][r]), beta[r])
      for r in range(k)
  ]
  w_pos = [
      tf.nest.map_structure(
          functools.partial(alpha_fn, dr=d[k][k - 1 - r]), beta[r]
      )
      for r in range(k)
  ]
  for r in range(k):
    w_neg_sum = tf.nest.map_structure(tf.math.add, w_neg_sum, w_neg[r])
    w_pos_sum = tf.nest.map_structure(tf.math.add, w_pos_sum, w_pos[r])

  for r in range(k):
    w_neg[r] = tf.nest.map_structure(tf.math.divide_no_nan, w_neg[r], w_neg_sum)
    w_pos[r] = tf.nest.map_structure(tf.math.divide_no_nan, w_pos[r], w_pos_sum)

  return w_neg, w_pos


def _reconstruct_weno_face_values(
    v: types.FlowFieldVal,
    kernel_op: get_kernel_fn.ApplyKernelConvOp,
    dim: str,
    k: int = 3,
) -> Tuple[Sequence[types.FlowFieldVal], Sequence[types.FlowFieldVal]]:
  """Computes the reconstructed face values from cell centered values.

  Args:
    v: A 3D tensor to which the interpolation is performed.
    kernel_op: A kernel of convolutional finite-difference operators.
    dim: The dimension along with the interpolation is performed.
    k: The order/stencil width of the interpolation.

  Returns:
    A tuple of the reconstructed face values for WENO interpolated values on the
    faces, with the first and second elements being the negative and postive
    weights at face i + 1/2, respectively.
  """
  kernel_fn = {
      'x': lambda u, name: kernel_op.apply_kernel_op_x(u, f'{name}x'),
      'y': lambda u, name: kernel_op.apply_kernel_op_y(u, f'{name}y'),
      'z': lambda u, name: kernel_op.apply_kernel_op_z(
          u, f'{name}z', f'{name}zsh'  # pylint: disable=g-long-lambda
      ),
  }[dim]

  # Compute the reconstructed values on faces.
  vr_neg = [kernel_fn(v, f'c{r}') for r in range(k)]
  vr_pos = [kernel_fn(v, f'cr{r}') for r in range(k)]

  return vr_neg, vr_pos


def _interpolate_with_weno_weights(
    v: types.FlowFieldVal,
    w_neg: Sequence[types.FlowFieldVal],
    w_pos: Sequence[types.FlowFieldVal],
    vr_neg: Sequence[types.FlowFieldVal],
    vr_pos: Sequence[types.FlowFieldVal],
    dim: str,
    k: int = 3,
) -> Tuple[types.FlowFieldVal, types.FlowFieldVal]:
  """Performs the WENO interpolation from cell centers to faces.

  Args:
    v: A 3D tensor to which the interpolation is performed.
    w_neg: A sequence of FlowFieldVal with weights for negative side of WENO
      interpolation.
    w_pos: A sequence of FlowFieldVal with weights for negative side of WENO
      interpolation.
    vr_neg: A sequence of FlowFieldVal with reconstructed face values for
      negative side of WENO interpolation.
    vr_pos: A sequence of FlowFieldVal with reconstructed face values for
      positive side of WENO interpolation.
    dim: The dimension along with the interpolation is performed.
    k: The order/stencil width of the interpolation.

  Returns:
    A tuple of the interpolated values on the faces, with the first and second
    elements being the negative and postive fluxes at face i + 1/2,
    respectively.
  """

  # Compute the weighted interpolated face values.
  v_neg = tf.nest.map_structure(tf.zeros_like, v)
  v_pos = tf.nest.map_structure(tf.zeros_like, v)
  prod_sum_fn = lambda s, w, u: s + w * u
  for r in range(k):
    v_neg = tf.nest.map_structure(prod_sum_fn, v_neg, w_neg[r], vr_neg[r])
    v_pos = tf.nest.map_structure(prod_sum_fn, v_pos, w_pos[r], vr_pos[r])

  # Shift the negative face flux on the i - 1/2 face that stored at i - 1 to i.
  # With this shift, both the positive and negative face flux at i - 1/2 will
  # be stored at location i. Values on the lower end of the negative flux Tensor
  # will be set to be the same as v in the first cell along `dim`.
  if dim == 'x':
    if isinstance(v, tf.Tensor):
      v_neg = tf.concat([v[:, :1, :], v_neg[:, :-1, :]], 1)
    else:  # v and v_neg are lists.

      def update_x0(v_n, v_0):
        res = tf.roll(v_n, 1, axis=0)
        return tf.tensor_scatter_nd_update(res, [[0]], [v_0[0, :]])

      v_neg = tf.nest.map_structure(update_x0, v_neg, v)
  elif dim == 'y':
    if isinstance(v, tf.Tensor):
      v_neg = tf.concat([v[:, :, :1], v_neg[:, :, :-1]], 2)
    else:  # v and v_neg are lists.

      def update_y0(v_n, v_0):
        res = tf.roll(v_n, 1, axis=1)
        res_t = tf.tensor_scatter_nd_update(
            tf.transpose(res), [[0]], [v_0[:, 0]]
        )
        return tf.transpose(res_t)

      v_neg = tf.nest.map_structure(update_y0, v_neg, v)
  else:  # dim == 'z':
    if isinstance(v, tf.Tensor):
      v_neg = tf.concat([v[:1, ...], v_neg[:-1, ...]], 0)
    else:  # v and v_neg are lists.
      v_neg = [v[0]] + v_neg[:-1]

  return v_neg, v_pos


def weno(
    v: types.FlowFieldVal,
    dim: str,
    k: int = 3,
) -> Tuple[types.FlowFieldVal, types.FlowFieldVal]:
  """Performs the WENO interpolation from cell centers to faces.

  Args:
    v: A 3D tensor to which the interpolation is performed.
    dim: The dimension along which the interpolation is performed.
    k: The order/stencil width of the interpolation.

  Returns:
    A tuple of the interpolated values on the faces, with the first and second
    elements being the negative and postive fluxes at face i - 1/2,
    respectively.
  """
  kernel_op = _get_weno_kernel_op(k)
  w_neg, w_pos = _calculate_weno_weights(v, kernel_op, dim, k)
  vr_neg, vr_pos = _reconstruct_weno_face_values(v, kernel_op, dim, k)
  v_neg, v_pos = _interpolate_with_weno_weights(
      v, w_neg, w_pos, vr_neg, vr_pos, dim, k
  )

  return v_neg, v_pos


def flux_limiter(
    v: types.FlowFieldVal,
    dim: str,
    limiter_type: FluxLimiterType,
) -> tuple[FlowFieldVal, FlowFieldVal]:
  """Performs interpolation using a limiter scheme.

  Computes the interpolation of the field variable to the faces using 2nd-order
  Upwind + a flux limiter.

  Args:
    v: A 3D tensor to which the interpolation is performed.
    dim: The dimension along which the interpolation is performed.
    limiter_type: The type of flux limiter.

  Returns:
    A tuple that represents the interpolated values of a variable onto faces
    that are normal to `dim`, with the first element being interpolated from the
    upwind/left-biased stencil and the second from the downwind/right-biased
    stencil.

  Raises:
    NotImplementedError if `limiter_type` is not one of the follwoing:
      'VAN_LEER', 'MUSCL'.
    ValueError if `dim` is not one of 'x', 'y', and 'z'.
  """

  def van_leer(r: FlowFieldVal) -> FlowFieldVal:
    """Computes the Van Leer flux limiter."""
    van_leer_fn = lambda x: 2 * x / (1 + x)

    return tf.nest.map_structure(
        lambda r_: tf.where(tf.greater_equal(r_, 0.0), van_leer_fn(r_), 0.0), r
    )

  def muscl(r: FlowFieldVal) -> FlowFieldVal:
    """Computes the MUSCL flux limiter."""
    min_arg_1 = tf.nest.map_structure(lambda r_: 2.0 * r_, r)
    min_arg_2 = tf.nest.map_structure(lambda r_: 0.5 * (1.0 + r_), r)
    min_arg = tf.nest.map_structure(tf.math.minimum, min_arg_1, min_arg_2)
    return tf.nest.map_structure(
        lambda min_arg_: tf.math.maximum(
            tf.zeros_like(min_arg_),
            tf.math.minimum(2.0 * tf.ones_like(min_arg_), min_arg_),
        ),
        min_arg,
    )

  if limiter_type == FluxLimiterType.VAN_LEER:
    limiter = van_leer
  elif limiter_type == FluxLimiterType.MUSCL:
    limiter = muscl
  else:
    raise NotImplementedError(
        f'{limiter_type.name} is not supported. Available flux limiters are:'
        f' {list(FluxLimiterType)}'
    )

  kernel_op = get_kernel_fn.ApplyKernelConvOp(
      4,
      {
          'shift': ([1.0, 0.0, 0.0], 1),
      },
  )

  if dim == 'x':
    shift_op = ['shiftx']
    backward_diff = ['kdx']
    forward_diff = ['kdx+']
    kernel_fn = kernel_op.apply_kernel_op_x
  elif dim == 'y':
    shift_op = ['shifty']
    backward_diff = ['kdy']
    forward_diff = ['kdy+']
    kernel_fn = kernel_op.apply_kernel_op_y
  elif dim == 'z':
    shift_op = ['shiftz', 'shiftzsh']
    backward_diff = ['kdz', 'kdzsh']
    forward_diff = ['kdz+', 'kdz+sh']
    kernel_fn = kernel_op.apply_kernel_op_z
  else:
    raise ValueError(f'`dim` has to be "x", "y", or "z". {dim} is provided.')

  # Define functions
  v_shifted = kernel_fn(v, *shift_op)  # v[i - 1]
  diff1 = kernel_fn(v, *forward_diff)  # v[i + 1] - v[i]
  diff2 = kernel_fn(v, *backward_diff)  # v[i] - v[i - 1]
  diff3 = kernel_fn(diff2, *shift_op)  # v[i - 1] - v[i - 2]

  # Compute r = (state_D - state_U) / (state_U - state_UU).
  r_pos = tf.nest.map_structure(tf.math.divide_no_nan, diff2, diff3)
  r_neg = tf.nest.map_structure(tf.math.divide_no_nan, diff2, diff1)

  psi_pos = limiter(r_pos)
  psi_neg = limiter(r_neg)

  v_neg = tf.nest.map_structure(
      lambda v_upstream, psi_, diff_: v_upstream + 0.5 * psi_ * diff_,
      v_shifted,
      psi_pos,
      diff3,
  )
  v_pos = tf.nest.map_structure(
      lambda v_upstream, psi_, diff_: v_upstream - 0.5 * psi_ * diff_,
      v,
      psi_neg,
      diff1,
  )
  return v_neg, v_pos


def trilinear_interpolation(
    field_data: tf.Tensor,
    points: tf.Tensor,
    grid_spacing: tf.Tensor,
    local_grid_min_pt: tf.Tensor,
    fill_value: float = 0.0,
) -> tf.Tensor:
  """Linear interpolation on a 3-D orthogonal, uniform grid with halos.

  Performs trilinear interpolation by determining the nearest surrounding field
  data to the points and interpolating between them. The locations of the
  field data are determined using `grid_spacing` and the shape of `field_data`.
  Points provided outside of the domain return `fill_value`, which is `0.0` by
  default. `field_data` should include either halos with size either 1 or 2.

  Note that coordinates in this function follow the order of the dimensions of
  `field_data` as a tensor instead of the physical-coordinates orientation in
  Swirl-LM. For instance, the first element in `grid_spacing` and
  `domain_min_pt`, as well as the first column in `points`, are associated with
  the 0th dimensions of `field_data`, instead of the `x` axis in Swirl-LM.

  Args:
    field_data: A 3D or 4D tensor of field data including halos. The field data
      can be a scalar or a vector of size `m` corresponding to its last
      dimension.
    points: An 2D tensor (n, 3) of n coordinate points in 3D space to
      interpolate at.
    grid_spacing: A three element tensor defining the grid spacing along the
      three dimensions. The ordering should follow the `field_data` shape. If
      `field_data` indexing is in order z, x, y, then `grid_spacing` should be
      in order z, x, y.
    local_grid_min_pt: A three element tensor defining the location of the  `0,
      0, 0` coordinate in the `field_data` tensor. The dimension ordering should
      follow that of `grid_spacing`.
    fill_value: A scalar value to fill the points outside of the domain.

  Returns:
    A tensor containing interpolated data values at the `n` supplied points.
    The output shape is a vector of length `n` for a 3-D `field_data` and tensor
    of shape `(n, m)` for 4-D `field_data`, respectively.
  """
  with tf.name_scope('preparing_trilinear_interpolation'):
    points_norm = (points - local_grid_min_pt) / grid_spacing
    ijk_unclipped = tf.floor(points_norm)
    with tf.name_scope('preventing_error_for_out_of_bounds_indexing'):
      ijk = tf.clip_by_value(
          ijk_unclipped,
          0.0,
          # The -2 is appropriate and prevents out of bounds indexing in the
          # gather_nd operation below.
          tf.cast(tf.shape(field_data)[:3] - 2, tf.float32),
      )
    points_norm -= ijk
    ijk = tf.cast(ijk, dtype=tf.int32)

    i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
    x0, x1, x2 = points_norm[:, 0], points_norm[:, 1], points_norm[:, 2]

  with tf.name_scope('indexing_field_data_for_interpolation'):
    v = tf.stack(
        [
            tf.gather_nd(field_data, tf.stack([i + p, j + q, k + l], axis=-1))
            for p, q, l in itertools.product(range(2), range(2), range(2))
        ],
        axis=-1,
    )

  with tf.name_scope('calculating_interpolation_weights'):
    w = tf.stack(
        [
            ((1 - p) + (2 * p - 1) * x0)
            * ((1 - q) + (2 * q - 1) * x1)
            * ((1 - l) + (2 * l - 1) * x2)
            for p, q, l in itertools.product(range(2), range(2), range(2))
        ],
        axis=-1,
    )

  with tf.name_scope('applying_interpolation_weights'):
    vals = tf.einsum('i...k,ik->i...', v, w)

  with tf.name_scope('filling_extrapolating_values'):
    out_of_bounds_locs = tf.math.reduce_any(
        tf.math.logical_or(
            tf.math.less(ijk_unclipped, 0),
            tf.math.greater_equal(
                ijk_unclipped,
                tf.cast(tf.shape(field_data)[:3], tf.float32) - 1,
            ),
        ),
        axis=1,
    )
    vals = tf.where(
        tf.tile(out_of_bounds_locs[:, tf.newaxis], (1, tf.shape(vals)[1]))
        if len(tf.shape(vals)) > 1
        else out_of_bounds_locs,
        tf.fill(tf.shape(vals), tf.constant(fill_value, dtype=vals.dtype)),
        vals,
    )

  return vals
