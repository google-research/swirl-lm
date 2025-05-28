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

"""A library of boundary conditions to be used in fluid simulations."""

from typing import List, Optional, Text, Tuple

import numpy as np
from swirl_lm.base import physical_variable_keys_manager
from swirl_lm.communication import halo_exchange
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf

FloatSequence = types.FloatSequence
IntSequence = types.IntSequence
FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap


_TF_DTYPE = types.TF_DTYPE
_MAX_SEED_NUM = 1024


def _gen_stateless_seed():
  """A utility function for the default seed for stateless random operations."""
  return tf.random.uniform(
      shape=[2], minval=0, maxval=65536, dtype=tf.dtypes.int32)


class SyntheticTurbulentInflow(object):
  """A library that generates inflow with synthetic turbulence.

  To use this library in a fluid simulation, the synthetic turbulent inflow
  needs to be generated at the beginning of each time step, and fed as Dirichlet
  boundary conditions to all velocity components at the specified boundary via
  `additional_states`.

  Reference:
  Klein, M., A. Sadiki, and J. Janicka. 2003. “A Digital Filter Based Generation
  of Inflow Data for Spatially Developing Direct Numerical or Large Eddy
  Simulations.” Journal of Computational Physics 186 (2): 652–65.

  To use this library, the following keys are required in `additional_states`:
    'bc_[u, v, w]_{inflow_dim}_{inflow_face}',
    'mean_[u, v, w]_{inflow_dim}_{inflow_face}',
    'rms_[u, v, w]_{inflow_dim}_{inflow_face}',
    'rand_[u, v, w]_{inflow_dim}_{inflow_face}',
  The shape of 'bc_[u, v, w]_{inflow_dim}_{inflow_face}' should be one of the
  following:
  * If the inflow is in dimension 0, [(halo_width + 1, ny)] * nz
  * If the inflow is in dimension 1, [(nx, halo_width + 1)] * nz
  * If the inflow is in dimension 2, [(nx, ny)] * (halo_width + 1)
  'mean_[u, v, w]_{inflow_dim}_{inflow_face}' and
  'rms_[u, v, w]_{inflow_dim}_{inflow_face}' are 2D tensors of the same size
  with the inflow plane.
  'rand_[u, v, w]_{inflow_dim}_{inflow_face}' are 3D tensors of which the shape
  depends on the filter width.
  """

  def __init__(
      self,
      length_scale: FloatSequence,
      delta: FloatSequence,
      mesh_size: IntSequence,
      inflow_dim: int,
      inflow_face: int,
  ):
    """Initializes fields and operators required for turbulence generation.

    Args:
      length_scale: Character length scales in three dimensions as a sequence
        of float in the order of x, y, z.
      delta: Grid sizes in three dimensions as a sequence of float in the order
        of x, y, z.
      mesh_size: Number of grid points in three dimensions as a sequence of
        float in the order of x, y, z.
      inflow_dim: The dimension along which the inflow is injected.
      inflow_face: The index of the face along `inflow_dim` where the inflow is
        imposed. 0 indicates the face with lower physical index, and 1 indicates
        the face with higher physical index.
    """
    # Construct a permutation sequence with the first element being the inflow
    # dimension and the remaining two being the dimension indices of the inflow
    # plane. The plane indices follows an ascending order.
    self.inflow_face = inflow_face
    self.inflow_dim = inflow_dim
    self.inflow_plane = [dim for dim in range(3) if dim != inflow_dim]
    inflow_permute = [
        inflow_dim,
    ] + self.inflow_plane

    # The widths of the digital filter.
    n = np.array([int(np.ceil(l / d)) for l, d in zip(length_scale, delta)
                 ])[inflow_permute]

    # Total number of points needs to be added to the compotational domain to
    # account for the filter width.
    self.n_pad = np.array([2 * n_i for n_i in n])

    # Number of grid points along the two dimensions in the inflow plane.
    self.m = np.array(mesh_size)[inflow_permute[1:]]

    # Total number of points required by the random fields.
    self.nr_total = [
        2 * self.n_pad[0], 2 * self.n_pad[1] + self.m[0],
        2 * self.n_pad[2] + self.m[1]
    ]

    # Initializes the weights of the filters.
    self.b = [
        self._compute_filter_weights(n_i, n_pad_i)
        for n_i, n_pad_i in zip(n, self.n_pad)
    ]

    # Generate boundary condition keys for the specified inflow.
    helper = physical_variable_keys_manager.BoundaryConditionKeysHelper()
    self._bc_keys = [
        helper.generate_bc_key(varname, inflow_dim, inflow_face)
        for varname in ['u', 'v', 'w']
    ]

    # Derive keys for other required fields.
    self._mean_keys = [
        self.helper_key('mean', vel, inflow_dim, inflow_face)
        for vel in ['u', 'v', 'w']
    ]
    self._rms_keys = [
        self.helper_key('rms', vel, inflow_dim, inflow_face)
        for vel in ['u', 'v', 'w']
    ]
    self._rand_keys = [
        self.helper_key('rand', vel, inflow_dim, inflow_face)
        for vel in ['u', 'v', 'w']
    ]
    self._required_keys = (
        self._bc_keys + self._mean_keys + self._rms_keys + self._rand_keys)

  def helper_key(
      self,
      helper_type: Text,
      velocity: Text,
      inflow_dim: int,
      inflow_face: int,
  ) -> Text:
    """Generates the a key of helper variable for the inflow generation.

    The key of the helper variable takes the following format:
      [helper_type]_[velocity]_[inflow_dim]_[inflow_face]

    Args:
      helper_type: The type of the helper variable. Should be one of 'bc',
        'mean', 'rms', 'rand'.
      velocity: The name of the velocity component. Should be one of 'u', 'v',
        'w'.
      inflow_dim: The inflow dimension. Should be one of 0, 1, or 2 indicates
        flow in the x, y, and z direction, respectively.
      inflow_face: The face at which the inflow is coming from. Should be either
        0 or 1, with 0 being the end of domain that has smaller indices, and 1
        being the end of domain with larger indices.

    Returns:
      The key of the helper variable.

    Raises:
      ValueError: If `helper_type` is not in ['mean', 'rms', 'rand'], or
        `velocity` is not in ['u', 'v', 'w'], or inflow_dim is not in [0, 1, 2],
        or `inflow_face` is not in [0, 1].
    """
    if helper_type not in ['bc', 'mean', 'rms', 'rand']:
      raise ValueError('`helper_type` has to be "bc", mean", "rms", or "rand".'
                       '{} is an invalid option.'.format(helper_type))

    if velocity not in ['u', 'v', 'w']:
      raise ValueError(
          '`velocity` has to be "u", "v", or "w". {} is an invalid option.'
          .format(velocity))

    if inflow_dim not in [0, 1, 2]:
      raise ValueError(
          '`inflow_dim` has to be 0, 1, or 2. {} is an invalid option.'.format(
              inflow_dim))

    if inflow_face not in [0, 1]:
      raise ValueError(
          '`inflow_face` has to be 0 or 1. {} is an invalid option.'.format(
              inflow_face))

    return '{}_{}_{}_{}'.format(helper_type, velocity, inflow_dim, inflow_face)

  def _compute_filter_weights(self, n: int, n_pad: int) -> List[float]:
    """Computes the filter weights for the turbulence flow field.

    Args:
      n: The number of mesh points required to capture the characteristic length
        scale.
      n_pad: The half width of the filter stencil.

    Returns:
      The weights of the filter with length being 2 * `n_pad`.
    """
    b_tilde = [np.exp(-np.pi * k**2 / n**2) for k in range(-n_pad, n_pad)]
    return b_tilde / np.sum(b_tilde)

  def _inflow_plane_to_bc(
      self,
      inflow: tf.Tensor,
      halo_width: int,
      use_3d_tf_tensor: bool = False,
  ) -> types.FlowFieldVal:
    """Arranges the inflow plane to the boundary condition format.

    Because the inflow planes in adjacent cores links directly with each other
    without considering the overlap, halo layers need to be added to form the
    boundary condition in the two dimensions perpendicular to the inflow
    dimension. Because these values in the halo layers are not used in any
    simulation, they are set to 0.

    Args:
      inflow: A 2D tensor that contains the inflow information.
      halo_width: The width of the halo layers.
      use_3d_tf_tensor: Returns the inflow plane as a 3D tf.Tensor if `True`,
        otherwise unstack the inflow plane along the z dimension, i.e. the 0th
        dimension following a z-x-y orientation.

    Returns:
      A list of 2D tensors representing a 3D structure, which will be used as
      the boundary condition.
    """
    plane = tf.tile(tf.expand_dims(inflow, 0), [halo_width + 1, 1, 1])
    plane = tf.pad(
        plane,
        paddings=[[0, 0], [halo_width, halo_width], [halo_width, halo_width]])
    if self.inflow_dim == 0:
      plane = tf.transpose(plane, perm=(2, 0, 1))
    elif self.inflow_dim == 1:
      plane = tf.transpose(plane, perm=(2, 1, 0))
    return plane if use_3d_tf_tensor else tf.unstack(plane, axis=0)

  def generate_random_fields(
      self,
      seed: Optional[Tuple[int, int]] = None,) -> List[tf.Tensor]:
    """Generates three random fields for the turbulence generation.

    Args:
      seed: A length 2 tuple of `int` as the seed for generating the
        corresponding random field. If not set, a random seed will be generated.

    Returns:
      A length 3 list of tensors with each component being a random field
      without halo update. The permutation of the tensor is: the list length is
    for the inflow dimension, the 2d tensors are in the rest 2 dimensions.
    """
    return [
        tf.random.stateless_normal(  # pylint:disable=g-complex-comprehension
            shape=self.nr_total, dtype=_TF_DTYPE,
            seed=seed if seed is not None else _gen_stateless_seed())
        for _ in range(3)
    ]

  def compute_inflow_velocity(
      self,
      r: List[tf.Tensor],
      velocity_mean: list[tf.Tensor],
      velocity_rms: list[tf.Tensor],
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      seed: Optional[Tuple[int, int]] = None,
  ) -> dict[str, list[tf.Tensor]]:
    """Computes the inflow velocity with synthetic turbulence.

    Args:
      r: A 3 element list of 3D tensors with each element being a random field.
      velocity_mean: The mean profile for velocity in three dimensions. Each
        velocity component is a 2D tensor that covers the inflow plane.
      velocity_rms: The rms profile for velocity in three dimensions. Each of
        the component is a 2D tensor that covers the inflow plane.
      replica_id: The index of the current TPU replica.
      replicas: The topography of the TPU replicas.
      seed: A length 2 tuple of `int` as the seed for generating the
        corresponding random field. If not set, a random seed will be generated.

    Returns:
      A dictionary that has two components: 'r' is the updated random fields for
      the next time step without halo update; 'u' is a list of 2D `tf.Tensor` of
      length 3, with its components being the velocity components in dimension
      0, 1, and 2, respectively.

    Raises:
      ValueError: If the shape of `r`, `velocity_mean`, or `velocity_rms` are
        not the same as `self.nr_total`, `self.m`, and `self.m`, respectively.
    """

    # Exchange the halos for the three 3D random fields.
    def halo_exchange_2d(value: tf.Tensor) -> tf.Tensor:
      """Performs halo exchange dimension by dimension."""

      # First transpose the first inflow plane dimension to be split into
      # z-list and perform the exchange for the `z-list` elements. After that,
      # transpose the second inflow plane dimension to be split into z-list,
      # and similarly perform halo exchange. This reduces the size of the graph
      # greatly and improves the compilation time significantly, from ~ 3000 sec
      # to ~ 100 sec. This change also seems to improve the run time
      # performance: for ground fire case, the step time improves from ~ 250 ms
      # to ~ 210 ms.
      for i in range(2):
        value = tf.transpose(value, [1, 2, 0])
        value = halo_exchange.inplace_halo_exchange(
            value,
            dims=[2],
            replica_id=replica_id,
            replicas=replicas,
            replica_dims=[self.inflow_plane[i]],
            boundary_conditions=[[
                (halo_exchange.BCType.NO_TOUCH, 0.0),
            ] * 2],
            width=self.n_pad[i + 1]
        )

      # Return the result in the original permutation of dimensions:
      # [inflow_dim, inflow_plane_dim_0, inflo_plane_dim_1].
      return tf.transpose(value, [1, 2, 0])

    def compute_u_alpha(r_alpha: tf.Tensor) -> tf.Tensor:
      """Computes the alpha component of u."""
      filter_dim_1 = np.sum(
          [
              np.eye(M=(self.m[1] + 2 * self.n_pad[2]),
                     N=self.m[1], k=i) * self.b[2][i]
              for i in range(2 * self.n_pad[2])], axis=0)
      filter_dim_0 = np.sum(
          [
              np.eye(M=(self.m[0] + 2 * self.n_pad[1]),
                     N=self.m[0], k=i) * self.b[1][i]
              for i in range(2 * self.n_pad[1])], axis=0)
      filter_inflow_dim = self.b[0]

      u_alpha_xy = tf.einsum('lk,ijk->lij', filter_dim_1, r_alpha)
      u_alpha_x = tf.einsum('lk,ijk->lij', filter_dim_0, u_alpha_xy)
      return tf.einsum('k,ijk->ij', filter_inflow_dim, u_alpha_x)

    # Check the shape of the input tensors.
    for i in range(3):
      if r[i].shape != self.nr_total:
        raise ValueError(
            'The shape of random field {} is not compatible. {} is given but {}'
            ' is requested.'.format(i, r[i].shape, self.nr_total))
      if velocity_mean[i].shape != self.m:
        raise ValueError(
            'The shape of velocity mean {} is not compatible. {} is given but'
            '{} is requested.'.format(i, velocity_mean[i].shape, self.m))
      if velocity_rms[i].shape != self.m:
        raise ValueError(
            'The shape of velocity rms {} is not compatible. {} is given but {}'
            ' is requested.'.format(i, velocity_rms[i].shape, self.m))

    r_halo_updated = [halo_exchange_2d(r_i) for r_i in r]
    u_alpha = [compute_u_alpha(r_alpha) for r_alpha in r_halo_updated]

    for i in range(3):
      r[i] = tf.concat(
          [
              r_halo_updated[i][1:, ...],
              tf.expand_dims(
                  tf.random.stateless_normal(
                      shape=[
                          self.n_pad[1] * 2 + self.m[0],
                          self.n_pad[2] * 2 + self.m[1]
                      ],
                      dtype=_TF_DTYPE,
                      seed=seed if seed is not None else _gen_stateless_seed()),
                  axis=0,
              )
          ],
          axis=0,
      )

    u = tf.nest.map_structure(
        lambda u_mean_i, u_rms_i, u_alpha_i: u_mean_i + u_rms_i * u_alpha_i,
        velocity_mean,
        velocity_rms,
        u_alpha,
    )

    return {'r': r, 'u': u}

  def generate_inflow_update_fn(self, seed: Optional[int] = None):
    """Generates an additional_state_update_fn that computes the inflow."""

    def additional_states_update_fn(
        kernel_op: get_kernel_fn.ApplyKernelOp,
        replica_id: tf.Tensor,
        replicas: np.ndarray,
        states: FlowFieldMap,
        additional_states: FlowFieldMap,
        params: grid_parametrization.GridParametrization,
    ) -> FlowFieldMap:
      """Updates the inflow boundary condition with synthetic turbulence."""
      del kernel_op

      use_3d_tf_tensor = isinstance(list(states.values())[0], tf.Tensor)

      for key in self._required_keys:
        if key not in additional_states.keys():
          raise ValueError('{} is required by the synthetic turbulent inflow '
                           'but was not found.'.format(key))

      inflow_info = self.compute_inflow_velocity(
          [additional_states[key] for key in self._rand_keys],
          [additional_states[key] for key in self._mean_keys],
          [additional_states[key] for key in self._rms_keys], replica_id,
          replicas, seed)

      additional_states_updated = {}
      for key, value in additional_states.items():
        if key in self._rand_keys:
          additional_states_updated.update(
              {key: inflow_info['r'][self._rand_keys.index(key)]})
        elif key in self._bc_keys:
          additional_states_updated.update({
              key: self._inflow_plane_to_bc(
                  inflow_info['u'][self._bc_keys.index(key)],
                  params.halo_width,
                  use_3d_tf_tensor,
              )
          })
        else:
          additional_states_updated.update({key: value})

      return additional_states_updated

    return additional_states_update_fn
