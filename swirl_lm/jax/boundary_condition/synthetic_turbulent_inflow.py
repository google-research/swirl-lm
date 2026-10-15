# Copyright 2026 The swirl_lm Authors.
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

# Copyright 2021 Google LLC
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
"""A library for synthetic turbulent inflow boundary conditions (JAX port).

This is the JAX port of
`swirl_lm.boundary_condition.synthetic_turbulent_inflow`.

Generates inflow with synthetic turbulence using the digital filter method of
Klein, Sadiki, and Janicka (2003).

Reference:
Klein, M., A. Sadiki, and J. Janicka. 2003. "A Digital Filter Based Generation
of Inflow Data for Spatially Developing Direct Numerical or Large Eddy
Simulations." Journal of Computational Physics 186 (2): 652-65.
"""


from collections.abc import Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
from jax.sharding import NamedSharding  # pylint: disable=g-importing-member
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
import numpy as np
from swirl_lm.jax.base import physical_variable_keys_manager
from swirl_lm.jax.boundary_condition import boundary_condition_utils
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap


def _resolve_prng_key(
    key: jax.Array | None,
    seed: tuple[int, int] | None,
) -> jax.Array:
  """Resolves the `key`/`seed` pair this module accepts into a JAX PRNG key.

  `key` takes precedence when both are given. `seed` is the older
  two-integer form, kept for callers that want a deterministic stream without
  building a key themselves; the two integers are folded together so each
  contributes independently.

  Args:
    key: A JAX PRNG key, or None.
    seed: A two-integer seed, or None.

  Returns:
    A JAX PRNG key.
  """
  if key is not None:
    return key
  if seed is not None:
    return jax.random.fold_in(jax.random.PRNGKey(seed[0]), seed[1])
  return jax.random.PRNGKey(0)


class SyntheticTurbulentInflow:
  """A library that generates inflow with synthetic turbulence.

  To use this library in a fluid simulation, the synthetic turbulent inflow
  needs to be generated at the beginning of each time step, and fed as Dirichlet
  boundary conditions to all velocity components at the specified boundary via
  `additional_states`.

  To use this library, the following keys are required in `additional_states`:
    'bc_[u, v, w]_{inflow_dim}_{inflow_face}',
    'mean_[u, v, w]_{inflow_dim}_{inflow_face}',
    'rms_[u, v, w]_{inflow_dim}_{inflow_face}',
    'rand_[u, v, w]_{inflow_dim}_{inflow_face}',
  """

  def __init__(
      self,
      length_scale: Sequence[float],
      delta: Sequence[float],
      mesh_size: Sequence[int],
      inflow_dim: int,
      inflow_face: int,
  ):
    """Initializes fields and operators required for turbulence generation.

    Args:
      length_scale: Character length scales in three dimensions (x, y, z).
      delta: Grid sizes in three dimensions (x, y, z).
      mesh_size: Number of grid points in three dimensions (x, y, z).
      inflow_dim: The dimension along which the inflow is injected.
      inflow_face: The index of the face along `inflow_dim` where the inflow is
        imposed. 0 indicates the face with lower physical index, and 1 indicates
        the face with higher physical index.
    """
    self.inflow_face = inflow_face
    self.inflow_dim = inflow_dim
    self.inflow_plane = [dim for dim in range(3) if dim != inflow_dim]
    inflow_permute = [inflow_dim] + self.inflow_plane

    # The widths of the digital filter.
    n = np.array([int(np.ceil(l / d)) for l, d in zip(length_scale, delta)])[
        inflow_permute
    ]

    # Total number of points needs to be added for the filter width.
    self.n_pad = np.array([2 * n_i for n_i in n])

    # Number of grid points along the two dimensions in the inflow plane.
    self.m = np.array(mesh_size)[inflow_permute[1:]]

    # Total number of points required by the random fields.
    self.nr_total = [
        2 * self.n_pad[0],
        2 * self.n_pad[1] + self.m[0],
        2 * self.n_pad[2] + self.m[1],
    ]

    # The random fields carry `n_pad` ghost points on each side of each
    # in-plane axis, and the cross-device exchange fills them from the
    # neighbour's interior by reading a `n_pad`-wide slab of it. That slab only
    # exists if each core's in-plane extent is at least `n_pad`; otherwise the
    # exchange silently reads the neighbour's own ghost region and the filtered
    # turbulence is corrupted in a band at every device boundary.
    for i, (pad, extent) in enumerate(zip(self.n_pad[1:], self.m)):
      if pad > extent:
        dim = self.inflow_plane[i]
        raise ValueError(
            f'The synthetic inflow needs at least {pad} grid points per core '
            f'along dimension {dim} (the filter half width derived from '
            f'`length_scale`), but the core has only {extent}. Either reduce '
            f'`length_scale[{dim}]`, coarsen the grid, or use fewer cores '
            f'along that dimension.'
        )

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
        self._bc_keys + self._mean_keys + self._rms_keys + self._rand_keys
    )

  def helper_key(
      self,
      helper_type: str,
      velocity: str,
      inflow_dim: int,
      inflow_face: int,
  ) -> str:
    """Generates the key for a helper variable for the inflow generation.

    The key of the helper variable takes the following format:
      [helper_type]_[velocity]_[inflow_dim]_[inflow_face]

    Args:
      helper_type: The type of the helper variable. Should be one of 'bc',
        'mean', 'rms', 'rand'.
      velocity: The name of the velocity component. Should be one of 'u', 'v',
        'w'.
      inflow_dim: The inflow dimension. Should be one of 0, 1, or 2.
      inflow_face: The face at which the inflow is coming from. 0 or 1.

    Returns:
      The key of the helper variable.

    Raises:
      ValueError: If any argument is out of range.
    """
    if helper_type not in ['bc', 'mean', 'rms', 'rand']:
      raise ValueError(
          '`helper_type` must be "bc", "mean", "rms", or "rand". '
          f'{helper_type} is invalid.'
      )
    if velocity not in ['u', 'v', 'w']:
      raise ValueError(
          f'`velocity` must be "u", "v", or "w". {velocity} is invalid.'
      )
    if inflow_dim not in [0, 1, 2]:
      raise ValueError(
          f'`inflow_dim` must be 0, 1, or 2. {inflow_dim} is invalid.'
      )
    if inflow_face not in [0, 1]:
      raise ValueError(
          f'`inflow_face` must be 0 or 1. {inflow_face} is invalid.'
      )
    return f'{helper_type}_{velocity}_{inflow_dim}_{inflow_face}'

  def partition_specs(self) -> dict[str, jax.sharding.PartitionSpec]:
    """Returns how this model's 2D helper states map onto the device mesh.

    `mean_*` and `rms_*` live on the inflow plane and are built per replica,
    so they are genuinely different on every device whose subdomain covers a
    different part of that plane. The driver cannot infer that -- a 2D state
    with no declared spec can only be replicated -- so register the result of
    this method in `params.additional_state_partition_specs`:

      params.additional_state_partition_specs.update(
          inflow.partition_specs())

    Without it, the driver has to assume every replica produced the same
    plane; it now raises instead of silently keeping replica 0's copy, which
    previously fed the wrong mean velocity profile to every device except the
    first.

    Returns:
      A dict mapping each 2D helper state name to its `PartitionSpec`.
    """
    # The plane's two axes are the physical dimensions other than the inflow
    # normal, in ascending order -- the same ordering `self.m` uses. Mesh axis
    # names are the physical dimension names ('x', 'y', 'z') whatever
    # `data_axis_order` is, so the spec simply names those two axes and does
    # not depend on the grid parametrization. Compare
    # `boundary_condition_utils.get_bc_partition_spec`, which derives the same
    # thing for the rank-3 boundary condition arrays, where the normal axis is
    # additionally present and unpartitioned.
    plane_axis_names = tuple(
        ('x', 'y', 'z')[dim] for dim in self.inflow_plane
    )
    spec = jax.sharding.PartitionSpec(*plane_axis_names)
    return {key: spec for key in self._mean_keys + self._rms_keys}

  def _compute_filter_weights(self, n: int, n_pad: int) -> list[float]:
    """Computes the filter weights for the turbulence flow field.

    Args:
      n: The number of mesh points required to capture the characteristic length
        scale.
      n_pad: The half width of the filter stencil.

    Returns:
      The weights of the filter with length being 2 * `n_pad`.
    """
    b_tilde = [np.exp(-np.pi * k**2 / n**2) for k in range(-n_pad, n_pad)]
    return (b_tilde / np.sum(b_tilde)).tolist()

  def _inflow_plane_to_bc(
      self,
      inflow: ScalarField,
      halo_width: int,
      grid_params: 'grid_parametrization.GridParametrization | None' = None,
      pad_mode: str = 'edge',
  ) -> ScalarField:
    """Arranges the inflow plane to the boundary condition format.

    Args:
      inflow: A 2D array that contains the inflow information.
      halo_width: The width of the halo layers.
      grid_params: Grid parametrization for data_axis_order-aware transposition.
        If None, assumes data_axis_order=('z','x','y').
      pad_mode: Padding mode for in-plane boundary extension.

    Returns:
      A 3D array for the boundary condition.
    """
    return boundary_condition_utils.boundary_plane_to_bc(
        inflow, self.inflow_dim, halo_width, grid_params, pad_mode=pad_mode
    )

  def generate_random_fields(
      self,
      key: jax.Array | None = None,
      seed: tuple[int, int] | None = None,
  ) -> list[ScalarField]:
    """Generates three random fields for the turbulence generation.

    The three fields are drawn from independent sub-keys, so the `u`, `v` and
    `w` fluctuations are uncorrelated.

    Args:
      key: A JAX PRNG key. Takes precedence over `seed`.
      seed: An optional two-integer seed, used when `key` is None.

    Returns:
      A length 3 list of 3D arrays, each being a random field.
    """
    keys = jax.random.split(_resolve_prng_key(key, seed), 3)
    return [jax.random.normal(keys[i], shape=self.nr_total) for i in range(3)]

  def _halo_exchange_1d(
      self,
      val: ScalarField,
      axis_dim: int,
      axis_name: str,
      width: int,
      inflow_plane_idx: int,
      replica_id: Any | None = None,
      replicas: np.ndarray | None = None,
  ) -> ScalarField:
    """Performs 1D halo exchange with NO_TOUCH boundary condition."""
    try:
      core_id = jax.lax.axis_index(axis_name)
      num_cores = jax.lax.axis_size(axis_name)
    except NameError:
      if replicas is not None and replica_id is not None:
        replica_dim = self.inflow_plane[inflow_plane_idx]
        pos = np.where(replicas == int(replica_id))
        core_id = int(pos[replica_dim][0])
        num_cores = int(replicas.shape[replica_dim])
      else:
        return val

    if num_cores <= 1:
      return val

    perm_up = [(j, (j + 1) % num_cores) for j in range(num_cores)]
    perm_down = [(j, (j - 1) % num_cores) for j in range(num_cores)]

    if axis_dim == 1:
      low_halo = val[:, :width, :]
      low_interior = val[:, width : 2 * width, :]
      middle = val[:, width:-width, :]
      high_interior = val[:, -2 * width : -width, :]
      high_halo = val[:, -width:, :]

      try:
        low_halo_received = jax.lax.ppermute(
            high_interior, axis_name, perm=perm_up
        )
        high_halo_received = jax.lax.ppermute(
            low_interior, axis_name, perm=perm_down
        )
      except NameError:
        return val

      new_low_halo = jnp.where(core_id == 0, low_halo, low_halo_received)
      new_high_halo = jnp.where(
          core_id == num_cores - 1, high_halo, high_halo_received
      )
      return jnp.concatenate([new_low_halo, middle, new_high_halo], axis=1)
    else:  # axis_dim == 2
      low_halo = val[:, :, :width]
      low_interior = val[:, :, width : 2 * width]
      middle = val[:, :, width:-width]
      high_interior = val[:, :, -2 * width : -width]
      high_halo = val[:, :, -width:]

      try:
        low_halo_received = jax.lax.ppermute(
            high_interior, axis_name, perm=perm_up
        )
        high_halo_received = jax.lax.ppermute(
            low_interior, axis_name, perm=perm_down
        )
      except NameError:
        return val

      new_low_halo = jnp.where(core_id == 0, low_halo, low_halo_received)
      new_high_halo = jnp.where(
          core_id == num_cores - 1, high_halo, high_halo_received
      )
      return jnp.concatenate([new_low_halo, middle, new_high_halo], axis=2)

  def compute_inflow_velocity(
      self,
      r: list[ScalarField],
      velocity_mean: list[ScalarField],
      velocity_rms: list[ScalarField],
      key: jax.Array | None = None,
      replica_id: Any | None = None,
      replicas: np.ndarray | None = None,
      seed: tuple[int, int] | None = None,
  ) -> dict[str, list[ScalarField]]:
    """Computes the inflow velocity with synthetic turbulence."""

    def halo_exchange_2d(value: ScalarField) -> ScalarField:
      """Performs halo exchange along both inflow plane dimensions."""
      val = value
      for i in range(2):
        dim = i + 1
        width = int(self.n_pad[i + 1])
        axis_name = ('x', 'y', 'z')[self.inflow_plane[i]]
        val = self._halo_exchange_1d(
            val,
            axis_dim=dim,
            axis_name=axis_name,
            width=width,
            inflow_plane_idx=i,
            replica_id=replica_id,
            replicas=replicas,
        )
      return val

    def compute_u_alpha(r_alpha: ScalarField) -> ScalarField:
      """Computes the alpha component of u via digital filtering."""
      filter_dim_1 = np.sum(
          [
              np.eye(
                  M=(self.m[1] + 2 * self.n_pad[2]),
                  N=self.m[1],
                  k=i,
              )
              * self.b[2][i]
              for i in range(2 * self.n_pad[2])
          ],
          axis=0,
      )
      filter_dim_0 = np.sum(
          [
              np.eye(
                  M=(self.m[0] + 2 * self.n_pad[1]),
                  N=self.m[0],
                  k=i,
              )
              * self.b[1][i]
              for i in range(2 * self.n_pad[1])
          ],
          axis=0,
      )
      filter_inflow_dim = jnp.array(self.b[0])

      u_alpha_xy = jnp.einsum('lk,ijk->lij', filter_dim_1, r_alpha)
      u_alpha_x = jnp.einsum('lk,ijk->lij', filter_dim_0, u_alpha_xy)
      return jnp.einsum('k,ijk->ij', filter_inflow_dim, u_alpha_x)

    # Check the shape of the input tensors.
    for i in range(3):
      if list(r[i].shape) != list(self.nr_total):
        raise ValueError(
            f'The shape of random field {i} is not compatible. '
            f'{r[i].shape} is given but {self.nr_total} is requested.'
        )
      if list(velocity_mean[i].shape) != list(self.m):
        raise ValueError(
            f'The shape of velocity mean {i} is not compatible. '
            f'{velocity_mean[i].shape} is given but {self.m} is requested.'
        )
      if list(velocity_rms[i].shape) != list(self.m):
        raise ValueError(
            f'The shape of velocity rms {i} is not compatible. '
            f'{velocity_rms[i].shape} is given but {self.m} is requested.'
        )

    r_halo_updated = [halo_exchange_2d(r_i) for r_i in r]
    u_alpha = [compute_u_alpha(r_alpha) for r_alpha in r_halo_updated]

    # The leading plane of each random field is replaced by a fresh draw every
    # step; this is what advances the turbulence in time.
    keys = jax.random.split(_resolve_prng_key(key, seed), 3)
    new_slice_shape = [
        self.n_pad[1] * 2 + self.m[0],
        self.n_pad[2] * 2 + self.m[1],
    ]
    new_slices = [
        jax.random.normal(keys[i], shape=new_slice_shape) for i in range(3)
    ]

    r_new = [
        jnp.concatenate(
            [
                r_halo_updated[i][1:, ...],
                jnp.expand_dims(new_slices[i], axis=0),
            ],
            axis=0,
        )
        for i in range(3)
    ]

    # u = mean + rms * u_alpha
    u = [velocity_mean[i] + velocity_rms[i] * u_alpha[i] for i in range(3)]

    return {'r': r_new, 'u': u}

  def generate_inflow_update_fn(
      self,
      key: Optional[jax.Array] = None,
      seed: tuple[int, int] | None = None,
      params: Any | None = None,
      grid_params: grid_parametrization.GridParametrization | None = None,
      halo_width: int | None = None,
  ):
    """Generates an additional_states update function for inflow.

    Args:
      key: An optional JAX PRNG key. If None, a default key is created.
      seed: Optional (int, int) tuple seed.
      params: Optional simulation parameters or GridParametrization.
      grid_params: Optional grid parametrization object.
      halo_width: Optional halo width.

    Returns:
      A function that updates additional_states with inflow BCs.
    """
    default_params = params
    default_grid_params = (
        grid_params  # pylint: disable=g-long-ternary
        if grid_params is not None
        else (
            params.grid_params  # pylint: disable=g-long-ternary
            if hasattr(params, 'grid_params')
            else (
                params
                if isinstance(params, grid_parametrization.GridParametrization)
                else None
            )
        )
    )
    default_halo_width = (
        halo_width
        if halo_width is not None
        else (params.halo_width if hasattr(params, 'halo_width') else None)
    )

    def additional_states_update_fn(
        *args,
        **kwargs,
    ) -> dict[str, ScalarField]:
      """Updates the inflow boundary condition with synthetic turbulence."""
      if len(args) >= 5:
        replica_id = args[1]
        replicas = args[2]
        additional_states = args[4]
        call_params = args[5] if len(args) > 5 else None
      elif len(args) == 3:
        replica_id = None
        replicas = None
        additional_states = args[1]
        call_params = args[2]
      elif len(args) == 2:
        replica_id = None
        replicas = None
        additional_states = args[1]
        call_params = None
      else:
        replica_id = kwargs.get('replica_id', None)
        replicas = kwargs.get('replicas', None)
        additional_states = kwargs.get('additional_states', None)
        call_params = kwargs.get('params', None)

      if additional_states is None:
        raise ValueError('`additional_states` must be provided.')

      for req_key in self._required_keys:
        if req_key not in additional_states:
          raise ValueError(
              f'{req_key} is required by the synthetic turbulent inflow '
              'but was not found.'
          )

      step_id = kwargs.get('step_id', None)
      current_key = key
      if key is not None and step_id is not None:
        current_key = jax.random.fold_in(key, step_id)

      inflow_info = self.compute_inflow_velocity(
          [additional_states[k] for k in self._rand_keys],
          [additional_states[k] for k in self._mean_keys],
          [additional_states[k] for k in self._rms_keys],
          key=current_key,
          replica_id=replica_id,
          replicas=replicas,
          seed=seed,
      )

      active_params = call_params if call_params is not None else default_params
      resolved_hw = (
          active_params.halo_width
          if hasattr(active_params, 'halo_width')
          else (default_halo_width if default_halo_width is not None else 2)
      )
      resolved_gp = (
          active_params.grid_params  # pylint: disable=g-long-ternary
          if hasattr(active_params, 'grid_params')
          else (
              active_params  # pylint: disable=g-long-ternary
              if isinstance(
                  active_params, grid_parametrization.GridParametrization
              )
              else default_grid_params
          )
      )

      additional_states_updated: dict[str, ScalarField] = {}
      for as_key, value in additional_states.items():
        if as_key in self._rand_keys:
          additional_states_updated[as_key] = inflow_info['r'][
              self._rand_keys.index(as_key)
          ]
        elif as_key in self._bc_keys:
          additional_states_updated[as_key] = self._inflow_plane_to_bc(
              inflow_info['u'][self._bc_keys.index(as_key)],
              resolved_hw,
              resolved_gp,
              pad_mode='edge',
          )
        else:
          additional_states_updated[as_key] = value

      return additional_states_updated

    return additional_states_update_fn


def parse_inflow_helper_key(
    key: str,
) -> tuple[str, str, int, int] | None:
  """Parses a synthetic turbulent inflow helper key.

  Keys follow format: `{helper_type}_{var}_{inflow_dim}_{inflow_face}`, e.g.,
  `mean_u_0_0`, `rms_v_1_0`, `rand_w_2_0`.

  Args:
    key: A variable name that may be an inflow helper key.

  Returns:
    (helper_type, var, inflow_dim, inflow_face) if valid, else None.
  """
  parts = key.split('_')
  if (
      len(parts) == 4
      and parts[0] in ('mean', 'rms', 'rand')
      and parts[1] in ('u', 'v', 'w')
      and parts[2].isdigit()
      and parts[3].isdigit()
  ):
    dim = int(parts[2])
    face = int(parts[3])
    if dim in (0, 1, 2) and face in (0, 1):
      return parts[0], parts[1], dim, face
  return None


def is_inflow_helper_key(key: str) -> bool:
  """Checks if `key` corresponds to a synthetic turbulent inflow helper."""
  return parse_inflow_helper_key(key) is not None


def get_inflow_partition_spec(
    key: str,
    mesh: Mesh,
    grid_params: grid_parametrization.GridParametrization | None = None,
) -> P:
  """Returns the `PartitionSpec` for an inflow helper variable."""
  info = parse_inflow_helper_key(key)
  if info is None:
    raise ValueError(f'{key} is not a valid synthetic turbulent inflow key.')

  helper_type, _, inflow_dim, _ = info
  inflow_plane = [d for d in range(3) if d != inflow_dim]
  axis_0 = ('x', 'y', 'z')[inflow_plane[0]]
  axis_1 = ('x', 'y', 'z')[inflow_plane[1]]

  mesh_axis_0 = (
      axis_0  # pylint: disable=g-long-ternary
      if axis_0 in mesh.axis_names
      else (
          mesh.axis_names[grid_params.data_axis_order.index(axis_0)]
          if grid_params is not None and axis_0 in grid_params.data_axis_order
          else mesh.axis_names[inflow_plane[0]]
      )
  )
  mesh_axis_1 = (
      axis_1  # pylint: disable=g-long-ternary
      if axis_1 in mesh.axis_names
      else (
          mesh.axis_names[grid_params.data_axis_order.index(axis_1)]
          if grid_params is not None and axis_1 in grid_params.data_axis_order
          else mesh.axis_names[inflow_plane[1]]
      )
  )

  if helper_type in ('mean', 'rms'):
    return P(mesh_axis_0, mesh_axis_1)
  elif helper_type == 'rand':
    return P(None, mesh_axis_0, mesh_axis_1)
  else:
    raise ValueError(f'Unknown helper type: {helper_type}')


def distribute_inflow_state(
    key: str,
    per_replica_states: Sequence[dict[str, Any]],
    mesh: Mesh,
    grid_params: grid_parametrization.GridParametrization | None = None,
) -> jax.Array:
  """Assembles per-replica inflow helper states into globally-sharded JAX arrays.

  Args:
    key: The inflow helper key, e.g. `mean_u_0_0`, `rms_v_1_0`, `rand_w_2_0`.
    per_replica_states: Sequence of state dicts from each replica.
    mesh: The JAX Mesh.
    grid_params: Optional grid parametrization.

  Returns:
    A globally-sharded JAX Array.
  """
  info = parse_inflow_helper_key(key)
  if info is None:
    raise ValueError(f'{key} is not a valid synthetic turbulent inflow key.')

  helper_type, _, inflow_dim, inflow_face = info
  inflow_axis = ('x', 'y', 'z')[inflow_dim]
  inflow_plane = [d for d in range(3) if d != inflow_dim]
  axis_0 = ('x', 'y', 'z')[inflow_plane[0]]
  axis_1 = ('x', 'y', 'z')[inflow_plane[1]]

  mesh_axis_0 = (
      axis_0  # pylint: disable=g-long-ternary
      if axis_0 in mesh.axis_names
      else (
          mesh.axis_names[grid_params.data_axis_order.index(axis_0)]
          if grid_params is not None and axis_0 in grid_params.data_axis_order
          else mesh.axis_names[inflow_plane[0]]
      )
  )
  mesh_axis_1 = (
      axis_1  # pylint: disable=g-long-ternary
      if axis_1 in mesh.axis_names
      else (
          mesh.axis_names[grid_params.data_axis_order.index(axis_1)]
          if grid_params is not None and axis_1 in grid_params.data_axis_order
          else mesh.axis_names[inflow_plane[1]]
      )
  )
  mesh_inflow = (
      inflow_axis  # pylint: disable=g-long-ternary
      if inflow_axis in mesh.axis_names
      else (
          mesh.axis_names[grid_params.data_axis_order.index(inflow_axis)]  # pylint: disable=g-long-ternary
          if grid_params is not None
          and inflow_axis in grid_params.data_axis_order
          else mesh.axis_names[inflow_dim]
      )
  )

  num_replicas = len(per_replica_states)
  sample = next((s[key] for s in per_replica_states if key in s), None)
  if sample is None:
    raise ValueError(f'Key {key} not found in any replica state.')
  computation_shape = tuple(mesh.shape[name] for name in mesh.axis_names)
  target_inflow_coord = 0 if inflow_face == 0 else (mesh.shape[mesh_inflow] - 1)

  if helper_type in ('mean', 'rms'):
    sharding_2d = NamedSharding(mesh, P(mesh_axis_0, mesh_axis_1))
    if num_replicas == 1:
      return jax.device_put(jnp.asarray(sample), sharding_2d)

    c_0 = mesh.shape[mesh_axis_0]
    c_1 = mesh.shape[mesh_axis_1]
    grid_2d = np.full((c_0, c_1), None, dtype=object)
    for idx in range(num_replicas):
      multi_idx = np.unravel_index(idx, computation_shape)
      replica_coords = {
          name: multi_idx[i] for i, name in enumerate(mesh.axis_names)
      }
      coord_0 = replica_coords[mesh_axis_0]
      coord_1 = replica_coords[mesh_axis_1]
      if key in per_replica_states[idx]:
        if replica_coords[mesh_inflow] == target_inflow_coord:
          grid_2d[coord_0, coord_1] = per_replica_states[idx][key]
        elif grid_2d[coord_0, coord_1] is None:
          grid_2d[coord_0, coord_1] = per_replica_states[idx][key]

    if any(v is None for v in grid_2d.flat):
      raise ValueError(
          f'Missing replica shards when assembling {key} on mesh {mesh.shape}.'
      )

    return jax.device_put(jnp.block(grid_2d.tolist()), sharding_2d)

  elif helper_type == 'rand':
    sharding_rand = NamedSharding(mesh, P(None, mesh_axis_0, mesh_axis_1))
    if num_replicas == 1:
      return jax.device_put(jnp.asarray(sample), sharding_rand)

    c_0 = mesh.shape[mesh_axis_0]
    c_1 = mesh.shape[mesh_axis_1]
    grid_rand = np.full((c_0, c_1), None, dtype=object)
    for idx in range(num_replicas):
      multi_idx = np.unravel_index(idx, computation_shape)
      replica_coords = {
          name: multi_idx[i] for i, name in enumerate(mesh.axis_names)
      }
      coord_0 = replica_coords[mesh_axis_0]
      coord_1 = replica_coords[mesh_axis_1]
      if key in per_replica_states[idx]:
        if replica_coords[mesh_inflow] == target_inflow_coord:
          grid_rand[coord_0, coord_1] = per_replica_states[idx][key]
        elif grid_rand[coord_0, coord_1] is None:
          grid_rand[coord_0, coord_1] = per_replica_states[idx][key]

    if any(v is None for v in grid_rand.flat):
      raise ValueError(
          f'Missing replica shards when assembling {key} on mesh {mesh.shape}.'
      )

    rows = []
    for i in range(c_0):
      row = jnp.concatenate([grid_rand[i, j] for j in range(c_1)], axis=2)
      rows.append(row)
    full_rand = jnp.concatenate(rows, axis=1)
    return jax.device_put(full_rand, sharding_rand)

  raise ValueError(f'Unsupported helper type: {helper_type}')
