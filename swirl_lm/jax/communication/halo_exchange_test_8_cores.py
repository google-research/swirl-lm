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

"""Tests for halo exchange."""

import copy
import functools
import itertools
from typing import Literal

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
from swirl_lm.jax.communication import halo_exchange
from swirl_lm.jax.communication import halo_exchange_utils
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import test_util

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'float32')

GP = grid_parametrization.GridParametrization
FaceBoundaryCondition = halo_exchange_utils.FaceBoundaryCondition
DimBoundaryConditions = halo_exchange_utils.DimBoundaryConditions
BCType = halo_exchange_utils.BCType

_axis_order = test_util.data_axis_order


def generate_arrays(
    array_shape: tuple[int, int, int],
    mesh_shape: tuple[int, int, int],
    hw: int,
    random_seed: int = 101,
    halo_value: float = 100.0,
) -> tuple[
    dict[tuple[int, int, int], np.ndarray],
    dict[tuple[int, int, int], np.ndarray],
]:
  """Generates arrays with and without halos.

  Args:
    array_shape: The shape of the array per core. This shape does not include
      the halos on both sides of each axis.
    mesh_shape: The shape of the mesh.
    hw: The width of the halo.
    random_seed: The random seed to use for generating the arrays.
    halo_value: The value to use for filling the halos.

  Returns:
    Two dicts of arrays, one with halos and one without. The keys of the dicts
    are (c0,c1,c2) taken from the mesh_shape. Each array without halos has shape
    `array_shape`, and each array with halos has shape
    `[a + 2 * hw for a in array_shape]`. Halos are filled with `halo_value`.
  """
  np.random.seed(random_seed)
  full_size = [a * c for a, c in zip(array_shape, mesh_shape)]
  full_array = np.random.normal(size=full_size)

  arrays_without_halo = {}
  for c0 in range(mesh_shape[0]):
    for c1 in range(mesh_shape[1]):
      for c2 in range(mesh_shape[2]):
        slice_3d = (
            slice(c0 * array_shape[0], (c0 + 1) * array_shape[0]),
            slice(c1 * array_shape[1], (c1 + 1) * array_shape[1]),
            slice(c2 * array_shape[2], (c2 + 1) * array_shape[2]),
        )
        arrays_without_halo[(c0, c1, c2)] = full_array[slice_3d]

  arrays_with_uniform_halo = {}
  for c0 in range(mesh_shape[0]):
    for c1 in range(mesh_shape[1]):
      for c2 in range(mesh_shape[2]):
        arrays_with_uniform_halo[(c0, c1, c2)] = np.copy(
            arrays_without_halo[(c0, c1, c2)]
        )
        arrays_with_uniform_halo[(c0, c1, c2)] = np.pad(
            arrays_with_uniform_halo[(c0, c1, c2)],
            pad_width=[(hw, hw)] * 3,
            constant_values=halo_value,
        )
  return arrays_without_halo, arrays_with_uniform_halo


def init_distributed_jax_array(
    np_arrays: dict[tuple[int, int, int], np.ndarray],
    mesh: Mesh,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
  """Initializes a distributed JAX array from NumPy arrays.

  Args:
    np_arrays: A dict of NumPy arrays, where the keys are core IDs and the
      values are the arrays for that core. Refer to `generate_arrays` for
      detailed explanation.
    mesh: The mesh to use for addressable data.
    dtype: The data type of the JAX array.

  Returns:
    The JAX array distributed across the mesh.
  """
  local_shape = np_arrays[(0, 0, 0)].shape
  mesh_shape = mesh.devices.shape
  global_shape = [a * c for a, c in zip(local_shape, mesh_shape)]
  # Initialize the array with zeros. Then fill-up individual shards.
  array_jax = jax.device_put(
      jnp.zeros(global_shape, jnp.float32),
      NamedSharding(mesh, P(*mesh.axis_names)),
  )

  def _update_shard(
      array: jax.Array,
      updated_shard: jax.Array,
      shard_indices: tuple[int, ...],
      core_numbers: tuple[int, ...],
  ):
    if len(shard_indices) != len(core_numbers):
      raise ValueError(
          f'shard_indices: {shard_indices} and core_numbers:'
          f' {core_numbers} must have the same length.'
      )
    if len(shard_indices) != array.ndim:
      raise ValueError(
          f'shard_indices: {shard_indices} and array.ndim: {array.ndim} should'
          ' be equal.'
      )
    starts = tuple(
        i * s // c for i, s, c in zip(shard_indices, array.shape, core_numbers)
    )
    return jax.lax.dynamic_update_slice(array, updated_shard, starts)

  for c0 in range(mesh_shape[0]):
    for c1 in range(mesh_shape[1]):
      for c2 in range(mesh_shape[2]):
        core_id = (c0, c1, c2)
        array_jax = _update_shard(
            array_jax,
            jnp.array(np_arrays[core_id], dtype=dtype),
            core_id,
            mesh_shape,
        )
  return array_jax


def is_first_core(
    core_id: tuple[int, int, int], mesh_shape: tuple[int, int, int], dim: int
) -> bool:
  """Returns True if the core is the first core in the given dimension."""
  del mesh_shape
  return core_id[dim] == 0


def is_last_core(
    core_id: tuple[int, int, int], mesh_shape: tuple[int, int, int], dim: int
) -> bool:
  """Returns True if the core is the last core in the given dimension."""
  return core_id[dim] == mesh_shape[dim] - 1


def get_3d_slice(slice_1d: slice, dim: int) -> tuple[slice, slice, slice]:
  """Returns a 3D slice with the given 1D slice in the specified dimension."""
  slices = [slice(None)] * 3
  slices[dim] = slice_1d
  return tuple(slices)


def get_succesor_core_id(
    core_id: tuple[int, int, int], mesh_shape: tuple[int, int, int], dim: int
) -> tuple[int, int, int]:
  """Returns the successor core ID in the given dimension."""
  idx = list(core_id)
  idx[dim] = (idx[dim] + 1) % mesh_shape[dim]
  return tuple(idx)


def get_predecessor_core_id(
    core_id: tuple[int, int, int], mesh_shape: tuple[int, int, int], dim: int
) -> tuple[int, int, int]:
  """Returns the predecessor core ID in the given dimension."""
  idx = list(core_id)
  idx[dim] = (idx[dim] - 1) % mesh_shape[dim]
  return tuple(idx)


def internal_halo_exchange(
    arrays: dict[tuple[int, int, int], np.ndarray],
    hw: int,
    dim: Literal[0, 1, 2],
    mesh_shape: tuple[int, int, int],
    periodic: bool,
) -> dict[tuple[int, int, int], np.ndarray]:
  """Performs internal halo exchange for all cores.

  This function leaves boundaries untouched if periodic is false. If periodic is
  true, it performs halo exchange for all the cores.

  Args:
    arrays: A dict of arrays, where the keys are core IDs and the values are the
      arrays for that core. Refer to `generate_arrays` for detailed explanation.
    hw: The width of the halo.
    dim: The dimension to perform halo exchange in.
    mesh_shape: The shape of the mesh.
    periodic: Whether the simulation is periodic.

  Returns:
    The arrays with internal halos exchanged. Format of the return value is the
    same as the input `arrays`.
  """
  for c0 in range(mesh_shape[0]):
    for c1 in range(mesh_shape[1]):
      for c2 in range(mesh_shape[2]):
        core_id = (c0, c1, c2)
        succ_core_id = get_succesor_core_id(core_id, mesh_shape, dim)
        pred_core_id = get_predecessor_core_id(core_id, mesh_shape, dim)

        if np.logical_or(not is_first_core(core_id, mesh_shape, dim), periodic):
          to_slice = get_3d_slice(slice(0, hw), dim)
          from_slice = get_3d_slice(slice(-2 * hw, -hw), dim)
          arrays[core_id][to_slice] = arrays[pred_core_id][from_slice]

        if np.logical_or(not is_last_core(core_id, mesh_shape, dim), periodic):
          to_slice = get_3d_slice(slice(-hw, None), dim)
          from_slice = get_3d_slice(slice(hw, 2 * hw), dim)
          arrays[core_id][to_slice] = arrays[succ_core_id][from_slice]
  return arrays


def set_face_dirichlet_bc(
    array: np.ndarray,
    sidetype: Literal['low', 'high'],
    plane_bc_value: jax.Array | float,
    dim: Literal[0, 1, 2],
    plane: int,
) -> np.ndarray:
  """Sets the face boundary condition to Dirichlet BC.

  Args:
    array: The array to set the boundary condition on.
    sidetype: Whether to set the low or high side of the boundary condition.
    plane_bc_value: The value of the boundary condition.
    dim: The dimension of the boundary condition.
    plane: The plane of the boundary condition.

  Returns:
    The array with the boundary condition set.
  """
  if sidetype == 'low':
    slice_3d = get_3d_slice(slice(plane, plane + 1), dim)
  elif sidetype == 'high':
    n = array.shape[dim]
    slice_3d = get_3d_slice(slice(n - plane - 1, n - plane), dim)
  else:
    raise ValueError(f'Unsupported sidetype: {sidetype}')
  array[slice_3d] = plane_bc_value
  return array


def set_face_additive_bc(
    array: np.ndarray,
    sidetype: Literal['low', 'high'],
    plane_bc_value: jax.Array | float,
    dim: Literal[0, 1, 2],
    plane: int,
) -> np.ndarray:
  """Sets the face boundary condition to Additive BC.

  Args:
    array: The array to set the boundary condition on.
    sidetype: Whether to set the low or high side of the boundary condition.
    plane_bc_value: The value of the boundary condition.
    dim: The dimension of the boundary condition.
    plane: The plane of the boundary condition.

  Returns:
    The array with the boundary condition set.
  """
  if sidetype == 'low':
    slice_3d = get_3d_slice(slice(plane, plane + 1), dim)
  elif sidetype == 'high':
    n = array.shape[dim]
    slice_3d = get_3d_slice(slice(n - plane - 1, n - plane), dim)
  else:
    raise ValueError(f'Unsupported sidetype: {sidetype}')
  array[slice_3d] = array[slice_3d] + plane_bc_value
  return array


def set_face_neumann_order_1_bc(
    array: np.ndarray,
    sidetype: Literal['low', 'high'],
    plane_bc_value: jax.Array | float,
    dim: Literal[0, 1, 2],
    plane: int,
) -> np.ndarray:
  """Sets the face boundary condition to Neumann BC with order 1.

  Args:
    array: The array to set the boundary condition on.
    sidetype: Whether to set the low or high side of the boundary condition.
    plane_bc_value: The value of the boundary condition.
    dim: The dimension of the boundary condition.
    plane: The plane of the boundary condition.

  Returns:
    The array with the boundary condition set.
  """
  if sidetype == 'low':
    slice_3d = get_3d_slice(slice(plane, plane + 1), dim)
    neighbor_slice_3d = get_3d_slice(slice(plane + 1, plane + 2), dim)
    sign = -1.0
  elif sidetype == 'high':
    n = array.shape[dim]
    slice_3d = get_3d_slice(slice(n - plane - 1, n - plane), dim)
    neighbor_slice_3d = get_3d_slice(slice(n - plane - 2, n - plane - 1), dim)
    sign = 1.0
  else:
    raise ValueError(f'Unsupported sidetype: {sidetype}')
  array[slice_3d] = sign * plane_bc_value + array[neighbor_slice_3d]
  return array


def set_face_neumann_order_2_bc(
    array: np.ndarray,
    sidetype: Literal['low', 'high'],
    plane_bc_value: jax.Array | float,
    dim: Literal[0, 1, 2],
    plane: int,
) -> np.ndarray:
  """Sets the face boundary condition to Neumann BC with order 2.

  Args:
    array: The array to set the boundary condition on.
    sidetype: Whether to set the low or high side of the boundary condition.
    plane_bc_value: The value of the boundary condition.
    dim: The dimension of the boundary condition.
    plane: The plane of the boundary condition.

  Returns:
    The array with the boundary condition set.
  """
  if sidetype == 'low':
    slice_3d = get_3d_slice(slice(plane, plane + 1), dim)
    neighbor_1_slice_3d = get_3d_slice(slice(plane + 1, plane + 2), dim)
    neighbor_2_slice_3d = get_3d_slice(slice(plane + 2, plane + 3), dim)
    sign = -1.0
  elif sidetype == 'high':
    n = array.shape[dim]
    slice_3d = get_3d_slice(slice(n - plane - 1, n - plane), dim)
    neighbor_1_slice_3d = get_3d_slice(slice(n - plane - 2, n - plane - 1), dim)
    neighbor_2_slice_3d = get_3d_slice(slice(n - plane - 3, n - plane - 2), dim)
    sign = 1.0
  else:
    raise ValueError(f'Unsupported sidetype: {sidetype}')
  array[slice_3d] = (
      sign * plane_bc_value
      + 4.0 / 3.0 * array[neighbor_1_slice_3d]
      - 1.0 / 3.0 * array[neighbor_2_slice_3d]
  )
  return array


def set_face_bc(
    array: np.ndarray,
    bc_face: FaceBoundaryCondition | None,
    dim: Literal[0, 1, 2],
    hw: int,
    sidetype: Literal['low', 'high'],
) -> np.ndarray:
  """Sets the face boundary condition on the given array.

  Args:
    array: The array to set the boundary condition on.
    bc_face: The face boundary condition to set. If None, the boundary condition
      is set to zero.
    dim: The dimension of the boundary condition.
    hw: The width of the halo.
    sidetype: Whether to set the low or high side of the boundary condition.

  Returns:
    The array with the boundary condition set.
  """

  if bc_face is None:
    if sidetype == 'low':
      array[get_3d_slice(slice(0, hw), dim)] = 0.0
    elif sidetype == 'high':
      array[get_3d_slice(slice(-hw, None), dim)] = 0.0
    return array
  bc_type, bc_value = bc_face
  if bc_type == BCType.NO_TOUCH:
    return array
  if isinstance(bc_value, float):
    bc_value = [bc_value] * hw

  for plane in range(hw - 1, -1, -1):
    if sidetype == 'low':
      plane_bc_value = bc_value[plane]
    elif sidetype == 'high':
      plane_bc_value = bc_value[hw - plane - 1]
    else:
      raise ValueError(f'Unsupported sidetype: {sidetype}')
    if bc_type == BCType.NEUMANN:
      array = set_face_neumann_order_1_bc(
          array, sidetype, plane_bc_value, dim, plane
      )
    elif bc_type == BCType.NEUMANN_2:
      array = set_face_neumann_order_2_bc(
          array, sidetype, plane_bc_value, dim, plane
      )
    elif bc_type in (BCType.DIRICHLET, BCType.NONREFLECTING):
      array = set_face_dirichlet_bc(array, sidetype, plane_bc_value, dim, plane)
    elif bc_type == BCType.ADDITIVE:
      array = set_face_additive_bc(array, sidetype, plane_bc_value, dim, plane)
    else:
      raise ValueError(f'Unsupported bc_type: {bc_type}')
  return array


def set_dim_bc(
    arrays: dict[tuple[int, int, int], np.ndarray],
    bc_dim: DimBoundaryConditions,
    dim: Literal[0, 1, 2],
    mesh_shape: tuple[int, int, int],
    hw: int,
    periodic: bool,
) -> dict[tuple[int, int, int], np.ndarray]:
  """Sets the boundary conditions on the given arrays.

  Args:
    arrays: A dict of arrays, where the keys are core IDs and the values are the
      arrays for that core. Refer to `generate_arrays` for detailed explanation.
    bc_dim: The boundary conditions to set.
    dim: The dimension of the boundary condition.
    mesh_shape: The shape of the mesh.
    hw: The width of the halo.
    periodic: Whether the simulation is periodic.

  Returns:
    The arrays with the boundary conditions set. Format of the return value is
    the same as the input `arrays`.
  """
  if periodic:
    return arrays
  for c0 in range(mesh_shape[0]):
    for c1 in range(mesh_shape[1]):
      for c2 in range(mesh_shape[2]):
        core_id = (c0, c1, c2)
        if is_first_core(core_id, mesh_shape, dim):
          arrays[core_id] = set_face_bc(
              arrays[core_id], bc_dim[0], dim, hw, 'low'
          )
        if is_last_core(core_id, mesh_shape, dim):
          arrays[core_id] = set_face_bc(
              arrays[core_id], bc_dim[1], dim, hw, 'high'
          )
  return arrays


def inplace_halo_exchange_1d(
    arrays: dict[tuple[int, int, int], np.ndarray],
    bc_dim: DimBoundaryConditions,
    dim: Literal[0, 1, 2],
    mesh_shape: tuple[int, int, int],
    hw: int,
    periodic: bool,
) -> dict[tuple[int, int, int], np.ndarray]:
  """Performs inplace halo exchange for all cores in the given dimension.

  This function performs inplace halo exchange for all cores in the given
  dimension. It also sets the boundary conditions on the arrays. Refer to
  `internal_halo_exchange` and `set_dim_bc` for detailed explanation.

  Args:
    arrays: A dict of arrays, where the keys are core IDs and the values are the
      arrays for that core. Refer to `generate_arrays` for detailed explanation.
    bc_dim: The boundary conditions to set.
    dim: The dimension to perform halo exchange in.
    mesh_shape: The shape of the mesh.
    hw: The width of the halo.
    periodic: Whether the simulation is periodic.

  Returns:
    The arrays with inplace halo exchange performed. Format of the return value
    is the same as the input `arrays`.
  """
  arrays = internal_halo_exchange(arrays, hw, dim, mesh_shape, periodic)
  arrays = set_dim_bc(arrays, bc_dim, dim, mesh_shape, hw, periodic)
  return arrays


def validate_arrays(
    np_arrays: dict[tuple[int, int, int], np.ndarray],
    jax_array: jax.Array,
    mesh: Mesh,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    strict: bool = True,
) -> None:
  """Validates that the JAX array is equal to the NumPy arrays.

  Args:
    np_arrays: A dict of NumPy arrays, where the keys are core IDs and the
      values are the arrays for that core. Refer to `generate_arrays` for
      detailed explanation.
    jax_array: The JAX array distributed across the mesh.
    mesh: The mesh to use for addressable data.
    atol: The absolute tolerance for comparing arrays.
    rtol: The relative tolerance for comparing arrays.
    strict: Whether to raise an exception if the arrays are not equal.
  """
  mesh_shape = mesh.devices.shape
  count = 0
  for c0 in range(mesh_shape[0]):
    for c1 in range(mesh_shape[1]):
      for c2 in range(mesh_shape[2]):
        core_id = (c0, c1, c2)
        np.testing.assert_allclose(
            np_arrays[core_id],
            np.array(
                jax_array.addressable_data(count),
                dtype=np_arrays[core_id].dtype,
            ),
            atol=atol,
            rtol=rtol,
            strict=strict,
        )
        count = count + 1


class HaloExchangeTest(parameterized.TestCase):
  """Tests for halo exchange."""

  AXIS = ('x', 'y', 'z')
  MESH_SHAPES = (
      (2, 2, 2),
      (8, 1, 1),
      (1, 8, 1),
      (1, 1, 8),
      (4, 2, 1),
      (4, 1, 2),
      (2, 1, 4),
      (2, 4, 1),
      (1, 2, 4),
      (1, 4, 2),
  )
  HALO_WIDTHS = (1, 2, 3)

  def setUp(self):
    super(HaloExchangeTest, self).setUp()

    self.array_shape = (4, 5, 6)  # Shape per core excluding halos.
    self.grid_params = GP.create_from_grid_lengths_and_etc_with_defaults(
        data_axis_order=_axis_order
    )

  BC_DIMS_SCALAR_1D = (
      [None, None],
      [(BCType.DIRICHLET, 2.0), (BCType.NEUMANN, 5.0)],
      [(BCType.NEUMANN_2, 2.0), (BCType.ADDITIVE, 5.0)],
      [(BCType.NO_TOUCH, 2.0), (BCType.NONREFLECTING, 5.0)],
  )

  @parameterized.parameters(
      *itertools.product(MESH_SHAPES, HALO_WIDTHS, AXIS, BC_DIMS_SCALAR_1D)
  )
  def test_inplace_halo_exchange_1d_scalar_bc(
      self, mesh_shape, halo_width, axis, bc_dim
  ):
    """Tests 1D halo exchange with scalar values of boundary conditions."""
    devices = jax.experimental.mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(devices, axis_names=tuple(_axis_order))
    _, np_arrays_with_uniform_halo = generate_arrays(
        self.array_shape, mesh_shape, halo_width, random_seed=101
    )
    jax_array = init_distributed_jax_array(np_arrays_with_uniform_halo, mesh)
    np_arrays_after_he = copy.deepcopy(np_arrays_with_uniform_halo)
    np_arrays_after_he = inplace_halo_exchange_1d(
        np_arrays_after_he,
        bc_dim=bc_dim,
        dim=self.grid_params.get_axis_index(axis),
        mesh_shape=mesh_shape,
        hw=halo_width,
        periodic=False,
    )

    he_func_jax = functools.partial(
        halo_exchange._inplace_halo_exchange_1d,  # pylint: disable=protected-access
        axis=axis,
        mesh=mesh,
        periodic=False,
        bc_low=bc_dim[0],
        bc_high=bc_dim[1],
        halo_width=halo_width,
        grid_params=self.grid_params,
    )
    for plane in range(halo_width - 1, -1, -1):
      he_sharded_func_jax = functools.partial(he_func_jax, plane=plane)
      he_sharded_func_jax = shard_map(
          he_sharded_func_jax,
          mesh=mesh,
          in_specs=P(*mesh.axis_names),
          out_specs=P(*mesh.axis_names),
          check_rep=False,
      )
      jax_array = he_sharded_func_jax(jax_array)
    validate_arrays(np_arrays_after_he, jax_array, mesh)

  @parameterized.parameters(*itertools.product(MESH_SHAPES, HALO_WIDTHS))
  def test_inplace_halo_exchange_3d_scalar_bc(self, mesh_shape, halo_width):
    """Tests 3D halo exchange with scalar values of boundary conditions."""
    devices = jax.experimental.mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(devices, axis_names=tuple(_axis_order))
    _, np_arrays_with_uniform_halo = generate_arrays(
        self.array_shape, mesh_shape, halo_width, random_seed=101
    )
    jax_array = init_distributed_jax_array(np_arrays_with_uniform_halo, mesh)

    boundary_conditions = (
        ((BCType.DIRICHLET, 2.0), (BCType.NEUMANN, 5.0)),
        ((BCType.NEUMANN_2, -2.0), (BCType.ADDITIVE, 3.0)),
        ((BCType.NO_TOUCH, -1.0), (BCType.NONREFLECTING, -2.5)),
    )
    he_func_jax = functools.partial(
        halo_exchange.inplace_halo_exchange,
        axes=self.grid_params.data_axis_order,
        mesh=mesh,
        grid_params=self.grid_params,
        periodic_dims=None,
        boundary_conditions=boundary_conditions,
        halo_width=halo_width,
    )
    he_func_jax = shard_map(
        he_func_jax,
        mesh=mesh,
        in_specs=P(*mesh.axis_names),
        out_specs=P(*mesh.axis_names),
        check_rep=False,
    )
    jax_array = he_func_jax(jax_array)
    np_arrays_after_he = copy.deepcopy(np_arrays_with_uniform_halo)
    for i_axis, axis in enumerate(self.grid_params.data_axis_order):
      np_arrays_after_he = inplace_halo_exchange_1d(
          np_arrays_after_he,
          bc_dim=boundary_conditions[i_axis],
          dim=self.grid_params.get_axis_index(axis),
          mesh_shape=mesh_shape,
          hw=halo_width,
          periodic=False,
      )
    validate_arrays(np_arrays_after_he, jax_array, mesh)

  BC_TYPES_ARRAY_1D = (
      [BCType.DIRICHLET, BCType.NEUMANN],
      [BCType.NEUMANN_2, BCType.ADDITIVE],
      [BCType.NO_TOUCH, BCType.NONREFLECTING],
  )

  @parameterized.parameters(
      *itertools.product(MESH_SHAPES, HALO_WIDTHS, AXIS, BC_TYPES_ARRAY_1D)
  )
  def test_inplace_halo_exchange_1d_array_bc(
      self, mesh_shape, halo_width, axis, bc_type
  ):
    """Tests 1D halo exchange with array values of boundary conditions."""
    devices = jax.experimental.mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(devices, axis_names=tuple(_axis_order))
    _, np_arrays_with_uniform_halo = generate_arrays(
        self.array_shape, mesh_shape, halo_width, random_seed=101
    )
    jax_array = init_distributed_jax_array(np_arrays_with_uniform_halo, mesh)

    bc_shape = list(self.array_shape)
    bc_shape = [s + 2 * halo_width for s in bc_shape]
    bc_shape[self.grid_params.get_axis_index(axis)] = 1
    bc_value_0 = [None] * halo_width
    bc_value_1 = [None] * halo_width
    for h in range(halo_width):
      np.random.seed(11 + h)
      bc_value_0[h] = jnp.array(np.random.normal(size=bc_shape))
      np.random.seed(101 + h)
      bc_value_1[h] = jnp.array(np.random.normal(size=bc_shape))
    bc_dim = [(bc_type[0], bc_value_0), (bc_type[1], bc_value_1)]

    np_arrays_after_he = copy.deepcopy(np_arrays_with_uniform_halo)
    np_arrays_after_he = inplace_halo_exchange_1d(
        np_arrays_after_he,
        bc_dim=bc_dim,
        dim=self.grid_params.get_axis_index(axis),
        mesh_shape=mesh_shape,
        hw=halo_width,
        periodic=False,
    )

    he_func_jax = functools.partial(
        halo_exchange._inplace_halo_exchange_1d,  # pylint: disable=protected-access
        axis=axis,
        mesh=mesh,
        periodic=False,
        halo_width=halo_width,
        grid_params=self.grid_params,
    )
    for plane in range(halo_width - 1, -1, -1):
      bc_low = (bc_dim[0][0], jnp.squeeze(bc_dim[0][1][plane]))
      bc_high = (
          bc_dim[1][0],
          jnp.squeeze(bc_dim[1][1][halo_width - plane - 1]),
      )
      he_sharded_func_jax = functools.partial(
          he_func_jax, bc_low=bc_low, bc_high=bc_high, plane=plane
      )
      he_sharded_func_jax = shard_map(
          he_sharded_func_jax,
          mesh=mesh,
          in_specs=P(*mesh.axis_names),
          out_specs=P(*mesh.axis_names),
          check_rep=False,
      )
      jax_array = he_sharded_func_jax(jax_array)
    validate_arrays(np_arrays_after_he, jax_array, mesh)

  @parameterized.parameters(*itertools.product(MESH_SHAPES, HALO_WIDTHS))
  def test_inplace_halo_exchange_3d_array_bc(self, mesh_shape, halo_width):
    """Tests 3D halo exchange with array values of boundary conditions."""
    devices = jax.experimental.mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(devices, axis_names=tuple(_axis_order))
    _, np_arrays_with_uniform_halo = generate_arrays(
        self.array_shape, mesh_shape, halo_width, random_seed=101
    )
    jax_array = init_distributed_jax_array(np_arrays_with_uniform_halo, mesh)

    bc_types = (
        (BCType.DIRICHLET, BCType.NEUMANN),
        (BCType.NEUMANN_2, BCType.ADDITIVE),
        (BCType.NO_TOUCH, BCType.NONREFLECTING),
    )
    boundary_conditions = [None] * 3
    for axis in self.grid_params.data_axis_order:
      bc_shape = list(self.array_shape)
      bc_shape = [s + 2 * halo_width for s in bc_shape]
      axis_index = self.grid_params.get_axis_index(axis)
      bc_shape[axis_index] = 1
      bc_value_0 = [None] * halo_width
      bc_value_1 = [None] * halo_width
      for h in range(halo_width):
        np.random.seed(axis_index + 11 + h)
        bc_value_0[h] = jnp.array(np.random.normal(size=bc_shape))
        np.random.seed(axis_index + 101 + h)
        bc_value_1[h] = jnp.array(np.random.normal(size=bc_shape))
      boundary_conditions[axis_index] = [
          (bc_types[axis_index][0], bc_value_0),
          (bc_types[axis_index][1], bc_value_1),
      ]

    he_func_jax = functools.partial(
        halo_exchange.inplace_halo_exchange,
        axes=self.grid_params.data_axis_order,
        mesh=mesh,
        grid_params=self.grid_params,
        periodic_dims=None,
        boundary_conditions=boundary_conditions,
        halo_width=halo_width,
    )
    he_func_jax = shard_map(
        he_func_jax,
        mesh=mesh,
        in_specs=P(*mesh.axis_names),
        out_specs=P(*mesh.axis_names),
        check_rep=False,
    )
    jax_array = he_func_jax(jax_array)

    np_arrays_after_he = copy.deepcopy(np_arrays_with_uniform_halo)
    for axis in self.grid_params.data_axis_order:
      axis_index = self.grid_params.get_axis_index(axis)
      np_arrays_after_he = inplace_halo_exchange_1d(
          np_arrays_after_he,
          bc_dim=boundary_conditions[axis_index],
          dim=self.grid_params.get_axis_index(axis),
          mesh_shape=mesh_shape,
          hw=halo_width,
          periodic=False,
      )
    validate_arrays(np_arrays_after_he, jax_array, mesh)

  @parameterized.parameters(*itertools.product(MESH_SHAPES, HALO_WIDTHS, AXIS))
  def test_inplace_halo_exchange_1d_periodic(
      self, mesh_shape, halo_width, axis
  ):
    """Tests 1D halo exchange with scalar values of boundary conditions."""
    devices = jax.experimental.mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(devices, axis_names=tuple(_axis_order))
    _, np_arrays_with_uniform_halo = generate_arrays(
        self.array_shape, mesh_shape, halo_width, random_seed=101
    )
    jax_array = init_distributed_jax_array(np_arrays_with_uniform_halo, mesh)
    np_arrays_after_he = copy.deepcopy(np_arrays_with_uniform_halo)
    np_arrays_after_he = inplace_halo_exchange_1d(
        np_arrays_after_he,
        bc_dim=(None, None),
        dim=self.grid_params.get_axis_index(axis),
        mesh_shape=mesh_shape,
        hw=halo_width,
        periodic=True,
    )

    he_func_jax = functools.partial(
        halo_exchange._inplace_halo_exchange_1d,  # pylint: disable=protected-access
        axis=axis,
        mesh=mesh,
        periodic=True,
        bc_low=None,
        bc_high=None,
        halo_width=halo_width,
        grid_params=self.grid_params,
    )
    for plane in range(halo_width - 1, -1, -1):
      he_sharded_func_jax = functools.partial(he_func_jax, plane=plane)
      he_sharded_func_jax = shard_map(
          he_sharded_func_jax,
          mesh=mesh,
          in_specs=P(*mesh.axis_names),
          out_specs=P(*mesh.axis_names),
          check_rep=False,
      )
      jax_array = he_sharded_func_jax(jax_array)
    validate_arrays(np_arrays_after_he, jax_array, mesh)

  @parameterized.parameters(*itertools.product(MESH_SHAPES, HALO_WIDTHS))
  def test_inplace_halo_exchange_3d_periodic(self, mesh_shape, halo_width):
    """Tests 3D halo exchange with scalar values of boundary conditions."""
    devices = jax.experimental.mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(devices, axis_names=tuple(_axis_order))
    _, np_arrays_with_uniform_halo = generate_arrays(
        self.array_shape, mesh_shape, halo_width, random_seed=101
    )
    jax_array = init_distributed_jax_array(np_arrays_with_uniform_halo, mesh)

    he_func_jax = functools.partial(
        halo_exchange.inplace_halo_exchange,
        axes=self.grid_params.data_axis_order,
        mesh=mesh,
        grid_params=self.grid_params,
        periodic_dims=[True] * 3,
        boundary_conditions=None,
        halo_width=halo_width,
    )
    he_func_jax = shard_map(
        he_func_jax,
        mesh=mesh,
        in_specs=P(*mesh.axis_names),
        out_specs=P(*mesh.axis_names),
        check_rep=False,
    )
    jax_array = he_func_jax(jax_array)
    np_arrays_after_he = copy.deepcopy(np_arrays_with_uniform_halo)
    for axis in self.grid_params.data_axis_order:
      np_arrays_after_he = inplace_halo_exchange_1d(
          np_arrays_after_he,
          bc_dim=(None, None),
          dim=self.grid_params.get_axis_index(axis),
          mesh_shape=mesh_shape,
          hw=halo_width,
          periodic=True,
      )
    validate_arrays(np_arrays_after_he, jax_array, mesh)

  @parameterized.parameters(*HALO_WIDTHS)
  def test_set_halos_to_zero(self, halo_width):
    """Tests whether halos are set to zero."""
    inner_shape = (6, 10, 14)
    full_shape = [s + 2 * halo_width for s in inner_shape]
    np.random.seed(101)
    full_array = np.random.normal(size=full_shape)
    expected_array = np.zeros_like(full_array)
    expected_array[
        halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
    ] = full_array[
        halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
    ]
    func = functools.partial(
        halo_exchange.set_halos_to_zero,
        halo_width=halo_width,
        grid_params=self.grid_params,
    )
    test_util.assert_jax_retval_allclose(
        func,
        (jnp.array(full_array),),
        expected_outputs=(expected_array,),
    )


if __name__ == '__main__':
  absltest.main()
