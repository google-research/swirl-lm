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

"""A library of boundary condition related utility functions."""

import enum
from typing import TypeAlias

from absl import logging
from swirl_lm.jax.base import physical_variable_keys_manager
from swirl_lm.jax.communication import halo_exchange
from swirl_lm.jax.utility import types

BoundaryConditionDict: TypeAlias = dict[
    str, halo_exchange.BoundaryConditionsSpec | None
]

ScalarFieldMap: TypeAlias = types.ScalarFieldMap


class BoundaryType(enum.Enum):
  """Defines the physical type of the boundary."""

  UNKNOWN = 0
  SLIP_WALL = 1
  NON_SLIP_WALL = 2
  PERIODIC = 3
  INFLOW = 4
  OUTFLOW = 5
  SHEAR_WALL = 6


def find_bc_type(
    bc: BoundaryConditionDict, periodic_dims: list[bool]
) -> list[list[BoundaryType | None]]:
  """Finds the type of each boundary based on boundary conditions."""
  bc_type = [[None, None], [None, None], [None, None]]

  if bc['u'] is None or bc['v'] is None or bc['w'] is None:
    return [[BoundaryType.PERIODIC] * 2] * 3

  def velocity_var(dim: int):
    """The name of the velocity variable in the given dimension."""
    if dim not in range(3):
      raise ValueError(
          'Dimension has to be one of 0, 1, and 2. Given {}.'.format(dim)
      )
    return ('u', 'v', 'w')[dim]

  def is_non_slip_wall(dim: int, face: int):
    """Checks if the boundary is a non-slip wall."""
    return (
        (
            bc['u'][dim][face][0] == halo_exchange.BCType.DIRICHLET
            and bc['u'][dim][face][1] == 0.0
        )
        and (
            bc['v'][dim][face][0] == halo_exchange.BCType.DIRICHLET
            and bc['v'][dim][face][1] == 0.0
        )
        and (
            bc['w'][dim][face][0] == halo_exchange.BCType.DIRICHLET
            and bc['w'][dim][face][1] == 0.0
        )
    )

  def is_slip_wall(dim: int, face: int):
    """Checks if the boundary is a free-slip wall."""
    wall_normal_velocity = velocity_var(dim)

    # The velocity component normal to the wall should be 0 to have no
    # penetration.
    if (
        bc[wall_normal_velocity][dim][face][0] != halo_exchange.BCType.DIRICHLET
        or bc[wall_normal_velocity][dim][face][1] != 0.0
    ):
      return False

    for velocity in ['u', 'v', 'w']:
      if velocity == wall_normal_velocity:
        continue
      # Zero shear needs to be applied at a slip wall.
      if (
          bc[velocity][dim][face][0]
          not in (halo_exchange.BCType.NEUMANN, halo_exchange.BCType.NEUMANN_2)
          or bc[velocity][dim][face][1] != 0.0
      ):
        return False

    return True

  def is_shear_wall(dim: int, face: int):
    """Checks if the boundary is a shear wall."""
    wall_normal_velocity = velocity_var(dim)

    # The velocity component normal to the wall should be 0 to have no
    # penetration.
    if (
        bc[wall_normal_velocity][dim][face][0] != halo_exchange.BCType.DIRICHLET
        or bc[wall_normal_velocity][dim][face][1] != 0.0
    ):
      return False

    non_zero_shear = False

    for velocity in ['u', 'v', 'w']:
      if velocity == wall_normal_velocity:
        continue
      # Zero shear needs to be applied at a slip wall.
      if bc[velocity][dim][face][0] not in (
          halo_exchange.BCType.NEUMANN,
          halo_exchange.BCType.NEUMANN_2,
      ):
        return False

      if bc[velocity][dim][face][1] != 0.0:
        non_zero_shear = True

    return non_zero_shear

  def is_inflow(dim: int, face: int):
    """Checks if the boundary is an inflow."""
    mainstream = velocity_var(dim)

    for velocity in ['u', 'v', 'w']:
      bc_local = bc[velocity][dim][face]
      if velocity == mainstream:
        # The mainstream velocity in the inflow has to be specified as a
        # non-zero Dirichlet boundary condition.
        if bc_local[0] != halo_exchange.BCType.DIRICHLET or bc_local[1] == 0.0:
          return False
      else:
        # The tangential velocity components in the inflow have to be specified
        # as Dirichlet boundary condition with arbitrary values.
        if bc_local[0] != halo_exchange.BCType.DIRICHLET:
          return False

    return True

  # Note, this is currently exclusively used for deriving the BC type for
  # pressure.
  def is_outflow(dim: int, face: int):
    """Checks if the boundary is an outflow."""

    # Here we only consider the case in which the outflow is specified by an
    # all-Neumann boundary condition.
    return (
        bc['u'][dim][face][0]
        in (
            halo_exchange.BCType.NEUMANN,
            halo_exchange.BCType.NEUMANN_2,
            halo_exchange.BCType.NONREFLECTING,
        )
        and bc['v'][dim][face][0]
        in (
            halo_exchange.BCType.NEUMANN,
            halo_exchange.BCType.NEUMANN_2,
            halo_exchange.BCType.NONREFLECTING,
        )
        and bc['w'][dim][face][0]
        in (
            halo_exchange.BCType.NEUMANN,
            halo_exchange.BCType.NEUMANN_2,
            halo_exchange.BCType.NONREFLECTING,
        )
    )

  for dim in range(3):
    if periodic_dims[dim]:
      bc_type[dim] = [BoundaryType.PERIODIC, BoundaryType.PERIODIC]
      continue

    for face in range(2):
      if is_non_slip_wall(dim, face):
        bc_type[dim][face] = BoundaryType.NON_SLIP_WALL
      elif is_slip_wall(dim, face):
        bc_type[dim][face] = BoundaryType.SLIP_WALL
      elif is_shear_wall(dim, face):
        bc_type[dim][face] = BoundaryType.SHEAR_WALL
      elif is_inflow(dim, face):
        bc_type[dim][face] = BoundaryType.INFLOW
      elif is_outflow(dim, face):
        bc_type[dim][face] = BoundaryType.OUTFLOW
      else:
        bc_type[dim][face] = BoundaryType.UNKNOWN

  return bc_type


def get_keys_for_boundary_condition(
    bc: BoundaryConditionDict, bc_type: halo_exchange.BCType
) -> list[str]:
  """Generates a list of string keys for storing boudary values for `bc_type`.

  Args:
    bc: The dictionary containing the boundary conditions.
    bc_type: The type of boundary condition to generate the keys for.

  Returns:
    A set of strings to be used as the key to `additional_states` for storing
    the corresponding boundary condition values.
  """
  keys_for_bc = []
  bc_manager = physical_variable_keys_manager.BoundaryConditionKeysHelper()
  for k, v in bc.items():
    if v is None:
      continue
    for dim in range(3):
      for face in range(2):
        if v[dim][face] is None:
          continue
        if v[dim][face][0] == bc_type:
          additional_state_key_for_bc = bc_manager.generate_bc_key(k, dim, face)
          logging.info(
              'Encountering %s BC for variable: %s, at dimension: '
              '%d and face: %d. New additional_state_key: %s is added.',
              str(bc_type),
              k,
              dim,
              face,
              additional_state_key_for_bc,
          )
          keys_for_bc.append(additional_state_key_for_bc)
  return keys_for_bc
