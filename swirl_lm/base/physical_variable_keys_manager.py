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

"""A library that manages `physical_variable_keys` in fluid simulations."""

import abc
import enum
import re
from typing import Dict, Optional, Sequence, Text, Tuple, Union

from absl import logging
import six
from swirl_lm.communication import halo_exchange
from swirl_lm.utility import common_ops
from swirl_lm.utility import types

BCKeyInfo = Tuple[Text, int, int]
SourceKeyInfo = Text
KeyInfo = Union[BCKeyInfo, SourceKeyInfo]


class PhysicalVariablesType(enum.Enum):
  UNKNOWN = 0
  BOUNDARY_CONDITION = 1
  SOURCE = 2
  IMMERSED_BOUNDARY = 3


@six.add_metaclass(abc.ABCMeta)
class PhysicalVariableKeysHelper(object):
  """Defines a template for processing `additional_state_keys` in NS solver."""

  def __init__(self, key_pattern):
    """Sets the key pattern for a particular type of `additional_states`."""
    self._key_pattern = key_pattern

  def _parse_key_text_info(
      self, additional_state_key: Text):
    """Parse the key with the pattern specified in `self._key_pattern`."""
    match = re.search(self._key_pattern, additional_state_key)
    if not match:
      return None

    return match.groups()

  def _parse_key(self, additional_state_key: Text):
    """Parse the key with the pattern specified in `self._key_pattern`.

    This function needs to be implemented in the derived class.

    Args:
      additional_state_key: The key of the `additional_states` to be processed.

    Returns:
      Information inferred from the key.

    Raises:
      NotImplementedError: If the derived class does not provide a definition to
        this function.
    """
    raise NotImplementedError(
        '{} is not processed.'.format(additional_state_key))

  def _update_helper_variable_from_additional_states(self, *args):
    """Retrieves helper variables from `additional_states`.

    This function needs to be implemented in the derived class.

    Args:
      *args: Inputs to process the helper variables. Exact arguments depend on
        the type of `additional_states`.

    Returns:
      The updated helper variables.

    Raises:
      NotImplementedError: If the derived class does not provide a definition to
        this function.
    """
    raise NotImplementedError(
        'No helper variables are processed. Key helpers need to define their '
        'own update function')

  def parse_key(self, additional_state_key: Text):
    """Parse the key with the pattern specified in `self._key_pattern`.

    Args:
      additional_state_key: The key of the `additional_states` to be processed.

    Returns:
      Information inferred from the key.
    """
    return self._parse_key(additional_state_key)

  def update_helper_variable_from_additional_states(self, *args):
    """Retrieves helper variables from `additional_states`.

    Args:
      *args: Inputs to process the helper variables. Exact arguments depend on
        the type of `additional_states`.

    Returns:
      The updated helper variables.
    """
    return self._update_helper_variable_from_additional_states(*args)


class BoundaryConditionKeysHelper(PhysicalVariableKeysHelper):
  r"""Processes `additional_states` for boundary conditions.

  The key is associated with a boundary condition if it follows the naming rule:
    'bc_(\w+)_([0-2])_([0-1])',
  with the pattern being in the format of:
    (variable_name]_(dimension)_(face).
  For example, 'bc_w_2_0' refers to the boundary condition of 'w' in dimension 2
  on face 0 (the lower face).
  """

  def __init__(self):
    super(BoundaryConditionKeysHelper,
          self).__init__(r'bc_(\w+)_([0-2])_([0-1])')

  def _parse_key(self, additional_state_key: Text):
    """Parse the key for boundary conditions.

    Args:
      additional_state_key: A string that might be the name of a boundary
        condition.

    Returns:
      A tuple of boundary information (variable_name, dimension, face).
    """
    key_info = self._parse_key_text_info(additional_state_key)

    if key_info is None:
      return key_info

    varname = key_info[0]
    dim = int(key_info[1])
    face = int(key_info[2])

    return (varname, dim, face)

  def _update_helper_variable_from_additional_states(
      self,
      additional_states: types.FlowFieldMap,
      halo_width: int,
      bc: Dict[Text, halo_exchange.BoundaryConditionsSpec],
  ) -> Dict[Text, halo_exchange.BoundaryConditionsSpec]:
    """Retrieves boundary conditions from `additional_states`.

    It is assumed that the updated boundary condition preserves the type but
    with its value being replaced by the additional state.

    Args:
      additional_states: A dictionary of keyed variables that provides
        additional information to the simulation.
      halo_width: The width of the halo layer, which indicates the number of
        planes needs to be provided as boundary conditions.
      bc: A container of boundary conditions.

    Returns:
      The updated boundary conditions.
    """
    for key, value in additional_states.items():
      boundary_info = self.parse_key(key)
      if not boundary_info:
        continue

      (varname, dim, face) = boundary_info
      if varname not in bc:
        continue

      bc_value = []
      for i in range(halo_width):
        bc_value += common_ops.get_face(value, dim, face, i)

      # The order of the slices is from the outermost to the innermost. On the
      # higher end it needs to be reversed.
      if face == 1:
        bc_value = bc_value[::-1]

      bc[varname][dim][face] = (bc[varname][dim][face][0], bc_value)

    return bc

  def check_boundaries_updates_from_additional_states(
      self,
      additional_states_keys: Sequence[Text],
  ):
    """Checks if any boundaries are updated from the `additional_states`.

    Args:
      additional_states_keys: The names of all additional states.
    """
    for key in additional_states_keys:
      boundary_info = self.parse_key(key)
      if not boundary_info:
        continue
      logging.info(
          'Boundary condition for %s on face %d in the %d dimension is '
          'overwritten by `additional_states`.', boundary_info[0],
          boundary_info[2], boundary_info[1])

  def generate_bc_key(self, varname: Text, dim: int, face: int) -> Text:
    """Generates a key for a boundary condition variable.

    Args:
      varname: The name of the variable to which the boundary condition is
        applied.
      dim: The dimension along which the boundary condition is applied. Should
        be one of 0, 1, and 2.
      face: The face in `dim` where the boundary condition is imposed. Should be
        either 0 or 1. If it's 0, then the boundary condition is applied at the
        lower plane; if it's 1, the boundary condition is applied at the higher
        plane.

    Returns:
      The boundary condition key for variable `varname` in the `dim` direction
      on the `face` plane.

    Raises:
      ValueError: If `dim` is not one of 0, 1, and 2.
      ValueError: If `face` is not one of 0 and 1.
    """
    if dim not in [0, 1, 2]:
      raise ValueError(
          'Dimension should be one of 0, 1, and 2. {} is given.'.format(dim))

    if face not in [0, 1]:
      raise ValueError(
          'Face should be one of 0 and 1. {} is given.'.format(face))

    return 'bc_{}_{}_{}'.format(varname, dim, face)


class SourceKeysHelper(PhysicalVariableKeysHelper):
  r"""Processes `additional_states` for source terms.

  The key in additional states associated with external sources follows the
  naming rule:
    'src_(\w+)',
  with patterns being the variable name.
  """

  def __init__(self):
    super(SourceKeysHelper, self).__init__(r'src_(\w+)')

  def _parse_key(self, additional_state_key: Text) -> Optional[SourceKeyInfo]:
    """Parse the key for variable name."""
    key_info = self._parse_key_text_info(additional_state_key)
    return None if key_info is None else key_info[0]

  def _update_helper_variable_from_additional_states(
      self,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldMap:
    R"""Updates external sources/forces that are specified in additinoal states.

    Args:
      additional_states: A dictionary that holds constants that will be used in
        the simulation, e.g. boundary conditions, source terms.

    Returns:
      The source/forcing term dictionary with corresponding items updated from
      `additional_states`.
    """
    src = {}

    for key, value in additional_states.items():
      varname = self._parse_key(key)
      if not varname:
        continue
      src.update({varname: value})

    return src

  def generate_src_key(self, varname: Text) -> Text:
    """Generates the name of the source term.

    Args:
      varname: The name of the variable.

    Returns:
      The name of the source term for `varname` following the naming rule
      'src_[varname]'.
    """
    return 'src_{}'.format(varname)
