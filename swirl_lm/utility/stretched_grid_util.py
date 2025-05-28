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

"""Stretched grid utilities."""

from typing import TypeAlias

from swirl_lm.utility import types

STRETCHED_GRID_KEY_PREFIX = 'stretched_grid'
FlowFieldMap: TypeAlias = types.FlowFieldMap


def h_key(dim: int) -> str:
  """Gets the key for the stretched grid scale factor on nodes, given `dim`."""
  return STRETCHED_GRID_KEY_PREFIX + f'_h{dim}'


def h_face_key(dim: int) -> str:
  """Gets the key for the stretched grid scale factor on faces, given `dim`."""
  return h_key(dim) + '_face'


def get_helper_variables(
    additional_states: FlowFieldMap,
) -> FlowFieldMap:
  """Returns a dictionary with just the stretched grid helper variables."""
  return {
      key: additional_states[key]
      for key in additional_states
      if key.startswith(STRETCHED_GRID_KEY_PREFIX)
  }


def get_use_stretched_grid(
    additional_states: FlowFieldMap,
) -> tuple[bool, bool, bool]:
  """Returns a tuple of boolean values indicating if stretched grid is used."""
  return tuple(h_key(dim) in additional_states for dim in (0, 1, 2))
