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

# Copyright 2026 Google LLC
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
"""The boundary conditions as the driver sees them.

The driver deals in boundary conditions as a whole. Which kinds exist -- the
nonreflecting boundary, the immersed boundary and its methods -- and what
state each of them keeps is decided here, so that adding one does not mean
teaching the driver about it.

This is a module of its own rather than part of `boundary_condition_utils`
because the libraries it delegates to import `boundary_condition_utils`, as
does `parameters`.
"""

from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.boundary_condition import immersed_boundary_method
from swirl_lm.jax.boundary_condition import nonreflecting_boundary
from swirl_lm.jax.utility import types

ScalarFieldMap = types.ScalarFieldMap


def required_keys(params: parameters_lib.SwirlLMParameters) -> list[str]:
  """Returns the additional state keys the configured boundary conditions need.

  A boundary condition declares the state it needs, so that the driver can
  allocate it and a configuration does not have to list internal arrays by
  hand. One that is not configured contributes nothing, so this can be called
  unconditionally.

  Args:
    params: The simulation parameters.

  Returns:
    The additional state keys to allocate. These may repeat keys that a
    configuration already lists, so the caller should deduplicate.
  """
  return immersed_boundary_method.required_keys(params)


def initial_states(params: parameters_lib.SwirlLMParameters) -> ScalarFieldMap:
  """Returns the initial values of the states the boundary conditions maintain.

  Args:
    params: The simulation parameters.

  Returns:
    A dict of initial states, empty if no configured boundary condition
    maintains any.
  """
  return nonreflecting_boundary.nonreflecting_bc_state_init_fn(params)


def update_additional_states(
    params: parameters_lib.SwirlLMParameters,
    states: ScalarFieldMap,
    additional_states: ScalarFieldMap,
    step_id: int,
) -> ScalarFieldMap:
  """Advances the states the boundary conditions maintain by one step.

  Args:
    params: The simulation parameters.
    states: Flow field variables.
    additional_states: The additional states, which hold the boundary
      condition states to advance.
    step_id: The current simulation step index.

  Returns:
    A dict of only the states that changed, to be merged into
    `additional_states` by the caller.
  """
  return nonreflecting_boundary.nonreflecting_bc_state_update_fn(
      params, states, additional_states, step_id
  )


def exclude_from_checkpoint(key: str) -> bool:
  """Whether `key` is boundary condition state that checkpoints leave out.

  A boundary condition may keep solver internals in the state that mean
  nothing outside the solver. Fields that post-processing uses, such as the
  immersed boundary masks, are not covered by this and are written.

  Args:
    key: A state name.

  Returns:
    True if the key should be left out of checkpoints.
  """
  return immersed_boundary_method.exclude_from_checkpoint(key)
