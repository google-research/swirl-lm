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
"""Tests for what the driver asks of the boundary conditions."""

from absl.testing import absltest
from google.protobuf import text_format
from swirl_lm.base import parameters_pb2
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.boundary_condition import boundary_condition


def _make_params_with_ib(ib_textpb: str) -> parameters_lib.SwirlLMParameters:
  """Creates `SwirlLMParameters` with the given immersed boundary config."""
  config_text = f"""
    grid_params {{
      computation_shape {{ dim_0: 1 dim_1: 1 dim_2: 1 }}
      length {{ dim_0: 1.0 dim_1: 1.0 dim_2: 1.0 }}
      grid_size {{ dim_0: 8 dim_1: 8 dim_2: 8 }}
      dt: 0.001
    }}
    boundary_models {{
      ib {{
        {ib_textpb}
      }}
    }}
  """
  return parameters_lib.SwirlLMParameters(
      text_format.Parse(config_text, parameters_pb2.SwirlLMParameters())
  )


_SPONGE = """
    sponge {
      damping_coeff: 0.5
      variables { name: "u" value: 0.0 }
    }
"""
_GHOST_POINT = 'ghost_point { variables { name: "u" } }'


class BoundaryConditionTest(absltest.TestCase):

  def test_required_keys_is_empty_without_a_method_that_needs_state(self):
    params = _make_params_with_ib(_SPONGE)
    self.assertEmpty(boundary_condition.required_keys(params))

  def test_ghost_point_keys_split_into_checkpointed_fields_and_internals(self):
    params = _make_params_with_ib(_GHOST_POINT)

    keys = boundary_condition.required_keys(params)
    checkpointed = [
        key
        for key in keys
        if not boundary_condition.exclude_from_checkpoint(key)
    ]

    # The masks are written by default and the signed distance is not; the
    # rest of what the method allocates is solver internals.
    self.assertCountEqual(checkpointed, ['ib_interior_mask', 'ib_boundary'])
    self.assertGreater(len(keys), len(checkpointed))

  def test_flow_variables_are_not_excluded_from_checkpoint(self):
    for key in ('u', 'p', 'bc_u_0_0'):
      self.assertFalse(boundary_condition.exclude_from_checkpoint(key))

  def test_no_states_are_maintained_without_a_nonreflecting_boundary(self):
    params = _make_params_with_ib(_SPONGE)

    self.assertEmpty(boundary_condition.initial_states(params))
    self.assertEmpty(
        boundary_condition.update_additional_states(params, {}, {}, 0)
    )


if __name__ == '__main__':
  absltest.main()
