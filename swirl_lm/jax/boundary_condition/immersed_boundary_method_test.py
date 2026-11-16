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
"""Tests for the JAX immersed boundary method module."""


from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
import jax.numpy as jnp
import numpy as np
from swirl_lm.base import parameters_pb2
from swirl_lm.boundary_condition import immersed_boundary_method_pb2
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.boundary_condition import immersed_boundary_method as ibm_lib

_GRID_SHAPE = (8, 8, 8)


def _make_params_with_ib(
    ib_textpb: str,
) -> parameters_lib.SwirlLMParameters:
  """Creates SwirlLMParameters with the given IB config."""
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
  config_proto = text_format.Parse(
      config_text, parameters_pb2.SwirlLMParameters()
  )
  return parameters_lib.SwirlLMParameters(config_proto)


class IbInfoMapTest(absltest.TestCase):
  """Tests for ib_info_map utility function."""

  def test_sponge_info_map(self):
    ib_proto = text_format.Parse(
        """
        sponge {
          damping_coeff: 0.5
          variables { name: "u" value: 0.0 }
          variables { name: "v" value: 0.0 }
        }
        """,
        immersed_boundary_method_pb2.ImmersedBoundaryMethod(),
    )
    result = ibm_lib.ib_info_map(ib_proto)
    self.assertIn('u', result)
    self.assertIn('v', result)
    self.assertEqual(result['u'].name, 'u')
    self.assertAlmostEqual(result['u'].value, 0.0)

  def test_direct_forcing_info_map(self):
    ib_proto = text_format.Parse(
        """
        direct_forcing {
          damping_coeff: 0.1
          variables { name: "u" value: 1.0 }
        }
        """,
        immersed_boundary_method_pb2.ImmersedBoundaryMethod(),
    )
    result = ibm_lib.ib_info_map(ib_proto)
    self.assertIn('u', result)
    self.assertAlmostEqual(result['u'].value, 1.0)


class ImmersedBoundaryMethodSpongeTest(parameterized.TestCase):
  """Tests for sponge (Rayleigh damping) IBM."""

  def _make_sponge_ibm(
      self, damping_coeff: float = 0.5
  ) -> ibm_lib.ImmersedBoundaryMethod:
    params = _make_params_with_ib(f"""
        sponge {{
          damping_coeff: {damping_coeff}
          variables {{ name: "u" value: 0.0 }}
          variables {{ name: "v" value: 0.0 }}
          variables {{ name: "w" value: 0.0 }}
        }}
    """)
    return ibm_lib.ImmersedBoundaryMethod(params)

  def test_type_is_sponge(self):
    ibm = self._make_sponge_ibm()
    self.assertEqual(ibm.type, 'sponge')

  def test_force_name(self):
    self.assertEqual(ibm_lib.ImmersedBoundaryMethod.ib_force_name('u'), 'src_u')

  def test_rhs_name(self):
    self.assertEqual(ibm_lib.ImmersedBoundaryMethod.ib_rhs_name('u'), 'rhs_u')

  def test_update_additional_states_computes_sponge_force(self):
    """Verify Rayleigh damping forces are computed in solid region."""
    ibm = self._make_sponge_ibm(damping_coeff=0.5)

    # Create states with non-zero velocity.
    u = jnp.ones(_GRID_SHAPE) * 2.0
    v = jnp.ones(_GRID_SHAPE) * 3.0
    w = jnp.zeros(_GRID_SHAPE)
    states = {'u': u, 'v': v, 'w': w}

    # Create mask: 1 = fluid, 0 = solid. Make half solid.
    mask = jnp.ones(_GRID_SHAPE)
    mask = mask.at[:4, :, :].set(0.0)  # First 4 slices are solid.

    additional_states = {
        'ib_interior_mask': mask,
        'src_u': jnp.zeros(_GRID_SHAPE),
        'src_v': jnp.zeros(_GRID_SHAPE),
        'src_w': jnp.zeros(_GRID_SHAPE),
    }

    result = ibm.update_additional_states(states, additional_states)

    # In solid region (mask=0), sponge force should be non-zero.
    # Force = -a_max * (value - target) * (1 - mask)
    # For u: -a_max * (2.0 - 0.0) * 1.0 in solid.
    src_u = result['src_u']
    # In solid region, force should be negative (pushing u toward 0).
    np.testing.assert_array_less(src_u[:4, :, :], 0.0)
    # In fluid region (mask=1), force should be zero.
    np.testing.assert_array_equal(src_u[4:, :, :], 0.0)

  def test_update_states_returns_copy(self):
    """For sponge type, update_states returns states unchanged."""
    ibm = self._make_sponge_ibm()
    states = {'u': jnp.ones(_GRID_SHAPE), 'v': jnp.zeros(_GRID_SHAPE)}
    additional_states = {'ib_interior_mask': jnp.ones(_GRID_SHAPE)}
    result = ibm.update_states(states, additional_states)
    np.testing.assert_array_equal(result['u'], states['u'])

  def test_generate_initial_states(self):
    """Verify initial states include mask and force variables."""
    ibm = self._make_sponge_ibm()
    mask = jnp.ones(_GRID_SHAPE)
    result = ibm.generate_initial_states(_GRID_SHAPE, mask)
    self.assertIn('ib_interior_mask', result)
    self.assertIn('src_u', result)
    self.assertIn('src_v', result)
    self.assertIn('src_w', result)


class ImmersedBoundaryMethodDirectForcingTest(parameterized.TestCase):
  """Tests for direct forcing IBM."""

  def _make_direct_forcing_ibm(self) -> ibm_lib.ImmersedBoundaryMethod:
    params = _make_params_with_ib("""
        direct_forcing {
          damping_coeff: 0.1
          variables { name: "u" value: 0.0 }
          variables { name: "v" value: 0.0 }
        }
    """)
    return ibm_lib.ImmersedBoundaryMethod(params)

  def test_type_is_direct_forcing(self):
    ibm = self._make_direct_forcing_ibm()
    self.assertEqual(ibm.type, 'direct_forcing')

  def test_update_forcing_modifies_rhs_in_solid(self):
    """Verify direct forcing updates RHS in solid region."""
    ibm = self._make_direct_forcing_ibm()

    u = jnp.ones(_GRID_SHAPE) * 5.0
    v = jnp.ones(_GRID_SHAPE) * 3.0
    states = {'u': u, 'v': v}

    mask = jnp.ones(_GRID_SHAPE)
    mask = mask.at[:4, :, :].set(0.0)

    additional_states = {
        'ib_interior_mask': mask,
        'rhs_u': jnp.ones(_GRID_SHAPE),
        'rhs_v': jnp.ones(_GRID_SHAPE),
    }

    result = ibm.update_forcing(states, additional_states)

    # In fluid (mask=1), rhs should be preserved (rhs * mask = rhs * 1).
    rhs_u = result['rhs_u']
    np.testing.assert_array_almost_equal(rhs_u[4:, :, :], 1.0)

    # In solid (mask=0), rhs should include forcing term toward target (0.0).
    # rhs = rhs * mask - coeff * (value - target) * (1 - mask)
    # In solid: 0 - coeff * (5.0 - 0.0) * 1.0 = -coeff * 5.0 < 0
    np.testing.assert_array_less(rhs_u[:4, :, :], 0.0)

  def test_generate_initial_states_direct_forcing(self):
    """Verify initial states include RHS variables for direct forcing."""
    ibm = self._make_direct_forcing_ibm()
    mask = jnp.ones(_GRID_SHAPE)
    result = ibm.generate_initial_states(_GRID_SHAPE, mask)
    self.assertIn('ib_interior_mask', result)
    self.assertIn('rhs_u', result)
    self.assertIn('rhs_v', result)


class ImmersedBoundaryMethodGhostPointTest(absltest.TestCase):
  """Tests for the ghost-point IBM configuration."""

  def test_unset_enforcement_selects_direct(self):
    params = _make_params_with_ib('ghost_point { variables { name: "u" } }')
    ibm = ibm_lib.ImmersedBoundaryMethod(params)
    self.assertEqual(ibm.type, 'ghost_point')
    self.assertEqual(
        params.boundary_models.ib.ghost_point.enforcement,
        immersed_boundary_method_pb2.GhostPointMethod.DIRECT,
    )

  def test_unknown_enforcement_raises(self):
    params = _make_params_with_ib("""
        ghost_point {
          enforcement: ENFORCEMENT_UNKNOWN
          variables { name: "u" }
        }
    """)
    with self.assertRaisesRegex(ValueError, 'enforcement'):
      ibm_lib.ImmersedBoundaryMethod(params)


if __name__ == '__main__':
  absltest.main()
