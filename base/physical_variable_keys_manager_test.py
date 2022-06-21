"""Tests for physical_variable_keys_manager."""

import numpy as np
from swirl_lm.base import physical_variable_keys_manager
from swirl_lm.communication import halo_exchange
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf


@test_util.run_all_in_graph_and_eager_modes
class AdditionalStatesKeyManagerTest(tf.test.TestCase):

  def testBoundaryConditionKeysHelperParsesKeysCorrectly(self):
    """Checks if the `BoundaryConditionKeysHelper` processes keys correctly."""
    helper = physical_variable_keys_manager.BoundaryConditionKeysHelper()

    with self.subTest(name='CorrectKeyReturnsCorrectInfoTuple'):
      key = 'bc_w_2_0'
      expected = ('w', 2, 0)
      result = helper.parse_key(key)
      self.assertAllEqual(expected, result)

    with self.subTest(name='IncorrectBoundaryKeyReturnsNone'):
      key = 'src_rho'
      self.assertIsNone(helper.parse_key(key))

  def testBoundaryConditionKeysHelperUpdatesBCCorrectly(self):
    """Checks if the `BoundaryConditionKeysHelper` updates BC correctly."""
    helper = physical_variable_keys_manager.BoundaryConditionKeysHelper()

    bc_u = np.ones((6, 2, 6), dtype=np.float32)
    bc_u[:, 1, :] = 2.0
    bc_u = tf.unstack(tf.convert_to_tensor(bc_u))

    bc_w = np.ones((2, 6, 6), dtype=np.float32)
    bc_w[1, ...] = 2.0
    bc_w = tf.unstack(tf.convert_to_tensor(bc_w))

    additional_states = {
        'bc_Z_1_0': bc_u,
        'bc_u_0_0': bc_u,
        'bc_w_2_0': bc_w,
    }
    bc = {
        'u': [[(halo_exchange.BCType.DIRICHLET, 0.0),
               (halo_exchange.BCType.NEUMANN, 0.0)], [None, None],
              [(halo_exchange.BCType.DIRICHLET, 0.0),
               (halo_exchange.BCType.NEUMANN, 0.0)]],
        'w': [[(halo_exchange.BCType.DIRICHLET, 0.0),
               (halo_exchange.BCType.NEUMANN, 0.0)], [None, None],
              [(halo_exchange.BCType.DIRICHLET, 0.0),
               (halo_exchange.BCType.NEUMANN, 0.0)]],
    }
    halo_width = 2

    expected = {
        'u': [
            [
                (halo_exchange.BCType.DIRICHLET, [
                    [np.ones((1, 6), dtype=np.float32)] * 6,
                    [2.0 * np.ones((1, 6), dtype=np.float32)] * 6,
                ]),
                (halo_exchange.BCType.NEUMANN, 0.0),
            ],
            [None, None],
            [(halo_exchange.BCType.DIRICHLET, 0.0),
             (halo_exchange.BCType.NEUMANN, 0.0)],
        ],
        'w': [
            [
                (halo_exchange.BCType.DIRICHLET, 0.0),
                (halo_exchange.BCType.NEUMANN, 0.0),
            ],
            [None, None],
            [(halo_exchange.BCType.DIRICHLET, [
                np.ones((6, 6), dtype=np.float32),
                2.0 * np.ones((6, 6), dtype=np.float32),
            ]), (halo_exchange.BCType.NEUMANN, 0.0)],
        ],
    }

    bc_new = helper.update_helper_variable_from_additional_states(
        additional_states, halo_width, bc)

    bc_u_0_0 = self.evaluate(bc_new['u'][0][0][1])
    bc_w_2_0 = self.evaluate(bc_new['w'][2][0][1])

    self.assertAllEqual(expected['u'][0][0][1], bc_u_0_0)
    self.assertAllEqual(expected['w'][2][0][1], bc_w_2_0)

  def testBoundaryConditionKeysHelperGeneratesCorrectBCKeys(self):
    """Checks if keys for boundary conditions can be generated correctly."""
    helper = physical_variable_keys_manager.BoundaryConditionKeysHelper()

    with self.subTest(name='ValidInputsReturnsCorrectKey'):
      expected = 'bc_rho_f_2_0'
      actual = helper.generate_bc_key('rho_f', 2, 0)
      self.assertEqual(expected, actual)

    with self.subTest(name='InvalidDimRaisesValueError'):
      with self.assertRaisesRegex(ValueError,
                                  'Dimension should be one of 0, 1, and 2'):
        _ = helper.generate_bc_key('u', 6, 1)

    with self.subTest(name='InvalidFaceRaisesValueError'):
      with self.assertRaisesRegex(ValueError, 'Face should be one of 0 and 1'):
        _ = helper.generate_bc_key('w', 0, 2)

  def testSourceKeysHelperParsesKeysCorrectly(self):
    """Checks if the `SourceKeysHelper` processes keys correctly."""
    helper = physical_variable_keys_manager.SourceKeysHelper()

    with self.subTest(name='CorrectKeyReturnsCorrectInfoTuple'):
      key = 'src_rho_f'
      expected = 'rho_f'
      result = helper.parse_key(key)
      self.assertAllEqual(expected, result)

    with self.subTest(name='IncorrectBoundaryKeyReturnsNone'):
      key = 'bc_u_0_1'
      self.assertIsNone(helper.parse_key(key))

  def testSourceKeysHelperUpdatesSourceTermCorrectly(self):
    """Checks if the `BoundaryConditionKeysHelper` updates BC correctly."""
    helper = physical_variable_keys_manager.SourceKeysHelper()
    additional_states = {
        'src_rho_f': tf.unstack(6.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
    }

    expected = {
        'rho_f': [6.0 * np.ones((6, 6), dtype=np.float32)] * 6,
    }

    src = self.evaluate(
        helper.update_helper_variable_from_additional_states(
            additional_states))

    self.assertEqual(expected.keys(), src.keys())
    self.assertAllEqual(expected['rho_f'], src['rho_f'])

  def testSourceKeysHelperGeneratesCorrectSourceKey(self):
    """Checks if `generate_src_key` generates correct source key."""
    helper = physical_variable_keys_manager.SourceKeysHelper()
    varname = 'rho_f'

    src_key = helper.generate_src_key(varname)

    self.assertEqual('src_rho_f', src_key)

  def testPhysicalVariableKeysManagerProvidesCorrectHelperObject(self):
    """Checks is the manager creates helper correctly."""
    with self.subTest(name='BoundaryConditionKeysHelper'):
      helper = physical_variable_keys_manager.physical_variable_keys_manager(
          physical_variable_keys_manager.PhysicalVariablesType
          .BOUNDARY_CONDITION)
      self.assertEqual(
          type(helper),
          physical_variable_keys_manager.BoundaryConditionKeysHelper)

    with self.subTest(name='SourceKeysHelper'):
      helper = physical_variable_keys_manager.physical_variable_keys_manager(
          physical_variable_keys_manager.PhysicalVariablesType.SOURCE)
      self.assertEqual(
          type(helper), physical_variable_keys_manager.SourceKeysHelper)


if __name__ == '__main__':
  tf.test.main()
