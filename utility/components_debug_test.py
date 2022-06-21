"""Tests for google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.utilities.components_debug."""

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import parameters_pb2
from swirl_lm.utility import components_debug
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.net.proto2.python.public import text_format


@test_util.run_all_in_graph_and_eager_modes
class ComponentsDebugTest(tf.test.TestCase):

  def setUp(self):
    """Initializes the debugging tool."""
    super(ComponentsDebugTest, self).setUp()

    pbtxt = (R'scalars { name: "Y_O"  solve_scalar: true }  '
             R'scalars { name: "T"  solve_scalar: true }  '
             R'scalars { name: "ambient"  solve_scalar: false }  '
             R'additional_state_keys: "src_T"  '
             R'additional_state_keys: "src_Y_O"  '
             R'additional_state_keys: "dbg_T_diff_z"  '
             R'additional_state_keys: "dbg_T_src"  '
             R'additional_state_keys: "dbg_T_D_t"  '
             R'additional_state_keys: "dbg_rhow_diff_z"  '
             R'use_sgs: true  ')
    config = text_format.Parse(pbtxt, parameters_pb2.SwirlLMParameters())
    params = parameters_lib.SwirlLMParameters(config)
    params.nx = 6
    params.ny = 6
    params.nz = 6

    self.tool = components_debug.ComponentsDebug(params)

    self.dbg_names = [
        'dbg_rhou_conv_x',
        'dbg_rhou_conv_y',
        'dbg_rhou_conv_z',
        'dbg_rhou_diff_x',
        'dbg_rhou_diff_y',
        'dbg_rhou_diff_z',
        'dbg_rhou_src',
        'dbg_rhou_gravity',
        'dbg_rhov_conv_x',
        'dbg_rhov_conv_y',
        'dbg_rhov_conv_z',
        'dbg_rhov_diff_x',
        'dbg_rhov_diff_y',
        'dbg_rhov_diff_z',
        'dbg_rhov_src',
        'dbg_rhov_gravity',
        'dbg_rhow_conv_x',
        'dbg_rhow_conv_y',
        'dbg_rhow_conv_z',
        'dbg_rhow_diff_x',
        'dbg_rhow_diff_y',
        'dbg_rhow_diff_z',
        'dbg_rhow_src',
        'dbg_rhow_gravity',
        'dbg_T_conv_x',
        'dbg_T_conv_y',
        'dbg_T_conv_z',
        'dbg_T_diff_x',
        'dbg_T_diff_y',
        'dbg_T_diff_z',
        'dbg_T_src',
        'dbg_T_D_t',
        'dbg_Y_O_conv_x',
        'dbg_Y_O_conv_y',
        'dbg_Y_O_conv_z',
        'dbg_Y_O_diff_x',
        'dbg_Y_O_diff_y',
        'dbg_Y_O_diff_z',
        'dbg_Y_O_src',
        'dbg_Y_O_D_t',
    ]

  def testDebuggingStatesNamesFullGeneratesCompletedListOfNames(self):
    """Checks if all debugging states names are generated."""
    self.assertCountEqual(self.dbg_names, self.tool.debugging_states_names_full)

  def testDebuggingStatesNamesGeneratesSpecifiedListOfNames(self):
    """Checks if debugging states names specified in config are generated."""
    expected = ['dbg_T_diff_z', 'dbg_T_src', 'dbg_T_D_t', 'dbg_rhow_diff_z']
    self.assertCountEqual(expected, self.tool.debugging_states_names())

  def testGenerateInitialStatesFullProvidesAllStatesWithZeros(self):
    """Checks if all debugging states are initialized with zeros."""
    output = self.evaluate(self.tool.generate_initial_states_full())

    self.assertCountEqual(self.dbg_names, output.keys())
    for value in output.values():
      self.assertAllEqual(np.zeros((6, 6, 6), dtype=np.float32), value)

  def testGenerateInitialStatesProvidesSpecifiedStatesWithZeros(self):
    """Checks if debugging states in config are initialized with zeros."""
    output = self.evaluate(self.tool.generate_initial_states())

    expected = ['dbg_T_diff_z', 'dbg_T_src', 'dbg_T_D_t', 'dbg_rhow_diff_z']
    self.assertCountEqual(expected, output.keys())
    for value in output.values():
      self.assertAllEqual(np.zeros((6, 6, 6), dtype=np.float32), value)

  def testUpdateScalarTermsReturnsCorrectDebuggingTerms(self):
    """Checks if correct debugging terms are extracted from inputs."""
    terms = {
        'conv_x': [tf.constant(1.0),],
        'conv_y': [tf.constant(2.0),],
        'conv_z': [tf.constant(-3.0),],
        'diff_x': [tf.constant(4.0),],
        'diff_y': [tf.constant(5.0),],
        'diff_z': [tf.constant(6.0),],
        'source': [tf.constant(7.0),],
    }
    key = 'T'

    with self.subTest(name='WithDtReturnsCorrectTerms'):
      diff_t = [tf.constant(8.0),]

      dbg_terms = self.evaluate(
          self.tool.update_scalar_terms(key, terms, diff_t))

      expected_terms_name = ['dbg_T_diff_z', 'dbg_T_src', 'dbg_T_D_t']
      self.assertCountEqual(expected_terms_name, dbg_terms.keys())

      self.assertAllEqual([6.0,], dbg_terms['dbg_T_diff_z'])
      self.assertAllEqual([7.0,], dbg_terms['dbg_T_src'])
      self.assertAllEqual([8.0,], dbg_terms['dbg_T_D_t'])

    with self.subTest(name='NoDtRaisesValueError'):
      with self.assertRaisesRegex(ValueError, 'not provided'):
        _ = self.tool.update_scalar_terms(key, terms)

  def testUpdateMomentumTermsReturnsCorrectDebuggingTerms(self):
    """Checks if correct debugging terms are extracted for momentum."""
    terms = [
        {
            'conv_x': [tf.constant(1.0),],
            'conv_y': [tf.constant(2.0),],
            'conv_z': [tf.constant(-3.0),],
            'diff_x': [tf.constant(4.0),],
            'diff_y': [tf.constant(5.0),],
            'diff_z': [tf.constant(6.0),],
            'force': [tf.constant(7.0),],
            'gravity': [tf.constant(0.0),],
        },
        {
            'conv_x': [2 * tf.constant(1.0),],
            'conv_y': [2 * tf.constant(2.0),],
            'conv_z': [2 * tf.constant(-3.0),],
            'diff_x': [2 * tf.constant(4.0),],
            'diff_y': [2 * tf.constant(5.0),],
            'diff_z': [2 * tf.constant(6.0),],
            'force': [2 * tf.constant(7.0),],
            'gravity': [tf.constant(0.0),],
        },
        {
            'conv_x': [3 * tf.constant(1.0),],
            'conv_y': [3 * tf.constant(2.0),],
            'conv_z': [3 * tf.constant(-3.0),],
            'diff_x': [3 * tf.constant(4.0),],
            'diff_y': [3 * tf.constant(5.0),],
            'diff_z': [3 * tf.constant(6.0),],
            'force': [3 * tf.constant(7.0),],
            'gravity': [tf.constant(9.0),],
        },
    ]

    dbg_terms = self.evaluate(self.tool.update_momentum_terms(terms))

    self.assertCountEqual(['dbg_rhow_diff_z'], dbg_terms.keys())
    self.assertAllEqual([18.0,], dbg_terms['dbg_rhow_diff_z'])


if __name__ == '__main__':
  tf.test.main()
