"""Tests for TensorFlow test utils in TF1 mode."""

import unittest
from swirl_lm.utility import tf_test_util as test_util

import tensorflow as tf


class GraphAndEagerDecoratorsTf1Test(tf.test.TestCase):

  @unittest.expectedFailure
  @test_util.run_in_graph_and_eager_modes
  def test_simple_failure(self):
    self.assertEqual(self.evaluate(tf.constant(1)), 2)

  def test_graph_and_eager_mode_runs_test_only_once_in_tf1(self):
    self.assertFalse(tf.executing_eagerly())

    invocation_count = 0
    def test(self):
      del self
      nonlocal invocation_count
      invocation_count += 1

    test_util.run_in_graph_and_eager_modes(test)(self)
    self.assertEqual(invocation_count, 1)

if __name__ == "__main__":
  tf.test.main()
