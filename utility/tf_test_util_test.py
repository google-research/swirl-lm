"""Tests for TensorFlow test utils."""

import unittest
from swirl_lm.utility import tf_test_util as test_util

import tensorflow as tf


class GraphAndEagerDecoratorsTest(tf.test.TestCase):

  def test_run_in_graph_and_eager_modes(self):
    l = []

    def inc(self, with_brackets):
      del self  # self argument is required by run_in_graph_and_eager_modes.
      mode = "eager" if tf.executing_eagerly() else "graph"
      with_brackets = "with_brackets" if with_brackets else "without_brackets"
      l.append((with_brackets, mode))

    f = test_util.run_in_graph_and_eager_modes(inc)
    f(self, with_brackets=False)
    f = test_util.run_in_graph_and_eager_modes()(inc)  # pylint: disable=assignment-from-no-return
    f(self, with_brackets=True)

    self.assertEqual(len(l), 4)
    self.assertEqual(
        set(l), {
            ("with_brackets", "graph"),
            ("with_brackets", "eager"),
            ("without_brackets", "graph"),
            ("without_brackets", "eager"),
        })

  def test_run_in_graph_and_eager_modes_test_class_error(self):
    msg = "`run_in_graph_and_eager_modes` only supports test methods.*"
    with self.assertRaisesRegex(ValueError, msg):

      @test_util.run_in_graph_and_eager_modes()
      class Foo(object):
        pass

      del Foo  # Make pylint unused happy.

  def test_run_all_in_graph_and_eager_modes(self):
    modes = []

    @test_util.run_all_in_graph_and_eager_modes
    class FooTest(tf.test.TestCase):

      def test1(self):
        if tf.executing_eagerly():
          modes.append(("test1", "eager"))
        else:
          modes.append(("test1", "graph"))

      def test2(self):
        if tf.executing_eagerly():
          modes.append(("test2", "eager"))
        else:
          modes.append(("test2", "graph"))

    footest = FooTest()
    for name in dir(footest):
      if not name.startswith(unittest.TestLoader.testMethodPrefix):
        continue
      test_method = getattr(footest, name)
      # Run the test.
      test_method()

    self.assertEqual(
        set(modes), {
            ("test1", "eager"),
            ("test1", "graph"),
            ("test2", "eager"),
            ("test2", "graph"),
        })

  def test_run_all_in_deprecated_graph_mode_only(self):
    modes = []

    @test_util.run_all_in_deprecated_graph_mode_only
    class FooTest(tf.test.TestCase):

      def test1(self):
        if tf.executing_eagerly():
          modes.append(("test1", "eager"))
        else:
          modes.append(("test1", "graph"))

      def test2(self):
        if tf.executing_eagerly():
          modes.append(("test2", "eager"))
        else:
          modes.append(("test2", "graph"))

    footest = FooTest()
    for name in dir(footest):
      if not name.startswith(unittest.TestLoader.testMethodPrefix):
        continue
      test_method = getattr(footest, name)
      # Run the test.
      test_method()

    self.assertEqual(
        set(modes), {
            ("test1", "graph"),
            ("test2", "graph"),
        })

  def test_run_in_eager_and_graph_modes_skip_eager_runs_graph(self):
    modes = []

    def test(self):
      del self
      if tf.executing_eagerly():
        return
      modes.append("eager" if tf.executing_eagerly() else "graph")

    test_util.run_in_graph_and_eager_modes(test)(self)
    self.assertEqual(modes, ["graph"])

  def test_run_in_eager_and_graph_modes_skip_graph_runs_eager(self):
    modes = []

    def test(self):
      del self
      if not tf.executing_eagerly():
        return
      modes.append("eager" if tf.executing_eagerly() else "graph")

    test_util.run_in_graph_and_eager_modes(test)(self)
    self.assertEqual(modes, ["eager"])

  def test_deprecated_graph_mode_runs_graph(self):
    modes = []

    def test(self):
      del self  # self argument is required by deprecated_graph_mode_only.
      modes.append("eager" if tf.executing_eagerly() else "graph")

    test_util.deprecated_graph_mode_only(test)(self)
    self.assertEqual(modes, ["graph"])

  def test_deprecated_graph_mode_works_with_sessions(self):
    test_results = []

    def test(self):
      del self
      sess = tf.compat.v1.Session()
      a_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[])
      b_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[])
      c = a_ph + b_ph
      result = sess.run(c, feed_dict={a_ph: 4., b_ph: 8.})
      test_results.append(result)

    test_util.deprecated_graph_mode_only(test)(self)
    self.assertEqual(test_results[0], 12.)

  def test_run_in_eager_mode_does_not_work_with_sessions(self):
    modes = []

    def test(self):
      if not tf.executing_eagerly():
        return
      modes.append("eager" if tf.executing_eagerly() else "graph")

      msg = r"placeholder.* not compatible with eager execution"
      with self.assertRaisesRegex(RuntimeError, msg):
        sess = tf.compat.v1.Session()
        a_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[])
        b_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[])
        c = a_ph + b_ph
        sess.run(c, feed_dict={a_ph: 4., b_ph: 8.})

    test_util.run_in_graph_and_eager_modes(test)(self)
    self.assertEqual(modes, ["eager"])


if __name__ == "__main__":
  tf.test.main()
