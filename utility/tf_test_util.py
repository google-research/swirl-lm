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

"""Utilities for testing TensorFlow code."""

import inspect
import unittest

import tensorflow as tf


def run_in_graph_and_eager_modes(func=None):
  """Executes the decorated test with and without enabling eager execution.

  This function is adapted from tensorflow/python/framework/test_util.py.
  Compared to the original version, this code is a lightweight version which
  does not allow running on GPU, or tf.Session config overrides. Furthermore, we
  removed some checks specific to TF internal memory management of EagerTensors.

  This function returns a decorator intended to be applied to test methods in
  a `tf.test.TestCase` class. Doing so will cause the contents of the test
  method to be executed twice - once normally, and once with eager execution
  enabled. This allows unittests to confirm the equivalence between eager
  and graph execution (see `tf.compat.v1.enable_eager_execution`).

  For example, consider the following unittest:

  ```python
  class MyTests(tf.test.TestCase):

    @run_in_graph_and_eager_modes
    def test_foo(self):
      x = tf.constant([1, 2])
      y = tf.constant([3, 4])
      z = tf.add(x, y)
      self.assertAllEqual([4, 6], self.evaluate(z))

  if __name__ == "__main__":
    tf.test.main()
  ```

  This test validates that `tf.add()` has the same behavior when computed with
  eager execution enabled as it does when constructing a TensorFlow graph and
  executing the `z` tensor in a session.

  See go/tf-test-decorator-cheatsheet for the decorators to use in different
  v1/v2/eager/graph combinations.


  Args:
    func: function to be annotated. If `func` is None, this method returns a
      decorator the can be applied to a function. If `func` is not None this
      returns the decorator applied to `func`.

  Returns:
    Returns a decorator that will run the decorated test method twice:
    once by constructing and executing a graph in a session and once with
    eager execution enabled.
  """

  def decorator(f):
    if inspect.isclass(f):
      raise ValueError(
          '`run_in_graph_and_eager_modes` only supports test methods. '
          'Did you mean to use `run_all_in_graph_and_eager_modes`?')

    def decorated(self, *args, **kwargs):
      if not tf.executing_eagerly():
        raise ValueError('Must be executing eagerly when using the '
                         'run_in_graph_and_eager_modes decorator.')

      # Run eager block
      f(self, *args, **kwargs)
      self.tearDown()

      # Run in graph mode block
      with tf.Graph().as_default():
        self.setUp()
        with self.test_session():
          f(self, *args, **kwargs)

    return decorated

  if func is not None:
    return decorator(func)

  return decorator


def deprecated_graph_mode_only(func=None):
  """Executes the decorated test in graph mode.

  This function is adapted from tensorflow/python/framework/test_util.py. The
  main change is that instead of using `graph_mode` context manager internal to
  TF code, this function uses the TF Migration guide's approach to migrating
  existing TF1 code to TF2. This consists of wrapping the existing test method
  in a `tf.Graph().as_default()` context.

  This function returns a decorator intended to be applied to tests that are not
  compatible with eager mode. When this decorator is applied, the test body will
  be run in an environment where API calls construct graphs instead of executing
  eagerly.

  Args:
    func: function to be annotated. If `func` is None, this method returns a
      decorator the can be applied to a function. If `func` is not None this
      returns the decorator applied to `func`.

  Returns:
    Returns a decorator that will run the decorated test method in graph mode.
  """

  def decorator(f):
    if inspect.isclass(f):
      setup = f.__dict__.get('setUp')
      if setup is not None:
        setattr(f, 'setUp', decorator(setup))

      for name, value in f.__dict__.copy().items():
        if (callable(value) and
            name.startswith(unittest.TestLoader.testMethodPrefix)):
          setattr(f, name, decorator(value))

      return f

    def decorated(self, *args, **kwargs):
      if tf.executing_eagerly():
        with tf.Graph().as_default():
          return f(self, *args, **kwargs)
      else:
        return f(self, *args, **kwargs)

    return decorated

  if func is not None:
    return decorator(func)

  return decorator


def run_all_in_graph_and_eager_modes(cls):
  """Executes all test methods in the given class with and without eager."""
  base_decorator = run_in_graph_and_eager_modes
  for name in dir(cls):
    if (not name.startswith(unittest.TestLoader.testMethodPrefix) or
        name.startswith('testSkipEager') or
        name.startswith('test_skip_eager') or name == 'test_session'):
      continue
    value = getattr(cls, name, None)
    if callable(value):
      setattr(cls, name, base_decorator(value))
  return cls


def run_all_in_deprecated_graph_mode_only(cls):
  """Executes all tests in a class in graph mode."""
  base_decorator = deprecated_graph_mode_only
  for name in dir(cls):
    if (not name.startswith(unittest.TestLoader.testMethodPrefix) or
        name == 'test_session'):
      continue
    value = getattr(cls, name, None)
    if callable(value):
      setattr(cls, name, base_decorator(value))
  return cls
