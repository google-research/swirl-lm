"""Utilities for testing TensorFlow code on TPU in both TF1 and TF2 modes."""

from typing import Any, Callable

import numpy as np
import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.framework import util
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner


def run_on_tpu_in_test(
    test_case: tf.test.TestCase,
    replicas: np.ndarray,
    fn: Callable[..., Any],
    *args: Any,
) -> Any:
  """Runs a function on TPU in a test, in either TF2 or TF1 mode.

  Args:
    test_case: The TestCase the caller is part of.
    replicas: Numpy array or nested list with the replica indices `(0..n-1)`
      assigned across dimensions.
    fn: The function to run.
    *args: Arguments for fn, each of length `num_replicas`.

  Returns:
    The gathered results of the computation.
  """
  if tf.executing_eagerly():
    # TF2
    return TpuRunner(replicas=replicas).run(fn, *args)
  else:
    # TF1
    computation_shape = np.array(replicas.shape)
    with test_case.session() as sess:
      topology = tf.tpu.experimental.Topology(
          sess.run(tf.compat.v1.tpu.initialize_system()))
      device_assignment, _ = util.tpu_device_assignment(computation_shape,
                                                        topology)
      inputs = [list(x) for x in zip(*args)]  # transpose args
      tpu_fn = tf.compat.v1.tpu.replicate(
          fn, inputs=inputs, device_assignment=device_assignment)

      return sess.run(tpu_fn)
