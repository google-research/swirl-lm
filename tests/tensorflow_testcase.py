from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

class TensorflowTestCase(parameterized.TestCase):
  def assertAllClose(self,
                     tensor_A : tf.Tensor,
                     tensor_B : tf.Tensor,
                     atol : tf.Tensor = tf.constant(1e-7)
      ) -> None:
    """Checks if two tensors have values near each other.

      Args:
          tensor_A: first tensor to test
          tensor_B: second tensor to test
          atol: absolute tolerance of the test. The difference must be less than
            this

    """
    abs_diff = tf.math.abs(tensor_A - tensor_B)
    within_tol = tf.less_equal(abs_diff, atol)
    self.assertTrue(tf.reduce_all(within_tol))

  def assertAllEqual(self,
                     tensor_A : tf.Tensor,
                     tensor_B : tf.Tensor
      ) -> None:
    """Checks if two tensors have equal values.

      Args:
          tensor_A: first tensor to test
          tensor_B: second tensor to test
    """
    if tensor_B.dtype is not tensor_A.dtype:
      raise TypeError("Tensor argument types should be the same, tensor A type : " \
                      + tensor_A.dtype.name \
                      + ", tensor B type: " + tensor_B.dtype.name)
    within_tol = tf.equal(tensor_A, tensor_B)
    self.assertTrue(tf.reduce_all(within_tol))

  def assertAllNonzero(self,
                       tensor : tf.Tensor
      ) -> None:
    """Checks if the input tensor has all nonzero values.

      Args:
          tensor: tensor to test
    """
    within_tol = tf.greater(tf.abs(tensor), tf.constant(0.0))
    self.assertTrue(tf.reduce_all(within_tol))
