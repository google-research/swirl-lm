"""Tests for outflow."""

from typing import List
from unittest import mock

from absl import flags
import numpy as np
from swirl_lm.boundary_condition import outflow
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
import tensorflow as tf
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized


FLAGS = flags.FLAGS

_REPLICAS = [
    np.array([[[0, 1]]]),
    np.array([[[0], [1]]]),
    np.array([[[0]], [[1]]])
]


def _split_by_two_replicas(
    data: np.ndarray,
    replicas: np.ndarray,
) -> List[np.ndarray]:
  """Splits a 3D numpy array into 2 according to the partitioning."""
  nz, nx, ny = data.shape
  if replicas.shape[0] == 2:
    return [data[:, :nx // 2, :], data[:, nx // 2:, :]]
  elif replicas.shape[1] == 2:
    return [data[..., :ny // 2], data[..., ny // 2:]]
  elif replicas.shape[2] == 2:
    return [data[:nz // 2, ...], data[nz // 2:, ...]]


def _merge_from_two_replicas(
    data_1: np.ndarray,
    data_2: np.ndarray,
    replicas: np.ndarray,
) -> np.ndarray:
  """Merges 2 3D numpy array into 1 according to the partitioning."""
  if replicas.shape[0] == 2:
    return np.concatenate([data_1, data_2], axis=1)
  elif replicas.shape[1] == 2:
    return np.concatenate([data_1, data_2], axis=2)
  elif replicas.shape[2] == 2:
    return np.concatenate([data_1, data_2], axis=0)


def _input_tensor_partition(
    data: np.ndarray,
    replicas: np.ndarray,
) -> List[List[tf.Tensor]]:
  """Converts a 3D numpy array into two lists of `tf.Tensor`."""
  data_1, data_2 = _split_by_two_replicas(data, replicas)
  return [
      tf.unstack(tf.convert_to_tensor(data_1)),
      tf.unstack(tf.convert_to_tensor(data_2))
  ]


class OutflowTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(*zip(_REPLICAS))
  def testOutflowProducesCorrectBoundaryConditionAtOutlet(self, replicas):
    """Checks if the outflow boundary condition is computed correctly."""
    u = np.ones((16, 16, 16), dtype=np.float32)
    rho = np.ones((16, 16, 16), dtype=np.float32)
    rho[:, 12:, :] = 0.5
    z = np.ones((16, 16, 16), dtype=np.float32)
    z[:, 12:, :] = 0.1
    bc_z = np.zeros((16, 16, 16), dtype=np.float32)
    bc_u = 0.5 * np.ones((16, 16, 16), dtype=np.float32)

    u_1, u_2 = _input_tensor_partition(u, replicas)
    rho_1, rho_2 = _input_tensor_partition(rho, replicas)
    z_1, z_2 = _input_tensor_partition(z, replicas)
    bc_z_1, bc_z_2 = _input_tensor_partition(bc_z, replicas)
    bc_u_1, bc_u_2 = _input_tensor_partition(bc_u, replicas)

    inputs = [
        [tf.constant(0), tf.constant(1)],
        [{
            'rho': rho_1,
            'u': u_1,
            'Z': z_1
        }, {
            'rho': rho_2,
            'u': u_2,
            'Z': z_2
        }],
        [{
            'bc_Z_0_1': bc_z_1,
            'bc_u_0_1': bc_u_1
        }, {
            'bc_Z_0_1': bc_z_2,
            'bc_u_0_1': bc_u_2
        }],
    ]

    outflow_fn = outflow.outflow_boundary_condition()

    params = mock.create_autospec(
        grid_parametrization.GridParametrization, instance=True)
    params.dx = 0.05
    params.dt = 0.01
    params.halo_width = 2

    def device_fn(replica_id, states, additional_states):
      """Wraps the outflow boundary condition function for TPU."""
      kernel_op = get_kernel_fn.ApplyKernelConvOp(4)

      return outflow_fn(kernel_op, replica_id, replicas, states,
                        additional_states, params)

    runner = TpuRunner(replicas=replicas)
    additional_states_new = runner.run(device_fn, *inputs)

    # Compares the domain that contains the outflow boundary condition plane.
    with self.subTest(name='BoundaryConditionOfZIs0.02'):
      output = _merge_from_two_replicas(additional_states_new[0]['bc_Z_0_1'],
                                        additional_states_new[1]['bc_Z_0_1'],
                                        replicas)

      expected = 0.02 * np.ones((16, 3, 16), dtype=np.float32)

      self.assertAllClose(expected, output[:, -3:, :])

    with self.subTest(name='BoundaryConditionOfUIs1.2'):
      output = _merge_from_two_replicas(additional_states_new[0]['bc_u_0_1'],
                                        additional_states_new[1]['bc_u_0_1'],
                                        replicas)

      expected = 1.2 * np.ones((16, 3, 16), dtype=np.float32)

      self.assertAllClose(expected, output[:, -3:, :])


if __name__ == '__main__':
  tf.test.main()
