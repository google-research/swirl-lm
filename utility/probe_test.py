"""Tests for probe."""

from absl import flags
import numpy as np
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import probe
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_parameters_pb2

FLAGS = flags.FLAGS


class ProbeTest(tf.test.TestCase):

  def setUp(self):
    """Initializes the probe object."""
    super().setUp()

    FLAGS.nx = 8
    FLAGS.ny = 8
    FLAGS.nz = 8
    FLAGS.cx = 2
    FLAGS.cy = 2
    FLAGS.cz = 2
    FLAGS.lx = 8.0
    FLAGS.ly = 8.0
    FLAGS.lz = 8.0
    FLAGS.dt = 0.01
    FLAGS.halo_width = 2

    textpb = (R'probe { '
              '  location {dim_0: 2.0 dim_1: 2.0 dim_2: 2.0 } '
              '  location {dim_0: 2.0 dim_1: 6.0 dim_2: 2.0 } '
              '  location {dim_0: 2.0 dim_1: 2.0 dim_2: 6.0 } '
              '  location {dim_0: 2.0 dim_1: 6.0 dim_2: 6.0 } '
              '  location {dim_0: 6.0 dim_1: 2.0 dim_2: 2.0 } '
              '  location {dim_0: 6.0 dim_1: 6.0 dim_2: 2.0 } '
              '  location {dim_0: 6.0 dim_1: 2.0 dim_2: 6.0 } '
              '  location {dim_0: 6.0 dim_1: 6.0 dim_2: 6.0 } '
              '  variable_name: "u" '
              '  variable_name: "T" '
              '  nt: 10 '
              '  start_step_id: 6 '
              '}')
    config = text_format.Parse(
        textpb,
        incompressible_structured_mesh_parameters_pb2
        .IncompressibleNavierStokesParameters())
    self.params = (
        incompressible_structured_mesh_config
        .IncompressibleNavierStokesParameters(config))
    self.model = probe.Probe(self.params)

  def testTheProbeLibraryInitializedCorrectly(self):
    """Checks if the indices in probe object are initialized correctly."""
    with self.subTest(name='VariablesAreUandT'):
      expected = ('u', 'T')
      self.assertAllEqual(expected, self.model.variable_names)

    with self.subTest(name='CoreIndicesAreCorrect'):
      expected = [
          [0, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
          [0, 1, 1],
          [1, 0, 0],
          [1, 1, 0],
          [1, 0, 1],
          [1, 1, 1],
      ]
      self.assertAllEqual(expected, self.model.c_indices)

    with self.subTest(name='LocalIndicesAreCorrect'):
      expected = [[3, 3, 3],] * 8
      self.assertAllEqual(expected, self.model.indices)

  def testProbeInitializesProbeVariableAsZerosWithCorrectShape(self):
    """Checks if probe arrays with size (10, 3) are created for `u` and `T`."""
    replica_id = tf.constant(0)
    coordinates = [0, 0, 0]
    probe_states = self.model.initialization(replica_id, coordinates)

    expected = {
        'PROBE_0': tf.zeros((10, 3), dtype=tf.float32),
        'PROBE_1': tf.zeros((10, 3), dtype=tf.float32),
        'PROBE_2': tf.zeros((10, 3), dtype=tf.float32),
        'PROBE_3': tf.zeros((10, 3), dtype=tf.float32),
        'PROBE_4': tf.zeros((10, 3), dtype=tf.float32),
        'PROBE_5': tf.zeros((10, 3), dtype=tf.float32),
        'PROBE_6': tf.zeros((10, 3), dtype=tf.float32),
        'PROBE_7': tf.zeros((10, 3), dtype=tf.float32),
    }
    self.assertDictEqual(expected, probe_states)

  def testProbeProvidesValuesAtCorrectNodes(self):
    """Checks if probes gets values from distributed 3D tensors correctly."""
    u = 0.1 * np.reshape(np.arange(16**3, dtype=np.float32), (16, 16, 16))
    p = np.reshape(np.arange(16**3, dtype=np.float32), (16, 16, 16))
    t = 10.0 * np.reshape(np.arange(16**3, dtype=np.float32), (16, 16, 16))

    # Inputs are partitioned following the order of x-y-z. Replicas are created
    # correspondingly.
    def partition_2x2x2(f):
      """Splits a 3D tensor into equal shaped 3D tensors in a 2x2x2 topology."""
      nx, ny, nz = f.shape
      return (
          f[:nx // 2, :ny // 2, :nz // 2],
          f[nx // 2:, :ny // 2, :nz // 2],
          f[:nx // 2, ny // 2:, :nz // 2],
          f[nx // 2:, ny // 2:, :nz // 2],
          f[:nx // 2, :ny // 2, nz // 2:],
          f[nx // 2:, :ny // 2, nz // 2:],
          f[:nx // 2, ny // 2:, nz // 2:],
          f[nx // 2:, ny // 2:, nz // 2:],
      )

    def to_list_tensor(f):
      """Converts a x-y-z tensor into z-x-y list of 2D tenors."""
      return tf.unstack(tf.convert_to_tensor(np.transpose(f, (2, 0, 1))))

    var_local = {
        'p': partition_2x2x2(p),
        'T': partition_2x2x2(t),
        'u': partition_2x2x2(u),
    }
    states = [{key: to_list_tensor(val[i])
               for key, val in var_local.items()}
              for i in range(8)]
    additional_states = [{
        'PROBE_0': tf.zeros((10, 3), dtype=tf.float32),
        'PROBE_1': tf.zeros((10, 3), dtype=tf.float32),
        'PROBE_2': tf.zeros((10, 3), dtype=tf.float32),
        'PROBE_3': tf.zeros((10, 3), dtype=tf.float32),
        'PROBE_4': tf.zeros((10, 3), dtype=tf.float32),
        'PROBE_5': tf.zeros((10, 3), dtype=tf.float32),
        'PROBE_6': tf.zeros((10, 3), dtype=tf.float32),
        'PROBE_7': tf.zeros((10, 3), dtype=tf.float32),
    },] * 8
    replica_id = [tf.constant(i) for i in range(8)]
    # Replicas are created such that the `replica_id` increases following the
    # x-y-z order, which is consistent with the way that 3D input tensors are
    # partitioned.
    replicas = np.transpose(np.reshape(np.arange(8), (2, 2, 2)), (2, 1, 0))
    kernel_op = get_kernel_fn.ApplyKernelConvOp(4)
    step_id = 9

    def device_fn(replica_id, states, additional_states):
      """Wraps the additional_states_update_fn that runs on TPU."""
      return self.model.additional_states_update_fn(kernel_op, replica_id,
                                                    replicas, step_id, states,
                                                    additional_states,
                                                    self.params)

    runner = TpuRunner(replicas=replicas)
    res = runner.run(device_fn, replica_id, states, additional_states)

    expected = {
        'PROBE_0': np.zeros((10, 3), dtype=np.float32),
        'PROBE_1': np.zeros((10, 3), dtype=np.float32),
        'PROBE_2': np.zeros((10, 3), dtype=np.float32),
        'PROBE_3': np.zeros((10, 3), dtype=np.float32),
        'PROBE_4': np.zeros((10, 3), dtype=np.float32),
        'PROBE_5': np.zeros((10, 3), dtype=np.float32),
        'PROBE_6': np.zeros((10, 3), dtype=np.float32),
        'PROBE_7': np.zeros((10, 3), dtype=np.float32),
    }
    expected['PROBE_0'][3, :] = [0.09, 819e-1, 819e1]
    expected['PROBE_1'][3, :] = [0.09, 947e-1, 947e1]
    expected['PROBE_2'][3, :] = [0.09, 827e-1, 827e1]
    expected['PROBE_3'][3, :] = [0.09, 955e-1, 955e1]
    expected['PROBE_4'][3, :] = [0.09, 2867e-1, 2867e1]
    expected['PROBE_5'][3, :] = [0.09, 2995e-1, 2995e1]
    expected['PROBE_6'][3, :] = [0.09, 2875e-1, 2875e1]
    expected['PROBE_7'][3, :] = [0.09, 3003e-1, 3003e1]

    for i in range(len(res)):
      with self.subTest(name='ProbeValueCorrectInReplica{}'.format(i)):
        self.assertDictEqual(expected, res[i])


if __name__ == '__main__':
  tf.test.main()
