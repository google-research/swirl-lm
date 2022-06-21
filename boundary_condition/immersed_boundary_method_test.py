"""Tests for swirl_lm.boundary_condition.immersed_boundary_method."""

from absl import flags
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import parameters_pb2
from swirl_lm.boundary_condition import immersed_boundary_method
from swirl_lm.communication import halo_exchange
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized

FLAGS = flags.FLAGS

_REPLICAS = [
    np.array([[[0], [1]]]), np.array([[[0, 1]]]), np.array([[[0]], [[1]]])
]


_PBTXT = [
    (R'boundary_models {  '
     R'  ib {  '
     R'    sponge {  '
     R'      variables { name: "u" value: 0.0 override: true }  '
     R'      variables { name: "T" value: 300.0 override: false }  '
     R'      damping_coeff: 0.5  '
     R'    }  '
     R'  }  '
     R'}  '),
    (R'boundary_models {  '
     R'  ib {  '
     R'    direct_forcing {  '
     R'      variables { name: "u" value: 0.0 override: true }  '
     R'      variables { name: "T" value: 300.0 override: false }  '
     R'      damping_coeff: 0.5  '
     R'    }  '
     R'  }  '
     R'}  '),
    (R'periodic {  '
     R'  dim_0: false dim_1: false dim_2: false  '
     R'}  '
     R'boundary_models {  '
     R'  ib {  '
     R'    cartesian_grid {  '
     R'      variables { name: "u" value: 0.0 bc: DIRICHLET }  '
     R'    }  '
     R'  }  '
     R'}  '),
    (R'periodic {  '
     R'  dim_0: false dim_1: false dim_2: true  '
     R'}  '
     R'boundary_models {  '
     R'  ib {  '
     R'    mac {  '
     R'      variables { name: "u" value: 10.0 bc: NEUMANN }  '
     R'    }  '
     R'  }  '
     R'}  '),
]


class ImmersedBoundaryMethodTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ImmersedBoundaryMethodTest, self).setUp()

    self.mask = [
        tf.constant([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
                    tf.float32),
    ] * 4
    self.boundary = [
        tf.constant([[0, 0, 0, 0], [1. / 4., 1. / 5., 1. / 5., 1. / 4.],
                     [0, 0, 0, 0], [0, 0, 0, 0]], tf.float32),
    ] * 4
    self.bc = [[(halo_exchange.BCType.NO_TOUCH, 0.),
                (halo_exchange.BCType.NO_TOUCH, 0.)]] * 3

  def _run_on_tpu(self, replicas, inputs, device_fn):
    """Wrapper for the Cartesian grid method."""
    computation_shape = np.array(replicas.shape)
    # transpose inputs.
    device_inputs = [list(x) for x in zip(*inputs)]
    runner = TpuRunner(computation_shape=computation_shape)
    return runner.run(device_fn, *device_inputs)

  def _set_up_immersed_boundary_method(self, pbtxt):
    """Generates the `ImmersedBoundaryMethod` object."""
    config = text_format.Parse(pbtxt, parameters_pb2.SwirlLMParameters())
    params = parameters_lib.SwirlLMParameters(config)
    FLAGS.dt = 0.01
    params.halo_width = 1
    return immersed_boundary_method.immersed_boundary_method_factory(params)

  def testGetFluidSolidInterfaceZGenerateCorrectBinaryMask(self):
    """Checks if the fluid-solid interface mask is retrieved correctly."""
    ib_interior_mask = np.ones((32, 8, 8), dtype=np.float32)
    for i in range(8):
      for j in range(8):
        ib_interior_mask[:8 + i + j, i, j] = 0.0
    ib_interior_mask = tf.unstack(tf.convert_to_tensor(ib_interior_mask))

    kernel_op = get_kernel_fn.ApplyKernelConvOp(8)
    replicas = np.array([[[0]]])

    def device_fn(replica_id, mask):
      """A wrapper for the get_fluid_solid_interface_z on TPU."""
      return immersed_boundary_method.get_fluid_solid_interface_z(
          kernel_op, replica_id, replicas, mask, 2)

    inputs = [[tf.constant(0), ib_interior_mask]]

    output = self._run_on_tpu(replicas, inputs, device_fn)
    mask = output[0]

    expected = np.zeros((32, 8, 8))
    for i in range(8):
      for j in range(8):
        expected[8 + i + j, i, j] = 1.0

    self.assertAllEqual(expected, mask)

  def testGetFluidSolidInterfaceValueZProvidesCorrectFluidLayerValues(self):
    """Checks if values are retrieved correctly at the fluid-solid interface."""
    ib_interior_mask = np.ones((32, 8, 8), dtype=np.float32)
    for i in range(8):
      for j in range(8):
        ib_interior_mask[:8 + i + j, i, j] = 0.0
    ib_interior_mask = tf.unstack(tf.convert_to_tensor(ib_interior_mask))

    values = np.reshape(np.arange(32 * 8 * 8, dtype=np.float32), (32, 8, 8))

    kernel_op = get_kernel_fn.ApplyKernelConvOp(8)
    replicas = np.array([[[0]]])

    def mask_fn(replica_id, mask):
      """A wrapper for the get_fluid_solid_interface_z on TPU."""
      return immersed_boundary_method.get_fluid_solid_interface_z(
          kernel_op, replica_id, replicas, mask, 2)

    inputs = [[tf.constant(0), ib_interior_mask]]

    output = self._run_on_tpu(replicas, inputs, mask_fn)
    ib_boudnary_mask = output[0]

    def device_fn(values, mask):
      """A wrapper for the get_fluid_solid_interface_value_z on TPU."""
      return immersed_boundary_method.get_fluid_solid_interface_value_z(
          replicas, values, mask)

    inputs = [[tf.unstack(tf.convert_to_tensor(values)), ib_boudnary_mask]]

    output = self._run_on_tpu(replicas, inputs, device_fn)
    target_value = np.squeeze(output[0])

    expected = np.zeros((8, 8), dtype=np.float32)
    for i in range(8):
      for j in range(8):
        expected[i, j] = values[8 + i + j, i, j]

    self.assertAllEqual(expected, target_value)

  @test_util.run_in_graph_and_eager_modes
  def testUpdateCartesianGridMethodBoundaryCoefficients(self):
    """Checks if the boundary coefficients update is applied correctly."""
    input_boundary = [tf.constant([[0, 0, 0, 0],
                                   [1, 1, 1, 1],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]], tf.float32),] * 4
    boundary_updated = (
        immersed_boundary_method
        .update_cartesian_grid_method_boundary_coefficients(
            input_boundary,
            self.mask,
            kernel_op=get_kernel_fn.ApplyKernelMulOp(4, 4),
        ))
    self.assertLen(boundary_updated, 4)
    boundary_edge = [
        tf.constant([[0, 0, 0, 0], [1. / 3., 1. / 4., 1. / 4., 1. / 3.],
                     [0, 0, 0, 0], [0, 0, 0, 0]], tf.float32),
    ]
    expected = boundary_edge + self.boundary[1:-1] + boundary_edge
    result = self.evaluate(boundary_updated)
    for i in range(len(input_boundary)):
      self.assertAllEqual(result[i], expected[i])

  @parameterized.parameters(*zip(_REPLICAS))
  def testApplyCartesianGridMethodMirrorFlow(self, replicas):
    """Checks if the Cartesian grid method updates Dirichlet BC correctly."""
    computation_shape = np.array(replicas.shape)

    pbtxt = (R'periodic {  '
             R'  dim_0: false dim_1: false dim_2: true  '
             R'}  '
             R'boundary_models {  '
             R'  ib {  '
             R'    cartesian_grid {  '
             R'      variables { name: "u" value: 0.0 bc: DIRICHLET }  '
             R'    }  '
             R'  }  '
             R'}  ')
    model = self._set_up_immersed_boundary_method(pbtxt)

    states = {'u': tf.unstack(tf.ones((4, 4, 4), dtype=tf.float32))}

    def device_fn(replica_id, additional_states):
      """Wraps the Cartesian grid method function on TPU."""
      return model._apply_cartesian_grid_method(
          get_kernel_fn.ApplyKernelConvOp(4), replica_id, replicas, states,
          additional_states, {'u': self.bc})

    # The mirror flow test case sets the values within the solid to 0, and the
    # values on the boundary layer to -1 x (average of the neighboring fluid
    # cells).
    mirror_flow = [
        np.array([[1, 1, 1, 1], [-1, -1, -1, -1], [0, 0, 0, 0], [0, 0, 0, 0]],
                 dtype=np.float32)
    ] * 4
    if computation_shape[0] == 2:
      zeros = tf.unstack(tf.zeros((4, 4, 4), dtype=tf.float32))
      inputs = [
          [
              tf.constant(0), {
                  'ib_boundary': self.boundary,
                  'ib_interior_mask': self.mask
              }
          ],
          [tf.constant(1), {
              'ib_boundary': zeros,
              'ib_interior_mask': zeros
          }],
      ]
      expected = [
          mirror_flow,
          [np.zeros((4, 4), dtype=np.float32),] * 4,
      ]
    else:
      inputs = [
          [
              tf.constant(0), {
                  'ib_boundary': self.boundary,
                  'ib_interior_mask': self.mask
              }
          ],
          [
              tf.constant(1), {
                  'ib_boundary': self.boundary,
                  'ib_interior_mask': self.mask
              }
          ],
      ]
      expected = [mirror_flow, mirror_flow]

    result = self._run_on_tpu(replicas, inputs, device_fn)

    for i in range(4):
      self.assertAllEqual(result[0]['u'][i], expected[0][i])
      self.assertAllEqual(result[1]['u'][i], expected[1][i])

  @parameterized.parameters(*zip(_REPLICAS))
  def testApplyCartesianGridMethodLocalAverage(self, replicas):
    """Checks if the Cartesian grid method updates Neumann BC correctly."""
    computation_shape = np.array(replicas.shape)

    pbtxt = (R'periodic {  '
             R'  dim_0: false dim_1: false dim_2: true  '
             R'}  '
             R'boundary_models {  '
             R'  ib {  '
             R'    cartesian_grid {  '
             R'      variables { name: "u" value: 0.0 bc: NEUMANN }  '
             R'    }  '
             R'  }  '
             R'}  ')
    model = self._set_up_immersed_boundary_method(pbtxt)

    states = {'u': tf.unstack(tf.ones((4, 4, 4), dtype=tf.float32))}

    def device_fn(replica_id, additional_states):
      """Wraps the Cartesian grid method function on TPU."""
      return model._apply_cartesian_grid_method(
          get_kernel_fn.ApplyKernelConvOp(4), replica_id, replicas, states,
          additional_states, {'u': self.bc})

    # The local average test case sets the values within the solid to 0, and the
    # values on the boundary layer to the average of the neighboring fluid
    # cells.
    neighbor_flow = [
        np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
                 dtype=np.float32)
    ] * 4
    if computation_shape[0] == 2:
      zeros = tf.unstack(tf.zeros((4, 4, 4), dtype=tf.float32))
      inputs = [
          [
              tf.constant(0), {
                  'ib_boundary': self.boundary,
                  'ib_interior_mask': self.mask
              }
          ],
          [tf.constant(1), {
              'ib_boundary': zeros,
              'ib_interior_mask': zeros
          }],
      ]
      expected = [
          neighbor_flow,
          [np.zeros((4, 4), dtype=np.float32),] * 4,
      ]
    else:
      inputs = [
          [
              tf.constant(0), {
                  'ib_boundary': self.boundary,
                  'ib_interior_mask': self.mask
              }
          ],
          [
              tf.constant(1), {
                  'ib_boundary': self.boundary,
                  'ib_interior_mask': self.mask
              }
          ],
      ]
      expected = [neighbor_flow, neighbor_flow]

    result = self._run_on_tpu(replicas, inputs, device_fn)

    for i in range(4):
      self.assertAllEqual(result[0]['u'][i], expected[0][i])
      self.assertAllEqual(result[1]['u'][i], expected[1][i])

  @parameterized.parameters(*zip(_REPLICAS))
  def testApplyCartesianGridMethodLocalAverageWithNonZeroMaskedValue(
      self, replicas):
    """Checks if the Cartesian grid method update is applied correctly."""
    computation_shape = np.array(replicas.shape)

    pbtxt = (R'periodic {  '
             R'  dim_0: false dim_1: false dim_2: true  '
             R'}  '
             R'boundary_models {  '
             R'  ib {  '
             R'    cartesian_grid {  '
             R'      variables { name: "u" value: 2.0 bc: NEUMANN }  '
             R'    }  '
             R'  }  '
             R'}  ')
    model = self._set_up_immersed_boundary_method(pbtxt)

    states = {'u': tf.unstack(tf.ones((4, 4, 4), dtype=tf.float32))}

    def device_fn(replica_id, additional_states):
      """Wraps the Cartesian grid method function on TPU."""
      return model._apply_cartesian_grid_method(
          get_kernel_fn.ApplyKernelConvOp(4), replica_id, replicas, states,
          additional_states, {'u': self.bc})

    # The local average test case sets the values within the solid to 2, and the
    # values on the boundary layer to the average of the neighboring fluid
    # cells.
    neighbor_flow = [
        np.array([[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]],
                 dtype=np.float32)
    ] * 4
    if computation_shape[0] == 2:
      zeros = tf.unstack(tf.zeros((4, 4, 4), dtype=tf.float32))
      inputs = [
          [
              tf.constant(0), {
                  'ib_boundary': self.boundary,
                  'ib_interior_mask': self.mask
              }
          ],
          [tf.constant(1), {
              'ib_boundary': zeros,
              'ib_interior_mask': zeros
          }],
      ]
      expected = [
          neighbor_flow,
          [2. * np.ones((4, 4), dtype=np.float32),] * 4,
      ]
    else:
      inputs = [
          [
              tf.constant(0), {
                  'ib_boundary': self.boundary,
                  'ib_interior_mask': self.mask
              }
          ],
          [
              tf.constant(1), {
                  'ib_boundary': self.boundary,
                  'ib_interior_mask': self.mask
              }
          ],
      ]
      expected = [neighbor_flow, neighbor_flow]

    result = self._run_on_tpu(replicas, inputs, device_fn)

    for i in range(4):
      self.assertAllEqual(result[0]['u'][i], expected[0][i])
      self.assertAllEqual(result[1]['u'][i], expected[1][i])

  @parameterized.parameters(*zip(_REPLICAS))
  def testApplyMarkerAndCellMethodWithNeumannBC(self, replicas):
    """Checks if the Cartesian grid method updates boundary by nodes in z."""
    partition_dim = replicas.shape.index(2)

    pbtxt = (R'periodic {  '
             R'  dim_0: false dim_1: false dim_2: true  '
             R'}  '
             R'boundary_models {  '
             R'  ib {  '
             R'    mac {  '
             R'      variables { name: "u" value: 10.0 bc: NEUMANN }  '
             R'    }  '
             R'  }  '
             R'}  ')
    model = self._set_up_immersed_boundary_method(pbtxt)

    def device_fn(replica_id, states, additional_states):
      """Wraps the Cartesian grid method function on TPU."""
      return model._apply_marker_and_cell_method(replica_id, replicas, states,
                                                 additional_states,
                                                 {'u': self.bc})

    def expand_dims_12(g):
      """Expands the dimension of g in the 1 and 2 dimension."""
      return np.expand_dims(np.expand_dims(g, 1), 2)

    f = np.tile(expand_dims_12(np.arange(8, dtype=np.float32)), (1, 8, 8))
    mask = np.ones((8, 8, 8), dtype=np.float32)
    mask[:3, ...] = 0.0
    boundary = np.zeros((8, 8, 8), dtype=np.float32)
    boundary[2, ...] = 1.0

    def split_by_partition(g):
      """Splits `g` in half in the partition dimension."""
      nz, nx, ny = g.shape
      if partition_dim == 0:
        return g[:, :nx // 2, :], g[:, nx // 2:, :]
      elif partition_dim == 1:
        return g[..., :ny // 2], g[..., ny // 2:]
      elif partition_dim == 2:
        return g[:nz // 2, ...], g[nz // 2:, ...]

    def merge_by_partition(output):
      """Merges `g_0` and `g_1` by the TPU topology."""
      if partition_dim == 0:
        merge_dim = 1
      elif partition_dim == 1:
        merge_dim = 2
      elif partition_dim == 2:
        merge_dim = 0
      return np.concatenate([output[0]['u'], output[1]['u']], axis=merge_dim)

    f1, f2 = split_by_partition(f)
    mask1, mask2 = split_by_partition(mask)
    boundary1, boundary2 = split_by_partition(boundary)

    inputs = [
        [
            tf.constant(0),
            {
                'u': tf.unstack(tf.convert_to_tensor(f1))
            },
            {
                'ib_interior_mask': tf.unstack(tf.convert_to_tensor(mask1)),
                'ib_boundary': tf.unstack(tf.convert_to_tensor(boundary1))
            },
        ],
        [
            tf.constant(1),
            {
                'u': tf.unstack(tf.convert_to_tensor(f2))
            },
            {
                'ib_interior_mask': tf.unstack(tf.convert_to_tensor(mask2)),
                'ib_boundary': tf.unstack(tf.convert_to_tensor(boundary2))
            },
        ],
    ]

    expected = f.copy()
    expected[0, ...] = 6.0
    expected[1, ...] = 10.0
    expected[2, ...] = 3.0
    expected[7, ...] = 10.0
    if partition_dim == 2:
      expected[3, ...] = 5.0
      expected[4, ...] = 3.0

    output = self._run_on_tpu(replicas, inputs, device_fn)

    result = merge_by_partition(output)

    self.assertAllClose(expected, result)

  @test_util.run_in_graph_and_eager_modes
  def testApplyRayleighDampingMethodWithAndWithoutOverrides(self):
    """Checks if the Rayleigh damping forces are computed correctly."""
    pbtxt = (R'boundary_models {  '
             R'  ib {  '
             R'    sponge {  '
             R'      variables { name: "u" value: 0.0 override: true }  '
             R'      variables { name: "T" value: 300.0 override: false }  '
             R'      variables { '
             R'        name: "Z" '
             R'        value: 1.0 '
             R'        override: true '
             R'        damping_coeff: 10.0 '
             R'      }  '
             R'      damping_coeff: 20.0  '
             R'    }  '
             R'  }  '
             R'}  ')
    model = self._set_up_immersed_boundary_method(pbtxt)
    model._params.dt = 0.1

    states = {
        'u': tf.unstack(10.0 * tf.ones((4, 4, 4), dtype=tf.float32)),
        'T': tf.unstack(800.0 * tf.ones((4, 4, 4), dtype=tf.float32)),
        'Z': tf.unstack(0.5 * tf.ones((4, 4, 4), dtype=tf.float32)),
    }
    additional_states = {
        'src_u': tf.unstack(6.0 * tf.ones((4, 4, 4), dtype=tf.float32)),
        'src_T': tf.unstack(100.0 * tf.ones((4, 4, 4), dtype=tf.float32)),
        'src_Z': tf.unstack(0.3 * tf.ones((4, 4, 4), dtype=tf.float32)),
        'ib_boundary': self.boundary,
        'ib_interior_mask': self.mask,
    }

    kernel_op = get_kernel_fn.ApplyKernelConvOp(8)
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])

    output = self.evaluate(
        model._apply_rayleigh_damping_method(kernel_op, replica_id, replicas,
                                             states, additional_states))

    self.assertCountEqual(
        ['src_u', 'src_T', 'src_Z', 'ib_boundary', 'ib_interior_mask'],
        output.keys())

    with self.subTest(name='SrcU'):
      expected = np.zeros((4, 4, 4), dtype=np.float32)
      expected[:, 2:, :] = -5.0
      self.assertAllClose(expected, output['src_u'])

    with self.subTest(name='SrcT'):
      expected = 100.0 * np.ones((4, 4, 4), dtype=np.float32)
      expected[:, 2:, :] = -150.0
      self.assertAllClose(expected, output['src_T'])

    with self.subTest(name='SrcZ'):
      expected = np.zeros((4, 4, 4), dtype=np.float32)
      expected[:, 2:, :] = 0.5
      self.assertAllClose(expected, output['src_Z'])

  @test_util.run_in_graph_and_eager_modes
  def testApplyDirectForcingMethodCorrectlyUpdatesRightHandSide(self):
    """Checks if the direct forcing method computes rhs correctly."""
    pbtxt = (R'boundary_models {  '
             R'  ib {  '
             R'    direct_forcing {  '
             R'      variables { name: "u" value: 0.0 }  '
             R'      variables { name: "T" value: 300.0 }  '
             R'      variables { name: "Z" value: 1.0 damping_coeff: 2.0 }  '
             R'      damping_coeff: 1.0  '
             R'    }  '
             R'  }  '
             R'}  ')
    model = self._set_up_immersed_boundary_method(pbtxt)

    states = {
        'u': tf.unstack(10.0 * tf.ones((4, 4, 4), dtype=tf.float32)),
        'T': tf.unstack(800.0 * tf.ones((4, 4, 4), dtype=tf.float32)),
        'Z': tf.unstack(0.5 * tf.ones((4, 4, 4), dtype=tf.float32)),
    }
    additional_states = {
        'rhs_u': tf.unstack(6.0 * tf.ones((4, 4, 4), dtype=tf.float32)),
        'rhs_T': tf.unstack(100.0 * tf.ones((4, 4, 4), dtype=tf.float32)),
        'rhs_Z': tf.unstack(0.3 * tf.ones((4, 4, 4), dtype=tf.float32)),
        'ib_interior_mask': self.mask,
    }

    output = self.evaluate(
        model._apply_direct_forcing_method(states, additional_states))

    self.assertCountEqual(['rhs_u', 'rhs_T', 'rhs_Z'], output.keys())

    with self.subTest(name='RhsU'):
      expected = 6.0 * np.ones((4, 4, 4), dtype=np.float32)
      expected[:, 2:, :] = -1e3
      self.assertAllClose(expected, output['rhs_u'])

    with self.subTest(name='RhsU'):
      expected = 100.0 * np.ones((4, 4, 4), dtype=np.float32)
      expected[:, 2:, :] = -5e4
      self.assertAllClose(expected, output['rhs_T'])

    with self.subTest(name='RhsZ'):
      expected = 0.3 * np.ones((4, 4, 4), dtype=np.float32)
      expected[:, 2:, :] = 25.0
      self.assertAllClose(expected, output['rhs_Z'])

  @parameterized.parameters(*zip(_PBTXT))
  @test_util.run_in_graph_and_eager_modes
  def testGenerateInitStatesProvidesCorrectIBFields(self, pbtxt):
    """Checks helper states are initialized correctly for IB models."""
    model = self._set_up_immersed_boundary_method(pbtxt)

    def mask_fn(xx, yy, zz, lx, ly, lz, coord):
      """Initializes the IB mask for fluid and solid."""
      del xx, yy, zz, lx, ly, lz, coord
      return tf.stack(self.mask)

    def boundary_fn(xx, yy, zz, lx, ly, lz, coord):
      """Initializes the IB boundary mask for fluid and solid."""
      del xx, yy, zz, lx, ly, lz, coord
      return tf.stack(self.boundary)

    coordinates = (0, 0, 0)

    output = self.evaluate(
        model.generate_initial_states(coordinates, mask_fn, boundary_fn))

    with self.subTest(name='IncludesAllKeys'):
      expected = ['ib_interior_mask']
      if model._ib_params.WhichOneof('type') == 'sponge':
        expected += ['src_u', 'src_T']
      elif model._ib_params.WhichOneof('type') in ('cartesian_grid', 'mac'):
        expected += ['ib_boundary']

      self.assertCountEqual(expected, output.keys())

    with self.subTest(name='MaskIsCorrect'):
      expected = np.array([
          np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                   np.float32),
      ] * 6)
      self.assertAllEqual(
          expected.transpose((2, 0, 1)), output['ib_interior_mask'])

    if model._ib_params.WhichOneof('type') in ('cartesian_grid', 'mac'):
      with self.subTest(name='BoundaryIsCorrect'):
        expected = np.array([
            np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                      [1. / 4., 1. / 4., 1. / 5., 1. / 5., 1. / 4., 1. / 4.],
                      [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]], np.float32),
        ] * 6)
        self.assertAllEqual(
            expected.transpose((2, 0, 1)), output['ib_boundary'])

    elif model._ib_params.WhichOneof('type') == 'sponge':
      with self.subTest(name='SrcUIsCorrect'):
        expected = [
            np.zeros_like(val, dtype=np.float32) for val in output['src_u']
        ]
        self.assertAllEqual(expected, output['src_u'])

      with self.subTest(name='SrcTIsCorrect'):
        expected = [
            np.zeros_like(val, dtype=np.float32) for val in output['src_T']
        ]
        self.assertAllEqual(expected, output['src_T'])


if __name__ == '__main__':
  tf.test.main()
