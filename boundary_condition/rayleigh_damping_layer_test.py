"""Tests for swirl_lm.boundary_condition.rayleigh_damping_layer."""
import itertools

from absl import flags
import numpy as np
from swirl_lm.boundary_condition import rayleigh_damping_layer
from swirl_lm.boundary_condition import rayleigh_damping_layer_pb2
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.research.simulation.tensorflow.fluid.framework import util
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized

_REPLICAS = [
    np.array([[[0, 1]]]),
    np.array([[[0], [1]]]),
    np.array([[[0]], [[1]]])
]
_DIM = [0, 1, 2]

FLAGS = flags.FLAGS


class RayleighDampingLayerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(RayleighDampingLayerTest, self).setUp()

    text_proto = (R'orientation {dim: 0 fraction: 0.5}  '
                  R'orientation {dim: 1 fraction: 0.5}  '
                  R'orientation {dim: 2 fraction: 0.5}  '
                  R'variable_info {  '
                  R'  name: "u" target_value: 5.0 override: false  '
                  R'}  '
                  R'variable_info {  '
                  R'  name: "w" override: false  '
                  R'}  '
                  R'variable_info {  '
                  R'  name: "T" target_value: 300.0 override: true  '
                  R'}  ')
    self.sponge = self.set_up_sponge(text_proto)

  def set_up_sponge(self, text_proto):
    """Initializes a sponge object."""
    return text_format.Parse(
        text_proto, rayleigh_damping_layer_pb2.RayleighDampingLayer())

  def testTargetValueLibFromProtoGeneratesCorrectTargetValueLibrary(self):
    """Checks if the target value library is generated correctly."""
    target_value_lib = rayleigh_damping_layer.target_value_lib_from_proto(
        self.sponge)

    expected = {'u': 5.0, 'w': None, 'T': 300.0}

    self.assertDictEqual(expected, target_value_lib)

  def testTargetStatusLibFromProtoGeneratesCorrectTargetStatusLibrary(self):
    """Checks if the target status library is generated correctly."""
    target_status_lib = rayleigh_damping_layer.target_status_lib_from_proto(
        self.sponge)

    expected = {'src_u': False, 'src_w': False, 'src_T': True}

    self.assertDictEqual(expected, target_status_lib)

  @test_util.run_in_graph_and_eager_modes
  def testGetSpongeForceProducesCorrectForceWithLinearSpongeLayerProfile(self):
    """Checkes if forcing term is linear in a linear-profile sponge layer."""
    beta = [
        tf.zeros((6, 6), dtype=tf.float32),
        tf.zeros((6, 6), dtype=tf.float32),
        tf.zeros((6, 6), dtype=tf.float32),
        1.0 * tf.ones((6, 6), dtype=tf.float32),
        2.0 * tf.ones((6, 6), dtype=tf.float32),
        3.0 * tf.ones((6, 6), dtype=tf.float32),
    ]
    values = tf.unstack(-1.0 * tf.ones((6, 6, 6), dtype=tf.float32))

    sponge = rayleigh_damping_layer.RayleighDampingLayer(self.sponge)
    sponge._target_values = {'u': 5.0}

    replicas = np.array([[[0]]])
    force = self.evaluate(sponge._get_sponge_force(replicas, values, beta, 5.0))

    expected = np.zeros((6, 6, 6), dtype=np.float32)
    expected[3, ...] = 6.0
    expected[4, ...] = 12.0
    expected[5, ...] = 18.0

    self.assertAllClose(expected, force)

  @test_util.run_in_graph_and_eager_modes
  def testInitFnGeneratesInitialStatesForCorrectVariables(self):
    """Checks if the generated initial states are expected."""
    dt = 0.1

    text_proto = (R'orientation {dim: 2 fraction: 0.5}  '
                  R'variable_info {  '
                  R'  name: "u" target_value: 5.0 override: false  '
                  R'}  '
                  R'variable_info {  '
                  R'  name: "T" target_value: 300.0 override: true  '
                  R'}  ')
    sponge_params = text_format.Parse(
        text_proto, rayleigh_damping_layer_pb2.RayleighDampingLayer())

    beta_fn = rayleigh_damping_layer.klemp_lilly_relaxation_coeff_fn(
        dt, sponge_params.orientation)

    sponge = rayleigh_damping_layer.RayleighDampingLayer(sponge_params)

    coordinates = (0, 0, 0)

    FLAGS.cx = 1
    FLAGS.cy = 1
    FLAGS.cz = 1
    FLAGS.lx = 1.0
    FLAGS.ly = 2.1
    FLAGS.lz = 3.6
    FLAGS.nx = 10
    FLAGS.ny = 12
    FLAGS.nz = 14
    FLAGS.halo_width = 2
    FLAGS.num_boundary_points = 0
    params = grid_parametrization.GridParametrization()

    states = self.evaluate(sponge.init_fn(params, coordinates, beta_fn))

    self.assertCountEqual(['sponge_beta', 'src_u', 'src_T'], states.keys())

    with self.subTest(name='Beta'):
      expected = np.zeros((10, 6, 8), dtype=np.float32)
      expected[5, ...] = 0.0150768448
      expected[6, ...] = 0.125
      expected[7, ...] = 0.2934120444
      expected[8, ...] = 0.4415111108
      expected[9, ...] = 0.5
      self.assertAllClose(expected, states['sponge_beta'][2:-2, 2:-2, 2:-2])

    with self.subTest(name='src_u'):
      expected = np.zeros((14, 10, 12), dtype=np.float32)
      self.assertAllClose(expected, states['src_u'])

    with self.subTest(name='src_T'):
      expected = np.zeros((14, 10, 12), dtype=np.float32)
      self.assertAllClose(expected, states['src_T'])

  @parameterized.parameters(*zip(_DIM))
  @test_util.run_in_graph_and_eager_modes
  def testKlempLillyRelaxationCoeffFnProvidesSpongeLayerInThreeDims(self, dim):
    """Checks correctness of Klemp Lilly relaxation coefficient in 3 dims."""
    dt = 0.1

    lx = 1.0
    ly = 2.1
    lz = 3.6
    nx = 6
    ny = 8
    nz = 10
    x = np.linspace(0, lx, nx, dtype=np.float32)
    y = np.linspace(0, ly, ny, dtype=np.float32)
    z = np.linspace(0, lz, nz, dtype=np.float32)
    xx, yy, zz = util.meshgrid(x, y, z)

    text_proto = 'dim: {} fraction: 0.5  '.format(dim)
    orientation = text_format.Parse(
        text_proto,
        rayleigh_damping_layer_pb2.RayleighDampingLayer.Orientation())

    beta_fn = rayleigh_damping_layer.klemp_lilly_relaxation_coeff_fn(
        dt, [orientation])

    beta = self.evaluate(beta_fn(xx, yy, zz, lx, ly, lz, None))

    expected = np.zeros((6, 8, 10), dtype=np.float32)
    if dim == 0:
      expected[3, ...] = 0.04774575141
      expected[4, ...] = 0.3272542486
      expected[5, ...] = 0.5
    elif dim == 1:
      expected[:, 4, :] = 0.02475778302
      expected[:, 5, :] = 0.1943697665
      expected[:, 6, :] = 0.4058724505
      expected[:, 7, :] = 0.5
    elif dim == 2:
      expected[..., 5] = 0.0150768448
      expected[..., 6] = 0.125
      expected[..., 7] = 0.2934120444
      expected[..., 8] = 0.4415111108
      expected[..., 9] = 0.5

    self.assertAllClose(expected, beta)

  @parameterized.parameters(*zip(_DIM))
  @test_util.run_in_graph_and_eager_modes
  def testKlempLillyRelaxationCoeffFnProvidesSpongeLayerWithCustomizedCoeff(
      self, dim):
    """Checks correctness of Klemp Lilly relaxation coefficient in 3 dims."""
    dt = 0.1

    lx = 1.0
    ly = 2.1
    lz = 3.6
    nx = 6
    ny = 8
    nz = 10
    x = np.linspace(0, lx, nx, dtype=np.float32)
    y = np.linspace(0, ly, ny, dtype=np.float32)
    z = np.linspace(0, lz, nz, dtype=np.float32)
    xx, yy, zz = util.meshgrid(x, y, z)

    text_proto = 'dim: {} fraction: 0.5  '.format(dim)
    orientation = text_format.Parse(
        text_proto,
        rayleigh_damping_layer_pb2.RayleighDampingLayer.Orientation())

    beta_fn = rayleigh_damping_layer.klemp_lilly_relaxation_coeff_fn(
        dt, [orientation], 40.0)

    beta = self.evaluate(beta_fn(xx, yy, zz, lx, ly, lz, None))

    expected = np.zeros((6, 8, 10), dtype=np.float32)
    if dim == 0:
      expected[3, ...] = 0.0238728757
      expected[4, ...] = 0.1636271243
      expected[5, ...] = 0.25
    elif dim == 1:
      expected[:, 4, :] = 0.01237889151
      expected[:, 5, :] = 0.09718488325
      expected[:, 6, :] = 0.2029362253
      expected[:, 7, :] = 0.25
    elif dim == 2:
      expected[..., 5] = 0.0075384224
      expected[..., 6] = 0.0625
      expected[..., 7] = 0.1467060222
      expected[..., 8] = 0.2207555554
      expected[..., 9] = 0.25

    self.assertAllClose(expected, beta)

  @test_util.run_in_graph_and_eager_modes
  def testKlempLillyRelaxationCoeffFnInThreeCombinedDims(self):
    """Checks correctness of Klemp Lilly relaxation coefficient in 3 dims."""
    dt = 0.1

    lx = 1.0
    ly = 2.1
    lz = 3.6
    nx = 6
    ny = 8
    nz = 10
    x = np.linspace(0, lx, nx, dtype=np.float32)
    y = np.linspace(0, ly, ny, dtype=np.float32)
    z = np.linspace(0, lz, nz, dtype=np.float32)
    xx, yy, zz = util.meshgrid(x, y, z)

    text_proto = (R'orientation {dim: 0 fraction: 0.5}  '
                  R'orientation {dim: 1 face: 1 fraction: 0.5}  '
                  R'orientation {dim: 2 fraction: 0.5}  '
                  R'orientation {dim: 0 face: 0 fraction: 0.3}  '
                  R'orientation {dim: 1 face: 0 fraction: 0.3}  '
                  R'orientation {dim: 2 face: 0 fraction: 0.3}  '
                  R'variable_info {  '
                  R'  name: "u" target_value: 5.0 override: false  '
                  R'}  '
                  R'variable_info {  '
                  R'  name: "w" override: false  '
                  R'}  '
                  R'variable_info {  '
                  R'  name: "T" target_value: 300.0 override: true  '
                  R'}  ')

    sponge = self.set_up_sponge(text_proto)

    beta_fn = rayleigh_damping_layer.klemp_lilly_relaxation_coeff_fn(
        dt, sponge.orientation)

    beta = self.evaluate(beta_fn(xx, yy, zz, lx, ly, lz, None))

    expected = np.zeros((6, 8, 10), dtype=np.float32)
    buf = np.zeros((6, 8, 10), dtype=np.float32)
    buf[0, ...] = 0.5
    buf[1, ...] = 0.125
    buf[3, ...] = 0.04774575141
    buf[4, ...] = 0.3272542486
    buf[5, ...] = 0.5
    expected = np.maximum(expected, buf)
    buf = np.zeros((6, 8, 10), dtype=np.float32)
    buf[:, 0, :] = 0.5
    buf[:, 1, :] = 0.2686825234
    buf[:, 2, :] = 0.002792293444
    buf[:, 4, :] = 0.02475778302
    buf[:, 5, :] = 0.1943697665
    buf[:, 6, :] = 0.4058724505
    buf[:, 7, :] = 0.5
    expected = np.maximum(expected, buf)
    buf = np.zeros((6, 8, 10), dtype=np.float32)
    buf[..., 0] = 0.5
    buf[..., 1] = 0.3490199415
    buf[..., 2] = 0.07843959053
    buf[..., 5] = 0.0150768448
    buf[..., 6] = 0.125
    buf[..., 7] = 0.2934120444
    buf[..., 8] = 0.4415111108
    buf[..., 9] = 0.5
    expected = np.maximum(expected, buf)

    self.assertAllClose(expected, beta)

  _PBTXT_WITH_MEAN_DIMS = (R'orientation { '
                           '  dim: 2 '
                           '  fraction: 0.5 '
                           '} '
                           'variable_info { '
                           '  name: "u" '
                           '  override: false '
                           '}'
                           'target_value_mean_dim: 0 '
                           'target_value_mean_dim: 1 ')
  _PBTXT_WITHOUT_MEAN_DIMS = (R'orientation { '
                              '  dim: 2 '
                              '  fraction: 0.5 '
                              '} '
                              'variable_info { '
                              '  name: "u" '
                              '  override: false '
                              '}')

  _MEAN_DIM_PARAMS = [(_PBTXT_WITH_MEAN_DIMS, (False, False, False)),
                      (_PBTXT_WITHOUT_MEAN_DIMS, (True, True, False))]

  @parameterized.parameters(*itertools.product(_REPLICAS, _MEAN_DIM_PARAMS))
  def testGetSpongeForceProducesCorrectForceWithoutTargetValue(
      self, replicas, mean_dim_params):
    """Checkes if forcing term is correct in the absence of a target type."""
    beta = tf.stack([
        tf.zeros((6, 6), dtype=tf.float32),
        tf.zeros((6, 6), dtype=tf.float32),
        tf.zeros((6, 6), dtype=tf.float32),
        1.0 * tf.ones((6, 6), dtype=tf.float32),
        2.0 * tf.ones((6, 6), dtype=tf.float32),
        3.0 * tf.ones((6, 6), dtype=tf.float32),
    ])
    # 6 x 6 diagonal tensor with diagonal [2, 7].
    diag_elements = np.arange(2, 8)
    diagonal = tf.cast(tf.linalg.diag(diag_elements), dtype=tf.float32)
    values = tf.stack([
        -1.0 * diagonal,
        -2.0 * diagonal,
        -3.0 * diagonal,
        -4.0 * diagonal,
        -5.0 * diagonal,
        -6.0 * diagonal,
    ])

    damping_layer = text_format.Parse(
        mean_dim_params[0], rayleigh_damping_layer_pb2.RayleighDampingLayer())

    periodic_dims = mean_dim_params[1]
    sponge = rayleigh_damping_layer.RayleighDampingLayer(
        damping_layer, periodic_dims)

    dim = np.where(np.array(replicas.shape) == 2)[0][0]
    if dim == 0:
      input1 = [tf.unstack(values[:, :3, :]), tf.unstack(beta[:, :3, :])]
      input2 = [tf.unstack(values[:, 3:, :]), tf.unstack(beta[:, 3:, :])]
    elif dim == 1:
      input1 = [tf.unstack(values[:, :, :3]), tf.unstack(beta[:, :, :3])]
      input2 = [tf.unstack(values[:, :, 3:]), tf.unstack(beta[:, :, 3:])]
    else:
      input1 = [tf.unstack(values[:3, :, :]), tf.unstack(beta[:3, :, :])]
      input2 = [tf.unstack(values[3:, :, :]), tf.unstack(beta[3:, :, :])]
    inputs = [input1, input2]

    def device_fn(val, b):
      """Wraps the `_get_sponge_force` function."""
      return sponge._get_sponge_force(replicas, val, b)

    # transpose inputs.
    device_inputs = [list(x) for x in zip(*inputs)]
    computation_shape = replicas.shape
    runner = TpuRunner(computation_shape=computation_shape)
    output = runner.run(device_fn, *device_inputs)
    if dim == 0:
      force = np.concatenate([np.stack(output[0]), np.stack(output[1])], 1)
    elif dim == 1:
      force = np.concatenate([np.stack(output[0]), np.stack(output[1])], 2)
    else:
      force = np.concatenate([np.stack(output[0]), np.stack(output[1])], 0)

    # Expected values for tensors in z = [0-2] will be 0 since beta is 0.
    # For z in [3 - 5], the expected values are -beta * (value - mean) where
    # mean is the mean of the x-y slice.

    # z = 3:
    # mean of x-y slice: -4 * 0.75 = -3
    # non-diagonal = -1. * (0. - (-3)) = -3.0
    expected_3 = -3.0 * np.ones((6, 6), dtype=np.float32)
    # diagonal: -1 * (-4 * diag_elements - (-3))
    expected_3_diag = -1. * (-4. * diag_elements + 3.)
    np.fill_diagonal(expected_3, expected_3_diag)

    # z = 4:
    # mean of x-y slice: -5 * 0.75 = -3.75
    # non-diagonal = -2 * (0 - (-3.75)) = -7.5
    expected_4 = -7.5 * np.ones((6, 6), dtype=np.float32)
    # diagonal: -2 * (-5 * diag_elements - (-3.75))
    expected_4_diag = -2. * (-5. * diag_elements + 3.75)
    np.fill_diagonal(expected_4, expected_4_diag)

    # z = 5:
    # mean of x-y slice: -6 * 0.75 = -4.5
    # non-diagonal = -3. * (0 - (-4.5)) = -13.5
    expected_5 = -13.5 * np.ones((6, 6), dtype=np.float32)
    # diagonal: -3 * (-6 * diag_elements - (-4.5))
    expected_5_diag = -3. * (-6. * diag_elements + 4.5)
    np.fill_diagonal(expected_5, expected_5_diag)

    expected = np.zeros((6, 6, 6), dtype=np.float32)
    expected[3, ...] = expected_3
    expected[4, ...] = expected_4
    expected[5, ...] = expected_5

    self.assertAllClose(expected, force)

  @parameterized.parameters(*zip(_REPLICAS))
  def testGetSpongeForceProducesCorrectForceWithTargetStateType(
      self, replicas):
    """Checkes forcing term is correct with `target_state_name` target type."""
    beta = np.array([
        np.zeros([6, 6]),
        np.zeros([6, 6]),
        np.zeros([6, 6]),
        1.0 * np.ones([6, 6]),
        2.0 * np.ones([6, 6]),
        3.0 * np.ones([6, 6]),
    ])
    values = np.random.normal(loc=0.1, scale=1, size=[6, 6, 6])
    init_values = np.random.normal(loc=0.1, scale=1, size=[6, 6, 6])

    damping_layer = text_format.Parse(
        """
        orientation {
          dim: 2
          fraction: 0.5
        }
        variable_info {
          target_state_name: 'u_init'
          name: "u"
          override: false
        }
        """, rayleigh_damping_layer_pb2.RayleighDampingLayer())

    sponge = rayleigh_damping_layer.RayleighDampingLayer(damping_layer)

    dim = np.where(np.array(replicas.shape) == 2)[0][0]
    if dim == 0:
      input1 = [tf.unstack(values[:, :3, :]), tf.unstack(beta[:, :3, :]),
                tf.unstack(init_values[:, :3, :])]
      input2 = [tf.unstack(values[:, 3:, :]), tf.unstack(beta[:, 3:, :]),
                tf.unstack(init_values[:, 3:, :])]
    elif dim == 1:
      input1 = [tf.unstack(values[:, :, :3]), tf.unstack(beta[:, :, :3]),
                tf.unstack(init_values[:, :, :3])]
      input2 = [tf.unstack(values[:, :, 3:]), tf.unstack(beta[:, :, 3:]),
                tf.unstack(init_values[:, :, 3:])]
    else:
      input1 = [tf.unstack(values[:3, :, :]), tf.unstack(beta[:3, :, :]),
                tf.unstack(init_values[:3, :, :])]
      input2 = [tf.unstack(values[3:, :, :]), tf.unstack(beta[3:, :, :]),
                tf.unstack(init_values[3:, :, :])]
    inputs = [input1, input2]

    def device_fn(val, b, init_val):
      """Wraps the `_get_sponge_force` function."""
      return sponge._get_sponge_force(replicas, val, b, init_val)

    # transpose inputs.
    device_inputs = [list(x) for x in zip(*inputs)]

    computation_shape = replicas.shape
    runner = TpuRunner(computation_shape=computation_shape)
    output = runner.run(device_fn, *device_inputs)
    if dim == 0:
      force = np.concatenate([np.stack(output[0]), np.stack(output[1])], 1)
    elif dim == 1:
      force = np.concatenate([np.stack(output[0]), np.stack(output[1])], 2)
    else:
      force = np.concatenate([np.stack(output[0]), np.stack(output[1])], 0)

    # Expected values for tensors in z = [0-2] will be 0 since beta is 0.
    # For z in [3 - 5], the expected values are -beta * (value - target_state).
    expected = np.zeros((6, 6, 6), dtype=np.float32)
    expected[3:, ...] = -beta[3:, ...] * (
        values[3:, ...] - init_values[3:, ...])

    self.assertAllClose(expected, force)

  @test_util.run_in_graph_and_eager_modes
  def testAdditionalStatesUpdateFnUpdatesSpongeForcesForMultipleVariables(self):
    """Checks if the sponge forces for 'u' and 'T' are updated."""
    additional_states = {
        'sponge_beta': [
            tf.zeros((6, 6), dtype=tf.float32),
            tf.zeros((6, 6), dtype=tf.float32),
            tf.zeros((6, 6), dtype=tf.float32),
            1.0 * tf.ones((6, 6), dtype=tf.float32),
            2.0 * tf.ones((6, 6), dtype=tf.float32),
            3.0 * tf.ones((6, 6), dtype=tf.float32),
        ],
        'src_u': tf.unstack(tf.ones((6, 6, 6), dtype=tf.float32)),
        'src_T': tf.unstack(200.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
    }
    states = {
        'u': tf.unstack(-1.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
        'T': tf.unstack(800.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
        'rho': tf.unstack(1.2 * tf.ones((6, 6, 6), dtype=tf.float32)),
    }

    sponge = rayleigh_damping_layer.RayleighDampingLayer(self.sponge)

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    params = grid_parametrization.GridParametrization()
    forces = self.evaluate(
        sponge.additional_states_update_fn(
            get_kernel_fn.ApplyKernelConvOp(4), replica_id, replicas, states,
            additional_states, params))

    with self.subTest(name='SpongeForT'):
      expected = np.zeros((6, 6, 6), dtype=np.float32)
      expected[3, ...] = -500.0
      expected[4, ...] = -1000.0
      expected[5, ...] = -1500.0

      self.assertAllClose(expected, forces['src_T'])

    with self.subTest(name='SpongeForU'):
      expected = np.ones((6, 6, 6), dtype=np.float32)
      expected[3, ...] = 7.0
      expected[4, ...] = 13.0
      expected[5, ...] = 19.0

      self.assertAllClose(expected, forces['src_u'])

  @test_util.run_in_graph_and_eager_modes
  def testAdditionalStatesUpdateFnUpdatesSpongeForcesForTargetStateType(self):
    """Checks if the sponge forces for 'u' are updated with the target state."""
    additional_states = {
        'sponge_beta': [
            tf.zeros((6, 6), dtype=tf.float32),
            tf.zeros((6, 6), dtype=tf.float32),
            tf.zeros((6, 6), dtype=tf.float32),
            1.0 * tf.ones((6, 6), dtype=tf.float32),
            2.0 * tf.ones((6, 6), dtype=tf.float32),
            3.0 * tf.ones((6, 6), dtype=tf.float32),
        ],
        'src_u': tf.unstack(tf.ones((6, 6, 6), dtype=tf.float32)),
        'u_init': tf.unstack(200.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
    }
    states = {
        'u': tf.unstack(-1.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
    }

    damping_layer = text_format.Parse(
        """
        orientation {
          dim: 2
          fraction: 0.5
        }
        variable_info {
          target_state_name: 'u_init'
          name: "u"
          override: true
        }
        """, rayleigh_damping_layer_pb2.RayleighDampingLayer())

    sponge = rayleigh_damping_layer.RayleighDampingLayer(damping_layer)

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    params = grid_parametrization.GridParametrization()
    forces = self.evaluate(
        sponge.additional_states_update_fn(
            get_kernel_fn.ApplyKernelConvOp(4), replica_id, replicas, states,
            additional_states, params))
    expected = self.evaluate([
        -b_i * (u_i - u_init_i) for b_i, u_i, u_init_i in zip(
            additional_states['sponge_beta'], states['u'],
            additional_states['u_init'])
    ])
    self.assertAllClose(expected, forces['src_u'])

  @test_util.run_in_graph_and_eager_modes
  def testAdditionalStatesUpdateFnUpdatesSpongeForcesForConservativeVariables(
      self):
    """Checks if the sponge for 'u' in the conservative form is correct."""

    text_proto = (R'orientation {dim: 0 fraction: 0.5}  '
                  R'orientation {dim: 1 fraction: 0.5}  '
                  R'orientation {dim: 2 fraction: 0.5}  '
                  R'variable_info {  '
                  R'  name: "u"  '
                  R'  target_value: 5.0  '
                  R'  override: false  '
                  R'  primitive: false  '
                  R'}  ')
    self.sponge = text_format.Parse(
        text_proto, rayleigh_damping_layer_pb2.RayleighDampingLayer())

    additional_states = {
        'sponge_beta': [
            tf.zeros((6, 6), dtype=tf.float32),
            tf.zeros((6, 6), dtype=tf.float32),
            tf.zeros((6, 6), dtype=tf.float32),
            1.0 * tf.ones((6, 6), dtype=tf.float32),
            2.0 * tf.ones((6, 6), dtype=tf.float32),
            3.0 * tf.ones((6, 6), dtype=tf.float32),
        ],
        'src_u': tf.unstack(tf.ones((6, 6, 6), dtype=tf.float32)),
    }
    states = {
        'u': tf.unstack(-1.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
        'rho': tf.unstack(1.2 * tf.ones((6, 6, 6), dtype=tf.float32)),
    }

    sponge = rayleigh_damping_layer.RayleighDampingLayer(self.sponge)

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    params = grid_parametrization.GridParametrization()
    forces = self.evaluate(
        sponge.additional_states_update_fn(
            get_kernel_fn.ApplyKernelConvOp(4), replica_id, replicas, states,
            additional_states, params))

    with self.subTest(name='SpongeForU'):
      expected = np.ones((6, 6, 6), dtype=np.float32)
      expected[3, ...] = 8.2
      expected[4, ...] = 15.4
      expected[5, ...] = 22.6

      self.assertAllClose(expected, forces['src_u'])


if __name__ == '__main__':
  tf.test.main()
