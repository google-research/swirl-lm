"""Tests for google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.equations.velocity."""

import os

from absl import flags
import numpy as np
from swirl_lm.equations import velocity
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.utility import components_debug
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import monitor
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.pyglib import gfile
from google3.pyglib import resources
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_parameters_pb2
from google3.testing.pybase import parameterized

FLAGS = flags.FLAGS

_TESTDATA_DIR = 'google3/third_party/py/swirl_lm/equations/testdata'


def ReadProto(filename):
  with gfile.Open(
      resources.GetResourceFilename(os.path.join(_TESTDATA_DIR,
                                                 filename))) as f:
    return f.read()


class VelocityTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Initializes shared fields for tests."""
    super(VelocityTest, self).setUp()

    self.kernel_op = get_kernel_fn.ApplyKernelConvOp(4)

    # Set up a (8, 8, 8) mesh. Only the point at (1, 1, 1) is tested as a
    # reference.
    self.u = [
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant([[0] * 8, [0, 2, 0, 0, 0, 0, 0, 0], [0] * 8, [0] * 8,
                     [0] * 8, [0] * 8, [0] * 8, [0] * 8],
                    dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
    ]

    self.v = [
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant([[0] * 8, [0, -3, 0, 0, 0, 0, 0, 0], [0] * 8, [0] * 8,
                     [0] * 8, [0] * 8, [0] * 8, [0] * 8],
                    dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
    ]

    self.w = [
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant([[0] * 8, [0, 4, 0, 0, 0, 0, 0, 0], [0] * 8, [0] * 8,
                     [0] * 8, [0] * 8, [0] * 8, [0] * 8],
                    dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
    ]

    self.p = [
        tf.constant(2, shape=(8, 8), dtype=tf.float32),
        tf.constant([[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11],
                     [8, 7, 6, 5, 4, 3, 2, 1], [4, 3, 2, 1, 0, -1, -2, -3],
                     [0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11],
                     [8, 7, 6, 5, 4, 3, 2, 1], [4, 3, 2, 1, 0, -1, -2, -3]],
                    dtype=tf.float32),
        tf.constant(6, shape=(8, 8), dtype=tf.float32),
        tf.constant(8, shape=(8, 8), dtype=tf.float32),
        tf.constant(10, shape=(8, 8), dtype=tf.float32),
        tf.constant(8, shape=(8, 8), dtype=tf.float32),
        tf.constant(6, shape=(8, 8), dtype=tf.float32),
        tf.constant(2, shape=(8, 8), dtype=tf.float32),
    ]

    self.halo_dims = [0, 1, 2]
    self.replica_id = tf.constant(0)
    self.replicas = np.array([[[0]]], dtype=np.int32)
    self.replica_dims = [0, 1, 2]

  def set_up_velocity(self, scheme, dbg):
    """Initializes the `Velocity` object."""
    pbtxt = ReadProto('velocity_config.textpb')
    convection_pbtxt = 'convection_scheme: {}  '.format(scheme)
    config = text_format.Parse(
        convection_pbtxt + pbtxt,
        incompressible_structured_mesh_parameters_pb2
        .IncompressibleNavierStokesParameters())
    FLAGS.cx = 1
    FLAGS.cy = 1
    FLAGS.cz = 1
    FLAGS.nx = 12
    FLAGS.ny = 12
    FLAGS.nz = 12
    FLAGS.lx = 7e-3
    FLAGS.ly = 1.4e-2
    FLAGS.lz = 3.5e-2
    FLAGS.halo_width = 2
    FLAGS.dt = 1e-3
    FLAGS.simulation_debug = dbg
    FLAGS.num_boundary_points = 0
    params = (
        incompressible_structured_mesh_config
        .IncompressibleNavierStokesParameters(config))

    thermodynamics = thermodynamics_manager.thermodynamics_factory(params)

    dbg_model = components_debug.ComponentsDebug(params) if dbg else None

    monitor_lib = monitor.Monitor(params)
    return velocity.Velocity(
        self.kernel_op, params, thermodynamics, monitor_lib, dbg=dbg_model)

  _CONVECTION_SCHEME = [
      velocity._ConvectionScheme.CONVECTION_SCHEME_CENTRAL_2,
      velocity._ConvectionScheme.CONVECTION_SCHEME_QUICK,
  ]

  @parameterized.parameters(*zip(_CONVECTION_SCHEME))
  @test_util.run_in_graph_and_eager_modes
  def testMomentumUpdateComputesCorrectRightHandSideFn(self, convection_scheme):
    """Momentum rhs near point [1, 1, 1] are computed correctly."""
    model = self.set_up_velocity(convection_scheme, False)

    nu = 0.0
    mu = [nu * tf.ones_like(u, dtype=tf.float32) for u in self.u]

    rho_mix = [0.5 * tf.ones_like(p_i, dtype=tf.float32) for p_i in self.p]

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    rhs_fn = model._momentum_update(replica_id, replicas, {}, mu, self.p,
                                    rho_mix, (None, None, None))

    rhs_rhou, rhs_rhov, rhs_rhow = self.evaluate(
        rhs_fn(self.u, self.v, self.w, self.u, self.v, self.w))

    if convection_scheme == (
        velocity._ConvectionScheme.CONVECTION_SCHEME_CENTRAL_2):
      with self.subTest(name='RhsRhoUAt121WithoutDiffusion'):
        self.assertAllClose(rhs_rhou[1][1, 2], -3500.0)

      with self.subTest(name='RhsRhoVAt121WithoutDiffusion'):
        self.assertAllClose(rhs_rhov[1][1, 2], 1750.0)

      with self.subTest(name='RhsRhoWAt121WithoutDiffusion'):
        self.assertAllClose(rhs_rhow[1][1, 2], -3395.0947)
    elif convection_scheme == (
        velocity._ConvectionScheme.CONVECTION_SCHEME_QUICK):
      with self.subTest(name='RhsRhoUAt121WithoutDiffusion'):
        self.assertAllClose(rhs_rhou[1][1, 2], -2562.5)

      with self.subTest(name='RhsRhoVAt121WithoutDiffusion'):
        self.assertAllClose(rhs_rhov[1][1, 2], 343.74997)

      with self.subTest(name='RhsRhoWAt121WithoutDiffusion'):
        self.assertAllClose(rhs_rhow[1][1, 2], -1520.095)

  @parameterized.parameters(*zip(_CONVECTION_SCHEME))
  @test_util.run_in_graph_and_eager_modes
  def testMomentumUpdateComputesCorrectRightHandSideFnWthDebugMode(
      self, convection_scheme):
    """Momentum rhs terms near point [1, 1, 1] are computed correctly."""
    model = self.set_up_velocity(convection_scheme, True)

    nu = 0.0
    mu = [nu * tf.ones_like(u, dtype=tf.float32) for u in self.u]

    rho_mix = [0.5 * tf.ones_like(p_i, dtype=tf.float32) for p_i in self.p]

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    rhs_fn = model._momentum_update(replica_id, replicas, {}, mu, self.p,
                                    rho_mix, (None, None, None), True)

    terms = self.evaluate(
        rhs_fn(self.u, self.v, self.w, self.u, self.v, self.w))

    if convection_scheme == (
        velocity._ConvectionScheme.CONVECTION_SCHEME_CENTRAL_2):
      with self.subTest(name='RhsRhoUAt121WithoutDiffusion'):
        self.assertAllClose(terms[0]['conv_x'][1][1, 2], 0.0)
        self.assertAllClose(terms[0]['conv_y'][1][1, 2], 1500.0)
        self.assertAllClose(terms[0]['conv_z'][1][1, 2], 0.0)
        self.assertAllClose(terms[0]['diff_x'][1][1, 2], 0.0)
        self.assertAllClose(terms[0]['diff_y'][1][1, 2], 0.0)
        self.assertAllClose(terms[0]['diff_z'][1][1, 2], 0.0)
        self.assertAllClose(terms[0]['gravity'][1][1, 2], 0.0)
        self.assertAllClose(terms[0]['force'][1][1, 2], 0.0)

      with self.subTest(name='RhsRhoVAt121WithoutDiffusion'):
        self.assertAllClose(terms[1]['conv_x'][1][1, 2], 0.0)
        self.assertAllClose(terms[1]['conv_y'][1][1, 2], -2250.0)
        self.assertAllClose(terms[1]['conv_z'][1][1, 2], 0.0)
        self.assertAllClose(terms[1]['diff_x'][1][1, 2], 0.0)
        self.assertAllClose(terms[1]['diff_y'][1][1, 2], 0.0)
        self.assertAllClose(terms[1]['diff_z'][1][1, 2], 0.0)
        self.assertAllClose(terms[1]['gravity'][1][1, 2], 0.0)
        self.assertAllClose(terms[1]['force'][1][1, 2], 0.0)

      with self.subTest(name='RhsRhoWAt121WithoutDiffusion'):
        self.assertAllClose(terms[2]['conv_x'][1][1, 2], 0.0)
        self.assertAllClose(terms[2]['conv_y'][1][1, 2], 3000.0)
        self.assertAllClose(terms[2]['conv_z'][1][1, 2], 0.0)
        self.assertAllClose(terms[2]['diff_x'][1][1, 2], 0.0)
        self.assertAllClose(terms[2]['diff_y'][1][1, 2], 0.0)
        self.assertAllClose(terms[2]['diff_z'][1][1, 2], 0.0)
        self.assertAllClose(terms[2]['gravity'][1][1, 2], 4.905)
        self.assertAllClose(terms[2]['force'][1][1, 2], 0.0)

    elif convection_scheme == (
        velocity._ConvectionScheme.CONVECTION_SCHEME_QUICK):
      with self.subTest(name='RhsRhoUAt121WithoutDiffusion'):
        self.assertAllClose(terms[0]['conv_x'][1][1, 2], 0.0)
        self.assertAllClose(terms[0]['conv_y'][1][1, 2], 562.5)
        self.assertAllClose(terms[0]['conv_z'][1][1, 2], 0.0)
        self.assertAllClose(terms[0]['diff_x'][1][1, 2], 0.0)
        self.assertAllClose(terms[0]['diff_y'][1][1, 2], 0.0)
        self.assertAllClose(terms[0]['diff_z'][1][1, 2], 0.0)
        self.assertAllClose(terms[0]['gravity'][1][1, 2], 0.0)
        self.assertAllClose(terms[0]['force'][1][1, 2], 0.0)

      with self.subTest(name='RhsRhoVAt121WithoutDiffusion'):
        self.assertAllClose(terms[1]['conv_x'][1][1, 2], 0.0)
        self.assertAllClose(terms[1]['conv_y'][1][1, 2], -843.75)
        self.assertAllClose(terms[1]['conv_z'][1][1, 2], 0.0)
        self.assertAllClose(terms[1]['diff_x'][1][1, 2], 0.0)
        self.assertAllClose(terms[1]['diff_y'][1][1, 2], 0.0)
        self.assertAllClose(terms[1]['diff_z'][1][1, 2], 0.0)
        self.assertAllClose(terms[1]['gravity'][1][1, 2], 0.0)
        self.assertAllClose(terms[1]['force'][1][1, 2], 0.0)

      with self.subTest(name='RhsRhoWAt121WithoutDiffusion'):
        self.assertAllClose(terms[2]['conv_x'][1][1, 2], 0.0)
        self.assertAllClose(terms[2]['conv_y'][1][1, 2], 1125.0)
        self.assertAllClose(terms[2]['conv_z'][1][1, 2], 0.0)
        self.assertAllClose(terms[2]['diff_x'][1][1, 2], 0.0)
        self.assertAllClose(terms[2]['diff_y'][1][1, 2], 0.0)
        self.assertAllClose(terms[2]['diff_z'][1][1, 2], 0.0)
        self.assertAllClose(terms[2]['gravity'][1][1, 2], 4.905)
        self.assertAllClose(terms[2]['force'][1][1, 2], 0.0)

  def testUpdateWallBCGeneratesCorrectVelocityBC(self):
    """Checks if wall boundary conditions for velocity is correctly updated."""
    pbtxt = ReadProto('velocity_config_mixed_wall.textpb')
    config = text_format.Parse(
        pbtxt,
        incompressible_structured_mesh_parameters_pb2
        .IncompressibleNavierStokesParameters())
    FLAGS.halo_width = 2
    params = (
        incompressible_structured_mesh_config
        .IncompressibleNavierStokesParameters(config))

    thermodynamics = thermodynamics_manager.thermodynamics_factory(params)

    monitor_lib = monitor.Monitor(params)
    model = velocity.Velocity(self.kernel_op, params, thermodynamics,
                              monitor_lib)

    u = np.ones((8, 8, 8), dtype=np.float32)
    v = 2.0 * np.ones((8, 8, 8), dtype=np.float32)
    w = 3.0 * np.ones((8, 8, 8), dtype=np.float32)

    states = {
        'u': tf.unstack(tf.convert_to_tensor(u)),
        'v': tf.unstack(tf.convert_to_tensor(v)),
        'w': tf.unstack(tf.convert_to_tensor(w)),
    }

    model._update_wall_bc(states)

    with self.subTest(name='InflowInDim0Face0'):
      self.assertEqual(6.0, model._bc['u'][0][0][1])
      self.assertEqual(0.0, model._bc['v'][0][0][1])
      self.assertEqual(0.0, model._bc['w'][0][0][1])

    with self.subTest(name='FreeSlipWallInDim0Face1'):
      expected_u_0 = [-1.0 * np.ones((1, 8), dtype=np.float32)] * 8
      expected_u_1 = [-3.0 * np.ones((1, 8), dtype=np.float32)] * 8
      self.assertAllEqual(expected_u_0,
                          self.evaluate(model._bc['u'][0][1][1][0]))
      self.assertAllEqual(expected_u_1,
                          self.evaluate(model._bc['u'][0][1][1][1]))
      self.assertEqual(0.0, model._bc['v'][0][1][1])
      self.assertEqual(0.0, model._bc['w'][0][1][1])

    with self.subTest(name='ShearWallInDim1Face0'):
      expected_v_0 = [-6.0 * np.ones((8, 1), dtype=np.float32)] * 8
      expected_v_1 = [-2.0 * np.ones((8, 1), dtype=np.float32)] * 8
      self.assertEqual(2.0, model._bc['u'][1][0][1])
      self.assertAllEqual(expected_v_0,
                          self.evaluate(model._bc['v'][1][0][1][0]))
      self.assertAllEqual(expected_v_1,
                          self.evaluate(model._bc['v'][1][0][1][1]))
      self.assertEqual(-3.0, model._bc['w'][1][0][1])

    with self.subTest(name='ShearWallInDim1Face1'):
      expected_v_0 = [-2.0 * np.ones((8, 1), dtype=np.float32)] * 8
      expected_v_1 = [-6.0 * np.ones((8, 1), dtype=np.float32)] * 8
      self.assertEqual(-2.0, model._bc['u'][1][1][1])
      self.assertAllEqual(expected_v_0,
                          self.evaluate(model._bc['v'][1][1][1][0]))
      self.assertAllEqual(expected_v_1,
                          self.evaluate(model._bc['v'][1][1][1][1]))
      self.assertEqual(3.0, model._bc['w'][1][1][1])

    with self.subTest(name='NonSlipWallInDim2Face0'):
      expected_u_0 = -3.0 * np.ones((8, 8), dtype=np.float32)
      expected_v_0 = -6.0 * np.ones((8, 8), dtype=np.float32)
      expected_w_0 = -9.0 * np.ones((8, 8), dtype=np.float32)
      expected_u_1 = -1.0 * np.ones((8, 8), dtype=np.float32)
      expected_v_1 = -2.0 * np.ones((8, 8), dtype=np.float32)
      expected_w_1 = -3.0 * np.ones((8, 8), dtype=np.float32)
      self.assertAllEqual(expected_u_0,
                          self.evaluate(model._bc['u'][2][0][1][0]))
      self.assertAllEqual(expected_v_0,
                          self.evaluate(model._bc['v'][2][0][1][0]))
      self.assertAllClose(expected_w_0,
                          self.evaluate(model._bc['w'][2][0][1][0]))
      self.assertAllEqual(expected_u_1,
                          self.evaluate(model._bc['u'][2][0][1][1]))
      self.assertAllEqual(expected_v_1,
                          self.evaluate(model._bc['v'][2][0][1][1]))
      self.assertAllEqual(expected_w_1,
                          self.evaluate(model._bc['w'][2][0][1][1]))

    with self.subTest(name='FreeSlipWallInDim2Face1'):
      expected_w_0 = -3.0 * np.ones((8, 8), dtype=np.float32)
      expected_w_1 = -9.0 * np.ones((8, 8), dtype=np.float32)
      self.assertEqual(0.0, model._bc['u'][2][1][1])
      self.assertEqual(0.0, model._bc['v'][2][1][1])
      self.assertAllEqual(expected_w_0,
                          self.evaluate(model._bc['w'][2][1][1][0]))
      self.assertAllEqual(expected_w_1,
                          self.evaluate(model._bc['w'][2][1][1][1]))


if __name__ == '__main__':
  tf.test.main()
