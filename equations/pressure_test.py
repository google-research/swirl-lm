"""Tests for google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.equations.pressure."""

import itertools
import os

from absl import flags
import numpy as np
from swirl_lm.communication import halo_exchange
from swirl_lm.equations import pressure
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import monitor
from swirl_lm.utility import tf_test_util as test_util
from swirl_lm.utility import types
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.pyglib import gfile
from google3.pyglib import resources
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_parameters_pb2
from google3.testing.pybase import googletest
from google3.testing.pybase import parameterized

FLAGS = flags.FLAGS

_TESTDATA_DIR = 'google3/third_party/py/swirl_lm/equations/testdata'


@test_util.run_all_in_graph_and_eager_modes
class PressureTest(tf.test.TestCase, parameterized.TestCase):

  _RHO_INFO = [
      pressure.ConstantDensityInfo,
      pressure.VariableDensityInfo,
  ]

  _SUBITER = [
      None,
      0,
      1
  ]

  def setUp(self):
    """Initializes shared fields for tests."""
    super(PressureTest, self).setUp()
    self.kernel_op = get_kernel_fn.ApplyKernelConvOp(4)
    self.halo_dims = [0, 1, 2]
    self.replica_id = tf.constant(0)
    self.replicas = np.array([[[0]]], dtype=np.int32)
    self.replica_dims = [0, 1, 2]
    self.periodic_dims = [False, False, False]

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

    self.bc_u = [[(halo_exchange.BCType.DIRICHLET, 1.0),
                  (halo_exchange.BCType.DIRICHLET, 1.0)],
                 [(halo_exchange.BCType.DIRICHLET, 2.0),
                  (halo_exchange.BCType.DIRICHLET, 2.0)],
                 [(halo_exchange.BCType.DIRICHLET, 3.0),
                  (halo_exchange.BCType.DIRICHLET, 3.0)]]

    self.bc_v = [[(halo_exchange.BCType.DIRICHLET, -1.0),
                  (halo_exchange.BCType.DIRICHLET, -1.0)],
                 [(halo_exchange.BCType.DIRICHLET, -2.0),
                  (halo_exchange.BCType.DIRICHLET, -2.0)],
                 [(halo_exchange.BCType.DIRICHLET, -3.0),
                  (halo_exchange.BCType.DIRICHLET, -3.0)]]

    self.bc_w = [[(halo_exchange.BCType.DIRICHLET, -2.0),
                  (halo_exchange.BCType.DIRICHLET, -2.0)],
                 [(halo_exchange.BCType.DIRICHLET, 3.0),
                  (halo_exchange.BCType.DIRICHLET, 3.0)],
                 [(halo_exchange.BCType.DIRICHLET, -1.0),
                  (halo_exchange.BCType.DIRICHLET, -1.0)]]

  def set_up_pressure(self, nx=10, ny=10, nz=10, halo_width=1):
    """Sets up the pressure object."""
    with gfile.Open(
        resources.GetResourceFilename(
            os.path.join(_TESTDATA_DIR, 'pressure_config.textpb')), 'r') as f:
      config = text_format.Parse(
          f.read(),
          incompressible_structured_mesh_parameters_pb2
          .IncompressibleNavierStokesParameters())

    FLAGS.cx = 1
    FLAGS.cy = 1
    FLAGS.cz = 1
    FLAGS.nx = nx
    FLAGS.ny = ny
    FLAGS.nz = nz
    FLAGS.lx = 7.0
    FLAGS.ly = 1.75
    FLAGS.lz = 3.5
    FLAGS.halo_width = halo_width
    FLAGS.dt = 1e-2
    FLAGS.num_boundary_points = 0
    params = (
        incompressible_structured_mesh_config
        .IncompressibleNavierStokesParameters(config))
    thermodynamics = thermodynamics_manager.thermodynamics_factory(params)
    monitor_lib = monitor.Monitor(params)

    return pressure.Pressure(self.kernel_op, params, thermodynamics,
                             monitor_lib)

  def testUpdatePressureHalosComputesPressureBCAtWallsCorrectly(self):
    """Checks if pressure boundary conditions are correct at different walls."""
    model = self.set_up_pressure(16, 16, 16, 2)

    u = np.zeros((16, 16, 16), dtype=np.float32)
    u[:, 1, :] = 1.0
    u[:, 2, :] = 3.0
    u[:, -2, :] = -2.0

    v = np.zeros((16, 16, 16), dtype=np.float32)
    v[..., 1] = 1.0
    v[..., 2] = 3.0
    v[..., -2] = -2.0

    w = np.zeros((16, 16, 16), dtype=np.float32)
    w[1, ...] = 1.0
    w[2, ...] = 3.0
    w[-2, ...] = -2.0

    states = {
        'u': tf.unstack(tf.convert_to_tensor(u, dtype=tf.float32)),
        'v': tf.unstack(tf.convert_to_tensor(v, dtype=tf.float32)),
        'w': tf.unstack(tf.convert_to_tensor(w, dtype=tf.float32)),
        'rho_u': tf.unstack(tf.convert_to_tensor(u, dtype=tf.float32)),
        'rho_v': tf.unstack(tf.convert_to_tensor(v, dtype=tf.float32)),
        'rho_w': tf.unstack(tf.convert_to_tensor(w, dtype=tf.float32)),
        'rho': tf.unstack(tf.ones((16, 16, 16), dtype=tf.float32)),
        'p': tf.unstack(tf.zeros((16, 16, 16), dtype=tf.float32)),
    }

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    output = self.evaluate(
        model.update_pressure_halos(replica_id, replicas, states))

    p = np.stack(output['p'])

    with self.subTest(name='Dim0Face0'):
      expected = -2.095238095e-05 * np.ones((12, 2, 12), dtype=np.float32)
      self.assertAllClose(expected, p[2:-2, :2, 2:-2])

    with self.subTest(name='Dim0Face1'):
      expected = 8.380952381e-05 * np.ones((12, 2, 12), dtype=np.float32)
      self.assertAllClose(expected, p[2:-2, -2:, 2:-2])

    with self.subTest(name='Dim1Face0'):
      expected = -8.380952381e-05 * np.ones((12, 12, 2), dtype=np.float32)
      self.assertAllClose(expected, p[2:-2, 2:-2, :2])

    with self.subTest(name='Dim1Face0'):
      expected = 3.352380952e-04 * np.ones((12, 12, 2), dtype=np.float32)
      self.assertAllClose(expected, p[2:-2, 2:-2, -2:])

    with self.subTest(name='Dim2Face0'):
      expected = -4.19047619e-05 * np.ones((2, 12, 12), dtype=np.float32)
      self.assertAllClose(expected, p[:2, 2:-2, 2:-2])

    with self.subTest(name='Dim2Face1'):
      expected = 1.676190476e-04 * np.ones((2, 12, 12), dtype=np.float32)
      self.assertAllClose(expected, p[-2:, 2:-2, 2:-2])

    with self.subTest(name='Interior'):
      expected = np.zeros((12, 12, 12), dtype=np.float32)
      self.assertAllClose(expected, p[2:-2, 2:-2, 2:-2])

  @parameterized.parameters(*itertools.product(_RHO_INFO, _SUBITER))
  def testPressureCorrectorUpdatesOutputsCorrectTensor(
      self, rho_info_fn, subiter):
    model = self.set_up_pressure()

    u = halo_exchange.inplace_halo_exchange(
        self.u,
        dims=self.halo_dims,
        replica_id=self.replica_id,
        replicas=self.replicas,
        replica_dims=self.replica_dims,
        periodic_dims=self.periodic_dims,
        boundary_conditions=self.bc_u)

    v = halo_exchange.inplace_halo_exchange(
        self.v,
        dims=self.halo_dims,
        replica_id=self.replica_id,
        replicas=self.replicas,
        replica_dims=self.replica_dims,
        periodic_dims=self.periodic_dims,
        boundary_conditions=self.bc_v)

    w = halo_exchange.inplace_halo_exchange(
        self.w,
        dims=self.halo_dims,
        replica_id=self.replica_id,
        replicas=self.replicas,
        replica_dims=self.replica_dims,
        periodic_dims=self.periodic_dims,
        boundary_conditions=self.bc_w)

    if (rho_info_fn ==
        pressure.ConstantDensityInfo):
      rho_info_val = 1.0
    elif (rho_info_fn ==
          pressure.VariableDensityInfo):
      rho_info_val = tf.unstack(
          tf.constant(np.reshape(np.arange(512, dtype=np.float32), (8, 8, 8))))

    rho_info = rho_info_fn(rho_info_val)

    dp, monitor_vars = self.evaluate(
        model._pressure_corrector_update(self.replica_id, self.replicas, {
            'u': u,
            'v': v,
            'w': w,
            'rho_u': u,
            'rho_v': v,
            'rho_w': w,
            'p': self.p,
        }, rho_info, subiter))

    expected_monitor_keys = [
        'MONITOR_pressure_convergence_l-1',
        'MONITOR_pressure_convergence_l-inf',
        'MONITOR_pressure_convergence_solver-iterations',
        # Missing for Jacobi solver:
        #   'MONITOR_pressure_convergence_solver-l-2'
        'MONITOR_pressure_raw_b',
        'MONITOR_pressure_raw_b-term-div',
        'MONITOR_pressure_raw_b-term-drho-dt',
        'MONITOR_pressure_raw_b-term-source-rho',
        'MONITOR_pressure_raw_convergence',
        'MONITOR_pressure_raw_dp',
        # As `rho` is not passed in:
        #   'MONITOR_pressure_raw_p-rho',
        'MONITOR_pressure_raw_p-rho-v',
        'MONITOR_pressure_raw_p-u',
        'MONITOR_pressure_raw_p-w',
        'MONITOR_pressure_scalar_b-l-inf',
        'MONITOR_pressure_scalar_b-term-div-l-1',
        'MONITOR_pressure_scalar_b-term-drho-dt-l-2',
        'MONITOR_pressure_scalar_b-term-source-rho-l-inf',
        'MONITOR_pressure_scalar_dp-l-2',
        'MONITOR_pressure_scalar_p-l-1',
        # As `rho` is not passed in:
        #   'MONITOR_pressure_scalar_p-rho-l-1',
        'MONITOR_pressure_scalar_p-rho-u-l-1',
        'MONITOR_pressure_scalar_p-rho-v-l-2',
        'MONITOR_pressure_scalar_p-rho-w-l-inf',
        'MONITOR_pressure_scalar_p-u-l-inf',
        'MONITOR_pressure_scalar_p-v-l-2',
        'MONITOR_pressure_scalar_p-w-l-1',
    ]
    if subiter is None:
      self.assertCountEqual(monitor_vars.keys(), expected_monitor_keys)
    else:
      self.assertCountEqual(
          monitor_vars.keys(),
          expected_monitor_keys + [
              # Monitor vars from sub iterations
              'MONITOR_pressure_subiter-scalar_convergence_l-inf',
              'MONITOR_pressure_subiter-scalar_convergence_l-2',
          ])

    self.assertLen(dp, 8)
    self.assertLen(dp[0], 8)
    self.assertAllClose(monitor_vars['MONITOR_pressure_raw_dp'], np.stack(dp))

    if isinstance(rho_info, pressure.ConstantDensityInfo):
      self.assertAllClose(dp[1][1, 1], np.float32(-7.2041254))
      self.assertAllClose(dp[1][2, 1], np.float32(-6.099423))
      self.assertAllClose(dp[1][1, 2], np.float32(-10.6219225))
      self.assertAllClose(dp[2][1, 1], np.float32(1.1349825))

      self.assertAllClose(monitor_vars['MONITOR_pressure_convergence_l-1'],
                          32439.387)
      self.assertAllClose(monitor_vars['MONITOR_pressure_convergence_l-inf'],
                          471.82547)

      self.assertAllClose(monitor_vars['MONITOR_pressure_scalar_b-l-inf'],
                          814.1482, 4)
      self.assertAllClose(
          monitor_vars['MONITOR_pressure_scalar_b-term-div-l-1'], 92682.0)
      self.assertAllClose(
          monitor_vars['MONITOR_pressure_scalar_b-term-drho-dt-l-2'], 0.0)
      self.assertAllClose(
          monitor_vars['MONITOR_pressure_scalar_b-term-source-rho-l-inf'], 0.0)
      self.assertAllClose(monitor_vars['MONITOR_pressure_scalar_dp-l-2'],
                          107.47147)
      self.assertAllClose(monitor_vars['MONITOR_pressure_scalar_p-l-1'], 2968.0)

      # Sampling the raw convergence
      self.assertAllClose(
          monitor_vars['MONITOR_pressure_raw_convergence'][1, 1, 3], -214.11798)
      self.assertAllClose(
          monitor_vars['MONITOR_pressure_raw_convergence'][3, 3, 2], -98.12996)
      self.assertAllClose(
          monitor_vars['MONITOR_pressure_raw_convergence'][6, 5, 5], 187.4802)

      if subiter is not None:
        l2_expected = np.zeros(3)
        l2_expected[subiter] = 2854.2053
        l_inf_expected = np.zeros(3)
        l_inf_expected[subiter] = 471.82547

        self.assertAllClose(
            monitor_vars['MONITOR_pressure_subiter-scalar_convergence_l-2'],
            l2_expected)
        self.assertAllClose(
            monitor_vars['MONITOR_pressure_subiter-scalar_convergence_l-inf'],
            l_inf_expected)

    elif isinstance(rho_info, pressure.VariableDensityInfo):
      self.assertAllClose(dp[1][1, 1], np.float32(283.92685))
      self.assertAllClose(dp[1][2, 1], np.float32(272.26965))
      self.assertAllClose(dp[1][1, 2], np.float32(278.9138))
      self.assertAllClose(dp[2][1, 1], np.float32(190.17072))

      self.assertAllClose(monitor_vars['MONITOR_pressure_convergence_l-1'],
                          2036889.5)
      self.assertAllClose(monitor_vars['MONITOR_pressure_convergence_l-inf'],
                          17619.986)

      self.assertAllClose(monitor_vars['MONITOR_pressure_scalar_b-l-inf'],
                          26064.148)
      self.assertAllClose(
          monitor_vars['MONITOR_pressure_scalar_b-term-div-l-1'], 92682.0)
      self.assertAllClose(
          monitor_vars['MONITOR_pressure_scalar_b-term-drho-dt-l-2'], 667894.1)
      self.assertAllClose(
          monitor_vars['MONITOR_pressure_scalar_b-term-source-rho-l-inf'], 0.0)
      self.assertAllClose(monitor_vars['MONITOR_pressure_scalar_dp-l-2'],
                          4485.634)
      self.assertAllClose(monitor_vars['MONITOR_pressure_scalar_p-l-1'], 2968.0)

      # Sampling the raw convergence
      self.assertAllClose(
          monitor_vars['MONITOR_pressure_raw_convergence'][1, 1, 3], 17414.74)
      self.assertAllClose(
          monitor_vars['MONITOR_pressure_raw_convergence'][3, 3, 2], 3651.8704)
      self.assertAllClose(
          monitor_vars['MONITOR_pressure_raw_convergence'][6, 5, 5], -16754.139)

      if subiter is not None:
        l2_expected = np.zeros(3)
        l2_expected[subiter] = 158304.92
        l_inf_expected = np.zeros(3)
        l_inf_expected[subiter] = 17619.986
        self.assertAllClose(
            monitor_vars['MONITOR_pressure_subiter-scalar_convergence_l-2'],
            l2_expected)
        self.assertAllClose(
            monitor_vars['MONITOR_pressure_subiter-scalar_convergence_l-inf'],
            l_inf_expected)

    self.assertIsInstance(
        monitor_vars['MONITOR_pressure_convergence_solver-iterations'],
        types.NP_DTYPE)
    self.assertEqual(
        monitor_vars['MONITOR_pressure_convergence_solver-iterations'], 1)

    self.assertAllClose(monitor_vars['MONITOR_pressure_scalar_p-rho-u-l-1'],
                        650.0)
    self.assertAllClose(monitor_vars['MONITOR_pressure_scalar_p-rho-v-l-2'],
                        40.211937)
    self.assertAllClose(monitor_vars['MONITOR_pressure_scalar_p-rho-w-l-inf'],
                        4.0)

    self.assertAllClose(monitor_vars['MONITOR_pressure_scalar_p-u-l-inf'], 3.0)
    self.assertAllClose(monitor_vars['MONITOR_pressure_scalar_p-v-l-2'],
                        40.211937)
    self.assertAllClose(monitor_vars['MONITOR_pressure_scalar_p-w-l-1'], 564.0)


if __name__ == '__main__':
  googletest.main()
