"""Tests for google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.equations.scalars."""

import itertools

from absl import flags
from absl.testing import parameterized
import numpy as np
from swirl_lm.equations import scalars
from swirl_lm.utility import components_debug
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_parameters_pb2


FLAGS = flags.FLAGS


class ScalarsTest(tf.test.TestCase, parameterized.TestCase):

  _GRAVITY_AND_THERMODYNAMICS_PBTXT = (R'gravity_direction { '
                                       R'  dim_0: 0.0 dim_1: 0.0 dim_2: -1.0 '
                                       R'}  '
                                       R'thermodynamics {  '
                                       R'  water {  '
                                       R'    r_v: 461.89  '
                                       R'    t_0: 273.0  '
                                       R'    t_min: 250.0  '
                                       R'    t_freeze: 273.15  '
                                       R'    t_triple: 273.16  '
                                       R'    p_triple: 611.7  '
                                       R'    e_int_v0: 2.132e6  '
                                       R'    e_int_i0: 3.34e5  '
                                       R'    lh_v0: 2.258e6  '
                                       R'    lh_s0: 2.592e6  '
                                       R'    cv_d: 716.9  '
                                       R'    cv_v: 1397.11  '
                                       R'    cv_l: 4217.4  '
                                       R'    cv_i: 2050.0  '
                                       R'    cp_v: 1859.0  '
                                       R'    cp_l: 4219.9  '
                                       R'    cp_i: 2050.0  '
                                       R'    max_temperature_iterations: 100  '
                                       R'    temperature_tolerance: 1e-3  '
                                       R'    num_density_iterations: 10  '
                                       R'    geo_static_reference_state {  '
                                       R'      t_s: 290.4 '
                                       R'      height: 8000.0  '
                                       R'      delta_t: 60.0  '
                                       R'    }'
                                       R'  } '
                                       R'}  ')

  def setUp(self):
    super(ScalarsTest, self).setUp()

    # Set up a (8, 8, 8) mesh. Only the point at (2, 2, 2) is tested as a
    # reference.
    self.u = [
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant([[0] * 8, [0] * 8, [0, 0, 2, 0, 0, 0, 0, 0], [0] * 8,
                     [0] * 8, [0] * 8, [0] * 8, [0] * 8],
                    dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
    ]

    self.v = [
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant([[0] * 8, [0] * 8, [0, 0, -3, 0, 0, 0, 0, 0], [0] * 8,
                     [0] * 8, [0] * 8, [0] * 8, [0] * 8],
                    dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
    ]

    self.w = [
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant([[0] * 8, [0] * 8, [0, 0, 4, 0, 0, 0, 0, 0], [0] * 8,
                     [0] * 8, [0] * 8, [0] * 8, [0] * 8],
                    dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
        tf.constant(0, shape=(8, 8), dtype=tf.float32),
    ]

    self.p = [
        tf.constant(2, shape=(8, 8), dtype=tf.float32),
        tf.constant(6, shape=(8, 8), dtype=tf.float32),
        tf.constant([[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11],
                     [8, 7, 6, 5, 4, 3, 2, 1], [4, 3, 2, 1, 0, -1, -2, -3],
                     [0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11],
                     [8, 7, 6, 5, 4, 3, 2, 1], [4, 3, 2, 1, 0, -1, -2, -3]],
                    dtype=tf.float32),
        tf.constant(8, shape=(8, 8), dtype=tf.float32),
        tf.constant(10, shape=(8, 8), dtype=tf.float32),
        tf.constant(8, shape=(8, 8), dtype=tf.float32),
        tf.constant(6, shape=(8, 8), dtype=tf.float32),
        tf.constant(2, shape=(8, 8), dtype=tf.float32),
    ]

    self.sc = [
        tf.constant(0.1, shape=(8, 8), dtype=tf.float32),
        tf.constant(0.5, shape=(8, 8), dtype=tf.float32),
        tf.constant([[0.4, 0.5, 0.6, 0.7, 0.0, 0.1, 0.2, 0.3],
                     [0.5, 0.6, 0.7, 0.0, 0.1, 0.2, 0.3, 0.4],
                     [0.6, 0.7, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                     [0.7, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0],
                     [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0, 0.1],
                     [0.3, 0.4, 0.5, 0.6, 0.7, 0.0, 0.1, 0.2]],
                    dtype=tf.float32),
        tf.constant(0.7, shape=(8, 8), dtype=tf.float32),
        tf.constant(0.9, shape=(8, 8), dtype=tf.float32),
        tf.constant(0.7, shape=(8, 8), dtype=tf.float32),
        tf.constant(0.5, shape=(8, 8), dtype=tf.float32),
        tf.constant(0.3, shape=(8, 8), dtype=tf.float32),
    ]

    self.kernel_op = get_kernel_fn.ApplyKernelConvOp(4)

  def set_up_scalars(self, dbg, pb_config=None):
    """Initializes the `Velocity` object."""
    if pb_config is not None:
      pbtxt = pb_config
    else:
      pbtxt = (R'scalars {  '
               R'  name: "Z"  '
               R'  diffusivity: 1e-2  '
               R'  density: 1.0   '
               R'  molecular_weight: 0.29  '
               R'  solve_scalar: true  '
               R'}  ')
    config = text_format.Parse(
        pbtxt,
        incompressible_structured_mesh_parameters_pb2
        .IncompressibleNavierStokesParameters())
    FLAGS.cx = 1
    FLAGS.cy = 1
    FLAGS.cz = 1
    FLAGS.nx = 8
    FLAGS.ny = 8
    FLAGS.nz = 8
    FLAGS.lx = 3.0
    FLAGS.ly = 0.75
    FLAGS.lz = 1.5
    FLAGS.halo_width = 2
    FLAGS.dt = 1e-2
    FLAGS.simulation_debug = dbg
    FLAGS.num_boundary_points = 0
    params = (
        incompressible_structured_mesh_config
        .IncompressibleNavierStokesParameters(config))

    dbg_model = components_debug.ComponentsDebug(params) if dbg else None

    return scalars.Scalars(self.kernel_op, params, dbg=dbg_model)

  @test_util.run_in_graph_and_eager_modes
  def testConservativeScalarUpdatesOutputsCorrectTensor(self):
    """Function value at [2, 2, 2] is correct."""
    model = self.set_up_scalars(False)

    states = {
        'u': self.u,
        'v': self.v,
        'w': self.w,
        'rho_u': self.u,
        'rho_v': self.v,
        'rho_w': self.w,
        'p': self.p,
        'rho': [tf.ones_like(u, dtype=tf.float32) for u in self.u],
    }
    additional_states = {
        'diffusivity': [
            1e-2 * tf.ones_like(u, dtype=tf.float32) for u in self.u
        ]
    }

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    scalar_rhs = model._generic_scalar_update(replica_id, replicas, 'Z', states,
                                              additional_states)

    rhs_sc = self.evaluate(scalar_rhs(self.sc))

    self.assertLen(rhs_sc, 8)
    self.assertAllClose(rhs_sc[2][2, 2], np.float32(0.134))

  @test_util.run_in_graph_and_eager_modes
  def testConservativeScalarUpdatesOutputsCorrectTensorWithDebugMode(self):
    """Function value at [2, 2, 2] is correct."""
    model = self.set_up_scalars(False)
    states = {
        'u': self.u,
        'v': self.v,
        'w': self.w,
        'rho_u': self.u,
        'rho_v': self.v,
        'rho_w': self.w,
        'p': self.p,
        'rho': [tf.ones_like(u, dtype=tf.float32) for u in self.u],
    }
    additional_states = {
        'diffusivity': [
            1e-2 * tf.ones_like(u, dtype=tf.float32) for u in self.u
        ]
    }

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    scalar_rhs = model._generic_scalar_update(replica_id, replicas, 'Z', states,
                                              additional_states, True)

    terms = self.evaluate(scalar_rhs(self.sc))

    with self.subTest(name='ConvectionX'):
      self.assertAllClose(terms['conv_x'][2][2, 2], -0.5)

    with self.subTest(name='ConvectionY'):
      self.assertAllClose(terms['conv_y'][2][2, 2], 1.2)

    with self.subTest(name='ConvectionZ'):
      self.assertAllClose(terms['conv_z'][2][2, 2], -0.65)

    with self.subTest(name='DiffusionX'):
      self.assertAllClose(terms['diff_x'][2][2, 2], 0.008)

    with self.subTest(name='DiffusionY'):
      self.assertAllClose(terms['diff_y'][2][2, 2], 0.128)

    with self.subTest(name='DiffusionZ'):
      self.assertAllClose(terms['diff_z'][2][2, 2], 0.048)

    with self.subTest(name='Source'):
      self.assertAllClose(terms['source'][2][2, 2], 0.0)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=3)))
  @test_util.run_in_graph_and_eager_modes
  def testEtUpdatesOutputsCorrectTensor(self, include_radiation,
                                        include_subsidence,
                                        include_precipitation):
    """Total energy RHS value at [4, 4, 4] is correct."""
    pbtxt = (
        self._GRAVITY_AND_THERMODYNAMICS_PBTXT + R'scalars {  '
        R'  name: "e_t"  '
        R'  total_energy {  '
        R'    include_radiation:  ' +
        (R'true' if include_radiation else R'false') +
        R'    include_subsidence:  ' +
        (R'true' if include_subsidence else R'false') +
        R'    include_precipitation:  ' +
        (R'true' if include_precipitation else R'false') + R'  }'
        R'}')
    model = self.set_up_scalars(False, pbtxt)

    ones = tf.ones((int(4), 8, 8), dtype=tf.float32)
    buf = np.zeros((8, 8, 8), dtype=np.float32)
    buf[4, 4, 4] = 2
    u = tf.unstack(tf.convert_to_tensor(buf))

    buf = np.zeros((8, 8, 8), dtype=np.float32)
    buf[4, 4, 4] = -3
    v = tf.unstack(tf.convert_to_tensor(buf))

    buf = np.zeros((8, 8, 8), dtype=np.float32)
    buf[4, 4, 4] = 4
    w = tf.unstack(tf.convert_to_tensor(buf))
    states = {
        'u': u,
        'v': v,
        'w': w,
        'rho_u': u,
        'rho_v': v,
        'rho_w': w,
        'p': self.p,
        'rho': [tf.ones_like(u, dtype=tf.float32) for u in self.u],
        'q_t': tf.unstack(tf.concat([0.009 * ones, 0.0015 * ones], axis=0)),
        'q_r': tf.unstack(tf.concat([0.0008 * ones, 0.0005 * ones], axis=0)),
    }
    zz = np.transpose(
        np.tile(np.linspace(600.0, 1000.0, 8, dtype=np.float32), (8, 8, 1)),
        (2, 0, 1))
    additional_states = {
        'diffusivity': [
            1e-2 * tf.ones_like(u, dtype=tf.float32) for u in self.u
        ],
        'nu_t': [1e-2 * tf.ones_like(u, dtype=tf.float32) for u in self.u],
        'zz': tf.unstack(tf.convert_to_tensor(zz)),
    }
    sc = tf.unstack(tf.concat([1.5e4 * ones, 1.8e4 * ones], axis=0))

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    scalar_rhs = model._e_t_update(replica_id, replicas, states,
                                   additional_states)

    rhs_e_t = self.evaluate(scalar_rhs(sc))

    self.assertLen(rhs_e_t, 8)
    expected_all = [[[-19442.453, -19451.49], [-19422.984, -19432.021]],
                    [[-19459.172, -19468.209], [-19439.701, -19448.74]]]
    expected = (
        expected_all[int(include_radiation)][int(include_subsidence)][int(
            include_precipitation)])
    actual = rhs_e_t[4][4, 4]
    self.assertAllClose(actual, np.float32(expected))

  @parameterized.parameters(
      {
          'scalar_name': 'q_t',
          'include_precipitation': False,
          'include_subsidence': False,
          'expected': 0.0
      }, {
          'scalar_name': 'q_t',
          'include_precipitation': False,
          'include_subsidence': True,
          'expected': 7.88e-7
      }, {
          'scalar_name': 'q_t',
          'include_precipitation': True,
          'include_subsidence': False,
          'expected': -0.001558865
      }, {
          'scalar_name': 'q_t',
          'include_precipitation': True,
          'include_subsidence': True,
          'expected': -0.001558077
      }, {
          'scalar_name': 'q_r',
          'include_precipitation': True,
          'include_subsidence': True,
          'expected': -0.044734500
      })
  @test_util.run_in_graph_and_eager_modes
  def testHumidityUpdate(self, scalar_name, include_precipitation,
                         include_subsidence, expected):
    """Total humidity RHS value at [3, 3, 3] is correct."""

    pbtxt = (
        self._GRAVITY_AND_THERMODYNAMICS_PBTXT + R'scalars {  '
        R'  name: "q_t"  '
        R'  diffusivity: 1e-5  '
        R'  density: 1.0   '
        R'  molecular_weight: 0.018  '
        R'  solve_scalar: true  '
        R'  humidity {  '
        R'    include_subsidence:  ' +
        (R'true' if include_subsidence else R'false') +
        R'    include_precipitation:  ' +
        (R'true' if include_precipitation else R'false') + R'  } '
        R'}  ')

    if include_precipitation:
      pbtxt = (
          pbtxt + R'scalars {  '
          R'  name: "q_r"  '
          R'  diffusivity: 1e-5  '
          R'  density: 1.0   '
          R'  molecular_weight: 0.018  '
          R'  solve_scalar: true  '
          R'  humidity {  '
          R'    include_subsidence:  ' +
          (R'true' if include_subsidence else R'false') +
          R'    include_precipitation:  ' +
          (R'true' if include_precipitation else R'false') + R'  } '
          R'}  ')

    model = self.set_up_scalars(False, pbtxt)

    ones = tf.ones((int(4), 8, 8), dtype=tf.float32)
    buf = np.zeros((8, 8, 8), dtype=np.float32)
    buf[4, 4, 4] = 2
    u = tf.unstack(tf.convert_to_tensor(buf))

    buf = np.zeros((8, 8, 8), dtype=np.float32)
    buf[4, 4, 4] = -3
    v = tf.unstack(tf.convert_to_tensor(buf))

    buf = np.zeros((8, 8, 8), dtype=np.float32)
    buf[4, 4, 4] = 4
    w = tf.unstack(tf.convert_to_tensor(buf))
    # Internal energy of ~41000 J/(kg/m^3) at the sea surface of
    # temperature 292.5K with a humidity 0.011 kg/kg.
    e = 41e3
    q_t = 0.011
    q_r = 0.005
    states = {
        'u': u,
        'v': v,
        'w': w,
        'rho_u': self.u,
        'rho_v': self.v,
        'rho_w': self.w,
        'p': self.p,
        'rho': [tf.ones_like(u, dtype=tf.float32) for u in self.u],
        'e_t': [e * tf.ones_like(u, dtype=tf.float32) for u in self.u],
        'q_t': [q_t * tf.ones_like(u, dtype=tf.float32) for u in self.u],
        'q_r': [q_r * tf.ones_like(u, dtype=tf.float32) for u in self.u],
    }
    additional_states = {
        'diffusivity': [
            1e-2 * tf.ones_like(u, dtype=tf.float32) for u in self.u
        ],
        'zz': tf.unstack(tf.linspace(600.0, 1000.0, 8)),
    }
    sc = tf.unstack(tf.concat([0.08 * ones, 0.1 * ones], axis=0))

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    scalar_rhs = model._humidity_update(
        replica_id,
        replicas,
        states,
        additional_states,
        scalar_name=scalar_name)

    rhs = self.evaluate(scalar_rhs(sc))

    self.assertLen(rhs, 8)
    self.assertAllClose(rhs[2][2, 2], np.float32(expected))


if __name__ == '__main__':
  tf.test.main()
