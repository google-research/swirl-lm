"""Tests for google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.equations.utils."""

import functools

import numpy as np
from swirl_lm.communication import halo_exchange
from swirl_lm.equations import utils
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf
from google3.net.proto2.python.public import text_format
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_parameters_pb2
from google3.testing.pybase import parameterized


@test_util.run_all_in_graph_and_eager_modes
class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Initializes shared fields for tests."""
    super(UtilsTest, self).setUp()

    self.kernel_op = get_kernel_fn.ApplyKernelConvOp(4)
    self.halo_dims = [0, 1, 2]
    self.replica_id = tf.constant(0)
    self.replicas = np.array([[[0]]], dtype=np.int32)
    self.replica_dims = [0, 1, 2]

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

  def _set_up_params(self, pbtxt=''):
    """Creates a simulation configuration handler."""
    proto = text_format.Parse(
        pbtxt,
        incompressible_structured_mesh_parameters_pb2
        .IncompressibleNavierStokesParameters())

    return (incompressible_structured_mesh_config
            .IncompressibleNavierStokesParameters(proto))

  def testShearStressOutputsCorrectStressTensors(self):
    """Shear stress near point [1, 1, 1] are non-trivial and correct."""
    nu = 1e-5
    dx = 1e-3
    dy = 2e-3
    dz = 5e-3

    mu = [nu * tf.ones_like(u, dtype=tf.float32) for u in self.u]
    tau = utils.shear_stress(self.kernel_op, mu, dx, dy, dz, self.u, self.v,
                             self.w)

    tau_value = self.evaluate(tau)

    with self.subTest(name='Tau11At211'):
      self.assertAlmostEqual(tau_value['xx'][1][2, 1], -0.0133333, 6)

    with self.subTest(name='Tau12At121'):
      self.assertAlmostEqual(tau_value['xy'][1][1, 2], -0.005)

    with self.subTest(name='Tau13At112'):
      self.assertAlmostEqual(tau_value['xz'][2][1, 1], -0.002)

    with self.subTest(name='Tau21At211'):
      self.assertAlmostEqual(tau_value['yx'][1][2, 1], 0.015)

    with self.subTest(name='Tau22At121'):
      self.assertAlmostEqual(tau_value['yy'][1][1, 2], 0.01)

    with self.subTest(name='Tau23At112'):
      self.assertAlmostEqual(tau_value['yz'][2][1, 1], 0.003)

    with self.subTest(name='Tau31At211'):
      self.assertAlmostEqual(tau_value['zx'][1][2, 1], -0.02)

    with self.subTest(name='Tau32At121'):
      self.assertAlmostEqual(tau_value['zy'][1][1, 2], -0.01)

    with self.subTest(name='Tau33At112'):
      self.assertAlmostEqual(tau_value['zz'][2][1, 1], -0.0053333)

  def testShearStressWithBCComputesCorrectly(self):
    """Checks if the shear stress is updated correctly."""
    mu = [
        tf.constant(1e-5, dtype=tf.float32),
    ] * len(self.u)
    dx = 0.1
    dy = 0.2
    dz = 0.5

    bc_tau_xz = [[None, None], [None, None],
                 [(halo_exchange.BCType.DIRICHLET, 6.0),
                  (halo_exchange.BCType.NEUMANN, 0.0)]]
    bc_fn = {
        'xz':
            functools.partial(
                halo_exchange.inplace_halo_exchange,
                dims=self.halo_dims,
                replica_id=self.replica_id,
                replicas=self.replicas,
                replica_dims=self.replica_dims,
                periodic_dims=[True, True, False],
                boundary_conditions=bc_tau_xz)
    }

    tau = self.evaluate(
        utils.shear_stress(self.kernel_op, mu, dx, dy, dz, self.u, self.v,
                           self.w, bc_fn))

    with self.subTest(name='tau00'):
      self.assertAlmostEqual(tau['xx'][1][2, 1], -1.3333332e-04)

    with self.subTest(name='tau01'):
      self.assertAlmostEqual(tau['xy'][1][1, 2], -5e-5)

    with self.subTest(name='tau02'):
      self.assertAlmostEqual(tau['xz'][0][1, 1], 6.0)
      self.assertAlmostEqual(tau['xz'][2][1, 1], -2e-5)

    with self.subTest(name='tau10'):
      self.assertAlmostEqual(tau['yx'][1][1, 2], -5e-5)

    with self.subTest(name='tau11'):
      self.assertAlmostEqual(tau['yy'][1][1, 2], 1e-4)

    with self.subTest(name='tau12'):
      self.assertAlmostEqual(tau['yz'][2][1, 1], 3e-5)

    with self.subTest(name='tau20'):
      self.assertAlmostEqual(tau['zx'][2][1, 1], -2e-5)

    with self.subTest(name='tau21'):
      self.assertAlmostEqual(tau['zy'][2][1, 1], 3e-5)

    with self.subTest(name='tau22'):
      self.assertAlmostEqual(tau['zz'][2][1, 1], -5.333333e-05)

  def testShearStressFluxComputesCorrectly(self):
    """Checks if the shear stress is updated correctly."""
    nu = 1e-5
    dx = 0.1
    dy = 0.2
    dz = 0.5

    u = tf.unstack(tf.convert_to_tensor(
        np.reshape(np.arange(64, dtype=np.float32), (4, 4, 4))))
    v = tf.unstack(tf.convert_to_tensor(
        2.0 * np.reshape(np.arange(64, dtype=np.float32), (4, 4, 4))))
    w = tf.unstack(tf.convert_to_tensor(
        3.0 * np.reshape(np.arange(64, dtype=np.float32), (4, 4, 4))))
    mu = [nu * tf.ones_like(u_i, dtype=tf.float32) for u_i in u]

    params = self._set_up_params()

    shear_flux_fn = utils.shear_flux(params)

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])

    tau = self.evaluate(
        shear_flux_fn(self.kernel_op, replica_id, replicas, mu, dx, dy, dz, u,
                      v, w))

    with self.subTest(name='Tau00'):
      expected = -0.0001733333333 * np.ones((2, 2, 2), dtype=np.float32)
      self.assertAllClose(expected,
                          np.stack(np.array(tau['xx'])[1:-1, 1:-1, 1:-1]))

    with self.subTest(name='Tau01'):
      expected = 0.00085 * np.ones((2, 2, 2), dtype=np.float32)
      self.assertAllClose(expected,
                          np.stack(np.array(tau['xy'])[1:-1, 1:-1, 1:-1]))

    with self.subTest(name='Tau02'):
      expected = 0.00152 * np.ones((2, 2, 2), dtype=np.float32)
      self.assertAllClose(expected,
                          np.stack(np.array(tau['xz'])[1:-1, 1:-1, 1:-1]))

    with self.subTest(name='Tau10'):
      expected = 0.00085 * np.ones((2, 2, 2), dtype=np.float32)
      self.assertAllClose(expected,
                          np.stack(np.array(tau['yx'])[1:-1, 1:-1, 1:-1]))

    with self.subTest(name='Tau11'):
      expected = -0.0007733333333 * np.ones((2, 2, 2), dtype=np.float32)
      self.assertAllClose(expected,
                          np.stack(np.array(tau['yy'])[1:-1, 1:-1, 1:-1]))

    with self.subTest(name='Tau12'):
      expected = 0.00079 * np.ones((2, 2, 2), dtype=np.float32)
      self.assertAllClose(expected,
                          np.stack(np.array(tau['yz'])[1:-1, 1:-1, 1:-1]))

    with self.subTest(name='Tau20'):
      expected = 0.00152 * np.ones((2, 2, 2), dtype=np.float32)
      self.assertAllClose(expected,
                          np.stack(np.array(tau['zx'])[1:-1, 1:-1, 1:-1]))

    with self.subTest(name='Tau21'):
      expected = 0.00079 * np.ones((2, 2, 2), dtype=np.float32)
      self.assertAllClose(expected,
                          np.stack(np.array(tau['zy'])[1:-1, 1:-1, 1:-1]))

    with self.subTest(name='Tau222'):
      expected = 0.0009466666667 * np.ones((2, 2, 2), dtype=np.float32)
      self.assertAllClose(expected,
                          np.stack(np.array(tau['zz'])[1:-1, 1:-1, 1:-1]))

  @parameterized.named_parameters(('Dim0', 0), ('Dim1', 1), ('Dim2', 2))
  def testShearStressFluxComputesCorrectlyWithMOS(self, dim):
    """Checks if the shear stress is updated correctly with the MOS model."""
    pbtxt = (R'diffusion_scheme: DIFFUSION_SCHEME_CENTRAL_3 '
             R'boundary_models { '
             R'  most { '
             R'    t_s: 265.0 '
             R'    z_0: 0.1 '
             R'    beta_m: 5.0 '
             R'    beta_h: 6.0 '
             R'    gamma_m: 15.0 '
             R'    gamma_h: 15.0 '
             R'  } '
             R'} ')
    if dim == 0:
      pbtxt += (R'gravity_direction { '
                R'  dim_0: -1.0 dim_1: 0.0 dim_2: 0.0 '
                R'} ')
    elif dim == 1:
      pbtxt += (R'gravity_direction { '
                R'  dim_0: 0.0 dim_1: -1.0 dim_2: 0.0 '
                R'} ')
    else:  # dim == 2
      pbtxt += (R'gravity_direction { '
                R'  dim_0: 0.0 dim_1: 0.0 dim_2: -1.0 '
                R'} ')

    params = self._set_up_params(pbtxt)
    params.cx = 1
    params.cy = 1
    params.cz = 1
    params.nx = 8
    params.ny = 8
    params.nz = 8
    params.lx = 6.0
    params.ly = 6.0
    params.lz = 6.0
    params.halo_width = 2

    velocity_keys = ('u', 'v', 'w')
    tangential_dims = [0, 1, 2]
    del tangential_dims[dim]

    helper_vars = {}
    helper_vars.update({
        velocity_keys[tangential_dims[0]]:
            tf.unstack(7.5 * tf.ones((8, 8, 8), dtype=tf.float32)),
        velocity_keys[dim]:
            tf.unstack(tf.zeros((8, 8, 8), dtype=tf.float32)),
        velocity_keys[tangential_dims[1]]:
            tf.unstack(-5.0 * tf.ones((8, 8, 8), dtype=tf.float32)),
        'theta':
            tf.unstack(300.0 * tf.ones((8, 8, 8), dtype=tf.float32)),
    })
    mu = tf.unstack(0.1 * tf.ones((8, 8, 8), dtype=tf.float32))

    params = self._set_up_params(pbtxt)

    shear_flux_fn = utils.shear_flux(params)

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])

    tau = self.evaluate(
        shear_flux_fn(self.kernel_op, replica_id, replicas, mu, params.dx,
                      params.dy, params.dz, helper_vars['u'], helper_vars['v'],
                      helper_vars['w'], helper_vars))

    for key, val in tau.items():
      with self.subTest(name=f'Tau{key}'):
        expected = np.zeros((6, 6, 6), dtype=np.float32)
        if dim == 0:
          if key == 'yx':
            expected[:, 0, :] = 1.675208
          elif key == 'zx':
            expected[:, 0, :] = -1.116805
        elif dim == 1:
          if key == 'xy':
            expected[..., 0] = 1.675208
          elif key == 'zy':
            expected[..., 0] = -1.116805
        else:  # dim == 2
          if key == 'xz':
            expected[0, ...] = 1.675208
          elif key == 'yz':
            expected[0, ...] = -1.116805

        self.assertAllClose(expected, np.array(val)[1:-1, 1:-1, 1:-1])

  def testSubsidenceVelocitySiebesma(self):
    """Checks if Siebesma's subsidence velocity is computed correctly."""
    z = tf.unstack(tf.linspace(0.0, 3000.0, 11))

    w = self.evaluate(utils.subsidence_velocity_siebesma(z))

    expected = -np.array([
        0.0, 0.0013, 0.0026, 0.0039, 0.0052, 0.0065, 0.00325, 0.0, 0.0, 0.0, 0.0
    ])

    self.assertAllClose(expected, w)

  def testSubsidenceVelocityStevens(self):
    """Checks if Stevens' subsidence velocity is computed correctly."""
    z = tf.unstack(tf.linspace(0.0, 3000.0, 11))

    w = self.evaluate(utils.subsidence_velocity_stevens(z))

    expected = -np.array([
        0.0, 0.001125, 0.00225, 0.003375, 0.0045, 0.005625, 0.00675, 0.007875,
        0.009, 0.010125, 0.01125
    ])

    self.assertAllClose(expected, w)

  _VERTICAL_DIMS = (0, 1, 2)

  @parameterized.parameters(*_VERTICAL_DIMS)
  def testSourceBySubsidenceVelocity(self, vertical_dim):
    """Checks if the source due to subsidence velocity is computed correctly."""
    n = 32
    dim_to_axis = (1, 2, 0)
    vertical_axis = dim_to_axis[vertical_dim]
    half_grid_shape = [n] * 3
    half_grid_shape[vertical_axis] = int(n/2)
    ones = tf.ones(half_grid_shape, dtype=tf.float32)

    rho = tf.unstack(tf.concat([1.2 * ones, 1.0 * ones], axis=vertical_axis))

    def linspace_tensor(start, stop, n):
      """Returns full state tensor with evenly spaced values along vertical."""
      tile_shape = [n] * 3
      tile_shape[vertical_axis] = 1
      t = tf.tile(
          tf.linspace([[start]], [[stop]], n, axis=vertical_axis),
          tile_shape)
      return tf.unstack(t)

    height = linspace_tensor(600.0, 1000.0, n)
    f = linspace_tensor(0.0, 0.05, n)
    dh = 400.0 / 27.0

    halos = [0, 0, 0]
    halos[vertical_dim] = 1
    src = self.evaluate(
        # Subsidence not defined at the boundaries.
        common_ops.strip_halos(
            utils.source_by_subsidence_velocity(self.kernel_op, rho, height, dh,
                                                f, vertical_dim), halos))

    z = np.linspace(600.0, 1000.0, n)
    expected_shape = [32] * 3
    expected_shape[vertical_axis] = 30
    expected = np.ones(expected_shape, dtype=np.float32)
    slices = [slice(0, None)] * 3
    for i in range(30):
      slices[vertical_axis] = i
      expected[slices] *= (0.05 / 31) / (400.0 / 27) * 3.75e-6 * z[i + 1]

    self.assertAllClose(expected, src)


if __name__ == '__main__':
  tf.test.main()
