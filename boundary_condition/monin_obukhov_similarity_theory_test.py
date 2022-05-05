"""Tests for google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.boundary_conditions.monin_obukhov_similarity_theory."""

import itertools
import os

from absl import flags
import numpy as np
from swirl_lm.boundary_condition import monin_obukhov_similarity_theory
from swirl_lm.equations import common
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.pyglib import gfile
from google3.pyglib import resources
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_parameters_pb2
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import physical_variable_keys_manager
from google3.testing.pybase import parameterized

FLAGS = flags.FLAGS


_TESTDATA_DIR = 'google3/third_party/py/swirl_lm/boundary_condition/test_data'

_MOST_CONFIG = 'monin_obukhov_similarity_theory_config.textpb'
_MOST_CONFIG_REG = 'monin_obukhov_similarity_theory_config_reg.textpb'
_MOST_CONFIG_NEGATIVE_HEAT_FLUX = (
    'monin_obukhov_similarity_theory_config_negative_heat_flux.textpb')
_MOST_CONFIG_ZERO_HEAT_FLUX = (
    'monin_obukhov_similarity_theory_config_zero_heat_flux.textpb')
_KEYS_VELOCITY = (common.KEY_U, common.KEY_V, common.KEY_W)
_DIM_TO_AXIS = (1, 2, 0)


def read_proto(filename):
  with gfile.GFile(
      resources.GetResourceFilename(os.path.join(_TESTDATA_DIR,
                                                 filename))) as f:
    return text_format.Parse(
        f.read(),
        incompressible_structured_mesh_parameters_pb2
        .IncompressibleNavierStokesParameters())


@test_util.run_all_in_graph_and_eager_modes
class MoninObukhovSimilarityTheoryTest(tf.test.TestCase,
                                       parameterized.TestCase):

  # pylint: disable=invalid-name
  def _setUp(self, filename=_MOST_CONFIG, vertical_dim=2, cx=2, cy=2, cz=2):
    """Initializes the `MoninObukhovSimilarityTheory` object from text proto."""
    FLAGS.cx = cx
    FLAGS.cy = cy
    FLAGS.cz = cz
    FLAGS.nx = 8
    FLAGS.ny = 8
    FLAGS.nz = 8
    FLAGS.lx = 6.0
    FLAGS.ly = 6.0
    FLAGS.lz = 6.0
    FLAGS.halo_width = 2
    self.params = (
        incompressible_structured_mesh_config
        .IncompressibleNavierStokesParameters(read_proto(filename)))
    if vertical_dim == 0:
      self.params.gravity_direction = [1.0, 0.0, 0.0]
    elif vertical_dim == 1:
      self.params.gravity_direction = [0.0, 1.0, 0.0]
    else:  # vertical_dim == 2
      self.params.gravity_direction = [0.0, 0.0, 1.0]

    self.model = (
        monin_obukhov_similarity_theory.monin_obukhov_similarity_theory_factory(
            self.params))
    self.bc_manager = (
        physical_variable_keys_manager.BoundaryConditionKeysHelper())
    if vertical_dim == 0:
      self.horizontal_dims = (1, 2)
    elif vertical_dim == 1:
      self.horizontal_dims = (0, 2)
    else:
      self.horizontal_dims = (0, 1)
    self.vertical_axis = _DIM_TO_AXIS[vertical_dim]

  def _linspace_tensor(self, start, stop, shape):
    """Returns full state tensor with evenly spaced values along vertical."""
    state_shape = [shape[i] for i in (2, 0, 1)]
    tile_shape = list(state_shape)
    tile_shape[self.vertical_axis] = 1
    t = tf.tile(
        tf.linspace([[start]], [[stop]],
                    state_shape[self.vertical_axis],
                    axis=self.vertical_axis), tile_shape)
    return tf.unstack(t)

  def _create_field_with_halos(self, value, halo_value, shape, vertical_dim):
    """Creates a diverse tensor with the first fluid layer set to `value`."""
    halo_shape = list(shape)
    halo_shape[vertical_dim] = self.params.halo_width
    # A partial state containing only the lower halos.
    halos = [halo_value * tf.ones(halo_shape[0:2], dtype=tf.float32)
            ] * halo_shape[2]
    shape[vertical_dim] -= self.params.halo_width
    partial_state = self._linspace_tensor(value, value + 100, shape)
    if vertical_dim == 2:
      return halos + partial_state
    else:
      return [
          tf.concat([h_i, f_i], axis=vertical_dim)
          for h_i, f_i in zip(halos, partial_state)
      ]

  @parameterized.named_parameters(('Tensor', 'TENSOR'),
                                  ('ListTensor', 'LIST_TENSOR'),
                                  ('OneListTensor', 'ONE_LIST_TENSOR'))
  def testStabilityCorrectionFunctionComputesCorrectly(self, option):
    """Checks if the stability function are correct with different buoyancy."""
    self._setUp()

    theta_val = np.array([260.0, 270.0, 265.0])
    zeta_val = np.array([0.01, 0.05, 0.025])

    theta = tf.convert_to_tensor(theta_val, dtype=tf.float32)
    zeta = tf.convert_to_tensor(zeta_val, dtype=tf.float32)
    if option == 'LIST_TENSOR':
      theta = tf.unstack(theta)
      zeta = tf.unstack(zeta)
    elif option == 'ONE_LIST_TENSOR':
      theta = [theta,]
      zeta = [zeta,]

    psi_m, psi_h = self.evaluate(
        self.model._stability_correction_function(zeta, theta))

    with self.subTest(name='PsiM'):
      expected = [-0.03940308019542371, -0.25, 0.0]
      if option == 'ONE_LIST_TENSOR':
        expected = np.expand_dims(expected, 0)
      self.assertAllClose(expected, psi_m)

    with self.subTest(name='PsiH'):
      expected = [-0.07960914357267133, -0.3, 0.0]
      if option == 'ONE_LIST_TENSOR':
        expected = np.expand_dims(expected, 0)
      self.assertAllClose(expected, psi_h)

  @parameterized.named_parameters(('Tensor', 'TENSOR'),
                                  ('ListTensor', 'LIST_TENSOR'),
                                  ('OneListTensor', 'ONE_LIST_TENSOR'))
  def testRichardsonNumberIsComputedCorrectly(self, option):
    """Checks if the Richardson number is computed correctly."""
    self._setUp()

    theta_val = np.array([260.0, 270.0, 265.0])
    u_val = np.array([2.0, -6.0, 10.0])
    v_val = np.array([-3.0, 8.0, -16.0])
    height = 2.0

    theta = tf.convert_to_tensor(theta_val, dtype=tf.float32)
    u = tf.convert_to_tensor(u_val, dtype=tf.float32)
    v = tf.convert_to_tensor(v_val, dtype=tf.float32)
    if option == 'LIST_TENSOR':
      theta = tf.unstack(theta)
      u = tf.unstack(u)
      v = tf.unstack(v)
    elif option == 'ONE_LIST_TENSOR':
      theta = [theta,]
      u = [u,]
      v = [v,]

    r = self.evaluate(self.model._richardson_number(theta, u, v, height))

    expected = [-0.029023668639053257, 0.0036333333333333335, 0.0]

    self.assertAllClose(expected, np.squeeze(r))

  @parameterized.named_parameters(('Tensor', 'TENSOR'),
                                  ('ListTensor', 'LIST_TENSOR'),
                                  ('OneListTensor', 'ONE_LIST_TENSOR'))
  def testNormlizedHeightIsComputedCorrectly(self, option):
    """Checks if the Obukhov length normalized height is computed correctly."""
    self._setUp()

    theta_val = np.array([260.0, 270.0, 265.0])
    u_val = np.array([2.0, -6.0, 10.0])
    v_val = np.array([-3.0, 8.0, -16.0])
    height = 2.0

    theta = tf.convert_to_tensor(theta_val, dtype=tf.float32)
    u = tf.convert_to_tensor(u_val, dtype=tf.float32)
    v = tf.convert_to_tensor(v_val, dtype=tf.float32)
    if option == 'LIST_TENSOR':
      theta = tf.unstack(theta)
      u = tf.unstack(u)
      v = tf.unstack(v)
    elif option == 'ONE_LIST_TENSOR':
      theta = [theta,]
      u = [u,]
      v = [v,]

    zeta = self.evaluate(self.model._normalized_height(theta, u, v, height))

    expected = [-0.086773, 0.011045, 0.0]

    if option == 'ONE_LIST_TENSOR':
      expected = np.expand_dims(expected, 0)

    self.assertAllClose(expected, zeta)

  @parameterized.named_parameters(('Tensor', 'TENSOR'),
                                  ('ListTensor', 'LIST_TENSOR'),
                                  ('OneListTensor', 'ONE_LIST_TENSOR'))
  def testSurfaceShearStressAndHeatFluxAreComputedCorrectly(self, option):
    """Checks if the surface shear stress and heat flux are correct."""
    self._setUp()

    theta_val = np.array([260.0, 270.0, 265.0])
    u_val = np.array([2.0, -6.0, 10.0])
    v_val = np.array([-3.0, 8.0, -16.0])
    height = 2.0

    theta = tf.convert_to_tensor(theta_val, dtype=tf.float32)
    u = tf.convert_to_tensor(u_val, dtype=tf.float32)
    v = tf.convert_to_tensor(v_val, dtype=tf.float32)
    if option == 'LIST_TENSOR':
      theta = tf.unstack(theta)
      u = tf.unstack(u)
      v = tf.unstack(v)
    elif option == 'ONE_LIST_TENSOR':
      theta = [theta,]
      u = [u,]
      v = [v,]

    tau_13, tau_23, q_3 = self.evaluate(
        self.model._surface_shear_stress_and_heat_flux(theta, u, v, height))

    with self.subTest(name='Tau13'):
      expected = [-0.1521845, 1.0313325, -3.3638682]

      if option == 'ONE_LIST_TENSOR':
        expected = np.expand_dims(expected, 0)

      self.assertAllClose(expected, tau_13)

    with self.subTest(name='Tau23'):
      expected = [0.22827676, -1.37511, 5.3821893]

      if option == 'ONE_LIST_TENSOR':
        expected = np.expand_dims(expected, 0)

      self.assertAllClose(expected, tau_23)

    with self.subTest(name='Q3'):
      expected = [0.4131136, -0.8563437, 0.0]

      if option == 'ONE_LIST_TENSOR':
        expected = np.expand_dims(expected, 0)

      self.assertAllClose(expected, q_3)

  @parameterized.named_parameters(('VerticalDimX', 0), ('VerticalDimY', 1),
                                  ('VerticalDimZ', 2))
  def testSurfaceShearStressAndHeatFluxUpdateFnProvidesCorrectValues(
      self, vertical_dim):
    """Checks if the surface shear stress and heat flux are correct."""
    self._setUp(vertical_dim=vertical_dim, cx=1, cy=1, cz=1)

    theta_val = 300.0 * np.ones((8, 8, 8), dtype=np.float32)
    u_val = np.zeros((8, 8, 8), dtype=np.float32)
    v_val = np.zeros((8, 8, 8), dtype=np.float32)
    w_val = np.zeros((8, 8, 8), dtype=np.float32)

    if vertical_dim == 0:
      v_val[:, 2, :] = 2.0
      w_val[:, 2, :] = -3.0
      theta_val[:, 2, :] = 260.0
    elif vertical_dim == 1:
      u_val[..., 2] = 2.0
      w_val[..., 2] = -3.0
      theta_val[..., 2] = 260.0
    else:  # vertical_dim == 2
      u_val[2, ...] = 2.0
      v_val[2, ...] = -3.0
      theta_val[2, ...] = 260.0

    states = {
        'theta': tf.unstack(tf.convert_to_tensor(theta_val, dtype=tf.float32)),
        'u': tf.unstack(tf.convert_to_tensor(u_val, dtype=tf.float32)),
        'v': tf.unstack(tf.convert_to_tensor(v_val, dtype=tf.float32)),
        'w': tf.unstack(tf.convert_to_tensor(w_val, dtype=tf.float32)),
    }

    tau_13, tau_23, q_3 = self.evaluate(
        self.model.surface_shear_stress_and_heat_flux_update_fn(states))

    results = {'tau_13': tau_13, 'tau_23': tau_23, 'q_3': q_3}
    for key, val in results.items():
      with self.subTest(name=f'{key}SliceShape'):
        if vertical_dim == 0:
          self.assertLen(val, 8)
          self.assertEqual(val[0].shape[0], 1)
          self.assertEqual(val[0].shape[1], 8)
        elif vertical_dim == 1:
          self.assertLen(val, 8)
          self.assertEqual(val[0].shape[0], 8)
          self.assertEqual(val[0].shape[1], 1)
        else:  # vertical_dim == 2
          self.assertEqual(val.shape[0], 8)
          self.assertEqual(val.shape[1], 8)

    with self.subTest(name='Tau13'):
      expected = -0.23988 * np.ones((8, 8), dtype=np.float32)

      self.assertAllClose(expected, np.squeeze(tau_13))

    with self.subTest(name='Tau23'):
      expected = 0.35982 * np.ones((8, 8), dtype=np.float32)

      self.assertAllClose(expected, np.squeeze(tau_23))

    with self.subTest(name='Q3'):
      expected = 0.62954 * np.ones((8, 8), dtype=np.float32)

      self.assertAllClose(expected, np.squeeze(q_3))

  _DIMS = (0, 1, 2)
  _THETA_AMBIENT = (400.0, 200.0)

  @parameterized.parameters(*itertools.product(_DIMS, _THETA_AMBIENT))
  def testSurfaceShearStressAndHeatFluxUpdateFnWithRegularizationIsCorrect(
      self, vertical_dim, theta_ambient):
    """Checks if the surface shear stress and heat flux are correct."""
    self._setUp(
        filename=_MOST_CONFIG_REG, vertical_dim=vertical_dim, cx=1, cy=1, cz=1)

    theta_val = theta_ambient * np.ones((8, 8, 8), dtype=np.float32)
    u_val = np.zeros((8, 8, 8), dtype=np.float32)
    v_val = np.zeros((8, 8, 8), dtype=np.float32)
    w_val = np.zeros((8, 8, 8), dtype=np.float32)

    if vertical_dim == 0:
      v_val[:, 2, :] = 2.0
      w_val[:, 2, :] = -3.0
    elif vertical_dim == 1:
      u_val[..., 2] = 2.0
      w_val[..., 2] = -3.0
    else:  # vertical_dim == 2
      u_val[2, ...] = 2.0
      v_val[2, ...] = -3.0

    states = {
        'theta': tf.unstack(tf.convert_to_tensor(theta_val, dtype=tf.float32)),
        'u': tf.unstack(tf.convert_to_tensor(u_val, dtype=tf.float32)),
        'v': tf.unstack(tf.convert_to_tensor(v_val, dtype=tf.float32)),
        'w': tf.unstack(tf.convert_to_tensor(w_val, dtype=tf.float32)),
    }

    tau_13, tau_23, q_3 = self.evaluate(
        self.model.surface_shear_stress_and_heat_flux_update_fn(states))

    results = {'tau_13': tau_13, 'tau_23': tau_23, 'q_3': q_3}
    for key, val in results.items():
      with self.subTest(name=f'{key}SliceShape'):
        if vertical_dim == 0:
          self.assertLen(val, 8)
          self.assertEqual(val[0].shape[0], 1)
          self.assertEqual(val[0].shape[1], 8)
        elif vertical_dim == 1:
          self.assertLen(val, 8)
          self.assertEqual(val[0].shape[0], 8)
          self.assertEqual(val[0].shape[1], 1)
        else:  # vertical_dim == 2
          self.assertEqual(val.shape[0], 8)
          self.assertEqual(val.shape[1], 8)

    with self.subTest(name='Tau13'):
      expected = -0.23988 * np.ones((8, 8), dtype=np.float32)

      self.assertAllClose(expected, np.squeeze(tau_13))

    with self.subTest(name='Tau23'):
      expected = 0.35982 * np.ones((8, 8), dtype=np.float32)

      self.assertAllClose(expected, np.squeeze(tau_23))

    with self.subTest(name='Q3'):
      expected = 0.62954 * np.ones((8, 8), dtype=np.float32)

      self.assertAllClose(expected, np.squeeze(q_3))

  @parameterized.named_parameters(('Tensor', 'TENSOR'),
                                  ('ListTensor', 'LIST_TENSOR'),
                                  ('OneListTensor', 'ONE_LIST_TENSOR'))
  def testExchangeCoefficientComputesCorrectly(self, option):
    """Checks if the exchange coefficient is correct with different buoyancy."""
    self._setUp()

    theta_val = np.array([260.0, 270.0, 265.0])
    u_val = np.array([2.0, -6.0, 10.0])
    v_val = np.array([-3.0, 8.0, -16.0])
    height = 2.0

    theta = tf.convert_to_tensor(theta_val, dtype=tf.float32)
    u = tf.convert_to_tensor(u_val, dtype=tf.float32)
    v = tf.convert_to_tensor(v_val, dtype=tf.float32)
    if option == 'LIST_TENSOR':
      theta = tf.unstack(theta)
      u = tf.unstack(u)
      v = tf.unstack(v)
    elif option == 'ONE_LIST_TENSOR':
      theta = [theta,]
      u = [u,]
      v = [v,]

    c_h = self.evaluate(self.model._exchange_coefficient(theta, u, v, height))

    expected = [0.02291542, 0.017126873, 0.017828466]
    if option == 'ONE_LIST_TENSOR':
      expected = np.expand_dims(expected, 0)
    self.assertAllClose(expected, c_h)

  @parameterized.named_parameters(('VerticalDimX', 0), ('VerticalDimY', 1),
                                  ('VerticalDimZ', 2))
  def testSurfaceEnergyFluxUpdateFnProvidesCorrectValues(self, vertical_dim):
    """Checks if the surface energy flux is correct."""
    self._setUp(vertical_dim=vertical_dim, cx=1, cy=1, cz=1)

    theta_val = 300.0 * np.ones((8, 8, 8), dtype=np.float32)
    u_val = np.zeros((8, 8, 8), dtype=np.float32)
    v_val = np.zeros((8, 8, 8), dtype=np.float32)
    w_val = np.zeros((8, 8, 8), dtype=np.float32)
    rho_val = np.ones((8, 8, 8), dtype=np.float32)
    h_t_val = 1.1e5 * np.ones((8, 8, 8), dtype=np.float32)

    if vertical_dim == 0:
      v_val[:, 2, :] = 2.0
      w_val[:, 2, :] = -3.0
      theta_val[:, 2, :] = 260.0
      rho_val[:, 2, :] = 1.2
      h_t_val[:, 1, :] = 1.2e5
    elif vertical_dim == 1:
      u_val[..., 2] = 2.0
      w_val[..., 2] = -3.0
      theta_val[..., 2] = 260.0
      rho_val[..., 2] = 1.2
      h_t_val[..., 1] = 1.2e5
    else:  # vertical_dim == 2
      u_val[2, ...] = 2.0
      v_val[2, ...] = -3.0
      theta_val[2, ...] = 260.0
      rho_val[2, ...] = 1.2
      h_t_val[1, ...] = 1.2e5

    states = {
        'theta': tf.unstack(tf.convert_to_tensor(theta_val, dtype=tf.float32)),
        'u': tf.unstack(tf.convert_to_tensor(u_val, dtype=tf.float32)),
        'v': tf.unstack(tf.convert_to_tensor(v_val, dtype=tf.float32)),
        'w': tf.unstack(tf.convert_to_tensor(w_val, dtype=tf.float32)),
        'rho': tf.unstack(tf.convert_to_tensor(rho_val, dtype=tf.float32)),
        'phi': tf.unstack(tf.convert_to_tensor(h_t_val, dtype=tf.float32)),
    }

    q_e = self.evaluate(self.model.surface_scalar_flux_update_fn(states))

    with self.subTest(name='SliceShape'):
      if vertical_dim == 0:
        self.assertLen(q_e, 8)
        self.assertEqual(q_e[0].shape[0], 1)
        self.assertEqual(q_e[0].shape[1], 8)
      elif vertical_dim == 1:
        self.assertLen(q_e, 8)
        self.assertEqual(q_e[0].shape[0], 8)
        self.assertEqual(q_e[0].shape[1], 1)
      else:  # vertical_dim == 2
        self.assertEqual(q_e.shape[0], 8)
        self.assertEqual(q_e.shape[1], 8)

    with self.subTest(name='Value'):
      expected = 1510.8957337735 * np.ones((8, 8), dtype=np.float32)

      self.assertAllClose(expected, np.squeeze(q_e))

  def testObukhovLengthIsComputedCorrectly(self):
    """Checks if the Obukhov length is computed correctly."""
    self._setUp()

    u = tf.constant(7.0, dtype=tf.float32)
    v = tf.constant(-5.5, dtype=tf.float32)
    m = tf.math.sqrt(u**2 + v**2)
    temperature = tf.constant(273.0, dtype=tf.float32)
    z_m = tf.constant(1.5, dtype=tf.float32)

    l = self.evaluate(self.model._compute_obukhov_length(m, temperature, z_m))

    expected = -101.02896973586881
    self.assertAllClose(expected, l, atol=1e-4)

  @parameterized.named_parameters(
      ('Case00', True, ('bc_u_2_0', 'bc_v_2_0', 'bc_T_2_0')),
      ('Case01', False, ('bc_u_2_0', 'bc_v_2_0')),
  )
  def testInitFnProvidesCorrectInitialStates(self, update_bc_t,
                                             expected_state_keys):
    """Checks if the `init_fn` generates correct initial states."""
    self._setUp()

    coordinates = (0, 0, 0)

    FLAGS.lx = 1.0
    FLAGS.ly = 2.1
    FLAGS.lz = 3.6
    FLAGS.nx = 10
    FLAGS.ny = 12
    FLAGS.nz = 14
    FLAGS.halo_width = 2
    params = grid_parametrization.GridParametrization()

    states = self.evaluate(self.model.init_fn(params, coordinates, update_bc_t))

    self.assertCountEqual(expected_state_keys, states.keys())

    with self.subTest(name='BcU'):
      expected = np.zeros((14, 10, 12), dtype=np.float32)
      self.assertAllClose(expected, states['bc_u_2_0'])

    with self.subTest(name='BcV'):
      expected = np.zeros((14, 10, 12), dtype=np.float32)
      self.assertAllClose(expected, states['bc_v_2_0'])

    if update_bc_t:
      with self.subTest(name='BcT'):
        expected = np.zeros((14, 10, 12), dtype=np.float32)
        self.assertAllClose(expected, states['bc_T_2_0'])

  _VERTICAL_DIMS = (0, 1, 2)

  @parameterized.parameters(*itertools.product(_VERTICAL_DIMS, (
      (_MOST_CONFIG, 1.048637, 1.048637, 0.076839),
      (_MOST_CONFIG_ZERO_HEAT_FLUX, 1.0407896, 1.0407896, 0),
      (_MOST_CONFIG_NEGATIVE_HEAT_FLUX, 1.0311368, 1.0311368, -0.0755569),
  )))
  def testMoengModelUpdateFnProvidesCorrectBCWithTInStates(
      self, vertical_dim, test_params):
    """Checks if the Moeng's model is computed correctly."""
    (config, expected_bc_0, expected_bc_1, expected_bc_upper_t) = test_params
    self._setUp(config, vertical_dim)

    dim0, dim1 = self.horizontal_dims
    states = {
        _KEYS_VELOCITY[dim0]:
            self._create_field_with_halos(2.0, -7.0, [8, 8, 8], vertical_dim),
        _KEYS_VELOCITY[dim1]:
            self._create_field_with_halos(1.0, -7.0, [8, 8, 8], vertical_dim),
        'T':
            self._create_field_with_halos(300.0, 300.0, [8, 8, 8],
                                          vertical_dim),
    }
    bc_velocity_key_0 = self.bc_manager.generate_bc_key(_KEYS_VELOCITY[dim0],
                                                        vertical_dim, 0)
    bc_velocity_key_1 = self.bc_manager.generate_bc_key(_KEYS_VELOCITY[dim1],
                                                        vertical_dim, 0)
    bc_t_key = self.bc_manager.generate_bc_key('T', vertical_dim, 0)
    additional_states = {
        bc_velocity_key_0: [tf.zeros((8, 8), dtype=tf.float32),] * 8,
        bc_velocity_key_1: [tf.zeros((8, 8), dtype=tf.float32),] * 8,
        bc_t_key: [tf.zeros((8, 8), dtype=tf.float32),] * 8,
    }
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    kernel_op = get_kernel_fn.ApplyKernelConvOp(4)

    new_additional_states = self.evaluate(
        self.model.moeng_model_update_fn(kernel_op, replica_id, replicas,
                                         states, additional_states,
                                         self.params))

    self.assertCountEqual([bc_velocity_key_0, bc_velocity_key_1, bc_t_key],
                          new_additional_states.keys())

    slices = [slice(self.params.halo_width, -self.params.halo_width)] * 3
    slices[self.vertical_axis] = slice(0, None)
    expected_shape = [4, 4, 4]
    expected_shape[self.vertical_axis] = 8

    with self.subTest(name='BCUCorrect'):
      self.assertAllClose(
          expected_bc_0 * np.ones(expected_shape, dtype=np.float32),
          np.array(new_additional_states[bc_velocity_key_0])[slices])

    with self.subTest(name='BCVCorrect'):
      self.assertAllClose(
          expected_bc_1 * np.ones(expected_shape, dtype=np.float32),
          np.array(new_additional_states[bc_velocity_key_1])[slices])

    with self.subTest(name='BCTCorrect'):
      self.assertAllClose(
          expected_bc_upper_t * np.ones(expected_shape, dtype=np.float32),
          np.array(new_additional_states[bc_t_key])[slices])

  @parameterized.parameters(*itertools.product(_VERTICAL_DIMS, (
      (_MOST_CONFIG, 1.048637),
      (_MOST_CONFIG_ZERO_HEAT_FLUX, 1.0407896),
      (_MOST_CONFIG_NEGATIVE_HEAT_FLUX, 1.0311368),
  )))
  def testMoengModelUpdateFnProvidesCorrectBCWithTInAdditionalStates(
      self, vertical_dim, test_params):
    """Checks if the Moeng's model is computed correctly."""
    config, expected_bc_0 = test_params
    self._setUp(config, vertical_dim)

    dim0, dim1 = self.horizontal_dims
    states = {
        _KEYS_VELOCITY[dim0]:
            self._create_field_with_halos(2.0, -7.0, [8, 8, 8], vertical_dim),
        _KEYS_VELOCITY[dim1]:
            self._create_field_with_halos(1.0, -7.0, [8, 8, 8], vertical_dim),
    }
    bc_velocity_key_0 = self.bc_manager.generate_bc_key(_KEYS_VELOCITY[dim0],
                                                        vertical_dim, 0)
    bc_velocity_key_1 = self.bc_manager.generate_bc_key(_KEYS_VELOCITY[dim1],
                                                        vertical_dim, 0)
    additional_states = {
        'T': [300.0 * tf.ones((8, 8), dtype=tf.float32),] * 8,
        bc_velocity_key_0: [tf.zeros((8, 8), dtype=tf.float32),] * 8,
        bc_velocity_key_1: [tf.zeros((8, 8), dtype=tf.float32),] * 8,
    }
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    kernel_op = get_kernel_fn.ApplyKernelConvOp(4)

    new_additional_states = self.evaluate(
        self.model.moeng_model_update_fn(kernel_op, replica_id, replicas,
                                         states, additional_states,
                                         self.params))

    self.assertCountEqual([bc_velocity_key_0, bc_velocity_key_1, 'T'],
                          new_additional_states.keys())

    slices = [slice(self.params.halo_width, -self.params.halo_width)] * 3
    slices[self.vertical_axis] = slice(0, None)
    expected_shape = [4, 4, 4]
    expected_shape[self.vertical_axis] = 8

    with self.subTest(name='BCUCorrect'):
      self.assertAllClose(
          expected_bc_0 * np.ones(expected_shape, dtype=np.float32),
          np.array(new_additional_states[bc_velocity_key_0])[slices])

    # TODO(b/218890889): Check the boundary conditions of u, v, and T.
    with self.subTest(name='BCVCorrect'):
      self.assertAllClose(
          expected_bc_0 * np.ones(expected_shape, dtype=np.float32),
          np.array(new_additional_states[bc_velocity_key_1])[slices])

    with self.subTest(name='BCTCorrect'):
      self.assertAllClose(
          300 * np.ones(expected_shape, dtype=np.float32),
          np.array(new_additional_states['T'])[slices])

  @parameterized.parameters(*_VERTICAL_DIMS)
  def testMoengModelUpdateFnRaisesValueErrorWhenRequiredAdditionalStatesMissing(
      self, vertical_dim):
    """Checks if ValueError is raised because of missing state keys."""
    self._setUp(vertical_dim=vertical_dim)

    dim0, dim1 = self.horizontal_dims
    states = {
        _KEYS_VELOCITY[dim0]: [
            2.0 * tf.ones((8, 8), dtype=tf.float32),
        ] * 8,
        _KEYS_VELOCITY[dim1]: [
            1.0 * tf.ones((8, 8), dtype=tf.float32),
        ] * 8,
    }
    bc_velocity_key_0 = self.bc_manager.generate_bc_key(_KEYS_VELOCITY[dim0],
                                                        vertical_dim, 0)
    bc_velocity_key_1 = self.bc_manager.generate_bc_key(_KEYS_VELOCITY[dim1],
                                                        vertical_dim, 0)
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    kernel_op = get_kernel_fn.ApplyKernelConvOp(4)

    with self.subTest(name='MissingFirstVelocityBC'):
      additional_states = {
          'T': [300.0 * tf.ones((8, 8), dtype=tf.float32),] * 8,
          bc_velocity_key_1: [tf.zeros((8, 8), dtype=tf.float32),] * 8,
      }
      with self.assertRaises(ValueError):
        self.evaluate(
            self.model.moeng_model_update_fn(kernel_op, replica_id, replicas,
                                             states, additional_states,
                                             self.params))

    with self.subTest(name='MissingSecondVelocityBC'):
      additional_states = {
          'T': [300.0 * tf.ones((8, 8), dtype=tf.float32),] * 8,
          bc_velocity_key_0: [tf.zeros((8, 8), dtype=tf.float32),] * 8,
      }
      with self.assertRaises(ValueError):
        self.evaluate(
            self.model.moeng_model_update_fn(kernel_op, replica_id, replicas,
                                             states, additional_states,
                                             self.params))

    with self.subTest(name='MissingTBC'):
      additional_states = {
          bc_velocity_key_0: [tf.zeros((8, 8), dtype=tf.float32),] * 8,
          bc_velocity_key_1: [tf.zeros((8, 8), dtype=tf.float32),] * 8,
      }
      with self.assertRaises(ValueError):
        self.evaluate(
            self.model.moeng_model_update_fn(kernel_op, replica_id, replicas,
                                             states, additional_states,
                                             self.params))

  @parameterized.parameters(*_VERTICAL_DIMS)
  def testPorteAgelModelUpdateFnGeneratesCorrectBCWithTInStates(
      self, vertical_dim):
    """Checks if the Porte Agel's model is computed correctly."""
    self._setUp(vertical_dim=vertical_dim)
    dim0, dim1 = self.horizontal_dims
    states = {
        _KEYS_VELOCITY[dim0]:
            self._create_field_with_halos(8.0, -7.0, [8, 8, 8], vertical_dim),
        _KEYS_VELOCITY[dim1]:
            self._create_field_with_halos(4.0, -7.0, [8, 8, 8], vertical_dim),
        'T':
            self._create_field_with_halos(266.0, -7.0, [8, 8, 8], vertical_dim),
    }
    nu_t = np.zeros((8, 8, 8), dtype=np.float32)
    slices = [slice(0, None)] * 3
    slices[self.vertical_axis] = self.params.halo_width
    nu_t[slices] = 2.0
    bc_tau_key = 'bc_tauT{vertical_dim}_{vertical_dim}_0'.format(
        vertical_dim=vertical_dim)
    bc_velocity_key_0 = self.bc_manager.generate_bc_key(_KEYS_VELOCITY[dim0],
                                                        vertical_dim, 0)
    bc_velocity_key_1 = self.bc_manager.generate_bc_key(_KEYS_VELOCITY[dim1],
                                                        vertical_dim, 0)
    bc_t_key = self.bc_manager.generate_bc_key('T', vertical_dim, 0)
    additional_states = {
        'nu_t': tf.unstack(tf.convert_to_tensor(nu_t)),
        bc_t_key: [265.0 * tf.ones((8, 8), dtype=tf.float32),] * 8,
        bc_velocity_key_0: [tf.zeros((8, 8), dtype=tf.float32),] * 8,
        bc_velocity_key_1: [tf.zeros((8, 8), dtype=tf.float32),] * 8,
        bc_tau_key: [tf.zeros((8, 8), dtype=tf.float32),] * 8,
    }
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    kernel_op = get_kernel_fn.ApplyKernelConvOp(4)

    new_additional_states = self.evaluate(
        self.model.porte_agel_model_update_fn(kernel_op, replica_id, replicas,
                                              states, additional_states,
                                              self.params))

    self.assertCountEqual(
        [bc_velocity_key_0, bc_velocity_key_1, bc_t_key, bc_tau_key, 'nu_t'],
        new_additional_states.keys())

    with self.subTest(name='BCUCorrect'):
      self.assertAllClose(
          1.0672182 * np.ones((8, 8, 8), dtype=np.float32),
          np.array(new_additional_states[bc_velocity_key_0]))

    with self.subTest(name='BCVCorrect'):
      self.assertAllClose(
          0.5336091 * np.ones((8, 8, 8), dtype=np.float32),
          np.array(new_additional_states[bc_velocity_key_1]))

    with self.subTest(name='BCTCorrect'):
      self.assertAllClose(
          0.11439023 * np.ones((8, 8, 8), dtype=np.float32),
          np.array(new_additional_states[bc_t_key]))

    with self.subTest(name='BCTauT2Correct'):
      self.assertAllClose(
          0.2669118872 * np.ones((8, 8, 8), dtype=np.float32),
          np.array(new_additional_states[bc_tau_key]))

  @parameterized.parameters(*_VERTICAL_DIMS)
  def testPorteAgelModelUpdateFnGeneratesCorrectBCWithTInAdditionalStates(
      self, vertical_dim):
    """Checks if the Porte Agel's model is computed correctly."""
    self._setUp(vertical_dim=vertical_dim)
    dim0, dim1 = self.horizontal_dims
    states = {
        _KEYS_VELOCITY[dim0]:
            self._create_field_with_halos(8.0, -7.0, [8, 8, 8], vertical_dim),
        _KEYS_VELOCITY[dim1]:
            self._create_field_with_halos(4.0, -7.0, [8, 8, 8], vertical_dim),
    }
    nu_t = np.zeros((8, 8, 8), dtype=np.float32)
    slices = [slice(0, None)] * 3
    slices[self.vertical_axis] = self.params.halo_width
    nu_t[slices] = 2.0
    bc_tau_key_0 = (
        'bc_tau{horizontal_dim}{vertical_dim}_{vertical_dim}_0').format(
            horizontal_dim=dim0, vertical_dim=vertical_dim)
    bc_tau_key_1 = (
        'bc_tau{horizontal_dim}{vertical_dim}_{vertical_dim}_0').format(
            horizontal_dim=dim1, vertical_dim=vertical_dim)
    bc_velocity_key_0 = self.bc_manager.generate_bc_key(_KEYS_VELOCITY[dim0],
                                                        vertical_dim, 0)
    bc_velocity_key_1 = self.bc_manager.generate_bc_key(_KEYS_VELOCITY[dim1],
                                                        vertical_dim, 0)
    additional_states = {
        'nu_t': tf.unstack(tf.convert_to_tensor(nu_t)),
        'T': [266.0 * tf.ones((8, 8), dtype=tf.float32),] * 8,
        bc_velocity_key_0: [tf.zeros((8, 8), dtype=tf.float32),] * 8,
        bc_velocity_key_1: [tf.zeros((8, 8), dtype=tf.float32),] * 8,
        bc_tau_key_0: [tf.zeros((8, 8), dtype=tf.float32),] * 8,
        bc_tau_key_1: [tf.zeros((8, 8), dtype=tf.float32),] * 8,
    }
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    kernel_op = get_kernel_fn.ApplyKernelConvOp(4)

    new_additional_states = self.evaluate(
        self.model.porte_agel_model_update_fn(kernel_op, replica_id, replicas,
                                              states, additional_states,
                                              self.params))

    self.assertCountEqual([
        bc_velocity_key_0, bc_velocity_key_1, bc_tau_key_0, bc_tau_key_1, 'T',
        'nu_t'
    ], new_additional_states.keys())

    with self.subTest(name='BCUCorrect'):
      self.assertAllClose(
          1.0672182 * np.ones((8, 8, 8), dtype=np.float32),
          np.array(new_additional_states[bc_velocity_key_0]))

    with self.subTest(name='BCVCorrect'):
      self.assertAllClose(
          0.5336091 * np.ones((8, 8, 8), dtype=np.float32),
          np.array(new_additional_states[bc_velocity_key_1]))

    with self.subTest(name='BCTau02Correct'):
      self.assertAllClose(
          2.4901884 * np.ones((8, 8, 8), dtype=np.float32),
          np.array(new_additional_states[bc_tau_key_0]))

    with self.subTest(name='BCTau12Correct'):
      self.assertAllClose(
          1.2450942 * np.ones((8, 8, 8), dtype=np.float32),
          np.array(new_additional_states[bc_tau_key_1]))


if __name__ == '__main__':
  tf.test.main()
