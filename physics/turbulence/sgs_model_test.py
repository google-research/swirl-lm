"""Tests for google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.sgs_model."""
import functools
import itertools

import numpy as np
from swirl_lm.base import parameters_pb2
from swirl_lm.physics.turbulence import sgs_model
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import tf_test_util as test_util
from swirl_lm.utility.tf_tpu_test_util import run_on_tpu_in_test
import tensorflow as tf

from google3.testing.pybase import parameterized


class SgsModelTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Initializes common flow field variables."""
    super(SgsModelTest, self).setUp()

    self.delta = (0.1, 0.2, 0.5)

    # Construct the velocity field so that the divergence is 3.5, and
    # the strain rate tensor is:
    # [8.833, -2.5, -1.0]
    # [-2.5, -8.667, 3.0]
    # [-1.0, 3.0, -0.167]
    # The strain rate magnitude is: 17.960604.

    u = np.ones((16, 16, 16), dtype=np.float32)
    u[8, 8, 8] = 2.0
    # dudz[8, 8, 8] = 1.0
    u[7, 8, 8] = 1.0
    u[9, 8, 8] = 2.0
    # dudx[8, 8, 8] = 10.0
    u[8, 7, 8] = 1.0
    u[8, 9, 8] = 3.0
    # dudy[8, 8, 8] = 7.5
    u[8, 8, 7] = 1.0
    u[8, 8, 9] = 4.0

    v = np.ones((16, 16, 16), dtype=np.float32)
    v[8, 8, 8] = 2.0
    # dvdz[8, 8, 8] = -1.0
    v[7, 8, 8] = 2.0
    v[9, 8, 8] = 1.0
    # dvdx[8, 8, 8] = -10.0
    v[8, 7, 8] = 3.0
    v[8, 9, 8] = 1.0
    # dvdy[8, 8, 8] = -7.5
    v[8, 8, 7] = 4.0
    v[8, 8, 9] = 1.0

    w = np.ones((16, 16, 16), dtype=np.float32)
    w[8, 8, 8] = 2.0
    # dwdz[8, 8, 8] = 1.0
    w[7, 8, 8] = 1.0
    w[9, 8, 8] = 2.0
    # dwdx[8, 8, 8] = -2.0
    w[8, 7, 8] = 1.4
    w[8, 9, 8] = 1.0
    # dwdy[8, 8, 8] = 4.0
    w[8, 8, 7] = 1.0
    w[8, 8, 9] = 2.6

    sc = 0.25 * np.ones((16, 16, 16), dtype=np.float32)
    sc[8, 8, 8] = 0.5
    # dsc_dz[8, 8, 8] = 0.25
    sc[7, 8, 8] = 0.25
    sc[9, 8, 8] = 0.5
    # dsc_dx[8, 8, 8] = 2.5
    sc[8, 7, 8] = 0.25
    sc[8, 9, 8] = 0.75
    # dsc_dy[8, 8, 8] = 1.875
    sc[8, 8, 7] = 0.25
    sc[8, 8, 9] = 1.0

    self.velocity = [
        tf.unstack(tf.convert_to_tensor(u, dtype=tf.float32)),
        tf.unstack(tf.convert_to_tensor(v, dtype=tf.float32)),
        tf.unstack(tf.convert_to_tensor(w, dtype=tf.float32)),
    ]
    self.scalar = tf.unstack(tf.convert_to_tensor(sc, dtype=tf.float32))

  @test_util.run_in_graph_and_eager_modes
  def testTestFilterProducesCorrectLocalSmering(self):
    """Test if test filtering a delta function redistributes value in a cube."""
    value = np.zeros((8, 8, 8), dtype=np.float32)
    value[4, 4, 4] = 27.0

    filtered = self.evaluate(
        sgs_model._test_filter(tf.unstack(tf.convert_to_tensor(value))))

    expected = np.zeros((8, 8, 8), dtype=np.float32)
    expected[3:6, 3:6, 3:6] = 1.0

    self.assertAllEqual(filtered, expected)

  @test_util.run_in_graph_and_eager_modes
  def testDotProducesCorrectTensorDotProduct(self):
    """Test if dot product of two tensors are computed correctly."""
    u = [tf.constant(2.0, dtype=tf.float32),]
    v = [tf.constant(3.0, dtype=tf.float32),]

    res = self.evaluate(sgs_model._dot(u, v))

    self.assertEqual(res[0], 6.0)

  @test_util.run_in_graph_and_eager_modes
  def testEinsumijProducesCorrectEinsteinSumOfTwoTensors(self):
    """Checks if einsum_ij computes the correct Einstein sum of two tensors."""
    tensor_1 = [
        [
            tf.unstack(tf.ones((6, 6, 6), dtype=tf.float32)),
            tf.unstack(2.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
            tf.unstack(3.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
        ],
        [
            tf.unstack(4.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
            tf.unstack(5.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
            tf.unstack(6.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
        ],
        [
            tf.unstack(7.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
            tf.unstack(8.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
            tf.unstack(9.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
        ],
    ]

    tensor_2 = [
        [
            tf.unstack(2.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
            tf.unstack(3.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
            tf.unstack(4.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
        ],
        [
            tf.unstack(5.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
            tf.unstack(6.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
            tf.unstack(7.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
        ],
        [
            tf.unstack(8.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
            tf.unstack(9.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
            tf.unstack(10.0 * tf.ones((6, 6, 6), dtype=tf.float32)),
        ],
    ]

    res = self.evaluate(sgs_model._einsum_ij(tensor_1, tensor_2))

    expected = 330.0 * np.ones((6, 6, 6), dtype=np.float32)

    self.assertAllEqual(res, expected)

  @test_util.run_in_graph_and_eager_modes
  def testStrainRateMagnitudeProducesCorrectResult(self):
    """Test the strain rate magnitude at [6, 6, 6] is correct."""
    strain_rate_np = [[np.zeros((8, 8, 8), dtype=np.float32)] * 3] * 3
    strain_rate_np[0][0][6, 6, 6] = 2.0
    strain_rate = [[tf.unstack(tf.convert_to_tensor(s_ij))
                    for s_ij in s_i]
                   for s_i in strain_rate_np]

    strain_rate_magnitude = self.evaluate(
        sgs_model._strain_rate_magnitude(strain_rate))

    self.assertAlmostEqual(strain_rate_magnitude[6][6, 6], 8.485281)

  @test_util.run_in_graph_and_eager_modes
  def testStrainRateTensorComputedCorrectly(self):
    """Test the strain rate tensor is computed correctly from gradients."""
    du_dx = [
        [
            [tf.constant(1.0, dtype=tf.float32)],
            [tf.constant(2.0, dtype=tf.float32)],
            [tf.constant(3.0, dtype=tf.float32)],
        ],
        [
            [tf.constant(4.0, dtype=tf.float32)],
            [tf.constant(5.0, dtype=tf.float32)],
            [tf.constant(6.0, dtype=tf.float32)],
        ],
        [
            [tf.constant(7.0, dtype=tf.float32)],
            [tf.constant(8.0, dtype=tf.float32)],
            [tf.constant(9.0, dtype=tf.float32)],
        ],
    ]

    strain_rate = self.evaluate(
        sgs_model._strain_rate_tensor(du_dx))

    self.assertLen(strain_rate, 3)
    self.assertLen(strain_rate[0], 3)

    with self.subTest(name='S11'):
      self.assertEqual(strain_rate[0][0][0], -4.0)

    with self.subTest(name='S12'):
      self.assertEqual(strain_rate[0][1][0], 3.0)

    with self.subTest(name='S13'):
      self.assertEqual(strain_rate[0][2][0], 5.0)

    with self.subTest(name='S21'):
      self.assertEqual(strain_rate[1][0][0], 3.0)

    with self.subTest(name='S22'):
      self.assertEqual(strain_rate[1][1][0], 0.0)

    with self.subTest(name='S23'):
      self.assertEqual(strain_rate[1][2][0], 7.0)

    with self.subTest(name='S31'):
      self.assertEqual(strain_rate[2][0][0], 5.0)

    with self.subTest(name='S32'):
      self.assertEqual(strain_rate[2][1][0], 7.0)

    with self.subTest(name='S33'):
      self.assertEqual(strain_rate[2][2][0], 4.0)

  _C_S = [None, 0.66]

  @parameterized.parameters(*zip(_C_S))
  @test_util.run_in_graph_and_eager_modes
  def testSmagorinskySGSModelProducesCorrectTurbulentViscosity(
      self, c_s):
    """Test if the turbulent viscosity is computed correctly at [8, 8, 8]."""
    if c_s is None:
      additional_states = None
    else:
      c_s_tensor = np.zeros((16, 16, 16), dtype=np.float32)
      c_s_tensor[8, 8, 8] = c_s
      additional_states = {
          'c_s': tf.unstack(tf.convert_to_tensor(c_s_tensor, dtype=tf.float32))
      }

    kernel_op = get_kernel_fn.ApplyKernelConvOp(8)
    params = parameters_pb2.SubGridScaleModel()
    params.smagorinsky.c_s = 0.18
    params.smagorinsky.pr_t = 0.3
    sgs = sgs_model.SgsModel(kernel_op, self.delta, params)

    nu_t = sgs.turbulent_viscosity(
        self.velocity, additional_states=additional_states)

    nu_t_val = self.evaluate(nu_t)

    self.assertLen(nu_t_val, 16)
    # Compare to result computed externally.
    if additional_states is None:
      self.assertAlmostEqual(nu_t_val[8][8, 8], 0.17457707)
    else:
      self.assertAlmostEqual(nu_t_val[8][8, 8], 2.3470917189, 5)

  @test_util.run_in_graph_and_eager_modes
  def testSmagorinskySGSModelProducesCorrectTurbulentDiffusivity(self):
    """Test if the turbulent viscosity is computed correctly at [8, 8, 8]."""
    scalar = [self.velocity[2]]

    kernel_op = get_kernel_fn.ApplyKernelConvOp(8)
    sgs = sgs_model.SgsModel(kernel_op, self.delta)

    diff_t = sgs.smagorinsky(scalar, self.delta)

    diff_t_val = self.evaluate(diff_t)

    self.assertLen(diff_t_val, 16)
    # Compare to result computed externally.
    self.assertAlmostEqual(diff_t_val[8][8, 8], 0.06299279959)

  @test_util.run_in_graph_and_eager_modes
  def testDynamicSmagorinskySGSModelProducesCorrectTurbulentViscosity(self):
    """Checks if turbulent viscosity from dynamic Smagorinsky is correct."""
    kernel_op = get_kernel_fn.ApplyKernelConvOp(8)
    sgs = sgs_model.SgsModel(kernel_op, self.delta)

    nu_t = sgs.dynamic_smagorinsky(self.delta, [False, False, True],
                                   np.array([[[0]]]), self.velocity)

    nu_t_val = self.evaluate(nu_t)

    self.assertLen(nu_t_val, 16)
    # Compare to result computed externally (http://go/dynamic_sgs_validation).
    self.assertAlmostEqual(nu_t_val[8][8, 8], 0.005617779, 6)

  @test_util.run_in_graph_and_eager_modes
  def testDynamicSmagorinskySGSModelProducesCorrectTurbulentDiffusivity(self):
    """Checks if turbulent diffusivity from dynamic Smagorinsky is correct."""
    kernel_op = get_kernel_fn.ApplyKernelConvOp(8)
    sgs = sgs_model.SgsModel(kernel_op, self.delta)

    nu_t = sgs.dynamic_smagorinsky(self.delta, [True, True, False],
                                   np.array([[[0]]]), self.velocity,
                                   self.scalar)

    nu_t_val = self.evaluate(nu_t)

    self.assertLen(nu_t_val, 16)
    # Compare to result computed externally (http://go/dynamic_sgs_validation).
    self.assertAlmostEqual(nu_t_val[8][8, 8], 0.053236455, 6)

  _REPLICAS = [
      np.array([[[0], [1]]]),
      np.array([[[0, 1]]]),
      np.array([[[0]], [[1]]])
  ]
  _PERIODIC_X = [True, False]
  _PERIODIC_Y = [True, False]
  _PERIODIC_Z = [True, False]

  @parameterized.parameters(
      *itertools.product(_REPLICAS, _PERIODIC_X, _PERIODIC_Y, _PERIODIC_Z))
  def testGermanoAveraging(self, replicas, periodic_x, periodic_y, periodic_z):
    """Checks Germano averaging in all periodicity combinations."""
    computation_shape = np.array(replicas.shape)
    inputs = [
        tf.unstack(tf.ones((8, 8, 8), dtype=tf.float32)),
        tf.unstack(2.0 * tf.ones((8, 8, 8), dtype=tf.float32))
    ]
    periodic_dims = (periodic_x, periodic_y, periodic_z)

    device_fn = functools.partial(
        sgs_model._germano_averaging,
        periodic_dims=periodic_dims,
        replicas=replicas)

    output = run_on_tpu_in_test(self, replicas, device_fn, inputs)

    if periodic_dims[np.where(computation_shape == 2)[0][0]]:
      self.assertAllEqual(output[0], 1.5 * np.ones(
          (8, 8, 8), dtype=np.float32))
      self.assertAllEqual(output[1], 1.5 * np.ones(
          (8, 8, 8), dtype=np.float32))
    else:
      self.assertAllEqual(output[0], 1.0 * np.ones(
          (8, 8, 8), dtype=np.float32))
      self.assertAllEqual(output[1], 2.0 * np.ones(
          (8, 8, 8), dtype=np.float32))

  @test_util.run_in_graph_and_eager_modes
  def testSmagorinskyLillyModelProducesCorrectTurbulentViscosityAndDiffusivity(
      self):
    """Tests if the turbulent viscosity is computed correctly at [8, 8, 8]."""
    t = 300.0 * np.ones((16, 16, 16), dtype=np.float32)
    # dTdz[8, 8, 8] = 10.0
    t[7, ...] = 250.0
    t[9, ...] = 350.0
    additional_states = {
        'theta_v': tf.unstack(tf.convert_to_tensor(t, dtype=tf.float32))
    }

    kernel_op = get_kernel_fn.ApplyKernelConvOp(8)
    params = parameters_pb2.SubGridScaleModel()
    params.smagorinsky_lilly.c_s = 0.18
    params.smagorinsky_lilly.pr_t = 0.1
    sgs = sgs_model.SgsModel(kernel_op, self.delta, params)

    with self.subTest(name='Viscosity'):
      nu_t = sgs.turbulent_viscosity(
          self.velocity, additional_states=additional_states)

      nu_t_val = self.evaluate(nu_t)

      self.assertLen(nu_t_val, 16)
      # Compare to result computed externally.
      self.assertAlmostEqual(nu_t_val[8][8, 8], 0.0256049111)

    with self.subTest(name='Diffusivity'):
      diff_t = sgs.turbulent_diffusivity({},
                                         velocity=self.velocity,
                                         additional_states=additional_states)

      diff_t_val = self.evaluate(diff_t)

      self.assertLen(diff_t_val, 16)
      # Compare to result computed externally.
      self.assertAlmostEqual(diff_t_val[8][8, 8], 0.256049111)

  @test_util.run_in_graph_and_eager_modes
  def testVremanModelProducesCorrectTurbulentViscosityAndDiffusivity(
      self):
    """Checks if the turbulent viscosity is computed correctly at [8, 8, 8]."""
    kernel_op = get_kernel_fn.ApplyKernelConvOp(8)
    params = parameters_pb2.SubGridScaleModel()
    params.vreman.c_s = 0.18
    params.vreman.pr_t = 0.1
    sgs = sgs_model.SgsModel(kernel_op, self.delta, params)

    with self.subTest(name='Viscosity'):
      nu_t = sgs.turbulent_viscosity(self.velocity, additional_states={})

      nu_t_val = self.evaluate(nu_t)

      self.assertLen(nu_t_val, 16)
      # Compare to result computed externally.
      self.assertAlmostEqual(nu_t_val[8][8, 8], 0.008136134008)

    with self.subTest(name='Diffusivity'):
      diff_t = sgs.turbulent_diffusivity({},
                                         velocity=self.velocity,
                                         additional_states={})

      diff_t_val = self.evaluate(diff_t)

      self.assertLen(diff_t_val, 16)
      # Compare to result computed externally.
      self.assertAlmostEqual(diff_t_val[8][8, 8], 0.08136134008)


if __name__ == '__main__':
  tf.test.main()
