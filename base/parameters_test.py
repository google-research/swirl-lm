"""Tests for swirl_lm.base.parameters."""

import os

from absl import flags
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.communication import halo_exchange
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.pyglib import resources
from google3.testing.pybase import parameterized

FLAGS = flags.FLAGS


_TESTDATA_DIR = ('google3/third_party/py/swirl_lm/base/testdata')


@test_util.run_all_in_graph_and_eager_modes
class ParametersTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ParametersTest, self).setUp()

    self.config_filepath = resources.GetResourceFilename(
        os.path.join(_TESTDATA_DIR, 'swirl_lm_parameters.textpb'))
    self.config_filepath_scalars = resources.GetResourceFilename(
        os.path.join(_TESTDATA_DIR, 'swirl_lm_parameters_scalars.textpb'))

  def testCorrectlyParsesFile(self):
    params = parameters_lib.SwirlLMParameters.config_from_proto(
        self.config_filepath)

    periodic_dims = [False, False, False]

    expected_u_bc = [[(halo_exchange.BCType.DIRICHLET, 1.0),
                      (halo_exchange.BCType.DIRICHLET, -1.0)],
                     [(halo_exchange.BCType.DIRICHLET, 1.0),
                      (halo_exchange.BCType.DIRICHLET, -1.0)],
                     [(halo_exchange.BCType.NEUMANN, 0.0),
                      (halo_exchange.BCType.NEUMANN, 0.0)]]

    expected_v_bc = [[(halo_exchange.BCType.DIRICHLET, 2.0),
                      (halo_exchange.BCType.DIRICHLET, -2.0)],
                     [(halo_exchange.BCType.NEUMANN, 0.0),
                      (halo_exchange.BCType.NEUMANN, 0.0)],
                     [(halo_exchange.BCType.DIRICHLET, 2.0),
                      (halo_exchange.BCType.DIRICHLET, -2.0)]]

    expected_w_bc = [[(halo_exchange.BCType.NEUMANN, 0.0),
                      (halo_exchange.BCType.NEUMANN, 0.0)],
                     [(halo_exchange.BCType.DIRICHLET, 3.0),
                      (halo_exchange.BCType.DIRICHLET, -3.0)],
                     [(halo_exchange.BCType.DIRICHLET, 3.0),
                      (halo_exchange.BCType.DIRICHLET, -3.0)]]

    expected_p_bc = [[(halo_exchange.BCType.DIRICHLET, 10.0),
                      (halo_exchange.BCType.DIRICHLET, 10.0)],
                     [(halo_exchange.BCType.NEUMANN, 0.0),
                      (halo_exchange.BCType.NEUMANN, 0.0)],
                     [(halo_exchange.BCType.NEUMANN, 0.0),
                      (halo_exchange.BCType.NEUMANN, 0.0)]]

    expected_y1_bc = [[(halo_exchange.BCType.DIRICHLET, 1.0),
                       (halo_exchange.BCType.DIRICHLET, 1.0)],
                      [(halo_exchange.BCType.NEUMANN, 0.0),
                       (halo_exchange.BCType.NEUMANN, 0.0)],
                      [(halo_exchange.BCType.NEUMANN, 0.0),
                       (halo_exchange.BCType.NEUMANN, 0.0)]]

    expected_y2_bc = [[(halo_exchange.BCType.DIRICHLET, 0.5),
                       (halo_exchange.BCType.DIRICHLET, 0.5)],
                      [(halo_exchange.BCType.NEUMANN, 0.0),
                       (halo_exchange.BCType.NEUMANN, 0.0)],
                      [(halo_exchange.BCType.NEUMANN, 0.0),
                       (halo_exchange.BCType.NEUMANN, 0.0)]]

    self.assertEqual(params.convection_scheme,
                     parameters_lib.ConvectionScheme.CONVECTION_SCHEME_UPWIND_1)
    self.assertEqual(params.time_integration_scheme,
                     parameters_lib.TimeIntegrationScheme.TIME_SCHEME_RK3)
    self.assertEqual(params.solver_procedure,
                     parameters_lib.SolverProcedure.SEQUENTIAL)

    self.assertListEqual(params.periodic_dims, periodic_dims)

    self.assertEqual(params.bc['u'], expected_u_bc)
    self.assertEqual(params.bc['v'], expected_v_bc)
    self.assertEqual(params.bc['w'], expected_w_bc)
    self.assertEqual(params.bc['p'], expected_p_bc)
    self.assertEqual(params.bc['Y1'], expected_y1_bc)
    self.assertEqual(params.bc['Y2'], expected_y2_bc)

    self.assertListEqual(params.scalars_names, ['Y1', 'Y2'])
    self.assertListEqual(params.transport_scalars_names, ['Y1', 'Y2'])

    self.assertAlmostEqual(params.diffusivity('Y1'), 1e-6)
    self.assertAlmostEqual(params.diffusivity('Y2'), 8e-7)

  def testInvalidScalarNameRaiseValueErrorWhenCallDiffusivity(self):
    params = parameters_lib.SwirlLMParameters.config_from_proto(
        self.config_filepath)

    with self.assertRaisesRegex(ValueError, 'Y3 is not in the flow field'):
      params.diffusivity('Y3')

  @parameterized.named_parameters(
      ('Case00', parameters_lib.ConvectionScheme.CONVECTION_SCHEME_UPWIND_1, 1),
      ('Case01', parameters_lib.ConvectionScheme.CONVECTION_SCHEME_QUICK, 2),
      ('Case02', parameters_lib.ConvectionScheme.CONVECTION_SCHEME_CENTRAL_2,
       1),
      ('Case03', parameters_lib.ConvectionScheme.CONVECTION_SCHEME_CENTRAL_4,
       2),
  )
  def testMaxHaloWidthCorrectlyDeterminedFromScheme(self, scheme,
                                                    expected_max_halo_width):
    params = parameters_lib.SwirlLMParameters.config_from_proto(
        self.config_filepath)
    params.convection_scheme = scheme

    self.assertEqual(params.convection_scheme, scheme)
    self.assertEqual(params.max_halo_width, expected_max_halo_width)

  def testMaxHaloWidthCorrectlyDeterminedFromSchemeFailure(self):
    params = parameters_lib.SwirlLMParameters.config_from_proto(
        self.config_filepath)
    params.convection_scheme = (
        parameters_lib.ConvectionScheme.CONVECTION_SCHEME_UNKNOWN)

    with self.assertRaisesRegex(
        ValueError,
        'Halo width is ambiguous because convection scheme is not recognized.'):
      _ = params.max_halo_width

  def testSourceUpdateFnLibProvidesCorrectSourceTermFunctions(self):
    """Checks if the source term update function library is correct."""

    def src_u(kernel_op, replcia_id, replicas, states, additional_states,
              params):
      """Source term update function for `u`."""
      del kernel_op, replcia_id, replicas, states, additional_states, params
      return {'src_u': tf.constant(6, dtype=tf.float32)}

    def src_t(kernel_op, replcia_id, replicas, states, additional_states,
              params):
      """Source term update function for `T`."""
      del kernel_op, replcia_id, replicas, states, additional_states, params
      return {'src_T': tf.constant(8)}

    source_update_fn_lib = {'u': src_u, 'T': src_t}

    params = parameters_lib.SwirlLMParameters.config_from_proto(
        self.config_filepath)

    params.source_update_fn_lib = source_update_fn_lib

    self.assertCountEqual(['u', 'T'],
                          params.source_update_fn_lib.keys())

    kernel_op = get_kernel_fn.ApplyKernelConvOp(4)
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    states = {}
    additional_states = {}
    config = grid_parametrization.GridParametrization()
    with self.subTest(name='SourceUIs6'):
      src = self.evaluate(
          params.source_update_fn('u')(kernel_op, replica_id, replicas, states,
                                       additional_states, config))
      self.assertEqual(6.0, src['src_u'])

    with self.subTest(name='SourceTIs8'):
      src = self.evaluate(
          params.source_update_fn('T')(kernel_op, replica_id, replicas, states,
                                       additional_states, config))
      self.assertEqual(8.0, src['src_T'])

    with self.subTest(name='SourceWIsNone'):
      src = params.source_update_fn('w')
      self.assertIsNone(src)

  @parameterized.named_parameters(
      ('Case00', 'Y0', 1e-7),
      ('Case01', 'Y1', 1e-6),
      ('Case02', 'Y2', 8e-7),
  )
  def testScalarDiffusivity(self, scalar_name, expected_diffusivity):
    params = parameters_lib.SwirlLMParameters.config_from_proto(
        self.config_filepath_scalars)

    self.assertAlmostEqual(
        params.diffusivity(scalar_name), expected_diffusivity)

  @parameterized.named_parameters(
      ('Case00', 'Y0', 0.1),
      ('Case01', 'Y1', 1.0),
      ('Case02', 'Y2', 0.5),
  )
  def testScalarDensity(self, scalar_name, expected_density):
    params = parameters_lib.SwirlLMParameters.config_from_proto(
        self.config_filepath_scalars)

    self.assertAlmostEqual(params.density(scalar_name), expected_density)

  @parameterized.named_parameters(
      # Missing time integration scheme in scalar: Fall back to shared time
      # integration scheme.
      ('Case00', 'Y0', parameters_lib.TimeIntegrationScheme.TIME_SCHEME_RK3),
      ('Case01', 'Y1',
       parameters_lib.TimeIntegrationScheme.TIME_SCHEME_CN_EXPLICIT_ITERATION),
      ('Case02', 'Y2',
       parameters_lib.TimeIntegrationScheme.TIME_SCHEME_UNKNOWN),
      # Missing in scalars: Fall back to shared time integration scheme.
      ('Case03', 'Y3', parameters_lib.TimeIntegrationScheme.TIME_SCHEME_RK3),
  )
  def testScalarTimeIntegrationScheme(self, scalar_name, expected_dt_scheme):
    params = parameters_lib.SwirlLMParameters.config_from_proto(
        self.config_filepath_scalars)
    self.assertEqual(
        params.scalar_time_integration_scheme(scalar_name), expected_dt_scheme)


if __name__ == '__main__':
  tf.test.main()
