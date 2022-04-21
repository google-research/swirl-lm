"""Tests for google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.utilities.monitor."""

import numpy as np
from swirl_lm.utility import monitor
from swirl_lm.utility import monitor_pb2
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.research.simulation.tensorflow.fluid.framework import test_util
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_parameters_pb2
from google3.testing.pybase import parameterized

_NP_DTYPE = np.float32


class MonitorTest(tf.test.TestCase, parameterized.TestCase):

  _PERIODIC_DIMS = [
      [False, False, True],
      [False, True, False],
      [False, True, True],
      [True, False, False],
      [True, False, True],
      [True, True, False],
      [True, True, True],
  ]

  _LOCAL_GRID_SIZE = [6, 7, 8]
  _NUM_SUBITER = 2

  def setUpMonitor(self,
                   periodic_dims=None,
                   local_grid_dims=None,
                   time_averaging=None,
                   dt=0.1):
    """Sets up the monitor config with optional periodic dimensions."""
    config = text_format.Parse(
        """
        helper_var_keys: "src_T"
        helper_var_keys: "src_Y_O"
        helper_var_keys: "MONITOR_pressure_convergence"
        helper_var_keys: "MONITOR_pressure_raw_convergence"
        helper_var_keys: "MONITOR_pressure_subiter-scalar_convergence"
        monitor_spec{
          state_analytics {
            state_name: 'u'
            analytics {
              key: 'MONITOR_velocity_moment_u_2'
              moment_statistic {
                order: 2
              }
            }
            analytics {
              key: 'MONITOR_cross_moment_u_v_2'
              moment_statistic {
                order: 2
                second_state: 'v'
              }
            }
          }
          state_analytics {
            state_name: 'v'
            analytics {
              key: 'MONITOR_velocity_raw_v'
              raw_state {}
            }
          }
        }
        """, incompressible_structured_mesh_parameters_pb2
        .IncompressibleNavierStokesParameters())
    config.num_sub_iterations = MonitorTest._NUM_SUBITER
    if time_averaging is not None:
      config.monitor_spec.time_averaging.MergeFrom(time_averaging)

    params = (
        incompressible_structured_mesh_config
        .IncompressibleNavierStokesParameters(config))

    params.halo_width = 2
    params.periodic_dims = periodic_dims
    params.dt = dt

    if local_grid_dims:
      params.nx = local_grid_dims[0]
      params.ny = local_grid_dims[1]
      params.nz = local_grid_dims[2]

    self.monitor = monitor.Monitor(params)

  def setUp(self):
    """Initializes the monitor object."""
    super(MonitorTest, self).setUp()
    self.setUpMonitor(local_grid_dims=self._LOCAL_GRID_SIZE)

  @parameterized.parameters(*zip(_PERIODIC_DIMS))
  def testMonitorVarInitGeneratesCorrectDictionaryOfAnalyticalQuantities(
      self, periodic_dims):
    """Checks if the analytics is initialized correctly."""

    self.setUpMonitor(periodic_dims, MonitorTest._LOCAL_GRID_SIZE)

    # Subtract halos or reduce dimension to 1 if in a homogeneous direction.
    raw_init = np.zeros(shape=np.roll(
        MonitorTest._LOCAL_GRID_SIZE, 1), dtype=_NP_DTYPE)
    subiter_scalar_init = np.zeros(
        shape=[MonitorTest._NUM_SUBITER], dtype=_NP_DTYPE)
    nx = 1 if periodic_dims[0] else 2
    ny = 1 if periodic_dims[1] else 3
    nz = 1 if periodic_dims[2] else 4
    moment_init = np.zeros(shape=[nz, nx, ny], dtype=_NP_DTYPE)

    expected = {
        'MONITOR_velocity_moment_u_2': moment_init,
        'MONITOR_cross_moment_u_v_2': moment_init,
        'MONITOR_velocity_raw_v': raw_init,
        'MONITOR_pressure_convergence': 0.0,
        'MONITOR_pressure_raw_convergence': raw_init,
        'MONITOR_pressure_subiter-scalar_convergence': subiter_scalar_init,
    }

    with self.subTest(name='ContainerInObject'):
      data = self.evaluate(self.monitor.data)

      self.assertCountEqual(expected.keys(), data.keys())
      for k, v in expected.items():
        self.assertAllEqual(v, data[k])

    with self.subTest(name='ExternalContainer'):
      data = self.evaluate(self.monitor.monitor_var_init())

      self.assertCountEqual(expected.keys(), data.keys())
      for k, v in expected.items():
        self.assertAllEqual(v, data[k])

  @parameterized.named_parameters(
      # Scalars.
      #   - Present.
      ('Case00', 'MONITOR_pressure_convergence', True),
      ('Case01', 'MONITOR_pressure_raw_convergence', True),
      ('Case02', 'MONITOR_pressure_subiter-scalar_convergence', True),
      ('Case03', 'MONITOR_velocity_moment_u_2', True),

      #   - Absent.
      ('Case10', 'MONITOR_velocity_moment_w_2', False),
      ('Case11', 'MONITOR_velocity_raw_convergence', False),
      ('Case12', 'MONITOR_velocity_subiter-scalar_convergence', False),
      ('Case13', 'src_T', False),
      # List.
      ('Case20', ('MONITOR_pressure_convergence',), True),
      ('Case21', ('MONITOR_pressure_convergence', 'MONITOR_unknown'), True),
      ('Case22', ('MONITOR_unknown',), False),
  )
  def testCheckKeyProducesCorrectInformationAboutKey(self, key,
                                                     expected_presence):
    """Checks if the `check_key` function provides correct judgement to keys."""
    self.assertEqual(self.monitor.check_key(key), expected_presence)

  def testUpdateCorrectlyUpdatesAnalytics(self):
    """Checks if the analytics data are updated correctly."""
    raw_expected = np.ones(shape=np.roll(
        MonitorTest._LOCAL_GRID_SIZE, 1), dtype=_NP_DTYPE)
    raw_initial = np.zeros(shape=np.roll(
        MonitorTest._LOCAL_GRID_SIZE, 1), dtype=_NP_DTYPE)
    subiter_scalar_expected = np.ones(shape=[MonitorTest._NUM_SUBITER],
                                      dtype=_NP_DTYPE)
    with self.subTest(name='PressureConvergenceUpdatesCorrectly'):
      self.monitor.update('MONITOR_pressure_convergence', tf.constant(1.0))
      self.monitor.update('MONITOR_pressure_raw_convergence',
                          tf.constant(raw_expected))
      self.monitor.update('MONITOR_pressure_subiter-scalar_convergence',
                          tf.constant(subiter_scalar_expected))

      data = self.evaluate(self.monitor.data)

      expected = {
          'MONITOR_velocity_moment_u_2': [[[0.0]]],
          'MONITOR_cross_moment_u_v_2': [[[0.0]]],
          'MONITOR_velocity_raw_v': raw_initial,
          'MONITOR_pressure_convergence': 1.0,
          'MONITOR_pressure_raw_convergence': raw_expected,
          'MONITOR_pressure_subiter-scalar_convergence':
              subiter_scalar_expected,
      }
      self.assertCountEqual(expected.keys(), data.keys())
      for k in data.keys():
        self.assertAllEqual(expected[k], data[k])

    with self.assertRaisesRegex(ValueError, 'not a valid analytical variable'):
      self.monitor.update('src_T', tf.constant(300.0))

    with self.assertRaisesRegex(
        ValueError,
        'MONITOR_pressure_raw_convergence shape mismatch in this simulation: '):
      self.monitor.update('MONITOR_pressure_raw_convergence',
                          np.ones(MonitorTest._LOCAL_GRID_SIZE))

  @parameterized.named_parameters(
      dict(
          testcase_name='2x1x1_periodic_x',
          replicas=np.array([[[0]], [[1]]]),
          periodic_dims=(True, False, False)),
      dict(
          testcase_name='1x1x2_periodic_y',
          replicas=np.array([[[0], [1]]]),
          periodic_dims=(False, True, False)),
      dict(
          testcase_name='1x1x2_periodic_z',
          replicas=np.array([[[0, 1]]]),
          periodic_dims=(False, False, True)),
      dict(
          testcase_name='1x8x1_periodic_y',
          replicas=np.array([[[0], [1], [2], [3], [4], [5], [6], [7]]]),
          periodic_dims=(False, True, False)),
      dict(
          testcase_name='2x4x1_periodic_xy',
          replicas=np.array([[[0], [1], [2], [3]], [[4], [5], [6], [7]]]),
          periodic_dims=(True, True, False)),
      dict(
          testcase_name='2x1x4_periodic_x',
          replicas=np.array([[[0, 3, 5, 7]], [[4, 1, 6, 2]]]),
          periodic_dims=(True, False, True)),
      dict(
          testcase_name='1x2x2_periodic_yz',
          replicas=np.array([[[0, 1], [2, 3]]]),
          periodic_dims=(False, True, True)),
      dict(
          testcase_name='2x2x2_periodic_xyz',
          replicas=np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
          periodic_dims=(True, True, True)),
  )
  def testComputeAnalyticsCorrectlyComputesMomentStatistics(
      self, replicas, periodic_dims):
    nx_full = 124
    ny_full = 120
    nz_full = 64
    halo_width = 2
    halos = [halo_width] * 3
    compute_shape = replicas.shape
    num_replicas = replicas.size
    nx = int(nx_full / compute_shape[0] + 2 * halos[0])
    ny = int(ny_full / compute_shape[1] + 2 * halos[1])
    nz = int(nz_full / compute_shape[2] + 2 * halos[2])
    self.setUpMonitor(periodic_dims, [nx, ny, nz])

    axis = [i for i, v in enumerate(periodic_dims) if v]

    v_full = np.random.normal(size=[nx_full, ny_full, nz_full])
    u_full = np.random.normal(size=[nx_full, ny_full, nz_full])
    w_full = np.random.normal(size=[nx_full, ny_full, nz_full])
    u_mean = np.mean(u_full, axis=tuple(axis), keepdims=True)
    v_mean = np.mean(v_full, axis=tuple(axis), keepdims=True)
    expected_u2 = np.transpose(
        np.mean((u_full - u_mean)**2, axis=tuple(axis), keepdims=True),
        axes=[2, 0, 1])
    expected_cross_moment_2 = np.transpose(
        np.mean(
            ((u_full - u_mean) * (v_full - v_mean))**2,
            axis=tuple(axis),
            keepdims=True),
        axes=[2, 0, 1])

    split_inputs = test_util.get_split_inputs(u_full, v_full, w_full, replicas,
                                              halos)
    # transpose split_inputs.
    split_inputs = [list(inputs) for inputs in zip(*split_inputs)]

    def computation(
        u,
        v,
        w,
    ):
      states = {'u': u, 'v': v, 'w': w}
      states.update(self.monitor.data)
      return self.monitor.compute_analytics(states, replicas)

    runner = TpuRunner(computation_shape=compute_shape)
    results = runner.run(computation, *split_inputs)
    with self.subTest('simple moment'):
      for i in range(num_replicas):
        self.assertAllClose(expected_u2,
                            results[i]['MONITOR_velocity_moment_u_2'])
    with self.subTest('cross moment'):
      for i in range(num_replicas):
        self.assertAllClose(expected_cross_moment_2,
                            results[i]['MONITOR_cross_moment_u_v_2'])

  @parameterized.named_parameters(
      dict(
          testcase_name='2x1x1_periodic_x',
          replicas=np.array([[[0]], [[1]]]),
          periodic_dims=(True, False, False)),
      dict(
          testcase_name='1x1x2_periodic_y',
          replicas=np.array([[[0], [1]]]),
          periodic_dims=(False, True, False)),
      dict(
          testcase_name='1x1x2_periodic_z',
          replicas=np.array([[[0, 1]]]),
          periodic_dims=(False, False, True)),
      dict(
          testcase_name='1x8x1_periodic_y',
          replicas=np.array([[[0], [1], [2], [3], [4], [5], [6], [7]]]),
          periodic_dims=(False, True, False)),
      dict(
          testcase_name='2x4x1_periodic_xy',
          replicas=np.array([[[0], [1], [2], [3]], [[4], [5], [6], [7]]]),
          periodic_dims=(True, True, False)),
      dict(
          testcase_name='2x1x4_periodic_x',
          replicas=np.array([[[0, 3, 5, 7]], [[4, 1, 6, 2]]]),
          periodic_dims=(True, False, True)),
      dict(
          testcase_name='1x2x2_periodic_yz',
          replicas=np.array([[[0, 1], [2, 3]]]),
          periodic_dims=(False, True, True)),
      dict(
          testcase_name='2x2x2_periodic_xyz',
          replicas=np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
          periodic_dims=(True, True, True)),
  )
  def testComputeAnalyticsCorrectlyComputesTimeAveragedMomentStatistics(
      self, replicas, periodic_dims):
    """Checks the moment statistics are correctly computed and time-averaged."""
    nx_full = 124
    ny_full = 120
    nz_full = 64
    halo_width = 2
    halos = [halo_width] * 3
    compute_shape = replicas.shape
    num_replicas = replicas.size
    nx = int(nx_full / compute_shape[0] + 2 * halos[0])
    ny = int(ny_full / compute_shape[1] + 2 * halos[1])
    nz = int(nz_full / compute_shape[2] + 2 * halos[2])
    dt = 0.1

    axis = [i for i, v in enumerate(periodic_dims) if v]

    def build_inputs_and_statistic():
      v_full = np.random.normal(size=[nx_full, ny_full, nz_full])
      u_full = np.random.normal(size=[nx_full, ny_full, nz_full])
      w_full = np.random.normal(size=[nx_full, ny_full, nz_full])
      u_mean = np.mean(u_full, axis=tuple(axis), keepdims=True)
      statistic = np.transpose(
          np.mean((u_full - u_mean)**2, axis=tuple(axis), keepdims=True),
          axes=[2, 0, 1])

      split_inputs = test_util.get_split_inputs(u_full, v_full, w_full,
                                                replicas, halos)
      return split_inputs, statistic

    split_inputs1, expected_moment1 = build_inputs_and_statistic()
    split_inputs2, expected_moment2 = build_inputs_and_statistic()
    split_inputs3, expected_moment3 = build_inputs_and_statistic()
    split_inputs = [
        inp1 + inp2 + inp3
        for inp1, inp2, inp3 in zip(split_inputs1, split_inputs2, split_inputs3)
    ]
    # transpose split_inputs.
    split_inputs = [list(inputs) for inputs in zip(*split_inputs)]
    expected_u2 = (expected_moment1 + expected_moment2 + expected_moment3) / 3
    expected_u2_last_two_steps = (expected_moment2 + expected_moment3) / 2

    def computation(
        u1,
        v1,
        w1,
        u2,
        v2,
        w2,
        u3,
        v3,
        w3,
    ):
      states1 = {'u': u1, 'v': v1, 'w': w1}
      states1.update(self.monitor.data)
      monitor_states = self.monitor.compute_analytics(states1, replicas, 0)
      states2 = {'u': u2, 'v': v2, 'w': w2}
      states2.update(monitor_states)
      monitor_states = self.monitor.compute_analytics(states2, replicas, 1)
      states3 = {'u': u3, 'v': v3, 'w': w3}
      states3.update(monitor_states)
      return self.monitor.compute_analytics(states3, replicas, 2)

    with self.subTest('Average over all time'):
      time_averaging = monitor_pb2.TimeAveraging()
      self.setUpMonitor(periodic_dims=periodic_dims,
                        local_grid_dims=[nx, ny, nz],
                        time_averaging=time_averaging,
                        dt=dt)
      runner = TpuRunner(computation_shape=compute_shape)
      results = runner.run(computation, *split_inputs)
      for i in range(num_replicas):
        self.assertAllClose(expected_u2,
                            results[i]['MONITOR_velocity_moment_u_2'])

    with self.subTest('Average over last two steps'):
      time_averaging = monitor_pb2.TimeAveraging(start_time_seconds=dt)
      self.setUpMonitor(periodic_dims=periodic_dims,
                        local_grid_dims=[nx, ny, nz],
                        time_averaging=time_averaging,
                        dt=dt)

      runner = TpuRunner(computation_shape=compute_shape)
      results = runner.run(computation, *split_inputs)

      for i in range(num_replicas):
        self.assertAllClose(expected_u2_last_two_steps,
                            results[i]['MONITOR_velocity_moment_u_2'])

  def testComputeAnalyticsCorrectlyReturnsRawStates(self):
    """Checks the `raw_state` analytics just passes the raw state through."""
    replicas = np.array([[[0]]])
    periodic_dims = (True, True, False)
    nx = 124
    ny = 120
    nz = 64
    halos = [0] * 3
    compute_shape = replicas.shape
    self.setUpMonitor(periodic_dims, [nx, ny, nz])

    v_full = np.random.normal(size=[nx, ny, nz])
    u_full = np.random.normal(size=[nx, ny, nz])
    w_full = np.random.normal(size=[nx, ny, nz])
    expected_raw_v = np.transpose(v_full, axes=[2, 0, 1])

    split_inputs = test_util.get_split_inputs(u_full, v_full, w_full, replicas,
                                              halos)
    # transpose split_inputs.
    split_inputs = [list(inputs) for inputs in zip(*split_inputs)]

    def computation(
        u,
        v,
        w,
    ):
      states = {'u': u, 'v': v, 'w': w}
      states.update(self.monitor.data)
      return self.monitor.compute_analytics(states, replicas)

    runner = TpuRunner(computation_shape=compute_shape)
    results = runner.run(computation, *split_inputs)

    self.assertAllClose(expected_raw_v, results[0]['MONITOR_velocity_raw_v'])


if __name__ == '__main__':
  tf.test.main()
