# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for Percipitation modeling."""

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import parameters_pb2
from swirl_lm.physics.atmosphere import precipitation
from swirl_lm.physics.thermodynamics import water
from swirl_lm.utility import tf_test_util as test_util
import tensorflow as tf

from google3.net.proto2.python.public import text_format


@test_util.run_all_in_graph_and_eager_modes
class PrecipitationTest(tf.test.TestCase):

  def setUp(self):
    super(PrecipitationTest, self).setUp()
    self.precipitation = self.set_up_model()

  def set_up_model(self):
    pbtxt = (R'gravity_direction { '
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
             R'}  '
             R'scalars {  '
             R'  name: "q_t"  '
             R'  diffusivity: 1e-5  '
             R'  density: 1.0   '
             R'  molecular_weight: 0.018  '
             R'  solve_scalar: true  '
             R'}  ')
    config = text_format.Parse(pbtxt, parameters_pb2.SwirlLMParameters())
    params = parameters_lib.SwirlLMParameters(config)
    return precipitation.Precipitation(water.Water(params))

  def testEvaporationRate(self):
    """Evaporation rate of rain drops is as expected."""
    np.random.seed(0)
    model = self.precipitation
    shape = [8, 16, 16]
    ones = np.ones(shape, dtype=np.float32)
    # Setting up values to be within reasonable physical range.
    rho = np.maximum(
        np.random.normal(loc=0.8 * ones, scale=0.1 * ones),
        0.0).astype(np.float32)
    temperature = np.maximum(
        np.random.normal(loc=283.0 * ones, scale=2.0 * ones),
        0.0).astype(np.float32)
    q_r = np.minimum(
        np.maximum(np.random.normal(loc=0.1 * ones, scale=0.02 * ones), 0.001),
        0.5).astype(np.float32)
    q_v = np.minimum(
        np.maximum(np.random.normal(loc=0.1 * ones, scale=0.02 * ones), 0.001),
        0.5).astype(np.float32)
    q_c = np.minimum(
        np.maximum(np.random.normal(loc=0.1 * ones, scale=0.02 * ones), 0.001),
        1.0 - q_v).astype(np.float32)
    q_l = np.maximum(
        q_c -
        np.maximum(np.random.normal(loc=0.03 * ones, scale=0.002 * ones), 0.0),
        0.0).astype(np.float32).astype(np.float32)

    c = 1.6 + 30.3922 * np.power((rho * q_r), 0.2046).astype(np.float32)
    q_vs = self.evaluate(
        model.water_model.saturation_q_vapor(temperature, rho, q_l, q_c))
    p_vs = q_vs * model.water_model.r_v * rho * temperature
    expected_evaporation_rate = ((1.0 / rho) * (1 - q_v / q_vs) * c *
                                 np.power((rho * q_r), 0.525) /
                                 (2.03e4 + 9.584e6 / p_vs))

    actual_evaporation_rate = self.evaluate(
        tf.stack(
            model.rain_evaporation_rate_kw1978(
                tf.unstack(rho), tf.unstack(temperature), tf.unstack(q_r),
                tf.unstack(q_v), tf.unstack(q_l), tf.unstack(q_c))))

    self.assertAllClose(
        np.stack(actual_evaporation_rate), expected_evaporation_rate)

  def testCloudLiquidToRainConversionRate(self):
    """Cloud liquid conversion rate to drops is as expected."""
    np.random.seed(0)
    model = self.precipitation
    shape = [8, 16, 16]
    ones = np.ones(shape, dtype=np.float32)
    # Setting up values to be within reasonable physical range.
    q_r = np.minimum(
        np.maximum(np.random.normal(loc=0.1 * ones, scale=0.02 * ones), 0.001),
        0.5).astype(np.float32)
    q_l = np.minimum(
        np.maximum(np.random.normal(loc=0.15 * ones, scale=0.02 * ones), 0.001),
        0.5).astype(np.float32)
    a_r = 0.001 * (q_l - 0.001)
    c_r = 2.2 * q_l * np.power(q_r, 0.875)
    expected_conversion_rate = a_r + c_r

    actual_conversion_rate_0 = self.evaluate(
        tf.stack(
            model.cloud_liquid_to_rain_conversion_rate_kw1978(
                tf.unstack(q_r), tf.unstack(q_l))))
    actual_conversion_rate_1 = self.evaluate(
        model.cloud_liquid_to_rain_conversion_rate_kw1978(q_r, q_l))

    self.assertAllClose(actual_conversion_rate_0, expected_conversion_rate)
    self.assertAllClose(actual_conversion_rate_1, expected_conversion_rate)

  def testWaterTerminalVelocity(self):
    """Rain water terminal velocity is as expected."""
    np.random.seed(0)
    model = self.precipitation
    shape = [8, 16, 16]
    ones = np.ones(shape, dtype=np.float32)
    # Setting up values to be within reasonable physical range.
    q_r = np.minimum(
        np.maximum(np.random.normal(loc=0.1 * ones, scale=0.02 * ones), 0.001),
        0.5).astype(np.float32)
    # Setting up values to be within reasonable physical range.
    rho = np.maximum(np.random.normal(loc=0.8 * ones, scale=0.1 * ones),
                     0.0).astype(np.float32)
    rho_ref = 1.15  # [kg/m^3]
    expected_terminal_velocity = 14.34 * np.power(rho * q_r, 0.1346) * np.sqrt(
        rho_ref / rho)

    actual_terminal_velocity_0 = self.evaluate(
        tf.stack(
            model.rain_water_terminal_velocity_kw1978(
                tf.unstack(rho), tf.unstack(q_r), rho_ref)))
    actual_terminal_velocity_1 = self.evaluate(
        model.rain_water_terminal_velocity_kw1978(rho, q_r, rho_ref))
    self.assertAllClose(actual_terminal_velocity_0, expected_terminal_velocity)
    self.assertAllClose(actual_terminal_velocity_1, expected_terminal_velocity)

    # Check using default value for rho_ref.
    actual_terminal_velocity_3 = self.evaluate(
        model.rain_water_terminal_velocity_kw1978(rho, q_r))
    self.assertAllClose(actual_terminal_velocity_3, expected_terminal_velocity)


if __name__ == '__main__':
  tf.test.main()
