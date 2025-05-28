# Copyright 2025 The swirl_lm Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fire uncertainty quantification."""
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf


_RANDOM_SEED_UQ = flags.DEFINE_integer(
    'random_seed_uq',
    0,
    'The random seed used for sampling the uncertain parameters.'
)
_N_SAMPLES_UQ = flags.DEFINE_integer(
    'n_samples_uq',
    100,
    'The number of samples collected for uncertainty quantification.'
)
_FUEL_DENSITY_LB = flags.DEFINE_float(
    'fuel_density_lb',
    0.2,
    'Lower bound of fuel density for uncertainty quantification.'
)
_FUEL_DENSITY_UB = flags.DEFINE_float(
    'fuel_density_ub',
    3.0,
    'Upper bound of fuel density for uncertainty quantification.'
)
_MOISTURE_CONTENT_LB = flags.DEFINE_float(
    'moisture_content_lb',
    0.03,
    'Lower bound of moisture content for uncertainty quantification.'
)
_MOISTURE_CONTENT_UB = flags.DEFINE_float(
    'moisture_content_ub',
    0.12,
    'Upper bound of moisture content for uncertainty quantification.'
)
_WIND_SPEED_LB = flags.DEFINE_float(
    'wind_speed_lb',
    5.0,
    'Lower bound of wind speed for uncertainty quantification.'
)
_WIND_SPEED_UB = flags.DEFINE_float(
    'wind_speed_ub',
    12.0,
    'Upper bound of wind speed for uncertainty quantification.'
)
_MODIFY_INDIVIDUAL = flags.DEFINE_bool(
    'modify_indiviual',
    False,
    'Whether to modify the different values used in uncertainty quantification'
    'one at a time. This produces only 4 trajectories and should only be used'
    'for validation.'
)
_READ_UQ_FILE = flags.DEFINE_bool(
    'read_uq_file',
    False,
    'Whether to read the UQ-parameters from a file',
)
_UQ_FILENAME = flags.DEFINE_string(
    'uq_filename',
    './large_scale_uq_params.npy',
    'The location of the .npy file that contains the uq values'
)
_UQ_START_ID = flags.DEFINE_integer(
    'uq_start_id',
    0,
    'The start id for uq.'
)
_UQ_END_ID = flags.DEFINE_integer(
    'uq_end_id',
    19,
    'The end id for uq.'
)


def load_uq_values_from_file():
  with tf.io.gfile.GFile(_UQ_FILENAME.value, 'rb') as f:
    return np.load(f)


class FireUQSampler:
  """Sampler for fire uncertainty quantification."""

  def __init__(self):
    self.sampler = np.random.default_rng(_RANDOM_SEED_UQ.value)
    self.start_id = _UQ_START_ID.value
    self.end_id = _UQ_END_ID.value
    if _MODIFY_INDIVIDUAL.value:
      logging.warn(
          'Modifying uncertain parameters one at a time. Only generating 4'
          'trajectories (1 trajectory for baseline and 1 for each varied'
          'parameter). Ignores n_samples_uq flag.'
      )
    pass

  def _get_norm_samples(self, n_samples, n_uq_vars=3):
    norm_samples = self.sampler.random(size=(n_samples, n_uq_vars),
                                       dtype=np.float32)
    return norm_samples

  def _shift_scale(self, lower_bound, upper_bound, norm_samples):
    """Shifts and scales to the interval [lower_bound, uppber_bound].

    We do not use `self.sampler.uniform` for compatibility with previous version
    of random number generation and for keeping the first n uq-samples the same.

    Args:
      lower_bound: Lower bound.
      upper_bound: Upper bound.
      norm_samples: Samples to shift and scale.

    Returns:
      Shifted and scaled samples.
    """
    return (upper_bound - lower_bound) * norm_samples + lower_bound

  def number_of_samples(self):
    if _MODIFY_INDIVIDUAL.value:
      return 4
    elif _READ_UQ_FILE.value:
      uq_values = load_uq_values_from_file()
      return uq_values.shape[0]
    else:
      return _N_SAMPLES_UQ.value

  def generate_samples(self):
    """Generates samples for random variables considered in UQ.

    Returns:
      fuel_density_samples, moisture_density_samples, wind_speed_samples
    """
    if _READ_UQ_FILE.value:
      uq_values = load_uq_values_from_file()
      fuel_density_samples = np.float32(uq_values[:, 0])
      moisture_density_samples = np.float32(uq_values[:, 1])
      wind_speed_samples = np.float32(uq_values[:, 2])
    else:
      if _MODIFY_INDIVIDUAL.value:
        fuel_density_samples = np.ones(4) * _FUEL_DENSITY_LB.value
        fuel_density_samples[1] = _FUEL_DENSITY_UB.value
        moisture_content_samples = np.ones(4) * _MOISTURE_CONTENT_LB.value
        moisture_content_samples[2] = _MOISTURE_CONTENT_UB.value
        moisture_density_samples = (moisture_content_samples *
                                    fuel_density_samples)
        wind_speed_samples = np.ones(4) * _WIND_SPEED_LB.value
        wind_speed_samples[3] = _WIND_SPEED_UB.value
      else:
        norm_samples = self._get_norm_samples(_N_SAMPLES_UQ.value, 3)
        fuel_density_samples = self._shift_scale(
            _FUEL_DENSITY_LB.value, _FUEL_DENSITY_UB.value, norm_samples[:, 0]
        )
        moisture_content_samples = self._shift_scale(
            _MOISTURE_CONTENT_LB.value,
            _MOISTURE_CONTENT_UB.value,
            norm_samples[:, 1]
        )
        moisture_density_samples = (moisture_content_samples *
                                    fuel_density_samples)
        wind_speed_samples = self._shift_scale(
            _WIND_SPEED_LB.value, _WIND_SPEED_UB.value, norm_samples[:, 2]
        )
    return fuel_density_samples, moisture_density_samples, wind_speed_samples

  def generate_data_dump_prefixes(self, data_dump_prefix):
    data_dump_prefix_base = data_dump_prefix[:-1] + '_'
    if _MODIFY_INDIVIDUAL.value:
      data_dump_prefix_base = data_dump_prefix_base + 'testrun_'
    data_dump_prefixes = []
    for i in range(self.number_of_samples()):
      data_dump_prefixes.append(data_dump_prefix_base + f'{i}/')
    return data_dump_prefixes


def main(_):
  uq_sampler = FireUQSampler()
  fd_samples, md_samples, ws_samples = uq_sampler.generate_samples()
  uq_params = np.stack((fd_samples, md_samples, ws_samples)).T
  logging.info('New uq-samples created as follows.\n %s', uq_params)


if __name__ == '__main__':
  app.run(main)
