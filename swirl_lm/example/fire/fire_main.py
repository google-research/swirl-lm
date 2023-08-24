# Copyright 2023 The swirl_lm Authors.
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

r"""The executable for the fire simulation."""

from absl import flags
from swirl_lm.base import driver as tf2_driver
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.example.fire import fire
from swirl_lm.example.shared import wildfire_utils
import tensorflow as tf

# Ignored.
_IS_TF2 = flags.DEFINE_boolean(
    'is_tf2', True, 'Whether to run the job as a tf2 binary.'
)


def main(_):
  params = parameters_lib.params_from_config_file_flag()
  fire_utils = wildfire_utils.WildfireUtils(params, None)
  simulation = fire.Fire(fire_utils)
  params.source_update_fn_lib = simulation.source_term_update_fn()
  params.additional_states_update_fn = simulation.additional_states_update_fn
  params.preprocessing_states_update_fn = simulation.pre_simulation_update_fn
  params.postprocessing_states_update_fn = simulation.post_simulation_update_fn
  tf2_driver.solver(simulation.initialization, params)


if __name__ == '__main__':
  tf.compat.v1.app.run(main)
