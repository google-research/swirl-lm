# Copyright 2026 The swirl_lm Authors.
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
"""Run buoyant bubble case
"""

from absl import app
from absl import flags
from google.protobuf import text_format
import jax

from swirl_lm.base import parameters_pb2
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.core import driver as driver_lib
from swirl_lm.jax.examples.buoyant_bubble import buoyant_bubble

jax.config.update("jax_enable_x64", False)

_NUM_CYCLES = flags.DEFINE_integer(
    'num_cycles',
    1,
    'number of cycles to run. Each cycle generates a set of output',
)
_NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 1, 'number of steps to run before generating an output.'
)
_CONFIG_FILEPATH = flags.DEFINE_string(
    'config_filepath', None,
    'The full path to a config file.',
    allow_override=True
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir', None,
    'The full path to a checkpoint directory.',
    allow_override=True
)
_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'checkpoint_interval',
    10,
    'interval at which to write solution.',
)

FLAGS = flags.FLAGS


def _make_params() -> parameters_lib.SwirlLMParameters:
  """Builds the bubble configuration used by the integration test."""
  # Initialize empty instance
  config = parameters_pb2.SwirlLMParameters()

  with open(FLAGS.config_filepath, "r") as file:
    text_data = file.read()
    text_format.Parse(text_data, config)
  return parameters_lib.SwirlLMParameters(config)


def main(_):
  params = _make_params()
  print(f'Apply preprocessing: {params.apply_preprocess}')
  print(f'Preprocessing step ID: {params.preprocess_step_id}')

  simulation = buoyant_bubble.BuoyantBubble(params)
  params.additional_states_update_fn = simulation.additional_states_update_fn
  if params.thermodynamics.WhichOneof('thermodynamics_type') == 'water':
    params.preprocessing_states_update_fn = (
        simulation.water_thermodynamics_preprocess_fn
    )
  driver_lib.run_simulation(
      params=params,
      init_fn=simulation.initial_states,
      num_steps=FLAGS.num_cycles,
      num_steps_per_cycle=FLAGS.num_steps,
      checkpoint_dir=FLAGS.checkpoint_dir,
      checkpoint_interval=FLAGS.checkpoint_interval,
  )


if __name__ == '__main__':
  app.run(main)
