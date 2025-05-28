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

r"""The executable for the channel flow simulation.

"""

from absl import app
from swirl_lm.base import driver
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.example.channel import channel


def main(_):
  params = parameters_lib.params_from_config_file_flag()
  simulation = channel.Channel(params)
  params.additional_states_update_fn = simulation.additional_states_update_fn
  driver.solver(simulation.init_fn, params)


if __name__ == '__main__':
  app.run(main)
