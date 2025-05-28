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

r"""The executable for the fire simulation.

# Flags to set up the fire simulation

Note: Details about how the flags are used to set up the fire simulation are in
`swirl_lm/example/fire/fire.py`, `swirl_lm/example/shared/wildfire_utils.py`,
and `swirl_lm/example/fire/terrain_utils.py`.

## Terrain related flags
`flat_surface_terrain_type`: The method used to specify the terrain in the
simulation. Supported types are `FILE`, `RAMP`, `BUMP`, and `WEDGE`. Please see
`swirl_lm/example/fire/fire.py` for details on how these terrain types are
generated.

### In case `flat_surface_terrain_type=FILE`
  - `terrain_filepath`: The path to a tensorflow serialized file or a numpy
  data file, which stores a 2D array that specifies the altitude at each point.
  It is assumed that the physical length and width of the map are the same as
  the computational domain.

### In case `flat_surface_terrain_type=RAMP`

Note: This option specifies a ramp with a constant slope along the x-axis. The
ramp may start and end with a plateau and may be customized with the following
flags.

  - `flat_surface_initial_height`: The height of the ramp at x = 0, in units of
  m.
  - `flat_surface_max_height`: The maximum allowed height of the terrain, in
  units of m.
  - `flat_surface_ramp_start_point`: The starting point of the ramp. The terrain
  remains flat from x = 0 to this point.
  - `flat_surface_ramp_length`: The length of the ramp. The terrain plateaus at
  the highest point of the ramp for x larger than
  `flat_surface_ramp_start_point` + `flat_surface_ramp_length`.
  - `flat_surface_slope`: The slope of the ramp in units of degrees.

## Fuel related flags
- `fuel_density`: The volume averaged bulk density of the fuel, which is
 computed as the fuel load ($$kg/m^2$$) / fuel height.
- `fuel_bed_height`: The height of the fuel bed.
- `moisture_density`: The volume averaged bulk density of the moisture, in units
of $$kg/m^3$$.
- `c_d`: The drag coefficient of the fuel

## Fire related flags
Note: When ignition is specified, a high temperature kernel will be imposed onto
both the gas and solid phase temperatures at `post_process_step_id`. The
`post_process_step_id` can be specified either through a flag or in the
the configuration proto file. All ignition kernels are applied in regions where
fuel density is non-zero, even if the original shape of the kernel extends
beyond that region.

- `flat_surface_ignite`: A boolean option that indicates whether to ignite a
fire during the simulation.
- `ignition_temperature`: The temperature of the ignition kernel, in units of K.
- `ignition_option`: The shape of the ignition kernel. Predefined options are
`SLANT_LINE`, `MULTI_LINE`, `SPHERE`, `BOX`.

### In case `ignition_option=SLANT_LINE`

The ignition kernel is a rectangle specified by its center coordinates, length,
width, and the angle with respect to the x axis with the following flags. This
kernel will be applied for fuels at all heights within the area.

- `ignition_center_x`: The x coordinate of the center of the rectangle.
- `ignition_center_y`: The y coordinate of the center of the rectangle.
- `ignition_line_length`: The length of the rectangle.
- `ignition_line_thickness`: The width of the rectangle.
- `ignition_line_angle`: The angle between the length of the rectangle and the
x axis.

### In case `ignition_option=SPHERE`

The ignition kernel is a sphere with its boundary smoothed by a tanh function.

- `ignition_center_x`: The x coordinate of the center of the sphere.
- `ignition_center_y`: The y coordinate of the center of the sphere.
- `ignition_center_z`: The z coordinate of the center of the sphere.
- `ignition_radius`: The radius of the sphere.
- `ignition_scale`: The smoothness factor of the tanh function.
"""

from absl import app
from swirl_lm.base import driver
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.example.fire import fire
from swirl_lm.example.shared import wildfire_utils


def main(_):
  params = parameters_lib.params_from_config_file_flag()
  fire_utils = wildfire_utils.WildfireUtils(params, None)
  simulation = fire.Fire(fire_utils)
  params.source_update_fn_lib = simulation.source_term_update_fn()
  params.additional_states_update_fn = simulation.additional_states_update_fn
  params.preprocessing_states_update_fn = simulation.pre_simulation_update_fn
  params.postprocessing_states_update_fn = simulation.post_simulation_update_fn
  driver.solver(simulation.initialization, params)


if __name__ == '__main__':
  app.run(main)
