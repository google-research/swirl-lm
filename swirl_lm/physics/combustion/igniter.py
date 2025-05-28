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

"""A library of ignition functions."""

from typing import Sequence

from swirl_lm.base import initializer
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal
_DTYPE = tf.float32


class Igniter(object):
  """A library that manages the ignition kernel."""

  def __init__(
      self,
      ignition_speed: float,
      ignition_start_point: Sequence[float],
      ignition_duration: float,
      start_step_id: int,
      igniter_radius: float,
      dt: float,
  ):
    """Initializes the ignition scheduling object.

    The ignition schedule is created based on the following assumptions:
    1. Ignition starts from a single point;
    2. Ignition progresses with a constant speed in all directions along the
    the radius of a sphere centered at the starting location;
    3. The shape of the ignition kernel is determined by an externally-defined
    (arbitrary shaped) binary mask where the positive valued pixels
    representing the locations where the fire will be ignited.

    More specifically, the time duration when a particular positive-valued mask
    pixel is ignited is determined by its distance from the
    `ignition_start_point`, the `ignition_speed`, the `start_step_id` and
    `igniter_radius` in the following way:

    (current_step_id - start_step_id) * dt * ignition_speed - igniter_radius
             <       distance_from_start_point      <
    (current_step_id - start_step_id) * dt * ignition_speed + igniter_radius.

    Args:
      ignition_speed: The speed that the ignition kernel moves.
      ignition_start_point: The (x, y, z) coordinates of the starting point of
        the ignition.
      ignition_duration: The duration of the ignition event, in units of
        seconds.
      start_step_id: The step id at which the ignition starts.
      igniter_radius: The radius (in units of meter) of the ignition kernel in
        the ignition event.
      dt: The time step size of the simulation.
    """
    self._speed = ignition_speed
    self._origin = ignition_start_point
    self._start_step_id = float(start_step_id)
    self._dt = dt

    # Compute the start and end time of the ignition event.
    self._start_time = self._start_step_id * self._dt
    self._end_time = self._start_time + ignition_duration

    # Represent the igniter radius in units of time.
    self._igniter_radius_in_time = igniter_radius / self._speed

  def ignition_schedule_init_fn(
      self,
      ignition_kernel_shape_fn: initializer.ValueFunction,
  ) -> initializer.ValueFunction:
    """Generates an init function that guides the ignition sequence.

    Args:
      ignition_kernel_shape_fn: A function that provides a binary kernel
        specifying the overall shape of the ignition kernel (after the ignition
        sequence is finished). The overall ignition kernel has value 1, and
        everywhere else is 0.

    Returns:
      A function that generates a kernel of time relative to the start of
      the simulation (i.e. step 0) inside the ignition kernel, which specifies
      the sequence of an ignition event. Locations that are outside of the
      ignition event are set to be -2 times the ignition radius in units of
      steps to avoid undesired ignition.
    """

    def init_fn(
        xx: tf.Tensor,
        yy: tf.Tensor,
        zz: tf.Tensor,
        lx: float,
        ly: float,
        lz: float,
        coordinates: initializer.ThreeIntTuple,
    ) -> tf.Tensor:
      """Initializes the ignition sequence tensor."""

      distance = tf.math.sqrt((xx - self._origin[0])**2 +
                              (yy - self._origin[1])**2 +
                              (zz - self._origin[2])**2)

      ignition_time = distance / self._speed + self._start_time

      ignition_kernel = ignition_kernel_shape_fn(xx, yy, zz, lx, ly, lz,
                                                 coordinates)

      return tf.where(
          tf.greater(ignition_kernel, 0.0), ignition_time,
          -2.0 * self._igniter_radius_in_time * tf.ones_like(ignition_time))

    return init_fn

  def ignition_kernel(
      self,
      step_id: tf.Tensor,
      ignition_schedule: FlowFieldVal,
  ) -> FlowFieldVal:
    """Generates a binary ignition kernel at the present step.

    Args:
      step_id: The current step id.
      ignition_schedule: A 3D matrix (in the format of list of 2D tensor slices)
        of integers representing the `ignition step` at each pixle in the space
        when the current`step_id` is within the temporal raius from this
        `ignition step`, the pixel will be `on fire`.

    Returns:
      A binary tensor where 1 represents the location of the ignition kernel and
      0 elsewhere.
    """
    t = tf.cast(step_id, _DTYPE) * self._dt

    # Get the binary ignition kernel around the current physical time in the
    # simulation.
    def local_ignition_kernel_fn(schedule: tf.Tensor) -> tf.Tensor:
      """Generates an ignition kernel local to current the time and space."""
      return tf.compat.v1.where(
          tf.math.logical_and(
              tf.greater_equal(schedule, t - self._igniter_radius_in_time),
              tf.less_equal(schedule, t + self._igniter_radius_in_time),
          ),
          tf.ones_like(schedule),
          tf.zeros_like(schedule),
      )

    ignition_kernel = tf.nest.map_structure(
        local_ignition_kernel_fn, ignition_schedule
    )

    def trim_time_interval(kernel: tf.Tensor) -> tf.Tensor:
      """Limits ignition only in the time interval specified."""
      return tf.cond(
          tf.math.logical_and(
              tf.greater_equal(t, self._start_time),
              tf.less_equal(t, self._end_time),
          ),
          lambda: kernel,
          lambda: tf.zeros_like(kernel),
      )

    return tf.nest.map_structure(trim_time_interval, ignition_kernel)
