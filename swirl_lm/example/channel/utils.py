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

"""Functions for input data preparation."""

from typing import Tuple

import numpy as np
from swirl_lm.boundary_condition import simulated_turbulent_inflow
from swirl_lm.utility.post_processing import data_processing
import tensorflow as tf


def interpolate_and_distribute_inflow_data(
    inflow_data: np.ndarray,
    target_prefix: str,
    target_step: int,
    target_n_core: Tuple[int, int, int],
    target_halo_width: Tuple[int, int, int],
    target_shape: Tuple[int, int, int],
) -> None:
  """Interpolates `inflow_data` to `shape` and writes to distributed files.

  This function assumes that the domain size remains the same before and after
  interpolation. It also assumes that the meshes before and after interpolation
  are uniform.

  Example:
    Inputs:
      # 'channel' is the actual prefix of the simulation.
      target_prefix = '/path/to/inflow/data/channel'
      target_step = 0
      target_n_core = (4, 1, 1)
      target_halo_width = (0, 2, 2)
      target_shape = (5120, 60, 60)

    Outputs:
      Inflow files that are partitioned and to be used as restart files for a
      simulation with Swirl-LM. The files will be written into the following
      directory: '/path/to/inflow/data/{target_step}'.

  Args:
    inflow_data: A 4D array that contains the 3 velocity components of the
      inflow, with the shape being [nt, ny, nz, 3], where y is the wall normal
      direction, and z is the spanwise direction.
    target_prefix: The prefix (with the full path included).
    target_step: The step number that the inflow data will be loaded. Note that
      this number has to match the `loading_step` flag when running the
      simulation.
    target_n_core: The number of cores of the partition along the x, y, and z
      axes, respectively.
    target_halo_width: The number of ghost cells to be allocated along the x (or
      t), y, and z axes. It should be equal to `halo_width` in the simulation
      flag for spatial dimensions, and 0 for the time dimension.
    target_shape: The new shape of the unpartitioned inflow data excluding ghost
      cells on the boundary.
  """
  inflow_varnames = simulated_turbulent_inflow.INFLOW_DATA_NAMES

  # The first dimension (t) of the inflow is partitioned along the x axis in
  # Swirl-LM (here we always assume that the channel flow is aligned along the
  # x-axis). The second and third dimension corresponds to the z and y axes,
  # respectively.
  inflow_data = np.transpose(inflow_data, [0, 2, 1, 3])

  for i in range(inflow_data.shape[-1]):
    buf = data_processing.interpolate_data(inflow_data[..., i], target_shape)
    tensor = tf.convert_to_tensor(buf, dtype=tf.float32)
    data_processing.distribute_and_write_serialized_tensor(
        tensor,
        target_prefix,
        inflow_varnames[i],
        target_step,
        target_n_core,
        target_halo_width,
        mode='xzy',
    )
