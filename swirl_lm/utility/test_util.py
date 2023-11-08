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

"""Library for common operations used in distributed tests."""

import numpy as np
from swirl_lm.utility import common_ops
import tensorflow as tf


def get_split_inputs(
    u_full,
    v_full,
    w_full,
    replicas,
    halos,
):
  """Creates split inputs from full field components with halos added.

  Args:
    u_full: The x component of the full-grid vector field.
    v_full: They y component of the full-grid vector field.
    w_full: The z component of the full-grid vector field.
    replicas: A 3D numpy array representing the mapping from the core replica
      coordinate to the `replica_id`. The number of cores in each dimension is
      the number of splits of the global input for the transformation.
    halos: The width of the (symmetric) halos for each dimension: for example
      [1, 2, 3] means the halos have width of 1, 2, 3 on both sides in x, y, z
      dimension respectively.

  Returns:
    An array mapping the replica id to the local vector field that was assigned
    to it. The local vector field consists of 3 z-lists, one for each vector
    component.
  """
  split_inputs = [[] for _ in range(replicas.size)]
  compute_shape = replicas.shape
  paddings = [[halos[0], halos[0]], [halos[1], halos[1]], [halos[2], halos[2]]]
  nx_core = u_full.shape[0] // compute_shape[0]
  ny_core = u_full.shape[1] // compute_shape[1]
  nz_core = u_full.shape[2] // compute_shape[2]

  for i in range(compute_shape[0]):
    for j in range(compute_shape[1]):
      for k in range(compute_shape[2]):
        u_core = tf.cast(
            tf.transpose(
                tf.pad(
                    u_full[i * nx_core:(i + 1) * nx_core,
                           j * ny_core:(j + 1) * ny_core,
                           k * nz_core:(k + 1) * nz_core],
                    paddings=paddings),
                perm=[2, 0, 1]), tf.float32)
        v_core = tf.cast(
            tf.transpose(
                tf.pad(
                    v_full[i * nx_core:(i + 1) * nx_core,
                           j * ny_core:(j + 1) * ny_core,
                           k * nz_core:(k + 1) * nz_core],
                    paddings=paddings),
                perm=[2, 0, 1]), tf.float32)
        w_core = tf.cast(
            tf.transpose(
                tf.pad(
                    w_full[i * nx_core:(i + 1) * nx_core,
                           j * ny_core:(j + 1) * ny_core,
                           k * nz_core:(k + 1) * nz_core],
                    paddings=paddings),
                perm=[2, 0, 1]), tf.float32)
        state = {'u_core': u_core, 'v_core': v_core, 'w_core': w_core}
        split_state = common_ops.split_state_in_z(
            state, ['u_core', 'v_core', 'w_core'], nz_core + 2 * halos[2])
        split_inputs[replicas[i, j, k]] = [
            [
                split_state[common_ops.get_tile_name('u_core', i)]
                for i in range(nz_core + 2 * halos[2])
            ],
            [
                split_state[common_ops.get_tile_name('v_core', i)]
                for i in range(nz_core + 2 * halos[2])
            ],
            [
                split_state[common_ops.get_tile_name('w_core', i)]
                for i in range(nz_core + 2 * halos[2])
            ]
        ]
  return split_inputs


def merge_output(
    split_result,
    nx_full,
    ny_full,
    nz_full,
    halos,
    replicas,
):
  """Merges output from TPU replicate computation into a single result."""
  compute_shape = replicas.shape
  num_replicas = replicas.size
  nx = nx_full // compute_shape[0]
  ny = ny_full // compute_shape[1]
  nz = nz_full // compute_shape[2]
  merged_result = np.zeros([nx_full, ny_full, nz_full])
  assert len(split_result) == num_replicas
  for i in range(num_replicas):
    coord = np.where(replicas == i)
    cx = coord[0][0]
    cy = coord[1][0]
    cz = coord[2][0]
    combined = np.stack(split_result[i], axis=2)
    merged_result[cx * nx:(cx + 1) * nx, cy * ny:(cy + 1) * ny,
                  cz * nz:(cz + 1) * nz] = combined[halos[0]:halos[0] + nx,
                                                    halos[1]:halos[1] + ny,
                                                    halos[2]:halos[2] + nz]
  return merged_result
