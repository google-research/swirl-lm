# Copyright 2022 The swirl_lm Authors.
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

"""A library with tools that processes simulation data."""

from typing import Sequence, Tuple

import numpy as np


def coordinates_to_indices(
    locations: np.ndarray,
    domain_size: Sequence[float],
    mesh_size_local: Sequence[int],
    partition: Sequence[int],
    halo_width: int,
) -> Tuple[Sequence[int], np.ndarray]:
  """Finds the indices of the partition and physical locations in each core.

  This function assumes that all probes are in the same core of partition.

  Args:
    locations: The indices of the probe locations. Stored in a 2D array of 3
      columns, with the columns being x, y, and z indices, respectively.
    domain_size: A three-element sequence that stores the physical size of the
      full domain.
    mesh_size_local: A length 3 tuple with elements being the number of mesh
      points in the x, y, and z directions in each core, respectively. Including
      halos.
    partition: A length 3 tuple with elements being the number of cores in the
      x, y, and directions, respectively.
    halo_width: The number of points contained in the halo layer on each side
      of the simulation mesh.

  Returns:
    A tuple of 2 elements. The first element is a length three sequence that
    stores the indices of the core in the partition. The second element is a 2D
    np.array. Each row of the array stores the index of the corresponding point
    in `locations` that's local to the core.
  """
  # Effective size of the mesh in each core.
  n = [core_n - 2 * halo_width for core_n in mesh_size_local]

  # Length of the domain in each core.
  core_l = [l_i / nc_i for l_i, nc_i in zip(domain_size, partition)]

  # Grid spacing.
  h = [
      l_i / (n_i * c_i - 1.0)
      for l_i, c_i, n_i in zip(domain_size, partition, n)
  ]

  # Find the indices of the core. Assumes that all probes are in the same
  # partition.
  c_indices = [np.int(locations[0][i] // core_l[i]) for i in range(3)]

  # Finds the indices of the physical coordinates inside the core.
  indices = np.zeros_like(locations, dtype=np.int)
  for i in range(3):
    indices[:, i] = np.array(
        (locations[:, i] - c_indices[i] * core_l[i]) // h[i] + halo_width,
        dtype=np.int)

  return c_indices, indices
