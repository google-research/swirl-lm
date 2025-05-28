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

"""Utilities for working with Lagrangian particles."""

from typing import Sequence, TypeAlias
import numpy as np
from swirl_lm.numerics import interpolation
from swirl_lm.physics.lpt import lpt_types
from swirl_lm.utility import common_ops
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldMap: TypeAlias = types.FlowFieldMap


def get_particle_replica_id(
    locations: tf.Tensor,
    core_spacings: tf.Tensor,
    replicas: np.ndarray,
    global_min_pt: tf.Tensor,
) -> tf.Tensor:
  """A fast determination of the replicas containing particles on a uniform grid.

  This function assumes that the core spacing is uniform along all dimensions,
  so it will not work directly with stretched grids and only their mapped
  variants. Returns -1 if the particle is out of bounds. For the below
  definitions, x0, x1, x2 correspond to the z, x, and y dimensions respectively.

  Args:
    locations: A tensor of shape (n, 3) containing the z, x, y particle
      locations.
    core_spacings: A three element tensor of floats z, x, y representing the
      length of each dimension contained within a replica's partial domain.
    replicas: A 3D numpy array of core replica IDs of shape cx, cy, cz for `cx`,
      `cy`, `cz` cores along each dimension.
    global_min_pt: A size three tuple containing the global minimum point of all
      cores in the domain of order (cz, cx, cy).

  Returns:
    A tensor of replica IDs of the particles, -1 if the particle is out of
    bounds.
  """
  x0_ind, x1_ind, x2_ind = [
      tf.cast(
          tf.floor((locations[:, i] - global_min_pt[i]) / core_spacings[i]),
          dtype=lpt_types.LPT_INT,
      )
      for i in range(3)
  ]

  core_n1, core_n2, core_n0 = replicas.shape

  out_of_bounds = (
      (x0_ind < 0)
      | (x0_ind >= core_n0)
      | (x1_ind < 0)
      | (x1_ind >= core_n1)
      | (x2_ind < 0)
      | (x2_ind >= core_n2)
  )

  replicas = tf.cast(replicas, dtype=lpt_types.LPT_INT)

  replica_ids = tf.where(
      out_of_bounds,
      -tf.ones_like(x0_ind),
      common_ops.gather(replicas, tf.stack([x1_ind, x2_ind, x0_ind], axis=-1)),
  )

  return replica_ids


def fluid_data_linear_interpolation(
    locations: tf.Tensor,
    states: FlowFieldMap,
    variables: Sequence[str],
    grid_spacings: tf.Tensor,
    local_grid_min_pt: tf.Tensor,
) -> tf.Tensor:
  """Interpolates local fluid data within the replica.

  Calculates fluid variables for the particles in the local replica assuming a
  uniform grid. The variables are trilinearly interpolated from the fluid grid
  points. Any locations exceeding the replica domain will give invalid
  solutions.

  Args:
    locations: An (n, 3) tensor containing `n` locations at (`z`, `x`, `y`).
    states: A FlowFieldMap containing the fluid states.
    variables: A tuple containing the names of the fluid variables to
      interpolate. This is used to query the `states` dictionary. Example, ("w",
      "u", "v").
    grid_spacings: The grid spacings in the `z`, `x`, and `y` dimensions.
    local_grid_min_pt: The minimum point of the local grid including halos. This
      should correspond to the location of the grid point located at (0, 0, 0)
      in the states tensors.

  Returns:
    The an (n, m) tensor containing fluid data at the `n` locations for the `m`
    variables.
  """
  return interpolation.trilinear_interpolation(
      field_data=tf.stack(
          [states[variable] for variable in variables], axis=-1
      ),
      points=locations,
      grid_spacing=grid_spacings,
      local_grid_min_pt=local_grid_min_pt,
  )


def get_active_rows(
    lpt_field_ints: lpt_types.LptFieldInts,
    float_field: lpt_types.LptFieldFloats,
) -> tuple[lpt_types.LptFieldFloats, tf.Tensor]:
  """Returns only the rows corresponding to the active particles.

  Args:
    lpt_field_ints: A size `n` tensor of integer particle states, the 2 columns
      are particle statuses (1=active, 0=inactive), particle ids (global).
    float_field: A tensor of floats with dimensions (n, m) where `n` corresponds
      to the above dimension, and `m` an arbitrary number of columns. This can
      be, for instance, the `lpt_field_floats` tensor.

  Returns:
    A tensor of float particle states of only the active particles with
    dimensions (q, m), where `q` is the number of active particles.
    A tensor of length `q` containing the indices of the rows of the active
    particles.
  """
  statuses = lpt_field_ints[:, 0]
  active_indices = tf.reshape(tf.where(statuses == 1), [-1])
  active_tensor = tf.one_hot(active_indices, tf.shape(float_field)[0])
  active_rows = tf.einsum("qn,nm->qm", active_tensor, float_field)
  return active_rows, active_indices


def tensor_scatter_update(
    tensor: tf.Tensor,
    indices: tf.Tensor,
    updates: tf.Tensor,
) -> tf.Tensor:
  """Updates the rows in a 2D tensor with the updates at the given indices.

  This function is similar to tf.tensor_scatter_nd_update, but works
  successfully with the XLA compiler when `updates` is of dynamic sizing.

  Args:
    tensor: A two dimensional tensor of shape (n, m) that will be updated.
    indices: A one_dimensional tensor of shape (q,) containing the indices of
      the rows to update in `tensor`.
    updates: A tensor of shape (q, m) containing the rows that will overwrite
      the rows in `tensor` at `indices`.

  Returns:
    `Tensor` with the `updates` applied at the rows denoted by `indices`. If
    `indices` is length zero, the `tensor` is returned unchanged.
  """
  one_hot = tf.cast(
      tf.one_hot(indices, depth=tf.shape(tensor)[0]),
      dtype=lpt_types.LPT_FLOAT,
  )
  substitute = tf.cast(
      tf.einsum("qj,qi->ij", updates, one_hot), dtype=lpt_types.LPT_FLOAT
  )
  inverted = tf.cast(
      tf.where(
          tf.greater(tf.abs(substitute), np.finfo(np.float32).resolution),
          tf.zeros_like(substitute),
          tf.ones_like(substitute),
      ),
      dtype=lpt_types.LPT_FLOAT,
  )
  return tensor * inverted + substitute
