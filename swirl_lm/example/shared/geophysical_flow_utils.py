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

"""Code shared among the simulation set up libraries (Dycoms2, SuperCell, ...).
"""
from typing import Optional, Tuple

from swirl_lm.base import initializer
from swirl_lm.utility import init_fn as init_fn_lib
import tensorflow as tf

# These prime numbers are used to space the seeding of the random number
# for different velocity components: u, v, and w to ensure there is no
# correlation between the random turbulence of these components.
U_SEED = 17167
V_SEED = 64567
W_SEED = 93097


def perturbed_constant_init_fn(
    seed: Tuple[int, int],
    mean: float,
    g_dim: int,
    local_grid_no_halos: Tuple[int, int, int],
    rms: Optional[float] = None,
    mean_init_fn: Optional[initializer.ValueFunction] = None,
    cloud_base=600.0,
) -> initializer.ValueFunction:
  """Initializes the field with `mean` perturbed by random noise.

  Args:
    seed: A tuple of (int, int) to seed the random number generation.
    mean: The mean field to be perturbed.
    g_dim: Gravity direction.
    local_grid_no_halos: The local grid size excluding the halos.
    rms: The root mean square/standard deviation to be applied.
    mean_init_fn: The initialization function for the mean flow field.
    cloud_base: The height of the bottom surface of the cloud, in units of m.
      The variable is only perturbed below this height.

  Returns:
    A function that generates the perturbed field with its mean being `mean`,
    and standard deviation being `rms` following a normal distribution.
  """

  def init_fn(xx: tf.Tensor, yy: tf.Tensor, zz: tf.Tensor, lx: float,
              ly: float, lz: float,
              coord: initializer.ThreeIntTuple) -> tf.Tensor:
    """Generates the initial field with a constant."""
    mean_val = init_fn_lib.constant_init_fn(
        mean) if mean_init_fn is None else mean_init_fn
    mean_val = mean_val(xx, yy, zz, lx, ly, lz, coord)

    if rms is None:
      return mean_val

    # We want to make sure different cors are seeded differently, so here we
    # incorporate the core cooridinates signature as part of the seed.
    seed_base = 702626207
    local_seed = (int.from_bytes(
        bytes('{}-{}-{}-{}'.format(coord[0], coord[1], coord[2], seed[0]),
              'utf-8'), 'big') % seed_base, seed[1])
    pert_val = tf.random.stateless_normal(
        local_grid_no_halos, local_seed, stddev=rms,
        dtype=mean_val.dtype) + mean_val

    height = (xx, yy, zz)[g_dim]
    return tf.compat.v1.where(
        tf.less(height, cloud_base), pert_val, mean_val)

  return init_fn
