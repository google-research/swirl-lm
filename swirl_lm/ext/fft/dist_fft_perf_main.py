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

"""Distributed FFT Performance main program."""

import functools

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

from swirl_lm.ext.fft import dist_fft


_NUM_CYCLES = flags.DEFINE_integer(
    'num_cycles',
    1,
    'number of cycles to run. Each cycle generates a set of output',
)
_NUM_TRANSFORMS = flags.DEFINE_integer(
    'num_transforms', 1,
    'number of transforms to run before generating an output.'
)
_CX = flags.DEFINE_integer('cx', 1, 'number of partitions in x')
_CY = flags.DEFINE_integer('cy', 1, 'number of partitions in y')
_NX = flags.DEFINE_integer('nx', 128, 'global grid size in x')
_NY = flags.DEFINE_integer('ny', 128, 'global grid size in y')
_BACKEND = flags.DEFINE_string('backend', 'tpu', '`tpu` or `gpu`')

FLAGS = flags.FLAGS

np.random.seed(123456)


def gen_random(local_shape, core_index, seeds):
  key_0 = jax.random.key(
      seeds[0] + core_index[0] + 10 * core_index[1] + 100 * core_index[2])
  key_1 = jax.random.key(
      seeds[1] + core_index[0] + 10 * core_index[1] + 100 * core_index[2])
  a = jax.random.normal(key_0, local_shape, dtype=jnp.float32)
  b = jax.random.normal(key_1, local_shape, dtype=jnp.float32)
  return jnp.complex64(a + jnp.complex64(1j) * b)


def main(_):
  partition = (_CX.value, _CY.value, 1)
  def input_fn(local_shape, core_index):
    return gen_random(local_shape, core_index, (10061, 23399))

  def kernel_fn(local_shape, core_index):
    kernel_local = gen_random(local_shape, core_index, (12143, 16573))
    kernel_norm = jax.numpy.linalg.norm(kernel_local)
    return kernel_local / kernel_norm

  transformer = dist_fft.DistFFT(('x', 'y', 'z'), partition,
                                 backend=_BACKEND.value)

  global_shape = (_NX.value, _NY.value, 1)

  @functools.partial(jax.jit, static_argnums=(0))
  def run_fft_2d_jit(num):
    return transformer.fft_2d_perf(global_shape, input_fn, kernel_fn, num=num)

  num_transforms = _NUM_TRANSFORMS.value
  for cycle in range(_NUM_CYCLES.value):
    logging.info('starting FFT cycle %d, for %d transforms',
                 cycle, num_transforms)
    run_fft_2d_jit(num=num_transforms)
    logging.info('FFT cycle %d done.', cycle)


if __name__ == '__main__':
  app.run(main)
