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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX implementation of distributed FFT.


An M-way split Cooley-Tuckey at the multi-core distributed level. Within a
single core, the computation is done with JAX single-core FFT.

This currently supports 3D inputs. NB: to get sufficeint precision on TPU,
please enable x64 support.
"""
import collections
import functools
from typing import Callable, Text, Tuple

import einops
import jax
from jax import lax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np

SUPPORTED_BACKENDS = ['tpu', 'gpu', 'cpu']


class DistFFT():
  """Distributed FFT.

  * Usage example to do distributed fast Fourier transform on unpartitioned
  global input:

      transformer = DistFFT(['x', 'y', 'z'], [2, 2, 2])
      output = transformer.fft_1d(unpartitioned_matrix, transfrom_dim=0)

  * Usage example for operating on partitioned input (within partition):

      # Outside the partition/pmap, creates a dist-FFT transformer with
      # the same `axis_names` and a consistent `partition` for the
      # partition/pmap:

      axis_names = ['x', 'y', 'z']
      partition = [2, 2, 2]  # partition into 2 x 2 x 2.

      transformer = DistFFT(axis_names, partition)

          ......
          ......

      # Inside the partition, use the above transformer to perform
      # the fft and subsequently use the result as needed:

      @functools.partial(jax.pmap, axis_name=axis_names[0], ..)
      @functools.partial(jax.pmap, axis_name=axis_names[1], ..)
      @functools.partial(jax.pmap, axis_name=axis_names[2], ..)
      def some_distributed_operation_fun(pmapped_input_array, .....):
          fft_input = some_operation(pmapped_input_array, ....)
          partitioned_transformed = transformer.partitioned_fft_1d(
            fft_input, ...)

          ......

    NB: All internal operations (einsum, matrix multiplication) will use the
    highest precision available for the backend platform, independent of the
    input. For example, on TPU, this is single precision (float32), while on CPU
    and GPU, it will be double precision (float64).
  """

  def __init__(
      self,
      axis_names: Tuple[Text, Text, Text],
      partition: Tuple[int, int, int],
      backend: str = 'tpu',
  ) -> None:
    """Set up a handler for the distributed FFT.

    Contains immutable settings `axis_names`, `partition`, and `backend` that
    the distributed FFT will be based on.

    Args:
      axis_names: A Tuple of three elements representing the names of the three
        axes representing dimension 0, 1, 2.
      partition: The global partition shape.
      backend: The backend type for the distributed FFT. One of 'tpu', 'gpu',
        and 'cpu'.

    Raises:
      ValueError if the `backend` is not one of the supported types.
    """
    if backend not in SUPPORTED_BACKENDS:
      raise ValueError('Unsupported bacend type %s. Supported types are %s' %
                       (backend, SUPPORTED_BACKENDS))
    self._backend = backend
    self._axis_names = axis_names
    self._partition = partition
    self._shuffle_eqs = ['ijk,il->ljk', 'ijk,jl->ilk', 'ijk,kl->ijl']
    self._fft_perms = [[2, 1, 0], [0, 2, 1], [0, 1, 2]]
    self._source_target_pairs = [
        [(i, (i + 1) % l) for i in range(l)] for l in self._partition
    ]
    # These will be set if fft_2d_perf is called.
    self._devices = None
    self._mesh = None

  def _create_mesh(self):
    self._devices = mesh_utils.create_device_mesh(
        (self._partition[0], self._partition[1], self._partition[2]),
        allow_split_physical_axes=True)
    self._mesh = Mesh(self._devices, axis_names=(
        self._axis_names[0], self._axis_names[1], self._axis_names[2]))

  def _exp_itheta(self, theta: jnp.ndarray) -> jnp.ndarray:
    """Calculates exp(i theta).

    Args:
      theta: theta.

    Returns:
      exp(i * theta)
    """
    return jnp.cos(theta) + jnp.complex64(1.0j) * jnp.sin(theta)

  def _get_shuffle_op(
      self,
      n: int,
      m: int,
      l: int,
      axis: int,
      margin_width: int = 0,
  ) -> jnp.ndarray:
    """Creates a shuffle operator to rearrange the elements on each core.

    This creates a 2D matrix that is used to contract the 3D partial input
    in the `axis` to shuffle all elements into `m` segments. The way the
    shuffling is performed is: in `axis`, lining up all elements globally in
    the original input order, and label each from 0 to m-1 cyclically, and each
    element labelled `k` will be shuffled to the corresponding k'th segment on
    each core. Note that on each core, the length in the `axis` `n` is not
    always divisible by `m`, so some of the segments will have missing elements.
    These missing elements are filled with 0. The matrix is then padded with 0
    on top and bottom with 0 for `margin_width`.

    For a concrete example, we consider a full input (without margins) with
    total length 12 (not including margins) in the `axis`. We further assume
    the input is distributed over 3 TPU cores and `margin_width` to be 1. We
    have the following distribution (where `M` represents margin):

      axis index:   |         0        |         1        |        2         |
      element id:   | M  0  1  2  3  M | M  4  5  6  7  M | M  8  9 10 11  M |
      label:        | M  0  1  2  0  M | M  1  2  0  1  M | M  2  0  1  2  M |

    The desired shuffled result should be:

      axis index:   |         0        |         1        |        2         |
      element id:   | 0  3  1  x  2  x | 6  x  4  7  5  x | 9  x 10  x  8 11 |
      label:        | 0  0  1  1  2  2 | 0  0  1  1  2  2 | 0  0  1  1  2  2 |

    Note that on each core, the elements are aggregated according to their
    respective label. The `x` is dummy filling -- which we can fill with 0. Also
    note that values in margin are not used.

    The desired transformation can be done through the contraction `ij,jk->ik`
    with the following `jk` matrix:

      axis index:   |         0        |         1        |        2         |

                    | .  .  .  .  .  . | .  .  .  .  .  . | .  .  .  .  .  . |
                    | 1  .  .  .  .  . | .  .  1  .  .  . | .  .  .  .  1  . |
      matrix:       | .  .  1  .  .  . | .  .  .  .  1  . | 1  .  .  .  .  . |
                    | .  .  .  .  1  . | 1  .  .  .  .  . | .  .  1  .  .  . |
                    | .  1  .  .  .  . | .  .  .  1  .  . | .  .  .  .  .  1 |
                    | .  .  .  .  .  . | .  .  .  .  .  . | .  .  .  .  .  . |

                                                           (`.` represents `0`)

    The function below generates the corresponding matrix that is needed.

    Args:
      n: The length of the partial input in the `axis`, excluding margins.
      m: The number of cores (or divisions) to distribute the input in the
        `axis`.
      l: The size of the segment of each of the `m` segments. This number is
        ceil(n / m).
      axis: The dimension the transformation is to be performed.
      margin_width: The width of the margin.

    Returns:
      A 2D matrix of the shape [n + 2 * margin_width, m * l] to be used to
      contract the partial input as demonstrated in the example above.
    """
    position_index = lax.axis_index(self._axis_names[axis])
    raw_indices = jnp.arange(0, n)
    shift_indices = jnp.fmod(position_index * n, m) + raw_indices
    indices = (raw_indices // m) + jnp.fmod(shift_indices, m) * l
    return jnp.pad(
        jax.nn.one_hot(indices, num_classes=m * l),
        [[margin_width, margin_width], [0, 0]])

  def _get_gather_adjust_op(
      self,
      n: int,
      m: int,
      l: int,
      axis: int,
  ) -> jnp.ndarray:
    """Creates a 2D matrix that removes the padding in the gathered input.

    This is used to contract the partial input in the `axis` to remove
    possible padding before performing local fft.

    Continuing the example used in the description of `_get_shuffle_op`, the
    input on each core, after the shuffling operation and the `all_to_all`
    operation:

      replica id:   |         0        |         1        |        2         |
      element id:   | 0  3  6  x  9  x | 1  x  4  7 10  x | 2  x  5  x  8 11 |
      label:        | 0  0  0  0  0  0 | 1  1  1  1  1  1 | 2  2  2  2  2  2 |

    While the desired input that can be used for transformation is

      replica id:   |      0     |      1     |     2      |
      element id:   | 0  3  6  9 | 1  4  7 10 | 2  5  8 11 |

    This can be achieved through the following 2D matrices used to contract the
    input:

      replica id:   |      0     |      1     |     2      |

                    | 1  .  .  . | 1  .  .  . | 1  .  .  . |
                    | .  1  .  . | .  .  .  . | .  .  .  . |
      matrix:       | .  .  1  . | .  1  .  . | .  1  .  . |
                    | .  .  .  . | .  .  1  . | .  .  .  . |
                    | .  .  .  1 | .  .  .  1 | .  .  1  . |
                    | .  .  .  . | .  .  .  . | .  .  .  1 |

                                                           (`.` represents `0`)

    The method here generates the needed matrices before the local FFT can be
    carried out.

    Args:
      n: The length of the partial input in the `axis`, excluding margins.
      m: The number of cores (or divisions) to split distribute the input in the
        `axis`.
      l: The size of the segment of each of the `m` segments. This number is
        ceil(n / m).
      axis: The dimension the transformation is to be performed: can be 0, 1, or
        2.

    Returns:
      A 2D matrix of the shape [m * l, n] to be used to contract the gathered
      partial input as demonstrated in the example above.
    """

    position_index = lax.axis_index(self._axis_names[axis])
    pos_indices = jnp.arange(0, m * l)
    source_position = pos_indices // l
    source_order = jnp.fmod(pos_indices, l)
    source_start = jnp.fmod(source_position * n, m)
    adjustment = jnp.where(
        jnp.less_equal(source_start, position_index), jnp.zeros([m * l]),
        m * jnp.ones([m * l]))
    mask = jnp.where(
        jnp.less(source_order * m + position_index - source_start + adjustment,
                 n), jnp.ones([m * l]), jnp.zeros([m * l]))
    upper_triangular = jnp.triu(jnp.ones([m * l, m * l]), 1)
    indices = jnp.int32(jnp.einsum('i,ij->j', mask, upper_triangular))
    indices = jnp.where(mask > 0, indices, -1 * jnp.ones([m * l]))
    return jax.nn.one_hot(indices, num_classes=n)

  def _get_correction_factor(
      self,
      n: int,
      m: int,
      axis: int,
      inverse: bool,
  ) -> jnp.ndarray:
    """Get the phase correction factors to adjust the local fft results.

    Args:
      n: The length of the partial input in the `axis`, excluding margins.
      m: The number of cores (or divisions) to split distribute the input in the
        `axis`.
      axis: The dimension the transformation is to be performed.
      inverse: Boolean. Whether the transformation is an inverse transformation.

    Returns:
      A 1D complex matrix of length `n` with the correction factors.
    """
    inverse_factor = -1.0 if inverse else 1.0
    inverse_amplitude = 1.0 / m if inverse else 1.0
    position_index = lax.axis_index(self._axis_names[axis])

    twiddle_index_rem = (position_index * inverse_factor * jnp.arange(0, n)) % (
        n * m)
    factor = self._exp_itheta(
        (-2.0 * jnp.pi * twiddle_index_rem) / (n * m)) * inverse_amplitude
    for _ in range(2 - axis):
      factor = jnp.expand_dims(factor, -1)
    return factor

  def fft_1d(self,
             global_input: jnp.ndarray,
             axis: int,
             inverse: bool = False,
             margin_width: int = 0,
             merge_output: bool = True) -> jnp.ndarray:
    """Performs a distributed 1D Fourier transform on an unpartitioned input.

    This operates on an unpartitioned input `global_input`. The input is
    expected to be in 3D array format with each dimension divisible by the
    corresponding partition of the `DistFFT` object. The result by default is
    merged back in the unpartitioned form, but if `merge_output` is set to
    `False`, it is in the format of

        [partition_0, partition_1, partition_2, local_0, local_1, local_2],

    where `partition_n` is the number of partiions along the axis `n`
    and `local_n` is the size (including of the margin on both sides) of the
    partitioned output in the axis `n`.


    Args:
      global_input: A 3D array representing the unpartiioned input for the fft.
      axis: An `int`. The dimension of the `global_input` along which to perform
        fft. Must be one of [0, 1, 2] or a `ValueError` will be raised.
      inverse: Whether the fft is an inverse fft or a forward fft. Default is
        `False`.
      margin_width: Default to 0. The 1-side width of the 2-sided margin along
        the `axis` dimension in the partitioned input. The values within the
        margins will be ignored and not used in the overall transformation. The
        output will maintain the same shape as the input and include the margins
        as well but the values within the margins should not be used. There is
        no guarantee on what these values will be. Negative `margin_width` will
        cause a `ValueError`.
      merge_output: A bool. Default is `True`. Whether to merge the output back
        into the unpartitioned 3D format or keep it as partitioned 6D format
        with the shape of  [partition_0, partition_1, partition_2, local_0,
        local_1, local_2],  where `partition_n` is the number of partiions
        along the dimension `n` and `local_n` is the size (including of the
        margin on both sides) of the partitioned output in the dimension `n`.

    Returns:
      A 3D or 6D array representing the transformed result.

    Raises:
      ValueError: (1) If `axis` is invalid (not 0, 1, or 2). (2) `margin_width`
      is negative.
    """

    partition_x = self._partition[0]
    partition_y = self._partition[1]
    partition_z = self._partition[2]
    global_x = global_input.shape[0]
    global_y = global_input.shape[1]
    global_z = global_input.shape[2]
    assert global_x % partition_x == 0
    assert global_y % partition_y == 0
    assert global_z % partition_z == 0
    global_input_reshaped = einops.rearrange(
        global_input,
        '(x1 x) (y1 y) (z1 z) -> x1 y1 z1 x y z',
        x1=partition_x,
        y1=partition_y,
        z1=partition_z)

    @functools.partial(
        jax.pmap,
        axis_name=self._axis_names[0],
        static_broadcasted_argnums=[1, 2, 3])
    @functools.partial(
        jax.pmap,
        axis_name=self._axis_names[1],
        static_broadcasted_argnums=[1, 2, 3])
    @functools.partial(
        jax.pmap,
        axis_name=self._axis_names[2],
        static_broadcasted_argnums=[1, 2, 3])
    def do_transform(global_input_reshaped, axis, inverse, margin_width):
      return self.partitioned_fft_1d(global_input_reshaped, axis, inverse,
                                     margin_width)

    split_output = do_transform(global_input_reshaped, axis, inverse,
                                margin_width)
    if merge_output:
      output = einops.rearrange(split_output,
                                'x1 y1 z1 x y z -> (x1 x) (y1 y) (z1 z)')
    else:
      output = split_output

    return output

  def fft_2d_perf(self,
                  global_shape: Tuple[int, int, int],
                  input_fn: Callable[[Tuple[int, int, int],
                                      Tuple[int, int, int]], jnp.ndarray],
                  kernel_fn: Callable[[Tuple[int, int, int],
                                       Tuple[int, int, int]], jnp.ndarray],
                  num: int = 1) -> jnp.ndarray:
    """Performs `num` sets of (1 FFT + 1 pointwise Mul + 1 iFFT) operations.

    This operates on input generated with `input_fn` and `kernel_fn`, both are
    fucntions that takes local 2D shape (nx, ny, 1) and the local core
    coordinate (cx, cy, 0) as input and returns a complex 2D array with shape
    (nx, ny, 1).

    Using the `input` and `kernel` as the initial values, this will perform
    `num` sets of FFT + pointwise Mul + iFFT operations repetedly.

    Args:
      global_shape: A 1D array specifing the unpartitioned 2D shape (Nx, Ny, 1):
        `Nx` must be divisible by the partition in the first dimension, `Ny`
        must be divisible by the partition in the 2nd dimension. The partiaion
        in the 3rd dimension and the 3rd dimension of the global_shape both has
        to be 1.
      input_fn: A function that takes the local shape (nx, ny, 1) and local core
        coordinate (cx, cy, 0) as the input and returns a 2D array with shape
        (nx, ny, 1). This is used as the initial input for the FFT.
      kernel_fn:  A function that takes the local shape (nx, ny, 1) and local
        core coordinate (cx, cy, 0) as the input and returns a 2D array with
        shape (nx, ny, 1). This is used for the point-wise multiplication.
      num: Number of cycles of the operation.

    Returns:
      A 2D array representing the transformed result.

    """

    partition_x = self._partition[0]
    partition_y = self._partition[1]
    partition_z = self._partition[2]
    global_x = global_shape[0]
    global_y = global_shape[1]
    global_z = global_shape[2]
    assert global_x % partition_x == 0
    assert global_y % partition_y == 0
    assert partition_z == 1
    assert global_z == 1

    nx = int(global_x / partition_x)
    ny = int(global_y / partition_y)

    self._create_mesh()

    @functools.partial(
        shard_map, mesh=self._mesh,
        in_specs=(),
        out_specs=P(None, None, None), check_rep=False)
    def do_transform():
      core_coord = (lax.axis_index(self._axis_names[0]),
                    lax.axis_index(self._axis_names[1]), 0)
      input_signal = input_fn((nx, ny, 1), core_coord)  # pytype: disable=wrong-arg-types  # lax-types
      kernel = kernel_fn((nx, ny, 1), core_coord)  # pytype: disable=wrong-arg-types  # lax-types
      out_signal = input_signal
      for _ in range(num):
        out_signal = self.partitioned_fft_1d(
            self.partitioned_fft_1d(
                self.partitioned_fft_1d(
                    self.partitioned_fft_1d(
                        out_signal, 0, False, 0),
                    1, False, 0) * kernel, 0, True, 0),
            1, True, 0)
      return out_signal

    split_output = do_transform()

    return split_output

  def partitioned_fft_1d(
      self,
      partitioned_input: jnp.ndarray,
      axis: int,
      inverse: bool = False,
      margin_width: int = 0,
  ) -> jnp.ndarray:
    """Performs a 1d fft or inverse fft on partitioned input.


    Args:
      partitioned_input: This represents the partial input handled by the
        corresponding core replica. It is expected to be a 3D matrix.
      axis: The dimension to perform the fft. Must be one of 0, 1, or 2.
      inverse: A boolean indicating whether the fft is to be done as an inverse
        fft.
      margin_width: The width of the margin in the local partition for the
        dimension to be transformed.

    Returns:
      A 3D matrix with the same shape as `partitioned_input` but with the
      dimension `axis` transformed to the global fft (or
      inverse fft), with the original input margin width. Note, after this
      transform, the margin does not contain the correct values. The caller
      need to perform the additional operations to get the corret values if
      needed.

    Raises:
      ValueError: (1) If `axis` is invalid (not 0, 1, or 2). (2) `margin_width`
      is negative.
    """
    # Validates input parameters.
    input_shape = partitioned_input.shape
    if len(input_shape) != 3:
      raise ValueError('Unsupported input shape %s. This currently only '
                       'supports 3D input.' % str(input_shape))
    supported_dims = [0, 1, 2]
    if axis not in supported_dims:
      raise ValueError('Unsupported `axis`: %d. This currently only '
                       'supports transform along dimensions %s' %
                       (axis, str(supported_dims)))
    if margin_width < 0:
      raise ValueError('Negative `margin_width` is not supported.')

    split_count = self._partition[axis]
    local_group_l = int(
        np.ceil((input_shape[axis] - 2 * margin_width) / split_count))
    # Forming the ops that can be used in einsum to shuffle the input.
    n = (input_shape[axis] - 2 * margin_width)
    shuffle_op = self._get_shuffle_op(n, split_count, local_group_l, axis,
                                      margin_width)
    shuffled_input = jnp.einsum(
        self._shuffle_eqs[axis],
        partitioned_input,
        jnp.float32(shuffle_op),
        precision=lax.Precision.HIGHEST)
    all_to_all_dim = axis

    # This is a TPU specific optimization for the `all_to_all` operation. If
    # the split is done on the inner-most (the last) dimension, the length in
    # that dimension to be 128x after the split. If not, the wasteful padding
    # occurs. We mitigate this issue by transposing the inner-most dimension to
    # an outer dimension.
    need_avoid_all_to_all_padding = (
        self._backend == 'tpu' and axis == 2 and (local_group_l % 128) != 0)
    if need_avoid_all_to_all_padding:
      shuffled_input = jnp.transpose(shuffled_input, [0, 2, 1])
      all_to_all_dim = 1
    gathered_input = lax.all_to_all(
        x=shuffled_input,
        axis_name=self._axis_names[axis],
        split_axis=all_to_all_dim,
        concat_axis=all_to_all_dim,
        tiled=True)
    if need_avoid_all_to_all_padding:
      gathered_input = jnp.transpose(gathered_input, [0, 2, 1])
    gather_adjust_op = self._get_gather_adjust_op(n, split_count, local_group_l,
                                                  axis)
    gathered_input = jnp.einsum(
        self._shuffle_eqs[axis],
        gathered_input,
        jnp.float32(gather_adjust_op),
        precision=lax.Precision.HIGHEST)
    gathered_input = jnp.transpose(gathered_input, self._fft_perms[axis])
    local_fft = (
        jnp.fft.ifft(jnp.complex64(gathered_input)) if inverse else jnp.fft.fft(
            jnp.complex64(gathered_input)))
    local_fft = jnp.transpose(local_fft, self._fft_perms[axis])

    corr_factor = self._get_correction_factor(n, split_count, axis, inverse)
    adjusted_local_fft = local_fft * corr_factor

    LoopVars = collections.namedtuple('LoopVars', [
        'dest_fft', 'dest_core_position', 'source_fft', 'source_core_position',
        'i'
    ])

    def cond(v):
      return v.i < self._partition[axis]

    def body(v):
      inverse_factor = -1.0 if inverse else 1.0
      factor = self._exp_itheta(-2.0 * np.pi * inverse_factor *
                                v.dest_core_position * v.source_core_position /
                                split_count)
      dest_fft = v.dest_fft + factor * v.source_fft
      source_core_position = lax.ppermute(
          x=v.source_core_position,
          axis_name=self._axis_names[axis],
          perm=self._source_target_pairs[axis])
      source_fft = lax.ppermute(
          x=v.source_fft,
          axis_name=self._axis_names[axis],
          perm=self._source_target_pairs[axis])
      i = v.i + 1
      return LoopVars(dest_fft, dest_core_position, source_fft,
                      source_core_position, i)

    source_core_position = lax.axis_index(self._axis_names[axis])
    dest_core_position = source_core_position
    source_fft = adjusted_local_fft
    dest_fft = jnp.zeros_like(source_fft)

    i = 0
    dest_fft, _, _, _, _ = lax.while_loop(
        cond,
        body,
        init_val=LoopVars(dest_fft, dest_core_position, source_fft,
                          source_core_position, i))

    padding = [[0, 0], [0, 0], [0, 0]]
    padding[axis] = [margin_width, margin_width]
    return jnp.pad(dest_fft, padding)
