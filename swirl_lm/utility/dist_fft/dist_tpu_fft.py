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

"""Distributed FFT on TPU.

"""

from typing import Sequence

import numpy as np
from swirl_lm.utility import types
import tensorflow.compat.v1 as tf

_CTYPE = types.TF_COMPLEX_DTYPE
_DTYPE = types.TF_DTYPE


class DistTPUFFT(object):
  """Distributed FFT on TPU."""

  def __init__(self, replicas: np.ndarray, replica_id: tf.Tensor) -> None:
    """Constructor.

    This class performs distributed FFT on TPU. Each class object is constructed
    with the global replica map, `replicas`, and the `replica_id` of the current
    core. `replicas` contains the replica id of each core at a given logical TPU
    core mesh coordinate. For example, assuming our compute shape [x, y, z] =
    [2, 4, 2], replicas[0, 3, 0] will be the core replica id for the TPU core
    that is logically at [0, 3, 0] position and so on.

    `transform_1d`, `transform_2d` and `transform_3d` are used to perfrom
    distributed FFT (forward and inverse) on TPU for 1D, 2D and 3D.

    Args:
      replicas: A 3D numpy array representing the mapping from the core replica
        coordinate to the `replica_id`. The number of cores in each dimension is
        the number of splits of the global input for the transformation.
      replica_id: A scalar integer tf.Tensor representing the `replica_id` of
        the current core.

    Raises:
      ValueError: If `replicas` isn't a 3d array.
    """

    if replicas.ndim != 3:
      raise ValueError('`replicas` must be a 3d array mapping core replica '
                       'coordinate to `replica_id`, but a %dd array is given.' %
                       replicas.ndim)

    self._replicas = replicas
    self._replica_id = replica_id
    self._shuffle_eqs = ['ijk,il->ljk', 'ijk,jl->ilk', 'ijk,kl->ijl']
    self._fft_perms = [[2, 1, 0], [0, 2, 1], [0, 1, 2]]
    self._replica_groups = []
    self._source_target_pairs = []
    for d in range(3):
      replica_groups = []
      source_target_pairs = []
      indices = [0, 0, 0]
      indices[d] = slice(0, replicas.shape[d])
      for i in range(replicas.shape[(d + 1) % 3]):
        for j in range(replicas.shape[(d + 2) % 3]):
          indices[(d + 1) % 3] = i
          indices[(d + 2) % 3] = j
          group_replicas = replicas[tuple(indices)].tolist()
          replica_groups.append(group_replicas)
          for c in range(replicas.shape[d]):
            source_target_pairs.append(
                (group_replicas[c],
                 group_replicas[(c + 1) % replicas.shape[d]]))
      self._replica_groups.append(replica_groups)
      self._source_target_pairs.append(source_target_pairs)

  @property
  def replicas(self) -> np.ndarray:
    return self._replicas

  @property
  def replica_id(self) -> tf.Tensor:
    return self._replica_id

  def _get_shuffle_op(self,
                      n: int,
                      m: int,
                      l: int,
                      transform_dim: int,
                      halo_width: int = 0) -> tf.Tensor:
    """Creates a shuffle operator for rearrange the elements on each core.

    This creates a 2D matrix that is used to contract the 3D partial input
    tensor in the `transform_dim` to shuffle all elements into `m` segments.
    The way the shuffling is performed is: in `transform_dim`, lining up all
    elements globally in the original input order, and label each from 0 to m-1
    cyclically, and each element labelled `k` will be shuffled to the
    corresponding k'th semegnt on each core. Note that on each core, the length
    in the `transform_dim` `n` is not always divisble by `m`, so some of the
    segments will have missing elements. These missing elements are filled with
    0. The matrix is then padded with 0 on top and bottom with 0 for
    `halo_width`.

    For a concrete example, we consider a full input (without halos) with total
    length 12 (not including halos) in the `transform_dim`. We further assume
    the input is distributed over 3 TPU cores and `halo_width` to be 1. We have
    the following distribution (where `H` represets halo):

      replica id:   |         0        |         1        |        2         |
      element id:   | H  0  1  2  3  H | H  4  5  6  7  H | H  8  9 10 11  H |
      label:        | H  0  1  2  0  H | H  1  2  0  1  H | H  2  0  1  2  H |

    The desired shuffled result should be:

      replica id:   |         0        |         1        |        2         |
      element id:   | 0  3  1  x  2  x | 6  x  4  7  5  x | 9  x 10  x  8 11 |
      label:        | 0  0  1  1  2  2 | 0  0  1  1  2  2 | 0  0  1  1  2  2 |

    Note that on each core, the elements are aggregated according to their
    respective label. The `x` is dummy filling -- which we can fill with 0. Also
    note that values in halo are not used.

    The desired transformation can be done through the contraction `ij,jk->ik`
    with the following `jk` matrix:

      replica id:   |         0        |         1        |        2         |

                    | 0  0  0  0  0  0 | 0  0  0  0  0  0 | 0  0  0  0  0  0 |
                    | 1  0  0  0  0  0 | 0  0  1  0  0  0 | 0  0  0  0  1  0 |
      matrix:       | 0  0  1  0  0  0 | 0  0  0  0  1  0 | 1  0  0  0  0  0 |
                    | 0  0  0  0  1  0 | 1  0  0  0  0  0 | 0  0  1  0  0  0 |
                    | 0  1  0  0  0  0 | 0  0  0  1  0  0 | 0  0  0  0  0  1 |
                    | 0  0  0  0  0  0 | 0  0  0  0  0  0 | 0  0  0  0  0  0 |

    The function below generates the corresponding matrix that is needed.

    Args:
      n: The length of the partial input in the `transform_dim`, excluding
        halos.
      m: The number of core (or division) to split distribute the input in the
        `transform_dim`.
      l: The size of the segment of each of the `m` segments. This number is
        ceil(n / m).
      transform_dim: The dimension the transformation is to be performed: can be
        0, 1, or 2.
      halo_width: The width of the halo.

    Returns:
      A 2D matrix of the shape [n + 2 * halo_width, m * l] to be used to
      contract the partial input as demonstrated in the example above.
    """
    position_index = tf.cast(
        self._get_core_position_index(transform_dim), dtype=tf.int32)
    raw_indices = tf.range(n, dtype=tf.int32)
    shift_indices = tf.math.floormod(position_index * n, m) + raw_indices
    indices = (raw_indices // m) + tf.math.floormod(shift_indices, m) * l
    return tf.pad(
        tf.one_hot(indices, depth=m * l, dtype=_CTYPE),
        [[halo_width, halo_width], [0, 0]])

  def _get_gather_adjust_op(self,
                            n: int,
                            m: int,
                            l: int,
                            transform_dim: int) -> tf.Tensor:
    """Creates a 2D matrixs that remove the padding in the gathered input.

    This is used to contract the partial input in the `transform_dim` to remove
    possible padding before performing local transform.

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

                    | 1  0  0  0 | 1  0  0  0 | 1  0  0  0 |
                    | 0  1  0  0 | 0  0  0  0 | 0  0  0  0 |
      matrix:       | 0  0  1  0 | 0  1  0  0 | 0  1  0  0 |
                    | 0  0  0  0 | 0  0  1  0 | 0  0  0  0 |
                    | 0  0  0  1 | 0  0  0  1 | 0  0  1  0 |
                    | 0  0  0  0 | 0  0  0  0 | 0  0  0  1 |

    The method here generates the needed matrices before the local FFT can be
    carried out.

    Args:
      n: The length of the partial input in the `transform_dim`, excluding
        halos.
      m: The number of core (or division) to split distribute the input in the
        `transform_dim`.
      l: The size of the segment of each of the `m` segments. This number if
        ceil(n / m).
      transform_dim: The dimension the transformation is to be performed: can be
        0, 1, or 2.

    Returns:
      A 2D matrix of the shape [m * l, n] to be used to contract the gathered
      partial input as demonstrated in the example above.
    """

    position_index = tf.cast(
        self._get_core_position_index(transform_dim), dtype=tf.int32)
    pos_indices = tf.range(m * l, dtype=tf.int32)
    source_position = pos_indices // l
    source_order = tf.math.floormod(pos_indices, l)
    source_start = tf.math.floormod(source_position * n, m)
    adjustment = tf.where(
        tf.less_equal(source_start, position_index),
        tf.zeros([m * l], dtype=tf.int32), m * tf.ones([m * l], dtype=tf.int32))
    mask = tf.where(
        tf.less(source_order * m + position_index - source_start + adjustment,
                n), tf.ones([m * l]), tf.zeros([m * l]))
    upper_triangular = (
        tf.ones([m * l, m * l], dtype=_DTYPE) -
        tf.matrix_band_part(tf.ones([m * l, m * l], dtype=_DTYPE), -1, 0))
    indices = tf.cast(
        tf.einsum('i,ij->j', mask, upper_triangular), dtype=tf.int32)
    indices = tf.where(mask > 0, indices, -1 * tf.ones([m * l], dtype=tf.int32))
    return tf.one_hot(indices, depth=n, dtype=_CTYPE)

  def _get_core_position_index(self, transform_dim: int):
    """Gets the relative position of the core in `transform_dim`.

    Returns a scalar tensor that represents the relative position in
    `transform_dim` of the current TPU core replica in the logical 3D mesh
    coordinates. For example, if the logical mesh is [2, 2, 2] with the
    following mapping from coordinate to core replica id:

        [[[0, 3], [1, 2]],
         [[4, 7], [6, 5]]]

    and if the current core has replica id = 6, its core position in dimension 0
    is 1, it is 1 in dimension 1, and it is 0 in dimension 2.

    Args:
      transform_dim: The dimension to perform transformation. This must be one
        on 0, 1, or 2.

    Returns:
      A scalar tensor representing the position of the current core in the
      global logical TPU core mesh in the `transform_dim` dimension.
    """
    # Note, if the emelemts within einsum are casted to integer, a weird XLA
    # compilation will be triggered, complaining about same dimension appearing
    # in convoultion twice.
    return tf.einsum(
        'i,i->',
        tf.cast(
            tf.where(tf.equal(self._replicas, self._replica_id))[0],
            dtype=_DTYPE), tf.one_hot(transform_dim, 3))

  def _get_correction_factor(self, n: int, m: int, transform_dim: int,
                             inverse: bool) -> tf.Tensor:
    """Get the phase correction factors to adjust the local transform results.

    Args:
      n: The length of the partial input in the `transform_dim`, excluding
        halos.
      m: The number of core (or division) to split distribute the input in the
        `transform_dim`.
      transform_dim: The dimension the transformation is to be performed: can be
        0, 1, or 2.
      inverse: Boolean. Whether the transformation is an inverse transformation.

    Returns:
      A 1D complex tensor of length `n` with the correction factors.
    """
    inverse_factor = -1.0 if inverse else 1.0
    inverse_amp = 1.0 / m if inverse else 1.0
    position_index = self._get_core_position_index(transform_dim)

    factor = tf.math.exp(-2.0j * np.pi * tf.cast(position_index, _CTYPE) *
                         inverse_factor * tf.cast(tf.range(n), dtype=_CTYPE) /
                         (n * m)) * inverse_amp
    for _ in range(2 - transform_dim):
      factor = tf.expand_dims(factor, -1)
    return factor

  def transform_1d(self,
                   partial_input: tf.Tensor,
                   transform_dim: int,
                   halo_width: int = 0,
                   inverse: bool = False) -> tf.Tensor:
    """Performs a 1d Fourier transform or inverse Fourier transform.


    Args:
      partial_input: This represents the partial input handled by the
        corresponding core replica. It is expected to be a 3D Tensor.
      transform_dim: The dimension to perform the transformation. Must be one of
        0, 1, or 2.
      halo_width: The width of the halo for the dimension to be transformed.
      inverse: A boolean indicating whether the transform is to be done as an
        inverse transform.

    Returns:
      A 3D tensor with the same shape as input but with the dimension
      `transform_dim` transformed to the global Fourier transform (or inverse
      transform), with the original input halo width. Note, after this
      transform, the halo does not contain the correct values. The caller needs
      to perform the needed halo exchange operations if correct values are
      needed.

    Raises:
      ValueEorror: (1) If `transform_dim` is invalid. (2) `halo_width` is
      invalid.
    """
    # Validates input parameters.
    input_shape = partial_input.shape.as_list()
    if len(input_shape) != 3:
      raise ValueError('Unsupported input shape %s. This currently only '
                       'supports 3D Tensor input.' % str(input_shape))
    supported_dims = [0, 1, 2]
    if transform_dim not in supported_dims:
      raise ValueError('Unsupported `transform_dim`: %d. This currently only '
                       'supports transform along dimensions %s' %
                       (transform_dim, str(supported_dims)))
    if halo_width < 0:
      raise ValueError('Negative `halo_width` is not supported.')

    split_count = self._replicas.shape[transform_dim]
    local_group_l = int(
        np.ceil((input_shape[transform_dim] - 2 * halo_width) / split_count))

    # Forming the ops that can be used in einsum to shuffle the input.
    n = (input_shape[transform_dim] - 2 * halo_width)
    shuffle_op = self._get_shuffle_op(n, split_count, local_group_l,
                                      transform_dim, halo_width)

    shuffled_input = tf.einsum(self._shuffle_eqs[transform_dim], partial_input,
                               shuffle_op)
    all_to_all_dim = transform_dim
    # This is to handle an issue associated with AllToAll. If the split is done
    # on the inner-most (the last) dimension, the length in that dimension
    # to be 128x after the split. If not, the wasteful padding occurs. We
    # mitigate this issue by transposing the inner-most dimension to an outer
    # dimension.
    need_avoid_all_to_all_padding = (transform_dim == 2 and
                                     (local_group_l % 128) != 0)
    if need_avoid_all_to_all_padding:
      shuffled_input = tf.transpose(shuffled_input, [0, 2, 1])
      all_to_all_dim = 1
    gathered_input = tf.raw_ops.AllToAll(
        input=shuffled_input,
        split_dimension=all_to_all_dim,
        concat_dimension=all_to_all_dim,
        split_count=split_count,
        group_assignment=self._replica_groups[transform_dim],
        name='fft-gather-input')
    if need_avoid_all_to_all_padding:
      gathered_input = tf.transpose(gathered_input, [0, 2, 1])

    gather_adjust_op = self._get_gather_adjust_op(n, split_count, local_group_l,
                                                  transform_dim)
    gathered_input = tf.einsum(self._shuffle_eqs[transform_dim],
                               gathered_input, gather_adjust_op)
    gathered_input = tf.transpose(gathered_input,
                                  self._fft_perms[transform_dim])
    local_transform = (
        tf.signal.ifft(gathered_input)
        if inverse else tf.signal.fft(gathered_input))
    local_transform = tf.transpose(local_transform,
                                   self._fft_perms[transform_dim])

    adjusted_local_transform = local_transform * self._get_correction_factor(
        n, split_count, transform_dim, inverse)

    def cond(dest_transform, dest_core_position, source_transform,
             source_core_position, i):
      del (dest_transform, dest_core_position, source_transform,
           source_core_position)
      return i < self._replicas.shape[transform_dim]

    def body(dest_transform, dest_core_position, source_transform,
             source_core_position, i):
      inverse_factor = -1.0 if inverse else 1.0
      factor = tf.math.exp(-2.0j * np.pi * inverse_factor *
                           tf.cast(dest_core_position, _CTYPE) *
                           tf.cast(source_core_position, _CTYPE) / split_count)
      dest_transform += factor * source_transform
      source_core_position = tf.raw_ops.CollectivePermute(
          input=source_core_position,
          source_target_pairs=self._source_target_pairs[transform_dim],
          name='source_core_position_permute')
      source_transform = tf.raw_ops.CollectivePermute(
          input=source_transform,
          source_target_pairs=self._source_target_pairs[transform_dim],
          name='source_transform_permute')
      i += 1
      return (dest_transform, dest_core_position, source_transform,
              source_core_position, i)

    source_core_position = self._get_core_position_index(transform_dim)
    dest_core_position = source_core_position
    source_transform = adjusted_local_transform
    dest_transform = tf.zeros_like(source_transform)
    i = 0
    dest_transform, _, _, _, _ = tf.while_loop(
        cond,
        body,
        loop_vars=(dest_transform, dest_core_position, source_transform,
                   source_core_position, i),
        return_same_structure=True,
        back_prop=False)

    padding = [[0, 0], [0, 0], [0, 0]]
    padding[transform_dim] = [halo_width, halo_width]
    return tf.pad(dest_transform, padding)

  def transform_2d(self,
                   partial_input: tf.Tensor,
                   transform_dims: Sequence[int],
                   halo_widths: Sequence[int] = 0,
                   inverse: bool = False) -> tf.Tensor:  # pytype: disable=annotation-type-mismatch
    """Performs a 2d Fourier transform or inverse Fourier transform.


    Args:
      partial_input: This represents the partial input handled by the
        corresponding core replica. It is expected to be a 3D Tensor.
      transform_dims: The dimensions to perform the transformation. Must be a
        sequence of 2 integer with value of 0, 1, or 2.
      halo_widths: The width of the halos for the input corresponding to the
        `transform_dims`.
      inverse: A boolean indicating whether the transform is to be done as an
        inverse transform.

    Returns:
      A 3D tensor with the same shape as input but with the dimensions
      `transform_dims` transformed to the global Fourier transform (or inverse
      transform), with the original input halo widths. Note, after this
      transform, the halos do not contain the correct values. The caller needs
      to perform the needed halo exchange operations if the correct values are
      needed.

    Raises:
      ValueEorror: (1) The `transform_dims` is invalid. (2) `halo_widths` is
      invalid.
    """
    if len(transform_dims) != 2 or len(halo_widths) != 2:
      raise ValueError('`transform_dims` and `halo_widths` must both have 2 '
                       'elements.')
    if transform_dims[0] == transform_dims[1]:
      raise ValueError('2 dimensions of the transform must be different.')

    result = partial_input
    for d, h in zip(transform_dims, halo_widths):
      result = self.transform_1d(result, d, h, inverse)
    return result

  def transform_3d(self,
                   partial_input: tf.Tensor,
                   halo_widths: Sequence[int] = (0, 0, 0),
                   inverse: bool = False) -> tf.Tensor:
    """Performs a 3d Fourier transform or inverse Fourier transform.


    Args:
      partial_input: This represents the partial input handled by the
        corresponding core replica. It is expected to be a 3D Tensor.
      halo_widths: The width of the halo for the dimension 0, 1 and 2, in this
        order.
      inverse: A boolean indicating whether the transform is to be done as an
        inverse transform.

    Returns:
      A 3D tensor with the same shape as input but with all 3 dimensions
      transformed to the global Fourier transform (or inverse transform), with
      the original input halo widths. Note, after this transform, the halos do
      not contain the correct values. The caller needs to perform the needed
      halo exchange operations if correct values are needed.

    Raises:
      ValueError: (1) The `transform_dims` is invalid. (2) `halo_widths` is
      invalid.
    """
    if len(halo_widths) != 3:
      raise ValueError('`halo_widths` must have 3 elements.')

    result = partial_input
    for d, h in enumerate(halo_widths):
      result = self.transform_1d(result, d, h, inverse)
    return result
