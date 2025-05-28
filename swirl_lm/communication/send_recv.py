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

"""A library for communicating information across replicas.

In an example of 4 replicas, with each replica has data with different sizes as:
  replica 0: data = tf.constant([])
  replica 1: data = tf.constant([1])
  replica 2: data = tf.constant([2, 2])
  replica 3: data = tf.constant([3, 3, 3])
If data is shared in an order of 0 -> 1 -> 2 -> 3, the corresponding
`source_dest_pairs` is [[0, 1], [1, 2], [2, 3]]. With a buffer size `n_max = 3`,
calling `send_recv(data, source_dest_pairs, n_max)` provides the following:
  replica 0: tf.constant([0, 0, 0])
  replica 1: tf.constant([])
  replica 2: tf.constant([1])
  replica 3: tf.constant([2, 2]).

Note that in the example above, the `source_dest_pairs` can be obtained by
calling `source_dest_pairs_along_dim(np.array([[[0]], [[1]], [[2]], [[3]]]), 0,
True, False)`,
or `source_dest_pairs_along_dim(np.array([[[0]], [[1]], [[2]], [[3]]])
*parse_dim('+x'))`.
"""

import re

import numpy as np
import tensorflow as tf


def parse_dim(dim_info: str) -> tuple[int, bool, bool]:
  """Parses a dimension string into a tuple (dim, forward, periodic).

  Args:
    dim_info: A string that has a structure '[-+][xyz]p?$'. The first character
      is '-' or '+', which indicates the negative or positive direction,
      respectively. The second character is one of 'x', 'y', and 'z', which
      corresponds to dimension 0, 1, and 2, respectively. The optional last
      character is 'p', which suggests the dimension is periodic if present.

  Returns:
    A 3-element tuple, with the first element being the dimension, the second
    indicating whether the dimension is along the positive direction, and the
    third indicating whether the dimension is periodic.

  Raises:
    ValueError if `dim_info` does not match '[-+][xyz]p?$'.
  """
  m = re.fullmatch(r'([-+])([xyz])(p?)', dim_info)
  if m is None:
    raise ValueError(
        f'{dim_info} does not conform with the string structure for dimension'
        ' info ("[-+][xyz]p?$").'
    )

  dim = 'xyz'.index(m.group(2))
  forward = m.group(1) == '+'
  periodic = m.group(3) == 'p'

  return dim, forward, periodic


def source_dest_pairs_along_dim(
    replicas: np.ndarray, dim: int, forward: bool, periodic: bool
) -> np.ndarray:
  """Generates a 2-D array of source-target pairs along `dim` in the topology.

  Args:
    replicas: A 3-D tensor representing the topology of the partitions.
    dim: The dimension of communication. Should be one of 0, 1, and 2.
    forward: A boolean argument that indicates sending data from replicas with
      lower indices to higher indices along the positive direction of the
      topology. If it is `False`, communication in performed the opposite
      direction, i.e. from the higher indices to lower indices.
    periodic: An indicator of whether the topology is periodic. When using the
      `source_dest_pairs` generated with this function, if `periodic` is
      `True`, data from the last replica along `dim` will be send to the first
      replica; otherwise the first replica returns all zeros with the same size
      as the input. The first and last replica follows the direction specified
      in `dim`.

  Returns:
    A 2-D array of size `[num_pairs, 2]`, with the columns being the
    `replica_id` of the senders and the receivers, respectively.
  """
  rolled = np.roll(replicas, -1 if forward else 1, axis=dim)
  trim = slice(
      None if periodic or forward else 1,
      None if periodic or not forward else -1,
  )
  stacked = np.moveaxis(np.stack([replicas, rolled]), dim + 1, 1)[:, trim]
  return np.reshape(stacked, (2, -1)).T


def send_recv(
    data: tf.Tensor, source_dest_pairs: np.ndarray, n_max: int
) -> tf.Tensor:
  """Exchanges N-D `tf.Tensor`s across a list of (sender, receiver) pairs.

  Args:
    data: The n-dimensional tensor to be sent to a different replica. Dimension
      0 of this tensor can have different sizes across replicas
    source_dest_pairs: A 2-D numpy array of shape `[num_replicas, 2]`, with the
      first column being the senders' `replica_id`, and the second one being the
      receiver's `replica_id`.
    n_max: The buffer size for the communication. It has to be greater or equal
      to the maximum number of `data.shape[0]` across all replicas, otherwise a
      runtime error will occur while padding the buffer for communication.

  Returns:
    An N-D tensor received from the sender replica specified in
    `source_dest_pairs`.
  """
  # Because `CollectivePermute` permits transferring data that has the same
  # shape across all replicas only, we need to pad the input data to satisfy
  # this condition.
  static_shape = data.get_shape()
  u = tf.scatter_nd(
      tf.range(tf.shape(data)[0])[:, tf.newaxis],
      data,
      (n_max, *static_shape[1:]),
  )

  n_received = tf.raw_ops.CollectivePermute(
      input=tf.shape(data)[0], source_target_pairs=source_dest_pairs
  )
  w = tf.raw_ops.CollectivePermute(
      input=u, source_target_pairs=source_dest_pairs
  )
  # Here we trim the padded data back to its original size.
  return tf.gather_nd(w, tf.where(tf.range(n_max) < n_received))
