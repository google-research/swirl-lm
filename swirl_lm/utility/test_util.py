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

"""Library for common operations used in tests."""

import os
from typing import Any

from absl.testing import absltest
import numpy as np
import numpy.typing as npt
from swirl_lm.base import parameters
from swirl_lm.base import parameters_pb2
from swirl_lm.utility import common_ops
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import grid_parametrization_pb2
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


def get_split_inputs_3d_tensor(
    u_full,
    v_full,
    w_full,
    replicas,
    halos,
) -> list[list[tf.Tensor]]:
  """Creates split inputs from full field components with halos added.

  This is the same as `get_split_inputs`, but outputs 3D tensors instead of
  lists of 2D tensors.

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
    to it. The output format is a list of list of 3D tensors, where output[i][j]
    gets the ith replica, and u, v, w for j=0, 1, 2.
  """
  split_inputs = get_split_inputs(u_full, v_full, w_full, replicas, halos)
  # Stack the output to return 3D tensors instead of lists of 2D tensors.
  return [
      [tf.stack(split_inputs[i][j]) for j in (0, 1, 2)]
      for i in range(len(split_inputs))
  ]


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


def extract_1d_slice_in_dim(
    f_3d: tf.Tensor, dim: int, other_idx: int
) -> tf.Tensor:
  """Extracts 1D slice of `f_3d` along `dim`.

  For example, if dim == 1 and other_idx == 4, return f[4, :, 4].

  Args:
    f_3d: The 3D tensor to extract the slice from.
    dim: The dimension along which to extract the slice.
    other_idx: The other indices to extract the slice from.

  Returns:
    The 1D slice of `f_3d` along `dim`.
  """
  observe_slice = [other_idx, other_idx, other_idx]
  observe_slice[dim] = slice(None)
  f_1d_slice = f_3d[tuple(observe_slice)]
  return f_1d_slice


def convert_to_3d_tensor_and_tile(
    f_1d: tf.Tensor, dim: int, num_repeats: int
) -> tf.Tensor:
  """Converts 1D tensor `f_1d` to a tiled 3D tensor.

  For example, if len(f_1d) == 8, dim == 1, and num_repeats == 4, then
  f.shape = (4, 8, 4).

  Args:
    f_1d: The 1D tensor to convert to 3D.
    dim: The dimension along which `f_1d` is laid out.
    num_repeats: The number of times to repeat the tensor in dimensions other
      than `dim`.

  Returns:
    The 3D tensor `f_3d`, where f_3d.shape[dim] == len(f_1d), and
      f_3d.shape[j] == num_repeats for j != dim.
  """
  # Convert f_1d to 3D tensor f where the direction of variation is along
  # dim. Result: f.shape[dim] = len(f_1d), and f.shape[j] = 1 for j != dim.
  slices = [tf.newaxis, tf.newaxis, tf.newaxis]
  slices[dim] = slice(None)
  f = f_1d[tuple(slices)]

  # Tile f to a 3D tensor that is repeated in other dimensions.
  repeats = [num_repeats, num_repeats, num_repeats]
  repeats[dim] = 1
  f_3d = tf.tile(f, multiples=repeats)
  return f_3d


def save_1d_array_to_tempfile(
    test: absltest.TestCase, array: tf.Tensor
) -> str:
  """Saves a 1D array to a tempfile and returns the path to the tempfile."""
  tempfile = test.create_tempfile()
  fname = os.path.join(tempfile)
  np.savetxt(fname, array)
  return fname


def get_np_array(
    state: Any,
    varname: str,
    computation_shape: npt.ArrayLike,
    halo_width: int,
) -> npt.NDArray:
  """Gets the numpy array of a particular variable."""
  if np.prod(computation_shape) == 1:
    result = state[varname].numpy()[
        halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
    ]
  else:
    raise NotImplementedError('Multiple cores not yet implemented.')
  return result


def l_infinity_norm(v: npt.ArrayLike) -> float:
  return np.max(np.abs(v))


def l_infinity_error(field1: npt.ArrayLike, field2: npt.ArrayLike) -> float:
  err = field1 - field2
  return l_infinity_norm(err)


def compute_power_exponent(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
  """Estimates a power-law exponent through regression.

  Assume that x, y have a power-law relationship, y = C * x^p. Estimate p
  through linear regression, using log y = log C + p log x.

  Args:
    x: Independent variable.
    y: Dependent variable.

  Returns:
    The exponent p characterizing the power law.
  """
  logy = np.log(y)
  logx = np.log(x)
  return np.polyfit(logx, logy, 1)[0]


def create_swirl_lm_params(
    config: parameters_pb2.SwirlLMParameters | str,
    grid_params: (grid_parametrization_pb2.GridParametrization | str |
                  None) = None,
) -> parameters.SwirlLMParameters:
  """Fills SwirlLMParams with backward-compatible defaults for tests.

  Do not use this function in new tests - create SwirlLMParameters using one of
  the functions in parameters.py instead. Also note that this function does not
  allow the input `config` to include the 'grid_params' message. New tests
  should set up SwirlLMParameters with 'grid_params'.

  Args:
    config: Either a SwirlLMParameters proto message or a string. If a string,
      then the string will be parsed as a text-formatted SwirlLMParameters
      proto.
    grid_params: A GridParametrization proto message, a string. If a string,
      then the string will be parsed as a text-formatted GridParametrization
      proto.

  Returns:
    A new SwirlLMParameters initialized from `config` and `grid_params`.

  """
  if grid_params is None:
    grid_params = grid_parametrization_pb2.GridParametrization()
  elif isinstance(grid_params, str):
    grid_params = grid_parametrization.params_from_text_proto(grid_params)

  if isinstance(config, str):
    config = parameters.parse_text_proto(config)

  assert not config.HasField('grid_params'), (
      'Use one of functions in parameters.py to create a SwirlLMParameters '
      'object from a SwirlLMParameters message that includes the grid_params '
      'message.')
  new_config = parameters_pb2.SwirlLMParameters()
  new_config.CopyFrom(config)
  new_config.grid_params.CopyFrom(grid_params)
  return parameters.SwirlLMParameters(new_config)


def params_from_text_proto(
    text_proto: str) -> grid_parametrization_pb2.GridParametrization:
  """Returns a GridParametrization protobuf from a text-formatted proto."""
  return grid_parametrization.params_from_text_proto(text_proto)
