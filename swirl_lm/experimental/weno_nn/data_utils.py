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

"""Data utilities."""

from collections.abc import Iterable
import itertools
import os
import pickle
from typing import Any, SupportsIndex

import grain.python as pygrain
import numpy as np
import tensorflow as tf


_KEYS_WENO3 = (
    'u_i_minus_one',
    'u_i',
    'u_i_plus_one',
    'u_i_plus_half',
)


class CustomDataSource:
  """A data source that stores the cell averages and the interface values."""

  def __init__(
      self,
      u_bar: np.ndarray,
      u_half_p: np.ndarray,
  ):
    """Data source constructor.

    Args:
      u_bar: An array containing the cell averages of the approximation
        solution. The dimension should be (num_samples, 3).
      u_half_p: Interface values of the approximation solution. The dimension
        should be (num_samples,).
    """
    if u_bar.shape[-1] != 3:
      raise ValueError(
          f'u_bar has {u_bar.shape[-1]} columns, but it should have 3.'
      )
    if u_half_p.shape[0] != u_bar.shape[0]:
      raise ValueError(
          'data should have the same number of training samples.'
          f'Instead we have {u_bar.shape[0]} samples in u_bar and '
          f'{u_half_p.shape[0]} samples in u_half_p.'
      )
    self._u_bar = u_bar
    self._u_half_p = u_half_p
    self._len = u_bar.shape[0]

  def __len__(self):
    return self._len

  def __getitem__(self, record_key: SupportsIndex) -> dict[str, np.ndarray]:
    """Returns the data record for a given key."""
    item = {}
    item['u_bar'] = self._u_bar[record_key]
    item['u_half_p'] = self._u_half_p[record_key]

    return item


def create_loader_from_pickle(
    batch_size: int,
    dataset_paths: tuple[str, ...],
    num_epochs: int,
    seed: int,
    worker_count: int = 0,
    drop_remainder: bool = True,
) -> pygrain.DataLoader:
  """Loads pre-computed trajectories stored in a pickle file.

  Arguments:
    batch_size: Batch size returned by dataloader. If set to -1, use entire
      dataset size as batch_size.
    dataset_paths: Tuple of absolute paths to dataset files.
    num_epochs: Number of epochs to iterate over the dataset.
    seed: Random seed to be used in data sampling.
    worker_count: Number of workers to use for data loading.
    drop_remainder: Whether to drop the last batch if it is smaller than
      batch_size.

  Returns:
    Dataloader object containing the data.
  """

  keys = _KEYS_WENO3

  data_dict = {key: [] for key in keys}

  for pickle_file in dataset_paths:
    data_temp = load_single_analytical_dataset(pickle_file, keys)
    for key in keys:
      data_dict[key].extend(data_temp[key])

  u_bar = np.stack(
      [
          np.array(data_dict['u_i_minus_one']),
          np.array(data_dict['u_i']),
          np.array(data_dict['u_i_plus_one']),
      ],
      axis=-1,
  )
  u_half_p = np.array(data_dict['u_i_plus_half'])

  source = CustomDataSource(u_bar=u_bar, u_half_p=u_half_p)

  data_loader = pygrain.load(
      source=source,
      num_epochs=num_epochs,
      shuffle=True,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=[],
      batch_size=batch_size,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )

  return data_loader


def create_loader_from_pickle_eval(
    dataset_main_dir: str,
    func_types: tuple[str, ...],
    noise_sigmas: tuple[str, ...],
    nx_values: tuple[int, ...],
) -> Iterable[Any]:
  """Load pre-computed pickle files for evaluation.

  Arguments:
    dataset_main_dir: Address of the directory.
    func_types: Keys of the different evaluation functions from which the
      samples have been extracted.
    noise_sigmas: Different levels of noise used to corrupt the samples.
    nx_values: Number of discretization points in which each sample was
      computed.

  Returns:
    Iterable object containing the evaluation data. In this case, we repeat the
    same data. The data set consist of a dictionaty with samples of a
    collection of functions.
  """

  keys = (
      'u_i_minus_one',
      'u_i',
      'u_i_plus_one',
      'u_i_plus_half',
      'expected_weno_wt',
  )

  data_dict = {}
  for func_type in func_types:
    data_dict[func_type] = {}
    for noise_sigma in noise_sigmas:
      data_dict[func_type][noise_sigma] = {}
      for nx in nx_values:
        data_dict[func_type][noise_sigma][nx] = {}
        filename = os.path.join(
            dataset_main_dir, noise_sigma, f'{func_type}_nx_{nx}.pickle'
        )
        data_temp = load_single_analytical_dataset(filename, keys)
        data_dict[func_type][noise_sigma][nx]['u_bar'] = np.stack(
            [
                np.array(data_temp['u_i_minus_one']),
                np.array(data_temp['u_i']),
                np.array(data_temp['u_i_plus_one']),
            ],
            axis=-1,
        )
        data_dict[func_type][noise_sigma][nx]['u_half_p'] = data_temp[
            'u_i_plus_half'
        ]
        data_dict[func_type][noise_sigma][nx]['expected_weno_wt'] = data_temp[
            'expected_weno_wt'
        ]
  return itertools.repeat(data_dict)


def load_single_analytical_dataset(
    file_path: str, keys: tuple[str, ...]
) -> dict[str, np.ndarray]:
  """Loads single analytical dataset from CNS in pickle.

  Args:
    file_path: Path to pickle file on CNS.
    keys: Keys to be extracted in the target dictionary.

  Returns:
    Values at 'u_i_minus_one', 'u_i', 'u_i_plus_one' and 'u_i_plus_half'.
  """
  with tf.io.gfile.Open(file_path, 'rb') as fpt:
    raw_data_dict = pickle.load(fpt)['data_list']

  def _single_var_func(key):
    """Concatenates the lists with the same key in the list of dictionaries."""
    var = []
    for data_dict in raw_data_dict:
      var.extend(data_dict[key].ravel())
    return np.array(var)

  merged_data = {}
  for key in keys:
    merged_data[key] = _single_var_func(key)
  return merged_data
