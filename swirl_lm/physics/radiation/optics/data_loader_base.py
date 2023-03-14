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

# Copyright 2023 Google LLC
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
"""A base loader class for the RRTMGP lookup tables."""

import abc
import dataclasses
import os
import tempfile
from typing import Sequence, Tuple
import zipfile

import netCDF4 as nc
import numpy as np
from swirl_lm.utility import types
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class DataLoaderBase(metaclass=abc.ABCMeta):
  """Generic data loader that reads a .nc file and constructs TF tensors."""

  @classmethod
  def _create_index(cls, name_arr: np.ndarray):
    """Utility function for generating an index from a sequence of names."""
    size = name_arr.shape[0]
    idx = {}
    for i in range(size):
      name = ''.join([g_i.decode('utf-8') for g_i in name_arr[i]]).strip()
      idx[name] = i
    return idx

  @classmethod
  def _parse_nc_file(
      cls,
      path: str,
      exclude_vars: Sequence[str] = (),
  ) -> Tuple[nc.Dataset, types.VariableMap, types.DimensionMap]:
    """Utility function for unpacking the RRTMGP files and loading tensors.

    Note that the lookup tables are loaded using `tf.Variable` in order to
    prevent constant folding, which would easily cause the Tensorflow graph to
    exceed the Protobuf hard limit of 2GB in a distributed setting.

    Args:
      path: Full path of a zipped netCDF dataset file.
      exclude_vars: Names of variables that should be skipped.

    Returns:
      A 3-tuple of 1) the original netCDF Dataset, 2) a dictionary containing
      the data as tf.Variable, and 3) a dictionary of dimensions.
    """
    filename = os.path.basename(path)
    with tempfile.TemporaryDirectory() as tmp_dir:
      with zipfile.ZipFile(path, 'r') as zip_file:
        zip_file.extractall(tmp_dir)
        filename = os.path.splitext(filename)[0]
        extracted_path = os.path.join(tmp_dir, filename + '.nc')
      ds = nc.Dataset(extracted_path, 'r')

    tensor_dict = {}
    dim_map = {k: v.size for k, v in ds.dimensions.items()}

    for key in ds.variables:
      val = ds[key][:].data
      if key in exclude_vars:
        continue
      if val.dtype == np.float64:
        val = val.astype(np.float32)
      # Create a tf.Variable that delays materializing the content until graph
      # execution.
      tensor_dict.update({key: tf.Variable(lambda: val, shape=val.shape)})  # pylint: disable=cell-var-from-loop
    return (ds, tensor_dict, dim_map)

  @classmethod
  @abc.abstractmethod
  def from_nc_file(
      cls,
      path: str,
      **kwargs,
  ) -> 'DataLoaderBase':
    """Loads lookup tables from NetCDF files and populates the attributes."""
