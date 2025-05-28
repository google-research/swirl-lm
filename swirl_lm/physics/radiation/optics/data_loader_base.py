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
from typing import Dict, Sequence, Tuple

import netCDF4 as nc
import numpy as np
from swirl_lm.utility import types
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class DataLoaderBase(metaclass=abc.ABCMeta):
  """Generic data loader that reads a .nc file and constructs `tf.Tensor`s."""

  _NETCDF_DATA_DIR = '/tmp/netcdf/data'

  @classmethod
  def _create_index(cls, name_arr: Sequence[str]) -> Dict[str, int]:
    """Utility function for generating an index from a sequence of names."""
    return {name: idx for idx, name in enumerate(name_arr)}

  @classmethod
  def _create_local_file(cls, cns_file_path):
    """Copies remote files locally so they can be ingested by netCDF reader."""
    # Create local directory.
    if os.path.exists(cls._NETCDF_DATA_DIR):
      assert os.path.isdir(cls._NETCDF_DATA_DIR)
    else:
      os.makedirs(cls._NETCDF_DATA_DIR)
    local_filename = os.path.join(
        cls._NETCDF_DATA_DIR, os.path.basename(cns_file_path)
    )
    # Copy the file from remote location if not already present.
    if os.path.exists(local_filename):
      assert os.path.isfile(local_filename)
    else:
      with tf.io.gfile.GFile(local_filename, 'w') as local_file:
        local_file.write(tf.io.gfile.GFile(cns_file_path, mode='rb').read())
    return local_filename

  @classmethod
  def _parse_nc_file(
      cls,
      path: str,
  ) -> Tuple[nc.Dataset, types.TensorMap, types.DimensionMap]:
    """Utility function for unpacking the RRTMGP files and loading tensors.

    Args:
      path: Full path of the netCDF dataset file.

    Returns:
      A 3-tuple of 1) the original netCDF Dataset, 2) a dictionary containing
      the data as tf.Tensor, and 3) a dictionary of dimensions.
    """
    local_path = cls._create_local_file(path)
    ds = nc.Dataset(local_path, 'r')

    tensor_dict = {}
    dim_map = {k: v.size for k, v in ds.dimensions.items()}

    for key in ds.variables:
      val = ds[key][:].data
      if val.dtype == np.float64:
        val = val.astype(np.float32)
      tensor_dict.update({key: tf.constant(val)})
    return (ds, tensor_dict, dim_map)

  @classmethod
  @abc.abstractmethod
  def from_nc_file(
      cls,
      path: str,
      **kwargs,
  ) -> 'DataLoaderBase':
    """Loads lookup tables from NetCDF files and populates the attributes."""
