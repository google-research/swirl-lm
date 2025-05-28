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
"""Data class for loading and accessing cloud optics."""

import abc
import dataclasses
import enum
from typing import Any, Dict

from swirl_lm.physics.radiation.optics import data_loader_base as loader
from swirl_lm.utility import types
import tensorflow as tf


class IceRoughness(enum.Enum):
  """Defines the levels of ice roughness that index the ice lookup tables."""
  # No roughness.
  SMOOTH = 0
  # Medium roughness.
  MEDIUM = 1
  # Rough.
  ROUGH = 2


@dataclasses.dataclass(frozen=True)
class LookupCloudOptics(loader.DataLoaderBase, metaclass=abc.ABCMeta):
  """Wrapper for lookup table of cloud optical properties."""
  # Number of liquid particle sizes.
  n_size_liq: int
  # Number of ice particle sizes.
  n_size_ice: int
  # Liquid particle size lower bound for interpolation.
  radius_liq_lower: float
  # Liquid particle size upper bound for interpolation.
  radius_liq_upper: float
  # Factor for calculating interpolation for liquid particle.
  diameter_ice_lower: float
  # Ice particle size upper bound for interpolation.
  diameter_ice_upper: float

  # Lookup table for liquid extinction coefficient (`nbnd, nsize_liq`) in
  # m²/g.
  ext_liq: tf.Tensor
  # Lookup table for liquid single-scattering albedo (`nbnd, nsize_liq`).
  ssa_liq: tf.Tensor
  # Lookup table for liquid asymmetry parameter (`nbnd, nsize_liq`).
  asy_liq: tf.Tensor

  # Lookup table for ice extinction coefficient (`nrghice, nband, nsize_ice`)
  # in m²/g.
  ext_ice: tf.Tensor
  # Lookup table for ice single-scattering albedo
  # (`nrghice, nband, nsize_ice`).
  ssa_ice: tf.Tensor
  # Lookup table for ice asymmetry parameter (`nrghice, nband, nsize_ice`).
  asy_ice: tf.Tensor
  # Ice roughness.
  ice_roughness: IceRoughness = IceRoughness.MEDIUM

  @classmethod
  def _load_data(
      cls,
      tables: types.TensorMap,
      dims: types.DimensionMap,
  ) -> Dict[str, Any]:
    """Preprocesses the cloud optics data.

    Args:
      tables: The extracted lookup tables and other parameters as a dictionary.
      dims: A dictionary containing dimension information for the tables.

    Returns:
      A dictionary containing dimension information and the preprocessed RRTMGP
      data as `tf.Tensor`s.
    """
    data = {}
    data['n_size_liq'] = dims['nsize_liq']
    data['n_size_ice'] = dims['nsize_ice']
    data['radius_liq_lower'] = tables['radliq_lwr']
    data['radius_liq_upper'] = tables['radliq_upr']
    data['diameter_ice_lower'] = tables['radice_lwr']
    data['diameter_ice_upper'] = tables['radice_upr']
    data['ext_liq'] = tables['lut_extliq']
    data['ssa_liq'] = tables['lut_ssaliq']
    data['asy_liq'] = tables['lut_asyliq']
    data['ext_ice'] = tables['lut_extice']
    data['ssa_ice'] = tables['lut_ssaice']
    data['asy_ice'] = tables['lut_asyice']
    return data

  @classmethod
  def from_nc_file(cls, path: str) -> 'LookupCloudOptics':
    """Instantiates a `LookupCloudOptics` object from a netCDF file.

    The netCDF file should contain the lookup tables for the extinction
    coefficients, the single-scattering albedo, and the asymmetry factor for
    liquid and ice particles.

    Args:
      path: The full path of the netCDF file containing the cloud optics lookup
        tables.

    Returns:
      A `LookupCloudOptics` instance.
    """
    _, tables, dims = cls._parse_nc_file(path)
    kwargs = cls._load_data(tables, dims)
    return cls(**kwargs)
