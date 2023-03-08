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

"""A data class for loading and accessing optical properties of gases."""

import dataclasses
from typing import Text

from swirl_lm.physics.radiation.optics import lookup_gas_optics_base as gas_base
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class LookupGasOpticsLongwave(gas_base.LookupGasOpticsBase):
  """Lookup table of gases' optical properties in the longwave bands."""

  # Planck fraction `(n_gpt, n_η, n_p_ref, n_t_ref)`
  planck_fraction: tf.Tensor
  # Number of reference temperatures, for Planck source calculations.
  n_t_plnk: int
  # reference temperatures for Planck source calculations `(n_t_plnk)`
  t_planck: tf.Tensor
  # total Planck source for each band `(n_t_plnk, n_bnd)`
  totplnk: tf.Tensor

  @classmethod
  def _load_data(cls, tables, dims):
    data = super()._load_data(tables, dims)
    data['n_t_plnk'] = dims['temperature_Planck']
    data['planck_fraction'] = tables['plank_fraction']
    data['t_planck'] = tables['temperature_Planck']
    data['totplnk'] = tables['totplnk']
    return data

  @classmethod
  def from_nc_file(
      cls, path: Text
  ) -> 'LookupGasOpticsLongwave':
    tables, dims = cls._parse_nc_file(path)
    kwargs = cls._load_data(tables, dims)
    return cls(**kwargs)
