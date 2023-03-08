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

"""A data class for loading and accessing shortwave optical properties of gases."""

import dataclasses
from typing import Text

from swirl_lm.physics.radiation.optics import lookup_gas_optics_base as gas_base
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class LookupGasOpticsShortwave(gas_base.LookupGasOpticsBase):
  """Lookup table of gases' optical properties in the shortwave bands."""

  # Total solar irradiation
  solar_src_tot: float
  # Relative solar source contribution from each `g-point` `(n_gpt)`
  solar_src_scaled: tf.Tensor
  # Rayleigh absorption coefficient for lower atmosphere `(n_gpt, n_η, n_t_ref)`
  rayl_lower: tf.Tensor
  # Rayleigh absorption coefficient for upper atmosphere `(n_gpt, n_η, n_t_ref)`
  rayl_upper: tf.Tensor

  @classmethod
  def _load_data(cls, tables, dims):
    data = super()._load_data(tables, dims)
    solar_src = tables['solar_source']
    data['solar_src_tot'] = tf.math.reduce_sum(solar_src)
    data['solar_src_scaled'] = solar_src / data['solar_src_tot']
    data['rayl_lower'] = tables['rayl_lower']
    data['rayl_upper'] = tables['rayl_upper']
    return data

  @classmethod
  def from_nc_file(
      cls, path: Text
  ) -> 'LookupGasOpticsShortwave':
    tables, dims = cls._parse_nc_file(path)
    kwargs = cls._load_data(tables, dims)
    return cls(**kwargs)
