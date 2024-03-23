# Copyright 2024 The swirl_lm Authors.
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
"""A base data class for the lookup tables of atmospheric optical properties."""

import dataclasses
from typing import Any, Dict

import netCDF4 as nc
from swirl_lm.physics.radiation.optics import constants
from swirl_lm.physics.radiation.optics import data_loader_base as loader
from swirl_lm.utility import types
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class LookupVolumeMixingRatio(loader.DataLoaderBase):
  """Lookup table of volume mixing ratio profiles of atmospheric gases."""

  # Reference pressure to be used when interpolating the vmr.
  p_ref: tf.Tensor
  # Volume mixing ratio of water vapor by pressure layer `(n_p_ref)`.
  vmr_h2o: tf.Tensor
  # Volume mixing ratio of Ozone by pressure layer `(n_p_ref)`.
  vmr_o3: tf.Tensor
  # Volume mixing ratio (vmr) global mean (gm) of all other gases keyed by
  # chemical formula.
  vmr_gm: types.TensorMap

  @classmethod
  def _load_data(
      cls,
      ds: nc.Dataset,
      tables: types.VariableMap,
      site_coord: int,
      exp_label: int,
  ) -> Dict[str, Any]:
    """Preprocesses volume mixing ratios of select atmospheric gases.

    Note that only ozone and water vapor vary with the pressure level. Global
    means are used for all the other gases.

    Args:
      ds: The original netCDF Dataset containing the RRTMGP atmospheric state.
      tables: The extracted data as a dictionary of `tf.Variable`s.
      site_coord: The site coordinate index, which uniquely identifies a
        (latitude, longitude, time) triplet.
      exp_label: The RFMIP experiment label.

    Returns:
      A dictionary containing dimension information and the RRTMGP data as
      `tf.Variable`s.
    """
    # Mapping from chemical formula to canonical molecular name as they appear
    # in the atmospheric state table. The _GM suffix denotes the global mean of
    # the volume mixing ratio.
    chem_formula_to_table_name_mapping = (
        ('co2', 'carbon_dioxide_GM'),
        ('n2o', 'nitrous_oxide_GM'),
        ('co', 'carbon_monoxide_GM'),
        ('ch4', 'methane_GM'),
        ('o2', 'oxygen_GM'),
        ('n2', 'nitrogen_GM'),
        ('ccl4', 'carbon_tetrachloride_GM'),
        ('cfc11', 'cfc11_GM'),
        ('cfc12', 'cfc12_GM'),
        ('cfc22', 'hcfc22_GM'),
        ('hfc143a', 'hfc143a_GM'),
        ('hfc125', 'hfc125_GM'),
        ('hfc23', 'hfc23_GM'),
        ('hfc32', 'hfc32_GM'),
        ('hfc134a', 'hfc134a_GM'),
        ('cf4', 'cf4_GM'),
    )
    vmr_gm = {}
    for chem_formula, varname in chem_formula_to_table_name_mapping:
      vmr_gm[chem_formula] = tables[varname][exp_label] * float(
          ds[varname].units
      )
    # Dry air is a special case that always has a volume mixing ratio of 1
    # since, by definition, vmr is normalized by the number of moles of dry air.
    vmr_gm[constants.DRY_AIR_KEY] = constants.DRY_AIR_VMR

    p_ref = ds['pres_layer'][:].data[site_coord, :]

    return dict(
        p_ref=tf.constant(p_ref),
        vmr_h2o=tables['water_vapor'][exp_label, site_coord, :],
        vmr_o3=tables['ozone'][exp_label, site_coord, :],
        vmr_gm=vmr_gm,
    )

  @classmethod
  def from_nc_file(
      cls,
      path: str,
      site_coord: int,
      exp_label: int,
  ) -> 'LookupVolumeMixingRatio':
    """Instantiates a `LookupVolumeMixingRatio` object from netCDF file.

    The compressed file should be netCDF parsable and contain the RRTMGP
    atmospheric state table with gas concentrations as well as data about the
    solar source and surface optical properties, indexed by `site` and
    and `exp_label`.

    Args:
      path: The full path of the zipped netCDF file containing the shortwave
        absorption coefficient lookup table.
      site_coord: The site coordinate index, which uniquely identifies a
        (latitude, longitude, time) triplet.
      exp_label: The RFMIP experiment label.

    Returns:
      A `LookupVolumeMixingRatio` object.
    """
    ds, tables, _ = cls._parse_nc_file(path, exclude_vars=['expt_label'])
    kwargs = cls._load_data(ds, tables, site_coord, exp_label)
    return cls(**kwargs)
