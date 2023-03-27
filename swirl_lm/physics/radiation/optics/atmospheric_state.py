# Copyright 2023 The swirl_lm Authors.
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

import numpy as np
from swirl_lm.physics.radiation.optics import data_loader_base as loader
from swirl_lm.physics.radiation.optics import lookup_volume_mixing_ratio as lookup_vmr
from swirl_lm.utility import types

# Default site corresponding to coordinates 13.5 N, 298.5 E.
DEFAULT_SITE_COORD = 9
# Default RFMIP experiment id.
DEFAULT_RFMIP_EXP_LABEL = 1


@dataclasses.dataclass(frozen=True)
class AtmosphericState(loader.DataLoaderBase):
  """Atmospheric gas concentrations and miscellaneous optical properties."""

  # Surface emissivity; the same for all bands.
  sfc_emis: float
  # Surface albedo; the same for all bands.
  sfc_alb: float
  # The solar zenith angle.
  zenith: float
  # The total solar irradiance.
  irrad: float
  # Volume mixing ratio lookup for each gas species. Only water vapor and ozone
  # are assumed to be variable. Global means are used for all the other species.
  vmr: lookup_vmr.LookupVolumeMixingRatio

  @classmethod
  def _load_data(
      cls,
      tables: types.VariableMap,
      site_coord: int,
      exp_label: int,
      vmr_path: str,
  ) -> Dict[str, Any]:
    """Preprocesses the RRTMGP atmospheric state data.

    Args:
      tables: The extracted data as a dictionary of `tf.Variable`s.
      site_coord: The site coordinate index, which uniquely identifies a
        (latitude, longitude, time) triplet.
      exp_label: The RFMIP experiment label.
      vmr_path: The full path of the zipped netCDF file containing the
        atmospheric gas concentrations.

    Returns:
      A dictionary containing dimension information and the preprocessed RRTMGP
      atmospheric state data as `tf.Variable`s.
    """
    return dict(
        sfc_emis=tables['surface_emissivity'][site_coord],
        sfc_alb=tables['surface_albedo'][site_coord],
        zenith=np.radians(tables['solar_zenith_angle'][site_coord]),
        irrad=tables['total_solar_irradiance'][site_coord],
        vmr=lookup_vmr.LookupVolumeMixingRatio.from_nc_file(
            vmr_path, site_coord, exp_label
        ),
    )

  @classmethod
  def from_nc_file(
      cls,
      path: str,
      site_coord: int = DEFAULT_SITE_COORD,
      exp_label: int = DEFAULT_RFMIP_EXP_LABEL,
  ) -> 'AtmosphericState':
    """Instantiates an `AtmosphericState` object from a zipped netCDF file.

    Args:
      path: The full path of the zipped netCDF file containing the atmospheric
        state.
      site_coord: The site coordinate index, which uniquely identifies a
        (latitude, longitude, time) triplet.
      exp_label: The RFMIP experiment label.

    Returns:
      An `AtmosphericState` object.
    """
    _, tables, _ = cls._parse_nc_file(path, exclude_vars=['expt_label'])
    kwargs = cls._load_data(tables, site_coord, exp_label, path)
    return cls(**kwargs)
