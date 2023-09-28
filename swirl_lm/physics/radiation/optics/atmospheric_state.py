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
from swirl_lm.physics.radiation.config import radiative_transfer_pb2
from swirl_lm.physics.radiation.optics import data_loader_base as loader
from swirl_lm.physics.radiation.optics import lookup_volume_mixing_ratio as lookup_vmr
from swirl_lm.utility import types


@dataclasses.dataclass(frozen=True)
class AtmosphericState(loader.DataLoaderBase):
  """Atmospheric gas concentrations and miscellaneous optical properties."""
  # Surface emissivity; the same for all bands.
  sfc_emis: float
  # Surface albedo; the same for all bands.
  sfc_alb: float
  # The solar zenith angle.
  zenith: float
  # The total solar irradiance (in W/m²).
  irrad: float
  # Volume mixing ratio lookup for each gas species. Only water vapor and ozone
  # are assumed to be variable. Global means are used for all the other species.
  vmr: lookup_vmr.LookupVolumeMixingRatio
  # The longwave incident flux at the top of the atmosphere (in W/m²).
  toa_flux_lw: float = 0.0

  @classmethod
  def _load_data(
      cls,
      params: radiative_transfer_pb2.AtmosphericState,
      tables: types.VariableMap,
      vmr_path: str,
  ) -> Dict[str, Any]:
    """Preprocesses the atmospheric state data.

    Args:
      params: A `radiative_transfer_pb2.AtmosphericState` proto instance
        containing RFMIP identifiers and override parameters.
      tables: The extracted data as a dictionary of `tf.Variable`s.
      vmr_path: The full path of the netCDF file containing the atmospheric gas
        concentrations.

    Returns:
      A dictionary containing dimension information and the preprocessed RRTMGP
      atmospheric state data as `tf.Variable`s.
    """
    rfmip_site = params.rfmip_site
    rfmip_expt_label = params.rfmip_expt_label

    sfc_emis = (
        params.sfc_emis
        if params.HasField('sfc_emis')
        else tables['surface_emissivity'][rfmip_site]
    )
    sfc_alb = (
        params.sfc_alb
        if params.HasField('sfc_alb')
        else tables['surface_albedo'][rfmip_site]
    )
    zenith = (
        params.zenith
        if params.HasField('zenith')
        else np.radians(tables['solar_zenith_angle'][rfmip_site])
    )
    irrad = (
        params.irrad
        if params.HasField('irrad')
        else tables['total_solar_irradiance'][rfmip_site]
    )
    kwargs = dict(
        sfc_emis=sfc_emis,
        sfc_alb=sfc_alb,
        zenith=zenith,
        irrad=irrad,
        vmr=lookup_vmr.LookupVolumeMixingRatio.from_nc_file(
            vmr_path, rfmip_site, rfmip_expt_label
        ),
    )
    if params.HasField('toa_flux_lw'):
      kwargs['toa_flux_lw'] = params.toa_flux_lw
    return kwargs

  @classmethod
  def from_proto(
      cls,
      params: radiative_transfer_pb2.AtmosphericState,
  ) -> 'AtmosphericState':
    """Instantiates an `AtmosphericState` object from a netCDF file.

    Args:
      params: A `radiative_transfer_pb2.AtmosphericState` proto instance
        containing RFMIP identifiers, override parameters, and a path to a
        netCDF file containing gas concentrations for various species.

    Returns:
      An `AtmosphericState` instance.
    """
    _, tables, _ = cls._parse_nc_file(
        params.atmospheric_state_nc_filepath, exclude_vars=['expt_label']
    )
    kwargs = cls._load_data(
        params, tables, params.atmospheric_state_nc_filepath
    )
    return cls(**kwargs)

  @classmethod
  def from_nc_file(
      cls,
      path: str,
  ) -> 'AtmosphericState':
    """Instantiates an `AtmosphericState` object from a netCDF file.

    Args:
      path: The full path of the netCDF file containing the atmospheric state.

    Returns:
      An `AtmosphericState` instance.
    """
    params = radiative_transfer_pb2.AtmosphericState()
    params.atmospheric_state_nc_filepath = path
    return cls.from_proto(params)
