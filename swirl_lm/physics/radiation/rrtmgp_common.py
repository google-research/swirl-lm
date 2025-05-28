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


"""Common variables for rrtmgp."""

from collections.abc import Sequence
from typing import Optional

from swirl_lm.physics.radiation.config import radiative_transfer_pb2

# Key for the applied radiative heating rate, in K/s.  When supercycling this
# field is only nonzero on the steps the radiative term is applied.
KEY_APPLIED_RADIATION = 'rad_heat_src_applied'
# Key for the stored radiative heating rate, in K/s.  This is the stored
# radiative heating rate and is nonzero for every step, even if supercycling.
KEY_STORED_RADIATION = 'rad_heat_src'

# Key for the longwave radiative fluxes, in W/m^2.
KEY_RADIATIVE_FLUX_LW = 'rad_flux_lw'
# Key for the longwave radiative fluxes in the extended grid, in W/m^2.
KEY_EXT_RADIATIVE_FLUX_LW = 'extended_rad_flux_lw'
# Key for the shortwave radiative fluxes, in W/m^2.
KEY_RADIATIVE_FLUX_SW = 'rad_flux_sw'
# Key for the shortwave radiative fluxes in the extended grid, in W/m^2.
KEY_EXT_RADIATIVE_FLUX_SW = 'extended_rad_flux_sw'
# Key for the longwave radiative fluxes with cloud effects removed, in W/m^2.
KEY_RADIATIVE_FLUX_LW_CLEAR = 'rad_flux_lw_clear'
# Key for the longwave radiative fluxes in the extended grid with cloud effects
# removed, in W/m^2.
KEY_EXT_RADIATIVE_FLUX_LW_CLEAR = 'extended_rad_flux_lw_clear'
# Key for the shortave radiative fluxes with cloud effects removed, in W/m^2.
KEY_RADIATIVE_FLUX_SW_CLEAR = 'rad_flux_sw_clear'
# Key for the shortave radiative fluxes in the extended grid with cloud effects
# removed, in W/m^2.
KEY_EXT_RADIATIVE_FLUX_SW_CLEAR = 'extended_rad_flux_sw_clear'


def required_keys(
    radiative_transfer_config: radiative_transfer_pb2.RadiativeTransfer,
) -> list[str]:
  """Returns the required keys for the rrtmgp radiative transfer library."""
  if radiative_transfer_config is None:
    return []
  else:
    return [KEY_APPLIED_RADIATION, KEY_STORED_RADIATION]


def additional_keys(
    radiative_transfer_config: radiative_transfer_pb2.RadiativeTransfer,
    additional_state_keys: Optional[Sequence[str]] = None,
) -> list[str]:
  """Returns all additional keys related to the radiative transfer library."""
  if radiative_transfer_config is None:
    return []
  else:
    diagnostic_keys = [
        KEY_RADIATIVE_FLUX_LW,
        KEY_RADIATIVE_FLUX_SW,
        KEY_RADIATIVE_FLUX_LW_CLEAR,
        KEY_RADIATIVE_FLUX_SW_CLEAR,
    ]
    diagnostic_keys += [f'extended_{k}' for k in diagnostic_keys]
    # Include diagnostic keys only if present in `additional_state_keys`.
    diagnostic_keys = [k for k in diagnostic_keys if k in additional_state_keys]
    return required_keys(radiative_transfer_config) + diagnostic_keys
