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
"""A data class for atmospheric optical properties."""

import dataclasses

from swirl_lm.physics.radiation.config import radiative_transfer_pb2
from swirl_lm.physics.radiation.optics import lookup_volume_mixing_ratio as lookup_vmr


@dataclasses.dataclass(frozen=True)
class AtmosphericState:
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
  def from_proto(
      cls,
      proto: radiative_transfer_pb2.AtmosphericState,
  ) -> 'AtmosphericState':
    """Instantiates an `AtmosphericState` object from a proto.

    Args:
      proto: A `radiative_transfer_pb2.AtmosphericState` proto instance
        containing atmospheric conditions and the path to a file containing
        volume mixing ratio sounding data.

    Returns:
      An `AtmosphericState` instance.
    """
    kwargs = dict(
        sfc_emis=proto.sfc_emis,
        sfc_alb=proto.sfc_alb,
        zenith=proto.zenith,
        irrad=proto.irrad,
        toa_flux_lw=proto.toa_flux_lw,
        vmr=lookup_vmr.LookupVolumeMixingRatio.from_proto(proto),
    )
    return cls(**kwargs)
