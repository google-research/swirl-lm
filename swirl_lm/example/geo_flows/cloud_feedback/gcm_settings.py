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

"""Settings of a GCM-driven LES."""
import dataclasses
from typing import Optional

GCM_PREFIX_KEY = 'gcm'

GCM_THETA_LI_KEY = f'{GCM_PREFIX_KEY}_theta_li'
GCM_Q_T_KEY = f'{GCM_PREFIX_KEY}_q_t'
GCM_TEMPERATURE_KEY = f'{GCM_PREFIX_KEY}_T'

GCM_ADV_TENDENCY_TEMPERATURE_KEY = f'{GCM_PREFIX_KEY}_T_adv'
GCM_ADV_TENDENCY_HUMIDITY_KEY = f'{GCM_PREFIX_KEY}_q_t_adv'

GCM_U_KEY = f'{GCM_PREFIX_KEY}_u'
GCM_V_KEY = f'{GCM_PREFIX_KEY}_v'
GCM_W_KEY = f'{GCM_PREFIX_KEY}_w'

GCM_Q_L_KEY = f'{GCM_PREFIX_KEY}_q_l'
GCM_Q_I_KEY = f'{GCM_PREFIX_KEY}_q_i'
GCM_CLD_FRAC_KEY = f'{GCM_PREFIX_KEY}_cld_frac'


@dataclasses.dataclass
class GCMSettings:
  """Defines parameters to drive a geophysical simulation with a GCM state."""

  # Comma-separated list of paths to the GCM column profiles.
  sounding_csv_filename: str = ''

  # This configuration assumes x, y, and z are being used for the streamwise,
  # lateral, and vertical directions, respectively.

  # The noise rms of the initial velocity (m/s) in the streamwise direction.
  u_rms: float = 1.0
  # The noise rms of the initial velocity (m/s) in the lateral direction.
  v_rms: float = 1.0
  # The noise rms of the initial velocity (m/s) in the vertical direction.
  w_rms: float = 0.0
  # The noise rms of the initial theta_li profile (K).
  theta_li_rms: float = 0.01
  # The latitude in radians where the GCM column is located. If not set, no
  # Coriolis force will be applied. Disabling the Coriolis force is acceptable
  # in the case of our GCM forcing framework because the horizontal winds are
  # relaxed to the GCM winds.
  latitude: Optional[float] = None
  # The mean surface temperature for the entire GCM column (K). This should
  # reflect the actual temperature of the surface (e.g. sea surface temperature)
  # instead of the temperature of the model's first fluid layer. Setting this
  # temperature appropriately is important for the accuracy of the radiative
  # transfer model, in particular the longwave solver.
  sfc_temperature: float = 290.5
  # Altitude thresholds for ramping up the relaxation coefficient.
  # Beginning of the free troposphere (m).
  z_i: float = 3000.0
  # End height for the troposphere relaxation ramp-up (m). Above this height the
  # relaxation timescale is exactly as specified in `tau_r_tropo_sec` below.
  z_r: float = 3500.0
  # Timescale of troposphere relaxation to GCM thermodynamic state (s).
  tau_r_tropo_sec: float = 86400.0
  # Timescale of horizontal momentum relaxation to geostrophic winds across the
  # entire domain (s).
  tau_r_wind_sec: float = 21600.0
  # Used to seed the random noise. Must be set.
  random_seed: int = 42
