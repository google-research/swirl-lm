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

"""Constants used in the one-moment microphysics model."""

# The density of liquid water [kg/m^3].
RHO_WATER = 1e3
# The density of crystal ice [kg/m^3].
RHO_ICE = 0.917e3
# The density of typical air [kg/m^3].
RHO_AIR = 1.0
# The thermal conductivity of air [J/(m s K)].
K_COND = 2.4e-2
# The kinematic visocity of air [m^2/s].
NU_AIR = 1.6e-5
# The molecular diffusivity of water vapor [m^2/s].
D_VAP = 2.26e-5
# The optical asymmetry factor of cloud droplets.
ASYMMETRY_CLOUD = 0.8
# The number of cloud droplets per cubic meter.
DROPLET_N = 1e8
