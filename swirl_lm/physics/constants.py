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

"""A library of commonly used physical constants."""

# Universal gas constant, in units of J/mol/K.
R_UNIVERSAL = 8.3145

# The precomputed gas constant for dry air, in units of J/kg/K.
R_D = 286.69

# The gravitational acceleration constant, in units of N/kg.
G = 9.81

# The heat capacity ratio of dry air, dimensionless.
GAMMA = 1.4

# The constant pressure heat capacity of dry air, in units of J/kg/K.
CP = GAMMA * R_D / (GAMMA - 1.0)

# The constant volume heat capacity of dry air, in units of J/kg/K.
CV = CP - R_D

# The molecular mass of dry air (kg/mol).
DRY_AIR_MOL_MASS = 0.0289647

# The molecular mass of water (kg/mol).
WATER_MOL_MASS = 0.0180153

# Avogadro's number.
AVOGADRO = 6.022e23
