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

"""Commonly used vars for equations sub directory."""

# Density keys.
KEY_RHO = 'rho'

# Pressure keys.
KEY_P = 'p'
KEY_DP = 'dp'

# Velocity keys.
KEY_U = 'u'
KEY_V = 'v'
KEY_W = 'w'

KEYS_VELOCITY = (KEY_U, KEY_V, KEY_W)

# Momentum keys.
KEY_RHO_U = 'rho_u'
KEY_RHO_V = 'rho_v'
KEY_RHO_W = 'rho_w'

KEYS_MOMENTUM = (KEY_RHO_U, KEY_RHO_V, KEY_RHO_W)

# Define keys for buoyancy diagnostics. Note these should be used for
# diagnostics only.
# TODO(b/274176115): Clean up all diagnostic variables and move it here.
KEYS_DIAGNOSTICS_BUOYANCY = ('buoyancy_u', 'buoyancy_v', 'buoyancy_w')
