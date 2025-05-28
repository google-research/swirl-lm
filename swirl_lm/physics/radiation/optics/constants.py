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

"""Central place for all the constants and common keys in the optics library."""

DRY_AIR_INDEX = 0
DRY_AIR_KEY = 'dry_air'
# Volume mixing ratio (VMR) of dry air is always 1 by definition, as VMR is
# normalized by the number of moles of dry air.
DRY_AIR_VMR = 1.0
# Stefan-Boltzmann constant.
STEFAN_BOLTZMANN = 5.67e-8
