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

"""Coefficients for terminal velocity parameterization from Chen et al."""

import dataclasses


@dataclasses.dataclass(frozen=True, init=False)
class TableB1Coeffs:
  """Coefficients from table B1 of Chen et al. (2022)."""
  a = [0.044612, -0.263166, 4.7178]
  b = [
      [2.2955, -0.038465],
      [2.2955, -0.038465],
      [1.1451, -0.038465]
  ]
  c = [0.0, 0.184325, 0.184325]
  rho_exp = -0.47335
  q_coeff = 0.115231


@dataclasses.dataclass(frozen=True, init=False)
class TableB3Coeffs:
  """Coefficients from table B3 of Chen et al. (2022)."""
  a = [-0.263503, 0.00174079, -0.0378769]
  b = [0.575231, 0.0909307, 0.515579]
  c = [-0.345387, 0.177362, -0.000427794, 0.00419647]
  e = [-0.156593, -0.0189334, 0.1377817]
  f = [-3.35641, -0.0156199, 0.765337]
  g = [-0.0309715, 1.55054, -0.518349]


@dataclasses.dataclass(frozen=True, init=False)
class TableB5Coeffs:
  """Coefficients from table B5 of Chen et al. (2022)."""
  a = [-0.475897, -0.00231270, 1.12293]
  b = [-2.56289, -0.00513504, 0.608459]
  c = [-0.756064, 0.935922, -1.70952]
  e = [0.00639847, 0.00906454, -0.108232]
  f = [0.515453, -0.0725042, -1.86810, 43.74911676]
  g = [2.65236, 0.00158269, 259.935]
  h = [-0.346044, -7.17829e-11, -1.24394, 46.05170186]
