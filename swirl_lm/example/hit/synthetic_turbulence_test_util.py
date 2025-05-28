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

"""Synthetic Turbulence test utilities."""

import numpy as np


def calculate_spectrum(
    dk: float,
    kc: float,
    tke: np.ndarray,
    kk: np.ndarray,
) -> np.ndarray:
  r"""Calculates the power spectrum.

  Args:
    dk: The angular wavenumber `2π/L` where L is the length of the domain.
    kc: The cut of angular frequency, above which the energy is zero.
    tke: A 3D array representing the turbulent kinetic energy at each grid
      point.
    kk: A 3D array representing the angular frequency mangitude \sqrt{k_x^2 +
      k_y^2 + k_z^2} at each grid point.

  Returns:
    A 1D array with each bin representing the total energy falling with the
    range of angular frenquency of [m dk, (m + 1) dk), where m is the index of
    the corresponding bin.
  """
  num_bins = int(round(kc / dk))
  spectrum = np.zeros([num_bins], dtype=float)
  for i in range(tke.shape[0]):
    for j in range(tke.shape[1]):
      for k in range(tke.shape[2]):
        ind = int(round(kk[i, j, k] / dk))
        if ind >= num_bins:
          continue
        spectrum[ind] = spectrum[ind] + tke[i, j, k]
  return spectrum


def get_slope_estimate(
    spectrum: np.ndarray,
    k_id: int,
    smooth: int = 3,
) -> float:
  """Calculates the smoothed log-log slope from a spectrum at one location.

  Args:
    spectrum: A 1D array representing the spectrum.
    k_id: The index representing the location of the spectrum where the slope is
      to be calculated.
    smooth: An integer representing the extent of the smoothing in calculating
      the slope. The log-log slope will be calculated (with centeral estimate)
      for all points in the range of [k_id - smooth, k_id + smooth] and the
      average is returned.

  Returns:
    The smoothed log-log slope at `k_id` in the spectrum.
  """
  slopes = []
  if k_id <= smooth or k_id + smooth + 1 >= len(list(spectrum)):
    raise ValueError('Argument smooth is out of allowed range.')
  for i in range(k_id - smooth, k_id + smooth + 1):
    hi = i + 1
    lo = i - 1
    slopes.append((np.log(spectrum[hi]) - np.log(spectrum[lo])) /
                  (np.log(float(hi)) - np.log(float(lo))))
  return np.average(slopes)


def get_expected_vkp_slope(
    k_id: int,
    dk: float,
    ke: float,
    kd: float,
    smooth: int = 3,
) -> float:
  """This generates smoothed log-log asymptotic slope of a VKP spectrum.

  The slope is calculated at the point `k_id * dk` of the angular
  frequency.

  Args:
    k_id: The slope will be calculated at `k_id * dk`.
    dk: The wavenumber interval.
    ke: The angular frequency corresponding to the turbulent eddy energetic
      length scale.
    kd: The angular frequency corresponding to the turbulent eddy dissipative
      length scale.
    smooth: An integer representing the extent of the smoothing in calculating
      the slope. The log-log slope will be calculated (with centeral estimate)
      for all points in the range of [k_id - smooth, k_id + smooth] and the
      average is returned.

  Returns:
    The smoothed log-log slope at `k_id` in the VKP asymptotic spectrum.
  """

  def slope_fn(
      k,
      ke,
      kd,
  ):
    alpha = 1.5
    return (4.0 - (17.0 / 3.0) * (k / ke)**2.0 / (1.0 + (k / ke)**2.0) -
            2.0 * alpha * np.power(k / kd, 4.0 / 3.0))

  slopes = []
  for i in range(-smooth, smooth + 1):
    slopes.append(slope_fn((k_id + i) * dk, ke, kd))
  return np.average(slopes)


def get_expected_dist_pope_slope(
    k_id: int,
    dk: float,
    le: float,
    ld: float,
    smooth: int = 3,
) -> float:
  """This generates smoothed log-log asymptotic slope of a VKP spectrum.

  Here the spectrum follows the model spectrum in `S. B. Pope, Turbulent Flows,
  Cambridge University Press, Cambridge, 2000`.

  The slope is calculated at the point `k_id * dk` of the angular
  frequency.

  Args:
    k_id: The slope will be calculated at `k_id * dk`.
    dk: The wavenumber interval.
    le: The energetic length scale of the turbulent eddy.
    ld: The dissipative length scale of the turbulent eddy.
    smooth: An integer representing the extent of the smoothing in calculating
      the slope. The log-log slope will be calculated (with centeral estimate)
      for all points in the range of [k_id - smooth, k_id + smooth] and the
      average is returned.

  Returns:
    The smoothed log-log slope at `k_id` in the model spectrum.
  """

  # Coefficients from the model spectrum in `S. B. Pope, Turbulent Flows,
  # Cambridge University Press, Cambridge, 2000, section 6.5.3.`
  c_l = 6.78
  c_eta = 0.4
  beta = 5.2
  p0 = 4.0
  f_le = 1.0
  def slope_fn(
      k,
      le,
      ld,
  ):
    # This is the theoretical asymptotic slope in log-log scale. This is based
    # on the model spectrum in `S. B. Pope, Turbulent Flows, Cambridge
    # University Press, Cambridge, 2000`, and perform d log(E(k)) / d log(k).
    return (-5.0 / 3.0 + (5.0 / 3.0 + p0) * c_l / ((k * le / f_le) ** 2 + c_l) -
            beta * (k * ld) ** 4 * ((k * ld) ** 4 + c_eta ** 4) ** (-3.0 / 4.0))

  slopes = []
  for i in range(-smooth, smooth + 1):
    slopes.append(slope_fn((k_id + i) * dk, le, ld))
  return np.average(slopes)
