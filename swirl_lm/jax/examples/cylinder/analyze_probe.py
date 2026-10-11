# Copyright 2026 The swirl_lm Authors.
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
"""Re-reads a saved wake probe and reports the shedding frequency.

The probe record is written by `cylinder_main` as `wake_probe.npy`, one
sample per step. Because it is saved, the frequency can be re-examined
without repeating the simulation -- which matters, because the single number
the run logs depends on how much of the record is discarded as transient, and
a result that moves when that choice moves is not yet converged.

  python -m swirl_lm.jax.examples.cylinder.analyze_probe \
      --probe=/path/to/wake_probe.npy \
      --config_filepath=swirl_lm/jax/examples/cylinder/cylinder.pbtxt
"""

from absl import app
from absl import flags
from google.protobuf import text_format
import numpy as np

from swirl_lm.jax.examples.cylinder import cylinder
from swirl_lm.jax.examples.cylinder import cylinder_pb2

_PROBE = flags.DEFINE_string('probe', None, 'Path to wake_probe.npy.')
_CONFIG_FILEPATH = flags.DEFINE_string(
    'config_filepath', None, 'The CylinderCase the run used.'
)

FLAGS = flags.FLAGS


def main(_):
  case = cylinder_pb2.CylinderCase()
  if _CONFIG_FILEPATH.value:
    with open(_CONFIG_FILEPATH.value, 'r') as f:
      text_format.Parse(f.read(), case)
  simulation = cylinder.Cylinder(case)
  signal = np.load(_PROBE.value)

  print(simulation.summary())
  total_periods = signal.size * simulation.dt * 0.165 / simulation.diameter
  print(
      f'\n{signal.size} samples at dt = {simulation.dt:g} s = '
      f'{signal.size * simulation.dt:.1f} s ~ {total_periods:.0f} shedding '
      'periods'
  )

  # The frequency should not care where the window starts, once the transient
  # is out. A number that drifts with the discard fraction is still measuring
  # the start-up, not the shedding.
  print('\n  discard   samples   periods   St       bin')
  previous = None
  values = []
  for discard in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75):
    keep = signal[int(discard * signal.size):]
    strouhal = simulation.strouhal_number(keep)
    values.append(strouhal)
    periods = keep.size * simulation.dt * 0.165 / simulation.diameter
    drift = '' if previous is None else f'  ({strouhal - previous:+.4f})'
    # The spectral bin width in Strouhal units: the floor on how precisely a
    # record of this length can place the peak, before any interpolation.
    bin_width = simulation.diameter / (
        keep.size * simulation.dt * simulation.u_inf
    )
    print(
        f'  {discard:5.0%}   {keep.size:7d}   {periods:7.0f}   '
        f'{strouhal:.4f}   {bin_width:.4f}{drift}'
    )
    previous = strouhal

  # Compare the spread against the longest window's bin width rather than a
  # fixed number: a short window cannot place the peak finely however clean
  # the signal is, so some of the spread is resolution, not transient.
  settled = values[2:]
  spread = max(settled) - min(settled)
  finest_bin = simulation.diameter / (
      signal.size * simulation.dt * simulation.u_inf
  )
  verdict = (
      'consistent with the window lengths'
      if spread < 4.0 * finest_bin
      else 'larger than the window lengths explain, so the record probably '
      'still contains transient'
  )
  print(
      f'\nspread over the settled windows: {spread:.4f}, against a finest '
      f'bin of {finest_bin:.4f} -- {verdict}.'
  )
  print(
      f'St = {np.median(settled):.4f} against a published 0.164 to 0.167 at '
      'Re = 100.'
  )


if __name__ == '__main__':
  app.run(main)
