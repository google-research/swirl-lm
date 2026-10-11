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
"""Runs the cylinder validation case.

Unlike the other examples the config file holds a `CylinderCase`, not a
`SwirlLMParameters`; the solver configuration is derived from it so the
Reynolds number and the viscosity cannot disagree. `--dump_config` writes the
derived configuration out so a run can record exactly what it solved.
"""

import os

from absl import app
from absl import flags
from absl import logging
from google.protobuf import text_format
import jax
import numpy as np

from swirl_lm.jax.core import driver as driver_lib
from swirl_lm.jax.examples.cylinder import cylinder
from swirl_lm.jax.examples.cylinder import cylinder_pb2

jax.config.update('jax_enable_x64', False)

_NUM_CYCLES = flags.DEFINE_integer(
    'num_cycles',
    1,
    'number of cycles to run. Each cycle generates a set of output',
)
_NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 1, 'number of steps to run before generating an output.'
)
_CONFIG_FILEPATH = flags.DEFINE_string(
    'config_filepath',
    None,
    'The full path to a CylinderCase config file.',
    allow_override=True,
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    None,
    'The full path to a checkpoint directory.',
    allow_override=True,
)
_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'checkpoint_interval',
    10,
    'interval at which to write solution.',
)
_REYNOLDS_NUMBER = flags.DEFINE_float(
    'reynolds_number',
    None,
    'Overrides the Reynolds number in the config. The viscosity follows.',
)
_CELLS_PER_DIAMETER = flags.DEFINE_integer(
    'cells_per_diameter',
    None,
    'Overrides the resolution in the config, for a convergence study.',
)
_CORES_X = flags.DEFINE_integer(
    'cores_x', None, 'Overrides cores_x in the config.'
)
_CORES_Y = flags.DEFINE_integer(
    'cores_y', None, 'Overrides cores_y in the config.'
)
_CORES_Z = flags.DEFINE_integer(
    'cores_z', None, 'Overrides cores_z in the config.'
)
_PROBE_DISCARD = flags.DEFINE_float(
    'probe_discard',
    0.5,
    'Fraction of the probe record to discard as transient before estimating '
    'the shedding frequency.',
)
_DUMP_CONFIG = flags.DEFINE_bool(
    'dump_config',
    True,
    'Write the derived solver configuration next to the checkpoints.',
)

FLAGS = flags.FLAGS


def _make_case() -> cylinder_pb2.CylinderCase:
  """Reads the case file, applying any command-line overrides."""
  case = cylinder_pb2.CylinderCase()
  if FLAGS.config_filepath:
    with open(FLAGS.config_filepath, 'r') as f:
      text_format.Parse(f.read(), case)
  if _REYNOLDS_NUMBER.value is not None:
    case.reynolds_number = _REYNOLDS_NUMBER.value
  if _CELLS_PER_DIAMETER.value is not None:
    case.cells_per_diameter = _CELLS_PER_DIAMETER.value
  for flag, field in (
      (_CORES_X, 'cores_x'), (_CORES_Y, 'cores_y'), (_CORES_Z, 'cores_z')
  ):
    if flag.value is not None:
      setattr(case, field, flag.value)
  return case


def _check_device_count(case: cylinder_pb2.CylinderCase) -> None:
  """Compares the requested decomposition with the devices actually present.

  `driver.create_mesh` takes `jax.devices()[:cx*cy*cz]`, so asking for fewer
  cores than there are devices silently leaves the rest idle -- a four-GPU job
  quietly running on one. That is worth saying out loud rather than
  discovering from the wall-clock time.

  Args:
    case: The resolved case parameters.

  Raises:
    ValueError: If there are fewer devices than the decomposition needs.
  """
  wanted = case.cores_x * case.cores_y * case.cores_z
  available = jax.device_count()
  if available < wanted:
    raise ValueError(
        f'The configuration asks for {wanted} cores '
        f'({case.cores_x}x{case.cores_y}x{case.cores_z}) but JAX sees '
        f'{available} device(s) on the {jax.default_backend()} backend. '
        'On GPU, request more devices from the scheduler; on CPU, set '
        f'XLA_FLAGS=--xla_force_host_platform_device_count={wanted}.'
    )
  if available > wanted:
    logging.warning(
        'Using %d of %d available %s device(s): the configuration asks for '
        '%dx%dx%d. The remaining %d will sit idle. Set --cores_x/--cores_y/'
        '--cores_z so their product is %d to use them all.',
        wanted, available, jax.default_backend(),
        case.cores_x, case.cores_y, case.cores_z,
        available - wanted, available,
    )


def main(_):
  case = _make_case()
  _check_device_count(case)
  simulation = cylinder.Cylinder(case)
  params = simulation.params
  logging.info('%s', simulation.summary())

  if _DUMP_CONFIG.value and FLAGS.checkpoint_dir:
    os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)
    path = os.path.join(FLAGS.checkpoint_dir, 'derived_config.pbtxt')
    with open(path, 'w') as f:
      f.write(simulation.config_text())
    logging.info('Derived solver configuration written to %s', path)

  # Sample the transverse velocity in the wake every step. This is the only
  # practical way to get the shedding frequency: one period is a few hundred
  # steps, so resolving it by checkpointing would need a full solution dump
  # every few tens of steps -- tens of gigabytes -- while the probe is one
  # float per step.
  mesh = driver_lib.create_mesh(params)
  record: list[np.ndarray] = []

  state = driver_lib.run_simulation(
      params=params,
      init_fn=simulation.initialization,
      num_steps=_NUM_CYCLES.value * _NUM_STEPS.value,
      num_steps_per_cycle=_NUM_STEPS.value,
      checkpoint_dir=FLAGS.checkpoint_dir,
      checkpoint_interval=_CHECKPOINT_INTERVAL.value,
      probe_fn=simulation.probe_fn(mesh),
      probe_sink=record.append,
  )
  signal = np.concatenate(record) if record else np.zeros(0)

  # Below about Re = 47 the wake is steady and the bubble length is the
  # quantity to compare; above it the wake sheds and the bubble length
  # oscillates, so it is reported but should not be read as a validation.
  if FLAGS.checkpoint_dir and signal.size:
    path = os.path.join(FLAGS.checkpoint_dir, 'wake_probe.npy')
    np.save(path, signal)
    logging.info('Wake probe record (%d samples) written to %s',
                 signal.size, path)

  length = simulation.recirculation_length(state)
  steady = simulation.case.reynolds_number < 47.0
  logging.info(
      'recirculation length L/d = %.3f%s',
      length,
      '' if steady else '  (wake is shedding; not a steady-state measure)',
  )
  if steady:
    logging.info('published value at Re = 40 is about 2.2')
  else:
    # Drop the start-up transient before estimating the frequency: the wake
    # takes a few flow-through times to establish, and those samples would
    # otherwise widen the spectral peak.
    keep = signal[int(_PROBE_DISCARD.value * signal.size):]
    periods = keep.size * simulation.dt * 0.165 / simulation.diameter
    if periods < 8.0:
      logging.warning(
          'the probe record after discarding the transient covers only %.1f '
          'shedding periods; the frequency estimate will be coarse. Run '
          'longer, or lower --probe_discard.',
          periods,
      )
    strouhal = simulation.strouhal_number(keep)
    logging.info(
        'Strouhal number St = %.4f from %d samples (~%.0f shedding '
        'periods); the published value at Re = 100 is 0.16 to 0.17',
        strouhal, keep.size, periods,
    )


if __name__ == '__main__':
  app.run(main)
