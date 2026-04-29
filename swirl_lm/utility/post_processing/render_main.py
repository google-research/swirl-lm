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

r"""Beam pipeline for volume rendering of clouds.

"""
from collections.abc import Iterable, Sequence
import logging

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from scipy.spatial import transform
from swirl_lm.utility.post_processing import render
import tensorflow as tf
import xarray

_INPUT = flags.DEFINE_string('input', '', 'Path to input zarr.')
_OUTPUT = flags.DEFINE_string('output', '', 'Output directory.')
_RESOLUTION = flags.DEFINE_float(
    'resolution', 250, 'Rendering grid resolution in meters.'
)
_EXTINCTION_COEFF_SUN = flags.DEFINE_float(
    'extinction_coeff_sun', 4e-4,
    'Extinction coefficient while ray-marching sun light.'
)
_EXTINCTION_COEFF_VIEW = flags.DEFINE_float(
    'extinction_coeff_view', 4e-3,
    'Extinction coefficient while ray-marching towards view plane.'
)
_SCATTERING_COEFF = flags.DEFINE_float(
    'scattering_coeff', 4e-3, 'Scattering coefficient.'
)
_SUN_DIR = flags.DEFINE_string(
    'sun_dir', 'xz,60,15', 'Euler rotation to locate sun. The comma separated '
    'args are passed to scipy\'s transform.Rotation.from_euler() function so '
    'see that function\'s documentation for details. The rotation is applied '
    'to [0, 0, 1] and the resulting vector points to the sun.'
)
_VIEW_DIR = flags.DEFINE_string(
    'view_dir', 'xz,75,-40', 'Euler rotation to locate viewing angle. The '
    'comma separated args are passed to scipy\'s '
    'transform.Rotation.from_euler() function so see that function\'s '
    'documentation for details. The rotation is applied to [0, 0, 1] and '
    'the resulting vector points from the viewer to the volume.'
)
_PIPELINE_OPTIONS = flags.DEFINE_list(
    'pipeline_options',
    ['--runner=DirectRunner'],
    'A comma-separated list of command line arguments to be used as options'
    ' for the Beam Pipeline.',
)


def get_qc(path: str) -> xarray.DataArray:
  """Extracts qc from CM1 or Swirl datasets."""
  ds = xarray.open_zarr(path, consolidated=True)
  if ds.attrs.get('CM1 version') is not None:
    qc = ds.qc.rename(xh='x', yh='y', zh='z', time='t')
    qc = qc.assign_coords(x=qc.x * 1e3, y=qc.y * 1e3, z=qc.z * 1e3)
  else:
    qc = ds.q_c
  return qc


def open_dataset(path: str) -> Iterable[tuple[int, xarray.DataArray]]:
  qc = get_qc(path)
  return enumerate(qc.t)


def render_at_time(
    e: tuple[int, xarray.DataArray],
    path: str,
    resolution: float,
    extinction_coeff_sun: float,
    extinction_coeff_view: float,
    scattering_coeff: float,
    sun_dir: transform.Rotation,
    view_dir: transform.Rotation,
    output: str,
) -> None:
  """Renders a single frame."""
  i, t = e
  m = 0.007  # "Max" qc for rendering.
  qc = get_qc(path)
  frame_data = qc.isel(t=i).compute()
  logging.info(
      'Loaded qc t=%s nbytes=%s shape=%s', t.item(), frame_data.nbytes, qc.shape
  )
  scattering, extinction, surface, surface_transform = render.render(
      sun_dir, view_dir, resolution, extinction_coeff_sun / m,
      extinction_coeff_view / m, scattering_coeff / m, frame_data
  )

  # TODO(bcg): Save as xarray.
  data = {
      't': t,
      'scattering': scattering,
      'extinction': extinction,
      'surface': surface,
      'surface_transform': surface_transform,
  }
  filename = f'{output}/{i}.npy'
  logging.info('Writing qc t=%s to %s', t.item(), filename)
  with tf.io.gfile.GFile(filename, 'wb') as f:
    np.save(f, data)


def rotation_from_euler_str(s: str) -> transform.Rotation:
  parts = s.split(',')
  assert len(parts[0]) == len(parts[1:]), (
      f'Expected {len(parts[0])} angles (one for each of `{parts[0]}`), '
      f'got {len(parts[1:])} angles: {parts[1:]}'
  )
  return transform.Rotation.from_euler(
      parts[0], [float(x) for x in parts[1:]], degrees=True
  )




def main(argv: Sequence[str]) -> None:
  del argv

  tf.io.gfile.makedirs(_OUTPUT.value)

  sun_dir = rotation_from_euler_str(_SUN_DIR.value)
  view_dir = rotation_from_euler_str(_VIEW_DIR.value)

  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      _PIPELINE_OPTIONS.value
  )

  with beam.Pipeline(options=pipeline_options) as root:
    _ = (
        root
        | beam.Create([_INPUT.value])
        | beam.FlatMap(open_dataset)
        | beam.Reshuffle()
        | beam.Map(
            render_at_time,
            _INPUT.value,
            _RESOLUTION.value,
            _EXTINCTION_COEFF_SUN.value,
            _EXTINCTION_COEFF_VIEW.value,
            _SCATTERING_COEFF.value,
            sun_dir,
            view_dir,
            _OUTPUT.value,
        )
    )


if __name__ == '__main__':
  app.run(main)
