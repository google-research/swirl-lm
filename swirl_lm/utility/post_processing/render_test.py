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

import os

from absl.testing import absltest
import matplotlib.pyplot as plt
import numpy as np
import PIL
from scipy.spatial import transform as sp_transform
from swirl_lm.utility.post_processing import render
import xarray



# Fallback for open-source: resolve testdata relative to this file.
if not globals().get('_get_testdata_path'):

  def _get_testdata_path(filename: str) -> str:  # pylint: disable=function-redefined
    return os.path.join(os.path.dirname(__file__), 'test_data', filename)


class TransformTest(absltest.TestCase):

  def test_apply(self):
    transform = render.transform_from_scale(sx=2)
    np.testing.assert_allclose(
        transform.apply([[1.0, 3, 4, 1]]), [[2.0, 3, 4, 1]]
    )

  def test_apply_3d(self):
    transform = render.transform_from_scale(sx=2)
    np.testing.assert_allclose(
        transform.apply_3d([[1.0, 3, 4, 1]]), [[2.0, 3, 4]]
    )

  def test_inverse(self):
    transform = render.transform_from_scale(sx=2)
    np.testing.assert_allclose(
        transform.inverse().apply([[1.0, 3, 4, 1]]), [[0.5, 3, 4, 1]]
    )

  def test_compose_empty_list(self):
    transform = render.compose_transforms()
    np.testing.assert_allclose(
        transform.apply([[1.0, 3, 4, 1]]), [[1.0, 3, 4, 1]]
    )

  def test_compose(self):
    transform = render.compose_transforms(
        render.transform_from_scale(sx=2),
        render.transform_from_scale(sy=3),
        render.transform_from_scale(sz=4),
    )
    np.testing.assert_allclose(
        transform.apply([[1.0, 3, 4, 1]]), [[2.0, 9, 16, 1]]
    )


def numpy_to_python(x):
  # Copies a nested tuple/list/ndarray structure while converting numpy
  # arrays to python lists by calling tolist().
  if isinstance(x, tuple):
    return tuple(numpy_to_python(e) for e in x)
  elif isinstance(x, list):
    return [numpy_to_python(e) for e in x]
  elif isinstance(x, np.ndarray):
    return x.tolist()
  else:
    return x


class PartitionByKeyTest(absltest.TestCase):

  def assertCountEqualNumpy(self, left, right):
    self.assertCountEqual(numpy_to_python(left), numpy_to_python(right))

  def test_partition_empty(self):
    self.assertEmpty(render.partition_by_key([]))

  def test_partition_by_simple_key(self):
    self.assertCountEqualNumpy(
        render.partition_by_key([10, 11, 10, 11, 11]),
        [(10, [0, 2]), (11, [1, 3, 4])],
    )

  def test_partition_one(self):
    self.assertCountEqualNumpy(render.partition_by_key([10]), [(10, [0])])

  def test_partition_by_list_key(self):
    keys = [[10, 0], [10, 1], [11, 0], [10, 1], [11, 0]]
    self.assertCountEqualNumpy(
        render.partition_by_key(keys),
        [([10, 0], [0]), ([10, 1], [1, 3]), ([11, 0], [2, 4])],
    )


def create_cloud(v, x0, y0, z0, xr, yr, zr, max_v):
  x, y, z = v.x, v.y, v.z
  d = np.sqrt(
      ((x - x0) / xr) ** 2 + ((y - y0) / yr) ** 2 + ((z - z0) / zr) ** 2
  )
  return xarray.where(
      d < 1, max_v * np.cos(0.5 * np.pi * d) ** 2, xarray.zeros_like(v)
  )


def create_scene():
  """Creates a small scene with two ellipsoidal clouds."""
  shape = 101, 101, 21
  data = xarray.DataArray(
      np.zeros(shape=shape),
      coords={
          'x': np.linspace(0, 120e3, shape[0]),
          'y': np.linspace(0, 120e3, shape[1]),
          'z': np.linspace(0, 20e3, shape[2]),
      },
  )
  data += create_cloud(data, 70e3, 60e3, 12e3, 30e3, 10e3, 3e3, 0.0014)
  data += create_cloud(data, 50e3, 70e3, 4e3, 20e3, 10e3, 3e3, 0.0014)
  return data


class RenderTest(absltest.TestCase):

  def assert_images_same(self, image1_filename, image2_filenames):
    diffs = []
    try:
      image1 = PIL.Image.open(image1_filename)
    except IOError:
      diffs.append(f'Failed to open first file `{image1_filename}`.')
    image2_list = []
    for image2_filename in image2_filenames:
      try:
        image2_list.append(PIL.Image.open(image2_filename))
      except IOError:
        diffs.append(f'Failed to open second file `{image2_filename}`.')
    if diffs:
      self.fail('\n'.join(diffs))
      return
    image1_data = np.asarray(image1)
    assert image1_data.dtype == np.uint8, (
        f'Expected image data to be uint8, got {image1_data.dtype} for '
        '{image1_filename}.'
    )
    found_match = False
    for image2 in image2_list:
      image2_data = np.asarray(image2)
      assert image2_data.dtype == np.uint8, (
          f'Expected image data to be uint8, got {image2_data.dtype} for '
          '{image2_filename}.'
      )
      # Now that we know image data is in the range 0-255, we can safely compare
      # with an absolute tolerance of 1. (If the data is float, the range could
      # be 0-1, in which case absolute tolerance of 1 would not detect changes.)
      if np.allclose(image1_data, image2_data, atol=1):
        found_match = True
        break
    if not found_match:
      self.fail('No matching images found.')

  def test_render(self):
    sun_dir = sp_transform.Rotation.from_euler('XZ', [60, 15], degrees=True)
    view_dir = sp_transform.Rotation.from_euler('xz', [75, -40], degrees=True)

    qc = create_scene()
    qc_max = qc.max().item()

    scattering, extinction, surface, surface_transform = render.render(
        sun_dir, view_dir, 1000, 2e-4 / qc_max, 2e-4 / qc_max, 4e-4 / qc_max, qc
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    render.plot(scattering, extinction, surface, surface_transform, ax)

    output_dir = os.environ['TEST_UNDECLARED_OUTPUTS_DIR']
    output_filename = 'render.png'

    output_path = os.path.join(output_dir, 'actual', output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
      fig.savefig(f, format='png', dpi=120)
    plt.close(fig)

    expected_filename1 = _get_testdata_path('render.png')
    expected_filename2 = _get_testdata_path('render_3_10_6.png')
    self.assert_images_same(
        output_path, [expected_filename1, expected_filename2]
    )


if __name__ == '__main__':
  absltest.main()
