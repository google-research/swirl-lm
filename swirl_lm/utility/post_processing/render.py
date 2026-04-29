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

"""Volume rendering via ray marching.

The implementation is based on ideas from these two sources:

1. [patapom]
http://patapom.com/topics/Revision2013/Revision%202013%20-%20Real-time%20Volumetric%20Rendering%20Course%20Notes.pdf

2. [mocz]
https://medium.com/swlh/create-your-own-volume-rendering-with-python-655ca839b097

To summarize:

  1. We first cast rays from the sun into the atmosphere. At each grid cell
     some of the light gets scattered, and some of it passes through. We collect
     the amount of scattering at each grid cell.

  2. We cast rays from the viewing point into the atmosphere. For each ray,
     we compute the amount of light visible along that ray. Currently we only
     take into account uniformly scattered sun light, (e.g., no ambient light).
     This gives us the total amount of light at each pixel (corresponding to
     each ray) and the transparency of the atmosphere along each ray.

  3. To render, we first plot the sun light (i.e., shades/light) on the earth's
     surface as a plane.  Then we blend the light from step 2 using the
     transparency onto this surface.
"""
import itertools
import logging
from typing import Union

from matplotlib import colors as mcolors
from matplotlib import transforms as mtransforms
import matplotlib.axes as mpl_axes
import numpy as np
import scipy
from scipy.spatial import transform as sp_transform
import xarray


class Transform:
  """3D transformation using homogeneous coordinates.

  See: https://en.wikipedia.org/wiki/Homogeneous_coordinates
  """

  def __init__(self, transform_matrix: np.typing.ArrayLike):
    assert (
        transform_matrix.ndim == 2
    ), f'Expected 2D matrix but got ndim={transform_matrix.ndim}'
    assert (
        transform_matrix.shape[0] == transform_matrix.shape[1]
    ), f'Expected square matrix but got shape={transform_matrix.shape}'
    self.matrix = np.asarray(transform_matrix, dtype=np.float64)

  def apply(self, points: np.typing.ArrayLike) -> np.ndarray:
    """Applies the transformation to points in homogeneous coords.

    Args:
      points: Nx4 array of N points in homogeneous coords.

    Returns:
      Nx4 array of points in homogeneous coords.
    """
    points = np.asarray(points)
    assert (
        points.ndim == 2
    ), f'Expected 2D array of points but got shape {points.shape}'
    assert points.shape[1] == 4, (
        'Expected points to be in homogeneous coordinates (with 4D coords) '
        f'but got {points.shape[1]} dimensions.'
    )
    return (self.matrix @ points.T).T

  def apply_3d(self, points: np.typing.ArrayLike) -> np.ndarray:
    """Returns the result of applying the transformation in cartesian coords.

    Args:
      points: Nx4 array of N points in homogeneous coords.

    Returns:
      Nx3 array of points in cartesian coords.
    """
    out = self.apply(points)
    return out[:, :3] / out[:, 3:]

  def inverse(self) -> 'Transform':
    return Transform(np.linalg.inv(self.matrix))


def compose_transforms(*transforms: Transform) -> Transform:
  """Composes Transforms into a single Transform.

  Args:
    *transforms: Zero or more transformations to be composed.

  Returns:
    Composition of the given transformations or the identity transformation
    if no transformation is given.
  """
  m = np.identity(4)
  for transform in transforms:
    m = m @ transform.matrix
  return Transform(m)


def transform_from_rotation(rotation: sp_transform.Rotation) -> Transform:
  """Returns the Transform for the given rotation.

  Args:
    rotation: The input rotation.

  Returns:
    The Transform for the rotation.
  """
  x = np.concatenate([rotation.as_matrix(), np.zeros((1, 3))], axis=0)
  return Transform(
      np.concatenate([x, [[0], [0], [0], [1]]], axis=1).astype(np.float32)
  )


def transform_from_scale(
    *, sx: float = 1, sy: float = 1, sz: float = 1
) -> Transform:
  """Returns the Transform for the given scaling transformation.

  Args:
    sx: The scaling factor in x.
    sy: The scaling factor in y.
    sz: The scaling factor in z.

  Returns:
    The Transform for scaling.
  """
  return Transform(
      np.array(
          [[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]],
          dtype=np.float32,
      )
  )


def transform_from_translation(
    xyz: Union[list[float], np.ndarray],
) -> Transform:
  """Returns the Transform for the given translation transformation.

  Args:
    xyz: The translation vector.

  Returns:
    The Transform for translation.
  """
  return Transform(
      np.array(
          [
              [1, 0, 0, xyz[0]],
              [0, 1, 0, xyz[1]],
              [0, 0, 1, xyz[2]],
              [0, 0, 0, 1],
          ],
          dtype=np.float32,
      )
  )


def partition_by_key(
    keys: np.typing.ArrayLike,
) -> list[tuple[np.typing.ArrayLike, np.ndarray]]:
  """Partitions unique values in keys and returns their indices.

  For example, given [10, 20, 10, 20, 20] returns [(10, [0, 2]), (20, [1, 3,
  4])]. This is useful for re-distributing (rectangular) slices to workers when
  changing coordinate systems.

  Args:
    keys: Numpy array of values, which will be passed to np.unique to determine
      unique partition keys. np.unique is called with axis=0 so if the array
      is n-dimensional, each (n-1)-dimensional row is a key.

  Returns:
    List of (key, array of indices) where key is a unique value in keys and
    array of indices contains the indices into keys for that key. Note that the
    type of key is the same as the type of elements in keys, e.g., it is
    np.ndarray if keys is multi-dimensional and np.int64 if keys is a list of
    ints.
  """
  # Handle the empty input case specially because when we apply np.diff below
  # we lose the distinction between one partition vs no partitions.
  if not keys:
    return []
  partitions, index = np.unique(keys, axis=0, return_inverse=True)
  sorted_index = np.argsort(index)
  partition_ends = np.nonzero(np.diff(index[sorted_index]))[0] + 1
  out = []
  start = 0
  for i, end in enumerate(itertools.chain(partition_ends, [len(sorted_index)])):
    out.append((partitions[i], sorted_index[start:end]))
    start = end
  return out


def align_coords(
    i_to_c: Transform, coords_list: list[xarray.DataArray]
) -> tuple[Transform, np.ndarray]:
  """Aligns coordinate system i_to_c to the bounds in coords_list.

  Args:
    i_to_c: A Transform that maps indices to physical coordinates.
    coords_list: x, y, z coordinates in physical space.

  Returns:
    (transform, [size_x, size_y, size_z]) where transform is a 4x4 affine
    transformation
    matrix and such that the region defined by coords_list lies in 0 <= i <=
    size_x, 0 <= j <= size_y, .... in the space defined by transform.
  """
  c_to_i = i_to_c.inverse()
  corners = []
  for i, j, k in itertools.product(*([(0, -1)] * 3)):
    corners.append(
        c_to_i.apply(
            [[coords_list[0][i], coords_list[1][j], coords_list[2][k], 1]]
        )[0]
    )
  corners = np.array(corners)
  min_ijk = np.min(corners, axis=0) - 1
  max_ijk = np.max(corners, axis=0) + 1
  return (
      compose_transforms(
          transform_from_translation(-min_ijk), c_to_i
      ).inverse(),
      np.ceil(max_ijk - min_ijk)[:3].astype(int),
  )


def regrid(x: xarray.DataArray, sz: float) -> np.ndarray:
  """Computes 1D coordinates spaced ~sz units covering the same region as x."""
  count = int(np.ceil((x.max() - x.min()) / sz))
  return np.linspace(x.min().item(), x.max().item(), count)


def compute_sun_color(
    v: xarray.DataArray,
    extinction_coeff: float,
    sun_transform: Transform,
    volume_size: np.ndarray,
    grid_size: float,
) -> xarray.DataArray:
  """Casts rays from the sun and computes scattering. See [patapom].

  Args:
    v: 3D variable representing the density of the volume to be rendered. The
      density together with extinction_coeff determines how much light is
      transmitted through the volume (as opposed to scattered or absorbed). For
      rendering clouds, this is q_c.
    extinction_coeff: Coefficient used to compute how much light is transmitted.
      Higher coefficients mean less light is transmitted.
    sun_transform: Transform that defines a coordinate system whose z axis is
      aligned with the direction of the sun.
    volume_size: 3D vector of ints. Sun color will be computed in a grid defined
      by applying sun_transform to all points with integer coordinates between
      corners [0, 0, 0] and volume_size.
    grid_size: Size of the output grid in physical units. The output is
      interpolated back into the coordinate system of v. grid_size determines
      the spacing of the output grid. Normally, it should match the grid size
      determined by sun_transform.

  Returns:
    3D DataArray that contains the computed sun color.
  """
  sun_color = []
  # pylint: disable=g-complex-comprehension
  start_grid = np.stack(
      [
          x.flatten()
          for x in np.meshgrid(
              range(volume_size[0]),
              range(volume_size[1]),
              [0],
              [1],
              indexing='ij',
          )
      ],
      axis=-1,
  )

  current_sun_color = np.ones(volume_size[:2])
  for _ in range(volume_size[2]):
    points = sun_transform.apply_3d(start_grid)
    # Interpolate to the new grid. See [mocz].
    density = scipy.interpolate.interpn(
        (v.x.data, v.y.data, v.z.data),
        v.transpose('x', 'y', 'z').data,
        points,
        method='linear',
        bounds_error=False,
        fill_value=0,
    )
    density = np.reshape(density, volume_size[:2])
    current_sun_color *= np.exp(-extinction_coeff * np.maximum(0.0, density))
    sun_color.append(current_sun_color.copy())
    start_grid += [0, 0, 1, 0]

  sun_color = np.stack(sun_color, axis=-1)

  xn, yn, zn = (
      regrid(v.x, grid_size),
      regrid(v.y, grid_size),
      regrid(v.z, grid_size),
  )
  orig_grid = np.stack(
      [x.flatten() for x in np.meshgrid(xn, yn, zn, [1], indexing='ij')],
      axis=-1,
  )
  orig_grid = sun_transform.inverse().apply_3d(orig_grid)

  out = scipy.interpolate.interpn(
      [np.arange(x, dtype=np.float32) for x in sun_color.shape],
      sun_color,
      orig_grid,
      method='linear',
      bounds_error=False,
      fill_value=1,
  )

  sun_color = xarray.DataArray(
      np.reshape(out, (len(xn), len(yn), len(zn))), {'x': xn, 'y': yn, 'z': zn}
  )
  return sun_color


def shade(
    v: xarray.DataArray,
    sun_color: xarray.DataArray,
    extinction_coeff: float,
    scattering_coeff: float,
    view_transform: Transform,
    volume_size: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Casts rays from the view point. See [patapom]."""
  # pylint: disable=g-complex-comprehension
  start_grid = np.stack(
      [
          x.flatten()
          for x in np.meshgrid(
              range(volume_size[0]),
              range(volume_size[1]),
              [0],
              [1],
              indexing='ij',
          )
      ],
      axis=-1,
  )

  extinction = np.ones(volume_size[:2], dtype=np.float32)
  scattering = np.zeros(volume_size[:2], dtype=np.float32)
  sun_color_data = sun_color.transpose('x', 'y', 'z').data
  # TODO(bcg): Sun color computation is a specialized version of the code
  # here. Merge them into one.
  for _ in range(volume_size[2]):
    points = view_transform.apply_3d(start_grid)
    density = scipy.interpolate.interpn(
        (v.x.data, v.y.data, v.z.data),
        v.transpose('x', 'y', 'z').data,
        points,
        method='linear',
        bounds_error=False,
        fill_value=0,
    )
    density = np.reshape(density, volume_size[:2])
    sun_color_l = scipy.interpolate.interpn(
        (sun_color.x.data, sun_color.y.data, sun_color.z.data),
        sun_color_data,
        points,
        method='linear',
        bounds_error=False,
        fill_value=1,
    )
    sun_color_l = np.reshape(sun_color_l, volume_size[:2])
    scattering_update = scattering_coeff * density
    extinction *= np.exp(-extinction_coeff * density)
    scattering += scattering_update * extinction * sun_color_l
    start_grid += [0, 0, 1, 0]

  return scattering, extinction


def render(
    sun_dir: sp_transform.Rotation,
    view_dir: sp_transform.Rotation,
    grid_size: float,
    extinction_coeff_sun: float,
    extinction_coeff_view: float,
    scattering_coeff: float,
    v: xarray.DataArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Transform]:
  """Wrapper around compute_sun_color() and shade()."""
  logging.info(
      'volume size: %s %s %s grid_size: %s',
      len(v.x),
      len(v.y),
      len(v.z),
      grid_size,
  )

  sun_transform = transform_from_scale(
      sx=grid_size, sy=grid_size, sz=-grid_size
  )
  sun_transform = compose_transforms(
      transform_from_rotation(sun_dir), sun_transform
  )
  sun_transform, sun_volume_sz = align_coords(sun_transform, [v.x, v.y, v.z])
  sun_color = compute_sun_color(
      v, extinction_coeff_sun * grid_size, sun_transform, sun_volume_sz,
      grid_size
  )
  logging.info('sun color.shape: %s', sun_color.shape)

  view_transform = transform_from_scale(
      sx=grid_size, sy=grid_size, sz=grid_size
  )
  view_transform = compose_transforms(
      transform_from_rotation(view_dir), view_transform
  )
  view_transform, view_volume_size = align_coords(
      view_transform, [v.x, v.y, v.z]
  )
  logging.info('view volume size: %s', view_volume_size)

  scattering, extinction = shade(
      v,
      sun_color,
      extinction_coeff_view * grid_size,
      scattering_coeff * grid_size,
      view_transform,
      view_volume_size,
  )

  surface = sun_color.sel(z=0, method='nearest').data

  surface_transform = compose_transforms(
      transform_from_translation([-v.x[0], -v.y[0], 0]), view_transform
  )
  surface_transform = compose_transforms(
      transform_from_scale(
          sx=(surface.shape[0] - 1) / (v.x[-1] - v.x[0]),
          sy=(surface.shape[1] - 1) / (v.y[-1] - v.y[0]),
      ),
      surface_transform,
  )

  return scattering, extinction, surface, surface_transform.inverse()


def plot(
    scattering: np.ndarray,
    extinction: np.ndarray,
    surface: np.ndarray,
    surface_transform: np.ndarray,
    ax: mpl_axes.Axes,
) -> None:
  """Plots volume rendering data on a matplotlib Axes object."""
  # TODO(bcg): Render sky color more realistically instead of a solid color.
  # Color blue background.
  im_bg = ax.imshow([[[0.01, 0.1, 0.3]]])
  im_bg.set_transform(mtransforms.Affine2D().translate(0.5, 0.5) + ax.transAxes)

  # Plot earth surface.
  cmap_earth = mcolors.LinearSegmentedColormap.from_list(None, ['#020', '#562'])
  im = ax.imshow(surface.T, cmap=cmap_earth, origin='lower', vmin=0, vmax=1)
  im_transform = mtransforms.Affine2D(
      surface_transform.matrix[[0, 1, 3]][:, [0, 1, 3]]
  )
  im.set_transform(im_transform + ax.transData)

  # Undo pre-multiplied alpha because matplotlib doesn't allow pre-multiplied
  # alphas and plot clouds.
  alpha = 1 - np.minimum(extinction, 1)
  scattering = np.divide(scattering, alpha, where=alpha != 0)
  ax.imshow(
      scattering.T,
      alpha=alpha.T,
      cmap='Greys_r',
      origin='lower',
      vmin=0,
      vmax=1,
  )

  ax.set_aspect('equal')
