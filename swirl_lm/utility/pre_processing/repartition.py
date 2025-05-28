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

"""Interpolates a checkpoint file (ser) and repartitions the data."""

import dataclasses
from typing import Dict, Iterable, Optional, Tuple
import uuid

import numpy as np
from swirl_lm.utility.post_processing import data_processing
import tensorflow as tf
import xarray
import xarray_beam as xbeam


@dataclasses.dataclass
class DataInfo:
  """Information about the mesh and partition of a simulation."""
  # The prefix of the simulation, including the full path and the simulation
  # prefix.
  prefix: str = ''
  # The number of cores of the partition along each dimension. The topology of
  # the partition is denoted as x, y, and z.
  cx: int = 1
  cy: int = 1
  cz: int = 1
  # The full path to the mesh for each dimension of the 3D data tensor. The
  # mesh is for the global 3D tensor without halos (if any).
  mesh_0: str = ''
  mesh_1: str = ''
  mesh_2: str = ''
  # The halo width in three dimensions.
  halo_width_0: int = 0
  halo_width_1: int = 0
  halo_width_2: int = 0
  # The step ID that the data files correspond to.
  step_id: int = 0
  # The orientation of the 3D data tensor with respect to the axis of partition.
  # E.g. 'zxy' refers to the case that the 0th dimension of the data tensor is
  # partitioned along the 'z' axis of the topology, the 1st dimension along x,
  # and the 2nd dimension along y.
  tensor_orientation: str = 'zxy'


FNAME_FMT_SER = '{}-field-{}-xyz-{}-{}-{}-step-{}.ser'
TF_DTYPE = tf.float32


def get_mesh(data_info: DataInfo) -> Dict[str, np.ndarray]:
  """Reads the mesh (in 3 dimensions) from files."""

  def read_mesh(filename: str) -> np.ndarray:
    """Reads a mesh file in txt format."""
    mesh = []
    with tf.io.gfile.GFile(filename, 'r') as f:
      while c := f.readline():
        mesh.append(float(c))
    return np.array(mesh)

  return {
      f'dim_{i}': read_mesh(filename)
      for i, filename in enumerate(
          (data_info.mesh_0, data_info.mesh_1, data_info.mesh_2)
      )
  }


def _dims_to_coords(dims: Dict[str, int], orientation: str) -> Dict[str, int]:
  """Reorders `dims` in (0, 1, 2) to (x, y, z) following `orientation`."""
  return {orientation[i]: dims[f'dim_{i}'] for i in range(3)}


def coords_to_dims(coords: Dict[str, int], orientation: str) -> Dict[str, int]:
  """Reorders `coords` in (x, y, z) corresponding to sequence in orientation."""
  return {f'dim_{i}': coords[orientation[i]] for i in range(3)}


def get_chunks(data_info: DataInfo) -> Dict[str, int]:
  """Finds the chunks for a dataset."""
  coords = get_mesh(data_info)

  data_shape = [len(mesh) for mesh in coords.values()]

  n_cores = coords_to_dims(
      {'x': data_info.cx, 'y': data_info.cy, 'z': data_info.cz},
      data_info.tensor_orientation,
  )
  return {
      f'dim_{i}': int(data_shape[i] // n_cores[f'dim_{i}']) for i in range(3)
  }


def load_source(
    shard_info: Tuple[int, int, int, str],
    source: DataInfo,
    mesh: Dict[str, np.ndarray],
) -> Iterable[Tuple[xbeam.Key, xarray.Dataset]]:
  """Loads a shard of the source data file."""
  core_x, core_y, core_z, varname = shard_info

  filename = FNAME_FMT_SER.format(
      source.prefix,
      varname,
      core_x,
      core_y,
      core_z,
      source.step_id,
  )
  data = data_processing.read_serialized_tensor(filename)
  n_0, n_1, n_2 = data.shape
  data = data[
      source.halo_width_0 : n_0 - source.halo_width_0,
      source.halo_width_1 : n_1 - source.halo_width_1,
      source.halo_width_2 : n_2 - source.halo_width_2,
  ]
  core_id = coords_to_dims(
      {'x': core_x, 'y': core_y, 'z': core_z}, source.tensor_orientation
  )

  key = xbeam.Key(
      {f'dim_{i}': n * core_id[f'dim_{i}'] for i, n in enumerate(data.shape)},
      vars={varname},
  )

  ds = xarray.Dataset(
      {varname: (('dim_0', 'dim_1', 'dim_2'), data)},
      coords={
          dim: mesh[dim][offset : offset + n]
          for n, (dim, offset) in zip(data.shape, key.offsets.items())
      },
  )

  return [(key, ds)]


def write_target_to_ser(
    key: xbeam.Key, ds: xarray.Dataset, target: DataInfo = DataInfo()
):
  """Writes variables in the xarray dataset to ser files."""
  n = get_chunks(target)
  core_id = _dims_to_coords(
      {dim: int(key.offsets[dim] // n_local) for dim, n_local in n.items()},
      target.tensor_orientation,
  )
  halos = (target.halo_width_0, target.halo_width_1, target.halo_width_2)

  for varname in ds.data_vars:
    filename = FNAME_FMT_SER.format(
        target.prefix,
        varname,
        core_id['x'],
        core_id['y'],
        core_id['z'],
        target.step_id,
    )
    tmp_uuid = uuid.uuid4().hex
    tmp_filename = '{}-{}'.format(filename, tmp_uuid)

    val = tf.convert_to_tensor(ds[varname], dtype=TF_DTYPE)
    data_processing.write_serialized_tensor(
        tmp_filename, tf.pad(val, [[halo] * 2 for halo in halos])
    )

    tf.io.gfile.rename(tmp_filename, filename, overwrite=True)


def interpolate(
    key: xbeam.Key,
    ds: xarray.Dataset,
    mesh_target: Optional[Dict[str, np.ndarray]] = None,
) -> Iterable[Tuple[xbeam.Key, xarray.Dataset]]:
  """Interpolates the source dataset onto target mesh.

  Args:
    key: The offset key of the source dataset.
    ds: The source dataset.
    mesh_target: A dictionary specifying the meshes for the data to be
      interpolated onto. It should contain dimensions for interpolation only.

  Returns:
    A sequence of (offset, interpolated dataset) pair.
  """
  if mesh_target is None:
    return [(key, ds)]

  # Becuase we always perform interpolation in dimensions that has all mesh
  # points, the offests in those dimenions are always 0, while the reminding
  # dimensions stay the same as the source.
  offset = xbeam.Key(
      {
          dim: 0 if dim in mesh_target else key.offsets[dim]
          for dim in ('dim_0', 'dim_1', 'dim_2')
      },
      vars=set(ds.data_vars),
  )
  return [(offset, ds.interp(coords=mesh_target))]
