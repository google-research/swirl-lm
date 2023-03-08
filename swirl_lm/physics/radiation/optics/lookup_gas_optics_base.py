# Copyright 2022 The swirl_lm Authors.
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

"""A base data class for the lookup tables of atmospheric optical properties."""

import abc
import dataclasses
import os
import tempfile
from typing import Any, Dict, Text, Tuple
import zipfile

import netCDF4 as nc
import numpy as np
from swirl_lm.utility import types
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class LookupGasOpticsBase(metaclass=abc.ABCMeta):
  """Lookup table of gases' optical properties in the longwave bands."""
  # Number of gases used in the lookup table.
  n_gases: int
  # Number of frequency bands.
  n_bnd: int
  # Number of `g-points`.
  n_gpt: int
  # Number of atmospheric layers (=2, lower and upper atmospheres).
  n_atmos_layers: int
  # Number of reference temperatures for absorption coefficient lookup table.
  n_t_ref: int
  # Number of reference pressures for absorption lookup table.
  n_p_ref: int
  # Number of reference binary mixing fractions, for absorption coefficient
  # lookup table.
  n_mixing_fraction: int
  # Number of major absorbing gases
  n_maj_absrb: int
  # Number of minor absorbing gases
  n_minor_absrb: int
  # Number of minor absorbers in lower atmosphere
  n_minor_absrb_lower: int
  # Number of minor absorbers in upper atmosphere
  n_minor_absrb_upper: int
  # Number of minor contributors in the lower atmosphere
  n_contrib_lower: int
  # Number of minor contributors in the upper atmosphere
  n_contrib_upper: int
  # Reference pressure separating upper and lower atmosphere
  p_ref_tropo: tf.Tensor
  # Reference temperature
  t_ref_absrb: tf.Tensor
  # Reference pressure
  p_ref_absrb: tf.Tensor
  # minimum pressure supported by RRTMGP lookup tables
  p_ref_min: tf.Tensor
  # Δt for reference temperature values (Δt is constant)
  dtemp: tf.Tensor
  # Δ for log of reference pressure values (Δp is constant)
  dln_p: tf.Tensor
  # major absorbing species in each band `(2, n_atmos_layers, n_bnd)`
  key_species: tf.Tensor
  # major absorption coefficient `(n_gpt, n_η, n_p_ref, n_t_ref)`
  kmajor: tf.Tensor
  # minor absorption coefficient in lower atmosphere `(n_contrib_lower, n_η,
  # n_t_ref)`
  kminor_lower: tf.Tensor
  # minor absorption coefficient in upper atmosphere `(n_contrib_upper, n_η,
  # n_t_ref)`
  kminor_upper: tf.Tensor
  # starting and ending `g-point` for each band `(2, n_bnd)`
  bnd_lims_gpt: tf.Tensor
  # starting and ending wavenumber for each band `(2, n_bnd)`
  bnd_lims_wn: tf.Tensor
  # `g-point` limits for minor contributors in lower atmosphere `(2,
  # n_contrib_lower)`
  minor_lower_gpt_lims: tf.Tensor
  # `g-point` limits for minor contributors in upper atmosphere `(2,
  # n_contrib_upper)`
  minor_upper_gpt_lims: tf.Tensor
  # minor gas (lower atmosphere) scales with density? `(n_minor_absrb_lower)`
  minor_lower_scales_with_density: tf.Tensor
  # minor gas (upper atmosphere) scales with density? `(n_minor_absrb_upper)`
  minor_upper_scales_with_density: tf.Tensor
  # minor gas (lower atmosphere) scales by compliment `(n_minor_absrb_lower)`
  lower_scale_by_complement: tf.Tensor
  # minor gas (upper atmosphere) scales by compliment `(n_minor_absrb_upper)`
  upper_scale_by_complement: tf.Tensor
  # reference pressures used by the lookup table `(n_p_ref)`
  p_ref: tf.Tensor
  # reference temperatures used by the lookup table `(n_t_ref)`
  t_ref: tf.Tensor
  # reference volume mixing ratios used by the lookup table `(2, n_gases,
  # n_t_ref)`
  vmr_ref: tf.Tensor

  @classmethod
  def _load_data(
      cls, tables: types.TensorMap,
      dims: types.DimensionMap,
  ) -> Dict[Text, Any]:
    p_ref = tables['press_ref']
    t_ref = tables['temp_ref']
    p_ref_min = tf.math.reduce_min(p_ref)
    dtemp = t_ref[1] - t_ref[0]
    dln_p = tf.math.log(p_ref[0]) - tf.math.log(p_ref[1])
    return dict(
        n_gases=dims['absorber'],
        n_bnd=dims['bnd'],
        n_gpt=dims['gpt'],
        n_atmos_layers=dims['atmos_layer'],
        n_t_ref=dims['temperature'],
        n_p_ref=dims['pressure'],
        n_maj_absrb=dims['absorber'],
        n_minor_absrb=dims['minor_absorber'],
        n_minor_absrb_lower=dims['minor_absorber_intervals_lower'],
        n_minor_absrb_upper=dims['minor_absorber_intervals_upper'],
        n_contrib_lower=dims['contributors_lower'],
        n_contrib_upper=dims['contributors_upper'],
        n_mixing_fraction=dims['mixing_fraction'],
        p_ref_tropo=tables['press_ref_trop'],
        t_ref_absrb=tables['absorption_coefficient_ref_T'],
        p_ref_absrb=tables['absorption_coefficient_ref_P'],
        key_species=tables['key_species'],
        kmajor=tables['kmajor'],
        kminor_lower=tables['kminor_lower'],
        kminor_upper=tables['kminor_upper'],
        bnd_lims_gpt=tables['bnd_limits_gpt'],
        bnd_lims_wn=tables['bnd_limits_wavenumber'],
        minor_lower_gpt_lims=tables['minor_limits_gpt_lower'],
        minor_upper_gpt_lims=tables['minor_limits_gpt_upper'],
        minor_lower_scales_with_density=tables[
            'minor_scales_with_density_lower'
        ],
        minor_upper_scales_with_density=tables[
            'minor_scales_with_density_upper'
        ],
        lower_scale_by_complement=tables['scale_by_complement_lower'],
        upper_scale_by_complement=tables['scale_by_complement_upper'],
        p_ref=p_ref,
        t_ref=t_ref,
        p_ref_min=p_ref_min,
        dtemp=dtemp,
        dln_p=dln_p,
        vmr_ref=tables['vmr_ref'],
    )

  @classmethod
  def _reload_tensors_as_single_graph_nodes(
      cls,
      tf_dict_in: types.TensorMap,
  ) -> types.TensorMap:
    """Uses tf.io API to load lookup tensors as single nodes in the TF graph."""
    tf_dict_out = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
      for k, v in tf_dict_in.items():
        fname = os.path.join(tmp_dir, k)
        serialized = tf.io.serialize_tensor(v)
        tf.io.write_file(fname, serialized)
        new_tensor = tf.io.parse_tensor(
            tf.io.read_file(fname), out_type=v.dtype
        )
        tf_dict_out.update({k: new_tensor})
    return tf_dict_out

  @classmethod
  def _parse_nc_file(
      cls,
      path: Text
  ) -> Tuple[types.TensorMap, types.DimensionMap]:
    """Utility function for unpacking the lookup files and loading tensors."""
    filename = os.path.basename(path)
    with tempfile.TemporaryDirectory() as tmp_dir:
      with zipfile.ZipFile(path, 'r') as zip_file:
        zip_file.extractall(tmp_dir)
        filename = os.path.splitext(filename)[0]
        extracted_path = os.path.join(tmp_dir, filename + '.nc')
      ds = nc.Dataset(extracted_path, 'r')

    tensor_dict = {}
    dim_map = {k: v.size for k, v in ds.dimensions.items()}

    for key in ds.variables:
      val = ds[key][:].data
      if val.dtype == np.float64:
        val = val.astype(np.float32)
      tensor_dict.update({key: tf.convert_to_tensor(val)})
    tensor_dict = cls._reload_tensors_as_single_graph_nodes(tensor_dict)
    return (tensor_dict, dim_map)

  @classmethod
  @abc.abstractmethod
  def from_nc_file(
      cls, path: Text
  ) -> 'LookupGasOpticsBase':
    """Loads lookup tables from NetCDF files and populates the attributes."""
