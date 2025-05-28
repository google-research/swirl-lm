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

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A base data class for the lookup tables of atmospheric optical properties."""

import collections
import dataclasses
import json
from typing import Callable, Dict, Optional

from swirl_lm.physics.radiation.config import radiative_transfer_pb2
from swirl_lm.physics.radiation.optics import constants
from swirl_lm.physics.radiation.optics import optics_utils
from swirl_lm.utility import file_io
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal
OrderedDict = collections.OrderedDict


@dataclasses.dataclass(frozen=True)
class LookupVolumeMixingRatio:
  """Lookup table of volume mixing ratio profiles of atmospheric gases."""

  # Volume mixing ratio (vmr) global mean of predominant atmospheric gas
  # species, keyed by chemical formula.
  global_means: types.TensorMap
  # Volume mixing ratio profiles, keyed by chemical formula.
  profiles: Optional[types.TensorMap] = None

  @classmethod
  def from_proto(
      cls,
      proto: radiative_transfer_pb2.AtmosphericState,
  ) -> 'LookupVolumeMixingRatio':
    """Instantiates a `LookupVolumeMixingRatio` object from a proto.

    The proto contains atmospheric conditions, the path to a json file
    containing globally averaged volume mixing ratio for various gas species,
    and the path to a file containing the volume mixing ratio sounding data for
    certain gas species. The gas species will be identified by their chemical
    formula in lowercase (e.g., 'h2o`, 'n2o', 'o3'). Each entry of the profile
    corresponds to the pressure level under 'p_ref', which is a required column.

    Args:
      proto: An instance of `radiative_transfer_pb2.AtmosphericState`.

    Returns:
      A `LookupVolumeMixingRatio` object.
    """
    vmr_sounding = (
        file_io.parse_csv_file(proto.vmr_sounding_filepath)
        if proto.HasField('vmr_sounding_filepath')
        else None
    )
    profiles = None
    if vmr_sounding is not None:
      assert (
          'p_ref' in vmr_sounding
      ), f'Missing p_ref column in sounding file {proto.vmr_sounding_filepath}'
      profiles = {
          key: tf.constant(values, dtype=tf.float32)
          for key, values in vmr_sounding.items()
      }

    # Dry air is a special case that always has a volume mixing ratio of 1
    # since, by definition, vmr is normalized by the number of moles of dry air.
    global_means = {
        constants.DRY_AIR_KEY: constants.DRY_AIR_VMR,
    }
    if proto.HasField('vmr_global_mean_filepath'):
      with tf.io.gfile.GFile(proto.vmr_global_mean_filepath, 'r') as f:
        global_means.update(json.loads(f.read()))

    kwargs = dict(
        global_means=global_means,
        profiles=profiles,
    )
    return cls(**kwargs)

  def _vmr_interpolant_fn(
      self,
      p_for_interp: tf.Tensor,
      vmr_profile: tf.Tensor,
  ) -> Callable[[tf.Tensor], tf.Tensor]:
    """Creates a volume mixing ratio interpolant for the given profile."""

    def interpolant_fn(p: tf.Tensor):
      interp = optics_utils.create_linear_interpolant(
          tf.math.log(p), tf.math.log(p_for_interp)
      )
      return optics_utils.interpolate(
          vmr_profile, OrderedDict({'p': lambda _: interp})
      )

    return interpolant_fn

  def reconstruct_vmr_fields_from_pressure(
      self,
      pressure: FlowFieldVal,
  ) -> Dict[str, FlowFieldVal]:
    """Reconstructs volume mixing ratio fields for a given pressure field.

    The volume mixing ratio fields are reconstructed for the gas species that
    have spatially variable profiles available from sounding data.

    Args:
      pressure: The pressure field, in Pa.

    Returns:
      A dictionary keyed by chemical formula of volume mixing ratio fields
      interpolated to the 3D grid.
    """
    if self.profiles is not None:
      p_for_interp = self.profiles['p_ref']

      return {
          k: tf.nest.map_structure(
              self._vmr_interpolant_fn(p_for_interp, profile), pressure
          )
          for k, profile in self.profiles.items()
          if k != 'p_ref'
      }

    return {}
