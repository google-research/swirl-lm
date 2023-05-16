# Copyright 2023 The swirl_lm Authors.
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
"""Utility functions for computing optical properties of atmospheric gases."""

import collections
from typing import Dict, Optional, Text

from swirl_lm.physics.radiation.optics import lookup_gas_optics_base
from swirl_lm.physics.radiation.optics import lookup_gas_optics_shortwave
from swirl_lm.physics.radiation.optics import lookup_volume_mixing_ratio
from swirl_lm.physics.radiation.optics import optics_utils
import tensorflow as tf

Interpolant = optics_utils.Interpolant
IndexAndWeight = optics_utils.IndexAndWeight
OrderedDict = collections.OrderedDict

AbstractLookupGasOptics = lookup_gas_optics_base.AbstractLookupGasOptics
LookupGasOpticsShortwave = lookup_gas_optics_shortwave.LookupGasOpticsShortwave
LookupVolumeMixingRatio = lookup_volume_mixing_ratio.LookupVolumeMixingRatio

_PASCAL_TO_HPASCAL_FACTOR = 0.01


def _pressure_interpolant(
    p: tf.Tensor,
    p_ref: tf.Tensor,
    troposphere_offset: Optional[tf.Tensor] = None,
) -> Interpolant:
  """Creates a pressure interpolant based on reference pressure values."""
  log_p = tf.math.log(p)
  log_p_ref = tf.math.log(p_ref)
  return optics_utils.create_linear_interpolant(
      log_p, log_p_ref, offset=troposphere_offset
  )


def _mixing_fraction_interpolant(
    f: tf.Tensor, n_mixing_fraction: int
) -> Interpolant:
  """Creates a mixing fraction interpolant based on desired number of points."""
  return optics_utils.create_linear_interpolant(
      f, tf.linspace(0.0, 1.0, n_mixing_fraction)
  )


def get_vmr(
    lookup_gas_optics: AbstractLookupGasOptics,
    vmr: LookupVolumeMixingRatio,
    species_idx: tf.Tensor,
    pressure_idx: tf.Tensor,
    vmr_water_vapor: Optional[tf.Tensor] = None,
) -> tf.Tensor:
  """Gets the volume mixing ratio, given major gas species index and pressure.

  Args:
    lookup_gas_optics: An `AbstractLookupGasOptics` object containing an index
      for the gas species.
    vmr: A `LookupVolumeMixingRatio` object containing the volume mixing ratio
      of all relevant atmospheric gases. Only ozone and water vapor are assumed
      to have vertically variable mixing ratios.
    species_idx: A `tf.Tensor` containing indices of gas species whose VMR will
      be computed.
    pressure_idx: A `tf.Tensor` holding indices of reference pressure values.
    vmr_water_vapor: An optional `tf.Tensor` containing the water vapor
      volume mixing ratio. If this is provided, the static values for water
        vapor from the `LookupVolumeMixingRatio` data object are ignored.

  Returns:
    A `tf.Tensor` of the same shape as `species_idx` or `pressure_idx`
    containing the pointwise volume mixing ratios of the corresponding gas
    species at that pressure level.
  """
  idx_gases = lookup_gas_optics.idx_gases
  # Indices of background gases for which a global mean VMR is available.
  vmr_gm = [0.0] * len(idx_gases)
  # Map the gas names in `vmr.vmr_gm` dict to indices consistent with the RRTMGP
  # `key_species` table.
  for k, v in vmr.vmr_gm.items():
    vmr_gm[idx_gases[k]] = v

  vmr_water_vapor = (
      vmr_water_vapor
      if vmr_water_vapor is not None
      else optics_utils.lookup_values(vmr.vmr_h2o, (pressure_idx,))
  )

  return tf.where(
      condition=tf.equal(species_idx, lookup_gas_optics.idx_h2o),
      x=vmr_water_vapor,
      y=tf.where(
          condition=tf.equal(species_idx, lookup_gas_optics.idx_o3),
          x=optics_utils.lookup_values(vmr.vmr_o3, (pressure_idx,)),
          y=optics_utils.lookup_values(tf.stack(vmr_gm), (species_idx,))
      )
  )


def compute_relative_abundance_interpolant(
    lookup_gas_optics: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    troposphere_idx: tf.Tensor,
    temperature_idx: tf.Tensor,
    pressure_idx: tf.Tensor,
    ibnd: int,
    scale_by_mixture: bool,
    vmr_water_vapor: Optional[tf.Tensor] = None,
) -> Interpolant:
  """Creates an `Interpolant` object for relative abundance of a major species.

  Args:
    lookup_gas_optics: An `AbstractLookupGasOptics` object containing a RRTMGP
      index for all relevant gas species.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases. Only ozone and water vapor are
      assumed to have vertically variable profiles.
    troposphere_idx: A `tf.Tensor` that is 1 where the corresponding pressure
      level is below the troposphere limit and 0 otherwise. This informs whether
      an offset should be added to the reference pressure indices when indexing
      into the `kmajor` table.
    temperature_idx: A `tf.Tensor` containing indices of reference temperature
      values.
    pressure_idx: A `tf.Tensor` containing indices of reference pressure values.
    ibnd: The frequency band for which the relative abundance is computed.
    scale_by_mixture: Whether to scale the weights by the gas mixture.
    vmr_water_vapor: An optional `tf.Tensor` containing the pointwise volume
      mixing ratio of water vapor.

  Returns:
    An `Interpolant` object for the relative abundance of the major gas species
      in a particular electromagnetic frequency band.
  """
  major_species_idx = []
  vmr_for_interp = []
  vmr_ref = []
  for i in range(2):
    major_species_idx.append(
        optics_utils.lookup_values(
            lookup_gas_optics.key_species[ibnd, :, i], (troposphere_idx,)
        )
    )
    vmr_for_interp.append(
        get_vmr(
            lookup_gas_optics,
            vmr_lib,
            major_species_idx[i],
            pressure_idx,
            vmr_water_vapor,
        )
    )
    vmr_ref.append(
        optics_utils.lookup_values(
            lookup_gas_optics.vmr_ref,
            (temperature_idx, major_species_idx[i], troposphere_idx),
        )
    )
  vmr_ref_ratio = tf.math.divide(vmr_ref[0], vmr_ref[1])
  combined_vmr = vmr_for_interp[0] + vmr_ref_ratio * vmr_for_interp[1]
  relative_abundance = tf.math.divide(vmr_for_interp[0], combined_vmr)
  interpolant = _mixing_fraction_interpolant(
      relative_abundance, lookup_gas_optics.n_mixing_fraction
  )
  if scale_by_mixture:
    interpolant.interp_low.weight *= combined_vmr
    interpolant.interp_high.weight *= combined_vmr
  return interpolant


def compute_major_optical_depth(
    lookup_gas_optics: AbstractLookupGasOptics,
    vmr: LookupVolumeMixingRatio,
    temperature: tf.Tensor,
    p: tf.Tensor,
    igpt: int,
    ibnd: int,
    vmr_water_vapor: Optional[tf.Tensor] = None,
) -> tf.Tensor:
  """Computes the optical depth contributions from major gases.

  Args:
    lookup_gas_optics: An `AbstractLookupGasOptics` object containing a RRTMGP
      index for all major gas species.
    vmr: A `LookupVolumeMixingRatio` object containing the volume mixing ratio
      of all relevant atmospheric gases. Only ozone and water vapor are assumed
      to have vertically variable mixing ratios.
    temperature: A `tf.Tensor` containing temperature values (in K).
    p: A `tf.Tensor` containing pressure values (in Pa).
    igpt: The absorption variable index (g-point) for which the optical depth
      is computed.
    ibnd: The frequency band for which the optical depth is computed.
    vmr_water_vapor: An optional `tf.Tensor` containing the pointwise water
      vapor volume mixing ratio.

  Returns:
    A `tf.Tensor` with the pointwise optical depth contributions from the major
    species.
  """
  # Take the troposphere limit into account when indexing into the major species
  # and absorption coefficients.
  troposphere_idx = tf.where(
      condition=tf.less_equal(p, lookup_gas_optics.p_ref_tropo),
      x=tf.ones_like(p, dtype=tf.int32),
      y=tf.zeros_like(p, dtype=tf.int32),
  )
  t_interp = optics_utils.create_linear_interpolant(
      temperature, lookup_gas_optics.t_ref
  )
  p_interp = _pressure_interpolant(
      p=p, p_ref=lookup_gas_optics.p_ref, troposphere_offset=troposphere_idx
  )

  def mix_interpolant_fn(dep: Dict[Text, IndexAndWeight]) -> Interpolant:
    """Relative abundance interpolant function that depends on `t` and `p`."""
    return compute_relative_abundance_interpolant(
        lookup_gas_optics,
        vmr,
        troposphere_idx,
        dep['t'].idx,
        dep['p'].idx,
        ibnd,
        vmr_water_vapor,
    )

  # Interpolant functions ordered according to the axes in `kmajor`.
  interpolant_fn_dict = OrderedDict((
      ('t', lambda _: t_interp),
      ('p', lambda _: p_interp),
      ('m', mix_interpolant_fn),
  ))
  return optics_utils.interpolate(
      lookup_gas_optics.kmajor[..., igpt], interpolant_fns=interpolant_fn_dict
  )


def compute_minor_optical_depth(
    lookup: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    tropo_idx: tf.Tensor,
    mols: tf.Tensor,
    temperature: tf.Tensor,
    p_idx: tf.Tensor,
    igpt: int,
    is_lower_atmosphere: bool,
    vmr_h2o: Optional[tf.Tensor] = None,
) -> tf.Tensor:
  """Computes the optical depth contributions from minor gases.

  Args:
    lookup: An `AbstractLookupGasOptics` object containing a RRTMGP index for
      all relevant gases and a lookup table for minor absorption coefficients.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases. Only ozone and water vapor are
      assumed to have vertically variable mixing ratios.
    tropo_idx: A `tf.Tensor` that is 1 where the corresponding pressure level is
      below the troposphere limit and 0 otherwise.
    mols: The number of molecules in an atmospheric grid cell per area
      [mols/m^2]
    temperature: The temperature of the flow field [K].
    p_idx: Index of reference pressure values.
    igpt: The absorption rank (g-point) index for which the optical depth
      will be computed.
    is_lower_atmosphere: A boolean indicating whether in the lower atmosphere.
    vmr_h2o: An optional `tf.Tensor` containing the pointwise water vapor volume
      mixing ratio.

  Returns:
    A `tf.Tensor` with the pointwise optical depth contributions from the minor
    species.
  """
  minor_absorber_to_bnd = (
      lookup.minor_lower_bnd if is_lower_atmosphere else lookup.minor_upper_bnd
  )
  minor_absorber_intervals = (
      lookup.n_minor_absrb_lower
      if is_lower_atmosphere
      else lookup.n_minor_absrb_upper
  )
  minor_bnd_start = (
      lookup.minor_lower_bnd_start
      if is_lower_atmosphere
      else lookup.minor_upper_bnd_start
  )
  idx_gases_minor = (
      lookup.idx_minor_gases_lower
      if is_lower_atmosphere
      else lookup.idx_minor_gases_upper
  )
  minor_scales_with_density = (
      lookup.minor_lower_scales_with_density
      if is_lower_atmosphere
      else lookup.minor_upper_scales_with_density
  )
  idx_scaling_gas = (
      lookup.idx_scaling_gases_lower
      if is_lower_atmosphere
      else lookup.idx_scaling_gases_upper
  )
  scale_by_complement = (
      lookup.lower_scale_by_complement
      if is_lower_atmosphere
      else lookup.upper_scale_by_complement
  )
  minor_gpt_shift = (
      lookup.minor_lower_gpt_shift
      if is_lower_atmosphere
      else lookup.minor_upper_gpt_shift
  )
  kminor = lookup.kminor_lower if is_lower_atmosphere else lookup.kminor_upper
  ibnd = lookup.g_point_to_bnd[igpt]

  loc_in_bnd = igpt - lookup.bnd_lims_gpt[ibnd, 0]

  temperature_interpolant = optics_utils.create_linear_interpolant(
      temperature, lookup.t_ref
  )
  p = optics_utils.lookup_values(lookup.p_ref, p_idx)

  dry_factor = 1.0 / (1.0 + vmr_h2o) if vmr_h2o is not None else 1.0

  tau_minor = tf.zeros_like(temperature)

  def mix_interpolant_fn(dep: Dict[Text, IndexAndWeight]) -> Interpolant:
    """Relative abundance interpolant that depends on `t`."""
    t_idx = dep['t'].idx
    return compute_relative_abundance_interpolant(
        lookup, vmr_lib, tropo_idx, t_idx, p_idx, ibnd, False, vmr_h2o
    )

  if ibnd not in minor_bnd_start:
    return tau_minor

  # Optical depth will be aggregated over all the minor absorbers contributing
  # to the g-point and frequency band.
  for i in range(minor_bnd_start[ibnd], minor_absorber_intervals):
    if minor_absorber_to_bnd[i] != ibnd:
      # No additional minor absorbers contributing to band 'ibnd'.
      break
    # Map the minor contributor to the RRTMGP gas index.
    gas_idx = idx_gases_minor[i] * tf.ones_like(tropo_idx)
    vmr_minor = get_vmr(lookup, vmr_lib, gas_idx, p_idx, vmr_h2o)
    scaling = vmr_minor * mols
    if minor_scales_with_density[i] == 1:
      scaling *= _PASCAL_TO_HPASCAL_FACTOR * p / temperature
      if i in idx_scaling_gas:
        sgas = idx_scaling_gas[i]
        sgas_idx = sgas * tf.ones_like(tropo_idx)
        scaling_vmr = get_vmr(lookup, vmr_lib, sgas_idx, p_idx, vmr_h2o)
        if scale_by_complement[i] == 1:
          scaling *= (1.0 - scaling_vmr) * dry_factor
        else:
          scaling *= scaling_vmr * dry_factor
    # Obtain the global contributor index needed to index into the `kminor`
    # table.
    k_loc = minor_gpt_shift[i] + loc_in_bnd

    tau_minor += (
        optics_utils.interpolate(
            kminor[..., k_loc],
            OrderedDict((
                ('t', lambda _: temperature_interpolant),
                ('m', mix_interpolant_fn),
            )),
        )
        * scaling
    )
  return tau_minor


def compute_rayleigh_optical_depth(
    lkp: LookupGasOpticsShortwave,
    vmr_lib: LookupVolumeMixingRatio,
    tropo_idx: tf.Tensor,
    mols: tf.Tensor,
    temperature: tf.Tensor,
    p_idx: tf.Tensor,
    igpt: int,
    vmr_h2o: Optional[tf.Tensor] = None,
) -> tf.Tensor:
  """Computes the optical depth contribution from Rayleigh scattering.

  Args:
    lkp: An `AbstractLookupGasOptics` object containing a RRTMGP
      index for all relevant gases and a lookup table for Rayleigh absorption
      coefficients.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases. Only ozone and water vapor are
      assumed to have vertically variable mixing ratios.
    tropo_idx: A `tf.Tensor` that is 1 where the corresponding pressure
      level is below the troposphere limit and 0 otherwise.
    mols: The number of molecules in an atmospheric grid cell per area
      [mols/m^2].
    temperature: Temperature variable (in K).
    p_idx: Index of reference pressure values.
    igpt: The absorption variable index (g-point) for which the optical depth
      will be computed.
    vmr_h2o: An optional `tf.Tensor` containing the pointwise water
      vapor volume mixing ratio.

  Returns:
    A `tf.Tensor` with the pointwise optical depth contributions from Rayleigh
      scattering.
  """
  temperature_interpolant = optics_utils.create_linear_interpolant(
      temperature, lkp.t_ref
  )
  ibnd = lkp.g_point_to_bnd[igpt]

  def mix_interpolant_fn(dep: Dict[Text, IndexAndWeight]) -> Interpolant:
    """Relative abundance interpolant function that depends on `t` and `p`."""
    t_idx = dep['t'].idx
    return compute_relative_abundance_interpolant(
        lkp, vmr_lib, tropo_idx, t_idx, p_idx, ibnd, False, vmr_h2o
    )

  interpolant_fns = OrderedDict(
      (('t', lambda _: temperature_interpolant), ('m', mix_interpolant_fn))
  )
  rayl_tau_lower = optics_utils.interpolate(
      lkp.rayl_lower[..., igpt], interpolant_fns
  )
  rayl_tau_upper = optics_utils.interpolate(
      lkp.rayl_upper[..., igpt], interpolant_fns
  )
  factor = 1.0 + vmr_h2o if vmr_h2o is not None else 1.0
  return (
      factor
      * mols
      * tf.where(
          condition=tf.equal(tropo_idx, 1),
          x=rayl_tau_upper,
          y=rayl_tau_lower,
      )
  )
