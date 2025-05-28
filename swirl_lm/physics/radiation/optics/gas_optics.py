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
"""Utility functions for computing optical properties of atmospheric gases."""

import collections
from typing import Dict, Optional, Text

from swirl_lm.physics.radiation.optics import lookup_gas_optics_base
from swirl_lm.physics.radiation.optics import lookup_gas_optics_longwave
from swirl_lm.physics.radiation.optics import lookup_gas_optics_shortwave
from swirl_lm.physics.radiation.optics import lookup_volume_mixing_ratio
from swirl_lm.physics.radiation.optics import optics_utils
import tensorflow as tf

Interpolant = optics_utils.Interpolant
IndexAndWeight = optics_utils.IndexAndWeight
OrderedDict = collections.OrderedDict

AbstractLookupGasOptics = lookup_gas_optics_base.AbstractLookupGasOptics
LookupGasOpticsLongwave = lookup_gas_optics_longwave.LookupGasOpticsLongwave
LookupGasOpticsShortwave = lookup_gas_optics_shortwave.LookupGasOpticsShortwave
LookupVolumeMixingRatio = lookup_volume_mixing_ratio.LookupVolumeMixingRatio

_PASCAL_TO_HPASCAL_FACTOR = 0.01
_M2_TO_CM2_FACTOR = 1e4


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
    vmr_lib: LookupVolumeMixingRatio,
    species_idx: tf.Tensor,
    vmr_fields: Optional[Dict[int, tf.Tensor]] = None,
) -> tf.Tensor:
  """Gets the volume mixing ratio, given major gas species index and pressure.

  Args:
    lookup_gas_optics: An `AbstractLookupGasOptics` object containing an index
      for the gas species.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    species_idx: A `tf.Tensor` containing indices of gas species whose VMR will
      be computed.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    A `tf.Tensor` of the same shape as `species_idx` or `pressure_idx`
    containing the pointwise volume mixing ratios of the corresponding gas
    species at that pressure level.
  """
  idx_gases = lookup_gas_optics.idx_gases
  # Indices of background gases for which a global mean VMR is available.
  vmr_gm = [0.0] * len(idx_gases)
  # Map the gas names in `vmr_lib.global_means` dict to indices consistent with
  # the RRTMGP `key_species` table.
  for k, v in vmr_lib.global_means.items():
    vmr_gm[idx_gases[k]] = v

  vmr = optics_utils.lookup_values(tf.stack(vmr_gm), (species_idx,))

  # Overwrite with available precomputed vmr.
  if vmr_fields is not None:
    for gas_idx, vmr_field in vmr_fields.items():
      vmr = tf.where(
          condition=tf.equal(species_idx, gas_idx),
          x=vmr_field,
          y=vmr,
      )

  assert_vmr_below_one = tf.compat.v1.assert_equal(
      tf.math.reduce_any(tf.math.greater(vmr, 1.0)),
      False,
      message='At least one volume mixing ratio (VMR) is above 1.')
  assert_vmr_nonnegative = tf.compat.v1.assert_equal(
      tf.math.reduce_any(tf.math.less(vmr, 0.0)),
      False,
      message='At least one volume mixing ratio (VMR) is negative.')

  with tf.control_dependencies([assert_vmr_below_one, assert_vmr_nonnegative]):
    return vmr


def _compute_relative_abundance_interpolant(
    lookup_gas_optics: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    troposphere_idx: tf.Tensor,
    temperature_idx: tf.Tensor,
    ibnd: tf.Tensor,
    scale_by_mixture: bool,
    vmr_fields: Optional[Dict[int, tf.Tensor]] = None,
) -> Interpolant:
  """Creates an `Interpolant` object for relative abundance of a major species.

  Args:
    lookup_gas_optics: An `AbstractLookupGasOptics` object containing a RRTMGP
      index for all relevant gas species.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    troposphere_idx: A `tf.Tensor` that is 1 where the corresponding pressure
      level is below the troposphere limit and 0 otherwise. This informs whether
      an offset should be added to the reference pressure indices when indexing
      into the `kmajor` table.
    temperature_idx: A `tf.Tensor` containing indices of reference temperature
      values.
    ibnd: The frequency band for which the relative abundance is computed.
    scale_by_mixture: Whether to scale the weights by the gas mixture.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

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
            vmr_fields,
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
  # Consistent with how the RRTM absorption coefficient tables are designed, the
  # relative abundance defaults to 0.5 when the volume mixing ratio of both
  # dominant species is exactly 0.
  relative_abundance = tf.where(
      condition=tf.greater(combined_vmr, 0.0),
      x=vmr_for_interp[0] / combined_vmr,
      y=0.5 * tf.ones_like(combined_vmr),
  )
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
    molecules: tf.Tensor,
    temperature: tf.Tensor,
    p: tf.Tensor,
    igpt: tf.Tensor,
    vmr_fields: Optional[Dict[int, tf.Tensor]] = None,
) -> tf.Tensor:
  """Computes the optical depth contributions from major gases.

  Args:
    lookup_gas_optics: An `AbstractLookupGasOptics` object containing a RRTMGP
      index for all major gas species.
    vmr: A `LookupVolumeMixingRatio` object containing the volume mixing ratio
      of all relevant atmospheric gases.
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m^2]
    temperature: A `tf.Tensor` containing temperature values (in K).
    p: A `tf.Tensor` containing pressure values (in Pa).
    igpt: The absorption variable index (g-point) for which the optical depth
      is computed.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    A `tf.Tensor` with the pointwise optical depth contributions from the major
    species.
  """
  # Take the troposphere limit into account when indexing into the major species
  # and absorption coefficients.
  # The troposphere index is 1 for levels above the troposphere limit and 0
  # otherwise.
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
  # The frequency band for which the optical depth is computed.
  ibnd = lookup_gas_optics.g_point_to_bnd[igpt]

  def mix_interpolant_fn(dep: Dict[Text, IndexAndWeight]) -> Interpolant:
    """Relative abundance interpolant function that depends on `t` and `p`."""
    return _compute_relative_abundance_interpolant(
        lookup_gas_optics,
        vmr,
        troposphere_idx,
        dep['t'].idx,
        ibnd,
        scale_by_mixture=True,
        vmr_fields=vmr_fields,
    )

  # Interpolant functions ordered according to the axes in `kmajor`.
  interpolant_fn_dict = OrderedDict((
      ('t', lambda _: t_interp),
      ('p', lambda _: p_interp),
      ('m', mix_interpolant_fn),
  ))
  return molecules / _M2_TO_CM2_FACTOR * optics_utils.interpolate(
      lookup_gas_optics.kmajor[..., igpt], interpolant_fns=interpolant_fn_dict
  )


def _compute_minor_optical_depth(
    lookup: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    molecules: tf.Tensor,
    temperature: tf.Tensor,
    p: tf.Tensor,
    igpt: tf.Tensor,
    is_lower_atmosphere: bool,
    vmr_fields: Optional[Dict[int, tf.Tensor]] = None,
) -> tf.Tensor:
  """Computes the optical depth from minor gases given atmosphere region.

  Args:
    lookup: An `AbstractLookupGasOptics` object containing a RRTMGP index for
      all relevant gases and a lookup table for minor absorption coefficients.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m^2]
    temperature: The temperature of the flow field [K].
    p: The pressure field (in Pa).
    igpt: The absorption rank (g-point) index for which the optical depth
      will be computed.
    is_lower_atmosphere: A boolean indicating whether in the lower atmosphere.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    A `tf.Tensor` with the pointwise optical depth contributions from the minor
    species.
  """
  # The troposphere index is 1 for levels above the troposphere limit and 0
  # otherwise.
  tropo_idx = tf.where(
      condition=tf.less_equal(p, lookup.p_ref_tropo),
      x=tf.ones_like(p, dtype=tf.int32),
      y=tf.zeros_like(p, dtype=tf.int32),
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
  minor_bnd_end = (
      lookup.minor_lower_bnd_end
      if is_lower_atmosphere
      else lookup.minor_upper_bnd_end
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

  if vmr_fields is not None and lookup.idx_h2o in vmr_fields:
    dry_factor = 1.0 / (1.0 + vmr_fields[lookup.idx_h2o])
  else:
    dry_factor = 1.0

  def mix_interpolant_fn(dep: Dict[Text, IndexAndWeight]) -> Interpolant:
    """Relative abundance interpolant that depends on `t`."""
    t_idx = dep['t'].idx
    return _compute_relative_abundance_interpolant(
        lookup, vmr_lib, tropo_idx, t_idx, ibnd, False, vmr_fields,
    )

  def scaling_fn(scaling_vmr, dry_factor):
    return lambda: scaling_vmr * dry_factor

  def scaling_by_complement_fn(scaling_vmr, dry_factor):
    return lambda: (1.0 - scaling_vmr * dry_factor)

  def scale_with_gas_fn(i):
    sgas = tf.maximum(idx_scaling_gas[i], 0)
    sgas_idx = sgas * tf.ones_like(tropo_idx)
    scaling_vmr = get_vmr(lookup, vmr_lib, sgas_idx, vmr_fields)
    scaling = tf.cond(
        pred=tf.equal(scale_by_complement[i], 1),
        true_fn=scaling_by_complement_fn(scaling_vmr, dry_factor),
        false_fn=scaling_fn(scaling_vmr, dry_factor),
    )
    return lambda: scaling

  def scale_with_density_fn(i):
    scaling = _PASCAL_TO_HPASCAL_FACTOR * p / temperature
    scaling *= tf.cond(
        pred=tf.greater(idx_scaling_gas[i], 0),
        true_fn=scale_with_gas_fn(i),
        false_fn=lambda: tf.ones_like(p),
    )
    return lambda: scaling

  # Optical depth will be aggregated over all the minor absorbers contributing
  # to the frequency band.
  def step_fn(i, tau_minor):
    # Map the minor contributor to the RRTMGP gas index.
    gas_idx = idx_gases_minor[i] * tf.ones_like(tropo_idx)
    vmr_minor = get_vmr(lookup, vmr_lib, gas_idx, vmr_fields)
    scaling = vmr_minor * molecules / _M2_TO_CM2_FACTOR
    scaling *= tf.cond(
        pred=tf.equal(minor_scales_with_density[i], 1),
        true_fn=scale_with_density_fn(i),
        false_fn=lambda: tf.ones_like(p),
    )
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
    return i + 1, tau_minor

  def stop_condition(i, tau_minor):
    del tau_minor
    return tf.logical_and(
        tf.less_equal(i, minor_bnd_end[ibnd]),
        tf.less(i, minor_absorber_intervals),
    )
  minor_start_idx = minor_bnd_start[ibnd]
  i0 = tf.cond(
      tf.greater_equal(minor_start_idx, 0),
      true_fn=lambda: minor_start_idx,
      false_fn=lambda: minor_absorber_intervals
  )
  tau_minor_0 = tf.zeros_like(temperature)
  return tf.nest.map_structure(
      tf.stop_gradient,
      tf.while_loop(
          cond=stop_condition,
          body=step_fn,
          loop_vars=(i0, tau_minor_0),
      ),
  )[1]


def compute_minor_optical_depth(
    lookup: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    molecules: tf.Tensor,
    temperature: tf.Tensor,
    p: tf.Tensor,
    igpt: tf.Tensor,
    vmr_fields: Optional[Dict[int, tf.Tensor]] = None,
) -> tf.Tensor:
  """Computes the optical depth contributions from minor gases.

  Args:
    lookup: An instance of `AbstractLookupGasOptics` containing a RRTMGP index
      for all relevant gases and a lookup table for minor absorption
      coefficients.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m^2]
    temperature: The temperature of the flow field [K].
    p: The pressure field (in Pa).
    igpt: The absorption rank (g-point) index for which the optical depth
      will be computed.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    A `tf.Tensor` with the pointwise optical depth contributions from the minor
    species.
  """
  # The troposphere index is 1 for levels above the troposphere limit and 0
  # otherwise.
  def minor_tau(is_lower_atmos: bool) -> tf.Tensor:
    """Computes the minor optical depth assuming an atmosphere level."""
    return _compute_minor_optical_depth(
        lookup,
        vmr_lib,
        molecules,
        temperature,
        p,
        igpt,
        is_lower_atmos,
        vmr_fields,
    )

  return tf.where(
      condition=tf.greater(p, lookup.p_ref_tropo),
      x=minor_tau(is_lower_atmos=True),
      y=minor_tau(is_lower_atmos=False),
  )


def compute_rayleigh_optical_depth(
    lkp: LookupGasOpticsShortwave,
    vmr_lib: LookupVolumeMixingRatio,
    molecules: tf.Tensor,
    temperature: tf.Tensor,
    p: tf.Tensor,
    igpt: tf.Tensor,
    vmr_fields: Optional[Dict[int, tf.Tensor]] = None,
) -> tf.Tensor:
  """Computes the optical depth contribution from Rayleigh scattering.

  Args:
    lkp: An instance of `AbstractLookupGasOptics` containing a RRTMGP index
      for all relevant gases and a lookup table for Rayleigh absorption
      coefficients.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m^2].
    temperature: Temperature variable (in K).
    p: The pressure field (in Pa).
    igpt: The absorption variable index (g-point) for which the optical depth
      will be computed.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    A `tf.Tensor` with the pointwise optical depth contributions from Rayleigh
      scattering.
  """
  # The troposphere index is 1 for levels above the troposphere limit and 0
  # otherwise.
  tropo_idx = tf.where(
      condition=tf.less_equal(p, lkp.p_ref_tropo),
      x=tf.ones_like(p, dtype=tf.int32),
      y=tf.zeros_like(p, dtype=tf.int32),
  )
  temperature_interpolant = optics_utils.create_linear_interpolant(
      temperature, lkp.t_ref
  )
  ibnd = lkp.g_point_to_bnd[igpt]

  def mix_interpolant_fn(dep: Dict[Text, IndexAndWeight]) -> Interpolant:
    """Relative abundance interpolant function that depends on `t` and `p`."""
    t_idx = dep['t'].idx
    return _compute_relative_abundance_interpolant(
        lkp, vmr_lib, tropo_idx, t_idx, ibnd, False, vmr_fields,
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
  if vmr_fields is not None and lkp.idx_h2o in vmr_fields:
    factor = 1.0 + vmr_fields[lkp.idx_h2o]
  else:
    factor = 1.0

  return (
      factor
      * molecules
      / _M2_TO_CM2_FACTOR
      * tf.where(
          condition=tf.equal(tropo_idx, 1),
          x=rayl_tau_upper,
          y=rayl_tau_lower,
      )
  )


def compute_planck_fraction(
    lookup: LookupGasOpticsLongwave,
    vmr_lib: LookupVolumeMixingRatio,
    p: tf.Tensor,
    temperature: tf.Tensor,
    igpt: tf.Tensor,
    vmr_fields: Optional[Dict[int, tf.Tensor]] = None,
) -> tf.Tensor:
  """Computes the Planck fraction that will be used to weight the Planck source.

  Args:
    lookup: An `LookupGasOpticsLongwave` object containing a RRTMGP index for
      all relevant gases and a lookup table for the Planck source.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    p: The pressure of the flow field [Pa].
    temperature: The temperature at the grid cell center [K].
    igpt: The absorption rank (g-point) index for which the optical depth will
      be computed.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    The pointwise Planck fraction associated with the temperature field.
  """
  # The troposphere index is 1 for levels above the troposphere limit and 0
  # otherwise.
  tropo_idx = tf.where(
      condition=tf.less_equal(p, lookup.p_ref_tropo),
      x=tf.ones_like(p, dtype=tf.int32),
      y=tf.zeros_like(p, dtype=tf.int32),
  )
  temperature_interpolant = optics_utils.create_linear_interpolant(
      temperature, lookup.t_ref
  )
  pressure_interpolant = _pressure_interpolant(
      p, lookup.p_ref, tropo_idx
  )
  ibnd = lookup.g_point_to_bnd[igpt]

  def mix_interpolant_fn(dep: Dict[Text, IndexAndWeight]) -> Interpolant:
    """Relative abundance interpolant function that depends on `temperature`."""
    return _compute_relative_abundance_interpolant(
        lookup,
        vmr_lib,
        tropo_idx,
        dep['t'].idx,
        ibnd,
        False,
        vmr_fields,
    )

  interpolants_fns = OrderedDict((
      ('t', lambda _: temperature_interpolant),
      ('p', lambda _: pressure_interpolant),
      ('m', mix_interpolant_fn),
  ))

  # 3-D interpolation of the Planck fraction.
  return optics_utils.interpolate(
      lookup.planck_fraction[..., igpt], interpolants_fns
  )


def compute_planck_sources(
    lookup: LookupGasOpticsLongwave,
    planck_fraction: tf.Tensor,
    temperature: tf.Tensor,
    igpt: tf.Tensor,
) -> tf.Tensor:
  """Computes the Planck source for the longwave problem.

  Args:
    lookup: An `LookupGasOpticsLongwave` object containing a RRTMGP index for
      all relevant gases and a lookup table for the Planck source.
    planck_fraction: The Planck fraction that scales the Planck source.
    temperature: The temperature [K] for which the Planck source will be
      computed.
    igpt: The absorption rank (g-point) index for which the optical depth will
      be computed.

  Returns:
    The planck source emanating from the points with given `temperature` [W/m²].
  """
  ibnd = lookup.g_point_to_bnd[igpt]

  # 1-D interpolation of the Planck source.
  interpolant = optics_utils.create_linear_interpolant(
      temperature, lookup.t_planck
  )
  return planck_fraction * optics_utils.interpolate(
      lookup.totplnk[ibnd, :], OrderedDict({'t': lambda _: interpolant})
  )
