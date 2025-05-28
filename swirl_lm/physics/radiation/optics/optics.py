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

"""Implementations of `OpticsScheme`s and a factory method."""

import typing
from typing import Any, Callable, Dict, Optional, Sequence
from absl import logging
import numpy as np
from swirl_lm.physics.radiation.config import radiative_transfer_pb2
from swirl_lm.physics.radiation.optics import cloud_optics
from swirl_lm.physics.radiation.optics import constants
from swirl_lm.physics.radiation.optics import gas_optics
from swirl_lm.physics.radiation.optics import lookup_cloud_optics as cloud_lookup_lib
from swirl_lm.physics.radiation.optics import lookup_gas_optics_base
from swirl_lm.physics.radiation.optics import lookup_gas_optics_longwave
from swirl_lm.physics.radiation.optics import lookup_gas_optics_shortwave
from swirl_lm.physics.radiation.optics import lookup_volume_mixing_ratio
from swirl_lm.physics.radiation.optics import optics_base
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf


AbstractLookupGasOptics = lookup_gas_optics_base.AbstractLookupGasOptics
LookupCloudOptics = cloud_lookup_lib.LookupCloudOptics
LookupGasOpticsLongwave = lookup_gas_optics_longwave.LookupGasOpticsLongwave
LookupGasOpticsShortwave = lookup_gas_optics_shortwave.LookupGasOpticsShortwave
LookupVolumeMixingRatio = lookup_volume_mixing_ratio.LookupVolumeMixingRatio
FlowFieldMap = types.FlowFieldMap
FlowFieldVal = types.FlowFieldVal


class RRTMOptics(optics_base.OpticsScheme):
  """The Rapid Radiative Transfer Model (RRTM) optics scheme implementation."""

  def __init__(
      self,
      vmr_lib: LookupVolumeMixingRatio,
      params: radiative_transfer_pb2.OpticsParameters,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      g_dim: int,
      halos: int,
  ):
    super().__init__(params, kernel_op, g_dim, halos)
    rrtm_params = params.rrtm_optics
    self.vmr_lib = vmr_lib
    self.cloud_optics_lw = LookupCloudOptics.from_nc_file(
        rrtm_params.cloud_longwave_nc_filepath
    )
    self.cloud_optics_sw = LookupCloudOptics.from_nc_file(
        rrtm_params.cloud_shortwave_nc_filepath
    )
    self.gas_optics_lw = LookupGasOpticsLongwave.from_nc_file(
        rrtm_params.longwave_nc_filepath
    )
    self.gas_optics_sw = LookupGasOpticsShortwave.from_nc_file(
        rrtm_params.shortwave_nc_filepath
    )

  @tf.function
  def _od_fn_graph(
      self,
      is_lw: bool,
      igpt: tf.Tensor,
      molecules: tf.Tensor,
      temperature: tf.Tensor,
      pressure: tf.Tensor,
      vmr_fields: Optional[Dict[int, tf.Tensor]]
  ) -> tf.Tensor:
    """The actual optical_depth calculation as a graph."""
    logging.info('Calling optical depth graph.')
    lookup_gas_optics = self.gas_optics_lw if is_lw else self.gas_optics_sw
    return (gas_optics.compute_minor_optical_depth(
        lookup_gas_optics,
        self.vmr_lib,
        molecules,
        temperature,
        pressure,
        igpt,
        vmr_fields,
    ) + gas_optics.compute_major_optical_depth(
        lookup_gas_optics,
        self.vmr_lib,
        molecules,
        temperature,
        pressure,
        igpt,
        vmr_fields,
    ))

  def _optical_depth_fn(
      self, igpt: tf.Tensor, is_lw: bool
  ):
    """Creates a callable Tensorflow graph for computing the optical depth.

    The returned callable graph exposes a signature that is in terms of
    `tf.Tensor`s and is less likely to be unnecessarily retraced by the
    compiler.

    Args:
      igpt: The g-point that will be used to index into the RRTMGP lookup table.
      is_lw: If `True`, uses the longwave lookup. Otherwise, uses the shortwave
        lookup.

    Returns:
      A callable Tensorflow graph that computes the optical depth given fields
      for the number of molecules per area, pressure, and volume mixing ratio of
      various gases.
    """
    def od_fn(
        molecules,
        temperature,
        pressure,
        vmr_fields
    ) -> tf.Tensor:
      return self._od_fn_graph(
          is_lw, igpt, molecules, temperature, pressure, vmr_fields)

    return od_fn

  @tf.function
  def _rayl_fn_graph(
      self,
      igpt: tf.Tensor,
      molecules: tf.Tensor,
      temperature: tf.Tensor,
      pressure: tf.Tensor,
      vmr_fields: Optional[Dict[int, tf.Tensor]]
  ) -> tf.Tensor:
    """The actual Rayleigh scattering calculation as a graph."""
    logging.info('Calling Rayleigh scattering graph.')
    return gas_optics.compute_rayleigh_optical_depth(
        self.gas_optics_sw,
        self.vmr_lib,
        molecules,
        temperature,
        pressure,
        igpt,
        vmr_fields,
    )

  def rayleigh_scattering_fn(self, igpt: tf.Tensor):
    """Creates a callable Tensorflow graph for computing Rayleigh scattering.

    Args:
      igpt: The g-point that will be used to index into the RRTMGP lookup table.

    Returns:
      A callable Tensorflow graph that computes the Rayleigh scattering optical
      depth given fields for the number of molecules per area, temperature,
      pressure, and volume mixing ratio of various gases.
    """
    def rayl_fn(
        molecules,
        temperature,
        pressure,
        vmr_fields,
    ) -> tf.Tensor:
      return self._rayl_fn_graph(
          igpt, molecules, temperature, pressure, vmr_fields)

    return rayl_fn

  @tf.function
  def _pf_fn_graph(
      self,
      igpt: tf.Tensor,
      pressure: tf.Tensor,
      temperature: tf.Tensor,
      vmr_fields: Optional[Dict[int, tf.Tensor]] = None,
  ) -> tf.Tensor:
    """The actual Planck fraction calculation as a graph."""
    logging.info('Calling Planck fraction graph.')
    return gas_optics.compute_planck_fraction(
        self.gas_optics_lw,
        self.vmr_lib,
        pressure,
        temperature,
        igpt,
        vmr_fields,
    )

  def planck_fraction_fn(
      self,
      igpt: tf.Tensor,
  ):
    """Creates a callable Tensorflow graph for computing the Planck fraction.

    Args:
      igpt: The g-point that will be used to index into the RRTMGP lookup table.

    Returns:
      A callable Tensorflow graph that computes the Planck fraction given fields
      for pressure, temperature and volume mixing ratio of various gases.
    """
    def pf_fn(
        pressure: tf.Tensor,
        temperature: tf.Tensor,
        vmr_fields: Optional[Dict[int, tf.Tensor]] = None,
    ) -> tf.Tensor:
      return self._pf_fn_graph(igpt, pressure, temperature, vmr_fields)
    return pf_fn

  @tf.function
  def _ps_fn_graph(
      self,
      igpt: tf.Tensor,
      planck_fraction: tf.Tensor,
      temperature: tf.Tensor,
  ) -> tf.Tensor:
    """The actual Planck source calculation as a graph."""
    logging.info('Calling Planck source graph.')
    return gas_optics.compute_planck_sources(
        self.gas_optics_lw,
        planck_fraction,
        temperature,
        igpt,
    )

  def planck_src_fn(
      self,
      igpt: tf.Tensor,
  ):
    """Creates a callable Tensorflow graph for computing the Planck source.

    Args:
      igpt: The g-point that will be used to index into the RRTMGP lookup table.

    Returns:
      A callable Tensorflow graph that computes the Planck source given fields
      for the precomputed pointwise Planck fraction and temperature.
    """
    def ps_fn(
        planck_fraction: tf.Tensor,
        temperature: tf.Tensor,
    ) -> tf.Tensor:
      return self._ps_fn_graph(igpt, planck_fraction, temperature)
    return ps_fn

  @tf.function
  def _cloud_props_graph(
      self, ibnd, is_lw, r_eff_liq, cloud_path_liq, r_eff_ice, cloud_path_ice
  ):
    """The actual cloud optical properties calculation as a graph."""
    logging.info('Calling cloud optical properties graph.')
    cloud_lookup = self.cloud_optics_lw if is_lw else self.cloud_optics_sw
    return cloud_optics.compute_optical_properties(
        cloud_lookup,
        cloud_path_liq,
        cloud_path_ice,
        r_eff_liq,
        r_eff_ice,
        ibnd=ibnd,
    )

  def cloud_properties_fn(
      self,
      ibnd: tf.Tensor,
      is_lw: bool,
  ):
    """Creates callable Tensorflow graph for computing cloud optical properties.

    Args:
      ibnd: The spectral band index that will be used to index into the lookup
        tables for cloud absorption coefficients.
      is_lw: If `True`, uses the longwave lookup. Otherwise, uses the shortwave
        lookup.

    Returns:
      A callable Tensorflow graph that returns a dictionary containing the cloud
        optical depth, single-scattering albedo, and asymmetry factor.
    """
    def cloud_props_fn(
        r_eff_liq, cloud_path_liq, r_eff_ice, cloud_path_ice
    ):
      return self._cloud_props_graph(
          ibnd,
          is_lw,
          r_eff_liq,
          cloud_path_liq,
          r_eff_ice,
          cloud_path_ice,
      )
    return cloud_props_fn

  def _map_fn(
      self,
      fn: Callable[..., tf.Tensor],
      *args: FlowFieldVal | Dict[Any, FlowFieldVal],
  ) -> FlowFieldVal | FlowFieldMap | list[tf.Tensor]:
    """Mimics `tf.nest.map_structure` with support for dictionary arguments.

    This assumes that the fields, including those in dictionary values, are
    either all `tf.Tensor`s or all sequences of `tf.Tensor`s, but not both.

    Args:
      fn: The function that will be called with `args`.
      *args: An arbitrary number of `FlowFieldVal` or dictionary arguments
        containing `FlowFieldVal`s as values.

    Returns:
      A `FlowFieldVal` that is the result of applying `fn` to the argument list
      as is, if the fields are represented by `tf.Tensor`s; or to each level of
      the inputs if the fields are represented by `Sequence[tf.Tensor]`.
    """
    # Extract a single argument variable and determine its type.
    single_input = [arg for arg in args if arg is not None][0]
    if isinstance(single_input, Dict):
      single_input = single_input.values()[0]
    field_is_tensor = isinstance(single_input, tf.Tensor)

    assert field_is_tensor or isinstance(
        single_input, Sequence
    ), 'Inputs must be either `tf.Tensor` or `Sequence`.'

    if field_is_tensor:
      return fn(*args)

    def extract_arg(
        arg: FlowFieldVal | Dict[Any, FlowFieldVal], i: int
    ) -> tf.Tensor | Dict[Any, tf.Tensor]:
      """Extracts the `tf.Tensor` or dictionary elements at index `i`."""
      if arg is None:
        return None
      elif isinstance(arg, Dict):
        return {k: v[i] for k, v in arg.items()}
      elif isinstance(arg, Sequence):
        return arg[i]
      else:
        raise ValueError(f'Unsupported nested input type: {type(arg)}')

    n = len(single_input)
    # Split arguments along z-list and apply the function to each level
    # separately.
    split_inputs = [[extract_arg(arg, i) for arg in args] for i in range(n)]
    return [fn(*split_inputs[i]) for i in range(n)]

  def _apply_delta_scaling_for_cloud(
      self,
      cloud_optical_props: FlowFieldMap,
  ) -> FlowFieldMap:
    """Delta-scales optical properties for shortwave bands."""
    def delta_scale_tau(
        tau: tf.Tensor, ssa: tf.Tensor, g: tf.Tensor
    ) -> tf.Tensor:
      wf = ssa * g**2
      return (1.0 - wf) * tau

    def delta_scale_ssa(ssa: tf.Tensor, g: tf.Tensor) -> tf.Tensor:
      wf = ssa * g**2
      return (ssa - wf) / tf.maximum(self._EPSILON, 1.0 - wf)

    def delta_scale_asy(g: tf.Tensor) -> tf.Tensor:
      f = g**2
      return (g - f) / tf.maximum(self._EPSILON, 1.0 - f)

    cloud_tau = tf.nest.map_structure(
        delta_scale_tau,
        cloud_optical_props['optical_depth'],
        cloud_optical_props['ssa'],
        cloud_optical_props['asymmetry_factor'],
    )
    cloud_ssa = tf.nest.map_structure(
        delta_scale_ssa,
        cloud_optical_props['ssa'],
        cloud_optical_props['asymmetry_factor'],
    )
    cloud_asy = tf.nest.map_structure(
        delta_scale_asy, cloud_optical_props['asymmetry_factor']
    )
    return {
        'optical_depth': cloud_tau,
        'ssa': cloud_ssa,
        'asymmetry_factor': cloud_asy,
    }

  def _combine_gas_and_cloud_properties(
      self,
      igpt: tf.Tensor,
      optical_props: FlowFieldMap,
      is_lw: bool,
      radius_eff_liq: Optional[FlowFieldVal] = None,
      cloud_path_liq: Optional[FlowFieldVal] = None,
      radius_eff_ice: Optional[FlowFieldVal] = None,
      cloud_path_ice: Optional[FlowFieldVal] = None,
  ) -> FlowFieldMap:
    """Combines the gas optical properties with the cloud optical properties."""
    gas_lookup = self.gas_optics_lw if is_lw else self.gas_optics_sw
    cloud_states = [
        radius_eff_liq,
        cloud_path_liq,
        radius_eff_ice,
        cloud_path_ice,
    ]
    cloud_states = [
        x
        if x is not None
        else tf.nest.map_structure(tf.zeros_like, optical_props['ssa'])
        for x in cloud_states
    ]
    ibnd = gas_lookup.g_point_to_bnd[igpt]

    compute_cloud_properties_fn = self.cloud_properties_fn(ibnd, is_lw)

    cloud_optical_props = tf.nest.map_structure(
        compute_cloud_properties_fn, *cloud_states
    )
    if isinstance(cloud_optical_props, Sequence):
      # Unnest dictionary.
      cloud_optical_props = {
          k: [cloud_prop[k] for cloud_prop in cloud_optical_props]
          for k in optical_props
      }
    if not is_lw:
      cloud_optical_props = self._apply_delta_scaling_for_cloud(
          cloud_optical_props
      )
    return self.combine_optical_properties(optical_props, cloud_optical_props)

  def compute_lw_optical_properties(
      self,
      pressure: FlowFieldVal,
      temperature: FlowFieldVal,
      molecules: FlowFieldVal,
      igpt: tf.Tensor,
      vmr_fields: Optional[Dict[int, FlowFieldVal]] = None,
      cloud_r_eff_liq: Optional[FlowFieldVal] = None,
      cloud_path_liq: Optional[FlowFieldVal] = None,
      cloud_r_eff_ice: Optional[FlowFieldVal] = None,
      cloud_path_ice: Optional[FlowFieldVal] = None,
  ) -> FlowFieldMap:
    """Computes the monochromatic longwave optical properties.

    Uses the RRTM optics scheme to compute the longwave optical depth, albedo,
    and asymmetry factor. These raw optical properties can be further
    transformed downstream to better suit the assumptions of the particular
    radiative transfer solver being used.

    Args:
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules / m^2].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m²].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m²].

    Returns:
      A dictionary containing (for a single g-point):
        'optical_depth': The longwave optical depth.
        'ssa': The longwave single-scattering albedo.
        'asymmetry_factor': The longwave asymmetry factor.
    """
    optical_depth_fn = self._optical_depth_fn(
        igpt, True
    )
    optical_depth_lw = self._map_fn(
        optical_depth_fn,
        molecules,
        temperature,
        pressure,
        vmr_fields,
    )
    optical_depth_lw = typing.cast(FlowFieldVal, optical_depth_lw)

    zeros = tf.nest.map_structure(tf.zeros_like, optical_depth_lw)
    optical_props = {
        'optical_depth': optical_depth_lw,
        'ssa': zeros,
        'asymmetry_factor': zeros,
    }
    if cloud_path_liq is not None or cloud_path_ice is not None:
      return self._combine_gas_and_cloud_properties(
          igpt,
          optical_props,
          is_lw=True,
          radius_eff_liq=cloud_r_eff_liq,
          cloud_path_liq=cloud_path_liq,
          radius_eff_ice=cloud_r_eff_ice,
          cloud_path_ice=cloud_path_ice,
      )
    return optical_props

  def compute_sw_optical_properties(
      self,
      pressure: FlowFieldVal,
      temperature: FlowFieldVal,
      molecules: FlowFieldVal,
      igpt: tf.Tensor,
      vmr_fields: Optional[Dict[int, FlowFieldVal]] = None,
      cloud_r_eff_liq: Optional[FlowFieldVal] = None,
      cloud_path_liq: Optional[FlowFieldVal] = None,
      cloud_r_eff_ice: Optional[FlowFieldVal] = None,
      cloud_path_ice: Optional[FlowFieldVal] = None,
  ) -> FlowFieldMap:
    """Computes the monochromatic shortwave optical properties.

    Uses the RRTM optics scheme to compute the shortwave optical depth, albedo,
    and asymmetry factor. These raw optical properties can be further
    transformed downstream to better suit the assumptions of the particular
    radiative transfer solver being used.

    Args:
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules / m^2].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m²].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m²].

    Returns:
      A dictionary containing (for a single g-point):
        'optical_depth': The shortwave optical depth.
        'ssa': The shortwave single-scattering albedo.
        'asymmetry_factor': The shortwave asymmetry factor.
    """
    optical_depth_fn = self._optical_depth_fn(
        igpt, False
    )
    optical_depth_sw = self._map_fn(
        optical_depth_fn,
        molecules,
        temperature,
        pressure,
        vmr_fields,
    )

    rayl_fn = self.rayleigh_scattering_fn(igpt)
    rayleigh_scattering = self._map_fn(
        rayl_fn, molecules, temperature, pressure, vmr_fields,
    )
    optical_depth_sw = tf.nest.map_structure(
        tf.math.add, optical_depth_sw, rayleigh_scattering
    )
    ssa = tf.nest.map_structure(
        tf.math.divide_no_nan, rayleigh_scattering, optical_depth_sw
    )
    gas_optical_props = {
        'optical_depth': optical_depth_sw,
        'ssa': ssa,
        'asymmetry_factor': tf.nest.map_structure(tf.zeros_like, ssa),
    }
    if cloud_path_liq is not None or cloud_path_ice is not None:
      return self._combine_gas_and_cloud_properties(
          igpt,
          gas_optical_props,
          is_lw=False,
          radius_eff_liq=cloud_r_eff_liq,
          cloud_path_liq=cloud_path_liq,
          radius_eff_ice=cloud_r_eff_ice,
          cloud_path_ice=cloud_path_ice,
      )
    return gas_optical_props

  def compute_planck_sources(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      pressure: FlowFieldVal,
      temperature: FlowFieldVal,
      igpt: tf.Tensor,
      vmr_fields: Optional[Dict[int, FlowFieldVal]] = None,
      sfc_temperature: Optional[FlowFieldVal] = None,
  ) -> FlowFieldMap:
    """Computes the monochromatic Planck sources given the atmospheric state.

    This requires interpolating the temperature at cell faces using a high-order
    scheme.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      sfc_temperature: The optional surface temperature [K] represented as
        either a 3D `tf.Tensor` or as a list of 2D `tf.Tensor`s but having a
        single vertical dimension.

    Returns:
      A dictionary containing the Planck source at the cell center
      (`planck_src`), the top cell boundary (`planck_src_top`), the bottom cell
      boundary (`planck_src_bottom`) and, if a `sfc_temperature` argument was
      provided, the surface cell boundary (`planck_src_sfc`). Note that the
      surface source will only be valid for the replicas in the first
      computational layer, as the local temperature field is used to compute it.
    """
    temperature_bottom, temperature_top = self._reconstruct_face_values(
        replica_id, replicas, temperature
    )

    planck_fraction = self._map_fn(
        self.planck_fraction_fn(igpt),
        pressure,
        temperature,
        vmr_fields,
    )

    temperature_fields = {
        'planck_src': temperature,
        'planck_src_top': temperature_top,
        'planck_src_bottom': temperature_bottom,
    }
    planck_srcs = {}
    planck_src_fn = self.planck_src_fn(igpt)
    for k, temperature in temperature_fields.items():
      planck_srcs[k] = tf.nest.map_structure(
          planck_src_fn,
          planck_fraction,
          temperature,
      )

    def slice_bottom(f: FlowFieldVal):
      """Extracts the first fluid layer for surface calculations."""
      f1 = common_ops.get_face(f, self._g_dim, face=0, index=self._halos)
      return f1[0] if isinstance(f, tf.Tensor) or self._g_dim != 2 else f1

    if sfc_temperature is not None:
      planck_fraction_0 = slice_bottom(planck_fraction)
      planck_src_sfc = tf.nest.map_structure(
          planck_src_fn,
          planck_fraction_0,
          sfc_temperature,
      )
      # Only allow the first computational layer of cores to have a nonzero
      # surface Planck source.
      core_idx = common_ops.get_core_coordinate(replicas, replica_id)[
          self._g_dim
      ]
      planck_srcs['planck_src_sfc'] = tf.cond(
          pred=tf.equal(core_idx, 0),
          true_fn=lambda: planck_src_sfc,
          false_fn=lambda: tf.nest.map_structure(tf.zeros_like, planck_src_sfc)
      )
    return planck_srcs

  @property
  def n_gpt_lw(self) -> int:
    """The number of g-points in the longwave bands."""
    return self.gas_optics_lw.n_gpt

  @property
  def n_gpt_sw(self) -> int:
    """The number of g-points in the shortwave bands."""
    return self.gas_optics_sw.n_gpt

  @property
  def solar_fraction_by_gpt(self) -> tf.Tensor:
    """Mapping from g-point to the fraction of total solar radiation."""
    return self.gas_optics_sw.solar_src_scaled


class GrayAtmosphereOptics(optics_base.OpticsScheme):
  """Implementation of the gray atmosphere optics scheme."""

  def __init__(
      self,
      params: radiative_transfer_pb2.OpticsParameters,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      g_dim: int,
      halos: int,
  ):
    super().__init__(params, kernel_op, g_dim, halos)
    self._p0 = params.gray_atmosphere_optics.p0
    self._alpha = params.gray_atmosphere_optics.alpha
    self._d0_lw = params.gray_atmosphere_optics.d0_lw
    self._d0_sw = params.gray_atmosphere_optics.d0_sw
    self._g_dim = g_dim
    self._grad_central = (
        lambda f: kernel_op.apply_kernel_op_x(f, 'kDx'),
        lambda f: kernel_op.apply_kernel_op_y(f, 'kDy'),
        lambda f: kernel_op.apply_kernel_op_z(f, 'kDz', 'kDzsh'),
    )[g_dim]

  def compute_lw_optical_properties(
      self,
      pressure: FlowFieldVal,
      *args,
      **kwargs,
  ) -> FlowFieldMap:
    """Computes longwave optical properties based on pressure and lapse rate.

    See Schneider 2004, J. Atmos. Sci. (2004) 61 (12): 1317–1340.
    DOI: https://doi.org/10.1175/1520-0469(2004)061<1317:TTATTS>2.0.CO;2
    To obtain the local optical depth of the layer, the expression for
    cumulative optical depth (from the top of the atmosphere to an arbitrary
    pressure level) was differentiated with respect to the pressure and
    multiplied by the pressure difference across the grid cell.

    Args:
      pressure: The pressure field [Pa].
      *args: Miscellaneous inherited arguments.
      **kwargs: Miscellaneous inherited keyword arguments.

    Returns:
      A dictionary containing the optical depth (`optical_depth`), the single-
      scattering albedo (`ssa`), and the asymmetry factor (`asymmetry_factor`)
      for longwave radiation.
    """

    def tau_fn(p, dp):
      """Computes the pointwise optical depth as a function of pressure only."""
      return tf.math.abs(
          self._alpha
          * self._d0_lw
          * tf.math.pow(p / self._p0, self._alpha)
          / p
          * dp
      )

    dp = tf.nest.map_structure(
        lambda dp_: dp_ / 2.0, self._grad_central(pressure)
    )

    return {
        'optical_depth': tf.nest.map_structure(tau_fn, pressure, dp),
        'ssa': tf.nest.map_structure(tf.zeros_like, pressure),
        'asymmetry_factor': tf.nest.map_structure(tf.zeros_like, pressure),
    }

  def compute_sw_optical_properties(
      self,
      pressure,
      *args,
      **kwargs,
  ):
    """Computes the shortwave optical properties of a gray atmosphere.

    See O'Gorman 2008, Journal of Climate Vol 21, Page(s): 3815–3832.
    DOI: https://doi.org/10.1175/2007JCLI2065.1. In particular, the cumulative
    optical depth expression shown in equation 3 inside the exponential is
    differentiated with respect to pressure and scaled by the pressure
    difference across the grid cell.

    Args:
      pressure: The pressure field [Pa].
      *args: Miscellaneous inherited arguments.
      **kwargs: Miscellaneous inherited keyword arguments.

    Returns:
      A dictionary containing the optical depth (`optical_depth`), the single-
      scattering albedo (`ssa`), and the asymmetry factor (`asymmetry_factor`)
      for shortwave radiation.
    """
    def tau_fn(p, dp):
      return tf.math.abs(2.0 * self._d0_sw * (p / self._p0) * (dp / self._p0))

    dp = tf.nest.map_structure(
        lambda dp_: dp_ / 2.0, self._grad_central(pressure)
    )
    optical_depth = tf.nest.map_structure(tau_fn, pressure, dp)

    return {
        'optical_depth': optical_depth,
        'ssa': tf.nest.map_structure(tf.zeros_like, optical_depth),
        'asymmetry_factor': tf.nest.map_structure(tf.zeros_like, optical_depth),
    }

  def compute_planck_sources(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      pressure: FlowFieldVal,
      temperature: FlowFieldVal,
      *args,
      sfc_temperature: Optional[FlowFieldVal] = None,
  ) -> FlowFieldMap:
    """Computes the Planck sources used in the longwave problem.

    The computation is based on Stefan-Boltzmann's law, which states that the
    thermal radiation emitted from a black body is directly proportional to the
    4-th power of its absolute temperature.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      *args: Miscellaneous inherited arguments.
      sfc_temperature: The optional surface temperature [K] represented as
        either a 3D `tf.Tensor` or as a list of 2D `tf.Tensor`s but having a
        single vertical dimension.

    Returns:
      A dictionary containing the Planck source at the cell center
      (`planck_src`), the top cell boundary (`planck_src_top`), and the bottom
      cell boundary (`planck_src_bottom`).
    """
    del pressure

    def src_fn(t: tf.Tensor) -> tf.Tensor:
      return constants.STEFAN_BOLTZMANN * t**4 / np.pi

    temperature_bottom, temperature_top = self._reconstruct_face_values(
        replica_id, replicas, temperature,
    )

    planck_srcs = {
        'planck_src': tf.nest.map_structure(src_fn, temperature),
        'planck_src_top': tf.nest.map_structure(src_fn, temperature_top),
        'planck_src_bottom': tf.nest.map_structure(src_fn, temperature_bottom),
    }
    if sfc_temperature is not None:
      planck_srcs['planck_src_sfc'] = tf.nest.map_structure(
          src_fn, sfc_temperature
      )
    return planck_srcs

  @property
  def n_gpt_lw(self) -> int:
    """The number of g-points in the longwave bands."""
    return 1

  @property
  def n_gpt_sw(self) -> int:
    """The number of g-points in the shortwave bands."""
    return 1

  @property
  def solar_fraction_by_gpt(self) -> tf.Tensor:
    """Mapping from g-point to the fraction of total solar radiation."""
    return tf.constant([1.0], dtype=tf.float32)


def optics_factory(
    params: radiative_transfer_pb2.OpticsParameters,
    kernel_op: get_kernel_fn.ApplyKernelOp,
    g_dim: int,
    halos: int,
    vmr_lib: Optional[LookupVolumeMixingRatio] = None,
) -> optics_base.OpticsScheme:
  """Constructs an instance of `OpticsScheme`.

  Args:
    params: The optics parameters.
    kernel_op: An object holding a library of kernel operations.
    g_dim: The vertical dimension.
    halos: The number of halo layers.
    vmr_lib: An instance of `LookupVolumeMixingRatio` containing gas
      concentrations.

  Returns:
    An instance of `OpticsScheme`.
  """
  if params.HasField('rrtm_optics'):
    assert vmr_lib is not None, '`vmr_lib` is required for `RRTMOptics`.'
    return RRTMOptics(vmr_lib, params, kernel_op, g_dim=g_dim, halos=halos)
  elif params.HasField('gray_atmosphere_optics'):
    return GrayAtmosphereOptics(params, kernel_op, g_dim=g_dim, halos=halos)
  else:
    raise ValueError('Unsupported optics scheme.')
