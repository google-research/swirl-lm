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

"""Implementations of `OpticsScheme`s and a factory method."""

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
from swirl_lm.physics.radiation.config import radiative_transfer_pb2
from swirl_lm.physics.radiation.optics import constants
from swirl_lm.physics.radiation.optics import gas_optics
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
    self.gas_optics_lw = LookupGasOpticsLongwave.from_nc_file(
        rrtm_params.longwave_nc_filepath
    )
    self.gas_optics_sw = LookupGasOpticsShortwave.from_nc_file(
        rrtm_params.shortwave_nc_filepath
    )

  def _compute_optical_depth_fn(
      self,
      lookup_gas_optics: AbstractLookupGasOptics,
      mols: tf.Tensor,
      temperature: tf.Tensor,
      pressure: tf.Tensor,
      igpt: int,
      vmr_fields: Optional[Dict[int, tf.Tensor]] = None,
  ) -> tf.Tensor:
    """Computes total optical depth from major and minor gas contributions."""
    major_optical_depth = gas_optics.compute_major_optical_depth(
        lookup_gas_optics,
        self.vmr_lib,
        mols,
        temperature,
        pressure,
        igpt,
        vmr_fields,
    )

    minor_optical_depth = gas_optics.compute_minor_optical_depth(
        lookup_gas_optics,
        self.vmr_lib,
        mols,
        temperature,
        pressure,
        igpt,
        vmr_fields,
    )

    return major_optical_depth + minor_optical_depth

  def _map_fn(
      self,
      fn: Callable[..., tf.Tensor],
      *args: FlowFieldVal | Dict[Any, FlowFieldVal],
  ) -> FlowFieldVal | FlowFieldMap:
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
      the inputs if the the fields are represented by `Sequence[tf.Tensor]`.
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

  def compute_lw_optical_properties(
      self,
      pressure: FlowFieldVal,
      temperature: FlowFieldVal,
      molecules: FlowFieldVal,
      igpt: int,
      vmr_fields: Optional[Dict[int, FlowFieldVal]] = None,
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

    Returns:
      A dictionary containing (for a single g-point):
        'optical_depth': The longwave optical depth.
        'ssa': The longwave single-scattering albedo.
        'asymmetry_factor': The longwave asymmetry factor.
    """

    def optical_depth_lw_fn(molecules, temperature, pressure, vmr_fields):
      return self._compute_optical_depth_fn(
          self.gas_optics_lw,
          molecules,
          temperature,
          pressure,
          igpt,
          vmr_fields,
      )

    optical_depth_lw = self._map_fn(
        optical_depth_lw_fn, molecules, temperature, pressure, vmr_fields,
    )

    zeros = tf.nest.map_structure(tf.zeros_like, optical_depth_lw)
    return {
        'optical_depth': optical_depth_lw,
        'ssa': zeros,
        'asymmetry_factor': zeros,
    }

  def compute_sw_optical_properties(
      self,
      pressure: FlowFieldVal,
      temperature: FlowFieldVal,
      molecules: FlowFieldVal,
      igpt: int,
      vmr_fields: Optional[Dict[int, FlowFieldVal]] = None,
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

    Returns:
      A dictionary containing (for a single g-point):
        'optical_depth': The shortwave optical depth.
        'ssa': The shortwave single-scattering albedo.
        'asymmetry_factor': The shortwave asymmetry factor.
    """

    def optical_depth_sw_fn(molecules, temperature, pressure, vmr_fields):
      return self._compute_optical_depth_fn(
          self.gas_optics_sw,
          molecules,
          temperature,
          pressure,
          igpt,
          vmr_fields,
      )

    def rayleigh_scattering_fn(molecules, temperature, pressure):
      return gas_optics.compute_rayleigh_optical_depth(
          self.gas_optics_sw,
          self.vmr_lib,
          molecules,
          temperature,
          pressure,
          igpt,
          vmr_fields,
      )

    optical_depth_sw = self._map_fn(
        optical_depth_sw_fn, molecules, temperature, pressure, vmr_fields,
    )

    rayleigh_scattering = tf.nest.map_structure(
        rayleigh_scattering_fn, molecules, temperature, pressure
    )
    optical_depth_sw = tf.nest.map_structure(
        tf.math.add, optical_depth_sw, rayleigh_scattering
    )
    ssa = tf.nest.map_structure(
        tf.math.divide_no_nan, rayleigh_scattering, optical_depth_sw
    )
    return {
        'optical_depth': optical_depth_sw,
        'ssa': ssa,
        'asymmetry_factor': tf.nest.map_structure(tf.zeros_like, ssa),
    }

  def compute_planck_sources(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      pressure: FlowFieldVal,
      temperature: FlowFieldVal,
      igpt: int,
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

    def planck_fraction_fn(
        pressure: tf.Tensor,
        temperature: tf.Tensor,
        vmr_fields: Optional[Dict[int, tf.Tensor]] = None,
    ):
      """Precomputes Planck fraction that is used for all Planck sources."""
      return gas_optics.compute_planck_fraction(
          self.gas_optics_lw,
          self.vmr_lib,
          pressure,
          temperature,
          igpt,
          vmr_fields,
      )

    planck_fraction = self._map_fn(
        planck_fraction_fn,
        pressure,
        temperature,
        vmr_fields,
    )

    def planck_src_fn(
        planck_fraction: tf.Tensor,
        temperature: tf.Tensor,
    ):
      """Computes Planck src for `temperature` scaled by `planck_fraction`."""
      return gas_optics.compute_planck_sources(
          self.gas_optics_lw,
          planck_fraction,
          temperature,
          igpt,
      )
    temperature_fields = {
        'planck_src': temperature,
        'planck_src_top': temperature_top,
        'planck_src_bottom': temperature_bottom,
    }
    planck_srcs = {}
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
      planck_src_sfc = self._map_fn(
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
    self._kernel_op = kernel_op
    self._g_dim = g_dim
    self._grad_central = (
        lambda f: self._kernel_op.apply_kernel_op_x(f, 'kDx'),
        lambda f: self._kernel_op.apply_kernel_op_y(f, 'kDy'),
        lambda f: self._kernel_op.apply_kernel_op_z(f, 'kDz', 'kDzsh'),
    )[g_dim]

  def compute_lw_optical_properties(
      self,
      pressure: FlowFieldVal,
      *args,
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
