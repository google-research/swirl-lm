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

# coding=utf-8
"""A library of the Rayleigh damping (sponge) layer.

This boundary treatment is applied as a forcing term that takes the form [1]:
  f(𝜙) = -𝛽 (𝜙 - 𝜙₀),
where 𝛽 is the coefficient that determines where the sponge layer is applied,
𝜙₀ is the desired value at the sponge layer.

To use this library, one or more variables for 'beta' coefficients are required
in the `additional_states`. These variables should have the same shape as the
mesh in each TPU replica.

One of the mostly known application of this boundary condition is to prevent the
reflection of gravitational wave from the top of the domain.

Reference:
1. Durran, Dale R., and Joseph B. Klemp. 1983. “A Compressible Model for the
   Simulation of Moist Mountain Waves.” Monthly Weather Review 111 (12):
   2341–61.
"""
from typing import Dict, Iterable, Mapping, Optional, Sequence, Text, Union

from absl import logging
import numpy as np
from swirl_lm.base import initializer
from swirl_lm.boundary_condition import rayleigh_damping_layer_pb2
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap
BoolMap = types.BoolMap
TargetValueLib = Dict[Text, Optional[Union[float, Text]]]

_InitFn = initializer.ValueFunction
_Orientation = rayleigh_damping_layer_pb2.RayleighDampingLayer.Orientation
_SpongeInfo = rayleigh_damping_layer_pb2.RayleighDampingLayer.VariableInfo
_RayleighDampingLayerSeq = Sequence[
    rayleigh_damping_layer_pb2.RayleighDampingLayer]


def get_sponge_force_name(varname: Text) -> Text:
  """Generates the name of the forcing term for the input variable name.

  Args:
    varname: The name of the variable for which the name of the forcing term
      is requested.

  Returns:
    The name of the forcing term, which takes the form 'src_[varname]'.
  """
  return 'src_{}'.format(varname)


def get_sponge_target_name(varname: Text) -> Text:
  """Generates the name of the target value term for the input variable name.

  Args:
    varname: The name of the variable for which the name of the target value
      is requested.

  Returns:
    The name of the target term, which takes the form 'sponge_target_[varname]'.
  """
  return 'sponge_target_{}'.format(varname)


def variable_type_lib_from_proto(sponges: _RayleighDampingLayerSeq) -> BoolMap:
  """Generates a library for the type of the variable from the proto.

  Args:
    sponges: A sequence of materialized sponge layer protos.

  Returns:
    A dictionary with keys being the variable names, and values being the type
    of the variable (primitive or conservative) that the sponge is associated
    with.
  """
  out = {}
  for sponge in sponges:
    for info in sponge.variable_info:
      out[info.name] = info.primitive
  return out


def _get_beta_name_from_sponge_info(
    sponge: rayleigh_damping_layer_pb2.RayleighDampingLayer) -> str:
  # Use default name 'sponge_beta' if beta_name is not explicitly set.
  return sponge.beta_name or 'sponge_beta'


def beta_name_by_var(sponges: _RayleighDampingLayerSeq) -> Dict[str, str]:
  """Returns the mapping from variable names to beta variable names."""
  out = {}
  seen_beta_names = set()
  for sponge in sponges:
    beta_name = _get_beta_name_from_sponge_info(sponge)
    if beta_name in seen_beta_names:
      raise ValueError(
          f'Sponge beta variable `{beta_name}` is defined more than once.')
    seen_beta_names.add(beta_name)
    for info in sponge.variable_info:
      if info.name in out:
        raise ValueError(
            f'Variable `{info.name}` participates in multiple sponge layers.')
      out[info.name] = beta_name
  return out


def sponge_info_map(
    sponges: _RayleighDampingLayerSeq) -> Dict[str, _SpongeInfo]:
  out = {}
  for sponge in sponges:
    for info in sponge.variable_info:
      out[info.name] = info
  return out


def target_value_mean_dims_by_var(
    sponges: _RayleighDampingLayerSeq,
    periodic_dims: Optional[Sequence[bool]]) -> Dict[str, Sequence[int]]:
  """Returns the mapping from variable names to target value mean dimensions."""
  out = {}
  for sponge in sponges:
    target_value_mean_dims = list(sponge.target_value_mean_dim)
    if not target_value_mean_dims and periodic_dims is not None:
      target_value_mean_dims = [
          i for i, val in enumerate(periodic_dims) if val
      ]
    for info in sponge.variable_info:
      out[info.name] = target_value_mean_dims
  return out


def klemp_lilly_relaxation_coeff_fn(
    orientation: Iterable[_Orientation],
    x0: tf.Tensor,
    y0: tf.Tensor,
    z0: tf.Tensor
) -> _InitFn:
  """Generates a function that computes 'sponge_beta' (Klemp & Lilly, 1978).

  The sponge layer coefficient is defined as:
  beta = 0, if h ⩽ h_d
  beta = 𝛼ₘₐₓ sin² (π/2 (h - h_d) / (hₜ - h_d)), if h > h_d.

  Args:
    orientation: A sequence of named variables that stores the orientation
      information of the sponge layer. Each orientation element includes a `dim`
      field that indicates the dimension of the sponge layer, and a `fraction`
      field that specifies the fraction that the sponge layer is taking at the
      higher end of `dim`.
    x0: Coordinate of face 0 along the x dimension.
    y0: Coordinate of face 0 along the y dimension.
    z0: Coordinate of face 0 along the z dimension.

  Returns:
    The `sponge_beta` coefficient following Klemp & Lilly, 1978.

  Raises:
    ValueError: If `dim` in `orientation` is not one of 0, 1, or 2.
    ValueError: If the sponge layer fraction is below 0 or above 1.
  """

  def init_fn(
      xx: tf.Tensor,
      yy: tf.Tensor,
      zz: tf.Tensor,
      lx: float,
      ly: float,
      lz: float,
      coord: initializer.ThreeIntTuple,
  ) -> tf.Tensor:
    """Initializes the sponge layer relaxation coefficient beta."""
    del coord
    beta = tf.zeros_like(xx, dtype=xx.dtype)

    for sponge in orientation:
      dim = sponge.dim

      if dim == 0:
        grid = xx
        h_t = lx
        c0 = x0
      elif dim == 1:
        grid = yy
        h_t = ly
        c0 = y0
      elif dim == 2:
        grid = zz
        h_t = lz
        c0 = z0
      else:
        raise ValueError(
            'Dimension has to be one of 0, 1, and 2. {} is given.'.format(dim))

      if sponge.fraction < 0 or sponge.fraction > 1:
        raise ValueError(
            'The fraction of sponge layer should be in (0, 1). {} is given in '
            'dim {}.'.format(sponge.fraction, dim))

      a_max = np.reciprocal(sponge.a_coeff)

      # Set the default face to the higher end for backward compatibility.
      face = sponge.face if sponge.HasField('face') else 1

      if face == 1:
        h_d = (1.0 - sponge.fraction) * h_t + c0
        buf = tf.compat.v1.where(
            tf.less_equal(grid, h_d), tf.zeros_like(grid),
            a_max * tf.math.pow(
                tf.math.sin(np.pi / 2.0 * (grid - h_d) / (h_t - h_d)), 2))
      elif face == 0:
        h_d = sponge.fraction * h_t + c0
        buf = tf.compat.v1.where(
            tf.greater_equal(grid, h_d), tf.zeros_like(grid),
            a_max *
            tf.math.pow(tf.math.sin(np.pi / 2.0 * tf.abs(grid - h_d) / h_d), 2))
      else:
        raise ValueError(
            'Face index has to be one of 0 and 1. {} is provided.'.format(face))

      beta = tf.maximum(beta, buf)

    return beta

  return init_fn


def klemp_lilly_relaxation_coeff_fns_for_sponges(
    sponge_infos: _RayleighDampingLayerSeq,
    x0: tf.Tensor,
    y0: tf.Tensor,
    z0: tf.Tensor
) -> Dict[str, _InitFn]:
  return {_get_beta_name_from_sponge_info(sponge_info):
          klemp_lilly_relaxation_coeff_fn(sponge_info.orientation, x0, y0, z0)
          for sponge_info in sponge_infos}


class RayleighDampingLayer(object):
  """A library of the sponge layer."""

  def __init__(
      self,
      sponge_infos: _RayleighDampingLayerSeq,
      periodic_dims: Optional[Sequence[bool]] = None):
    """Initializes the sponge layer library.

    Args:
      sponge_infos: Sequence of materialized RayleighDampingLayer protos.
      periodic_dims: An optional list of booleans indicating the periodic
        dimensions.
    """
    self._is_primitive = variable_type_lib_from_proto(sponge_infos)
    self._beta_name_by_var = beta_name_by_var(sponge_infos)
    self._sponge_info_map = sponge_info_map(sponge_infos)

    # Get the dimensions over which to compute the mean as the target values
    # when they are not provided. If target value mean dimensions are not
    # defined in the config, it will use periodic dimensions by default.
    self._target_value_mean_dims_by_var = target_value_mean_dims_by_var(
        sponge_infos, periodic_dims)

    logging.info(
        'Sponge layer will be applied for the following variables with '
        'following values: %s', self._sponge_info_map)

  def _get_sponge_force(
      self,
      replicas: np.ndarray,
      field: FlowFieldVal,
      beta: FlowFieldVal,
      dt: FlowFieldVal,
      target_value_mean_dims: Sequence[int],
      target_state: Optional[Union[float, FlowFieldVal]] = None,
  ) -> FlowFieldVal:
    """Computes the sponge forcing term.

    Args:
      replicas: The replicas. In particular, a numpy array that maps grid
        coordinates to replica id numbers.
      field: The value of the variable to which the sponge forcing is applied.
      beta: The coefficients to be applied as the sponge.
      dt: The time step size.
      target_value_mean_dims: Dimensions over which to compute the target value
        mean.
      target_state: An optional reference state from which to compute the
        sponge target.

    Returns:
      The sponge force.
    """
    if target_state is not None:
      target_value = target_state
    else:
      target_value = common_ops.global_mean(
          field, replicas, axis=target_value_mean_dims)

    return beta / dt * (target_value - field)

  @property
  def varnames(self) -> Sequence[str]:
    """Generates a tuple of variable names to which sponge is applied."""
    return tuple(self._sponge_info_map.keys())

  def init_fn(
      self,
      config: grid_parametrization.GridParametrization,
      coordinates: initializer.ThreeIntTuple,
      beta_fn_by_var: Dict[str, _InitFn],
  ) -> Mapping[Text, tf.Tensor]:
    """Generates the required initial fields by the simulation.

    Args:
      config: An instance of `grid_parametrization.GridParametrization`.
      coordinates: A tuple that specifies the replica's grid coordinates in
        physical space.
      beta_fn_by_var: A mapping from sponge beta variable names to functions
        that initialize the beta coefficients.

    Returns:
      A dictionary of state variables that are required by the Rayleigh damping
      layer.
    """

    def states_init(initial_value_fn) -> tf.Tensor:
      """Assigns value to a tensor with `initial_value_fn`."""
      return initializer.partial_mesh_for_core(
          config,
          coordinates,
          initial_value_fn,
          pad_mode='SYMMETRIC',
          mesh_choice=initializer.MeshChoice.PARAMS,
      )
    # pylint: disable=g-long-lambda
    init_fn_zeros = lambda xx, yy, zz, lx, ly, lz, coord: tf.zeros_like(
        xx, dtype=xx.dtype)
    # pylint: enable=g-long-lambda

    output = {beta_name: states_init(beta_fn)
              for beta_name, beta_fn in beta_fn_by_var.items()}
    for variable in self._sponge_info_map:
      output.update({
          get_sponge_force_name(variable): states_init(init_fn_zeros)
      })
    return output

  def additional_states_update_fn(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Updates the forcing term due to the sponge layer.

    The forcing term will be added to the existing forcing term in
    `additional_states` for variables that are in the scope of
    `self._sponge_info_map` following the specification of target values stored
    in the values of this dictionary.

    Args:
      kernel_op: An object holding a library of kernel operations.
      replica_id: The id of the replica.
      replicas: The replicas. In particular, a numpy array that maps grid
        coordinates to replica id numbers.
      states: A keyed dictionary of states that will be updated.
      additional_states: A list of states that are needed by the update fn, but
        will not be updated by the main governing equations.
      params: An instance of `grid_parametrization.GridParametrization`.

    Returns:
      A dictionary with updated sponge-layer forcing.

    Raises:
      ValueError: if 'sponge_beta' or `target_state_name` is not present in
      `additional_states`.
    """
    del kernel_op, replica_id

    beta_names_not_in_additional_states = (
        set(self._beta_name_by_var.values()) - set(additional_states.keys()))
    if beta_names_not_in_additional_states:
      raise ValueError(f'{beta_names_not_in_additional_states} not found '
                       'in `additional_states.`')

    def add_to_additional_states(
        name: Text,
        value: FlowFieldVal,
    ) -> FlowFieldVal:
      """Adds two states elementwise."""
      return additional_states[name] + value

    additional_states_updated = {}
    additional_states_updated.update(additional_states)
    for varname, var_info in self._sponge_info_map.items():
      if varname not in states.keys():
        logging.warn('%s is not a valid state. Available states are: %r',
                     varname, states.keys())
        continue

      sponge_name = get_sponge_force_name(varname)
      target_val = None
      if var_info.HasField('target_state_name'):
        if var_info.target_state_name not in additional_states_updated:
          raise ValueError(
              'Target_state_name {} is not among the states.'.format(
                  var_info.target_state_name))
        target_val = additional_states_updated[var_info.target_state_name]
      elif var_info.HasField('target_value'):
        target_val = var_info.target_value

      sponge_force = self._get_sponge_force(
          replicas, states[varname],
          additional_states[self._beta_name_by_var[varname]],
          tf.convert_to_tensor(params.dt),
          self._target_value_mean_dims_by_var[varname],
          target_val)
      if not self._is_primitive[varname]:
        sponge_force = states['rho'] * sponge_force
      additional_states_updated.update(
          {sponge_name: add_to_additional_states(sponge_name, sponge_force)})

    return additional_states_updated
