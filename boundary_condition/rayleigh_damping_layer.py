# coding=utf-8
"""A library of the Rayleigh damping (sponge) layer.

This boundary treatment is applied as a forcing term that takes the form [1]:
  f(𝜙) = -𝛽 (𝜙 - 𝜙₀),
where 𝛽 is the coefficient that determines where the sponge layer is applied,
𝜙₀ is the desired value at the sponge layer.

To use this library, a variable with name 'sponge_beta' is required in the
`additional_states`. This variable should have the same shape as the mesh in
each TPU replica.

One of the mostly known application of this boundary condition is useful to
prevent the reflection of gravitational wave from the top of the domain.

Reference:
1. Durran, Dale R., and Joseph B. Klemp. 1983. “A Compressible Model for the
   Simulation of Moist Mountain Waves.” Monthly Weather Review 111 (12):
   2341–61.
"""
from typing import Dict, Iterable, Mapping, Optional, Sequence, Text, Union

from absl import logging
import numpy as np
from swirl_lm.base import parameters_pb2
from swirl_lm.boundary_condition import rayleigh_damping_layer_pb2
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.framework import initializer

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap
BoolMap = types.BoolMap
TargetValueLib = Dict[Text, Optional[Union[float, Text]]]

_InitFn = initializer.ValueFunction
_Orientation = rayleigh_damping_layer_pb2.RayleighDampingLayer.Orientation
_SpongeInfo = rayleigh_damping_layer_pb2.RayleighDampingLayer.VariableInfo
_PeriodicDimensionInfo = parameters_pb2.PeriodicDimensions


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


def target_value_lib_from_proto(
    sponge: rayleigh_damping_layer_pb2.RayleighDampingLayer) -> TargetValueLib:
  """Generates a target value library from the proto.

  Args:
    sponge: A materialized sponge layer proto.

  Returns:
    A dictionary with keys being the variable names, and values being the target
    value in the sponge layer.
  """
  lib = {}
  for info in sponge.variable_info:
    if info.HasField('target_value'):
      lib.update({info.name: info.target_value})
    elif info.HasField('target_state_name'):
      lib.update({info.name: info.target_state_name})
    else:
      lib.update({info.name: None})
  return lib


def variable_type_lib_from_proto(
    sponge: rayleigh_damping_layer_pb2.RayleighDampingLayer) -> BoolMap:
  """Generates a library for the type of the variable from the proto.

  Args:
    sponge: A materialized sponge layer proto.

  Returns:
    A dictionary with keys being the variable names, and values being the type
    of the variable (primitive or conservative) that the sponge is associated
    with.
  """
  return {info.name: info.primitive for info in sponge.variable_info}


def target_status_lib_from_proto(
    sponge: rayleigh_damping_layer_pb2.RayleighDampingLayer) -> BoolMap:
  """Generates a sponge forcing status library from the proto.

  Args:
    sponge: A materialized sponge layer proto.

  Returns:
    A dictionary with keys being the name of the forcing term, and values being
    the target behavior of the forcing term. `False` if there are other forcing
    terms need to be combined with the sponge force, and `True` if the sponge
    force is the only forcing term for that variable.
  """
  return {
      get_sponge_force_name(info.name): info.override
      for info in sponge.variable_info
  }


def klemp_lilly_relaxation_coeff_fn(
    dt: float,
    orientation: Iterable[_Orientation],
    a_coeff: Optional[float] = 20.0,
) -> _InitFn:
  """Generates a function that computes 'sponge_beta' (Klemp & Lilly, 1978).

  The sponge layer coefficient is defined as:
  beta = 0, if h ⩽ h_d
  beta = 𝛼ₘₐₓ sin² (π/2 (h - h_d) / (hₜ - h_d)), if h > h_d.

  Args:
    dt: The time step size.
    orientation: A sequence of named variables that stores the orientation
      information of the sponge layer. Each orientation element includes a `dim`
      field that indicates the dimension of the sponge layer, and a `fraction`
      field that specifies the fraction that the sponge layer is taking at the
      higher end of `dim`.
    a_coeff: The coefficient used to compute the maximum magitude of the sponge
      force. Scales inversely with the force.

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
      elif dim == 1:
        grid = yy
        h_t = ly
      elif dim == 2:
        grid = zz
        h_t = lz
      else:
        raise ValueError(
            'Dimension has to be one of 0, 1, and 2. {} is given.'.format(dim))

      if sponge.fraction < 0 or sponge.fraction > 1:
        raise ValueError(
            'The fraction of sponge layer should be in (0, 1). {} is given in '
            'dim {}.'.format(sponge.fraction, dim))

      a_max = np.power(a_coeff * dt, -1)

      # Set the default face to the higher end for backward compatibility.
      face = sponge.face if sponge.HasField('face') else 1

      if face == 1:
        h_d = (1.0 - sponge.fraction) * h_t
        buf = tf.compat.v1.where(
            tf.less_equal(grid, h_d), tf.zeros_like(grid),
            a_max * tf.math.pow(
                tf.math.sin(np.pi / 2.0 * (grid - h_d) / (h_t - h_d)), 2))
      elif face == 0:
        h_d = sponge.fraction * h_t
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


class RayleighDampingLayer(object):
  """A library of the sponge layer."""

  def __init__(self,
               sponge_info: rayleigh_damping_layer_pb2.RayleighDampingLayer,
               periodic_dims: Optional[Sequence[bool]] = None):
    """Initializes the sponge layer library.

    Args:
      sponge_info: An instance of the materialized RayleighDampingLayer proto.
      periodic_dims: An optional list of booleans indicating the periodic
        dimensions.
    """
    self._target_values = target_value_lib_from_proto(sponge_info)
    self._target_status = target_status_lib_from_proto(sponge_info)
    self._is_primitive = variable_type_lib_from_proto(sponge_info)
    self._orientation = [ori.dim for ori in sponge_info.orientation]
    self._sponge_info_map = {
        info.name: info for info in sponge_info.variable_info
    }

    # Get the dimensions over which to compute the mean as the target values
    # when they are not provided. If target value mean dimensions are not
    # defined in the config, it will use periodic dimensions by default.
    self._target_value_mean_dims = list(sponge_info.target_value_mean_dim)
    if not self._target_value_mean_dims and periodic_dims is not None:
      self._target_value_mean_dims = [
          i for i, val in enumerate(periodic_dims) if val
      ]

    logging.info(
        'Sponge layer will be applied for the following variables with'
        'following values: %r', self._target_values)

  def _get_sponge_force(
      self,
      replicas: np.ndarray,
      field: FlowFieldVal,
      beta: FlowFieldVal,
      target_state: Optional[Union[float, FlowFieldVal]] = None,
  ) -> FlowFieldVal:
    """Computes the sponge forcing term.

    Args:
      replicas: The replicas. In particular, a numpy array that maps grid
        coordinates to replica id numbers.
      field: The value of the variable to which the sponge forcing is applied.
      beta: The coefficients to be applied as the sponge.
      target_state: An optional reference state from which to compute the
        sponge target.

    Returns:
      The sponge force.
    """
    if target_state is not None:
      target_value = target_state
    else:
      target_value = common_ops.global_mean(
          field, replicas, axis=self._target_value_mean_dims)

    diff = common_ops.linear_combination(field, target_value, 1.0, -1.0)
    return [-b * diff_i for b, diff_i in zip(beta, diff)]

  def init_fn(
      self,
      config: grid_parametrization.GridParametrization,
      coordinates: initializer.ThreeIntTuple,
      beta_fn: _InitFn,
  ) -> Mapping[Text, tf.Tensor]:
    """Generates the required initial fields by the simulation.

    Args:
      config: An instance of `grid_parametrization.GridParametrization`.
      coordinates: A tuple that specifies the replica's grid coordinates in
        physical space.
      beta_fn: A function that initializes the coefficients in the sponge layer.

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
          num_boundary_points=0,
          mesh_choice=initializer.MeshChoice.PARAMS,
      )
    # pylint: disable=g-long-lambda
    init_fn_zeros = lambda xx, yy, zz, lx, ly, lz, coord: tf.zeros_like(
        xx, dtype=xx.dtype)
    # pylint: enable=g-long-lambda

    output = {'sponge_beta': states_init(beta_fn)}
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

    The forcing term will replace or add to the existing forcing term in
    `additional_states` for variables that are in the scope of
    `self._target_values`, following the indicator stated in
    `self._target_status`: if `False`, the sponge force will be added to the
    input force with the same name; if `True`, the sponge force will override
    the existing one. If other forcing terms needs to be applied to a same
    variable, the values of all these forcing terms needs to be updated ahead of
    the sponge forces.

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
    del kernel_op, replica_id, params

    if 'sponge_beta' not in additional_states.keys():
      raise ValueError('"sponge_beta" not found in `additional_states.`')

    def add_to_additional_states(
        name: Text,
        value: FlowFieldVal,
    ) -> FlowFieldVal:
      """Adds two states elementwise."""
      return [
          state_1 + state_2
          for state_1, state_2 in zip(additional_states[name], value)
      ]

    additional_states_updated = {}
    additional_states_updated.update(additional_states)
    for varname, var_info in self._sponge_info_map.items():
      if varname not in states.keys():
        # TODO(wqing): Revise this condition so that the function captures the
        # case with true incomplete states.
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
      sponge_force = self._get_sponge_force(replicas, states[varname],
                                            additional_states['sponge_beta'],
                                            target_val)
      if not self._is_primitive[varname]:
        sponge_force = [rho * f for rho, f in zip(states['rho'], sponge_force)]
      if self._target_status[sponge_name]:
        additional_states_updated.update({sponge_name: sponge_force})
      else:
        additional_states_updated.update(
            {sponge_name: add_to_additional_states(sponge_name, sponge_force)})

    return additional_states_updated
