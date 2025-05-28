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

"""A class that computes terms in a generic scalar transport equation."""

import abc
from collections.abc import Sequence
import copy
from typing import List, Tuple
from absl import logging
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.boundary_condition import boundary_condition_utils
from swirl_lm.equations import common
from swirl_lm.equations import scalars_pb2
from swirl_lm.numerics import convection
from swirl_lm.numerics import diffusion
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.physics.turbulence import sgs_model
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import stretched_grid_util
from swirl_lm.utility import types
import tensorflow as tf

# A small number that's used as the threshold for the gravity vector. If the
# absolute value of a gravity component is less than this threshold, it is
# considered as 0 when computing the free slip wall boundary condition.
_G_THRESHOLD = 1e-6


def _get_config_for_scalar(
    scalars: Sequence[scalars_pb2.Scalar],
    name: str,
) -> scalars_pb2.Scalar:
  """Gets the config for a scalar based on the name."""
  for scalar in scalars:
    if scalar.name == name:
      return scalar
  raise AssertionError(
      f'{name} is not configured. Valid scalars are:'
      f' {tuple(scalar.name for scalar in scalars)}.'
  )


class ScalarGeneric(abc.ABC):
  """A class that defines terms in a generic scalar transport equation."""

  def __init__(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      params: parameters_lib.SwirlLMParameters,
      scalar_name: str,
      thermodynamics: thermodynamics_manager.ThermodynamicsManager,
  ):
    """Prepares helper variables for function evaluations."""
    self._kernel_op = kernel_op
    self._params = params
    self._bc_types = copy.deepcopy(params.bc_type)
    self._scalar_name = scalar_name
    self._thermodynamics = thermodynamics
    self._deriv_lib = params.deriv_lib
    self._h = params.grid_spacings

    self._scalar_params = _get_config_for_scalar(
        self._params.scalars, self._scalar_name
    )

    for override_bc in self._scalar_params.override_bc_type:
      dim = override_bc.dim
      face = override_bc.face
      logging.info(
          'BC type for scalar: %s '
          'will be reset to %s (originally %s) for dimension %d at '
          'face %d',
          self._scalar_name,
          str(boundary_condition_utils.BoundaryType.UNKNOWN),
          str(self._bc_types[dim][face]),
          dim,
          face,
      )
      self._bc_types[dim][face] = boundary_condition_utils.BoundaryType.UNKNOWN

    # Dimension in which gravity acts.
    self._g_dim = params.g_dim

    # Prepare diffusion related models.
    self._diffusion_fn = diffusion.diffusion_scalar(self._params)
    self._use_sgs = self._params.use_sgs
    if self._use_sgs:
      self._sgs_model = sgs_model.SgsModel(self._kernel_op, params)

  def _get_momentum_for_convection(
      self,
      phi: types.FlowFieldVal,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> Tuple[types.FlowFieldVal, types.FlowFieldVal, types.FlowFieldVal]:
    """Determines the momentum to be used to compute the convection term."""
    del phi, additional_states

    return tuple(states[key] for key in common.KEYS_MOMENTUM)

  def _get_scalar_for_convection(
      self,
      phi: types.FlowFieldVal,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldVal:
    """Determines the scalar to be used to compute the convection term."""
    del states, additional_states

    return phi

  def _get_scalar_for_diffusion(
      self,
      phi: types.FlowFieldVal,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldVal:
    """Determines the scalar to be used to compute the diffusion term."""
    del states, additional_states

    return phi

  def get_diffusivity(
      self,
      replicas: np.ndarray,
      phi: types.FlowFieldVal,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldVal:
    """Computes the overall diffusivity.

    Args:
      replicas: A 3D array specifying the topology of the partition.
      phi: The variable `scalar_name` at the present iteration.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.

    Returns:
      The overall diffusivity of the scalar at the present iteration.
    """
    if self._use_sgs:
      velocity = tuple(
          states[key] for key in (common.KEY_U, common.KEY_V, common.KEY_W)
      )
      diff_t = self._sgs_model.turbulent_diffusivity(
          (phi,), additional_states, velocity, replicas
      )
      diffusivity = tf.nest.map_structure(
          lambda diff_t_i: self._params.diffusivity(self._scalar_name)  # pylint: disable=g-long-lambda
          + diff_t_i,
          diff_t,
      )
    else:
      diffusivity = tf.nest.map_structure(
          lambda sc: self._params.diffusivity(self._scalar_name)  # pylint: disable=g-long-lambda
          * tf.ones_like(sc),
          phi,
      )

    if self._params.turbulent_combustion is not None:
      diffusivity = self._params.turbulent_combustion.update_diffusivity(
          diffusivity, states, additional_states
      )

    return diffusivity

  def _get_wall_diffusive_flux_helper_variables(
      self,
      phi: types.FlowFieldVal,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldMap:
    """Prepares the helper variables for the diffusive flux in wall models.

    Args:
      phi: The variable `scalar_name` at the present iteration.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.

    Returns:
      A dictionary of variables required by wall diffusive flux closure models.
    """
    helper_variables = {key: states[key] for key in common.KEYS_VELOCITY}

    # Because only the layer close to the ground will be used in the Monin
    # Obukhov similarity closure model, the temperature and potential
    # temperature are equal. Note that the helper variables are used only
    # in the 'T' and 'theta' transport equations.
    for varname in ('theta', 'T'):
      if self._scalar_name == varname:
        helper_variables.update({'theta': phi})
        break
      elif varname in states:
        helper_variables.update({'theta': states[varname]})
        break
      elif varname in additional_states:
        helper_variables.update({'theta': additional_states[varname]})
        break

    return helper_variables

  def convection_fn(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      phi: types.FlowFieldVal,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> List[types.FlowFieldVal]:
    """Computes the convection term in the transport equation.

    Args:
      replica_id: The index of the local core replica.
      replicas: A 3D array specifying the topology of the partition.
      phi: The variable `scalar_name` at the present iteration.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.

    Returns:
      The convection term of the transport equation.
    """
    momentum = self._get_momentum_for_convection(phi, states, additional_states)
    phi_convect = self._get_scalar_for_convection(
        phi, states, additional_states
    )
    helper_variables = {}
    # Helper variables required for stretched grids.
    helper_variables.update(
        stretched_grid_util.get_helper_variables(additional_states)
    )

    def convection_1d(dim: int) -> types.FlowFieldVal:
      """Computes the convection term for the conservative scalar."""
      # Computes the gravitational force for the face flux correction.
      momentum_component = common.KEYS_MOMENTUM[dim]

      return convection.convection_term(
          self._kernel_op,
          self._deriv_lib,
          replica_id,
          replicas,
          phi_convect,
          momentum[dim],
          states[common.KEY_P],
          self._h[dim],
          self._params.dt,
          dim,
          helper_variables,
          bc_types=tuple(self._bc_types[dim]),
          varname=momentum_component,
          halo_width=self._params.halo_width,
          scheme=self._scalar_params.scheme,
          flux_scheme=self._params.numerical_flux,
          src=None,
          apply_correction=False,
      )

    return [convection_1d(dim) for dim in range(3)]

  def diffusion_fn(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      phi: types.FlowFieldVal,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> List[types.FlowFieldVal]:
    """Computes the diffusion term in the transport equation.

    Args:
      replica_id: The index of the local core replica.
      replicas: A 3D array specifying the topology of the partition.
      phi: The variable `scalar_name` at the mid-point between the previous and
        current scalar step in the latest sub-iteration.
      states: A dictionary that holds all flow field variables at the mid-point
        between the previous and current scalar step.
      additional_states: A dictionary that holds all helper variables.

    Returns:
      The diffusion term of the transport equation.
    """
    # Get the variable for which the diffusion term is computed.
    phi_diffuse = self._get_scalar_for_diffusion(phi, states, additional_states)

    helper_variables = {}

    # Helper variables required for stretched grids.
    helper_variables.update(
        stretched_grid_util.get_helper_variables(additional_states)
    )

    # Helper variables required by the Monin-Obukhov similarity theory.
    helper_variables.update(
        self._get_wall_diffusive_flux_helper_variables(
            phi, states, additional_states
        )
    )

    # Gather all diffusion fluxes specified by a variable.
    for flux_info in self._scalar_params.diffusive_flux:
      if flux_info.WhichOneof('flux') == 'varname':
        helper_variables[flux_info.varname] = additional_states[
            flux_info.varname
        ]

    return self._diffusion_fn(
        self._kernel_op,
        self._deriv_lib,
        replica_id,
        replicas,
        phi_diffuse,
        states[common.KEY_RHO],
        additional_states['diffusivity'],
        scalar_name=self._scalar_name,
        helper_variables=helper_variables,
    )

  def source_fn(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      phi: types.FlowFieldVal,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.ScalarSource:
    """Computes the source term in the transport equation.

    Args:
      replica_id: The index of the local core replica.
      replicas: A 3D array specifying the topology of the partition.
      phi: The variable `scalar_name` at the present iteration.
      states: A dictionary that holds all flow field variables.
      additional_states: A dictionary that holds all helper variables.

    Returns:
      The source term of this scalar transport equation.
    """
    del replica_id, replicas, states, additional_states
    return types.ScalarSource(total=tf.nest.map_structure(tf.zeros_like, phi))
