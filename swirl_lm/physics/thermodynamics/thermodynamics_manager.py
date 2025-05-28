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

"""The manager of thermodyanmics models."""

from typing import Optional

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.communication import halo_exchange
from swirl_lm.numerics import filters
from swirl_lm.physics.thermodynamics import constant_density
from swirl_lm.physics.thermodynamics import ideal_gas
from swirl_lm.physics.thermodynamics import linear_mixing
from swirl_lm.physics.thermodynamics import thermodynamics_generic
from swirl_lm.physics.thermodynamics import thermodynamics_pb2
from swirl_lm.physics.thermodynamics import thermodynamics_utils
from swirl_lm.physics.thermodynamics import water
from swirl_lm.utility import get_kernel_fn
import tensorflow as tf

DensityUpdateOption = parameters_lib.DensityUpdateOption
FlowFieldVal = thermodynamics_utils.FlowFieldVal
FlowFieldMap = thermodynamics_utils.FlowFieldMap


class ThermodynamicsManager(object):
  """A manager class for models of thermodynamics."""

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
      model_params: thermodynamics_pb2.Thermodynamics,
      model: thermodynamics_generic.ThermodynamicModel,
  ):
    """Initializes the thermodynamics model manager."""
    self._params = params
    self._model_params = model_params
    self.model = model

    self._replica_dims = (0, 1, 2)
    self._halo_dims = (0, 1, 2)

  @property
  def solver_mode(self):
    """Returns the mode of the thermodynamics model."""
    return self._model_params.solver_mode

  def rho_ref(
      self,
      zz: Optional[FlowFieldVal] = None,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Generates the reference density."""
    return self.model.rho_ref(zz, additional_states)

  def p_ref(
      self,
      zz: Optional[FlowFieldVal] = None,
      additional_states: Optional[FlowFieldMap] = None,
  ) -> FlowFieldVal:
    """Generates the reference pressure."""
    return self.model.p_ref(zz, additional_states)

  def update_density(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      states_0: Optional[FlowFieldMap] = None,
  ) -> tuple[tf.Tensor, tf.Tensor]:
    """Updates the density based on field values provided.

    Args:
      kernel_op: A kernel operation library for finite difference operations.
      replica_id: The index of the current TPU replica.
      replicas: The topology of the TPU partition.
      states: Flow field variables. Must include 'rho'.
      additional_states: Helper variables in the simulation.
      states_0: Optional flow field variables at the previous time step. When
        included, density is updated from its filtered change with respect to
        the previous time step. `states_0` must include 'rho'.

    Returns:
      The density at the current step, and the change of density from previous
      time step. Note that the change of density will be 0 if `states_0` is
      absent.

    Raises:
      NotImplementedError: If the solver mode is not one of the following:
        'LOW_MACH', 'ANELASTIC'.
      ValueError: If 'rho' is not found in `states` or `state_0` (if it's used).
    """
    del kernel_op

    if 'rho' not in states:
      raise ValueError('"rho" is not found in `states`.')

    if self.solver_mode == thermodynamics_pb2.Thermodynamics.LOW_MACH:
      # In this mode, the thermodynamic density is fully coupled with the
      # momentum and scalars, i.e. the conservative variables are computed with
      # the thermodynamic density.
      rho = self.model.update_density(states, additional_states)

      if states_0 is not None:
        if 'rho' not in states_0:
          raise ValueError(
              'Density change from state 0 is requested but "rho" is not found.'
          )
        # Applies filtering to the density change from previous step to
        # eliminate spurious heat release that's caused by dispersion errors in
        # the convection terms of the scalar transport equations. It's assumed
        # that Assumes density changes in the halos of the physical boundary
        # are the same as the first fluid layer.
        drho = halo_exchange.inplace_halo_exchange(
            filters.filter_op(
                self._params,
                tf.nest.map_structure(tf.math.subtract, rho, states_0['rho']),
                additional_states,
                order=2,
            ),
            self._halo_dims,
            replica_id,
            replicas,
            self._replica_dims,
            self._params.periodic_dims,
            [[(halo_exchange.BCType.NEUMANN, 0.0)] * 2] * 3,
            width=self._params.halo_width,
        )

        rho = tf.nest.map_structure(tf.math.add, states_0['rho'], drho)
      else:
        drho = tf.nest.map_structure(tf.zeros_like, states['rho'])

      return rho, drho

    elif self.solver_mode == thermodynamics_pb2.Thermodynamics.ANELASTIC:
      # In this mode, the reference density is used to compute conservative
      # variables. The thermodynamic density is only coupled with the buoyancy
      # following the Boussinesq approximation.
      if 'zz' not in additional_states:
        zz = tf.nest.map_structure(tf.zeros_like, states['rho'])
      else:
        zz = additional_states['zz']
      return (
          self.model.rho_ref(zz, additional_states),
          tf.nest.map_structure(tf.zeros_like, states['rho']),
      )
    else:
      raise NotImplementedError(
          '{} is not a valid solver model for density update'.format(
              thermodynamics_pb2.Thermodynamics.SolverMode.Name(
                  self.solver_mode)))

  def update_thermal_density(self, states, additional_states):
    """Computes the density based on field values provided.

    Args:
      states: Flow field variables. Must include 'rho'.
      additional_states: Helper variables in the simulation.

    Returns:
      The thermodynamic density at the current step.
    """
    return self.model.update_density(states, additional_states)


def thermodynamics_factory(params: parameters_lib.SwirlLMParameters):
  """Creates an object of the `ThermodynamicsManager`.

  The thermodynamic library will be created based on parameters specified in the
  simulation configuration file. In the absence of `thermodynamics` in the
  configuration file, we assume that the simulation follows the low-Mach number
  approach.

  Args:
    params: The context object that holds parameters in a simulation.

  Returns:
    An object of the thermodynamics manager, which provides interfaces for the
    evaluations of thermodynamics quantities.
  """
  # The `density_option` will be deprecated.
  if params.thermodynamics is None:
    density_option = params.density_update_option
    if density_option == DensityUpdateOption.DENSITY_UPDATE_LINEAR_MIXING:
      model = linear_mixing.LinearMixing(params)
    elif density_option == DensityUpdateOption.DENSITY_UPDATE_EOS:
      model = ideal_gas.IdealGas(params)
    else:
      model = constant_density.ConstantDensity(params)
    model_params = thermodynamics_pb2.Thermodynamics()
    model_params.solver_mode = thermodynamics_pb2.Thermodynamics.LOW_MACH
    return ThermodynamicsManager(params, model_params, model)

  model_params = params.thermodynamics
  model_type = model_params.WhichOneof('thermodynamics_type')

  if model_type == 'linear_mixing':
    model = linear_mixing.LinearMixing(params)
  elif model_type == 'ideal_gas_law':
    model = ideal_gas.IdealGas(params)
  elif model_type == 'water':
    model = water.Water(params)
  elif model_type == 'constant_density':
    model = constant_density.ConstantDensity(params)
  else:
    raise NotImplementedError(
        '{} is not a valid thermodynamics model.'.format(model_type))

  return ThermodynamicsManager(params, model_params, model)
