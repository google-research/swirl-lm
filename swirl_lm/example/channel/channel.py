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

"""A Channel Flow Simulation."""

from absl import logging

import numpy as np
from swirl_lm.base import initializer
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.boundary_condition import simulated_turbulent_inflow
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import init_fn as init_fn_lib
from swirl_lm.utility import types
import tensorflow as tf


FlowFieldMap = types.FlowFieldMap


# Defines the function that initializes state variables.
def _initialize_states(
    coordinates,
    init_fn,
    params,
    pad_mode='SYMMETRIC',
) -> tf.Tensor:
  """Assigns value to a tensor with `init_fn`.

  Args:
    coordinates: A tuple that specifies the replica's grid coordinates in
      physical space.
    init_fn: A function that takes the local mesh_grid tensor for the core (in
      order x, y, z) and the global characteristic length floats (in order x, y,
      z) and returns a 3-D tensor representing the value for the local core
      (without including the margin/overlap between the cores).
    params: A grid_parametrization.GridParametrization object containing the
      required grid spec information.
    pad_mode: The mode for filling in values in the halo layers. Should be one
      of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive). If 'CONSTANT'
      is used, 0 will be assigned in the halos.

  Returns:
    A 3D tensor with values assigned by `init_fn`.
  """
  return initializer.partial_mesh_for_core(
      params,
      coordinates,
      init_fn,
      pad_mode=pad_mode,
      mesh_choice=initializer.MeshChoice.PARAMS)


class Channel:
  """Defines initial condition and functions for the channel flow simulation."""

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    """Initializes the simulation setup."""
    self.params = params

    self.inflow = None
    if self.params.boundary_models is not None:
      if self.params.boundary_models.HasField('simulated_inflow'):
        self.inflow = (
            simulated_turbulent_inflow.simulated_turbulent_inflow_factory(
                self.params
            )
        )

  def init_fn(self, replica_id, coordinates):
    """Initializes state variables in a channel flow."""
    # Parameters of a parabola.
    c = 1.0

    # Parameters of the temperature profile.
    t_min = 300.0
    t_max = 300.0

    # Initialize helper variables for boundary condition
    def init_fn_u_parabola(xx, yy, zz, lx, ly, lz, coord):
      """Generates a parabola as inflow condition."""
      del xx, zz, lx, lz, coord

      if self.params.use_stretched_grid[1]:
        # Here we assume that the stretched mesh is defined from -L / 2 to
        # L / 2.
        y_min = -0.5 * self.params.ly
        y_max = 0.5 * self.params.ly
        p_max = y_min * y_max
      else:
        y_min = -0.5 * self.params.dy
        y_max = ly + 0.5 * self.params.dy
        p_max = (0.5 * ly - y_min) * (0.5 * ly - y_max)
      return c / p_max * (yy - y_min) * (yy - y_max)

    def init_fn_t_ramp(xx, yy, zz, lx, ly, lz, coord):
      """Generates a ramp for temperature decreases with increasing height."""
      del xx, zz, lx, lz, coord
      return t_max * tf.ones_like(yy) + (yy - self.params.y[0]) / ly * (
          t_min - t_max
      )

    logging.info('Creating initial fields.')
    output = {
        'replica_id': replica_id,
        'u': _initialize_states(coordinates, init_fn_u_parabola, self.params),
        'v': _initialize_states(
            coordinates, init_fn_lib.constant_init_fn(0), self.params
        ),
        'w': _initialize_states(
            coordinates, init_fn_lib.constant_init_fn(0), self.params
        ),
        'p': _initialize_states(
            coordinates, init_fn_lib.constant_init_fn(0), self.params
        ),
    }

    t_var = 'T'
    if t_var in self.params.transport_scalars_names:
      output[t_var] = _initialize_states(
          coordinates, init_fn_t_ramp, self.params
      )

    if 'Y_O' in self.params.transport_scalars_names:
      output['Y_O'] = _initialize_states(
          coordinates, init_fn_lib.constant_init_fn(0.23), self.params
      )

    logging.info('Updating density.')
    thermo_manager = thermodynamics_manager.thermodynamics_factory(self.params)
    thermo_additional_states = {}
    output.update({
        'rho': thermo_manager.update_thermal_density(
            output, thermo_additional_states
        )
    })

    logging.info('Initializing inflow boundary condition.')
    if 'bc_u_0_0' in self.params.additional_state_keys:
      output['bc_u_0_0'] = output['u']

    if f'bc_{t_var}_0_0' in self.params.additional_state_keys:
      output[f'bc_{t_var}_0_0'] = output[t_var]

    # Override the inflow helper variables with those from the inflow model if
    # configured.
    if self.inflow is not None:
      if isinstance(
          self.inflow, simulated_turbulent_inflow.SimulatedTurbulentInflow
      ):
        output.update(self.inflow.initialize_inflow())
      else:
        raise NotImplementedError(
            f'Inflow model {self.inflow} is not supported. Supported models'
            ' are: "simulated_inflow".'
        )

    return output

  def additional_states_update_fn(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      step_id: tf.Tensor,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Updates `additional_states` for drag forces and fire physics if required.

    Args:
      kernel_op: An object holding a library of kernel operations.
      replica_id: The id of the replica.
      replicas: The replicas. In particular, a numpy array that maps grid
        coordinates to replica id numbers.
      step_id: The index of the current time step.
      states: A keyed dictionary of states that will be updated.
      additional_states: A list of states that are needed by the update fn, but
        will not be updated by the main governing equations.
      params: An instance of `grid_parametrization.GridParametrization`.

    Returns:
      Updated `additional_states`.
    """
    del kernel_op, params

    additional_states_updated = {}
    additional_states_updated.update(additional_states)

    if isinstance(
        self.inflow, simulated_turbulent_inflow.SimulatedTurbulentInflow
    ):
      additional_states_updated = dict(
          self.inflow.additional_states_update_fn(
              replica_id, replicas, step_id, states, additional_states_updated
          )
      )

    return additional_states_updated
