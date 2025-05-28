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

"""Variable-density low Mach number Navier-Stokes solver."""

from typing import Optional

from absl import logging
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import physical_variable_keys_manager
from swirl_lm.boundary_condition import immersed_boundary_method
from swirl_lm.communication import halo_exchange
from swirl_lm.equations import common
from swirl_lm.equations import pressure as pressure_model
from swirl_lm.equations import scalars as scalars_model
from swirl_lm.equations import velocity as velocity_model
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.physics.thermodynamics import thermodynamics_pb2
from swirl_lm.utility import common_ops
from swirl_lm.utility import components_debug
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import monitor
from swirl_lm.utility.types import FlowFieldMap
import tensorflow as tf


TIME_VARNAME = 'simulation_time'


class Simulation:
  """Defines the step function for a variable-density low Mach solver.

  This class is used for the TF2-compatible simulation while it also defines
  the core solver functionality for the TF1 version.
  """

  def __init__(self, kernel_op: get_kernel_fn.ApplyKernelOp,
               params: parameters_lib.SwirlLMParameters):
    """Initializes the simulation step.

    Args:
      kernel_op: An ApplyKernelOp instance to use in computing the step update.
      params: An instance of `SwirlLMParameters`.
    """
    state_keys = ['rho', 'u', 'v', 'w', 'p']

    if params.transport_scalars_names:
      state_keys += params.transport_scalars_names

    additional_state_keys = (
        params.additional_state_keys if params.additional_state_keys else [])

    self._replica_dims = (0, 1, 2)
    self._halo_dims = (0, 1, 2)

    self._kernel_op = kernel_op
    self._params = params
    self._bc = params.bc
    self._bc['p'] = [[None, None], [None, None], [None, None]
                    ] if self._bc['p'] is None else self._bc['p']
    self._bc_manager = (
        physical_variable_keys_manager.BoundaryConditionKeysHelper())
    self._bc_manager.check_boundaries_updates_from_additional_states(
        additional_state_keys)

    self._ib = immersed_boundary_method.immersed_boundary_method_factory(
        self._params)

    self.thermodynamics = thermodynamics_manager.thermodynamics_factory(
        self._params)

    logging.info('Boundary condition types in this simulation are: %r',
                 self._params.bc_type)

    logging.info('Time integration scheme is predefined by the solver. The '
                 ' time integration scheme specified in the config file is not '
                 'activated.')

    self.dbg = components_debug.ComponentsDebug(
        self._params) if self._params.dbg else None

    self.monitor = monitor.Monitor(params)
    self.velocity = velocity_model.Velocity(
        self._kernel_op,
        self._params,
        self.thermodynamics,
        self.monitor,
        self._ib,
    )

    self.scalars = scalars_model.Scalars(self._kernel_op, self._params,
                                         self._ib, self.dbg)
    self.pressure = pressure_model.Pressure(self._kernel_op, self._params,
                                            self.thermodynamics, self.monitor)

    self._update_additional_states = (
        params.additional_states_update_fn is not None)

    self._updated_additional_states_keys = self.dbg.debugging_states_names(
    ) if self.dbg is not None else []
    if params.use_sgs:
      self._updated_additional_states_keys += ['nu_t', 'drho']
    self._updated_additional_states_keys += self.monitor.data.keys()
    # Diagnostic variables for output.
    self._diagnostic_var_names = common.KEYS_DIAGNOSTICS_BUOYANCY
    self._updated_additional_states_keys += list(
        self._diagnostic_var_names)
    # These are states used in inner simulation step and they are "transient" in
    # the sense unless explicitly specified in the additional states
    # configuration, they are not available outside the inner simulation steps.
    self._transient_var_names = ('rho_thermal', 'drho', 'dp')
    self._updated_additional_states_keys += list(
        self._transient_var_names)

  def _exchange_halos(self, f, bc_f, replica_id, replicas):
    """Performs halo exchange for the variable f."""
    return halo_exchange.inplace_halo_exchange(
        f,
        self._halo_dims,
        replica_id,
        replicas,
        self._replica_dims,
        self._params.periodic_dims,
        bc_f,
        width=self._params.halo_width)

  def _update_initial_states(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Keeps the initial states before time advancement."""
    states_0 = {
        'rho': states['rho'],
        'u': states['u'],
        'v': states['v'],
        'w': states['w'],
        'p': states['p'],
    }

    for varname in self._params.transport_scalars_names:
      states_0.update({
          varname:
              self.scalars.exchange_scalar_halos(states[varname], varname,
                                                 replica_id, replicas)
      })

    # Needs linear extrapolation for the guess of density to start each
    # iteration. Not doing it here because it's hard to keep track of the old
    # states.
    rho_0, _ = self.thermodynamics.update_density(self._kernel_op, replica_id,
                                                  replicas, states_0,
                                                  additional_states)
    states_0.update({
        'rho':
            rho_0,
        'rho_thermal':
            self.thermodynamics.update_thermal_density(states_0,
                                                       additional_states),
        'drho': tf.nest.map_structure(tf.zeros_like, rho_0),
        'buoyancy_u': tf.nest.map_structure(tf.zeros_like, rho_0),
        'buoyancy_v': tf.nest.map_structure(tf.zeros_like, rho_0),
        'buoyancy_w': tf.nest.map_structure(tf.zeros_like, rho_0),
    })

    states_0.update(
        self.velocity.update_velocity_halos(replica_id, replicas, states_0,
                                            states_0))

    states_0.update({
        'rho_u': tf.nest.map_structure(tf.multiply, rho_0, states_0['u']),
        'rho_v': tf.nest.map_structure(tf.multiply, rho_0, states_0['v']),
        'rho_w': tf.nest.map_structure(tf.multiply, rho_0, states_0['w']),
    })
    for varname in self._params.transport_scalars_names:
      states_0.update({
          'rho_{}'.format(varname): tf.nest.map_structure(
              tf.multiply, rho_0, states_0[varname])
      })

    states_0.update(
        self.pressure.update_pressure_halos(replica_id, replicas, states_0,
                                            additional_states))
    states_0.update({'dp': tf.nest.map_structure(tf.zeros_like, states_0['p'])})

    # Reserve a dictionary of variables for diagnostics. The name of these
    # diagnostic variables has to be in the `additional_states` for them to be
    # outputs.
    if 'nu_t' in additional_states.keys():
      states_0.update({'nu_t': additional_states['nu_t']})

    if self.dbg is not None:
      states_0.update(self.dbg.generate_initial_states(True))

    for monitor_var in self.monitor.data:
      states_0.update({monitor_var: additional_states[monitor_var]})

    return states_0

  def step(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      step_id: tf.Tensor,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      keyed_infeed_queue_elements: Optional[FlowFieldMap] = None,
  ):
    """Simulation step update function.

    Args:
      replica_id: The tf.Tensor containing the replica id.
      replicas: The replicas. In particular, a numpy array that maps grid
        coordinates to replica id numbers.
      step_id: A `tf.Tensor` denoting the current step id.
      states: A ordered dictionary that holds: 1. The density (in kg/m^3). 2.
        The velocity component in dim 0 (in m/s). 3. The velocity component in
        dim 1 (in m/s). 4. The velocity component in dim 2 (in m/s). 5. The
        pressure (in Pa) 6. Scalars.
      additional_states: A dictionary that holds constants that will be used in
        the simulation, e.g. boundary conditions.
      keyed_infeed_queue_elements: Unused.

    Returns:
      (rho, u, v, w, rho_u, rho_v, rho_w, p, sc): The state of density,
      velocity, momentum, pressure, and scalars for the next time step.
    """
    del keyed_infeed_queue_elements

    self.scalars.prestep(replica_id, replicas, additional_states)
    self.velocity.prestep(replica_id, replicas, additional_states)
    self.pressure.prestep(replica_id, replicas, additional_states)

    # Step 1: store the variables at the previous time step with boundary
    # conditions updated.
    with tf.name_scope('init_states_update'):
      init_states = self._update_initial_states(replica_id, replicas, states,
                                                additional_states)
    states_0 = init_states

    def update_step(i, states_k):
      """Defines a predictor-corrector iteration."""
      rho_mid = tf.nest.map_structure(
          common_ops.average, states_k['rho'], states_0['rho'])
      states_k.update({
          'rho_u': tf.nest.map_structure(tf.multiply, rho_mid, states_k['u']),
          'rho_v': tf.nest.map_structure(tf.multiply, rho_mid, states_k['v']),
          'rho_w': tf.nest.map_structure(tf.multiply, rho_mid, states_k['w']),
      })

      # Step 2: Update all scalars in conservative form. Boundary conditions are
      # not enforced for the conservative scalars, but they are enforced for the
      # temporary primitive scalars.
      with tf.name_scope('scalar_prediction_step'):
        scalar_prediction_states, mass_source = self.scalars.prediction_step(
            replica_id, replicas, states_k, states_0, additional_states)
      states_k.update(scalar_prediction_states)

      # Step 3: Update the density with the temporary primitive scalars. NB:
      # Because the boundary conditions are enforced for the temporary primitive
      # variables, the density at the boundary is valid.
      if self._params.solver_mode == thermodynamics_pb2.Thermodynamics.LOW_MACH:
        with tf.name_scope('density_update_low_mach'):
          rho, drho = self.thermodynamics.update_density(self._kernel_op,
                                                         replica_id, replicas,
                                                         states_k,
                                                         additional_states,
                                                         states_0)
          rho_thermal = self.thermodynamics.update_thermal_density(
              states_k, additional_states)
        states_k.update({
            'rho': rho,
            'rho_thermal': rho_thermal,
            'drho': drho,
        })
      else:
        with tf.name_scope('density_update_anelastic'):
          rho_thermal = self.thermodynamics.update_thermal_density(
              states_k, additional_states)
        states_k.update({
            'rho_thermal': rho_thermal,
        })

      with tf.name_scope('pressure_halo_update'):
        pressure_update_halo_states = self.pressure.update_pressure_halos(
            replica_id, replicas, {
                'rho_u': states_k['rho_u'],
                'rho_v': states_k['rho_v'],
                'rho_w': states_k['rho_w'],
                'u': states_k['u'],
                'v': states_k['v'],
                'w': states_k['w'],
                'rho': rho_mid,
                'rho_thermal': states_k['rho_thermal'],
                'p': states_k['p'],
            },
            additional_states,
        )
      states_k.update(pressure_update_halo_states)

      # Step 4: Update all primitive scalars with the latest density. Boundary
      # conditions are enforced for these scalars.
      if (
          self._params.enable_scalar_recorrection
          and self._params.solver_mode
          != thermodynamics_pb2.Thermodynamics.ANELASTIC
      ):
        with tf.name_scope('scalar_correction'):
          scalar_correction_states = self.scalars.correction_step(
              replica_id, replicas, states_k, states_0, additional_states
          )
        states_k.update(scalar_correction_states)

      # Step 5: Time advance the momentum equations to yield provisional
      # estimates for the velocity components. Boundary conditions are enforced
      # for velocity components only.
      with tf.name_scope('velocity_prediction'):
        velocity_prediction_states = self.velocity.prediction_step(
            replica_id, replicas, states_k, states_0, additional_states
        )
      states_k.update(velocity_prediction_states)

      # Step 6: Get the pressure correction. NB: the boundary condition for
      # density is set to be Neumann everywhere.
      with tf.name_scope('pressure_step'):
        additional_states_with_mass_source = dict(additional_states) | {
            'mass_source': mass_source
        }
        pressure_step_states = self.pressure.step(
            replica_id,
            replicas,
            states_k,
            states_0,
            additional_states_with_mass_source,
            i,
            step_id,
        )
      states_k.update(pressure_step_states)

      # Step 7: Update the velocity and pressure.
      with tf.name_scope('velocity_correction'):
        velocity_correction_states = self.velocity.correction_step(
            replica_id, replicas, states_k, states_0, additional_states
        )
      states_k.update(velocity_correction_states)

      return (i + 1, states_k)

    condition = lambda i, states_k: i < self._params.corrector_nit

    i0 = tf.constant(0)
    states_init = dict(states_0)

    _, states_new = tf.while_loop(
        cond=condition,
        body=update_step,
        loop_vars=(i0, states_init),
        back_prop=False)

    # For those additional states values that are not meant to be changed by the
    # inner solver (indicated by the `updated_additional_state_keys`), revert
    # their values back to the original values.
    states_new.update({
        key: val
        for key, val in additional_states.items()
        if key not in self._updated_additional_states_keys
    })

    monitor_states = self.monitor.compute_analytics(
        states_new, replicas, (tf.convert_to_tensor(self._params.dt),
                               additional_states[TIME_VARNAME]))
    states_new.update(monitor_states)

    for varname in ['u', 'v', 'w'] + self._params.transport_scalars_names:
      states_new.pop('rho_{}'.format(varname))

    # Removes diagnostic variables and transient variables that are not
    # specified as additional_states.
    for var_name in (
        list(self._diagnostic_var_names) + list(self._transient_var_names)):
      if var_name not in additional_states:
        states_new.pop(var_name)

    return states_new
