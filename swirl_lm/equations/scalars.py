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

"""A library for solving scalar transport equations.

   Because of the staggering in time, and density is at the same time step as
   scalars, the average density (old & new) is at the same time step as the
   velocity at the new time step.
"""

import functools
from typing import Optional, Text

from absl import logging
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import physical_variable_keys_manager
from swirl_lm.boundary_condition import immersed_boundary_method
from swirl_lm.communication import halo_exchange
from swirl_lm.equations import common
from swirl_lm.equations import utils as eq_utils
from swirl_lm.equations.source_function import humidity
from swirl_lm.equations.source_function import potential_temperature
from swirl_lm.equations.source_function import scalar_generic
from swirl_lm.equations.source_function import total_energy
from swirl_lm.numerics import diffusion
from swirl_lm.numerics import numerics_pb2
from swirl_lm.physics.atmosphere import microphysics_utils
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.physics.thermodynamics import thermodynamics_pb2
from swirl_lm.utility import common_ops
from swirl_lm.utility import components_debug
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap

# Density keys.
_KEY_RHO = common.KEY_RHO

# Velocity keys.
_KEY_U = common.KEY_U
_KEY_V = common.KEY_V
_KEY_W = common.KEY_W


class Scalars(object):
  """A library for solving scalar transport equations."""

  def __init__(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      params: parameters_lib.SwirlLMParameters,
      ib: Optional[immersed_boundary_method.ImmersedBoundaryMethod] = None,
      dbg: Optional[components_debug.ComponentsDebug] = None,
  ):
    """Initializes the velocity update library."""
    self._kernel_op = kernel_op
    self._params = params
    self._halo_dims = (0, 1, 2)
    self._replica_dims = (0, 1, 2)

    self.diffusion_fn = diffusion.diffusion_scalar(self._params)

    self._bc_manager = (
        physical_variable_keys_manager.BoundaryConditionKeysHelper())
    self._src_manager = (
        physical_variable_keys_manager.SourceKeysHelper())

    self.thermodynamics = thermodynamics_manager.thermodynamics_factory(
        self._params)

    self.microphysics = None
    if self._params.microphysics is not None:
      self.microphysics = microphysics_utils.select_microphysics(
          self._params, self.thermodynamics)

    self._bc = {
        varname: bc_val
        for varname, bc_val in self._params.bc.items()
        if varname in self._params.transport_scalars_names
    }

    self._source = {
        sc.name: None for sc in self._params.scalars if sc.solve_scalar
    }

    self._ib = ib
    if self._ib is None:
      self._ib = immersed_boundary_method.immersed_boundary_method_factory(
          self._params
      )

    self._dbg = dbg

    # Get functions that computes terms in tranport equations for all scalars.
    self._scalar_model = {}
    for scalar_name in self._params.transport_scalars_names:
      common_args = (
          self._kernel_op,
          self._params,
          scalar_name,
          self.thermodynamics,
      )
      if scalar_name in potential_temperature.POTENTIAL_TEMPERATURE_VARNAME:
        scalar_model = potential_temperature.PotentialTemperature(
            *common_args, self.microphysics
        )
      elif scalar_name in humidity.HUMIDITY_VARNAME:
        scalar_model = humidity.Humidity(*common_args, self.microphysics)
      elif scalar_name == 'e_t':
        scalar_model = total_energy.TotalEnergy(*common_args, self.microphysics)
      else:
        scalar_model = scalar_generic.ScalarGeneric(*common_args)
      self._scalar_model[scalar_name] = scalar_model

  @tf.function
  def _scalar_transport_equation(
      self, conv_x, conv_y, conv_z, diff_x, diff_y, diff_z, src
  ):
    """Defines right-hand side function of a scalar transport equation."""
    logging.info('Tracing `_scalar_transport_equation`.')
    return -(conv_x + conv_y + conv_z) + (diff_x + diff_y + diff_z) + src

  def _exchange_halos(self, f, bc_f, replica_id, replicas):
    """Performs halo exchange for the variable f."""
    @tf.function
    def do_exchange_halos(f, bc_f, replica_id):
      logging.info('Tracing `do_exchange_halos`.')
      return halo_exchange.inplace_halo_exchange(
          f,
          self._halo_dims,
          replica_id,
          replicas,
          self._replica_dims,
          self._params.periodic_dims,
          bc_f,
          width=self._params.halo_width)

    return do_exchange_halos(f, bc_f, replica_id)

  def exchange_scalar_halos(
      self,
      f: FlowFieldVal,
      name: Text,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
  ) -> FlowFieldVal:
    """Performs halo exchange with updated boundary conditions.

    Note that the boundary condition can be adjusted prior to the halo exchange.
    For example, values in the ghost cells can be updated based on the
    transient fluid field and the boundary if the boundary condition type is
    specified as Dirichlet (not included currently for better stability).

    Args:
      f: The 3D tensor field to which the halo exhange is performed.
      name: The name of the variable. It's used to retrieve the boundary
        condition from the boundary condition library.
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.

    Returns:
      A 3D tensor with values in halos updated.
    """
    bc = self._bc
    return self._exchange_halos(f, bc[name], replica_id, replicas)

  def _scalar_update(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      scalar_name: Text,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      dbg: bool = False,
  ):
    """Provides a function that computes the RHS function of a generic scalar.

    This function provides a wrapper for the function that computes the rhs
    `f(phi)` of the scalar equation in functional form, i.e.
    `drho_phi / dt = f(phi)`.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      scalar_name: The name of the scalar.
      states: A dictionary that holds field variables that are essential to
        compute the right hand side function of the scalar transport equation.
        Must include the following fields: 'rho_u', 'rho_v', 'rho_w', 'p',
        'rho', 'u', 'v', 'w'.
      additional_states: Helper states that are required by the scalar transport
        equation. Must contain 'diffusivity'.
      dbg: A flag of whether to use the debug mode. If `True`, the returned RHS
        function returns the convection terms, the diffusion terms, and the
        external source term instead of the sum of all these terms (i.e. the
        actual RHS term).

    Returns:
      scalar_function: A function that computes the `f(phi)` and the potential
      source term to the mass.
    """
    logging.info('Tracing `_scalar_update`.')
    source = (
        self._source[scalar_name]
        if self._source[scalar_name] is not None
        else tf.nest.map_structure(tf.zeros_like, states[_KEY_RHO])
    )

    @tf.function
    def scalar_function(phi: FlowFieldVal):
      """Computes the functional RHS for the three momentum equations.

      Args:
        phi: The scalar field.

      Returns:
        A `FlowFieldVal` representing the RHS of the scalar transport
        equation.
      """
      logging.info('Tracing `scalar_function`.')
      conv = self._scalar_model[scalar_name].convection_fn(
          replica_id, replicas, phi, states, additional_states
      )

      diff = self._scalar_model[scalar_name].diffusion_fn(
          replica_id, replicas, phi, states, additional_states
      )

      source_additional = self._scalar_model[scalar_name].source_fn(
          replica_id, replicas, phi, states, additional_states
      )

      source_all = tf.nest.map_structure(
          tf.math.add, source, source_additional.total
      )

      if dbg:
        return {
            'conv_x': conv[0],
            'conv_y': conv[1],
            'conv_z': conv[2],
            'diff_x': diff[0],
            'diff_y': diff[1],
            'diff_z': diff[2],
            'source': source_all,
        }

      def scalar_transport_equation(
          conv_x, conv_y, conv_z, diff_x, diff_y, diff_z, src
      ):
        """Defines right-hand side function of a scalar transport equation."""
        return self._scalar_transport_equation(
            conv_x, conv_y, conv_z, diff_x, diff_y, diff_z, src)

      rhs = tf.nest.map_structure(
          scalar_transport_equation,
          conv[0],
          conv[1],
          conv[2],
          diff[0],
          diff[1],
          diff[2],
          source_all,
      )

      if self._ib is not None:
        rho_sc_name = 'rho_{}'.format(scalar_name)
        rhs_name = self._ib.ib_rhs_name(rho_sc_name)
        helper_states = {rhs_name: rhs}
        for helper_var_name in ('ib_interior_mask', 'ib_boundary'):
          if helper_var_name in additional_states:
            helper_states[helper_var_name] = additional_states[
                helper_var_name
            ]

        rhs_ib_updated = self._ib.update_forcing(
            self._kernel_op, replica_id, replicas, {
                rho_sc_name:
                    tf.nest.map_structure(tf.math.multiply, states[_KEY_RHO],
                                          phi)
            }, helper_states)
        rhs = rhs_ib_updated[rhs_name]

      return rhs, source_additional.mass

    return scalar_function

  def prestep(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      additional_states: FlowFieldMap,
  ) -> None:
    """Updates additional information required for scalars step.

    This function is called before the beginning of each time step. It updates
    the boundary conditions of all scalars. It also updates the source term of
    each scalar. These information will be hold within this helper object.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      additional_states: A dictionary that holds constants that will be used in
        the simulation, e.g. boundary conditions, forcing terms.
    """
    del replica_id, replicas

    # Parse additional states to extract boundary conditions.
    self._bc = self._bc_manager.update_helper_variable_from_additional_states(
        additional_states, self._params.halo_width, self._bc)

    # Parse additional states to extract external source/forcing terms.
    self._source.update(
        self._src_manager.update_helper_variable_from_additional_states(
            additional_states))

  def prediction_step(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      states_0: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> tuple[FlowFieldMap, FlowFieldVal]:
    """Predicts the scalars from the generic scalar transport equation.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      states: A dictionary that holds flow field variables from the latest
        prediction.
      states_0: A dictionary that holds flow field variables from the previous
        time step.
      additional_states: A dictionary that holds constants that will be used in
        the simulation, e.g. boundary conditions, forcing terms.

    Returns:
      The predicted scalars and all debugging terms (if required).
    """
    exchange_halos = functools.partial(
        self.exchange_scalar_halos, replica_id=replica_id, replicas=replicas)

    states_mid = {}
    states_mid.update(states)
    states_mid.update(
        {_KEY_RHO: common_ops.average(states[_KEY_RHO], states_0[_KEY_RHO])})
    states_mid.update(
        {'rho_thermal': common_ops.average(states['rho_thermal'],
                                           states_0['rho_thermal'])})

    for sc_name in self._params.transport_scalars_names:
      states_mid.update(
          {sc_name: common_ops.average(states[sc_name], states_0[sc_name])})

    updated_scalars = {}
    mass_source = tf.nest.map_structure(tf.zeros_like, states[_KEY_RHO])

    for sc_name in self._params.transport_scalars_names:
      sc_mid = states_mid[sc_name]
      diffusivity = self._scalar_model[sc_name].get_diffusivity(
          replicas, sc_mid, states, additional_states
      )
      # Bounds diffusivity here instead of in the `get_diffusivity` function
      # to apply the constraint to all scalars regardless of their specific
      # implementations.
      diffusivity = eq_utils.bound_viscosity(
          diffusivity, additional_states, self._params
      )
      helper_states = {'diffusivity': diffusivity}
      helper_states.update(additional_states)
      scalar_rhs_fn = self._scalar_update(
          replica_id, replicas, sc_name, states_mid, helper_states
      )

      @tf.function
      def time_advance_cn_explicit(rhs, sc_name):
        updated_vars = {}
        if (self._params.solver_mode ==
            thermodynamics_pb2.Thermodynamics.ANELASTIC):
          alpha = tf.nest.map_structure(tf.math.reciprocal, states[_KEY_RHO])
          new_sc = tf.nest.map_structure(
              lambda sc, b, a: sc + self._params.dt * b * a, states_0[sc_name],
              rhs, alpha)
          updated_vars.update({sc_name: exchange_halos(new_sc, sc_name)})
        else:  # solver_mode == thermodynamics_pb2.Thermodynamics.LOW_MACH
          new_sc = tf.nest.map_structure(lambda a, b: a + self._params.dt * b,
                                         states_0['rho_{}'.format(sc_name)],
                                         rhs)
          updated_vars.update({'rho_{}'.format(sc_name): new_sc})

          # Updates scalar, to be consistent with rho * scalar.
          updated_vars.update({
              sc_name:
                  exchange_halos(
                      tf.nest.map_structure(
                          tf.math.divide,
                          updated_vars['rho_{}'.format(sc_name)],
                          states[_KEY_RHO]), sc_name),
          })
        return updated_vars

      # Time advancement for rho * scalar.
      time_scheme = self._params.scalar_time_integration_scheme(sc_name)
      if (time_scheme ==
          numerics_pb2.TimeIntegrationScheme.TIME_SCHEME_CN_EXPLICIT_ITERATION):
        rhs, src_rho = scalar_rhs_fn(sc_mid)
        updated_scalars.update(time_advance_cn_explicit(rhs, sc_name))
        if src_rho is not None:
          mass_source = tf.nest.map_structure(tf.math.add, mass_source, src_rho)
      else:
        raise ValueError(
            'Time integration scheme %s is not supported yet for scalars.' %
            time_scheme)

      if self._dbg is not None:
        terms = (
            self._scalar_update(replica_id, replicas, sc_name, states_mid,
                                helper_states, True)(sc_mid))
        updated_scalars.update(
            self._dbg.update_scalar_terms(sc_name, terms, diffusivity))

    # Applies the marker-and-cell or Cartesian grid method if requested in the
    # config file.
    if self._ib is not None:
      updated_scalars = self._ib.update_states(self._kernel_op, replica_id,
                                               replicas, updated_scalars,
                                               additional_states, self._bc)

    return updated_scalars, mass_source  # pytype: disable=bad-return-type

  def correction_step(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      states_0: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Updates the primitive scalars after the density correction.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      states: A dictionary that holds flow field variables from the latest
        prediction.
      states_0: A dictionary that holds flow field variables from the previous
        time step.
      additional_states: A dictionary that holds constants that will be used in
        the simulation.

    Returns:
      The updated primitive scalars with density correction.
    """
    del states_0

    exchange_halos = functools.partial(
        self.exchange_scalar_halos, replica_id=replica_id, replicas=replicas)

    scalars = {}

    for sc_name in self._params.transport_scalars_names:
      sc_buf = tf.nest.map_structure(tf.math.divide,
                                     states['rho_{}'.format(sc_name)],
                                     states[_KEY_RHO])

      # Applies the marker-and-cell or Cartesian grid method if requested in the
      # config file. Halo exchange will be performed after the solid boundary
      # condition is applied.
      if self._ib is not None:
        sc_buf = self._ib.update_states(self._kernel_op, replica_id, replicas,
                                        {sc_name: sc_buf}, additional_states,
                                        self._bc)[sc_name]

      scalars.update({sc_name: exchange_halos(sc_buf, sc_name)})

    return scalars
