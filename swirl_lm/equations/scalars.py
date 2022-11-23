# Copyright 2022 The swirl_lm Authors.
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

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import physical_variable_keys_manager
from swirl_lm.boundary_condition import immersed_boundary_method
from swirl_lm.communication import halo_exchange
from swirl_lm.equations import common
from swirl_lm.equations import utils as eq_utils
from swirl_lm.numerics import calculus
from swirl_lm.numerics import convection
from swirl_lm.numerics import diffusion
from swirl_lm.numerics import filters
from swirl_lm.numerics import numerics_pb2
from swirl_lm.physics import constants
from swirl_lm.physics.atmosphere import cloud
from swirl_lm.physics.atmosphere import precipitation
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.physics.thermodynamics import thermodynamics_pb2
from swirl_lm.physics.thermodynamics import water
from swirl_lm.physics.turbulence import sgs_model
from swirl_lm.utility import common_ops
from swirl_lm.utility import components_debug
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap

# A small number that's used as the threshold for the gravity vector. If the
# absolute value of a gravity component is less than this threshold, it is
# considered as 0 when computing the free slip wall boundary condition.
_G_THRESHOLD = 1e-6

# Parameters required by the radiation model. Reference:
# Stevens, Bjorn, Chin-Hoh Moeng, Andrew S. Ackerman, Christopher S.
# Bretherton, Andreas Chlond, Stephan de Roode, James Edwards, et al. 2005.
# “Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine
# Stratocumulus.” Monthly Weather Review 133 (6): 1443–62.
_F0 = 70.0
_F1 = 22.0
_KAPPA = 85.0
_ALPHA_Z = 1.0
# The subsidence velocity coefficient.
_D = 3.75e-6
# The initial height of the cloud, in units of m
_ZI = 840.0

# Density keys.
_KEY_RHO = common.KEY_RHO
# Pressure keys.
_KEY_P = common.KEY_P

# Velocity keys.
_KEY_U = common.KEY_U
_KEY_V = common.KEY_V
_KEY_W = common.KEY_W

# Momentum keys.
_KEY_RHO_U = common.KEY_RHO_U
_KEY_RHO_V = common.KEY_RHO_V
_KEY_RHO_W = common.KEY_RHO_W


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

    if isinstance(self.thermodynamics.model, water.Water):
      self.precipitation = precipitation.Precipitation(
          self.thermodynamics.model)
      self.cloud = cloud.Cloud(
          self.thermodynamics.model)

    self._use_sgs = self._params.use_sgs
    filter_widths = (self._params.dx, self._params.dy, self._params.dz)
    if self._use_sgs:
      self._sgs_model = sgs_model.SgsModel(self._kernel_op, filter_widths,
                                           params.sgs_model)

    self._bc = {
        varname: bc_val
        for varname, bc_val in self._params.bc.items()
        if varname in self._params.transport_scalars_names
    }

    self._source = {
        sc.name: None for sc in self._params.scalars if sc.solve_scalar
    }

    self._g_vec = (
        self._params.gravity_direction if self._params.gravity_direction else [
            0.0,
        ] * 3)

    # Find the direction of gravity. Only vector along a particular dimension is
    # considered currently.
    self._g_dim = None
    for i in range(3):
      if np.abs(np.abs(self._g_vec[i]) - 1.0) < _G_THRESHOLD:
        self._g_dim = i
        break

    self._ib = ib if ib is not None else (
        immersed_boundary_method.immersed_boundary_method_factory(self._params))

    self._dbg = dbg

    self._scalars = {}
    for scalar in self._params.scalars:
      self._scalars.update({scalar.name: scalar})

    self._grad_central = [
        lambda f: self._kernel_op.apply_kernel_op_x(f, 'kDx'),
        lambda f: self._kernel_op.apply_kernel_op_y(f, 'kDy'),
        lambda f: self._kernel_op.apply_kernel_op_z(f, 'kDz', 'kDzsh'),
    ]

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

  def _conservative_scalar_convection(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      phi: FlowFieldVal,
      rho_u: FlowFieldVal,
      p: FlowFieldVal,
      rho: FlowFieldVal,
      h: float,
      dt: float,
      dim: int,
      sc_name: Text,
      zz: Optional[FlowFieldVal] = None,
      apply_correction: bool = False,
  ) -> FlowFieldVal:
    """Computes the convection term for the conservative scalar."""
    # Computes the gravitational force for the face flux correction.
    if np.abs(self._g_vec[dim]) < _G_THRESHOLD or not apply_correction:
      gravity = None
    else:
      zz = zz if zz is not None else [tf.zeros_like(phi_i) for phi_i in phi]

      drho = filters.filter_op(
          self._kernel_op, [
              rho_mix_i - rho_ref_i for rho_mix_i, rho_ref_i in zip(
                  rho, self.thermodynamics.rho_ref(zz))
          ],
          order=2)
      gravity = [drho_i * self._g_vec[dim] * constants.G for drho_i in drho]

    momentum_component = common.KEYS_MOMENTUM[dim]

    return convection.convection_term(
        self._kernel_op,
        replica_id,
        replicas,
        phi,
        rho_u,
        p,
        h,
        dt,
        dim,
        bc_types=tuple(self._params.bc_type[dim]),
        varname=momentum_component,
        halo_width=self._params.halo_width,
        scheme=self._scalars[sc_name].scheme,
        src=gravity,
        apply_correction=apply_correction)

  def _generic_scalar_update(
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
      scalar_function: A function that computes the `f(phi)`.
    """

    h = (self._params.dx, self._params.dy, self._params.dz)
    dt = self._params.dt
    momentum = [states[_KEY_RHO_U], states[_KEY_RHO_V], states[_KEY_RHO_W]]

    source = self._source[scalar_name] if self._source[scalar_name] else [
        tf.zeros_like(rho_i) for rho_i in states[_KEY_RHO]
    ]

    # Helper variables required by the Monin-Obukhov similarity theory.
    helper_variables_most = {
        'u': states[_KEY_U],
        'v': states[_KEY_V],
        'w': states[_KEY_W],
    }

    def scalar_function(phi: FlowFieldVal):
      """Computes the functional RHS for the three momentum equations.

      Args:
        phi: The scalar field.

      Returns:
        A `FlowFieldVal` representing the RHS of the scalar transport
        equation.
      """
      conv = [
          self._conservative_scalar_convection(replica_id, replicas, phi,  # pylint: disable=g-complex-comprehension
                                               momentum[i], states[_KEY_P],
                                               states[_KEY_RHO], h[i], dt, i,
                                               scalar_name) for i in range(3)
      ]

      # Because only the layer close to the ground will be used in the Monin
      # Obukhov similarity closure model, the temperature and potential
      # temperature are equal. Note that the helper variables are used only
      # in the 'T' and 'theta' transport equations.
      if scalar_name in ('T', 'theta'):
        helper_variables_most.update({'theta': phi})

      diff = self.diffusion_fn(
          self._kernel_op,
          replica_id,
          replicas,
          phi,
          states[_KEY_RHO],
          additional_states['diffusivity'],
          h,
          scalar_name=scalar_name,
          helper_variables=helper_variables_most)

      if dbg:
        return {
            'conv_x': conv[0],
            'conv_y': conv[1],
            'conv_z': conv[2],
            'diff_x': diff[0],
            'diff_y': diff[1],
            'diff_z': diff[2],
            'source': source,
        }

      equation_terms = zip(conv[0], conv[1], conv[2], diff[0], diff[1], diff[2],
                           source)

      rhs = [
          -(conv_x_i + conv_y_i + conv_z_i) + (diff_x_i + diff_y_i + diff_z_i) +
          src_i for conv_x_i, conv_y_i, conv_z_i, diff_x_i, diff_y_i, diff_z_i,
          src_i in equation_terms
      ]

      if self._ib is not None:
        rho_sc_name = 'rho_{}'.format(scalar_name)
        rhs_name = self._ib.ib_rhs_name(rho_sc_name)
        helper_states = {
            rhs_name: rhs,
            'ib_interior_mask': additional_states['ib_interior_mask'],
        }
        rhs_ib_updated = self._ib.update_forcing(
            self._kernel_op, replica_id, replicas, {
                rho_sc_name:
                    tf.nest.map_structure(tf.math.multiply, states[_KEY_RHO],
                                          phi)
            }, helper_states)
        rhs = rhs_ib_updated[rhs_name]

      return rhs

    return scalar_function

  def _theta_li_update(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      dbg: bool = False,
  ):
    """Generates a functor that computes RHS of potential temperature equation.

    This function returns a functor that computes the rhs `f(theta_li)` of the
    liquid-ice potential temperature update equation
    `d theta_li / dt = f(theta_li)`.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      states: A dictionary that holds field variables that are essential to
        compute the right hand side function of the scalar transport equation.
        Must include the following fields: 'u', 'v', 'w', 'rho_u', 'rho_v',
          'rho_w', 'p', 'rho', 'theta_li', 'q_t'.
      additional_states: Helper states that are required by the scalar transport
        equation. Must contain 'diffusivity'. If SGS model is used, must also
        contain 'nu_t'. If 'zz' is not in `additional_states`, assumes the flow
        field is independent of height.
      dbg: A flag of whether to use the debug mode. If `True`, the returned RHS
        function returns the convection terms, the diffusion terms, and the
        external source term instead of the sum of all these terms (i.e. the
        actual RHS term).

    Returns:
      scalar_function: A function that computes the `f(theta_li)`.

    Raises:
      ValueError: If the thermodynamics model is not `Water`.
    """
    if not isinstance(self.thermodynamics.model, water.Water):
      raise ValueError('The thermodynamics model has to be `Water` for the '
                       'liquid-ice potential temperature equation. Current '
                       'model is {}'.format(
                           self.thermodynamics.model))

    h = (self._params.dx, self._params.dy, self._params.dz)
    dt = self._params.dt
    momentum = [states[_KEY_RHO_U], states[_KEY_RHO_V], states[_KEY_RHO_W]]
    zz = additional_states.get(
        'zz', tf.nest.map_structure(tf.zeros_like, states[_KEY_RHO_U]))

    # Helper variables required by the Monin-Obukhov similarity theory.
    helper_variables_most = {
        'u': states[_KEY_U],
        'v': states[_KEY_V],
        'w': states[_KEY_W],
    }

    include_radiation = (
        self._scalars['theta_li'].HasField('potential_temperature') and
        self._scalars['theta_li'].potential_temperature.include_radiation and
        self._g_dim is not None)

    include_subsidence = (
        self._scalars['theta_li'].HasField('potential_temperature') and
        self._scalars['theta_li'].potential_temperature.include_subsidence and
        self._g_dim is not None)

    source_ext = (
        self._source['theta_li'] if self._source['theta_li'] else
        tf.nest.map_structure(tf.zeros_like, states[_KEY_RHO]))

    def scalar_function(theta_li: FlowFieldVal):
      """Computes functional RHS for liquid-ice potential temperature equation.

      Args:
        theta_li: The liquid-ice potential temperature field.

      Returns:
        A `FlowFieldVal` representing the RHS of the liquid-ice potential
        temperature equation.
      """
      # Compute the temperature.
      q_t = states['q_t']
      rho = states[_KEY_RHO]
      rho_thermal = states['rho_thermal']

      temperature = self.thermodynamics.model.saturation_adjustment(
          'theta_li', theta_li, rho_thermal, q_t, zz)

      # Compute the potential temperature.
      buf = self.thermodynamics.model.potential_temperatures(
          temperature, q_t, rho_thermal, zz)
      theta = buf['theta']
      helper_variables_most.update({'theta': theta})

      q_l, q_i = self.thermodynamics.model.equilibrium_phase_partition(
          temperature, rho_thermal, q_t)

      # Compute the source terms due to dilatation, wind shear, radiation,
      # subsidence velocity, and precipitation.
      source = tf.nest.map_structure(tf.zeros_like, zz)

      if include_radiation:
        halos = [self._params.halo_width] * 3
        f_r = self.cloud.source_by_radiation(q_l, rho_thermal, zz,
                                             h[self._g_dim], self._g_dim, halos,
                                             replica_id, replicas)

        def radiation_source_fn(rho_i, f_r_i):
          return - rho_i * f_r_i

        rad_src = tf.nest.map_structure(radiation_source_fn, rho, f_r)
        source = tf.nest.map_structure(lambda f: f / (2.0 * h[self._g_dim]),
                                       self._grad_central[self._g_dim](rad_src))

      cp_m = self.thermodynamics.model.cp_m(q_t, q_l, q_i)
      cp_m_inv = tf.nest.map_structure(tf.math.reciprocal, cp_m)
      exner_inv = self.thermodynamics.model.exner_inverse(
          rho_thermal, q_t, temperature, zz)
      cp_m_exner_inv = tf.nest.map_structure(tf.math.multiply, cp_m_inv,
                                             exner_inv)
      source = tf.nest.map_structure(tf.math.multiply, cp_m_exner_inv, source)

      if include_subsidence:
        src_subsidence = eq_utils.source_by_subsidence_velocity(
            self._kernel_op, rho, zz, h[self._g_dim], theta_li, self._g_dim)
        source = tf.nest.map_structure(tf.math.add, source, src_subsidence)

      # Add external source, e.g. sponge forcing.
      source = tf.nest.map_structure(tf.math.add, source, source_ext)

      # Compute the convection and diffusion terms.
      conv = [
          self._conservative_scalar_convection(  # pylint: disable=g-complex-comprehension
              replica_id, replicas, theta_li, momentum[i], states[_KEY_P], rho,
              h[i], dt, i, 'theta_li', zz) for i in range(3)
      ]

      diff = self.diffusion_fn(
          self._kernel_op,
          replica_id,
          replicas,
          theta_li,
          rho,
          additional_states['diffusivity'],
          h,
          scalar_name='theta_li',
          helper_variables=helper_variables_most)

      if dbg:
        return {
            'conv_x': conv[0],
            'conv_y': conv[1],
            'conv_z': conv[2],
            'diff_x': diff[0],
            'diff_y': diff[1],
            'diff_z': diff[2],
            'source': source,
        }

      equation_terms = zip(conv[0], conv[1], conv[2], diff[0], diff[1], diff[2],
                           source)

      rhs = [
          -(conv_x_i + conv_y_i + conv_z_i) + (diff_x_i + diff_y_i + diff_z_i) +
          src_i for conv_x_i, conv_y_i, conv_z_i, diff_x_i, diff_y_i, diff_z_i,
          src_i in equation_terms
      ]

      if self._ib is not None:
        rhs_name = self._ib.ib_rhs_name('rho_theta_li')
        helper_states = {
            rhs_name: rhs,
            'ib_interior_mask': additional_states['ib_interior_mask'],
        }
        rhs_ib_updated = self._ib.update_forcing(
            self._kernel_op, replica_id, replicas, {
                'rho_theta_li':
                    tf.nest.map_structure(tf.math.multiply, rho, theta_li)
            }, helper_states)
        rhs = rhs_ib_updated[rhs_name]

      return rhs

    return scalar_function

  def _e_t_update(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      dbg: bool = False,
  ):
    """Generates a functor that computes the RHS of the total energy equation.

    This function returns a functor that computes the rhs `f(e_t)` of the
    total energy update equation `d rho_e_t / dt = f(e_t)`.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      states: A dictionary that holds field variables that are essential to
        compute the right hand side function of the scalar transport equation.
        Must include the following fields: 'u', 'v', 'w', 'rho_u', 'rho_v',
          'rho_w', 'p', 'rho', 'e_t', 'q_t', (and 'q_r' if including
          precipitation).
      additional_states: Helper states that are required by the scalar transport
        equation. Must contain 'diffusivity'. If SGS model is used, must also
        contain 'nu_t'. If 'zz' is not in `additional_states`, assumes the flow
        field is independent of height.
      dbg: A flag of whether to use the debug mode. If `True`, the returned RHS
        function returns the convection terms, the diffusion terms, and the
        external source term instead of the sum of all these terms (i.e. the
        actual RHS term).

    Returns:
      scalar_function: A function that computes the `f(e_t)`.

    Raises:
      ValueError: If the thermodynamics model is not `Water`.
    """
    if not isinstance(self.thermodynamics.model, water.Water):
      raise ValueError('The thermodynamics model has to be `Water` for the '
                       'total energy equation. Current model is {}'.format(
                           self.thermodynamics.model))

    h = (self._params.dx, self._params.dy, self._params.dz)
    dt = self._params.dt
    velocity = [states[_KEY_U], states[_KEY_V], states[_KEY_W]]
    momentum = [states[_KEY_RHO_U], states[_KEY_RHO_V], states[_KEY_RHO_W]]
    zz = additional_states.get(
        'zz', tf.nest.map_structure(tf.zeros_like, states[_KEY_RHO_U]))
    p = self.thermodynamics.model.p_ref(zz)
    bc_p = [[(halo_exchange.BCType.NEUMANN, 0.0),] * 2] * 3
    p = self._exchange_halos(p, bc_p, replica_id, replicas)

    # Helper variables required by the Monin-Obukhov similarity theory.
    helper_variables_most = {
        'u': states[_KEY_U],
        'v': states[_KEY_V],
        'w': states[_KEY_W],
    }

    # Compute the shear stress tensor.
    if self._params.use_sgs:
      mu = [(self._params.nu + nu_t) * rho_i
            for nu_t, rho_i in zip(additional_states['nu_t'], states[_KEY_RHO])]
    else:
      mu = [self._params.nu * rho_i for rho_i in states[_KEY_RHO]]

    tau = eq_utils.shear_stress(self._kernel_op, mu, h[0], h[1], h[2],
                                states[_KEY_U], states[_KEY_V], states[_KEY_W])

    include_radiation = (
        self._scalars['e_t'].HasField('total_energy') and
        self._scalars['e_t'].total_energy.include_radiation and
        self._g_dim is not None)
    include_subsidence = (
        self._scalars['e_t'].HasField('total_energy') and
        self._scalars['e_t'].total_energy.include_subsidence and
        self._g_dim is not None)
    include_precipitation = (
        self._scalars['e_t'].HasField('total_energy') and
        self._scalars['e_t'].total_energy.include_precipitation and
        self._g_dim is not None)

    def compute_rho_u_tau(
        tau_0j: FlowFieldVal,
        tau_1j: FlowFieldVal,
        tau_2j: FlowFieldVal,
    ) -> FlowFieldVal:
      """Computes 'rho u_i tau_ij'."""
      return [
          u * tau_0j_l + v * tau_1j_l + w * tau_2j_l
          for u, v, w, tau_0j_l, tau_1j_l, tau_2j_l in zip(
              velocity[0], velocity[1], velocity[2], tau_0j, tau_1j, tau_2j)
      ]

    rho_u_tau = [
        compute_rho_u_tau(tau['xx'], tau['yx'], tau['zx']),
        compute_rho_u_tau(tau['xy'], tau['yy'], tau['zy']),
        compute_rho_u_tau(tau['xz'], tau['yz'], tau['zz'])
    ]

    # Compute the divergence of the combined source terms due to dilatation and
    # wind shear.
    div_terms = rho_u_tau

    # Prepare source terms that are computed externally.
    source_ext = self._source['e_t'] if self._source['e_t'] else [
        tf.zeros_like(rho_i) for rho_i in states[_KEY_RHO]
    ]

    def scalar_function(e_t: FlowFieldVal):
      """Computes the functional RHS for the total energy equation.

      Args:
        e_t: The total energy field.

      Returns:
        A `FlowFieldVal` representing the RHS of the total energy equation.
      """
      # Compute the temperature.
      q_t = states['q_t']
      rho = states[_KEY_RHO]
      rho_thermal = states['rho_thermal']

      if 'T' in additional_states.keys():
        temperature = additional_states['T']
      else:
        e = self.thermodynamics.model.internal_energy_from_total_energy(
            e_t, states[_KEY_U], states[_KEY_V], states[_KEY_W], zz)
        temperature = self.thermodynamics.model.saturation_adjustment(
            'e_int', e, rho_thermal, q_t)

      # Compute the potential temperature.
      if 'theta' in additional_states:
        theta = additional_states['theta']
      else:
        buf = self.thermodynamics.model.potential_temperatures(
            temperature, q_t, rho_thermal, zz)
        theta = buf['theta']
      helper_variables_most.update({'theta': theta})

      # Compute the total enthalpy.
      h_t = self.thermodynamics.model.total_enthalpy(e_t, rho_thermal, q_t,
                                                     temperature)
      if include_radiation or include_precipitation:
        q_l, _ = self.thermodynamics.model.equilibrium_phase_partition(
            temperature, rho_thermal, q_t)

      # Compute the source terms due to dilatation, wind shear, radiation,
      # subsidence velocity, and precipitation.
      if include_radiation:
        halos = [self._params.halo_width] * 3
        f_r = self.cloud.source_by_radiation(q_l, rho_thermal, zz,
                                             h[self._g_dim], self._g_dim, halos,
                                             replica_id, replicas)
        div_terms[self._g_dim] = [
            div_term_i - rho_i * f_r_i for div_term_i, rho_i, f_r_i in zip(
                div_terms[self._g_dim], rho, f_r)
        ]
      source = calculus.divergence(self._kernel_op, div_terms, h)
      if include_subsidence:
        src_subsidence = eq_utils.source_by_subsidence_velocity(
            self._kernel_op, rho, zz, h[self._g_dim], h_t, self._g_dim)
        source = tf.nest.map_structure(tf.math.add, source, src_subsidence)

      if include_precipitation:
        e_v, e_l, _ = (
            self.thermodynamics.model.internal_energy_components(temperature))
        # Get conversion rates from cloud water to rain water (for liquid and
        # vapor phase).
        q_r = states['q_r']
        cloud_liquid_to_rain_water_rate = (
            self.precipitation.cloud_liquid_to_rain_conversion_rate_kw1978(
                q_r, q_l))
        q_c = self.thermodynamics.model.saturation_excess(
            temperature, rho_thermal, q_t)
        # Find q_v from the invariant q_t = q_c + q_v = q_l + q_i + q_v.
        q_v = tf.nest.map_structure(tf.math.subtract, q_t, q_c)
        rain_water_evaporation_rate = (
            self.precipitation.rain_evaporation_rate_kw1978(
                rho_thermal, temperature, q_r, q_v, q_l, q_c))
        # Get potential energy.
        phi = tf.nest.map_structure(lambda zz_i: constants.G * zz_i, zz)
        # Calculate source terms for vapor and liquid conversions, respectively.
        # Use that c_{q_v->q_l} = -c_{q_l->q_v}, i.e. minus the evaporation
        # rate.
        source_v = tf.nest.map_structure(
            lambda e, phi, rho, c_lv: (e + phi) * rho * (-c_lv), e_v, phi, rho,
            rain_water_evaporation_rate)
        source_l = tf.nest.map_structure(
            lambda e, phi, rho, c_lr: (e + phi) * rho * c_lr, e_l, phi, rho,
            cloud_liquid_to_rain_water_rate)
        source = tf.nest.map_structure(lambda s, sv, sl: s + sv + sl, source,
                                       source_v, source_l)

      # Add external source, e.g. sponge forcing.
      source = tf.nest.map_structure(tf.math.add, source, source_ext)

      # Compute the convection and diffusion terms.
      conv = [
          self._conservative_scalar_convection(  # pylint: disable=g-complex-comprehension
              replica_id, replicas, h_t, momentum[i], states[_KEY_P], rho, h[i],
              dt, i, 'e_t', zz) for i in range(3)
      ]

      diff = self.diffusion_fn(
          self._kernel_op,
          replica_id,
          replicas,
          h_t,
          rho,
          additional_states['diffusivity'],
          h,
          scalar_name='e_t',
          helper_variables=helper_variables_most)

      if dbg:
        return {
            'conv_x': conv[0],
            'conv_y': conv[1],
            'conv_z': conv[2],
            'diff_x': diff[0],
            'diff_y': diff[1],
            'diff_z': diff[2],
            'source': source,
        }

      equation_terms = zip(conv[0], conv[1], conv[2], diff[0], diff[1], diff[2],
                           source)

      rhs = [
          -(conv_x_i + conv_y_i + conv_z_i) + (diff_x_i + diff_y_i + diff_z_i) +
          src_i for conv_x_i, conv_y_i, conv_z_i, diff_x_i, diff_y_i, diff_z_i,
          src_i in equation_terms
      ]

      if self._ib is not None:
        rhs_name = self._ib.ib_rhs_name('rho_e_t')
        helper_states = {
            rhs_name: rhs,
            'ib_interior_mask': additional_states['ib_interior_mask'],
        }
        rhs_ib_updated = self._ib.update_forcing(
            self._kernel_op, replica_id, replicas,
            {'rho_e_t': tf.nest.map_structure(tf.math.multiply, rho, e_t)},
            helper_states)
        rhs = rhs_ib_updated[rhs_name]

      return rhs

    return scalar_function

  def _humidity_update(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      scalar_name: str = 'q_t',
      dbg: bool = False,
  ):
    """Generate a functor that computes the RHS of the humidity equation.

    This function returns a functor that computes the rhs `f(q)` of the
    humidity update equation `d rho_q / dt = f(q)`, where `q` is either total
    humidity `q_t` or liquid precipitation `q_r`.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      states: A dictionary that holds field variables that are essential to
        compute the right hand side function of the scalar transport equation.
        Must include the following fields: 'rho_u', 'rho_v', 'rho_w', 'u', 'v',
          'w', 'p', 'rho', and 'e_t'.
      additional_states: Helper states that are required by the scalar transport
        equation. Must contain 'diffusivity'. If 'zz' is not in
        `additional_states`, assumes the flow field is independent of height.
      scalar_name: Name of humidity field to update, If scalar_name is 'q_t',
        update total humidity. If scalar_name is 'q_r' update the humidity field
        corresponding to liquid precipitation.
      dbg: A flag of whether to use the debug mode. If `True`, the returned RHS
        function returns the convection terms, the diffusion terms, and the
        external source term instead of the sum of all these terms (i.e. the
        actual RHS term).

    Returns:
      scalar_function: A function that computes `f(q)`.

    Raises:
      ValueError: If the thermodynamics model is not `Water` or `e_t` is not
         found in states.
    """
    if not isinstance(self.thermodynamics.model, water.Water):
      raise ValueError('The thermodynamics model has to be `Water` for the '
                       'total humidity equation. Current model is {}'.format(
                           self.thermodynamics.model))
    if 'e_t' not in states and 'theta_li' not in states:
      raise ValueError('Expected e_t or theta_li in states used for total '
                       'humidity equation, got: {}'.format(states.keys()))

    valid_names = ['q_t', 'q_r']
    if scalar_name not in valid_names:
      raise ValueError('Expected scalar_name in {}, got: {}'.format(
          valid_names, scalar_name))

    h = (self._params.dx, self._params.dy, self._params.dz)
    dt = self._params.dt

    momentum = [states[_KEY_RHO_U], states[_KEY_RHO_V], states[_KEY_RHO_W]]

    zz = additional_states.get(
        'zz', tf.nest.map_structure(tf.zeros_like, states[_KEY_RHO_U]))

    source_ext = self._source[scalar_name] if (  # pylint: disable=g-long-ternary
        scalar_name in self._source.keys() and
        self._source[scalar_name]) else tf.nest.map_structure(
            tf.zeros_like, states[_KEY_RHO])

    # Helper variables required by the Monin-Obukhov similarity theory.
    helper_variables_most = {
        'u': states[_KEY_U],
        'v': states[_KEY_V],
        'w': states[_KEY_W],
    }
    include_subsidence = (
        self._scalars[scalar_name].HasField('humidity') and
        self._scalars[scalar_name].humidity.include_subsidence and
        self._g_dim is not None)
    include_precipitation = (
        self._scalars[scalar_name].HasField('humidity') and
        self._scalars[scalar_name].humidity.include_precipitation)

    def scalar_function(q: FlowFieldVal):
      """Computes the RHS of humidity update equation.

      This return rhs `f(q)` of the humidity update equation
      `d rho_q / dt = f(q)`, where `q` is either total humidity
      `q_t` or liquid precipitation `q_r`.

      Args:
        q: The humidity field to be updated (can be q_t or q_r, depending on
          'scalar_name').

      Returns:
        A `FlowFieldVal` representing the RHS of the humidity equation.
      """

      valid_names = ['q_t', 'q_r']
      if scalar_name not in valid_names:
        raise ValueError('Expected scalar_name in {}, got: {}'.format(
            valid_names, scalar_name))
      if scalar_name == 'q_r' and not include_precipitation:
        raise ValueError(
            'Calculating q_r without setting include_precipitation to '
            ' True is not well-defined.')

      rho = states[_KEY_RHO]
      rho_thermal = states['rho_thermal']

      # Compute momentum term
      corrected_momentum = momentum
      if scalar_name == 'q_r':
        # Subtract term for rain terminal velocity from the vertical momentum.
        rain_water_terminal_velocity = (
            self.precipitation.rain_water_terminal_velocity_kw1978(
                rho_thermal, q))
        rain_water_momentum = tf.nest.map_structure(
            tf.math.multiply, rain_water_terminal_velocity, rho)
        corrected_momentum[self._g_dim] = tf.nest.map_structure(
            tf.math.subtract, momentum[self._g_dim], rain_water_momentum)

      # pylint: disable=g-complex-comprehension
      conv = [
          self._conservative_scalar_convection(replica_id, replicas, q,
                                               corrected_momentum[i],
                                               states[_KEY_P], rho, h[i], dt, i,
                                               scalar_name, zz)
          for i in range(3)
      ]

      # Compute the sum of the various source terms including the external
      # sources, e.g. sponge.
      source = source_ext

      # Compute the temperature.
      q_t = q if scalar_name == 'q_t' else states['q_t']

      if 'theta_li' in states:
        temperature = self.thermodynamics.model.saturation_adjustment(
            'theta_li', states['theta_li'], rho_thermal, q_t, zz=zz)
      elif 'e_t' in states:
        e = self.thermodynamics.model.internal_energy_from_total_energy(
            states['e_t'], states[_KEY_U], states[_KEY_V], states[_KEY_W], zz)
        temperature = self.thermodynamics.model.saturation_adjustment(
            'e_int', e, rho_thermal, q_t)
      elif 'T' in additional_states.keys():
        temperature = additional_states['T']

      # Compute the potential temperature.
      if 'theta' in additional_states:
        theta = additional_states['theta']
      else:
        buf = self.thermodynamics.model.potential_temperatures(
            temperature, q_t, rho_thermal, zz)
        theta = buf['theta']
      helper_variables_most.update({'theta': theta})

      if include_precipitation or include_subsidence:
        # Compute condensate mass fraction.
        q_c = self.thermodynamics.model.saturation_excess(
            temperature, rho_thermal, q_t)

      # Compute vapor to rain water conversion term if needed.
      if include_precipitation:
        q_r = q if scalar_name == 'q_r' else states['q_r']
        q_l, _ = self.thermodynamics.model.equilibrium_phase_partition(
            temperature, rho_thermal, q_t)
        cloud_liquid_to_rain_water_rate = (
            self.precipitation.cloud_liquid_to_rain_conversion_rate_kw1978(
                q_r, q_l))
        # q_v = q_t - q_l - q_i. Not: We assume q_i == 0 here.
        q_v = tf.nest.map_structure(tf.math.subtract, q_t, q_l)
        rain_water_evaporation_rate = (
            self.precipitation.rain_evaporation_rate_kw1978(
                rho_thermal, temperature, q_r, q_v, q_l, q_c))
        # Net vapor to rain water rate is
        #   (vapor to rain water rate) - (evaporation rate).
        net_cloud_liquid_to_rain_water_rate = tf.nest.map_structure(
            tf.math.subtract, cloud_liquid_to_rain_water_rate,
            rain_water_evaporation_rate)
        cloud_liquid_to_water_source = tf.nest.map_structure(
            tf.math.multiply, net_cloud_liquid_to_rain_water_rate, rho)
        # Add term for q_r, subtract for q_t.
        op = tf.math.subtract if scalar_name == 'q_t' else tf.math.add
        source = tf.nest.map_structure(op, source, cloud_liquid_to_water_source)

      # Compute source terms
      if scalar_name == 'q_t' and include_subsidence:
        subsidence_source = eq_utils.source_by_subsidence_velocity(
            self._kernel_op, rho, zz, h[self._g_dim], q_c, self._g_dim)

        # Add external source, e.g. sponge forcing and subsidence.
        source = tf.nest.map_structure(tf.math.add, source, subsidence_source)

      diff = self.diffusion_fn(
          self._kernel_op,
          replica_id,
          replicas,
          q,
          rho,
          additional_states['diffusivity'],
          h,
          scalar_name=scalar_name,
          helper_variables=helper_variables_most)

      if dbg:
        return {
            'conv_x': conv[0],
            'conv_y': conv[1],
            'conv_z': conv[2],
            'diff_x': diff[0],
            'diff_y': diff[1],
            'diff_z': diff[2],
            'source': source,
        }

      equation_terms = zip(conv[0], conv[1], conv[2], diff[0], diff[1], diff[2],
                           source)

      rhs = [
          -(conv_x_i + conv_y_i + conv_z_i) + (diff_x_i + diff_y_i + diff_z_i) +
          src_i for conv_x_i, conv_y_i, conv_z_i, diff_x_i, diff_y_i, diff_z_i,
          src_i in equation_terms
      ]

      if self._ib is not None:
        rhs_name = self._ib.ib_rhs_name('rho_' + scalar_name)
        helper_states = {
            rhs_name: rhs,
            'ib_interior_mask': additional_states['ib_interior_mask'],
        }
        rhs_ib_updated = self._ib.update_forcing(
            self._kernel_op, replica_id, replicas, {
                'rho_' + scalar_name:
                    tf.nest.map_structure(tf.math.multiply, rho, q_t)
            }, helper_states)
        rhs = rhs_ib_updated[rhs_name]

      return rhs

    return scalar_function

  def _scalar_update(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      scalar_name: Text,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      dbg: bool = False,
  ):
    """Provides a function that computes the RHS of a scalar transport equation.

    This function provides a wrapper for the function that computes the rhs
    `f(phi)` of the scalar equation in functional form, i.e.
    `drho_phi / dt = f(phi)`.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      scalar_name: The name of the scalar.
      states: A dictionary that holds field variables that are essential to
        compute the right hand side function of the scalar transport equation.
      additional_states: Helper states that are required by the scalar transport
        equation.
      dbg: A flag of whether to use the debug mode. If `True`, the returned RHS
        function returns the convection terms, the diffusion terms, and the
        external source term instead of the sum of all these terms (i.e. the
        actual RHS term).

    Returns:
      scalar_function: A function that computes the `f(phi)`.
    """
    if scalar_name == 'e_t':
      return self._e_t_update(replica_id, replicas, states, additional_states,
                              dbg)
    elif scalar_name in ['q_t', 'q_r']:
      return self._humidity_update(replica_id, replicas, states,
                                   additional_states, scalar_name, dbg)
    elif scalar_name == 'theta_li':
      return self._theta_li_update(replica_id, replicas, states,
                                   additional_states, dbg)
    else:
      return self._generic_scalar_update(replica_id, replicas, scalar_name,
                                         states, additional_states, dbg)

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
  ) -> FlowFieldMap:
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
    for sc_name in self._params.transport_scalars_names:
      sc_mid = states_mid[sc_name]
      if self._use_sgs:
        diff_t = self._sgs_model.turbulent_diffusivity(
            (sc_mid,), (states[_KEY_U], states[_KEY_V], states[_KEY_W]),
            replicas, additional_states)
        diffusivity = [
            self._params.diffusivity(sc_name) + diff_t_i for diff_t_i in diff_t
        ]
      else:
        diffusivity = [
            self._params.diffusivity(sc_name) * tf.ones_like(sc)
            for sc in sc_mid
        ]
      helper_states = {'diffusivity': diffusivity}
      helper_states.update(additional_states)
      scalar_rhs_fn = self._scalar_update(replica_id, replicas, sc_name,
                                          states_mid, helper_states)

      def time_advance_cn_explicit(rhs, sc_name):
        updated_vars = {}
        if (self._params.solver_mode ==
            thermodynamics_pb2.Thermodynamics.ANELASTIC):
          alpha = tf.nest.map_structure(tf.math.reciprocal, states[_KEY_RHO])
          new_sc = tf.nest.map_structure(
              lambda sc, b, a: sc + self._params.dt * b * a, states_0[sc_name],
              rhs, alpha)
          updated_vars.update({sc_name: exchange_halos(new_sc, sc_name)})
        else:
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
        updated_scalars.update(
            time_advance_cn_explicit(scalar_rhs_fn(sc_mid), sc_name))
      else:
        raise ValueError(
            'Time integration scheme %s is not supported yet for scalars.' %
            time_scheme)

      if self._dbg is not None:
        terms = (
            self._scalar_update(replica_id, replicas, sc_name, states_mid,
                                helper_states, True)(sc_mid))
        diff_t = diff_t if self._use_sgs else None
        updated_scalars.update(
            self._dbg.update_scalar_terms(sc_name, terms, diff_t))

    # Applies the marker-and-cell or Cartesian grid method if requested in the
    # config file.
    if self._ib is not None:
      updated_scalars = self._ib.update_states(self._kernel_op, replica_id,
                                               replicas, updated_scalars,
                                               additional_states, self._bc)

    return updated_scalars  # pytype: disable=bad-return-type

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
