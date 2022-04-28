"""A library for solving scalar transport equations.

   Because of the staggering in time, and density is at the same time step as
   scalars, the average density (old & new) is at the same time step as the
   velocity at the new time step.
"""

import functools
from typing import List, Optional, Sequence, Text

import numpy as np
from swirl_lm.boundary_condition import immersed_boundary_method
from swirl_lm.communication import halo_exchange
from swirl_lm.equations import common
from swirl_lm.equations import utils as eq_utils
from swirl_lm.numerics import convection
from swirl_lm.numerics import diffusion
from swirl_lm.numerics import filters
from swirl_lm.numerics import numerics_pb2
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.physics.thermodynamics import water
from swirl_lm.utility import common_ops
from swirl_lm.utility import components_debug
from swirl_lm.utility import get_kernel_fn
import tensorflow as tf
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_numerics
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import physical_variable_keys_manager
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import sgs_model

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
# The gravitational acceleration.
_GRAVITY = 9.81

_FlowFieldVar = eq_utils.FlowFieldVar
_FlowFieldMap = eq_utils.FlowFieldMap

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
      params: incompressible_structured_mesh_config
      .IncompressibleNavierStokesParameters,
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
      f: List[tf.Tensor],
      name: Text,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
  ) -> List[tf.Tensor]:
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
      phi: Sequence[tf.Tensor],
      rho_u: Sequence[tf.Tensor],
      p: Sequence[tf.Tensor],
      rho: Sequence[tf.Tensor],
      h: float,
      dt: float,
      dim: int,
      sc_name: Text,
      zz: Optional[Sequence[tf.Tensor]] = None,
      apply_correction: bool = False,
  ) -> List[tf.Tensor]:
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
      gravity = [drho_i * self._g_vec[dim] * _GRAVITY for drho_i in drho]

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
        scheme=self._scalars[sc_name].scheme,
        src=gravity,
        apply_correction=apply_correction)

  def _generic_scalar_update(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      scalar_name: Text,
      states: _FlowFieldMap,
      additional_states: _FlowFieldMap,
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

    def scalar_function(phi: Sequence[tf.Tensor]):
      """Computes the functional RHS for the three momentum equations.

      Args:
        phi: The scalar field.

      Returns:
        A `List[tf.Tensor]` representing the RHS of the scalar transport
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
            self._kernel_op, replica_id, replicas,
            {rho_sc_name: common_ops.multiply(states[_KEY_RHO], phi)},
            helper_states)
        rhs = rhs_ib_updated[rhs_name]

      return rhs

    return scalar_function

  def _e_t_update(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: _FlowFieldMap,
      additional_states: _FlowFieldMap,
      dbg: bool = False,
  ):
    """Provides a function that computes the RHS of the total energy equation.

    This function provides a wrapper for the function that computes the rhs
    `f(e_t)` in functional form, i.e. `d rho_e_t / dt = f(e_t)`.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      states: A dictionary that holds field variables that are essential to
        compute the right hand side function of the scalar transport equation.
        Must include the following fields: 'u', 'v', 'w', 'rho_u', 'rho_v',
        'rho_w', 'p', 'rho', 'e_t', 'q_t'.
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
    zz = additional_states['zz'] if 'zz' in additional_states.keys() else [
        tf.zeros_like(rho_u_i, dtype=rho_u_i.dtype)
        for rho_u_i in states[_KEY_RHO_U]
    ]
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

    def compute_rho_u_tau(
        tau_0j: Sequence[tf.Tensor],
        tau_1j: Sequence[tf.Tensor],
        tau_2j: Sequence[tf.Tensor],
    ) -> List[tf.Tensor]:
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

    def radiation(
        q_h: tf.Tensor,
        q_l: tf.Tensor,
        rho: tf.Tensor,
        z: tf.Tensor,
    ) -> tf.Tensor:
      """Computes the radiation term based on given parameters.

      Args:
        q_h: The integral of the liquid water specific mass from `z` to the
          maximum height of the simulation.
        q_l: The integral of the liquid water specific mass from 0 to `z`.
        rho: The density of air at `z`
        z: The current height.

      Returns:
        The radiation source term.
      """
      return (_F0 * tf.math.exp(-_KAPPA * q_h) +
              _F1 * tf.math.exp(-_KAPPA * q_l) +
              rho * self.thermodynamics.model.cp_d * _D * _ALPHA_Z *
              (0.25 * tf.math.pow(tf.maximum(z - _ZI, 0.0), 4.0 / 3.0) +
               _ZI * tf.math.pow(tf.maximum(z - _ZI, 0.0), 1.0 / 3.0)))

    def source_by_radiation(
        q_l: Sequence[tf.Tensor],
        g_dim: int,
    ) -> List[tf.Tensor]:
      """Computes the energy source term due to radiation.

      Reference:
      Stevens, Bjorn, Chin-Hoh Moeng, Andrew S. Ackerman, Christopher S.
      Bretherton, Andreas Chlond, Stephan de Roode, James Edwards, et al. 2005.
      “Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine
      Stratocumulus.” Monthly Weather Review 133 (6): 1443–62.

      Args:
        q_l: The liquid humidity.
        g_dim: The dimension of the gravity.

      Returns:
        The source term in the total energy equation due to radiation.
      """
      rho_q_l = common_ops.multiply(states[_KEY_RHO], q_l)
      q_below, q_above = common_ops.integration_in_dim(replica_id, replicas,
                                                       rho_q_l, h[g_dim], g_dim)

      f_r = [
          radiation(q_h, q_l, rho_i, z)
          for q_h, q_l, rho_i, z in zip(q_above, q_below, states[_KEY_RHO], zz)
      ]

      horizontal_dims = [0, 1, 2]
      horizontal_dims.remove(g_dim)

      mean = common_ops.global_mean(f_r, replicas, [0] * 3, horizontal_dims)
      if 2 in horizontal_dims:
        mean = mean * self._params.nz
      return mean

    def scalar_function(e_t: Sequence[tf.Tensor]):
      """Computes the functional RHS for the total energy equation.

      Args:
        e_t: The total energy field.

      Returns:
        A `List[tf.Tensor]` representing the RHS of the total energy equation.
      """
      # Compute the temperature.
      if 'T' in additional_states.keys():
        temperature = additional_states['T']
      else:
        e = self.thermodynamics.model.internal_energy_from_total_energy(
            e_t, states[_KEY_U], states[_KEY_V], states[_KEY_W], zz)
        temperature = self.thermodynamics.model.saturation_adjustment(
            e, states[_KEY_RHO], states['q_t'])

      # Compute the potential temperature.
      if 'theta' in additional_states:
        theta = additional_states['theta']
      else:
        buf = self.thermodynamics.model.potential_temperatures(
            temperature, states['q_t'], states[_KEY_RHO], zz)
        theta = buf['theta_v']
      helper_variables_most.update({'theta': theta})

      # Compute the total enthalpy.
      h_t = self.thermodynamics.model.total_enthalpy(e_t, states[_KEY_RHO],
                                                     states['q_t'], temperature)

      # Compute the source term due to radiation.
      if (self._scalars['e_t'].HasField('total_energy') and
          self._scalars['e_t'].total_energy.include_radiation and
          self._g_dim is not None):
        q_l, _ = self.thermodynamics.model.equilibrium_phase_partition(
            temperature, states[_KEY_RHO], states['q_t'])
        f_r = source_by_radiation(q_l, self._g_dim)
        div_terms[self._g_dim] = [
            div_term_i - rho_i * f_r_i for div_term_i, rho_i, f_r_i in zip(
                div_terms[self._g_dim], states[_KEY_RHO], f_r)
        ]

      # Compute the source term due to dilatation, wind shear, and radiation.
      source = incompressible_structured_mesh_numerics.divergence(
          self._kernel_op, div_terms, h)

      # Compute the source term due to the subsidence velocity.
      if (self._scalars['e_t'].HasField('total_energy') and
          self._scalars['e_t'].total_energy.include_subsidence):
        src_subsidence = eq_utils.source_by_subsidence_velocity(
            self._kernel_op, states[_KEY_RHO], zz, h[self._g_dim], h_t,
            self._g_dim)
        source = tf.nest.map_structure(tf.math.add, source, src_subsidence)

      # Add external source, e.g. sponge forcing.
      source = tf.nest.map_structure(tf.math.add, source, source_ext)

      # Compute the convection and diffusion terms.
      conv = [
          self._conservative_scalar_convection(replica_id, replicas, h_t,  # pylint: disable=g-complex-comprehension
                                               momentum[i], states[_KEY_P],
                                               states[_KEY_RHO], h[i], dt, i,
                                               'e_t', zz) for i in range(3)
      ]

      diff = self.diffusion_fn(
          self._kernel_op,
          replica_id,
          replicas,
          h_t,
          states[_KEY_RHO],
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
            {'rho_e_t': common_ops.multiply(states[_KEY_RHO], e_t)},
            helper_states)
        rhs = rhs_ib_updated[rhs_name]

      return rhs

    return scalar_function

  def _q_t_update(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: _FlowFieldMap,
      additional_states: _FlowFieldMap,
      dbg: bool = False,
  ):
    """Provides a function that computes the RHS of the total humidity equation.

    This function provides a wrapper for the function that computes the rhs
    `f(q_t)` in functional form, i.e. `d rho_q_t / dt = f(q_t)`.

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
      dbg: A flag of whether to use the debug mode. If `True`, the returned RHS
        function returns the convection terms, the diffusion terms, and the
        external source term instead of the sum of all these terms (i.e. the
        actual RHS term).

    Returns:
      scalar_function: A function that computes the `f(q_t)`.

    Raises:
      ValueError: If the thermodynamics model is not `Water` or `e_t` is not
         found in states.
    """
    if not isinstance(self.thermodynamics.model, water.Water):
      raise ValueError('The thermodynamics model has to be `Water` for the '
                       'total humidity equation. Current model is {}'.format(
                           self.thermodynamics.model))
    if 'e_t' not in states.keys():
      raise ValueError('Expected e_t in states used for total humidty '
                       'equation, got: {}'.format(states.keys()))

    h = (self._params.dx, self._params.dy, self._params.dz)
    dt = self._params.dt
    momentum = [states[_KEY_RHO_U], states[_KEY_RHO_V], states[_KEY_RHO_W]]

    zz = additional_states['zz'] if 'zz' in additional_states.keys() else [
        tf.zeros_like(rho_u_i, dtype=rho_u_i.dtype)
        for rho_u_i in states[_KEY_RHO_U]
    ]

    source_ext = self._source['q_t'] if self._source['q_t'] else [
        tf.zeros_like(rho_i) for rho_i in states[_KEY_RHO]
    ]

    # Helper variables required by the Monin-Obukhov similarity theory.
    helper_variables_most = {
        'u': states[_KEY_U],
        'v': states[_KEY_V],
        'w': states[_KEY_W],
    }
    include_subsidence = (
        self._scalars['q_t'].HasField('total_humidity') and
        self._scalars['q_t'].total_humidity.include_subsidence)

    def scalar_function(q_t: Sequence[tf.Tensor]):
      """Computes the functional RHS for the three momentum equations.

      Args:
        q_t: The total humidity field.

      Returns:
        A `List[tf.Tensor]` representing the RHS of the total humidity equation.
      """
      conv = [
          self._conservative_scalar_convection(replica_id, replicas, q_t,  # pylint: disable=g-complex-comprehension
                                               momentum[i], states[_KEY_P],
                                               states[_KEY_RHO], h[i], dt, i,
                                               'q_t', zz) for i in range(3)
      ]

      # Compute the temperature if needed.
      if include_subsidence or 'theta' not in additional_states:
        if 'T' in additional_states.keys():
          temperature = additional_states['T']
        else:
          e = self.thermodynamics.model.internal_energy_from_total_energy(
              states['e_t'], states[_KEY_U], states[_KEY_V], states[_KEY_W], zz)
          temperature = self.thermodynamics.model.saturation_adjustment(
              e, states[_KEY_RHO], q_t)

      if include_subsidence:
        # Compute condensate mass fraction.
        q_c = self.thermodynamics.model.saturation_excess(
            temperature, states[_KEY_RHO], q_t)

        # Source term from falling water (subsidence).
        source = eq_utils.source_by_subsidence_velocity(self._kernel_op,
                                                        states[_KEY_RHO], zz,
                                                        h[self._g_dim], q_c,
                                                        self._g_dim)
        # Add external source, e.g. sponge forcing and subsidence.
        source = tf.nest.map_structure(tf.math.add, source, source_ext)
      else:
        source = source_ext

      # Compute the potential temperature.
      if 'theta' in additional_states:
        theta = additional_states['theta']
      else:
        buf = self.thermodynamics.model.potential_temperatures(
            temperature, q_t, states[_KEY_RHO], zz)
        theta = buf['theta_v']
      helper_variables_most.update({'theta': theta})

      diff = self.diffusion_fn(
          self._kernel_op,
          replica_id,
          replicas,
          q_t,
          states[_KEY_RHO],
          additional_states['diffusivity'],
          h,
          scalar_name='q_t',
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
        rhs_name = self._ib.ib_rhs_name('rho_q_t')
        helper_states = {
            rhs_name: rhs,
            'ib_interior_mask': additional_states['ib_interior_mask'],
        }
        rhs_ib_updated = self._ib.update_forcing(
            self._kernel_op, replica_id, replicas,
            {'rho_q_t': common_ops.multiply(states[_KEY_RHO], q_t)},
            helper_states)
        rhs = rhs_ib_updated[rhs_name]

      return rhs

    return scalar_function

  def _scalar_update(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      scalar_name: Text,
      states: _FlowFieldMap,
      additional_states: _FlowFieldMap,
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
    elif scalar_name == 'q_t':
      return self._q_t_update(replica_id, replicas, states, additional_states,
                              dbg)
    else:
      return self._generic_scalar_update(replica_id, replicas, scalar_name,
                                         states, additional_states, dbg)

  def prestep(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      additional_states: _FlowFieldMap,
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
      states: _FlowFieldMap,
      states_0: _FlowFieldMap,
      additional_states: _FlowFieldMap,
  ) -> _FlowFieldMap:
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

    updated_scalars = {}
    for sc_name in self._params.transport_scalars_names:
      sc_mid = common_ops.average(states[sc_name], states_0[sc_name])
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
      # Time advancement for rho * scalar.
      time_scheme = self._params.scalar_time_integration_scheme(sc_name)
      if (time_scheme ==
          numerics_pb2.TimeIntegrationScheme.TIME_SCHEME_CN_EXPLICIT_ITERATION):
        new_sc = common_ops.linear_combination(
            states_0['rho_{}'.format(sc_name)], scalar_rhs_fn(sc_mid), 1.0,
            self._params.dt)
      else:
        raise ValueError(
            'Time integration scheme %s is not supported yet for scalars.' %
            time_scheme)

      updated_scalars.update({'rho_{}'.format(sc_name): new_sc})

      # Updates scalar, to be consistent with rho * scalar.
      updated_scalars.update({
          sc_name:
              exchange_halos(
                  common_ops.divide(updated_scalars['rho_{}'.format(sc_name)],
                                    states[_KEY_RHO]), sc_name),
      })

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
      states: _FlowFieldMap,
      states_0: _FlowFieldMap,
      additional_states: _FlowFieldMap,
  ) -> _FlowFieldMap:
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
      sc_buf = common_ops.divide(states['rho_{}'.format(sc_name)],
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
