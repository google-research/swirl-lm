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

"""A library that computes the diffusion term in the Navier-Stokes solver.

The diffusion term for scalars has 3 components only, in each component the 2
first order derivatives are performed along the same direction, i.e. d/dx(d/dx),
d/dy(d/dy), and d/dz(d/dz).

The diffusion term for velocity considers not only derivatives along same
directions, but also in perpendicular directions, e.g. d/dy(d/dx). There are 3
methods to compute these terms, which are:

DIFFUSION_SCHEME_CENTRAL_5: both the inner and outer first order derivatives are
computed with 3-node stencil central difference. As a result, derivatives
performed in the same direction has a stencil of width 5.

DIFFUSION_SCHEME_CENTRAL_3: the inner derivatives are computed with neighboring
nodes, so that their values fall on the faces. The outer derivatives are
performed to the face flux so that the diffusion terms fall back on nodes.
Interpolations across faces in different directions are required in this
approach.

DIFFUSION_SCHEME_STENCIL_3: the inner derivatives are computed with the 3-node
stencil central difference, except when the outer derivative is in the same
direction as the inner one, in which case both derivatives are computed from
neighboring nodes/faces. In this approach the width of the stencil in each
direction is 3.
"""

from typing import Callable, Dict, List, Optional, Sequence, Text, Tuple

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.boundary_condition import monin_obukhov_similarity_theory
from swirl_lm.equations import common
from swirl_lm.equations import utils as eq_utils
from swirl_lm.numerics import numerics_pb2
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap


def diffusion_scalar(
    params: parameters_lib.SwirlLMParameters,
) -> Callable[..., List[FlowFieldVal]]:
  """Generates a function that computes the scalar diffusion term.

  Args:
    params: A object of the simulation parameter context. `boundary_models.most`
      and `nu` are used here.

  Returns:
    A function that computes the diffusion terms in a scalar transport equation.
  """

  if (params.boundary_models is not None and
      params.boundary_models.HasField('most')):
    most = (
        monin_obukhov_similarity_theory.monin_obukhov_similarity_theory_factory(
            params))
  else:
    most = None

  def diffusion_fn(
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      phi: FlowFieldVal,
      rho: FlowFieldVal,
      diffusivity: FlowFieldVal,
      grid_spacing: Tuple[float, float, float],
      scalar_name: Optional[Text] = None,
      helper_variables: Optional[Dict[Text, FlowFieldVal]] = None,
  ) -> List[FlowFieldVal]:
    """Computes the diffusion term for the conservative scalar.

    Args:
      kernel_op: An object holding a library of kernel operations.
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      phi: The scalar for which the diffusion term is computed.
      rho: The density of the fluid.
      diffusivity: The kinematic diffusivity of the scalar.
      grid_spacing: A tuple that has the grid spacing in the x, y, and z,
        directions, respectively.
      scalar_name: The name of the scalar. This is useful for determining if
        special treatments needs to be applied for specific scalars, e.g.
        modelled heat fluxes for temperature and energy equations.
      helper_variables: A dictionarry that stores variables that provides
        additional information for computing the diffusion term, e.g. the
        velocity and potential temperature for Monin-Obukhov similarity theory.

    Returns:
      A list that contains the 3 diffusion components of the scalar.
    """
    sum_backward_fn = (
        lambda f: kernel_op.apply_kernel_op_x(f, 'ksx'),
        lambda f: kernel_op.apply_kernel_op_y(f, 'ksy'),
        lambda f: kernel_op.apply_kernel_op_z(f, 'ksz', 'kszsh'),
    )
    flux_backward_fn = (
        lambda f: kernel_op.apply_kernel_op_x(f, 'kdx'),
        lambda f: kernel_op.apply_kernel_op_y(f, 'kdy'),
        lambda f: kernel_op.apply_kernel_op_z(f, 'kdz', 'kdzsh'),
    )
    grad_forward_fn = (
        lambda f: kernel_op.apply_kernel_op_x(f, 'kdx+'),
        lambda f: kernel_op.apply_kernel_op_y(f, 'kdy+'),
        lambda f: kernel_op.apply_kernel_op_z(f, 'kdz+', 'kdz+sh'),
    )

    rho_d = [rho_i * d_i for rho_i, d_i in zip(rho, diffusivity)]

    rho_d_dim = [[0.5 * rho_d_sum
                  for rho_d_sum in sum_backward_fn[i](rho_d)]
                 for i in range(3)]

    f_diff = [
        [  # pylint: disable=g-complex-comprehension
            rho_d_i * d_phi / grid_spacing[i]
            for rho_d_i, d_phi in zip(rho_d_dim[i], flux_backward_fn[i](phi))
        ]
        for i in range(3)
    ]

    # Add the closure from Monin-Obukhov similarity theory if requested.
    if most is not None and most.is_active_scalar(scalar_name):
      required_variables = ('u', 'v', 'w', 'theta')
      for varname in required_variables:
        if varname not in helper_variables:
          raise ValueError(f'{varname} is missing for the MOS model.')

      scalar_flux_helper_variables = {'rho': rho, 'phi': phi}
      scalar_flux_helper_variables.update(helper_variables)
      q_3 = most.surface_flux_update_fn(scalar_flux_helper_variables,
                                        scalar_name)

      # The sign of the heat flux needs to be reversed to be consistent with
      # the diffusion scheme. In the MOS formulation, the heat flux is positive
      # if heat is flowing into the control volume, which indicates that the
      # control volume has a lower temperature than its surounding environment.
      # This corresponds to a negative temperature gradient in the present
      # diffusion scheme. Therefore, the sign of this flux needs to be reversed
      # for consistency.
      q_3 = tf.nest.map_structure(lambda q: -q, q_3)

      if most.vertical_dim == 2:
        q_3 = [q_3]

      # Replace the diffusion flux at the ground surface with the MOS closure.
      core_index = 0
      plane_index = params.halo_width
      f_diff[most.vertical_dim] = common_ops.tensor_scatter_1d_update_global(
          replica_id, replicas, f_diff[most.vertical_dim], most.vertical_dim,
          core_index, plane_index, q_3)

    # Assign the diffusive flux specified in the simulation configuration. This
    # prescribed flux will override values computed from other models.
    if scalar_name in params.scalar_lib:
      for flux_info in params.scalar_lib[scalar_name].diffusive_flux:
        core_index = (0 if flux_info.face == 0 else
                      replicas.shape[flux_info.dim] - 1)
        plane_index = (
            params.halo_width if flux_info.face == 0 else
            (params.nx, params.ny, params.nz)[flux_info.dim] -
            params.halo_width)
        if flux_info.WhichOneof('flux') == 'value':
          flux = flux_info.value
        else:
          flux = helper_variables[flux_info.varname]
          assert isinstance(flux, tf.Tensor), (
              f'The diffusive flux {flux_info.varname} for {scalar_name} in'
              f' {flux_info.dim} has to be a tf.Tensor, but {type(flux)} is'
              ' provided.'
          )
          # The 2D flux tensor needs prepared as a 3D tensor with its size being
          # 1 along the flux dimension. If it's provided as a 2D tensor, we need
          # to add a dummy dimension along the flux dimension.
          axis = int((flux_info.dim + 1) % 3)
          if len(flux.shape) == 3:
            assert flux.shape[axis] == 1, (
                f'The diffusive flux of {scalar_name} in dim {flux_info.dim} is'
                f' specified by a 3D tensor {flux_info.varname}, but its size'
                f' in dim {flux_info.dim} is {flux.shape[axis]} instead of 1.'
            )
          else:
            flux = tf.expand_dims(flux, axis)
          # If 3D tensors are represented as List[tf.Tensor], we need to convert
          # the flux variable that is a 2D tf.Tensor to a List[tf.Tensor].
          # Specifically, if the dimension of the flux is 0, the flux tensor
          # has to be reshaped as nz x [(1, ny)]; if the dimension is 1, the
          # shape is nz x [(nx, 1)]; and if the dimension is 2, the shape is
          # 1 x [(nx, ny)].
          if not isinstance(f_diff, tf.Tensor):
            flux = tf.unstack(flux)
        f_diff[flux_info.dim] = common_ops.tensor_scatter_1d_update_global(
            replica_id, replicas, f_diff[flux_info.dim], flux_info.dim,
            core_index, plane_index, flux)

    return [[
        d_f_diff / grid_spacing[i] for d_f_diff in grad_forward_fn[i](f_diff[i])
    ] for i in range(3)]

  return diffusion_fn


def _diffusion_momentum_stencil_3(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    mu: FlowFieldVal,
    grid_spacing: Tuple[float, float, float],
    velocity: FlowFieldMap,
) -> Dict[Text, Sequence[FlowFieldVal]]:
  """Computes diffusion terms of momentum equations with 3-node stencil.

  Args:
    kernel_op: An object holding a library of kernel operations.
    mu: The dynamic viscosity.
    grid_spacing: A tuple that holds (dx, dy, dz).
    velocity: A dictionary that has flow vield variables u, v, and w.

  Returns:
    A dictionary that holds the diffusion terms in all momentum equations. The
    dictionary is indexed by the name of the velocity components, i.e. 'u', 'v',
    and 'w'. For each velocity component, the 3 diffusion terms are stored in a
    list of 3 elements, with the elements being the diffusion component in the
    x, y, and z directions, respectively.
  """
  # Functions that interpolates viscosity onto faces.
  sum_backward_fn = (
      lambda f: kernel_op.apply_kernel_op_x(f, 'ksx'),
      lambda f: kernel_op.apply_kernel_op_y(f, 'ksy'),
      lambda f: kernel_op.apply_kernel_op_z(f, 'ksz', 'kszsh'),
  )
  # Functions that computes the diffusion flux on faces.
  flux_backward_fn = (
      lambda f: kernel_op.apply_kernel_op_x(f, 'kdx'),
      lambda f: kernel_op.apply_kernel_op_y(f, 'kdy'),
      lambda f: kernel_op.apply_kernel_op_z(f, 'kdz', 'kdzsh'),
  )
  # Functions that computes the diffusion term from face fluxes.
  grad_forward_fn = (
      lambda f: kernel_op.apply_kernel_op_x(f, 'kdx+'),
      lambda f: kernel_op.apply_kernel_op_y(f, 'kdy+'),
      lambda f: kernel_op.apply_kernel_op_z(f, 'kdz+', 'kdz+sh'),
  )
  # Functions that computes the second order central gradients.
  grad_central_fn = (
      lambda f: kernel_op.apply_kernel_op_x(f, 'kDx'),
      lambda f: kernel_op.apply_kernel_op_y(f, 'kDy'),
      lambda f: kernel_op.apply_kernel_op_z(f, 'kDz', 'kDzsh'),
  )

  # Prepares the scaled/unscaled viscosity on faces.
  mu_dim = [
      [0.5 * mu_sum for mu_sum in sum_backward_fn[i](mu)] for i in range(3)
  ]
  four_thirds_mu = [[4.0 / 3.0 * mu_i for mu_i in mu_dim[i]] for i in range(3)]
  two_thirds_mu = [2.0 / 3.0 * mu_i for mu_i in mu]

  # Computes velocity gradients on faces along all directions. These gradients
  # used to compute second order derivatives of velocity along the gradient
  # direction.
  flux_u = {  # pylint: disable=g-complex-comprehension
      k: [[
          flux / grid_spacing[dim]
          for flux in flux_backward_fn[dim](velocity[k])
      ]
          for dim in range(3)] for k in common.KEYS_VELOCITY
  }
  # Computes velocity gradients with central difference. These gradients are
  # used to compute the cross terms in second order derivatives of velocity.
  grad_central_u = {  # pylint: disable=g-complex-comprehension
      k: [[
          grad / (2.0 * grid_spacing[dim])
          for grad in grad_central_fn[dim](velocity[k])
      ]
          for dim in range(3)] for k in common.KEYS_VELOCITY
  }

  # Functions that are used to compute the diffusion terms.
  def tangential_diffusion_fn(dim):
    """Computes the diffusion term along direction of a velocity component."""
    # Get keys for velocity perpendicular to `dim`.
    dims_n = [0, 1, 2]
    dims_n.remove(dim)

    four_thirds_mu_flux_u = tf.nest.map_structure(
        tf.math.multiply, four_thirds_mu[dim],
        flux_u[common.KEYS_VELOCITY[dim]][dim])
    output = tf.nest.map_structure(lambda grad: grad / grid_spacing[dim],
                                   grad_forward_fn[dim](four_thirds_mu_flux_u))
    for i in dims_n:
      two_thirds_mu_grad_central_u = tf.nest.map_structure(
          tf.math.multiply, two_thirds_mu,
          grad_central_u[common.KEYS_VELOCITY[i]][i])
      buf = tf.nest.map_structure(
          lambda grad: grad / (2 * grid_spacing[dim]),
          grad_central_fn[dim](two_thirds_mu_grad_central_u))
      output = tf.nest.map_structure(tf.math.subtract, output, buf)

    return output

  def normal_diffusion_fn(dim, dim_n):
    """Computes the diffusion term normal to a velocity component."""
    mu_flux_u = tf.nest.map_structure(tf.math.multiply, mu_dim[dim_n],
                                      flux_u[common.KEYS_VELOCITY[dim]][dim_n])
    mu_grad_central_u = tf.nest.map_structure(
        tf.math.multiply, mu, grad_central_u[common.KEYS_VELOCITY[dim_n]][dim])
    dx = grid_spacing[dim_n]
    return tf.nest.map_structure(lambda g0, g1: g0 / dx + g1 / (2 * dx),
                                 grad_forward_fn[dim_n](mu_flux_u),
                                 grad_central_fn[dim_n](mu_grad_central_u))

  def diffusion_fn(vel):
    """Computes the diffusion terms of velocity component `vel`."""
    vel_id = common.KEYS_VELOCITY.index(vel)

    output = []
    for i in range(3):
      if i == vel_id:
        output.append(tangential_diffusion_fn(vel_id))
      else:
        output.append(normal_diffusion_fn(vel_id, i))

    return output

  return {k: diffusion_fn(k) for k in common.KEYS_VELOCITY}


def diffusion_momentum(
    params: parameters_lib.SwirlLMParameters,
) -> Callable[..., Dict[Text, Sequence[FlowFieldVal]]]:
  """Generates a function that computes the scalar diffusion term.

  Args:
    params: A object of the simulation parameter context. `boundary_models.most`
      and `nu` are used here.

  Returns:
    A function that computes the diffusion terms in a scalar transport equation.
  """

  shear_flux_fn_stencil_3 = eq_utils.shear_flux(params)

  def diffusion_fn(
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      scheme: numerics_pb2.DiffusionScheme,
      mu: FlowFieldVal,
      grid_spacing: Tuple[float, float, float],
      velocity: FlowFieldMap,
      tau_bc_update_fn: Optional[Dict[Text, Callable[[FlowFieldVal],
                                                     FlowFieldVal]]] = None,
      helper_variables: Optional[FlowFieldMap] = None,
  ) -> Dict[Text, Sequence[FlowFieldVal]]:
    """Computes the diffusion term in momentum equations of u, v, and w.

    Args:
      kernel_op: An object holding a library of kernel operations.
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      scheme: The numerical scheme used to compute the diffusion term.
      mu: The dynamic viscosity.
      grid_spacing: A tuple that holds (dx, dy, dz).
      velocity: A dictionary that has flow vield variables u, v, and w.
      tau_bc_update_fn: A dictionary of halo_exchange functions for the shear
        stress tensor.
      helper_variables: A dictionarry that stores variables that provides
        additional information for computing the diffusion term, e.g. the
        potential temperature for the Monin-Obukhov similarity theory.

    Returns:
      A dictionary that holds the diffusion terms in all momentum equations. The
      dictionary is indexed by the name of the velocity components, i.e. 'u',
      'v', and 'w'. For each velocity component, the 3 diffusion terms are
      stored in a list of 3 elements, with the elements being the diffusion
      component in the x, y, and z directions, respectively.
    """
    grad_central = [
        lambda f: kernel_op.apply_kernel_op_x(f, 'kDx'),
        lambda f: kernel_op.apply_kernel_op_y(f, 'kDy'),
        lambda f: kernel_op.apply_kernel_op_z(f, 'kDz', 'kDzsh'),
    ]
    grad_forward = [
        lambda f: kernel_op.apply_kernel_op_x(f, 'kdx+'),
        lambda f: kernel_op.apply_kernel_op_y(f, 'kdy+'),
        lambda f: kernel_op.apply_kernel_op_z(f, 'kdz+', 'kdz+sh'),
    ]

    shear_key = {
        'u': ('xx', 'xy', 'xz'),
        'v': ('yx', 'yy', 'yz'),
        'w': ('zx', 'zy', 'zz')
    }

    if scheme == numerics_pb2.DiffusionScheme.DIFFUSION_SCHEME_CENTRAL_5:
      tau = eq_utils.shear_stress(kernel_op, mu, grid_spacing[0],
                                  grid_spacing[1], grid_spacing[2],
                                  velocity['u'], velocity['v'], velocity['w'],
                                  tau_bc_update_fn)
      diff_op = grad_central
      grad_width = 2.0
    elif scheme == numerics_pb2.DiffusionScheme.DIFFUSION_SCHEME_CENTRAL_3:
      tau = shear_flux_fn_stencil_3(kernel_op, replica_id, replicas, mu,
                                    grid_spacing[0], grid_spacing[1],
                                    grid_spacing[2], velocity['u'],
                                    velocity['v'], velocity['w'],
                                    helper_variables)
      diff_op = grad_forward
      grad_width = 1.0
    elif scheme == numerics_pb2.DiffusionScheme.DIFFUSION_SCHEME_STENCIL_3:
      return _diffusion_momentum_stencil_3(kernel_op, mu, grid_spacing,
                                           velocity)
    else:
      raise ValueError('{} is not implemented. Available options are: '
                       '"DIFFUSION_SCHEME_CENTRAL_3", '
                       '"DIFFUSION_SCHEME_CENTRAL_5", '
                       '"DIFFUSION_SCHEME_STENCIL_3".'.format(scheme))

    def diffusion_fn(key, dim):
      """Computes the diffusion term for `key` in direction `dim`."""
      shear = tau[shear_key[key][dim]]
      h = grid_spacing[dim]
      return [d_flux / (grad_width * h) for d_flux in diff_op[dim](shear)]

    diff = {
        key: [diffusion_fn(key, i) for i in range(3)
             ] for key in common.KEYS_VELOCITY
    }

    return diff

  return diffusion_fn
