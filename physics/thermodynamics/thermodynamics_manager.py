"""The manager of thermodyanmics models."""

from typing import List, Optional, Tuple

import numpy as np
from swirl_lm.communication import halo_exchange
from swirl_lm.numerics import filters
from swirl_lm.physics.thermodynamics import constant_density
from swirl_lm.physics.thermodynamics import ideal_gas
from swirl_lm.physics.thermodynamics import linear_mixing
from swirl_lm.physics.thermodynamics import thermodynamics_generic
from swirl_lm.physics.thermodynamics import thermodynamics_pb2
from swirl_lm.physics.thermodynamics import thermodynamics_utils
from swirl_lm.physics.thermodynamics import water
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
import tensorflow as tf
from google3.research.simulation.tensorflow.fluid.framework.tf1 import model_function
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config

DensityUpdateOption = incompressible_structured_mesh_config.DensityUpdateOption
FlowFieldVar = thermodynamics_utils.FlowFieldVar


class ThermodynamicsManager(object):
  """A manager class for models of thermodynamics."""

  def __init__(
      self,
      params: incompressible_structured_mesh_config
      .IncompressibleNavierStokesParameters,
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

  def rho_ref(self, zz: Optional[FlowFieldVar] = None) -> FlowFieldVar:
    """Generates the reference density."""
    return self.model.rho_ref(zz)

  def p_ref(self, zz: Optional[FlowFieldVar] = None) -> FlowFieldVar:
    """Generates the reference pressure."""
    return self.model.p_ref(zz)

  def update_density(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: model_function.StatesMap,
      additional_states: model_function.StatesMap,
      states_0: Optional[model_function.StatesMap] = None,
  ) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    """Updates the density based on field values provided.

    Args:
      kernel_op: An kernel operation library for finite difference operations.
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
        # TODO(b/210157369): update the BC for drho to be consistent with that
        # for scalars.
        drho = halo_exchange.inplace_halo_exchange(
            filters.filter_op(
                kernel_op, common_ops.subtract(rho, states_0['rho']), order=2),
            self._halo_dims,
            replica_id,
            replicas,
            self._replica_dims,
            self._params.periodic_dims,
            [[(halo_exchange.BCType.NEUMANN, 0.0),] * 2] * 3,
            width=self._params.halo_width)

        rho = tf.nest.map_structure(tf.math.add, states_0['rho'], drho)
      else:
        drho = [tf.zeros_like(rho_i) for rho_i in states['rho']]

      return rho, drho

    elif self.solver_mode == thermodynamics_pb2.Thermodynamics.ANELASTIC:
      # In this mode, the reference density is used to compute conservative
      # variables. The thermodynamic density is only coupled with the buoyancy
      # following the Boussinesq approximation.
      if 'zz' not in additional_states:
        zz = [tf.zeros_like(rho_i) for rho_i in states['rho']]
      else:
        zz = additional_states['zz']
      return self.model.rho_ref(zz), [
          tf.zeros_like(rho_i) for rho_i in states['rho']
      ]
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


def thermodynamics_factory(params: incompressible_structured_mesh_config
                           .IncompressibleNavierStokesParameters):
  """Creates an object of the `ThermodynamicsManager`.

  The thermodynamic library will be created based on parameters specified in the
  simulation configuration file. In the absence of `thermodynamcis` in the
  configuration file, we assume that the simulation follows the low-Mach number
  approach.

  Args:
    params: The context object that holds parameters in a simulation.

  Returns:
    An object of the thermodynamcis manager, which provides interfaces for the
    evaluations of thermodynamcis quantities.
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
