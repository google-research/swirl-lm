"""A library of density update with a constant."""

from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics.thermodynamics import thermodynamics_generic
import tensorflow as tf

TF_DTYPE = thermodynamics_generic.TF_DTYPE

FlowFieldVal = thermodynamics_generic.FlowFieldVal
FlowFieldMap = thermodynamics_generic.FlowFieldMap


class ConstantDensity(thermodynamics_generic.ThermodynamicModel):
  """A library of constant density."""

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    """Initializes the constant density object."""
    super(ConstantDensity, self).__init__(params)

    self.rho = params.rho

  def update_density(
      self,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldVal:
    """Updates the density with the ideal gas law."""
    del additional_states
    return [
        self.rho * tf.ones_like(rho_i, dtype=TF_DTYPE)
        for rho_i in list(states.values())[0]
    ]
