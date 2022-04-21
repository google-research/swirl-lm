"""A library of density update with a constant."""

from swirl_lm.physics.thermodynamics import thermodynamics_generic
import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config

TF_DTYPE = thermodynamics_generic.TF_DTYPE

FlowFieldVar = thermodynamics_generic.FlowFieldVar
FlowFieldMap = thermodynamics_generic.FlowFieldMap


class ConstantDensity(thermodynamics_generic.ThermodynamicModel):
  """A library of constant density."""

  def __init__(
      self,
      params: incompressible_structured_mesh_config
      .IncompressibleNavierStokesParameters,
  ):
    """Initializes the constant density object."""
    super(ConstantDensity, self).__init__(params)

    self.rho = params.rho

  def update_density(
      self,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldVar:
    """Updates the density with the ideal gas law."""
    del additional_states
    return [
        self.rho * tf.ones_like(rho_i, dtype=TF_DTYPE)
        for rho_i in list(states.values())[0]
    ]
