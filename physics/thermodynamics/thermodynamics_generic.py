"""An abstract class for thermodynamic models."""

import abc
from typing import Optional

import six
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics.thermodynamics import thermodynamics_utils
import tensorflow as tf

TF_DTYPE = thermodynamics_utils.TF_DTYPE

FlowFieldVal = thermodynamics_utils.FlowFieldVal
FlowFieldMap = thermodynamics_utils.FlowFieldMap


@six.add_metaclass(abc.ABCMeta)
class ThermodynamicModel(object):
  """A generic class for thermodynamic models."""

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    """Initializes the thermodynamics library."""
    self._params = params
    self._rho = params.rho

  def rho_ref(
      self,
      zz: Optional[FlowFieldVal] = None,
  ) -> FlowFieldVal:
    """Generates the reference density.

    The default reference density is a constant whose value is specified in the
    input config.

    Args:
      zz: The coordinates along the direction of height/gravitation. Useful in
        geophysical flows.

    Returns:
      The reference density in the simulation.
    """
    del zz
    return [self._rho * tf.constant(1.0, dtype=TF_DTYPE),] * self._params.nz

  def p_ref(
      self,
      zz: FlowFieldVal,
  ) -> FlowFieldVal:
    """Generates the reference pressure.

    The default reference pressure is a constant whose value is specified in the
    input config.

    Args:
      zz: The coordinates along the direction of height/gravitation. Useful in
        geophysical flows.

    Returns:
      The reference pressure in the simulation.
    """
    return [
        self._params.p_thermal * tf.ones_like(zz_i, dtype=TF_DTYPE)
        for zz_i in zz
    ]

  def update_density(
      self,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldVal:
    """Defines a pure virtual interface for the density update function."""
    raise NotImplementedError(
        'A thermodynamic model needs to provide a definition for the density '
        'update function.'
    )
