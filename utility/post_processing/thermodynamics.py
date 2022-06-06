"""A library of thermodynamics to be used during post-processing.

In this library, all variables are computed with the underlying modules used in
the simulation. No duplication of real logic is introduced.

This library can be imported from colab with adhoc_import. For example:
from colabtools import adhoc_import

with adhoc_import.Google3(
    build_targets=['//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2']):
    #pylint: disable=line-too-long
  from
  google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.utilities.post_processing
  import thermodynamics  #pylint: disable=line-too-long
"""

from typing import Optional, Sequence, Text

import numpy as np
from swirl_lm.physics.thermodynamics import water
from swirl_lm.utility import types
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.pyglib import gfile
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_parameters_pb2

_TF_DTYPE = types.TF_DTYPE


class Water(object):
  """A thermodynamics library with water phase transfer."""

  def __init__(self, config_filepath: Text, tf1: bool = False):
    """Initializes the thermodynamics library used in the NS solver."""
    with gfile.GFile(config_filepath) as f:
      config = text_format.Parse(
          f.read(),
          incompressible_structured_mesh_parameters_pb2
          .IncompressibleNavierStokesParameters())

    params = (
        incompressible_structured_mesh_config
        .IncompressibleNavierStokesParameters(config))

    self.water = water.Water(params)

    def tf1_return_fn(result):
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        return sess.run(result)

    def tf2_return_fn(result):
      return tf.nest.map_structure(lambda x: x.numpy(), result)

    self.return_fn = tf1_return_fn if tf1 else tf2_return_fn

  def _potential_temperatures(
      self,
      varname: Text,
      t: np.ndarray,
      q_t: np.ndarray,
      rho: np.ndarray,
      zz: np.ndarray,
  ) -> np.ndarray:
    """Computes the potential temperature requested.

    Args:
      varname: The name of the potential temperature to be returned. Should be
        one of the following: 'theta_l' (the liquid potential temperature),
          'theta_v' (the virtual potential temperature).
      t: The temperature, in units of K.
      q_t: The total humidity, in units of kg/kg.
      rho: The density of the water-air mixture, in units of kg/m^3.
      zz: The coordinates in the vertical direction, in units of m.

    Returns:
      The potential temperature specified by `varname`.
    """
    t = [tf.constant(t, dtype=_TF_DTYPE)]
    q_t = [tf.constant(q_t, dtype=_TF_DTYPE)]
    rho = [tf.constant(rho, dtype=_TF_DTYPE)]
    zz = [tf.constant(zz, dtype=_TF_DTYPE)]
    res = self.return_fn(self.water.potential_temperatures(t, q_t, rho, zz))  # pytype: disable=wrong-arg-types

    return res[varname][0]

  def _equilibrium_phase_partition(
      self,
      t: np.ndarray,
      rho: np.ndarray,
      q_t: np.ndarray,
  ) -> Sequence[np.ndarray]:
    """Partitions the water phases in equilibrium.

    Args:
      t: The temperature, in units of K.
      rho: The density of the water-air mixture, in units of kg/m^3.
      q_t: The total humidity, in units of kg/kg.

    Returns:
      A tuple with its first element being the liquid phase water mass fraction,
      and the second element being the ice phase water mass fraction.
    """
    t = [tf.constant(t, dtype=_TF_DTYPE)]
    q_t = [tf.constant(q_t, dtype=_TF_DTYPE)]
    rho = [tf.constant(rho, dtype=_TF_DTYPE)]

    res = self.return_fn(self.water.equilibrium_phase_partition(t, rho, q_t))  # pytype: disable=wrong-arg-types

    return res[0][0], res[1][0]

  def p_ref(self, zz: np.ndarray) -> np.ndarray:
    """Computes the reference pressure considering the geopotential.

    Args:
      zz: The geopotential height.

    Returns:
      The reference pressure as a function of height.
    """
    zz = [tf.constant(zz, dtype=_TF_DTYPE)]

    res = self.return_fn(self.water.p_ref(zz))  # pytype: disable=wrong-arg-types

    return res[0]

  def t_ref(self, zz: np.ndarray) -> np.ndarray:
    """Computes the reference temperature considering the geopotential.

    Args:
      zz: The geopotential height.

    Returns:
      The reference temperature as a function of height.
    """
    zz = [tf.constant(zz, dtype=_TF_DTYPE)]
    res = self.return_fn(self.water.t_ref(zz))  # pytype: disable=wrong-arg-types

    return res[0]

  def rho_ref(self, zz: np.ndarray) -> np.ndarray:
    """Computes the reference density.

    Args:
      zz: The geopotential height.

    Returns:
      The reference density as a function of height.
    """
    zz = [tf.constant(zz, dtype=_TF_DTYPE)]

    res = self.return_fn(self.water.rho_ref(zz))  # pytype: disable=wrong-arg-types

    return res[0]

  def virtual_potential_temperature(
      self,
      t: np.ndarray,
      q_t: np.ndarray,
      rho: np.ndarray,
      zz: np.ndarray,
  ) -> np.ndarray:
    """Computes the virtual potential temperature.

    Args:
      t: The temperature, in units of K.
      q_t: The total humidity, in units of kg/kg.
      rho: The density of the water-air mixture, in units of kg/m^3.
      zz: The coordinates in the vertical direction, in units of m.

    Returns:
      The virtual potential temperature specified.
    """
    return self._potential_temperatures('theta_v', t, q_t, rho, zz)

  def liquid_ice_potential_temperature(
      self,
      t: np.ndarray,
      q_t: np.ndarray,
      rho: np.ndarray,
      zz: np.ndarray,
  ) -> np.ndarray:
    """Computes the liquid-ice potential temperature.

    Args:
      t: The temperature, in units of K.
      q_t: The total humidity, in units of kg/kg.
      rho: The density of the water-air mixture, in units of kg/m^3.
      zz: The coordinates in the vertical direction, in units of m.

    Returns:
      The liquid-ice potential temperature specified.
    """
    return self._potential_temperatures('theta_li', t, q_t, rho, zz)

  def temperature(
      self,
      e_int: np.ndarray,
      rho: np.ndarray,
      q_t: np.ndarray,
  ) -> np.ndarray:
    """Computes the temperature for the water-air mixture.

    Args:
      e_int: The mass specific internal energy, in units of J/kg.
      rho: The density of the water-air mixture, in units of kg/m^3.
      q_t: The total humidity, in units of kg/kg.

    Returns:
      The temperature of the flow field.
    """
    e_int = [tf.constant(e_int, dtype=_TF_DTYPE)]
    rho = [tf.constant(rho, dtype=_TF_DTYPE)]
    q_t = [tf.constant(q_t, dtype=_TF_DTYPE)]

    res = self.return_fn(self.water.saturation_adjustment(
        'e_int', e_int, rho, q_t))  # pytype: disable=wrong-arg-types

    return res[0]

  def density(
      self,
      e_tot: np.ndarray,
      q_tot: np.ndarray,
      u: np.ndarray,
      v: np.ndarray,
      w: np.ndarray,
      rho_0: Optional[np.ndarray] = None,
      zz: Optional[np.ndarray] = None,
  ) -> np.ndarray:
    """Computes the density that is consistent with the input state.

    Args:
      e_tot: The total energy.
      q_tot: The total specific humidity.
      u: The velocity component in the x direction.
      v: The velocity component in the y direction.
      w: The velocity component in the z direction. the Secant solver.
      rho_0: A guess of the density, which is used as the initial condition for
      zz: The cooridinates in the vertical direction.

    Returns:
      The density at the given state.
    """
    e_tot = [tf.constant(e_tot, dtype=_TF_DTYPE)]
    q_tot = [tf.constant(q_tot, dtype=_TF_DTYPE)]
    u = [tf.constant(u, dtype=_TF_DTYPE)]
    v = [tf.constant(v, dtype=_TF_DTYPE)]
    w = [tf.constant(w, dtype=_TF_DTYPE)]
    if rho_0 is not None:
      rho_0 = [tf.constant(rho_0, dtype=_TF_DTYPE)]
    if zz is not None:
      zz = [tf.constant(zz, dtype=_TF_DTYPE)]

    res = self.return_fn(
        self.water.saturation_density(
            prognostic_var_name='e_t',
            prognostic_var=e_tot,
            q_tot=q_tot,
            u=u,
            v=v,
            w=w,
            rho_0=rho_0,
            zz=zz))  # pytype: disable=wrong-arg-types

    return res[0]

  def liquid_mass_fraction(
      self,
      t: np.ndarray,
      rho: np.ndarray,
      q_t: np.ndarray,
  ) -> np.ndarray:
    """Computes the liquid water mass fraction at equilibrium.

    Args:
      t: The temperature, in units of K.
      rho: The density of the water-air mixture, in units of kg/m^3.
      q_t: The total humidity, in units of kg/kg.

    Returns:
      The liquid phase water mass fraction.
    """
    q_l, _ = self._equilibrium_phase_partition(t, rho, q_t)
    return q_l

  def condensed_mass_fraction(
      self,
      t: np.ndarray,
      rho: np.ndarray,
      q_t: np.ndarray,
  ) -> np.ndarray:
    """Computes the condensed water mass fraction at equilibrium.

    Args:
      t: The temperature, in units of K.
      rho: The density of the water-air mixture, in units of kg/m^3.
      q_t: The total humidity, in units of kg/kg.

    Returns:
      The condensed phase water mass fraction.
    """
    q_l, q_i = self._equilibrium_phase_partition(t, rho, q_t)
    return q_l + q_i

  def internal_energy(
      self,
      temperature: np.ndarray,
      q_tot: np.ndarray,
      q_liq: np.ndarray,
      q_ice: np.ndarray,
  ) -> np.ndarray:
    """Computes the specific internal energy.

    e = cvₘ (T - T₀) + (qₜ - qₗ) eᵥ₀ - qᵢ (eᵥ₀ + eᵢ₀)

    Args:
      temperature: The temperature of the flow field.
      q_tot: The total specific humidity.
      q_liq: The specific humidity of the liquid phase.
      q_ice: The specific humidity of the solid phase.

    Returns:
      The specific internal energy at the given temperature and humidity
      condition.
    """
    temperature = [tf.constant(temperature, dtype=_TF_DTYPE)]
    q_tot = [tf.constant(q_tot, dtype=_TF_DTYPE)]
    q_liq = [tf.constant(q_liq, dtype=_TF_DTYPE)]
    q_ice = [tf.constant(q_ice, dtype=_TF_DTYPE)]

    res = self.return_fn(
        self.water.internal_energy(temperature, q_tot, q_liq, q_ice))  # pytype: disable=wrong-arg-types

    return res[0]

  def total_enthalpy(
      self,
      e_tot: np.ndarray,
      rho: np.ndarray,
      q_tot: np.ndarray,
      temperature: np.ndarray,
  ) -> np.ndarray:
    """Computes the total enthalpy.

    hₜ = eₜ + Rₘ T

    Args:
      e_tot: The total energy, in units of J/(kg/m^3).
      rho: The moist air density, in units of kg/m^3.
      q_tot: The total specific humidity.
      temperature: The temperature obtained from saturation adjustment that's
        consistent with the other 3 input parameters.

    Returns:
      The total enthalpy, in units of J/(kg/m^3).
    """
    e_tot = [tf.constant(e_tot, dtype=_TF_DTYPE)]
    rho = [tf.constant(rho, dtype=_TF_DTYPE)]
    q_tot = [tf.constant(q_tot, dtype=_TF_DTYPE)]
    temperature = [tf.constant(temperature, dtype=_TF_DTYPE)]

    res = self.return_fn(
        self.water.total_enthalpy(e_tot, rho, q_tot, temperature))  # pytype: disable=wrong-arg-types

    return res[0]
