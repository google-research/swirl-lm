"""A library of combustion to be used during post-processing.

In this library, all variables are computed with the underlying modules used in
the simulation. No duplication of real logic is introduced.

This library can be imported from colab with adhoc_import. For example:
from colabtools import adhoc_import

with adhoc_import.Google3(
    build_targets=['//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2']):
    #pylint: disable=line-too-long
  from
  google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh.utilities.post_processing
  import combustion  #pylint: disable=line-too-long
"""

from typing import Optional

import numpy as np
from swirl_lm.physics.combustion import wood
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_parameters_pb2


class Wood():
  """A combustion library with wood burning chemistry."""

  def __init__(self, config_pbtxt: str, tf1: bool = False):
    """Initializes the thermodynamics library used in the NS solver."""
    config = text_format.Parse(
        config_pbtxt,
        incompressible_structured_mesh_parameters_pb2
        .IncompressibleNavierStokesParameters())

    params = (
        incompressible_structured_mesh_config
        .IncompressibleNavierStokesParameters(config))

    self.wood = wood.wood_combustion_factory(params)

    def tf1_return_fn(result):
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        return sess.run(result)

    def tf2_return_fn(result):
      return {
          key: tf.nest.map_structure(lambda x: x.numpy(), val)
          for key, val in result.items()
      }

    self.return_fn = tf1_return_fn if tf1 else tf2_return_fn

  def reaction_step(
      self,
      dt: float,
      rho: np.ndarray,
      t_g: np.ndarray,
      y_o: np.ndarray,
      rho_f: np.ndarray,
      t_s: np.ndarray,
      tke: np.ndarray,
      rho_f_init: Optional[np.ndarray] = None,
      rho_m: Optional[np.ndarray] = None,
      phi_w: Optional[np.ndarray] = None,
  ) -> np.ndarray:
    """Produces updated states for one step."""
    if rho_f_init is None:
      update_fn = self.wood.update_fn()
    else:
      update_fn = self.wood.update_fn([tf.convert_to_tensor(rho_f_init)])

    kernel_op = get_kernel_fn.ApplyKernelSliceOp()
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    params = grid_parametrization.GridParametrization()
    params.dt = dt

    states = {
        'rho': [tf.convert_to_tensor(rho)],
        'T': [tf.convert_to_tensor(t_g)],
        'Y_O': [tf.convert_to_tensor(y_o)],
    }
    additional_states = {
        'rho_f': [tf.convert_to_tensor(rho_f)],
        'T_s': [tf.convert_to_tensor(t_s)],
        'tke': [tf.convert_to_tensor(tke)],
    }
    additional_states.update({
        'src_T': [tf.zeros_like(states['T'][0])],
        'src_rho': [tf.zeros_like(states['rho'][0])],
        'src_Y_O': [tf.zeros_like(states['Y_O'][0])],
    })
    if rho_m is not None:
      additional_states.update({
          'rho_m': [tf.convert_to_tensor(rho_m)],
      })
    if phi_w is not None:
      additional_states.update({
          'phi_w': [tf.convert_to_tensor(phi_w)],
      })

    return self.return_fn(
        update_fn(kernel_op, replica_id, replicas, states, additional_states,
                  params))
