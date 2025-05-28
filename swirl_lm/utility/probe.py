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

"""A library for retrieving values at specific points in the flow field."""

from typing import Optional, Text

import numpy as np
from swirl_lm.base import initializer
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
from swirl_lm.utility.post_processing import data_processing
import tensorflow as tf


FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap
_DTYPE = tf.float32


class Probe:
  """A library for getting values from the flow field."""

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    """Initializes the probe library."""
    assert (
        probe := params.probe
    ) is not None, 'Probe must be set in the config.'
    self.variable_names = probe.variable_name

    domain_size = (params.lx, params.ly, params.lz)
    mesh_size_local = (params.nx, params.ny, params.nz)
    partition = (params.cx, params.cy, params.cz)

    self.c_indices = np.zeros((len(probe.location), 3), dtype=np.int32)
    self.indices = np.zeros((len(probe.location), 3), dtype=np.int32)
    for i in range(len(probe.location)):
      location = np.array([[
          probe.location[i].dim_0, probe.location[i].dim_1,
          probe.location[i].dim_2
      ]])
      self.c_indices[i, :], self.indices[i, :] = (
          data_processing.coordinates_to_indices(location, domain_size,
                                                 mesh_size_local, partition,
                                                 params.halo_width))

    self.start_step_id = probe.start_step_id
    self.nt = probe.nt

  def probe_name(self, index: int) -> Text:
    """Generates the variable name for the `index`th probe."""
    return 'PROBE_{}'.format(index)

  def initialization(
      self,
      replica_id: tf.Tensor,
      coordinates: initializer.ThreeIntTuple,
  ) -> types.TensorMap:
    """Initializes probe variables with zeros for the simulation.

    Args:
      replica_id: The ID number of the replica.
      coordinates: A tuple that specifies the replica's grid coordinates in
        physical space.

    Returns:
      A dictionary of probes with values inialized as zeros.
    """
    del replica_id, coordinates

    probes = {}
    for i in range(self.c_indices.shape[0]):
      # Different variables for the same timestamp are recorded in a row. The
      # first column is the time of a specific step.
      probes.update({
          self.probe_name(i):
              tf.zeros((self.nt, len(self.variable_names) + 1), dtype=_DTYPE)
      })

    return probes

  def additional_states_update_fn(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      step_id: tf.Tensor,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Updates values in tables of the probes.

    Args:
      kernel_op: An object holding a library of kernel operations.
      replica_id: The id of the replica.
      replicas: The replicas. In particular, a numpy array that maps grid
        coordinates to replica id numbers.
      step_id: The index of the current time step.
      states: A keyed dictionary of states that will be updated.
      additional_states: A list of states that are needed by the update fn, but
        will not be updated by the main governing equations.
      params: An instance of `grid_parametrization.GridParametrization`.

    Returns:
      The updated probes.
    """
    del replica_id, kernel_op

    t = tf.cast(step_id, _DTYPE) * params.dt

    # Grab values at locations specified in `self.indices` in each replica.
    probe_local = tf.zeros((len(self.indices), len(self.variable_names)),
                           dtype=_DTYPE)
    for i in range(self.indices.shape[0]):
      iz = self.indices[i][2]
      ix = self.indices[i][0]
      iy = self.indices[i][1]
      for j in range(len(self.variable_names)):
        probe_local = tf.tensor_scatter_nd_update(
            probe_local, [[i, j]], [states[self.variable_names[j]][iz][ix, iy]])

    # Gather information from all replicas.
    group_assignment = np.array([np.arange(np.prod(replicas.shape))])
    probe_all = common_ops.global_reduce(probe_local, lambda x: x,
                                         group_assignment)

    # Update values from the correct replica.
    probes = {}
    row_id = step_id - self.start_step_id
    for i in range(self.indices.shape[0]):
      probe_old = additional_states[self.probe_name(i)]
      ci = replicas[self.c_indices[i][0], self.c_indices[i][1],
                    self.c_indices[i][2]]
      # Put the current time into the first column of the corresponding row.
      probe_old = tf.tensor_scatter_nd_update(probe_old, [[row_id, 0]], [t])
      for j in range(len(self.variable_names)):
        # Put other variables into columns following their order in the config
        # file.
        probe_old = tf.tensor_scatter_nd_update(probe_old, [[row_id, j + 1]],
                                                [probe_all[ci, i, j]])
      probes.update({self.probe_name(i): probe_old})

    return probes


def probe_factory(params: parameters_lib.SwirlLMParameters) -> Optional[Probe]:
  """Creates an object of the probe library if requested."""
  return None if params.probe is None else Probe(params)
