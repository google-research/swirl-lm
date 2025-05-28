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

"""A library generates/enforces inflow BC from an independent simulation."""

import re
from typing import List, Optional, Tuple

import numpy as np
from swirl_lm.base import driver_tpu
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import physical_variable_keys_manager
from swirl_lm.boundary_condition import simulated_turbulent_inflow_pb2
from swirl_lm.equations import common
from swirl_lm.utility import common_ops
from swirl_lm.utility import types
import tensorflow as tf

_T_EPS = 0.5

INFLOW_DATA_NAMES = ('INFLOW_U', 'INFLOW_V', 'INFLOW_W')

_INFLOW_DATA_FILE_FORMAT = (
    '{prefix}-field-{{}}-xyz-{rx}-{ry}-{rz}-step-{step}.ser')


def _required_inflow_data_names(
    inflow_params: simulated_turbulent_inflow_pb2.SimulatedTurbulentInflow,
) -> List[str]:
  """Generates a list of names for all inflow variables."""
  inflow_scalars = []
  for sc_name in inflow_params.scalar:
    inflow_scalars.append(f'INFLOW_{sc_name.upper()}')
  return list(INFLOW_DATA_NAMES) + inflow_scalars


def _required_bc_names(
    inflow_params: simulated_turbulent_inflow_pb2.SimulatedTurbulentInflow,
) -> List[str]:
  """Generates the boundary condition variables names based on the config."""
  operation = inflow_params.WhichOneof('operation')
  if operation == 'generation':
    return []
  elif operation == 'enforcement':
    return [
        f'bc_{var}_{inflow_params.inflow_dim}_{inflow_params.enforcement.face}'
        for var in common.KEYS_VELOCITY
    ]
  else:
    raise ValueError(
        f'Unknown simulated turbulent inflow operation '
        f'{operation}. Supported operations are: "generation", "enforcement".')


class SimulatedTurbulentInflow():
  """A library generates/enforces inflow BC from an independent simulation.

  Remarks:
  1. The distribution topology for the inflow generation and enforcement have
     to be the same.
  2. The number of mesh points of the inflow plane in the inflow data must match
     those in the simulation where this BC is enforced.
  3. The inflow data is a 3D tensor, with the last 2 dimensions being the
     spatial dimension, and the first one being the temporal one. Specifically,
     for inflow_dim = 0, 1, and 2, the shapes of the inflow data are
     (nt, nz, ny), (nt, nz, nx), and (nt, nx, ny), respectively.
  """

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    """Initializes the inflow library."""
    self._params = params
    assert (
        boundary_models := params.boundary_models
    ) is not None, '`boundary_models` must be set in the config.'
    self._model_params = boundary_models.simulated_inflow

    # Get the inflow dimension (in order x-y-z) and inflow axis
    # (in order z-x-y). The inflow dimension refers to the physical dimension,
    # i.e. x = 0, y = 1, and z = 2; the inflow axis refers to the actual
    # dimension in a 3D tensor that the physical dimension is corresponding to
    # (recall that the 3D tensor in this framework is oriented as z-x-y).
    # Specifically, axis 0 corresponds to dimension 2, axis 1 is dimension 0,
    # and axis 2 is dimension 1.
    self._inflow_dim = self._model_params.inflow_dim
    self._inflow_axis = (self._inflow_dim + 1) % 3

    # Get the data shape in each core.
    self._dim_t_nc = (self._params.cx, self._params.cy,
                      self._params.cz)[self._inflow_dim]
    nt_local = int(self._model_params.nt / self._dim_t_nc)

    # The shape of the inflow data is (nt, n_dim_0, n_dim_1). Specifically,
    # inflow_dim == 0: (nt, nz, ny)
    # inflow_dim == 1: (nt, nz, nx)
    # inflow_dim == 2: (nt, nx, ny)
    # This treatment is for the convenience of adapting the inflow data to the
    # 3D tensors that are oriented in the order of z-x-y in this simulation
    # framework.
    self._inflow_data_shape = [
        self._params.nz, self._params.nx, self._params.ny
    ]
    del self._inflow_data_shape[self._inflow_axis]
    self._inflow_data_shape.insert(0, nt_local)

    if self._model_params.WhichOneof('operation') == 'generation':
      # Get the index of the plane to extract the inflow data.
      mesh_size = self._params.grid_spacings[self._inflow_dim]
      mesh_count = (self._params.nx, self._params.ny,
                    self._params.nz)[self._inflow_dim]
      idx = int(self._model_params.generation.location // mesh_size)
      self._sender_core_id = int(idx / mesh_count)
      self._mesh_idx = int(idx % mesh_count)
    else:
      self._sender_core_id = None
      self._mesh_idx = None

    self._key_manager = (
        physical_variable_keys_manager.BoundaryConditionKeysHelper())

    # Get the names of inflow variables.
    self.inflow_data_names = _required_inflow_data_names(self._model_params)

  @property
  def inflow_dim(self):
    """The dimension along which the inflow is imposed."""
    return self._inflow_dim

  def _inflow_data_to_bc(
      self,
      inflow_data: tf.Tensor,
      index: int,
  ) -> types.FlowFieldVal:
    """Extracts the inflow plane at `index`.

    Args:
      inflow_data: The 3D tensor storing the partitioned inflow data.
      index: The local index where the plane is extracted. It has to be within
        the range of the last dimension of `inflow_data`.

    Returns:
      A list of 2D tensors representing a 3D structure, which will be used as
      the boundary condition. The orientation of the 2D tensors follows an
      order of z-x-y.

    Raises:
      ValueError: If `index` is out of the range of the last dimension of
        `inflow_data`.
    """
    inflow_shape = tf.shape(inflow_data)
    check_op = tf.debugging.assert_equal(
        tf.math.logical_and(
            tf.math.less(index, inflow_shape[0]),
            tf.math.greater_equal(index, 0)), True,
        (f'Index ({index}) is out of range [0, {inflow_shape[2]}] when '
         f'extracting the inflow plane.'))

    with tf.control_dependencies([check_op]):
      plane = tf.squeeze(inflow_data[index, ...])
      tiles = [1, 1, 1]
      tiles[self._inflow_axis] = self._params.halo_width + 1
      inflow_bc = tf.tile(tf.expand_dims(plane, self._inflow_axis), tiles)
      return inflow_bc

  def _get_source_data(
      self,
      data: types.FlowFieldVal,
      replicas: np.ndarray,
      source_id: int,
  ) -> types.FlowFieldVal:
    """Gets the data from `source_id` along the inflow dimension."""
    group_assignment = common_ops.group_replicas(replicas, self._inflow_dim)
    num_replicas = len(group_assignment[0])
    buf = tf.repeat([data,], num_replicas, 0)

    data_all = tf.raw_ops.AllToAll(
        input=buf,
        group_assignment=group_assignment,
        concat_dimension=0,
        split_dimension=0,
        split_count=num_replicas)

    return tf.gather(data_all, source_id)

  def _get_time_indices_and_fraction(
      self,
      step_id: tf.Tensor,
      start_step_id: int,
  ) -> Tuple[int, int, float]:
    """Computes the core and local indices for time with fraction of delta t."""
    t = self._params.dt * tf.cast(step_id - start_step_id, dtype=types.TF_DTYPE)
    t_index_global = tf.cast(t // self._model_params.delta_t, dtype=tf.int32)
    core_index = tf.cast(
        t_index_global // self._inflow_data_shape[0], dtype=tf.int32)
    plane_index = tf.cast(
        t_index_global % self._inflow_data_shape[0], dtype=tf.int32)
    t_scale = tf.cast(
        tf.math.round(self._model_params.delta_t / self._params.dt),
        dtype=tf.int32)
    t_fraction = tf.math.divide(
        tf.math.floormod(
            tf.cast(step_id - start_step_id, dtype=tf.int32), t_scale), t_scale)
    return core_index, plane_index, tf.cast(t_fraction, dtype=types.TF_DTYPE)

  def initialize_inflow(self) -> types.FlowFieldMap:
    """Initializes the inflow data."""
    inflow_data = {
        key: tf.zeros(self._inflow_data_shape, dtype=types.TF_DTYPE)
        for key in self.inflow_data_names
    }

    if self._model_params.WhichOneof('operation') == 'enforcement':
      # Initializes the boundary condition variables.
      required_bc_names = _required_bc_names(self._model_params)
      inflow_data.update({
          bc_key: tf.stack(self._inflow_data_to_bc(inflow_data[data_key], 0))
          for bc_key, data_key in zip(required_bc_names, self.inflow_data_names)
      })

    return inflow_data

  # This function is not currently used in simulations.
  def read_inflow(
      self,
      strategy: tf.distribute.TPUStrategy,
      coordinates: List[Tuple[int, int, int]],
  ) -> dict[str, driver_tpu.PerReplica]:
    """Loads inflow data from files for BC enforcement."""
    if self._model_params.WhichOneof('operation') != 'enforcement':
      raise ValueError(
          'The read_inflow operation is only available for inflow enforcement.')

    states = tuple([
        {
            key: tf.constant(0, dtype=types.TF_DTYPE)
            for key in self.inflow_data_names
        },
    ] * len(coordinates))
    return driver_tpu.distributed_read_state(
        strategy, states, coordinates,
        self._model_params.enforcement.inflow_data_dir,
        self._model_params.enforcement.inflow_data_prefix,
        tf.constant(
            self._model_params.enforcement.inflow_data_step, tf.int32))

  # This function is not currently used in simulations.
  def update_inflow_states(
      self,
      strategy: tf.distribute.TPUStrategy,
      states: tf.distribute.DistributedValues,
      coordinates: List[Tuple[int, int, int]],
  ) -> tf.distribute.DistributedValues:
    """Updates states with the inflow data read from files."""
    states.update(self.read_inflow(strategy, coordinates))
    return states

  def collect_inflow(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      step_id: tf.Tensor,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldMap:
    """Collects inflow data from a simulation."""
    if self._model_params.WhichOneof('operation') != 'generation':
      raise ValueError(
          '`collect_inflow` should only be called for inflow generation. Please'
          ' check the simulation configuration.')

    def get_inflow_plane(varname):
      """Gets the plane of `varname` at `self._mesh_idx` for all cores."""
      return tf.squeeze(tf.stack(
          common_ops.get_face(states[varname], self._inflow_dim, 0,
                              self._mesh_idx)[0]))

    # Get the core and local plane indices of the receiver.
    receiver_core_id, local_plane_id, t_fraction = (
        self._get_time_indices_and_fraction(step_id,
                                            self._model_params.start_step_id))

    # Send the inflow plane to the desired replica and update the inflow data.
    core_coordinates = common_ops.get_core_coordinate(replicas, replica_id)

    def maybe_update_inflow_data(inflow_original, inflow_new):
      return tf.cond(
          tf.math.logical_and(
              tf.math.less(t_fraction * self._model_params.delta_t,
                           _T_EPS * self._params.dt),
              tf.math.equal(core_coordinates[self._inflow_dim],
                            receiver_core_id)), lambda: inflow_new,
          lambda: inflow_original)

    inflow_data = {}
    for inflow_name in self.inflow_data_names:
      m = re.match(r'INFLOW_(\w+)', inflow_name)
      assert m is not None, (
          f'Invalid inflow name {inflow_name}. Inflow data should start with'
          ' prefix "INFLOW_".'
      )
      varname = m.group(1).lower()
      inflow_plane = self._get_source_data(
          get_inflow_plane(varname), replicas, self._sender_core_id)
      updated_inflow_data_dim = tf.tensor_scatter_nd_update(
          additional_states[inflow_name], [[local_plane_id]],
          inflow_plane[tf.newaxis, ...])
      inflow_data.update({
          inflow_name:
              maybe_update_inflow_data(additional_states[inflow_name],
                                       updated_inflow_data_dim)
      })

    return inflow_data

  def update_inflow(
      self,
      replicas: np.ndarray,
      step_id: tf.Tensor,
      additional_states: types.FlowFieldMap,
      use_3d_tf_tensor: bool = False,
  ) -> types.FlowFieldMap:
    """Updates the boundary condition with inflow data."""
    if self._model_params.WhichOneof('operation') != 'enforcement':
      raise ValueError(
          '`update_inflow` should only be called for inflow enforcement. '
          'Please check the simulation configuration.')

    check_op = tf.debugging.assert_greater_equal(
        step_id, self._model_params.start_step_id,
        (f'The start step id should be smaller than the simulation step id. '
         f'Currently start step id: {self._model_params.start_step_id}, '
         f'step id: {step_id}')
    )
    with tf.control_dependencies([check_op]):
      # In case the number of time steps in the new simulation where the inflow
      # data is used is greater than those generated, the inflow BC re-iterates
      # from the beginning of the inflow data after it reaches the last plane.
      t = tf.cast(
          step_id - self._model_params.start_step_id,
          dtype=types.TF_DTYPE) * self._params.dt
      t_cycle = self._model_params.nt * self._model_params.delta_t
      step_id_recoil = tf.math.round((t % t_cycle) / self._params.dt)
      sender_core_id, local_inflow_time_index, t_fraction = (
          self._get_time_indices_and_fraction(step_id_recoil, 0))
      next_core_id, next_time_index = tf.cond(
          tf.math.less(local_inflow_time_index + 1, self._inflow_data_shape[0]),
          lambda: (sender_core_id, local_inflow_time_index + 1), lambda:  # pylint: disable=g-long-lambda
          (sender_core_id + 1, 0))

      inflow_bc = {}
      for inflow_name in self.inflow_data_names:
        m = re.match(r'INFLOW_(\w+)', inflow_name)
        assert m is not None, (
            f'Invalid inflow name {inflow_name}. Inflow data should start with'
            ' prefix "INFLOW_".'
        )
        varname = m.group(1).lower()
        bc_key = self._key_manager.generate_bc_key(
            varname, self._inflow_dim, self._model_params.enforcement.face)
        bc_val_0 = self._get_source_data(
            self._inflow_data_to_bc(
                additional_states[inflow_name],
                local_inflow_time_index,
            ),
            replicas,
            sender_core_id,
        )
        bc_val_1 = self._get_source_data(
            self._inflow_data_to_bc(
                additional_states[inflow_name],
                next_time_index,
            ),
            replicas,
            next_core_id,
        )
        bc_val = tf.nest.map_structure(
            lambda bc_0, bc_1: (1.0 - t_fraction) * bc_0 + t_fraction * bc_1,
            bc_val_0, bc_val_1)

        if not use_3d_tf_tensor:
          bc_val = tf.unstack(bc_val)

        inflow_bc.update({bc_key: bc_val})

    return inflow_bc

  def additional_states_update_fn(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      step_id: tf.Tensor,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ) -> types.FlowFieldMap:
    """Updates/Collects the inflow boundary condition."""
    use_3d_tf_tensor = isinstance(list(states.values())[0], tf.Tensor)

    updated_additional_states = dict(additional_states)

    operation = self._model_params.WhichOneof('operation')
    if operation == 'generation':
      updated_additional_states.update(
          self.collect_inflow(replica_id, replicas, step_id, states,
                              additional_states))
    elif operation == 'enforcement':
      updated_additional_states.update(
          self.update_inflow(
              replicas, step_id, additional_states, use_3d_tf_tensor
          )
      )
    else:
      raise ValueError(
          f'{operation} is not supported. Available options: "generation", '
          f'"enforcement".')

    return updated_additional_states


def simulated_turbulent_inflow_factory(
    params: parameters_lib.SwirlLMParameters,
) -> Optional[SimulatedTurbulentInflow]:
  """Creates a `SimulatedTurbulentInflow` object if requested.

  Args:
    params: The configuration context of a simulation.

  Returns:
    An `SimulatedTurbulentInflow` object if simulated turbulent inflow is
    requested in the config, otherwise returns `None`.

  Raises:
    ValueError: If required helper variables are not found in the simulation
      configuration file.
  """
  if params.boundary_models is None or not params.boundary_models.HasField(
      'simulated_inflow'):
    return None

  inflow_params = params.boundary_models.simulated_inflow
  for required_varname in _required_inflow_data_names(
      inflow_params
  ) + _required_bc_names(inflow_params):
    if required_varname not in list(params.helper_var_keys) + list(
        params.additional_state_keys
    ):
      raise ValueError(
          f'Simulated turbulent inflow is requested by {required_varname} is '
          'not provided as a `helper_var`.'
      )

  return SimulatedTurbulentInflow(params)
