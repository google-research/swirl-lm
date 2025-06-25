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

"""Code shared among the simulation set up libraries (Dycoms2, SuperCell, ...).
"""
import csv
from typing import Callable, Dict, List, Optional

from absl import flags
import numpy as np
from scipy import integrate
from swirl_lm.base import initializer
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.physics import constants
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap
_HaloUpdateFn = Callable[..., FlowFieldVal]

# TODO(wqing): Remove this flag and use the 'PHYSICAL' mode only after all
# simulations are fully verified.
_INIT_MODE = flags.DEFINE_enum(
    'init_mode',
    'PAD',
    ['PAD', 'PHYSICAL'],
    'Defines the mode of initialization. If "PAD" is used, the mesh used to'
    ' construct the initial condition contains the valid fluid domain only.'
    ' Values in the halos are filled with "SYMMETRIC" padding. If "PHYSICAL"'
    ' is used, the full mesh including halos will be used to construct the'
    ' initial conditions.',
    allow_override=True,
)


class GeophysicalFlowSetup:
  """Defines a generic framework for geophysical flow simulation setups."""

  def __init__(self, config: parameters_lib.SwirlLMParameters):
    """Initializes the library."""
    self.init_mode = _INIT_MODE.value

    self.config = config
    self.init_wind = {'u': 0.0, 'v': 0.0, 'w': 0.0}
    self.coriolis_force_fn = None
    self.radiation_src_update_fn = None
    self.sounding = {}

    self.g_vec = (
        self.config.gravity_direction if self.config.gravity_direction else [
            0.0,
        ] * 3)
    self.g_dim = config.g_dim
    self.core_n = (
        self.config.core_nx,
        self.config.core_ny,
        self.config.core_nz,
    )
    self.core_n_full = (self.config.nx, self.config.ny, self.config.nz)

  def _init_from_sounding(
      self,
      varname: str,
      z_coord: int,
  ) -> tf.Tensor:
    """Initializes a full 3D `tf.Tensor` from the sounding profile."""
    assert varname in self.sounding, (
        f'`{varname}` not available in sounding. Available vars are: '
        f'{self.sounding.keys}'
    )
    z = self.core_n[self.g_dim] * z_coord
    nz = (
        self.core_n[self.g_dim]
        if self.init_mode == 'PAD'
        else self.core_n_full[self.g_dim]
    )

    thin_shape = [1, 1, 1]
    thin_shape[self.g_dim] = -1
    profile_slice = tf.reshape(
        self.sounding[varname][z : z + nz],
        thin_shape
    )

    repeats = (
        list(self.core_n) if self.init_mode == 'PAD' else list(self.core_n_full)
    )
    repeats[self.g_dim] = 1
    return tf.tile(profile_slice, repeats)

  def _init_fn_from_sounding(
      self,
      varname: str,
  ):
    # pylint: disable=g-long-lambda
    return lambda xx, yy, zz, lx, ly, lz, coord: self._init_from_sounding(
        varname,
        coord[self.g_dim],
    )

  def initialize_states(
      self,
      value_fn: initializer.ValueFunction,
      coordinates: initializer.ThreeIntTuple,
  ) -> tf.Tensor:
    """Generates partial states for core using `value_fn`.

    Args:
      value_fn: A function that takes coordinates information that computes the
        flow field.
      coordinates: The index of the TPU replica in the partition.

    Returns:
      A 3D flow field variable.
    """
    pad_mode = 'SYMMETRIC' if self.init_mode == 'PAD' else 'PHYSICAL'
    return initializer.partial_mesh_for_core(
        self.config,
        coordinates,
        value_fn,
        pad_mode=pad_mode,
        mesh_choice=initializer.MeshChoice.PARAMS,
    )

  def thermodynamics_states(
      self,
      zz: FlowFieldVal,
      xx: Optional[FlowFieldVal] = None,
      yy: Optional[FlowFieldVal] = None,
      lx: Optional[float] = None,
      ly: Optional[float] = None,
      coord: Optional[initializer.ThreeIntTuple] = None,
  ) -> FlowFieldMap:
    """Initial conditions of thermodynamic states from simulation setup.

    Args:
      zz: The coordinates along the vertical direction.
      xx: The coordinates along the first dimension in the horizontal direction.
      yy: The coordinates along the second dimension in the horizontal
        direction.
      lx: The total physical length of the first horizontal direction.
      ly: The total physical length of the second horizontal direction.
      coord: The coordinate of the local core.

    Returns:
      A dictionary that holds variables and must contain at least 'temperature',
     'q_t', 'r_m', and 'cp_m'.

    Returns:
      A dictionary that holds variables and must contain at least 'temperature',
      'q_t', 'r_m', and 'cp_m'.
    """
    raise NotImplementedError(
        'User defined initial conditions are required for a specific flow '
        'configuration.'
    )

  def velocity_init_fn(
      self,
      varname: str,
  ) -> initializer.ValueFunction:
    """Generates the init functions for `varname`.

    The velocity specified by `varname` should be one of 'u', 'v', or 'w'.

    Args:
      varname: The name of the velocity component for which the initial state is
        generated.

    Returns:
      The initial states for variable `varname`.
    """
    raise NotImplementedError(
        'User defined initialization functions are required for a specific flow'
        'configuration.'
    )

  def momentum_source_fn(
      self,
  ) -> parameters_lib.SourceUpdateFnLib:
    """Constructs the source functions for the momentum equations.

    Returns:
      A dictionary containing functions for updating the source terms of the
      momentum equations.
    """
    return {}

  def scalar_source_fn(
      self,
  ) -> parameters_lib.SourceUpdateFnLib:
    """Constructs the source functions for the transport scalar equations.

    Returns:
      A dictionary containing functions for updating the source terms of the
      transported scalar equations.
    """
    return {}

  def initial_halo_update_fn(
      self,
  ) -> Dict[str, _HaloUpdateFn]:
    """Returns functions for updating the halos of certain additional states."""
    return {}

  def post_simulation_update_fn(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
      params: grid_parametrization.GridParametrization,
  ) -> FlowFieldMap:
    """Updates flow field variables at a specific step in the simulation.

    This function is invoked only once after the step specified in the
    commandline flag.

    Args:
      kernel_op: An object holding a library of kernel operations. Not used in
        the default base class function.
      replica_id: The id of the replica. Not used in the default base class
        function.
      replicas: The replicas. In particular, a numpy array that maps grid
        coordinates to replica id numbers. Not used in the default base class
        function.
      states: A keyed dictionary of states that will be updated.
      additional_states: A list of states that are needed by the update fn, but
        will not be updated by the main governing equations.
      params: An instance of `grid_parametrization.GridParametrization`. Not
        used in the default base class function.

    Returns:
      A dictionary that is a union of `states` and `additional_states`.
    """
    del kernel_op, replica_id, replicas, params
    output = dict(states)
    output.update(additional_states)
    return output

  def initial_states(
      self,
      replica_id: tf.Tensor,
      coordinates: initializer.ThreeIntTuple,
  ) -> FlowFieldMap:
    """Initializes the simulation with the geophysical flow setup."""
    del replica_id, coordinates

    return {}

  def helper_states_fn(self) -> Dict[str, initializer.ValueFunction]:
    """Provides helper variables for thermodynamic states determination."""

    return {}

  def additional_states_update(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      step_id: tf.Tensor,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Updates additional states for the simulation."""
    del replica_id, replicas, step_id, states, additional_states
    return {}


def bubble(
    pert: float,
    rh: float,
    rv: float,
    z_loc: float,
) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor, float, float], tf.Tensor]:
  """Generates a function that specifies a bubble perturbation."""

  def bubble_fn(
      x: tf.Tensor,
      y: tf.Tensor,
      z: tf.Tensor,
      lx: float,
      ly: float,
  ) -> tf.Tensor:
    """Computes the perturbation of the bubble centered in the x-y plane."""
    normalized_distance = tf.sqrt(
        ((x - 0.5 * lx) / rh) ** 2
        + ((y - 0.5 * ly) / rh) ** 2
        + ((z - z_loc) / rv) ** 2
    )
    return tf.where(
        tf.math.less(normalized_distance, 1.0),
        pert * tf.math.cos(0.5 * np.pi * normalized_distance) ** 2,
        tf.zeros_like(normalized_distance),
    )

  return bubble_fn


def _load_csv(filename: str) -> Dict[str, np.ndarray]:
  """Reads and returns the columns of the input csv file (with header row)."""
  out = {}
  with tf.io.gfile.GFile(filename, 'r') as f:
    for i, row in enumerate(csv.DictReader(f)):
      for key, value in row.items():
        out.setdefault(key, []).append(float(value))
      assert all(
          len(values) == i + 1 for values in out.values()
      ), f'Missing values in {filename} while processing {row}.'
  return {
      key: np.array(values, dtype=np.float32) for key, values in out.items()
  }


def interpolate_sounding(
    sounding: Dict[str, np.ndarray], zz: np.ndarray, sounding_z_varname: str
) -> Dict[str, np.ndarray]:
  """Interpolates all variables in 'sounding' to new coordinates 'zz'."""
  out = {}
  for varname, values in sounding.items():
    if varname == sounding_z_varname:
      continue
    out[varname] = np.interp(zz, sounding[sounding_z_varname], values)
  return out


def load_sounding(
    filenames: List[str],
    zz: np.ndarray,
) -> Dict[str, np.ndarray]:
  """Loads sounding profiles from files and interpolates to mesh.

  The files will be read in order and for each variable, only the values from
  the latest file in which the variable is defined will be used.

  The files don't need to all have the same number of rows or z values -
  variable values are interpolated to the grid as the files are being loaded.

  Args:
    filenames: The list of filenames containing the soundings.
    zz: The coordinates in the vertical direction, in units of m.

  Returns:
    A dictionary with the interpolated profiles keyed by variable name.
  """
  sounding = {}
  for filename in filenames:
    sounding.update(interpolate_sounding(_load_csv(filename), zz, 'z'))
  return sounding


def broadcast_vertical_profile_for_inflow(
    u: tf.Tensor, g_dim: int, inflow_dim: int
) -> tf.Tensor:
  """Makes a 1D vertical profile broadcastable for 3D inflows."""
  u = tf.squeeze(tf.convert_to_tensor(u))
  if g_dim == 1:
    # t-z-y if inflow is along x, t-x-y if inflow is along z.
    u = u[tf.newaxis, tf.newaxis, :]
  elif g_dim == 2:
    # t-z-y if inflow is along x, t-z-x if inflow is along y.
    u = u[tf.newaxis, :, tf.newaxis]
  else:  # g_dim == 0:
    if inflow_dim == 1:
      # t-z-x.
      u = u[tf.newaxis, tf.newaxis, :]
    elif inflow_dim == 2:
      # t-x-y.
      u = u[tf.newaxis, :, tf.newaxis]
    else:
      raise ValueError('Both inflow and gravity are in the x direction!')
  return u


def broadcast_vertical_profile_for_flow_field(
    u: tf.Tensor, g_dim: int
) -> tf.Tensor:
  """Makes a 1D vertical profile broadcastable for 3D flow fields."""
  shape_3d = [1, 1, 1]
  shape_3d[(g_dim + 1) % 3] = -1
  return tf.reshape(tf.squeeze(u), shape_3d)


def compute_hydrostatic_pressure_from_theta(
    zz: np.ndarray, theta: np.ndarray, pressure_0: float
) -> np.ndarray:
  """Computes hydrostatic pressure via integration of potential temperature.

  Args:
    zz: Heights to compute pressure.
    theta: Potential temperature at points given by 'zz'.
    pressure_0: Pressure at 'zz[0]'.

  Returns:
    Pressure at points given by 'zz'.
  """
  theta_recip_integral = integrate.cumulative_trapezoid(
      1 / theta, zz, initial=0
  )
  return pressure_0 * np.power(
      (1 - constants.G / constants.CP * theta_recip_integral),
      constants.CP / constants.R_D,
  )


def compute_hydrostatic_pressure_from_temperature(
    zz: np.ndarray, temperature: np.ndarray, pressure_0: float
) -> np.ndarray:
  """Computes hydrostatic pressure via numerical integration of temperature.

  Numerically computes p = p₀exp{- G / R ∫ (1 / T(z)) dz}.

  Args:
    zz: Heights to compute pressure.
    temperature: Temperature at points given by 'zz'.
    pressure_0: Pressure at 'zz[0]'.

  Returns:
    Pressure at points given by 'zz'.
  """
  temperature_recip_integral = integrate.cumulative_trapezoid(
      1 / temperature, zz, initial=0
  )
  return pressure_0 * np.exp(
      -constants.G / constants.R_D * temperature_recip_integral
  )
