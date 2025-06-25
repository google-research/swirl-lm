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

"""Setup for the Dycoms II RF01 simulation."""
import dataclasses
from typing import Optional

from swirl_lm.base import initializer
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.example.geo_flows import geophysical_flow_common
from swirl_lm.example.shared import cloud_utils
from swirl_lm.example.shared import geophysical_flow_utils
from swirl_lm.utility import init_fn as init_fn_lib
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldMap = types.FlowFieldMap
FlowFieldVal = types.FlowFieldVal


@dataclasses.dataclass
class DycomsSettings:
  """Defines parameters to set up the DYCOMS simulation."""
  # The mean initial velocity (m/s) in the x direction.
  u_init: float = 7.0
  # The mean initial velocity (m/s) in the y direction.
  v_init: float = -5.5
  # The mean initial velocity (m/s) in the z direction.
  w_init: float = 0.0
  # The rms initial velocity (m/s) in the streamwise direction.
  u_rms: float = 1.0
  # The rms initial velocity (m/s) in the lateral direction.
  v_rms: float = 1.0
  # The rms initial velocity (m/s) in the vertical direction.
  w_rms: float = 1.0
  # The latitude in radian where the cloud is located. Default value results in
  # the Coriolis parameter f = 7.62e-5 s⁻¹.
  latitude: float = 0.5497607357
  # The distance in meter where the Blasius boundary layer profile is
  # extracted. This profile is used as the initial condition for u, v, and w.
  # Set to None (by not setting this flag in command) to initialize velocity
  # with a constant.
  bl_distance: Optional[float] = None
  # Used to seed the random noise. Must be set.
  random_seed: int = 42


class Dycoms2RF01(geophysical_flow_common.GeophysicalFlowSetup):
  """Defines initial conditions and essential fields based on DYCOMS II.

  Reference:
  Stevens, Bjorn, Chin-Hoh Moeng, Andrew S. Ackerman, Christopher S. Bretherton,
  Andreas Chlond, Stephan de Roode, James Edwards, et al. 2005. “Evaluation of
  Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus.”
  Monthly Weather Review 133 (6): 1443–62.
  """

  def __init__(
      self,
      config: parameters_lib.SwirlLMParameters,
      sim_params: DycomsSettings,
      cloud_util: cloud_utils.CloudUtils,
  ):
    """Initializes the parametric library for the DYCOMS II RF01 simulation."""
    super().__init__(config)

    self.sim_params = sim_params
    self.cloud_utils = cloud_util
    self.init_wind = {
        'u': self.sim_params.u_init,
        'v': self.sim_params.v_init,
        'w': self.sim_params.w_init,
    }
    self.coriolis_force_fn = cloud_utils.coriolis_force(
        self.sim_params.latitude, self.init_wind, self.g_dim)

    # Temperature below the cloud, in units of K.
    self.t_base = 289.0
    # Temperature right above the cloud, in units of K.
    self.t_top = 297.5
    # Total water mass fraction below the cloud, in units of kg/kg.
    self.q_t_base = 9e-3
    # Total water mass fraction above the cloud, in units of kg/kg.
    self.q_t_top = 1.5e-3

  def thermodynamics_states(
      self,
      zz: FlowFieldVal,
      xx: Optional[FlowFieldVal] = None,
      yy: Optional[FlowFieldVal] = None,
      lx: Optional[float] = None,
      ly: Optional[float] = None,
      coord: Optional[initializer.ThreeIntTuple] = None,
  ) -> FlowFieldMap:
    """Provides the liquid potential temperature and total humidity of DYCOMS.

    Constants listed in this function are obtained from:
    Stevens, Bjorn, Chin-Hoh Moeng, Andrew S. Ackerman, Christopher S.
    Bretherton, Andreas Chlond, Stephan de Roode, James Edwards, et al. 2005.
    “Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine
    Stratocumulus.” Monthly Weather Review 133 (6): 1443–62.

    Args:
      zz: The geopotential height.
      xx: The coordinates along the first dimension in the horizontal direction.
      yy: The coordinates along the second dimension in the horizontal
        direction.
      lx: The total physical length of the first horizontal direction.
      ly: The total physical length of the second horizontal direction.
      coord: The coordinate of the local core.

    Returns:
      The equilibrium profile of liquid water potential temperature and total
      humidity for the DYCOMS simulation.
    """
    del xx, yy, lx, ly
    # This condition is specified in Eq. (1) in Stevens, et. al., 2005.
    theta_li = tf.compat.v1.where(
        tf.less_equal(zz, self.cloud_utils.zi),
        self.t_base * tf.ones_like(zz),
        self.t_top
        + tf.math.pow(tf.maximum(zz - self.cloud_utils.zi, 0.0), 1.0 / 3.0),
    )

    # This condition is specified in Eq. (2) in Stevens, et. al., 2005.
    q_t = tf.compat.v1.where(
        tf.less_equal(zz, self.cloud_utils.zi),
        self.q_t_base * tf.ones_like(zz),
        self.q_t_top * tf.ones_like(zz),
    )

    thermal_states = {'theta_li': theta_li, 'q_t': q_t}
    thermal_states.update(
        self.cloud_utils.temperature(theta_li, q_t, zz, 'theta_li'))

    return thermal_states

  def velocity_init_fn(self, varname: str) -> initializer.ValueFunction:
    """Generates the init functions for `varname` following DYCOMS setup.

    The velocity specified by `varname` should be one of 'u', 'v', or 'w'.

    Args:
      varname: The name of the velocity component for which the initial state is
        generated.

    Returns:
      The initial states for variable `varname`.
    """
    if self.sim_params.bl_distance is None:
      velocity_init_fn = None
    else:
      velocity_init_fn = init_fn_lib.blasius_boundary_layer(
          self.sim_params.u_init,
          self.sim_params.v_init,
          self.config.nu,
          self.config.dx,
          self.config.dy,
          self.config.lz,
          self.config.fz,
          self.sim_params.bl_distance,
          apply_transition=False)[varname]

    seed = {'u': geophysical_flow_utils.U_SEED,
            'v': geophysical_flow_utils.V_SEED,
            'w': geophysical_flow_utils.W_SEED}[varname]
    mean = {
        'u': self.sim_params.u_init,
        'v': self.sim_params.v_init,
        'w': self.sim_params.w_init,
    }[varname]
    rms = {
        'u': self.sim_params.u_rms,
        'v': self.sim_params.v_rms,
        'w': self.sim_params.w_rms,
    }[varname]
    core_n = self.core_n if self.init_mode == 'PAD' else self.core_n_full
    return geophysical_flow_utils.perturbed_constant_init_fn(
        seed + self.sim_params.random_seed,
        mean,
        self.g_dim,
        core_n,
        rms,
        velocity_init_fn,
    )
