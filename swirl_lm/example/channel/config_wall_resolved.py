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

"""Example 2: A Turbulent Channel Flow."""

# Solver parameters and physical conditions.
GRID_PBTXT = """
  # The number of cores in 3 dimensions.
    computation_shape {
      dim_0: 4
      dim_1: 2
      dim_2: 1
    }
  # The physical size of the simulation domain in units of m.
  length {
    dim_0: 25.1327412287
    dim_1: 2.0
    dim_2: 3.1415926536
  }
  # The number of grid points per core in 3 dimensions including ghost cells
  # (halos).
  grid_size {
    dim_0: 64
    dim_1: 64
    dim_2: 64
  }
  periodic {
    dim_0: false
    dim_1: false
    dim_2: true
  }
  # An indicator of mesh stretching in each dimension.
  stretched_grid_files {
    dim_1 {
      path: 'swirl_lm/example/channel/test_data/y_120.txt'
    }
  }
  # The width of the ghost cells on each side of the domain. It is set to 2
  # considering the stencil width of the QUICK scheme.
  halo_width: 2
  # The time step size in units of s.
  dt: 1e-2
  # The size of the convolution kernel to be used for fundamental numerical
  # operations.
  kernel_size: 16
"""

SIM_PBTXT = """
  # proto-file: swirl_lm/base/parameters.proto
  # proto-message: SwirlLMParameters
  solver_procedure: VARIABLE_DENSITY
  convection_scheme: CONVECTION_SCHEME_QUICK
  time_integration_scheme: TIME_SCHEME_CN_EXPLICIT_ITERATION
  enable_rhie_chow_correction: false
  enable_scalar_recorrection: true
  use_3d_tf_tensor: true
  num_sub_iterations: 3
  pressure {
    solver {
      jacobi {
        max_iterations: 20 halo_width: 2 omega: 0.67
      }
    }
    num_d_rho_filter: 0
    pressure_outlet: false
  }
  thermodynamics {
    constant_density {}
  }
  # To be included when inflow data is provided.
  # boundary_models {
  #   simulated_inflow {
  #     inflow_dim: 0
  #     enforcement {
  #       inflow_data_dir: ""
  #       inflow_data_prefix: ""
  #       inflow_data_step: 0
  #       face: 0
  #     }
  #     nt: 5120
  #     start_step_id: 0
  #     delta_t: 5e-2
  #   }
  # }
  use_sgs: true
  sgs_model {
    smagorinsky {
      c_s: 0.18
      pr_t: 1.0
      use_pr_t: false
    }
  }
  density: 1.0
  p_thermal: 1.01325e5
  kinematic_viscosity: 0.00025
  additional_state_keys: "bc_u_0_0"
  # To be included when inflow data is provided.
  # additional_state_keys: "bc_v_0_0"
  # additional_state_keys: "bc_w_0_0"
  # additional_state_keys: "INFLOW_U"
  # additional_state_keys: "INFLOW_V"
  # additional_state_keys: "INFLOW_W"
  # states_from_file: "INFLOW_U"
  # states_from_file: "INFLOW_V"
  # states_from_file: "INFLOW_W"
  boundary_conditions {
    name: "u"
    boundary_info {
      dim: 0
      location: 0
      type: BC_TYPE_DIRICHLET
      value: 1.0
    }
    boundary_info {
      dim: 0
      location: 1
      type: BC_TYPE_NEUMANN
      value: 0.0
    }
    boundary_info {
      dim: 1
      location: 0
      type: BC_TYPE_DIRICHLET
      value: 0.0
    }
    boundary_info {
      dim: 1
      location: 1
      type: BC_TYPE_DIRICHLET
      value: 0.0
    }
  }
  boundary_conditions {
    name: "v"
    boundary_info {
      dim: 0
      location: 0
      type: BC_TYPE_DIRICHLET
      value: 0.0
    }
    boundary_info {
      dim: 0
      location: 1
      type: BC_TYPE_NEUMANN
      value: 0.0
    }
    boundary_info {
      dim: 1
      location: 0
      type: BC_TYPE_DIRICHLET
      value: 0.0
    }
    boundary_info {
      dim: 1
      location: 1
      type: BC_TYPE_DIRICHLET
      value: 0.0
    }
  }
  boundary_conditions {
    name: "w"
    boundary_info {
      dim: 0
      location: 0
      type: BC_TYPE_DIRICHLET
      value: 0.0
    }
    boundary_info {
      dim: 0
      location: 1
      type: BC_TYPE_NEUMANN
      value: 0.0
    }
    boundary_info {
      dim: 1
      location: 0
      type: BC_TYPE_DIRICHLET
      value: 0.0
    }
    boundary_info {
      dim: 1
      location: 1
      type: BC_TYPE_DIRICHLET
      value: 0.0
    }
  }
"""
