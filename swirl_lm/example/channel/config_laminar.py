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

"""Example 1: A Laminar Channel Flow."""

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
    dim_0: 4.0
    dim_1: 1.0
    dim_2: 0.1
  }
  # The number of grid points per core in 3 dimensions including ghost cells
  # (halos).
  grid_size {
    dim_0: 128
    dim_1: 128
    dim_2: 6
  }
  periodic {
    dim_0: false
    dim_1: false
    dim_2: true
  }
  # The width of the ghost cells on each side of the domain. It is set to 2
  # considering the stencil width of the QUICK scheme.
  halo_width: 2
  # The time step size in units of s. Note that the CFL number corresponding to
  # this time step size is very small. This is used for the investigation of
  # the convergence with respect to the number of subiterations per time step,
  # which requires the time step size to be sufficiently small.
  dt: 1e-5  # 5e-4
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
  enable_rhie_chow_correction: true
  enable_scalar_recorrection: true
  num_sub_iterations: 10
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
    ideal_gas_law {}
  }
  density: 1.0
  p_thermal: 1.01325e5
  kinematic_viscosity: 1.0e-2
  scalars {
    name: "T"
    diffusivity: 1.0e-2
    reference_value: 283.0
  }
  scalars {
    name: "Y_O"
    diffusivity: 1.0e-2
    molecular_weight: 0.029
    density: 1.2228709548
    solve_scalar: true
  }
  scalars {
    name: "ambient"
    diffusivity: 1.0e-2
    molecular_weight: 0.029
    density: 1.2228709548
    solve_scalar: false
  }
  additional_state_keys: "bc_u_0_0"
  # additional_state_keys: "bc_T_0_0"
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
  boundary_conditions {
    name: "T"
    boundary_info {
      dim: 0 location: 0 type: BC_TYPE_DIRICHLET value: 300.0
    }
    boundary_info {
      dim: 0 location: 1 type: BC_TYPE_NEUMANN value: 0.0
    }
    boundary_info {
      dim: 1 location: 0 type: BC_TYPE_DIRICHLET value: 300.0
    }
    boundary_info {
      dim: 1 location: 1 type: BC_TYPE_DIRICHLET value: 300.0
    }
  }
  boundary_conditions {
    name: "Y_O"
    boundary_info {
      dim: 0 location: 0 type: BC_TYPE_DIRICHLET value: 0.23
    }
    boundary_info {
      dim: 0 location: 1 type: BC_TYPE_NEUMANN value: 0.0
    }
    boundary_info {
      dim: 1 location: 0 type: BC_TYPE_NEUMANN value: 0.0
    }
    boundary_info {
      dim: 1 location: 1 type: BC_TYPE_NEUMANN value: 0.0
    }
  }
"""
