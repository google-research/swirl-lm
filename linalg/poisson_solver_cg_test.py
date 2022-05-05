# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for google3.research.simulation.tensorflow.fluid.framework.poisson_solver."""

import itertools

import numpy as np
from swirl_lm.linalg import base_poisson_solver
from swirl_lm.linalg import poisson_solver
from swirl_lm.linalg import poisson_solver_pb2
from swirl_lm.linalg import poisson_solver_testutil
from swirl_lm.utility import get_kernel_fn
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.research.simulation.tensorflow.fluid.framework import util
from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized

_X = base_poisson_solver.X
_RESIDUAL_L2_NORM = base_poisson_solver.RESIDUAL_L2_NORM
_COMPONENT_WISE_DISTANCE = base_poisson_solver.COMPONENT_WISE_DISTANCE
_ITERATIONS = base_poisson_solver.ITERATIONS

_CG_CONFIG = """
  conjugate_gradient {
    max_iterations: 2000
    halo_width: 1
    atol: 1e-10
    reprojection: true
  }
"""

_CG_CONFIG_COMPONENT_WISE_DISTANCE = """
  conjugate_gradient {
    max_iterations: 2000
    halo_width: 1
    atol: 1e-10
    reprojection: true
    component_wise_convergence {
      symmetric: false
      atol: 2e-7
      rtol: 1e-7
    }
  }
"""

_CG_CONFIG_COMPONENT_WISE_DISTANCE_SYMMETRIC = """
  conjugate_gradient {
    max_iterations: 2000
    halo_width: 1
    atol: 1e-10
    reprojection: true
    component_wise_convergence {
      symmetric: true
      atol: 2e-7
      rtol: 1e-7
    }
  }
"""

# Preconditioner with different halo_widths (`m`) and coefficients for the `M`
# matrix with width (at most distance `n` from the diagonal), combined with
# component wise distance as a convergence criterion, should be the fastest
# combination. Named as:
#   _CG_CONFIG_WITH_RPECONDITIONER_BAND_{m}{n}_COMPONENT_WISE_DISTANCE
_CG_CONFIG_WITH_RPECONDITIONER_BAND_11_COMPONENT_WISE_DISTANCE = """
  conjugate_gradient {
    max_iterations: 2000
    halo_width: 1
    atol: 1e-10
    component_wise_convergence {
      symmetric: true
      atol: 2e-7
      rtol: 1e-7
    }
    reprojection: true
    preconditioner {
      band_preconditioner {
        halo_width: 1
        coefficients: -0.5
        coefficients: -1.
        coefficients: -0.5
      }
    }
  }
"""

_CG_CONFIG_WITH_RPECONDITIONER_BAND_21_COMPONENT_WISE_DISTANCE = """
  conjugate_gradient {
    max_iterations: 2000
    halo_width: 2
    atol: 1e-10
    component_wise_convergence {
      symmetric: true
      atol: 2e-7
      rtol: 1e-7
    }
    reprojection: true
    preconditioner {
      band_preconditioner {
        halo_width: 2
        coefficients: -0.5
        coefficients: -1.
        coefficients: -0.5
      }
    }
  }
"""

_CG_CONFIG_WITH_RPECONDITIONER_BAND_22_COMPONENT_WISE_DISTANCE = """
  conjugate_gradient {
    max_iterations: 2000
    halo_width: 2
    atol: 1e-10
    reprojection: true
    component_wise_convergence {
      symmetric: true
      atol: 2e-7
      rtol: 1e-7
    }
    preconditioner {
      band_preconditioner {
        halo_width: 2
        coefficients: -0.5
        coefficients: -1.0
        coefficients: -1.5
        coefficients: -1.0
        coefficients: -0.5
        taylor_expansion_order: %s
        symmetric: %s
      }
    }
  }
"""


def get_kernel_op(name):
  if name == 'ApplyKernelConvOp':
    return get_kernel_fn.ApplyKernelConvOp(4)
  elif name == 'ApplyKernelSliceOp':
    return get_kernel_fn.ApplyKernelSliceOp()

  return None


class PoissonSolverCGTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      # Band preconditioner.
      ('Case00', """
          conjugate_gradient {
            halo_width: 1
            preconditioner {
              band_preconditioner {
                halo_width: 2
                coefficients: -1
                coefficients: -1
                coefficients: -1
              }
            }
          }
       """, 'band preconditioner: 1 != 2, '),
      ('Case01', """
          conjugate_gradient {
            halo_width: 2
            preconditioner {
              band_preconditioner {
                halo_width: 1
                coefficients: -1
                coefficients: -1
                coefficients: -1
              }
            }
          }
       """, 'band preconditioner: 2 != 1, '),
      ('Case02', """
          conjugate_gradient {
            halo_width: 2
            preconditioner {
              band_preconditioner {
                halo_width: 2
                coefficients: -1
                coefficients: -1
                coefficients: -1
                coefficients: -1
              }
            }
          }
       """, ('band preconditioner: 2 != 2, coefficients length is not odd as '
             'expected: len = 4, ')),
      ('Case03', """
          conjugate_gradient {
            halo_width: 1
            preconditioner {
              band_preconditioner {
                halo_width: 1
                coefficients: -1
                coefficients: -1
                coefficients: -1
                coefficients: -1
                coefficients: -1
              }
            }
          }
       """, ('band preconditioner: 1 != 1, coefficients length is not odd as '
             'expected: len = 5, halo_width < len // 2,')),
      ('Case04', """
          conjugate_gradient {
            halo_width: 2
            preconditioner {
              band_preconditioner {
                halo_width: 2
                coefficients: 0
                coefficients: 0
                coefficients: 0
              }
            }
          }
       """,
       ('band preconditioner: 2 != 2, coefficients length is not odd as '
        'expected: len = 3, halo_width < len // 2, or coefficients abs sum < '
        '1e-6.')),
  )
  def testConjugateGradientPreconditionerConfigFailure(self, cg_config, regex):
    solver_option = text_format.Parse(cg_config,
                                      poisson_solver_pb2.PoissonSolver())

    with self.assertRaisesRegex(ValueError, regex):
      poisson_solver.validate_cg_config(solver_option)

  _ARGS_TF32 = ((tf.float32, 'ApplyKernelConvOp'),)
  _ARGS_TF64 = ((tf.float64, 'ApplyKernelSliceOp'),)

  _L_XYZ = (
      # lx = ly = lz
      (2, 2, 2),
      # lx != ly != lz
      (2, 4, 8),
      # One of {lx, ly, lz} is much smaller.
      (2, 8, 8),
      (8, 2, 8),
      (8, 8, 2),
      # One of {lx, ly, lz} is much larger.
      (8, 2, 2),
      (2, 8, 2),
      (2, 2, 8),
  )

  _CONFIGS_00 = (
      _CG_CONFIG,
      _CG_CONFIG_COMPONENT_WISE_DISTANCE,
      _CG_CONFIG_COMPONENT_WISE_DISTANCE_SYMMETRIC,
      _CG_CONFIG_WITH_RPECONDITIONER_BAND_11_COMPONENT_WISE_DISTANCE,
      _CG_CONFIG_WITH_RPECONDITIONER_BAND_21_COMPONENT_WISE_DISTANCE,
      _CG_CONFIG_WITH_RPECONDITIONER_BAND_22_COMPONENT_WISE_DISTANCE %
      (0, 'true'),
      _CG_CONFIG_WITH_RPECONDITIONER_BAND_22_COMPONENT_WISE_DISTANCE %
      (0, 'false'),
  )
  _CONFIGS_01 = (
      _CG_CONFIG_WITH_RPECONDITIONER_BAND_22_COMPONENT_WISE_DISTANCE %
      (1, 'false'),)
  _CONFIGS_02 = (
      _CG_CONFIG_WITH_RPECONDITIONER_BAND_22_COMPONENT_WISE_DISTANCE %
      (2, 'false'),
  )

  @parameterized.parameters(
      # tf.float32
      list(itertools.product(_ARGS_TF32, _CONFIGS_00, _L_XYZ)) +
      # - Only guaranteed to work when there is a dominant direction.
      list(itertools.product(_ARGS_TF32, _CONFIGS_01, _L_XYZ[1:5])) +
      # - Only guaranteed to work for `taylor_expansion_order >= 2`: (2, 8, 8).
      list(itertools.product(_ARGS_TF32, _CONFIGS_02, ((2, 8, 8),))) +
      # tf.float64
      list(itertools.product(_ARGS_TF64, _CONFIGS_00[:3], _L_XYZ[:1])))
  def testConjugateGradientSingularSystemSubtractMean(self, args, cg_config,
                                                      l_xyz):
    internal_dtype, kernel_op = args
    lx, ly, lz = [l_i * np.pi for l_i in l_xyz]

    def rhs_fn(xx, yy, zz, lx, ly, lz, coord):
      """Defines the right hand side tensor."""
      del lx, ly, lz, yy, zz, coord
      return tf.zeros_like(xx)

    replicas = np.array([[[0]], [[1]]], dtype=np.int32)
    computation_shape = np.array(replicas.shape)

    solver_option = text_format.Parse(cg_config,
                                      poisson_solver_pb2.PoissonSolver())
    halo_width = solver_option.conjugate_gradient.halo_width

    nx = ny = nz = 32

    solver = poisson_solver_testutil.PoissonSolverRunner(
        get_kernel_op(kernel_op),
        rhs_fn,
        replicas,
        nx,
        ny,
        nz,
        lx,
        ly,
        lz,
        halo_width,
        solver_option,
        internal_dtype=internal_dtype)

    coordinates = util.grid_coordinates(computation_shape)
    runner = TpuRunner(computation_shape=computation_shape)
    states = [solver.init_fn_tf2(i, coordinates[i])
              for i in range(np.prod(computation_shape))]
    tpu_res = runner.run_with_replica_args(solver.step_fn_tf2, states)

    poisson_solver_solution = tpu_res[0]
    l2_norm = poisson_solver_solution[_RESIDUAL_L2_NORM]
    component_wise_distance = poisson_solver_solution[_COMPONENT_WISE_DISTANCE]
    iterations = poisson_solver_solution[_ITERATIONS]

    tf.compat.v1.logging.info(
        '(iterations, l2_norm, component_wise_distance) = (%d, %g, %g): %s.',
        iterations, l2_norm, component_wise_distance, cg_config)

    if solver_option.conjugate_gradient.HasField('component_wise_convergence'):
      self.assertGreaterEqual(2e-5, l2_norm)

      # Actual convergence criterion: component wise distance.
      self.assertGreaterEqual(0, component_wise_distance)
    else:
      # Actual convergence criterion: l2 norm.
      self.assertGreaterEqual(1e-10, l2_norm)

      self.assertAllClose(1, component_wise_distance)

    # Nondeterministic with `step_fn_zeromean_random_initial_values`.
    self.assertGreaterEqual(2000, iterations)

    tpu_res = [np.stack(tpu_res_i[_X]) for tpu_res_i in tpu_res]
    interior = slice(halo_width, -halo_width, 1)
    res = np.concatenate([
        tpu_res[0][interior, interior, interior],
        tpu_res[1][interior, interior, interior]
    ],
                         axis=0)

    nx = (nx - 2 * halo_width) * computation_shape[0]
    ny = (ny - 2 * halo_width) * computation_shape[1]
    nz = (nz - 2 * halo_width) * computation_shape[2]
    expected = np.zeros((nx, ny, nz))

    self.assertAllClose(expected, res, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
  tf.test.main()
