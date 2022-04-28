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
"""Util class for tests of google3.research.simulation.tensorflow.fluid.framework.poisson_solver."""

import matplotlib.pyplot as plt
import numpy as np
from swirl_lm.linalg import multigrid_utils
from swirl_lm.linalg import poisson_solver
from swirl_lm.utility import grid_parametrization
import tensorflow as tf

from google3.pyglib import gfile
from google3.research.simulation.tensorflow.fluid.framework import initializer

_MULTIGRID_ADDITIONAL_STATES = ('coordinates', 'ps', 'rs')


def _plot_surface(ax, x, y, z, title, x_min_max, y_min_max, z_min_max):
  xx, yy = np.meshgrid(x, y, indexing='ij')
  ax.plot_surface(xx, yy, z, rstride=1, cstride=1,
                  cmap=plt.get_cmap('coolwarm'), linewidth=0, antialiased=False)
  ax.set_xlim(*x_min_max)
  ax.set_ylim(*y_min_max)
  ax.set_zlim(*z_min_max)
  ax.view_init(30, 30)
  ax.set_title(title + f' ({z.min():.3g}, {z.max():.3g})', fontsize=7,
               wrap=True)


def _two_plots_side_by_side(x, y, z1, z2, t1, t2, main_title, subtitle, f_name):
  """Create a plot with 2D slices side by side."""
  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3),
                                 subplot_kw={'projection': '3d'})
  x_min_max = x.min(), x.max()
  y_min_max = y.min(), y.max()
  z_min_max = min(z1.min(), z2.min()), max(z1.max(), z2.max())
  _plot_surface(ax1, x, y, z1, t1, x_min_max, y_min_max, z_min_max)
  _plot_surface(ax2, x, y, z2, t2, x_min_max, y_min_max, z_min_max)
  plt.suptitle(main_title, fontsize=9, wrap=True)
  fig.text(0.5, 0.1, subtitle, ha='center', fontsize=7)
  with gfile.GFile(f_name, 'wb') as f:
    plt.savefig(f)


def create_2d_slice_pngs(expected, actual, x, y, z, params, write_dir,
                         solver_name, l2_norm, slice_divisor=2,
                         additional_title='', additional_file_name=''):
  """Create pngs of 2D slices of 3D fields, for expected and actual."""
  additional_title += ', ' if additional_title else ''
  additional_file_name += '_' if additional_file_name else ''
  for i, axis in enumerate(('x', 'y', 'z')):
    axes = [x, y, z]
    del axes[i]
    l2 = '' if l2_norm is None else f', L2={l2_norm:.3g}'
    index = expected.shape[i] // slice_divisor
    slices = [slice(None)] * 3
    slices[i] = slice(index, index + 1)
    expected_2d = np.squeeze(expected[tuple(slices)])
    actual_2d = np.squeeze(actual[tuple(slices)])
    expected_title, actual_title = 'expected', 'actual'
    main_title = (f'{solver_name}: {additional_title}grid=({params.nx}, '
                  f'{params.ny}, {params.nz}), TPU layout=({params.cx}, '
                  f'{params.cy}, {params.cz}), {axis}-index={index}, '
                  f'halo={params.halo_width}{l2}')
    diffs = expected_2d - actual_2d
    subtitle = f'diffs in ({diffs.min():.3g}, {diffs.max():.3g})'
    short_solver_name = {'Fast Diagonalization': 'fd',
                         'Multigrid': 'mg',
                         'Jacobi': 'j',
                         'Conjugate Gradient': 'cg'}[solver_name]
    file_name = (f'{short_solver_name}_{params.cx}{params.cy}{params.cz}_'
                 f'{params.nx}x{params.ny}x{params.nz}_halo{params.halo_width}_'
                 f'{additional_file_name}{axis}-slice.png')
    _two_plots_side_by_side(axes[0], axes[1], expected_2d, actual_2d,
                            expected_title, actual_title, main_title, subtitle,
                            f'{write_dir}/{file_name}')


class PoissonSolverRunner(object):
  """Util class to set up Poisson solver in tests."""

  def __init__(
      self,
      kernel_op,
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
      internal_dtype=tf.float32,
  ):
    self._kernel_op = kernel_op
    self._rhs_fn = rhs_fn
    self._halo_width = halo_width
    self._solver_option = solver_option
    self._internal_dtype = internal_dtype
    self.is_multigrid = solver_option.HasField('multigrid')

    self._computation_shape = np.array(replicas.shape)

    self._params = (grid_parametrization.GridParametrization.
                    create_from_grid_lengths_and_etc(
                        grid_lengths=(lx, ly, lz),
                        computation_shape=self._computation_shape,
                        subgrid_shape=(nx, ny, nz),
                        halo_width=halo_width))

    self._solver = poisson_solver.poisson_solver_factory(
        self._params, self._kernel_op, self._solver_option)

    self._multigrid_init_fn = (
        multigrid_utils.get_init_fn_from_value_fn_for_homogeneous_bcs(
            rhs_fn, self._params))

  def _get_additional_states(self, state):
    """Maybe copy some fields from `state` to `additional_states`."""
    additional_states = {}
    for key in _MULTIGRID_ADDITIONAL_STATES:
      if key in state:
        additional_states[key] = state[key]

    return additional_states

  def init_fn(self, replica_id, coordinates):
    """Initializes the state for the fluid framework."""
    return {
        'replica_id':
            replica_id,
        'rhs':
            initializer.partial_mesh_for_core(self._params, coordinates,
                                              self._rhs_fn)
    }

  def step_fn(self, inputs, keyed_queue_elements, state, replicas):
    """Solves the Poisson equation."""
    del inputs, keyed_queue_elements

    replica_id = state['replica_id']

    rhs = tf.unstack(state['rhs'], axis=0)
    p0 = [tf.zeros_like(rhs_i) for rhs_i in rhs]

    return (self._solver.solve(
        replica_id, replicas, rhs, p0,
        internal_dtype=self._internal_dtype), state)

  def init_fn_tf2(self, replica_id, coordinates):
    """Initializes the state (to be used with TpuRunner)."""
    if self.is_multigrid:
      return self._multigrid_init_fn(replica_id, coordinates)
    else:
      state = {
          'rhs': initializer.partial_mesh_for_core(
              self._params, coordinates, self._rhs_fn)
          }
      state['p'] = tf.zeros_like(state['rhs'])
      return state

  def step_fn_tf2(self, state, replicas, replica_id):
    """Computes solution to the Poisson equation (to be used with TpuRunner)."""
    if self.is_multigrid:
      rhs = tf.unstack(state['b_minus_a_xb'])
      p = tf.unstack(state['x0'])
      additional_states = self._get_additional_states(state)
    else:
      rhs = tf.unstack(state['rhs'])
      p = tf.unstack(state['p'])
      additional_states = {}

    return self._solver.solve(
        replica_id, replicas, rhs, p,
        internal_dtype=self._internal_dtype,
        additional_states=additional_states)

  def step_fn_zeromean_random_initial_values(
      self, inputs, keyed_queue_elements, state, replicas):
    """Solves the Poisson equation with zero-mean random initial values."""
    del inputs, keyed_queue_elements

    replica_id = state['replica_id']

    rhs = tf.unstack(state['rhs'], axis=0)

    p0 = []
    for rhs_i in rhs:
      p0_core = tf.random.normal(rhs_i.shape, mean=0.0, stddev=1.0)
      mean_value = tf.math.reduce_mean(p0_core)
      p0.append(p0_core - mean_value)

    return (self._solver.solve(
        replica_id, replicas, rhs, p0,
        internal_dtype=self._internal_dtype), state)

  @property
  def params(self) -> grid_parametrization.GridParametrization:
    """The grid parametrization instance."""
    return self._params
