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

r"""Demo for running Taylor-Green Vortex Flow with Swirl-LM on cloud TPUs.

Example command line:

  python3 main.py --cx=2 --cy=2 --cz=8 \
    --data_dump_prefix=gs://<GCS_DIR>/data/tgv \
    --data_load_prefix=gs://<GCS_DIR>/data/tgv \
    --config_filepath=tgv_3d.textpb \
    --num_steps=2000 --nx=128 --ny=128 --nz=6 --kernel_size=16 --halo_width=2 \
    --lx=6.28 --ly=6.28 --lz=6.28 --dt=2e-3 \
    --u_mag=1.0 --p_ref=0.0 --rho_ref=1.0 --target=<TPU> \
    --output_fn_template=gs://<GCS_DIR>/output/tgv_{var}.png
"""
import functools

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
from swirl_lm.base import driver
from swirl_lm.base import initializer
from swirl_lm.base import parameters
from swirl_lm.utility import tpu_util
import tensorflow as tf

FLAGS = flags.FLAGS

_U_MAG = flags.DEFINE_float(
    'u_mag',
    1.0,
    'The magnitude of the velocity component in dim 0.',
    allow_override=True)
_P_REF = flags.DEFINE_float(
    'p_ref', 0.0,
    'The reference pressure used in pressure-induced Taylor-Green vortex.')
_RHO_REF = flags.DEFINE_float(
    'rho_ref', 1.0,
    'The reference density used in pressure-induced Taylor-Green vortex.')
_OUTPUT_FN_TEMPLATE = flags.DEFINE_string(
    'output_fn_template', 'tgv_{var}.png',
    'Output image filename template - should contain {var} as a substring.')


def taylor_green_vortices(config, v0, p0, rho0, replica_id, coordinates):
  """Initialize the u, v, w, and p field in each TPU replica.

  The velocity and pressure fields are initialized following the reference:

  J. R. DeBonis, Solutions of the Taylor-Green vortex problem using
  high-resolution explicit finite difference methods, 51st AIAA Aerospace
  Sciences Meeting including the New Horizons Forum and Aerospace Exposition,
  2013.

  Args:
    config: SwirlLMParameters for running the solver.
    v0: The magnitude of the velocity component in dim 0.
    p0: The reference pressure.
    rho0: The reference density.
    replica_id: The ID number of the replica.
    coordinates: A tuple that specifies the replica's grid coordinates in
      physical space.

  Returns:
    A dictionary of states and values that are stored as string and 3D tensor
    pairs.
  """

  def get_vortices(state_key):
    """Generates the vortex field for each flow variable."""

    def get_u(
        xx,
        yy,
        zz,
        lx,
        ly,
        lz,
        coord,
    ):
      """Generates the velocity component in dim 0.

      Args:
        xx: The sub-mesh in dimension 0 in the present replica.
        yy: The sub-mesh in dimension 1 in the present replica.
        zz: The sub-mesh in dimension 2 in the present replica.
        lx: Length in dimension 0.
        ly: Length in dimension 1.
        lz: Length in dimension 2.
        coord: The coordinate of the local core.

      Returns:
        The 3D velocity field in dimension 0 in the present replica.
      """
      del coord
      x_corr = config.dx / (lx + config.dx) * 2.0 * np.pi
      y_corr = config.dy / (ly + config.dy) * 2.0 * np.pi
      z_corr = config.dz / (lz + config.dz) * 2.0 * np.pi
      return v0 * tf.math.sin((2.0 * np.pi - x_corr) * xx / lx) * tf.math.cos(
          (2.0 * np.pi - y_corr) * yy / ly) * tf.math.cos(
              (2.0 * np.pi - z_corr) * zz / lz)

    def get_v(
        xx,
        yy,
        zz,
        lx,
        ly,
        lz,
        coord,
    ):
      """Generates the velocity component in dim 1.

      Args:
        xx: The sub-mesh in dimension 0 in the present replica.
        yy: The sub-mesh in dimension 1 in the present replica.
        zz: The sub-mesh in dimension 2 in the present replica.
        lx: Length in dimension 0.
        ly: Length in dimension 1.
        lz: Length in dimension 2.
        coord: The coordinate of the local core.

      Returns:
        The 3D velocity field in dimension 1 in the present replica.
      """
      del coord
      x_corr = config.dx / (lx + config.dx) * 2.0 * np.pi
      y_corr = config.dy / (ly + config.dy) * 2.0 * np.pi
      z_corr = config.dz / (lz + config.dz) * 2.0 * np.pi
      return -v0 * tf.math.cos((2.0 * np.pi - x_corr) * xx / lx) * tf.math.sin(
          (2.0 * np.pi - y_corr) * yy / ly) * tf.math.cos(
              (2.0 * np.pi - z_corr) * zz / lz)

    def get_w(
        xx,
        yy,
        zz,
        lx,
        ly,
        lz,
        coord,
    ):
      """Generates the velocity component in dim 2.

      Args:
        xx: The sub-mesh in dimension 0 in the present replica.
        yy: Not used.
        zz: Not used.
        lx: Not used.
        ly: Not used.
        lz: Not used.
        coord: The coordinate of the local core.

      Returns:
        The 3D velocity field in dimension 2 in the present replica.
      """
      del yy, zz, lx, ly, lz, coord
      return tf.zeros_like(xx, dtype=tf.float32)

    def get_p(
        xx,
        yy,
        zz,
        lx,
        ly,
        lz,
        coord,
    ):
      """Generates the pressure field.

      Args:
        xx: The sub-mesh in dimension 0 in the present replica.
        yy: The sub-mesh in dimension 1 in the present replica.
        zz: The sub-mesh in dimension 2 in the present replica.
        lx: Length in dimension 0.
        ly: Length in dimension 1.
        lz: Length in dimension 2.
        coord: The coordinate of the local core.

      Returns:
        The 3D pressure field in the present replica.
      """
      del coord
      x_corr = config.dx / (lx + config.dx) * 2.0 * np.pi
      y_corr = config.dy / (ly + config.dy) * 2.0 * np.pi
      z_corr = config.dz / (lz + config.dz) * 2.0 * np.pi
      return p0 + rho0 * v0**2 / 16.0 * (
          (tf.math.cos(2.0 * (2.0 * np.pi - z_corr) * zz / lz) + 2.) *
          (tf.math.cos(2.0 * (2.0 * np.pi - x_corr) * xx / lx) +
           tf.math.cos(2.0 * (2.0 * np.pi - y_corr) * yy / ly)))

    if state_key == 'u':
      return get_u
    elif state_key == 'v':
      return get_v
    elif state_key == 'w':
      return get_w
    elif state_key == 'p':
      return get_p
    else:
      raise ValueError(
          'State key must be one of u, v, w, p. {} is given.'.format(state_key))

  output = {'replica_id': replica_id}

  for key in ['u', 'v', 'w', 'p']:
    output.update({
        key:
            initializer.partial_mesh_for_core(
                config, coordinates, get_vortices(key))
    })

  if config.solver_procedure == parameters.SolverProcedure.VARIABLE_DENSITY:
    output.update({'rho': tf.ones_like(output['u'], dtype=tf.float32)})

  return output


def merge_result(values, var, coordinates, halo_width):
  """Merges results from multiple TPU replicas following the topology."""
  if len(values) != len(coordinates):
    raise ValueError(
        f'The length of `value` and `coordinates` must equal. Now `values` '
        f'has length {len(values)}, but `coordinates` has length '
        f'{len(coordinates)}.')

  # The results are oriented in order z-x-y.
  nz, nx, ny = values[0][var].shape
  nz_0, nx_0, ny_0 = [n - 2 * halo_width for n in (nz, nx, ny)]

  # The topology is oriented in order x-y-z.
  cx, cy, cz = np.array(np.max(coordinates, axis=0)) + 1

  # Compute the total size without ghost cells/halos.
  shape = [n * c for n, c in zip([nz_0, nx_0, ny_0], [cz, cx, cy])]

  result = np.empty(shape, dtype=np.float32)

  for replica in range(len(coordinates)):
    s = np.roll(
        [c * n for c, n in zip(coordinates[replica], (nx_0, ny_0, nz_0))],
        shift=1)
    e = [s_i + n for s_i, n in zip(s, (nz_0, nx_0, ny_0))]
    result[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = (
        values[replica][var][halo_width:nz_0 + halo_width,
                             halo_width:nx_0 + halo_width,
                             halo_width:ny_0 + halo_width])

  return result


def contour_plot(result, lx, ly, output):
  """Saves contour-plot of the middle slice of 3d data in result in output."""
  nz, ny, nx = result.shape

  x = np.linspace(0.0, lx, nx)
  y = np.linspace(0.0, ly, ny)

  fig, ax = plt.subplots(figsize=(18, 6))
  c = ax.contourf(x, y, result[nz // 2, ...].transpose(), cmap='jet', levels=21)
  fig.colorbar(c)
  ax.axis('equal')

  with tf.io.gfile.GFile(output, 'wb') as f:
    fig.savefig(f)


def main(args):
  del args

  # Prepares the simulation configuration.
  params = parameters.params_from_config_file_flag()
  init_fn = functools.partial(taylor_green_vortices, params, _U_MAG.value,
                              _P_REF.value, _RHO_REF.value)

  state = driver.solver(init_fn, params)

  # Post-processing.
  computation_shape = np.array([params.cx, params.cy, params.cz])
  logical_coordinates = tpu_util.grid_coordinates(computation_shape).tolist()
  sim_vars = ['u', 'v', 'w', 'p', 'rho']
  for var in sim_vars:
    result = merge_result(state, var, logical_coordinates, FLAGS.halo_width)
    contour_plot(result, FLAGS.lx, FLAGS.ly,
                 _OUTPUT_FN_TEMPLATE.value.format(var=var))


if __name__ == '__main__':
  app.run(main)
