"""Tests for diffusion."""

import itertools
import os

import numpy as np
from swirl_lm.numerics import diffusion
from swirl_lm.numerics import numerics_pb2
from swirl_lm.utility import get_kernel_fn
import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.pyglib import gfile
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_config
from google3.research.simulation.tensorflow.fluid.models.incompressible_structured_mesh import incompressible_structured_mesh_parameters_pb2
from google3.testing.pybase import parameterized

_NP_DTYPE = np.float32
_TF_DTYPE = tf.float32


class DiffusionTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Initializes the shared variables in the test."""
    super().setUp()

    self.kernel_op = get_kernel_fn.ApplyKernelConvOp(4)

    nx, ny, nz = (16, 16, 16)
    x = np.linspace(0, 2.0 * np.pi, nx + 1)[:-1]
    y = np.linspace(0, 2.0 * np.pi, ny + 1)[:-1]
    z = np.linspace(0, 2.0 * np.pi, nz + 1)[:-1]

    self.grid_spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    # Reshape the coordinates from x-y-z to z-x-y.
    xx = np.transpose(xx, (2, 0, 1))
    yy = np.transpose(yy, (2, 0, 1))
    zz = np.transpose(zz, (2, 0, 1))

    # Initialize the dynamic viscosity.
    mu = 0.1 * np.ones((nx, ny, nz), dtype=_NP_DTYPE)
    self.mu = tf.unstack(tf.convert_to_tensor(mu, dtype=_TF_DTYPE))

    # Initialize the density.
    rho = 1.2 * np.ones((nx, ny, nz), dtype=_NP_DTYPE)
    self.rho = tf.unstack(tf.convert_to_tensor(rho, dtype=_TF_DTYPE))

    # Initialize the diffusivity.
    d = 0.1 * np.ones((nx, ny, nz), dtype=_NP_DTYPE)
    self.d = tf.unstack(tf.convert_to_tensor(d, dtype=_TF_DTYPE))

    # Initialize the velocity.
    u = np.cos(xx) * np.sin(yy) * np.sin(zz)
    v = np.sin(xx) * np.cos(yy) * np.sin(zz)
    w = np.sin(xx) * np.sin(yy) * np.cos(zz)
    self.velocity = {
        'u': tf.unstack(tf.convert_to_tensor(u, dtype=_TF_DTYPE)),
        'v': tf.unstack(tf.convert_to_tensor(v, dtype=_TF_DTYPE)),
        'w': tf.unstack(tf.convert_to_tensor(w, dtype=_TF_DTYPE)),
    }

    # Initializes a scalar. Assume it takes the same profiles as u.
    self.phi = tf.unstack(tf.convert_to_tensor(u, dtype=_TF_DTYPE))

    # Initialize the expected/analytical solution to the velocity diffusion.
    self.velocity_diffusion_analytical = {
        'u': (
            np.zeros((nx, ny, nz), dtype=_NP_DTYPE),
            -2.0 * mu * u,
            -2.0 * mu * u,
        ),
        'v': (
            -2.0 * mu * v,
            np.zeros((nx, ny, nz), dtype=_NP_DTYPE),
            -2.0 * mu * v,
        ),
        'w': (
            -2.0 * mu * w,
            -2.0 * mu * w,
            np.zeros((nx, ny, nz), dtype=_NP_DTYPE),
        ),
    }

    # Initialize the expected/analytical solution to the scalar diffusion.
    self.scalar_diffusion_analytical = [-rho * d * u,] * 3

    # Initialize the analytical solution to the shear stress.
    self.tau = {
        'xx': np.zeros_like(u),
        'xy': 2.0 * mu * np.cos(xx) * np.cos(yy) * np.sin(zz),
        'xz': 2.0 * mu * np.cos(xx) * np.sin(yy) * np.cos(zz),
        'yx': 2.0 * mu * np.cos(xx) * np.cos(yy) * np.sin(zz),
        'yy': np.zeros_like(v),
        'yz': 2.0 * mu * np.sin(xx) * np.cos(yy) * np.cos(zz),
        'zx': 2.0 * mu * np.cos(xx) * np.sin(yy) * np.cos(zz),
        'zy': 2.0 * mu * np.sin(xx) * np.cos(yy) * np.cos(zz),
        'zz': np.zeros_like(w),
    }

  def dump_result(self, prefix, data):
    """Writes result to file."""
    filename = os.path.join(
        os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), '{}.npy'.format(prefix))

    with gfile.Open(filename, 'wb') as f:
      np.save(f, data)

  def _set_up_params(self, pbtxt=''):
    """Creates a simulation configuration handler."""
    proto = text_format.Parse(
        pbtxt,
        incompressible_structured_mesh_parameters_pb2
        .IncompressibleNavierStokesParameters())

    return (incompressible_structured_mesh_config
            .IncompressibleNavierStokesParameters(proto))

  @parameterized.named_parameters(
      ('Central5', numerics_pb2.DIFFUSION_SCHEME_CENTRAL_5),
      ('Central3', numerics_pb2.DIFFUSION_SCHEME_CENTRAL_3),
      ('Stencil3', numerics_pb2.DIFFUSION_SCHEME_STENCIL_3))
  def testDiffusionMomentumProvidesCorrectDiffusion(self, scheme):
    """Checks if the diffusion term for the momentum equation is correct."""
    params = self._set_up_params()

    diffusion_fn = diffusion.diffusion_momentum(params)

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])

    result = self.evaluate(
        diffusion_fn(self.kernel_op, replica_id, replicas, scheme, self.mu,
                     self.grid_spacing, self.velocity))

    for k in ('u', 'v', 'w'):
      for dim in range(3):
        with self.subTest(name='DiffusionOf{}In{}'.format(k, dim)):
          self.assertAllClose(
              self.velocity_diffusion_analytical[k][dim][2:-2, 2:-2, 2:-2],
              np.array(result[k][dim])[2:-2, 2:-2, 2:-2],
              rtol=1e-2,
              atol=1e-2)

  @parameterized.named_parameters(('Dim0', 0), ('Dim1', 1), ('Dim2', 2))
  def testDiffusionMomentumProvidesCorrectDiffusionWithMOS(self, dim):
    """Checks if diffusion term for scalar equation is correct with MOS."""
    pbtxt = (R'diffusion_scheme: DIFFUSION_SCHEME_CENTRAL_3 '
             R'boundary_models { '
             R'  most { '
             R'    t_s: 265.0 '
             R'    z_0: 0.1 '
             R'    beta_m: 5.0 '
             R'    beta_h: 6.0 '
             R'    gamma_m: 15.0 '
             R'    gamma_h: 15.0 '
             R'  } '
             R'} ')
    if dim == 0:
      pbtxt += (R'gravity_direction { '
                R'  dim_0: -1.0 dim_1: 0.0 dim_2: 0.0 '
                R'} ')
    elif dim == 1:
      pbtxt += (R'gravity_direction { '
                R'  dim_0: 0.0 dim_1: -1.0 dim_2: 0.0 '
                R'} ')
    else:  # dim == 2
      pbtxt += (R'gravity_direction { '
                R'  dim_0: 0.0 dim_1: 0.0 dim_2: -1.0 '
                R'} ')

    params = self._set_up_params(pbtxt)
    params.cx = 1
    params.cy = 1
    params.cz = 1
    params.nx = 8
    params.ny = 8
    params.nz = 8
    params.lx = 6.0
    params.ly = 6.0
    params.lz = 6.0
    params.halo_width = 2

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])

    velocity_keys = ('u', 'v', 'w')
    tangential_dims = [0, 1, 2]
    del tangential_dims[dim]

    velocity = {
        velocity_keys[tangential_dims[0]]:
            tf.unstack(7.5 * tf.ones((8, 8, 8), dtype=tf.float32)),
        velocity_keys[dim]:
            tf.unstack(tf.zeros((8, 8, 8), dtype=tf.float32)),
        velocity_keys[tangential_dims[1]]:
            tf.unstack(-5.0 * tf.ones((8, 8, 8), dtype=tf.float32)),
    }
    helper_vars = {
        'theta':
            tf.unstack(300.0 * tf.ones((8, 8, 8), dtype=tf.float32)),
    }
    mu = tf.unstack(0.1 * tf.ones((8, 8, 8), dtype=tf.float32))

    diffusion_fn = diffusion.diffusion_momentum(params)

    result = self.evaluate(
        diffusion_fn(
            self.kernel_op,
            replica_id,
            replicas,
            numerics_pb2.DIFFUSION_SCHEME_CENTRAL_3,
            mu, (params.dx, params.dy, params.dz),
            velocity,
            helper_variables=helper_vars))

    for i in range(3):
      expected = [np.zeros((6, 6, 6), dtype=np.float32) for _ in range(3)]

      if i == dim:
        tangential_velocity = ['u', 'v', 'w']
        del tangential_velocity[dim]

        if dim == 0:
          expected[tangential_dims[0]][:, 0, :] = 0.883276
          expected[tangential_dims[0]][:, 1, :] = -0.883276
        elif dim == 1:
          expected[tangential_dims[0]][..., 0] = 0.883276
          expected[tangential_dims[0]][..., 1] = -0.883276
        else:  # dim == 2
          expected[tangential_dims[0]][0, ...] = 0.883276
          expected[tangential_dims[0]][1, ...] = -0.883276

        if dim == 0:
          expected[tangential_dims[1]][:, 0, :] = -0.588851
          expected[tangential_dims[1]][:, 1, :] = 0.588851
        elif dim == 1:
          expected[tangential_dims[1]][..., 0] = -0.588851
          expected[tangential_dims[1]][..., 1] = 0.588851
        else:  # dim == 2
          expected[tangential_dims[1]][0, ...] = -0.588851
          expected[tangential_dims[1]][1, ...] = 0.588851

      for j in range(3):
        with self.subTest(name=f'Diff{velocity_keys[j]}{i}'):
          self.assertAllClose(
              expected[j],
              np.array(result[velocity_keys[j]][i])[1:-1, 1:-1, 1:-1])

  def testDiffusionScalarProvidesCorrectDiffusion(self):
    """Checks if diffusion term for scalar equation is correct."""
    params = self._set_up_params()

    diffusion_fn = diffusion.diffusion_scalar(params)

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])

    result = self.evaluate(
        diffusion_fn(self.kernel_op, replica_id, replicas, self.velocity['u'],
                     self.rho, self.d, self.grid_spacing))

    for dim in range(3):
      with self.subTest(name='Dim{}'.format(dim)):
        self.assertAllClose(
            self.scalar_diffusion_analytical[dim][1:-1, 1:-1, 1:-1],
            np.array(result[dim])[1:-1, 1:-1, 1:-1],
            rtol=1e-2,
            atol=1e-2)

  @parameterized.named_parameters(('Dim0', 0), ('Dim1', 1), ('Dim2', 2))
  def testDiffusionScalarProvidesCorrectDiffusionWithMOS(self, dim):
    """Checks if diffusion term for scalar equation is correct with MOS."""
    pbtxt = (R'diffusion_scheme: DIFFUSION_SCHEME_CENTRAL_3 '
             R'boundary_models { '
             R'  most { '
             R'    t_s: 265.0 '
             R'    z_0: 0.1 '
             R'    beta_m: 5.0 '
             R'    beta_h: 6.0 '
             R'    gamma_m: 15.0 '
             R'    gamma_h: 15.0 '
             R'  } '
             R'} ')
    if dim == 0:
      pbtxt += (R'gravity_direction { '
                R'  dim_0: -1.0 dim_1: 0.0 dim_2: 0.0 '
                R'} ')
    elif dim == 1:
      pbtxt += (R'gravity_direction { '
                R'  dim_0: 0.0 dim_1: -1.0 dim_2: 0.0 '
                R'} ')
    else:  # dim == 2
      pbtxt += (R'gravity_direction { '
                R'  dim_0: 0.0 dim_1: 0.0 dim_2: -1.0 '
                R'} ')

    params = self._set_up_params(pbtxt)
    params.cx = 1
    params.cy = 1
    params.cz = 1
    params.nx = 8
    params.ny = 8
    params.nz = 8
    params.lx = 6.0
    params.ly = 6.0
    params.lz = 6.0
    params.halo_width = 2

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])

    velocity_keys = ('u', 'v', 'w')
    tangential_dims = [0, 1, 2]
    del tangential_dims[dim]

    helper_vars = {
        velocity_keys[tangential_dims[0]]:
            tf.unstack(7.5 * tf.ones((8, 8, 8), dtype=tf.float32)),
        velocity_keys[dim]:
            tf.unstack(tf.zeros((8, 8, 8), dtype=tf.float32)),
        velocity_keys[tangential_dims[1]]:
            tf.unstack(-5.0 * tf.ones((8, 8, 8), dtype=tf.float32)),
        'theta':
            tf.unstack(300.0 * tf.ones((8, 8, 8), dtype=tf.float32)),
    }
    rho = tf.unstack(1.2 * tf.ones((8, 8, 8), dtype=tf.float32))
    d = tf.unstack(0.1 * tf.ones((8, 8, 8), dtype=tf.float32))

    diffusion_fn = diffusion.diffusion_scalar(params)

    result = self.evaluate(
        diffusion_fn(self.kernel_op, replica_id, replicas, helper_vars['theta'],
                     rho, d, (params.dx, params.dy, params.dz), 'theta',
                     helper_vars))

    for i in range(3):
      with self.subTest(name='Dim{}'.format(i)):
        expected = np.zeros((6, 6, 6), dtype=np.float32)
        if i == dim:
          if dim == 0:
            expected[:, 0, :] = 0.005672
            expected[:, 1, :] = -0.005672
          elif dim == 1:
            expected[..., 0] = 0.005672
            expected[..., 1] = -0.005672
          else:  # dim == 2
            expected[0, ...] = 0.005672
            expected[1, ...] = -0.005672

        self.assertAllClose(expected, np.array(result[i])[1:-1, 1:-1, 1:-1])

  _DIMS = (0, 1, 2)
  _FACES = (0, 1)

  @parameterized.parameters(*itertools.product(_DIMS, _FACES))
  def testDiffusionScalarProvidesCorrectDiffusionWithPrescribedFlux(
      self, dim, face):
    """Checks if scalar diffusion term is correct with prescribed flux."""
    pbtxt = (R'scalars { '
             R'  name: "phi" '
             R'  diffusive_flux { ')
    pbtxt += f'dim: {dim} face: {face} value: 1.0'
    pbtxt += R'}} '

    params = self._set_up_params(pbtxt)
    params.cx = 1
    params.cy = 1
    params.cz = 1
    params.nx = 8
    params.ny = 8
    params.nz = 8
    params.lx = 6.0
    params.ly = 6.0
    params.lz = 6.0
    params.halo_width = 2

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])

    phi = tf.unstack(tf.zeros((9, 8, 8), dtype=tf.float32))
    rho = tf.unstack(1.2 * tf.ones((8, 8, 8), dtype=tf.float32))
    d = tf.unstack(0.1 * tf.ones((8, 8, 8), dtype=tf.float32))

    diffusion_fn = diffusion.diffusion_scalar(params)

    result = self.evaluate(
        diffusion_fn(self.kernel_op, replica_id, replicas, phi, rho, d,
                     (params.dx, params.dy, params.dz), 'phi'))

    for i in range(3):
      with self.subTest(name='Dim{}'.format(i)):
        expected = np.zeros((6, 6, 6), dtype=np.float32)
        if i == dim:
          if dim == 0:
            if face == 0:
              expected[:, 0, :] = 0.5
              expected[:, 1, :] = -0.5
            else:  # face == 1
              expected[:, -3, :] = 0.5
              expected[:, -2, :] = -0.5
          elif dim == 1:
            if face == 0:
              expected[..., 0] = 0.5
              expected[..., 1] = -0.5
            else:  # face == 1
              expected[..., -3] = 0.5
              expected[..., -2] = -0.5
          else:  # dim == 2
            if face == 0:
              expected[0, ...] = 0.5
              expected[1, ...] = -0.5
            else:  # face == 1
              expected[-3, ...] = 0.5
              expected[-2, ...] = -0.5

        self.assertAllClose(expected, np.array(result[i])[1:-1, 1:-1, 1:-1])


if __name__ == '__main__':
  tf.test.main()
