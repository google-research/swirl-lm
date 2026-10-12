# Copyright 2026 The swirl_lm Authors.
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

"""Checks that the slice and convolution finite-difference paths agree."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.utility import common_ops
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import grid_parametrization_pb2 as gpb

AXES = ('x', 'y', 'z')
# The slice path zero-pads the outermost cells, so only the interior (inside
# the widest stencil reach of 2) is compared.
MARGIN = 2
TOL = 1e-5


def _make_grid(kernel_size):
  proto = gpb.GridParametrization(
      computation_shape=gpb.CoordinateInt(dim_x=1, dim_y=1, dim_z=1),
      length=gpb.CoordinateFloat(dim_x=1.0, dim_y=1.0, dim_z=1.0),
      grid_size=gpb.CoordinateInt(dim_x=16, dim_y=16, dim_z=16),
      halo_width=2,
      dt=0.1,
      kernel_size=kernel_size,
      periodic=gpb.CoordinateBool(dim_x=False, dim_y=False, dim_z=False),
  )
  return grid_parametrization.GridParametrization(proto)


def _random_array(n):
  return np.random.default_rng(1000 + n).standard_normal(
      (n, n, n)).astype(np.float32)


def _delta(n):
  a = np.zeros((n, n, n), np.float32)
  a[n // 2, n // 2, n // 2] = 1.0
  return a


def _stencil(out, axis, grid_params):
  """Returns {d: w} with out[i] += w * a[i + d] for a delta at the middle."""
  ai = grid_params.get_axis_index(axis)
  mid = out.shape[ai] // 2
  idx = [mid, mid, mid]
  idx[ai] = slice(None)
  line = np.asarray(out, np.float64)[tuple(idx)]
  return {mid - i: float(w) for i, w in enumerate(line) if abs(w) > TOL}


def _assert_interior_close(conv, sl, axis, grid_params):
  ai = grid_params.get_axis_index(axis)
  inner = [slice(None)] * 3
  inner[ai] = slice(MARGIN, conv.shape[ai] - MARGIN)
  np.testing.assert_allclose(
      np.asarray(sl)[tuple(inner)],
      np.asarray(conv)[tuple(inner)],
      rtol=TOL,
      atol=TOL,
  )


def _assert_same_stencil(test, a, b):
  test.assertEqual(sorted(a), sorted(b))
  for d in a:
    test.assertAlmostEqual(a[d], b[d], places=5)


class GetKernelFnTest(absltest.TestCase):

  def test_builtin_slice_matches_conv(self):
    grid_params = _make_grid(4)
    conv_kernels = get_kernel_fn._convop_kernel_dict(4)
    slice_kernels = get_kernel_fn._slice_kernel_dict(None)
    self.assertEqual(set(conv_kernels), set(slice_kernels))
    arr = jnp.array(_random_array(16))
    for name in conv_kernels:
      kd = slice_kernels[name]
      for axis in AXES:
        with self.subTest(kernel=name, axis=axis):
          conv = common_ops.apply_convolutional_op(
              arr, jnp.array(conv_kernels[name]), axis, grid_params
          )
          sl = common_ops.finite_diff_with_slice(
              arr, kd['coeff'], kd['shift'], axis, grid_params
          )
          _assert_interior_close(conv, sl, axis, grid_params)

  def test_custom_slice_matches_conv(self):
    grid_params = _make_grid(4)
    custom = {
        'kf2-': ([-0.125, 0.75, 0.375], 2),
        'shift': ([0.0, 0.0, 1.0], 1),
        'shiftm': ([1.0, 0.0, 0.0], 1),
        'krc': ([-1.0, 1.0, 1.0, -1.0], 2),
    }
    op = get_kernel_fn.ApplyKernelConvOp(4, grid_params, custom)
    arr = jnp.array(_random_array(16))
    for name in custom:
      kd = op._slice_kernels[name]
      for axis in AXES:
        with self.subTest(kernel=name, axis=axis):
          conv = common_ops.apply_convolutional_op(
              arr, jnp.array(op._get_kernel(name)), axis, grid_params
          )
          sl = common_ops.finite_diff_with_slice(
              arr, kd['coeff'], kd['shift'], axis, grid_params
          )
          _assert_interior_close(conv, sl, axis, grid_params)

  def test_fallback_is_oriented(self):
    grid_params = _make_grid(4)
    custom = {'kf2-': ([-0.125, 0.75, 0.375], 2)}
    op = get_kernel_fn.ApplyKernelConvOp(4, grid_params, custom)
    names = ['kf2', 'kf2+', 'kd', 'kd+', 'k3d1+', 'kD', 'kf2-']
    for axis in AXES:
      for name in names:
        with self.subTest(kernel=name, axis=axis):
          # 18 % 4 != 0 takes the slice fallback; 20 % 4 == 0 takes conv.
          fallback = op.apply_kernel_op(jnp.array(_delta(18)), name, axis)
          conv = op.apply_kernel_op(jnp.array(_delta(20)), name, axis)
          _assert_same_stencil(
              self,
              _stencil(fallback, axis, grid_params),
              _stencil(conv, axis, grid_params),
          )


if __name__ == '__main__':
  absltest.main()
