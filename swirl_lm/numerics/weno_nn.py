# Copyright 2023 The swirl_lm Authors.
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

"""A library for the WENO schemes based on neural network."""

import pickle
from typing import Sequence, Tuple

import numpy as np
from swirl_lm.numerics import interpolation
from swirl_lm.utility import types
import tensorflow as tf


# Reference [1]: Bezgin, D. A., Schmidt, S. J., & Adams, N. A. (2022).
# WENO3-NN: A maximum-order three-point data-driven weighted essentially
# non-oscillatory scheme. Journal of Computational Physics, 452, 110920.

# Location of the binary file storing the dictionary of the neural network:
_WENO_NN_PATH = 'swirl_lm/numerics/testdata/weno_nn_neg_03_17_2023_11_13_45.pickle'


class WenoNN:
  """Functions to perform WENO interpolations based on neural network."""

  def __init__(
      self,
      k: int = 2,
      mlp_network_path: str = _WENO_NN_PATH,
  ):
    """Reads and initializes networks weights.

    Args:
      k: The order/stencil width of the interpolation.
      mlp_network_path: Location of the binary file storing the dictionary of
        the neural network. Dictionary containing the weights and
        hyper-parameters of multilayer perceptron neural network has following
        key-value pairs:
        'n_features': Number of features of network including bias layer. It
          takes a value of 5 for WENO-3 (4 values of Delta and a bias layer)
        'n_outputs': Number of outputs of network. It takes a value of 1 for
          WENO-3. Second weight of WENO-3 is obtained by subtracting first
          weight from unity.
        'n_hidden_units': A 1D tensor listing number of neurons for each hidden
          layer. Size is number of hidden layers.
        'weights': A list of 2D tensors consisting of weights of each hidden
          layer. Weights include the bias term. Size of this list is one higher
          than number of hidden layers.
    """
    assert k == 2, (
        f'WenoNN is defined only for third order WENO with `k` = 2; current'
        f' value of `k`: {k}.'
    )
    self._k = k
    self._kernel_op = interpolation._get_weno_kernel_op(k=k)  # pylint: disable=protected-access

    with tf.io.gfile.GFile(mlp_network_path, 'rb') as f:
      self._mlp_network = pickle.load(f)
    for i in range(len(self._mlp_network['weights'])):
      if self._mlp_network['weights'][i].dtype is not types.TF_DTYPE:
        self._mlp_network['weights'][i] = tf.cast(
            self._mlp_network['weights'][i], dtype=types.TF_DTYPE
        )

  def _mlp_nn_forward_prop(
      self,
      delta: tf.Tensor,
  ) -> tf.Tensor:
    """Forward propagation of mlp_nn to estimate WENO-weights from Delta values.

    mlp_nn denotes Multi-Layer Perceptron Neural Network.
    Delta layer includes normalized amplitudes of first and second-order
    derivatives of the local input field (equations 14 and 15 of [1]).

    Args:
      delta: A 3D or a 4D tensor based on whether flow field variables are
        stored as list[tf.Tensor] or full 3D tensors respectively.
        3D shape: (nx, ny, mlp_network['n_features']).
        4D shape: (nz, nx, ny, mlp_network['n_features']).
        This includes the Delta and a bias layer.

    Returns:
      Output of the neural network is a 3D or 4D tensor based on whether flow
      field variables are stored as list[tf.Tensor] or full 3D tensors
      respectively. 3D shape: (nx, ny, n_wts), 4D shape: (nz, nx, ny, n_wts). It
      represents the WENO-weights. For WENO-3, number of weights should be unity
      (n_wts = 1). Second weight of WENO-3 is obtained by subtracting first
      weight from unity.
    """

    assert (
        len(delta.shape) in [3, 4]
    ), f'`delta` should be a 3D or 4D tensor. Current shape is {delta.shape}.'
    assert delta.shape[-1] == self._mlp_network['weights'][0].shape[0], (
        f'Length of last axis of `delta`: {delta.shape[-1]} should be equal'
        f' to the number of features of the network:'
        f' {self._mlp_network["weights"][0].shape[0]}'
    )

    n_hid = len(self._mlp_network['n_hidden_units'])
    hidden_out = tf.constant(delta)
    for i in range(n_hid):
      hidden_act = tf.einsum(
          '...ji,...kj->...ki', self._mlp_network['weights'][i], hidden_out
      )
      hidden_out = tf.nn.swish(hidden_act)
      hidden_out = tf.concat(
          [
              hidden_out,
              tf.expand_dims(tf.ones_like(hidden_out[..., 0]), axis=-1)
          ],
          axis=-1,
      )

    output_act = tf.einsum(
        '...ji,...kj->...ki', self._mlp_network['weights'][n_hid], hidden_out
    )
    return tf.nn.sigmoid(output_act)

  def _process_weno_nn_delta_layer(
      self,
      delta: Sequence[types.FlowFieldVal],
  ) -> types.FlowFieldVal:
    """Normalizes delta layer, add bias and convert it to a list of tensors.

    Args:
      delta: A list consisting of the delta layer [1].
        Each entry of the dictionary corresponds to a normalized amplitude of
        numerical derivatives. For WENO-3, there are 4 entries with keys from 0
        to 3. Each dict entry is list of length [nz] with each tensor of
        shape (nx, ny), if flow variables are stored as list[tf.Tensor].
        Otherwise, each dict entry is a 3D tensor of shape (nz, nx, ny).

    Returns:
      Output is a list of 3D tensors or a 4D tensor.
      List is of length [nz] with shape of each tensor: (nx, ny, n_delta) if
      flow variables are stored as list[tf.Tensor].
      Shape of 4D tensor: (nz, nx, ny, n_delta) if flow variables are stored as
      tf.Tensor.
      Value of n_delta is 5 for WENO-3 including the bias layer.
    """

    delta = tf.nest.map_structure(tf.abs, delta)

    # Small positive number to avoid division by zero. Set close to machine
    # precision.
    epsilon = np.finfo(types.NP_DTYPE).resolution
    max_delta_0 = tf.math.reduce_max(delta[0])
    max_delta_1 = tf.math.reduce_max(delta[1])
    max_val = tf.maximum(
        tf.maximum(max_delta_0, epsilon), tf.maximum(max_delta_1, epsilon)
    )
    if isinstance(delta[0], list):
      # Transpose the order of list of lists (get Z along outer axis and
      # features along inner axis).
      delta = [[row[i] for row in delta] for i in range(len(delta[0]))]
      delta = [tf.stack(delta_z, axis=2) for delta_z in delta]
    else:  # For flow field as 3D tensor, get features along last axis
      delta = tf.stack(delta, axis=-1)
    delta = tf.nest.map_structure(lambda delta_z: delta_z / max_val, delta)
    # Add a column of ones to account for the bias layer.
    delta = tf.nest.map_structure(
        lambda delta_z: tf.concat(  # pylint: disable=g-long-lambda
            [delta_z, tf.expand_dims(tf.ones_like(delta_z[..., 0]), axis=-1)],
            axis=-1,
        ),
        delta,
    )
    return delta

  def _calculate_weno_nn_delta_layer(
      self,
      v: types.FlowFieldVal,
      dim: str,
  ) -> Tuple[types.FlowFieldVal, types.FlowFieldVal]:
    """Calculates the delta layer: input to the neural network [1].

    Args:
      v: A list of 2D tensors or a 3D tensor to which the interpolation is
        performed.
      dim: The dimension along which the interpolation is performed.

    Returns:
      Output is a tuple containing lists of 3D tensors of length [nz] or 4D
      tensors. Each entry of the list is of shape
      (nx, ny, mlp_network['n_features']) where, mlp_network['n_features'] = 5
      for WENO-3. Shape of 4D tensor is (nz, nx, ny, mlp_network['n_features']).
      This includes the Delta and a bias layer.
    """
    kernel_fn = [None] * 4
    kernel_fn[0] = {
        'x': lambda u: self._kernel_op.apply_kernel_op_x(u, 'kdx'),
        'y': lambda u: self._kernel_op.apply_kernel_op_y(u, 'kdy'),
        'z': lambda u: self._kernel_op.apply_kernel_op_z(u, 'kdz', 'kdzsh')
    }[dim]
    kernel_fn[1] = {
        'x': lambda u: self._kernel_op.apply_kernel_op_x(u, 'kdx+'),
        'y': lambda u: self._kernel_op.apply_kernel_op_y(u, 'kdy+'),
        'z': lambda u: self._kernel_op.apply_kernel_op_z(u, 'kdz+', 'kdz+sh')
    }[dim]
    kernel_fn[2] = {
        'x': lambda u: self._kernel_op.apply_kernel_op_x(u, 'kDx'),
        'y': lambda u: self._kernel_op.apply_kernel_op_y(u, 'kDy'),
        'z': lambda u: self._kernel_op.apply_kernel_op_z(u, 'kDz', 'kDzsh')
    }[dim]
    kernel_fn[3] = {
        'x': lambda u: self._kernel_op.apply_kernel_op_x(u, 'kddx'),
        'y': lambda u: self._kernel_op.apply_kernel_op_y(u, 'kddy'),
        'z': lambda u: self._kernel_op.apply_kernel_op_z(u, 'kddz', 'kddzsh')
    }[dim]

    delta_neg = [kernel_fn[r](v) for r in range(4)]

    delta_neg = self._process_weno_nn_delta_layer(delta_neg)
    # Note: delta_pos is identical to delta_neg with features 0 and 1 swapped.
    delta_pos = tf.nest.map_structure(
        lambda delta_z: tf.stack(  # pylint: disable=g-long-lambda
            [
                delta_z[..., 1],
                delta_z[..., 0],
                delta_z[..., 2],
                delta_z[..., 3],
                delta_z[..., 4],
            ],
            axis=-1,
        ),
        delta_neg,
    )
    return delta_neg, delta_pos

  def _calculate_weno_nn_weights(
      self,
      delta_neg: types.FlowFieldVal,
      delta_pos: types.FlowFieldVal,
      c_eno: types.TF_DTYPE = 2E-4,
  ) -> Tuple[Sequence[types.FlowFieldVal], Sequence[types.FlowFieldVal]]:
    """Calculates the weights for WENO interpolation using the neural network.

    Args:
      delta_neg: Negative side (biased on the left side) of the delta layer [1].
        It is a list of 3D tensors or a 4D tensor.
        If flow variables are stored as list[tf.Tensor], its a list of
        length [nz]. Shape of each tensor in the list: (nx, ny, n_delta).
        If flow variables are stored as tf.Tensor, it a 4D tensor of shape:
        (nz, nx, ny, n_delta).
        Value of n_delta is 5 for WENO-3 including the bias layer.
      delta_pos: Positive side of the delta layer [1]. Details of shape and
        contents are same as delta_neg.
      c_eno: A cutoff threshold to ensure zero weights near discontinuities [1].

    Returns:
      A tuple of the weights for WENO interpolated values on the faces between
      the contiguous cells, with the first and second elements being the
      negative and positive weights at face i + 1/2, respectively. Index of the
      cell center is denoted by i.
    """
    weno_wt_neg = [tf.zeros_like(delta_neg)] * 2
    weno_wt_pos = [tf.zeros_like(delta_neg)] * 2
    def mlp_nn_forward_prop(delta):
      return self._mlp_nn_forward_prop(delta)[..., 0]
    weno_wt_neg[1] = tf.nest.map_structure(mlp_nn_forward_prop, delta_neg)
    weno_wt_pos[0] = tf.nest.map_structure(mlp_nn_forward_prop, delta_pos)

    calc_second_wt = lambda wt: 1.0 - wt
    weno_wt_neg[0] = tf.nest.map_structure(calc_second_wt, weno_wt_neg[1])
    weno_wt_pos[1] = tf.nest.map_structure(calc_second_wt, weno_wt_pos[0])

    # ENO layer: equation (16) of reference [1]:
    def apply_eno_layer(weno_wt):
      zero_cutoff = lambda wt: tf.where(wt < c_eno, tf.zeros_like(wt), wt)
      weno_wt = [
          tf.nest.map_structure(zero_cutoff, weno_wt_i) for weno_wt_i in weno_wt
      ]
      weno_wt_sum = tf.nest.map_structure(tf.math.add, weno_wt[0], weno_wt[1])
      weno_wt = [
          tf.nest.map_structure(tf.math.divide, weno_wt_i, weno_wt_sum)
          for weno_wt_i in weno_wt
      ]
      return weno_wt

    weno_wt_neg = apply_eno_layer(weno_wt_neg)
    weno_wt_pos = apply_eno_layer(weno_wt_pos)
    return weno_wt_neg, weno_wt_pos

  def weno_nn(
      self,
      v: types.FlowFieldVal,
      dim: str,
  ) -> Tuple[types.FlowFieldVal, types.FlowFieldVal]:
    """Performs WENO interpolation with weights estimated using a neural network.

    Args:
      v: A tensor or a list of tensor representing a cell-averaged flow field
        to which the interpolation is performed.
      dim: The dimension along with the interpolation is performed.

    Returns:
      A tuple of the interpolated values on the faces, with the first and second
      elements being the negative and positive fluxes at face i - 1/2,
      respectively.
    """
    delta_neg, delta_pos = self._calculate_weno_nn_delta_layer(v, dim)
    weno_wt_neg, weno_wt_pos = self._calculate_weno_nn_weights(
        delta_neg, delta_pos,
    )
    vr_neg, vr_pos = interpolation._reconstruct_weno_face_values(  # pylint: disable=protected-access
        v, self._kernel_op, dim=dim, k=self._k
    )
    v_neg, v_pos = interpolation._interpolate_with_weno_weights(  # pylint: disable=protected-access
        v, weno_wt_neg, weno_wt_pos, vr_neg, vr_pos, dim=dim, k=self._k
    )
    return v_neg, v_pos

