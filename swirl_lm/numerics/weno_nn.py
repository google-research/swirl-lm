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

"""A library for the WENO schemes based on neural network."""

import functools
from typing import Any, Mapping, Sequence, Tuple

import h5py
import numpy as np
from swirl_lm.numerics import interpolation
from swirl_lm.utility import types
import tensorflow as tf


# Reference [1]: Bezgin, D. A., Schmidt, S. J., & Adams, N. A. (2022).
# WENO3-NN: A maximum-order three-point data-driven weighted essentially
# non-oscillatory scheme. Journal of Computational Physics, 452, 110920.

# Location of the binary file storing the dictionary of the neural network:
_WENO_NN_PATH = (
    'swirl_lm/numerics/testdata/xid_94741459_model_113.hdf5'
)
_EPSILON = 1e-6


def _read_group(
    group: h5py.Group, array_dtype: Any = np.float32
) -> Mapping[str, Any]:
  """Recursively reads a hdf5 group."""
  out = {}
  for key in group.keys():
    if isinstance(group[key], h5py.Group):
      out[key] = _read_group(group[key])
    elif isinstance(group[key], h5py.Dataset):
      if group[key].shape:  # pytype: disable=attribute-error
        out[key] = np.asarray(group[key], dtype=array_dtype)
      else:
        out[key] = group[key][()]
    else:
      raise ValueError(f'Unknown type for key {key}.')
  return out


def _read_all_arrays_as_dict(
    file_path: str, array_dtype: Any = np.float32
) -> Mapping[str, Any]:
  """Reads the entire contents of a file as a (possibly nested) dictionary."""
  if not tf.io.gfile.exists(file_path):
    raise FileNotFoundError(f'No data file found at {file_path}.')

  with tf.io.gfile.GFile(file_path, 'rb') as f:
    with h5py.File(f, 'r') as hf:
      return _read_group(hf, array_dtype)


class WenoNN:
  """Functions to perform WENO interpolations based on neural network.

  Attributes:
    k: The order/stencil width of the interpolation.
    mlp_network_path: Location of the binary file storing the dictionary of the
      neural network config and weights. Dictionary has structure: {'config':
      config, 'params': params}. `config`: {'features_fun': str, 'act_fun': str,
      'features': list[int]}. `params` has the following structure based on the
      standard FLAX model: {'Dense_0': {'bias': (b0,), 'kernel': (k0_0, k0_1)},
      'Dense_1': {'bias': (b1,), 'kernel': (k1_0, k1_1)}, ...}, where b0, b1,
      k0_0, ... are appropriate dimensions.
    epsilon: Small positive number to avoid division by zero. Set close to
      machine precision.
    _act_fun_out: Activation function for the last (output) layer.
    _act_fun_name: Name of the activation function.
    _feature_fun: Function operated on the input fields to generate the features
      before the forward propagation.
    _hidden_layer_neurons: List with number of neurons for the hidden layers.
    _n_hidden_layers: Number of hidden layers.
  """

  def __init__(
      self,
      k: int = 2,
      mlp_network_path: str = _WENO_NN_PATH,
      epsilon: float = _EPSILON,
  ):
    """Reads and initializes networks weights."""
    assert k == 2, (
        'WenoNN is defined only for third order WENO with `k` = 2; current'
        f' value of `k`: {k}.'
    )
    self._epsilon = epsilon
    self._k = k
    self._kernel_op = interpolation._get_weno_kernel_op(k=k)  # pylint: disable=protected-access
    self._act_fun_out = tf.nn.softmax
    self._dense_layer_key = 'Dense'
    hdf5_dict = _read_all_arrays_as_dict(mlp_network_path)
    self._feature_fun = hdf5_dict['config']['features_fun'].decode()
    self._act_fun_name = hdf5_dict['config']['act_fun'].decode()
    self._mlp_network = hdf5_dict['params']
    self._hidden_layer_neurons = hdf5_dict['config']['features'].astype(int)
    self._n_hidden_layers = len(self._hidden_layer_neurons)
    for k1 in self._mlp_network.keys():
      for k2 in self._mlp_network[k1].keys():
        if not isinstance(self._mlp_network[k1][k2], dict):
          if self._mlp_network[k1][k2].dtype is not types.TF_DTYPE:
            self._mlp_network[k1][k2] = tf.cast(
                self._mlp_network[k1][k2], dtype=types.TF_DTYPE
            )
        else:
          for k3 in self._mlp_network[k1][k2].keys():
            self._mlp_network[k1][k2][k3] = tf.cast(
                self._mlp_network[k1][k2][k3], dtype=types.TF_DTYPE
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
      delta: `delta` is a 3D tensor of shape `(nx, ny, n_delta)` if flow
        variables are stored as `list[tf.Tensor]`. `delta` is a 4D tensor of
        shape `(nz, nx, ny, n_delta)`, if flow variables are stored as 3D
        `tf.Tensor`. Value of `n_delta` is 4 for WENO-3.

    Returns:
      Output of the neural network is a 3D tensor of shape `(nx, ny, n_wts)`, if
      flow variables are stored as `list[tf.Tensor]`. Output is a 4D tensor of
      shape `(nz, nx, ny, n_wts)`, if flow variables are stored as 3D
      `tf.Tensor`. It represents the WENO-weights. For WENO-3, number of weights
      should be two (`n_wts` = 2).
    """

    assert len(delta.shape) in [
        3,
        4,
    ], f'`delta` should be a 3D or 4D tensor. Current shape is {delta.shape}.'

    n_hid = self._n_hidden_layers
    hidden_out = tf.identity(delta)
    key = self._dense_layer_key
    for i in range(n_hid):
      hidden_act = (
          tf.einsum(
              '...ji,...kj->...ki',
              self._mlp_network[f'{key}_{i}']['kernel'],
              hidden_out,
          )
          + self._mlp_network[f'{key}_{i}']['bias']
      )
      hidden_out = self._act_fun(hidden_act, i)

    output_act = (
        tf.einsum(
            '...ji,...kj->...ki',
            self._mlp_network[f'{key}_{n_hid}']['kernel'],
            hidden_out,
        )
        + self._mlp_network[f'{key}_{n_hid}']['bias']
    )
    return self._act_fun_out(output_act)

  def _act_fun(self, x: tf.Tensor, i: int) -> tf.Tensor:
    """Calculates the activation function.

    Args:
      x: Input to the activation function.
      i: Layer number.

    Returns:
      Activation function applied to the input.
    """
    if self._act_fun_name == 'swish':
      hidden_out = tf.nn.swish(x)
    elif self._act_fun_name == 'rational_act_fun':
      hidden_out = self._rational_function(
          x,
          self._mlp_network[f'RationalLayer_{i}']['p_coeffs'],
          self._mlp_network[f'RationalLayer_{i}']['q_coeffs'],
      )
    else:
      raise NotImplementedError(
          f'Activation function: {self._act_fun_name} is not supported.'
      )
    return hidden_out

  def _rational_function(
      self,
      x: types.FlowFieldVal,
      p_params: tf.Tensor,
      q_params: tf.Tensor,
  ) -> tf.Tensor:
    """Calculates the rational function.

    Args:
      x: Variable on which rational function is to be calculated.
      p_params: Coefficients of the numerator polynomial.
      q_params: Coefficients of the denominator polynomial.

    Returns:
      Rational function evaluated on `x`. The denominator values are clipped at
      `epsilon` to avoid division by zero.
    """
    p = tf.math.polyval(tf.unstack(p_params), x)
    q = tf.math.polyval(tf.unstack(q_params), x)
    sign_func = lambda q0: tf.where(
        q0 >= 0, tf.ones_like(q0), -tf.ones_like(q0)
    )
    q = tf.where(
        tf.abs(q) < self._epsilon,
        self._epsilon * sign_func(q),
        q,
    )
    return p / q

  def _eno_layer(
      self,
      weno_wt: tf.Tensor,
      c_eno: types.TF_DTYPE,
  ) -> tf.Tensor:
    """ENO layer: equation (16) of reference [1].

    Args:
      weno_wt: WENO-weights used for polynomial interpolation. The two weights
        for WENO-3 should be stacked along the last axis.
      c_eno: A cutoff threshold to ensure zero weights near discontinuities [1].

    Returns:
      WENO-weights after the application of ENO layer.
    """
    weno_wt = tf.where(weno_wt < c_eno, tf.zeros_like(weno_wt), weno_wt)
    weno_wt_sum = tf.reduce_sum(weno_wt, axis=-1, keepdims=True)
    return weno_wt / weno_wt_sum

  def _process_weno_nn_delta_layer(
      self,
      delta: Sequence[types.FlowFieldVal],
  ) -> types.FlowFieldVal:
    """Normalizes delta layer and converts it to a list of tensors.

    Args:
      delta: A list consisting of the delta layer [1]. Each entry of the list
        corresponds to a normalized amplitude of numerical derivatives. For
        WENO-3, there are 4 entries. Each entry is list of length `(nz)` with
        each tensor of shape `(nx, ny)`, if flow variables are stored as
        `list[tf.Tensor]`. Each entry is a 3D tensor of shape `(nz, nx, ny)`, if
        flow variables are stored as 3D `tf.Tensor`. This argument is expected
        to be non-negative.

    Returns:
      Output is a list of 3D tensors or a 4D tensor.
      List is of length `(nz)` with shape of each tensor: `(nx, ny, n_delta)` if
      flow variables are stored as `list[tf.Tensor]`. Shape of 4D tensor: `(nz,
      nx,
      ny, n_delta)` if flow variables are stored as 3D `tf.Tensor`. Value of
      `n_delta` is 4 for WENO-3.
    """

    if isinstance(delta[0], list):
      # Transpose the order of list of lists (get Z along outer axis and
      # features along inner axis).
      delta = [[row[i] for row in delta] for i in range(len(delta[0]))]
      delta = [tf.stack(delta_z, axis=2) for delta_z in delta]
    else:  # For flow field as 3D tensor, get features along last axis
      delta = tf.stack(delta, axis=-1)
    max_delta = tf.nest.map_structure(
        lambda delta_z: tf.expand_dims(
            tf.maximum(delta_z[..., 0], delta_z[..., 1]), axis=-1
        ),
        delta,
    )
    max_val = tf.nest.map_structure(
        lambda max_delta_z: tf.maximum(max_delta_z, self._epsilon), max_delta
    )
    delta = tf.nest.map_structure(
        lambda delta_z, max_val_z: delta_z / max_val_z, delta, max_val
    )
    return delta

  def _delta_neg_kernel_op(
      self,
      v: types.FlowFieldVal,
      dim: str,
  ) -> Sequence[types.FlowFieldVal]:
    """Perform the kernel operations to compute the neg side of delta.

    Args:
      v: A list of 2D tensors or a 3D tensor to which the interpolation is
        performed.
      dim: The dimension along which the interpolation is performed.

    Returns:
      Delta layers without the normalization [1]. Note that the output is
      non-negative.
    """
    kernel_fn = [None] * 4
    kernel_fn[0] = {
        'x': lambda u: self._kernel_op.apply_kernel_op_x(u, 'kdx'),
        'y': lambda u: self._kernel_op.apply_kernel_op_y(u, 'kdy'),
        'z': lambda u: self._kernel_op.apply_kernel_op_z(u, 'kdz', 'kdzsh'),
    }[dim]
    kernel_fn[1] = {
        'x': lambda u: self._kernel_op.apply_kernel_op_x(u, 'kdx+'),
        'y': lambda u: self._kernel_op.apply_kernel_op_y(u, 'kdy+'),
        'z': lambda u: self._kernel_op.apply_kernel_op_z(u, 'kdz+', 'kdz+sh'),
    }[dim]
    kernel_fn[2] = {
        'x': lambda u: self._kernel_op.apply_kernel_op_x(u, 'kDx'),
        'y': lambda u: self._kernel_op.apply_kernel_op_y(u, 'kDy'),
        'z': lambda u: self._kernel_op.apply_kernel_op_z(u, 'kDz', 'kDzsh'),
    }[dim]
    kernel_fn[3] = {
        'x': lambda u: self._kernel_op.apply_kernel_op_x(u, 'kddx'),
        'y': lambda u: self._kernel_op.apply_kernel_op_y(u, 'kddy'),
        'z': lambda u: self._kernel_op.apply_kernel_op_z(u, 'kddz', 'kddzsh'),
    }[dim]
    delta_neg = [kf(v) for kf in kernel_fn]
    return tf.nest.map_structure(tf.abs, delta_neg)

  def _rational_layer_single_side(
      self,
      delta: Sequence[types.FlowFieldVal],
  ) -> types.FlowFieldVal:
    """Computes the rational function for a single side.

    Args:
      delta: Delta layer without the normalization. The outer list should
        consist of features. Features consist of a fully rational layer applied
        on the un-normalized delta layer.  This argument is expected to be
        non-negative.

    Returns:
      The output of the rational functions applied on the delta layer with
      features along the last axis.
    """
    p_params = tf.unstack(
        self._mlp_network['features_fun']['UnsharedRationalLayer_0'][
            'p_params'
        ],
        axis=0,
    )
    q_params = tf.unstack(
        self._mlp_network['features_fun']['UnsharedRationalLayer_0'][
            'q_params'
        ],
        axis=0,
    )
    rat_layer = [
        self._rational_function(delta0, p_params0, q_params0)
        for delta0, p_params0, q_params0 in zip(delta, p_params, q_params)
    ]
    # If `delta[0]` is a list of tensors, `tf.math.polyval` inside
    # `self._rational_function` stacks the list of tensors along axis=0. Hence,
    # rat_layer[0], rat_layer[1] etc. are all 3D tensors irrespective of
    # whether delta[0] is a 3D tensor or list of tensors.
    # Stack features along the last axis.
    rat_layer = tf.stack(rat_layer, axis=-1)
    rat_layer_norm = tf.norm(rat_layer, axis=-1, keepdims=True)
    rat_layer_norm = tf.where(
        rat_layer_norm < self._epsilon,
        self._epsilon * tf.ones_like(rat_layer_norm),
        rat_layer_norm,
    )
    rat_layer = rat_layer / rat_layer_norm
    # Following unstacking along the `Z` axis is needed if `delta[0]` is a
    # list of 2D tensors since the list of tensors are stacked along axis=0
    # (described above).
    if isinstance(delta[0], list):
      rat_layer = tf.unstack(rat_layer, axis=0)
    return rat_layer

  def _calculate_rational_function_delta_layer(
      self,
      v: types.FlowFieldVal,
      dim: str,
  ) -> Tuple[types.FlowFieldVal, types.FlowFieldVal]:
    """Calculates the rational function delta layer.

    Args:
      v: A list of 2D tensors or a 3D tensor to which the interpolation is
        performed.
      dim: The dimension along which the interpolation is performed.

    Returns:
      The rational layer for both neg and pos sides.
    """
    delta_neg = self._delta_neg_kernel_op(v, dim)
    # Note: delta_pos is identical to delta_neg with features 0 and 1 swapped.
    delta_pos = [delta_neg[1], delta_neg[0], delta_neg[2], delta_neg[3]]
    delta_neg = self._rational_layer_single_side(delta_neg)
    delta_pos = self._rational_layer_single_side(delta_pos)
    return delta_neg, delta_pos

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
      A tuple containing the delta layers on the neg and pos sides.
      Each entry of the tuple is a list of length `(nz)` with shape of each
      tensor: `(nx, ny, n_delta)` if flow variables are stored as
      `list[tf.Tensor]`. Each entry of the tuple is a 4D tensor of shape: `(nz,
      nx, ny, n_delta)` if flow variables are stored as 3D `tf.Tensor`. Value of
      `n_delta` is 4 for WENO-3.
    """
    delta_neg = self._delta_neg_kernel_op(v, dim)

    delta_neg = self._process_weno_nn_delta_layer(delta_neg)
    # Note: delta_pos is identical to delta_neg with features 0 and 1 swapped.
    delta_pos = tf.nest.map_structure(
        lambda delta_z: tf.stack(  # pylint: disable=g-long-lambda
            [
                delta_z[..., 1],
                delta_z[..., 0],
                delta_z[..., 2],
                delta_z[..., 3],
            ],
            axis=-1,
        ),
        delta_neg,
    )
    return delta_neg, delta_pos

  def weno_nn(
      self,
      v: types.FlowFieldVal,
      dim: str,
      c_eno: types.TF_DTYPE = 2e-4,
      apply_eno_layer: bool = True,
  ) -> Tuple[types.FlowFieldVal, types.FlowFieldVal]:
    """Performs WENO interpolation with weights estimated using a neural network.

    Args:
      v: A tensor or a list of tensor representing a cell-averaged flow field to
        which the interpolation is performed.
      dim: The dimension along with the interpolation is performed.
      c_eno: A cutoff threshold to ensure zero weights near discontinuities [1].
      apply_eno_layer: ENO layer is applied if this is set true (inference
        mode). ENO layer is not used in the training mode [1].

    Returns:
      A tuple of the interpolated values on the faces, with the first and second
      elements being the negative and positive fluxes at face i - 1/2,
      respectively.
    """
    if self._feature_fun == 'delta_layer':
      delta_neg, delta_pos = self._calculate_weno_nn_delta_layer(v, dim)
    elif self._feature_fun == 'rational':
      delta_neg, delta_pos = self._calculate_rational_function_delta_layer(
          v, dim
      )
    else:
      raise NotImplementedError(
          f'feature_func: {self._feature_fun} is not supported.'
      )

    def _calculate_weno_nn_wt_single_side(
        delta: types.FlowFieldVal,
    ) -> Sequence[types.FlowFieldVal]:
      """Calculates the WENO weights on each side: `neg` and `pos`."""

      def mlp_nn_forward_prop(delta: tf.Tensor) -> tf.Tensor:
        return self._mlp_nn_forward_prop(delta)

      weno_wt = tf.nest.map_structure(mlp_nn_forward_prop, delta)
      if apply_eno_layer:
        weno_wt = tf.nest.map_structure(
            functools.partial(self._eno_layer, c_eno=c_eno), weno_wt
        )
      if isinstance(weno_wt, list):
        weno_wt = [tf.unstack(wt, axis=-1) for wt in weno_wt]
        # Transpose the order of list of lists (get WENO-weights along outer
        # axis and Z along inner axis).
        weno_wt = [[row[i] for row in weno_wt] for i in range(len(weno_wt[0]))]
      else:
        weno_wt = tf.unstack(weno_wt, axis=-1)

      return weno_wt  # pytype: disable=bad-return-type

    weno_wt_neg = _calculate_weno_nn_wt_single_side(delta_neg)
    # Invert the order of neg side of WENO-weights to match the training of the
    # network.
    weno_wt_neg = [weno_wt_neg[1], weno_wt_neg[0]]
    weno_wt_pos = _calculate_weno_nn_wt_single_side(delta_pos)

    vr_neg, vr_pos = interpolation._reconstruct_weno_face_values(  # pylint: disable=protected-access
        v, self._kernel_op, dim=dim, k=self._k
    )
    v_neg, v_pos = interpolation._interpolate_with_weno_weights(  # pylint: disable=protected-access
        v, weno_wt_neg, weno_wt_pos, vr_neg, vr_pos, dim=dim, k=self._k
    )
    return v_neg, v_pos
