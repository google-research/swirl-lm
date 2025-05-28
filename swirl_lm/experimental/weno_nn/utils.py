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

"""Utilities for handling parameters.

This file contains three utilities that are useful for transforming parameters
from trees to vectors and vice versa.
"""
from collections.abc import Callable
from typing import Any, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from swirl_lm.experimental.weno_nn import rational_networks
from swirl_lm.experimental.weno_nn import weno_nn

PyTree = Any
PyTreeDef = Any

# List of feature functions.
_FEATURES_FUNCTIONS = {
    'delta_layer': weno_nn.delta_layer,
    'rational': weno_nn.FeaturesRationalLayer(),
    'rational_descentered': weno_nn.FeaturesRationalLayerDescentered(),
}

# List of activation functions.
_ACTIVATION_FUNCTIONS = {
    'gelu': nn.gelu,
    'relu': nn.relu,
    'rational': rational_networks.RationalLayer(),
    'rational_unshared': rational_networks.UnsharedRationalLayer(),
    'selu': nn.selu,
    'swish': nn.swish,
}

# List of activation functions that are trainable and different at each layer.
_UNSHARED_ACTIVATION_FUNCTIONS = {
    'rational_act_fun': rational_networks.RationalLayer,
    'unshared_rational_act_fun': rational_networks.UnsharedRationalLayer,
}


def flat_dim(params: PyTree) -> int:
  """Computes the total number of scalar elements in a `PyTree`.

  Args:
    params: PyTree containing the parameters/scalar values.

  Returns:
    Total number of scalars within all the leaves of the PyTree.
  """
  flat_params, _ = jax.tree_util.tree_flatten(params)
  return sum([p.size for p in flat_params])


def flatten_params(
    params: PyTree, from_axis: int = 0
) -> tuple[jax.Array, list[jax.Array], PyTreeDef]:
  """Function to flatten the dictionary containing the parameters.

  Args:
    params: pytree containing the parameters of the network.
    from_axis: axis from which the flatten is applied. All dimensions before
      axis, will be kept. Thus, the shape of the leaves of the PyTree need to be
      the same.

  Returns:
    Tuple containing:
    i) Array with all the data in the leaves of the Pytree concatenated, where
       we preserve the original dimensions, and we flatten everything after
       from_axis. By default (from_axis=0) everything is flattened to a vector.
    ii) List of arrays with the shapes of the leaves in the Pytree.
    iii) PyTreeDef, containing the names and labels of the nodes in the Pytree.
  """
  flat_params, tree_def = jax.tree_util.tree_flatten(params)
  shapes = [p.shape for p in flat_params]

  flattened = []

  for p in flat_params:
    d = np.int64(p.size // (np.prod(p.shape[:from_axis])))
    flattened.append(jnp.reshape(p, p.shape[:from_axis] + (d,)))

  return jnp.concatenate(flattened, axis=-1), shapes, tree_def


def get_feature_func(
    func_name: Literal[
        'z_layer', 'rational', 'rational_descentered', 'delta_layer'
    ]
) -> Callable[[jax.Array], jax.Array] | None:
  """Returns the feature function for the given function name.

  Args:
    func_name: Name of the function.

  Returns:
    The feature function for the given function name.
  """

  if func_name in _FEATURES_FUNCTIONS:
    return _FEATURES_FUNCTIONS[func_name]
  else:
    return None


def get_act_func(
    func_name: Literal[
        'relu',
        'gelu',
        'selu',
        'rational',
        'rational_unshared',
        'swish',
        'rational_act_fun',
        'unshared_rational_act_fun',
    ]
) -> Callable[[jax.Array], jax.Array] | str | None:
  """Returns the activation function for the given function name.

  Args:
    func_name: Name of the function.

  Returns:
    The activation function for the given function name, or the string for
    defining the associated nn.Module inside the OmegaNN models.
  """

  if func_name in _ACTIVATION_FUNCTIONS:
    return _ACTIVATION_FUNCTIONS[func_name]
  elif func_name in _UNSHARED_ACTIVATION_FUNCTIONS:
    return func_name
  else:
    return None
