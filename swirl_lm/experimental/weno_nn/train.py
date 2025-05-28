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

"""Training functions for WENO_NN following [1].

Refs:

[1] D. Bezgin, S.J. Schmidt and N.A. Klaus, "WENO3-NN: A maximum-order
three-point data-driven weighted essentially non-oscillatory scheme",
Journal of Computational Physics, Volume 452, Issue C, Mar 2022.
"""

from flax.training import train_state
import grain.python as pygrain
import jax
import jax.numpy as jnp
import numpy as np
from swirl_lm.experimental.weno_nn import utils
from swirl_lm.experimental.weno_nn import weno_nn


@jax.jit
def train_step(
    state: train_state.TrainState,
    batch: dict[str, jax.Array],
    alpha: float = 0.03,
    beta_d: float = 0.1,
    beta_w: float = 1e-9,
) -> tuple[train_state.TrainState, dict[str, jax.Array]]:
  """Performs one step of the training.

  We assume that the batch is given by a dictionary of the form
  batch = {"u_bar": jax.Array, "u_half_p" : jax.Array}, where:
  - batch["u_bar"].shape = (batch_size, 3), contains the local averages, i.e.,
      (u_bar_{n-1}, u_bar_n, u_bar_{n+1}), and
  - batch["u_half_p"].shape = (batch_size,), and it contains the correct value
      u at the right boundary of the n-th cell, i.e., u_{n+1/2}.

  Args:
    state: Structure encapsulating all the variables for training.
    batch: Dictionary containing the data to train.
    alpha: Exponent in Eqs. (18) and (19) in [1], which controls the deviation
      of the weights from the optimal ones as function becomes smoother.
    beta_d: Coefficients in Eq. (17) that controls the weights of the deviation
      from the optimal weights for smooth functions.
    beta_w: Coefficients in Eq. (17) controlling the Frobenious regularization.

  Returns:
    The TrainState after advancing one step in the optimization.
  """

  # Getting the upwind weights.
  _, d_k = weno_nn.upwind_weights()

  def batch_loss(params_omega):
    """Defining the loss with the current batch."""

    # Computing gamma in Eq. (22) in [1].
    gamma_vmap = jax.vmap(weno_nn.gamma)
    # shape : (batch_size,).
    gamma_batch = gamma_vmap(batch['u_bar'])
    # Computing (\gamma^s)^{\alpha} in (17)-(18) in [1].
    # shape : (batch_size,).
    gamma_alpha = jnp.power(gamma_batch, alpha)

    # Computing ℒ_r in Eq. (18) in [1].
    # Vmapping the weno interpolation.
    inter_p = jax.vmap(weno_nn.weno_interpolation_plus, in_axes=(0, None))
    # shape: (batch_size,).
    u_half_batch = inter_p(
        batch['u_bar'],
        lambda x, order: state.apply_fn({'params': params_omega}, x),
    )

    # shape : (batch_size,).
    diff_sqr = jnp.square(u_half_batch - batch['u_half_p'])
    # shape : (1,)
    loss_r = jnp.mean(gamma_alpha * diff_sqr)

    # Computing ℒ_d in Eq. (19) in [1].
    # shape: (batch_size, 2).
    omega_batch = jax.vmap(
        lambda x: state.apply_fn({'params': params_omega}, x)
    )(batch['u_bar'])
    # shape: (batch_size,).
    diff_sqr_omega = jnp.square(omega_batch[:, 0] - d_k[0]) + jnp.square(
        omega_batch[:, 1] - d_k[1]
    )
    # shape: (1,)
    loss_d = jnp.mean((1 - gamma_alpha) * diff_sqr_omega)

    # Computing the Frobenius regularization.
    flattened, _, _ = utils.flatten_params(params_omega)
    loss_w = jnp.sum(jnp.square(flattened))

    # Assembling the loss in Eq. (17) of [1].
    loss = loss_r + beta_d * loss_d + beta_w * loss_w
    return loss, {'loss_r': loss_r, 'loss_d': loss_d, 'loss_w': loss_w}

  grad_fn = jax.value_and_grad(batch_loss, has_aux=True)
  (loss, losses_dict), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)

  return state, {'loss': loss, **losses_dict}


def train(
    state: train_state.TrainState,
    train_ds: pygrain.DataLoader,
    alpha: float,
    beta_d: float,
    beta_w: float,
    aggregation_period: int = 1000,
) -> tuple[train_state.TrainState, dict[str, np.ndarray]]:
  """Trains the model.

  Args:
    state: Structure encapsulating all the variables for training.
    train_ds: Data for training.
    alpha: Exponent in Eqs. (18) and (19) in [1], which controls the deviation
      of the weights from the optimal ones as function becomes smoother.
    beta_d: Coefficient in Eq. (17) that controls the weights of the deviation
      from the optimal weights for smooth functions.
    beta_w: Coefficient in Eq. (17) controlling the Frobenious regularization.
    aggregation_period: Period (in number of steps) for printing the metrics.

  Returns:
    The TrainState after advancing one step in the optimization.
  """

  batch_metrics = []

  for ii, batch in enumerate(train_ds):
    state, metrics = train_step(
        state,
        batch,
        alpha,
        beta_d,
        beta_w,
    )

    batch_metrics.append(metrics)
    if aggregation_period and ii % aggregation_period == 0:
      print(f'Training step: {ii:d}:')
      for key, item in metrics.items():
        print(f'\t {key}: {item:.6f}')

  # Extracting the metrics in time to be plotted later.
  batch_metrics_np = jax.device_get(batch_metrics)
  metrics_np = {}
  for k in batch_metrics_np[0].keys():
    metrics_np[k] = np.array([metrics[k] for metrics in batch_metrics_np])

  return state, metrics_np
