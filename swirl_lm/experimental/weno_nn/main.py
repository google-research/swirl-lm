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

r"""The main entry point for running training loops."""

import json
from os import path as osp

from absl import app
from absl import flags
from flax.training import train_state
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import optax
from orbax import checkpoint
from swirl_lm.experimental.weno_nn import data_utils
from swirl_lm.experimental.weno_nn import train
from swirl_lm.experimental.weno_nn import utils
from swirl_lm.experimental.weno_nn import weno_nn
import tensorflow as tf

jax.config.update("jax_enable_x64", True)

_WORKDIR = flags.DEFINE_string(
    "workdir", None, "Directory to store model data."
)

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  config = _CONFIG.value
  # Dump config as json to workdir.
  workdir = _WORKDIR.value
  if not tf.io.gfile.exists(workdir):
    tf.io.gfile.makedirs(workdir)
  # Only 0-th process should write the json file to disk, in order to avoid
  # race conditions.
  if jax.process_index() == 0:
    with tf.io.gfile.GFile(
        name=osp.join(workdir, "config.json"), mode="w"
    ) as f:
      conf_json = config.to_json_best_effort()
      if isinstance(conf_json, str):  # Sometimes `.to_json()` returns string
        conf_json = json.loads(conf_json)
      json.dump(conf_json, f)
  tf.config.experimental.set_visible_devices([], "GPU")

  rng = jax.random.PRNGKey(config.seed)

  _, init_rng = jax.random.split(rng)

  # Defining the data loader.
  data_loader_train = data_utils.create_loader_from_pickle(
      batch_size=config.batch_size,
      dataset_paths=config.dataset_paths,
      num_epochs=1,
      seed=config.seed,
      worker_count=0,
  )

  # Building the network.
  features_fun = utils.get_feature_func(config.features_fun)
  act_fun = utils.get_act_func(config.act_fun)
  omega_nn = weno_nn.OmegaNN(
      features=config.features,
      features_fun=features_fun,
      act_fun=act_fun,
      dtype=jnp.float64,
  )

  # Initializing the parameters.
  init_params = omega_nn.init(init_rng, jnp.zeros((3,)))["params"]
  print(f"number of parameters of the network {utils.flat_dim(init_params)}")

  # Defining the optimizer.
  schedule = optax.warmup_cosine_decay_schedule(
      init_value=config.initial_lr,
      peak_value=config.peak_lr,
      warmup_steps=config.warmup_steps,
      decay_steps=config.num_train_steps,
      end_value=config.end_lr,
  )

  optimizer = optax.chain(
      optax.clip(config.clip),
      optax.adam(
          learning_rate=schedule,
          b1=config.beta1,
      ),
  )

  # Defining the train state.
  state_omega = train_state.TrainState.create(
      apply_fn=omega_nn.apply, params=init_params, tx=optimizer
  )

  # Training loop of the model.
  state, dict_losses = train.train(
      state_omega,
      data_loader_train,
      alpha=config.alpha,
      beta_d=config.beta_d,
      beta_w=config.beta_w,
  )

  orbax_checkpointer = checkpoint.PyTreeCheckpointer()
  bundle_ckpt = {"model": state, "config": config, "losses": dict_losses}

  orbax_checkpointer.save(
      workdir + "/checkpoints", bundle_ckpt
  )


if __name__ == "__main__":
  app.run(main)
