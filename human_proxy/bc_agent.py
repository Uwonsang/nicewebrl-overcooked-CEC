"""Behaviour cloning on the Overcooked-AI 2019 human-human trial data
(see `preprocess.py` for how the raw CSV is turned into (obs, action) pairs).

Trains a CNN policy via supervised cross-entropy on human actions, using the same
observation format (9x9x26) as the CEC/IPPO Overcooked baselines so the resulting
policy can be dropped into the same evaluation pipeline as a human-proxy agent.
"""

import os
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from jax_tqdm import scan_tqdm
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import flax.linen as nn
from omegaconf import OmegaConf


ACTION_DIM = 6  # right, down, left, up, stay, interact (matches env.action_set)


class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Conv(features=32, kernel_size=(5, 5), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        return x


class BCPolicy(nn.Module):
    action_dim: int = ACTION_DIM
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        embedding = CNN(self.activation)(x)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)
        return logits


def load_split(data_dir, stem, split):
    npz = np.load(Path(data_dir) / f"{stem}_{split}.npz")
    return jnp.asarray(npz["obs"]), jnp.asarray(npz["actions"])


def make_train(config):
    batch_size = config["BATCH_SIZE"]
    network = BCPolicy(activation=config["ACTIVATION"])

    def train(rng, train_obs, train_actions, test_obs, test_actions):
        n_train = train_obs.shape[0]
        n_batches = n_train // batch_size

        rng, init_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((1, 9, 9, 26), dtype=jnp.float32)
        params = network.init(init_rng, dummy_obs)

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"]),
        )
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        def loss_and_acc(params, obs, actions):
            logits = network.apply(params, obs.astype(jnp.float32))
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, actions).mean()
            acc = (jnp.argmax(logits, axis=-1) == actions).mean()
            return loss, acc

        def _update_minibatch(train_state, batch):
            obs_b, act_b = batch

            def _loss_fn(params):
                return loss_and_acc(params, obs_b, act_b)

            (loss, acc), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, (loss, acc)

        @scan_tqdm(config["NUM_EPOCHS"])
        def _update_epoch(runner_state, epoch_idx):
            train_state, rng = runner_state
            rng, perm_rng = jax.random.split(rng)
            perm = jax.random.permutation(perm_rng, n_train)[: n_batches * batch_size]

            batched_obs = train_obs[perm].reshape((n_batches, batch_size) + train_obs.shape[1:])
            batched_actions = train_actions[perm].reshape((n_batches, batch_size))

            train_state, (losses, accs) = jax.lax.scan(_update_minibatch, train_state, (batched_obs, batched_actions))

            test_loss, test_acc = loss_and_acc(train_state.params, test_obs, test_actions)

            def callback(metric):
                wandb.log(metric)

            metric = {
                "epoch": epoch_idx,
                "train_loss": losses.mean(),
                "train_acc": accs.mean(),
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
            jax.debug.callback(callback, metric)

            return (train_state, rng), metric

        (train_state, rng), metrics = jax.lax.scan(
            _update_epoch, (train_state, rng), jnp.arange(config["NUM_EPOCHS"]), config["NUM_EPOCHS"]
        )
        return {"train_state": train_state, "metrics": metrics}

    return train


@hydra.main(version_base=None, config_path="config", config_name="bc")
def main(config):
    config = OmegaConf.to_container(config)

    run_name = f'bc_overcooked_{config["LAYOUT"]}_seed{config["SEED"]}'
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["BC", "human_proxy", config["LAYOUT"]],
        config=config,
        mode=config["WANDB_MODE"],
        name=run_name,
    )

    data_stem = f'{config["DATA_STEM"]}_{config["LAYOUT"]}'
    train_obs, train_actions = load_split(config["DATA_DIR"], data_stem, "train")
    test_obs, test_actions = load_split(config["DATA_DIR"], data_stem, "test")
    print(f"loaded {train_obs.shape[0]} train / {test_obs.shape[0]} test samples for layout {config['LAYOUT']}")

    rng = jax.random.PRNGKey(config["SEED"])
    train_fn = jax.jit(make_train(config))
    out = train_fn(rng, train_obs, train_actions, test_obs, test_actions)

    ckpt_dir = Path(config["CKPT_DIR"]) / config["LAYOUT"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(ckpt_dir / f"{run_name}.pkl", "wb") as f:
        pickle.dump(jax.device_get(out["train_state"].params), f)
    print(f"saved checkpoint to {ckpt_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
