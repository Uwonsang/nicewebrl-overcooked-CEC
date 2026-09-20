"""Evaluate directional cross-play between selected ICRL algorithms.

Each invocation evaluates one layout. Agent 0 uses ``algo_1`` and agent 1
uses ``algo_2``; both role directions are retained in the output CSV. For
same-algorithm cells, XP_ONLY excludes identical checkpoint seeds.
"""
from __future__ import annotations

import glob
import pickle
import re
from pathlib import Path

import distrax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import jaxmarl
from jaxmarl.environments.overcooked import overcooked_layouts

from actor_networks import (
    ActorCriticE3T,
    ActorCriticRNN,
    IDAACActorRNN,
    ScannedRNN,
)


DEFAULT_MODELS = [
    "IPPO",
    "E3T",
    "FCP",
    "CEC_envs64",
    "CEC_IDAAC_envs32",
    "CEC_IDAAC_envs256",
]

MODEL_SPECS = {
    "IPPO": {"family": "IPPO", "network": "rnn", "seeds": [0, 1, 2, 3, 5, 6]},
    "E3T": {"family": "E3T", "network": "e3t", "seeds": [0, 1, 2, 3, 4, 5]},
    "FCP": {"family": "FCP", "network": "rnn", "seeds": [0, 1, 2, 3, 4, 5]},
    "CEC_envs64": {
        "family": "CEC", "network": "rnn", "num_envs": 64,
        "seeds": [0, 1, 2, 3, 4, 5],
    },
    "CEC_IDAAC_envs32": {
        "family": "CEC_IDAAC", "network": "idaac", "num_envs": 32,
        "seeds": [0, 1, 2, 3, 4, 5],
    },
    "CEC_IDAAC_envs256": {
        "family": "CEC_IDAAC", "network": "idaac", "num_envs": 256,
        "seeds": [0, 1, 2, 3, 4, 5],
    },
}


def initialize_environment(config):
    layout_name = config["ENV_KWARGS"]["layout"]
    if layout_name not in overcooked_layouts:
        raise ValueError(f"Unknown Overcooked layout: {layout_name}")
    config["layout_name"] = layout_name
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    return env


def checkpoint_patterns(model_name: str, seed: int, model_root: Path, layout: str):
    spec = MODEL_SPECS[model_name]
    family = spec["family"]
    root = model_root / family
    if family in {"CEC", "CEC_IDAAC"}:
        num_envs = spec["num_envs"]
        return [str(root / str(num_envs) / f"seed{seed}" / f"seed{seed}_ckpt*.pkl")]
    if family == "IPPO":
        return [str(root / layout / f"seed{seed}" / f"seed{seed}_best.pkl")]
    if family == "E3T":
        return [str(root / layout / f"seed{seed}" / f"seed{seed}_best_e3t.pkl")]
    if family == "FCP":
        return [str(root / layout / f"seed{seed}" / f"fcp_seed{seed}_best.pkl")]
    raise ValueError(f"Unsupported model family: {family}")


def checkpoint_sort_key(path: str):
    numbers = re.findall(r"(?:updates|ckpt)(\d+)", Path(path).name)
    return tuple(int(number) for number in numbers) or (0,)


def find_checkpoint(model_name: str, seed: int, model_root: Path, layout: str):
    matches = []
    for pattern in checkpoint_patterns(model_name, seed, model_root, layout):
        matches.extend(glob.glob(pattern))
    return max(matches, key=checkpoint_sort_key) if matches else None


def load_model_group(model_name: str, config, layout: str):
    if model_name not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model {model_name}. Available: {', '.join(MODEL_SPECS)}"
        )
    spec = MODEL_SPECS[model_name]
    model_root = Path(config["MODEL_PATH"])
    params, seeds, paths = [], [], []
    for seed in spec["seeds"]:
        checkpoint = find_checkpoint(model_name, seed, model_root, layout)
        if checkpoint is None:
            print(f"Missing checkpoint: model={model_name}, seed={seed}")
            continue
        try:
            with open(checkpoint, "rb") as file:
                params.append(pickle.load(file)["params"])
            seeds.append(seed)
            paths.append(checkpoint)
            print(f"Loaded {model_name} seed {seed}: {checkpoint}")
        except (OSError, KeyError, pickle.UnpicklingError) as exc:
            print(f"Failed checkpoint: model={model_name}, seed={seed}: {exc}")
    if not params:
        raise RuntimeError(f"No checkpoints found for model={model_name}")
    param_stack = jax.tree.map(lambda *values: jnp.stack(values), *params)
    return {
        "name": model_name,
        "params": param_stack,
        "seeds": jnp.asarray(seeds),
        "paths": paths,
        "network_type": spec["network"],
    }


def make_network(network_type: str, env, config):
    action_dim = env.action_space(env.agents[0]).n
    if network_type == "idaac":
        return IDAACActorRNN(action_dim, config=config)
    if network_type == "e3t":
        return ActorCriticE3T(action_dim, config=config)
    return ActorCriticRNN(action_dim, config=config)


def apply_policy(network, params, hidden, actor_input, beta: float):
    result = network.apply(params, hidden, actor_input)
    hidden, policy = result[0], result[1]
    return hidden, distrax.Categorical(logits=policy.logits * beta)


def get_rollout_returns(params_1, params_2, network_1, network_2, config, env):
    """Return one cumulative reward per trajectory without retaining states."""
    beta = float(config["TEST_KWARGS"]["beta"])
    use_argmax = bool(config["TEST_KWARGS"]["argmax"])

    def one_rollout(rng):
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key)
        hidden_1 = ScannedRNN.initialize_carry(env.num_agents, config["GRU_HIDDEN_DIM"])
        hidden_2 = ScannedRNN.initialize_carry(env.num_agents, config["GRU_HIDDEN_DIM"])
        dones = jnp.zeros(env.num_agents, dtype=bool)

        def step(carry, _):
            state, observations, last_dones, h_1, h_2, step_rng = carry
            step_rng, action_key_0, action_key_1, env_key = jax.random.split(step_rng, 4)
            obs_batch = jnp.stack(
                [observations[agent].flatten() for agent in env.agents]
            )
            actor_input = (
                obs_batch[jnp.newaxis, :],
                last_dones[jnp.newaxis, :],
                state.agent_pos[jnp.newaxis, ...],
            )
            h_1, policy_1 = apply_policy(network_1, params_1, h_1, actor_input, beta)
            h_2, policy_2 = apply_policy(network_2, params_2, h_2, actor_input, beta)
            sampled_0 = policy_1.sample(seed=action_key_0)[0, 0]
            sampled_1 = policy_2.sample(seed=action_key_1)[0, 1]
            action_0 = jnp.where(
                use_argmax, jnp.argmax(policy_1.probs[0, 0]), sampled_0
            )
            action_1 = jnp.where(
                use_argmax, jnp.argmax(policy_2.probs[0, 1]), sampled_1
            )
            actions = {env.agents[0]: action_0, env.agents[1]: action_1}
            next_obs, next_state, reward, done, _ = env.step(env_key, state, actions)
            next_dones = jnp.asarray([done[agent] for agent in env.agents])
            next_carry = (
                next_state, next_obs, next_dones, h_1, h_2, step_rng
            )
            return next_carry, reward[env.agents[0]]

        carry = (env_state, obs, dones, hidden_1, hidden_2, rng)
        _, rewards = jax.lax.scan(step, carry, None, length=config["NUM_STEPS"])
        return rewards.sum()

    keys = jax.random.split(
        jax.random.PRNGKey(config["SEED"]),
        config["TEST_KWARGS"]["num_trajs"],
    )
    return jax.vmap(one_rollout)(keys)


def evaluate_algorithm_pair(group_1, group_2, network_1, network_2, config, env):
    indices_1 = jnp.arange(len(group_1["seeds"]))
    indices_2 = jnp.arange(len(group_2["seeds"]))
    pairs = jnp.asarray(jnp.meshgrid(indices_1, indices_2)).reshape(2, -1).T
    if config.get("XP_ONLY", True) and group_1["name"] == group_2["name"]:
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    def evaluate(pair, params_stack_1, params_stack_2):
        index_1, index_2 = pair
        params_1 = jax.tree.map(lambda value: value[index_1], params_stack_1)
        params_2 = jax.tree.map(lambda value: value[index_2], params_stack_2)
        rewards = get_rollout_returns(
            params_1, params_2, network_1, network_2, config, env
        )
        return group_1["seeds"][index_1], group_2["seeds"][index_2], rewards

    evaluate_all = jax.jit(jax.vmap(evaluate, in_axes=(0, None, None)))
    return evaluate_all(pairs, group_1["params"], group_2["params"])


@hydra.main(version_base=None, config_path="repro_config", config_name="cross_algo")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    requested_models = [str(model) for model in (config.get("MODEL_NAMES") or DEFAULT_MODELS)]
    unknown = [model for model in requested_models if model not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unsupported MODEL_NAMES: {unknown}")

    layout = config["ENV_KWARGS"]["layout"]
    output_dir = Path(config["SAVE_PATH"])
    output_dir.mkdir(parents=True, exist_ok=True)
    env = initialize_environment(config)
    groups = {
        model: load_model_group(model, config, layout) for model in requested_models
    }
    networks = {
        model: make_network(groups[model]["network_type"], env, config)
        for model in requested_models
    }

    records = []
    for model_1 in requested_models:
        for model_2 in requested_models:
            print(f"Evaluating {model_1} (agent 0) vs {model_2} (agent 1)")
            seeds_1, seeds_2, rewards = evaluate_algorithm_pair(
                groups[model_1], groups[model_2],
                networks[model_1], networks[model_2], config, env,
            )
            seeds_1 = np.asarray(jax.device_get(seeds_1))
            seeds_2 = np.asarray(jax.device_get(seeds_2))
            rewards = np.asarray(jax.device_get(rewards))
            for pair_index in range(len(seeds_1)):
                for trajectory_index in range(rewards.shape[1]):
                    records.append({
                        "layout": layout,
                        "algo_1": model_1,
                        "algo_2": model_2,
                        "seed_1": int(seeds_1[pair_index]),
                        "seed_2": int(seeds_2[pair_index]),
                        "trajectory": trajectory_index,
                        "reward": float(rewards[pair_index, trajectory_index]),
                    })

    output_path = output_dir / f"{layout}_cross_algo_results.csv"
    pd.DataFrame.from_records(records).to_csv(output_path, index=False)
    print(f"Saved data to {output_path}")


if __name__ == "__main__":
    main()
