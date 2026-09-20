"""Evaluate cross-algorithm play on 100 procedurally generated tasks.

The task set is created by ``test_general_pcg.initialize_environment``, so task
indices 0--99 are identical to the existing per-algorithm PCG evaluation.
``ENV_KWARGS.layout`` selects the checkpoint cohort for layout-specific models
(IPPO, E3T, and FCP); it is not used as the evaluation layout.
"""
from __future__ import annotations

from pathlib import Path

import distrax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax.core import unfreeze
from omegaconf import OmegaConf

from actor_networks import ScannedRNN
from cross_algo import DEFAULT_MODELS, MODEL_SPECS, load_model_group, make_network
from test_general_pcg import initialize_environment


def get_rollout_returns(
    params_1, params_2, network_1, network_2, config, env, reset_layout
):
    """Return cumulative rewards for one model pair on one fixed PCG task."""
    beta = float(config["TEST_KWARGS"]["beta"])
    use_argmax = bool(config["TEST_KWARGS"]["argmax"])

    def apply_policy(network, params, hidden, actor_input):
        result = network.apply(params, hidden, actor_input)
        hidden, policy = result[0], result[1]
        return hidden, distrax.Categorical(logits=policy.logits * beta)

    def one_rollout(rng):
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.custom_reset(
            reset_key,
            random_reset=False,
            shuffle_inv_and_pot=False,
            layout=reset_layout,
        )
        hidden_1 = ScannedRNN.initialize_carry(
            env.num_agents, config["GRU_HIDDEN_DIM"]
        )
        hidden_2 = ScannedRNN.initialize_carry(
            env.num_agents, config["GRU_HIDDEN_DIM"]
        )
        dones = jnp.zeros(env.num_agents, dtype=bool)

        def step(carry, _):
            state, observations, last_dones, h_1, h_2, step_rng = carry
            step_rng, action_key_0, action_key_1, env_key = jax.random.split(
                step_rng, 4
            )
            obs_batch = jnp.stack(
                [observations[agent].flatten() for agent in env.agents]
            )
            actor_input = (
                obs_batch[jnp.newaxis, :],
                last_dones[jnp.newaxis, :],
                state.agent_pos[jnp.newaxis, ...],
            )
            h_1, policy_1 = apply_policy(network_1, params_1, h_1, actor_input)
            h_2, policy_2 = apply_policy(network_2, params_2, h_2, actor_input)
            sampled_0 = policy_1.sample(seed=action_key_0)[0, 0]
            sampled_1 = policy_2.sample(seed=action_key_1)[0, 1]
            action_0 = jnp.where(
                use_argmax, jnp.argmax(policy_1.probs[0, 0]), sampled_0
            )
            action_1 = jnp.where(
                use_argmax, jnp.argmax(policy_2.probs[0, 1]), sampled_1
            )
            actions = {env.agents[0]: action_0, env.agents[1]: action_1}
            next_obs, next_state, reward, done, _ = env.step(
                env_key, state, actions
            )
            next_dones = jnp.asarray([done[agent] for agent in env.agents])
            next_carry = (
                next_state, next_obs, next_dones, h_1, h_2, step_rng
            )
            return next_carry, reward[env.agents[0]]

        initial = (env_state, obs, dones, hidden_1, hidden_2, rng)
        _, rewards = jax.lax.scan(
            step, initial, None, length=config["NUM_STEPS"]
        )
        return rewards.sum()

    rollout_keys = jax.random.split(
        jax.random.PRNGKey(config["SEED"]),
        config["TEST_KWARGS"]["num_trajs"],
    )
    return jax.vmap(one_rollout)(rollout_keys)


def seed_pairs(group_1, group_2, xp_only):
    """Return parameter indices and their checkpoint seed labels."""
    index_pairs = np.asarray(
        [
            (i, j)
            for i in range(len(group_1["seeds"]))
            for j in range(len(group_2["seeds"]))
            if not (
                xp_only
                and group_1["name"] == group_2["name"]
                and int(group_1["seeds"][i]) == int(group_2["seeds"][j])
            )
        ],
        dtype=np.int32,
    )
    if not len(index_pairs):
        raise ValueError(
            f"No seed pairs for {group_1['name']} and {group_2['name']}"
        )
    labels_1 = np.asarray(group_1["seeds"])[index_pairs[:, 0]]
    labels_2 = np.asarray(group_2["seeds"])[index_pairs[:, 1]]
    return jnp.asarray(index_pairs), labels_1, labels_2


def make_pair_evaluator(group_1, group_2, network_1, network_2, config, env):
    """Build one compiled evaluator for an ordered algorithm pair."""
    pairs, labels_1, labels_2 = seed_pairs(
        group_1, group_2, bool(config["XP_ONLY"])
    )

    def evaluate(pair, params_stack_1, params_stack_2, reset_layout):
        params_1 = jax.tree.map(lambda value: value[pair[0]], params_stack_1)
        params_2 = jax.tree.map(lambda value: value[pair[1]], params_stack_2)
        return get_rollout_returns(
            params_1, params_2, network_1, network_2, config, env, reset_layout
        )

    evaluate_all = jax.jit(
        jax.vmap(evaluate, in_axes=(0, None, None, None))
    )
    return evaluate_all, pairs, labels_1, labels_2


@hydra.main(
    version_base=None,
    config_path="repro_config",
    config_name="cross_algo_pcg",
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    requested_models = [
        str(model) for model in (config.get("MODEL_NAMES") or DEFAULT_MODELS)
    ]
    unknown = [model for model in requested_models if model not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unsupported MODEL_NAMES: {unknown}")
    if len(requested_models) != len(set(requested_models)):
        raise ValueError("MODEL_NAMES must not contain duplicates")
    if config["ENV_NAME"] != "overcooked":
        raise ValueError("cross_algo_pcg.py currently supports Overcooked only")
    if int(config["PCG_NUM_TASKS"]) != 100:
        raise ValueError("PCG_NUM_TASKS must be 100 to match test_general_pcg.py")
    if int(config["NUM_STEPS"]) != int(config["ENV_KWARGS"]["max_steps"]):
        raise ValueError("NUM_STEPS must equal ENV_KWARGS.max_steps")

    checkpoint_layout = str(config["ENV_KWARGS"]["layout"])
    output_dir = Path(config["SAVE_PATH"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{checkpoint_layout}_pcg_cross_algo_results.csv"
    if output_path.exists() and not config.get("OVERWRITE", False):
        raise FileExistsError(
            f"Output exists: {output_path}. Set OVERWRITE=true to replace it."
        )

    # This call generates the exact task list used by test_general_pcg.py.
    env = initialize_environment(config)
    generated_tasks = unfreeze(config["eval_held_out_layouts"])
    if len(generated_tasks) != 100:
        raise RuntimeError(f"Expected 100 generated tasks, got {len(generated_tasks)}")

    groups = {
        model: load_model_group(model, config, checkpoint_layout)
        for model in requested_models
    }
    networks = {
        model: make_network(groups[model]["network_type"], env, config)
        for model in requested_models
    }

    if output_path.exists():
        output_path.unlink()
    wrote_header = False
    for model_1 in requested_models:
        for model_2 in requested_models:
            evaluate, pairs, labels_1, labels_2 = make_pair_evaluator(
                groups[model_1],
                groups[model_2],
                networks[model_1],
                networks[model_2],
                config,
                env,
            )
            records = []
            for task_index, reset_layout in enumerate(generated_tasks):
                rewards = np.asarray(
                    jax.device_get(
                        evaluate(
                            pairs,
                            groups[model_1]["params"],
                            groups[model_2]["params"],
                            reset_layout,
                        )
                    )
                )
                if not np.isfinite(rewards).all():
                    raise RuntimeError(
                        f"Nonfinite reward for {model_1} vs {model_2}, "
                        f"task {task_index}"
                    )
                num_trajectories = rewards.shape[1]
                for pair_index in range(len(labels_1)):
                    for trajectory in range(num_trajectories):
                        records.append(
                            {
                                "layout": (
                                    f"{checkpoint_layout}/pcg_{task_index:03d}"
                                ),
                                "held_out_layout_idx": task_index,
                                "checkpoint_layout": checkpoint_layout,
                                "algo_1": model_1,
                                "algo_2": model_2,
                                "seed_1": int(labels_1[pair_index]),
                                "seed_2": int(labels_2[pair_index]),
                                "trajectory": trajectory,
                                "reward": float(rewards[pair_index, trajectory]),
                            }
                        )
            pd.DataFrame.from_records(records).to_csv(
                output_path,
                mode="a",
                header=not wrote_header,
                index=False,
            )
            wrote_header = True
            print(
                f"Completed {model_1} (agent 0) vs {model_2} (agent 1): "
                "100 tasks",
                flush=True,
            )

    print(f"Saved data to {output_path}")


if __name__ == "__main__":
    main()
