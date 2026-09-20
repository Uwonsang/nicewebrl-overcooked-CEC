"""Shared evaluation rollouts for CEC/E3T training scripts."""

import distrax
import jax
import jax.numpy as jnp
import numpy as np

from baselines.CEC_UED.evaluation_metrics import (
    compute_evaluation_critic_statistics,
)


def evaluate_self_play_layout(
    eval_env,
    params,
    eval_rng,
    network_apply,
    initialize_carry,
    batchify,
    unbatchify,
    num_eval_envs,
    num_steps,
    hidden_dim,
    beta,
    argmax,
    return_critic_stats=False,
    gamma=0.99,
):
    """Evaluate one recurrent policy controlling both seats of one layout."""
    num_actors = eval_env.num_agents * num_eval_envs
    eval_rng, reset_rng = jax.random.split(eval_rng)
    reset_rngs = jax.random.split(reset_rng, num_eval_envs)
    init_obs, init_state = jax.vmap(eval_env.reset, in_axes=(0,))(reset_rngs)
    init_hstate = initialize_carry(num_actors, hidden_dim)
    init_done = jnp.zeros((num_actors,), dtype=bool)
    init_returns = jnp.zeros((num_eval_envs,), dtype=jnp.float32)
    runner_state = (
        init_state, init_obs, init_done, init_hstate, init_returns, eval_rng,
    )

    def _eval_step(carry, _):
        env_state, obs, done, hstate, returns, rng = carry
        rng, action_rng, step_rng = jax.random.split(rng, 3)
        obs_batch = batchify(obs, eval_env.agents, num_actors)
        positions = {
            "agent_0": env_state.env_state.agent_pos,
            "agent_1": env_state.env_state.agent_pos,
        }
        positions = batchify(positions, eval_env.agents, num_actors)
        network_outputs = network_apply(
            params,
            hstate,
            (
                obs_batch[np.newaxis, :],
                done[np.newaxis, :],
                positions[np.newaxis, :],
            ),
        )
        hstate_next, policy, value = network_outputs[:3]
        policy = distrax.Categorical(logits=policy.logits * beta)
        sampled_action = policy.sample(seed=action_rng)[0]
        greedy_action = jnp.argmax(policy.probs, axis=-1)[0]
        action = jnp.where(argmax, greedy_action, sampled_action)
        env_action = unbatchify(
            action, eval_env.agents, num_eval_envs, eval_env.num_agents,
        )
        env_action = {key: value.squeeze() for key, value in env_action.items()}
        step_rngs = jax.random.split(step_rng, num_eval_envs)
        obs_next, state_next, reward, done_next_dict, _ = jax.vmap(
            eval_env.step, in_axes=(0, 0, 0),
        )(step_rngs, env_state, env_action)
        done_next = batchify(
            done_next_dict, eval_env.agents, num_actors,
        ).squeeze()
        returns_next = returns + reward["agent_0"]
        reward_batch = batchify(
            reward, eval_env.agents, num_actors,
        ).squeeze()
        critic_transition = (
            (value.squeeze(), reward_batch, done_next)
            if return_critic_stats else None
        )
        return (
            state_next, obs_next, done_next, hstate_next, returns_next, rng,
        ), critic_transition

    runner_state, critic_trajectory = jax.lax.scan(
        _eval_step, runner_state, None, int(num_steps),
    )
    mean_return = runner_state[4].mean()
    if not return_critic_stats:
        return mean_return

    values, rewards, dones = critic_trajectory
    critic_stats = compute_evaluation_critic_statistics(
        values,
        rewards,
        dones,
        jnp.zeros_like(values[-1]),
        gamma,
    )
    return mean_return, critic_stats


def _evaluate_cross_play_direction(
    eval_env,
    main_params,
    bc_params,
    eval_rng,
    main_agent_id,
    network_apply,
    bc_network_apply,
    initialize_carry,
    num_eval_envs,
    num_steps,
    hidden_dim,
    beta,
    argmax,
    return_critic_stats=False,
    gamma=0.99,
):
    """Evaluate one main-policy seat against one BC-policy partner."""
    other_agent_id = "agent_1" if main_agent_id == "agent_0" else "agent_0"
    eval_rng, reset_rng = jax.random.split(eval_rng)
    reset_rngs = jax.random.split(reset_rng, num_eval_envs)
    init_obs, init_state = jax.vmap(eval_env.reset, in_axes=(0,))(reset_rngs)
    init_hstate = initialize_carry(num_eval_envs, hidden_dim)
    init_done = jnp.zeros((num_eval_envs,), dtype=bool)
    init_returns = jnp.zeros((num_eval_envs,), dtype=jnp.float32)
    runner_state = (
        init_state, init_obs, init_done, init_hstate, init_returns, eval_rng,
    )

    def _eval_step(carry, _):
        env_state, obs, main_done, hstate, returns, rng = carry
        rng, main_rng, other_rng, step_rng = jax.random.split(rng, 4)
        positions = env_state.env_state.agent_pos.reshape(num_eval_envs, -1)
        main_input = (
            obs[main_agent_id].reshape(num_eval_envs, -1)[np.newaxis, :],
            main_done[np.newaxis, :],
            positions[np.newaxis, :],
        )
        network_outputs = network_apply(main_params, hstate, main_input)
        hstate_next, main_policy, main_value = network_outputs[:3]
        main_policy = distrax.Categorical(logits=main_policy.logits * beta)
        main_sampled = main_policy.sample(seed=main_rng)[0]
        main_greedy = jnp.argmax(main_policy.probs, axis=-1)[0]
        main_action = jnp.where(argmax, main_greedy, main_sampled)

        other_logits = bc_network_apply(
            bc_params, obs[other_agent_id].astype(jnp.float32),
        )
        other_policy = distrax.Categorical(logits=other_logits * beta)
        other_sampled = other_policy.sample(seed=other_rng)
        other_greedy = jnp.argmax(other_policy.probs, axis=-1)
        other_action = jnp.where(argmax, other_greedy, other_sampled)

        env_action = {
            main_agent_id: main_action,
            other_agent_id: other_action,
        }
        step_rngs = jax.random.split(step_rng, num_eval_envs)
        obs_next, state_next, reward, done, _ = jax.vmap(
            eval_env.step, in_axes=(0, 0, 0),
        )(step_rngs, env_state, env_action)
        returns_next = returns + reward["agent_0"]
        critic_transition = (
            (main_value.squeeze(), reward[main_agent_id], done[main_agent_id])
            if return_critic_stats else None
        )
        return (
            state_next,
            obs_next,
            done[main_agent_id],
            hstate_next,
            returns_next,
            rng,
        ), critic_transition

    runner_state, critic_trajectory = jax.lax.scan(
        _eval_step, runner_state, None, int(num_steps),
    )
    mean_return = runner_state[4].mean()
    if not return_critic_stats:
        return mean_return

    values, rewards, dones = critic_trajectory
    critic_stats = compute_evaluation_critic_statistics(
        values,
        rewards,
        dones,
        jnp.zeros_like(values[-1]),
        gamma,
    )
    return mean_return, critic_stats


def evaluate_cross_play_layout(
    eval_env,
    main_params,
    bc_params_stacked,
    eval_rng,
    network_apply,
    bc_network_apply,
    initialize_carry,
    num_eval_envs,
    num_steps,
    hidden_dim,
    beta,
    argmax,
    num_human_proxy_seeds,
    return_critic_stats=False,
    gamma=0.99,
):
    """Average XP return over both seats and all human-proxy seeds."""
    def _one_seed(bc_params, seed_rng):
        rng_agent_0, rng_agent_1 = jax.random.split(seed_rng)
        main_as_agent_0 = _evaluate_cross_play_direction(
            eval_env, main_params, bc_params, rng_agent_0, "agent_0",
            network_apply, bc_network_apply, initialize_carry,
            num_eval_envs, num_steps, hidden_dim, beta, argmax,
            return_critic_stats, gamma,
        )
        main_as_agent_1 = _evaluate_cross_play_direction(
            eval_env, main_params, bc_params, rng_agent_1, "agent_1",
            network_apply, bc_network_apply, initialize_carry,
            num_eval_envs, num_steps, hidden_dim, beta, argmax,
            return_critic_stats, gamma,
        )
        if not return_critic_stats:
            return (main_as_agent_0 + main_as_agent_1) / 2.0

        return_0, stats_0 = main_as_agent_0
        return_1, stats_1 = main_as_agent_1
        return (
            (return_0 + return_1) / 2.0,
            jax.tree.map(lambda a, b: (a + b) / 2.0, stats_0, stats_1),
        )

    seed_rngs = jax.random.split(eval_rng, num_human_proxy_seeds)
    per_seed_results = jax.vmap(_one_seed)(bc_params_stacked, seed_rngs)
    if not return_critic_stats:
        return per_seed_results.mean()

    per_seed_returns, per_seed_stats = per_seed_results
    return (
        per_seed_returns.mean(),
        jax.tree.map(lambda value: value.mean(), per_seed_stats),
    )
