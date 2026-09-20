"""
ON LARGE-BATCH TRAINING FOR DEEP LEARNING:GENERALIZATION GAP AND SHARP MINIMA
Keskar et al. (2017) sharpness metric for JAX parameter pytrees.
"""

from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import Bounds, minimize


class SharpnessBatch(NamedTuple):
    initial_hstate: tuple[jax.Array, jax.Array]
    obs: jax.Array
    done: jax.Array
    agent_positions: jax.Array
    action: jax.Array
    value: jax.Array
    log_prob: jax.Array
    advantages: jax.Array
    targets: jax.Array


class E3TSharpnessBatch(NamedTuple):
    initial_hstate: tuple[jax.Array, jax.Array]
    obs: jax.Array
    done: jax.Array
    agent_positions: jax.Array
    action: jax.Array
    other_action: jax.Array
    value: jax.Array
    log_prob: jax.Array
    advantages: jax.Array
    targets: jax.Array


def collect_final_sharpness_batch(
    env,
    network,
    final_runner_state,
    final_update_count,
    config,
    batchify_fn,
    unbatchify_fn,
    value_mu=0.0,
    value_sigma=1.0,
    include_other_action=False,
):
    """Collect a fixed post-training rollout and compute its GAE targets."""
    train_state, env_state, obs, done, hstate, rng = final_runner_state
    num_actors = int(config["NUM_ACTORS"])
    num_envs = int(config["NUM_ENVS"])
    sample_size = min(int(config["SHARPNESS"]["NUM_ACTORS"]), num_actors)
    actor_indices = jnp.linspace(
        0, num_actors - 1, sample_size
    ).round().astype(jnp.int32)
    initial_hstate = jax.tree.map(
        lambda value: jnp.take(value, actor_indices, axis=0), hstate
    )
    shaping_fraction = jnp.maximum(
        0.0,
        1.0 - final_update_count / config["NUM_REWARD_SHAPING_STEPS"],
    )

    def rollout_step(carry, unused):
        env_state_s, obs_s, done_s, hstate_s, rng_s = carry
        rng_s, action_rng, step_rng = jax.random.split(rng_s, 3)
        obs_batch = batchify_fn(obs_s, env.agents, num_actors)
        positions = batchify_fn(
            {
                "agent_0": env_state_s.env_state.agent_pos,
                "agent_1": env_state_s.env_state.agent_pos,
            },
            env.agents,
            num_actors,
        )
        network_output = network.apply(
            train_state.params,
            hstate_s,
            (
                obs_batch[jnp.newaxis, :],
                done_s[jnp.newaxis, :],
                positions[jnp.newaxis, :],
            ),
        )
        next_hstate, pi, value = network_output[:3]
        action = pi.sample(seed=action_rng).squeeze(0)
        log_prob = pi.log_prob(action).squeeze(0)
        env_action = unbatchify_fn(
            action, env.agents, num_envs, env.num_agents
        )
        env_action = {key: value.squeeze() for key, value in env_action.items()}
        step_rngs = jax.random.split(step_rng, num_envs)
        next_obs, next_env_state, reward, next_done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(step_rngs, env_state_s, env_action)
        reward = jax.tree.map(
            lambda raw, shaped: raw + shaping_fraction * shaped,
            reward,
            info["shaped_reward"],
        )
        reward_batch = batchify_fn(reward, env.agents, num_actors).squeeze()
        next_done_batch = batchify_fn(
            next_done, env.agents, num_actors
        ).squeeze()

        transition_prefix = (
            jnp.take(obs_batch, actor_indices, axis=0),
            jnp.take(done_s, actor_indices, axis=0),
            jnp.take(positions, actor_indices, axis=0),
            jnp.take(action, actor_indices, axis=0),
        )
        if include_other_action:
            other_env_action = {
                "agent_0": env_action["agent_1"],
                "agent_1": env_action["agent_0"],
            }
            other_action = batchify_fn(
                other_env_action, env.agents, num_actors
            ).squeeze()
            transition_prefix += (
                jnp.take(other_action, actor_indices, axis=0),
            )
        transition = transition_prefix + (
            jnp.take(value.squeeze(0), actor_indices, axis=0),
            jnp.take(log_prob, actor_indices, axis=0),
            jnp.take(reward_batch, actor_indices, axis=0),
            jnp.take(
                jnp.tile(next_done["__all__"], env.num_agents),
                actor_indices,
                axis=0,
            ),
        )
        next_carry = (
            next_env_state, next_obs, next_done_batch, next_hstate, rng_s
        )
        return next_carry, transition

    final_carry, trajectory = jax.lax.scan(
        rollout_step,
        (env_state, obs, done, hstate, rng),
        None,
        int(config["SHARPNESS"]["NUM_STEPS"]),
    )
    final_env_state, final_obs, final_done, final_hstate, _ = final_carry
    final_obs_batch = batchify_fn(final_obs, env.agents, num_actors)
    final_positions = batchify_fn(
        {
            "agent_0": final_env_state.env_state.agent_pos,
            "agent_1": final_env_state.env_state.agent_pos,
        },
        env.agents,
        num_actors,
    )
    final_network_output = network.apply(
        train_state.params,
        final_hstate,
        (
            final_obs_batch[jnp.newaxis, :],
            final_done[jnp.newaxis, :],
            final_positions[jnp.newaxis, :],
        ),
    )
    final_value = final_network_output[2]
    final_value = jnp.take(
        final_value.squeeze(0), actor_indices, axis=0
    )
    final_value_real = final_value * value_sigma + value_mu
    if include_other_action:
        (
            obs_batch,
            done_batch,
            positions_batch,
            actions,
            other_actions,
            old_values,
            old_log_probs,
            rewards,
            global_dones,
        ) = trajectory
    else:
        (
            obs_batch,
            done_batch,
            positions_batch,
            actions,
            old_values,
            old_log_probs,
            rewards,
            global_dones,
        ) = trajectory

    def gae_step(carry, transition):
        gae, next_value = carry
        value_normalized, reward, global_done = transition
        value_real = value_normalized * value_sigma + value_mu
        delta = (
            reward
            + config["GAMMA"] * next_value * (1 - global_done)
            - value_real
        )
        gae = (
            delta
            + config["GAMMA"]
            * config["GAE_LAMBDA"]
            * (1 - global_done)
            * gae
        )
        return (gae, value_real), gae

    _, advantages = jax.lax.scan(
        gae_step,
        (jnp.zeros_like(final_value_real), final_value_real),
        (old_values, rewards, global_dones),
        reverse=True,
    )
    targets_real = advantages + old_values * value_sigma + value_mu
    targets_normalized = (targets_real - value_mu) / value_sigma
    batch_fields = {
        "initial_hstate": initial_hstate,
        "obs": obs_batch,
        "done": done_batch,
        "agent_positions": positions_batch,
        "action": actions,
        "value": old_values,
        "log_prob": old_log_probs,
        "advantages": advantages,
        "targets": targets_normalized,
    }
    if include_other_action:
        return E3TSharpnessBatch(
            other_action=other_actions,
            **batch_fields,
        )
    return SharpnessBatch(**batch_fields)


def compute_keskar_sharpness(
    loss_fn: Callable[[Any], jax.Array],
    params: Any,
    epsilons: Iterable[float],
    maxiter: int = 10,
) -> dict[str, float]:
    """Approximately maximize ``loss_fn`` in Keskar's full-space box.

    For each epsilon, the perturbation coordinate ``delta_i`` is constrained by
    ``|delta_i| <= epsilon * (|x_i| + 1)``.  As in the paper, the constrained
    problem is solved inexactly with L-BFGS-B.

    The loss data captured by ``loss_fn`` must remain fixed throughout this
    function. This is particularly important for on-policy RL losses.
    """
    flat_params, unravel = ravel_pytree(params)
    flat_params = jnp.asarray(flat_params)

    def flat_loss(delta):
        return loss_fn(unravel(flat_params + delta))

    value_and_grad = jax.jit(jax.value_and_grad(flat_loss))
    zero = np.zeros(flat_params.shape, dtype=np.asarray(flat_params).dtype)
    base_loss = float(flat_loss(jnp.asarray(zero)))
    results: dict[str, float] = {}

    def scipy_objective(delta):
        value, grad = value_and_grad(jnp.asarray(delta, dtype=flat_params.dtype))
        # scipy minimizes, whereas sharpness requires maximizing the loss.
        return -float(value), -np.asarray(grad, dtype=np.float64)

    abs_params = np.abs(np.asarray(flat_params, dtype=np.float64))
    for epsilon in epsilons:
        epsilon = float(epsilon)
        radii = epsilon * (abs_params + 1.0)
        result = minimize(
            scipy_objective,
            zero,
            method="L-BFGS-B",
            jac=True,
            bounds=Bounds(-radii, radii),
            options={"maxiter": int(maxiter)},
        )

        # The starting point is feasible, so never report an approximate
        # maximum below the unperturbed loss if the solver terminates early.
        max_loss = max(base_loss, -float(result.fun))
        increase = max_loss - base_loss
        denominator = 1.0 + base_loss
        sharpness = (
            100.0 * increase / denominator
            if denominator > 0.0
            else float("nan")
        )
        suffix = f"eps_{epsilon:g}"
        results[f"sharpness/keskar_{suffix}"] = sharpness

    return results
