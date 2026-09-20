"""
Based on PureJaxRL Implementation of PPO.

Note, this file will only work for MPE environments with homogenous agents (e.g. Simple Spread).

"""
import os
import pickle
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import OmegaConf

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts

import wandb
import functools
import pdb
from jax_tqdm import scan_tqdm
import yaml
from pathlib import Path
import time
from baselines.CEC_UED.algo_utils import (
    BCPolicy,
    EVAL_LAYOUTS_9,
    load_human_proxy_params,
    make_eval_envs_overcooked,
)

def initialize_environment(config):
    layout_name = config["ENV_KWARGS"]["layout"]
    config['layout_name'] = layout_name
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if config["ENV_NAME"] == "overcooked":
        temp_reset = lambda key: env.custom_reset(key, random_reset=True, shuffle_inv_and_pot=False, layout=env.layout)
        reset_env = jax.jit(temp_reset)
        def gen_held_out(runner_state, unused):
            (i,) = runner_state
            _, ho_state = reset_env(jax.random.key(i))
            res = (ho_state.goal_pos, ho_state.wall_map, ho_state.pot_pos)
            carry = (i+1,)
            return carry, res
        carry, res = jax.lax.scan(gen_held_out, (0,), jnp.arange(100), 100)
        ho_goal, ho_wall, ho_pot = [], [], []
        for layout_name, padded_layout in overcooked_layouts.items():  # add hand crafted ones to heldout set
            if "padded" in layout_name:
                _, ho_state = env.custom_reset(jax.random.PRNGKey(0), random_reset=False, shuffle_inv_and_pot=False, layout=padded_layout)
                ho_goal.append(ho_state.goal_pos)
                ho_wall.append(ho_state.wall_map)
                ho_pot.append(ho_state.pot_pos)
        ho_goal = jnp.stack(ho_goal, axis=0)
        ho_wall = jnp.stack(ho_wall, axis=0)
        ho_pot = jnp.stack(ho_pot, axis=0)
        ho_goal = jnp.concatenate([res[0], ho_goal], axis=0)
        ho_wall = jnp.concatenate([res[1], ho_wall], axis=0)
        ho_pot = jnp.concatenate([res[2], ho_pot], axis=0)
        env.held_out_goal, env.held_out_wall, env.held_out_pot = (ho_goal, ho_wall, ho_pot)
    elif config["ENV_NAME"] == "ToyCoop":
        # Generate 100 held-out states for ToyCoop
        @scan_tqdm(100)
        def gen_held_out_toycoop(runner_state, unused):
            (i,) = runner_state
            key = jax.random.key(i)
            state = env.custom_reset_fn(key, random_reset=True)
            res = (state.agent_pos, state.goal_pos, state.other_goal_pos)
            carry = (i+1,)
            return carry, res
        
        carry, res = jax.lax.scan(gen_held_out_toycoop, (0,), jnp.arange(100), 100)
        ho_agent_pos, ho_goal_pos, ho_other_goal_pos = res
        
        # Set the held-out states in the environment
        env.held_out_agent_pos = ho_agent_pos
        env.held_out_goal_pos = ho_goal_pos
        env.held_out_other_goal_pos = ho_other_goal_pos
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    return env

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        lstm_state = carry
        ins, resets = x
        
        # Reset LSTM state on episode boundaries
        lstm_state = jax.tree.map(
            lambda x: jnp.where(resets[:, np.newaxis], jnp.zeros_like(x), x),
            lstm_state
        )
        
        new_lstm_state, y = nn.OptimizedLSTMCell(features=ins.shape[-1])(lstm_state, ins)
        return new_lstm_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.OptimizedLSTMCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, agent_positions = x
        if self.config["CONV_NET"]:
            batch_size, num_envs, flattened_obs_dim = obs.shape
            if self.config["ENV_NAME"] == "overcooked":
                reshaped_obs = obs.reshape(-1, 9,9,26)
            else:
                reshaped_obs = obs.reshape(-1, 5,5,4)

            embedding = nn.Conv(
                features=64 if "9" in self.config['layout_name'] else 2 * self.config["FC_DIM_SIZE"],
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(reshaped_obs)
            embedding = nn.relu(embedding)
            embedding = nn.Conv(
                features=32 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"],
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.relu(embedding)

            embedding = embedding.reshape((batch_size, num_envs, -1))
        else:
            embedding = obs

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"], 
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        if self.config["LSTM"]:
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN()(hidden, rnn_in)
        else:
            # embedding = embedding.reshape((batch_size, num_envs, -1))
            embedding = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
            embedding = nn.relu(embedding)

        #########
        # Actor
        #########
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] , kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            actor_mean
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        if self.config["ENV_NAME"] == "overcooked":
            actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                actor_mean
            )
            actor_mean = nn.relu(actor_mean)  # extra layer 1

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)        

        pi = distrax.Categorical(logits=actor_mean)

        #########
        # Critic
        #########
        critic = nn.Dense(self.config["FC_DIM_SIZE"]*2, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            critic
        )
        critic = nn.relu(critic)
        if self.config["ENV_NAME"] == "overcooked":
            critic = nn.Dense(self.config["FC_DIM_SIZE"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 1
            critic = nn.Dense(self.config["FC_DIM_SIZE"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 2
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    agent_positions: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(
    config, update_step=0, save_info=None, opt_state=None,
    train_state_step=None,
):
    # env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = initialize_environment(config)
    
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    resume_update_step = (
        0 if opt_state is not None
        else update_step * (
            config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]
        )
    )
    config["MAX_TRAIN_UPDATES"] = (
        config["MAX_TRAIN_STEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    env = LogWrapper(env, env_params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})

    eval_envs = make_eval_envs_overcooked(config)

    eval_xp_enabled = (
        config["ENV_NAME"] == "overcooked"
        and len(eval_envs) > 0
        and bool(config["EVAL_KWARGS"]["eval_xp"])
    )
    layout_name = config["layout_name"]
    if eval_xp_enabled and layout_name not in EVAL_LAYOUTS_9:
        raise ValueError(
            f"XP evaluation does not support layout: {layout_name}"
        )

    human_proxy_params = {}
    if eval_xp_enabled:
        human_proxy_params = load_human_proxy_params(
            config["EVAL_KWARGS"]["human_proxy_ckpt_dir"],
            int(config["EVAL_KWARGS"]["human_proxy_num_seeds"]),
            layout_names=(layout_name,),
        )

    LOG_INTERVAL = max(1, int(config["NUM_UPDATES"]) // 100)
    remaining_updates = int(config["NUM_UPDATES"]) - update_step

    def linear_schedule(count):
        frac = (
            1.0
            - ((count + resume_update_step) // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["MAX_TRAIN_UPDATES"]
        )
        frac = jnp.maximum(1e-9, frac)
        return config["LR"] * frac

    def train(
        rng, frozen_param_stack, model_params=None,
        resume_runner_state=None, num_stacked_params=1,
    ):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        bc_network = BCPolicy()
        rng, _rng = jax.random.split(rng)
        # get flattened obs dim
        flattened_obs_dim = 1
        for dim in env.observation_space(env.agents[0]).shape:
            flattened_obs_dim *= dim
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], flattened_obs_dim)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], 2, 2)).astype(jnp.int32)
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        network_params = network.init(_rng, init_hstate, init_x)
        if model_params is not None:
            network_params = model_params
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        if opt_state is not None:
            train_state = train_state.replace(
                opt_state=opt_state,
                step=(
                    train_state.step
                    if train_state_step is None else train_state_step
                ),
            )

        # INIT OR RESTORE ENV RUNNER STATE
        if resume_runner_state is None:
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            init_hstate = ScannedRNN.initialize_carry(
                config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
            )
            rng, runner_rng = jax.random.split(rng)
        else:
            (
                env_state, obsv, restored_done, init_hstate, runner_rng,
            ) = resume_runner_state

        def eval_layout_xp_direction(eval_env, main_params, bc_params, eval_rng, main_agent_id):
            """Rolls out `main_params` (recurrent) paired against a human_proxy BC policy.

            `main_agent_id` picks which env seat the main agent controls; the other seat
            is controlled by the (stateless) BC policy.
            """
            other_agent_id = "agent_1" if main_agent_id == "agent_0" else "agent_0"
            num_eval_envs = int(config["EVAL_KWARGS"]["num_envs"])

            eval_rng, reset_rng = jax.random.split(eval_rng)
            reset_rngs = jax.random.split(reset_rng, num_eval_envs)
            init_obs, init_state = jax.vmap(eval_env.reset, in_axes=(0,))(reset_rngs)
            init_hstate = ScannedRNN.initialize_carry(num_eval_envs, config["GRU_HIDDEN_DIM"])
            init_done = jnp.zeros((num_eval_envs,), dtype=bool)
            init_returns = jnp.zeros((num_eval_envs,), dtype=jnp.float32)
            runner_state = (init_state, init_obs, init_done, init_hstate, init_returns, eval_rng)

            def _eval_step(carry, _):
                env_state_e, obs_e, main_done_e, hstate_e, returns_e, rng_e = carry
                rng_e, main_rng_e, other_rng_e = jax.random.split(rng_e, 3)

                agent_positions_e = env_state_e.env_state.agent_pos.reshape(num_eval_envs, -1)
                main_ac_in = (
                    obs_e[main_agent_id].reshape(num_eval_envs, -1)[np.newaxis, :],
                    main_done_e[np.newaxis, :],
                    agent_positions_e[np.newaxis, :],
                )
                hstate_next, main_pi, _ = network.apply(main_params, hstate_e, main_ac_in)
                main_pi = distrax.Categorical(logits=main_pi.logits * config["EVAL_KWARGS"]["beta"])
                main_sampled = main_pi.sample(seed=main_rng_e)[0]
                main_greedy = jnp.argmax(main_pi.probs, axis=-1)[0]
                main_action = jnp.where(config["EVAL_KWARGS"]["argmax"], main_greedy, main_sampled)

                other_logits = bc_network.apply(bc_params, obs_e[other_agent_id].astype(jnp.float32))
                other_pi = distrax.Categorical(logits=other_logits * config["EVAL_KWARGS"]["beta"])
                other_sampled = other_pi.sample(seed=other_rng_e)
                other_greedy = jnp.argmax(other_pi.probs, axis=-1)
                other_action = jnp.where(config["EVAL_KWARGS"]["argmax"], other_greedy, other_sampled)

                env_act = {main_agent_id: main_action, other_agent_id: other_action}

                rng_e, _rng_e = jax.random.split(rng_e)
                rng_step_e = jax.random.split(_rng_e, num_eval_envs)
                obs_next, state_next, reward, done, _info = jax.vmap(
                    eval_env.step, in_axes=(0, 0, 0)
                )(rng_step_e, env_state_e, env_act)

                returns_next = returns_e + reward["agent_0"]
                return (state_next, obs_next, done[main_agent_id], hstate_next, returns_next, rng_e), None

            runner_state, _ = jax.lax.scan(_eval_step, runner_state, None, int(config["EVAL_KWARGS"]["num_steps"]))
            _, _, _, _, returns, _ = runner_state
            return returns.mean()

        def eval_layout_xp(
            eval_env, main_params, bc_params_stacked, eval_rng,
        ):
            """Cross-play score averaged over human_proxy seeds and both seats."""
            def _one_seed(bc_params, seed_rng):
                rng_a, rng_b = jax.random.split(seed_rng)
                r_main_as_0 = eval_layout_xp_direction(
                    eval_env, main_params, bc_params, rng_a, "agent_0"
                )
                r_main_as_1 = eval_layout_xp_direction(
                    eval_env, main_params, bc_params, rng_b, "agent_1"
                )
                return (r_main_as_0 + r_main_as_1) / 2.0

            num_hp_seeds = int(config["EVAL_KWARGS"]["human_proxy_num_seeds"])
            seed_rngs = jax.random.split(eval_rng, num_hp_seeds)
            return jax.vmap(_one_seed)(bc_params_stacked, seed_rngs).mean()

        # TRAIN LOOP
        @scan_tqdm(remaining_updates)
        def _update_step(update_runner_state, unused, frozen_param_stack=frozen_param_stack, num_stacked_params=num_stacked_params):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng, frozen_param, other_hstate, frozen_is_agent_1 = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                agent_positions = {'agent_0': env_state.env_state.agent_pos, 'agent_1': env_state.env_state.agent_pos}  
                agent_positions = batchify(agent_positions, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    agent_positions[np.newaxis, :],
                )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )

                # Get other agent actions
                other_hstate, other_pi, other_value = network.apply(frozen_param, other_hstate, ac_in)
                other_action = other_pi.sample(seed=_rng)
                other_log_prob = other_pi.log_prob(other_action)
                other_env_act = unbatchify(
                    other_action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                other_env_act = {k: v.squeeze() for k, v in other_env_act.items()}
                env_act = {k: v.squeeze() for k, v in env_act.items()}
                env_act['agent_0'] = jnp.where(frozen_is_agent_1, env_act['agent_0'], other_env_act['agent_0'])
                env_act['agent_1'] = jnp.where(frozen_is_agent_1, other_env_act['agent_1'], env_act['agent_1'])

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                
                # remove shaped rewards
                del info['shaped_reward']

                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    agent_positions
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, frozen_param, other_hstate, frozen_is_agent_1)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            init_other_hstate = jax.tree.map(lambda x: jnp.zeros_like(x), initial_hstate)
            # init_other_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            # sample param from 3 * 6 possible params
            seed = jax.random.randint(rng, (1,), minval=0, maxval=num_stacked_params)[0]
            frozen_param = jax.tree.map(lambda x: x[seed], frozen_param_stack)
            rng, _rng = jax.random.split(rng)
            frozen_is_agent_1 = jax.random.bernoulli(_rng, 0.5)

            rollout_runner_state = train_state, env_state, last_obs, last_done, hstate, rng, frozen_param, init_other_hstate, frozen_is_agent_1
            rollout_runner_state, traj_batch = jax.lax.scan(
                _env_step, rollout_runner_state, None, config["NUM_STEPS"]
            )
            train_state, env_state, last_obs, last_done, hstate, rng, frozen_param, init_other_hstate, frozen_is_agent_1 = rollout_runner_state
            runner_state = train_state, env_state, last_obs, last_done, hstate, rng

            # Only the actor slots actually played by the trained network (not the frozen
            # partner) should contribute to the loss; the other slot's action/log_prob never
            # corresponds to what was actually executed in the env.
            agent_0_is_trained = frozen_is_agent_1.astype(jnp.float32)
            agent_1_is_trained = 1.0 - agent_0_is_trained
            actor_mask = jnp.concatenate([
                jnp.full((config["NUM_ENVS"],), agent_0_is_trained),
                jnp.full((config["NUM_ENVS"],), agent_1_is_trained),
            ])
            actor_mask = jnp.broadcast_to(actor_mask[None, :], (config["NUM_STEPS"], config["NUM_ACTORS"]))

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            agent_positions = {'agent_0': env_state.env_state.agent_pos, 'agent_1': env_state.env_state.agent_pos}
            agent_positions = batchify(agent_positions, env.agents, config["NUM_ACTORS"])
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
                agent_positions[np.newaxis, :],
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets, actor_mask = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets, actor_mask):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            jax.tree.map(lambda h: h.squeeze(), init_hstate),
                            (traj_batch.obs, traj_batch.done, traj_batch.agent_positions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        mask_sum = jnp.maximum(actor_mask.sum(), 1.0)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * (jnp.maximum(
                            value_losses, value_losses_clipped
                        ) * actor_mask).sum() / mask_sum

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae_mean = (gae * actor_mask).sum() / mask_sum
                        gae_var = (jnp.square(gae - gae_mean) * actor_mask).sum() / mask_sum
                        gae = (gae - gae_mean) / (jnp.sqrt(gae_var) + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = (loss_actor * actor_mask).sum() / mask_sum
                        entropy = (pi.entropy() * actor_mask).sum() / mask_sum

                        # debug
                        approx_kl = (((ratio - 1) - logratio) * actor_mask).sum() / mask_sum
                        clip_frac = ((jnp.abs(ratio - 1) > config["CLIP_EPS"]).astype(jnp.float32) * actor_mask).sum() / mask_sum

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets, actor_mask
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    actor_mask,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate = jax.tree.map(lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)), init_hstate)
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                    actor_mask,
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    jax.tree.map(lambda x: x.squeeze(), init_hstate),
                    traj_batch,
                    advantages,
                    targets,
                    actor_mask,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                actor_mask,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            ratio_0 = loss_info[1][3].at[0,0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }
            rng = update_state[-1]

            if eval_xp_enabled:
                run_eval = (
                    (update_steps % LOG_INTERVAL == 0)
                    | (update_steps == int(config["NUM_UPDATES"]) - 1)
                )

                def _do_eval(_):
                    base = jax.random.fold_in(rng, update_steps)
                    layout_return = eval_layout_xp(
                        eval_envs[layout_name],
                        train_state.params,
                        human_proxy_params[layout_name],
                        base,
                    )
                    return {
                        f"{layout_name}_xp": layout_return,
                        "mean_xp": layout_return,
                    }

                def _skip_eval(_):
                    nan = jnp.array(jnp.nan, dtype=jnp.float32)
                    return {
                        f"{layout_name}_xp": nan,
                        "mean_xp": nan,
                    }

                metric["eval_returns"] = jax.lax.cond(
                    run_eval, _do_eval, _skip_eval, operand=None
                )

            def callback(metric):
                log_data = {
                    "returns": metric["returns"],
                    "env_step": metric["update_steps"]
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"],
                    **metric["loss"],
                }
                if "eval_returns" in metric:
                    xp_mean = float(metric["eval_returns"]["mean_xp"])
                    if np.isfinite(xp_mean):
                        log_data["eval_xp/mean"] = xp_mean
                        log_data[f"eval_xp/{layout_name}"] = float(
                            metric["eval_returns"][f"{layout_name}_xp"]
                        )
                wandb.log(log_data, step=int(metric["update_steps"]))
                current_return = float(metric["returns"])
                if current_return > best_return[0]:
                    best_return[0] = current_return
                    os.makedirs(config['filepath'], exist_ok=True)
                    ckpt_path = (
                        f"{config['filepath']}/"
                        f"fcp_fixed_seed{config['SEED']}_best.pkl"
                    )
                    with open(ckpt_path, "wb") as f:
                        pickle.dump({
                            'params': metric["params"],
                            'returns': current_return,
                            'update_steps': int(metric['update_steps']),
                        }, f)

            returns = metric["returned_episode_returns"][:, :, 0][
                            metric["returned_episode"][:, :, 0].astype(jnp.int32)
                        ].mean()
            metric["returns"] = returns
            metric["update_steps"] = update_steps
            metric["params"] = train_state.params

            jax.experimental.io_callback(callback, None, metric)

            def ckpt_callback(
                params, opt_state_, tx_step, step, env_state_, last_obs_,
                last_done_, hstate_, rng_,
            ):
                mid_ckpt_dir = config["MID_CKPT_DIR"]
                os.makedirs(mid_ckpt_dir, exist_ok=True)
                mid_ckpt_path = os.path.join(
                    mid_ckpt_dir, "resume_ckpt.pkl"
                )
                tmp_ckpt_path = f"{mid_ckpt_path}.tmp"
                with open(tmp_ckpt_path, "wb") as f:
                    pickle.dump({
                        "params": params,
                        "opt_state": opt_state_,
                        "tx_step": tx_step,
                        "final_update_step": int(step) + 1,
                        "wandb_run_id": wandb.run.id,
                        "runner_state": (
                            env_state_, last_obs_, last_done_, hstate_, rng_,
                        ),
                    }, f)
                os.replace(tmp_ckpt_path, mid_ckpt_path)

            is_scheduled_ckpt = jnp.equal(
                update_steps % LOG_INTERVAL, 0
            )
            jax.lax.cond(
                is_scheduled_ckpt,
                lambda _: jax.experimental.io_callback(
                    ckpt_callback, None,
                    train_state.params, train_state.opt_state,
                    train_state.step, update_steps, env_state, last_obs,
                    last_done, hstate, rng, ordered=True,
                ),
                lambda _: None,
                operand=None,
            )
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), metric

        best_return = [float('-inf')]

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            (
                jnp.zeros((config["NUM_ACTORS"]), dtype=bool)
                if resume_runner_state is None else restored_done
            ),
            init_hstate,
            runner_rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step,
            (runner_state, update_step),
            jnp.arange(remaining_updates),
            remaining_updates,
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="repro_config", config_name="fcp_final_baseline")
def main(config):
    save_xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
    config = OmegaConf.to_container(config)
    config["model_name"] = "FCP_FIXED"
    if config['TRAIN_KWARGS']['finetune']:
        config['LR'] = config['LR'] / 10
        finetune_appendage = "_improved_finetuneIK"
    else:
        finetune_appendage = "_improved"
    
    if config['ENV_KWARGS']['partial_obs']:
        finetune_appendage += "_partial_obs"
    if not config['LSTM']:
        finetune_appendage += "_no_lstm"
    if config['ENV_KWARGS']['incentivize_strat'] != 2:
        finetune_appendage += f"_incentivize_strat_{config['ENV_KWARGS']['incentivize_strat']}"
    
    if config["WANDB_MODE"] == "online":
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])

    if config["ENV_NAME"] == "ToyCoop":
        run_name = (
            f"FCP_FIXED_ToyCoop_ik{config['ENV_KWARGS']['random_reset']}"
            f"_seed{config['SEED']}"
        )
    else:
        run_name = (
            f"FCP_FIXED_{config['ENV_KWARGS']['layout']}_"
            f"{config['SEED']}"
        )
    resume_xpid = config.get("RESUME_XPID", "")
    active_xpid = resume_xpid if resume_xpid else save_xpid
    filepath = f"ckpts/fcp_fixed/{config['ENV_NAME']}"
    if config["ENV_NAME"] == "overcooked":
        filepath += f"/{config['ENV_KWARGS']['layout']}"
    filepath = f"{filepath}/ik{config['ENV_KWARGS']['random_reset']}/{config['ENV_KWARGS']['random_reset_fn']}/{active_xpid}"
    config['filepath'] = filepath
    config["MID_CKPT_DIR"] = os.path.join(
        filepath, f"seed{config['SEED']}_mid_ckpts"
    )
    mid_ckpt_path = os.path.join(
        config["MID_CKPT_DIR"], "resume_ckpt.pkl"
    )
    has_mid_ckpt = bool(resume_xpid) and os.path.exists(mid_ckpt_path)
    mid_ckpt = None
    wandb_resume_id = None
    if has_mid_ckpt:
        with open(mid_ckpt_path, "rb") as f:
            mid_ckpt = pickle.load(f)
        wandb_resume_id = mid_ckpt.get("wandb_run_id")

    if wandb_resume_id:
        wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            id=wandb_resume_id,
            resume="must",
            mode=config["WANDB_MODE"],
        )
    else:
        wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=["IPPO", "RNN", "FCP", "FCP_FIXED"],
            config=config,
            mode=config["WANDB_MODE"],
            name=run_name
        )

    #####################
    # Load frozen params
    #####################
    frozen_param_stack = []

    if config['FCP_KWARGS']['train_oracle']:
        # only load 2 strategies
        path = f"{filepath}/seed3_ckpt16_improved_incentivize_strat_0.pkl"
        with open(path, "rb") as f:
            frozen_ckpt = pickle.load(f)
            frozen_param_stack.append(frozen_ckpt['params'])
        path = f"{filepath}/seed3_ckpt16_improved_incentivize_strat_1.pkl"
        with open(path, "rb") as f:
            frozen_ckpt = pickle.load(f)
            frozen_param_stack.append(frozen_ckpt['params'])
        finetune_appendage += "_oracle"

    if not config['TRAIN_KWARGS']['overwrite_ckpt']:
        # check if ckpt exists
        if os.path.exists(f"{filepath}/fcp_fixed_seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}{finetune_appendage}.pkl"):
            print(f"Checkpoint {config['TRAIN_KWARGS']['ckpt_id']} already exists, exiting")
            print(f"filepath: {filepath}")
            exit(0)

    resume_runner_state = None
    resume_train_state_step = None
    opt_state = None
    if has_mid_ckpt:
        print(f"Found mid-run checkpoint: {mid_ckpt_path}")
        model_params = mid_ckpt["params"]
        opt_state = mid_ckpt.get("opt_state")
        resume_train_state_step = mid_ckpt.get("tx_step")
        resume_runner_state = mid_ckpt.get("runner_state")
        final_update_step = int(mid_ckpt["final_update_step"])
        rng = jax.random.PRNGKey(config["SEED"])
        print(f"Resuming from update step {final_update_step}")
    elif config['TRAIN_KWARGS']['ckpt_id'] > 0:
        print("Loading checkpoint")
        with open(f"{filepath}/fcp_fixed_seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id'] - 1}{finetune_appendage}.pkl", "rb") as f:
            previous_ckpt = pickle.load(f)
            model_params = previous_ckpt['params']
            final_update_step = previous_ckpt['final_update_step']
            rng = previous_ckpt['key']
            rng, _rng = jax.random.split(jax.random.PRNGKey(rng))
    elif config['TRAIN_KWARGS']['finetune']:
        print("Loading finetune checkpoint")
        ik_filepath = f"ckpts/ippo/{config['ENV_NAME']}"
        if config["ENV_NAME"] == "overcooked":
            ik_filepath += f"/{config['ENV_KWARGS']['layout']}"
        ik_filepath += f"/ikTrue"
        with open(f"{ik_filepath}/seed{config['SEED']}_ckpt19_improved.pkl", "rb") as f:
            previous_ckpt = pickle.load(f)
            model_params = previous_ckpt['params']
            # final_update_step = previous_ckpt['final_update_step']
            final_update_step = 0
            rng = previous_ckpt['key']
            rng, _rng = jax.random.split(jax.random.PRNGKey(rng))
    else:
        model_params = None
        final_update_step = 0
        rng = jax.random.PRNGKey(config["SEED"])
    
    if len(frozen_param_stack) == 0:
        ckpt_id_list = ['init', 'final', 'mid']
        if config['ENV_KWARGS']['random_reset']:
            ckpt_id_list = [9, 19, 29]
        elif config['ENV_KWARGS']['partial_obs']:  # handle partial obs for toy env 
            ckpt_id_list = [0, 1, 3]
        elif config['ENV_KWARGS']['incentivize_strat'] == 3:
            ckpt_id_list = [1, 2, 3]
        seed_list = [0, 1, 2, 3, 5, 6]
        CKPT_ROOT = Path(__file__).resolve().parents[2] / config['FCP_filepath']
        if config["ENV_NAME"] == "ToyCoop":
            custom_path = str(CKPT_ROOT)
        else:
            custom_path = os.path.join(CKPT_ROOT, config['ENV_KWARGS']['layout'])
        for ckpt_id in ckpt_id_list:
            for ckpt_seed in seed_list:
                path_to_open = (
                    Path(custom_path)
                    / f"seed{ckpt_seed}"
                    / "fcp_pool"
                    / f"seed{ckpt_seed}_{ckpt_id}.pkl"
                )
                print(f"Looking for FCP partner: {path_to_open}")
                if not path_to_open.exists():
                    continue
                with open(path_to_open, "rb") as f:
                    frozen_ckpt= pickle.load(f)
                    frozen_param_stack.append(frozen_ckpt['params'])
    num_stacked_params = len(frozen_param_stack)
    frozen_param_stack = jax.tree.map(lambda *x: jnp.stack(x), *frozen_param_stack)
    
    save_info = {"filepath": filepath}
    train_jit = jax.jit(
        make_train(
            config, final_update_step, save_info, opt_state,
            resume_train_state_step,
        ),
        device=jax.devices()[0],
    )
    out = train_jit(
        rng, frozen_param_stack, model_params, resume_runner_state,
        num_stacked_params,
    )
    final_runner_state, final_update_step = out['runner_state']
    model_state = final_runner_state[0]
    rng = final_runner_state[-1]
    
    num_updates = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
        
    # save model
    os.makedirs(filepath, exist_ok=True)
    final_checkpoint = (
        f"{filepath}/seed{config['SEED']}_ckpt"
        f"{config['TRAIN_KWARGS']['ckpt_id']}_fcp_fixed"
        f"{finetune_appendage}_updates{num_updates}.pkl"
    )
    with open(final_checkpoint, "wb") as f:
        ckpt = {'key': rng, 'params': model_state.params, 'update_steps': num_updates}
        pickle.dump(ckpt, f)

    print(f"Saved model to {final_checkpoint}")
    print(f"Finished training for seed {config['SEED']} with ckpt {config['TRAIN_KWARGS']['ckpt_id']}_updates{num_updates}")
    print(f"--------------------------------")
    

if __name__ == "__main__":
    main()
