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
from typing import Sequence, NamedTuple, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import OmegaConf

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.overcooked.layouts import make_counter_circuit_9x9, make_forced_coord_9x9, make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9

import wandb
import functools
from jax_tqdm import scan_tqdm
import time
import yaml
from algo_utils import make_eval_envs_overcooked, EVAL_LAYOUTS_9, load_human_proxy_params, BCPolicy

from gradient_conflict_utils import (
    compute_layout_gradient_metrics,
    empty_layout_gradient_metrics,
)
from representation_metrics import (
    compute_optimizer_update_metrics,
    compute_minibatch_penultimate_metrics,
    empty_penultimate_metrics,
    first_epoch_first_minibatch_indices,
)


# Parameter groups used consistently by gradient diagnostics and norm metrics.
# Each group contains the shared encoder/RNN, its branch, and its output head.
ACTOR_TRUNK_KEYS = (
    "Conv_0", "Conv_1", "Dense_0", "Dense_1", "ScannedRNN_0",
    "Dense_2", "Dense_3", "Dense_4", "Dense_5", "Dense_6",
)
VALUE_TRUNK_KEYS = (
    "Conv_0", "Conv_1", "Dense_0", "Dense_1", "ScannedRNN_0",
    "Dense_7", "Dense_8", "Dense_9", "Dense_10", "Dense_11",
)
SHARED_TRUNK_KEYS = (
    "Conv_0", "Conv_1", "Dense_0", "Dense_1", "ScannedRNN_0",
)


def initialize_environment(config):
    layout_name = config["ENV_KWARGS"]["layout"]
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if config["ENV_NAME"] == "overcooked":
        def reset_env(key):
            def reset_sub_dict(key, fn):
                key, subkey = jax.random.split(key)
                sampled_layout_dict = fn(subkey, ik=True)
                temp_o, temp_s = env.custom_reset(key, layout=sampled_layout_dict, random_reset=False, shuffle_inv_and_pot=False)
                key, subkey = jax.random.split(key)
                return (temp_o, temp_s), key
                
            asymm_reset, key = reset_sub_dict(key, make_asymm_advantages_9x9)
            coord_ring_reset, key = reset_sub_dict(key, make_coord_ring_9x9)
            counter_circuit_reset, key = reset_sub_dict(key, make_counter_circuit_9x9)
            forced_coord_reset, key = reset_sub_dict(key, make_forced_coord_9x9)
            cramped_room_reset, key = reset_sub_dict(key, make_cramped_room_9x9)
            layout_resets = [asymm_reset, coord_ring_reset, counter_circuit_reset, forced_coord_reset, cramped_room_reset]
            # stack all layouts
            stacked_layout_reset = jax.tree.map(lambda *x: jnp.stack(x), *layout_resets)
            # sample an index from 0 to 4
            index = jax.random.randint(key, (), minval=0, maxval=5)
            sampled_reset = jax.tree.map(lambda x: x[index], stacked_layout_reset)
            return sampled_reset
        @scan_tqdm(100)
        def gen_held_out(runner_state, unused):
            (i,) = runner_state
            _, ho_state = reset_env(jax.random.key(i))
            res = (ho_state.goal_pos, ho_state.wall_map, ho_state.pot_pos)
            carry = (i+1,)
            return carry, res
        carry, res = jax.lax.scan(gen_held_out, (0,), jnp.arange(100), 100)
        ho_goal, ho_wall, ho_pot = [], [], []
        for layout_name, layout_dict in overcooked_layouts.items():  # add hand crafted ones to heldout set
            if "9" in layout_name:
                _, ho_state = env.custom_reset(jax.random.PRNGKey(0), random_reset=False, shuffle_inv_and_pot=False, layout=layout_dict)
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

def _is_connected_jax(passable_9x9):
    """Flood fill: True iff all passable cells form one connected component."""
    flat = passable_9x9.astype(jnp.float32).flatten()
    visited = jnp.zeros_like(flat).at[jnp.argmax(flat)].set(1.0).reshape(9, 9)
    def _spread(v, _):
        nbr = jnp.maximum(
            jnp.maximum(jnp.pad(v[1:],   ((0,1),(0,0))), jnp.pad(v[:-1],  ((1,0),(0,0)))),
            jnp.maximum(jnp.pad(v[:,1:], ((0,0),(0,1))), jnp.pad(v[:,:-1],((0,0),(1,0)))),
        )
        return passable_9x9.astype(jnp.float32) * jnp.maximum(v, nbr), None
    visited, _ = jax.lax.scan(_spread, visited, None, 18)
    return jnp.sum(visited) == jnp.sum(passable_9x9.astype(jnp.float32))

def _classify_layout_jax(maze_map_9x9_ch0):
    """Return layout ID: 0=cramped_room_9, 1=asymm_advantages_9, 2=coord_ring_9,
                         3=counter_circuit_9, 4=forced_coord_9"""
    passable = (maze_map_9x9_ch0 == 1) | (maze_map_9x9_ch0 == 10)
    n = passable.sum()
    conn = _is_connected_jax(passable)
    return jnp.where(n == 8, 2,
           jnp.where(n == 6,
               jnp.where(conn, 0, 4),
               jnp.where(conn, 3, 1)
           )).astype(jnp.int32)


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
        batch_size, num_envs, flattened_obs_dim = obs.shape

        # Intermediate feature norms are only materialized during the
        # diagnostic apply(..., mutable=["intermediates"]) call. Recording
        # per-sample norms avoids retaining every Conv/Dense activation while
        # adding no norm-computation overhead to ordinary rollout/update
        # forwards.

        collect_intermediates = (
            not self.is_initializing()
            and self.is_mutable_collection("intermediates")
        )

        def record_feature_norm(name, features):
            if collect_intermediates:
                feature_vectors = features.reshape(
                    (batch_size, num_envs, -1)
                )
                sample_norms = jnp.linalg.norm(feature_vectors, axis=-1)
                self.sow(
                    "intermediates", f"feature_norm_{name}", sample_norms
                )

        if self.config["CONV_NET"]:
            if self.config["ENV_NAME"] == "overcooked":
                reshaped_obs = obs.reshape(-1, 9,9,26)
            else:
                reshaped_obs = obs.reshape(-1, 5,5,4)

            embedding = nn.Conv(
                # features=64 if "9" in self.config['layout_name'] and self.config["ENV_NAME"] == "overcooked")else 2 * self.config["FC_DIM_SIZE"],
                features=64,
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(reshaped_obs)
            embedding = nn.relu(embedding)
            record_feature_norm("shared_conv_0", embedding)

            embedding = nn.Conv(
                # features=32 if "9" in self.config['layout_name'] and self.config["ENV_NAME"] == "overcooked") else self.config["FC_DIM_SIZE"],
                features=32,
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.relu(embedding)
            record_feature_norm("shared_conv_1", embedding)

            embedding = embedding.reshape((batch_size, num_envs, -1))
        else:
            embedding = obs

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)
        record_feature_norm("shared_dense_0", embedding)

        embedding = nn.Dense(
            # self.config["FC_DIM_SIZE"] * 2 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"], 
            self.config["FC_DIM_SIZE"] * 2,
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)
        record_feature_norm("shared_dense_1", embedding)

        if self.config["LSTM"]:
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN()(hidden, rnn_in)
        else:
            embedding = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
            embedding = nn.relu(embedding)
        embedding = embedding.reshape((batch_size, num_envs, -1))
        record_feature_norm("shared_recurrent", embedding)

        if not self.is_initializing():
            self.sow("intermediates", "shared_penultimate", embedding)

        #########
        # Actor
        #########
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] , kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        record_feature_norm("actor_hidden_0", actor_mean)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            actor_mean
        )
        actor_mean = nn.relu(actor_mean)
        record_feature_norm("actor_hidden_1", actor_mean)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        record_feature_norm("actor_hidden_2", actor_mean)
        if self.config["ENV_NAME"] == "overcooked":
            actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                actor_mean
            )
            actor_mean = nn.relu(actor_mean)  # extra layer 1
            record_feature_norm("actor_hidden_3", actor_mean)

        if not self.is_initializing():
            self.sow("intermediates", "actor_penultimate", actor_mean)

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
        record_feature_norm("critic_hidden_0", critic)
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            critic
        )
        critic = nn.relu(critic)
        record_feature_norm("critic_hidden_1", critic)
        if self.config["ENV_NAME"] == "overcooked":
            critic = nn.Dense(self.config["FC_DIM_SIZE"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 1
            record_feature_norm("critic_hidden_2", critic)
            critic = nn.Dense(self.config["FC_DIM_SIZE"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 2
            record_feature_norm("critic_hidden_3", critic)

        if not self.is_initializing():
            self.sow("intermediates", "critic_penultimate", critic)
            
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
    layout_id: jnp.ndarray


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
    # If opt_state is restored from a mid-run checkpoint, the optimizer's own step
    # count already reflects progress, so the manual offset would double-count it.
    resume_update_step = 0 if opt_state is not None else update_step * (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
    config["MAX_TRAIN_UPDATES"] = (
        config["MAX_TRAIN_STEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_REWARD_SHAPING_STEPS"] = config["MAX_TRAIN_UPDATES"] // 2  # used for annealing reward shaping
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )
    config["obs_dim"] = env.observation_space(env.agents[0]).shape

    obs, state = env.reset(jax.random.PRNGKey(0), params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})

    env = LogWrapper(env, env_params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})

    eval_envs = make_eval_envs_overcooked(config)

    eval_xp_enabled = (
        config["ENV_NAME"] == "overcooked"
        and len(eval_envs) > 0
        and bool(config["EVAL_KWARGS"]["eval_xp"])
    )
    human_proxy_params = {}
    if eval_xp_enabled:
        human_proxy_params = load_human_proxy_params(
            config["EVAL_KWARGS"]["human_proxy_ckpt_dir"],
            int(config["EVAL_KWARGS"]["human_proxy_num_seeds"]),
        )

    def linear_schedule(count):
        frac = (
            1.0
            - ((count + resume_update_step) // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["MAX_TRAIN_UPDATES"]
        )
        frac = jnp.maximum(1e-9, frac)
        return config["LR"] * frac

    remaining_updates = int(config["NUM_UPDATES"]) - update_step

    def train(rng, model_params=None, resume_runner_state=None):
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
                step=train_state.step if train_state_step is None else train_state_step,
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
            env_state, obsv, restored_done, init_hstate, runner_rng = resume_runner_state

        # WandB logging: cap at ~100 points over the full run. PPO scalars are
        # update-averaged; target/critic/TD metrics are logging-step snapshots.
        LOG_INTERVAL = max(1, int(config["NUM_UPDATES"]) // 100)
        _log_accum = {
            "sum": {},
            "count": {},
            "layout_sum": {},
            "layout_count": {},
            "eval_last": None,
        }

        # TRAIN LOOP
        @scan_tqdm(remaining_updates)
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng, update_step = runner_state

                # layout BEFORE env.step: the layout this transition's action/reward belong to
                pre_maze_map = env_state.env_state.maze_map
                layout_id = jax.vmap(_classify_layout_jax)(pre_maze_map[:, 4:13, 4:13, 0])  # (NUM_ENVS,)
                layout_id = jnp.tile(layout_id, [env.num_agents])  # (NUM_ACTORS,), matches agent_positions

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
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                shaped_reward = info['shaped_reward']
                reward_shaping_frac = jnp.maximum(0.0, 1.0 - (update_step / config["NUM_REWARD_SHAPING_STEPS"]))
                reward = jax.tree.map(lambda x, y: x + y * reward_shaping_frac, reward, shaped_reward)
                
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
                    agent_positions,
                    layout_id,
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, update_step)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            (train_state, env_state, obsv, done_batch, hstate, rng) = runner_state
            runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, update_steps)
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng, update_steps = runner_state
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
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
                    return (gae, value), (gae, delta)

                _, (advantages, td_errors) = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value, td_errors

            advantages, targets, td_errors = _calculate_gae(traj_batch, last_val)

            # ── per-layout gradient conflict ──────────────────────────────
            _LAYOUT_NAMES = [
                "cramped_room_9", "asymm_advantages_9", "coord_ring_9",
                "counter_circuit_9", "forced_coord_9",
            ]
            original_params = train_state.params

            # per-step layout id, classified pre-step inside _env_step, already tiled to actors
            _actor_layout_full = traj_batch.layout_id  # (NUM_STEPS, NUM_ACTORS)
            _layout_ids_full = _actor_layout_full[:, :config["NUM_ENVS"]]  # (NUM_STEPS, NUM_ENVS)

            # ── value target / critic quality statistics (raw scale, no PopArt here) ──
            target_stats = {}
            target_stats["target_raw/mean"] = targets.mean()

            _err = targets - traj_batch.value
            target_stats["critic/rmse"] = jnp.sqrt((_err ** 2).mean())

            target_stats["td_error/rmse"] = jnp.sqrt((td_errors ** 2).mean())

            for _lid, _name in enumerate(_LAYOUT_NAMES):
                _mask = (_actor_layout_full == _lid).astype(jnp.float32)
                _cnt_raw = _mask.sum()
                _cnt = jnp.maximum(_cnt_raw, 1.0)
                _valid = _cnt_raw > 0
                target_stats[f"target_raw/{_name}/mean"] = jnp.where(
                    _valid, (targets * _mask).sum() / _cnt, jnp.nan
                )
                target_stats[f"critic/{_name}/rmse"] = jnp.where(
                    _valid,
                    jnp.sqrt(((_err ** 2) * _mask).sum() / _cnt),
                    jnp.nan,
                )
                target_stats[f"td_error/{_name}/rmse"] = jnp.where(
                    _valid,
                    jnp.sqrt(((td_errors ** 2) * _mask).sum() / _cnt),
                    jnp.nan,
                )

            # ── end value target / critic quality statistics ───────────────

            run_eval = jnp.logical_or(
                jnp.equal(update_steps % LOG_INTERVAL, 0),
                jnp.equal(update_steps, int(config["NUM_UPDATES"]) - 1),
            )

            layout_gradient_window_steps = int(
                config["GRAD_CONFLICT_WINDOW_STEPS"]
            )
            def _compute_layout_gradient(_):
                layout_gradient_traj = jax.tree.map(
                    lambda x: x[:layout_gradient_window_steps],
                    traj_batch,
                )
                return compute_layout_gradient_metrics(
                    network=network,
                    original_params=original_params,
                    initial_hstate=initial_hstate,
                    traj_batch=layout_gradient_traj,
                    advantages=advantages[:layout_gradient_window_steps],
                    value_targets=targets[:layout_gradient_window_steps],
                    layout_ids_full=(
                        _layout_ids_full[:layout_gradient_window_steps]
                    ),
                    layout_names=_LAYOUT_NAMES,
                    config=config,
                    num_agents=env.num_agents,
                )

            layout_gradient_metrics = jax.lax.cond(
                run_eval,
                _compute_layout_gradient,
                lambda _: empty_layout_gradient_metrics(_LAYOUT_NAMES),
                operand=None,
            )

            # Use exactly the actor subset that the first epoch's first PPO
            # minibatch will consume. Time remains contiguous and each actor's
            # matching initial recurrent state is preserved.
            def _compute_representation_metrics(_):
                first_minibatch_indices = first_epoch_first_minibatch_indices(
                    rng,
                    config["NUM_ACTORS"],
                    config["NUM_MINIBATCHES"],
                )
                representation_hstate = jax.tree.map(
                    lambda h: jnp.take(h, first_minibatch_indices, axis=0),
                    initial_hstate,
                )
                representation_traj = jax.tree.map(
                    lambda x: jnp.take(x, first_minibatch_indices, axis=1),
                    traj_batch,
                )
                return compute_minibatch_penultimate_metrics(
                    network,
                    original_params,
                    representation_hstate,
                    (
                        representation_traj.obs,
                        representation_traj.done,
                        representation_traj.agent_positions,
                    ),
                )

            representation_metrics = jax.lax.cond(
                run_eval,
                _compute_representation_metrics,
                lambda _: empty_penultimate_metrics(),
                operand=None,
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _family_means(per_sample_values, layout_ids):
                        """Return within-family means and a present-family mask."""
                        family_ids = jnp.arange(len(_LAYOUT_NAMES))[:, jnp.newaxis, jnp.newaxis] # 5, 1, 1
                        family_masks = (layout_ids[jnp.newaxis, ...] == family_ids).astype(per_sample_values.dtype) # mask (5, time, actor)
                        family_counts = family_masks.sum(axis=(1, 2))
                        family_sums = (family_masks * per_sample_values[jnp.newaxis, ...]).sum(axis=(1, 2))
                        family_is_present = family_counts > 0

                        family_means = family_sums / jnp.maximum(family_counts, 1.0)

                        return family_means, family_is_present

                    def _family_losses_fn(params):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            jax.tree.map(lambda h: h.squeeze(), init_hstate),
                            (traj_batch.obs, traj_batch.done, traj_batch.agent_positions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        # loss by layout family
                        family_value_losses, family_is_present = _family_means(
                            0.5 * jnp.maximum(
                                value_losses, value_losses_clipped
                            ),
                            traj_batch.layout_id,
                        )

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (
                            advantages - advantages.mean()
                        ) / (advantages.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        family_actor_losses, _ = _family_means(
                            -jnp.minimum(loss_actor1, loss_actor2),
                            traj_batch.layout_id)

                        family_entropies, _ = _family_means(
                            pi.entropy(),
                            traj_batch.layout_id)

                        present_family_count = jnp.maximum(family_is_present.sum(), 1)

                        family_total_losses = (
                            family_actor_losses
                            + config["VF_COEF"] * family_value_losses
                            - config["ENT_COEF"] * family_entropies
                        ) * family_is_present

                        total_loss = (family_total_losses.sum() / present_family_count)

                        value_loss = (family_value_losses * family_is_present).sum() / present_family_count
                        loss_actor = (family_actor_losses * family_is_present).sum() / present_family_count
                        entropy = (family_entropies * family_is_present).sum() / present_family_count

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        return family_total_losses, (
                            total_loss,
                            (
                                value_loss,
                                loss_actor,
                                entropy,
                                ratio,
                                approx_kl,
                                clip_frac,
                            ),
                            family_is_present,
                        )

                    (
                        family_total_losses,
                        family_loss_vjp,
                        (loss, loss_aux, family_is_present)
                        ) = jax.vjp(_family_losses_fn, train_state.params, has_aux=True)

                    family_gradients = []
                    for family_id in range(len(_LAYOUT_NAMES)):
                        family_cotangent = jax.nn.one_hot(family_id, len(_LAYOUT_NAMES),)
                        family_gradient, = family_loss_vjp(family_cotangent)
                        family_gradients.append(family_gradient)

                    def _gradient_l2_norm(gradient):
                        squared_norm = sum(
                            jnp.sum(leaf ** 2)
                            for leaf in jax.tree_util.tree_leaves(gradient)
                        )
                        return jnp.sqrt(squared_norm)

                    family_gradient_norms = jnp.stack([_gradient_l2_norm(family_gradient) for family_gradient in family_gradients])
                    present_family_count = jnp.maximum(family_is_present.sum(), 1)
                    target_gradient_norm = jax.lax.stop_gradient((family_gradient_norms * family_is_present).sum() / present_family_count)

                    equalized_family_gradients = [
                        jax.tree.map(
                            lambda gradient: (
                                gradient * family_is_present[family_id] * target_gradient_norm
                                / (family_gradient_norms[family_id] + 1e-8)
                            ),
                            family_gradient,
                        )
                        for family_id, family_gradient in enumerate(
                            family_gradients
                        )
                    ]
                    grads = jax.tree.map(
                        lambda *family_gradient_leaves: (
                            sum(family_gradient_leaves)
                            / present_family_count
                        ),
                        *equalized_family_gradients,
                    )
                    equalized_family_gradient_norms = jnp.stack(
                        [
                            _gradient_l2_norm(family_gradient)
                            for family_gradient in equalized_family_gradients
                        ]
                    )

                    # Match SimBaV2's optimizer-update semantics: measure the
                    # parameter state used by this minibatch immediately before
                    # applying its gradient, then average across minibatches.
                    optimizer_update_metrics = compute_optimizer_update_metrics(
                        gradients=grads,
                        params=train_state.params,
                        actor_param_keys=ACTOR_TRUNK_KEYS,
                        critic_param_keys=VALUE_TRUNK_KEYS,
                        shared_param_keys=SHARED_TRUNK_KEYS,
                    )

                    optimizer_update_metrics["family_gradient_norm/target"] = target_gradient_norm
                    for family_id, family_name in enumerate(_LAYOUT_NAMES):
                        optimizer_update_metrics[
                            f"family_gradient_norm/raw/{family_name}"
                        ] = jnp.where(
                            family_is_present[family_id],
                            family_gradient_norms[family_id],
                            jnp.nan,
                        )
                        optimizer_update_metrics[
                            f"family_gradient_norm/equalized/{family_name}"
                        ] = jnp.where(
                            family_is_present[family_id],
                            equalized_family_gradient_norms[family_id],
                            jnp.nan,
                        )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss, loss_aux, optimizer_update_metrics)

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate = jax.tree.map(lambda h: jnp.reshape(h, (1, config["NUM_ACTORS"], -1)), init_hstate)
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
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
                    jax.tree.map(lambda h: h.squeeze(), init_hstate),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
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

            # 'returned_episode', 'returned_episode_lengths', 'returned_episode_returns'
            returns = metric["returned_episode_returns"][:, :, 0][
                metric["returned_episode"][:, :, 0].astype(jnp.int32)
            ].mean()
            # Save before reduction for per-layout return logging in callback
            episode_returns_step = metric["returned_episode_returns"][:, :, 0]  # (NUM_STEPS, NUM_ENVS)
            episode_done_step = metric["returned_episode"][:, :, 0]             # (NUM_STEPS, NUM_ENVS)
            # Reduce to scalars so scan output stays O(NUM_UPDATES), not O(NUM_UPDATES*NUM_STEPS*...)
            metric = jax.tree.map(lambda x: x.mean(), metric)
            
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
                **loss_info[2],
                **target_stats,
            }
            metric["layout_gradient"] = layout_gradient_metrics
            rng = update_state[-1]

            def eval_layout(eval_env, params, eval_rng):
                num_eval_envs = int(config["EVAL_KWARGS"]["num_envs"])
                num_actors_eval = eval_env.num_agents * num_eval_envs 

                eval_rng, reset_rng = jax.random.split(eval_rng)
                reset_rngs = jax.random.split(reset_rng, num_eval_envs)
                init_obs, init_state = jax.vmap(eval_env.reset, in_axes=(0,))(reset_rngs)
                init_hstate = ScannedRNN.initialize_carry(num_actors_eval, config["GRU_HIDDEN_DIM"])
                init_done = jnp.zeros((num_actors_eval,), dtype=bool)
                init_returns = jnp.zeros((num_eval_envs,), dtype=jnp.float32)
                runner_state = (init_state, init_obs, init_done, init_hstate, init_returns, eval_rng)

                def _eval_step(carry, _):
                    env_state_e, obs_e, done_e, hstate_e, returns_e, rng_e = carry

                    rng_e, _rng_e = jax.random.split(rng_e)
                    obs_batch = batchify(obs_e, eval_env.agents, num_actors_eval)
                    agent_positions = {'agent_0': env_state_e.env_state.agent_pos, 'agent_1': env_state_e.env_state.agent_pos}
                    agent_positions = batchify(agent_positions, eval_env.agents, num_actors_eval)
                    ac_in = (
                        obs_batch[np.newaxis, :],
                        done_e[np.newaxis, :],
                        agent_positions[np.newaxis, :],
                    )
                    hstate_next, pi, _ = network.apply(params, hstate_e, ac_in)
                    pi = distrax.Categorical(logits=pi.logits * config["EVAL_KWARGS"]["beta"])
                    sampled_action = pi.sample(seed=_rng_e)[0]
                    greedy_action = jnp.argmax(pi.probs, axis=-1)[0]
                    action = jnp.where(config["EVAL_KWARGS"]["argmax"], greedy_action, sampled_action)

                    env_act = unbatchify(action, eval_env.agents, num_eval_envs, eval_env.num_agents)
                    env_act = {k: v.squeeze() for k, v in env_act.items()}

                    rng_e, _rng_e = jax.random.split(rng_e)
                    rng_step_e = jax.random.split(_rng_e, num_eval_envs)
                    obs_next, state_next, reward, done, _info = jax.vmap(
                        eval_env.step, in_axes=(0, 0, 0)
                    )(rng_step_e, env_state_e, env_act)

                    done_next = batchify(done, eval_env.agents, num_actors_eval).squeeze()
                    returns_next = returns_e + reward["agent_0"]

                    return (state_next, obs_next, done_next, hstate_next, returns_next, rng_e), None
                
                runner_state, _ = jax.lax.scan(_eval_step, runner_state, None, int(config["EVAL_KWARGS"]["num_steps"]))
                state, obs, done, h_state, returns, rng = runner_state

                return returns.mean()

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

            def eval_layout_xp(eval_env, main_params, bc_params_stacked, eval_rng):
                """Cross-play score for one layout: averaged over human_proxy seeds and over
                which env seat (agent_0/agent_1) the main agent occupies."""
                def _one_seed(bc_params_seed, rng_seed):
                    rng_a, rng_b = jax.random.split(rng_seed)
                    r_main_as_0 = eval_layout_xp_direction(eval_env, main_params, bc_params_seed, rng_a, "agent_0")
                    r_main_as_1 = eval_layout_xp_direction(eval_env, main_params, bc_params_seed, rng_b, "agent_1")
                    return (r_main_as_0 + r_main_as_1) / 2.0

                num_hp_seeds = int(config["EVAL_KWARGS"]["human_proxy_num_seeds"])
                seed_rngs = jax.random.split(eval_rng, num_hp_seeds)
                per_seed_returns = jax.vmap(_one_seed)(bc_params_stacked, seed_rngs)
                return per_seed_returns.mean()

            metric["representation"] = representation_metrics

            if config["ENV_NAME"] == "overcooked" and len(eval_envs) > 0:
                def _do_eval(_):
                    out = {}
                    base = jax.random.fold_in(rng, update_steps)
                    for i, layout_name in enumerate(EVAL_LAYOUTS_9):
                        out[layout_name] = eval_layout(
                            eval_envs[layout_name],
                            train_state.params,
                            jax.random.fold_in(base, i),
                        )
                    out["mean"] = jnp.mean(jnp.stack([out[n] for n in EVAL_LAYOUTS_9]))
                    if eval_xp_enabled:
                        xp_base = jax.random.fold_in(base, 1000)
                        for i, layout_name in enumerate(EVAL_LAYOUTS_9):
                            out[f"{layout_name}_xp"] = eval_layout_xp(
                                eval_envs[layout_name],
                                train_state.params,
                                human_proxy_params[layout_name],
                                jax.random.fold_in(xp_base, i),
                            )
                        out["mean_xp"] = jnp.mean(jnp.stack([out[f"{n}_xp"] for n in EVAL_LAYOUTS_9]))
                    return out

                def _skip_eval(_):
                    out = {n: jnp.array(jnp.nan, dtype=jnp.float32) for n in EVAL_LAYOUTS_9}
                    out["mean"] = jnp.array(jnp.nan, dtype=jnp.float32)
                    if eval_xp_enabled:
                        for n in EVAL_LAYOUTS_9:
                            out[f"{n}_xp"] = jnp.array(jnp.nan, dtype=jnp.float32)
                        out["mean_xp"] = jnp.array(jnp.nan, dtype=jnp.float32)
                    return out

                metric["eval_returns"] = jax.lax.cond(run_eval, _do_eval, _skip_eval, operand=None)

            def callback(metric):
                step = int(metric["update_steps"])
                snapshot_prefixes = (
                    "target_raw/",
                    "target_popart/",
                    "critic/",
                    "td_error/",
                )

                # Average finite scalar training metrics over the interval.
                def _accumulate(key, value):
                    value = float(value)
                    _log_accum["sum"].setdefault(key, 0.0)
                    _log_accum["count"].setdefault(key, 0)
                    if np.isfinite(value):
                        _log_accum["sum"][key] += value
                        _log_accum["count"][key] += 1

                _accumulate("returns", metric["returns"])
                for k, v in metric["loss"].items():
                    if not k.startswith(snapshot_prefixes):
                        _accumulate(k, v)

                if "eval_returns" in metric:
                    if np.isfinite(float(metric["eval_returns"]["mean"])):
                        eval_last = {
                            "mean": float(metric["eval_returns"]["mean"]),
                            **{_ln: float(metric["eval_returns"][_ln]) for _ln in EVAL_LAYOUTS_9},
                        }
                        if eval_xp_enabled and np.isfinite(float(metric["eval_returns"]["mean_xp"])):
                            eval_last["mean_xp"] = float(metric["eval_returns"]["mean_xp"])
                            for _ln in EVAL_LAYOUTS_9:
                                eval_last[f"{_ln}_xp"] = float(metric["eval_returns"][f"{_ln}_xp"])
                        _log_accum["eval_last"] = eval_last

                if config["ENV_NAME"] == "overcooked":
                    ep_rets = np.array(metric["episode_returns_step"])   # (NUM_STEPS, NUM_ENVS)
                    ep_done = np.array(metric["episode_done_step"]).astype(bool)
                    layout_ids = np.array(metric["layout_ids"])  # (NUM_STEPS, NUM_ENVS), pre-step layout
                    for t, e in np.argwhere(ep_done):
                        label = EVAL_LAYOUTS_9[int(layout_ids[t, e])]
                        _log_accum["layout_sum"][label] = _log_accum["layout_sum"].get(label, 0.0) + float(ep_rets[t, e])
                        _log_accum["layout_count"][label] = _log_accum["layout_count"].get(label, 0) + 1

                if (
                    step % LOG_INTERVAL == 0
                    or step == int(config["NUM_UPDATES"]) - 1
                ):
                    log_dict = {
                        "update_step": step,
                        "env_step": int(step * config["NUM_ENVS"] * config["NUM_STEPS"]),
                    }
                    for k, s in _log_accum["sum"].items():
                        cnt = _log_accum["count"][k]
                        log_dict[k] = s / cnt if cnt > 0 else float("nan")

                    # Target/critic/TD statistics are snapshots from this
                    # logging update, for both total and per-layout metrics.
                    for k, v in metric["loss"].items():
                        if k.startswith(snapshot_prefixes):
                            log_dict[k] = float(v)

                    if _log_accum["eval_last"] is not None:
                        log_dict["eval/mean"] = _log_accum["eval_last"]["mean"]
                        for _ln in EVAL_LAYOUTS_9:
                            log_dict[f"eval/{_ln}"] = _log_accum["eval_last"][_ln]
                        if eval_xp_enabled and "mean_xp" in _log_accum["eval_last"]:
                            log_dict["eval_xp/mean"] = _log_accum["eval_last"]["mean_xp"]
                            for _ln in EVAL_LAYOUTS_9:
                                log_dict[f"eval_xp/{_ln}"] = _log_accum["eval_last"][f"{_ln}_xp"]

                    # Expensive diagnostics are evaluated only at this update and
                    # logged directly rather than averaged across the interval.
                    for k, v in metric["layout_gradient"].items():
                        log_dict[k] = float(v)

                    for k, v in metric["representation"].items():
                        if np.isfinite(float(v)):
                            log_dict[k] = float(v)

                    if config["ENV_NAME"] == "overcooked":
                        for name in EVAL_LAYOUTS_9:
                            c = _log_accum["layout_count"].get(name, 0)
                            log_dict[f"train_returns/{name}"] = (
                                _log_accum["layout_sum"][name] / c if c > 0 else float("nan")
                            )

                    # Use the actual PPO update as WandB's global x-axis, rather than
                    # the number of times logging has occurred.
                    wandb.log(log_dict, step=step)

                    _log_accum["sum"] = {}
                    _log_accum["count"] = {}
                    _log_accum["layout_sum"] = {}
                    _log_accum["layout_count"] = {}

            metric["returns"] = returns
            metric["update_steps"] = update_steps

            callback_metric = {
                **metric,
                "episode_returns_step": episode_returns_step,
                "episode_done_step": episode_done_step,
                "layout_ids": _layout_ids_full,
            }

            jax.experimental.io_callback(
                callback, None, callback_metric, ordered=True
            )

            def ckpt_callback(
                params, opt_state_, tx_step, step,
                env_state_, last_obs_, last_done_, hstate_, rng_,
            ):
                step = int(step)
                mid_ckpt_dir = config["MID_CKPT_DIR"]
                os.makedirs(mid_ckpt_dir, exist_ok=True)
                mid_ckpt_path = os.path.join(mid_ckpt_dir, "resume_ckpt.pkl")
                with open(mid_ckpt_path, "wb") as f:
                    pickle.dump({
                        'params': params,
                        'opt_state': opt_state_,
                        'tx_step': tx_step,
                        'final_update_step': step + 1,
                        'wandb_run_id': wandb.run.id,
                        'runner_state': (
                            env_state_, last_obs_, last_done_, hstate_, rng_,
                        ),
                    }, f)

            # Keep resume checkpoints aligned with WandB aggregation boundaries.
            save_ckpt_interval = LOG_INTERVAL
            if save_ckpt_interval > 0:
                run_save_ckpt = jnp.equal(update_steps % save_ckpt_interval, 0)
                jax.lax.cond(
                    run_save_ckpt,
                    lambda _: jax.experimental.io_callback(
                        ckpt_callback, None,
                        train_state.params, train_state.opt_state, train_state.step,
                        update_steps, env_state, last_obs, last_done, hstate, rng,
                        ordered=True,
                    ),
                    lambda _: None,
                    operand=None,
                )

            if save_info is not None:
                num_updates_total = save_info["num_updates"]
                def final_save_callback(params):
                    fp = save_info["filepath"]
                    prefix = save_info["fcp_prefix"]
                    appendage = save_info["finetune_appendage"]
                    rng_key = save_info["rng"]
                    os.makedirs(fp, exist_ok=True)
                    ckpt_path = f"{fp}/{prefix}seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}{appendage}_updates{num_updates_total}.pkl"
                    with open(ckpt_path, "wb") as f:
                        pickle.dump({'key': rng_key, 'params': params, 'update_steps': num_updates_total}, f)
                    print(f"Saved final model to {ckpt_path}")
                    print(f"Finished training for seed {config['SEED']} with ckpt {config['TRAIN_KWARGS']['ckpt_id']}_updates{num_updates_total}")
                    print("--------------------------------")

                is_last_step = jnp.equal(update_steps, num_updates_total - 1)
                jax.lax.cond(
                    is_last_step,
                    lambda _: jax.experimental.io_callback(
                        final_save_callback,
                        None,
                        train_state.params,
                        ordered=True,
                    ),
                    lambda _: None,
                    operand=None,
                )

            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)  # hstate resets automatically
            return (runner_state, update_steps), metric

        initial_done = (
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool)
            if resume_runner_state is None else restored_done
        )
        runner_state = (
            train_state,
            env_state,
            obsv,
            initial_done,
            init_hstate,
            runner_rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, update_step), jnp.arange(remaining_updates), remaining_updates
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_overcooked_CEC_gradient")
def main(config):
    config = OmegaConf.to_container(config)
    config['model_name'] = "CEC_const"
    xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")

    if config['TRAIN_KWARGS']['finetune']:
        config['LR'] = config['LR'] / 10
        finetune_appendage = "_improved_finetune"
        if config['FCP']:
            fcp_prefix = "fcp_"
        else:
            fcp_prefix = ""
    elif config['ENV_NAME'] == 'overcooked':
        fcp_prefix = ""
        finetune_appendage = "_improved"
    else:
        fcp_prefix = ""
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

    resume_xpid = config["RESUME_XPID"]
    active_xpid = resume_xpid if resume_xpid else xpid

    filepath_base = f"ckpts/ippo/{config['ENV_NAME']}"
    if config["ENV_NAME"] == "overcooked":
        filepath_base += f"/{config['ENV_KWARGS']['layout']}"
    filepath_base += f"/ik{config['ENV_KWARGS']['random_reset']}/{config['ENV_KWARGS']['random_reset_fn']}"
    filepath = f"{filepath_base}/{active_xpid}"
    print(f"Working on: \n{filepath}\n")

    config['MID_CKPT_DIR'] = os.path.join(filepath, f"seed{config['SEED']}_mid_ckpts")

    mid_ckpt_path = os.path.join(config['MID_CKPT_DIR'], "resume_ckpt.pkl")
    _has_mid_ckpt = bool(resume_xpid) and os.path.exists(mid_ckpt_path)
    wandb_resume_id = None
    resume_runner_state = None
    resume_train_state_step = None
    if _has_mid_ckpt:
        with open(mid_ckpt_path, "rb") as f:
            _peek = pickle.load(f)
        wandb_resume_id = _peek.get('wandb_run_id', None)

    layout_name = config["ENV_KWARGS"]["layout"]
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
            tags=["IPPO", "RNN", "SP"],
            config=config,
            mode=config["WANDB_MODE"],
            name=f"CEC_gradient_const_{layout_name}_seed{config['SEED']}"
        )

    if not config['TRAIN_KWARGS']['overwrite_ckpt']:
        # check if ckpt exists
        if os.path.exists(f"{filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}{finetune_appendage}.pkl"):
            print(f"Checkpoint {config['TRAIN_KWARGS']['ckpt_id']} already exists, exiting")
            exit(0)

    if _has_mid_ckpt:
        print(f"Found mid-run checkpoint: {mid_ckpt_path}")
        model_params = _peek['params']
        opt_state = _peek.get('opt_state', None)
        resume_train_state_step = _peek.get('tx_step', None)
        resume_runner_state = _peek.get('runner_state', None)
        final_update_step = _peek['final_update_step']
        rng = jax.random.PRNGKey(config["SEED"])
        print(f"Resuming from update step {final_update_step}")
    elif config['TRAIN_KWARGS']['ckpt_id'] > 0:
        print("Loading checkpoint")
        with open(f"{filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id'] - 1}{finetune_appendage}.pkl", "rb") as f:
            previous_ckpt = pickle.load(f)
            model_params = previous_ckpt['params']
            opt_state = None
            final_update_step = previous_ckpt['final_update_step']
            rng = previous_ckpt['key']
            rng, _rng = jax.random.split(jax.random.PRNGKey(rng))

    elif config['TRAIN_KWARGS']['finetune']:
        finetune_filepath =f"ckpts/ippo/{config['ENV_NAME']}"
        if config["ENV_NAME"] == "overcooked":
            finetune_filepath += f"/cramped_room_9"
        if config['FCP']:
            finetune_filepath = f"{finetune_filepath}/ikFalse/{xpid}"
            finetune_ckpt_num = 19 if config['ENV_NAME'] == 'ToyCoop' else 6
        else:
            finetune_filepath = f"{finetune_filepath}/ikTrue/{config['ENV_KWARGS']['random_reset_fn']}/{xpid}"
            finetune_ckpt_num = 29 if config['ENV_NAME'] == 'overcooked' else 19
        print(f"Loading checkpoint for finetuning: {finetune_filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{finetune_ckpt_num}_improved.pkl")
        with open(f"{finetune_filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{finetune_ckpt_num}_improved.pkl", "rb") as f:  # need to resume from last checkpoint
            previous_ckpt = pickle.load(f)
            model_params = previous_ckpt['params']
            opt_state = None
            # final_update_step = previous_ckpt['final_update_step']
            final_update_step = 0
            rng = previous_ckpt['key']
            rng, _rng = jax.random.split(jax.random.PRNGKey(rng))
    else:
        model_params = None
        opt_state = None
        final_update_step = 0
        rng = jax.random.PRNGKey(config["SEED"])

    num_updates = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    save_info = {
        "filepath": filepath,
        "fcp_prefix": fcp_prefix,
        "finetune_appendage": finetune_appendage,
        "rng": rng,
        "num_updates": num_updates,
    }

    print(f"Starting from update step {final_update_step}")
    train_jit = jax.jit(
        make_train(
            config, final_update_step, save_info, opt_state,
            resume_train_state_step,
        ),
        device=jax.devices()[0],
    )
    train_jit(rng, model_params, resume_runner_state)

    jax.effects_barrier()
    jax.clear_caches()
    wandb.finish()


if __name__ == "__main__":
    main()
