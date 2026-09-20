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
from jaxmarl.environments.overcooked.layouts import make_counter_circuit_9x9, make_forced_coord_9x9, make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9

import wandb
import functools
import pdb
from jax_tqdm import scan_tqdm
import time
import yaml
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from flax import struct
import chex
import imageio
from algo_utils import init_hdf5, save_to_hdf5, make_eval_envs_overcooked, classify_layout, EVAL_LAYOUTS_9

def initialize_environment(config):
    layout_name = config["ENV_KWARGS"]["layout"]
    config['layout_name'] = layout_name
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


class CriticHead(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, embedding):
        x = nn.Dense(self.config["FC_DIM_SIZE"] * 2, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        x = nn.relu(x)
        x = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        if self.config["ENV_NAME"] == "overcooked":
            x = nn.Dense(self.config["FC_DIM_SIZE"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
            x = nn.relu(x)
            x = nn.Dense(self.config["FC_DIM_SIZE"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
            x = nn.relu(x)
        x = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return x.squeeze(-1)


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
            embedding = nn.Conv(
                # features=32 if "9" in self.config['layout_name'] and self.config["ENV_NAME"] == "overcooked") else self.config["FC_DIM_SIZE"],
                features=32,
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
            # self.config["FC_DIM_SIZE"] * 2 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"], 
            self.config["FC_DIM_SIZE"] * 2,
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        if self.config["LSTM"]:
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN()(hidden, rnn_in)
        else:
            embedding = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
            embedding = nn.relu(embedding)
        embedding = embedding.reshape((batch_size, num_envs, -1))

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
        # Critic — 5 independent heads, one per base layout
        #########
        values_all = jnp.stack(
            [CriticHead(config=self.config, name=f"critic_{i}")(embedding) for i in range(5)],
            axis=-1,
        )  # (*batch, 5)

        return hidden, pi, values_all


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
    layout_id: jnp.ndarray  # int32 in [0,4], one per actor

@struct.dataclass
class FilteredState:
    agent_dir_idx: chex.Array
    agent_inv: chex.Array
    maze_map: chex.Array


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config, update_step=0, save_info=None, opt_state=None):
    # env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = initialize_environment(config)
    agent_view_size = env.agent_view_size
    viz = OvercookedVisualizer()
    
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
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )
    config["obs_dim"] = env.observation_space(env.agents[0]).shape

    obs, state = env.reset(jax.random.PRNGKey(0), params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})

    if config["save_env_state"]:
        state_names  = ["agent_dir_idx", "agent_inv", "maze_map"]
        state_shapes = {k: getattr(state, k).shape for k in state_names}
        state_dtypes = {k: getattr(state, k).dtype for k in state_names}
        save_every = int(config["save_env_state_interval"])
        data_len = (int(config["NUM_UPDATES"]) + save_every - 1) // save_every

        save_path = f"/app/baselines/CEC_UED/dataset/train_env_states.h5"
        init_hdf5(save_path, state_names, state_shapes, state_dtypes, int(data_len), config["NUM_ENVS"])


    env = LogWrapper(env, env_params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})

    eval_envs = make_eval_envs_overcooked(config)

    def linear_schedule(count):
        frac = (
            1.0
            - ((count + resume_update_step) // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["MAX_TRAIN_UPDATES"]
        )
        frac = jnp.maximum(1e-9, frac)
        return config["LR"] * frac

    def train(rng, model_params=None, update_step=0):
        save_xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
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
            train_state = train_state.replace(opt_state=opt_state)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # TRAIN LOOP
        @scan_tqdm(int(config["NUM_UPDATES"]))
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng, update_step = runner_state

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
                hstate, pi, values_all = network.apply(train_state.params, hstate, ac_in)
                # classify layout per env, tile to actors
                _cur_maze = env_state.env_state.maze_map[:, 4:13, 4:13, 0]  # (NUM_ENVS, 9, 9)
                _layout_ids = jax.vmap(_classify_layout_jax)(_cur_maze)     # (NUM_ENVS,)
                _actor_layout_id = jnp.tile(_layout_ids, env.num_agents)    # (NUM_ACTORS,)
                # select the value from the correct head per actor
                value = values_all.squeeze(0)[jnp.arange(config["NUM_ACTORS"]), _actor_layout_id]
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

                filtered_state = {
                    "agent_dir_idx": env_state.env_state.agent_dir_idx,
                    "agent_inv": env_state.env_state.agent_inv,
                    "maze_map": env_state.env_state.maze_map}

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
                    _actor_layout_id,
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, update_step)
                return runner_state, (
                    transition,
                    FilteredState(
                        filtered_state["agent_dir_idx"],
                        filtered_state["agent_inv"],
                        filtered_state["maze_map"],
                    ),
                )

            initial_hstate = runner_state[-2]
            (train_state, env_state, obsv, done_batch, hstate, rng) = runner_state
            runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, update_steps)
            runner_state, (traj_batch, train_filtered_state) = jax.lax.scan(
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
            _, _, last_values_all = network.apply(train_state.params, hstate, ac_in)
            _last_maze = env_state.env_state.maze_map[:, 4:13, 4:13, 0]  # (NUM_ENVS, 9, 9)
            _last_layout_ids = jax.vmap(_classify_layout_jax)(_last_maze)  # (NUM_ENVS,)
            _last_actor_layout_id = jnp.tile(_last_layout_ids, env.num_agents)  # (NUM_ACTORS,)
            last_val = last_values_all.squeeze(0)[jnp.arange(config["NUM_ACTORS"]), _last_actor_layout_id]

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

            # ── per-layout gradient conflict ──────────────────────────────
            _LAYOUT_NAMES = [
                "cramped_room_9", "asymm_advantages_9", "coord_ring_9",
                "counter_circuit_9", "forced_coord_9",
            ]
            original_params = train_state.params

            # subsample: use only the first _GC_STEPS steps to reduce activation memory
            _GC_STEPS = config["GRAD_CONFLICT_WINDOW_STEPS"]
            _gc_traj = jax.tree.map(lambda x: x[:_GC_STEPS], traj_batch)
            _gc_adv  = advantages[:_GC_STEPS]
            _gc_tgt  = targets[:_GC_STEPS]

            # classify each (step, env) from maze_map → (_GC_STEPS, NUM_ENVS)
            _n_se = _GC_STEPS * config["NUM_ENVS"]
            _active_flat = train_filtered_state.maze_map[:_GC_STEPS, :, 4:13, 4:13, 0].reshape(_n_se, 9, 9)
            _layout_ids = jax.vmap(_classify_layout_jax)(_active_flat).reshape(
                _GC_STEPS, config["NUM_ENVS"]
            )
            # tile to actors: [agent0_envs..., agent1_envs...] → (_GC_STEPS, NUM_ACTORS)
            _actor_layout = jnp.tile(_layout_ids, [1, env.num_agents])

            def _tdot(g1, g2):
                return sum(
                    jnp.sum(a * b)
                    for a, b in zip(jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(g2))
                )

            def _tnorm2(g):
                return sum(jnp.sum(a ** 2) for a in jax.tree_util.tree_leaves(g))

            # Per-loss-type scalar accumulators: norms_sq[lid], dots[(i,j)], prev[lid]
            # actor and value are logged; entropy is excluded (less interpretable).
            _gc_state = {
                'actor': {'norms_sq': [], 'dots': {}, 'prev': []},
                'value': {'norms_sq': [], 'dots': {}, 'prev': []},
            }

            for _lid in range(5):
                _mask = (_actor_layout == _lid).astype(jnp.float32)  # (_GC_STEPS, NUM_ACTORS)
                _cnt = _mask.sum() + 1e-8

                # Single forward pass per layout; 2 backward passes via vjp cotangents.
                def _fwd(p, mask=_mask, cnt=_cnt, lid=_lid):
                    _, pi, values_all = jax.checkpoint(network.apply)(
                        p, initial_hstate,
                        (_gc_traj.obs, _gc_traj.done, _gc_traj.agent_positions),
                    )
                    lp = pi.log_prob(_gc_traj.action)
                    value = values_all[..., lid]  # use this layout's value head
                    adv_mean = (_gc_adv * mask).sum() / cnt
                    adv_std = jnp.sqrt(((_gc_adv - adv_mean) ** 2 * mask).sum() / cnt + 1e-8)
                    gae = (_gc_adv - adv_mean) / (adv_std + 1e-8)
                    ratio = jnp.exp(lp - _gc_traj.log_prob)
                    al = -(jnp.minimum(
                        ratio * gae,
                        jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]) * gae,
                    ) * mask).sum() / cnt
                    vpc = _gc_traj.value + (value - _gc_traj.value).clip(
                        -config["CLIP_EPS"], config["CLIP_EPS"]
                    )
                    vl = 0.5 * (jnp.maximum(
                        jnp.square(value - _gc_tgt), jnp.square(vpc - _gc_tgt)
                    ) * mask).sum() / cnt
                    return al, vl

                _, _vjp_fn = jax.vjp(_fwd, original_params)
                # cotangent (1, 0) → actor gradient; (0, 1) → value gradient
                _g_actor, = _vjp_fn((1.0, 0.0))
                _g_value, = _vjp_fn((0.0, 1.0))

                for _loss_type, _g in [('actor', _g_actor), ('value', _g_value)]:
                    _s = _gc_state[_loss_type]
                    _s['norms_sq'].append(_tnorm2(_g))
                    for _prev_lid, _g_prev in enumerate(_s['prev']):
                        _s['dots'][(_prev_lid, _lid)] = _tdot(_g_prev, _g)
                    _s['prev'].append(_g)

            grad_conflict = {}

            for _loss_type, _s in _gc_state.items():
                # per-layout gradient norms
                for _i in range(5):
                    grad_conflict[f"grad_conflict_{_loss_type}/norm/{_LAYOUT_NAMES[_i]}"] = (
                        jnp.sqrt(_s['norms_sq'][_i])
                    )
                # pairwise cosine similarities
                for _i in range(5):
                    for _j in range(_i + 1, 5):
                        cos = _s['dots'][(_i, _j)] / (
                            jnp.sqrt(_s['norms_sq'][_i] * _s['norms_sq'][_j]) + 1e-8
                        )
                        grad_conflict[
                            f"grad_conflict_{_loss_type}/{_LAYOUT_NAMES[_i]}_vs_{_LAYOUT_NAMES[_j]}"
                        ] = cos
                # joint update alignment: cos(g_i, g_all) where g_all = sum of all 5 layout gradients
                # measures how much each layout's update aligns with the combined training direction
                _g_all = jax.tree.map(lambda *gs: sum(gs), *_s['prev'])
                _norm_all_sq = _tnorm2(_g_all)
                for _i in range(5):
                    _dot_i_all = _tdot(_s['prev'][_i], _g_all)
                    _align = _dot_i_all / (jnp.sqrt(_s['norms_sq'][_i] * _norm_all_sq) + 1e-8)
                    grad_conflict[f"grad_conflict_{_loss_type}/alignment/{_LAYOUT_NAMES[_i]}"] = _align
                    # leave-one-out: cos(g_i, g_all - g_i) — removes self-contribution
                    _g_others = jax.tree.map(lambda ga, gi: ga - gi, _g_all, _s['prev'][_i])
                    _norm_others_sq = _tnorm2(_g_others)
                    _dot_i_others = _tdot(_s['prev'][_i], _g_others)
                    _align_loo = _dot_i_others / (
                        jnp.sqrt(_s['norms_sq'][_i] * _norm_others_sq) + 1e-8
                    )
                    grad_conflict[f"grad_conflict_{_loss_type}/alignment_loo/{_LAYOUT_NAMES[_i]}"] = _align_loo
            # ── end gradient conflict ──────────────────────────────────────

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, values_all = network.apply(
                            params,
                            jax.tree.map(lambda h: h.squeeze(), init_hstate),
                            (traj_batch.obs, traj_batch.done, traj_batch.agent_positions),
                        )
                        # values_all: (NUM_STEPS, MINIBATCH_SIZE, 5) — select per-layout head
                        value = jnp.take_along_axis(
                            values_all, traj_batch.layout_id[..., None], axis=-1
                        ).squeeze(-1)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
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
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

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
                **grad_conflict,
            }
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
                    env_act = {k: v.squeeze() for k, v in env_act.items()} #TODO: check

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

            run_eval = jnp.equal(update_steps % config["EVAL_KWARGS"]["eval_interval"], 0)

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
                    return out

                def _skip_eval(_):
                    out = {n: jnp.array(jnp.nan, dtype=jnp.float32) for n in EVAL_LAYOUTS_9}
                    out["mean"] = jnp.array(jnp.nan, dtype=jnp.float32)
                    return out

                metric["eval_returns"] = jax.lax.cond(run_eval, _do_eval, _skip_eval, operand=None)

            def callback(metric):
                log_dict = {
                    "returns": metric["returns"],
                    "env_step": int(metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]),
                    **metric["loss"],
                }
                if "eval_returns" in metric:
                    if np.isfinite(float(metric["eval_returns"]["mean"])):
                        log_dict["eval/mean"] = float(metric["eval_returns"]["mean"])
                        for _ln in EVAL_LAYOUTS_9:
                            log_dict[f"eval/{_ln}"] = float(metric["eval_returns"][_ln])

                if config["ENV_NAME"] == "overcooked":
                    maze_map = np.array(metric["env_state"].env_state.maze_map)  # (num_envs, 17, 17, 3)
                    active = maze_map[:, 4:13, 4:13, 0]  # (num_envs, 9, 9)
                    layout_counts = {name: 0 for name in EVAL_LAYOUTS_9}
                    for e in range(maze_map.shape[0]):
                        label = classify_layout(active[e])
                        if label in layout_counts:
                            layout_counts[label] += 1
                    total = maze_map.shape[0]
                    for name in EVAL_LAYOUTS_9:
                        log_dict[f"layout_ratio/{name}"] = layout_counts[name] / total

                    ep_rets = np.array(metric["episode_returns_step"])   # (NUM_STEPS, NUM_ENVS)
                    ep_done = np.array(metric["episode_done_step"]).astype(bool)
                    step_maze = np.array(metric["train_filtered_state"].maze_map)  # (NUM_STEPS, NUM_ENVS, H, W, C)
                    layout_returns = {name: [] for name in EVAL_LAYOUTS_9}
                    for t in range(ep_done.shape[0]):
                        for e in range(ep_done.shape[1]):
                            if ep_done[t, e]:
                                label = classify_layout(step_maze[t, e, 4:13, 4:13, 0])
                                layout_returns[label].append(float(ep_rets[t, e]))
                    for name in EVAL_LAYOUTS_9:
                        returns_for_layout = layout_returns[name]
                        log_dict[f"train_returns/{name}"] = (
                            float(np.mean(returns_for_layout))
                            if len(returns_for_layout) > 0
                            else float("nan")
                        )
                wandb.log(log_dict)
                
                step = int(metric["update_steps"])
                def save_frames(filtered_state, step, file_path):
                    frames = [viz.custom_get_frame(jax.tree.map(lambda x: x[step], filtered_state), agent_view_size)
                        for step in range(config["NUM_STEPS"])]
                    
                    os.makedirs(file_path, exist_ok=True)
                    filename = f"step_{step:03}_animation.gif"
                    save_path = os.path.join(file_path, filename)
                    imageio.mimsave(save_path, frames, 'GIF', duration=0.5)
            
                if config["save_frames"] and (step % config["save_frames_interval"] == 0):
                    save_frames(metric["train_filtered_state"], step, f"/app/viz_results/{config['ENV_NAME']}/{save_xpid}/train_images")
                
                if config["save_env_state"] and (step % config["save_env_state_interval"] == 0):
                    save_slot = step // int(config["save_env_state_interval"])
                    arrays = [getattr(metric["env_state"].env_state, k) for k in state_names]
                    save_to_hdf5(save_path, state_names, arrays, save_slot, step)

            metric["returns"] = returns
            metric["update_steps"] = update_steps

            callback_metric = {
                **metric,
                "train_filtered_state": train_filtered_state,
                "env_state": env_state,
                "episode_returns_step": episode_returns_step,
                "episode_done_step": episode_done_step,
            }
            jax.experimental.io_callback(callback, None, callback_metric)

            def ckpt_callback(params, opt_state_, tx_step, step):
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
                    }, f)

            save_ckpt_interval = int(config.get("SAVE_CKPT_INTERVAL", 5000))
            if save_ckpt_interval > 0:
                run_save_ckpt = jnp.equal(update_steps % save_ckpt_interval, 0)
                jax.lax.cond(
                    run_save_ckpt,
                    lambda _: jax.experimental.io_callback(ckpt_callback, None, train_state.params, train_state.opt_state, train_state.step, update_steps),
                    lambda _: None,
                    operand=None,
                )

            if save_info is not None:
                num_updates_total = save_info["num_updates"]
                def final_save_callback(params, step):
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
                    print(f"--------------------------------")

                is_last_step = jnp.equal(update_steps, num_updates_total - 1)
                jax.lax.cond(
                    is_last_step,
                    lambda _: jax.experimental.io_callback(final_save_callback, None, train_state.params, update_steps),
                    lambda _: None,
                    operand=None,
                )

            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)  # hstate resets automatically
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, update_step), jnp.arange(int(config["NUM_UPDATES"])), int(config["NUM_UPDATES"])
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_overcooked_CEC_gradient")
def main(config):
    config = OmegaConf.to_container(config)
    config['model_name'] = "CEC_INDI"
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
    
    resume_xpid = config.get("RESUME_XPID", "")
    active_xpid = resume_xpid if resume_xpid else xpid

    layout_name = config["ENV_KWARGS"]["layout"]
    filepath = f"ckpts/ippo/{config['ENV_NAME']}"
    if config["ENV_NAME"] == "overcooked":
        filepath += f"/{config['ENV_KWARGS']['layout']}"
    filepath = f'{filepath}/ik{config["ENV_KWARGS"]["random_reset"]}/{config["ENV_KWARGS"]["random_reset_fn"]}/{active_xpid}'
    print(f"Working on: \n{filepath}\n")

    config['MID_CKPT_DIR'] = os.path.join(filepath, f"seed{config['SEED']}_mid_ckpts")
    mid_ckpt_path = os.path.join(config['MID_CKPT_DIR'], "resume_ckpt.pkl")
    _has_mid_ckpt = bool(resume_xpid) and os.path.exists(mid_ckpt_path)
    wandb_resume_id = None
    if _has_mid_ckpt:
        with open(mid_ckpt_path, "rb") as f:
            _peek = pickle.load(f)
        wandb_resume_id = _peek.get('wandb_run_id', None)

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
            name=f"CEC_gradient_indi_{layout_name}_seed{config['SEED']}"
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
    train_jit = jax.jit(make_train(config, final_update_step, save_info, opt_state), device=jax.devices()[0])
    out = train_jit(rng, model_params, final_update_step)

    jax.effects_barrier()
    jax.clear_caches()
    wandb.finish()

if __name__ == "__main__":
    main()
