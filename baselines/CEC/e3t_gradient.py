"""
Based on PureJaxRL Implementation of PPO.

Note, this file will only work for MPE environments with homogenous agents (e.g. Simple Spread).

"""
import os
import glob
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
import yaml
import time
from baselines.CEC_UED.algo_utils import (
    BCPolicy,
    EVAL_LAYOUTS_9,
    load_human_proxy_params,
    make_eval_envs_overcooked,
)
from baselines.CEC.evaluation_rollout_utils import (
    evaluate_cross_play_layout,
    evaluate_self_play_layout,
)
from baselines.CEC_UED.value_diagnostics import compute_value_diagnostics
from baselines.CEC_UED.representation_metrics import (
    compute_optimizer_update_metrics,
    compute_minibatch_penultimate_metrics,
    empty_penultimate_metrics,
    first_epoch_first_minibatch_indices,
)
from baselines.CEC_UED.evaluation_metrics import (
    EVAL_CRITIC_STAT_NAMES,
    add_evaluation_metrics_to_log_dict,
    empty_evaluation_metrics,
)
from baselines.CEC_UED.sharpness import (
    collect_final_sharpness_batch,
    compute_keskar_sharpness,
)
from baselines.CEC_UED.critic_loss_surface import (
    build_critic_loss_surface_settings,
    save_critic_loss_surface_snapshots,
)
from baselines.CEC_UED.gradient_conflict_utils import (
    compute_layout_gradient_metrics,
    empty_layout_gradient_metrics,
)


LAYOUT_NAMES = tuple(EVAL_LAYOUTS_9)


def _e3t_parameter_groups(config):
    """Return parameter module names for the instantiated E3T network."""
    shared_keys = ["Dense_0", "Dense_1", "ScannedRNN_0"]
    if config["CONV_NET"]:
        shared_keys = ["Conv_0", "Conv_1", *shared_keys]

    # Dense_2..Dense_6 form the model-of-other-agent branch and therefore
    # belong to the actor path. Overcooked adds one actor hidden layer and two
    # critic hidden layers, shifting the subsequent Flax module indices.
    if config["ENV_NAME"] == "overcooked":
        actor_branch_keys = [f"Dense_{index}" for index in range(2, 12)]
        value_branch_keys = [f"Dense_{index}" for index in range(12, 17)]
    else:
        actor_branch_keys = [f"Dense_{index}" for index in range(2, 11)]
        value_branch_keys = [f"Dense_{index}" for index in range(11, 14)]

    shared_keys = tuple(shared_keys)
    return (
        (*shared_keys, *actor_branch_keys),
        (*shared_keys, *value_branch_keys),
        shared_keys,
    )


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
            res = (state.agent_pos, state.goal_pos)
            carry = (i+1,)
            return carry, res
        
        carry, res = jax.lax.scan(gen_held_out_toycoop, (0,), jnp.arange(100), 100)
        ho_agent_pos, ho_goal_pos = res
        
        # Set the held-out states in the environment
        env.held_out_agent_pos = ho_agent_pos
        env.held_out_goal_pos = ho_goal_pos
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    return env


def _is_connected_jax(passable_9x9):
    """Return whether all passable cells form one connected component."""
    flat = passable_9x9.astype(jnp.float32).flatten()
    visited = jnp.zeros_like(flat).at[jnp.argmax(flat)].set(1.0).reshape(9, 9)

    def _spread(current, _):
        neighbors = jnp.maximum(
            jnp.maximum(
                jnp.pad(current[1:], ((0, 1), (0, 0))),
                jnp.pad(current[:-1], ((1, 0), (0, 0))),
            ),
            jnp.maximum(
                jnp.pad(current[:, 1:], ((0, 0), (0, 1))),
                jnp.pad(current[:, :-1], ((0, 0), (1, 0))),
            ),
        )
        return (
            passable_9x9.astype(jnp.float32)
            * jnp.maximum(current, neighbors),
            None,
        )

    visited, _ = jax.lax.scan(_spread, visited, None, 18)
    return jnp.sum(visited) == jnp.sum(passable_9x9.astype(jnp.float32))


def _classify_layout_jax(maze_map_9x9_ch0):
    """Map a 9x9 Overcooked maze to its index in ``LAYOUT_NAMES``."""
    passable = (maze_map_9x9_ch0 == 1) | (maze_map_9x9_ch0 == 10)
    num_passable = passable.sum()
    connected = _is_connected_jax(passable)
    return jnp.where(
        num_passable == 8,
        2,
        jnp.where(
            num_passable == 6,
            jnp.where(connected, 0, 4),
            jnp.where(connected, 3, 1),
        ),
    ).astype(jnp.int32)


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
        batch_size, num_envs, _ = obs.shape
        collect_intermediates = (
            not self.is_initializing()
            and self.is_mutable_collection("intermediates")
        )

        def record_feature_norm(name, features):
            if collect_intermediates:
                feature_vectors = features.reshape(
                    (batch_size, num_envs, -1)
                )
                self.sow(
                    "intermediates",
                    f"feature_norm_{name}",
                    jnp.linalg.norm(feature_vectors, axis=-1),
                )

        if self.config["CONV_NET"]:
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
            record_feature_norm("shared_conv_0", embedding)
            embedding = nn.Conv(
                features=32 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"],
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
        # embedding = nn.Dense(
        #     self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        # )(embedding)
        # embedding = nn.relu(embedding)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)
        record_feature_norm("shared_dense_1", embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        embedding = embedding.reshape((batch_size, num_envs, -1))
        record_feature_norm("shared_recurrent", embedding)
        if not self.is_initializing():
            self.sow("intermediates", "shared_penultimate", embedding)

        #########
        # Model of other agent (patner_prediction module-> 7,8,9 index in original e3t paper)
        #########
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = nn.tanh(prediction_other)
        prediction_other = nn.Dense(self.action_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = prediction_other / jnp.sqrt(jnp.sum(prediction_other**2, axis=-1, keepdims=True) + 1e-10)  # L2 normalization
        other_pi = distrax.Categorical(logits=prediction_other)

        #########
        # Actor
        #########
        actor_embedding = jnp.concatenate([embedding, prediction_other], axis=-1)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] , kernel_init=orthogonal(2), bias_init=constant(0.0))(
            actor_embedding
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

        return hidden, pi, jnp.squeeze(critic, axis=-1), other_pi


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
    other_action: jnp.ndarray


def e3t_ppo_loss(
    network, params, initial_hstate, traj_batch, advantages, targets, config
):
    """E3T PPO objective shared by training and final sharpness."""
    _, pi, value, other_pi = network.apply(
        params,
        initial_hstate,
        (traj_batch.obs, traj_batch.done, traj_batch.agent_positions),
    )
    log_prob = pi.log_prob(traj_batch.action)
    other_log_prob = other_pi.log_prob(traj_batch.other_action)
    moa_nll_loss = -jnp.mean(other_log_prob)

    value_pred_clipped = traj_batch.value + (
        value - traj_batch.value
    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
    value_loss = 0.5 * jnp.maximum(
        jnp.square(value - targets),
        jnp.square(value_pred_clipped - targets),
    ).mean()

    normalized_advantages = (
        (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    )
    logratio = log_prob - traj_batch.log_prob
    ratio = jnp.exp(logratio)
    actor_loss = -jnp.minimum(
        ratio * normalized_advantages,
        jnp.clip(
            ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
        ) * normalized_advantages,
    ).mean()
    entropy = pi.entropy().mean()
    approx_kl = ((ratio - 1) - logratio).mean()
    clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
    total_loss = (
        actor_loss
        + config["MOA_COEF"] * moa_nll_loss
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    return total_loss, (
        value_loss,
        actor_loss,
        entropy,
        ratio,
        approx_kl,
        clip_frac,
        moa_nll_loss,
    )

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
    surface_layout_name = config["ENV_KWARGS"]["layout"]
    env = initialize_environment(config)
    
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
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
    config["ACTION_DIM"] = env.action_space(env.agents[0]).n
    actor_trunk_keys, value_trunk_keys, shared_trunk_keys = (
        _e3t_parameter_groups(config)
    )

    surface_settings = build_critic_loss_surface_settings(
        config,
        algorithm="E3T",
        layout=surface_layout_name,
        actor_trunk_keys=actor_trunk_keys,
        value_trunk_keys=value_trunk_keys,
        shared_trunk_keys=shared_trunk_keys,
        value_coordinates="raw",
    )

    obs, state = env.reset(jax.random.PRNGKey(0), params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})
    

    env = LogWrapper(env, env_params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})

    eval_envs = make_eval_envs_overcooked(config)
    eval_enabled = (
        config["ENV_NAME"] == "overcooked"
        and all(name in eval_envs for name in EVAL_LAYOUTS_9)
    )
    eval_xp_enabled = (
        eval_enabled
        and bool(config["EVAL_KWARGS"]["eval_xp"])
    )
    human_proxy_params = (
        load_human_proxy_params(
            config["EVAL_KWARGS"]["human_proxy_ckpt_dir"],
            int(config["EVAL_KWARGS"]["human_proxy_num_seeds"]),
            layout_names=EVAL_LAYOUTS_9,
        )
        if eval_xp_enabled
        else {}
    )
    LOG_INTERVAL = max(1, int(config["NUM_UPDATES"]) // 100)

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
            initial_done = jnp.zeros((config["NUM_ACTORS"]), dtype=bool)
        else:
            env_state, obsv, initial_done, init_hstate, runner_rng = resume_runner_state

        # Match ippo_general_gradient.py: aggregate frequent training scalars
        # and emit roughly 100 WandB points over the complete run.
        _log_accum = {
            "sum": {},
            "count": {},
            "layout_sum": {},
            "layout_count": {},
            "eval_last": None,
        }

        def eval_layout_sp(eval_env, params, eval_rng):
            return evaluate_self_play_layout(
                eval_env=eval_env,
                params=params,
                eval_rng=eval_rng,
                network_apply=network.apply,
                initialize_carry=ScannedRNN.initialize_carry,
                batchify=batchify,
                unbatchify=unbatchify,
                num_eval_envs=int(config["EVAL_KWARGS"]["num_envs"]),
                num_steps=int(config["EVAL_KWARGS"]["num_steps"]),
                hidden_dim=config["GRU_HIDDEN_DIM"],
                beta=config["EVAL_KWARGS"]["beta"],
                argmax=config["EVAL_KWARGS"]["argmax"],
                return_critic_stats=True,
                gamma=config["GAMMA"],
            )

        def eval_layout_xp(
            eval_env, main_params, bc_params_stacked, eval_rng,
        ):
            return evaluate_cross_play_layout(
                eval_env=eval_env,
                main_params=main_params,
                bc_params_stacked=bc_params_stacked,
                eval_rng=eval_rng,
                network_apply=network.apply,
                bc_network_apply=bc_network.apply,
                initialize_carry=ScannedRNN.initialize_carry,
                num_eval_envs=int(config["EVAL_KWARGS"]["num_envs"]),
                num_steps=int(config["EVAL_KWARGS"]["num_steps"]),
                hidden_dim=config["GRU_HIDDEN_DIM"],
                beta=config["EVAL_KWARGS"]["beta"],
                argmax=config["EVAL_KWARGS"]["argmax"],
                num_human_proxy_seeds=int(
                    config["EVAL_KWARGS"]["human_proxy_num_seeds"]
                ),
                return_critic_stats=True,
                gamma=config["GAMMA"],
            )

        # TRAIN LOOP
        @scan_tqdm(remaining_updates)
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng, update_step, beta_agent = runner_state

                if config["ENV_NAME"] == "overcooked":
                    pre_maze_map = env_state.env_state.maze_map
                    layout_id = jax.vmap(_classify_layout_jax)(
                        pre_maze_map[:, 4:13, 4:13, 0]
                    )
                    layout_id = jnp.tile(layout_id, [env.num_agents])
                else:
                    layout_id = jnp.zeros(
                        (config["NUM_ACTORS"],), dtype=jnp.int32
                    )

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
                hstate, pi, value, other_pi = network.apply(train_state.params, hstate, ac_in)

                unbatched_logits = unbatchify(pi.logits, env.agents, config["NUM_ENVS"], env.num_agents)
                # agent 0 mask is 1 if beta_agent is 0, 0 otherwise
                agent_0_mask = jnp.where(beta_agent == 0, config["TRAIN_KWARGS"]["e3t_beta"], 1.00)
                agent_1_mask = jnp.where(beta_agent == 1, config["TRAIN_KWARGS"]["e3t_beta"], 1.00)
                multiply_row = lambda x, y: x * y
                unbatched_logits['agent_0'] = jax.vmap(multiply_row)(unbatched_logits['agent_0'], agent_0_mask)
                unbatched_logits['agent_1'] = jax.vmap(multiply_row)(unbatched_logits['agent_1'], agent_1_mask)
                batched_logits = batchify(unbatched_logits, env.agents, config["NUM_ACTORS"])
                pi = distrax.Categorical(logits=batched_logits)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}
                other_env_act = {'agent_0': env_act['agent_1'], 'agent_1': env_act['agent_0']}  # get other agent's action
                other_action = batchify(other_env_act, env.agents, config["NUM_ACTORS"])

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
                    other_action.squeeze(),
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, update_step, beta_agent)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            (train_state, env_state, obsv, done_batch, hstate, rng) = runner_state
            # sample which agent we'll increase beta to
            beta_agent = jax.random.choice(rng, jnp.arange(env.num_agents), shape=(config["NUM_ENVS"],))
            rng, _rng = jax.random.split(rng)
            runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, update_steps, beta_agent)
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng, update_steps, beta_agent = runner_state
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            agent_positions = {'agent_0': env_state.env_state.agent_pos, 'agent_1': env_state.env_state.agent_pos}
            agent_positions = batchify(agent_positions, env.agents, config["NUM_ACTORS"])
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
                agent_positions[np.newaxis, :],
            )
            _, _, last_val, _ = network.apply(train_state.params, hstate, ac_in)
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
            original_params = train_state.params
            actor_layout_ids = traj_batch.layout_id
            environment_layout_ids = actor_layout_ids[:, :config["NUM_ENVS"]]
            diagnostic_layout_names = (
                LAYOUT_NAMES if config["ENV_NAME"] == "overcooked" else ()
            )
            target_stats = compute_value_diagnostics(
                raw_targets=targets,
                critic_targets=targets,
                critic_values=traj_batch.value,
                td_errors=td_errors,
                rewards=traj_batch.reward,
                actor_layout_ids=actor_layout_ids,
                layout_names=diagnostic_layout_names,
            )

            run_eval = jnp.logical_or(
                jnp.logical_or(
                    jnp.equal(update_steps % LOG_INTERVAL, 0),
                    jnp.equal(update_steps, int(config["NUM_UPDATES"]) - 1),
                ),
                jnp.equal(update_steps, update_step),
            )

            gradient_window_steps = int(
                config["GRAD_CONFLICT_WINDOW_STEPS"]
            )

            def _compute_layout_gradient(_):
                gradient_traj = jax.tree.map(
                    lambda value: value[:gradient_window_steps],
                    traj_batch,
                )
                return compute_layout_gradient_metrics(
                    network=network,
                    original_params=original_params,
                    initial_hstate=initial_hstate,
                    traj_batch=gradient_traj,
                    advantages=advantages[:gradient_window_steps],
                    value_targets=targets[:gradient_window_steps],
                    layout_ids_full=(
                        environment_layout_ids[:gradient_window_steps]
                    ),
                    layout_names=LAYOUT_NAMES,
                    config=config,
                    num_agents=env.num_agents,
                )

            if config["ENV_NAME"] == "overcooked":
                layout_gradient_metrics = jax.lax.cond(
                    run_eval,
                    _compute_layout_gradient,
                    lambda _: empty_layout_gradient_metrics(LAYOUT_NAMES),
                    operand=None,
                )
            else:
                layout_gradient_metrics = empty_layout_gradient_metrics(
                    LAYOUT_NAMES
                )

            def _compute_representation_metrics(_):
                first_minibatch_indices = first_epoch_first_minibatch_indices(
                    rng,
                    config["NUM_ACTORS"],
                    config["NUM_MINIBATCHES"],
                )
                representation_hstate = jax.tree.map(
                    lambda h: jnp.take(
                        h, first_minibatch_indices, axis=0
                    ),
                    initial_hstate,
                )
                representation_traj = jax.tree.map(
                    lambda value: jnp.take(
                        value, first_minibatch_indices, axis=1
                    ),
                    traj_batch,
                )
                return compute_minibatch_penultimate_metrics(
                    network,
                    train_state.params,
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

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        return e3t_ppo_loss(
                            network,
                            params,
                            jax.tree.map(lambda h: h.squeeze(), init_hstate),
                            traj_batch,
                            gae,
                            targets,
                            config,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    optimizer_metrics = compute_optimizer_update_metrics(
                        gradients=grads,
                        params=train_state.params,
                        actor_param_keys=actor_trunk_keys,
                        critic_param_keys=value_trunk_keys,
                        shared_param_keys=shared_trunk_keys,
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    loss, loss_aux = total_loss
                    return train_state, (
                        loss,
                        loss_aux,
                        optimizer_metrics,
                    )

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

            # Save the final model immediately after the last optimizer update.
            # This mirrors ippo_general_gradient.py so later evaluation/logging
            # cannot prevent the final checkpoint from being written.
            if save_info is not None:
                num_updates_total = save_info["num_updates"]

                def final_save_callback(params):
                    ckpt_path = save_info["final_ckpt_path"]
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    with open(ckpt_path, "wb") as f:
                        pickle.dump(
                            {
                                "key": save_info["rng"],
                                "params": params,
                                "update_steps": num_updates_total,
                            },
                            f,
                        )
                    print(f"Saved final model to {ckpt_path}")
                    print(
                        f"Finished training for seed {config['SEED']} with "
                        f"ckpt {config['TRAIN_KWARGS']['ckpt_id']}"
                        f"_updates{num_updates_total}"
                    )
                    print("--------------------------------")

                is_last_step = jnp.equal(
                    update_steps, num_updates_total - 1
                )
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

            save_critic_loss_surface_snapshots(
                completed_updates=update_steps + 1,
                total_updates=config["NUM_UPDATES"],
                settings=surface_settings,
                params=train_state.params,
                initial_hstate=initial_hstate,
                traj_batch=traj_batch,
                advantages=advantages,
                targets=targets,
            )

            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            returns = metric["returned_episode_returns"][:, :, 0][
                metric["returned_episode"][:, :, 0].astype(jnp.int32)
            ].mean()

            episode_returns_step = metric["returned_episode_returns"][:, :, 0]
            episode_done_step = metric["returned_episode"][:, :, 0].astype(bool)

            # Keep the scan output and host callback payload scalar-sized.
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
                "moa_nll_loss": loss_info[1][6],
                **loss_info[2],
                **target_stats,
            }
            metric["layout_gradient"] = layout_gradient_metrics
            metric["representation"] = representation_metrics
            rng = update_state[-1]

            if eval_enabled:
                def _do_eval(_):
                    base = jax.random.fold_in(rng, update_steps)
                    out = {}
                    for index, eval_layout_name in enumerate(EVAL_LAYOUTS_9):
                        layout_return, critic_stats = eval_layout_sp(
                            eval_envs[eval_layout_name],
                            train_state.params,
                            jax.random.fold_in(base, index),
                        )
                        out[eval_layout_name] = layout_return
                        for stat_name in EVAL_CRITIC_STAT_NAMES:
                            value = critic_stats[
                                "value_mse" if stat_name == "value_rmse"
                                else "td_error_mse"
                                if stat_name == "td_error_rmse"
                                else stat_name
                            ]
                            out[f"{eval_layout_name}_critic_{stat_name}"] = (
                                jnp.sqrt(value)
                                if stat_name.endswith("_rmse")
                                else value
                            )
                    out["mean"] = jnp.mean(
                        jnp.stack([out[name] for name in EVAL_LAYOUTS_9])
                    )

                    if eval_xp_enabled:
                        xp_base = jax.random.fold_in(base, 1000)
                        for index, eval_layout_name in enumerate(
                            EVAL_LAYOUTS_9
                        ):
                            xp_return, xp_critic_stats = eval_layout_xp(
                                eval_envs[eval_layout_name],
                                train_state.params,
                                human_proxy_params[eval_layout_name],
                                jax.random.fold_in(xp_base, index),
                            )
                            out[f"{eval_layout_name}_xp"] = xp_return
                            for stat_name in EVAL_CRITIC_STAT_NAMES:
                                value = xp_critic_stats[
                                    "value_mse"
                                    if stat_name == "value_rmse"
                                    else "td_error_mse"
                                    if stat_name == "td_error_rmse"
                                    else stat_name
                                ]
                                out[
                                    f"{eval_layout_name}_xp_critic_{stat_name}"
                                ] = (
                                    jnp.sqrt(value)
                                    if stat_name.endswith("_rmse")
                                    else value
                                )
                        out["mean_xp"] = jnp.mean(
                            jnp.stack(
                                [out[f"{name}_xp"] for name in EVAL_LAYOUTS_9]
                            )
                        )
                    return out

                metric["eval_returns"] = jax.lax.cond(
                    run_eval,
                    _do_eval,
                    lambda _: empty_evaluation_metrics(
                        EVAL_LAYOUTS_9,
                        eval_xp_enabled,
                    ),
                    operand=None,
                )

            def callback(metric):
                step = int(metric["update_steps"])
                snapshot_prefixes = (
                    "target_raw/", "target_popart/", "critic/", "td_error/",
                )

                def _accumulate(key, value):
                    value = float(value)
                    _log_accum["sum"].setdefault(key, 0.0)
                    _log_accum["count"].setdefault(key, 0)
                    if np.isfinite(value):
                        _log_accum["sum"][key] += value
                        _log_accum["count"][key] += 1

                _accumulate("returns", metric["returns"])
                for key, value in metric["loss"].items():
                    if not key.startswith(snapshot_prefixes):
                        _accumulate(key, value)

                if "eval_returns" in metric:
                    eval_metrics = metric["eval_returns"]
                    if np.isfinite(float(eval_metrics["mean"])):
                        _log_accum["eval_last"] = {
                            key: float(value)
                            for key, value in eval_metrics.items()
                            if np.isfinite(float(value))
                        }

                if config["ENV_NAME"] == "overcooked":
                    episode_returns_array = np.asarray(
                        metric["episode_returns_step"]
                    )
                    episode_done_array = np.asarray(
                        metric["episode_done_step"]
                    ).astype(bool)
                    layout_ids_array = np.asarray(metric["layout_ids"])
                    for time_index, env_index in np.argwhere(
                        episode_done_array
                    ):
                        name = EVAL_LAYOUTS_9[
                            int(layout_ids_array[time_index, env_index])
                        ]
                        _log_accum["layout_sum"][name] = (
                            _log_accum["layout_sum"].get(name, 0.0)
                            + float(
                                episode_returns_array[time_index, env_index]
                            )
                        )
                        _log_accum["layout_count"][name] = (
                            _log_accum["layout_count"].get(name, 0) + 1
                        )

                if (
                    step % LOG_INTERVAL == 0
                    or step == int(config["NUM_UPDATES"]) - 1
                    or step == update_step
                ):
                    log_data = {
                        "update_step": step,
                        "env_step": int(
                            step
                            * config["NUM_ENVS"]
                            * config["NUM_STEPS"]
                        ),
                    }
                    for key, value_sum in _log_accum["sum"].items():
                        count = _log_accum["count"][key]
                        log_data[key] = (
                            value_sum / count if count > 0 else float("nan")
                        )
                    for key, value in metric["loss"].items():
                        if key.startswith(snapshot_prefixes):
                            log_data[key] = float(value)
                    for key, value in metric["representation"].items():
                        if np.isfinite(float(value)):
                            log_data[key] = float(value)
                    for key, value in metric["layout_gradient"].items():
                        log_data[key] = float(value)
                    add_evaluation_metrics_to_log_dict(
                        log_data,
                        _log_accum["eval_last"],
                        EVAL_LAYOUTS_9,
                        eval_xp_enabled,
                    )
                    for name in EVAL_LAYOUTS_9:
                        count = _log_accum["layout_count"].get(name, 0)
                        log_data[f"train_returns/{name}"] = (
                            _log_accum["layout_sum"].get(name, 0.0) / count
                            if count > 0
                            else float("nan")
                        )
                    wandb.log(log_data, step=step)
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
                "layout_ids": environment_layout_ids,
            }
            jax.experimental.io_callback(
                callback, None, callback_metric, ordered=True,
            )

            def checkpoint_callback(
                params,
                opt_state_,
                tx_step,
                step,
                env_state_,
                last_obs_,
                last_done_,
                hstate_,
                rng_,
            ):
                step = int(step)
                mid_ckpt_dir = config["MID_CKPT_DIR"]
                os.makedirs(mid_ckpt_dir, exist_ok=True)
                mid_ckpt_path = os.path.join(
                    mid_ckpt_dir, "resume_ckpt.pkl"
                )
                with open(mid_ckpt_path, "wb") as f:
                    pickle.dump(
                        {
                            "params": params,
                            "opt_state": opt_state_,
                            "tx_step": tx_step,
                            "final_update_step": step + 1,
                            "wandb_run_id": wandb.run.id,
                            "runner_state": (
                                env_state_,
                                last_obs_,
                                last_done_,
                                hstate_,
                                rng_,
                            ),
                        },
                        f,
                    )

            is_scheduled_ckpt = jnp.equal(update_steps % LOG_INTERVAL, 0)
            jax.lax.cond(
                is_scheduled_ckpt,
                lambda _: jax.experimental.io_callback(
                    checkpoint_callback,
                    None,
                    train_state.params,
                    train_state.opt_state,
                    train_state.step,
                    update_steps,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    rng,
                    ordered=True,
                ),
                lambda _: None,
                operand=None,
            )
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)  # hstate resets automatically
            return (runner_state, update_steps), metric

        runner_state = (
            train_state,
            env_state,
            obsv,
            initial_done,
            init_hstate,
            runner_rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step,
            (runner_state, update_step),
            jnp.arange(remaining_updates),
            remaining_updates,
        )
        final_runner_state, final_update_count = runner_state
        final_train_state = final_runner_state[0]
        return {
            "runner_state": runner_state,
            "sharpness_params": final_train_state.params,
            "sharpness_batch": collect_final_sharpness_batch(
                env,
                network,
                final_runner_state,
                final_update_count,
                config,
                batchify,
                unbatchify,
                include_other_action=True,
            ),
        }

    return train


@hydra.main(version_base=None, config_path="repro_config", config_name="e3t_final_baseline")
def main(config):
    config = OmegaConf.to_container(config)
    config["model_name"] = "E3T"
    xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")

    if config['TRAIN_KWARGS']['finetune']:
        config['LR'] = config['LR'] / 10
        finetune_appendage = "_e3t_finetune"
        fcp_prefix = "fcp_"
    elif config['ENV_NAME'] == 'overcooked':
        fcp_prefix = ""
        finetune_appendage = "_e3t"
    else:
        fcp_prefix = ""
        finetune_appendage = "_e3t"

    save_variant = "e3t"
    if config["TRAIN_KWARGS"]["finetune"]:
        save_variant += "_finetune"

    if config["WANDB_MODE"] == "online":
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])

    resume_xpid = config.get("RESUME_XPID")
    active_xpid = resume_xpid if resume_xpid else xpid

    filepath_base = f"ckpts/e3t/{config['ENV_NAME']}"
    if config["ENV_NAME"] == "overcooked":
        filepath_base += f"/{config['ENV_KWARGS']['layout']}"
    filepath_base += (
        f"/ik{config['ENV_KWARGS']['random_reset']}"
        f"/{config['ENV_KWARGS']['random_reset_fn']}"
    )
    filepath = f"{filepath_base}/{active_xpid}"
    config['filepath'] = filepath
    config['fcp_prefix'] = fcp_prefix
    config["MID_CKPT_DIR"] = os.path.join(
        filepath, f"seed{config['SEED']}_mid_ckpts"
    )
    print(f"Working on: \n{filepath}\n")

    mid_ckpt_path = os.path.join(
        config["MID_CKPT_DIR"], "resume_ckpt.pkl"
    )
    has_mid_ckpt = bool(resume_xpid) and os.path.exists(mid_ckpt_path)
    checkpoint_data = None
    wandb_resume_id = None
    if has_mid_ckpt:
        with open(mid_ckpt_path, "rb") as f:
            checkpoint_data = pickle.load(f)
        wandb_resume_id = checkpoint_data.get("wandb_run_id")

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
            tags=["E3T", "RNN", "SP"],
            config=config,
            mode=config["WANDB_MODE"],
            name=(
                f"e3t_{config['ENV_KWARGS']['layout']}"
                f"_seed{config['SEED']}"
            ),
        )

    num_updates = int(
        config["TOTAL_TIMESTEPS"]
        // config["NUM_STEPS"]
        // config["NUM_ENVS"]
    )
    final_ckpt_path = os.path.join(
        filepath,
        f"{fcp_prefix}seed{config['SEED']}_ckpt"
        f"{config['TRAIN_KWARGS']['ckpt_id']}_{save_variant}"
        f"_updates{num_updates}.pkl",
    )
    legacy_final_ckpt_path = os.path.join(
        filepath,
        f"{fcp_prefix}seed{config['SEED']}_ckpt"
        f"{config['TRAIN_KWARGS']['ckpt_id']}{finetune_appendage}.pkl",
    )
    if not config['TRAIN_KWARGS']['overwrite_ckpt']:
        if (
            os.path.exists(final_ckpt_path)
            or os.path.exists(legacy_final_ckpt_path)
        ):
            print(f"Checkpoint {config['TRAIN_KWARGS']['ckpt_id']} already exists, exiting")
            return

    resume_runner_state = None
    resume_train_state_step = None
    if has_mid_ckpt:
        print(f"Found mid-run checkpoint: {mid_ckpt_path}")
        model_params = checkpoint_data["params"]
        opt_state = checkpoint_data.get("opt_state")
        resume_train_state_step = checkpoint_data.get("tx_step")
        resume_runner_state = checkpoint_data.get("runner_state")
        final_update_step = checkpoint_data["final_update_step"]
        rng = jax.random.PRNGKey(config["SEED"])
        print(f"Resuming from update step {final_update_step}")
    elif config['TRAIN_KWARGS']['ckpt_id'] > 0:
        print("Loading checkpoint")
        previous_ckpt_prefix = (
            f"{fcp_prefix}seed{config['SEED']}_ckpt"
            f"{config['TRAIN_KWARGS']['ckpt_id'] - 1}"
        )
        checkpoint_candidates = [
            os.path.join(
                filepath,
                f"{previous_ckpt_prefix}_{save_variant}"
                f"_updates{num_updates}.pkl",
            ),
            os.path.join(
                filepath,
                f"{previous_ckpt_prefix}{finetune_appendage}.pkl",
            ),
        ]
        checkpoint_candidates.extend(
            sorted(
                glob.glob(
                    os.path.join(
                        filepath,
                        f"{previous_ckpt_prefix}_{save_variant}"
                        "_updates*.pkl",
                    )
                ),
                reverse=True,
            )
        )
        previous_ckpt_path = next(
            (path for path in checkpoint_candidates if os.path.exists(path)),
            None,
        )
        if previous_ckpt_path is None:
            raise FileNotFoundError(
                "Previous E3T checkpoint was not found under "
                f"{filepath!r}. Set RESUME_XPID to the run directory that "
                "contains the previous checkpoint."
            )
        with open(previous_ckpt_path, "rb") as f:
            previous_ckpt = pickle.load(f)
        model_params = previous_ckpt['params']
        opt_state = None
        final_update_step = previous_ckpt.get(
            'final_update_step', previous_ckpt.get('update_steps', 0)
        )
        rng = previous_ckpt['key']

    elif config['TRAIN_KWARGS']['finetune']:
        finetune_filepath = f"ckpts/e3t/{config['ENV_NAME']}"
        if config["ENV_NAME"] == "overcooked":
            finetune_filepath += f"/{config['ENV_KWARGS']['layout']}"
        finetune_filepath = f"{finetune_filepath}/ikFalse"
        fcp_ckpt_num = 19 if config['ENV_NAME'] == 'ToyCoop' else 6
        print("Loading fcp checkpoint for finetuning")
        with open(f"{finetune_filepath}/{fcp_prefix}seed{config['SEED']}_e3t_ckpt{fcp_ckpt_num}.pkl", "rb") as f:  # need to resume from last checkpoint
            previous_ckpt = pickle.load(f)
            model_params = previous_ckpt['params']
            opt_state = None
            final_update_step = 0
            rng = previous_ckpt['key']
    else:
        model_params = None
        opt_state = None
        final_update_step = 0
        rng = jax.random.PRNGKey(config["SEED"])

    save_info = {
        "rng": rng,
        "num_updates": num_updates,
        "final_ckpt_path": final_ckpt_path,
    }

    print(f"Starting from update step {final_update_step}")
    train_jit = jax.jit(
        make_train(
            config,
            final_update_step,
            save_info,
            opt_state,
            resume_train_state_step,
        ),
        device=jax.devices()[0],
    )
    train_output = train_jit(rng, model_params, resume_runner_state)
    jax.effects_barrier()

    print("Computing final Keskar sharpness with L-BFGS-B...")
    sharpness_network = ActorCriticRNN(
        int(config["ACTION_DIM"]), config=config
    )
    sharpness_batch = train_output["sharpness_batch"]

    def sharpness_loss(params):
        loss, _ = e3t_ppo_loss(
            sharpness_network,
            params,
            sharpness_batch.initial_hstate,
            sharpness_batch,
            sharpness_batch.advantages,
            sharpness_batch.targets,
            config,
        )
        return loss

    sharpness_metrics = compute_keskar_sharpness(
        sharpness_loss,
        train_output["sharpness_params"],
        config["SHARPNESS"]["EPSILONS"],
        maxiter=int(config["SHARPNESS"]["LBFGSB_MAXITER"]),
    )
    wandb.log(sharpness_metrics, step=num_updates)
    print(
        "Final sharpness: "
        + ", ".join(
            f"{key}={value:.6g}"
            for key, value in sharpness_metrics.items()
            if key.startswith("sharpness/keskar_")
        )
    )
    jax.clear_caches()
    wandb.finish()
    

if __name__ == "__main__":
    main()
