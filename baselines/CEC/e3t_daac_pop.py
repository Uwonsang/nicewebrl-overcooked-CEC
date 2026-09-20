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
import yaml
import time
import flax.core
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
    def __call__(self, hidden, x, return_advantages=False):
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
        # embedding = nn.Dense(
        #     self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        # )(embedding)
        # embedding = nn.relu(embedding)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

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

        # DAAC auxiliary task: predict the normalized GAE for every action
        # from the same policy features used by the actor output. The loss
        # selects only the prediction for the sampled action.
        advantage_predictions = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="advantage_output",
        )(actor_mean)

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
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="critic_output",
        )(
            critic
        )

        outputs = (hidden, pi, jnp.squeeze(critic, axis=-1), other_pi)
        if return_advantages:
            return outputs + (advantage_predictions,)
        return outputs


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
    other_action: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config, update_step=0):
    config.setdefault("DAAC_ADV_COEF", 0.25)
    # env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = initialize_environment(config)
    
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    resume_update_step = update_step * (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
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
    

    env = LogWrapper(env, env_params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})

    eval_envs = make_eval_envs_overcooked(config)
    eval_all_layouts = (
        config["ENV_NAME"] == "overcooked"
        and bool(config["ENV_KWARGS"]["random_reset"])
        and config["ENV_KWARGS"]["random_reset_fn"] == "reset_all"
        and bool(config["ENV_KWARGS"]["check_held_out"])
        and len(eval_envs) > 0
    )
    human_proxy_params = (
        load_human_proxy_params(
            config["EVAL_KWARGS"]["human_proxy_ckpt_dir"],
            int(config["EVAL_KWARGS"]["human_proxy_num_seeds"]),
        )
        if eval_all_layouts
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

    def train(
        rng, model_params=None, update_step=0,
        init_popart_mu=None, init_popart_sigma=None,
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
            params=flax.core.freeze(network_params),
            tx=tx,
        )
        popart_mu = init_popart_mu
        popart_sigma = init_popart_sigma

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

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
            )

        # TRAIN LOOP
        @scan_tqdm(int(config["NUM_UPDATES"]))
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps, popart_mu, popart_sigma = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng, update_step, beta_agent = runner_state

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
                    other_action.squeeze()
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
            last_val_real = last_val.squeeze() * popart_sigma + popart_mu

            def _calculate_gae(traj_batch, last_val_real):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value_norm, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    value_real = value_norm * popart_sigma + popart_mu
                    delta = (
                        reward
                        + config["GAMMA"] * next_value * (1 - done)
                        - value_real
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value_real), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val_real), last_val_real),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                targets_real = (
                    advantages + traj_batch.value * popart_sigma + popart_mu
                )
                return advantages, targets_real

            advantages, targets = _calculate_gae(traj_batch, last_val_real)
            targets_norm_all = (targets - popart_mu) / popart_sigma
            value_real_all = traj_batch.value * popart_sigma + popart_mu
            target_stats = {
                "popart/mu": popart_mu,
                "popart/sigma": popart_sigma,
                "target_raw/mean": targets.mean(),
                "target_raw/std": targets.std(),
                "target_popart/mean": targets_norm_all.mean(),
                "target_popart/std": targets_norm_all.std(),
                "critic/raw_rmse": jnp.sqrt(
                    jnp.square(targets - value_real_all).mean()
                ),
            }

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value, other_pi, advantage_predictions = network.apply(
                            params,
                            jax.tree.map(lambda h: h.squeeze(), init_hstate),
                            (traj_batch.obs, traj_batch.done, traj_batch.agent_positions),
                            return_advantages=True,
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        other_log_prob = other_pi.log_prob(traj_batch.other_action)
                        moa_nll_loss = -jnp.mean(other_log_prob)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        targets_norm = (targets - popart_mu) / popart_sigma
                        value_losses = jnp.square(value - targets_norm)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - targets_norm
                        )
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        predicted_advantage = jnp.take_along_axis(
                            advantage_predictions,
                            traj_batch.action[..., None],
                            axis=-1,
                        ).squeeze(-1)
                        advantage_loss = 0.5 * jnp.square(
                            predicted_advantage - jax.lax.stop_gradient(gae)
                        ).mean()
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
                            + config["MOA_COEF"] * moa_nll_loss
                            + config["VF_COEF"] * value_loss
                            + config["DAAC_ADV_COEF"] * advantage_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            advantage_loss,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                        )

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

            popart_alpha = config["POPART_ALPHA"]
            batch_mu = targets.mean()
            batch_var = targets.var()
            mu_new = (
                (1.0 - popart_alpha) * popart_mu
                + popart_alpha * batch_mu
            )
            sigma_new = jnp.sqrt(jnp.maximum(
                (1.0 - popart_alpha)
                * (popart_sigma ** 2 + popart_mu ** 2)
                + popart_alpha * (batch_var + batch_mu ** 2)
                - mu_new ** 2,
                1e-8,
            ))
            popart_params = flax.core.unfreeze(train_state.params)
            critic_output = popart_params["params"]["critic_output"]
            critic_output["kernel"] = (
                popart_sigma / sigma_new
            ) * critic_output["kernel"]
            critic_output["bias"] = (
                popart_sigma * critic_output["bias"]
                + popart_mu
                - mu_new
            ) / sigma_new
            train_state = train_state.replace(
                params=flax.core.freeze(popart_params)
            )
            popart_mu, popart_sigma = mu_new, sigma_new

            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            ratio_0 = loss_info[1][4].at[0,0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "advantage_loss": loss_info[1][2],
                "entropy": loss_info[1][3],
                "ratio": loss_info[1][4],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][5],
                "clip_frac": loss_info[1][6],
                **target_stats,
            }
            rng = update_state[-1]

            if eval_all_layouts:
                run_eval = (
                    (update_steps % LOG_INTERVAL == 0)
                    | (update_steps == int(config["NUM_UPDATES"]) - 1)
                )

                def _do_eval(_):
                    base = jax.random.fold_in(rng, update_steps)
                    eval_layout_names = EVAL_LAYOUTS_9
                    out = {}
                    for i, eval_layout_name in enumerate(eval_layout_names):
                        out[eval_layout_name] = eval_layout_sp(
                            eval_envs[eval_layout_name],
                            train_state.params,
                            jax.random.fold_in(base, i),
                        )
                        out[f"{eval_layout_name}_xp"] = eval_layout_xp(
                            eval_envs[eval_layout_name],
                            train_state.params,
                            human_proxy_params[eval_layout_name],
                            jax.random.fold_in(base, 1000 + i),
                        )
                    out["mean"] = jnp.mean(jnp.stack([
                        out[name] for name in eval_layout_names
                    ]))
                    out["mean_xp"] = jnp.mean(jnp.stack([
                        out[f"{name}_xp"] for name in eval_layout_names
                    ]))
                    return out

                def _skip_eval(_):
                    nan = jnp.array(jnp.nan, dtype=jnp.float32)
                    eval_layout_names = EVAL_LAYOUTS_9
                    out = {name: nan for name in eval_layout_names}
                    out.update({
                        f"{name}_xp": nan for name in eval_layout_names
                    })
                    out["mean"] = nan
                    out["mean_xp"] = nan
                    return out

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
                    eval_layout_names = EVAL_LAYOUTS_9
                    sp_mean = float(metric["eval_returns"]["mean"])
                    xp_mean = float(metric["eval_returns"]["mean_xp"])
                    if np.isfinite(sp_mean):
                        log_data["eval/mean"] = sp_mean
                        for eval_layout_name in eval_layout_names:
                            log_data[f"eval/{eval_layout_name}"] = float(
                                metric["eval_returns"][eval_layout_name]
                            )
                    if np.isfinite(xp_mean):
                        log_data["eval_xp/mean"] = xp_mean
                        for eval_layout_name in eval_layout_names:
                            log_data[f"eval_xp/{eval_layout_name}"] = float(
                                metric["eval_returns"][f"{eval_layout_name}_xp"]
                            )
                wandb.log(log_data, step=int(metric["update_steps"]))
                current_return = float(metric["returns"])
                if current_return > best_return[0]:
                    best_return[0] = current_return
                    os.makedirs(config['filepath'], exist_ok=True)
                    ckpt_path = f"{config['filepath']}/{config['fcp_prefix']}seed{config['SEED']}_best_e3t_daac_pop.pkl"
                    with open(ckpt_path, "wb") as f:
                        pickle.dump({
                            'params': metric["params"],
                            'returns': current_return,
                            'update_steps': int(metric['update_steps']),
                            'popart_mu': metric['popart_mu'],
                            'popart_sigma': metric['popart_sigma'],
                        }, f)

            returns = metric["returned_episode_returns"][:, :, 0][
                            metric["returned_episode"][:, :, 0].astype(jnp.int32)
                        ].mean()
            metric["returns"] = returns
            metric["update_steps"] = update_steps
            metric["params"] = train_state.params
            metric["popart_mu"] = popart_mu
            metric["popart_sigma"] = popart_sigma
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)  # hstate resets automatically
            return (
                runner_state, update_steps, popart_mu, popart_sigma,
            ), metric

        best_return = [float('-inf')]
        
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
            _update_step,
            (runner_state, update_step, popart_mu, popart_sigma),
            jnp.arange(int(config["NUM_UPDATES"])),
            int(config["NUM_UPDATES"]),
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="repro_config", config_name="e3t_final_baseline")
def main(config):
    save_xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
    config = OmegaConf.to_container(config)
    config["model_name"] = "E3T_DAAC_POP"
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

    save_variant = "e3t_daac_pop"
    if config["TRAIN_KWARGS"]["finetune"]:
        save_variant += "_finetune"

    if config["WANDB_MODE"] == "online":
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["E3T", "DAAC", "RNN", "SP", "PopArt"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"e3t_daac_pop_{config['ENV_KWARGS']['layout']}_seed{config['SEED']}"
    )
    filepath = f"ckpts/e3t_daac_pop/{config['ENV_NAME']}"
    if config["ENV_NAME"] == "overcooked":
        filepath += f"/{config['ENV_KWARGS']['layout']}"
    filepath = f"{filepath}/ik{config['ENV_KWARGS']['random_reset']}/{config['ENV_KWARGS']['random_reset_fn']}/{save_xpid}"
    config['filepath'] = filepath
    config['fcp_prefix'] = fcp_prefix
    print(f"Working on: \n{filepath}\n")

    if not config['TRAIN_KWARGS']['overwrite_ckpt']:
        # check if ckpt exists
        if os.path.exists(f"{filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}{finetune_appendage}.pkl"):
            print(f"Checkpoint {config['TRAIN_KWARGS']['ckpt_id']} already exists, exiting")
            exit(0)

    init_popart_mu = None
    init_popart_sigma = None
    if config['TRAIN_KWARGS']['ckpt_id'] > 0:
        print("Loading checkpoint")
        with open(f"{filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id'] - 1}{finetune_appendage}.pkl", "rb") as f:
            previous_ckpt = pickle.load(f)
            model_params = previous_ckpt['params']
            init_popart_mu = previous_ckpt.get('popart_mu')
            init_popart_sigma = previous_ckpt.get('popart_sigma')
            final_update_step = previous_ckpt['final_update_step']
            rng = previous_ckpt['key']
            rng, _rng = jax.random.split(jax.random.PRNGKey(rng))

    elif config['TRAIN_KWARGS']['finetune']:
        finetune_filepath =f"ckpts/e3t/{config['ENV_NAME']}"
        if config["ENV_NAME"] == "overcooked":
            finetune_filepath += f"/{config['ENV_KWARGS']['layout']}"
        finetune_filepath = f"{finetune_filepath}/ikFalse"
        fcp_ckpt_num = 19 if config['ENV_NAME'] == 'ToyCoop' else 6
        print("Loading fcp checkpoint for finetuning")
        with open(f"{finetune_filepath}/{fcp_prefix}seed{config['SEED']}_e3t_ckpt{fcp_ckpt_num}.pkl", "rb") as f:  # need to resume from last checkpoint
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

    if init_popart_mu is None:
        init_popart_mu = jnp.zeros(())
    if init_popart_sigma is None:
        init_popart_sigma = jnp.ones(())

    print(f"Starting from update step {final_update_step}")
    train_jit = jax.jit(make_train(config, final_update_step), device=jax.devices()[0])
    out = train_jit(
        rng, model_params, final_update_step,
        init_popart_mu, init_popart_sigma,
    )
    runner_state = out['runner_state']
    train_state = runner_state[0]
    model_state = train_state[0]
    rng = train_state[-1]
    popart_mu = runner_state[2]
    popart_sigma = runner_state[3]
    num_updates = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    
    # save model
    os.makedirs(filepath, exist_ok=True)
    with open(f"{filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}_{save_variant}_updates{num_updates}.pkl", "wb") as f:
        ckpt = {
            'key': rng,
            'params': model_state.params,
            'update_steps': num_updates,
            'popart_mu': popart_mu,
            'popart_sigma': popart_sigma,
        }
        pickle.dump(ckpt, f)

    print(f"Saved model to {filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}_{save_variant}_updates{num_updates}.pkl")
    print(f"Finished training for seed {config['SEED']} with ckpt {config['TRAIN_KWARGS']['ckpt_id']}_updates{num_updates}")
    print(f"--------------------------------")
    

if __name__ == "__main__":
    main()
