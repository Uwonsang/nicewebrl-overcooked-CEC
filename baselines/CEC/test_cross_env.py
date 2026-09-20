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
import gc
# from sklearn.manifold import TSNE

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.overcooked.layouts import make_counter_circuit_9x9, make_forced_coord_9x9, make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9


import imageio
import matplotlib.pyplot as plt

import wandb
import functools
import pdb
from jax_tqdm import scan_tqdm
import pandas as pd
from tqdm import tqdm
# import tsnex
from pathlib import Path


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
    if config["ENV_NAME"] == "ToyCoop":
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
        config["obs_dim"] = (5,5,4)
    else:
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
        batch_size, num_envs, flattened_obs_dim = obs.shape
        if self.config["CONV_NET"]:
            if self.config["ENV_NAME"] == "overcooked":
                reshaped_obs = obs.reshape(-1, 9,9,26)
            else:
                reshaped_obs = obs.reshape(-1, 5,5,4)

            embedding = nn.Conv(
                features=64,
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(reshaped_obs)
            embedding = nn.relu(embedding)
            embedding = nn.Conv(
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


def get_rollouts(model_param, config, env, network, seed=0):
    
    def _step(carry, unused):
        train_state_params, env_state, last_obs, last_done, hstate_1, hstate_2, rng = carry
        
        # Select action
        rng, _rng = jax.random.split(rng)
        obs_batch = jnp.stack([last_obs[a].flatten() for a in env.agents])

        agent_positions = jnp.stack([env_state.env_state.agent_pos for a in env.agents])
        ac_in = (
            obs_batch[np.newaxis, :],
            last_done[np.newaxis, :],
            agent_positions[np.newaxis, :]
        )
        hstate_1, pi_1, value_1 = network.apply(train_state_params, hstate_1, ac_in)
        pi_1 = distrax.Categorical(logits=pi_1.logits * config["TEST_KWARGS"]["beta"])
        hstate_2, pi_2, value_2 = network.apply(train_state_params, hstate_2, ac_in)
        pi_2 = distrax.Categorical(logits=pi_2.logits * config["TEST_KWARGS"]["beta"])

        _rng, _a0, _a1 = jax.random.split(_rng, 3)
        action_1 = pi_1.sample(seed=_a0)[0]
        action_1 = jnp.where(config["TEST_KWARGS"]["argmax"], jnp.argmax(pi_1.probs, 2)[0], action_1)
        action_1_prob_distrib = pi_1.probs[0, 0, :]
        action_2 = pi_2.sample(seed=_a1)[0]
        action_2 = jnp.where(config["TEST_KWARGS"]["argmax"], jnp.argmax(pi_2.probs, 2)[0], action_2)
        action_2_prob_distrib = pi_2.probs[0, 1, :]
        action_prob_dict = {env.agents[0]: action_1_prob_distrib, env.agents[1]: action_2_prob_distrib}

        # Convert action to env format
        env_act = {env.agents[0]: action_1[0], env.agents[1]: action_2[1]}

        # Step environment
        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward, done, info = env.step(_rng, env_state, env_act)
        
        done_batch = jnp.array([done[a] for a in env.agents])
        transition = (env_state.env_state, obsv, done_batch, env_act, reward, action_prob_dict)
        carry = (train_state_params, env_state, obsv, done_batch, hstate_1, hstate_2, rng)
        return carry, transition
    
    # Initialize environment and RNN state
    rng = jax.random.PRNGKey(seed)

    def get_rollout(rng, train_state_params=model_param, env=env, config=config):
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng)
        init_hstate_1 = ScannedRNN.initialize_carry(env.num_agents, config["GRU_HIDDEN_DIM"])
        init_hstate_2 = ScannedRNN.initialize_carry(env.num_agents, config["GRU_HIDDEN_DIM"])
        done_batch = jnp.zeros(env.num_agents, dtype=bool)
        
        init_carry = (train_state_params, env_state, obsv, done_batch, init_hstate_1, init_hstate_2, rng)
        _, trajectory = jax.lax.scan(_step, init_carry, None, config["NUM_STEPS"])
        return trajectory, env_state.env_state, obsv

    rollouts_fn = jax.jit(jax.vmap(get_rollout, in_axes=(0,)))
    rollouts_res = rollouts_fn(jax.random.split(rng, config["TEST_KWARGS"]["num_trajs"]))
    trajectories, init_env_states, init_obsvs = rollouts_res
    return (trajectories, init_env_states, init_obsvs)


@hydra.main(version_base=None, config_path="repro_config", config_name="cross_env")
def main(config):
    config = OmegaConf.to_container(config)
    if config["FCP"]:
        fcp_str = "fcp_"
    else:
        fcp_str = ""
    if config["TRAIN_KWARGS"]["finetune"]:
        finetune_appendage = "_improved_finetune"
    else:
        finetune_appendage = "_improved"

    if config['ENV_KWARGS']['partial_obs']:
        finetune_appendage += "_partial_obs"
    if not config['LSTM']:
        finetune_appendage += "_no_lstm"

    if config['ENV_NAME'] == "overcooked":
        from jaxmarl.viz.overcooked_jitted_visualizer import render_fn
    else:
        from jaxmarl.viz.toy_coop_jitted_visualizer import render_fn

    config["ENV_KWARGS"]["shuffle_inv_and_pot"] = False
    config["ENV_KWARGS"]["check_held_out"] = False
    filepath = f"ckpts/ippo/{config['ENV_NAME']}"
    if config["ENV_NAME"] == "overcooked":
        filepath += f"/{config['ENV_KWARGS']['layout']}"
        filepath = f'{filepath}/ik{config["TEST_KWARGS"]["ik"]}/{config["ENV_KWARGS"]["random_reset_fn"]}'
    else:
        filepath = f'{filepath}/ik{config["TEST_KWARGS"]["ik"]}/{config["ENV_KWARGS"]["random_reset_fn"]}'
    # make path if it doesn't exist
    os.makedirs(filepath, exist_ok=True)

    if config['FCP_KWARGS']['train_oracle'] and config['FCP'] and config['ENV_KWARGS']['incentivize_strat'] == 2:
        finetune_appendage += '_oracle'
    if config['ENV_KWARGS']['incentivize_strat'] != 2:
        finetune_appendage += f"_incentivize_strat_{config['ENV_KWARGS']['incentivize_strat']}"

    CKPT_ROOT = Path(__file__).resolve().parents[2] / "ckpts"
    model_dir = CKPT_ROOT / config["MODEL_PATH"].lstrip("/")
    ckpt_name = config.get("CKPT_FILE")
    if not ckpt_name:
        ckpt_name = (
            f"seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}_improved.pkl"
        )
    ckpt_path = model_dir / ckpt_name
    with open(ckpt_path, "rb") as f:
        previous_ckpt = pickle.load(f)
        model_params = previous_ckpt["params"]

    ##################
    # Initialize environment and network
    ##################
    env = initialize_environment(config)
    env = LogWrapper(env, env_params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})
    network = ActorCriticRNN(env.action_space("agent_0").n, config=config)

    ##################
    # Self-play eval (single shared policy, separate LSTM carries per agent)
    ##################
    @jax.jit
    def eval_sp(model_params, config=config, env=env, network=network):
        trajectories, init_env_states, init_obsvs = get_rollouts(model_params, config, env, network)
        rewards = trajectories[4]["agent_0"].sum(axis=1)
        return (rewards, trajectories, init_env_states)

    if not config['TEST_KWARGS']['plot']:
        rewards, trajectories, init_env_states = eval_sp(model_params)
        rewards_np = np.asarray(rewards)
        print(f"Mean reward_agent0: {rewards_np.mean():.4f}")
    else:
        rewards, trajectories, init_env_states = eval_sp(model_params)
        for traj_num in tqdm(range(config['TEST_KWARGS']['num_trajs']), desc="trajectories"): 
            traj = jax.tree.map(lambda x: x[traj_num], trajectories)  # get current trajectory from (num_trajs, num_timesteps, ...) output
            env_states = [jax.tree.map(lambda x: x[traj_num], init_env_states)]
            traj_rewards = [0]
            for timepoint in tqdm(range(config['NUM_STEPS']), desc="num_steps"):
                timestep = jax.tree.map(lambda x: x[timepoint], traj)  # get the current timestep from trajectories

                env_state = timestep[0]
                traj_rewards.append(timestep[4]['agent_0'])
                env_states.append(env_state)

            env_images = [render_fn(s) for s in env_states]
            traj_rewards = jnp.cumsum(jnp.array(traj_rewards))

            frames = []
            for t in tqdm(range(len(env_images)), desc="render_images"):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

                ax1.imshow(env_images[t], interpolation="nearest", aspect="equal")
                ax1.axis("off")
                ax1.set_title("Environment")

                ax2.plot(range(len(traj_rewards)), traj_rewards, "b-")
                ax2.scatter(t, traj_rewards[t], color="red", s=100)
                ax2.set_xlabel("Timestep")
                ax2.set_ylabel("Cumulative Reward")
                ax2.set_title("Reward Over Time")

                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(image)
                plt.close(fig)

            gif_path = (
                f"{filepath}/{fcp_str}sp_eval_on_ik{config['ENV_KWARGS']['random_reset']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}_traj_{traj_num}.gif"
            )
            imageio.mimsave(gif_path, frames, fps=10, loop=0)
            print(f"Saved gif to {gif_path}")

if __name__ == "__main__":
    main()


    # FOR FUTURE REFERENCE:
    '''
        loop over graph/no graph  (this will be config)
        loop over ik train vs sk train  (this will be test kwargs)
        loop over ckpt id  (this will be train kwargs)
        loop over eval on ik vs eval on sk  (this will be env kwargs)
    '''

    # For overcooked
    '''
    # first eval sk grids on sk model
    for layout in "cramped_room_padded" "counter_circuit_padded" "forced_coord_padded" "asymm_advantages_padded" "coord_ring_padded"
        for graph vs no graph
            for train sk
                for test ik = False vs True
                    for ckpt id
                        run eval
    '''