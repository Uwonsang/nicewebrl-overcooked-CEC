"""Evaluate trained Overcooked policies against human-proxy BC policies."""
import os
import sys
import glob as glob_module
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import distrax
import hydra
from omegaconf import OmegaConf
# from sklearn.manifold import TSNE

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.overcooked.layouts import make_counter_circuit_9x9, make_forced_coord_9x9, make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9


from jax_tqdm import scan_tqdm
import pandas as pd
from tqdm import tqdm
# import tsnex
from actor_networks import (
    ScannedRNN,
    ActorCriticE3T,
    ActorCriticRNN,
    IDAACActorRNN,
)

HUMAN_PROXY_MODULE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "human_proxy")
)
if HUMAN_PROXY_MODULE_DIR not in sys.path:
    sys.path.append(HUMAN_PROXY_MODULE_DIR)

from bc_agent import BCPolicy


def load_human_proxy_params(ckpt_dir, num_seeds, layout_name_9):
    """Load the same per-layout BC checkpoints used during training eval."""
    layout_name = (
        layout_name_9[:-2]
        if layout_name_9.endswith("_9")
        else layout_name_9
    )
    seed_params = []
    for seed in range(num_seeds):
        ckpt_path = os.path.join(
            ckpt_dir,
            layout_name,
            f"bc_overcooked_{layout_name}_seed{seed}.pkl",
        )
        with open(ckpt_path, "rb") as f:
            seed_params.append(pickle.load(f))
        print(f"Loaded BC seed {seed}: {ckpt_path}")
    return jax.tree.map(lambda *x: jnp.stack(x), *seed_params)


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

def get_rollouts(
    model_params,
    bc_params,
    config,
    env,
    network,
    bc_network,
    main_agent_id,
    seed=0,
):
    other_agent_id = "agent_1" if main_agent_id == "agent_0" else "agent_0"
    main_agent_index = 0 if main_agent_id == "agent_0" else 1
    
    def _step(carry, unused):
        env_state, last_obs, last_done, hstate, rng = carry
        
        # Select action
        rng, main_rng, bc_rng = jax.random.split(rng, 3)
        obs_batch = jnp.stack([last_obs[a].flatten() for a in env.agents])

        agent_positions = jnp.stack([env_state.env_state.agent_pos for a in env.agents])
        ac_in = (
            obs_batch[np.newaxis, :],
            last_done[np.newaxis, :],
            agent_positions[np.newaxis, :]
        )
        if config["model_name"] == "E3T":
            hstate, main_pi, _value, _ = network.apply(
                model_params, hstate, ac_in
            )
        else:
            hstate, main_pi, _value = network.apply(
                model_params, hstate, ac_in
            )
        main_pi = distrax.Categorical(
            logits=main_pi.logits * config["TEST_KWARGS"]["beta"]
        )
        main_sampled = main_pi.sample(seed=main_rng)[0, main_agent_index]
        main_greedy = jnp.argmax(main_pi.probs, axis=-1)[0, main_agent_index]
        main_action = jnp.where(
            config["TEST_KWARGS"]["argmax"], main_greedy, main_sampled
        )

        bc_obs = last_obs[other_agent_id][np.newaxis, :].astype(jnp.float32)
        bc_logits = bc_network.apply(bc_params, bc_obs)
        bc_pi = distrax.Categorical(
            logits=bc_logits * config["TEST_KWARGS"]["beta"]
        )
        bc_sampled = bc_pi.sample(seed=bc_rng)[0]
        bc_greedy = jnp.argmax(bc_pi.probs, axis=-1)[0]
        bc_action = jnp.where(
            config["TEST_KWARGS"]["argmax"], bc_greedy, bc_sampled
        )

        action_prob_dict = {
            main_agent_id: main_pi.probs[0, main_agent_index, :],
            other_agent_id: bc_pi.probs[0],
        }

        # Convert action to env format
        env_act = {main_agent_id: main_action, other_agent_id: bc_action}

        # Step environment
        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward, done, info = env.step(_rng, env_state, env_act)
        
        done_batch = jnp.array([done[a] for a in env.agents])
        transition = (env_state.env_state, obsv, done_batch, env_act, reward, action_prob_dict)
        carry = (env_state, obsv, done_batch, hstate, rng)
        return carry, transition
    
    # Initialize environment and RNN state
    rng = jax.random.PRNGKey(seed)

    def get_rollout(rng, env=env, config=config):
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng)
        init_hstate = ScannedRNN.initialize_carry(
            env.num_agents, config["GRU_HIDDEN_DIM"]
        )
        done_batch = jnp.zeros(env.num_agents, dtype=bool)
        
        init_carry = (env_state, obsv, done_batch, init_hstate, rng)
        _, trajectory = jax.lax.scan(_step, init_carry, None, config["NUM_STEPS"])
        return trajectory, env_state.env_state, obsv

    rollouts_fn = jax.jit(jax.vmap(get_rollout, in_axes=(0,)))
    rollouts_res = rollouts_fn(jax.random.split(rng, config["TEST_KWARGS"]["num_trajs"]))
    trajectories, init_env_states, init_obsvs = rollouts_res
    return (trajectories, init_env_states, init_obsvs)


@hydra.main(version_base=None, config_path="repro_config", config_name="test_general")
def main(config):
    config = OmegaConf.to_container(config)
    model_name = config["model_name"]
    if config["ENV_NAME"] != "overcooked":
        raise ValueError("Human-proxy BC evaluation only supports overcooked")
    if config["TEST_KWARGS"]["plot"]:
        raise ValueError("Human-proxy BC evaluation currently saves CSV only")
    popart_model_names = {
        "CEC_POP_ART",
        "CEC_POP_ART_PREV",
        "CEC_POP_ART_64",
        "CEC_POP_ART_TEST",
    }
    rnn_model_names = {
        "CEC",
        "CEC_Finetune",
        "CEC_IDAAC",
        "CEC_IDAAC_Finetune",
        "CEC_PREV",
        "CEC_64",
        "FCP",
        "FCP_Fixed",
        "IPPO",
        *popart_model_names,
    }

    ##################
    # Load all models for current ckpt id
    ##################

    scalable_model_names = {"CEC", "CEC_IDAAC"}
    finetune_model_names = {"CEC_Finetune", "CEC_IDAAC_Finetune"}
    idaac_model_names = {"CEC_IDAAC", "CEC_IDAAC_Finetune"}
    if config.get("OUTPUT_DIR"):
        save_path_final = config["OUTPUT_DIR"]
    else:
        save_path_final = config['SAVE_PATH'] + "_" + str(config["NUM_MODELS"])
        if model_name in scalable_model_names:
            save_path_final += f"_envs{config['MODEL_NUM_ENVS']}"
        save_path_final += f"/{config['ENV_NAME']}"
    config['SAVE_PATH_FINAL'] = save_path_final
    os.makedirs(config['SAVE_PATH_FINAL'], exist_ok=True)
    param_list = []
    seed_list = []
    configured_seeds = config.get("MODEL_SEEDS")
    iter_range = (
        [int(seed) for seed in configured_seeds]
        if configured_seeds is not None
        else range(config['NUM_MODELS'])
    )

    def find_model_path(seed):
        model_root = f"{config['MODEL_PATH']}/{model_name}"

        if model_name in scalable_model_names:
            patterns = [
                f"{model_root}/{config['MODEL_NUM_ENVS']}/seed{seed}/"
                f"seed{seed}_ckpt*.pkl"
            ]
        elif model_name in finetune_model_names:
            patterns = [
                f"{model_root}/{config['ENV_KWARGS']['layout']}/seed{seed}/"
                f"seed{seed}_ckpt*_finetune_updates*.pkl"
            ]
        elif model_name == "CEC_PREV":
            patterns = [
                f"{model_root}/seed{seed}/"
                f"seed{seed}_ckpt0_improved_updates58593.pkl"
            ]
        elif model_name == "CEC_64":
            patterns = [
                f"{model_root}/seed{seed}/"
                f"seed{seed}_ckpt0_improved_updates183105.pkl"
            ]
        elif model_name == "FCP":
            patterns = [
                f"{model_root}/{config['ENV_KWARGS']['layout']}/"
                f"seed{seed}/fcp_seed{seed}_best.pkl",
            ]
        elif model_name == "FCP_Fixed":
            patterns = [
                f"{model_root}/{config['ENV_KWARGS']['layout']}/"
                f"seed{seed}/fcp_fixed_seed{seed}_best.pkl",
            ]
        elif model_name == "E3T":
            patterns = [
                f"{model_root}/{config['ENV_KWARGS']['layout']}/"
                f"seed{seed}/seed{seed}_best_e3t.pkl"
            ]
        elif model_name == "IPPO":
            patterns = [
                f"{model_root}/{config['ENV_KWARGS']['layout']}/"
                f"seed{seed}/seed{seed}_best.pkl"
            ]
        elif model_name == "CEC_POP_ART_PREV":
            patterns = [
                f"{model_root}/seed{seed}/"
                f"seed{seed}_ckpt0_improved_pop_updates29296.pkl"
            ]
        elif model_name == "CEC_POP_ART_64":
            patterns = [
                f"{model_root}/seed{seed}/"
                f"seed{seed}_ckpt0_improved_pop_updates183105.pkl"
            ]
        elif model_name == "CEC_POP_ART_TEST":
            patterns = [
                f"{model_root}/seed{seed}/"
                f"seed{seed}_ckpt0_improved_pop_updates29.pkl"
            ]
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        for pat in patterns:
            matches = sorted(glob_module.glob(pat))
            if matches:
                return matches[-1]
        return None
    
    def find_toy_model_path(seed):
        model_root = os.path.join(config['MODEL_PATH'], "ToyCoop", model_name)
        if model_name == "CEC":
            patterns = [f"{model_root}/seed{seed}/seed{seed}_best.pkl"]
        elif model_name == "FCP":
            patterns = [
                f"{model_root}/seed{seed}/fcp_seed{seed}_best.pkl"
            ]
        elif model_name == "IPPO":
            patterns = [
                f"{model_root}/seed{seed}/seed{seed}_best.pkl"
            ]
        for pat in patterns:
            matches = sorted(glob_module.glob(pat))
            if matches:
                return matches[-1]
        return None

    for seed in iter_range:
        if config["ENV_NAME"] == "ToyCoop":
            filepath = find_toy_model_path(seed)
        else:
            filepath = find_model_path(seed)
        if filepath is None:
            print(f"Missing checkpoint: model={model_name}, seed={seed}")
            continue
        try:
            with open(filepath, "rb") as f:
                previous_ckpt = pickle.load(f)
                model_params = previous_ckpt['params']
                if model_name in popart_model_names:
                    import flax.core
                    p = flax.core.unfreeze(model_params)
                    if 'critic_output' in p.get('params', {}):
                        p['params']['Dense_11'] = p['params'].pop('critic_output')
                    model_params = flax.core.freeze(p)
                param_list.append(model_params)
                seed_list.append(seed)
                del previous_ckpt
                print(f"Loaded seed {seed}: {filepath}")
        except (OSError, KeyError, pickle.UnpicklingError) as exc:
            print(f"Failed to load {filepath}: {exc}")
            continue
    
    if len(param_list) == 0:
        raise RuntimeError(f"No checkpoints found for model={model_name}")
    seed_list = jnp.array(seed_list)

    param_stack = jax.tree.map(lambda *x: jnp.stack(x), *param_list)
    
    ##################
    # Initialize environment and network
    ##################
    layout_name = config['ENV_KWARGS']['layout']
    env = initialize_environment(config)
    env = LogWrapper(env, env_params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})
    if model_name in idaac_model_names:
        network = IDAACActorRNN(
            env.action_space("agent_0").n,
            config=config,
        )
    elif model_name in rnn_model_names:
        network = ActorCriticRNN(env.action_space("agent_0").n, config=config)
    elif model_name == "E3T":
        network = ActorCriticE3T(env.action_space("agent_0").n, config=config)
    bc_network = BCPolicy()
    num_bc_seeds = int(config["HUMAN_PROXY_NUM_SEEDS"])
    bc_params = load_human_proxy_params(
        config["HUMAN_PROXY_CKPT_DIR"],
        num_bc_seeds,
        layout_name,
    )

    model_indices, bc_indices = jnp.meshgrid(
        jnp.arange(len(seed_list)),
        jnp.arange(num_bc_seeds),
        indexing="ij",
    )
    eval_pairs = jnp.stack(
        (model_indices.reshape(-1), bc_indices.reshape(-1)), axis=-1
    )
    
    ##################
    # Evaluate every trained seed against every BC seed in both agent seats.
    ##################
    @jax.jit
    def eval_pair(pair, seed_list, param_stack, bc_params):
        model_index, bc_seed = pair
        model_params = jax.tree.map(lambda x: x[model_index], param_stack)
        bc_seed_params = jax.tree.map(lambda x: x[bc_seed], bc_params)
        rollout_seed = model_index * num_bc_seeds + bc_seed

        trajectories_as_0, _, _ = get_rollouts(
            model_params,
            bc_seed_params,
            config,
            env,
            network,
            bc_network,
            "agent_0",
            seed=rollout_seed * 2,
        )
        trajectories_as_1, _, _ = get_rollouts(
            model_params,
            bc_seed_params,
            config,
            env,
            network,
            bc_network,
            "agent_1",
            seed=rollout_seed * 2 + 1,
        )
        rewards_as_0 = trajectories_as_0[4]["agent_0"].sum(axis=1)
        rewards_as_1 = trajectories_as_1[4]["agent_0"].sum(axis=1)
        return seed_list[model_index], bc_seed, rewards_as_0, rewards_as_1

    print("Evaluating trained policies against human-proxy BC policies")
    eval_pair_fn = jax.jit(
        jax.vmap(eval_pair, in_axes=(0, None, None, None))
    )
    model_seeds, bc_seeds, rewards_as_0, rewards_as_1 = eval_pair_fn(
        eval_pairs, seed_list, param_stack, bc_params
    )

    rows = []
    for pair_index in tqdm(range(len(model_seeds))):
        model_seed = int(model_seeds[pair_index])
        bc_seed = int(bc_seeds[pair_index])
        for model_agent, pair_rewards in (
            ("agent_0", rewards_as_0[pair_index]),
            ("agent_1", rewards_as_1[pair_index]),
        ):
            for trajectory, reward in enumerate(pair_rewards):
                rows.append(
                    {
                        "model_seed": model_seed,
                        "bc_seed": bc_seed,
                        "model_agent": model_agent,
                        "trajectory": trajectory,
                        "reward": float(reward),
                    }
                )
    df = pd.DataFrame(rows)
    savefile = (
        f"{config['SAVE_PATH_FINAL']}/{model_name}_{layout_name}_BC_results.csv"
    )
    df.to_csv(savefile, index=False)
    print(f"Saved data to {savefile}")

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
