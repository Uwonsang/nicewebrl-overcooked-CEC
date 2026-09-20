"""
Based on PureJaxRL Implementation of PPO.

Note, this file will only work for MPE environments with homogenous agents (e.g. Simple Spread).

"""
import os
import glob as glob_module
import jax
import jax.numpy as jnp
import hydra
from omegaconf import OmegaConf

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.overcooked.layouts import make_counter_circuit_9x9, make_forced_coord_9x9, make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9
import imageio

from jax_tqdm import scan_tqdm
from tqdm import tqdm
from baselines.CEC_UED.minimax.plr_utils import pad_wall_idx
from pathlib import Path
from flax.core import unfreeze
# import tsnex

def initialize_environment(config):
    layout_name = config["ENV_KWARGS"]["layout"]
    config['layout_name'] = layout_name
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if config["ENV_NAME"] == "overcooked":
        def reset_env(key):
            def reset_sub_dict(key, fn):
                key, subkey = jax.random.split(key)
                sampled_layout_dict = fn(subkey)
                temp_o, temp_s = env.custom_reset(key, layout=sampled_layout_dict, random_reset=False, shuffle_inv_and_pot=False)
                key, subkey = jax.random.split(key)
                return (temp_o, temp_s), sampled_layout_dict, key

            def mk(fn):
                def f(k):
                    k, sk = jax.random.split(k)
                    layout = fn(sk, ik=True)
                    return pad_wall_idx(layout)
                return f
                
            asymm_reset, asymm_layout_dict, key = reset_sub_dict(key, mk(make_asymm_advantages_9x9))
            coord_ring_reset, coord_ring_layout_dict, key = reset_sub_dict(key, mk(make_coord_ring_9x9))
            counter_circuit_reset, counter_circuit_layout_dict, key = reset_sub_dict(key, mk(make_counter_circuit_9x9))
            forced_coord_reset, forced_coord_layout_dict, key = reset_sub_dict(key, mk(make_forced_coord_9x9))
            cramped_room_reset, cramped_room_layout_dict, key = reset_sub_dict(key, mk(make_cramped_room_9x9))
            layout_resets = [asymm_reset, coord_ring_reset, counter_circuit_reset, forced_coord_reset, cramped_room_reset]
            layout_dicts = [asymm_layout_dict, coord_ring_layout_dict, counter_circuit_layout_dict, forced_coord_layout_dict, cramped_room_layout_dict]
            # stack all layouts
            stacked_layout_reset = jax.tree.map(lambda *x: jnp.stack(x), *layout_resets)
            stacked_layout_dicts = jax.tree.map(lambda *x: jnp.stack(x), *layout_dicts)
            # sample an index from 0 to 4
            index = jax.random.randint(key, (), minval=0, maxval=5)
            sampled_reset = jax.tree.map(lambda x: x[index], stacked_layout_reset)
            sampled_layout_dict = jax.tree.map(lambda x: x[index], stacked_layout_dicts)
            return sampled_reset, sampled_layout_dict
        @scan_tqdm(100)
        def gen_held_out(runner_state, unused):
            (i,) = runner_state
            (_, ho_state), ho_layout_dict = reset_env(jax.random.key(i))
            res = (ho_state.goal_pos, ho_state.wall_map, ho_state.pot_pos, ho_layout_dict)
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
        # Build held-out layout dict as: {held_out_idx: single_layout_dict}
        eval_held_out_layouts = [
            jax.tree.map(lambda x, i=i: x[i], res[3]) for i in range(res[3]["agent_idx"].shape[0])
        ]
        config["eval_held_out_layouts"] = eval_held_out_layouts
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
    

@hydra.main(version_base=None, config_path="../repro_config", config_name="test_general")
def main(config):
    config = OmegaConf.to_container(config)
    if config['ENV_NAME'] == "overcooked":
        from jaxmarl.viz.overcooked_jitted_visualizer import render_fn
    else:
        from jaxmarl.viz.toy_coop_jitted_visualizer import render_fn

    out_path = Path(__file__).resolve().parent / "map_check"
    os.makedirs(out_path, exist_ok=True)

    ##################
    # Initialize environment and network
    ##################
    env = initialize_environment(config)
    env = LogWrapper(env, env_params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})

    stacked_layouts = jax.device_get(config["eval_held_out_layouts"])
    eval_held_out_layouts = unfreeze(stacked_layouts)

    @jax.jit
    def render_reset(key, reset_layout):
        obsv, state = env._env.custom_reset(
                key, random_reset=False, shuffle_inv_and_pot=False, layout=reset_layout)
        return render_fn(state)

    images = []
    base_key = jax.random.PRNGKey(0)
    for i, reset_layout in enumerate(tqdm(eval_held_out_layouts)):
        key = jax.random.fold_in(base_key, i)
        images.append(render_reset(key, reset_layout))

    for i, image in enumerate(images):
        imageio.imwrite(f"{out_path}/held_out_layout_{i}.png", image)


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