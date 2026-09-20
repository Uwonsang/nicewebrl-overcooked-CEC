"""
Reset-only data collection: resets env (diverse layouts) NUM_UPDATES times.
Saves (agent_0 obs, env_state) pairs that are always time-aligned.
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
import hydra
import h5py
from omegaconf import OmegaConf

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.overcooked.layouts import make_counter_circuit_9x9, make_forced_coord_9x9, make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9

from jax_tqdm import scan_tqdm
import time
from jax.experimental import io_callback


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
            stacked_layout_reset = jax.tree.map(lambda *x: jnp.stack(x), *layout_resets)
            index = jax.random.randint(key, (), minval=0, maxval=5)
            sampled_reset = jax.tree.map(lambda x: x[index], stacked_layout_reset)
            return sampled_reset
        @scan_tqdm(100)
        def gen_held_out(runner_state, unused):
            (i,) = runner_state
            _, ho_state = reset_env(jax.random.key(i))
            res = (ho_state.goal_pos, ho_state.wall_map, ho_state.pot_pos)
            carry = (i + 1,)
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
        @scan_tqdm(100)
        def gen_held_out_toycoop(runner_state, unused):
            (i,) = runner_state
            key = jax.random.key(i)
            state = env.custom_reset_fn(key, random_reset=True)
            res = (state.agent_pos, state.goal_pos, state.other_goal_pos)
            carry = (i + 1,)
            return carry, res
        carry, res = jax.lax.scan(gen_held_out_toycoop, (0,), jnp.arange(100), 100)
        ho_agent_pos, ho_goal_pos, ho_other_goal_pos = res
        env.held_out_agent_pos = ho_agent_pos
        env.held_out_goal_pos = ho_goal_pos
        env.held_out_other_goal_pos = ho_other_goal_pos
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    return env


def init_hdf5(save_path, field_names, field_shapes, field_dtypes, num_updates, num_envs):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with h5py.File(save_path, "w") as f:
        for k in field_names:
            f.create_dataset(
                k,
                shape=(num_updates, num_envs, *field_shapes[k]),
                dtype=field_dtypes[k],
                chunks=(1, num_envs, *field_shapes[k]),
            )
    print(f"Initialized HDF5 at {save_path}")


def make_rollout(config, save_path):
    env = initialize_environment(config)

    config["NUM_UPDATES"] = (
        config["NUM_LAYOUTS"] // config["NUM_ENVS"]
    )
    init_obs, init_state = env.reset(
        jax.random.PRNGKey(0),
        params={"random_reset_fn": config["ENV_KWARGS"]["random_reset_fn"]}
    )
    state_names  = ["agent_dir_idx", "agent_inv", "maze_map"]
    state_shapes = {k: getattr(init_state, k).shape for k in state_names}
    state_dtypes = {k: getattr(init_state, k).dtype for k in state_names}
    obs_shapes   = {"agent_0": init_obs["agent_0"].shape}
    obs_dtypes   = {"agent_0": init_obs["agent_0"].dtype}

    field_shapes = {**obs_shapes, **state_shapes}
    field_dtypes = {**obs_dtypes, **state_dtypes}
    field_names  = list(obs_shapes.keys()) + list(state_names)

    env = LogWrapper(env, env_params={"random_reset_fn": config["ENV_KWARGS"]["random_reset_fn"]})

    init_hdf5(save_path, field_names, field_shapes, field_dtypes,
              int(config["NUM_UPDATES"]), config["NUM_ENVS"])

    def run(rng):
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        def save_to_hdf5(arrays, update_idx):
            idx = int(update_idx)
            with h5py.File(save_path, "a") as f:
                for k, v in zip(field_names, arrays):
                    f[k][idx] = np.asarray(v)

        @scan_tqdm(int(config["NUM_UPDATES"]))
        def _collect(runner_state, update_idx):
            env_state, obsv, rng = runner_state

            arrays = (
                [obsv[k] for k in obs_shapes.keys()]
                + [getattr(env_state.env_state, k) for k in state_names]
            )
            io_callback(save_to_hdf5, None, arrays, update_idx, ordered=True)

            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

            return (env_state, obsv, rng), None

        rng, _rng = jax.random.split(rng)
        runner_state = (env_state, obsv, _rng)
        runner_state, _ = jax.lax.scan(
            _collect,
            runner_state,
            jnp.arange(int(config["NUM_UPDATES"])),
            int(config["NUM_UPDATES"]),
        )

        return {"runner_state": runner_state}

    return run


@hydra.main(version_base=None, config_path="config", config_name="collect_overcooked")
def main(config):
    config = OmegaConf.to_container(config)
    xpid    = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
    ts_str  = f"{int(config['NUM_LAYOUTS']):.0e}".replace("+0", "")
    reset_str = config['ENV_KWARGS']['random_reset_fn'].replace('reset_', '')
    save_path = f"/app/baselines/CEC_UED/VAE/dataset/layout_dataset_{ts_str}_{reset_str}.h5"

    rng = jax.random.PRNGKey(config["SEED"])
    rollout_fn = jax.jit(make_rollout(config, save_path), device=jax.devices()[0])
    rollout_fn(rng)
    print("Done.")

if __name__ == "__main__":
    main()