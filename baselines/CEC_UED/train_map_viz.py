import os
from tqdm import tqdm
import h5py
import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked import overcooked_layouts

from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from flax import struct
import chex
import hydra
from omegaconf import OmegaConf
import jaxmarl
import imageio

def load_h5(path):
    """Load env_state dict from h5 (keys = dataset names, values = arrays)."""
    with h5py.File(path, "r") as f:
        return {k: f[k][:] for k in f.keys()}

@struct.dataclass
class FilteredState:
    agent_dir_idx: chex.Array
    agent_inv: chex.Array
    maze_map: chex.Array
    update_step: chex.Array

def filtered_state(ep_state):
    dtype_map = {
        "agent_dir_idx": jnp.int32,
        "agent_inv": jnp.int32,
        "maze_map": jnp.uint8,
        "update_step": jnp.uint8,
    }
    state_dict = {
        k: jnp.asarray(v, dtype=dtype_map.get(k, v.dtype))
        for k, v in ep_state.items()
        if k in dtype_map
    }

    return FilteredState(**state_dict)

def layout_render(env_state_obs, config, save_dir):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    agent_view_size = env.agent_view_size
    viz = OvercookedVisualizer()
    state = filtered_state(env_state_obs)
    os.makedirs(save_dir, exist_ok=True)

    num_updates = int(state.agent_dir_idx.shape[0])
    num_envs = int(state.agent_dir_idx.shape[1])

    for upd_idx in tqdm(range(num_updates), desc="update steps"):
        upd_step = int(state.update_step[upd_idx])
        step_dir = os.path.join(save_dir, f"step_{upd_step:06d}")
        os.makedirs(step_dir, exist_ok=True)

        for env_idx in range(num_envs):
            single_state = FilteredState(
                agent_dir_idx=state.agent_dir_idx[upd_idx, env_idx],
                agent_inv=state.agent_inv[upd_idx, env_idx],
                maze_map=state.maze_map[upd_idx, env_idx],
                update_step=state.update_step[upd_idx],
            )
            frame = viz.custom_get_frame(single_state, agent_view_size)
            imageio.imwrite(os.path.join(step_dir, f"env_{env_idx:03d}.png"), frame)

    
@hydra.main(version_base=None, config_path="config", config_name="ippo_overcooked_CEC")
def visualize_layout(config):
    config = OmegaConf.to_container(config)
    data_path = '/app/baselines/CEC_UED/dataset/train_env_states.h5'
    # layout_name = data_path.split('/')[-1].split('.h5')[0].split('layout_dataset_')[1].split('_', 1)[1]
    env_state_obs = load_h5(data_path)
    save_dir = f"/app/baselines/CEC_UED/dataset/img_train_env_states"
    layout_render(env_state_obs, config, save_dir)


if __name__ == "__main__":
    visualize_layout()