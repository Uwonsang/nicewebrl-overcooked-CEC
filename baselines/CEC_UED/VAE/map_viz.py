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
from utils import restore_from_obs


def load_h5(path, config):
    """Load env_state dict from h5 (keys = dataset names, values = arrays)."""
    with h5py.File(path, "r") as f:
        return {k: f[k][:config["VIZ_STEPS"]] for k in f.keys()}

@struct.dataclass
class FilteredState:
    agent_dir_idx: chex.Array
    agent_inv: chex.Array
    maze_map: chex.Array

def filtered_state(ep_state):
    dtype_map = {
        "agent_dir_idx": jnp.int32,
        "agent_inv": jnp.int32,
        "maze_map": jnp.uint8,
    }
    state_dict = {
        k: jnp.asarray(v, dtype=dtype_map.get(k, v.dtype))
        for k, v in ep_state.items()
        if k in dtype_map
    }

    state_dict = {
        k: v.reshape(-1, *v.shape[2:]) for k, v in state_dict.items()
    }
    
    return FilteredState(**state_dict)

def layout_render(env_state_obs, config, save_dir):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    agent_view_size = env.agent_view_size
    viz = OvercookedVisualizer()

    if config["use_obs"]:
        restore_from_obs_batch = jax.vmap(jax.vmap(restore_from_obs, in_axes=0), in_axes=0)
        restored = restore_from_obs_batch(env_state_obs["agent_0"])
        state = filtered_state(restored)
    else:
        state = filtered_state(env_state_obs)

    frame = [
        viz.custom_get_frame(jax.tree.map(lambda x: x[step], state), agent_view_size)
        for step in tqdm(range(config["VIZ_STEPS"]))
    ]

    os.makedirs(save_dir, exist_ok=True)
    for i, frame in enumerate(frame):
        imageio.imwrite(os.path.join(save_dir, f"layout_{i:03d}.png"), frame)

    
@hydra.main(version_base=None, config_path="config", config_name="collect_overcooked")
def visualize_layout(config):
    config = OmegaConf.to_container(config)
    data_path = '/app/baselines/CEC_UED/VAE/dataset/layout_dataset_1e6_coord_ring.h5'
    layout_name = data_path.split('/')[-1].split('.h5')[0].split('layout_dataset_')[1].split('_', 1)[1]
    env_state_obs = load_h5(data_path, config)
    save_dir = f"/app/baselines/CEC_UED/VAE/dataset/img_use_obs_{config['use_obs']}_{layout_name}"
    layout_render(env_state_obs, config, save_dir)


if __name__ == "__main__":
    visualize_layout()