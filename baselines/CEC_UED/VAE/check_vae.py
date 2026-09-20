import jax
import jax.numpy as jnp
import hydra
from omegaconf import OmegaConf

from tqdm import tqdm
import pickle
import os
import jaxmarl
import imageio

from Models.vae import Decoder
from utils import restore_from_obs, restore_to_26ch, load_checkpoint
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from map_viz import FilteredState
import jaxmarl

def check_decoder(config):
    rng = jax.random.PRNGKey(config["SEED"])
    
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    agent_view_size = env.agent_view_size
    viz = OvercookedVisualizer()
    
    ckpt_path = f"/app/baselines/CEC_UED/VAE/checkpoints/layout_1e7_all/lr-20260307-080520/vae_seed0_kl50.pkl"
    params, ckpt_config = load_checkpoint(ckpt_path)

    decoder_params = {"params": params["params"]["Decoder_0"]}
    decoder = Decoder(ckpt_config["output_channels"])

    restore_to_26ch_batch = jax.vmap(restore_to_26ch, in_axes=0)
    restore_from_obs_batch = jax.vmap(restore_from_obs, in_axes=0)

    num_samples = 16
    frames = []
    for i in tqdm(range(num_samples)):
        rng, _rng = jax.random.split(rng)
        z = jax.random.normal(_rng, (1, config["latent_dim"]))

        recons = decoder.apply(decoder_params, z)
        recons = (jax.nn.sigmoid(recons) > 0.5).astype(jnp.uint8)

        recons_26 = restore_to_26ch_batch(recons)
        restored = restore_from_obs_batch(recons_26)
        states = FilteredState(**restored)
        
        frame = viz.custom_get_frame(jax.tree.map(lambda x: x[i], states), agent_view_size)
        frames.append(frame)

    save_dir = f"/app/baselines/CEC_UED/VAE/img_check_decoder"
    os.makedirs(save_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        imageio.imwrite(os.path.join(save_dir, f"layout_{i:03d}.png"), frame)


@hydra.main(version_base=None, config_path="config", config_name="vae_config")
def main(config):
    config = OmegaConf.to_container(config)
    check_decoder(config)

if __name__ == "__main__":
    main()