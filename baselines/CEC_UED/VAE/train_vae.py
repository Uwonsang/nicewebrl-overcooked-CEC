import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf

import wandb
import yaml
from tqdm import tqdm
import pickle
import os
import jaxmarl
import time

from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from map_viz import FilteredState
from Models.vae import VAE, vae_loss
from utils import (
    restore_from_obs, 
    split_dataset, 
    DataLoader, 
    concat_images_with_labels, 
    load_h5, 
    input_processing, 
    restore_to_26ch
)

def make_train(config, train_data, test_data):
    xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
    checkpoint_path = f"/app/baselines/CEC_UED/VAE/checkpoints/layout_{config['LAYOUT_DATA_FILE'].replace('.h5', '').split('layout_dataset_')[1]}/{xpid}"
    print(f"Checkpoint path: {checkpoint_path}")
    # for viz
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    agent_view_size = env.agent_view_size
    viz = OvercookedVisualizer()

    input_shape = train_data.shape[1:]
    num_epochs = config["NUM_EPOCHS"]

    steps_per_epoch = len(train_data) // config["BATCH_SIZE_TRAIN"]
    total_steps = num_epochs * steps_per_epoch
    
    def linear_schedule(count):
        frac = 1.0 - count / total_steps
        frac = jnp.maximum(1e-9, frac)
        return config["LR"] * frac

    train_loader = DataLoader(train_data, config["BATCH_SIZE_TRAIN"], shuffle=True)
    test_loader = DataLoader(test_data, config["BATCH_SIZE_TEST"], shuffle=True)

    kl_spectrum = [10, 20, 30, 50, 70, 100, 200]
    best_recon_wrt_kl = {kl: 1e9 for kl in kl_spectrum}

    def train(rng):
        model = VAE(config)
        rng, _rng, _rng_init = jax.random.split(rng, 3)
        params = model.init(_rng, jnp.zeros((1, *input_shape)), _rng_init)

        tx = optax.adamw(learning_rate=linear_schedule, eps=1e-5, weight_decay=1e-4)
        train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        
        jit_update = jax.jit(jax.value_and_grad(vae_loss, has_aux=True), static_argnums=(1,))
        jit_eval   = jax.jit(vae_loss, static_argnums=(1,))

        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            beta = config["init_kl_penalty"] + epoch / num_epochs * (config["kl_penalty"] - config["init_kl_penalty"])
            epoch_losses, epoch_recons, epoch_kls = [], [], []
            epoch_test_losses, epoch_test_recons, epoch_test_kls = [], [], []

            for train_batch in tqdm(train_loader, desc="Training"):
                rng, _rng_vae = jax.random.split(rng)
                (loss, (recon_loss, kl_loss)), grads = jit_update(
                    train_state.params, train_state.apply_fn, train_batch, _rng_vae, beta)

                train_state = train_state.apply_gradients(grads=grads)
                epoch_losses.append(float(loss))
                epoch_recons.append(float(recon_loss))
                epoch_kls.append(float(kl_loss))
            
            wandb.log({
                "loss": np.mean(epoch_losses), "recon_loss": np.mean(epoch_recons), "kl_loss": np.mean(epoch_kls)}, step=epoch)

            if epoch % config["validation_freq"] == 0 and epoch != 0:
                for test_idx, (test_batch) in enumerate(tqdm(test_loader, desc="Testing")):
                    rng, _rng_test = jax.random.split(rng)
                    test_loss, (test_recon, test_kl) = jit_eval(train_state.params, train_state.apply_fn, test_batch, _rng_test, beta)
                    epoch_test_losses.append(float(test_loss))
                    epoch_test_recons.append(float(test_recon))
                    epoch_test_kls.append(float(test_kl))


                    if epoch % config["render_freq"] == 0 and epoch != 0 and test_idx == 0:
                        restore_to_26ch_batch = jax.vmap(restore_to_26ch, in_axes=0)
                        restore_from_obs_batch = jax.vmap(restore_from_obs, in_axes=0)
                        
                        # GT_render
                        restored_gt_26 = restore_to_26ch_batch(test_batch)
                        restored_gt = restore_from_obs_batch(restored_gt_26)
                        test_states = FilteredState(**restored_gt)
                        test_frame = [
                            viz.custom_get_frame(jax.tree.map(lambda x: x[step], test_states), agent_view_size)
                            for step in range(config["n_render_samples"])]
                        
                        # VAE_render
                        rng, _rng_render = jax.random.split(rng)
                        vae_render, _, _ = train_state.apply_fn(train_state.params, test_batch, _rng_render)
                        vae_render = (jax.nn.sigmoid(vae_render) > 0.5).astype(jnp.uint8)

                        vae_render_26 = restore_to_26ch_batch(vae_render)
                        restored_pred = restore_from_obs_batch(vae_render_26)
                        test_states_pred = FilteredState(**restored_pred)
                        
                        vae_frame = [
                            viz.custom_get_frame(jax.tree.map(lambda x: x[step], test_states_pred), agent_view_size)
                            for step in range(config["n_render_samples"])]

                        combined = concat_images_with_labels([test_frame, vae_frame])                           
                        for i, canvas in enumerate(combined):
                            wandb.log({f"comparison_{i:03d}": wandb.Image(canvas)}, step=epoch)
                
                
                mean_recon = np.mean(epoch_test_recons)
                mean_kl = np.mean(epoch_test_kls)
                
                wandb.log({"test_recon_loss": mean_recon, "test_kl_loss": mean_kl}, step=epoch)

                os.makedirs(checkpoint_path, exist_ok=True)
                for kl in best_recon_wrt_kl:
                    if mean_kl < kl and mean_recon < best_recon_wrt_kl[kl]:
                        best_recon_wrt_kl[kl] = mean_recon
                        ckpt_path = f"{checkpoint_path}/vae_seed{config['SEED']}_kl{kl}.pkl"
                        with open(ckpt_path, "wb") as f:
                            pickle.dump({'params': train_state.params, 'config': config, "key": rng}, f)

        return {"train_state": train_state, "key": rng}

    return train

def single_run(config):
    config = OmegaConf.to_container(config)

    if config["WANDB_MODE"] == "online":
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])

    name = f"VAE_layout_{config['LAYOUT_DATA_FILE'].replace('.h5', '').split('layout_dataset_')[1]}_seed{config['SEED']}"
    
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "SP"],
        config=config,
        mode=config["WANDB_MODE"],
        name=name
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rng, rng_train = jax.random.split(rng)

    data_path = os.path.join(config["LAYOUT_DATA_PATH"], config["LAYOUT_DATA_FILE"])
    data = load_h5(data_path)
    images_train, images_test = split_dataset(
        data["agent_0"], config["VALIDATION_RATIO"])

    processed_train = input_processing(images_train)
    processed_test = input_processing(images_test)

    train_fn = make_train(config, processed_train, processed_test)
    out = train_fn(rng_train)
    print("Done.")

def tune(config):
    import copy

    config = OmegaConf.to_container(config)

    def wrapped_train():
        
        wandb.init(
            tags=["IPPO", "RNN", "SP"],
            entity=config["ENTITY"],
            config=config,
            project=config["PROJECT"],
            mode=config["WANDB_MODE"],
        )

        run_config = copy.deepcopy(config)
        for k, v in dict(wandb.config).items():
            run_config[k] = v

        wandb.run.name = f"VAE_layout_{run_config['LAYOUT_DATA_FILE'].replace('.h5', '').split('layout_dataset_')[1]}_seed{run_config['SEED']}"
        
        rng = jax.random.PRNGKey(run_config["SEED"])
        rng, rng_train = jax.random.split(rng)

        data_path = os.path.join(run_config["LAYOUT_DATA_PATH"], run_config["LAYOUT_DATA_FILE"])
        data = load_h5(data_path)
        images_train, images_test = split_dataset(data["agent_0"], run_config["VALIDATION_RATIO"])

        processed_train = input_processing(images_train)
        processed_test  = input_processing(images_test)

        train_fn = make_train(run_config, processed_train, processed_test)
        train_fn(rng_train)

    sweep_config = {
        "name": "vae_sweep",
        "method": "grid",
        "metric": {
            "name": "test_recon_loss",
            "goal": "minimize",
        },
        "parameters": {
            "BATCH_SIZE_TRAIN": {"values": [512, 1024, 2048, 4096]},
            "latent_dim": {"values": [32, 64, 128]},
            "LR": {"values": [1e-4, 1e-5]},
        },
    }

    with open("private.yaml") as f:
        private_info = yaml.load(f, Loader=yaml.FullLoader)
    wandb.login(key=private_info["wandb_key"])
   
    sweep_id = wandb.sweep(
        sweep_config, entity=config["ENTITY"], project=config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_train)


@hydra.main(version_base=None, config_path="config", config_name="vae_config")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)

if __name__ == "__main__":
    main()