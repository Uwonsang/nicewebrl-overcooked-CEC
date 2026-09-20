import os
import sys
import pickle
import h5py
import numpy as np
from scipy import ndimage
import jax
import jax.numpy as jnp
from baselines.human_proxy.bc_agent import BCPolicy

EVAL_LAYOUTS_9 = [
    "cramped_room_9",
    "asymm_advantages_9",
    "coord_ring_9",
    "counter_circuit_9",
    "forced_coord_9",
]

FINETUNE_UPDATES_BY_FOLDER = {
    32: 366210,
    64: 183105,
    128: 91552,
    256: 45776,
}


def get_finetune_checkpoint_path(checkpoint_root, seed):
    folder_size = int(os.path.basename(os.path.normpath(checkpoint_root)))
    num_updates = FINETUNE_UPDATES_BY_FOLDER[folder_size]
    return os.path.join(
        checkpoint_root,
        f"seed{seed}",
        f"seed{seed}_ckpt0_improved_updates{num_updates}.pkl",
    )

def classify_layout(maze_map_9x9_ch0: np.ndarray) -> str:
    passable = (maze_map_9x9_ch0 == 1) | (maze_map_9x9_ch0 == 10)
    n_passable = int(passable.sum())
    _, n_components = ndimage.label(passable)

    if n_passable == 6 and n_components == 1:
        return "cramped_room_9"
    elif n_passable == 6 and n_components == 2:
        return "forced_coord_9"
    elif n_passable == 8:
        return "coord_ring_9"
    elif n_passable == 14 and n_components == 1:
        return "counter_circuit_9"
    elif n_passable == 14 and n_components >= 2:
        return "asymm_advantages_9"
    return "unknown"


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
        f.create_dataset(
            "update_step",
            shape=(num_updates,),
            dtype=np.int32,
            chunks=(1,),
        )
    print(f"Initialized HDF5 at {save_path}")

def save_to_hdf5(save_path, field_names, arrays, update_idx, update_step):
    idx = int(update_idx)
    with h5py.File(save_path, "a") as f:
        for k, v in zip(field_names, arrays):
            f[k][idx] = np.asarray(v)
        f["update_step"][idx] = int(update_step)

def make_eval_envs_overcooked(config):
    import jaxmarl
    from jaxmarl.environments.overcooked import overcooked_layouts
    from jaxmarl.wrappers.baselines import LogWrapper
    
    if config["ENV_NAME"] != "overcooked":
        return {}
    if "EVAL_KWARGS" not in config:
        return {}

    allowed = {"max_steps", "random_reset", "check_held_out", "shuffle_inv_and_pot"}
    base_env_kwargs = {k: v for k, v in config["EVAL_KWARGS"].items() if k in allowed}

    envs = {}
    for layout_name in EVAL_LAYOUTS_9:
        env_kwargs = dict(base_env_kwargs)
        env_kwargs["layout"] = overcooked_layouts[layout_name]
        base_env = jaxmarl.make(config["ENV_NAME"], **env_kwargs)
        envs[layout_name] = LogWrapper(
            base_env,
            env_params={"random_reset_fn": config["EVAL_KWARGS"]["random_reset_fn"]},
        )
    return envs

def load_human_proxy_params(ckpt_dir, num_seeds, layout_names=None):
    """Load human_proxy (BC) checkpoints, stacked across seeds.

    Returns {layout_name_9: params_pytree} where each leaf has a leading `num_seeds` axis.
    When ``layout_names`` is omitted, checkpoints for every evaluation layout are loaded.
    """
    if layout_names is None:
        layout_names = EVAL_LAYOUTS_9

    params_by_layout = {}
    for layout_name_9 in layout_names:
        if layout_name_9 not in EVAL_LAYOUTS_9:
            supported_layouts = ", ".join(EVAL_LAYOUTS_9)
            raise ValueError(
                f"Unsupported BC evaluation layout '{layout_name_9}'. "
                f"Choose one of: {supported_layouts}"
            )
        layout_name = layout_name_9[:-2]  # strip the CEC_UED-only "_9" suffix
        seed_params = []
        for seed in range(num_seeds):
            ckpt_path = os.path.join(ckpt_dir, layout_name, f"bc_overcooked_{layout_name}_seed{seed}.pkl")
            with open(ckpt_path, "rb") as f:
                seed_params.append(pickle.load(f))
        params_by_layout[layout_name_9] = jax.tree.map(lambda *x: jnp.stack(x), *seed_params)
    return params_by_layout
