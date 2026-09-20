import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked.common import (
    OBJECT_TO_INDEX,
    COLOR_TO_INDEX,
)
from jaxmarl.environments.overcooked.overcooked import POT_EMPTY_STATUS
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import h5py
import pickle


# Static env only: pot(10), wall(11), onion_pile(12), plate_pile(14), goal(15)
STATIC_TRAIN_CHANNELS = [10, 11, 12, 14, 15]


def input_processing(images):
    """images: (..., H, W, 26) → (..., H, W, 5)"""
    return images[:, :, :, STATIC_TRAIN_CHANNELS]


def input_processing_crop(images):
    """images: (..., H, W, 26) → (..., 5, 5, 5)"""
    
    
    images = images[..., STATIC_TRAIN_CHANNELS]  # (batch, H, W, 5) -> 

    cropped_images = images[..., :5, :5, :]     # (batch, 5, 5, 5)
    
    return cropped_images

def restore_to_26ch(pred_obs):
    """pred_obs: (..., H, W, 5) → (..., H, W, 26)"""
    H, W = pred_obs.shape[:2]
    full_obs = jnp.zeros((H, W, 26), dtype=jnp.uint8)
    full_obs = full_obs.at[:, :, STATIC_TRAIN_CHANNELS].set(pred_obs)
    return full_obs

    
def load_h5(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][:].reshape(-1, *f[k].shape[2:]) for k in f.keys()}

def concat_images_with_labels(images):
    font = ImageFont.load_default()

    canvas_list = []
    for f_idx, (image1, image2) in enumerate(zip(images[0], images[1])):
        image1, image2 = map(Image.fromarray, (image1, image2))

        width, height = image1.size

        canvas = Image.new("RGB", (width * 2, height), color=(255, 255, 255))
        canvas.paste(image1, (0, 0))
        canvas.paste(image2, (width, 0))

        draw = ImageDraw.Draw(canvas)

        label_text = f"Layout: {f_idx}"
        draw.text((10, 10), label_text, fill=(0, 0, 0), font=font)

        canvas_list.append(canvas)

    return canvas_list


def split_dataset(dataset, validation_ratio):
    num_data = len(dataset)
    random_indices = np.random.permutation(np.arange(num_data))
    validation_split = int(validation_ratio * num_data)
    train_indices = random_indices[:-validation_split]
    validation_indices = random_indices[-validation_split:]
    return dataset[train_indices], dataset[validation_indices]


class DataLoader:
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(data)

    def __len__(self):
        return self.n_samples // self.batch_size

    def __iter__(self):
        indices = np.random.permutation(self.n_samples) if self.shuffle else np.arange(self.n_samples)
        for i in range(len(self)):
            batch_idx = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield jnp.array(self.data[batch_idx])

def restore_from_obs(obs, agent_view_size=5):
    """
    obs: (H, W, 26) agent_0 기준
    returns: agent_dir_idx (scalar), agent_inv (scalar), maze_map (H+pad*2, W+pad*2, 3)
    """
    H, W = obs.shape[:2]
    padding = agent_view_size - 1  # 4

    # agent_dir_idx
    agent_0_dir_idx = jnp.argmax(jnp.sum(obs[:, :, 2:6], axis=(0, 1)))
    agent_1_dir_idx = jnp.argmax(jnp.sum(obs[:, :, 6:10], axis=(0, 1)))

    agent_dir_idx = jnp.array([agent_0_dir_idx, agent_1_dir_idx])


    # agent_inv (empty for both agents)
    agent_inv =  jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['empty']])

    # static-only 복원 시 agent 채널 0,1이 전부 0 → 에이전트 타일/색 그리기 스킵
    has_agent_0 = jnp.any(obs[:, :, 0])
    has_agent_1 = jnp.any(obs[:, :, 1])

    # maze_map channel 0: object type (H, W)
    obj_map = jnp.zeros((H, W), dtype=jnp.uint8)
    obj_map = jnp.where(obs[:, :, 11], OBJECT_TO_INDEX['wall'],       obj_map)
    obj_map = jnp.where(obs[:, :, 10], OBJECT_TO_INDEX['pot'],        obj_map)
    obj_map = jnp.where(obs[:, :, 12], OBJECT_TO_INDEX['onion_pile'], obj_map)
    obj_map = jnp.where(obs[:, :, 14], OBJECT_TO_INDEX['plate_pile'], obj_map)
    obj_map = jnp.where(obs[:, :, 15], OBJECT_TO_INDEX['goal'],       obj_map)
    obj_map = jnp.where(obs[:, :, 23], OBJECT_TO_INDEX['onion'],      obj_map)
    obj_map = jnp.where(obs[:, :, 22], OBJECT_TO_INDEX['plate'],      obj_map)
    obj_map = jnp.where(obs[:, :, 21], OBJECT_TO_INDEX['dish'],       obj_map)
    obj_map = jax.lax.cond(has_agent_0, lambda: jnp.where(obs[:, :, 0], OBJECT_TO_INDEX['agent'], obj_map), lambda: obj_map)
    obj_map = jax.lax.cond(has_agent_1, lambda: jnp.where(obs[:, :, 1], OBJECT_TO_INDEX['agent'], obj_map), lambda: obj_map)
    obj_map = jnp.where(obj_map == 0, OBJECT_TO_INDEX['empty'], obj_map)

    # maze_map channel 1: color
    agent_0_pos = jnp.argwhere(obs[:, :, 0], size=1)[0]
    y0, x0 = agent_0_pos[0], agent_0_pos[1]
    agent_1_pos = jnp.argwhere(obs[:, :, 1], size=1)[0]
    y1, x1 = agent_1_pos[0], agent_1_pos[1]
    color_map = jnp.zeros((H, W), dtype=jnp.uint8)
    color_map = jnp.where(obs[:, :, 11], COLOR_TO_INDEX['grey'],   color_map)  # wall
    color_map = jnp.where(obs[:, :, 10], COLOR_TO_INDEX['black'],  color_map)  # pot
    color_map = jnp.where(obs[:, :, 12], COLOR_TO_INDEX['yellow'], color_map)  # onion_pile
    color_map = jnp.where(obs[:, :, 14], COLOR_TO_INDEX['white'],  color_map)  # plate_pile
    color_map = jnp.where(obs[:, :, 15], COLOR_TO_INDEX['green'],  color_map)  # goal
    color_map = jnp.where(obs[:, :, 23], COLOR_TO_INDEX['yellow'], color_map)  # onion
    color_map = jnp.where(obs[:, :, 22], COLOR_TO_INDEX['white'],  color_map)  # plate
    color_map = jnp.where(obs[:, :, 21], COLOR_TO_INDEX['white'],  color_map)  # dish
    color_map = jax.lax.cond(has_agent_0, lambda: color_map.at[y0, x0].set(COLOR_TO_INDEX['red']), lambda: color_map)
    color_map = jax.lax.cond(has_agent_1, lambda: color_map.at[y1, x1].set(COLOR_TO_INDEX['blue']), lambda: color_map)

    # maze_map channel 2: pot status
    pot_status_map = jnp.zeros((H, W), dtype=jnp.uint8)
    pot_status_map = jnp.where(obs[:, :, 10],
                               POT_EMPTY_STATUS - obs[:, :, 16] - obs[:, :, 20],
                               pot_status_map)

    # (H, W, 3)
    maze_map = jnp.stack([obj_map, color_map, pot_status_map], axis=-1)

    pad_value_obj = OBJECT_TO_INDEX['wall']
    padded_maze_map = jnp.pad(
        maze_map,
        pad_width=((padding, padding), (padding, padding), (0, 0)),
        mode='constant',
        constant_values=0
    )
    padded_obj = jnp.pad(
        obj_map,
        pad_width=((padding, padding), (padding, padding)),
        mode='constant',
        constant_values=pad_value_obj
    )
    padded_maze_map = padded_maze_map.at[:, :, 0].set(padded_obj)
    # Padding border: channel 1 = grey for wall (env convention)
    pad_grey = COLOR_TO_INDEX['grey']
    padded_maze_map = padded_maze_map.at[:padding, :, 1].set(pad_grey)
    padded_maze_map = padded_maze_map.at[-padding:, :, 1].set(pad_grey)
    padded_maze_map = padded_maze_map.at[padding:-padding, :padding, 1].set(pad_grey)
    padded_maze_map = padded_maze_map.at[padding:-padding, -padding:, 1].set(pad_grey)
    # Channel 2 at agent cells = agent_dir_idx (static-only면 스킵)
    padded_maze_map = jax.lax.cond(
        has_agent_0,
        lambda: padded_maze_map.at[padding + y0, padding + x0, 2].set(agent_0_dir_idx.astype(jnp.uint8)),
        lambda: padded_maze_map,
    )
    padded_maze_map = jax.lax.cond(
        has_agent_1,
        lambda: padded_maze_map.at[padding + y1, padding + x1, 2].set(agent_1_dir_idx.astype(jnp.uint8)),
        lambda: padded_maze_map,
    )

    state = {
        "agent_dir_idx": agent_dir_idx,
        "agent_inv": agent_inv,
        "maze_map": padded_maze_map,
    }

    return state

def load_checkpoint(ckpt_path):
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt["params"], ckpt["config"]

if __name__ == "__main__":
    obs = jnp.load('/app/baselines/CEC_UED/VAE/dataset/env_states_3e6_lr-20260224-071423.h5')
    agent_dir_idx, agent_inv, maze_map = restore_from_obs(obs)
    print(agent_dir_idx)
    print(agent_inv)
    print(maze_map)