"""Converts raw Overcooked-AI human-human trial CSVs (2019/2020_hh_trials.csv) into
(obs, action) supervised pairs for behaviour cloning, using jaxmarl's own `Overcooked.get_obs`
so the resulting observations exactly match what the live jaxmarl env produces.

Only the 5 layouts in `layouts.LAYOUT_NAME_MAP` are supported (the ones whose raw grid maps
onto an existing jaxmarl 9x9 layout family). Rows for any other layout are skipped.

Usage:
    python -m baselines.human_proxy.preprocess \
        --csv baselines/human_proxy/data/2019_hh_trials.csv \
        --out_dir baselines/human_proxy/data/processed
"""

import argparse
import ast
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxmarl.environments.overcooked import Overcooked, State
from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX, make_overcooked_map

from baselines.human_proxy.layouts import LAYOUT_NAME_MAP, build_padded_layout

DIR_VEC_TO_IDX = {(0, -1): 0, (0, 1): 1, (1, 0): 2, (-1, 0): 3}
# raw overcooked_ai_py held item name -> jaxmarl OBJECT_TO_INDEX value
# NB: raw "dish" is an empty plate/bowl (jaxmarl calls this "plate"); raw "soup" is a
# cooked soup (jaxmarl calls this "dish").
RAW_ITEM_TO_JAX_INDEX = {
    "onion": OBJECT_TO_INDEX["onion"],
    "dish": OBJECT_TO_INDEX["plate"],
    "soup": OBJECT_TO_INDEX["dish"],
}
POT_EMPTY_STATUS = 23
GRID_SIZE = 9
CHUNK_SIZE = 4096


def action_to_index(action_pair):
    """A single agent's raw action: [dx, dy] or 'interact'/'stay' -> jaxmarl Actions int."""
    if isinstance(action_pair, str):
        if action_pair.lower() == "interact":
            return 5
        raise ValueError(f"unexpected string action {action_pair!r}")
    dx, dy = action_pair
    if (dx, dy) == (0, 0):
        return 4
    return DIR_VEC_TO_IDX[(dx, dy)]


def soup_pot_status(soup):
    n = len(soup["_ingredients"])
    if soup["is_ready"]:
        return 0
    if soup["is_cooking"]:
        return max(0, soup["cook_time"] - soup["cooking_tick"])
    return POT_EMPTY_STATUS - n


def build_base_maze_map(layout, env):
    """Static (no agents, empty pots) 9x9 maze_map, padded like `custom_reset` does."""
    idx = lambda arr: (jnp.asarray(arr) % GRID_SIZE, jnp.asarray(arr) // GRID_SIZE)

    def pos_of(key):
        x, y = idx(layout[key])
        return jnp.stack([x, y], axis=1)

    wall_map = jnp.zeros(GRID_SIZE * GRID_SIZE, dtype=jnp.uint32)
    wall_map = wall_map.at[jnp.asarray(layout["wall_idx"])].set(1)
    wall_map = wall_map.reshape(GRID_SIZE, GRID_SIZE).astype(jnp.bool_)

    pot_pos = pos_of("pot_idx")
    return make_overcooked_map(
        wall_map,
        goal_pos=pos_of("goal_idx"),
        agent_pos=jnp.zeros((2, 2), dtype=jnp.uint32),
        agent_dir_idx=jnp.zeros((2,), dtype=jnp.int32),
        plate_pile_pos=pos_of("plate_pile_idx"),
        onion_pile_pos=pos_of("onion_pile_idx"),
        pot_pos=pot_pos,
        pot_status=jnp.full((pot_pos.shape[0],), POT_EMPTY_STATUS, dtype=jnp.uint32),
        onion_pos=jnp.array([[-1, -1]]),
        plate_pos=jnp.array([[-1, -1]]),
        dish_pos=jnp.array([[-1, -1]]),
        pad_obs=True,
        num_agents=2,
        agent_view_size=env.agent_view_size,
    )


def build_state_arrays(group_df, layout, base_maze_map, row_offset):
    """Parses every row of `group_df` (all sharing one layout) into batched State fields.

    `row_offset` is the number of leading all-wall rows `build_padded_layout` trimmed off
    the raw grid (see `layouts.py`); raw y-positions must be shifted by the same amount to
    land in the padded grid's coordinate system.
    """
    pad = (base_maze_map.shape[0] - GRID_SIZE) // 2
    pot_positions = {
        (int(layout["pot_idx"][i]) % GRID_SIZE, int(layout["pot_idx"][i]) // GRID_SIZE)
        for i in range(len(layout["pot_idx"]))
    }
    base_maze_np = np.asarray(base_maze_map)

    n = len(group_df)
    maze_maps = np.tile(base_maze_np[None], (n, 1, 1, 1))
    agent_pos = np.zeros((n, 2, 2), dtype=np.uint32)
    agent_dir_idx = np.zeros((n, 2), dtype=np.int32)
    agent_inv = np.zeros((n, 2), dtype=np.uint8)
    times = np.zeros((n,), dtype=np.int32)
    actions = np.zeros((n, 2), dtype=np.int32)
    valid = np.ones((n,), dtype=bool)

    for row_i, (_, row) in enumerate(group_df.iterrows()):
        try:
            state = json.loads(row["state"])
            joint_action = ast.literal_eval(row["joint_action"]) if isinstance(row["joint_action"], str) else row["joint_action"]
            actions[row_i, 0] = action_to_index(joint_action[0])
            actions[row_i, 1] = action_to_index(joint_action[1])
        except (ValueError, KeyError):
            valid[row_i] = False
            continue

        times[row_i] = state.get("timestep", row.get("cur_gameloop", 0))

        for p_i, player in enumerate(state["players"]):
            x, y = player["position"]
            y -= row_offset
            agent_pos[row_i, p_i] = (x, y)
            agent_dir_idx[row_i, p_i] = DIR_VEC_TO_IDX[tuple(player["orientation"])]
            held = player.get("held_object")
            agent_inv[row_i, p_i] = RAW_ITEM_TO_JAX_INDEX[held["name"]] if held else OBJECT_TO_INDEX["empty"]

        objects = state["objects"]
        objects_iter = objects.values() if isinstance(objects, dict) else objects
        for obj in objects_iter:
            x, y = obj["position"]
            y -= row_offset
            if obj["name"] == "soup" and (x, y) in pot_positions:
                maze_maps[row_i, pad + y, pad + x, 2] = soup_pot_status(obj)
            else:
                maze_maps[row_i, pad + y, pad + x, 0] = RAW_ITEM_TO_JAX_INDEX[obj["name"]]
                maze_maps[row_i, pad + y, pad + x, 1] = 0
                maze_maps[row_i, pad + y, pad + x, 2] = 0

    return {
        "maze_map": maze_maps,
        "agent_pos": agent_pos,
        "agent_dir_idx": agent_dir_idx,
        "agent_inv": agent_inv,
        "time": times,
        "actions": actions,
        "valid": valid,
    }


def compute_obs(env, arrays):
    """Runs `env.get_obs` over every parsed timestep via vmap, chunked to bound memory."""
    n = arrays["maze_map"].shape[0]
    dummy = jnp.zeros((2, 2), dtype=jnp.uint32)

    def make_state(chunk):
        b = chunk["maze_map"].shape[0]
        return State(
            agent_pos=jnp.asarray(chunk["agent_pos"]),
            agent_dir=jnp.zeros((b, 2, 2), dtype=jnp.int8),
            agent_dir_idx=jnp.asarray(chunk["agent_dir_idx"]),
            agent_inv=jnp.asarray(chunk["agent_inv"]),
            goal_pos=jnp.tile(dummy[None], (b, 1, 1)),
            pot_pos=jnp.tile(dummy[None], (b, 1, 1)),
            wall_map=jnp.zeros((b, GRID_SIZE, GRID_SIZE), dtype=jnp.bool_),
            maze_map=jnp.asarray(chunk["maze_map"]),
            time=jnp.asarray(chunk["time"]),
            terminal=jnp.zeros((b,), dtype=jnp.bool_),
        )

    get_obs_batched = jax.jit(jax.vmap(env.get_obs))

    obs0_chunks, obs1_chunks = [], []
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        chunk = {k: v[start:end] for k, v in arrays.items() if k in ("maze_map", "agent_pos", "agent_dir_idx", "agent_inv", "time")}
        obs = get_obs_batched(make_state(chunk))
        obs0_chunks.append(np.asarray(obs["agent_0"]))
        obs1_chunks.append(np.asarray(obs["agent_1"]))

    return np.concatenate(obs0_chunks), np.concatenate(obs1_chunks)


def process_csv(csv_path, max_rows=None):
    """Returns {jax_layout_name: (obs, actions, group_id)}, kept separate per layout so
    a BC model can be trained on a single layout at a time."""
    usecols = ["state", "joint_action", "layout", "layout_name", "cur_gameloop", "trial_id", "player_0_id", "player_1_id"]
    df = pd.read_csv(csv_path, usecols=usecols, nrows=max_rows)
    df = df[df["layout_name"].isin(LAYOUT_NAME_MAP)]

    per_layout = {}
    for raw_name, jax_name in LAYOUT_NAME_MAP.items():
        group_df = df[df["layout_name"] == raw_name]
        if len(group_df) == 0:
            continue
        print(f"[{raw_name} -> {jax_name}] {len(group_df)} rows")

        grid_rows = ast.literal_eval(group_df.iloc[0]["layout"])
        layout, row_offset = build_padded_layout(grid_rows, jax_name)
        env = Overcooked(layout=layout, random_reset=False)
        base_maze_map = build_base_maze_map(layout, env)

        trial_key_full = (
            group_df["trial_id"].astype(str)
            + "_" + group_df["player_0_id"].astype(str)
            + "_" + group_df["player_1_id"].astype(str)
        ).values

        arrays = build_state_arrays(group_df, layout, base_maze_map, row_offset)
        valid = arrays.pop("valid")
        n_invalid = (~valid).sum()
        if n_invalid:
            print(f"  dropping {n_invalid} unparseable rows")
        arrays = {k: v[valid] for k, v in arrays.items()}
        trial_key = trial_key_full[valid]

        obs0, obs1 = compute_obs(env, arrays)
        obs = np.concatenate([obs0, obs1], axis=0)
        actions = np.concatenate([arrays["actions"][:, 0], arrays["actions"][:, 1]], axis=0)
        group_id = np.concatenate([trial_key, trial_key])

        per_layout[jax_name] = (obs, actions, group_id)

    return per_layout


def train_test_split_by_group(group_id, train_frac=0.8, seed=0):
    unique_groups = np.unique(group_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    n_train = int(len(unique_groups) * train_frac)
    train_groups = set(unique_groups[:n_train])
    is_train = np.array([g in train_groups for g in group_id])
    return is_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default= "baselines/human_proxy/data/origin/2019_hh_trials.csv")
    parser.add_argument("--out_dir", default="baselines/human_proxy/data/processed")
    parser.add_argument("--max_rows", type=int, default=None, help="for quick smoke tests")
    args = parser.parse_args()

    per_layout = process_csv(args.csv, max_rows=args.max_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.csv).stem
    for jax_name, (obs, actions, group_id) in per_layout.items():
        is_train = train_test_split_by_group(group_id)
        np.savez_compressed(out_dir / f"{stem}_{jax_name}_train.npz", obs=obs[is_train], actions=actions[is_train])
        np.savez_compressed(out_dir / f"{stem}_{jax_name}_test.npz", obs=obs[~is_train], actions=actions[~is_train])
        print(f"[{jax_name}] saved {is_train.sum()} train / {(~is_train).sum()} test samples to {out_dir}")


if __name__ == "__main__":
    main()
