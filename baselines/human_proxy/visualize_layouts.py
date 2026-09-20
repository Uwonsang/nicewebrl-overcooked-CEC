"""Renders the 5 padded 9x9 layouts used by `preprocess.py` (see `layouts.py`) to PNGs,
so they can be sanity-checked visually against the original human-data grids.

Usage:
    python -m baselines.human_proxy.visualize_layouts --out_dir /tmp/layout_previews
"""

import argparse
import ast
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

from jaxmarl.environments.overcooked import Overcooked, State
from jaxmarl.environments.overcooked.common import make_overcooked_map
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer

from baselines.human_proxy.layouts import LAYOUT_NAME_MAP, build_padded_layout
from baselines.human_proxy.preprocess import GRID_SIZE, POT_EMPTY_STATUS


def render_layout(jax_name, layout):
    env = Overcooked(layout=layout, random_reset=False)

    idx_to_pos = lambda arr: jnp.stack([jnp.asarray(arr) % GRID_SIZE, jnp.asarray(arr) // GRID_SIZE], axis=1)

    wall_map = jnp.zeros(GRID_SIZE * GRID_SIZE, dtype=jnp.uint32)
    wall_map = wall_map.at[jnp.asarray(layout["wall_idx"])].set(1)
    wall_map = wall_map.reshape(GRID_SIZE, GRID_SIZE).astype(jnp.bool_)

    agent_pos = idx_to_pos(layout["agent_idx"])
    pot_pos = idx_to_pos(layout["pot_idx"])

    # `custom_get_frame` (unlike `get_obs`) reads agent placement straight off maze_map,
    # so the real agent positions need to be baked in here (not just set on `state`).
    maze_map = make_overcooked_map(
        wall_map,
        goal_pos=idx_to_pos(layout["goal_idx"]),
        agent_pos=agent_pos,
        agent_dir_idx=jnp.zeros((2,), dtype=jnp.int32),
        plate_pile_pos=idx_to_pos(layout["plate_pile_idx"]),
        onion_pile_pos=idx_to_pos(layout["onion_pile_idx"]),
        pot_pos=pot_pos,
        pot_status=jnp.full((pot_pos.shape[0],), POT_EMPTY_STATUS, dtype=jnp.uint32),
        onion_pos=jnp.array([[-1, -1]]),
        plate_pos=jnp.array([[-1, -1]]),
        dish_pos=jnp.array([[-1, -1]]),
        pad_obs=True,
        num_agents=2,
        agent_view_size=env.agent_view_size,
    )

    state = State(
        agent_pos=agent_pos,
        agent_dir=jnp.zeros((2, 2), dtype=jnp.int8),
        agent_dir_idx=jnp.zeros((2,), dtype=jnp.int32),
        agent_inv=jnp.array([1, 1], dtype=jnp.uint8),  # empty
        goal_pos=jnp.zeros((2, 2), dtype=jnp.uint32),
        pot_pos=pot_pos,
        wall_map=wall_map,
        maze_map=maze_map,
        time=0,
        terminal=False,
    )

    viz = OvercookedVisualizer()
    return viz.custom_get_frame(state, agent_view_size=env.agent_view_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="baselines/human_proxy/data/origin/2019_hh_trials.csv")
    parser.add_argument("--out_dir", default="baselines/human_proxy/data/processed/layout_previews")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, usecols=["layout_name", "layout"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for raw_name, jax_name in LAYOUT_NAME_MAP.items():
        row = df[df["layout_name"] == raw_name].iloc[0]
        grid_rows = ast.literal_eval(row["layout"])
        layout, _row_offset = build_padded_layout(grid_rows, jax_name)

        frame = render_layout(jax_name, layout)
        out_path = out_dir / f"{raw_name}_{jax_name}.png"
        plt.imsave(out_path, frame)
        print(f"[{raw_name} -> {jax_name}] saved {out_path}")


if __name__ == "__main__":
    main()
