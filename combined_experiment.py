"""Combine configured layout experiments into one participant session."""

import importlib
import os

import jax
import jax.numpy as jnp
import nicewebrl
from nicewebrl.nicejax import new_rng


LAYOUT_MODULES = {
  "counter_circuit": "counter_circuit_experiment",
  "coord_ring": "coord_ring_experiment",
  "asymm_advantages": "asymm_advantages_experiment",
  "forced_coord": "forced_coord_experiment",
  "cramped_room": "cramped_room_experiment",
}

selected_layout_text = os.environ.get(
  "NICEWEBRL_LAYOUTS",
  "counter_circuit,coord_ring,asymm_advantages,forced_coord,cramped_room",
)
selected_layout_names = selected_layout_text.split(",")

selected_layouts = []
for layout_name in selected_layout_names:
  layout_name = layout_name.strip()
  if layout_name:
    selected_layouts.append(layout_name)

all_blocks = []
layout_block_groups = []
for layout_index, layout_name in enumerate(selected_layouts):
  if layout_index == 0:
    os.environ["NICEWEBRL_INCLUDE_TUTORIAL"] = "1"
  else:
    os.environ["NICEWEBRL_INCLUDE_TUTORIAL"] = "0"

  module = importlib.import_module(LAYOUT_MODULES[layout_name])
  layout_blocks = module.all_blocks

  # Show the common instructions and tutorial once, before the first layout.
  if layout_index == 0:
    instruction_block = layout_blocks[0]
    algorithm_blocks = layout_blocks[1:]
    all_blocks.append(instruction_block)
  else:
    algorithm_blocks = layout_blocks[1:]

  group_start = len(all_blocks)
  all_blocks.extend(algorithm_blocks)
  group_end = len(all_blocks)
  layout_block_groups.append(list(range(group_start, group_end)))


class LayoutGroupedExperiment(nicewebrl.Experiment):
  """Keep layouts fixed while randomizing algorithms inside each layout."""

  async def get_block_order(self):
    saved_order = self.get_user_data("block_order")
    if saved_order is not None:
      return saved_order

    block_order = [0]
    rng_key = new_rng()

    for block_group in layout_block_groups:
      rng_key, group_key = jax.random.split(rng_key)
      group_indices = jnp.asarray(block_group)
      shuffled_group = jax.random.permutation(group_key, group_indices)
      block_order.extend(int(index) for index in shuffled_group)

    await self.set_user_data(block_order=block_order)
    return block_order

# Keep every layout together. The layout order is fixed, while algorithms are
# shuffled independently inside each layout:
#
#   first layout:  all selected algorithms in random order
#   second layout: all selected algorithms in a new random order
#   ...
#
# Each game remains immediately followed by its survey because both stages are
# contained in one block.
experiment = LayoutGroupedExperiment(
  blocks=all_blocks,
  randomize=True,
  name="combined_" + "_".join(selected_layouts),
)
