"""Combine configured layout experiments into one participant session."""

import importlib
import os

import nicewebrl


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
for layout_index, layout_name in enumerate(selected_layouts):
  if layout_index == 0:
    os.environ["NICEWEBRL_INCLUDE_TUTORIAL"] = "1"
  else:
    os.environ["NICEWEBRL_INCLUDE_TUTORIAL"] = "0"

  module = importlib.import_module(LAYOUT_MODULES[layout_name])
  layout_blocks = module.all_blocks
  # Show the common instructions and tutorial once, before the first layout.
  if layout_index == 0:
    all_blocks.extend(layout_blocks)
  else:
    all_blocks.extend(layout_blocks[1:])

# The tutorial stays first. Every algorithm-layout pair is randomized after it,
# while each game remains immediately followed by its survey.
experiment = nicewebrl.Experiment(
  blocks=all_blocks,
  randomize=[False] + [True] * (len(all_blocks) - 1),
  name="combined_" + "_".join(selected_layouts),
)
