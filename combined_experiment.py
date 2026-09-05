"""Combine configured layout experiments into one participant session."""

import importlib
import os

import nicewebrl


LAYOUT_MODULES = {
  "counter_circuit": "counter_circuit_experiment",
  "coord_ring": "coord_ring_experiment",
}

selected_layouts = [
  name.strip()
  for name in os.environ.get(
    "NICEWEBRL_LAYOUTS", "counter_circuit,coord_ring"
  ).split(",")
  if name.strip()
]

all_blocks = []
for layout_index, layout_name in enumerate(selected_layouts):
  module = importlib.import_module(LAYOUT_MODULES[layout_name])
  layout_blocks = module.all_blocks
  # Show the common instructions and tutorial once, before the first layout.
  all_blocks.extend(layout_blocks if layout_index == 0 else layout_blocks[1:])

# The tutorial stays first. Every algorithm-layout pair is randomized after it,
# while each game remains immediately followed by its survey.
experiment = nicewebrl.Experiment(
  blocks=all_blocks,
  randomize=[False] + [True] * (len(all_blocks) - 1),
  name="combined_" + "_".join(selected_layouts),
)
