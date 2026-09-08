"""Utilities for loading map-specific and shared policy checkpoints."""

from pathlib import Path
import pickle

import jax
import jax.numpy as jnp


_PARAMETER_STACK_CACHE = {}


def load_parameter_stack(model_dir, *, pattern="*.pkl"):
  """Load matching pickle checkpoints into one parameter stack."""
  model_dir = Path(model_dir)
  cache_key = (str(model_dir.resolve()), pattern)

  if cache_key in _PARAMETER_STACK_CACHE:
    return _PARAMETER_STACK_CACHE[cache_key]

  checkpoint_paths = sorted(model_dir.glob(pattern))
  if not checkpoint_paths:
    raise FileNotFoundError(
      f"No checkpoints matching '{pattern}' found in {model_dir}"
    )

  parameters = []
  for checkpoint_path in checkpoint_paths:
    with checkpoint_path.open("rb") as f:
      checkpoint = pickle.load(f)
    if not isinstance(checkpoint, dict) or "params" not in checkpoint:
      raise ValueError(f"Checkpoint has no 'params' entry: {checkpoint_path}")
    parameters.append(checkpoint["params"])

  try:
    parameter_stack = jax.tree_util.tree_map(
      lambda *values: jnp.stack(values), *parameters
    )
  except Exception as exc:
    raise ValueError(
      f"Checkpoints in {model_dir} do not have the same parameter structure"
    ) from exc

  result = (
    parameter_stack,
    len(parameters),
    [str(path) for path in checkpoint_paths],
  )
  _PARAMETER_STACK_CACHE[cache_key] = result
  return result
