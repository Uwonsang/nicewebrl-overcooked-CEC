"""Evaluate trained policies with BC human proxies using human-study logs.

Each rollout is written as a length-prefixed MessagePack ``.json`` file with
the same EnvStage record shape produced by :mod:`paper_metrics`.  The BC proxy
occupies the fields that normally describe the human participant.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import jaxmarl
import msgpack
import numpy as np
import yaml
from flax.serialization import to_bytes
from jaxmarl.environments.overcooked.layouts import overcooked_layouts
from nicewebrl import TimestepWrapper

from actor_networks import ActorCriticE3T, ActorCriticRNN, ScannedRNN
from human_proxy.bc_agent import BCPolicy
from idaac_actor_networks import ActorCriticIDAAC
from paper_metrics import ACTION_NAMES, DELIVERY_REWARD, _collision_details


DEFAULT_MODEL_ROOT = Path("/mnt/nas/wonsang/crossenv_ued/models/ICRL")
DEFAULT_PROXY_ROOT = Path("human_proxy/checkpoints")
DEFAULT_OUTPUT_DIR = Path("data/xp_human_proxy")
DEFAULT_MODELS = (
  "ippo",
  "e3t",
  "fcp",
  "cec_64",
  "cec_idaac_32",
  "cec_idaac_256",
)
DEFAULT_LAYOUTS = (
  "counter_circuit",
  "coord_ring",
  "asymm_advantages",
  "forced_coord",
  "cramped_room",
)

UI_ACTION_INDEX = {3: 0, 1: 1, 2: 2, 0: 3, 4: 4, 5: 5}
ACTION_KEYS = {
  0: "ArrowRight",
  1: "ArrowDown",
  2: "ArrowLeft",
  3: "ArrowUp",
  4: "s",
  5: " ",
}
KOREAN_ACTION_NAMES = {
  0: "오른쪽",
  1: "아래",
  2: "왼쪽",
  3: "위",
  4: "대기",
  5: "상호작용",
}


@dataclass(frozen=True)
class ModelSpec:
  name: str
  network: str
  checkpoint_pattern: str
  map_specific: bool


MODEL_SPECS = {
  "ippo": ModelSpec("ippo", "rnn", "IPPO/{layout}/seed*/seed*_best.pkl", True),
  "e3t": ModelSpec("e3t", "e3t", "E3T/{layout}/seed*/seed*_best_e3t.pkl", True),
  "fcp": ModelSpec("fcp", "rnn", "FCP/{layout}/seed*/fcp_seed*_best.pkl", True),
  "cec_64": ModelSpec(
    "cec_64", "rnn", "CEC/64/seed*/seed*_ckpt0_improved_updates*.pkl", False
  ),
  "cec_idaac_32": ModelSpec(
    "cec_idaac_32",
    "idaac",
    "CEC_IDAAC/32/seed*/seed*_ckpt0_improved_updates*.pkl",
    False,
  ),
  "cec_idaac_256": ModelSpec(
    "cec_idaac_256",
    "idaac",
    "CEC_IDAAC/256/seed*/seed*_ckpt0_improved_updates*.pkl",
    False,
  ),
}

# IPPO seed 4 is excluded from the evaluation set. The remaining IPPO seeds
# (0, 1, 2, 3, 5, 6) keep its seed count aligned with the other algorithms.
EXCLUDED_MODEL_SEEDS = {"ippo": {4}}


def normalize_layout(layout: str) -> tuple[str, str]:
  short_name = layout[:-2] if layout.endswith("_9") else layout
  layout_name = f"{short_name}_9"
  if layout_name not in overcooked_layouts:
    raise ValueError(f"Unknown Overcooked layout: {layout}")
  return short_name, layout_name


def parse_seed(path: Path) -> int:
  for part in reversed(path.parts):
    match = re.fullmatch(r"seed(\d+)", part)
    if match:
      return int(match.group(1))
  match = re.search(r"seed(\d+)", path.name)
  if not match:
    raise ValueError(f"Could not determine seed from {path}")
  return int(match.group(1))


def discover_model_checkpoints(
  model_root: Path,
  spec: ModelSpec,
  layout_name: str,
  requested_seeds: set[int] | None,
  checkpoint_path: Path | None = None,
) -> list[tuple[int, Path]]:
  relative_pattern = spec.checkpoint_pattern.format(layout=layout_name)
  if checkpoint_path is None:
    searched_location = model_root / relative_pattern
    paths = sorted(model_root.glob(relative_pattern))
  elif checkpoint_path.is_file():
    searched_location = checkpoint_path
    paths = [checkpoint_path]
  elif checkpoint_path.is_dir():
    # First interpret the supplied directory as the common ICRL root.
    paths = sorted(checkpoint_path.glob(relative_pattern))
    if paths:
      searched_location = checkpoint_path / relative_pattern
    else:
      # Also accept an algorithm root, layout root, or shared CEC root.
      filename_pattern = Path(spec.checkpoint_pattern).name
      if spec.map_specific and (checkpoint_path / layout_name).is_dir():
        search_root = checkpoint_path / layout_name
      else:
        search_root = checkpoint_path
        if spec.map_specific and search_root.name != layout_name:
          raise ValueError(
            f"Checkpoint directory for map-specific model '{spec.name}' must "
            f"be an ICRL root, contain a '{layout_name}' directory, or itself "
            f"be that directory: {checkpoint_path}"
          )
      searched_location = search_root / "seed*" / filename_pattern
      paths = sorted(search_root.glob(f"seed*/{filename_pattern}"))
  else:
    raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

  by_seed: dict[int, Path] = {}
  for path in paths:
    seed = parse_seed(path)
    if seed in EXCLUDED_MODEL_SEEDS.get(spec.name, set()):
      continue
    if requested_seeds is None or seed in requested_seeds:
      by_seed[seed] = path
  if not by_seed:
    raise FileNotFoundError(
      f"No checkpoints for {spec.name} matching {searched_location}"
    )
  return sorted(by_seed.items())


def discover_proxy_checkpoints(
  proxy_root: Path, layout_short: str, requested_seeds: set[int] | None
) -> list[tuple[int, Path]]:
  paths = sorted(
    (proxy_root / layout_short).glob(
      f"bc_overcooked_{layout_short}_seed*.pkl"
    )
  )
  selected = []
  for path in paths:
    seed = parse_seed(path)
    if requested_seeds is None or seed in requested_seeds:
      selected.append((seed, path))
  if not selected:
    raise FileNotFoundError(
      f"No human-proxy checkpoints found under {proxy_root / layout_short}"
    )
  return selected


def load_pickle(path: Path) -> Any:
  with path.open("rb") as stream:
    return pickle.load(stream)


def load_model_params(path: Path) -> Any:
  checkpoint = load_pickle(path)
  if not isinstance(checkpoint, dict) or "params" not in checkpoint:
    raise ValueError(f"Checkpoint has no 'params' entry: {path}")
  return checkpoint["params"]


def make_environment(layout_name: str, max_timesteps: int):
  env = jaxmarl.make(
    "overcooked",
    layout=overcooked_layouts[layout_name],
    random_reset=False,
    check_held_out=False,
    shuffle_inv_and_pot=False,
    random_reset_fn="reset_all",
    max_steps=max_timesteps,
  )
  wrapped = TimestepWrapper(
    env, autoreset=False, reset_w_batch_dim=False, use_params=False
  )
  return env, jax.jit(wrapped.reset), jax.jit(wrapped.step)


def make_network(network_name: str, config: dict[str, Any], action_dim: int):
  if network_name == "rnn":
    return ActorCriticRNN(action_dim=action_dim, config=config)
  if network_name == "e3t":
    return ActorCriticE3T(action_dim=action_dim, config=config)
  if network_name == "idaac":
    return ActorCriticIDAAC(action_dim=action_dim, config=config)
  raise ValueError(f"Unknown network type: {network_name}")


def initial_hidden(network_name: str, config: dict[str, Any]):
  hidden_size = int(config["GRU_HIDDEN_DIM"])
  if network_name == "idaac":
    return ActorCriticIDAAC.initialize_carry(1, hidden_size)
  return ScannedRNN.initialize_carry(1, hidden_size)


def make_model_step(network, *, argmax: bool, beta: float):
  def step(params, hidden, observation, done, agent_positions, rng):
    inputs = (
      observation.reshape(1, 1, -1),
      jnp.asarray(done).reshape(1, 1),
      agent_positions.astype(jnp.int32).reshape(1, 1, 2, 2),
    )
    outputs = network.apply(params, hidden, inputs)
    next_hidden, policy = outputs[:2]
    logits = policy.logits * beta
    probabilities = jax.nn.softmax(logits, axis=-1)
    if argmax:
      action = jnp.argmax(probabilities, axis=-1)
    else:
      action = jax.random.categorical(rng, logits, axis=-1)
    return next_hidden, action.reshape(()), probabilities.reshape(-1)

  return jax.jit(step)


def make_proxy_step(*, argmax: bool, beta: float):
  network = BCPolicy()

  def step(params, observation, rng):
    logits = network.apply(
      params, observation[jnp.newaxis, ...].astype(jnp.float32)
    ) * beta
    probabilities = jax.nn.softmax(logits, axis=-1)
    if argmax:
      action = jnp.argmax(probabilities, axis=-1)
    else:
      action = jax.random.categorical(rng, logits, axis=-1)
    return action.reshape(()), probabilities.reshape(-1)

  return jax.jit(step)


def write_record(stream, record: dict[str, Any]) -> None:
  payload = msgpack.packb(record)
  stream.write(len(payload).to_bytes(4, byteorder="big"))
  stream.write(payload)


def pseudo_user_id(
  algorithm: str,
  layout_short: str,
  model_seed: int,
  proxy_seed: int,
  model_agent: int,
  episode: int,
) -> str:
  text = (
    f"xp-{algorithm.replace('_', '-')}-{layout_short.replace('_', '-')}"
    f"-m{model_seed}-bc{proxy_seed}-a{model_agent}-e{episode}"
  )
  # Keep filenames compact while preserving a deterministic, readable prefix.
  digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
  return f"{text}-{digest}"


def record_for_step(
  *,
  timestep,
  next_timestep,
  model_action: int | None,
  proxy_action: int | None,
  model_probabilities: list[float] | None,
  proxy_probabilities: list[float] | None,
  model_agent: int,
  algorithm: str,
  model_seed: int,
  model_checkpoint: Path,
  proxy_seed: int,
  proxy_checkpoint: Path,
  layout_short: str,
  episode: int,
  rollout_seed: int,
  cumulative_collisions: int,
  collision: dict[str, bool] | None,
  user_id: str,
  session_id: str,
  nsteps: int,
  final: bool,
) -> dict[str, Any]:
  proxy_agent = 1 - model_agent
  reward = float(np.asarray(timestep.reward).sum())
  proxy_action_index = (
    UI_ACTION_INDEX[proxy_action] if proxy_action is not None else -1
  )
  terminal = bool(np.asarray(timestep.state.terminal))
  collision = collision or {
    "collision_event": None,
    "same_target_collision": None,
    "swap_collision": None,
  }
  data = {
    "image_seen_time": None,
    "action_taken_time": None,
    "computer_interaction": "timer"
    if final
    else ACTION_KEYS[proxy_action],
    "action_name": "timer"
    if final
    else KOREAN_ACTION_NAMES[proxy_action],
    "action_idx": proxy_action_index,
    "human_environment_action": proxy_action,
    "human_environment_action_name": ACTION_NAMES.get(proxy_action),
    "ai_action": model_action,
    "ai_action_name": ACTION_NAMES.get(model_action),
    "human_id": proxy_agent,
    "model_index": model_seed,
    "model_seed": model_seed,
    "model_checkpoint": str(model_checkpoint),
    "step_reward": reward,
    "delivery_event": reward / DELIVERY_REWARD,
    "collision_event": collision["collision_event"],
    "same_target_collision": collision["same_target_collision"],
    "swap_collision": collision["swap_collision"],
    "cumulative_collisions": cumulative_collisions,
    "agent_positions_before": None
    if final
    else np.asarray(timestep.state.agent_pos).tolist(),
    "agent_positions_after": None
    if final or next_timestep is None
    else np.asarray(next_timestep.state.agent_pos).tolist(),
    "timelimit": None,
    "timestep": to_bytes(jax.device_get(timestep)),
    # XP-only additions. Existing human-data readers ignore extra fields.
    "xp_episode": episode,
    "xp_rollout_seed": rollout_seed,
    "model_agent": model_agent,
    "human_proxy_seed": proxy_seed,
    "human_proxy_checkpoint": str(proxy_checkpoint),
    "model_action_probabilities": model_probabilities,
    "human_proxy_action_probabilities": proxy_probabilities,
  }
  return {
    "stage_idx": 0,
    "session_id": session_id,
    "data": data,
    "user_data": {"user_id": user_id, "age": None, "sex": None},
    "metadata": {
      "desc": "trained policy x human-proxy BC cross-play",
      "type": "EnvStage",
      "nsteps": nsteps,
      "nepisodes": 0,
      "nsuccesses": int(terminal),
      "xp": True,
      "algorithm": algorithm,
      "layout": layout_short,
    },
    "name": f"{algorithm}_{layout_short}",
    "body": "",
  }


def run_episode(
  *,
  env,
  reset_fn,
  env_step_fn,
  model_step_fn,
  proxy_step_fn,
  model_params,
  proxy_params,
  network_name: str,
  config: dict[str, Any],
  algorithm: str,
  layout_short: str,
  model_seed: int,
  model_checkpoint: Path,
  proxy_seed: int,
  proxy_checkpoint: Path,
  model_agent: int,
  episode: int,
  rollout_seed: int,
  max_timesteps: int,
  output_path: Path,
) -> dict[str, Any]:
  proxy_agent = 1 - model_agent
  model_agent_name = f"agent_{model_agent}"
  proxy_agent_name = f"agent_{proxy_agent}"
  rng = jax.random.PRNGKey(rollout_seed)
  rng, reset_rng = jax.random.split(rng)
  timestep = reset_fn(reset_rng)
  hidden = initial_hidden(network_name, config)
  cumulative_collisions = 0
  total_reward = 0.0
  deliveries = 0.0
  session_id = f"xp-{rollout_seed}"
  user_id = pseudo_user_id(
    algorithm,
    layout_short,
    model_seed,
    proxy_seed,
    model_agent,
    episode,
  )

  output_path.parent.mkdir(parents=True, exist_ok=True)
  temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
  with temporary_path.open("wb") as stream:
    nsteps = 0
    while nsteps < max_timesteps and not bool(np.asarray(timestep.last())):
      observations = env.get_obs(timestep.state)
      rng, model_rng, proxy_rng, env_rng = jax.random.split(rng, 4)
      hidden, model_action_array, model_probs_array = model_step_fn(
        model_params,
        hidden,
        observations[model_agent_name],
        timestep.last(),
        timestep.state.agent_pos,
        model_rng,
      )
      proxy_action_array, proxy_probs_array = proxy_step_fn(
        proxy_params, observations[proxy_agent_name], proxy_rng
      )
      model_action = int(np.asarray(model_action_array))
      proxy_action = int(np.asarray(proxy_action_array))
      joint_actions = [0, 0]
      joint_actions[model_agent] = model_action
      joint_actions[proxy_agent] = proxy_action
      collision = _collision_details(timestep.state, joint_actions)
      cumulative_collisions += int(collision["collision_event"])
      action_dict = {
        model_agent_name: jnp.asarray(model_action, dtype=jnp.int32),
        proxy_agent_name: jnp.asarray(proxy_action, dtype=jnp.int32),
      }
      next_timestep = env_step_fn(env_rng, timestep, action_dict)

      write_record(
        stream,
        record_for_step(
          timestep=timestep,
          next_timestep=next_timestep,
          model_action=model_action,
          proxy_action=proxy_action,
          model_probabilities=np.asarray(model_probs_array).tolist(),
          proxy_probabilities=np.asarray(proxy_probs_array).tolist(),
          model_agent=model_agent,
          algorithm=algorithm,
          model_seed=model_seed,
          model_checkpoint=model_checkpoint,
          proxy_seed=proxy_seed,
          proxy_checkpoint=proxy_checkpoint,
          layout_short=layout_short,
          episode=episode,
          rollout_seed=rollout_seed,
          cumulative_collisions=cumulative_collisions,
          collision=collision,
          user_id=user_id,
          session_id=session_id,
          nsteps=nsteps,
          final=False,
        ),
      )
      timestep = next_timestep
      step_reward = float(np.asarray(timestep.reward).sum())
      total_reward += step_reward
      deliveries += step_reward / DELIVERY_REWARD
      nsteps += 1

    # Human EnvStage writes one final "timer" record. It carries the reward
    # from the final transition, so analysis_extend sees the complete return.
    write_record(
      stream,
      record_for_step(
        timestep=timestep,
        next_timestep=None,
        model_action=None,
        proxy_action=None,
        model_probabilities=None,
        proxy_probabilities=None,
        model_agent=model_agent,
        algorithm=algorithm,
        model_seed=model_seed,
        model_checkpoint=model_checkpoint,
        proxy_seed=proxy_seed,
        proxy_checkpoint=proxy_checkpoint,
        layout_short=layout_short,
        episode=episode,
        rollout_seed=rollout_seed,
        cumulative_collisions=cumulative_collisions,
        collision=None,
        user_id=user_id,
        session_id=session_id,
        nsteps=nsteps,
        final=True,
      ),
    )
  temporary_path.replace(output_path)

  return {
    "user_id": user_id,
    "algorithm": algorithm,
    "layout": layout_short,
    "model_seed": model_seed,
    "human_proxy_seed": proxy_seed,
    "model_agent": f"agent_{model_agent}",
    "episode": episode,
    "rollout_seed": rollout_seed,
    "steps": nsteps,
    "reward": total_reward,
    "deliveries": deliveries,
    "collisions": cumulative_collisions,
    "model_checkpoint": str(model_checkpoint),
    "human_proxy_checkpoint": str(proxy_checkpoint),
    "log_file": str(output_path),
  }


def parse_int_set(values: list[int] | None) -> set[int] | None:
  return None if values is None else {int(value) for value in values}


def write_summary(path: Path, rows: Iterable[dict[str, Any]]) -> None:
  rows = list(rows)
  if not rows:
    return
  with path.open("w", newline="", encoding="utf-8-sig") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def read_summary(path: Path) -> list[dict[str, Any]]:
  if not path.exists():
    return []
  with path.open("r", newline="", encoding="utf-8-sig") as stream:
    return list(csv.DictReader(stream))


def infer_output_dir(model_root: Path, checkpoint_path: Path | None) -> Path:
  """Place results beside ``models/`` regardless of its mount prefix."""
  source_path = (checkpoint_path or model_root).resolve()
  if source_path.is_file():
    source_path = source_path.parent
  for candidate in (source_path, *source_path.parents):
    if candidate.name == "ICRL" and candidate.parent.name == "models":
      return candidate.parent.parent / "proxy_data"
  return DEFAULT_OUTPUT_DIR


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Cross-play trained policies with map-specific BC human proxies."
  )
  parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
  parser.add_argument(
    "--checkpoint-path",
    type=Path,
    help=(
      "Override checkpoint discovery for one selected algorithm. Accepts an "
      "algorithm directory, a layout directory, or one .pkl checkpoint."
    ),
  )
  parser.add_argument("--human-proxy-root", type=Path, default=DEFAULT_PROXY_ROOT)
  parser.add_argument(
    "--output-dir",
    type=Path,
    help=(
      "Result directory. By default it is inferred beside models/ as "
      "<crossenv_ued>/proxy_data."
    ),
  )
  parser.add_argument(
    "--models",
    nargs="+",
    choices=sorted(MODEL_SPECS),
    default=list(DEFAULT_MODELS),
  )
  parser.add_argument("--layouts", nargs="+", default=list(DEFAULT_LAYOUTS))
  parser.add_argument("--model-seeds", nargs="+", type=int)
  parser.add_argument("--human-proxy-seeds", nargs="+", type=int)
  parser.add_argument("--episodes", type=int, default=1)
  parser.add_argument("--max-timesteps", type=int, default=200)
  parser.add_argument("--world-seed", type=int, default=1)
  parser.add_argument("--beta", type=float, default=1.0)
  parser.add_argument(
    "--argmax",
    action="store_true",
    help="Choose the highest-probability action instead of sampling (default: sample).",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Replace existing rollout files. By default completed files are skipped.",
  )
  return parser


def main() -> None:
  args = build_parser().parse_args()
  if args.episodes < 1 or args.max_timesteps < 1:
    raise ValueError("--episodes and --max-timesteps must be positive")
  if args.beta <= 0:
    raise ValueError("--beta must be positive")
  if args.checkpoint_path is not None and len(args.models) != 1:
    raise ValueError("--checkpoint-path requires exactly one --models value")

  if args.output_dir is None:
    args.output_dir = infer_output_dir(args.model_root, args.checkpoint_path)

  args.output_dir.mkdir(parents=True, exist_ok=True)
  base_config = yaml.safe_load(Path("overcooked_config.yaml").read_text())
  requested_model_seeds = parse_int_set(args.model_seeds)
  requested_proxy_seeds = parse_int_set(args.human_proxy_seeds)
  proxy_step_fn = make_proxy_step(argmax=args.argmax, beta=args.beta)
  summary_path = args.output_dir / "summary.csv"
  existing_summaries = read_summary(summary_path)
  summaries_by_file = {
    str(row["log_file"]): row
    for row in existing_summaries
    if row.get("log_file")
  }
  rollout_index = 0

  manifest = {
    "model_root": str(args.model_root.resolve()),
    "checkpoint_path": str(args.checkpoint_path.resolve())
    if args.checkpoint_path is not None
    else None,
    "human_proxy_root": str(args.human_proxy_root.resolve()),
    "models": args.models,
    "layouts": args.layouts,
    "model_seeds": args.model_seeds,
    "human_proxy_seeds": args.human_proxy_seeds,
    "episodes": args.episodes,
    "max_timesteps": args.max_timesteps,
    "world_seed": args.world_seed,
    "beta": args.beta,
    "argmax": args.argmax,
  }
  (args.output_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
  )

  for layout_value in args.layouts:
    layout_short, layout_name = normalize_layout(layout_value)
    env, reset_fn, env_step_fn = make_environment(layout_name, args.max_timesteps)
    config = dict(base_config)
    config["ENV_KWARGS"] = dict(base_config["ENV_KWARGS"])
    config["layout_name"] = layout_name
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    proxy_checkpoints = discover_proxy_checkpoints(
      args.human_proxy_root, layout_short, requested_proxy_seeds
    )

    for algorithm in args.models:
      spec = MODEL_SPECS[algorithm]
      model_checkpoints = discover_model_checkpoints(
        args.model_root,
        spec,
        layout_name,
        requested_model_seeds,
        args.checkpoint_path,
      )
      network = make_network(
        spec.network, config, env.action_space(env.agents[0]).n
      )
      model_step_fn = make_model_step(
        network, argmax=args.argmax, beta=args.beta
      )

      for model_seed, model_checkpoint in model_checkpoints:
        model_params = load_model_params(model_checkpoint)
        for proxy_seed, proxy_checkpoint in proxy_checkpoints:
          proxy_params = load_pickle(proxy_checkpoint)
          for model_agent in (0, 1):
            for episode in range(args.episodes):
              rollout_seed = (
                args.world_seed
                + rollout_index * 1_000_003
                + model_seed * 10_007
                + proxy_seed * 101
                + model_agent * 17
                + episode
              ) % (2**32 - 1)
              user_id = pseudo_user_id(
                algorithm,
                layout_short,
                model_seed,
                proxy_seed,
                model_agent,
                episode,
              )
              output_path = (
                args.output_dir
                / f"user={user_id}_name={layout_short}_debug=0.json"
              )
              rollout_index += 1
              if output_path.exists() and not args.overwrite:
                print(f"skip existing: {output_path}")
                continue
              print(
                f"run {algorithm} m{model_seed} x BC{proxy_seed} "
                f"on {layout_short}, model=agent_{model_agent}, episode={episode}"
              )
              summary = run_episode(
                env=env,
                reset_fn=reset_fn,
                env_step_fn=env_step_fn,
                model_step_fn=model_step_fn,
                proxy_step_fn=proxy_step_fn,
                model_params=model_params,
                proxy_params=proxy_params,
                network_name=spec.network,
                config=config,
                algorithm=algorithm,
                layout_short=layout_short,
                model_seed=model_seed,
                model_checkpoint=model_checkpoint,
                proxy_seed=proxy_seed,
                proxy_checkpoint=proxy_checkpoint,
                model_agent=model_agent,
                episode=episode,
                rollout_seed=rollout_seed,
                max_timesteps=args.max_timesteps,
                output_path=output_path,
              )
              summaries_by_file[str(output_path)] = summary
              write_summary(summary_path, summaries_by_file.values())

  print(
    f"finished sweep; {len(summaries_by_file)} rollouts indexed; "
    f"output={args.output_dir}"
  )


if __name__ == "__main__":
  main()
