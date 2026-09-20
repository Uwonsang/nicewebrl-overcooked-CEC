"""Post-hoc critic-only sharpness for CEC and CEC_IDAAC checkpoints.

Only the two model families below are supported:

* CEC: ``ippo_general_gradient.ActorCriticRNN``
* CEC_IDAAC: ``idaac_general_gradient.ActorCriticRNN``

The script creates a fresh, fixed on-policy CEC rollout for every checkpoint,
then approximately maximizes the identical clipped critic loss in Keskar et
al.'s parameter-wise box with L-BFGS-B. Only the equally-sized critic paths are
perturbed. The rollout does not change while an epsilon is optimized.

Example (run from the repository root)::

    python -m baselines.CEC_UED.measure_cec_sharpness \
        --training-num-envs 256 --seeds 0 1 2 3 4 5 \
        --epsilons 0.001 0.0005 \
        --output sharpness_cec.json

Use ``--dry-run`` to inspect which checkpoints would be measured without
importing JAX or constructing an environment.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "baselines/CEC_UED/config/ippo_overcooked_CEC_gradient.yaml"
)
DEFAULT_ROOTS = {
    "CEC": Path("/app/nas/models/ICRL/CEC"),
    "CEC_IDDAC": Path(
        "/app/nas/models/ICRL/CEC_IDDAC"
    ),
}
CHECKPOINT_RE = re.compile(r"seed(?P<seed>\d+).*\.pkl$")


def _ensure_legacy_import_path() -> None:
    """Support training modules whose local imports predate packaging."""
    module_directory = str(Path(__file__).resolve().parent)
    if module_directory not in sys.path:
        sys.path.insert(0, module_directory)


@dataclass(frozen=True)
class Checkpoint:
    model: str
    training_num_envs: int
    seed: int
    path: Path


def discover_checkpoints(
    roots: Mapping[str, Path],
    models: Iterable[str],
    training_num_envs_filter: set[int] | None,
    seeds: set[int] | None,
) -> list[Checkpoint]:
    """Discover ``<num_envs>/seedN/*.pkl`` checkpoint files."""
    checkpoints: list[Checkpoint] = []
    for model in models:
        root = roots[model]
        if not root.is_dir():
            raise FileNotFoundError(f"Checkpoint root does not exist: {root}")
        for path in root.glob("*/seed*/*.pkl"):
            relative = path.relative_to(root)
            try:
                training_num_envs = int(relative.parts[0])
            except (ValueError, IndexError):
                continue
            match = CHECKPOINT_RE.search(path.name)
            if match is None:
                continue
            seed = int(match.group("seed"))
            if (
                training_num_envs_filter is not None
                and training_num_envs not in training_num_envs_filter
            ):
                continue
            if seeds is not None and seed not in seeds:
                continue
            checkpoints.append(
                Checkpoint(model, training_num_envs, seed, path.resolve())
            )
    return sorted(
        checkpoints,
        key=lambda item: (
            item.training_num_envs,
            item.seed,
            item.model,
            str(item.path),
        ),
    )


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in config file: {path}")
    return config


def _checkpoint_updates(payload: Mapping[str, Any], item: Checkpoint) -> int:
    for key in ("update_steps", "final_update_step"):
        if key in payload:
            return int(payload[key])
    match = re.search(r"updates(\d+)", item.path.name)
    if match:
        return int(match.group(1))
    # This only affects reward shaping. These are final checkpoints, so using
    # the expected final update count correctly makes its coefficient zero.
    return int(3e9 // 256 // item.training_num_envs)


def _prepare_config(
    config_path: Path,
    eval_num_envs: int,
    rollout_steps: int,
    sampled_actors: int,
) -> dict[str, Any]:
    config = _load_config(config_path)
    # PyYAML treats values such as ``3e9`` as strings. Normalize the fields
    # used in integer schedule arithmetic before applying ``//``.
    for key in ("TOTAL_TIMESTEPS", "MAX_TRAIN_STEPS"):
        try:
            config[key] = int(config[key])
        except (TypeError, ValueError):
            config[key] = int(float(config[key]))
    config["NUM_ENVS"] = int(eval_num_envs)
    config["NUM_ACTORS"] = 2 * int(eval_num_envs)
    config["NUM_STEPS"] = int(rollout_steps)
    config.setdefault("SHARPNESS", {})
    config["SHARPNESS"]["NUM_ACTORS"] = int(sampled_actors)
    config["SHARPNESS"]["NUM_STEPS"] = int(rollout_steps)
    config.setdefault("DAAC_ADV_COEF", 0.25)
    config.setdefault("IDAAC_ORDER_COEF", 0.001)
    config.setdefault("IDAAC_USE_NONLINEAR_CLF", False)
    config.setdefault("IDAAC_CLF_HIDDEN_SIZE", 4)
    return config


def _make_environment(config: dict[str, Any]):
    """Construct the same procedural CEC environment used during training."""
    _ensure_legacy_import_path()
    from baselines.CEC_UED.ippo_general_gradient import initialize_environment
    from jaxmarl.wrappers.baselines import LogWrapper

    env = initialize_environment(config)
    config["ACTION_DIM"] = env.action_space(env.agents[0]).n
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    return LogWrapper(
        env,
        env_params={
            "random_reset_fn": config["ENV_KWARGS"]["random_reset_fn"]
        },
    )


def _collect_batch(
    item: Checkpoint,
    payload: Mapping[str, Any],
    params: Any,
    config: dict[str, Any],
    env: Any,
    rollout_seed: int,
):
    _ensure_legacy_import_path()
    import jax
    import jax.numpy as jnp

    from baselines.CEC_UED.sharpness import collect_final_sharpness_batch

    if item.model == "CEC":
        from baselines.CEC_UED.ippo_general_gradient import (
            ActorCriticRNN,
            ScannedRNN,
            batchify,
            unbatchify,
        )

        network = ActorCriticRNN(int(config["ACTION_DIM"]), config=config)
        hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
    elif item.model == "CEC_IDDAC":
        from baselines.CEC_UED.idaac_general_gradient import (
            ActorCriticRNN,
            batchify,
            unbatchify,
        )

        network = ActorCriticRNN(int(config["ACTION_DIM"]), config=config)
        hstate = ActorCriticRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
    else:  # Kept explicit so another model family cannot silently enter.
        raise ValueError(f"Unsupported model: {item.model}")

    reset_key, runner_key = jax.random.split(jax.random.PRNGKey(rollout_seed))
    reset_keys = jax.random.split(reset_key, config["NUM_ENVS"])
    obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)
    done = jnp.zeros((config["NUM_ACTORS"],), dtype=bool)
    runner_state = (
        SimpleNamespace(params=params),
        env_state,
        obs,
        done,
        hstate,
        runner_key,
    )

    # These are final checkpoints. Their recorded update count reconstructs the
    # completed training schedule without conflating training NUM_ENVS with the
    # smaller post-hoc evaluation environment count.
    max_updates = _checkpoint_updates(payload, item)
    config["NUM_REWARD_SHAPING_STEPS"] = max(1, max_updates // 2)
    return network, collect_final_sharpness_batch(
        env=env,
        network=network,
        final_runner_state=runner_state,
        final_update_count=_checkpoint_updates(payload, item),
        config=config,
        batchify_fn=batchify,
        unbatchify_fn=unbatchify,
    )


def _cec_loss(network: Any, batch: Any, config: Mapping[str, Any]):
    _ensure_legacy_import_path()
    from baselines.CEC_UED.ippo_general_gradient import ppo_loss

    def loss(params):
        value, _ = ppo_loss(
            network,
            params,
            batch.initial_hstate,
            batch,
            batch.advantages,
            batch.targets,
            config,
        )
        return value

    return loss


def _idaac_loss(
    network: Any,
    batch: Any,
    config: Mapping[str, Any],
    order_seed: int,
):
    """Return IDAAC's main (non-classifier-step) training objective."""
    import jax
    import jax.numpy as jnp
    import optax

    order_swap = jax.random.bernoulli(
        jax.random.PRNGKey(order_seed), shape=batch.done.shape
    )
    not_last = (
        jnp.arange(batch.done.shape[0])[:, None]
        < batch.done.shape[0] - 1
    )
    next_is_reset = jnp.roll(batch.done, shift=-1, axis=0)
    order_mask = (not_last & ~next_is_reset).astype(jnp.float32)

    def masked_mean(values):
        return (values * order_mask).sum() / jnp.maximum(
            order_mask.sum(), 1.0
        )

    def loss(params):
        _, pi, value, advantage_predictions, order_logits = network.apply(
            params,
            batch.initial_hstate,
            (batch.obs, batch.done, batch.agent_positions),
            return_auxiliary=True,
            order_swap=order_swap,
        )
        log_prob = pi.log_prob(batch.action)
        value_pred_clipped = batch.value + (value - batch.value).clip(
            -config["CLIP_EPS"], config["CLIP_EPS"]
        )
        value_loss = 0.5 * jnp.maximum(
            jnp.square(value - batch.targets),
            jnp.square(value_pred_clipped - batch.targets),
        ).mean()

        advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
        )
        predicted_advantage = jnp.take_along_axis(
            advantage_predictions, batch.action[..., None], axis=-1
        ).squeeze(-1)
        advantage_loss = 0.5 * jnp.square(
            predicted_advantage - jax.lax.stop_gradient(advantages)
        ).mean()
        order_loss = masked_mean(
            optax.sigmoid_binary_cross_entropy(
                order_logits, jnp.full_like(order_logits, 0.5)
            )
        )

        logratio = log_prob - batch.log_prob
        ratio = jnp.exp(logratio)
        actor_loss = -jnp.minimum(
            ratio * advantages,
            jnp.clip(
                ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"],
            )
            * advantages,
        ).mean()
        entropy = pi.entropy().mean()
        return (
            actor_loss
            + config["VF_COEF"] * value_loss
            + config["DAAC_ADV_COEF"] * advantage_loss
            + config["IDAAC_ORDER_COEF"] * order_loss
            - config["ENT_COEF"] * entropy
        )

    return loss


def _critic_loss(network: Any, batch: Any, config: Mapping[str, Any]):
    """The identical PPO clipped critic objective used by both models."""
    import jax.numpy as jnp

    def loss(params):
        _, _, value = network.apply(
            params,
            batch.initial_hstate,
            (batch.obs, batch.done, batch.agent_positions),
        )
        value_pred_clipped = batch.value + (value - batch.value).clip(
            -config["CLIP_EPS"], config["CLIP_EPS"]
        )
        return 0.5 * jnp.maximum(
            jnp.square(value - batch.targets),
            jnp.square(value_pred_clipped - batch.targets),
        ).mean()

    return loss


def _critic_parameter_scope(model: str, params: Any):
    """Return critic-path variables and a function that merges them back."""
    import flax

    if model == "CEC":
        from baselines.CEC_UED.ippo_general_gradient import VALUE_TRUNK_KEYS
    elif model == "CEC_IDDAC":
        from baselines.CEC_UED.idaac_general_gradient import VALUE_TRUNK_KEYS
    else:
        raise ValueError(f"Unsupported model: {model}")

    was_frozen = isinstance(params, flax.core.FrozenDict)
    full_variables = flax.core.unfreeze(params)
    available = full_variables["params"]
    missing = set(VALUE_TRUNK_KEYS).difference(available)
    if missing:
        raise ValueError(
            f"Checkpoint is missing critic modules: {sorted(missing)}"
        )
    scoped_variables = {
        "params": {key: available[key] for key in VALUE_TRUNK_KEYS}
    }
    if was_frozen:
        scoped_variables = flax.core.freeze(scoped_variables)

    def merge(candidate_scope):
        merged = flax.core.unfreeze(params)
        candidate = flax.core.unfreeze(candidate_scope)
        merged["params"].update(candidate["params"])
        return flax.core.freeze(merged) if was_frozen else merged

    return scoped_variables, merge, tuple(VALUE_TRUNK_KEYS)


def _write_results(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(temporary, path)


def summarize_results(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate successful sharpness values across seeds."""
    grouped: dict[tuple[str, int, str, str], list[float]] = {}
    for result in results:
        if "error" in result:
            continue
        for metric, value in result["sharpness"].items():
            key = (
                str(result["model"]),
                int(result["training_num_envs"]),
                str(result.get("loss_scope", "full")),
                str(metric),
            )
            grouped.setdefault(key, []).append(float(value))

    summary = []
    for (
        model, training_num_envs, loss_scope, metric
    ), values in sorted(grouped.items()):
        epsilon = float(metric.rsplit("eps_", maxsplit=1)[-1])
        summary.append(
            {
                "model": model,
                "training_num_envs": training_num_envs,
                "loss_scope": loss_scope,
                "epsilon": epsilon,
                "metric": metric,
                "num_seeds": len(values),
                "mean": statistics.mean(values),
                # Across-seed sample standard deviation. It is undefined for
                # a single successful seed, so JSON records null in that case.
                "std": statistics.stdev(values) if len(values) > 1 else None,
                "values": values,
            }
        )
    return summary


def _summary_path(output_path: Path) -> Path:
    suffix = output_path.suffix or ".json"
    stem = output_path.stem if output_path.suffix else output_path.name
    return output_path.with_name(f"{stem}_summary{suffix}")


def _write_summary(
    output_path: Path, results: Sequence[Mapping[str, Any]]
) -> Path:
    path = _summary_path(output_path)
    _write_results(path, summarize_results(results))
    return path


def _print_summary(summary: Sequence[Mapping[str, Any]]) -> None:
    if not summary:
        print("\nNo successful results to summarize.")
        return
    print("\nAcross-seed sharpness summary (mean ± sample std)")
    print("  model       train_envs  scope    epsilon     n  mean ± std")
    for row in summary:
        std = "N/A" if row["std"] is None else f"{row['std']:.6g}"
        print(
            f"  {row['model']:<11s} {row['training_num_envs']:>10d}  "
            f"{row['loss_scope']:<7s}  "
            f"{row['epsilon']:<9g} {row['num_seeds']:>2d}  "
            f"{row['mean']:.6g} ± {std}"
        )


def measure_checkpoint(
    item: Checkpoint,
    config: dict[str, Any],
    env: Any,
    epsilons: Sequence[float],
    maxiter: int,
    eval_num_envs: int,
    rollout_steps: int,
    sampled_actors: int,
    rollout_seed: int,
    loss_scope: str,
) -> dict[str, Any]:
    import jax

    from baselines.CEC_UED.sharpness import compute_keskar_sharpness

    started = time.time()
    with item.path.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, Mapping) or "params" not in payload:
        raise ValueError(f"Checkpoint has no 'params' mapping: {item.path}")
    params = payload["params"]

    network, batch = _collect_batch(
        item, payload, params, config, env, rollout_seed
    )
    perturbation_modules: tuple[str, ...] | None = None
    if loss_scope != "critic":
        raise ValueError("Only critic sharpness is supported")
    full_critic_loss = _critic_loss(network, batch, config)
    sharpness_params, merge_scope, perturbation_modules = (
        _critic_parameter_scope(item.model, params)
    )

    def loss_fn(candidate_scope):
        return full_critic_loss(merge_scope(candidate_scope))

    base_loss = float(loss_fn(sharpness_params))
    metrics = compute_keskar_sharpness(
        loss_fn, sharpness_params, epsilons, maxiter=maxiter
    )
    parameter_count = sum(
        int(leaf.size) for leaf in jax.tree.leaves(sharpness_params)
    )
    result = {
        "model": item.model,
        "training_num_envs": item.training_num_envs,
        "seed": item.seed,
        "checkpoint": str(item.path),
        "checkpoint_updates": _checkpoint_updates(payload, item),
        "parameter_count": parameter_count,
        "total_parameter_count": sum(
            int(leaf.size) for leaf in jax.tree.leaves(params)
        ),
        "loss_scope": loss_scope,
        "perturbed_modules": perturbation_modules,
        "base_loss": base_loss,
        "eval_num_envs": eval_num_envs,
        "sampled_actors": min(sampled_actors, 2 * eval_num_envs),
        "rollout_steps": rollout_steps,
        "rollout_seed": rollout_seed,
        "epsilons": list(epsilons),
        "lbfgsb_maxiter": maxiter,
        "sharpness": metrics,
        "elapsed_seconds": time.time() - started,
    }
    jax.clear_caches()
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(DEFAULT_ROOTS),
        default=list(DEFAULT_ROOTS),
    )
    parser.add_argument("--cec-root", type=Path, default=DEFAULT_ROOTS["CEC"])
    parser.add_argument(
        "--cec-idaac-root", type=Path, default=DEFAULT_ROOTS["CEC_IDDAC"]
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--training-num-envs",
        "--training-rollout-sizes",
        "--batch-sizes",
        dest="training_num_envs",
        nargs="+",
        type=int,
        help="checkpoint directory names to include (training NUM_ENVS)",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--epsilons", nargs="+", type=float, default=[0.001, 0.0005]
    )
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--eval-num-envs", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=400)
    parser.add_argument("--sampled-actors", type=int, default=16)
    parser.add_argument("--rollout-seed", type=int, default=20260823)
    parser.add_argument(
        "--loss-scope",
        choices=("critic",),
        default="critic",
        help="critic-only (the sole supported sharpness scope)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("sharpness_cec.json")
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="only list matching checkpoints"
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="stop on the first failure"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.maxiter < 1 or args.eval_num_envs < 1 or args.rollout_steps < 1:
        raise ValueError("maxiter, eval-num-envs, and rollout-steps must be positive")
    if args.sampled_actors < 1:
        raise ValueError("sampled-actors must be positive")
    if any(epsilon <= 0 for epsilon in args.epsilons):
        raise ValueError("all epsilons must be positive")

    roots = {
        "CEC": args.cec_root,
        "CEC_IDDAC": args.cec_idaac_root,
    }
    checkpoints = discover_checkpoints(
        roots,
        args.models,
        (
            set(args.training_num_envs)
            if args.training_num_envs
            else None
        ),
        set(args.seeds) if args.seeds else None,
    )
    if not checkpoints:
        print("No matching checkpoints found.", file=sys.stderr)
        return 2

    print(f"Found {len(checkpoints)} checkpoint(s):")
    for item in checkpoints:
        print(
            f"  {item.model:10s} train_envs={item.training_num_envs:<3d} "
            f"seed={item.seed}: {item.path}"
        )
    if args.dry_run:
        return 0

    results: list[dict[str, Any]] = []
    runtime: tuple[dict[str, Any], Any] | None = None
    for index, item in enumerate(checkpoints, start=1):
        print(
            f"\n[{index}/{len(checkpoints)}] Measuring {item.model}, "
            f"train_envs={item.training_num_envs}, seed={item.seed}"
        )
        try:
            # The CEC held-out set is deterministic and expensive to compile.
            # Reuse one environment for every checkpoint in this invocation.
            if runtime is None:
                runtime_config = _prepare_config(
                    args.config.resolve(),
                    args.eval_num_envs,
                    args.rollout_steps,
                    args.sampled_actors,
                )
                runtime = (
                    runtime_config,
                    _make_environment(runtime_config),
                )
            runtime_config, runtime_env = runtime
            result = measure_checkpoint(
                item=item,
                config=runtime_config,
                env=runtime_env,
                epsilons=args.epsilons,
                maxiter=args.maxiter,
                eval_num_envs=args.eval_num_envs,
                rollout_steps=args.rollout_steps,
                sampled_actors=args.sampled_actors,
                rollout_seed=args.rollout_seed,
                loss_scope=args.loss_scope,
            )
            results.append(result)
            values = ", ".join(
                f"{key}={value:.6g}"
                for key, value in result["sharpness"].items()
            )
            print(f"  base_loss={result['base_loss']:.6g}, {values}")
        except Exception as error:  # Preserve earlier expensive results.
            failure = {
                "model": item.model,
                "training_num_envs": item.training_num_envs,
                "seed": item.seed,
                "checkpoint": str(item.path),
                "loss_scope": args.loss_scope,
                "error": f"{type(error).__name__}: {error}",
            }
            results.append(failure)
            print(f"  FAILED: {failure['error']}", file=sys.stderr)
            if args.fail_fast:
                _write_results(args.output.resolve(), results)
                _write_summary(args.output.resolve(), results)
                raise
        _write_results(args.output.resolve(), results)
        _write_summary(args.output.resolve(), results)

    summary = summarize_results(results)
    summary_path = _summary_path(args.output.resolve())
    _print_summary(summary)
    print(f"\nWrote {len(results)} result(s) to {args.output.resolve()}")
    print(f"Wrote across-seed summary to {summary_path}")
    return int(any("error" in result for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
