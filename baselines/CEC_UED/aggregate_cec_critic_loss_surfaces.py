"""Aggregate CEC critic loss surfaces by parameter case across seeds."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


UPDATE_RE = re.compile(r"^(?P<label>.+)_update(?P<step>\d+)_")
CASE_ORDER = ("encoder_rnn", "critic_mlp", "critic_full")
LABEL_ORDER = ("early", "middle", "final")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["CEC", "CEC_IDDAC"])
    parser.add_argument("--training-num-envs", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--cases", nargs="+", choices=CASE_ORDER, default=list(CASE_ORDER))
    return parser.parse_args(argv)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(temporary, path)


def _discover(args: argparse.Namespace):
    groups = defaultdict(list)
    model_filter = set(args.models)
    env_filter = set(args.training_num_envs) if args.training_num_envs else None
    seed_filter = set(args.seeds) if args.seeds else None

    for case in args.cases:
        pattern = f"*/env*/seed*/{case}/*_{case}_loss_surface.npz"
        for path in args.output_root.glob(pattern):
            model = path.parents[3].name
            env_name = path.parents[2].name
            seed_name = path.parents[1].name
            if not env_name.startswith("env") or not seed_name.startswith("seed"):
                continue
            try:
                training_num_envs = int(env_name[3:])
                seed = int(seed_name[4:])
            except ValueError:
                continue
            match = UPDATE_RE.match(path.name)
            if match is None:
                continue
            if model not in model_filter:
                continue
            if env_filter is not None and training_num_envs not in env_filter:
                continue
            if seed_filter is not None and seed not in seed_filter:
                continue
            groups[(case, model, training_num_envs, match.group("label"))].append(
                (seed, int(match.group("step")), path)
            )
    return groups


def _aggregate(groups, aggregate_root: Path):
    records = []
    surfaces = {}
    for key, sources in sorted(groups.items()):
        case, model, training_num_envs, label = key
        sources = sorted(sources)
        loaded = [np.load(path) for _, _, path in sources]
        reference_x = loaded[0]["x"]
        reference_y = loaded[0]["y"]
        for data in loaded[1:]:
            if not (
                np.array_equal(reference_x, data["x"])
                and np.array_equal(reference_y, data["y"])
            ):
                raise ValueError(f"Grid mismatch while aggregating {key}")
        losses = np.stack([data["loss"] for data in loaded], axis=0)
        mean_loss = losses.mean(axis=0)
        std_loss = (
            losses.std(axis=0, ddof=1)
            if losses.shape[0] > 1
            else np.zeros_like(mean_loss)
        )
        seeds = [seed for seed, _, _ in sources]
        update_steps = [step for _, step, _ in sources]
        case_dir = aggregate_root / case
        case_dir.mkdir(parents=True, exist_ok=True)
        output_path = case_dir / (
            f"{model}_env{training_num_envs}_{label}_seed_aggregate.npz"
        )
        np.savez_compressed(
            output_path,
            x=reference_x,
            y=reference_y,
            mean_loss=mean_loss,
            std_loss=std_loss,
            num_seeds=np.asarray(len(seeds)),
            seeds=np.asarray(seeds),
            update_steps=np.asarray(update_steps),
        )
        surfaces[key] = {
            "x": reference_x,
            "y": reference_y,
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "num_seeds": len(seeds),
            "seeds": seeds,
            "update_steps": update_steps,
        }
        records.append(
            {
                "parameter_case": case,
                "model": model,
                "training_num_envs": training_num_envs,
                "label": label,
                "num_seeds": len(seeds),
                "seeds": seeds,
                "update_steps": update_steps,
                "mean_center_loss": float(
                    mean_loss[mean_loss.shape[0] // 2, mean_loss.shape[1] // 2]
                ),
                "mean_surface_min": float(mean_loss.min()),
                "mean_surface_max": float(mean_loss.max()),
                "npz": str(output_path),
                "sources": [str(path) for _, _, path in sources],
            }
        )
    return surfaces, records


def _plot_case(
    case: str,
    case_surfaces: dict,
    output_path: Path,
    shared_scale: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    row_keys = sorted(
        {(model, envs) for _, model, envs, _ in case_surfaces},
        key=lambda value: (value[0], value[1]),
    )
    labels = [
        label
        for label in LABEL_ORDER
        if any(key[3] == label for key in case_surfaces)
    ]
    extra_labels = sorted(
        {key[3] for key in case_surfaces}.difference(labels)
    )
    labels.extend(extra_labels)
    values = np.concatenate(
        [surface["mean_loss"].ravel() for surface in case_surfaces.values()]
    )
    vmin, vmax = float(values.min()), float(values.max())
    if vmin == vmax:
        vmax = vmin + max(abs(vmin), 1.0) * 1e-8
    shared_normalization = Normalize(vmin=vmin, vmax=vmax)

    figure = plt.figure(figsize=(5.2 * len(labels), 4.8 * len(row_keys)))
    figure.subplots_adjust(
        left=0.03, right=0.88, bottom=0.08, top=0.90, wspace=0.12, hspace=0.30
    )
    axes = []
    for row, (model, envs) in enumerate(row_keys):
        for column, label in enumerate(labels):
            axis = figure.add_subplot(
                len(row_keys), len(labels), row * len(labels) + column + 1,
                projection="3d",
            )
            axes.append(axis)
            surface = case_surfaces.get((case, model, envs, label))
            if surface is None:
                axis.set_axis_off()
                continue
            panel_min = float(surface["mean_loss"].min())
            panel_max = float(surface["mean_loss"].max())
            if panel_min == panel_max:
                panel_max = panel_min + max(abs(panel_min), 1.0) * 1e-8
            normalization = (
                shared_normalization
                if shared_scale
                else Normalize(vmin=panel_min, vmax=panel_max)
            )
            axis.plot_surface(
                surface["x"], surface["y"], surface["mean_loss"],
                cmap="viridis", norm=normalization, linewidth=0, antialiased=True,
            )
            axis.set_xlabel("Weight Subspace 1")
            axis.set_ylabel("Weight Subspace 2")
            axis.set_zlabel("Mean Critic Loss")
            axis.set_zlim(
                (vmin, vmax) if shared_scale else (panel_min, panel_max)
            )
            axis.view_init(elev=28, azim=-58)
            axis.set_title(
                f"{model}, NUM_ENVS={envs}\n"
                f"{label.capitalize()} · n={surface['num_seeds']} seed(s)"
            )
    if shared_scale:
        colorbar = figure.colorbar(
            ScalarMappable(norm=shared_normalization, cmap="viridis"),
            ax=axes, shrink=0.72, pad=0.03,
        )
        colorbar.set_label("Across-seed Mean Critic Loss")
    title = {
        "encoder_rnn": "Encoder/RNN",
        "critic_mlp": "Critic MLP",
        "critic_full": "Full Critic Path",
    }[case]
    figure.suptitle(
        f"{title} Loss Surfaces Aggregated by Algorithm and NUM_ENVS\n"
        "(pointwise mean across seeds; "
        + (
            "common z-axis and color scale)"
            if shared_scale
            else "independent z-axis and color scale per panel)"
        )
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.35)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = args.output_root.resolve()
    aggregate_root = output_root / "aggregated_by_parameter_case"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    groups = _discover(args)
    if not groups:
        print("No matching loss-surface NPZ files found for aggregation.")
        return 0
    surfaces, records = _aggregate(groups, aggregate_root)
    for case in args.cases:
        case_surfaces = {
            key: value for key, value in surfaces.items() if key[0] == case
        }
        if not case_surfaces:
            continue
        output_path = aggregate_root / f"{case}_aggregate_comparison.png"
        _plot_case(case, case_surfaces, output_path, shared_scale=False)
        shared_output_path = aggregate_root / (
            f"{case}_aggregate_comparison_shared_scale.png"
        )
        _plot_case(case, case_surfaces, shared_output_path, shared_scale=True)
        print(f"Saved {case} independent-scale comparison: {output_path}")
        print(f"Saved {case} shared-scale comparison: {shared_output_path}")
    _atomic_json(
        aggregate_root / "aggregation_metadata.json",
        {
            "output_root": str(output_root),
            "models": args.models,
            "training_num_envs": args.training_num_envs,
            "seeds": args.seeds,
            "parameter_cases": args.cases,
            "aggregation": "pointwise arithmetic mean across seeds",
            "std": "sample std (ddof=1); zero when n=1",
            "default_figure_scale": "independent per panel",
            "additional_figure_scale": "shared across all panels in a case",
            "records": records,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
