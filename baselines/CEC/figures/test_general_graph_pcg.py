"""Visualize cross-play scores on procedurally generated Overcooked layouts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "figure.titlesize": 22,
        "legend.fontsize": 18,
    }
)


DEFAULT_RESULTS_DIR = Path(
    "/app/nas/models/ICRL/pcg_xp_results"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "test_general_graph_pcg_ICRL"
)

ALGORITHM_ORDER = [
    "IPPO",
    "E3T",
    "FCP",
    "CEC_envs64",
    "CEC_IDAAC_envs32",
    "CEC_IDAAC_envs256",
]

# graph key: (relative directory, filename prefix, layout-generalist model)
ALGORITHM_SOURCES = {
    "IPPO": ("IPPO", "IPPO", False),
    "E3T": ("E3T", "E3T", False),
    "FCP": ("FCP", "FCP", False),
    "CEC_envs64": ("CEC/envs64", "CEC_envs64", True),
    "CEC_IDAAC_envs32": (
        "CEC_IDAAC/envs32",
        "CEC_IDAAC_envs32",
        True,
    ),
    "CEC_IDAAC_envs256": (
        "CEC_IDAAC/envs256",
        "CEC_IDAAC_envs256",
        True,
    ),
}

ALGORITHM_LABELS = {
    "IPPO": "IPPO",
    "E3T": "E3T",
    "FCP": "FCP",
    "CEC_envs64": "CEC",
    "CEC_IDAAC_envs32": "DCEC (8K)",
    "CEC_IDAAC_envs256": "DCEC (65K)",
}

ALGORITHM_COLORS = {
    "IPPO": "#d62728",
    "E3T": "#7b126b",
    "FCP": "#e3a21a",
    "CEC_envs64": "#117733",
    "CEC_IDAAC_envs32": "#56B4E9",
    "CEC_IDAAC_envs256": "#0072B2",
}

GENERATED_LAYOUT_ORDER = [
    "CEC_envs64",
    "CEC_IDAAC_envs32",
    "CEC_IDAAC_envs64",
    "CEC_IDAAC_envs128",
    "CEC_IDAAC_envs256",
]

GENERATED_LAYOUT_LABELS = {
    "CEC_envs64": "CEC",
    "CEC_IDAAC_envs32": "DCEC (8K)",
    "CEC_IDAAC_envs64": "DCEC (64)",
    "CEC_IDAAC_envs128": "DCEC (128)",
    "CEC_IDAAC_envs256": "DCEC (65K)",
}

CHECKPOINT_LAYOUT_ORDER = [
    "asymm_advantages",
    "coord_ring",
    "counter_circuit",
    "cramped_room",
    "forced_coord",
]

CHECKPOINT_LAYOUT_LABELS = {
    "asymm_advantages": "asymm advantages",
    "coord_ring": "coord ring",
    "counter_circuit": "counter circuit",
    "cramped_room": "cramped room",
    "forced_coord": "forced coord",
}


def _sem(values: pd.Series) -> float:
    return (
        float(values.std(ddof=1) / np.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )


def _checkpoint_layout_from_filename(
    filename: str,
    prefix: str,
    is_generalist: bool,
) -> str:
    if is_generalist:
        expected = f"{prefix}_PCG_XP_results.csv"
        if filename != expected:
            raise ValueError(f"Unexpected generalist filename: {filename}")
        return "general"

    match = re.fullmatch(
        rf"{re.escape(prefix)}_(?P<layout>.+)_9_PCG_XP_results\.csv",
        filename,
    )
    if match is None:
        raise ValueError(f"Unexpected specialist filename: {filename}")
    return match.group("layout")


def load_pcg_results(results_dir: Path, xp_only: bool) -> pd.DataFrame:
    parts = []
    for algorithm in ALGORITHM_ORDER:
        relative_dir, prefix, is_generalist = ALGORITHM_SOURCES[algorithm]
        source_dir = results_dir / relative_dir
        paths = sorted(source_dir.glob("*_PCG_XP_results.csv"))
        if not paths:
            print(f"Warning: no PCG CSV found in {source_dir}")
        for path in paths:
            try:
                checkpoint_layout = _checkpoint_layout_from_filename(
                    path.name, prefix, is_generalist
                )
            except ValueError:
                continue
            df = pd.read_csv(path)
            required = {
                "seed_1",
                "seed_2",
                "reward",
                "held_out_layout_idx",
            }
            if not required.issubset(df.columns):
                print(f"Warning: skipping malformed CSV: {path}")
                continue
            if xp_only:
                df = df[df["seed_1"] != df["seed_2"]]
            if df.empty:
                print(f"Warning: skipping empty PCG data: {path}")
                continue
            df = df.copy()
            df["algorithm"] = algorithm
            df["checkpoint_layout"] = checkpoint_layout
            df["source_file"] = str(path)
            parts.append(df)

    if not parts:
        raise RuntimeError(f"No matching PCG XP CSVs found under {results_dir}")
    return pd.concat(parts, ignore_index=True)


def summarize_pair_units(data: pd.DataFrame) -> pd.DataFrame:
    """Average trajectories/layouts before treating a seed pair as one unit."""
    return (
        data.groupby(
            ["algorithm", "checkpoint_layout", "seed_1", "seed_2"],
            as_index=False,
        )["reward"]
        .mean()
        .rename(columns={"reward": "pair_mean_reward"})
    )


def make_overall_summary(pair_units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm in ALGORITHM_ORDER:
        values = pair_units.loc[
            pair_units["algorithm"] == algorithm, "pair_mean_reward"
        ]
        if values.empty:
            continue
        rows.append(
            {
                "algorithm": algorithm,
                "mean_reward": float(values.mean()),
                "sem_pair_units": _sem(values),
                "n_pair_units": len(values),
            }
        )
    return pd.DataFrame(rows)


def make_checkpoint_summary(pair_units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for checkpoint_layout in CHECKPOINT_LAYOUT_ORDER:
        for algorithm in ALGORITHM_ORDER:
            model_layout = (
                "general"
                if ALGORITHM_SOURCES[algorithm][2]
                else checkpoint_layout
            )
            values = pair_units.loc[
                (pair_units["algorithm"] == algorithm)
                & (pair_units["checkpoint_layout"] == model_layout),
                "pair_mean_reward",
            ]
            if values.empty:
                continue
            rows.append(
                {
                    "checkpoint_layout": checkpoint_layout,
                    "algorithm": algorithm,
                    "mean_reward": float(values.mean()),
                    "sem_pair_units": _sem(values),
                    "n_pair_units": len(values),
                }
            )
    return pd.DataFrame(rows)


def plot_overall(summary: pd.DataFrame, output_path: Path) -> None:
    indexed = summary.set_index("algorithm").reindex(ALGORITHM_ORDER)
    fig, ax = plt.subplots(figsize=(14.0, 7.5))
    x = np.arange(len(ALGORITHM_ORDER))
    ax.bar(
        x,
        indexed["mean_reward"],
        yerr=indexed["sem_pair_units"],
        capsize=5,
        color=[ALGORITHM_COLORS[a] for a in ALGORITHM_ORDER],
        edgecolor="black",
        linewidth=0.7,
        alpha=0.92,
    )
    ax.set_xticks(x)
    overall_labels = [
        ALGORITHM_LABELS[algorithm].replace(" ", "\n", 1)
        if ALGORITHM_LABELS[algorithm].startswith("DCEC ")
        else ALGORITHM_LABELS[algorithm]
        for algorithm in ALGORITHM_ORDER
    ]
    ax.set_xticklabels(overall_labels, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=24, pad=12)
    ax.tick_params(axis="y", labelsize=24)
    ax.set_ylabel("mean reward", fontsize=26)
    ax.set_title(
        "(b) Procedurally generated tasks",
        fontsize=28,
        fontweight="bold",
        pad=16,
    )
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_per_checkpoint_layout(
    summary: pd.DataFrame, output_path: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    axes_flat = axes.ravel()
    x = np.arange(len(ALGORITHM_ORDER))
    for index, checkpoint_layout in enumerate(CHECKPOINT_LAYOUT_ORDER):
        ax = axes_flat[index]
        sub = summary[summary["checkpoint_layout"] == checkpoint_layout]
        indexed = sub.set_index("algorithm").reindex(ALGORITHM_ORDER)
        ax.bar(
            x,
            indexed["mean_reward"],
            yerr=indexed["sem_pair_units"],
            capsize=4,
            color=[ALGORITHM_COLORS[a] for a in ALGORITHM_ORDER],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [ALGORITHM_LABELS[a] for a in ALGORITHM_ORDER],
            rotation=30,
            ha="right",
        )
        ax.set_title(CHECKPOINT_LAYOUT_LABELS[checkpoint_layout])
        ax.set_ylabel("mean reward")
        ax.grid(axis="y", alpha=0.35)
        ax.set_axisbelow(True)
    axes_flat[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_generated_layout_heatmap(
    summary: pd.DataFrame, output_path: Path
) -> None:
    """Plot cached mean rewards for each generated held-out layout."""
    heatmap = summary.pivot(
        index="algorithm",
        columns="held_out_layout_idx",
        values="mean_reward",
    ).reindex(GENERATED_LAYOUT_ORDER)
    fig, ax = plt.subplots(figsize=(14.5, 4.7))
    image = ax.imshow(heatmap.to_numpy(dtype=float), aspect="auto")
    ax.set_yticks(np.arange(len(GENERATED_LAYOUT_ORDER)))
    ax.set_yticklabels(
        [GENERATED_LAYOUT_LABELS[key] for key in GENERATED_LAYOUT_ORDER]
    )
    ax.set_xlabel("held-out PCG layout index")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("mean reward")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize test_general_pcg.py cross-play CSV results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"PCG XP result root (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"graph output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--include-sp",
        action="store_true",
        help="include same-seed self-play rows; default uses XP rows only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_pcg_results(args.results_dir, xp_only=not args.include_sp)
    pair_units = summarize_pair_units(data)
    overall = make_overall_summary(pair_units)
    by_checkpoint = make_checkpoint_summary(pair_units)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = (
        "test_general_pcg_xp_with_sp"
        if args.include_sp
        else "test_general_pcg_xp"
    )
    output_paths = [
        args.output_dir / f"{prefix}_overall_table.csv",
        args.output_dir / f"{prefix}_per_checkpoint_layout_table.csv",
        args.output_dir / f"{prefix}_overall.pdf",
        args.output_dir / f"{prefix}_per_checkpoint_layout.pdf",
    ]
    overall.to_csv(output_paths[0], index=False)
    by_checkpoint.to_csv(output_paths[1], index=False)
    plot_overall(overall, output_paths[2])
    plot_per_checkpoint_layout(by_checkpoint, output_paths[3])

    print(f"Loaded {len(data):,} XP rows from {args.results_dir}")
    for path in output_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
