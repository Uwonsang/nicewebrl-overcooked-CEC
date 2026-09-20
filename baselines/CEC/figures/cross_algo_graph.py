"""Cross-algorithm XP result heatmaps — directional (role-labeled) and symmetric versions."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    }
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DEFAULT_RESULTS_DIR = Path(
    "/mnt/nas/wonsang/crossenv_ued/models/ICRL/xp_results_diff_algo"
)
DEFAULT_SAVE_DIR = (
    Path(__file__).resolve().parents[3] / "artifacts" / "cross_algo_graph"
)

ALGO_RENAME = {
    "IPPO": "IPPO",
    "E3T": "E3T",
    "FCP": "FCP",
    "CEC_envs64": "CEC",
    "CEC_IDAAC_envs32": "DCEC (8K)",
    "CEC_IDAAC_envs256": "DCEC (65K)",
}
ALGO_ORDER = [
    "IPPO", "E3T", "FCP", "CEC",
    "DCEC (8K)", "DCEC (65K)",
]

LAYOUT_ORDER = [
    "asymm_advantages_9",
    "coord_ring_9",
    "counter_circuit_9",
    "cramped_room_9",
    "forced_coord_9",
]
LAYOUT_LABEL = {
    "asymm_advantages_9": "Asymm Advantages",
    "coord_ring_9":        "Coord Ring",
    "counter_circuit_9":   "Counter Circuit",
    "cramped_room_9":      "Cramped Room",
    "forced_coord_9":      "Forced Coord",
}

CEC_IDAAC_BLUE = LinearSegmentedColormap.from_list(
    "cec_idaac_blue",
    ["#C6E8F8", "#35A9E0", "#087FBD", "#004B87"],
)


# ──────────────────────────────────────────────
# Data loading / transforms
# ──────────────────────────────────────────────
def load_pivot(csv_path: Path) -> pd.DataFrame:
    """Return (algo_1 × algo_2) mean-reward pivot, with display names, ordered."""
    df = pd.read_csv(csv_path)
    df["algo_1"] = df["algo_1"].map(ALGO_RENAME)
    df["algo_2"] = df["algo_2"].map(ALGO_RENAME)
    df = df.dropna(subset=["algo_1", "algo_2"])

    pivot = df.groupby(["algo_1", "algo_2"])["reward"].mean().unstack("algo_2")
    present = [a for a in ALGO_ORDER if a in pivot.index]
    return pivot.reindex(index=present, columns=present)


def symmetrize_pivot(pivot: pd.DataFrame) -> pd.DataFrame:
    """Average (i,j) and (j,i) to produce a symmetric matrix."""
    p = pivot.copy().astype(float)
    return (p + p.T) / 2


def normalize_pivot(pivot: pd.DataFrame) -> pd.DataFrame:
    """Divide by this pivot's own max value."""
    return pivot.copy().astype(float) / pivot.max().max()


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
def plot_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    vmin: float,
    vmax: float,
    *,
    label_fontsize: int = 20,
    axis_fontsize: int = 22,
    title_fontsize: int = 24,
    value_fontsize: int = 17,
):
    algos = list(data.index)
    display_algos = [
        algorithm.replace(" ", "\n", 1)
        if algorithm.startswith("DCEC ")
        else algorithm
        for algorithm in algos
    ]
    mat = data.values.astype(float)

    im = ax.imshow(
        mat,
        cmap=CEC_IDAAC_BLUE,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(
        display_algos, fontsize=label_fontsize, rotation=0, ha="center"
    )
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(display_algos, fontsize=label_fontsize)
    ax.set_xlabel(xlabel, fontsize=axis_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_fontsize)
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=12)

    for i in range(len(algos)):
        for j in range(len(algos)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=value_fontsize, color="black")
    return im


def save_figures(
    pivots_raw: dict, tag: str, xlabel: str, ylabel: str, suptitle: str,
    save_dir: Path,
):
    """Generate per-layout + overall figures for a given pivot set."""
    finite_values = np.concatenate([
        pivot.to_numpy(dtype=float).ravel() for pivot in pivots_raw.values()
    ])
    finite_values = finite_values[np.isfinite(finite_values)]
    color_min = min(0.0, float(finite_values.min()))
    color_max = float(finite_values.max())

    # per-layout
    # Five layouts are easier to read as three panels on the first row and
    # two centered panels on the second row than as one very wide strip.
    fig = plt.figure(figsize=(24.6, 12.4))
    grid = fig.add_gridspec(2, 6)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]

    for ax, (layout, pivot) in zip(axes, pivots_raw.items()):
        im = plot_heatmap(
            ax,
            pivot,
            LAYOUT_LABEL[layout],
            xlabel=xlabel,
            ylabel=ylabel,
            vmin=color_min,
            vmax=color_max,
            label_fontsize=14,
            axis_fontsize=17,
            title_fontsize=20,
            value_fontsize=14,
        )
    for ax in axes[len(pivots_raw):]:
        ax.set_visible(False)

    fig.subplots_adjust(
        left=0.06, right=0.92, bottom=0.10, top=0.92,
        wspace=0.90, hspace=0.65,
    )
    cbar_ax = fig.add_axes([0.94, 0.17, 0.012, 0.66])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Mean Reward", fontsize=18)
    cbar.ax.tick_params(labelsize=15)

    out = save_dir / f"cross_algo_per_layout_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # Also save one large, publication-friendly PDF per layout. Individual
    # panels use their own reward range to make within-layout differences
    # visible; the combined figure above retains common cross-layout limits.
    for layout, pivot in pivots_raw.items():
        layout_values = pivot.to_numpy(dtype=float)
        layout_values = layout_values[np.isfinite(layout_values)]
        layout_color_min = float(layout_values.min())
        layout_color_max = float(layout_values.max())
        if np.isclose(layout_color_min, layout_color_max):
            layout_color_min -= 0.5
            layout_color_max += 0.5
        single_fig, single_ax = plt.subplots(figsize=(14.0, 7.5))
        single_im = plot_heatmap(
            single_ax,
            pivot,
            LAYOUT_LABEL[layout],
            xlabel=xlabel,
            ylabel=ylabel,
            vmin=layout_color_min,
            vmax=layout_color_max,
        )
        single_cbar = single_fig.colorbar(
            single_im, ax=single_ax, pad=0.04, fraction=0.05
        )
        single_cbar.set_label("Mean Reward", fontsize=22)
        single_cbar.ax.tick_params(labelsize=18)
        single_fig.tight_layout()
        single_out = save_dir / f"cross_algo_{layout}_{tag}.pdf"
        single_fig.savefig(single_out, bbox_inches="tight")
        print(f"Saved: {single_out}")
        plt.close(single_fig)

    # Overall raw reward: mean across layouts.
    stacked = np.stack([p.values for p in pivots_raw.values()], axis=0)
    mean_mat = np.nanmean(stacked, axis=0)
    ref = next(iter(pivots_raw.values()))
    overall_pivot = pd.DataFrame(
        mean_mat, index=ref.index, columns=ref.columns
    )

    fig2, ax2 = plt.subplots(figsize=(14.0, 7.5))
    im2 = plot_heatmap(ax2, overall_pivot, "Overall",
                       xlabel=xlabel, ylabel=ylabel,
                       vmin=color_min, vmax=color_max)
    cbar2 = fig2.colorbar(im2, ax=ax2, pad=0.04, fraction=0.05)
    cbar2.set_label("Mean Reward", fontsize=22)
    cbar2.ax.tick_params(labelsize=18)
    fig2.tight_layout()

    out2 = save_dir / f"cross_algo_overall_{tag}.pdf"
    fig2.savefig(out2, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close(fig2)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    csv_files = {
        layout: args.results_dir / f"{layout}_cross_algo_results.csv"
        for layout in LAYOUT_ORDER
    }
    missing = [l for l, p in csv_files.items() if not p.exists()]
    if missing:
        print(f"Warning: missing CSVs for {missing}")

    pivots_raw = {}
    for layout, path in csv_files.items():
        if path.exists():
            pivots_raw[layout] = load_pivot(path)

    if not pivots_raw:
        print("No data found.")
        return

    # Version 1: directional — rows = Agent 0, columns = Agent 1
    save_figures(
        pivots_raw,
        tag="directional",
        xlabel="Agent 1",
        ylabel="Agent 0",
        suptitle="Cross-Algorithm XP (Directional)",
        save_dir=args.save_dir,
    )

    # Version 2: symmetric — average both role assignments
    pivots_sym = {layout: symmetrize_pivot(p) for layout, p in pivots_raw.items()}
    save_figures(
        pivots_sym,
        tag="symmetric",
        xlabel="Algorithm",
        ylabel="Algorithm",
        suptitle="Cross-Algorithm XP (Symmetric)",
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
