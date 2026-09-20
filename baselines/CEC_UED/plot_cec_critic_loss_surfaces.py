"""Plot checkpoint-aligned critic loss surfaces for CEC (IPPO).

The snapshot files written by ``critic_loss_surface.py`` contain both model
parameters and a fixed recurrent PPO batch.  This script evaluates the clipped
critic loss around every snapshot along two checkpoint-local, filter-normalized
random directions following Li et al. (2018). Three parameter cases extend the
encoder/critic-MLP split used in Cetin et al. (2022):

* ``encoder_rnn`` perturbs the shared Conv/Dense/RNN representation modules.
* ``critic_mlp`` perturbs only the critic's post-RNN Dense head.
* ``critic_full`` perturbs the encoder/RNN and critic MLP together.

Actor-only parameters and biases stay fixed in all cases.

Outputs for every snapshot:

* ``<case>/<label>_update<step>_<case>_loss_surface.png`` (3-D surface)
* ``<case>/<label>_update<step>_<case>_loss_surface.npz``
* ``<case>/loss_surfaces_comparison.png`` (all steps for one case)
* ``critic_parameter_cases_comparison.png`` (selected cases and all steps)
* ``critic_loss_surfaces_metadata.json`` (directions and run settings)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_SNAPSHOT_DIR = Path(
    "/app/nas/models/ICRL/CEC/256/seed0/"
    "seed0_mid_ckpts/critic_loss_surface"
)
UPDATE_RE = re.compile(r"update(?P<step>\d+)\.pkl$")


def _ensure_legacy_import_path() -> None:
    module_directory = str(Path(__file__).resolve().parent)
    if module_directory not in sys.path:
        sys.path.insert(0, module_directory)


def discover_snapshots(snapshot_dir: Path) -> list[Path]:
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"Snapshot directory does not exist: {snapshot_dir}")

    def update_step(path: Path) -> int:
        match = UPDATE_RE.search(path.name)
        return int(match.group("step")) if match else sys.maxsize

    snapshots = sorted(snapshot_dir.glob("*_update*.pkl"), key=update_step)
    if not snapshots:
        raise FileNotFoundError(f"No *_update*.pkl snapshots found in {snapshot_dir}")
    return snapshots


def load_snapshot(path: Path) -> dict[str, Any]:
    _ensure_legacy_import_path()
    with path.open("rb") as file:
        payload = pickle.load(file)
    required = {"params", "batch", "metadata", "label", "update_step"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"{path} is missing snapshot keys: {sorted(missing)}")
    algorithm = payload["metadata"].get("algorithm")
    if algorithm not in {"IPPO", "IDAAC"}:
        raise ValueError(
            f"Expected a CEC/IPPO or CEC_IDDAC/IDAAC snapshot, "
            f"got algorithm={algorithm!r}: {path}"
        )
    return payload


def _network_from_metadata(metadata: Mapping[str, Any]):
    """Rebuild only the shared representation and critic used by the loss.

    Importing the training module also imports environments, VAE utilities,
    Hydra, and logging packages that are irrelevant to post-hoc evaluation.
    Explicit module names below match the IPPO checkpoint names.
    """
    import functools

    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax.linen.initializers import constant, orthogonal

    config = dict(metadata["network_config"])
    config.update(metadata["loss_config"])

    class SurfaceScannedRNN(nn.Module):
        @functools.partial(
            nn.scan,
            variable_broadcast="params",
            in_axes=0,
            out_axes=0,
            split_rngs={"params": False},
        )
        @nn.compact
        def __call__(self, carry, inputs):
            features, resets = inputs
            carry = jax.tree.map(
                lambda value: jnp.where(
                    resets[:, np.newaxis], jnp.zeros_like(value), value
                ),
                carry,
            )
            return nn.OptimizedLSTMCell(
                features=features.shape[-1], name="OptimizedLSTMCell_0"
            )(carry, features)

    class SurfaceIPPOCriticRNN(nn.Module):
        config: Mapping[str, Any]

        @nn.compact
        def __call__(self, hidden, inputs):
            obs, dones, agent_positions = inputs
            del agent_positions
            batch_size, num_envs, _ = obs.shape

            if self.config["CONV_NET"]:
                if self.config["ENV_NAME"] == "overcooked":
                    embedding = obs.reshape(-1, 9, 9, 26)
                else:
                    embedding = obs.reshape(-1, 5, 5, 4)
                embedding = nn.relu(
                    nn.Conv(
                        features=64,
                        kernel_size=(2, 2),
                        kernel_init=orthogonal(np.sqrt(2)),
                        bias_init=constant(0.0),
                        name="Conv_0",
                    )(embedding)
                )
                embedding = nn.relu(
                    nn.Conv(
                        features=32,
                        kernel_size=(2, 2),
                        kernel_init=orthogonal(np.sqrt(2)),
                        bias_init=constant(0.0),
                        name="Conv_1",
                    )(embedding)
                )
                embedding = embedding.reshape((batch_size, num_envs, -1))
            else:
                embedding = obs

            embedding = nn.relu(
                nn.Dense(
                    self.config["FC_DIM_SIZE"] * 2,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    name="Dense_0",
                )(embedding)
            )
            embedding = nn.relu(
                nn.Dense(
                    self.config["FC_DIM_SIZE"] * 2,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    name="Dense_1",
                )(embedding)
            )
            if self.config["LSTM"]:
                hidden, embedding = SurfaceScannedRNN(
                    name="ScannedRNN_0"
                )(hidden, (embedding, dones))
            else:
                embedding = nn.relu(
                    nn.Dense(
                        self.config["GRU_HIDDEN_DIM"],
                        kernel_init=orthogonal(2),
                        bias_init=constant(0.0),
                        name="Dense_2",
                    )(embedding)
                )
            embedding = embedding.reshape((batch_size, num_envs, -1))

            critic = nn.relu(
                nn.Dense(
                    self.config["FC_DIM_SIZE"] * 2,
                    kernel_init=orthogonal(2),
                    bias_init=constant(0.0),
                    name="Dense_7",
                )(embedding)
            )
            critic = nn.relu(
                nn.Dense(
                    self.config["FC_DIM_SIZE"],
                    kernel_init=orthogonal(2),
                    bias_init=constant(0.0),
                    name="Dense_8",
                )(critic)
            )
            if self.config["ENV_NAME"] == "overcooked":
                critic = nn.relu(
                    nn.Dense(
                        self.config["FC_DIM_SIZE"] * 3 // 4,
                        kernel_init=orthogonal(2),
                        bias_init=constant(0.0),
                        name="Dense_9",
                    )(critic)
                )
                critic = nn.relu(
                    nn.Dense(
                        self.config["FC_DIM_SIZE"] // 2,
                        kernel_init=orthogonal(2),
                        bias_init=constant(0.0),
                        name="Dense_10",
                    )(critic)
                )
            critic = nn.Dense(
                1,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="Dense_11",
            )(critic)
            return hidden, None, jnp.squeeze(critic, axis=-1)

    if metadata["algorithm"] == "IPPO":
        return SurfaceIPPOCriticRNN(config=config), config

    class SurfaceIDAACCriticTrunk(nn.Module):
        config: Mapping[str, Any]

        @nn.compact
        def __call__(self, hidden, obs, dones):
            time_size, actor_size, _ = obs.shape
            if self.config["CONV_NET"]:
                if self.config["ENV_NAME"] == "overcooked":
                    embedding = obs.reshape(-1, 9, 9, 26)
                else:
                    embedding = obs.reshape(-1, 5, 5, 4)
                embedding = nn.relu(
                    nn.Conv(
                        features=64,
                        kernel_size=(2, 2),
                        kernel_init=orthogonal(np.sqrt(2)),
                        bias_init=constant(0.0),
                        name="conv_0",
                    )(embedding)
                )
                embedding = nn.relu(
                    nn.Conv(
                        features=32,
                        kernel_size=(2, 2),
                        kernel_init=orthogonal(np.sqrt(2)),
                        bias_init=constant(0.0),
                        name="conv_1",
                    )(embedding)
                )
                embedding = embedding.reshape((time_size, actor_size, -1))
            else:
                embedding = obs

            embedding = nn.relu(
                nn.Dense(
                    self.config["FC_DIM_SIZE"] * 2,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    name="dense_0",
                )(embedding)
            )
            embedding = nn.relu(
                nn.Dense(
                    self.config["FC_DIM_SIZE"] * 2,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    name="dense_1",
                )(embedding)
            )
            if self.config["LSTM"]:
                hidden, embedding = SurfaceScannedRNN(name="recurrent")(
                    hidden, (embedding, dones)
                )
            else:
                embedding = nn.relu(
                    nn.Dense(
                        self.config["GRU_HIDDEN_DIM"],
                        kernel_init=orthogonal(2),
                        bias_init=constant(0.0),
                        name="recurrent_dense",
                    )(embedding)
                )
            return hidden, embedding.reshape((time_size, actor_size, -1))

    class SurfaceIDAACCriticRNN(nn.Module):
        config: Mapping[str, Any]

        @nn.compact
        def __call__(self, hidden, inputs):
            obs, dones, agent_positions = inputs
            del agent_positions
            actor_hidden, critic_hidden = hidden
            critic_hidden, critic = SurfaceIDAACCriticTrunk(
                config=self.config, name="critic_trunk"
            )(critic_hidden, obs, dones)
            critic = nn.relu(
                nn.Dense(
                    self.config["FC_DIM_SIZE"] * 2,
                    kernel_init=orthogonal(2),
                    bias_init=constant(0.0),
                    name="critic_hidden_0",
                )(critic)
            )
            critic = nn.relu(
                nn.Dense(
                    self.config["FC_DIM_SIZE"],
                    kernel_init=orthogonal(2),
                    bias_init=constant(0.0),
                    name="critic_hidden_1",
                )(critic)
            )
            if self.config["ENV_NAME"] == "overcooked":
                critic = nn.relu(
                    nn.Dense(
                        self.config["FC_DIM_SIZE"] * 3 // 4,
                        kernel_init=orthogonal(2),
                        bias_init=constant(0.0),
                        name="critic_hidden_2",
                    )(critic)
                )
                critic = nn.relu(
                    nn.Dense(
                        self.config["FC_DIM_SIZE"] // 2,
                        kernel_init=orthogonal(2),
                        bias_init=constant(0.0),
                        name="critic_hidden_3",
                    )(critic)
                )
            critic = nn.Dense(
                1,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="critic_output",
            )(critic)
            return (
                (actor_hidden, critic_hidden),
                None,
                jnp.squeeze(critic, axis=-1),
            )

    return SurfaceIDAACCriticRNN(config=config), config


def critic_loss_fn(network: Any, params: Any, batch: Any, clip_eps: float):
    """PPO clipped value loss evaluated on the snapshot's fixed batch."""
    import jax.numpy as jnp

    del params  # Documents that params enter the returned closure explicitly.

    def loss(candidate_params):
        _, _, value = network.apply(
            candidate_params,
            batch.initial_hstate,
            (batch.obs, batch.done, batch.agent_positions),
        )
        value_pred_clipped = batch.value + (value - batch.value).clip(
            -clip_eps, clip_eps
        )
        return 0.5 * jnp.maximum(
            jnp.square(value - batch.targets),
            jnp.square(value_pred_clipped - batch.targets),
        ).mean()

    return loss


def make_filter_normalized_directions(
    params: Any,
    value_modules: Sequence[str],
    seed: int,
) -> tuple[Any, Any, dict[str, float]]:
    """Create two filter-normalized directions on critic-relevant kernels."""
    import flax
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax.traverse_util import flatten_dict, unflatten_dict

    was_frozen = isinstance(params, flax.core.FrozenDict)
    flat_params = flatten_dict(flax.core.unfreeze(params))
    value_modules = set(value_modules)
    keys = jax.random.split(jax.random.PRNGKey(seed), 2 * len(flat_params))
    directions = [{}, {}]

    for index, (path, parameter) in enumerate(flat_params.items()):
        module = str(path[1]) if len(path) > 2 and path[0] == "params" else ""
        is_kernel = str(path[-1]) == "kernel" and parameter.ndim >= 2
        selected = module in value_modules and is_kernel
        for direction_index in range(2):
            if not selected:
                direction = jnp.zeros_like(parameter)
            else:
                random_direction = jax.random.normal(
                    keys[2 * index + direction_index],
                    parameter.shape,
                    dtype=parameter.dtype,
                )
                # Flax Dense/Conv kernels store output filters on the last
                # axis. Match each direction filter's norm to its parameter.
                filter_axes = tuple(range(parameter.ndim - 1))
                parameter_norm = jnp.sqrt(
                    jnp.sum(jnp.square(parameter), axis=filter_axes, keepdims=True)
                )
                direction_norm = jnp.sqrt(
                    jnp.sum(
                        jnp.square(random_direction),
                        axis=filter_axes,
                        keepdims=True,
                    )
                )
                direction = random_direction * parameter_norm / jnp.maximum(
                    direction_norm, jnp.finfo(parameter.dtype).eps
                )
            directions[direction_index][path] = direction

    reconstructed = [unflatten_dict(direction) for direction in directions]
    if was_frozen:
        reconstructed = [flax.core.freeze(value) for value in reconstructed]

    flat_x = np.concatenate(
        [np.asarray(value).reshape(-1) for value in directions[0].values()]
    )
    flat_y = np.concatenate(
        [np.asarray(value).reshape(-1) for value in directions[1].values()]
    )
    x_norm = float(np.linalg.norm(flat_x))
    y_norm = float(np.linalg.norm(flat_y))
    cosine = float(np.dot(flat_x, flat_y) / max(x_norm * y_norm, 1e-30))
    diagnostics = {
        "x_direction_norm": x_norm,
        "y_direction_norm": y_norm,
        "direction_cosine_similarity": cosine,
    }
    return reconstructed[0], reconstructed[1], diagnostics


def evaluate_surface(
    loss_fn: Any,
    params: Any,
    x_direction: Any,
    y_direction: Any,
    coordinates: Any,
):
    """Evaluate every coordinate sequentially on device via ``lax.map``."""
    import jax

    def loss_at_coordinate(coordinate):
        x, y = coordinate
        candidate = jax.tree.map(
            lambda center, dx, dy: center + x * dx + y * dy,
            params,
            x_direction,
            y_direction,
        )
        return loss_fn(candidate)

    return jax.jit(lambda points: jax.lax.map(loss_at_coordinate, points))(
        coordinates
    )


def _plot_individual(
    output_path: Path,
    x_values: Any,
    y_values: Any,
    z_values: Any,
    label: str,
    update_step: int,
    center_loss: float,
    case_title: str,
    model_title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    surface_axis = figure.add_subplot(1, 1, 1, projection="3d")
    surface = surface_axis.plot_surface(
        x_values,
        y_values,
        z_values,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
    )
    surface_axis.set_xlabel("x coefficient")
    surface_axis.set_ylabel("y coefficient")
    surface_axis.set_zlabel("Critic Loss")
    surface_axis.set_title("3-D surface")
    figure.colorbar(surface, ax=surface_axis, shrink=0.72, pad=0.08,
                    label="Critic Loss")
    figure.suptitle(
        f"{model_title} {case_title} loss surface — "
        f"{label}, update {update_step:,}\n"
        f"center critic loss={center_loss:.6g}"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_comparison(
    output_path: Path,
    surfaces: Sequence[Mapping[str, Any]],
    case_title: str,
    model_title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    values = np.concatenate([surface["plot_grid"].ravel() for surface in surfaces])
    vmin, vmax = float(values.min()), float(values.max())
    if vmin == vmax:
        vmax = vmin + max(abs(vmin), 1.0) * 1e-8
    normalization = Normalize(vmin=vmin, vmax=vmax)
    figure = plt.figure(
        figsize=(5.3 * len(surfaces), 5.0), constrained_layout=True
    )
    axes = [
        figure.add_subplot(1, len(surfaces), index + 1, projection="3d")
        for index in range(len(surfaces))
    ]
    for axis, surface in zip(axes, surfaces):
        axis.plot_surface(
            surface["x_grid"],
            surface["y_grid"],
            surface["plot_grid"],
            cmap="viridis",
            norm=normalization,
            linewidth=0,
            antialiased=True,
        )
        axis.scatter(
            [0],
            [0],
            [surface["plot_grid"][surface["plot_grid"].shape[0] // 2,
                                  surface["plot_grid"].shape[1] // 2]],
            marker="x",
            color="red",
            s=42,
            linewidth=2,
        )
        axis.set_xlabel("Weight Subspace 1")
        axis.set_ylabel("Weight Subspace 2")
        axis.set_zlabel("Critic Loss")
        axis.set_zlim(vmin, vmax)
        axis.view_init(elev=28, azim=-58)
        axis.set_title(
            f"{surface['label'].capitalize()}\n"
            f"{surface['update_step']:,} Training Updates"
        )
    colorbar = figure.colorbar(
        ScalarMappable(norm=normalization, cmap="viridis"),
        ax=axes,
        shrink=0.72,
        pad=0.04,
    )
    colorbar.set_label("Critic Loss")
    figure.suptitle(
        f"{model_title} {case_title} Loss Surface During Training"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_case_comparison(
    output_path: Path,
    surfaces_by_case: Mapping[str, Sequence[Mapping[str, Any]]],
    model_title: str,
) -> None:
    """Plot both parameter cases with a common loss scale for direct comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    case_order = tuple(surfaces_by_case)
    case_titles = {
        "encoder_rnn": (
            "Shared Encoder/RNN"
            if model_title == "CEC"
            else "Critic Encoder/RNN"
        ),
        "critic_mlp": "Critic MLP",
        "critic_full": "Full Critic Path",
    }
    all_surfaces = [
        surface
        for case in case_order
        for surface in surfaces_by_case[case]
    ]
    values = np.concatenate(
        [surface["plot_grid"].ravel() for surface in all_surfaces]
    )
    vmin, vmax = float(values.min()), float(values.max())
    if vmin == vmax:
        vmax = vmin + max(abs(vmin), 1.0) * 1e-8
    normalization = Normalize(vmin=vmin, vmax=vmax)
    num_steps = len(surfaces_by_case[case_order[0]])
    figure = plt.figure(figsize=(5.2 * num_steps, 5.0 * len(case_order)))
    figure.subplots_adjust(
        left=0.025,
        right=0.88,
        bottom=0.10,
        top=0.88,
        wspace=0.10,
        hspace=0.32,
    )
    axes = []
    for row, case in enumerate(case_order):
        for column, surface in enumerate(surfaces_by_case[case]):
            axis = figure.add_subplot(
                len(case_order), num_steps, row * num_steps + column + 1,
                projection="3d",
            )
            axes.append(axis)
            axis.plot_surface(
                surface["x_grid"],
                surface["y_grid"],
                surface["plot_grid"],
                cmap="viridis",
                norm=normalization,
                linewidth=0,
                antialiased=True,
            )
            axis.set_xlabel("Weight Subspace 1")
            axis.set_ylabel("Weight Subspace 2")
            axis.set_zlabel("Critic Loss")
            axis.set_zlim(vmin, vmax)
            axis.view_init(elev=28, azim=-58)
            axis.set_title(
                f"{case_titles[case]}\n"
                f"{surface['update_step']:,} Training Updates"
            )
    colorbar = figure.colorbar(
        ScalarMappable(norm=normalization, cmap="viridis"),
        ax=axes,
        shrink=0.72,
        pad=0.03,
    )
    colorbar.set_label("Critic Loss")
    figure.suptitle(
        f"{model_title} Critic Parameter-Case Loss Surfaces\n"
        "(common z-axis and color scale)"
    )
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close(figure)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(temporary, path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--direction-seed", type=int, default=0)
    parser.add_argument(
        "--parameter-case",
        choices=("all", "both", "encoder_rnn", "critic_mlp", "critic_full"),
        default="all",
        help=(
            "parameter subset to perturb; all evaluates encoder_rnn, "
            "critic_mlp, and critic_full (both is a backward-compatible alias)"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list snapshots without importing JAX"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    snapshot_dir = args.snapshot_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else snapshot_dir / "plots"
    )
    snapshots = discover_snapshots(snapshot_dir)
    print(f"Found {len(snapshots)} critic snapshot(s):")
    for path in snapshots:
        print(f"  {path.name}")
    print(f"Output directory: {output_dir}")
    if args.dry_run:
        return 0
    if args.grid_size < 3 or args.grid_size % 2 == 0:
        raise ValueError("grid-size must be an odd integer >= 3 so (0, 0) is sampled")
    if args.radius <= 0:
        raise ValueError("radius must be positive")

    import jax
    import jax.numpy as jnp
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [load_snapshot(path) for path in snapshots]
    reference = payloads[-1]
    algorithm = str(reference["metadata"]["algorithm"])
    model_title = "CEC" if algorithm == "IPPO" else "CEC_IDDAC"
    if any(payload["metadata"]["algorithm"] != algorithm for payload in payloads):
        raise ValueError("All snapshots in one directory must use one algorithm")
    parameter_groups = reference["metadata"]["parameter_groups"]
    value_modules = tuple(parameter_groups["value_trunk"])
    shared_modules = tuple(parameter_groups["shared_trunk"])
    encoder_modules = (
        shared_modules
        if shared_modules
        else tuple(module for module in value_modules if module == "critic_trunk")
    )
    encoder_module_set = set(encoder_modules)
    critic_mlp_modules = tuple(
        module for module in value_modules if module not in encoder_module_set
    )
    available_cases = {
        "encoder_rnn": {
            "title": (
                "Shared Encoder/RNN"
                if algorithm == "IPPO"
                else "Critic Encoder/RNN"
            ),
            "modules": encoder_modules,
        },
        "critic_mlp": {
            "title": "Critic MLP",
            "modules": critic_mlp_modules,
        },
        "critic_full": {
            "title": "Full Critic Path",
            "modules": value_modules,
        },
    }
    if args.parameter_case in {"all", "both"}:
        selected_case_names = tuple(available_cases)
    else:
        selected_case_names = (args.parameter_case,)
    for case_name in selected_case_names:
        if not available_cases[case_name]["modules"]:
            raise ValueError(
                f"No modules found for parameter case {case_name!r}. "
                "The snapshots may use an older metadata format."
            )

    axis = np.linspace(-args.radius, args.radius, args.grid_size, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="xy")
    coordinates = jnp.asarray(
        np.stack((x_grid.ravel(), y_grid.ravel()), axis=-1)
    )
    surfaces_by_case = {}
    case_metadata = {}
    total_evaluations = len(selected_case_names) * len(payloads)
    evaluation_index = 0
    for case_name in selected_case_names:
        case = available_cases[case_name]
        case_output_dir = output_dir / case_name
        case_output_dir.mkdir(parents=True, exist_ok=True)
        surfaces = []
        direction_diagnostics = []
        for path, payload in zip(snapshots, payloads):
            evaluation_index += 1
            label = str(payload["label"])
            update_step = int(payload["update_step"])
            print(
                f"[{evaluation_index}/{total_evaluations}] Evaluating "
                f"{case_name}: {label}, update {update_step} on "
                f"{args.grid_size}x{args.grid_size} grid..."
            )
            # Li et al. filter normalization is relative to the checkpoint
            # being visualized. The random seed is stable, but directions are
            # rescaled independently using this checkpoint's filter norms.
            x_direction, y_direction, diagnostics = (
                make_filter_normalized_directions(
                    payload["params"], case["modules"], args.direction_seed
                )
            )
            network, config = _network_from_metadata(payload["metadata"])
            loss_fn = critic_loss_fn(
                network,
                payload["params"],
                payload["batch"],
                float(config["CLIP_EPS"]),
            )
            losses = np.asarray(
                evaluate_surface(
                    loss_fn,
                    payload["params"],
                    x_direction,
                    y_direction,
                    coordinates,
                )
            ).reshape(x_grid.shape)
            center_index = args.grid_size // 2
            center_loss = float(losses[center_index, center_index])
            base_name = f"{label}_update{update_step}_{case_name}_loss_surface"
            np.savez_compressed(
                case_output_dir / f"{base_name}.npz",
                x=x_grid,
                y=y_grid,
                loss=losses,
                center_loss=np.asarray(center_loss),
                update_step=np.asarray(update_step),
                parameter_case=np.asarray(case_name),
                x_direction_norm=np.asarray(diagnostics["x_direction_norm"]),
                y_direction_norm=np.asarray(diagnostics["y_direction_norm"]),
                direction_cosine_similarity=np.asarray(
                    diagnostics["direction_cosine_similarity"]
                ),
            )
            _plot_individual(
                case_output_dir / f"{base_name}.png",
                x_grid,
                y_grid,
                losses,
                label,
                update_step,
                center_loss,
                case["title"],
                model_title,
            )
            surfaces.append(
                {
                    "label": label,
                    "update_step": update_step,
                    "center_loss": center_loss,
                    "x_grid": x_grid,
                    "y_grid": y_grid,
                    "plot_grid": losses,
                }
            )
            direction_diagnostics.append(
                {
                    "label": label,
                    "update_step": update_step,
                    **diagnostics,
                }
            )
            print(f"  center critic loss: {center_loss:.6g}")
            jax.clear_caches()

        _plot_comparison(
            case_output_dir / "loss_surfaces_comparison.png",
            surfaces,
            case["title"],
            model_title,
        )
        surfaces_by_case[case_name] = surfaces
        case_metadata[case_name] = {
            "title": case["title"],
            "perturbed_modules": case["modules"],
            "directions_by_snapshot": direction_diagnostics,
        }

    if len(selected_case_names) > 1:
        _plot_case_comparison(
            output_dir / "critic_parameter_cases_comparison.png",
            surfaces_by_case,
            model_title,
        )
    metadata = {
        "snapshot_dir": str(snapshot_dir),
        "model": model_title,
        "algorithm": algorithm,
        "snapshots": [
            {
                "path": str(path),
                "label": str(payload["label"]),
                "update_step": int(payload["update_step"]),
                "center_loss_by_case": {
                    case_name: surfaces_by_case[case_name][index]["center_loss"]
                    for case_name in selected_case_names
                },
            }
            for index, (path, payload) in enumerate(zip(snapshots, payloads))
        ],
        "grid_size": args.grid_size,
        "radius": args.radius,
        "direction_seed": args.direction_seed,
        "parameter_case": args.parameter_case,
        "cases": case_metadata,
        "biases_perturbed": False,
        "direction_normalization": "per-output-filter parameter norm",
        "direction_reference": "each snapshot's own parameter filters",
        "directions_shared_across_snapshots": False,
    }
    _atomic_json(output_dir / "critic_loss_surfaces_metadata.json", metadata)
    print(f"Saved plots and grids to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
