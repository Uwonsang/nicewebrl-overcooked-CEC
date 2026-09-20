"""Checkpoint-aligned PPO batches for post-hoc critic loss surfaces."""

from __future__ import annotations

import math
import os
import pickle
from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


DEFAULT_SNAPSHOT_FRACTIONS = {
    "early": 0.1,
    "middle": 0.5,
    "final": 1.0,
}


class CriticLossSurfaceBatch(NamedTuple):
    """The fixed data needed to reproduce PPO's critic loss."""

    initial_hstate: Any
    obs: jax.Array
    done: jax.Array
    agent_positions: jax.Array
    action: jax.Array
    value: jax.Array
    log_prob: jax.Array
    advantages: jax.Array
    targets: jax.Array
    actor_indices: jax.Array


class CriticLossSurfaceSettings(NamedTuple):
    """Static settings captured by the jitted training function."""

    schedule: tuple[tuple[str, int], ...]
    num_actors: int
    snapshot_dir: str
    metadata: Mapping[str, Any]


def critic_surface_snapshot_schedule(
    total_updates: int,
    snapshot_fractions: Mapping[str, float],
) -> tuple[tuple[str, int], ...]:
    """Convert fractions into one-based, global completed-update counts."""

    total_updates = int(total_updates)
    if total_updates < 1:
        raise ValueError("total_updates must be positive")

    schedule = []
    for label, fraction in snapshot_fractions.items():
        fraction = float(fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"Snapshot fraction for {label!r} must be in (0, 1], "
                f"got {fraction}."
            )
        completed_updates = min(
            total_updates,
            max(1, int(math.ceil(fraction * total_updates))),
        )
        schedule.append((str(label), completed_updates))
    return tuple(schedule)


def critic_surface_snapshot_schedule_from_config(
    config: Mapping[str, Any],
    total_updates: int | None = None,
) -> tuple[tuple[str, int], ...]:
    surface_config = config.get("CRITIC_LOSS_SURFACE", {})
    return critic_surface_snapshot_schedule(
        config["NUM_UPDATES"] if total_updates is None else total_updates,
        surface_config.get(
            "SNAPSHOT_FRACTIONS", DEFAULT_SNAPSHOT_FRACTIONS
        ),
    )


def critic_surface_snapshot_dir(config: Mapping[str, Any]) -> str:
    return os.path.join(config["MID_CKPT_DIR"], "critic_loss_surface")


def build_critic_loss_surface_settings(
    config: Mapping[str, Any],
    algorithm: str,
    layout: str,
    actor_trunk_keys: tuple[str, ...],
    value_trunk_keys: tuple[str, ...],
    shared_trunk_keys: tuple[str, ...],
    value_coordinates: str,
) -> CriticLossSurfaceSettings:
    """Build the shared schedule, output path, and reconstruction metadata."""

    surface_config = config.get("CRITIC_LOSS_SURFACE", {})
    metadata = {
        "algorithm": algorithm,
        "env_name": config["ENV_NAME"],
        "layout": layout,
        "network_config": {
            key: config[key]
            for key in (
                "ENV_NAME", "ACTION_DIM", "FC_DIM_SIZE",
                "GRU_HIDDEN_DIM", "CONV_NET", "LSTM",
            )
        },
        "loss_config": {
            key: config[key]
            for key in ("CLIP_EPS", "VF_COEF", "ENT_COEF")
        },
        "value_coordinates": value_coordinates,
        "parameter_groups": {
            "actor_trunk": actor_trunk_keys,
            "value_trunk": value_trunk_keys,
            "shared_trunk": shared_trunk_keys,
        },
    }
    return CriticLossSurfaceSettings(
        schedule=critic_surface_snapshot_schedule_from_config(config),
        num_actors=int(surface_config.get("NUM_ACTORS", 16)),
        snapshot_dir=critic_surface_snapshot_dir(config),
        metadata=metadata,
    )


def select_critic_loss_surface_batch(
    initial_hstate: Any,
    traj_batch: Any,
    advantages: jax.Array,
    targets: jax.Array,
    sample_size: int,
    values: jax.Array | None = None,
) -> CriticLossSurfaceBatch:
    """Select evenly-spaced, intact recurrent trajectories from a rollout."""

    num_actors = int(traj_batch.obs.shape[1])
    sample_size = min(int(sample_size), num_actors)
    if sample_size < 1:
        raise ValueError("sample_size must be positive")

    actor_indices = jnp.linspace(
        0, num_actors - 1, sample_size
    ).round().astype(jnp.int32)

    def select_actor_axis(value):
        return jnp.take(value, actor_indices, axis=1)

    return CriticLossSurfaceBatch(
        initial_hstate=jax.tree.map(
            lambda value: jnp.take(value, actor_indices, axis=0),
            initial_hstate,
        ),
        obs=select_actor_axis(traj_batch.obs),
        done=select_actor_axis(traj_batch.done),
        agent_positions=select_actor_axis(traj_batch.agent_positions),
        action=select_actor_axis(traj_batch.action),
        value=select_actor_axis(
            traj_batch.value if values is None else values
        ),
        log_prob=select_actor_axis(traj_batch.log_prob),
        advantages=select_actor_axis(advantages),
        targets=select_actor_axis(targets),
        actor_indices=actor_indices,
    )


def critic_surface_snapshot_path(
    snapshot_dir: str,
    label: str,
    completed_updates: int,
) -> str:
    return os.path.join(
        snapshot_dir,
        f"{label}_update{int(completed_updates)}.pkl",
    )


def save_critic_loss_surface_snapshot(
    snapshot_dir: str,
    label: str,
    completed_updates: int,
    total_updates: int,
    params: Any,
    batch: CriticLossSurfaceBatch,
    metadata: Mapping[str, Any],
    popart_mu: Any | None = None,
    popart_sigma: Any | None = None,
) -> None:
    """Atomically save a model/data pair, without overwriting on resume."""

    path = critic_surface_snapshot_path(
        snapshot_dir, label, completed_updates
    )
    if os.path.exists(path):
        print(f"Critic loss-surface snapshot already exists: {path}")
        return

    os.makedirs(snapshot_dir, exist_ok=True)
    payload = {
        "format_version": 1,
        "label": label,
        "update_step": int(completed_updates),
        "total_updates": int(total_updates),
        "params": params,
        "batch": batch,
        "metadata": dict(metadata),
    }
    if popart_mu is not None:
        payload["popart_mu"] = popart_mu
    if popart_sigma is not None:
        payload["popart_sigma"] = popart_sigma

    temporary_path = f"{path}.tmp-{os.getpid()}"
    with open(temporary_path, "wb") as file:
        pickle.dump(payload, file)
    os.replace(temporary_path, path)
    print(f"Saved critic loss-surface snapshot: {path}")


def save_critic_loss_surface_snapshots(
    completed_updates: jax.Array,
    total_updates: int,
    settings: CriticLossSurfaceSettings,
    params: Any,
    initial_hstate: Any,
    traj_batch: Any,
    advantages: jax.Array,
    targets: jax.Array,
    values: jax.Array | None = None,
    popart_mu: jax.Array | None = None,
    popart_sigma: jax.Array | None = None,
) -> None:
    """Save any snapshot scheduled for this global completed-update count."""

    for snapshot_label, snapshot_update in settings.schedule:
        def save_snapshot(
            _, label=snapshot_label, scheduled_update=snapshot_update
        ):
            batch = select_critic_loss_surface_batch(
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                settings.num_actors,
                values=values,
            )

            if popart_mu is None:
                def callback(callback_params, callback_batch):
                    save_critic_loss_surface_snapshot(
                        settings.snapshot_dir,
                        label,
                        scheduled_update,
                        total_updates,
                        callback_params,
                        callback_batch,
                        settings.metadata,
                    )

                return jax.experimental.io_callback(
                    callback, None, params, batch, ordered=True
                )

            def popart_callback(
                callback_params, callback_batch, callback_mu, callback_sigma
            ):
                save_critic_loss_surface_snapshot(
                    settings.snapshot_dir,
                    label,
                    scheduled_update,
                    total_updates,
                    callback_params,
                    callback_batch,
                    settings.metadata,
                    popart_mu=callback_mu,
                    popart_sigma=callback_sigma,
                )

            return jax.experimental.io_callback(
                popart_callback,
                None,
                params,
                batch,
                popart_mu,
                popart_sigma,
                ordered=True,
            )

        jax.lax.cond(
            jnp.equal(completed_updates, snapshot_update),
            save_snapshot,
            lambda _: None,
            operand=None,
        )
