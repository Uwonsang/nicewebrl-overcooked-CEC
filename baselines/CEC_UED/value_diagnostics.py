"""Value-target, critic-quality, and TD-error diagnostics."""

import jax.numpy as jnp


def compute_value_diagnostics(
    *,
    raw_targets,
    critic_targets,
    critic_values,
    td_errors,
    rewards,
    actor_layout_ids,
    layout_names,
    normalized_target_prefix=None,
):
    """Compute rollout-wide and per-layout value diagnostics.

    ``critic_targets`` and ``critic_values`` must use the same scale. For a
    standard critic this is the raw return scale; for PopArt it is the
    normalized scale. ``td_errors`` must use the raw reward scale so results
    remain directly comparable between PopArt and non-PopArt runs.
    """
    critic_errors = critic_targets - critic_values
    metrics = {
        "target_raw/mean": raw_targets.mean(),
        "critic/rmse": jnp.sqrt(jnp.square(critic_errors).mean()),
        "td_error/rmse": jnp.sqrt(jnp.square(td_errors).mean()),
    }

    zero_reward_mask = (jnp.abs(rewards) < 1e-8).astype(jnp.float32)
    nonzero_reward_mask = 1.0 - zero_reward_mask
    zero_reward_count = zero_reward_mask.sum()
    nonzero_reward_count = nonzero_reward_mask.sum()

    metrics["td_error/zero_reward_rmse"] = jnp.where(
        zero_reward_count > 0,
        jnp.sqrt(
            (jnp.square(td_errors) * zero_reward_mask).sum()
            / jnp.maximum(zero_reward_count, 1.0)
        ),
        jnp.nan,
    )
    metrics["td_error/nonzero_reward_rmse"] = jnp.where(
        nonzero_reward_count > 0,
        jnp.sqrt(
            (jnp.square(td_errors) * nonzero_reward_mask).sum()
            / jnp.maximum(nonzero_reward_count, 1.0)
        ),
        jnp.nan,
    )

    if normalized_target_prefix is not None:
        metrics[f"{normalized_target_prefix}/mean"] = critic_targets.mean()

    for layout_id, layout_name in enumerate(layout_names):
        mask = (actor_layout_ids == layout_id).astype(jnp.float32)
        raw_count = mask.sum()
        count = jnp.maximum(raw_count, 1.0)
        valid = raw_count > 0

        metrics[f"target_raw/{layout_name}/mean"] = jnp.where(
            valid,
            (raw_targets * mask).sum() / count,
            jnp.nan,
        )
        metrics[f"critic/{layout_name}/rmse"] = jnp.where(
            valid,
            jnp.sqrt((jnp.square(critic_errors) * mask).sum() / count),
            jnp.nan,
        )
        metrics[f"td_error/{layout_name}/rmse"] = jnp.where(
            valid,
            jnp.sqrt((jnp.square(td_errors) * mask).sum() / count),
            jnp.nan,
        )

        if normalized_target_prefix is not None:
            normalized_mean = (critic_targets * mask).sum() / count
            normalized_variance = (
                jnp.square(critic_targets - normalized_mean) * mask
            ).sum() / count
            metrics[
                f"{normalized_target_prefix}/{layout_name}/mean"
            ] = jnp.where(valid, normalized_mean, jnp.nan)
            metrics[
                f"{normalized_target_prefix}/{layout_name}/std"
            ] = jnp.where(
                valid,
                jnp.sqrt(normalized_variance),
                jnp.nan,
            )

    return metrics
