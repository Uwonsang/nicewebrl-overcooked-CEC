"""Small, model-independent helpers for evaluation diagnostics and logging."""

import jax
import jax.numpy as jnp


EVAL_CRITIC_STAT_NAMES = (
    "value_mean",
    "target_mean",
    "value_rmse",
    "td_error_rmse",
)


def compute_evaluation_critic_statistics(
    values, rewards, dones, final_values, gamma,
):
    """Compute raw-reward critic diagnostics for one evaluation rollout."""
    dones = dones.astype(values.dtype)
    next_values = jnp.concatenate(
        [values[1:], final_values[jnp.newaxis, :]], axis=0,
    )
    td_errors = rewards + gamma * (1.0 - dones) * next_values - values

    def _discounted_return(carry, transition):
        reward, done = transition
        target = reward + gamma * (1.0 - done) * carry
        return target, target

    _, value_targets = jax.lax.scan(
        _discounted_return,
        final_values,
        (rewards, dones),
        reverse=True,
    )
    value_errors = value_targets - values
    return {
        "value_mean": values.mean(),
        "target_mean": value_targets.mean(),
        # XP direction별 통계를 합친 뒤 RMSE를 계산할 수 있도록 MSE를 반환한다.
        "value_mse": jnp.square(value_errors).mean(),
        "td_error_mse": jnp.square(td_errors).mean(),
    }


def empty_evaluation_metrics(
    layout_names, eval_xp_enabled, xp_layout_names=None,
):
    """Return the fixed NaN pytree required by the skipped lax.cond branch."""
    if xp_layout_names is None:
        xp_layout_names = layout_names

    nan = jnp.array(jnp.nan, dtype=jnp.float32)
    metrics = {layout_name: nan for layout_name in layout_names}
    metrics["mean"] = nan

    for layout_name in layout_names:
        for stat_name in EVAL_CRITIC_STAT_NAMES:
            metrics[f"{layout_name}_critic_{stat_name}"] = nan

    if eval_xp_enabled:
        for layout_name in xp_layout_names:
            metrics[f"{layout_name}_xp"] = nan
            for stat_name in EVAL_CRITIC_STAT_NAMES:
                metrics[f"{layout_name}_xp_critic_{stat_name}"] = nan
        metrics["mean_xp"] = nan

    return metrics


def add_evaluation_metrics_to_log_dict(
    log_dict, eval_metrics, layout_names, eval_xp_enabled,
    xp_layout_names=None,
):
    """Add the flat internal evaluation results under their W&B key names."""
    if eval_metrics is None:
        return log_dict
    if xp_layout_names is None:
        xp_layout_names = layout_names

    log_dict["eval/mean"] = eval_metrics["mean"]
    for layout_name in layout_names:
        log_dict[f"eval/{layout_name}"] = eval_metrics[layout_name]
        for stat_name in EVAL_CRITIC_STAT_NAMES:
            source_key = f"{layout_name}_critic_{stat_name}"
            log_dict[f"eval_critic/{layout_name}/{stat_name}"] = eval_metrics[source_key]

    if eval_xp_enabled and "mean_xp" in eval_metrics:
        log_dict["eval_xp/mean"] = eval_metrics["mean_xp"]
        for layout_name in xp_layout_names:
            log_dict[f"eval_xp/{layout_name}"] = eval_metrics[f"{layout_name}_xp"]
            for stat_name in EVAL_CRITIC_STAT_NAMES:
                source_key = f"{layout_name}_xp_critic_{stat_name}"
                log_dict[f"eval_xp_critic/{layout_name}/{stat_name}"] = eval_metrics[source_key]

    return log_dict
