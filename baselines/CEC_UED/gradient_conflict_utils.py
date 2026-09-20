"""Shared gradient-conflict diagnostics for CEC IPPO variants."""

import jax
import jax.numpy as jnp


def empty_layout_gradient_metrics(layout_names, dtype=jnp.float32):
    """Shape-compatible skipped output for layout-gradient diagnostics."""
    names = []
    for layout_name in layout_names:
        names.append(f"sample_share/{layout_name}")

    for loss_type in ("actor", "value"):
        norm_role = "actor" if loss_type == "actor" else "critic"
        for layout_name in layout_names:
            names.append(f"grad_norm_{norm_role}/{layout_name}")
            names.append(
                f"grad_contribution_signed_{norm_role}/{layout_name}"
            )
        names.append(f"grad_norm_{norm_role}/total")

    return {
        name: jnp.asarray(jnp.nan, dtype=dtype)
        for name in names
    }


def compute_layout_gradient_metrics(
    network,
    original_params,
    initial_hstate,
    traj_batch,
    advantages,
    value_targets,
    layout_ids_full,
    layout_names,
    config,
    num_agents):

    num_layouts = len(layout_names)
    advantage_mean = jnp.mean(advantages)
    advantage_std = jnp.std(advantages)

    def _tree_dot_product(tree_a, tree_b):
        return sum(
            jnp.sum(leaf_a * leaf_b)
            for leaf_a, leaf_b in zip(
                jax.tree_util.tree_leaves(tree_a),
                jax.tree_util.tree_leaves(tree_b),
            )
        )

    def _tree_squared_l2_norm(tree):
        return sum(
            jnp.sum(leaf ** 2)
            for leaf in jax.tree_util.tree_leaves(tree)
        )

    def _normalize_rollout_advantages(unnormalized_advantages):
        return (
            (unnormalized_advantages - advantage_mean)
            / (advantage_std + 1e-8)
        )

    def _ppo_value_loss(predicted_values, old_values, targets):

        clipped_values = old_values + (
            predicted_values - old_values
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_losses = jnp.square(predicted_values - targets)
        value_losses_clipped = jnp.square(clipped_values - targets)

        return  0.5 * jnp.maximum(value_losses, value_losses_clipped) 

    def _ppo_actor_loss(new_log_probs, old_log_probs, normalized_advantages):

        probability_ratio = jnp.exp(new_log_probs - old_log_probs)
        loss_actor1 = probability_ratio * normalized_advantages
        loss_actor2 = (jnp.clip(
                probability_ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"],
            )
            * normalized_advantages
        ) 

        return -jnp.minimum(loss_actor1, loss_actor2)

    layout_gradient_state = {
        "actor": {
            "squared_norms": [],
            "gradients": [],
        },
        "value": {
            "squared_norms": [],
            "gradients": [],
        },
    }

    # Compute actor/value gradients for each layout.
    layout_sample_counts = []
    for layout_id in range(num_layouts):
        # Classification is environment-level. Repeat the mask for every agent
        environment_mask = (layout_ids_full == layout_id).astype(jnp.float32)
        actor_mask = jnp.tile(environment_mask, (1, num_agents))
        actor_sample_count = actor_mask.sum() + 1e-8
        layout_sample_counts.append(environment_mask.sum())

        def _layout_losses(params, mask=actor_mask, sample_count=actor_sample_count):

            network_output = jax.checkpoint(network.apply)(
                params,
                initial_hstate,
                (traj_batch.obs, traj_batch.done, traj_batch.agent_positions),
            )
            _, policy, value = network_output[:3]
            log_prob = policy.log_prob(traj_batch.action)

            value_loss_per_sample = _ppo_value_loss(
                value,
                traj_batch.value,
                value_targets,
            )
            value_loss = (
                (value_loss_per_sample * mask).sum() / sample_count
            )

            actor_loss_per_sample = _ppo_actor_loss(
                log_prob,
                traj_batch.log_prob,
                _normalize_rollout_advantages(advantages),
            )
            actor_loss = (
                actor_loss_per_sample * mask
            ).sum() / sample_count

            return value_loss, actor_loss 

        _, layout_loss_vjp = jax.vjp(_layout_losses, original_params)
        # Two VJP calls then select the actor and value gradients without running two independent forwards.
        # Cotangent (1, 0) selects value loss, (0, 1) selects actor loss.
        value_gradient, = layout_loss_vjp((1.0, 0.0))
        actor_gradient, = layout_loss_vjp((0.0, 1.0))
        
        for loss_type, gradient in (
            ("actor", actor_gradient),
            ("value", value_gradient),
        ):
            state = layout_gradient_state[loss_type]
            state["squared_norms"].append(_tree_squared_l2_norm(gradient))
            state["gradients"].append(gradient)

    metrics = {}
    layout_sample_counts_array = jnp.stack(layout_sample_counts)
    total_layout_samples = layout_sample_counts_array.sum() + 1e-8
    valid_layouts = layout_sample_counts_array > 0

    # sample_share about layout
    for layout_id, layout_name in enumerate(layout_names):
        metrics[f"sample_share/{layout_name}"] = (
            layout_sample_counts_array[layout_id] / total_layout_samples
        )

    for loss_type, state in layout_gradient_state.items():
        parameter_role = "actor" if loss_type == "actor" else "critic"

        # gradient norm = the square root of the squared norm
        for layout_id, layout_name in enumerate(layout_names):
            metrics[f"grad_norm_{parameter_role}/{layout_name}"] = jnp.where(
                valid_layouts[layout_id],
                jnp.sqrt(state["squared_norms"][layout_id]),
                jnp.nan,
            )

        # Each stored gradient is a within-layout mean.
        # Weighting it by the layout count reconstructs the complete-rollout gradient sum.
        count_weighted_gradient_sum = jax.tree.map(
            lambda *layout_gradients: sum(
                layout_sample_counts_array[index] * gradient
                for index, gradient in enumerate(layout_gradients)
            ),
            *state["gradients"],
        )
        combined_gradient_squared_norm = _tree_squared_l2_norm(
            count_weighted_gradient_sum
        )
        metrics[f"grad_norm_{parameter_role}/total"] = (
            jnp.sqrt(combined_gradient_squared_norm) / total_layout_samples
        )

        # Signed contribution is the sample-weighted projection of each
        # layout gradient onto the combined-gradient direction.
        for layout_id, layout_name in enumerate(layout_names):
            layout_gradient = state["gradients"][layout_id]

            layout_combined_dot_product = _tree_dot_product(
                layout_gradient, count_weighted_gradient_sum)

            sample_share = (
                layout_sample_counts_array[layout_id] / total_layout_samples
                )

            metrics[
                f"grad_contribution_signed_{parameter_role}/{layout_name}"
            ] = jnp.where(
                valid_layouts[layout_id],
                sample_share
                * layout_combined_dot_product
                / (jnp.sqrt(combined_gradient_squared_norm) + 1e-8),
                jnp.nan,
            )

    return metrics
