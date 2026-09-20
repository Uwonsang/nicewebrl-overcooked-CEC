"""Shared gradient-conflict diagnostics for CEC IPPO variants."""

import jax
import jax.numpy as jnp


def empty_gradient_conflict_metrics(layout_names, dtype=jnp.float32):
    """Shape-compatible skipped output for interval-only diagnostics."""
    names = []
    for layout_name in layout_names:
        names.append(f"sample_share/{layout_name}")

    for loss_type in ("actor", "value"):
        norm_role = "actor" if loss_type == "actor" else "critic"
        for layout_name in layout_names:
            names.append(f"grad_norm_{norm_role}/{layout_name}")
            names.append(
                f"grad_contribution_magnitude_{norm_role}/{layout_name}"
            )
            names.append(
                f"grad_contribution_signed_{norm_role}/{layout_name}"
            )
        names.append(f"grad_norm_{norm_role}/total")

        for i, layout_i in enumerate(layout_names):
            for layout_j in layout_names[i + 1:]:
                names.append(
                    f"grad_conflict_{loss_type}/{layout_i}_vs_{layout_j}"
                )

        for metric_name in (
            "avg_pairwise_cosine",
            "conflict_rate",
            "avg_negative_cosine",
            "alignment",
        ):
            names.append(f"grad_conflict_sample_{loss_type}/{metric_name}")

    return {
        name: jnp.asarray(jnp.nan, dtype=dtype)
        for name in names
    }


def compute_gradient_conflict_metrics(
    network,
    original_params,
    initial_hstate,
    traj_batch,
    advantages,
    value_targets,
    layout_ids_full,
    layout_names,
    config,
    num_agents,
    pairing_key,
    value_trunk_keys,
    actor_trunk_keys):

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
            "pairwise_dot_products": {},
            "gradients": [],
        },
        "value": {
            "squared_norms": [],
            "pairwise_dot_products": {},
            "gradients": [],
        },
    }

    # Compute squared_norm, dot_products, gradient per layouts
    layout_sample_counts = []
    for layout_id in range(num_layouts):
        # Classification is environment-level. Repeat the mask for every agent
        environment_mask = (layout_ids_full == layout_id).astype(jnp.float32)
        actor_mask = jnp.tile(environment_mask, (1, num_agents))
        actor_sample_count = actor_mask.sum() + 1e-8
        layout_sample_counts.append(environment_mask.sum())

        def _layout_losses(params, mask=actor_mask, sample_count=actor_sample_count):

            _, policy, value = jax.checkpoint(network.apply)(
                params,
                initial_hstate,
                (traj_batch.obs, traj_batch.done, traj_batch.agent_positions),
            )
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
            for previous_layout_id, previous_gradient in enumerate(state["gradients"]):
                state["pairwise_dot_products"][
                    (previous_layout_id, layout_id)
                ] = _tree_dot_product(previous_gradient, gradient)
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

        # Cosine similarity using pairwise dot product and gradient norms
        for layout_i in range(num_layouts):
            for layout_j in range(layout_i + 1, num_layouts):
                pair_is_valid = valid_layouts[layout_i] & valid_layouts[layout_j]
                pair_dot_product = state["pairwise_dot_products"][
                    (layout_i, layout_j)
                ]
                pair_norm_product = jnp.sqrt(
                    state["squared_norms"][layout_i]
                    * state["squared_norms"][layout_j]
                )
                cosine_similarity = jnp.where(
                    pair_is_valid,
                    pair_dot_product / (pair_norm_product + 1e-8),
                    jnp.nan,
                )
                layout_pair_name = (
                    f"{layout_names[layout_i]}_vs_{layout_names[layout_j]}"
                )
                metrics[
                    f"grad_conflict_{loss_type}/{layout_pair_name}"
                ] = cosine_similarity

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

        # grad_contribution_magnitude & grad_contribution_signed
        for layout_id, layout_name in enumerate(layout_names):
            layout_gradient = state["gradients"][layout_id]

            layout_combined_dot_product = _tree_dot_product(
                layout_gradient, count_weighted_gradient_sum)

            # cosine simliarity between gradient of layout and average gradient    
            alignment_with_combined_gradient = jnp.where(
                valid_layouts[layout_id],
                layout_combined_dot_product
                / (
                    jnp.sqrt(
                        state["squared_norms"][layout_id]
                        * combined_gradient_squared_norm
                    )
                    + 1e-8
                ),
                jnp.nan,
            )

            sample_share = (
                layout_sample_counts_array[layout_id] / total_layout_samples
                )

            weighted_gradient_magnitude = jnp.where(
                valid_layouts[layout_id],
                sample_share * jnp.sqrt(
                    state["squared_norms"][layout_id]
                ),
                jnp.nan,
            )

            metrics[
                f"grad_contribution_magnitude_{parameter_role}/{layout_name}"
            ] = weighted_gradient_magnitude

            metrics[
                f"grad_contribution_signed_{parameter_role}/{layout_name}"
            ] = jnp.where(
                valid_layouts[layout_id],
                weighted_gradient_magnitude
                * alignment_with_combined_gradient,
                jnp.nan,
            )

    # Sample-level (per-environment) gradient conflict.
    # One sample is one environment slot's complete rollout and contains every
    # agent in that slot.

    def _select_parameter_group(parameter_tree, parameter_keys):
        return {
            key: parameter_tree["params"][key]
            for key in parameter_keys
        }

    # Compute the complete-rollout mean gradients with the same two losses used
    # for each environment sample. They are needed for the alignment statistic.
    def _global_losses(params):
        _, policy, predicted_values = jax.checkpoint(network.apply)(
            params,
            initial_hstate,
            (traj_batch.obs, traj_batch.done, traj_batch.agent_positions),
        )

        actor_loss = _ppo_actor_loss(
            policy.log_prob(traj_batch.action),
            traj_batch.log_prob,
            _normalize_rollout_advantages(advantages),
        ).mean()

        value_loss = _ppo_value_loss(
            predicted_values,
            traj_batch.value,
            value_targets,
        ).mean()

        return value_loss, actor_loss

    _, global_loss_vjp = jax.vjp(_global_losses, original_params)
    global_value_gradient, = global_loss_vjp((1.0, 0.0))
    global_actor_gradient, = global_loss_vjp((0.0, 1.0))

    num_environment_chunks = config["NUM_ENVS"] // config["GRAD_CONFLICT_CHUNK_SIZE"]
    if config["NUM_ENVS"] % config["GRAD_CONFLICT_CHUNK_SIZE"] != 0:
        raise ValueError(
            "GRAD_CONFLICT_CHUNK_SIZE must evenly divide NUM_ENVS"
        )

    # A fresh permutation creates NUM_ENVS / 2 disjoint random pairs. Chunking
    # changes peak memory only; it does not change the permutation or pair set.
    environment_permutation = jax.random.permutation(pairing_key, config["NUM_ENVS"])

    def _to_environment_chunks(array, actor_axis):
        actor_environment_shape = (
            *array.shape[:actor_axis],
            num_agents,
            config["NUM_ENVS"],
            *array.shape[actor_axis + 1:],
        )
        actor_environment_view = array.reshape(actor_environment_shape)

        environment_axis = actor_axis + 1
        environment_major = jnp.moveaxis(
            actor_environment_view,
            environment_axis,
            0,
        )
        shuffled_environments = jnp.take(
            environment_major, environment_permutation, axis=0
        )
        chunked_environment_shape = (
            num_environment_chunks,
            config["GRAD_CONFLICT_CHUNK_SIZE"],
            *shuffled_environments.shape[1:],
        )
        return shuffled_environments.reshape(chunked_environment_shape)

    # Rollout arrays have shape [T, NUM_ACTORS, ...], so their combined
    # actor-environment axis is 1. Initial hidden-state leaves have no time
    # axis and use [NUM_ACTORS, ...], so their combined axis is 0.
    environment_chunks = {
        "observations": _to_environment_chunks(traj_batch.obs, actor_axis=1),
        "dones": _to_environment_chunks(traj_batch.done, actor_axis=1),
        "agent_positions": _to_environment_chunks(traj_batch.agent_positions, actor_axis=1),
        "old_values": _to_environment_chunks(traj_batch.value, actor_axis=1),
        "targets": _to_environment_chunks(value_targets, actor_axis=1),
        "actions": _to_environment_chunks(traj_batch.action, actor_axis=1),
        "old_log_probs": _to_environment_chunks(traj_batch.log_prob, actor_axis=1),
        "advantages": _to_environment_chunks(advantages, actor_axis=1),
        "initial_hstate": jax.tree.map(lambda hidden_state: _to_environment_chunks(
                hidden_state, actor_axis=0),initial_hstate),
    }

    def _accumulate_environment_gradient(accumulator, environment_sample):
        def _environment_losses(params):
            _, policy, predicted_values = jax.checkpoint(network.apply)(
                params,
                environment_sample["initial_hstate"],
                (
                    environment_sample["observations"],
                    environment_sample["dones"],
                    environment_sample["agent_positions"],
                ),
            )
            value_loss = _ppo_value_loss(
                predicted_values,
                environment_sample["old_values"],
                environment_sample["targets"],
            ).mean()
            actor_loss = _ppo_actor_loss(
                policy.log_prob(environment_sample["actions"]),
                environment_sample["old_log_probs"],
                _normalize_rollout_advantages(
                    environment_sample["advantages"]
                ),
            ).mean()
            return value_loss, actor_loss

        _, environment_loss_vjp = jax.vjp(
            _environment_losses, original_params
        )

        full_value_gradient, = environment_loss_vjp((1.0, 0.0))
        full_actor_gradient, = environment_loss_vjp((0.0, 1.0))

        value_gradient = _select_parameter_group(
            full_value_gradient, value_trunk_keys
        )
        actor_gradient = _select_parameter_group(
            full_actor_gradient, actor_trunk_keys
        )

        value_squared_norm = _tree_squared_l2_norm(value_gradient)
        actor_squared_norm = _tree_squared_l2_norm(actor_gradient)
        value_norm_denominator = jnp.sqrt(value_squared_norm) + 1e-8
        actor_norm_denominator = jnp.sqrt(actor_squared_norm) + 1e-8
        normalized_value_gradient = jax.tree.map(
            lambda gradient: gradient / value_norm_denominator,
            value_gradient,
        )
        normalized_actor_gradient = jax.tree.map(
            lambda gradient: gradient / actor_norm_denominator,
            actor_gradient,
        )

        value_pair_cosine = _tree_dot_product(
            accumulator["pending_value_gradient"],
            normalized_value_gradient,
        )
        actor_pair_cosine = _tree_dot_product(
            accumulator["pending_actor_gradient"],
            normalized_actor_gradient,
        )
        next_pending_value_gradient = jax.tree.map(
            lambda pending, current: jnp.where(
                accumulator["has_pending_gradient"],
                jnp.zeros_like(pending),
                current,
            ),
            accumulator["pending_value_gradient"],
            normalized_value_gradient,
        )
        next_pending_actor_gradient = jax.tree.map(
            lambda pending, current: jnp.where(
                accumulator["has_pending_gradient"],
                jnp.zeros_like(pending),
                current,
            ),
            accumulator["pending_actor_gradient"],
            normalized_actor_gradient,
        )

        return {
            "value_normalized_gradient_sum": jax.tree.map(
                lambda total, current: total + current,
                accumulator["value_normalized_gradient_sum"],
                normalized_value_gradient,
            ),
            "actor_normalized_gradient_sum": jax.tree.map(
                lambda total, current: total + current,
                accumulator["actor_normalized_gradient_sum"],
                normalized_actor_gradient,
            ),
            "value_squared_norm_sum": (
                accumulator["value_squared_norm_sum"]
                + value_squared_norm
            ),
            "actor_squared_norm_sum": (
                accumulator["actor_squared_norm_sum"]
                + actor_squared_norm
            ),
            "value_normalized_squared_norm_sum": (
                accumulator["value_normalized_squared_norm_sum"]
                + value_squared_norm / (value_norm_denominator ** 2)
            ),
            "actor_normalized_squared_norm_sum": (
                accumulator["actor_normalized_squared_norm_sum"]
                + actor_squared_norm / (actor_norm_denominator ** 2)
            ),
            "pending_value_gradient": next_pending_value_gradient,
            "pending_actor_gradient": next_pending_actor_gradient,
            "has_pending_gradient": jnp.logical_not(
                accumulator["has_pending_gradient"]
            ),
            "value_conflict_count": (
                accumulator["value_conflict_count"]
                + accumulator["has_pending_gradient"]
                * (value_pair_cosine < 0)
            ),
            "actor_conflict_count": (
                accumulator["actor_conflict_count"]
                + accumulator["has_pending_gradient"]
                * (actor_pair_cosine < 0)
            ),
            "value_negative_cosine_sum": (
                accumulator["value_negative_cosine_sum"]
                + accumulator["has_pending_gradient"]
                * jnp.maximum(0.0, -value_pair_cosine)
            ),
            "actor_negative_cosine_sum": (
                accumulator["actor_negative_cosine_sum"]
                + accumulator["has_pending_gradient"]
                * jnp.maximum(0.0, -actor_pair_cosine)
            ),
            "matched_pair_count": (
                accumulator["matched_pair_count"]
                + accumulator["has_pending_gradient"]
            ),
        }, None

    def _accumulate_environment_chunk(accumulator, environment_chunk):
        updated_accumulator, _ = jax.lax.scan(
            _accumulate_environment_gradient,
            accumulator,
            environment_chunk,
        )
        return updated_accumulator, None

    value_gradient_template = _select_parameter_group(
        original_params, value_trunk_keys
    )
    actor_gradient_template = _select_parameter_group(
        original_params, actor_trunk_keys
    )
    initial_accumulator = {
        "value_normalized_gradient_sum": jax.tree.map(
            jnp.zeros_like, value_gradient_template
        ),
        "actor_normalized_gradient_sum": jax.tree.map(
            jnp.zeros_like, actor_gradient_template
        ),
        "value_squared_norm_sum": jnp.array(0.0),
        "actor_squared_norm_sum": jnp.array(0.0),
        "value_normalized_squared_norm_sum": jnp.array(0.0),
        "actor_normalized_squared_norm_sum": jnp.array(0.0),
        "pending_value_gradient": jax.tree.map(
            jnp.zeros_like, value_gradient_template
        ),
        "pending_actor_gradient": jax.tree.map(
            jnp.zeros_like, actor_gradient_template
        ),
        "has_pending_gradient": jnp.array(False),
        "value_conflict_count": jnp.array(0.0),
        "actor_conflict_count": jnp.array(0.0),
        "value_negative_cosine_sum": jnp.array(0.0),
        "actor_negative_cosine_sum": jnp.array(0.0),
        "matched_pair_count": jnp.array(0.0),
    }
    accumulated, _ = jax.lax.scan(
        _accumulate_environment_chunk,
        initial_accumulator,
        environment_chunks,
    )

    loss_summaries = (
        (
            "value",
            accumulated["value_squared_norm_sum"],
            accumulated["value_normalized_gradient_sum"],
            accumulated["value_normalized_squared_norm_sum"],
            global_value_gradient,
            value_trunk_keys,
            accumulated["value_conflict_count"],
            accumulated["value_negative_cosine_sum"],
        ),
        (
            "actor",
            accumulated["actor_squared_norm_sum"],
            accumulated["actor_normalized_gradient_sum"],
            accumulated["actor_normalized_squared_norm_sum"],
            global_actor_gradient,
            actor_trunk_keys,
            accumulated["actor_conflict_count"],
            accumulated["actor_negative_cosine_sum"],
        ),
    )

    for (
        loss_type,
        individual_squared_norm_sum,
        normalized_gradient_sum,
        normalized_squared_norm_sum,
        global_gradient,
        parameter_keys,
        conflict_count,
        negative_cosine_sum,
    ) in loss_summaries:
        selected_global_gradient = _select_parameter_group(
            global_gradient, parameter_keys
        )
        global_mean_gradient_squared_norm = _tree_squared_l2_norm(
            selected_global_gradient
        )
        all_gradient_sum_squared_norm = (
            config["NUM_ENVS"] ** 2
        ) * global_mean_gradient_squared_norm
        alignment = (
            all_gradient_sum_squared_norm
            / (
                config["NUM_ENVS"] * individual_squared_norm_sum
                + 1e-8
            )
        )
        # For u_i = g_i / (||g_i|| + eps):
        # sum_{i != j} cos(g_i, g_j) = ||sum_i u_i||^2 - sum_i ||u_i||^2.
        average_pairwise_cosine = (
            _tree_squared_l2_norm(normalized_gradient_sum)
            - normalized_squared_norm_sum
        ) / (
            config["NUM_ENVS"] * (config["NUM_ENVS"] - 1) + 1e-8
        )

        metrics[
            f"grad_conflict_sample_{loss_type}/avg_pairwise_cosine"
        ] = average_pairwise_cosine
        metrics[
            f"grad_conflict_sample_{loss_type}/conflict_rate"
        ] = conflict_count / (accumulated["matched_pair_count"] + 1e-8)
        metrics[
            f"grad_conflict_sample_{loss_type}/avg_negative_cosine"
        ] = negative_cosine_sum / (
            accumulated["matched_pair_count"] + 1e-8
        )
        metrics[
            f"grad_conflict_sample_{loss_type}/alignment"
        ] = alignment

    return metrics
