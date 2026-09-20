"""JAX/Flax representation diagnostics."""

import jax
import jax.numpy as jnp


INTERMEDIATE_FEATURE_GROUPS = {
    "shared": (
        "shared_conv_0",
        "shared_conv_1",
        "shared_dense_0",
        "shared_dense_1",
        "shared_recurrent",
    ),
    "actor": (
        "actor_hidden_0",
        "actor_hidden_1",
        "actor_hidden_2",
        "actor_hidden_3",
    ),
    "critic": (
        "critic_hidden_0",
        "critic_hidden_1",
        "critic_hidden_2",
        "critic_hidden_3",
    ),
}

INTERMEDIATE_FEATURE_NAMES = tuple(
    name
    for names in INTERMEDIATE_FEATURE_GROUPS.values()
    for name in names
)

SEPARATE_TRUNK_FEATURE_GROUPS = {
    "actor_trunk": (
        "actor_trunk_conv_0",
        "actor_trunk_conv_1",
        "actor_trunk_dense_0",
        "actor_trunk_dense_1",
        "actor_trunk_recurrent",
    ),
    "critic_trunk": (
        "critic_trunk_conv_0",
        "critic_trunk_conv_1",
        "critic_trunk_dense_0",
        "critic_trunk_dense_1",
        "critic_trunk_recurrent",
    ),
    "actor": INTERMEDIATE_FEATURE_GROUPS["actor"],
    "critic": INTERMEDIATE_FEATURE_GROUPS["critic"],
}


def _sum_group(layer_norms, feature_groups, group):
    """Sum recorded feature norms belonging to one named layer group."""
    group_feature_norm = sum(
        layer_norms[name]
        for name in feature_groups[group])
    
    return group_feature_norm 

def tree_global_l2_norm(tree):
    """Global L2 norm over every array leaf in a pytree."""
    squared_norm = sum(
        (
            jnp.sum(jnp.square(x))
            for x in jax.tree_util.tree_leaves(tree)
        ),
        jnp.asarray(0.0),
    )
    global_l2_norm = jnp.sqrt(squared_norm)

    return global_l2_norm


def parameter_group_l2_norm(params, module_names):
    """All-leaf global L2 norm for selected Flax parameter modules."""
    param_tree = params["params"] if "params" in params else params
    selected = {name: param_tree[name] for name in module_names}
    group_l2_norm = tree_global_l2_norm(selected)

    return group_l2_norm


def tree_leaf_count_weighted_rms_norm(tree):
    """Leaf-size-weighted RMS of leaf L2 norms for an array pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    leaf_l2_norms = jnp.stack([jnp.linalg.norm(leaf) for leaf in leaves])
    leaf_sizes = jnp.asarray(
        [leaf.size for leaf in leaves],
        dtype=leaf_l2_norms.dtype,
    )
    weighted_rms_norm = jnp.sqrt(
        jnp.average(
            jnp.square(leaf_l2_norms),
            weights=leaf_sizes,
        )
    )

    return weighted_rms_norm


def tree_group_leaf_count_weighted_rms_norm(tree, module_names):
    """Leaf-size-weighted RMS for selected Flax parameter modules."""
    module_tree = tree["params"] if "params" in tree else tree
    selected = {name: module_tree[name] for name in module_names}
    group_weighted_rms_norm = tree_leaf_count_weighted_rms_norm(selected)

    return group_weighted_rms_norm


def compute_gradient_norm_metrics(
    gradients,
    actor_param_keys,
    critic_param_keys,
    shared_param_keys=None,
):
    """Measure global and parameter-group RMS norms of PPO gradients."""
    gradients_norm = {
        "gradient_norm/global_norm": tree_global_l2_norm(gradients),
        "gradient_norm/weighted_rms_norm": (
            tree_leaf_count_weighted_rms_norm(gradients)
        ),
        "gradient_norm/actor_weighted_rms_norm": (
            tree_group_leaf_count_weighted_rms_norm(
                gradients, actor_param_keys
            )
        ),
        "gradient_norm/critic_weighted_rms_norm": (
            tree_group_leaf_count_weighted_rms_norm(
                gradients, critic_param_keys
            )
        ),
    }
    if shared_param_keys is not None:
        gradients_norm["gradient_norm/shared_weighted_rms_norm"] = (
            tree_group_leaf_count_weighted_rms_norm(
                gradients, shared_param_keys
            )
        )

    return gradients_norm


def gradient_kurtosis(gradients, epsilon=1e-8):
    """Pearson kurtosis of log-transformed absolute gradients."""
    leaves = jax.tree_util.tree_leaves(gradients)
    flattened_gradients = jnp.concatenate([leaf.reshape(-1) for leaf in leaves])

    log_absolute_gradients = jnp.log(jnp.abs(flattened_gradients) + epsilon)
    centered_gradients = (
        log_absolute_gradients - jnp.mean(log_absolute_gradients)
    )
    variance = jnp.mean(centered_gradients ** 2)
    fourth_central_moment = jnp.mean(centered_gradients ** 4)
    kurtosis = fourth_central_moment / (jnp.square(variance) + 1e-12)

    return kurtosis


def compute_gradient_kurtosis_metrics(
    gradients,
    actor_param_keys,
    critic_param_keys,
    shared_param_keys=None,
):
    """Measure log-absolute-gradient kurtosis for PPO parameter groups."""
    gradient_tree = gradients["params"] if "params" in gradients else gradients
    actor_gradients = {name: gradient_tree[name] for name in actor_param_keys}
    critic_gradients = {name: gradient_tree[name] for name in critic_param_keys}

    gradient_kurtosis_metrics = {
        "gradient_kurtosis/global": gradient_kurtosis(gradients),
        "gradient_kurtosis/actor": gradient_kurtosis(actor_gradients),
        "gradient_kurtosis/critic": gradient_kurtosis(critic_gradients),
    }
    if shared_param_keys is not None:
        shared_gradients = {
            name: gradient_tree[name] for name in shared_param_keys
        }
        gradient_kurtosis_metrics["gradient_kurtosis/shared"] = (
            gradient_kurtosis(shared_gradients)
        )

    return gradient_kurtosis_metrics


def compute_weight_metrics(
    params,
    actor_param_keys,
    critic_param_keys,
    shared_param_keys=None,
):
    """Measure SimBaV2-style parameter norms at one optimizer-update state."""

    representation_weight = {
        "representation_weight/weight_norm": tree_global_l2_norm(params),
        "representation_weight/actor_weight_norm": parameter_group_l2_norm(
            params, actor_param_keys
        ),
        "representation_weight/critic_weight_norm": parameter_group_l2_norm(
            params, critic_param_keys
        ),
        "representation_weight/weighted_rms_norm": (
            tree_leaf_count_weighted_rms_norm(params)
        ),
        "representation_weight/actor_weighted_rms_norm": (
            tree_group_leaf_count_weighted_rms_norm(
                params, actor_param_keys
            )
        ),
        "representation_weight/critic_weighted_rms_norm": (
            tree_group_leaf_count_weighted_rms_norm(
                params, critic_param_keys
            )
        ),
    }
    if shared_param_keys is not None:
        representation_weight["representation_weight/shared_weight_norm"] = (
            parameter_group_l2_norm(params, shared_param_keys)
        )
        representation_weight[
            "representation_weight/shared_weighted_rms_norm"
        ] = tree_group_leaf_count_weighted_rms_norm(
            params, shared_param_keys
        )

    return representation_weight 
    

def compute_optimizer_update_metrics(
    gradients,
    params,
    actor_param_keys,
    critic_param_keys,
    shared_param_keys=None,
):
    group_keys = {
        "actor_param_keys": actor_param_keys,
        "critic_param_keys": critic_param_keys,
        "shared_param_keys": shared_param_keys,
    }

    optimizer_metrics = {
        **compute_gradient_norm_metrics(gradients, **group_keys),
        **compute_gradient_kurtosis_metrics(gradients, **group_keys),
        **compute_weight_metrics(params, **group_keys),
    }

    return optimizer_metrics 


def feature_scale_metrics(feature_matrix, singular_values):
    """Compute feature magnitude and leading-spectrum scale metrics."""
    n_samples = feature_matrix.shape[0]
    normalized_singular_values = singular_values / jnp.sqrt(
        jnp.asarray(n_samples, dtype=feature_matrix.dtype)
    )

    feature_scale = {
        "feature_norm": jnp.mean(
            jnp.linalg.norm(feature_matrix, axis=-1)
        ),
        "normalized_sigma_1": normalized_singular_values[0],
        "sigma_1_ratio": singular_values[0] / jnp.sum(singular_values),
    }

    return feature_scale


# source: https://github.com/DAVIAN-Robotics/SimbaV2/blob/86899c277cdc697b2b02d827243de1ea93f20a1d/scale_rl/agents/jax_utils/metrics.py#L390
def feature_rank_metrics(feature_matrix, svals, cutoff=0.01):
    """Compute rank metrics from one feature matrix and its spectrum."""
    threshold = 1 - cutoff

    # Roy & Vetterli (2007): exp(entropy) of the normalized singular-value
    # distribution. Unlike the threshold feature rank, this is invariant to a
    # uniform rescaling of all features.
    sval_sum = jnp.sum(svals)
    sval_dist = svals / sval_sum
    # Replace 0 with 1. This is a safe trick to avoid log(0) = -inf
    # as Roy & Vetterli assume 0*log(0) = 0 = 1*log(1).
    sval_dist_fixed = jnp.where(sval_dist == 0, jnp.ones_like(sval_dist), sval_dist)
    effective_rank_vetterli = jnp.exp(-jnp.sum(sval_dist_fixed * jnp.log(sval_dist_fixed)))

    # Yang et al. (2020): smallest k explaining 1 - cutoff of the PCA
    # variance, i.e. the squared singular-value sum.
    sval_squares = svals**2
    sval_squares_sum = jnp.sum(sval_squares)
    cumsum_squares = jnp.cumsum(sval_squares)
    threshold_crossed = cumsum_squares >= (threshold * sval_squares_sum)
    approximate_rank_pca = ((~threshold_crossed).sum() + 1) 

    # Kumar et al. (2020): smallest k whose leading singular values explain
    # 1 - cutoff of the nuclear norm.
    cumsum_svals = jnp.cumsum(svals)
    threshold_crossed = cumsum_svals >= threshold * sval_sum
    srank_kumar = (~threshold_crossed).sum() + 1

    # Lyle et al. (2022): singular values of Phi / sqrt(N) above epsilon.
    n_obs = feature_matrix.shape[0]
    svals_of_normalized = svals / jnp.sqrt(n_obs)
    over_cutoff = svals_of_normalized > cutoff
    feature_rank = over_cutoff.sum()

    # JAX chooses a numerical tolerance from the input shape and precision.
    jnp_ranks = jnp.linalg.matrix_rank(feature_matrix)

    rank_metrics = {
        "feature_rank": feature_rank.astype(jnp.float32),
        "effective_rank_vetterli": effective_rank_vetterli.astype(jnp.float32),
        "srank_kumar": srank_kumar.astype(jnp.float32),
        "approximate_rank_pca": approximate_rank_pca.astype(jnp.float32),
        "matrix_rank": jnp_ranks.astype(jnp.float32),
    }

    return rank_metrics


def compute_feature_metrics(features, cutoff=0.01):
    """Compute feature-scale and rank metrics with one shared SVD."""
    feature_matrix = features.reshape((-1, features.shape[-1]))
    singular_values = jnp.linalg.svdvals(feature_matrix)

    scale_metrics = feature_scale_metrics(
        feature_matrix,
        singular_values,
    )
    rank_metrics = feature_rank_metrics(
        feature_matrix,
        singular_values,
        cutoff=cutoff,
    )

    return scale_metrics, rank_metrics


def first_epoch_first_minibatch_indices(
    update_rng,
    num_actors,
    num_minibatches,
):
    """Reproduce the actor indices used by the first PPO minibatch.

    ``_update_epoch`` splits its incoming RNG once and uses the second key for
    the actor-axis permutation. Calling this helper with that same incoming
    RNG therefore selects exactly the actors that the first epoch's first
    minibatch will consume, without advancing the training RNG.
    """
    if num_actors % num_minibatches != 0:
        raise ValueError(
            "NUM_ACTORS must be divisible by NUM_MINIBATCHES, got "
            f"{num_actors} and {num_minibatches}"
        )
    _, permutation_key = jax.random.split(update_rng)
    permutation = jax.random.permutation(permutation_key, num_actors)
    actors_per_minibatch = num_actors // num_minibatches
    return permutation[:actors_per_minibatch]


def compute_minibatch_penultimate_metrics(
    network,
    params,
    initial_hstate,
    network_inputs,
    cutoff=0.01,
):
    """Measure representation metrics on one recurrent PPO minibatch.

    The input keeps the complete rollout-time axis and a subset of the actor
    axis, together with the matching initial recurrent states. The network
    therefore reconstructs features with the same sequence and reset handling
    as PPO. Time and actor axes are flattened only after features are computed;
    actors from the same environment remain separate rows.
    """
    _, intermediates = network.apply(
        params,
        initial_hstate,
        network_inputs,
        mutable=["intermediates"],
    )

    captured = intermediates["intermediates"]
    role_features = (
        ("shared", captured["shared_penultimate"][0]),
        ("actor", captured["actor_penultimate"][0]),
        ("critic", captured["critic_penultimate"][0])
    )
 

    metrics = {}
    for role, features in role_features:
        scale_metrics, rank_metrics = compute_feature_metrics(features,cutoff=cutoff)
        for name, value in scale_metrics.items():
            metrics[f"representation_feature/{role}_{name}"] = value
        for name, value in rank_metrics.items():
            metrics[f"representation_rank/{role}_{name}"] = value

    layer_norms = {
        name: jnp.mean(captured[f"feature_norm_{name}"][0])
        for name in INTERMEDIATE_FEATURE_NAMES
    }
    # Match SimbaV2's featnorm_total: sum the mean per-sample feature norm of
    # each layer. Actor/critic totals both include the shared feature path.
    shared_total = _sum_group(layer_norms, INTERMEDIATE_FEATURE_GROUPS, "shared")
    actor_branch_total = _sum_group(layer_norms, INTERMEDIATE_FEATURE_GROUPS, "actor")
    critic_branch_total = _sum_group(layer_norms, INTERMEDIATE_FEATURE_GROUPS, "critic")
    metrics["representation_feature_total/actor_feature_norm"] = (
        shared_total + actor_branch_total)
    metrics["representation_feature_total/critic_feature_norm"] = (
        shared_total + critic_branch_total)
    metrics["representation_feature_total/shared_feature_norm"] = shared_total

    return metrics


def empty_penultimate_metrics(dtype=jnp.float32):
    """Shape/dtype-compatible output for skipped logging steps."""
    names = (
        "representation_feature/shared_feature_norm",
        "representation_feature/shared_normalized_sigma_1",
        "representation_feature/shared_sigma_1_ratio",
        "representation_feature/actor_feature_norm",
        "representation_feature/actor_normalized_sigma_1",
        "representation_feature/actor_sigma_1_ratio",
        "representation_feature/critic_feature_norm",
        "representation_feature/critic_normalized_sigma_1",
        "representation_feature/critic_sigma_1_ratio",
        "representation_rank/shared_feature_rank",
        "representation_rank/shared_effective_rank_vetterli",
        "representation_rank/shared_srank_kumar",
        "representation_rank/shared_approximate_rank_pca",
        "representation_rank/shared_matrix_rank",
        "representation_rank/actor_feature_rank",
        "representation_rank/actor_effective_rank_vetterli",
        "representation_rank/actor_srank_kumar",
        "representation_rank/actor_approximate_rank_pca",
        "representation_rank/actor_matrix_rank",
        "representation_rank/critic_feature_rank",
        "representation_rank/critic_effective_rank_vetterli",
        "representation_rank/critic_srank_kumar",
        "representation_rank/critic_approximate_rank_pca",
        "representation_rank/critic_matrix_rank",
    )
    names = names + (
        "representation_feature_total/actor_feature_norm",
        "representation_feature_total/critic_feature_norm",
        "representation_feature_total/shared_feature_norm",
    )
    empty_metrics = {
        name: jnp.asarray(jnp.nan, dtype=dtype)
        for name in names
    }

    return empty_metrics


def compute_separate_trunk_penultimate_metrics(
    network,
    params,
    initial_hstate,
    network_inputs,
    cutoff=0.01,
):
    """Measure actor/critic representations for independent recurrent trunks."""
    _, intermediates = network.apply(
        params,
        initial_hstate,
        network_inputs,
        mutable=["intermediates"],
    )
    captured = intermediates["intermediates"]

    role_features = (
        ("actor_trunk", captured["actor_trunk_penultimate"][0]),
        ("critic_trunk", captured["critic_trunk_penultimate"][0]),
        ("actor", captured["actor_penultimate"][0]),
        ("critic", captured["critic_penultimate"][0]),
    )

    metrics = {}
    for role, features in role_features:
        scale_metrics, rank_metrics = compute_feature_metrics(features, cutoff=cutoff)
        for name, value in scale_metrics.items():
            metrics[f"representation_feature/{role}_{name}"] = value
        for name, value in rank_metrics.items():
            metrics[f"representation_rank/{role}_{name}"] = value
    
    layer_norms = {
        name: jnp.mean(
            captured["actor_trunk"][f"feature_norm_{name}"][0]
        )
        for name in SEPARATE_TRUNK_FEATURE_GROUPS["actor_trunk"]
    }
    layer_norms.update({
        name: jnp.mean(
            captured["critic_trunk"][f"feature_norm_{name}"][0]
        )
        for name in SEPARATE_TRUNK_FEATURE_GROUPS["critic_trunk"]
    })
    layer_norms.update({
        name: jnp.mean(captured[f"feature_norm_{name}"][0])
        for group in ("actor", "critic")
        for name in SEPARATE_TRUNK_FEATURE_GROUPS[group]
    })

    metrics["representation_feature_total/actor_feature_norm"] = (
        _sum_group(layer_norms, SEPARATE_TRUNK_FEATURE_GROUPS, "actor_trunk")
        + _sum_group(layer_norms, SEPARATE_TRUNK_FEATURE_GROUPS, "actor")
    )
    metrics["representation_feature_total/critic_feature_norm"] = (
        _sum_group(layer_norms, SEPARATE_TRUNK_FEATURE_GROUPS, "critic_trunk")
        + _sum_group(layer_norms, SEPARATE_TRUNK_FEATURE_GROUPS, "critic")
    )
    
    return metrics


def empty_separate_trunk_penultimate_metrics(dtype=jnp.float32):
    """Shape-compatible skipped output for separate-trunk representations."""
    roles = ("actor_trunk", "critic_trunk", "actor", "critic")
    scale_names = (
        "feature_norm",
        "normalized_sigma_1",
        "sigma_1_ratio",
    )
    rank_names = (
        "feature_rank",
        "effective_rank_vetterli",
        "srank_kumar",
        "approximate_rank_pca",
        "matrix_rank",
    )
    names = tuple(
        f"representation_feature/{role}_{name}"
        for role in roles
        for name in scale_names
    ) + tuple(
        f"representation_rank/{role}_{name}"
        for role in roles
        for name in rank_names
    ) + (
        "representation_feature_total/actor_feature_norm",
        "representation_feature_total/critic_feature_norm",
    )
    return {
        name: jnp.asarray(jnp.nan, dtype=dtype)
        for name in names
    }
