"""Inference network used by the CEC-IDAAC checkpoints."""

from typing import Dict, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from actor_networks import ScannedRNN


class RecurrentFeatureTrunk(nn.Module):
  """Independent CNN, dense, and LSTM path for an actor or critic."""

  config: Dict

  @nn.compact
  def __call__(self, hidden, observations, dones):
    batch_size, num_environments, _ = observations.shape

    if self.config["GRAPH_NET"]:
      observation_shape = self.config["obs_dim"]
      reshaped_observations = observations.reshape(-1, *observation_shape)

      embedding = nn.Conv(
        features=64,
        kernel_size=(2, 2),
        kernel_init=orthogonal(np.sqrt(2)),
        bias_init=constant(0.0),
        name="conv_0",
      )(reshaped_observations)
      embedding = nn.relu(embedding)

      embedding = nn.Conv(
        features=32,
        kernel_size=(2, 2),
        kernel_init=orthogonal(np.sqrt(2)),
        bias_init=constant(0.0),
        name="conv_1",
      )(embedding)
      embedding = nn.relu(embedding)

      embedding = embedding.reshape((batch_size, num_environments, -1))
    else:
      embedding = observations

    embedding = nn.Dense(
      self.config["FC_DIM_SIZE"] * 2,
      kernel_init=orthogonal(np.sqrt(2)),
      bias_init=constant(0.0),
      name="dense_0",
    )(embedding)
    embedding = nn.relu(embedding)

    embedding = nn.Dense(
      self.config["FC_DIM_SIZE"] * 2,
      kernel_init=orthogonal(np.sqrt(2)),
      bias_init=constant(0.0),
      name="dense_1",
    )(embedding)
    embedding = nn.relu(embedding)

    hidden, embedding = ScannedRNN(name="recurrent")(
      hidden,
      (embedding, dones),
    )

    embedding = embedding.reshape((batch_size, num_environments, -1))
    return hidden, embedding


class ActorCriticIDAAC(nn.Module):
  """Actor and critic with completely separate recurrent feature trunks."""

  action_dim: Sequence[int]
  config: Dict

  @staticmethod
  def initialize_carry(batch_size, hidden_size):
    actor_hidden = ScannedRNN.initialize_carry(batch_size, hidden_size)
    critic_hidden = ScannedRNN.initialize_carry(batch_size, hidden_size)
    return actor_hidden, critic_hidden

  @nn.compact
  def __call__(
    self,
    hidden,
    inputs,
    return_auxiliary=False,
    order_swap=None,
  ):
    observations, dones, agent_positions = inputs
    del agent_positions

    actor_hidden, critic_hidden = hidden

    actor_hidden, actor_embedding = RecurrentFeatureTrunk(
      config=self.config,
      name="actor_trunk",
    )(
      actor_hidden,
      observations,
      dones,
    )

    critic_hidden, critic_embedding = RecurrentFeatureTrunk(
      config=self.config,
      name="critic_trunk",
    )(
      critic_hidden,
      observations,
      dones,
    )

    actor_features = nn.Dense(
      self.config["GRU_HIDDEN_DIM"],
      kernel_init=orthogonal(2),
      bias_init=constant(0.0),
      name="actor_hidden_0",
    )(actor_embedding)
    actor_features = nn.relu(actor_features)

    actor_features = nn.Dense(
      self.config["GRU_HIDDEN_DIM"] * 3 // 4,
      kernel_init=orthogonal(2),
      bias_init=constant(0.0),
      name="actor_hidden_1",
    )(actor_features)
    actor_features = nn.relu(actor_features)

    actor_features = nn.Dense(
      self.config["GRU_HIDDEN_DIM"] // 2,
      kernel_init=orthogonal(2),
      bias_init=constant(0.0),
      name="actor_hidden_2",
    )(actor_features)
    actor_features = nn.relu(actor_features)

    actor_features = nn.Dense(
      self.config["GRU_HIDDEN_DIM"] // 4,
      kernel_init=orthogonal(2),
      bias_init=constant(0.0),
      name="actor_hidden_3",
    )(actor_features)
    actor_features = nn.relu(actor_features)

    actor_logits = nn.Dense(
      self.action_dim,
      kernel_init=orthogonal(0.01),
      bias_init=constant(0.0),
      name="actor_output",
    )(actor_features)
    policy = distrax.Categorical(logits=actor_logits)

    advantage_predictions = nn.Dense(
      self.action_dim,
      kernel_init=orthogonal(1.0),
      bias_init=constant(0.0),
      name="advantage_output",
    )(actor_features)

    next_actor_features = jnp.roll(actor_features, shift=-1, axis=0)
    if order_swap is None:
      order_swap = jnp.zeros(actor_features.shape[:2], dtype=bool)

    first_order_features = jnp.where(
      order_swap[..., None],
      next_actor_features,
      actor_features,
    )
    second_order_features = jnp.where(
      order_swap[..., None],
      actor_features,
      next_actor_features,
    )
    order_features = jnp.concatenate(
      (first_order_features, second_order_features),
      axis=-1,
    )

    if self.config.get("IDAAC_USE_NONLINEAR_CLF", False):
      classifier_size = self.config.get("IDAAC_CLF_HIDDEN_SIZE", 4)
      order_features = nn.Dense(
        classifier_size,
        kernel_init=orthogonal(np.sqrt(2)),
        bias_init=constant(0.0),
        name="order_classifier_hidden",
      )(order_features)
      order_features = nn.relu(order_features)

    order_logits = nn.Dense(
      1,
      kernel_init=orthogonal(1.0),
      bias_init=constant(0.0),
      name="order_classifier_output",
    )(order_features)
    order_logits = order_logits.squeeze(-1)

    critic = nn.Dense(
      self.config["FC_DIM_SIZE"] * 2,
      kernel_init=orthogonal(2),
      bias_init=constant(0.0),
      name="critic_hidden_0",
    )(critic_embedding)
    critic = nn.relu(critic)

    critic = nn.Dense(
      self.config["FC_DIM_SIZE"],
      kernel_init=orthogonal(2),
      bias_init=constant(0.0),
      name="critic_hidden_1",
    )(critic)
    critic = nn.relu(critic)

    critic = nn.Dense(
      self.config["FC_DIM_SIZE"] * 3 // 4,
      kernel_init=orthogonal(2),
      bias_init=constant(0.0),
      name="critic_hidden_2",
    )(critic)
    critic = nn.relu(critic)

    critic = nn.Dense(
      self.config["FC_DIM_SIZE"] // 2,
      kernel_init=orthogonal(2),
      bias_init=constant(0.0),
      name="critic_hidden_3",
    )(critic)
    critic = nn.relu(critic)

    critic = nn.Dense(
      1,
      kernel_init=orthogonal(1.0),
      bias_init=constant(0.0),
      name="critic_output",
    )(critic)
    value = jnp.squeeze(critic, axis=-1)

    new_hidden = (actor_hidden, critic_hidden)
    outputs = (new_hidden, policy, value)

    if return_auxiliary:
      return outputs + (advantage_predictions, order_logits)

    return outputs
