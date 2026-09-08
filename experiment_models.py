"""Build and load the policy models selected by ``web_app.py``."""

import json
import os

from actor_networks import ActorCriticE3T, ActorCriticRNN, ScannedRNN
from idaac_actor_networks import ActorCriticIDAAC
from model_loading import load_parameter_stack


def make_rnn_hidden_state(config):
  hidden_size = config["GRU_HIDDEN_DIM"]
  return ScannedRNN.initialize_carry(1, hidden_size)


def make_idaac_hidden_state(config):
  hidden_size = config["GRU_HIDDEN_DIM"]
  return ActorCriticIDAAC.initialize_carry(1, hidden_size)


def load_tutorial_ippo_model(config, action_dim):
  """Load IPPO partners trained on the fixed tutorial layout."""
  tutorial_model = ActorCriticRNN(
    action_dim=action_dim,
    config=config,
  )

  tutorial_model_dir = "models/IPPO/asymm_advantages_9"
  tutorial_checkpoint_pattern = "seed*/seed*_best.pkl"

  tutorial_params, number_of_seeds, checkpoint_names = load_parameter_stack(
    tutorial_model_dir,
    pattern=tutorial_checkpoint_pattern,
  )

  def initialize_tutorial_hidden_state():
    return make_rnn_hidden_state(config)

  return (
    tutorial_model,
    tutorial_params,
    number_of_seeds,
    checkpoint_names,
    initialize_tutorial_hidden_state,
  )


def load_experiment_models(config, action_dim):
  """Return selected models, checkpoints, and matching hidden-state factories."""
  serialized_specs = os.environ.get("NICEWEBRL_ALGORITHM_SPECS", "{}")
  algorithm_specs = json.loads(serialized_specs)

  requested_text = os.environ.get("NICEWEBRL_ALGORITHMS", "")
  requested_names = requested_text.split(",")

  requested_algorithms = []
  for algorithm_name in requested_names:
    algorithm_name = algorithm_name.strip()
    if algorithm_name:
      requested_algorithms.append(algorithm_name)

  if not requested_algorithms:
    raise ValueError("No algorithms were selected")

  base_model = ActorCriticRNN(
    action_dim=action_dim,
    config=config,
  )
  e3t_model = ActorCriticE3T(
    action_dim=action_dim,
    config=config,
  )
  idaac_model = ActorCriticIDAAC(
    action_dim=action_dim,
    config=config,
  )

  network_models = {
    "rnn": base_model,
    "e3t": e3t_model,
    "idaac": idaac_model,
  }

  hidden_state_factories = {
    "rnn": lambda: make_rnn_hidden_state(config),
    "e3t": lambda: make_rnn_hidden_state(config),
    "idaac": lambda: make_idaac_hidden_state(config),
  }

  model_dict = {}
  param_dict = {}
  num_seed_dict = {}
  checkpoint_name_dict = {}
  init_hidden_state_fn_dict = {}

  layout_name = config["layout_name"]

  for algorithm_name in requested_algorithms:
    if algorithm_name not in algorithm_specs:
      raise ValueError(f"No model specification for '{algorithm_name}'")

    algorithm_spec = algorithm_specs[algorithm_name]
    network_name = algorithm_spec["network"]

    if network_name not in network_models:
      raise ValueError(
        f"Unknown network '{network_name}' for '{algorithm_name}'"
      )

    model_path_template = algorithm_spec["path"]
    model_dir = model_path_template.format(layout=layout_name)
    checkpoint_pattern = algorithm_spec["glob"]

    loaded_params, number_of_seeds, checkpoint_names = load_parameter_stack(
      model_dir,
      pattern=checkpoint_pattern,
    )

    model_dict[algorithm_name] = network_models[network_name]
    param_dict[algorithm_name] = loaded_params
    num_seed_dict[algorithm_name] = number_of_seeds
    checkpoint_name_dict[algorithm_name] = checkpoint_names
    init_hidden_state_fn_dict[algorithm_name] = hidden_state_factories[network_name]

  return (
    model_dict,
    param_dict,
    num_seed_dict,
    checkpoint_name_dict,
    init_hidden_state_fn_dict,
  )
