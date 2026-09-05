"""NiceWebRL stage extension that records the CEC paper's human metrics.

The upstream stage stores the current timestep and the human key press.  This
extension additionally records the selected model checkpoint, the AI action,
recipe deliveries, agent positions, and agent-to-agent movement collisions.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import Any

import aiofiles
import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.viz.overcooked_jitted_visualizer import overlay_score_text
from nicegui import app, ui
from nicewebrl import MultiAgentEnvStage
from nicewebrl.nicejax import base64_npimage, make_serializable, new_rng
from nicewebrl.utils import write_msgpack_record


DELIVERY_REWARD = 20.0
ACTION_NAMES = {
  0: "right",
  1: "down",
  2: "left",
  3: "up",
  4: "stay",
  5: "interact",
}
DIR_TO_VEC = np.asarray(
  [
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1],
  ],
  dtype=np.int64,
)

try:
  jax_tree_map = jax.tree.map
except AttributeError:
  jax_tree_map = jax.tree_map


def _as_scalar(value: Any) -> float:
  return float(np.asarray(value).sum())


def _collision_details(state: Any, joint_actions: list[int]) -> dict[str, bool]:
  """Match JAXMARL Overcooked's same-cell and position-swap blocking rules."""
  positions = np.asarray(state.agent_pos, dtype=np.int64)
  wall_map = np.asarray(state.wall_map, dtype=bool)
  goals = {tuple(pos) for pos in np.asarray(state.goal_pos, dtype=np.int64)}
  height, width = wall_map.shape
  desired = positions.copy()

  for agent_id, action in enumerate(joint_actions):
    if action not in range(4):
      continue
    target = positions[agent_id] + DIR_TO_VEC[action]
    target[0] = np.clip(target[0], 0, width - 1)
    target[1] = np.clip(target[1], 0, height - 1)
    if wall_map[target[1], target[0]] or tuple(target) in goals:
      continue
    desired[agent_id] = target

  same_target = bool(np.array_equal(desired[0], desired[1]))
  swap_places = bool(
    np.array_equal(desired[0], positions[1])
    and np.array_equal(desired[1], positions[0])
  )
  return {
    "collision_event": same_target or swap_places,
    "same_target_collision": same_target,
    "swap_collision": swap_places,
  }


@dataclass
class PaperMetricsMultiAgentEnvStage(MultiAgentEnvStage):
  """Multi-agent stage with paper-compatible per-step data collection."""

  # Values passed to the environment in the same order as ``action_keys``.
  action_values: Any = None
  checkpoint_names: list[str] | None = None

  def _environment_action(self, action_index: int) -> int | None:
    if self.action_values is None or action_index < 0:
      return None
    values = np.asarray(self.action_values)
    if action_index >= len(values):
      return None
    return int(values[action_index])

  async def step_and_send_timestep(self, container, update_display: bool = True):
    """Precompute human moves while retaining the AI action and collisions."""
    rng = new_rng()
    timestep = self.get_user_data("stage_state").timestep

    if self.model is not None:
      all_obs = self.web_env.env._env.get_obs(timestep.state)
      human_id = int(self.get_user_data("human_id"))
      other_agent_obs = all_obs["agent_1" if human_id == 0 else "agent_0"]
      other_agent_obs = other_agent_obs.flatten()[None, None]
      agent_pos = timestep.state.agent_pos.astype(jnp.int32)[None, None]
      sim_done = timestep.last().astype(jnp.int32).reshape((1, 1))
      hidden_state = self.get_user_data("hidden_state")
      model_index = self.get_user_data("model_index")
      if self.using_param_stack:
        model_params = jax_tree_map(lambda x: x[model_index], self.model_params)
      else:
        model_params = self.model_params
      hidden_state, pi = self.model.apply(
        model_params,
        hidden_state,
        (other_agent_obs, sim_done, agent_pos),
      )[:2]
      await self.set_user_data(hidden_state=hidden_state)
      other_agent_action = jnp.argmax(pi.probs, 2)[0].squeeze().astype(jnp.int32)
    else:
      other_agent_action = jnp.asarray(4, dtype=jnp.int32)

    human_id = int(self.get_user_data("human_id"))
    ai_action = int(np.asarray(other_agent_action))
    action_values = [
      self._environment_action(index) for index in range(len(self.action_keys))
    ]
    collision_details = []
    for human_action in action_values:
      joint_actions = [ai_action, ai_action]
      joint_actions[human_id] = int(human_action)
      joint_actions[1 - human_id] = ai_action
      collision_details.append(_collision_details(timestep.state, joint_actions))

    next_timesteps = self.web_env.next_steps(
      rng,
      timestep,
      self.env_params,
      other_action=other_agent_action,
      h_id=human_id,
    )
    current_score = self.get_user_data("score", 0.0)
    next_images = self.vmap_render_fn(next_timesteps)
    next_images = {
      self.action_keys[idx]: base64_npimage(
        overlay_score_text(
          np.array(image), current_score + float(next_timesteps.reward[idx])
        )
      )
      for idx, image in enumerate(next_images)
    }
    ui.run_javascript(f"window.next_states = {next_images};")
    await self.set_user_data(
      next_timesteps=next_timesteps,
      pending_ai_action=ai_action,
      pending_collision_details=collision_details,
    )

    if update_display:
      await self.display_timestep(container, timestep)
    else:
      ui.run_javascript(
        "window.imageSeenTime = window.next_imageSeenTime;", timeout=10
      )

  async def save_key_data(self, event):
    """Attach the selected precomputed transition to the normal save queue."""
    stage_state = self.get_user_data("stage_state")
    if stage_state is None:
      return

    args = dict(event.args)
    action_index = self.key_to_action.get(args.get("key"), -1)
    details = self.get_user_data("pending_collision_details", [])
    collision = (
      details[action_index]
      if 0 <= action_index < len(details)
      else {
        "collision_event": None,
        "same_target_collision": None,
        "swap_collision": None,
      }
    )
    collision_count = int(self.get_user_data("collision_count", 0))
    if collision.get("collision_event"):
      collision_count += 1

    timestep = stage_state.timestep
    next_timesteps = self.get_user_data("next_timesteps")
    positions_after = None
    if next_timesteps is not None and action_index >= 0:
      positions_after = np.asarray(
        next_timesteps.state.agent_pos[action_index]
      ).tolist()

    args.update(
      human_environment_action=self._environment_action(action_index),
      ai_action=self.get_user_data("pending_ai_action"),
      human_id=int(self.get_user_data("human_id")),
      model_index=self.get_user_data("model_index"),
      agent_positions_before=np.asarray(timestep.state.agent_pos).tolist(),
      agent_positions_after=positions_after,
      cumulative_collisions=collision_count,
      **collision,
    )
    await self.set_user_data(collision_count=collision_count)

    processed_timestep = self.preprocess_timestep(timestep)
    async with self.get_user_lock():
      await self.get_user_queue().put((args, processed_timestep, self.user_stats()))
    asyncio.create_task(self._process_save_queue())

  async def save_experiment_data(self, args, timestep, user_stats):
    """Write the upstream record plus explicit paper metrics."""
    key = args["key"]
    action_index = self.key_to_action.get(key, -1)
    action_name = self.action_to_name.get(action_index, key)
    step_reward = _as_scalar(timestep.reward)
    model_index = args.get("model_index", self.get_user_data("model_index"))
    model_index = int(model_index) if model_index is not None else None
    checkpoint_name = None
    if (
      model_index is not None
      and self.checkpoint_names
      and 0 <= model_index < len(self.checkpoint_names)
    ):
      checkpoint_name = self.checkpoint_names[model_index]

    timestep_data = {}
    if self.custom_data_fn is not None:
      timestep_data = self.custom_data_fn(timestep)
      timestep_data = jax_tree_map(make_serializable, timestep_data)

    metadata = copy.deepcopy(self.metadata)
    metadata.update(type="EnvStage", **user_stats)
    save_data = {
      "stage_idx": app.storage.user.get("stage_idx"),
      "session_id": app.storage.browser["id"],
      "data": {
        "image_seen_time": args.get("imageSeenTime"),
        "action_taken_time": args.get("keydownTime"),
        "computer_interaction": key,
        "action_name": action_name,
        "action_idx": action_index,
        "human_environment_action": args.get("human_environment_action"),
        "human_environment_action_name": ACTION_NAMES.get(
          args.get("human_environment_action")
        ),
        "ai_action": args.get("ai_action"),
        "ai_action_name": ACTION_NAMES.get(args.get("ai_action")),
        "human_id": int(
          args.get("human_id", self.get_user_data("human_id"))
        ),
        "model_index": model_index,
        "model_checkpoint": checkpoint_name,
        "step_reward": step_reward,
        "delivery_event": step_reward / DELIVERY_REWARD,
        "collision_event": args.get("collision_event"),
        "same_target_collision": args.get("same_target_collision"),
        "swap_collision": args.get("swap_collision"),
        "cumulative_collisions": args.get("cumulative_collisions"),
        "agent_positions_before": args.get("agent_positions_before"),
        "agent_positions_after": args.get("agent_positions_after"),
        "timelimit": self.duration,
        "timestep": self.serializer.serialize(timestep),
        **timestep_data,
      },
      "user_data": {
        "user_id": app.storage.user["seed"],
        "age": app.storage.user.get("age"),
        "sex": app.storage.user.get("sex"),
      },
      "metadata": metadata,
      "name": self.name,
      "body": self.body,
    }

    async with aiofiles.open(self.user_save_file_fn(), "ab") as stream:
      await write_msgpack_record(stream, save_data)
    await self.set_user_data(saved_data=True)
