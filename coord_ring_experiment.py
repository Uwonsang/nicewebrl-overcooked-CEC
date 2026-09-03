from dotenv import load_dotenv
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import os
import pickle

from jaxmarl.viz.overcooked_jitted_visualizer import render_fn as overcooked_render_fn
from jaxmarl.viz.overcooked_jitted_visualizer import overlay_score_text
from jaxmarl.environments.overcooked import Overcooked, Actions, State
from jaxmarl.environments.overcooked.layouts import overcooked_layouts
from jaxmarl.environments.overcooked.layouts import (
  make_asymm_advantages_9x9,
  make_coord_ring_9x9,
  make_counter_circuit_9x9,
  make_forced_coord_9x9,
  make_cramped_room_9x9,
)

# import jaxmarl.make as make
import jaxmarl

from nicegui import ui, app
import nicewebrl
from nicewebrl import MultiAgentJaxWebEnv, base64_npimage, TimestepWrapper
from nicewebrl import (
  Stage,
  MultiAgentEnvStage,
  FeedbackStage,
  Block,
  prepare_blocks,
  generate_stage_order,
)
from nicewebrl import get_logger
from actor_networks import ActorCriticRNN, ScannedRNN, ActorCriticE3T
import pdb
import asyncio

load_dotenv()

logger = get_logger(__name__)
VERBOSITY = int(os.environ.get("VERBOSITY", 0))
DEBUG = int(os.environ.get("DEBUG", 0))
WORLD_SEED = int(os.environ.get("WORLD_SEED", 1))
NAME = os.environ.get("NAME", "coord_ring")
DATA_DIR = os.environ.get("DATA_DIR", "data")

MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 201
MIN_SUCCESS_EPISODES = 100


def get_user_save_file_fn():
  return (
    f"{DATA_DIR}/user={app.storage.user.get('seed')}_name={NAME}_debug={DEBUG}.json"
  )


########################################
# Define actions and corresponding keys
########################################
actions = [
  Actions.up,
  Actions.down,
  Actions.left,
  Actions.right,
  Actions.stay,
  Actions.interact,
]
action_array = jnp.array([a.value for a in actions])
action_keys = ["ArrowLeft", "ArrowDown", "ArrowRight", "ArrowUp", "s", " "]
action_to_name = ["위", "아래", "왼쪽", "오른쪽", "대기", "상호작용"]


def initialize_environment(config):
  layout_name = config["ENV_KWARGS"]["layout"]
  config["layout_name"] = layout_name
  config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
  env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

  if config["ENV_NAME"] == "overcooked":

    def reset_env(key):
      def reset_sub_dict(key, fn):
        key, subkey = jax.random.split(key)
        sampled_layout_dict = fn(subkey, ik=True)
        temp_o, temp_s = env.custom_reset(
          key, layout=sampled_layout_dict, random_reset=False, shuffle_inv_and_pot=False
        )
        key, subkey = jax.random.split(key)
        return (temp_o, temp_s), key

      asymm_reset, key = reset_sub_dict(key, make_asymm_advantages_9x9)
      coord_ring_reset, key = reset_sub_dict(key, make_coord_ring_9x9)
      counter_circuit_reset, key = reset_sub_dict(key, make_counter_circuit_9x9)
      forced_coord_reset, key = reset_sub_dict(key, make_forced_coord_9x9)
      cramped_room_reset, key = reset_sub_dict(key, make_cramped_room_9x9)
      layout_resets = [
        asymm_reset,
        coord_ring_reset,
        counter_circuit_reset,
        forced_coord_reset,
        cramped_room_reset,
      ]
      # stack all layouts
      stacked_layout_reset = jax.tree_map(lambda *x: jnp.stack(x), *layout_resets)
      # sample an index from 0 to 4
      index = jax.random.randint(key, (), minval=0, maxval=5)
      sampled_reset = jax.tree_map(lambda x: x[index], stacked_layout_reset)
      return sampled_reset

    def gen_held_out(runner_state, unused):
      (i,) = runner_state
      _, ho_state = reset_env(jax.random.key(i))
      res = (ho_state.goal_pos, ho_state.wall_map, ho_state.pot_pos)
      carry = (i + 1,)
      return carry, res

    carry, res = jax.lax.scan(gen_held_out, (0,), jnp.arange(100), 100)
    ho_goal, ho_wall, ho_pot = [], [], []
    for (
      layout_name,
      layout_dict,
    ) in overcooked_layouts.items():  # add hand crafted ones to heldout set
      if "9" in layout_name:
        _, ho_state = env.custom_reset(
          jax.random.PRNGKey(0),
          random_reset=False,
          shuffle_inv_and_pot=False,
          layout=layout_dict,
        )
        ho_goal.append(ho_state.goal_pos)
        ho_wall.append(ho_state.wall_map)
        ho_pot.append(ho_state.pot_pos)
    ho_goal = jnp.stack(ho_goal, axis=0)
    ho_wall = jnp.stack(ho_wall, axis=0)
    ho_pot = jnp.stack(ho_pot, axis=0)
    ho_goal = jnp.concatenate([res[0], ho_goal], axis=0)
    ho_wall = jnp.concatenate([res[1], ho_wall], axis=0)
    ho_pot = jnp.concatenate([res[2], ho_pot], axis=0)
    env.held_out_goal, env.held_out_wall, env.held_out_pot = (ho_goal, ho_wall, ho_pot)
  config["obs_dim"] = env.observation_space(env.agents[0]).shape
  return env


import yaml
from pathlib import Path

base_config = yaml.safe_load(Path("overcooked_config.yaml").read_text())
base_config["ENV_NAME"] = "overcooked"
base_config["ENV_KWARGS"]["check_held_out"] = False
base_config["ENV_KWARGS"]["shuffle_inv_and_pot"] = False
base_config["ENV_KWARGS"]["random_reset"] = False
base_config["ENV_KWARGS"]["random_reset_fn"] = "reset_all"
base_config["ENV_KWARGS"]["layout"] = "coord_ring_9"
base_config["ENV_KWARGS"]["max_steps"] = MAX_EPISODE_TIMESTEPS - 1
base_config["GRAPH_NET"] = True

tutorial_config = pickle.loads(pickle.dumps(base_config))
tutorial_config["ENV_KWARGS"]["layout"] = "asymm_advantages_9"


########################################
# Define Overcooked environment
########################################
jax_env = initialize_environment(base_config)
jax_env_tutorial = initialize_environment(tutorial_config)
default_params = {"random_reset_fn": 0}

########################################
# Load agent models
########################################
base_agent_model = ActorCriticRNN(action_dim=len(actions), config=base_config)
e3t_agent_model = ActorCriticE3T(action_dim=len(actions), config=base_config)

model_dict = {
  "ik": base_agent_model,
  "ik_finetune": base_agent_model,
  "sk": base_agent_model,
  "sk_e3t": e3t_agent_model,
  "sk_fcp": base_agent_model,
  "counter_sk": base_agent_model,
  "counter_fcp": base_agent_model,
}
param_dict = {
  "ik": [],
  "ik_finetune": [],
  "sk": [],
  "sk_e3t": [],
  "sk_fcp": [],
  "counter_sk": [],
  "counter_fcp": [],
}
num_seed_dict = {
  "ik": 0,
  "ik_finetune": 0,
  "sk": 0,
  "sk_e3t": 0,
  "sk_fcp": 0,
  "counter_sk": 0,
  "counter_fcp": 0,
}

for model_name in model_dict.keys():
  if "counter" in model_name:
    continue
  model_dir = f"models/{model_name}/coord_ring/"
  # load all files in model_dir
  files = os.listdir(model_dir)
  for file in files:
    with open(os.path.join(model_dir, file), "rb") as f:
      params = pickle.load(f)["params"]
      param_dict[model_name].append(params)
      num_seed_dict[model_name] += 1
  param_dict[model_name] = jax.tree_map(
    lambda *x: jnp.stack(x), *param_dict[model_name]
  )

for model_name in ["sk", "sk_fcp"]:  # add useless counter circuit models
  model_dir = f"models/{model_name}/counter_circuit/"
  dict_name = "counter_sk" if model_name == "sk" else "counter_fcp"
  # load all files in model_dir
  files = os.listdir(model_dir)
  for file in files:
    with open(os.path.join(model_dir, file), "rb") as f:
      params = pickle.load(f)["params"]
      param_dict[dict_name].append(params)
      num_seed_dict[dict_name] += 1
  param_dict[dict_name] = jax.tree_map(lambda *x: jnp.stack(x), *param_dict[dict_name])

init_hidden_state_fn = lambda: ScannedRNN.initialize_carry(
  1, base_config["GRU_HIDDEN_DIM"]
)


# NiceWebRL exploits a `TimeStep` object for checking episode conditions
# wrap environment in wrapper if needed
jax_env = TimestepWrapper(
  jax_env, autoreset=True, reset_w_batch_dim=False, use_params=False
)
jax_env_tutorial = TimestepWrapper(
  jax_env_tutorial, autoreset=True, reset_w_batch_dim=False, use_params=False
)
# create web environment wrapper
jax_web_env = MultiAgentJaxWebEnv(env=jax_env, actions=action_array)
jax_web_env_tutorial = MultiAgentJaxWebEnv(env=jax_env_tutorial, actions=action_array)

# Call this function to pre-compile jax functions before experiment starts.
jax_web_env.precompile(dummy_env_params=default_params)
jax_web_env_tutorial.precompile(dummy_env_params=default_params)


# Define rendering function
def render_fn(timestep: nicewebrl.Timestep):
  image = overcooked_render_fn(timestep.state)
  return image.astype(jnp.uint8)


def render_fn_tutorial(timestep: nicewebrl.Timestep):
  image = overcooked_render_fn(timestep.state)
  return image.astype(jnp.uint8)


# precompile vmapped render fn that will vmap over all actions
vmap_render_fn = jax_web_env.precompile_vmap_render_fn(render_fn, default_params)
vmap_render_fn_tutorial = jax_web_env_tutorial.precompile_vmap_render_fn(
  render_fn_tutorial, default_params
)

# compile it so fast
render_fn = (
  jax.jit(render_fn)
  .lower(jax_web_env.reset(jax.random.PRNGKey(0), default_params))
  .compile()
)
render_fn_tutorial = (
  jax.jit(render_fn_tutorial)
  .lower(jax_web_env_tutorial.reset(jax.random.PRNGKey(0), default_params))
  .compile()
)


async def user_survey_display_fn(stage, container):
  nicewebrl.clear_element(container)
  with container.style("align-items: center;"):
    ui.markdown("## 사용자 설문")

    ui.markdown("아래에 Prolific ID를 입력해 주세요.")
    prolific_id = ui.input(placeholder="Prolific ID")

    ui.markdown("방금 AI 에이전트와 플레이한 경험을 바탕으로 답해 주세요.")

    questions = [
      ("The agent adapted to me when making decisions.", "AI 에이전트는 의사결정을 할 때 나에게 맞춰 행동했다."),
      ("The agent was consistent in its actions.", "AI 에이전트의 행동은 일관적이었다."),
      ("The agent's actions were human-like.", "AI 에이전트의 행동은 사람처럼 자연스러웠다."),
      ("The agent frequently got in my way.", "AI 에이전트는 자주 내 움직임을 방해했다."),
      ("The agent's behavior was frustrating.", "AI 에이전트의 행동 때문에 답답함을 느꼈다."),
      ("Overall, I enjoyed playing with the agent.", "전반적으로 AI 에이전트와 함께 플레이하는 것이 즐거웠다."),
      ("Overall, I felt that the agent's ability to coordinate with me was:", "전반적으로 AI 에이전트가 나와 협력하는 능력은 어떠했습니까?"),
    ]

    responses = {"prolific_id": prolific_id}
    completed = {}
    completed_all = asyncio.Event()

    def create_on_change(q_idx):
      def on_change(val):
        completed[q_idx] = True
        if len(completed) == len(questions):  # +1 for prolific_id
          completed_all.set()

      return on_change

    # # Add prolific ID completion check
    # prolific_id.on_change(create_on_change('prolific'))

    for i, (question_key, question_text) in enumerate(questions):
      ui.markdown(question_text)
      options = (
        {
          "Strongly disagree": "전혀 그렇지 않다",
          "Disagree": "그렇지 않다",
          "Neutral": "보통이다",
          "Agree": "그렇다",
          "Strongly agree": "매우 그렇다",
        }
        if i < len(questions) - 1
        else {
          "Very poor": "매우 부족했다",
          "Poor": "부족했다",
          "Neutral": "보통이었다",
          "Good": "좋았다",
          "Very good": "매우 좋았다",
        }
      )
      dropdown = ui.select(options, on_change=create_on_change(i))
      responses[question_key] = dropdown

    ui.markdown(f"{stage.body}")

    await completed_all.wait()
    return {k: v.value for k, v in responses.items()}


def make_survey_stage(name="사용자 설문"):
  stage = FeedbackStage(
    name=name,
    body="",
    display_fn=user_survey_display_fn,
    user_save_file_fn=get_user_save_file_fn,
    next_button=True,
  )
  return stage


########################################
# Define Stages of experiment
########################################
all_stages = []
all_blocks = []


# ------------------
# Instruction stage
# ------------------
async def instruction_display_fn(stage, container):
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    ui.markdown(f"## {stage.name}")
    ui.markdown(
      "AI 에이전트와 함께 Overcooked를 플레이합니다. AI 에이전트는 과제를 완료할 수 있도록 여러분과 협력합니다."
    )
    ui.markdown(
      "여러분은 한 캐릭터를 조작하고, AI 에이전트는 다른 캐릭터를 조작합니다."
    )
    ui.markdown(
      "주어진 과제를 AI 에이전트와 협력하여 완료해 주세요."
    )
    ui.markdown("키보드의 방향키를 사용하여 위, 아래, 왼쪽, 오른쪽으로 이동합니다.")
    ui.markdown("물건을 집거나 내려놓는 등 상호작용하려면 스페이스바를 누릅니다.")
    ui.markdown("제자리에서 기다리려면 S 키를 누릅니다.")


async def tutorial_display_fn(stage, container):
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    ui.markdown(
      "먼저 조작 방법에 익숙해질 수 있도록 튜토리얼을 진행합니다."
    )
    ui.markdown(
      "실험이 끝날 때까지 이 페이지를 닫거나 벗어나지 마세요. 페이지를 나가면 다시 돌아올 수 없습니다."
    )


async def post_tutorial_display_fn(stage, container):
  with container.style("align-items: center;"):
    ui.markdown(f"## {stage.name}")
    ui.markdown(
      "게임 방법을 확인했습니다. 이제 본 실험을 시작합니다."
    )


# ------------------
# Environment stage
# ------------------
env_params = default_params


def make_image_html(src):
  html = f'''
    <div id="stateImageContainer" style="display: flex; justify-content: center; align-items: center;">
        <img id="stateImage" src="{src}" style="width: 100%; height: 100%; object-fit: contain;">
    </div>
    '''
  return html


async def env_stage_display_fn(
  stage: MultiAgentEnvStage, container: ui.element, timestep: nicewebrl.Timestep
):
  state_image = stage.render_fn(timestep)
  score = stage.get_user_data("score", 0.0)
  state_image = overlay_score_text(np.array(state_image), score)
  state_image = base64_npimage(state_image)
  stage_state = stage.get_user_data("stage_state")
  human_color = stage.get_user_data("human_color")
  human_color_ko = {"blue": "파란색", "green": "초록색"}.get(
    human_color, human_color
  )

  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    # --------------------------------
    # tell person how many episodes completed
    # --------------------------------
    with ui.row():
      with ui.element("div").classes("p-2 bg-green-100"):
        ui.label().bind_text_from(
          stage_state,
          "nepisodes",
          lambda n: f"진행: {n}/{stage.max_episodes}. 여러분은 {human_color_ko} 캐릭터를 조작합니다.",
        )

    # --------------------------------
    # display environment
    # --------------------------------
    ui.html(make_image_html(src=state_image))


def evaluate_success_fn(timestep: nicewebrl.Timestep, env_params: struct.PyTreeNode):
  """Episode finishes if person every gets 1 achievement"""
  success = int(timestep.state.terminal)
  return success


# ------------------
# Transition stage
# ------------------
async def transition_display_fn(stage, container):
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    ui.markdown(f"## {stage.name}")
    ui.markdown(
      "설문을 완료한 후 아래 버튼을 눌러 계속해 주세요."
    )


instruction_stage = Stage(name="게임 안내", display_fn=instruction_display_fn)
tutorial_stage = Stage(name="튜토리얼 안내", display_fn=tutorial_display_fn)
tutorial_env_stage = MultiAgentEnvStage(
  name=f"tutorial",
  web_env=jax_web_env_tutorial,
  action_keys=action_keys,
  action_to_name=action_to_name,
  env_params=env_params,
  render_fn=render_fn_tutorial,
  vmap_render_fn=vmap_render_fn_tutorial,
  display_fn=env_stage_display_fn,
  evaluate_success_fn=evaluate_success_fn,
  notify_success=False,
  min_success=MIN_SUCCESS_EPISODES,
  max_episodes=MAX_STAGE_EPISODES,
  verbosity=VERBOSITY,
  user_save_file_fn=get_user_save_file_fn,
  metadata=dict(
    # nothing required, just for bookkeeping
    desc="some description",
    key1="value1",
    key2="value2",
  ),
  model=base_agent_model,  # temporarily have other agent stay in place
  model_params=param_dict["ik"],
  num_seeds=num_seed_dict["ik"],
  using_param_stack=True,
  init_hidden_state_fn=init_hidden_state_fn,
  max_timesteps=MAX_EPISODE_TIMESTEPS,
  human_id=None,  # will randomly shuffle human id
)

post_tutorial_stage = Stage(name="본 실험 안내", display_fn=post_tutorial_display_fn)
all_stages.append(instruction_stage)
all_stages.append(tutorial_stage)
all_stages.append(tutorial_env_stage)
all_stages.append(post_tutorial_stage)
instruction_block = Block(
  stages=[
    instruction_stage,
    tutorial_stage,
    tutorial_env_stage,
    post_tutorial_stage,
  ],
  metadata=dict(desc="Instructions"),
  randomize=False,
)
all_blocks.append(instruction_block)


for model_name, model in model_dict.items():
  environment_stage = MultiAgentEnvStage(
    name=f"{model_name}_coord_ring",
    web_env=jax_web_env,
    action_keys=action_keys,
    action_to_name=action_to_name,
    env_params=env_params,
    render_fn=render_fn,
    vmap_render_fn=vmap_render_fn,
    display_fn=env_stage_display_fn,
    evaluate_success_fn=evaluate_success_fn,
    notify_success=False,
    min_success=MIN_SUCCESS_EPISODES,
    max_episodes=MAX_STAGE_EPISODES,
    verbosity=VERBOSITY,
    user_save_file_fn=get_user_save_file_fn,
    metadata=dict(
      # nothing required, just for bookkeeping
      desc="some description",
      key1="value1",
      key2="value2",
    ),
    model=model,  # temporarily have other agent stay in place
    model_params=param_dict[model_name],
    num_seeds=num_seed_dict[model_name],
    using_param_stack=True,
    init_hidden_state_fn=init_hidden_state_fn,
    max_timesteps=MAX_EPISODE_TIMESTEPS,
    human_id=None,  # will randomly shuffle human id
  )

  transition_stage = Stage(
    name="설문 완료",
    display_fn=transition_display_fn,
  )
  survey_stage = make_survey_stage(f"{model_name} Coord Ring 설문")

  env_block = Block(
    stages=[
      environment_stage,
      survey_stage,
      # transition_stage,
    ],
    metadata=dict(desc=f"{model_name} Environment"),
    randomize=False,
  )
  all_blocks.append(env_block)


experiment = nicewebrl.Experiment(
    blocks=all_blocks,
    randomize=[False] + [True] * (len(all_blocks) - 1),
    name=f"coord_ring",
)
