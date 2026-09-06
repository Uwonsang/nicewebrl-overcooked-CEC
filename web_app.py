import aiofiles
import os.path
import asyncio
import importlib
from datetime import datetime
from nicegui import app, ui
import zipfile

# from gcs import save_to_gcs_with_retries
import nicewebrl
from nicewebrl.logging import get_logger
from nicewebrl.utils import write_msgpack_record
import sys

from session_manager import (
  create_new_session,
  load_snapshot,
  replace_current_session,
  save_current_snapshot,
)

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Experiment selection
# Comment out or add entries here to choose layouts and algorithm partners.
# Running ``python web_app.py`` uses every entry listed below.
# -----------------------------------------------------------------------------
LAYOUTS_TO_TEST = [
  # The original paper assigned one layout per participant. Keeping both here
  # runs every layout-algorithm combination in one session, as configured.
  "counter_circuit",
  "coord_ring",
]

ALGORITHMS_TO_TEST = [
  # For the paper's full seven-condition comparison, enable every entry below.
  # The current shorter selection is preserved so test length does not change
  # unexpectedly; comment/uncomment entries as needed.
  "ik",
  # "ik_finetune",
  # "sk",
  "sk_e3t",
  "sk_fcp",
  # Optional cross-layout baselines. These use SK/FCP checkpoints trained on
  # the other layout and are named coord_* or counter_* in saved records.
  # "cross_sk",
  # "cross_fcp",
]

LAYOUT_FILES = {
  "counter_circuit": "counter_circuit_experiment.py",
  "coord_ring": "coord_ring_experiment.py",
}

selected_layouts = list(LAYOUTS_TO_TEST)
invalid_layouts = [name for name in selected_layouts if name not in LAYOUT_FILES]
if invalid_layouts:
  print(f"Invalid experiment layout(s): {', '.join(invalid_layouts)}")
  print(f"Available layouts: {', '.join(LAYOUT_FILES)}")
  sys.exit(1)
if not selected_layouts:
  print("At least one layout must be selected")
  sys.exit(1)
if not ALGORITHMS_TO_TEST:
  print("At least one algorithm must be selected")
  sys.exit(1)

os.environ["NICEWEBRL_LAYOUTS"] = ",".join(selected_layouts)
os.environ["NICEWEBRL_ALGORITHMS"] = ",".join(ALGORITHMS_TO_TEST)

EXPERIMENT_CONFIG_ID = (
  "participant-session-v1|"
  f"layouts={','.join(selected_layouts)}|"
  f"algorithms={','.join(ALGORITHMS_TO_TEST)}"
)

experiment_name = (
  selected_layouts[0] if len(selected_layouts) == 1 else "combined"
)
experiment_file = (
  LAYOUT_FILES[selected_layouts[0]]
  if len(selected_layouts) == 1
  else "combined_experiment.py"
)

NAME = os.environ.get("NAME", experiment_name)
DEBUG = int(os.environ.get("DEBUG", 0))

def download_models_if_needed():
  """Download and extract models.zip if models directory doesn't exist"""
  import requests

  models_dir = "models"
  models_zip = "models.zip"

  if os.path.exists(models_dir):
    return
  else:
    print("model directory not found. end this session")
    sys.exit(1)

  # print("Models directory not found. Downloading models.zip...")

  # # Download the file
  # dropbox_url = "https://www.dropbox.com/scl/fi/a5bazgpl4hpsnz2pwmqae/models.zip?rlkey=n8fl1a4xebqc45cf97uv019oq&dl=1"

  # response = requests.get(dropbox_url, stream=True)
  # if response.status_code == 200:
  #   with open(models_zip, 'wb') as f:
  #     for chunk in response.iter_content(chunk_size=8192):
  #       f.write(chunk)
  #   print("Downloaded models.zip successfully")
  # else:
  #   raise Exception(
  #       f"Failed to download models.zip: HTTP {response.status_code}")

  # # Extract the zip file
  # with zipfile.ZipFile(models_zip, 'r') as zip_ref:
  #   zip_ref.extractall('.')

  # # Clean up the zip file
  # os.remove(models_zip)
  # print("Extracted models.zip and cleaned up")

download_models_if_needed()


async def save_data(final_save=True, feedback=None, **kwargs):
  user_data_file = nicewebrl.user_data_file()

  if final_save:
    # --------------------------------
    # save user data to final line of file
    # --------------------------------
    user_storage = nicewebrl.make_serializable(dict(app.storage.user))
    last_line = dict(
        finished=True,
        feedback=feedback,
        user_storage=user_storage,
        **kwargs,
    )
    async with aiofiles.open(user_data_file, "ab") as f:  # Changed to binary mode
      await write_msgpack_record(f, last_line)



async def make_consent_form(container):
  consent_given = asyncio.Event()
  with container:
    ui.markdown("## 연구 참여 동의")
    with open("consent.md", "r") as consent_file:
      consent_text = consent_file.read()
    ui.markdown(consent_text)
    ui.checkbox("위 내용을 확인했으며 연구 참여에 동의합니다.",
                on_change=lambda: consent_given.set())

  await consent_given.wait()


async def collect_demographic_info(container):
  # Create a markdown title for the section
  nicewebrl.clear_element(container)
  with container:
    ui.markdown("## 기본 정보")
    ui.markdown("아래 정보를 입력해 주세요.")

    with ui.column():
      with ui.column():
        ui.label("성별")
        sex_input = ui.radio(["남성", "여성"], value="남성").props("inline")

      # Collect age with a textbox input
      age_input = ui.input("나이")

    # Button to submit and store the data
    async def submit():
      age = age_input.value
      sex = sex_input.value

      # Validation for age input
      if not age.isdigit() or not (0 < int(age) < 100):
        ui.notify("1세부터 99세 사이의 나이를 입력해 주세요.", type="warning")
        return
      app.storage.user["age"] = int(age)
      app.storage.user["sex"] = sex
      logger.info(f"age: {int(age)}, sex: {sex}")

    button = ui.button("제출", on_click=submit)
    await button.clicked()


async def prepare_landing_user(request):
  """Detach the start screen from whichever participant used this browser last."""
  del request  # NiceWebRL requires this callback signature.

  # Browser storage is writable only while the initial HTTP response is being
  # built. New/resumed sessions therefore apply their DB key here after reload.
  pending_browser_session_id = app.storage.user.pop(
    "pending_browser_session_id", None
  )
  if pending_browser_session_id is not None:
    # Changing the signed browser session ID also changes which server-side
    # NiceGUI user-storage file is selected. Prepare that storage first, then
    # copy the queued participant state into it.
    source_user_storage = app.storage.user
    queued_user_storage = dict(source_user_storage)
    if pending_browser_session_id not in app.storage._users:
      await app.storage._create_user_storage(pending_browser_session_id)
    app.storage.browser["id"] = pending_browser_session_id
    app.storage.user.clear()
    app.storage.user.update(queued_user_storage)
    if source_user_storage is not app.storage.user:
      source_user_storage.clear()
    return

  # A prepared session may be reloading again before its start screen appears.
  if app.storage.user.get("participant_session_ready", False):
    return

  # Before detaching the browser, migrate its latest cookie-backed state to the
  # ID-based save. This also makes a refresh immediately resumable by ID.
  if (
    app.storage.user.get("experiment_started", False)
    or int(app.storage.user.get("stage_idx", 0) or 0) > 0
  ):
    saved_config = app.storage.user.get("experiment_config_id", "")
    await save_current_snapshot(saved_config)

  # The landing page must not display or log as the previous participant.
  # Its RNG key is already initialized, so this temporary seed can be a label.
  app.storage.user.clear()
  app.storage.user.update(
    seed="landing",
    user_id="landing",
    rng_splits=0,
    rng_key=[0, 1],
    init_rng_key=[0, 1],
    session_start=datetime.now().isoformat(),
    stage_idx=-1,
    block_idx=0,
    session_duration=0,
  )


def get_experiment():
  runner = importlib.import_module("nicewebrl.run_experiment")
  return runner.experiment_obj


async def normalize_experiment_progress():
  """Map legacy random block names onto the new stable block names."""
  experiment = get_experiment()
  await experiment.initialize()
  block_order = await experiment.get_block_order()
  if sorted(block_order) != list(range(experiment.num_blocks)):
    raise ValueError("저장된 실험 순서가 현재 실험과 맞지 않습니다.")

  remaining = max(0, int(app.storage.user.get("stage_idx", 0)))
  calculated_block_idx = 0
  for ordered_position, block_index in enumerate(block_order):
    block = experiment.blocks[block_index]
    completed_in_block = min(remaining, len(block.stages))
    existing = dict(app.storage.user.get(f"{block.name}_data", {}))
    existing["stage_idx"] = completed_in_block
    app.storage.user[f"{block.name}_data"] = existing
    if completed_in_block >= len(block.stages):
      calculated_block_idx = ordered_position + 1
    remaining = max(0, remaining - len(block.stages))

  app.storage.user["block_idx"] = calculated_block_idx
  app.storage.user["num_blocks"] = experiment.num_blocks
  app.storage.user["num_stages"] = experiment.num_stages


async def start_new_participant():
  participant_id = create_new_session()
  app.storage.user["experiment_config_id"] = EXPERIMENT_CONFIG_ID
  return participant_id


async def resume_participant(participant_id):
  snapshot = await load_snapshot(participant_id)
  saved = snapshot["user_storage"]
  experiment = get_experiment()
  saved_config = snapshot.get("config_id") or saved.get("experiment_config_id")
  if saved_config and saved_config != EXPERIMENT_CONFIG_ID:
    raise ValueError(
      "이 ID는 현재와 다른 맵/알고리즘 구성으로 진행된 기록입니다."
    )
  if not saved_config and (
    saved.get("num_blocks") != experiment.num_blocks
    or saved.get("num_stages") != experiment.num_stages
  ):
    raise ValueError("이 ID의 실험 구성이 현재 설정과 다릅니다.")
  if saved.get("experiment_finished", False):
    raise ValueError("이 ID의 실험은 이미 완료되었습니다.")

  replace_current_session(saved, snapshot["browser_session_id"])
  app.storage.user["experiment_config_id"] = EXPERIMENT_CONFIG_ID
  return app.storage.user["seed"]


async def choose_participant_session(container):
  """Wait for either a new experiment or an ID-based resume request."""
  nicewebrl.clear_element(container)
  selected = asyncio.get_running_loop().create_future()

  async def new_game():
    try:
      await start_new_participant()
      ui.navigate.to("/?session_reload=new")
    except Exception as exc:
      logger.exception("Failed to create participant session")
      ui.notify(f"새 실험을 시작하지 못했습니다: {exc}", type="negative")

  async def resume_game():
    try:
      await resume_participant(resume_id.value)
      ui.navigate.to("/?session_reload=resume")
    except ValueError as exc:
      ui.notify(str(exc), type="warning")
    except Exception as exc:
      logger.exception("Failed to restore participant session")
      ui.notify(f"저장 기록을 불러오지 못했습니다: {exc}", type="negative")

  prepared_mode = app.storage.user.pop("participant_session_ready", None)
  if prepared_mode:
    await normalize_experiment_progress()
    await save_current_snapshot(EXPERIMENT_CONFIG_ID)
    selected.set_result((prepared_mode, app.storage.user["seed"]))
  else:
    with container.style("align-items: center;"):
      ui.markdown("# Overcooked 협력 실험")
      ui.markdown("새 참가자는 새 실험을, 진행 중이던 참가자는 발급받은 ID로 이어하기를 선택해 주세요.")
      ui.button("새 실험 시작", on_click=new_game).props("color=primary")
      ui.separator()
      resume_id = ui.input("참가자 ID", placeholder="예: 3509644287")
      ui.button("ID로 이어하기", on_click=resume_game)

  mode, participant_id = await selected
  nicewebrl.clear_element(container)
  if mode == "new":
    confirmed = asyncio.Event()
    with container.style("align-items: center;"):
      ui.markdown("## 참가자 ID가 발급되었습니다")
      ui.label(str(participant_id)).classes("text-3xl font-bold")
      ui.markdown("이어하기에 필요하므로 ID를 따로 기록해 주세요.")
      ui.button("ID를 기록했습니다", on_click=confirmed.set)
    await confirmed.wait()
  else:
    ui.notify(f"ID {participant_id}의 진행 상태를 불러왔습니다.", type="positive")
  nicewebrl.clear_element(container)


async def on_startup(stage_container):
  """Called when experiment starts - UI is available"""
  await choose_participant_session(stage_container)

  # Browser cookies are no longer the only progress store. Keep a current
  # server-side snapshot so the participant ID is sufficient for resuming.
  async def snapshot_progress():
    await save_current_snapshot(EXPERIMENT_CONFIG_ID)

  ui.timer(1.0, snapshot_progress)

  if not app.storage.user.get("experiment_started", False):
    await make_consent_form(stage_container)
    await collect_demographic_info(stage_container)
    app.storage.user["experiment_started"] = True
    await save_current_snapshot(EXPERIMENT_CONFIG_ID)

async def finish_experiment(meta_container, stage_container):
  nicewebrl.clear_element(meta_container)
  nicewebrl.clear_element(stage_container)
  logger.info("Finishing experiment")
  experiment_finished = app.storage.user.get("experiment_finished", False)

  if experiment_finished and not DEBUG:
    # in case called multiple times
    return

  #########################
  # Save data
  #########################
  async def submit(feedback):
    app.storage.user["experiment_finished"] = True
    with meta_container:
      nicewebrl.clear_element(meta_container)
      ui.markdown("## 데이터를 저장하고 있습니다. 잠시만 기다려 주세요.")
      ui.markdown(
          "**데이터 저장이 완료되면 자동으로 다음 화면으로 이동합니다.**"
      )

    # when over, delete user data.
    await save_data(final_save=True, feedback=feedback)
    app.storage.user["data_saved"] = True
    print("data saved")

  app.storage.user["data_saved"] = app.storage.user.get("data_saved", False)
  if not app.storage.user["data_saved"]:
    with meta_container:
      nicewebrl.clear_element(meta_container)
      ui.markdown(
          "실험에 대한 의견을 자유롭게 적어 주세요. 문제가 있었거나 개선할 점이 있다면 함께 알려 주세요."
      )
      text = ui.textarea().style("width: 80%;")  # Set width to 80% of the container
      button = ui.button("제출")
      await button.clicked()
      await submit(text.value)

  #########################
  # Final screen
  #########################
  async def next_participant():
    await start_new_participant()
    ui.navigate.to("/?session_reload=next")

  with meta_container:
    nicewebrl.clear_element(meta_container)

    ui.markdown("# 실험이 종료되었습니다")
    ui.markdown("## 데이터가 저장되었습니다")
    ui.markdown(
        "### 보상 지급에 필요한 아래 코드를 기록해 주세요."
    )
    ui.markdown(f"### socialrl.cook")
    ui.markdown("#### 이제 브라우저를 닫아도 됩니다.")
    ui.button("다음 참가자 시작", on_click=next_participant)
    await save_current_snapshot(EXPERIMENT_CONFIG_ID)


nicewebrl.run(
    storage_secret="a_very_secret_key_for_testing_only_12345",
    experiment_file=experiment_file,
    port=int(os.environ.get("PORT", 8080)),
    title="Overcooked 인간-AI 협력 실험",
    reload=False,
    init_user_fn=prepare_landing_user,
    on_startup_fn=on_startup,
    on_termination_fn=finish_experiment,
)
