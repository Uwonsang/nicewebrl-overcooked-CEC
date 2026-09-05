import aiofiles
import os.path
import asyncio
from nicegui import app, ui
import zipfile

# from gcs import save_to_gcs_with_retries
import nicewebrl
from nicewebrl.logging import get_logger
from nicewebrl.utils import write_msgpack_record
import sys

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


async def on_startup(stage_container):
  """Called when experiment starts - UI is available"""
  if not app.storage.user.get("experiment_started", False):
    await make_consent_form(stage_container)
    await collect_demographic_info(stage_container)
    app.storage.user["experiment_started"] = True

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
  with meta_container:
    nicewebrl.clear_element(meta_container)

    ui.markdown("# 실험이 종료되었습니다")
    ui.markdown("## 데이터가 저장되었습니다")
    ui.markdown(
        "### 보상 지급에 필요한 아래 코드를 기록해 주세요."
    )
    ui.markdown(f"### socialrl.cook")
    ui.markdown("#### 이제 브라우저를 닫아도 됩니다.")


nicewebrl.run(
    storage_secret="a_very_secret_key_for_testing_only_12345",
    experiment_file=experiment_file,
    title="Overcooked 인간-AI 협력 실험",
    reload=False,
    on_startup_fn=on_startup,
    on_termination_fn=finish_experiment,
)
