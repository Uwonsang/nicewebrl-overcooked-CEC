"""Save and restore NiceWebRL progress using a participant ID."""

from __future__ import annotations

import json
import os
import asyncio
from datetime import datetime
from pathlib import Path
import random
import re
import uuid

import aiofiles
from nicegui import app

import nicewebrl


SESSION_DIR = Path("data/session_progress")
LEGACY_STORAGE_DIR = Path(".nicegui")
PARTICIPANT_ID_RE = re.compile(r"^[0-9]{1,10}$")
LEGACY_STORAGE_RE = re.compile(r"storage-user-(.+)\.json$")
_snapshot_locks: dict[int, asyncio.Lock] = {}


def _session_path(participant_id: int) -> Path:
  return SESSION_DIR / f"{participant_id}.json"


def _validate_participant_id(value: str | int) -> int:
  text = str(value).strip()
  if not PARTICIPANT_ID_RE.fullmatch(text):
    raise ValueError("참가자 ID는 숫자로 입력해 주세요.")
  return int(text)


def _legacy_snapshot(participant_id: int) -> dict | None:
  """Read an old cookie-backed save so IDs created before this change work."""
  matches = []
  if not LEGACY_STORAGE_DIR.exists():
    return None

  for path in LEGACY_STORAGE_DIR.glob("storage-user-*.json"):
    try:
      user_storage = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      continue
    if user_storage.get("seed") != participant_id:
      continue
    match = LEGACY_STORAGE_RE.fullmatch(path.name)
    if match:
      matches.append((path.stat().st_mtime, match.group(1), user_storage))

  if not matches:
    return None
  _, browser_session_id, user_storage = max(matches, key=lambda item: item[0])
  return {
    "participant_id": participant_id,
    "browser_session_id": browser_session_id,
    "user_storage": user_storage,
    "legacy": True,
  }


async def load_snapshot(participant_id: str | int) -> dict:
  participant_id = _validate_participant_id(participant_id)
  path = _session_path(participant_id)
  if path.exists():
    try:
      async with aiofiles.open(path, "r", encoding="utf-8") as f:
        snapshot = json.loads(await f.read())
    except (OSError, json.JSONDecodeError) as exc:
      raise ValueError("저장된 진행 정보를 읽을 수 없습니다.") from exc
  else:
    snapshot = _legacy_snapshot(participant_id)

  if snapshot is None:
    raise ValueError("해당 참가자 ID의 저장 기록이 없습니다.")
  if snapshot.get("user_storage", {}).get("seed") != participant_id:
    raise ValueError("참가자 ID와 저장 기록이 일치하지 않습니다.")
  return snapshot


async def save_current_snapshot(config_id: str) -> None:
  participant_id = app.storage.user.get("seed")
  browser_session_id = app.storage.browser.get("id")
  if participant_id is None or browser_session_id is None:
    return

  lock = _snapshot_locks.setdefault(participant_id, asyncio.Lock())
  async with lock:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = {
      "participant_id": participant_id,
      "browser_session_id": browser_session_id,
      "config_id": config_id,
      "saved_at": datetime.now().isoformat(),
      "user_storage": nicewebrl.make_serializable(dict(app.storage.user)),
    }
    path = _session_path(participant_id)
    temporary_path = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    async with aiofiles.open(temporary_path, "w", encoding="utf-8") as f:
      await f.write(json.dumps(snapshot, ensure_ascii=False))
    os.replace(temporary_path, path)


def _participant_id_exists(participant_id: int) -> bool:
  if _session_path(participant_id).exists():
    return True
  if _legacy_snapshot(participant_id) is not None:
    return True
  return any(Path("data").glob(f"user={participant_id}_name=*_debug=*.json"))


def new_participant_id() -> int:
  while True:
    # NiceWebRL uses this ID as a JAX uint32 random seed.
    participant_id = random.SystemRandom().randrange(1_000_000_000, 4_294_967_296)
    if not _participant_id_exists(participant_id):
      return participant_id


def replace_current_session(user_storage: dict, browser_session_id: str) -> None:
  """Queue a restored session for application during the next page request."""
  app.storage.user.clear()
  app.storage.user.update(user_storage)
  app.storage.user["user_id"] = app.storage.user["seed"]
  app.storage.user["pending_browser_session_id"] = browser_session_id
  app.storage.user["participant_session_ready"] = "resume"


def create_new_session() -> int:
  """Queue a fresh session for application during the next page request."""
  participant_id = new_participant_id()
  app.storage.user.clear()
  nicewebrl.initialize_user(seed=participant_id)
  app.storage.user["user_id"] = participant_id
  app.storage.user["pending_browser_session_id"] = str(uuid.uuid4())
  app.storage.user["participant_session_ready"] = "new"
  app.storage.user["stage_idx"] = 0
  app.storage.user["block_idx"] = 0
  app.storage.user["experiment_started"] = False
  app.storage.user["experiment_finished"] = False
  app.storage.user["data_saved"] = False
  return participant_id
