"""Create per-user, per-map, per-algorithm, and total experiment reports.

The experiment files use a .json suffix, but contain length-prefixed MessagePack
records. Run this script from the repository root:

    uv run python analysis_extend.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault(
  "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "overcooked-matplotlib-cache")
)

import msgpack
import matplotlib
import numpy as np
from flax.serialization import msgpack_restore

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DELIVERY_REWARD = 20.0
RATING_VALUES = {
  "Strongly disagree": 1,
  "Disagree": 2,
  "Neutral": 3,
  "Agree": 4,
  "Strongly agree": 5,
  "Very poor": 1,
  "Poor": 2,
  "Good": 4,
  "Very good": 5,
  "전혀 그렇지 않다": 1,
  "그렇지 않다": 2,
  "보통이다": 3,
  "보통이었다": 3,
  "그렇다": 4,
  "좋았다": 4,
  "매우 그렇다": 5,
  "매우 좋았다": 5,
  "매우 부족했다": 1,
  "부족했다": 2,
}
NEGATIVE_QUESTIONS = {
  "The agent frequently got in my way.",
  "The agent's behavior was frustrating.",
}

USER_FILE_RE = re.compile(
  r"^user=(?P<user_id>[^_]+)_name=(?P<map_name>.+)_debug=(?P<debug>\d+)\.json$"
)


def read_records(path: Path) -> list[dict[str, Any]]:
  """Read length-prefixed MessagePack records, tolerating a truncated tail."""
  records = []
  with path.open("rb") as stream:
    while True:
      length_bytes = stream.read(4)
      if not length_bytes:
        break
      if len(length_bytes) != 4:
        print(f"경고: {path} 마지막 레코드의 길이 정보가 손상되었습니다.")
        break
      length = int.from_bytes(length_bytes, byteorder="big")
      payload = stream.read(length)
      if len(payload) != length:
        print(f"경고: {path} 마지막 레코드가 완전히 저장되지 않았습니다.")
        break
      try:
        record = msgpack.unpackb(payload, strict_map_key=False)
      except Exception as exc:
        print(f"경고: {path} 레코드를 읽지 못했습니다: {exc}")
        continue
      if isinstance(record, dict):
        records.append(record)
  return records


def json_ready(value: Any) -> Any:
  if isinstance(value, np.ndarray):
    return value.item() if value.ndim == 0 else value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, bytes):
    return f"<binary: {len(value)} bytes>"
  if isinstance(value, dict):
    return {str(key): json_ready(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [json_ready(item) for item in value]
  return value


def timestep_reward(record: dict[str, Any]) -> float:
  timestep_blob = record.get("data", {}).get("timestep")
  if not isinstance(timestep_blob, bytes):
    return 0.0
  try:
    timestep = msgpack_restore(timestep_blob)
    return float(np.asarray(timestep.get("reward", 0.0)).sum())
  except Exception as exc:
    print(f"경고: timestep reward를 복원하지 못했습니다: {exc}")
    return 0.0


def algorithm_from_stage(stage_name: str, map_name: str) -> str | None:
  suffix = f"_{map_name}"
  if stage_name == "tutorial":
    return None
  if stage_name.endswith(suffix):
    return stage_name[: -len(suffix)]
  return None


def algorithm_from_survey(stage_name: str, map_name: str) -> str | None:
  labels = {
    "counter_circuit": "Counter Circuit",
    "coord_ring": "Coord Ring",
  }
  label = labels.get(map_name, map_name.replace("_", " ").title())
  marker = f" {label} "
  if marker in stage_name:
    return stage_name.split(marker, 1)[0]
  return None


def numeric_survey(data: dict[str, Any]) -> dict[str, int]:
  return {
    question: RATING_VALUES[answer]
    for question, answer in data.items()
    if question != "prolific_id" and answer in RATING_VALUES
  }


def scored_survey(data: dict[str, Any]) -> dict[str, int]:
  """Return 1-5 cooperation scores, reverse-scoring negative questions."""
  raw_scores = numeric_survey(data)
  return {
    question: 6 - score if question in NEGATIVE_QUESTIONS else score
    for question, score in raw_scores.items()
  }


def analyze_user_file(path: Path) -> tuple[str, str, dict[str, Any]] | None:
  match = USER_FILE_RE.match(path.name)
  if not match:
    return None
  user_id = match.group("user_id")
  map_name = match.group("map_name")
  records = read_records(path)
  demographics: dict[str, Any] = {}
  plays: dict[str, dict[str, Any]] = {}
  surveys: dict[str, dict[str, Any]] = {}

  env_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for record in records:
    if record.get("user_data"):
      demographics.update(json_ready(record["user_data"]))
    record_type = record.get("metadata", {}).get("type")
    stage_name = str(record.get("name", ""))
    if record_type == "EnvStage":
      algorithm = algorithm_from_stage(stage_name, map_name)
      if algorithm:
        env_records[algorithm].append(record)
    elif record_type == "FeedbackStage":
      algorithm = algorithm_from_survey(stage_name, map_name)
      if algorithm:
        raw = json_ready(record.get("data", {}))
        scores = numeric_survey(raw)
        scored = scored_survey(raw)
        surveys[algorithm] = {
          "responses": raw,
          "numeric_scores": scores,
          "scored_numeric_scores": scored,
          "mean_rating": round(float(np.mean(list(scored.values()))), 4)
          if scored
          else None,
        }

  for algorithm, items in sorted(env_records.items()):
    total_reward = sum(timestep_reward(item) for item in items)
    latest_metadata = max(
      (item.get("metadata", {}) for item in items),
      key=lambda metadata: int(metadata.get("nsteps", 0)),
    )
    successful_deliveries = total_reward / DELIVERY_REWARD
    completed = any(
      item.get("data", {}).get("computer_interaction") == "timer"
      for item in items
    )
    plays[algorithm] = {
      "total_reward": round(total_reward, 4),
      "successful_deliveries": round(successful_deliveries, 4),
      "success": total_reward > 0,
      "success_rate": 1.0 if total_reward > 0 else 0.0,
      "completed": completed,
      "nsteps": int(latest_metadata.get("nsteps", 0)),
      "nepisodes": int(latest_metadata.get("nepisodes", 0)),
      "record_count": len(items),
      "collisions": None,
      "collision_status": "not_recorded",
      "survey": surveys.get(algorithm),
    }

  return user_id, map_name, {
    "source_file": str(path),
    "debug": int(match.group("debug")),
    "demographics": demographics,
    "algorithms": plays,
  }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
  if not runs:
    return {}
  rewards = [float(run["total_reward"]) for run in runs]
  deliveries = [float(run["successful_deliveries"]) for run in runs]
  successes = [bool(run["success"]) for run in runs]
  completed_runs = [run for run in runs if run.get("completed")]
  completed_successes = [bool(run["success"]) for run in completed_runs]
  survey_values: dict[str, list[float]] = defaultdict(list)
  scored_survey_values: dict[str, list[float]] = defaultdict(list)
  survey_means = []
  for run in runs:
    survey = run.get("survey") or {}
    for question, score in survey.get("numeric_scores", {}).items():
      survey_values[question].append(float(score))
    for question, score in survey.get("scored_numeric_scores", {}).items():
      scored_survey_values[question].append(float(score))
    if survey.get("mean_rating") is not None:
      survey_means.append(float(survey["mean_rating"]))

  return {
    "runs": len(runs),
    "completed_runs": len(completed_runs),
    "incomplete_runs": len(runs) - len(completed_runs),
    "successful_completed_runs": sum(completed_successes),
    "success_rate": round(sum(completed_successes) / len(completed_runs), 4)
    if completed_runs
    else None,
    "reward_positive_runs_including_incomplete": sum(successes),
    "total_reward": round(sum(rewards), 4),
    "mean_reward": round(float(np.mean(rewards)), 4),
    "total_successful_deliveries": round(sum(deliveries), 4),
    "mean_successful_deliveries": round(float(np.mean(deliveries)), 4),
    "collisions": None,
    "collision_status": "not_recorded",
    "mean_qualitative_rating": round(float(np.mean(survey_means)), 4)
    if survey_means
    else None,
    "survey_question_means": {
      question: round(float(np.mean(values)), 4)
      for question, values in sorted(survey_values.items())
    },
    "survey_scored_question_means": {
      question: round(float(np.mean(values)), 4)
      for question, values in sorted(scored_survey_values.items())
    },
  }


def write_json(path: Path, data: Any) -> None:
  path.write_text(
    json.dumps(json_ready(data), ensure_ascii=False, indent=2), encoding="utf-8"
  )


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  fieldnames = [
    "user_id",
    "map",
    "algorithm",
    "total_reward",
    "successful_deliveries",
    "success",
    "success_rate",
    "completed",
    "nsteps",
    "nepisodes",
    "mean_qualitative_rating",
    "collisions",
    "collision_status",
  ]
  with path.open("w", newline="", encoding="utf-8-sig") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def _draw_bars(
  axis,
  labels: list[str],
  values: list[float | None],
  title: str,
  ylabel: str,
  ylim: tuple[float, float] | None = None,
) -> None:
  numeric = [np.nan if value is None else float(value) for value in values]
  bars = axis.bar(range(len(labels)), numeric, color="#4C78A8")
  axis.set_title(title)
  axis.set_ylabel(ylabel)
  axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
  axis.grid(axis="y", alpha=0.25)
  if ylim:
    axis.set_ylim(*ylim)
  for bar, value in zip(bars, numeric):
    if not np.isnan(value):
      axis.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
      )


def plot_user(path: Path, user_id: str, rows: list[dict[str, Any]]) -> None:
  if not rows:
    return
  labels = [f"{row['map']}\n{row['algorithm']}" for row in rows]
  figure, axes = plt.subplots(1, 3, figsize=(max(12, len(rows) * 1.8), 4.8))
  _draw_bars(
    axes[0], labels, [row["total_reward"] for row in rows], "Total reward", "Reward"
  )
  _draw_bars(
    axes[1],
    labels,
    [row["successful_deliveries"] for row in rows],
    "Successful deliveries",
    "Deliveries",
  )
  _draw_bars(
    axes[2],
    labels,
    [row.get("mean_qualitative_rating") for row in rows],
    "Qualitative rating",
    "Rating (1-5)",
    (0, 5.5),
  )
  figure.suptitle(f"User {user_id}")
  figure.tight_layout()
  figure.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(figure)


def plot_group(path: Path, title: str, groups: dict[str, dict[str, Any]]) -> None:
  if not groups:
    return
  labels = list(groups)
  summaries = [groups[label] for label in labels]
  figure, axes = plt.subplots(1, 3, figsize=(max(12, len(labels) * 1.8), 4.8))
  _draw_bars(
    axes[0],
    labels,
    [summary.get("success_rate") for summary in summaries],
    "Success rate (completed runs)",
    "Rate",
    (0, 1.1),
  )
  _draw_bars(
    axes[1],
    labels,
    [summary.get("mean_reward") for summary in summaries],
    "Mean reward",
    "Reward",
  )
  _draw_bars(
    axes[2],
    labels,
    [summary.get("mean_qualitative_rating") for summary in summaries],
    "Mean qualitative rating",
    "Rating (1-5)",
    (0, 5.5),
  )
  figure.suptitle(title)
  figure.tight_layout()
  figure.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(figure)


def plot_total(
  path: Path,
  map_results: dict[str, dict[str, Any]],
  algorithm_results: dict[str, dict[str, Any]],
) -> None:
  map_groups = {name: result["overall"] for name, result in map_results.items()}
  algorithm_groups = {
    name: result["overall"] for name, result in algorithm_results.items()
  }
  figure, axes = plt.subplots(2, 3, figsize=(max(14, len(algorithm_groups) * 1.5), 9))
  specifications = [
    ("success_rate", "Success rate", "Rate", (0, 1.1)),
    ("mean_reward", "Mean reward", "Reward", None),
    ("mean_qualitative_rating", "Qualitative rating", "Rating (1-5)", (0, 5.5)),
  ]
  for column, (metric, title, ylabel, ylim) in enumerate(specifications):
    _draw_bars(
      axes[0, column],
      list(map_groups),
      [summary.get(metric) for summary in map_groups.values()],
      f"By map: {title}",
      ylabel,
      ylim,
    )
    _draw_bars(
      axes[1, column],
      list(algorithm_groups),
      [summary.get(metric) for summary in algorithm_groups.values()],
      f"By algorithm: {title}",
      ylabel,
      ylim,
    )
  figure.suptitle("Total experiment results")
  figure.tight_layout()
  figure.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(figure)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data-dir", type=Path, default=Path("data"))
  parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
  args = parser.parse_args()

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  per_user_dir = args.output_dir / "result_per_user"
  per_map_dir = args.output_dir / "result_per_map"
  per_algorithm_dir = args.output_dir / "result_per_algorithm"
  for directory in (per_user_dir, per_map_dir, per_algorithm_dir):
    directory.mkdir(parents=True, exist_ok=True)

  users: dict[str, dict[str, Any]] = defaultdict(
    lambda: {"demographics": {}, "maps": {}}
  )
  rows: list[dict[str, Any]] = []
  map_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
  algorithm_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
  map_algorithm_runs: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
    lambda: defaultdict(list)
  )
  algorithm_map_runs: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
    lambda: defaultdict(list)
  )

  for path in sorted(args.data_dir.glob("user=*_name=*_debug=*.json")):
    analyzed = analyze_user_file(path)
    if analyzed is None:
      continue
    user_id, map_name, result = analyzed
    users[user_id]["demographics"].update(result["demographics"])
    users[user_id]["maps"][map_name] = result
    for algorithm, run in result["algorithms"].items():
      row = {
        "user_id": user_id,
        "map": map_name,
        "algorithm": algorithm,
        **{key: value for key, value in run.items() if key != "survey"},
        "mean_qualitative_rating": (run.get("survey") or {}).get("mean_rating"),
      }
      rows.append(row)
      map_runs[map_name].append(run)
      algorithm_runs[algorithm].append(run)
      map_algorithm_runs[map_name][algorithm].append(run)
      algorithm_map_runs[algorithm][map_name].append(run)

  for user_id, result in sorted(users.items()):
    user_rows = [row for row in rows if row["user_id"] == user_id]
    write_json(per_user_dir / f"user_{user_id}_{timestamp}.json", result)
    write_summary_csv(per_user_dir / f"user_{user_id}_{timestamp}.csv", user_rows)
    plot_user(per_user_dir / f"user_{user_id}_{timestamp}.png", user_id, user_rows)

  map_results = {}
  for map_name, runs in sorted(map_runs.items()):
    result = {
      "map": map_name,
      "overall": aggregate_runs(runs),
      "algorithms": {
        algorithm: aggregate_runs(items)
        for algorithm, items in sorted(map_algorithm_runs[map_name].items())
      },
    }
    map_results[map_name] = result
    write_json(per_map_dir / f"{map_name}_{timestamp}.json", result)
    write_summary_csv(
      per_map_dir / f"{map_name}_{timestamp}.csv",
      [row for row in rows if row["map"] == map_name],
    )
    plot_group(
      per_map_dir / f"{map_name}_{timestamp}.png",
      f"Map: {map_name}",
      result["algorithms"],
    )

  algorithm_results = {}
  for algorithm, runs in sorted(algorithm_runs.items()):
    result = {
      "algorithm": algorithm,
      "overall": aggregate_runs(runs),
      "maps": {
        map_name: aggregate_runs(items)
        for map_name, items in sorted(algorithm_map_runs[algorithm].items())
      },
    }
    algorithm_results[algorithm] = result
    write_json(per_algorithm_dir / f"{algorithm}_{timestamp}.json", result)
    write_summary_csv(
      per_algorithm_dir / f"{algorithm}_{timestamp}.csv",
      [row for row in rows if row["algorithm"] == algorithm],
    )
    plot_group(
      per_algorithm_dir / f"{algorithm}_{timestamp}.png",
      f"Algorithm: {algorithm}",
      result["maps"],
    )

  total_result = {
    "generated_at": datetime.now().astimezone().isoformat(),
    "source_data_dir": str(args.data_dir),
    "user_count": len(users),
    "run_count": len(rows),
    "metric_definitions": {
      "success": "한 알고리즘-맵 실행의 total_reward가 0보다 크면 성공",
      "success_rate": "successful_completed_runs / completed_runs; 중단된 실행은 분모에서 제외",
      "successful_deliveries": f"total_reward / {DELIVERY_REWARD:g}",
      "collisions": "현재 실험 데이터에 명시적으로 저장되지 않아 null",
      "qualitative_rating": "설문 선택지를 1~5로 변환하고 부정 문항을 역채점한 평균",
    },
    "overall": aggregate_runs(list(run for runs in map_runs.values() for run in runs)),
    "maps": map_results,
    "algorithms": algorithm_results,
  }
  write_json(args.output_dir / f"Total_result_{timestamp}.json", total_result)
  write_summary_csv(args.output_dir / f"Total_result_{timestamp}.csv", rows)
  plot_total(
    args.output_dir / f"Total_result_{timestamp}.png",
    map_results,
    algorithm_results,
  )

  print(f"분석 완료: {len(users)}명, {len(rows)}개 알고리즘-맵 실행")
  print(f"결과 위치: {args.output_dir.resolve()}")


if __name__ == "__main__":
  main()
