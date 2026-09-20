import re
from pathlib import Path
from typing import Counter

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "layout_log.txt"

# 결과 파일명: 원본파일명_Analyzed.txt
OUTPUT_PATH = LOG_PATH.with_name(f"{LOG_PATH.stem}_Analyzed.txt")

pattern = re.compile(
    r"\[layout check\]\s+"
    r"pot=(\d+),\s+"
    r"onion=(\d+),\s+"
    r"plate=(\d+),\s+"
    r"goal=(\d+),\s+"
    r"count_ok=(True|False),\s+"
    r"onion_reachable=(True|False),\s+"
    r"plate_reachable=(True|False),\s+"
    r"pot_reachable=(True|False),\s+"
    r"goal_reachable=(True|False),\s+"
    r"reachable_ok=(True|False),\s+"
    r"valid=(True|False)"
)

total_count = 0
valid_count = 0
invalid_count = 0

records = []
invalid_reason_counter = Counter()

def to_bool(s: str) -> bool:
    return s == "True"

with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if not match:
            continue

        pot = int(match.group(1))
        onion = int(match.group(2))
        plate = int(match.group(3))
        goal = int(match.group(4))

        count_ok = to_bool(match.group(5))
        onion_reachable = to_bool(match.group(6))
        plate_reachable = to_bool(match.group(7))
        pot_reachable = to_bool(match.group(8))
        goal_reachable = to_bool(match.group(9))
        reachable_ok = to_bool(match.group(10))
        valid = to_bool(match.group(11))

        total_count += 1
        if valid:
            valid_count += 1
        else:
            invalid_count += 1

        record = {
            "pot": pot,
            "onion": onion,
            "plate": plate,
            "goal": goal,
            "count_ok": count_ok,
            "onion_reachable": onion_reachable,
            "plate_reachable": plate_reachable,
            "pot_reachable": pot_reachable,
            "goal_reachable": goal_reachable,
            "reachable_ok": reachable_ok,
            "valid": valid,
        }
        records.append(record)

        if not valid:
            reasons = []

            # count 실패 원인
            if pot <= 0:
                reasons.append("missing_pot")
            if onion <= 0:
                reasons.append("missing_onion")
            if plate <= 0:
                reasons.append("missing_plate")
            if goal <= 0:
                reasons.append("missing_goal")

            # reachable 실패 원인
            if not onion_reachable:
                reasons.append("onion_unreachable")
            if not plate_reachable:
                reasons.append("plate_unreachable")
            if not pot_reachable:
                reasons.append("pot_unreachable")
            if not goal_reachable:
                reasons.append("goal_unreachable")

            if not reasons:
                reasons.append("unknown_invalid_reason")

            for reason in reasons:
                invalid_reason_counter[reason] += 1

yield_rate = valid_count / total_count if total_count > 0 else 0.0

result_lines = []
result_lines.append("===== Yield Summary =====")
result_lines.append(f"source_file   : {LOG_PATH.name}")
result_lines.append(f"total_count   : {total_count}")
result_lines.append(f"valid_count   : {valid_count}")
result_lines.append(f"invalid_count : {invalid_count}")
result_lines.append(f"yield_rate    : {yield_rate:.4f}")
result_lines.append("")

result_lines.append("===== Invalid Reason Summary =====")
if invalid_reason_counter:
    for reason, count in invalid_reason_counter.most_common():
        result_lines.append(f"{reason:20s} : {count}")
else:
    result_lines.append("No invalid samples")
result_lines.append("")

result_lines.append("===== Invalid Samples =====")
for i, r in enumerate(records):
    if not r["valid"]:
        reasons = []
        if r["pot"] <= 0:
            reasons.append("missing_pot")
        if r["onion"] <= 0:
            reasons.append("missing_onion")
        if r["plate"] <= 0:
            reasons.append("missing_plate")
        if r["goal"] <= 0:
            reasons.append("missing_goal")
        if not r["onion_reachable"]:
            reasons.append("onion_unreachable")
        if not r["plate_reachable"]:
            reasons.append("plate_unreachable")
        if not r["pot_reachable"]:
            reasons.append("pot_unreachable")
        if not r["goal_reachable"]:
            reasons.append("goal_unreachable")

        reason_str = ", ".join(reasons) if reasons else "unknown_invalid_reason"

        result_lines.append(
            f"[{i}] "
            f"pot={r['pot']}, onion={r['onion']}, plate={r['plate']}, goal={r['goal']}, "
            f"count_ok={r['count_ok']}, "
            f"onion_reachable={r['onion_reachable']}, "
            f"plate_reachable={r['plate_reachable']}, "
            f"pot_reachable={r['pot_reachable']}, "
            f"goal_reachable={r['goal_reachable']}, "
            f"reachable_ok={r['reachable_ok']}, "
            f"valid={r['valid']}, "
            f"reasons=[{reason_str}]"
        )

result_text = "\n".join(result_lines)

# 터미널에도 출력
print(result_text)

# 분석 결과 파일로 저장
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(result_text)

print(f"\nSaved analyzed result to: {OUTPUT_PATH}")