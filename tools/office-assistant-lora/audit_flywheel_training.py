from __future__ import annotations

import json
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path


database = Path("/data/office.db")
root = Path("/data")
connection = sqlite3.connect(database)
connection.row_factory = sqlite3.Row
rows = connection.execute(
    "SELECT c.*, l.label, l.subtype, COALESCE(p.name, CASE WHEN c.person_id IS NULL "
    "THEN '未识别人物' ELSE '人员 ' || c.person_id END) person_name "
    "FROM flywheel_candidates c JOIN flywheel_labels l ON l.candidate_id=c.id "
    "LEFT JOIN people p ON p.id=c.person_id ORDER BY c.started_at"
).fetchall()
connection.close()


def duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, check=True, text=True, timeout=10,
        )
        return round(float(result.stdout.strip()), 2)
    except Exception:
        return None


positive = [row for row in rows if row["label"] == "positive"]
negative = [row for row in rows if row["label"] == "negative"]
trimmed = [row for row in positive if row["training_clip_path"]]
trim_durations = [
    value for row in trimmed
    if (value := duration(root / row["training_clip_path"])) is not None
]
positive_times = sorted(float(row["started_at"]) for row in positive)
near_duplicates = sum(
    1 for left, right in zip(positive_times, positive_times[1:]) if right - left < 20
)
report = {
    "labeled_total": len(rows),
    "positive": len(positive),
    "negative": len(negative),
    "uncertain": sum(row["label"] == "uncertain" for row in rows),
    "positive_trimmed": len(trimmed),
    "trim_files_present": sum((root / row["training_clip_path"]).is_file() for row in trimmed),
    "trim_duration_min": min(trim_durations, default=None),
    "trim_duration_max": max(trim_durations, default=None),
    "trim_duration_avg": round(sum(trim_durations) / len(trim_durations), 2) if trim_durations else None,
    "positive_people": Counter(row["person_name"] for row in positive),
    "positive_person_ids": Counter(str(row["person_id"]) for row in positive),
    "positive_sensors": Counter(row["sensor_id"] for row in positive),
    "positive_rules": Counter(row["rule_version"] for row in positive),
    "positive_unique_tracks": len({row["track_id"] for row in positive}),
    "positive_object_confirmed": sum(
        bool(json.loads(row["object_json"] or "{}").get("confirmed")) for row in positive
    ),
    "negative_subtypes": Counter((row["subtype"] or "未分类") for row in negative),
    "positive_pairs_within_20_seconds": near_duplicates,
}
print(json.dumps(report, ensure_ascii=False, indent=2))
