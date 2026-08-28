# SPDX-License-Identifier: Apache-2.0
"""Durable storage and HTTP-facing helpers for the office data flywheel."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import subprocess
import time
from contextlib import closing
from pathlib import Path
from typing import Any


LABELS = {"positive", "negative", "uncertain"}


def initialize_flywheel_schema(database_file: Path) -> None:
    with closing(sqlite3.connect(database_file, timeout=30)) as connection, connection:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS flywheel_candidates ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id TEXT NOT NULL UNIQUE, "
            "sensor_id TEXT NOT NULL, track_id TEXT NOT NULL, person_id INTEGER, "
            "activity TEXT NOT NULL, started_at REAL NOT NULL, ended_at REAL NOT NULL, "
            "rule_version TEXT NOT NULL, score REAL NOT NULL, trigger_json TEXT NOT NULL, "
            "bbox_json TEXT NOT NULL, clip_path TEXT NOT NULL DEFAULT '', "
            "object_json TEXT NOT NULL DEFAULT '{}', object_status TEXT NOT NULL DEFAULT 'pending', "
            "object_error TEXT NOT NULL DEFAULT '', "
            "status TEXT NOT NULL DEFAULT 'pending_clip', clip_error TEXT NOT NULL DEFAULT '', "
            "created_at REAL NOT NULL, updated_at REAL NOT NULL)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_flywheel_candidate_status_time "
            "ON flywheel_candidates(status, started_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_flywheel_candidate_person_time "
            "ON flywheel_candidates(person_id, started_at DESC)"
        )
        existing = {row[1] for row in connection.execute("PRAGMA table_info(flywheel_candidates)")}
        for name, definition in {
            "object_json": "TEXT NOT NULL DEFAULT '{}'",
            "object_status": "TEXT NOT NULL DEFAULT 'pending'",
            "object_error": "TEXT NOT NULL DEFAULT ''",
            "training_clip_path": "TEXT NOT NULL DEFAULT ''",
            "trim_start": "REAL",
            "trim_end": "REAL",
            "trim_updated_at": "REAL",
        }.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE flywheel_candidates ADD COLUMN {name} {definition}")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS flywheel_labels ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id INTEGER NOT NULL UNIQUE, "
            "label TEXT NOT NULL, subtype TEXT NOT NULL DEFAULT '', note TEXT NOT NULL DEFAULT '', "
            "annotator TEXT NOT NULL DEFAULT 'local-reviewer', created_at REAL NOT NULL, "
            "updated_at REAL NOT NULL, FOREIGN KEY(candidate_id) REFERENCES flywheel_candidates(id))"
        )


class FlywheelStore:
    def __init__(self, database_file: Path, data_dir: Path | None = None) -> None:
        self.database_file = database_file
        self.data_dir = data_dir or database_file.parent
        self.clip_dir = self.data_dir / "flywheel" / "clips"
        self.training_clip_dir = self.data_dir / "flywheel" / "training_clips"
        self.export_dir = self.data_dir / "flywheel" / "exports"
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.training_clip_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        initialize_flywheel_schema(database_file)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_file, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def status(self) -> dict[str, Any]:
        with closing(self.connect()) as connection:
            statuses = dict(connection.execute(
                "SELECT status, COUNT(*) FROM flywheel_candidates GROUP BY status"
            ).fetchall())
            labels = dict(connection.execute(
                "SELECT label, COUNT(*) FROM flywheel_labels GROUP BY label"
            ).fetchall())
            objects = dict(connection.execute(
                "SELECT object_status, COUNT(*) FROM flywheel_candidates GROUP BY object_status"
            ).fetchall())
            total = int(connection.execute("SELECT COUNT(*) FROM flywheel_candidates").fetchone()[0])
            ready = int(connection.execute(
                "SELECT COUNT(*) FROM flywheel_candidates WHERE clip_path != ''"
            ).fetchone()[0])
            object_confirmed = int(connection.execute(
                "SELECT COUNT(*) FROM flywheel_candidates "
                "WHERE object_status='complete' AND json_extract(object_json, '$.confirmed')=1"
            ).fetchone()[0])
            trimmed_ready = int(connection.execute(
                "SELECT COUNT(*) FROM flywheel_candidates WHERE training_clip_path != ''"
            ).fetchone()[0])
        return {
            "activity": "drinking",
            "rule_version": "drink-wrist-mouth-v2",
            "total_candidates": total,
            "clips_ready": ready,
            "object_confirmed": object_confirmed,
            "trimmed_ready": trimmed_ready,
            "statuses": statuses,
            "labels": labels,
            "object_statuses": objects,
        }

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        for key in ("trigger_json", "bbox_json", "object_json"):
            try:
                value[key.removesuffix("_json")] = json.loads(value.pop(key) or "{}")
            except json.JSONDecodeError:
                value[key.removesuffix("_json")] = {}
        candidate_id = int(value["id"])
        value["clip_url"] = f"/api/flywheel/candidates/{candidate_id}/clip" if value.get("clip_path") else ""
        value["training_clip_url"] = (
            f"/api/flywheel/candidates/{candidate_id}/training-clip"
            if value.get("training_clip_path") else ""
        )
        return value

    def candidates(
        self, *, review: str = "all", person_id: int | None = None, limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if review == "unlabeled":
            clauses.append("l.id IS NULL")
        elif review in LABELS:
            clauses.append("l.label = ?")
            values.append(review)
        if person_id is not None:
            clauses.append("c.person_id = ?")
            values.append(person_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(max(1, min(500, int(limit))))
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT c.*, l.label, l.subtype, l.note, l.annotator, l.updated_at AS labeled_at, "
                "COALESCE(p.name, CASE WHEN c.person_id IS NULL THEN '未识别人物' "
                "ELSE '人员 ' || c.person_id END) AS person_name "
                "FROM flywheel_candidates c "
                "LEFT JOIN flywheel_labels l ON l.candidate_id = c.id "
                "LEFT JOIN people p ON p.id = c.person_id "
                f"{where} ORDER BY c.started_at DESC LIMIT ?",
                values,
            ).fetchall()
        return [self._row(row) for row in rows]

    def label(
        self, candidate_id: int, label: str, *, subtype: str = "", note: str = "",
        annotator: str = "local-reviewer",
    ) -> dict[str, Any] | None:
        if label not in LABELS:
            raise ValueError("label must be positive, negative, or uncertain")
        now = time.time()
        with closing(self.connect()) as connection, connection:
            candidate = connection.execute(
                "SELECT id FROM flywheel_candidates WHERE id = ?", (candidate_id,),
            ).fetchone()
            if not candidate:
                return None
            connection.execute(
                "INSERT INTO flywheel_labels(candidate_id, label, subtype, note, annotator, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(candidate_id) DO UPDATE SET label=excluded.label, subtype=excluded.subtype, "
                "note=excluded.note, annotator=excluded.annotator, updated_at=excluded.updated_at",
                (candidate_id, label, subtype[:100], note[:1000], annotator[:100], now, now),
            )
            connection.execute(
                "UPDATE flywheel_candidates SET status='labeled', updated_at=? WHERE id=?",
                (now, candidate_id),
            )
        return self.candidate(candidate_id)

    def candidate(self, candidate_id: int) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT c.*, l.label, l.subtype, l.note, l.annotator, l.updated_at AS labeled_at, "
                "COALESCE(p.name, CASE WHEN c.person_id IS NULL THEN '未识别人物' "
                "ELSE '人员 ' || c.person_id END) AS person_name "
                "FROM flywheel_candidates c LEFT JOIN flywheel_labels l ON l.candidate_id=c.id "
                "LEFT JOIN people p ON p.id=c.person_id WHERE c.id=?", (candidate_id,),
            ).fetchone()
        return self._row(row) if row else None

    def clip(self, candidate_id: int) -> Path | None:
        candidate = self.candidate(candidate_id)
        if not candidate or not candidate.get("clip_path"):
            return None
        path = (self.data_dir / str(candidate["clip_path"])).resolve()
        root = self.clip_dir.resolve()
        return path if path.is_file() and (path == root or root in path.parents) else None

    def training_clip(self, candidate_id: int) -> Path | None:
        candidate = self.candidate(candidate_id)
        if not candidate or not candidate.get("training_clip_path"):
            return None
        path = (self.data_dir / str(candidate["training_clip_path"])).resolve()
        root = self.training_clip_dir.resolve()
        return path if path.is_file() and (path == root or root in path.parents) else None

    def trim_candidate(self, candidate_id: int, start: float, end: float) -> dict[str, Any] | None:
        """Create a short training clip while retaining the original evidence clip."""
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("trim offsets must be finite and end must be after start")
        selected_duration = end - start
        if selected_duration < 2 or selected_duration > 8:
            raise ValueError("training clip must be between 2 and 8 seconds")
        candidate = self.candidate(candidate_id)
        if not candidate:
            return None
        if candidate.get("label") != "positive":
            raise ValueError("only confirmed drinking candidates can be trimmed")
        source = self.clip(candidate_id)
        if not source:
            raise ValueError("original candidate clip is not ready")
        try:
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(source),
                ],
                check=True, capture_output=True, timeout=20,
            )
            source_duration = float(probe.stdout.decode("utf-8").strip())
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise ValueError(f"could not read original clip duration: {error}") from error
        if end > source_duration + 0.05:
            raise ValueError(f"trim end exceeds source duration {source_duration:.2f}s")
        destination = self.training_clip_dir / f"candidate-{candidate_id}.mp4"
        temporary = destination.with_suffix(".tmp.mp4")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}",
                    "-i", str(source), "-t", f"{selected_duration:.3f}", "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-movflags", "+faststart", str(temporary),
                ],
                check=True, capture_output=True, timeout=90,
            )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise ValueError("ffmpeg returned an empty training clip")
            temporary.replace(destination)
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError(f"could not create training clip: {error}") from error
        finally:
            temporary.unlink(missing_ok=True)
        relative = str(destination.relative_to(self.data_dir))
        now = time.time()
        with closing(self.connect()) as connection, connection:
            connection.execute(
                "UPDATE flywheel_candidates SET training_clip_path=?,trim_start=?,trim_end=?,"
                "trim_updated_at=?,updated_at=? WHERE id=?",
                (relative, start, end, now, now, candidate_id),
            )
        return self.candidate(candidate_id)

    @staticmethod
    def _split(sample: dict[str, Any]) -> str:
        day = time.strftime("%Y-%m-%d", time.gmtime(float(sample["started_at"])))
        group = f"{sample.get('person_id')}:{sample['sensor_id']}:{day}"
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
        return "train" if bucket < 80 else "validation" if bucket < 90 else "test"

    def export_jsonl(self) -> tuple[Path, dict[str, int]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT c.*, l.label, l.subtype, l.note, l.annotator, l.updated_at AS labeled_at, "
                "COALESCE(p.name, CASE WHEN c.person_id IS NULL THEN '未识别人物' "
                "ELSE '人员 ' || c.person_id END) AS person_name "
                "FROM flywheel_candidates c JOIN flywheel_labels l ON l.candidate_id=c.id "
                "LEFT JOIN people p ON p.id=c.person_id WHERE c.clip_path != '' "
                "ORDER BY c.started_at"
            ).fetchall()
        samples = [self._row(row) for row in rows if row["label"] in LABELS]
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        destination = self.export_dir / f"drinking-{stamp}.jsonl"
        counts = {"train": 0, "validation": 0, "test": 0}
        with destination.open("w", encoding="utf-8") as handle:
            for sample in sorted(samples, key=lambda value: value["started_at"]):
                split = self._split(sample)
                counts[split] += 1
                object_value = sample.get("object") or {}
                target = {
                    "activity": "drinking",
                    "confirmed": sample["label"] == "positive",
                    "uncertain": sample["label"] == "uncertain",
                }
                training_path = str(sample.get("training_clip_path") or "")
                use_trimmed = bool(sample["label"] == "positive" and training_path)
                if use_trimmed and not (self.data_dir / training_path).is_file():
                    use_trimmed = False
                record = {
                    "sample_id": sample["sample_id"],
                    "split": split,
                    "video": training_path if use_trimmed else sample["clip_path"],
                    "prompt": "只判断对应人物是否正在喝水。忽略环境和其他人，并输出结构化JSON。",
                    "response": target,
                    "metadata": {
                        "person_id": sample.get("person_id"),
                        "sensor_id": sample["sensor_id"],
                        "started_at": sample["started_at"],
                        "ended_at": sample["ended_at"],
                        "rule_version": sample["rule_version"],
                        "candidate_score": sample["score"],
                        "label_subtype": sample.get("subtype") or "",
                        "is_trimmed": use_trimmed,
                        "source_video": sample["clip_path"],
                        "trim_start": float(sample.get("trim_start") or 0) if use_trimmed else None,
                        "trim_end": float(sample.get("trim_end") or 0) if use_trimmed else None,
                        "object_evidence": {
                            "detector": object_value.get("detector", ""),
                            "detector_version": object_value.get("detector_version", ""),
                            "confirmed": bool(object_value.get("confirmed", False)),
                            "best_score": float(object_value.get("best_score", 0)),
                            "confirmed_frames": int(object_value.get("confirmed_frames", 0)),
                            "threshold": float(object_value.get("threshold", 0)),
                        },
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        return destination, counts
