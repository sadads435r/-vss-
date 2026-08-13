# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-seat, privacy-preserving workstation analytics."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from datetime import date, datetime, time as datetime_time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from PIL import Image
import yaml

ACTIVITIES = ("computer", "reading", "writing", "phone", "conversation", "rest", "unknown")
ACTIVITY_LABELS = {
    "computer": "电脑操作",
    "reading": "阅读",
    "writing": "书写",
    "phone": "使用手机",
    "conversation": "交谈",
    "rest": "休息",
    "unknown": "无法判断",
}


def default_config() -> dict[str, Any]:
    """Defaults keep existing office-config.yaml files upgrade-compatible."""
    return {
        "enabled": True,
        "id": "desk-main",
        "name": "Main Workstation",
        "chair_roi": [[0.35, 0.25], [0.75, 0.25], [0.75, 0.95], [0.35, 0.95]],
        "sample_seconds": 20,
        "departure_seconds": 60,
        "state_confirmation_samples": 2,
        "frame_stale_seconds": 15,
        "focused_activities": ["computer", "reading", "writing"],
        "activities": list(ACTIVITIES),
        "cosmos3_url": "http://127.0.0.1:8018",
        "cosmos3_model": "auto",
        "report_retention_days": 365,
        "event_clip_retention_days": 7,
    }


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    """Return whether a normalized point is inside a normalized polygon."""
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def person_in_chair(frame: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str | None]:
    """Select a person whose bounding-box foot point is inside the chair ROI."""
    camera = config["camera"]
    width, height = (float(value) for value in camera.get("resolution", [1920, 1080]))
    polygon = config["workstation"]["chair_roi"]
    minimum = float(config["occupancy"].get("minimum_person_confidence", 0.3))
    candidates: list[tuple[float, str]] = []
    for item in frame.get("objects", []):
        if str(item.get("type", "")).casefold() != "person":
            continue
        bbox = item.get("bbox") or {}
        try:
            confidence = float(item.get("confidence", -1))
            bbox_confidence = float(bbox.get("confidence", -1))
            # Some DeepStream protobuf profiles report -0.1/0 at the object layer.
            # Zero/-0.1 are protobuf sentinel values, not real confidences.
            if 0 < max(confidence, bbox_confidence) < minimum:
                continue
            foot = (
                (float(bbox["leftX"]) + float(bbox["rightX"])) / 2 / width,
                float(bbox["bottomY"]) / height,
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if point_in_polygon(foot, polygon):
            candidates.append((max(confidence, bbox_confidence), str(item.get("id", "anonymous"))))
    if not candidates:
        return False, None
    return True, max(candidates)[1]


def validate_workstation_config(config: dict[str, Any]) -> None:
    workstation = config.get("workstation")
    if not isinstance(workstation, dict):
        raise ValueError("missing required section: workstation")
    polygon = workstation.get("chair_roi", [])
    if len(polygon) < 3:
        raise ValueError("workstation.chair_roi needs at least three points")
    for point in polygon:
        if len(point) != 2 or any(not 0 <= float(value) <= 1 for value in point):
            raise ValueError("workstation.chair_roi coordinates must be normalized into [0, 1]")
    if float(workstation.get("sample_seconds", 0)) < 5:
        raise ValueError("workstation.sample_seconds must be at least 5")
    if float(workstation.get("departure_seconds", 0)) < float(workstation["sample_seconds"]):
        raise ValueError("workstation.departure_seconds must be at least sample_seconds")
    if int(workstation.get("state_confirmation_samples", 0)) < 1:
        raise ValueError("workstation.state_confirmation_samples must be at least 1")
    if not workstation.get("cosmos3_url", workstation.get("rtvi_vlm_url")):
        raise ValueError("workstation.cosmos3_url is required")
    activities = workstation.get("activities", [])
    if set(activities) != set(ACTIVITIES):
        raise ValueError(f"workstation.activities must contain exactly: {', '.join(ACTIVITIES)}")
    if not set(workstation.get("focused_activities", [])).issubset(ACTIVITIES):
        raise ValueError("workstation.focused_activities contains an unsupported activity")
    retention = int(workstation.get("report_retention_days", 0))
    if retention < 1 or retention > 3650:
        raise ValueError("workstation.report_retention_days must be between 1 and 3650")


def initialize_schema(database_file: Path) -> None:
    with closing(sqlite3.connect(database_file)) as connection, connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS workstation_sessions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, started_at REAL NOT NULL, last_seen_at REAL NOT NULL, "
            "ended_at REAL, track_id TEXT)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS workstation_samples ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, observed_at REAL NOT NULL, occupied INTEGER, "
            "raw_activity TEXT, confidence REAL, data_status TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '')"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS workstation_activity_intervals ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL, activity TEXT NOT NULL, "
            "started_at REAL NOT NULL, ended_at REAL, FOREIGN KEY(session_id) REFERENCES workstation_sessions(id))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS workstation_away_events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, started_at REAL NOT NULL, ended_at REAL, "
            "clip_path TEXT NOT NULL DEFAULT '')"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS workstation_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS workstation_events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL, occurred_at REAL NOT NULL, "
            "activity TEXT, clip_path TEXT NOT NULL DEFAULT '', archive_error TEXT NOT NULL DEFAULT '')"
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_ws_samples_time ON workstation_samples(observed_at)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_ws_sessions_time ON workstation_sessions(started_at)")


def overlap(start: float, end: float, window_start: float, window_end: float) -> float:
    return max(0.0, min(end, window_end) - max(start, window_start))


def day_bounds(day: date, timezone: ZoneInfo) -> tuple[float, float]:
    start = datetime.combine(day, datetime_time.min, timezone).timestamp()
    return start, (datetime.combine(day + timedelta(days=1), datetime_time.min, timezone)).timestamp()


class WorkstationEngine:
    def __init__(
        self,
        config: dict[str, Any],
        database_file: Path,
        config_file: Path,
        elasticsearch_url: str,
        vst_url: str,
        clip_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.database_file = database_file
        self.config_file = config_file
        self.elasticsearch_url = elasticsearch_url.rstrip("/")
        self.vst_url = vst_url.rstrip("/")
        self.clip_dir = clip_dir or database_file.parent / "clips" / "workstation"
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.last_frame: dict[str, Any] | None = None
        self.last_frame_received_at: float | None = None
        self.last_vlm_at = 0.0
        self.last_vlm_error: str | None = None
        self.model_id: str | None = None
        self.last_health_check = 0.0
        self.last_health_result = False

    @property
    def workstation(self) -> dict[str, Any]:
        return self.config["workstation"]

    @property
    def cosmos3_url(self) -> str:
        """Keep old office-config files compatible while preferring the new name."""
        return str(self.workstation.get("cosmos3_url", self.workstation.get("rtvi_vlm_url", ""))).rstrip("/")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_file, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _state(self, connection: sqlite3.Connection, key: str, default: str = "") -> str:
        row = connection.execute("SELECT value FROM workstation_state WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row else default

    def _set_state(self, connection: sqlite3.Connection, key: str, value: Any) -> None:
        connection.execute(
            "INSERT OR REPLACE INTO workstation_state(key, value) VALUES (?, ?)", (key, str(value))
        )

    def process_frame(self, frame: dict[str, Any] | None, now: float | None = None) -> None:
        current = time.time() if now is None else now
        stale_seconds = float(self.workstation.get("frame_stale_seconds", 15))
        if not frame:
            self._record_missing(current, "rtcv_no_frame")
            return
        try:
            observed = datetime.fromisoformat(str(frame.get("timestamp", "")).replace("Z", "+00:00")).timestamp()
        except ValueError:
            observed = current
        if current - observed > stale_seconds:
            self._record_missing(current, "rtcv_stale_frame")
            return
        token = f"{frame.get('sensorId', '')}:{frame.get('id', '')}:{frame.get('timestamp', '')}"
        occupied, track_id = person_in_chair(frame, self.config)
        with self.lock, closing(self._connect()) as connection, connection:
            if self._state(connection, "frame_token") == token:
                self._check_departure(connection, current)
                return
            self._set_state(connection, "frame_token", token)
            self._set_state(connection, "last_good_frame_at", observed)
            previous_status = self._state(connection, "data_status")
            self._set_state(connection, "data_status", "healthy")
            self._set_state(connection, "chair_occupied", "1" if occupied else "0")
            self.last_frame = frame
            self.last_frame_received_at = current
            connection.execute(
                "INSERT INTO workstation_samples(observed_at, occupied, data_status) VALUES (?, ?, 'healthy')",
                (observed, int(occupied)),
            )
            if occupied:
                self._observe_occupied(connection, observed, track_id)
            elif previous_status == "missing":
                self._close_uncertain_session(connection, observed)
            else:
                self._check_departure(connection, current)

    def _record_missing(self, observed: float, detail: str) -> None:
        with self.lock, closing(self._connect()) as connection, connection:
            previous = self._state(connection, "data_status")
            self._set_state(connection, "data_status", "missing")
            self._set_state(connection, "chair_occupied", "0")
            if previous != "missing":
                self._set_state(connection, "missing_since", observed)
                connection.execute(
                    "UPDATE workstation_activity_intervals SET ended_at = ? WHERE ended_at IS NULL", (observed,)
                )
                connection.execute(
                    "INSERT INTO workstation_samples(observed_at, occupied, data_status, detail) "
                    "VALUES (?, NULL, 'missing', ?)",
                    (observed, detail),
                )

    def _close_uncertain_session(self, connection: sqlite3.Connection, recovered_at: float) -> None:
        """Close an open session across an outage without inventing a departure event."""
        session = connection.execute(
            "SELECT * FROM workstation_sessions WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not session:
            return
        ended = float(self._state(connection, "missing_since", str(recovered_at)))
        ended = max(float(session["started_at"]), min(ended, recovered_at))
        connection.execute("UPDATE workstation_sessions SET ended_at = ? WHERE id = ?", (ended, session["id"]))
        connection.execute(
            "UPDATE workstation_activity_intervals SET ended_at = ? WHERE session_id = ? AND ended_at IS NULL",
            (ended, session["id"]),
        )

    def _observe_occupied(self, connection: sqlite3.Connection, observed: float, track_id: str | None) -> None:
        session = connection.execute(
            "SELECT * FROM workstation_sessions WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not session:
            session_id = connection.execute(
                "INSERT INTO workstation_sessions(started_at, last_seen_at, track_id) VALUES (?, ?, ?)",
                (observed, observed, track_id),
            ).lastrowid
            away = connection.execute(
                "SELECT id FROM workstation_away_events WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if away:
                connection.execute("UPDATE workstation_away_events SET ended_at = ? WHERE id = ?", (observed, away[0]))
            self._set_state(connection, "current_activity", "unknown")
            self._set_state(connection, "pending_activity", "")
            self._set_state(connection, "pending_count", "0")
            connection.execute(
                "INSERT INTO workstation_activity_intervals(session_id, activity, started_at) VALUES (?, 'unknown', ?)",
                (session_id, observed),
            )
        else:
            connection.execute(
                "UPDATE workstation_sessions SET last_seen_at = ?, track_id = ? WHERE id = ?",
                (observed, track_id, session["id"]),
            )
            open_interval = connection.execute(
                "SELECT 1 FROM workstation_activity_intervals WHERE session_id = ? AND ended_at IS NULL",
                (session["id"],),
            ).fetchone()
            if not open_interval:
                connection.execute(
                    "INSERT INTO workstation_activity_intervals(session_id, activity, started_at) VALUES (?, ?, ?)",
                    (session["id"], self._state(connection, "current_activity", "unknown"), observed),
                )

    def _check_departure(self, connection: sqlite3.Connection, current: float) -> None:
        session = connection.execute(
            "SELECT * FROM workstation_sessions WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not session:
            return
        departure = float(self.workstation.get("departure_seconds", 60))
        if current - float(session["last_seen_at"]) < departure:
            return
        ended = float(session["last_seen_at"])
        connection.execute("UPDATE workstation_sessions SET ended_at = ? WHERE id = ?", (ended, session["id"]))
        connection.execute(
            "UPDATE workstation_activity_intervals SET ended_at = ? WHERE session_id = ? AND ended_at IS NULL",
            (ended, session["id"]),
        )
        connection.execute("INSERT INTO workstation_away_events(started_at) VALUES (?)", (ended,))
        connection.execute(
            "INSERT INTO workstation_events(event_type, occurred_at) VALUES ('away', ?)", (ended,)
        )

    def needs_vlm_sample(self, now: float | None = None) -> bool:
        current = time.time() if now is None else now
        if current - self.last_vlm_at < float(self.workstation.get("sample_seconds", 20)):
            return False
        with closing(self._connect()) as connection:
            if self._state(connection, "data_status") != "healthy" or self._state(connection, "chair_occupied") != "1":
                return False
            return connection.execute(
                "SELECT 1 FROM workstation_sessions WHERE ended_at IS NULL LIMIT 1"
            ).fetchone() is not None

    def current_picture(self) -> bytes:
        sensor_id = str(self.config["camera"].get("vss_sensor_id", "")).strip()
        if not sensor_id:
            raise ValueError("camera.vss_sensor_id is required for snapshots")
        url = f"{self.vst_url}/vst/api/v1/storage/stream/{urllib.parse.quote(sensor_id)}/picture"
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read(15 * 1024 * 1024)
        if not body:
            raise ValueError("VST returned an empty picture")
        return body

    def cropped_picture(self) -> bytes:
        picture = self.current_picture()
        polygon = self.workstation["chair_roi"]
        with Image.open(io.BytesIO(picture)) as image:
            image = image.convert("RGB")
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            margin = 0.08
            left = max(0, int((min(xs) - margin) * image.width))
            top = max(0, int((min(ys) - margin) * image.height))
            right = min(image.width, int((max(xs) + margin) * image.width))
            bottom = min(image.height, int((max(ys) + margin) * image.height))
            cropped = image.crop((left, top, right, bottom))
            output = io.BytesIO()
            cropped.save(output, format="JPEG", quality=85)
            return output.getvalue()

    def analyze_activity(self, now: float | None = None) -> dict[str, Any]:
        observed = time.time() if now is None else now
        self.last_vlm_at = observed
        try:
            image = self.cropped_picture()
            result = self._call_vlm(image)
            activity = str(result.get("activity", "unknown")).lower()
            if activity not in ACTIVITIES:
                activity = "unknown"
            confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
            detail = str(result.get("detail", ""))[:300]
            self.last_vlm_error = None
            self._record_activity(observed, activity, confidence, detail)
            return {"activity": activity, "confidence": confidence, "detail": detail}
        except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
            self.last_vlm_error = str(error)
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    "UPDATE workstation_activity_intervals SET ended_at = ? WHERE ended_at IS NULL", (observed,)
                )
                connection.execute(
                    "INSERT INTO workstation_samples(observed_at, occupied, raw_activity, data_status, detail) "
                    "VALUES (?, 1, 'unknown', 'vlm_missing', ?)",
                    (observed, str(error)[:500]),
                )
            return {"activity": "unknown", "confidence": 0, "error": str(error)}

    def _model(self) -> str:
        configured = str(self.workstation.get("cosmos3_model", self.workstation.get("rtvi_vlm_model", "auto")))
        if configured != "auto":
            return configured
        if self.model_id:
            return self.model_id
        with urllib.request.urlopen(f"{self.cosmos3_url}/v1/models", timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
        self.model_id = str(result["data"][0]["id"])
        return self.model_id

    def cosmos3_healthy(self, now: float | None = None) -> bool:
        current = time.time() if now is None else now
        if current - self.last_health_check < 15:
            return self.last_health_result
        self.last_health_check = current
        try:
            with urllib.request.urlopen(f"{self.cosmos3_url}/health", timeout=2) as response:
                self.last_health_result = response.status < 500
        except (OSError, urllib.error.URLError):
            self.last_health_result = False
        return self.last_health_result

    def _call_vlm(self, image: bytes) -> dict[str, Any]:
        data_url = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
        prompt = (
            "Analyze only the anonymous person at this single workstation. Return one JSON object and no markdown: "
            '{"activity":"computer|reading|writing|phone|conversation|rest|unknown",'
            '"confidence":0.0,"detail":"short Chinese description"}. '
            "computer means actively using a computer; reading means reading paper/book; writing means handwriting; "
            "phone means using a phone; conversation means talking with someone; rest means seated but not working; "
            "use unknown when unclear. The detail may describe a more specific visible action such as typing, using "
            "a mouse, looking at a monitor, handwriting, drinking water, eating, stretching, or tidying the desk, "
            "but activity must remain one of the fixed categories. Do not identify the person, read private screen "
            "content, or infer sensitive traits."
        )
        payload = {
            "model": self._model(),
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
            "stream": False,
            "temperature": 0,
            "max_tokens": 512,
        }
        request = urllib.request.Request(
            f"{self.cosmos3_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = str(result["choices"][0]["message"]["content"]).strip()
        if "```" in content:
            content = content.replace("```json", "").replace("```", "").strip()
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end < start:
            raise ValueError("VLM response did not contain JSON")
        return json.loads(content[start:end + 1])

    def _record_activity(self, observed: float, activity: str, confidence: float, detail: str) -> None:
        confirmations = int(self.workstation.get("state_confirmation_samples", 2))
        with self.lock, closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO workstation_samples(observed_at, occupied, raw_activity, confidence, data_status, detail) "
                "VALUES (?, 1, ?, ?, 'healthy', ?)",
                (observed, activity, confidence, detail),
            )
            session = connection.execute(
                "SELECT id FROM workstation_sessions WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if not session:
                return
            current = self._state(connection, "current_activity", "unknown")
            pending = self._state(connection, "pending_activity")
            count = int(self._state(connection, "pending_count", "0"))
            if activity == current:
                pending, count = "", 0
                open_interval = connection.execute(
                    "SELECT 1 FROM workstation_activity_intervals WHERE session_id = ? AND ended_at IS NULL",
                    (session["id"],),
                ).fetchone()
                if not open_interval:
                    connection.execute(
                        "INSERT INTO workstation_activity_intervals(session_id, activity, started_at) VALUES (?, ?, ?)",
                        (session["id"], current, observed),
                    )
            elif activity == pending:
                count += 1
            else:
                pending, count = activity, 1
            if count >= confirmations:
                connection.execute(
                    "UPDATE workstation_activity_intervals SET ended_at = ? "
                    "WHERE session_id = ? AND ended_at IS NULL",
                    (observed, session["id"]),
                )
                connection.execute(
                    "INSERT INTO workstation_activity_intervals(session_id, activity, started_at) VALUES (?, ?, ?)",
                    (session["id"], activity, observed),
                )
                connection.execute(
                    "INSERT INTO workstation_events(event_type, occurred_at, activity) VALUES ('activity_change', ?, ?)",
                    (observed, activity),
                )
                current, pending, count = activity, "", 0
            self._set_state(connection, "current_activity", current)
            self._set_state(connection, "pending_activity", pending)
            self._set_state(connection, "pending_count", count)

    def live(self, now: float | None = None) -> dict[str, Any]:
        current = time.time() if now is None else now
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            session = connection.execute(
                "SELECT * FROM workstation_sessions WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1"
            ).fetchone()
            status = self._state(connection, "data_status", "missing")
            activity = self._state(connection, "current_activity", "unknown")
            chair_occupied = self._state(connection, "chair_occupied", "0") == "1"
            last_frame = float(self._state(connection, "last_good_frame_at", "0") or 0)
            latest_sample = connection.execute(
                "SELECT detail, confidence FROM workstation_samples "
                "WHERE raw_activity IS NOT NULL ORDER BY observed_at DESC LIMIT 1"
            ).fetchone()
        report = self.report(datetime.fromtimestamp(current, ZoneInfo(self.config["timezone"])).date(), current)
        return {
            "workstation_id": self.workstation.get("id", "desk-main"),
            "occupied": bool(session) and status == "healthy" and chair_occupied,
            "activity": activity if session and chair_occupied else "away",
            "activity_label": ACTIVITY_LABELS.get(activity, "离座") if session and chair_occupied else "离座",
            "activity_detail": str(latest_sample[0]) if latest_sample and session and chair_occupied else "",
            "activity_confidence": float(latest_sample[1] or 0) if latest_sample and session and chair_occupied else 0,
            "session_started_at": session["started_at"] if session else None,
            "continuous_seated_seconds": round(current - session["started_at"]) if session and chair_occupied else 0,
            "camera_online": bool(last_frame and current - last_frame <= float(self.workstation.get("frame_stale_seconds", 15)) * 2),
            "rtcv_healthy": status == "healthy",
            "vlm_healthy": self.cosmos3_healthy(current),
            "vlm_error": self.last_vlm_error,
            "today": report,
            "disclaimer": "行为由模型估计，仅供个人时间复盘。",
        }

    def report(self, day: date, now: float | None = None) -> dict[str, Any]:
        current = time.time() if now is None else now
        timezone = ZoneInfo(self.config["timezone"])
        start, end = day_bounds(day, timezone)
        effective_end = min(end, current)
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            sessions = connection.execute(
                "SELECT * FROM workstation_sessions WHERE started_at < ? AND COALESCE(ended_at, ?) > ? ORDER BY started_at",
                (end, current, start),
            ).fetchall()
            intervals = connection.execute(
                "SELECT * FROM workstation_activity_intervals WHERE started_at < ? AND COALESCE(ended_at, ?) > ? ORDER BY started_at",
                (end, current, start),
            ).fetchall()
            away = connection.execute(
                "SELECT * FROM workstation_away_events WHERE started_at < ? AND COALESCE(ended_at, ?) > ? ORDER BY started_at",
                (end, current, start),
            ).fetchall()
            events = connection.execute(
                "SELECT * FROM workstation_events WHERE occurred_at >= ? AND occurred_at < ? ORDER BY occurred_at",
                (start, end),
            ).fetchall()
        durations = {activity: 0.0 for activity in ACTIVITIES}
        timeline = []
        for row in intervals:
            interval_start = max(float(row["started_at"]), start)
            interval_end = min(float(row["ended_at"] or current), effective_end)
            if interval_end <= interval_start:
                continue
            durations[row["activity"]] += interval_end - interval_start
            timeline.append({"activity": row["activity"], "started_at": interval_start, "ended_at": interval_end})
        # Activity intervals are deliberately closed during RT-CV/VLM outages. Summing them
        # keeps missing data out of seated, focused, and overtime totals.
        seated = sum(durations.values())
        focused = sum(durations[name] for name in self.workstation.get("focused_activities", []))
        overtime = sum(
            self._overtime_overlap(row["started_at"], row["ended_at"] or current, day, timezone)
            for row in intervals
        )
        away_duration = sum(overlap(row["started_at"], row["ended_at"] or current, start, effective_end) for row in away)
        return {
            "date": day.isoformat(),
            "seated_seconds": round(seated),
            "focused_seconds": round(focused),
            "focus_rate": round(focused / seated * 100, 1) if seated else 0,
            "activity_seconds": {key: round(value) for key, value in durations.items()},
            "away_count": len(away),
            "away_seconds": round(away_duration),
            "first_arrival_at": min((row["started_at"] for row in sessions), default=None),
            "last_departure_at": max((row["ended_at"] for row in sessions if row["ended_at"]), default=None),
            "overtime_seconds": round(overtime),
            "timeline": timeline,
            "away_events": [dict(row) for row in away],
            "events": [dict(row) for row in events],
        }

    def _overtime_overlap(self, session_start: float, session_end: float, day: date, timezone: ZoneInfo) -> float:
        start, end = day_bounds(day, timezone)
        schedule = self.config["schedule"]
        workday = day.strftime("%A").lower() in {str(value).lower() for value in schedule["weekdays"]}
        workday = workday and day.isoformat() not in set(schedule.get("holidays", []))
        if not workday:
            return overlap(session_start, session_end, start, end)
        overtime_start = datetime.combine(day, datetime_time.fromisoformat(str(schedule["end"])), timezone).timestamp()
        return overlap(session_start, session_end, overtime_start, end)

    def reports(self, start: date, end: date) -> list[dict[str, Any]]:
        if end < start or (end - start).days > 366:
            raise ValueError("report date range must be between 0 and 366 days")
        result = []
        day = start
        while day <= end:
            result.append(self.report(day))
            day += timedelta(days=1)
        return result

    def roi(self) -> dict[str, Any]:
        return {"chair_roi": self.workstation["chair_roi"]}

    def save_roi(self, polygon: list[list[float]]) -> dict[str, Any]:
        candidate = json.loads(json.dumps(self.config))
        candidate["workstation"]["chair_roi"] = polygon
        validate_workstation_config(candidate)
        with self.lock:
            self.config["workstation"]["chair_roi"] = polygon
            with self.config_file.open("r", encoding="utf-8") as handle:
                document = yaml.safe_load(handle)
            document.setdefault("workstation", default_config())["chair_roi"] = polygon
            temporary = self.config_file.with_suffix(".yaml.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(document, handle, allow_unicode=True, sort_keys=False)
            temporary.replace(self.config_file)
        return self.roi()

    def cleanup(self, now: float | None = None) -> None:
        current = time.time() if now is None else now
        cutoff = current - int(self.workstation.get("report_retention_days", 365)) * 86400
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM workstation_samples WHERE observed_at < ?", (cutoff,))
            connection.execute("DELETE FROM workstation_activity_intervals WHERE COALESCE(ended_at, started_at) < ?", (cutoff,))
            connection.execute("DELETE FROM workstation_sessions WHERE COALESCE(ended_at, started_at) < ?", (cutoff,))
            connection.execute("DELETE FROM workstation_away_events WHERE COALESCE(ended_at, started_at) < ?", (cutoff,))
        clip_cutoff = current - int(self.workstation.get("event_clip_retention_days", 7)) * 86400
        with closing(self._connect()) as connection, connection:
            expired = connection.execute(
                "SELECT id, clip_path FROM workstation_events WHERE occurred_at < ?", (clip_cutoff,)
            ).fetchall()
            for event_id, clip_path in expired:
                if clip_path:
                    Path(clip_path).unlink(missing_ok=True)
                connection.execute("DELETE FROM workstation_events WHERE id = ?", (event_id,))

    def mark_overtime(self, now: float | None = None) -> None:
        current = time.time() if now is None else now
        timezone = ZoneInfo(self.config["timezone"])
        local = datetime.fromtimestamp(current, timezone)
        schedule = self.config["schedule"]
        workday = local.strftime("%A").lower() in {str(value).lower() for value in schedule["weekdays"]}
        workday = workday and local.date().isoformat() not in set(schedule.get("holidays", []))
        if workday and local.time().replace(tzinfo=None) < datetime_time.fromisoformat(str(schedule["end"])):
            return
        marker = f"overtime:{local.date().isoformat()}"
        with closing(self._connect()) as connection, connection:
            occupied = (
                self._state(connection, "data_status") == "healthy"
                and self._state(connection, "chair_occupied") == "1"
                and connection.execute("SELECT 1 FROM workstation_sessions WHERE ended_at IS NULL LIMIT 1").fetchone()
            )
            if occupied and not self._state(connection, marker):
                connection.execute(
                    "INSERT INTO workstation_events(event_type, occurred_at) VALUES ('overtime_start', ?)", (current,)
                )
                self._set_state(connection, marker, "1")

    def archive_pending_clips(self, now: float | None = None) -> None:
        current = time.time() if now is None else now
        after = int(self.config.get("retention", {}).get("clip_after_seconds", 20))
        before = int(self.config.get("retention", {}).get("clip_before_seconds", 10))
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT id, occurred_at FROM workstation_events "
                "WHERE clip_path = '' AND archive_error = '' AND occurred_at <= ? ORDER BY id LIMIT 5",
                (current - after,),
            ).fetchall()
            for event_id, occurred_at in rows:
                try:
                    destination = self._archive_clip(event_id, occurred_at - before, occurred_at + after)
                    connection.execute(
                        "UPDATE workstation_events SET clip_path = ? WHERE id = ?", (str(destination), event_id)
                    )
                except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
                    connection.execute(
                        "UPDATE workstation_events SET archive_error = ? WHERE id = ?",
                        (str(error)[:500], event_id),
                    )

    def _archive_clip(self, event_id: int, start: float, end: float) -> Path:
        sensor_id = str(self.config["camera"].get("vss_sensor_id", "")).strip()
        if not sensor_id:
            raise ValueError("camera.vss_sensor_id is required for event clips")
        iso = lambda value: datetime.fromtimestamp(value, ZoneInfo("UTC")).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        query = urllib.parse.urlencode({
            "startTime": iso(start), "endTime": iso(end), "blocking": "true", "disableAudio": "true",
        })
        url = f"{self.vst_url}/vst/api/v1/storage/file/{urllib.parse.quote(sensor_id)}/url?{query}"
        with urllib.request.urlopen(url, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
        video_url = urllib.parse.urljoin(f"{self.vst_url}/", str(result.get("videoUrl", "")))
        if not video_url:
            raise ValueError("VST did not return videoUrl")
        parsed = urllib.parse.urlparse(video_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("VST returned an unsupported clip URL")
        destination = self.clip_dir / f"{hashlib.sha256(str(event_id).encode()).hexdigest()}.mp4"
        temporary = destination.with_suffix(".part")
        with urllib.request.urlopen(video_url, timeout=60) as source, temporary.open("wb") as output:
            remaining = 256 * 1024 * 1024
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                output.write(chunk)
                remaining -= len(chunk)
            if source.read(1):
                raise ValueError("workstation clip exceeds 256 MB")
        temporary.replace(destination)
        return destination

    def event_clip(self, event_id: int) -> Path | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT clip_path FROM workstation_events WHERE id = ?", (event_id,)).fetchone()
        if not row or not row[0]:
            return None
        path = Path(row[0])
        return path if path.is_file() and self.clip_dir in path.parents else None


def worker(engine: WorkstationEngine, fetch_frame: Any, poll_seconds: float = 2.0) -> None:
    while True:
        try:
            engine.process_frame(fetch_frame())
            if engine.needs_vlm_sample():
                engine.analyze_activity()
            engine.mark_overtime()
            engine.archive_pending_clips()
            if int(time.time()) % 3600 < max(1, int(poll_seconds)):
                engine.cleanup()
        except Exception as error:  # Worker must remain alive across dependent-service outages.
            print(f"[office-api] workstation worker failed: {error}", flush=True)
        time.sleep(max(0.5, poll_seconds))
