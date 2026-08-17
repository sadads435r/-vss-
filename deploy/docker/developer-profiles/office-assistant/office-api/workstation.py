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

ACTIVITIES = ("computer", "reading", "writing", "phone", "conversation", "eating", "rest", "unknown")
ACTIVITY_LABELS = {
    "computer": "电脑操作",
    "reading": "阅读",
    "writing": "书写",
    "phone": "使用手机",
    "conversation": "交谈",
    "eating": "吃东西",
    "rest": "睡觉",
    "unknown": "无法判断",
}


def _ensure_columns(connection: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """Add missing SQLite columns without changing existing rows."""
    existing = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, definition in columns.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


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
        "minimum_activity_confidence": 0.6,
        "frame_stale_seconds": 15,
        "focused_activities": ["computer", "reading", "writing"],
        "activities": list(ACTIVITIES),
        "cosmos3_url": "http://127.0.0.1:8018",
        "cosmos3_model": "auto",
        "report_retention_days": 365,
        "event_clip_retention_days": 7,
        # 人员识别：新 track 需稳定出现该秒数才触发 VLM 判定，避免一闪而过的误检
        "person_verify_seconds": 3.0,
        # 身份比对时最多与最近活跃的 N 个已注册人员逐一比对（控制 VLM 成本）
        "person_max_compare": 8,
        # 判定为"非真人"的 track 在冷却期内不重复判定
        "person_reject_cooldown_seconds": 600,
        # 按人活动采样节流（画面内所有人员轮流采样，缩短可加快多人工作时长累积）
        "person_sample_seconds": 10,
        # 在场/在椅区间融合为连续"在工位"段的最大间隙（人坐下后短暂漏检不算离席）
        "person_presence_merge_seconds": 300,
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
            # Zero/-0.1 are protobuf sentinel values, not real confidences; treat them as
            # below the minimum so sentinel frames never count as occupied.
            if max(confidence, bbox_confidence) < minimum:
                continue
            bottom = float(bbox["bottomY"])
            top = float(bbox.get("topY", bottom))
            # Use the point 80% down the box (torso/hip level) instead of the foot:
            # person boxes often extend to the bottom of the frame while seated,
            # so the foot point can fall outside the chair ROI even when seated.
            foot = (
                (float(bbox["leftX"]) + float(bbox["rightX"])) / 2 / width,
                (top + (bottom - top) * 0.8) / height,
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
        # 按人记录：画面内所有 track 的在椅区间 / 活动区间 / 活动状态机
        connection.execute(
            "CREATE TABLE IF NOT EXISTS person_seated_intervals ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL, track_id TEXT NOT NULL, "
            "started_at REAL NOT NULL, last_seen_at REAL NOT NULL, ended_at REAL, workstation_session_id INTEGER)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_psi_track ON person_seated_intervals(track_id, started_at)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS person_activity_intervals ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL, track_id TEXT NOT NULL, "
            "seated_interval_id INTEGER, activity TEXT NOT NULL, confidence REAL, "
            "started_at REAL NOT NULL, ended_at REAL, person_id INTEGER, "
            "description TEXT NOT NULL DEFAULT '', last_observed_at REAL, "
            "observation_count INTEGER NOT NULL DEFAULT 1)"
        )
        _ensure_columns(connection, "person_activity_intervals", {
            "person_id": "INTEGER",
            "description": "TEXT NOT NULL DEFAULT ''",
            "last_observed_at": "REAL",
            "observation_count": "INTEGER NOT NULL DEFAULT 1",
        })
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_pai_track ON person_activity_intervals(track_id, started_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_pai_person_time "
            "ON person_activity_intervals(person_id, started_at)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS person_activity_state ("
            "track_id TEXT PRIMARY KEY, current TEXT, pending TEXT, count INTEGER, last_sampled_at REAL, "
            "current_description TEXT NOT NULL DEFAULT '', pending_description TEXT NOT NULL DEFAULT '', "
            "pending_started_at REAL)"
        )
        _ensure_columns(connection, "person_activity_state", {
            "current_description": "TEXT NOT NULL DEFAULT ''",
            "pending_description": "TEXT NOT NULL DEFAULT ''",
            "pending_started_at": "REAL",
        })
        # 人员库：每个已确认的真人一条记录，reference_image 存参考图路径
        connection.execute(
            "CREATE TABLE IF NOT EXISTS people ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, "
            "first_seen_at REAL NOT NULL, last_seen_at REAL NOT NULL, "
            "reference_image TEXT NOT NULL DEFAULT '', active INTEGER NOT NULL DEFAULT 1)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_people_active ON people(active)"
        )
        # track -> person 映射：同一人跨重进/跨 track 归并到同一 person
        connection.execute(
            "CREATE TABLE IF NOT EXISTS person_track_map ("
            "track_id TEXT PRIMARY KEY, person_id INTEGER NOT NULL, matched_at REAL NOT NULL)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ptm_person ON person_track_map(person_id)"
        )
        # 判定记录：每次新 track 的 VLM 判定结果（新人/真人/匹配到谁）
        connection.execute(
            "CREATE TABLE IF NOT EXISTS person_verifications ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, track_id TEXT NOT NULL, person_id INTEGER, "
            "is_person INTEGER NOT NULL DEFAULT 0, matched_person_id INTEGER, "
            "confidence REAL NOT NULL DEFAULT 0, reason TEXT NOT NULL DEFAULT '', "
            "image_path TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL)"
        )


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
        # 人员参考图存储目录：数据库同级 people/
        self.people_dir = database_file.parent / "people"
        self.people_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.last_frame: dict[str, Any] | None = None
        self.last_frame_received_at: float | None = None
        self.last_vlm_at = 0.0
        self.last_vlm_error: str | None = None
        self._empty_confirmations = 0
        # 按人记录内存态：track_id -> person_seated_intervals.id；round-robin 采样游标
        self.person_seated: dict[str, int] = {}
        self._person_sample_index = 0
        # 人员识别内存态：新 track 出现时间戳/累计出现秒数、被拒 track 冷却
        self._person_first_seen: dict[str, float] = {}
        self._person_seen_seconds: dict[str, float] = {}
        self._person_reject_until: dict[str, float] = {}
        self._person_last_verify_at: dict[str, float] = {}
        # 待判定的新 track 队列（track_id -> bbox），由 worker 主循环异步消费（不阻塞 process_frame）
        self._person_pending: dict[str, dict[str, Any]] = {}
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
        connection = sqlite3.connect(self.database_file, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
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
            # 按人记录：遍历画面内所有 person，维护在场/进椅区间
            self._update_people(frame, observed, connection)

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

    def _update_people(self, frame: dict[str, Any], observed: float, connection: sqlite3.Connection) -> None:
        """按人记录：遍历帧内所有 person，维护在场/进椅区间 + 新 track 判定队列。"""
        minimum = float(self.config["occupancy"].get("minimum_person_confidence", 0.2))
        width, height = (float(value) for value in self.config["camera"].get("resolution", [1920, 1080]))
        polygon = self.workstation["chair_roi"]
        camera_id = str(self.config["camera"].get("id", "office-main"))
        seen: set[str] = set()
        # 新 track 判定参数
        verify_seconds = max(0.0, float(self.workstation.get("person_verify_seconds", 3.0)))
        pending: list[tuple[str, dict[str, Any]]] = []
        for item in frame.get("objects", []):
            if str(item.get("type", "")).casefold() != "person":
                continue
            bbox = item.get("bbox") or {}
            if "leftX" not in bbox or "rightX" not in bbox or "bottomY" not in bbox:
                continue
            try:
                conf = float(item.get("confidence", -1))
                bbox_conf = float(bbox.get("confidence", -1))
                if max(conf, bbox_conf) < minimum:
                    continue
                bottom = float(bbox["bottomY"])
                top = float(bbox.get("topY", bottom))
                foot = (
                    (float(bbox["leftX"]) + float(bbox["rightX"])) / 2 / width,
                    (top + (bottom - top) * 0.8) / height,
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            track_id = str(item.get("id", "")).strip()
            if not track_id:
                continue
            seen.add(track_id)
            if point_in_polygon(foot, polygon):
                self._observe_person_seated(connection, camera_id, track_id, observed)
            else:
                self._close_person_seated(connection, track_id, observed)
            # 新 track 判定：尚未入库映射的 track，累计出现时长，达标后加入待判定
            row = connection.execute(
                "SELECT 1 FROM person_track_map WHERE track_id = ?", (track_id,)
            ).fetchone()
            if row is None and observed >= self._person_reject_until.get(track_id, 0.0):
                if track_id not in self._person_first_seen:
                    self._person_first_seen[track_id] = observed
                    self._person_seen_seconds[track_id] = 0.0
                else:
                    self._person_seen_seconds[track_id] = observed - self._person_first_seen[track_id]
                if self._person_seen_seconds[track_id] >= verify_seconds:
                    pending.append((track_id, bbox))
        for track_id in list(self.person_seated):
            if track_id not in seen:
                self._close_person_seated(connection, track_id, observed)
        # 达标的新 track 进入内存待判队列，由 worker 主循环异步消费（VLM 判定不能占用 DB 事务）
        for track_id, bbox in pending:
            self._person_pending.setdefault(track_id, bbox)

    def _drain_person_pending(self, now: float | None = None) -> None:
        """worker 主循环调用：每次处理一个待判新 track（VLM 判定耗时，异步执行）。"""
        current = time.time() if now is None else now
        for track_id, bbox in list(self._person_pending.items()):
            cooldown = max(0.0, float(self.workstation.get("person_reject_cooldown_seconds", 600)))
            # 已入库/被拒冷却中的 track 跳过
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT 1 FROM person_track_map WHERE track_id = ?", (track_id,)
                ).fetchone()
                rejected = current < self._person_reject_until.get(track_id, 0.0)
            if row is not None or rejected:
                self._person_pending.pop(track_id, None)
                continue
            verify_seconds = max(0.0, float(self.workstation.get("person_verify_seconds", 3.0)))
            if current - self._person_last_verify_at.get(track_id, 0.0) < verify_seconds:
                continue
            self._person_last_verify_at[track_id] = current
            self._person_pending.pop(track_id, None)
            self._verify_new_track(track_id, bbox, current)
            return  # 每轮只处理一个，避免 VLM 长时间阻塞 worker

    def _verify_new_track(self, track_id: str, bbox: dict[str, Any], observed: float) -> None:
        """新 track 判定流程：截图 → VLM 单图判定是否真人 → 多图与人员库比对 → 入库/归并。"""
        reject_cooldown = max(0.0, float(self.workstation.get("person_reject_cooldown_seconds", 600)))
        camera_id = str(self.config["camera"].get("id", "office-main"))
        try:
            image = self.crop_person_bbox(bbox)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"[office-api] person verify crop failed track={track_id}: {error}", flush=True)
            return
        # Step 1: 单图判定是否真人（过滤布偶/海报/显示器人像等误检）
        try:
            verdict = self._call_vlm_verdict(image)
        except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
            self.last_vlm_error = str(error)
            print(f"[office-api] person verify VLM failed track={track_id}: {error}", flush=True)
            return
        is_person = bool(verdict.get("is_person", False))
        reason = str(verdict.get("reason", ""))[:200]
        with self.lock, closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO person_verifications(track_id, is_person, confidence, reason, image_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (track_id, int(is_person), float(verdict.get("confidence", 0)), reason, "", observed),
            )
            if not is_person:
                # 非真人：冷却期内不再重复判定（布偶/海报类稳定误检）
                self._person_reject_until[track_id] = observed + reject_cooldown
                return
            # Step 2: 与人员库逐一多图比对
            matched_person_id = self._match_person_identity(image, connection)
            image_path = self._store_person_image(track_id, image)
            if matched_person_id is not None:
                connection.execute(
                    "INSERT INTO person_track_map(track_id, person_id, matched_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(track_id) DO UPDATE SET person_id=excluded.person_id, matched_at=excluded.matched_at",
                    (track_id, matched_person_id, observed),
                )
                connection.execute(
                    "UPDATE people SET last_seen_at = ? WHERE id = ?", (observed, matched_person_id)
                )
                connection.execute(
                    "UPDATE person_verifications SET matched_person_id = ?, image_path = ? WHERE track_id = ? "
                    "AND created_at = ?",
                    (matched_person_id, image_path, track_id, observed),
                )
                print(f"[office-api] track {track_id} -> 人员 #{matched_person_id} (匹配), image={image_path}", flush=True)
                return
            # 未匹配：注册为新人
            cursor = connection.execute(
                "INSERT INTO people(name, first_seen_at, last_seen_at, reference_image) "
                "VALUES (?, ?, ?, ?)",
                (f"人员 {self._next_person_number(connection)}", observed, observed, image_path),
            )
            person_id = int(cursor.lastrowid)
            connection.execute(
                "INSERT INTO person_track_map(track_id, person_id, matched_at) VALUES (?, ?, ?)",
                (track_id, person_id, observed),
            )
            connection.execute(
                "UPDATE person_verifications SET matched_person_id = ?, image_path = ? WHERE track_id = ? "
                "AND created_at = ?",
                (person_id, image_path, track_id, observed),
            )
            print(f"[office-api] 新人注册: track {track_id} -> 人员 #{person_id}, image={image_path}", flush=True)

    def _next_person_number(self, connection: sqlite3.Connection) -> int:
        row = connection.execute("SELECT COUNT(*) FROM people").fetchone()
        return int(row[0]) + 1

    def _store_person_image(self, track_id: str, image: bytes) -> str:
        """将单人截图存入 people 目录，返回相对路径（相对于 database_file.parent）。"""
        import hashlib as _hashlib
        stamp = int(time.time())
        digest = _hashlib.sha256(image).hexdigest()[:10]
        path = self.people_dir / f"track_{track_id}_{stamp}_{digest}.jpg"
        path.write_bytes(image)
        try:
            return str(path.relative_to(self.database_file.parent))
        except ValueError:
            return str(path)

    def _match_person_identity(self, image: bytes, connection: sqlite3.Connection) -> int | None:
        """与人员库中最近活跃的人员逐一多图比对，返回匹配的 person_id 或 None。"""
        max_compare = max(1, int(self.workstation.get("person_max_compare", 8)))
        rows = connection.execute(
            "SELECT id, reference_image FROM people WHERE active = 1 AND reference_image != '' "
            "ORDER BY last_seen_at DESC LIMIT ?",
            (max_compare,),
        ).fetchall()
        if not rows:
            return None
        best_id: int | None = None
        best_score = 0.0
        for row in rows:
            reference = self.database_file.parent / row["reference_image"]
            if not reference.is_file():
                continue
            try:
                result = self._call_vlm_identity(reference.read_bytes(), image)
            except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
                self.last_vlm_error = str(error)
                continue
            same = bool(result.get("same_person", False))
            score = float(result.get("confidence", 0))
            if same and score > best_score:
                best_id = int(row["id"])
                best_score = score
        return best_id

    def _call_vlm_verdict(self, image: bytes) -> dict[str, Any]:
        """单图判定：检测框里是否是一个真实的人（排除玩具/海报/显示器画面/阴影等）。"""
        data_url = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
        prompt = (
            "这是一台办公摄像头抓拍的一个人形检测框截图。请判断里面是否是一个真实的人。"
            "注意排除：玩具人偶、海报/打印图片上的人、显示器屏幕里的人物、影子、雕塑等误检。"
            "只依据画面内容判断，不要猜测。返回 JSON（不要 markdown）："
            '{"is_person": true或false, "confidence": 0.0~1.0, "reason": "简短中文理由"}'
        )
        payload = {
            "model": self._model(),
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
            "stream": False,
            "temperature": 0,
            "max_tokens": 256,
        }
        return self._post_vlm(payload)

    def _call_vlm_identity(self, reference: bytes, current: bytes) -> dict[str, Any]:
        """双图比对：参考图（人员库）与当前截图是否同一人。"""
        def data_url(image: bytes) -> str:
            return "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
        prompt = (
            "这是同一台办公摄像头在不同时刻抓拍的两张人物截图。第一张是人员档案中的参考照片，"
            "第二张是当前检测到的人。请判断两张图中是否是同一个人。"
            "比较衣服颜色和款式、发型、体型、是否戴眼镜、面部特征等。"
            "注意：同一人换衣服/换角度仍算同一人；不同人即使衣着相似也不算同一人。"
            "返回 JSON（不要 markdown）："
            '{"same_person": true或false, "confidence": 0.0~1.0, "reason": "简短中文理由"}'
        )
        payload = {
            "model": self._model(),
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url(reference)}},
                {"type": "image_url", "image_url": {"url": data_url(current)}},
            ]}],
            "stream": False,
            "temperature": 0,
            "max_tokens": 256,
        }
        return self._post_vlm(payload)

    def _post_vlm(self, payload: dict[str, Any]) -> dict[str, Any]:
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

    def _observe_person_seated(self, connection: sqlite3.Connection, camera_id: str, track_id: str, observed: float) -> None:
        interval_id = self.person_seated.get(track_id)
        if interval_id is not None:
            connection.execute(
                "UPDATE person_seated_intervals SET last_seen_at = ? WHERE id = ?", (observed, interval_id)
            )
            return
        cursor = connection.execute(
            "INSERT INTO person_seated_intervals(camera_id, track_id, started_at, last_seen_at) VALUES (?, ?, ?, ?)",
            (camera_id, track_id, observed, observed),
        )
        self.person_seated[track_id] = int(cursor.lastrowid)

    def _close_person_seated(self, connection: sqlite3.Connection, track_id: str, observed: float) -> None:
        interval_id = self.person_seated.get(track_id)
        if interval_id is None:
            return
        row = connection.execute(
            "SELECT last_seen_at FROM person_seated_intervals WHERE id = ?", (interval_id,)
        ).fetchone()
        if not row:
            self.person_seated.pop(track_id, None)
            return
        departure = float(self.workstation.get("departure_seconds", 60))
        if observed - float(row[0]) < departure:
            return  # 短暂消失（漏检/遮挡），保留区间
        connection.execute(
            "UPDATE person_seated_intervals SET ended_at = ? WHERE id = ?", (row[0], interval_id)
        )
        connection.execute(
            "UPDATE person_activity_intervals SET ended_at = COALESCE(last_observed_at, ?) "
            "WHERE track_id = ? AND ended_at IS NULL",
            (row[0], track_id),
        )
        self.person_seated.pop(track_id, None)

    def _current_person_event(self, connection: sqlite3.Connection, track_id: str) -> sqlite3.Row | None:
        mapping = connection.execute(
            "SELECT person_id FROM person_track_map WHERE track_id = ?", (track_id,)
        ).fetchone()
        if mapping:
            return connection.execute(
                "SELECT * FROM person_activity_intervals "
                "WHERE person_id = ? AND ended_at IS NULL ORDER BY started_at DESC LIMIT 1",
                (int(mapping[0]),),
            ).fetchone()
        return connection.execute(
            "SELECT * FROM person_activity_intervals "
            "WHERE track_id = ? AND ended_at IS NULL ORDER BY started_at DESC LIMIT 1",
            (track_id,),
        ).fetchone()

    def _record_person_activity(
        self, connection: sqlite3.Connection, camera_id: str, track_id: str,
        activity: str, description: str, confidence: float, continues_current: bool, observed: float,
    ) -> None:
        """Merge repeated observations and split only after a confirmed activity change."""
        connection.row_factory = sqlite3.Row
        confirmations = int(self.workstation.get("state_confirmation_samples", 2))
        minimum_conf = float(self.workstation.get("minimum_activity_confidence", 0.6))
        description = " ".join(str(description).strip().split())[:200]
        if not description:
            description = ACTIVITY_LABELS.get(activity, ACTIVITY_LABELS["unknown"])
        mapping = connection.execute(
            "SELECT person_id FROM person_track_map WHERE track_id = ?", (track_id,)
        ).fetchone()
        person_id = int(mapping[0]) if mapping else None
        if person_id is not None:
            connection.execute(
                "UPDATE person_activity_intervals SET person_id = ? "
                "WHERE track_id = ? AND person_id IS NULL",
                (person_id, track_id),
            )
        open_event = self._current_person_event(connection, track_id)
        row = connection.execute(
            "SELECT current, pending, count, current_description, pending_description, pending_started_at "
            "FROM person_activity_state WHERE track_id = ?", (track_id,)
        ).fetchone()
        if row:
            current, pending, count = str(row[0] or "unknown"), str(row[1] or ""), int(row[2] or 0)
            current_description = str(row[3] or "")
            pending_description = str(row[4] or "")
            pending_started_at = float(row[5]) if row[5] is not None else None
        elif open_event:
            current = str(open_event["activity"])
            pending, count = "", 0
            current_description = str(open_event["description"] or ACTIVITY_LABELS.get(current, current))
            pending_description, pending_started_at = "", None
        else:
            current, pending, count = "unknown", "", 0
            current_description, pending_description, pending_started_at = "", "", None

        valid = confidence >= minimum_conf and activity != "unknown"
        same_event = bool(
            valid
            and open_event
            and activity == str(open_event["activity"])
            and continues_current
        )
        if same_event:
            previous_count = max(1, int(open_event["observation_count"] or 1))
            previous_confidence = float(open_event["confidence"] or 0)
            average_confidence = ((previous_confidence * previous_count) + confidence) / (previous_count + 1)
            connection.execute(
                "UPDATE person_activity_intervals SET last_observed_at = ?, observation_count = ?, confidence = ? "
                "WHERE id = ?",
                (observed, previous_count + 1, average_confidence, int(open_event["id"])),
            )
            current = activity
            current_description = str(open_event["description"] or description)
            pending, count, pending_description, pending_started_at = "", 0, "", None
        elif not valid:
            # Uncertain samples update the throttle timestamp but never fragment a confirmed event.
            pending, count, pending_description, pending_started_at = "", 0, "", None
        elif activity == pending:
            count += 1
            if count >= confirmations:
                event_start = pending_started_at if pending_started_at is not None else observed
                if open_event:
                    connection.execute(
                        "UPDATE person_activity_intervals SET ended_at = ? WHERE id = ?",
                        (event_start, int(open_event["id"])),
                    )
                current, current_description = activity, pending_description or description
                pending, count, pending_description, pending_started_at = "", 0, "", None
                seat = connection.execute(
                    "SELECT id FROM person_seated_intervals WHERE track_id = ? AND ended_at IS NULL "
                    "ORDER BY id DESC LIMIT 1",
                    (track_id,),
                ).fetchone()
                connection.execute(
                    "INSERT INTO person_activity_intervals("
                    "camera_id, track_id, person_id, seated_interval_id, activity, description, confidence, "
                    "started_at, last_observed_at, observation_count) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        camera_id,
                        track_id,
                        person_id,
                        seat[0] if seat else None,
                        activity,
                        current_description,
                        confidence,
                        event_start,
                        observed,
                        confirmations,
                    ),
                )
        else:
            pending, count = activity, 1
            pending_description = description
            pending_started_at = observed
        connection.execute(
            "INSERT INTO person_activity_state("
            "track_id, current, pending, count, last_sampled_at, current_description, "
            "pending_description, pending_started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(track_id) DO UPDATE SET current=excluded.current, pending=excluded.pending, "
            "count=excluded.count, last_sampled_at=excluded.last_sampled_at, "
            "current_description=excluded.current_description, pending_description=excluded.pending_description, "
            "pending_started_at=excluded.pending_started_at",
            (
                track_id, current, pending, count, observed, current_description,
                pending_description, pending_started_at,
            ),
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
        # 直接从 MediaMTX 8554 清晰流用 ffmpeg 抓一帧 JPEG，不经过 VST（VST 输出损坏/502）
        rtsp_url = str(self.config["camera"].get("rtsp_url", "")).strip()
        if not rtsp_url:
            raise ValueError("camera.rtsp_url is required for snapshots")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-frames:v", "1",
            "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        ]
        import subprocess
        proc = subprocess.run(command, capture_output=True, timeout=15)
        if proc.returncode != 0 or not proc.stdout:
            detail = proc.stderr.decode(errors="replace")[-300:]
            raise ValueError(f"ffmpeg failed to grab frame: {detail}")
        return proc.stdout

    def cropped_picture(self) -> bytes:
        picture = self.current_picture()
        # VLM 看图区域与占用判定的 chair_roi 解耦：默认整帧（保证能看到人脸/眼睛），
        # 可用 workstation.vlm_roi 覆盖为更聚焦的区域（如人脸附近）。
        polygon = self.workstation.get(
            "vlm_roi",
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        )
        with Image.open(io.BytesIO(picture)) as image:
            image = image.convert("RGB")
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            margin = 0.02
            left = max(0, int((min(xs) - margin) * image.width))
            top = max(0, int((min(ys) - margin) * image.height))
            right = min(image.width, int((max(xs) + margin) * image.width))
            bottom = min(image.height, int((max(ys) + margin) * image.height))
            cropped = image.crop((left, top, right, bottom))
            output = io.BytesIO()
            cropped.save(output, format="JPEG", quality=85)
            return output.getvalue()

    def crop_person_bbox(self, bbox: dict[str, Any]) -> bytes:
        """按检测框（像素坐标）裁取单人的图，供按人 VLM 采样。"""
        picture = self.current_picture()
        with Image.open(io.BytesIO(picture)) as image:
            image = image.convert("RGB")
            margin = 0.08
            left = max(0, int(float(bbox["leftX"]) - margin * image.width))
            top = max(0, int(float(bbox.get("topY", float(bbox["bottomY"]))) - margin * image.height))
            right = min(image.width, int(float(bbox["rightX"]) + margin * image.width))
            bottom = min(image.height, int(float(bbox["bottomY"]) + margin * image.height))
            cropped = image.crop((left, top, right, bottom))
            output = io.BytesIO()
            cropped.save(output, format="JPEG", quality=85)
            return output.getvalue()

    def sample_next_person(self, now: float | None = None) -> dict[str, Any] | None:
        """对当前画面中的各人轮流做 VLM 活动采样（已注册人员优先，每 track 独立 20s 节流）。"""
        current = time.time() if now is None else now
        frame = getattr(self, "last_frame", None)
        if not frame:
            return None
        present: list[str] = []
        for item in frame.get("objects", []):
            if str(item.get("type", "")).casefold() != "person":
                continue
            track_id = str(item.get("id", "")).strip()
            if track_id and track_id not in present:
                present.append(track_id)
        if not present:
            return None
        with closing(self._connect()) as connection:
            mapped = {
                str(row[0]) for row in connection.execute(
                    "SELECT track_id FROM person_track_map"
                ).fetchall()
            }
        # 已注册人员优先采样；当前画面全是未注册 track 时退回全部
        candidates = [t for t in present if t in mapped] or present
        candidates.sort()
        if self._person_sample_index >= len(candidates):
            self._person_sample_index = 0
        track_id = candidates[self._person_sample_index]
        self._person_sample_index += 1
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT last_sampled_at FROM person_activity_state WHERE track_id = ?", (track_id,)
            ).fetchone()
        if row and row[0] and current - float(row[0]) < float(self.workstation.get("person_sample_seconds", 10)):
            return None
        bbox: dict[str, Any] | None = None
        for item in frame.get("objects", []):
            if str(item.get("type", "")).casefold() != "person":
                continue
            if str(item.get("id", "")) == track_id:
                bbox = item.get("bbox") or {}
                break
        if not bbox or "leftX" not in bbox or "rightX" not in bbox or "bottomY" not in bbox:
            return None
        with closing(self._connect()) as connection:
            current_event = self._current_person_event(connection, track_id)
            current_context = None if not current_event else {
                "activity": str(current_event["activity"]),
                "description": str(
                    current_event["description"]
                    or ACTIVITY_LABELS.get(str(current_event["activity"]), "")
                ),
            }
        try:
            image = self.crop_person_bbox(bbox)
            result = self._call_vlm(image, current_context)
        except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
            self.last_vlm_error = str(error)
            return None
        activity = str(result.get("activity", "unknown")).lower()
        if activity not in ACTIVITIES:
            activity = "unknown"
        if activity == "rest" and result.get("eyes_open") is True:
            activity = "computer"
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
        description = str(result.get("description", result.get("detail", "")))[:200]
        continues_current = result.get("continues_current") is True
        camera_id = str(self.config["camera"].get("id", "office-main"))
        with self.lock, closing(self._connect()) as connection, connection:
            self._record_person_activity(
                connection,
                camera_id,
                track_id,
                activity,
                description,
                confidence,
                continues_current,
                current,
            )
        return {
            "track_id": track_id,
            "activity": activity,
            "description": description,
            "continues_current": continues_current,
            "confidence": confidence,
        }

    def person_activity_today(self, now: float | None = None) -> dict[str, Any]:
        """聚合今日各人的在场/进椅/活动区间，返回按人分组的 segments（多 track 归并到同一人）。"""
        current = time.time() if now is None else now
        timezone = ZoneInfo(self.config["timezone"])
        today = datetime.fromtimestamp(current, timezone).date()
        start, end = day_bounds(today, timezone)
        camera_id = str(self.config["camera"].get("id", "office-main"))
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            presence = connection.execute(
                "SELECT track_id, arrived_at, COALESCE(left_at, last_seen_at) AS ended_at FROM presence_sessions "
                "WHERE camera_id = ? AND arrived_at < ? AND COALESCE(left_at, last_seen_at) > ? ORDER BY arrived_at",
                (camera_id, end, start),
            ).fetchall()
            seated = connection.execute(
                "SELECT track_id, started_at, COALESCE(ended_at, ?) AS ended_at FROM person_seated_intervals "
                "WHERE camera_id = ? AND started_at < ? AND COALESCE(ended_at, ?) > ? ORDER BY started_at",
                (current, camera_id, end, current, start),
            ).fetchall()
            activities = connection.execute(
                "SELECT track_id, activity, confidence, started_at, COALESCE(ended_at, ?) AS ended_at "
                "FROM person_activity_intervals "
                "WHERE camera_id = ? AND started_at < ? AND COALESCE(ended_at, ?) > ? ORDER BY started_at",
                (current, camera_id, end, current, start),
            ).fetchall()
            # track -> person 映射（含人员名/参考图）
            mapping = {
                str(row["track_id"]): int(row["person_id"])
                for row in connection.execute("SELECT track_id, person_id FROM person_track_map").fetchall()
            }
            people_rows = {
                int(row["id"]): row
                for row in connection.execute("SELECT id, name, reference_image FROM people").fetchall()
            }
        # 按 person 分组；未映射的 track 各自独立（待判定阶段）
        groups: dict[str, dict[str, Any]] = {}

        def group_key(track_id: str) -> tuple[str, dict[str, Any]]:
            person_id = mapping.get(track_id)
            if person_id is not None:
                row = people_rows.get(person_id)
                name = str(row["name"]) if row else f"人员 {person_id}"
                image = str(row["reference_image"]) if row else ""
                key = f"p{person_id}"
                item = groups.setdefault(key, {"key": key, "label": name, "person_id": person_id, "image": image, "segments": []})
            else:
                key = f"t{track_id}"
                item = groups.setdefault(key, {"key": key, "label": f"人员 {track_id}", "person_id": None, "image": "", "segments": []})
            return key, item

        for row in presence:
            _, item = group_key(str(row["track_id"]))
            item["segments"].append(
                {"kind": "present", "started_at": max(float(row["arrived_at"]), start),
                 "ended_at": min(float(row["ended_at"]), end)}
            )
        for row in seated:
            _, item = group_key(str(row["track_id"]))
            item["segments"].append(
                {"kind": "seated", "started_at": max(float(row["started_at"]), start),
                 "ended_at": min(float(row["ended_at"]), end)}
            )
        for row in activities:
            _, item = group_key(str(row["track_id"]))
            item["segments"].append(
                {"kind": "activity", "activity": str(row["activity"]),
                 "started_at": max(float(row["started_at"]), start),
                 "ended_at": min(float(row["ended_at"]), end)}
            )
        result: list[dict[str, Any]] = []
        # 人坐下后不瞬移：present/seated 融合为连续"在工位"区间，间隙不超过该秒数视为同一段
        presence_merge_gap = max(60.0, float(self.workstation.get("person_presence_merge_seconds", 300)))
        for item in groups.values():
            segments = [s for s in item["segments"] if s["ended_at"] > s["started_at"]]
            segments.sort(key=lambda s: s["started_at"])
            # 先融合在场/在椅 → 连续在工位段
            presence_intervals: list[tuple[float, float]] = []
            for seg in segments:
                if seg["kind"] in ("present", "seated"):
                    presence_intervals.append((seg["started_at"], seg["ended_at"]))
            presence_intervals.sort()
            merged_presence: list[tuple[float, float]] = []
            for seg_start, seg_end in presence_intervals:
                if merged_presence and seg_start - merged_presence[-1][1] <= presence_merge_gap:
                    merged_presence[-1] = (merged_presence[-1][0], max(merged_presence[-1][1], seg_end))
                else:
                    merged_presence.append((seg_start, seg_end))
            # 活动段独立保留（与在工位段重叠展示）
            activity_segments = [
                {"kind": "activity", "activity": seg.get("activity"),
                 "started_at": seg["started_at"], "ended_at": seg["ended_at"]}
                for seg in segments if seg["kind"] == "activity"
            ]
            merged: list[dict[str, Any]] = [
                {"kind": "seated", "activity": None, "started_at": start_at, "ended_at": end_at}
                for start_at, end_at in merged_presence
            ]
            merged.extend(activity_segments)
            merged.sort(key=lambda s: s["started_at"])
            # 工作时长：人在工位上的总时长（present/seated 融合段）——
            # 只要没识别到这个人离开（间隙 <= person_presence_merge_seconds），就一直算在工作
            work_seconds = sum(end - start for start, end in merged_presence)
            # 在场状态：最近 presence_merge_gap 秒内出现过即视为"仍在工位"
            present = any(s["kind"] == "present" and s["ended_at"] >= current - presence_merge_gap for s in segments)
            result.append({
                "track_id": item["key"],
                "person_id": item["person_id"],
                "label": item["label"],
                "image": item["image"],
                "work_seconds": round(work_seconds),
                "present": present,
                "segments": list(merged),
            })
        result.sort(key=lambda p: p["segments"][0]["started_at"] if p["segments"] else 0)
        return {"date": today.isoformat(), "people": result, "working_count": sum(1 for p in result if p["present"])}

    def analyze_activity(self, now: float | None = None) -> dict[str, Any]:
        observed = time.time() if now is None else now
        self.last_vlm_at = observed
        try:
            image = self.cropped_picture()
            result = self._call_vlm(image)
            activity = str(result.get("activity", "unknown")).lower()
            if activity not in ACTIVITIES:
                activity = "unknown"
            # Hard rule: a person with open eyes is never resting, regardless of what the model
            # guessed for the activity. The model explicitly reports eyes_open, so enforce it here.
            if activity == "rest" and result.get("eyes_open") is True:
                activity = "computer"
            confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
            detail = str(result.get("description", result.get("detail", "")))[:300]
            self.last_vlm_error = None
            # VLM 空座兜底：Cosmos3 连续 2 次看到"空座位/无人" → 强制置离座。
            # 占用置信度门槛放宽后，用 VLM 整帧视觉结果纠正误检占用（如后景物体被检出）。
            empty_markers = ("空座位", "空着的", "无人", "没有人", "没人", "empty", "no one", "nobody", "vacant", "unoccupied")
            if any(marker in detail.lower() for marker in empty_markers):
                self._empty_confirmations += 1
            else:
                self._empty_confirmations = 0
            if self._empty_confirmations >= 2:
                self._empty_confirmations = 0
                with closing(self._connect()) as connection, connection:
                    self._set_state(connection, "chair_occupied", "0")
                    connection.execute(
                        "INSERT INTO workstation_samples(observed_at, occupied, data_status, detail) "
                        "VALUES (?, 0, 'healthy', ?)",
                        (observed, f"VLM 空座确认: {detail[:80]}"),
                    )
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

    def _call_vlm(self, image: bytes, current_event: dict[str, str] | None = None) -> dict[str, Any]:
        data_url = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
        current_context = (
            f'The current confirmed event is activity="{current_event["activity"]}", '
            f'description="{current_event["description"]}". '
            if current_event else
            "There is no current confirmed event. "
        )
        prompt = (
            "Analyze only visible evidence about the anonymous person in this office-camera image. "
            + current_context
            + "Return one JSON object and no markdown: "
            '{"activity":"computer|reading|writing|phone|conversation|eating|rest|unknown",'
            '"eyes_open":true,"confidence":0.0,"description":"保守、简短的中文动作描述",'
            '"continues_current":true}. '
            "Set continues_current=true only when the visible action is a continuation of the current confirmed "
            "event; set it false when the action clearly changed or when there is no current event. "
            '"eyes_open" MUST be true whenever the person\'s eyes are visible and open, and false only when the '
            'eyes are clearly closed or covered; '
            "computer means the person is seated and awake with open eyes, facing the workstation/camera "
            "direction — simply sitting at the desk facing the camera counts as working (looking toward the "
            "camera or monitor direction is working, not rest); "
            "the camera is mounted above the monitor at this workstation: looking toward the camera means the "
            "person is looking at the screen, which is working (computer); "
            "reading means reading paper/book; writing means handwriting; "
            "phone means using a phone; conversation means talking with someone; "
            "eating means eating, drinking, or snacking at the desk; "
            "eating with open eyes is NEVER rest and NEVER unknown; "
            "rest means the person's eyes are CLOSED and they are sleeping or dozing; closed eyes are the ONLY "
            "rest condition; "
            "if the eyes are open, the person is awake and NOT resting — open eyes never count as rest, regardless "
            "of posture, gaze direction, or expression (even looking down, slouched, or resting the chin on a hand); "
            "use unknown when unclear. A seated person with open eyes facing the camera or monitor direction must "
            "be classified as computer, not rest — facing the camera counts as working, even if slouched, relaxed, "
            "or resting the chin on a hand. "
            "The description may name only a specific visible action such as typing, using a mouse, looking at a "
            "monitor, handwriting, drinking water, eating, stretching, or tidying the desk, but activity must remain "
            "one of the fixed categories. Be conservative: never invent a document name, project, meeting topic, "
            "website, business purpose, or screen content that is not clearly visible. If the exact task is not "
            "visible, use a general description such as 在电脑前工作. Do not identify the person, read private "
            "screen content, or infer sensitive traits."
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
        min_confidence = float(self.workstation.get("minimum_activity_confidence", 0.6))
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
            # 低置信度分类不推进状态切换（防止 computer+0.2 连续两次就误切换）。
            # sample 仍记录（供 UI 显示 detail），但 pending 计数保持等待更高置信。
            if confidence < min_confidence:
                if activity == current:
                    open_interval = connection.execute(
                        "SELECT 1 FROM workstation_activity_intervals WHERE session_id = ? AND ended_at IS NULL",
                        (session["id"],),
                    ).fetchone()
                    if not open_interval:
                        connection.execute(
                            "INSERT INTO workstation_activity_intervals(session_id, activity, started_at) VALUES (?, ?, ?)",
                            (session["id"], current, observed),
                        )
                return
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

    # ---- 每日活动日志与人员库查询（供 API/前端/Agent 使用） ----

    def activity_events(
        self,
        start: float,
        end: float,
        person_id: int | None = None,
        query: str = "",
        now: float | None = None,
    ) -> dict[str, Any]:
        """Return clipped, searchable activity intervals grouped by recognized person."""
        current = time.time() if now is None else now
        effective_now = min(current, end)
        timezone = ZoneInfo(self.config["timezone"])
        camera_id = str(self.config["camera"].get("id", "office-main"))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT a.*, COALESCE(a.person_id, m.person_id) AS resolved_person_id, p.name AS person_name "
                "FROM person_activity_intervals a "
                "LEFT JOIN person_track_map m ON m.track_id = a.track_id "
                "LEFT JOIN people p ON p.id = COALESCE(a.person_id, m.person_id) "
                "WHERE a.camera_id = ? AND a.started_at < ? "
                "AND COALESCE(a.ended_at, a.last_observed_at, ?) > ? "
                "ORDER BY a.started_at, a.id",
                (camera_id, end, effective_now, start),
            ).fetchall()

        needle = " ".join(query.strip().casefold().split())
        events: list[dict[str, Any]] = []
        for row in rows:
            resolved_person_id = int(row["resolved_person_id"]) if row["resolved_person_id"] is not None else None
            if person_id is not None and resolved_person_id != person_id:
                continue
            activity = str(row["activity"])
            description = str(row["description"] or ACTIVITY_LABELS.get(activity, activity))
            person_name = str(row["person_name"] or f"人员 {resolved_person_id or row['track_id']}")
            searchable = " ".join((person_name, activity, ACTIVITY_LABELS.get(activity, ""), description)).casefold()
            if needle and needle not in searchable:
                continue
            raw_end = (
                float(row["ended_at"])
                if row["ended_at"] is not None
                else float(row["last_observed_at"] or effective_now)
            )
            event_start = max(start, float(row["started_at"]))
            event_end = min(end, max(event_start, raw_end))
            if event_end <= event_start:
                continue
            ongoing = bool(
                row["ended_at"] is None
                and raw_end >= current - float(self.workstation.get("departure_seconds", 60))
            )
            events.append({
                "id": int(row["id"]),
                "person_id": resolved_person_id,
                "person_name": person_name,
                "track_id": str(row["track_id"]),
                "activity": activity,
                "activity_label": ACTIVITY_LABELS.get(activity, activity),
                "description": description,
                "started_at": event_start,
                "ended_at": event_end,
                "duration_seconds": round(event_end - event_start),
                "ongoing": ongoing,
                "confidence": round(float(row["confidence"] or 0), 3),
                "observation_count": max(1, int(row["observation_count"] or 1)),
            })

        # Tracker IDs can change while the same person and action remain visible.
        # Merge identical adjacent events; genuine action changes remain separate.
        merged: list[dict[str, Any]] = []
        merge_gap = max(1.0, float(self.workstation.get("person_sample_seconds", 10)) * 2)
        for event in events:
            previous = next((
                item for item in reversed(merged)
                if item["person_id"] == event["person_id"]
                and (event["person_id"] is not None or item["track_id"] == event["track_id"])
                and item["activity"] == event["activity"]
                and item["description"] == event["description"]
            ), None)
            if previous and event["started_at"] <= previous["ended_at"] + merge_gap:
                total_samples = previous["observation_count"] + event["observation_count"]
                previous["confidence"] = round((
                    previous["confidence"] * previous["observation_count"]
                    + event["confidence"] * event["observation_count"]
                ) / total_samples, 3)
                previous["ended_at"] = max(previous["ended_at"], event["ended_at"])
                previous["duration_seconds"] = round(previous["ended_at"] - previous["started_at"])
                previous["observation_count"] = total_samples
                previous["ongoing"] = previous["ongoing"] or event["ongoing"]
                previous["source_ids"].append(event["id"])
            else:
                event["source_ids"] = [event["id"]]
                merged.append(event)

        summaries: dict[str, dict[str, Any]] = {}
        for event in merged:
            key = str(event["person_id"]) if event["person_id"] is not None else f"track:{event['track_id']}"
            summary = summaries.setdefault(key, {
                "person_id": event["person_id"],
                "person_name": event["person_name"],
                "total_seconds": 0,
                "event_count": 0,
                "categories": {},
            })
            summary["total_seconds"] += event["duration_seconds"]
            summary["event_count"] += 1
            categories = summary["categories"]
            categories[event["activity"]] = categories.get(event["activity"], 0) + event["duration_seconds"]

        return {
            "timezone": str(timezone),
            "start": datetime.fromtimestamp(start, timezone).isoformat(),
            "end": datetime.fromtimestamp(end, timezone).isoformat(),
            "people": sorted(summaries.values(), key=lambda item: str(item["person_name"])),
            "events": merged,
            "event_count": len(merged),
            "total_seconds": sum(int(item["duration_seconds"]) for item in merged),
        }

    def people_list(self, now: float | None = None) -> list[dict[str, Any]]:
        """已注册人员列表：名字、参考图（相对路径）、首见/末见时间、关联 track 数。"""
        current = time.time() if now is None else now
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT p.id, p.name, p.reference_image, p.first_seen_at, p.last_seen_at, "
                "(SELECT COUNT(*) FROM person_track_map m WHERE m.person_id = p.id) AS track_count "
                "FROM people p WHERE p.active = 1 ORDER BY p.last_seen_at DESC"
            ).fetchall()
        return [
            {
                "person_id": int(row["id"]),
                "name": str(row["name"]),
                "image": str(row["reference_image"]),
                "first_seen_at": float(row["first_seen_at"]),
                "last_seen_at": float(row["last_seen_at"]),
                "track_count": int(row["track_count"]),
            }
            for row in rows
        ]

    def person_image(self, person_id: int) -> Path | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT reference_image FROM people WHERE id = ? AND active = 1", (person_id,)
            ).fetchone()
        if not row or not row[0]:
            return None
        path = self.database_file.parent / row[0]
        return path if path.is_file() else None

    def delete_person(self, person_id: int) -> bool:
        with self.lock, closing(self._connect()) as connection, connection:
            cursor = connection.execute("UPDATE people SET active = 0 WHERE id = ?", (person_id,))
            if cursor.rowcount == 0:
                return False
            connection.execute(
                "UPDATE person_activity_intervals SET person_id = ? WHERE person_id IS NULL "
                "AND track_id IN (SELECT track_id FROM person_track_map WHERE person_id = ?)",
                (person_id, person_id),
            )
            connection.execute(
                "DELETE FROM person_track_map WHERE person_id = ?", (person_id,)
            )
        return True

    def rename_person(self, person_id: int, name: str) -> bool:
        name = str(name).strip()[:50]
        if not name:
            return False
        with self.lock, closing(self._connect()) as connection, connection:
            cursor = connection.execute("UPDATE people SET name = ? WHERE id = ?", (name, person_id))
            return cursor.rowcount > 0


def worker(engine: WorkstationEngine, fetch_frame: Any, poll_seconds: float = 2.0) -> None:
    while True:
        try:
            engine.process_frame(fetch_frame())
            if engine.needs_vlm_sample():
                engine.analyze_activity()
            engine.sample_next_person()
            engine._drain_person_pending()  # noqa: SLF001 - worker 主循环异步消费新 track 判定
            engine.mark_overtime()
            engine.archive_pending_clips()
            if int(time.time()) % 3600 < max(1, int(poll_seconds)):
                engine.cleanup()
        except Exception as error:  # Worker must remain alive across dependent-service outages.
            print(f"[office-api] workstation worker failed: {error}", flush=True)
        time.sleep(max(0.5, poll_seconds))
