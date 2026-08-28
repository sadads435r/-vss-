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

from motion import (
    MediaPipeHandAnalyzer,
    MediaPipePoseAnalyzer,
    RtspFrameBuffer,
    build_pose_observation,
    build_storyboard,
    cosine_similarity,
    crop_person_frames,
    extract_embedding,
    facts_json,
    parse_pose,
    parse_timestamp,
    select_storyboard_frames,
    summarize_motion,
)

ACTIVITIES = ("computer", "reading", "writing", "phone", "conversation", "eating", "rest", "unknown")
ACTIVITY_LABELS = {
    "computer": "使用电脑",
    "reading": "阅读",
    "writing": "书写",
    "phone": "使用手机",
    "conversation": "交谈",
    "eating": "吃东西",
    "rest": "睡眠",
    "unknown": "其他动作",
    "left_workstation": "离开工位",
    "returned_to_workstation": "返回工位",
}

# Negative lifecycle IDs normally encode a motion-window ID.  Keep inferred
# presence transitions in a separate range so their detail endpoint can expose
# the seated-interval evidence without pretending that a VLM saw the departure.
PRESENCE_LIFECYCLE_ID_BASE = 1_000_000_000_000

WORKSTATION_COMPUTER_EXPLICIT = (
    "电脑", "屏幕", "键盘", "鼠标", "触控板", "笔记本电脑", "桌面",
)
WORKSTATION_COMPUTER_POSTURE = (
    "坐姿", "坐在", "办公椅", "身体前倾", "上半身", "低头", "后仰",
    "头部", "手部", "右手", "左手", "双手", "手臂", "双臂",
)
WORKSTATION_COMPUTER_BLOCKERS = (
    "站起", "起身", "行走", "走动", "离开", "手机", "进食", "吃", "饮水",
    "喝水", "交谈", "说话", "闭眼", "打盹", "睡眠",
)

TARGET_ACTION_FORBIDDEN = (
    "背景", "环境", "办公室", "开放式", "工位区", "其他人", "另一人", "另一名", "另一个",
    "同事", "周围", "后方", "远处", "两人", "多人", "戴眼镜", "身穿", "穿着", "中年",
    "年轻", "男子", "女士", "男士", "女子", "光线", "镜头前", "似乎", "看起来", "氛围",
)


def clean_target_actions(value: Any) -> list[str]:
    """Keep short target-person actions and reject appearance, scene, and bystander text."""
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        action = " ".join(str(item).split()).strip(" ，。；、")
        if not action or any(marker in action for marker in TARGET_ACTION_FORBIDDEN):
            continue
        if action not in cleaned:
            cleaned.append(action[:80])
    return cleaned[:4]


def infer_activity_from_actions(
    category: str, actions: list[str], current_category: str = "",
) -> str:
    """Repair an unknown VLM label only when its own visible actions are explicit."""
    if category != "unknown" or not actions:
        return category
    text = "；".join(actions).casefold()
    keyword_groups = (
        ("phone", ("手机", "移动电话")),
        ("computer", ("电脑", "屏幕", "键盘", "鼠标", "笔记本电脑")),
        ("reading", ("阅读", "读书", "翻书", "纸张", "书本")),
        ("writing", ("手写", "书写", "写字", "记笔记")),
        ("eating", ("进食", "吃", "饮水", "喝水", "杯子", "餐具")),
        ("conversation", ("开口说话", "持续说话", "交谈手势")),
    )
    for inferred, keywords in keyword_groups:
        if any(keyword in text for keyword in keywords):
            return inferred
    # At a calibrated workstation the monitor itself is often outside the tight
    # person crop. Seated desk posture and small head/arm adjustments are still
    # part of computer use unless a visible, incompatible action says otherwise.
    if not any(keyword in text for keyword in WORKSTATION_COMPUTER_BLOCKERS):
        if any(keyword in text for keyword in WORKSTATION_COMPUTER_EXPLICIT):
            return "computer"
        posture_hits = sum(keyword in text for keyword in WORKSTATION_COMPUTER_POSTURE)
        if posture_hits >= 2 or (current_category == "computer" and posture_hits >= 1):
            return "computer"
    return category


def clean_target_description(value: Any, fallback: str = "动作无法确定") -> str:
    """Sanitize legacy free text before returning it through the activity API."""
    clauses = [" ".join(str(value or "").split())]
    for delimiter in ("。", "；", "，", "\n"):
        clauses = [piece for clause in clauses for piece in clause.split(delimiter)]
    cleaned = clean_target_actions(clauses)
    return "；".join(cleaned)[:300] if cleaned else fallback


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
        # Continuous motion pipeline. The legacy single-image classifier remains the rollback path.
        "motion_pipeline": {
            "enabled": True,
            "pose_source": "mediapipe",
            "mediapipe_pose_model": "/models/mediapipe/pose_landmarker_lite.task",
            "pose_frame_tolerance_seconds": 1.5,
            "window_seconds": 8,
            "step_seconds": 2,
            "semantic_interval_seconds": 10,
            "minimum_semantic_interval_seconds": 5,
            "pose_minimum_confidence": 0.35,
            "frame_buffer_seconds": 120,
            "frame_buffer_fps": 2,
            "person_storyboard_frames": 6,
            "scene_storyboard_frames": 4,
            "mediapipe_hand_model": "/models/mediapipe/hand_landmarker.task",
            "storyboard_retention_days": 7,
            "queue_limit": 100,
        },
        "identity": {
            "confirmation_samples": 2,
            "cosmos_match_confidence": 0.85,
            "strong_match_confidence": 0.95,
            "strong_reid_similarity": 0.80,
            "known_match_retry_seconds": 0.5,
            "pending_stale_seconds": 6.0,
            "pending_limit": 24,
            "low_quality_retry_seconds": 15.0,
            "fast_known_match_confidence": 0.95,
            "fast_known_person_confidence": 0.95,
            "fast_known_minimum_quality": 0.65,
            "fast_known_recent_seconds": 1800,
            "candidate_limit": 3,
            "gallery_limit": 5,
            "minimum_image_height": 256,
            "minimum_detection_confidence": 0.3,
            "minimum_visible_keypoints_ratio": 0.8,
            "gallery_diversity_similarity": 0.95,
        },
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


def allow_fast_known_match(
    *,
    match_confidence: float,
    verdict_confidence: float,
    quality: float,
    in_chair_roi: bool,
    seconds_since_target_seen: float,
    identity: dict[str, Any],
) -> bool:
    """Allow one-sample binding only for recent, high-quality known identity evidence."""
    return bool(
        in_chair_roi
        and match_confidence >= float(identity.get("fast_known_match_confidence", 0.95))
        and verdict_confidence >= float(identity.get("fast_known_person_confidence", 0.95))
        and quality >= float(identity.get("fast_known_minimum_quality", 0.65))
        and seconds_since_target_seen <= float(identity.get("fast_known_recent_seconds", 1800))
    )


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
    motion = workstation.get("motion_pipeline", {})
    if not isinstance(motion, dict):
        raise ValueError("workstation.motion_pipeline must be a mapping")
    if float(motion.get("window_seconds", 8)) < 4:
        raise ValueError("workstation.motion_pipeline.window_seconds must be at least 4")
    if float(motion.get("step_seconds", 2)) < 1:
        raise ValueError("workstation.motion_pipeline.step_seconds must be at least 1")
    if float(motion.get("minimum_semantic_interval_seconds", 5)) < 1:
        raise ValueError("workstation.motion_pipeline.minimum_semantic_interval_seconds must be at least 1")
    identity = workstation.get("identity", {})
    if not isinstance(identity, dict) or int(identity.get("confirmation_samples", 2)) < 2:
        raise ValueError("workstation.identity.confirmation_samples must be at least 2")


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
        connection.execute(
            "CREATE TABLE IF NOT EXISTS person_merge_history ("
            "source_person_id INTEGER PRIMARY KEY, target_person_id INTEGER NOT NULL, "
            "source_name TEXT NOT NULL, target_name TEXT NOT NULL, "
            "migrated_json TEXT NOT NULL DEFAULT '{}', merged_at REAL NOT NULL)"
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
        _ensure_columns(connection, "person_verifications", {
            "decision": "TEXT NOT NULL DEFAULT ''",
            "candidate_person_id": "INTEGER",
            "reid_similarity": "REAL NOT NULL DEFAULT 0",
            "candidate_json": "TEXT NOT NULL DEFAULT '[]'",
            "quality_score": "REAL NOT NULL DEFAULT 0",
        })
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_person_verification_track_time "
            "ON person_verifications(track_id, created_at)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS person_reference_images ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER NOT NULL, path TEXT NOT NULL UNIQUE, "
            "captured_at REAL NOT NULL, quality_score REAL NOT NULL DEFAULT 0, "
            "embedding_json TEXT NOT NULL DEFAULT '[]', is_cover INTEGER NOT NULL DEFAULT 0, "
            "active INTEGER NOT NULL DEFAULT 1)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_reference_person_active "
            "ON person_reference_images(person_id, active, quality_score DESC)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS person_motion_windows ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL, track_id TEXT NOT NULL, "
            "person_id INTEGER, started_at REAL NOT NULL, ended_at REAL NOT NULL, "
            "facts_json TEXT NOT NULL, motion_summary TEXT NOT NULL DEFAULT '', "
            "person_storyboard TEXT NOT NULL DEFAULT '', scene_storyboard TEXT NOT NULL DEFAULT '', "
            "hand_json TEXT NOT NULL DEFAULT '{}', quality REAL NOT NULL DEFAULT 0, "
            "status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0, "
            "next_retry_at REAL NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL, "
            "UNIQUE(camera_id, track_id, started_at, ended_at))"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_motion_status_retry "
            "ON person_motion_windows(status, next_retry_at, ended_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_motion_person_time "
            "ON person_motion_windows(person_id, started_at)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS person_activity_observations ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, interval_id INTEGER, motion_window_id INTEGER NOT NULL UNIQUE, "
            "category TEXT NOT NULL, description TEXT NOT NULL, observed_actions_json TEXT NOT NULL DEFAULT '[]', "
            "continues_current INTEGER NOT NULL DEFAULT 0, confidence REAL NOT NULL DEFAULT 0, "
            "uncertainty TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS motion_frame_tokens ("
            "token TEXT PRIMARY KEY, observed_at REAL NOT NULL)"
        )
        # Upgrade legacy one-image profiles into the reference gallery exactly once.
        connection.execute(
            "INSERT OR IGNORE INTO person_reference_images("
            "person_id, path, captured_at, quality_score, is_cover) "
            "SELECT id, reference_image, last_seen_at, 0.5, 1 FROM people WHERE reference_image != ''"
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
        self.storyboard_dir = database_file.parent / "storyboards"
        self.storyboard_dir.mkdir(parents=True, exist_ok=True)
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
        # 已通过一次 VLM 身份确认的 track 优先复核，并只复核上次命中的人物。
        self._person_match_priority: dict[str, int] = {}
        # Synchronized per-track pose samples are kept in memory only; aggregate windows are persisted.
        self._motion_samples: dict[str, list[dict[str, Any]]] = {}
        self._motion_last_window_at: dict[str, float] = {}
        self._motion_last_semantic_at: dict[str, float] = {}
        self.frame_buffer: RtspFrameBuffer | None = None
        self.hand_analyzer: MediaPipeHandAnalyzer | None = None
        self.pose_analyzer: MediaPipePoseAnalyzer | None = None
        self._motion_last_pose_frame_at: dict[str, float] = {}
        # The dedicated motion worker may attach a non-blocking Kafka sink. The API
        # process leaves this unset, so pose collection cannot affect request handling.
        self.pose_observer: Any | None = None
        self.pose_observe_hands = True
        self.pose_observation_interval_seconds = 0.5
        self._motion_last_observation_at: dict[str, float] = {}
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
        pending: list[tuple[str, dict[str, Any], bool]] = []
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
            mapped = connection.execute(
                "SELECT person_id FROM person_track_map WHERE track_id = ?", (track_id,)
            ).fetchone()
            if mapped:
                connection.execute("UPDATE people SET last_seen_at = ? WHERE id = ?", (observed, int(mapped[0])))
            in_chair_roi = point_in_polygon(foot, polygon)
            if in_chair_roi:
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
                    pending.append((track_id, dict(item), in_chair_roi))
        for track_id in list(self.person_seated):
            if track_id not in seen:
                self._close_person_seated(connection, track_id, observed)
        self._close_departed_person_activities(connection, seen, observed)
        # 达标的新 track 进入内存待判队列，由 worker 主循环异步消费（VLM 判定不能占用 DB 事务）
        for track_id, item, in_chair_roi in pending:
            previous = self._person_pending.get(track_id, {})
            item["_identity_in_chair_roi"] = in_chair_roi
            item["_identity_last_seen_at"] = observed
            item["_identity_enqueued_at"] = float(previous.get("_identity_enqueued_at", observed))
            bbox = item.get("bbox") or {}
            item["_identity_bbox_height"] = max(
                0.0, float(bbox.get("bottomY", 0)) - float(bbox.get("topY", 0))
            )
            # Replace stale screenshots for the same tracker with the newest crop.
            self._person_pending[track_id] = item
        identity = self.workstation.get("identity", {})
        pending_limit = max(1, int(identity.get("pending_limit", 24)))
        if len(self._person_pending) > pending_limit:
            ranked = sorted(
                self._person_pending.items(),
                key=lambda pair: (
                    pair[0] in self._person_match_priority,
                    bool(pair[1].get("_identity_in_chair_roi", False)),
                    float(pair[1].get("_identity_last_seen_at", 0)),
                    float(pair[1].get("_identity_bbox_height", 0)),
                ),
                reverse=True,
            )[:pending_limit]
            self._person_pending = dict(ranked)

    def _drain_person_pending(self, now: float | None = None) -> None:
        """worker 主循环调用：每次处理一个待判新 track（VLM 判定耗时，异步执行）。"""
        current = time.time() if now is None else now
        identity = self.workstation.get("identity", {})
        stale_seconds = max(1.0, float(identity.get("pending_stale_seconds", 6.0)))
        for track_id, item in list(self._person_pending.items()):
            if current - float(item.get("_identity_last_seen_at", current)) > stale_seconds:
                self._person_pending.pop(track_id, None)
                self._person_match_priority.pop(track_id, None)
        ranked = sorted(
            self._person_pending.items(),
            key=lambda pair: (
                pair[0] in self._person_match_priority,
                bool(pair[1].get("_identity_in_chair_roi", False)),
                float(pair[1].get("_identity_last_seen_at", 0)),
                float(pair[1].get("_identity_bbox_height", 0)),
            ),
            reverse=True,
        )
        for track_id, item in ranked:
            cooldown = max(0.0, float(self.workstation.get("person_reject_cooldown_seconds", 600)))
            # 已入库/被拒冷却中的 track 跳过
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT 1 FROM person_track_map WHERE track_id = ?", (track_id,)
                ).fetchone()
                rejected = current < self._person_reject_until.get(track_id, 0.0)
            if row is not None or rejected:
                self._person_pending.pop(track_id, None)
                self._person_match_priority.pop(track_id, None)
                continue
            verify_seconds = max(0.0, float(self.workstation.get("person_verify_seconds", 3.0)))
            retry_seconds = (
                max(0.0, float(identity.get("known_match_retry_seconds", 0.5)))
                if track_id in self._person_match_priority else verify_seconds
            )
            if current - self._person_last_verify_at.get(track_id, 0.0) < retry_seconds:
                continue
            self._person_last_verify_at[track_id] = current
            self._person_pending.pop(track_id, None)
            self._verify_new_track(track_id, item, current)
            return  # 每轮只处理一个，避免 VLM 长时间阻塞 worker

    def _verify_new_track(self, track_id: str, item: dict[str, Any], observed: float) -> None:
        """Verify a new track twice before matching or enrolling it."""
        reject_cooldown = max(0.0, float(self.workstation.get("person_reject_cooldown_seconds", 600)))
        bbox = item.get("bbox") or {}
        embedding = extract_embedding(item)
        try:
            image = self.crop_person_bbox(bbox)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"[office-api] person verify crop failed track={track_id}: {error}", flush=True)
            return
        quality = self._person_image_quality(image, item)
        if quality <= 0:
            identity = self.workstation.get("identity", {})
            retry_seconds = max(1.0, float(identity.get("low_quality_retry_seconds", 15.0)))
            with self.lock, closing(self._connect()) as connection, connection:
                connection.execute(
                    "INSERT INTO person_verifications("
                    "track_id, is_person, confidence, reason, decision, quality_score, created_at) "
                    "VALUES (?, 1, 0, '截图未达到建档质量阈值', 'low_quality', 0, ?)",
                    (track_id, observed),
                )
            # Reject locally before VLM; retry later in case pose or visibility improves.
            self._person_reject_until[track_id] = observed + retry_seconds
            self._person_match_priority.pop(track_id, None)
            return
        preferred_person_id = self._person_match_priority.get(track_id)
        # A second confirmation for the same live tracker reuses the first human
        # verdict and spends VLM time only on the previously matched identity.
        if preferred_person_id is not None:
            verdict = {"is_person": True, "confidence": 1.0, "reason": "已通过首轮真人验证"}
        else:
            try:
                verdict = self._call_vlm_verdict(image)
            except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
                self.last_vlm_error = str(error)
                print(f"[office-api] person verify VLM failed track={track_id}: {error}", flush=True)
                return
        is_person = bool(verdict.get("is_person", False))
        verdict_confidence = max(0.0, min(1.0, float(verdict.get("confidence", 0))))
        reason = str(verdict.get("reason", ""))[:200]
        with self.lock, closing(self._connect()) as connection, connection:
            if not is_person:
                connection.execute(
                    "INSERT INTO person_verifications("
                    "track_id, is_person, confidence, reason, decision, quality_score, created_at) "
                    "VALUES (?, 0, ?, ?, 'rejected', ?, ?)",
                    (track_id, verdict_confidence, reason, quality, observed),
                )
                # 非真人：冷却期内不再重复判定（布偶/海报类稳定误检）
                self._person_reject_until[track_id] = observed + reject_cooldown
                self._person_match_priority.pop(track_id, None)
                return
            (
                matched_person_id, match_confidence, nonmatch_confidence,
                reid_similarity, candidates, match_reason,
            ) = self._match_person_identity(
                image, embedding, connection, preferred_person_id=preferred_person_id,
            )
            identity = self.workstation.get("identity", {})
            required = max(2, int(identity.get("confirmation_samples", 2)))
            threshold = float(identity.get("cosmos_match_confidence", 0.85))
            if matched_person_id is not None and match_confidence >= threshold:
                decision = "match_candidate"
                decision_confidence = match_confidence
            elif matched_person_id is None and (not candidates or nonmatch_confidence >= threshold):
                decision = "new_candidate"
                decision_confidence = nonmatch_confidence if candidates else verdict_confidence
            else:
                decision = "ambiguous"
                decision_confidence = max(match_confidence, nonmatch_confidence)
            connection.execute(
                "INSERT INTO person_verifications("
                "track_id, is_person, matched_person_id, confidence, reason, image_path, created_at, decision, "
                "candidate_person_id, reid_similarity, candidate_json, quality_score) "
                "VALUES (?, 1, ?, ?, ?, '', ?, ?, ?, ?, ?, ?)",
                (
                    track_id, matched_person_id, decision_confidence,
                    match_reason or reason, observed, decision, matched_person_id, reid_similarity,
                    json.dumps(candidates, ensure_ascii=False), quality,
                ),
            )
            if decision == "ambiguous":
                return
            if decision == "match_candidate":
                self._person_match_priority[track_id] = int(matched_person_id)
                confirmations = connection.execute(
                    "SELECT COUNT(DISTINCT CAST(created_at AS INTEGER)) FROM person_verifications WHERE track_id = ? "
                    "AND decision = 'match_candidate' AND candidate_person_id = ? AND confidence >= ?",
                    (track_id, matched_person_id, threshold),
                ).fetchone()[0]
                strong_match = (
                    match_confidence >= float(identity.get("strong_match_confidence", 0.95))
                    and reid_similarity >= float(identity.get("strong_reid_similarity", 0.80))
                )
                target_seen = connection.execute(
                    "SELECT last_seen_at FROM people WHERE id = ? AND active = 1",
                    (matched_person_id,),
                ).fetchone()
                fast_known_match = bool(target_seen) and allow_fast_known_match(
                    match_confidence=match_confidence,
                    verdict_confidence=verdict_confidence,
                    quality=quality,
                    in_chair_roi=bool(item.get("_identity_in_chair_roi", False)),
                    seconds_since_target_seen=max(0.0, observed - float(target_seen[0])),
                    identity=identity,
                )
                if int(confirmations) < (1 if strong_match or fast_known_match else required):
                    return
                image_path = self._store_person_image(track_id, image)
                connection.execute(
                    "INSERT INTO person_track_map(track_id, person_id, matched_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(track_id) DO UPDATE SET person_id=excluded.person_id, matched_at=excluded.matched_at",
                    (track_id, matched_person_id, observed),
                )
                connection.execute(
                    "UPDATE people SET last_seen_at = ? WHERE id = ?", (observed, matched_person_id)
                )
                connection.execute(
                    "UPDATE person_verifications SET image_path = ?, decision = 'matched' WHERE track_id = ? "
                    "AND decision = 'match_candidate'",
                    (image_path, track_id),
                )
                self._add_reference_image(connection, int(matched_person_id), image_path, observed, quality, embedding)
                self._backfill_person(connection, track_id, int(matched_person_id), observed)
                self._person_match_priority.pop(track_id, None)
                print(f"[office-api] track {track_id} -> 人员 #{matched_person_id} (匹配), image={image_path}", flush=True)
                return
            self._person_match_priority.pop(track_id, None)
            confirmations = connection.execute(
                "SELECT COUNT(DISTINCT CAST(created_at AS INTEGER)) FROM person_verifications WHERE track_id = ? "
                "AND decision = 'new_candidate' AND is_person = 1 AND confidence >= ?",
                (track_id, threshold),
            ).fetchone()[0]
            if int(confirmations) < required:
                return
            image_path = self._store_person_image(track_id, image)
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
                "UPDATE person_verifications SET matched_person_id = ?, image_path = ?, decision = 'enrolled' "
                "WHERE track_id = ? AND decision = 'new_candidate'",
                (person_id, image_path, track_id),
            )
            self._add_reference_image(connection, person_id, image_path, observed, quality, embedding, is_cover=True)
            self._backfill_person(connection, track_id, person_id, observed)
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

    def _match_person_identity(
        self,
        image: bytes,
        embedding: list[float],
        connection: sqlite3.Connection,
        preferred_person_id: int | None = None,
    ) -> tuple[int | None, float, float, float, list[dict[str, Any]], str]:
        """Use RT-CV ReID for top-k recall and Cosmos only for final identity confirmation."""
        identity = self.workstation.get("identity", {})
        limit = max(1, int(identity.get("candidate_limit", 3)))
        rows = connection.execute(
            "SELECT p.id, p.name, p.last_seen_at, r.path, r.embedding_json, r.quality_score "
            "FROM people p JOIN person_reference_images r ON r.person_id = p.id "
            "WHERE p.active = 1 AND r.active = 1 ORDER BY p.last_seen_at DESC, r.quality_score DESC"
        ).fetchall()
        profiles: dict[int, dict[str, Any]] = {}
        for row in rows:
            person_id = int(row["id"])
            profile = profiles.setdefault(person_id, {
                "person_id": person_id,
                "name": str(row["name"]),
                "last_seen_at": float(row["last_seen_at"]),
                "similarity": 0.0,
                "references": [],
            })
            try:
                reference_embedding = json.loads(str(row["embedding_json"] or "[]"))
            except json.JSONDecodeError:
                reference_embedding = []
            similarity = cosine_similarity(embedding, reference_embedding)
            profile["similarity"] = max(float(profile["similarity"]), similarity)
            profile["references"].append((str(row["path"]), float(row["quality_score"])))
        if preferred_person_id is not None and preferred_person_id in profiles:
            candidates = [profiles[preferred_person_id]]
        else:
            candidates = sorted(
                profiles.values(),
                key=lambda value: (float(value["similarity"]), float(value["last_seen_at"])),
                reverse=True,
            )[:limit]
        public_candidates = [
            {"person_id": item["person_id"], "name": item["name"], "reid_similarity": round(item["similarity"], 4)}
            for item in candidates
        ]
        best_id: int | None = None
        best_score = 0.0
        best_similarity = 0.0
        best_reason = ""
        mismatch_scores: list[float] = []
        for candidate in candidates:
            references = sorted(candidate["references"], key=lambda value: value[1], reverse=True)[:3]
            candidate_score = 0.0
            candidate_reason = ""
            candidate_mismatch = 0.0
            compared = False
            for path_value, _quality in references:
                reference = self.database_file.parent / path_value
                if not reference.is_file():
                    continue
                try:
                    result = self._call_vlm_identity(reference.read_bytes(), image)
                except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
                    self.last_vlm_error = str(error)
                    continue
                compared = True
                if bool(result.get("same_person", False)):
                    score = max(0.0, min(1.0, float(result.get("confidence", 0))))
                    if score > candidate_score:
                        candidate_score = score
                        candidate_reason = str(result.get("reason", ""))[:200]
                else:
                    candidate_mismatch = max(
                        candidate_mismatch,
                        max(0.0, min(1.0, float(result.get("confidence", 0)))),
                    )
            if compared and candidate_score == 0:
                mismatch_scores.append(candidate_mismatch)
            if candidate_score > best_score:
                best_id = int(candidate["person_id"])
                best_score = candidate_score
                best_similarity = float(candidate["similarity"])
                best_reason = candidate_reason
        nonmatch_confidence = (
            min(mismatch_scores)
            if best_id is None and len(mismatch_scores) == len(candidates) and candidates
            else 0.0
        )
        return best_id, best_score, nonmatch_confidence, best_similarity, public_candidates, best_reason

    def _person_image_quality(self, image: bytes, item: dict[str, Any]) -> float:
        """Score enrollment evidence without using identity or sensitive attributes."""
        identity = self.workstation.get("identity", {})
        bbox = item.get("bbox") or {}
        height = max(0.0, float(bbox.get("bottomY", 0)) - float(bbox.get("topY", 0)))
        confidence = max(float(item.get("confidence", 0)), float(bbox.get("confidence", 0)))
        pose = parse_pose(item)
        visible = sum(point.confidence >= 0.35 for point in pose.values()) / 34 if pose else 1.0
        with Image.open(io.BytesIO(image)) as source:
            grayscale = source.convert("L").resize((64, 64))
            pixels = list(grayscale.getdata())
        mean = sum(pixels) / max(1, len(pixels))
        contrast = min(1.0, (sum((value - mean) ** 2 for value in pixels) / max(1, len(pixels))) ** 0.5 / 64)
        # DeepStream rtdetr protobuf emits confidence = -0.1 (sentinel) when the
        # detector does not provide a real confidence. Treat non-positive values as
        # "no confidence reported" and skip that gate instead of rejecting the
        # enrollment evidence outright. Real rtdetr person confidences are commonly
        # in the 0.3-0.6 band, so the gate uses minimum_detection_confidence (0.3)
        # as a coarse filter; precision is left to the VLM person verdict.
        valid_confidence = confidence > 0.0
        passed = (
            height >= float(identity.get("minimum_image_height", 256))
            and (not valid_confidence or confidence >= float(identity.get("minimum_detection_confidence", 0.3)))
            and visible >= float(identity.get("minimum_visible_keypoints_ratio", 0.8))
        )
        score_confidence = max(0.0, confidence)
        return round((0.4 * min(1.0, height / 512) + 0.3 * score_confidence + 0.2 * visible + 0.1 * contrast) if passed else 0.0, 4)

    def _add_reference_image(
        self,
        connection: sqlite3.Connection,
        person_id: int,
        path: str,
        captured_at: float,
        quality: float,
        embedding: list[float],
        is_cover: bool = False,
    ) -> None:
        identity = self.workstation.get("identity", {})
        if quality <= 0:
            return
        existing = connection.execute(
            "SELECT id, embedding_json FROM person_reference_images WHERE person_id = ? AND active = 1",
            (person_id,),
        ).fetchall()
        diversity = float(identity.get("gallery_diversity_similarity", 0.95))
        for row in existing:
            try:
                stored = json.loads(str(row["embedding_json"] or "[]"))
            except json.JSONDecodeError:
                stored = []
            if stored and embedding and cosine_similarity(stored, embedding) >= diversity:
                return
        if is_cover:
            connection.execute("UPDATE person_reference_images SET is_cover = 0 WHERE person_id = ?", (person_id,))
        connection.execute(
            "INSERT OR IGNORE INTO person_reference_images("
            "person_id, path, captured_at, quality_score, embedding_json, is_cover) VALUES (?, ?, ?, ?, ?, ?)",
            (person_id, path, captured_at, quality, json.dumps(embedding), int(is_cover)),
        )
        limit = max(1, int(identity.get("gallery_limit", 5)))
        rows = connection.execute(
            "SELECT id FROM person_reference_images WHERE person_id = ? AND active = 1 "
            "ORDER BY is_cover DESC, quality_score DESC, captured_at DESC",
            (person_id,),
        ).fetchall()
        for row in rows[limit:]:
            connection.execute("UPDATE person_reference_images SET active = 0, is_cover = 0 WHERE id = ?", (int(row[0]),))
        cover = connection.execute(
            "SELECT path FROM person_reference_images WHERE person_id = ? AND active = 1 "
            "ORDER BY is_cover DESC, quality_score DESC LIMIT 1",
            (person_id,),
        ).fetchone()
        if cover:
            connection.execute("UPDATE people SET reference_image = ? WHERE id = ?", (str(cover[0]), person_id))

    @staticmethod
    def _backfill_person(
        connection: sqlite3.Connection, track_id: str, person_id: int, observed: float,
    ) -> None:
        """Backfill only this track lifecycle; tracker IDs can be reused after a restart."""
        lifecycle_start = observed - 600
        lifecycle_end = observed + 60
        connection.execute(
            "UPDATE person_activity_intervals SET person_id = ? WHERE track_id = ? AND person_id IS NULL "
            "AND COALESCE(last_observed_at, started_at) >= ? AND started_at <= ?",
            (person_id, track_id, lifecycle_start, lifecycle_end),
        )
        connection.execute(
            "UPDATE person_motion_windows SET person_id = ? WHERE track_id = ? AND person_id IS NULL "
            "AND ended_at >= ? AND started_at <= ?",
            (person_id, track_id, lifecycle_start, lifecycle_end),
        )

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
        self.person_seated.pop(track_id, None)

    def _close_departed_person_activities(
        self, connection: sqlite3.Connection, seen: set[str], observed: float,
    ) -> None:
        """Close events only after a track is absent, not merely outside the chair ROI."""
        departure = float(self.workstation.get("departure_seconds", 60))
        rows = connection.execute(
            "SELECT DISTINCT track_id, MAX(COALESCE(last_observed_at, started_at)) AS last_seen "
            "FROM person_activity_intervals WHERE ended_at IS NULL GROUP BY track_id"
        ).fetchall()
        for row in rows:
            track_id = str(row["track_id"])
            last_seen = float(row["last_seen"])
            if track_id not in seen and observed - last_seen >= departure:
                connection.execute(
                    "UPDATE person_activity_intervals SET ended_at = COALESCE(last_observed_at, ?) "
                    "WHERE track_id = ? AND ended_at IS NULL",
                    (last_seen, track_id),
                )

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
    ) -> int | None:
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

        # An unknown category can still carry useful, directly observed motion.
        # Keep those windows in the timeline; discard only genuinely empty evidence.
        meaningful_unknown = (
            activity == "unknown"
            and description not in {"动作无法确定", ACTIVITY_LABELS["unknown"]}
        )
        valid = confidence >= minimum_conf and (activity != "unknown" or meaningful_unknown)
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
        current_event = self._current_person_event(connection, track_id)
        return int(current_event["id"]) if current_event else None

    def enable_motion_runtime(self) -> None:
        """Start evidence buffering only in the dedicated motion worker process."""
        if self.frame_buffer is None:
            motion = self.workstation.get("motion_pipeline", {})
            self.frame_buffer = RtspFrameBuffer(
                str(self.config["camera"].get("rtsp_url", "")),
                retention_seconds=float(motion.get(
                    "frame_buffer_seconds",
                    self.config.get("retention", {}).get("transient_buffer_seconds", 120),
                )),
                fps=float(motion.get("frame_buffer_fps", 2)),
            )
            self.frame_buffer.start()
        if self.hand_analyzer is None:
            model_path = str(self.workstation.get("motion_pipeline", {}).get("mediapipe_hand_model", ""))
            self.hand_analyzer = MediaPipeHandAnalyzer(model_path)
        if self.pose_analyzer is None:
            model_path = str(self.workstation.get("motion_pipeline", {}).get("mediapipe_pose_model", ""))
            self.pose_analyzer = MediaPipePoseAnalyzer(model_path)

    def process_motion_frame(
        self, frame: dict[str, Any], now: float | None = None, observation_only: bool = False,
    ) -> int:
        """Combine RT-CV bbox/ReID with MediaPipe pose and persist due motion windows."""
        if not self.workstation.get("motion_pipeline", {}).get("enabled", True):
            return 0
        current = time.time() if now is None else now
        try:
            observed = parse_timestamp(frame.get("timestamp", current))
        except (TypeError, ValueError):
            observed = current
        token = f"{frame.get('sensorId', '')}:{frame.get('id', '')}:{frame.get('timestamp', '')}"
        if not observation_only:
            with self.lock, closing(self._connect()) as connection, connection:
                try:
                    connection.execute("INSERT INTO motion_frame_tokens(token, observed_at) VALUES (?, ?)", (token, observed))
                except sqlite3.IntegrityError:
                    return 0
                self._set_state(connection, "motion_worker_last_seen", current)
                self._set_state(connection, "motion_worker_last_frame_at", observed)
        motion_config = self.workstation.get("motion_pipeline", {})
        window_seconds = max(4.0, float(motion_config.get("window_seconds", 8)))
        step_seconds = max(1.0, float(motion_config.get("step_seconds", 2)))
        semantic_seconds = max(5.0, float(motion_config.get("semantic_interval_seconds", 10)))
        minimum_semantic = max(1.0, float(motion_config.get("minimum_semantic_interval_seconds", 5)))
        minimum_pose = max(0.0, min(1.0, float(motion_config.get("pose_minimum_confidence", 0.35))))
        width, height = (float(value) for value in self.config["camera"].get("resolution", [1920, 1080]))
        camera_id = str(self.config["camera"].get("id", "office-main"))
        for stale_track, stale_samples in list(self._motion_samples.items()):
            if not stale_samples or observed - stale_samples[-1]["timestamp"] > window_seconds * 2:
                self._motion_samples.pop(stale_track, None)
                self._motion_last_window_at.pop(stale_track, None)
                self._motion_last_semantic_at.pop(stale_track, None)
                self._motion_last_pose_frame_at.pop(stale_track, None)
                self._motion_last_observation_at.pop(stale_track, None)
        created = 0
        for item in frame.get("objects", []):
            if str(item.get("type", "")).casefold() != "person":
                continue
            track_id = str(item.get("id", "")).strip()
            bbox = item.get("bbox") or {}
            pose_source = str(motion_config.get("pose_source", "mediapipe")).casefold()
            pose: dict[str, Any] = {}
            pose_frame_stamp = observed
            image_data: bytes | None = None
            if pose_source in {"mediapipe", "auto"} and self.pose_analyzer is not None and self.frame_buffer is not None:
                tolerance = max(0.1, float(motion_config.get("pose_frame_tolerance_seconds", 1.5)))
                buffered = self.frame_buffer.nearest(observed, tolerance)
                if buffered is not None:
                    pose_frame_stamp, image_data = buffered
                    if pose_frame_stamp > self._motion_last_pose_frame_at.get(track_id, 0):
                        self._motion_last_pose_frame_at[track_id] = pose_frame_stamp
                        pose = self.pose_analyzer.analyze(image_data, bbox, minimum_pose)
            if not pose and pose_source in {"rtcv", "auto"}:
                pose = parse_pose(item)
                pose = {name: point for name, point in pose.items() if point.confidence >= minimum_pose}
            if not track_id or not pose or not bbox:
                continue
            sample = {
                "timestamp": pose_frame_stamp,
                "pose": pose,
                "bbox": dict(bbox),
                "embedding": extract_embedding(item),
                "pose_source": pose_source,
            }
            samples = [sample] if observation_only else self._motion_samples.setdefault(track_id, [])
            if not observation_only:
                samples.append(sample)
            observation_interval = max(0.1, float(self.pose_observation_interval_seconds))
            observation_due = (
                self.pose_observer is not None
                and image_data is not None
                and pose_source in {"mediapipe", "auto"}
                and pose_frame_stamp - self._motion_last_observation_at.get(track_id, 0) >= observation_interval
            )
            if observation_due:
                self._motion_last_observation_at[track_id] = pose_frame_stamp
                hand_facts: dict[str, Any] | None = None
                try:
                    if self.pose_observe_hands and self.hand_analyzer is not None:
                        person_frames = crop_person_frames(
                            [(pose_frame_stamp, image_data)], [samples[-1]],
                        )
                        hand_facts = self.hand_analyzer.analyze(person_frames)
                    with closing(self._connect()) as mapping_connection:
                        mapping = mapping_connection.execute(
                            "SELECT person_id FROM person_track_map WHERE track_id = ?", (track_id,),
                        ).fetchone()
                    person_id = int(mapping[0]) if mapping else None
                    observation = build_pose_observation(
                        sensor_id=camera_id,
                        frame_id=str(frame.get("id", "")),
                        track_id=track_id,
                        timestamp=pose_frame_stamp,
                        source_timestamp=observed,
                        bbox=bbox,
                        pose=pose,
                        frame_width=width,
                        frame_height=height,
                        person_id=person_id,
                        hands=hand_facts,
                    )
                    self.pose_observer(observation)
                except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
                    print(f"[office-motion-worker] pose observation skipped track={track_id}: {error}", flush=True)
            if observation_only:
                created += int(observation_due)
                continue
            cutoff = observed - window_seconds
            samples[:] = [sample for sample in samples if sample["timestamp"] >= cutoff]
            if len(samples) < 2 or observed - self._motion_last_window_at.get(track_id, 0) < step_seconds:
                continue
            facts = summarize_motion(samples, width, height)
            self._motion_last_window_at[track_id] = observed
            strong_change = bool(facts.get("posture_transitions"))
            span_seconds = samples[-1]["timestamp"] - samples[0]["timestamp"]
            semantic_due = (
                span_seconds >= window_seconds * 0.75
                and observed - self._motion_last_semantic_at.get(track_id, 0) >= semantic_seconds
            )
            semantic_allowed = observed - self._motion_last_semantic_at.get(track_id, 0) >= minimum_semantic
            needs_semantic = semantic_due or (span_seconds >= 2.0 and strong_change and semantic_allowed)
            if needs_semantic:
                self._motion_last_semantic_at[track_id] = observed
            self._persist_motion_window(connection=None, camera_id=camera_id, track_id=track_id, samples=list(samples), facts=facts, semantic=needs_semantic, now=current)
            created += 1
        return created

    def _persist_motion_window(
        self,
        connection: sqlite3.Connection | None,
        camera_id: str,
        track_id: str,
        samples: list[dict[str, Any]],
        facts: dict[str, Any],
        semantic: bool,
        now: float,
    ) -> int | None:
        start = float(facts.get("window", {}).get("start", samples[0]["timestamp"]))
        end = float(facts.get("window", {}).get("end", samples[-1]["timestamp"]))
        selected_person: list[tuple[float, bytes]] = []
        selected_scene: list[tuple[float, bytes]] = []
        person_path = ""
        scene_path = ""
        hand_facts: dict[str, Any] = {"available": False, "reason": "no synchronized frames", "observations": []}
        if semantic and self.frame_buffer is not None:
            buffered = self.frame_buffer.between(start, end)
            motion = self.workstation.get("motion_pipeline", {})
            selected_person = select_storyboard_frames(
                buffered, int(motion.get("person_storyboard_frames", 6)), samples,
            )
            selected_scene = select_storyboard_frames(
                buffered, int(motion.get("scene_storyboard_frames", 4)), samples,
            )
            base_name = f"{camera_id}_{track_id}_{int(start * 1000)}"
            person_file = self.storyboard_dir / f"{base_name}_person.jpg"
            scene_file = self.storyboard_dir / f"{base_name}_scene.jpg"
            person_path = build_storyboard(selected_person, samples, person_file, person_crop=True)
            scene_path = build_storyboard(selected_scene, samples, scene_file, person_crop=False)
            if self.hand_analyzer is not None:
                hand_facts = self.hand_analyzer.analyze(crop_person_frames(selected_person, samples))
        mapping_connection = connection or self._connect()
        owns_connection = connection is None
        try:
            mapping = mapping_connection.execute(
                "SELECT person_id FROM person_track_map WHERE track_id = ?", (track_id,)
            ).fetchone()
            person_id = int(mapping[0]) if mapping else None
            relative_person = str(Path(person_path).relative_to(self.database_file.parent)) if person_path else ""
            relative_scene = str(Path(scene_path).relative_to(self.database_file.parent)) if scene_path else ""
            summary = self._motion_summary(facts)
            cursor = mapping_connection.execute(
                "INSERT OR IGNORE INTO person_motion_windows("
                "camera_id, track_id, person_id, started_at, ended_at, facts_json, motion_summary, "
                "person_storyboard, scene_storyboard, hand_json, quality, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    camera_id, track_id, person_id, start, end, facts_json(facts), summary,
                    relative_person, relative_scene, json.dumps(hand_facts, ensure_ascii=False),
                    float(facts.get("quality", {}).get("pose_confidence", 0)),
                    "pending" if semantic else "facts_only", now,
                ),
            )
            if owns_connection:
                mapping_connection.commit()
            window_id = int(cursor.lastrowid) if cursor.lastrowid else None
        finally:
            if owns_connection:
                mapping_connection.close()
        if semantic:
            self._trim_motion_queue(now)
        return window_id

    @staticmethod
    def _motion_summary(facts: dict[str, Any]) -> str:
        motion = facts.get("body_motion", {})
        direction = str(motion.get("direction_in_image", "stationary"))
        labels = {
            "right_in_image": "在画面中向右移动",
            "left_in_image": "在画面中向左移动",
            "up_in_image": "在画面中向上移动",
            "down_in_image": "在画面中向下移动",
            "stationary": "身体位置基本稳定",
        }
        transitions = [str(item.get("type", "")) for item in facts.get("posture_transitions", [])]
        return "；".join([labels.get(direction, direction), *transitions])[:300]

    def _trim_motion_queue(self, now: float) -> None:
        limit = max(10, int(self.workstation.get("motion_pipeline", {}).get("queue_limit", 100)))
        with self.lock, closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT id FROM person_motion_windows WHERE status IN ('pending', 'retry') "
                "ORDER BY ended_at DESC"
            ).fetchall()
            for row in rows[limit:]:
                connection.execute(
                    "UPDATE person_motion_windows SET status = 'superseded', error = 'queue limit exceeded', "
                    "next_retry_at = ? WHERE id = ?",
                    (now, int(row[0])),
                )

    def analyze_pending_motion(self, now: float | None = None) -> dict[str, Any] | None:
        """Analyze one latest due motion window; retry failures without blocking ingestion."""
        current = time.time() if now is None else now
        with self.lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT * FROM person_motion_windows WHERE status IN ('pending', 'retry') "
                "AND next_retry_at <= ? ORDER BY ended_at DESC LIMIT 1",
                (current,),
            ).fetchone()
            if not row:
                return None
            connection.execute(
                "UPDATE person_motion_windows SET status = 'processing', attempts = attempts + 1 WHERE id = ?",
                (int(row["id"]),),
            )
        try:
            result = self._call_motion_vlm(dict(row))
            category = str(result.get("category", result.get("activity", "unknown"))).lower()
            if category not in ACTIVITIES:
                category = "unknown"
            confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
            continues = result.get("continues_current") is True
            observed_actions = clean_target_actions(result.get("observed_actions", []))
            with self.lock, closing(self._connect()) as context_connection:
                current_event = self._current_person_event(context_connection, str(row["track_id"]))
            current_category = str(current_event["activity"]) if current_event else ""
            category = infer_activity_from_actions(category, observed_actions, current_category)
            uncertainty = str(result.get("uncertainty", ""))[:300]
            if category == "rest" and result.get("sleep_evidence") is not True:
                category = "unknown"
                confidence = min(confidence, 0.59)
                continues = False
                uncertainty = (uncertainty + "；未观察到持续闭眼或打盹证据").strip("；")[:300]
            if not observed_actions:
                category = "unknown"
                confidence = min(confidence, 0.59)
                continues = False
                description = "动作无法确定"
            else:
                description = "；".join(observed_actions)[:300]
            with self.lock, closing(self._connect()) as connection, connection:
                interval_id = self._record_person_activity(
                    connection, str(row["camera_id"]), str(row["track_id"]), category,
                    description, confidence, continues, float(row["ended_at"]),
                )
                connection.execute(
                    "INSERT OR REPLACE INTO person_activity_observations("
                    "interval_id, motion_window_id, category, description, observed_actions_json, "
                    "continues_current, confidence, uncertainty, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        interval_id, int(row["id"]), category, description,
                        json.dumps(observed_actions if isinstance(observed_actions, list) else [], ensure_ascii=False),
                        int(continues), confidence, uncertainty, current,
                    ),
                )
                if interval_id is not None:
                    event_start = connection.execute(
                        "SELECT started_at FROM person_activity_intervals WHERE id = ?", (interval_id,)
                    ).fetchone()
                    if event_start:
                        connection.execute(
                            "UPDATE person_activity_observations SET interval_id = ? "
                            "WHERE interval_id IS NULL AND motion_window_id IN ("
                            "SELECT id FROM person_motion_windows WHERE camera_id = ? AND track_id = ? "
                            "AND ended_at >= ?)",
                            (interval_id, str(row["camera_id"]), str(row["track_id"]), float(event_start[0])),
                        )
                connection.execute(
                    "UPDATE person_motion_windows SET status = 'complete', error = '', next_retry_at = 0 "
                    "WHERE id = ?",
                    (int(row["id"]),),
                )
            return {"window_id": int(row["id"]), "category": category, "description": description, "confidence": confidence}
        except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError, KeyError) as error:
            attempts = int(row["attempts"] or 0) + 1
            delay = min(3600, 10 * (2 ** min(attempts, 8)))
            with self.lock, closing(self._connect()) as connection, connection:
                connection.execute(
                    "UPDATE person_motion_windows SET status = 'retry', error = ?, next_retry_at = ? WHERE id = ?",
                    (str(error)[:500], current + delay, int(row["id"])),
                )
            self.last_vlm_error = str(error)
            return None

    def _call_motion_vlm(self, window: dict[str, Any]) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            current_event = self._current_person_event(connection, str(window["track_id"]))
        context = None if not current_event else {
            "category": str(current_event["activity"]),
            "description": str(current_event["description"] or ""),
        }
        evidence = {
            "person_id": window.get("person_id"),
            "track_id": window["track_id"],
            "window": {"start": window["started_at"], "end": window["ended_at"]},
            "body_motion": json.loads(str(window["facts_json"])),
            "hand_motion": json.loads(str(window["hand_json"] or "{}")),
            "current_event": context,
        }
        prompt = (
            "你将收到同一目标人物连续时间窗的结构化运动测量和人物近景故事板。"
            "测量是证据，不是活动标签；z只表示单目模型估计的相对前后变化。"
            "只分析裁剪框中的目标人物；忽略并且绝不描述背景、场所、光线、服装、外貌、身份、其他人物或人与人的关系。"
            "observed_actions必须列出1到4个只属于目标人物的可见动作短语，不要写人物外貌或环境；"
            "description只能由这些动作组成，禁止猜测屏幕内容、文件名、谈话主题、业务目的或敏感属性。"
            "computer仅在可见看屏幕、使用键鼠或打字时选择；reading仅限阅读纸张或书；writing仅限手写；"
            "phone仅限可见使用手机；conversation仅在目标人物持续开口说话或有明确交谈手势时选择，且不得描述对方；"
            "eating仅限进食或饮水。rest仅表示睡眠：必须在多个画面持续闭眼，并伴随打盹姿态且没有目的性动作。"
            "在已标定的电脑工位内，人物保持坐姿面向桌面或屏幕、低头观看、身体前倾，或只有头部、上半身、"
            "手臂和手部的小幅调整，均视为连续使用电脑；紧裁剪画面未拍到显示器本身不能作为unknown的理由。"
            "静坐、发呆、托腮、思考、等待、身体稳定、低头或后仰都不能单独判为rest；动作不明确时用unknown。"
            "sleep_evidence只有满足上述睡眠证据时才可为true。"
            "只返回JSON："
            '{"category":"computer|reading|writing|phone|conversation|eating|rest|unknown",'
            '"description":"自由中文描述","observed_actions":["可见动作"],'
            '"sleep_evidence":false,"continues_current":true,"confidence":0.0,"uncertainty":"不确定原因"}\n'
            f"运动证据：{json.dumps(evidence, ensure_ascii=False)}"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for key in ("person_storyboard",):
            relative = str(window.get(key, ""))
            path = self.database_file.parent / relative
            if relative and path.is_file():
                content.append({
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")},
                })
        payload = {
            "model": self._model(),
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "temperature": 0,
            "max_tokens": 768,
        }
        return self._post_vlm(payload)

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
            connection.execute(
                "DELETE FROM person_activity_intervals WHERE COALESCE(ended_at, last_observed_at, started_at) < ?",
                (cutoff,),
            )
            connection.execute(
                "DELETE FROM person_seated_intervals WHERE COALESCE(ended_at, last_seen_at, started_at) < ?",
                (cutoff,),
            )
            connection.execute("DELETE FROM person_verifications WHERE created_at < ?", (cutoff,))
        clip_cutoff = current - int(self.workstation.get("event_clip_retention_days", 7)) * 86400
        with closing(self._connect()) as connection, connection:
            expired = connection.execute(
                "SELECT id, clip_path FROM workstation_events WHERE occurred_at < ?", (clip_cutoff,)
            ).fetchall()
            for event_id, clip_path in expired:
                if clip_path:
                    Path(clip_path).unlink(missing_ok=True)
                connection.execute("DELETE FROM workstation_events WHERE id = ?", (event_id,))
        motion = self.workstation.get("motion_pipeline", {})
        evidence_days = int(motion.get("storyboard_retention_days", motion.get("retention_days", 7)))
        evidence_cutoff = current - evidence_days * 86400
        with closing(self._connect()) as connection, connection:
            expired_windows = connection.execute(
                "SELECT id, person_storyboard, scene_storyboard FROM person_motion_windows WHERE ended_at < ?",
                (evidence_cutoff,),
            ).fetchall()
            root = self.database_file.parent.resolve()
            for window_id, person_storyboard, scene_storyboard in expired_windows:
                for relative in (person_storyboard, scene_storyboard):
                    if not relative:
                        continue
                    path = (root / str(relative)).resolve()
                    if root in path.parents:
                        path.unlink(missing_ok=True)
                connection.execute(
                    "DELETE FROM person_activity_observations WHERE motion_window_id = ?", (window_id,)
                )
                connection.execute("DELETE FROM person_motion_windows WHERE id = ?", (window_id,))
            connection.execute("DELETE FROM motion_frame_tokens WHERE observed_at < ?", (current - 86400,))

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

    def _person_lifecycle_events(
        self,
        connection: sqlite3.Connection,
        start: float,
        end: float,
        effective_now: float,
        person_id: int | None = None,
        needle: str = "",
    ) -> list[dict[str, Any]]:
        """Derive evidence-backed leave/return events from recognized seated tracks."""
        camera_id = str(self.config["camera"].get("id", "office-main"))
        merge_gap = max(60.0, float(self.workstation.get("person_presence_merge_seconds", 300)))
        rows = connection.execute(
            "SELECT s.*, m.person_id AS resolved_person_id, p.name AS person_name, p.active AS person_active "
            "FROM person_seated_intervals s "
            "JOIN person_track_map m ON m.track_id = s.track_id "
            "JOIN people p ON p.id = m.person_id "
            "WHERE s.camera_id = ? AND s.started_at < ? "
            "AND COALESCE(s.ended_at, s.last_seen_at) > ? AND p.active = 1 "
            "ORDER BY m.person_id, s.started_at, s.id",
            (camera_id, end, start - merge_gap),
        ).fetchall()

        grouped: dict[int, dict[str, Any]] = {}
        for row in rows:
            resolved = int(row["resolved_person_id"])
            if person_id is not None and resolved != person_id:
                continue
            item = grouped.setdefault(resolved, {
                "person_name": str(row["person_name"]),
                "intervals": [],
            })
            interval_end = float(row["ended_at"] or row["last_seen_at"] or row["started_at"])
            if interval_end <= float(row["started_at"]):
                continue
            item["intervals"].append({
                "ids": {int(row["id"])},
                "started_at": float(row["started_at"]),
                "ended_at": interval_end,
                "tracks": {str(row["track_id"])},
            })

        def evidence_for(tracks: set[str], boundary: float, kind: str) -> sqlite3.Row | None:
            if not tracks:
                return None
            placeholders = ",".join("?" for _ in tracks)
            if kind == "leave":
                lower, upper = boundary - 45, boundary + 120
                preferred = (
                    "CASE WHEN o.description LIKE '%离开%' OR o.description LIKE '%消失%' "
                    "OR w.motion_summary LIKE '%stood_up%' OR w.motion_summary LIKE '%moved_away%' "
                    "THEN 0 ELSE 1 END"
                )
            else:
                lower, upper = boundary, boundary + 300
                preferred = "CASE WHEN w.person_storyboard != '' THEN 0 ELSE 1 END"
            return connection.execute(
                "SELECT w.*, o.id AS observation_id, o.description AS observation_description, "
                "o.observed_actions_json, o.confidence AS observation_confidence, o.uncertainty "
                "FROM person_motion_windows w "
                "LEFT JOIN person_activity_observations o ON o.motion_window_id = w.id "
                f"WHERE w.track_id IN ({placeholders}) AND w.started_at >= ? AND w.started_at <= ? "
                f"ORDER BY {preferred}, CASE WHEN w.person_storyboard != '' THEN 0 ELSE 1 END, "
                "ABS(w.started_at - ?) LIMIT 1",
                (*sorted(tracks), lower, upper, boundary),
            ).fetchone()

        events: list[dict[str, Any]] = []
        for resolved, item in grouped.items():
            merged_presence: list[dict[str, Any]] = []
            for interval in item["intervals"]:
                if merged_presence and interval["started_at"] <= merged_presence[-1]["ended_at"] + merge_gap:
                    merged_presence[-1]["ended_at"] = max(
                        merged_presence[-1]["ended_at"], interval["ended_at"],
                    )
                    merged_presence[-1]["ids"].update(interval["ids"])
                    merged_presence[-1]["tracks"].update(interval["tracks"])
                else:
                    merged_presence.append(interval)

            for index, interval in enumerate(merged_presence):
                following = merged_presence[index + 1] if index + 1 < len(merged_presence) else None
                away_end = following["started_at"] if following else effective_now
                if away_end - interval["ended_at"] <= merge_gap:
                    continue
                leave_evidence = evidence_for(interval["tracks"], interval["ended_at"], "leave")
                leave_text = "" if not leave_evidence else " ".join((
                    str(leave_evidence["observation_description"] or ""),
                    str(leave_evidence["motion_summary"] or ""),
                )).casefold()
                explicit_leave = any(marker in leave_text for marker in (
                    "离开座位", "离开工位", "从画面中完全消失", "走出画面",
                    "站起并离开", "起身并离开", "stood_up_or_approached_camera",
                ))
                published_leave = False
                if interval["ended_at"] < end and away_end > start:
                    description = "离开座位并离开工位"
                    searchable = " ".join((item["person_name"], "离开工位", description)).casefold()
                    if not needle or needle in searchable:
                        window_id = int(leave_evidence["id"]) if leave_evidence else None
                        seated_id = max(interval["ids"])
                        event_id = (
                            -(window_id * 2)
                            if leave_evidence and explicit_leave
                            else -(PRESENCE_LIFECYCLE_ID_BASE + seated_id * 2)
                        )
                        event_start = max(start, interval["ended_at"])
                        event_end = min(end, away_end)
                        events.append({
                            "id": event_id,
                            "person_id": resolved,
                            "person_name": item["person_name"],
                            "track_id": (
                                str(leave_evidence["track_id"])
                                if leave_evidence else sorted(interval["tracks"])[-1]
                            ),
                            "activity": "left_workstation",
                            "activity_label": ACTIVITY_LABELS["left_workstation"],
                            "description": description,
                            "started_at": event_start,
                            "ended_at": event_end,
                            "duration_seconds": round(event_end - event_start),
                            "ongoing": following is None,
                            "confidence": round(float(
                                leave_evidence["observation_confidence"]
                                or leave_evidence["quality"] or 1
                            ), 3) if leave_evidence and explicit_leave else 1.0,
                            "observation_count": 1,
                            "source_ids": [],
                            "motion_summary": (
                                str(leave_evidence["motion_summary"] or "")
                                if leave_evidence and explicit_leave
                                else "识别人员离开座椅区域并持续缺席"
                            ),
                            "evidence_status": (
                                str(leave_evidence["status"] or "unavailable")
                                if leave_evidence and explicit_leave else "presence_timeout"
                            ),
                            "storyboard_url": (
                                f"/api/activity/evidence/{window_id}/person"
                                if leave_evidence and explicit_leave and leave_evidence["person_storyboard"] else ""
                            ),
                        })
                        published_leave = True
                if not following or not published_leave:
                    continue
                return_evidence = evidence_for(following["tracks"], following["started_at"], "return")
                if start <= following["started_at"] < end:
                    description = "返回工位并重新坐下"
                    searchable = " ".join((item["person_name"], "返回工位", description)).casefold()
                    if not needle or needle in searchable:
                        window_id = int(return_evidence["id"]) if return_evidence else None
                        seated_id = min(following["ids"])
                        event_id = (
                            -(window_id * 2 + 1)
                            if return_evidence
                            else -(PRESENCE_LIFECYCLE_ID_BASE + seated_id * 2 + 1)
                        )
                        event_start = following["started_at"]
                        events.append({
                            "id": event_id,
                            "person_id": resolved,
                            "person_name": item["person_name"],
                            "track_id": (
                                str(return_evidence["track_id"])
                                if return_evidence else sorted(following["tracks"])[0]
                            ),
                            "activity": "returned_to_workstation",
                            "activity_label": ACTIVITY_LABELS["returned_to_workstation"],
                            "description": description,
                            "started_at": event_start,
                            "ended_at": min(end, event_start + 1),
                            "duration_seconds": 1,
                            "ongoing": False,
                            "confidence": round(float(
                                return_evidence["observation_confidence"]
                                or return_evidence["quality"] or 1
                            ), 3) if return_evidence else 1.0,
                            "observation_count": 1,
                            "source_ids": [],
                            "motion_summary": (
                                str(return_evidence["motion_summary"] or "")
                                if return_evidence else "识别人员重新进入座椅区域"
                            ),
                            "evidence_status": (
                                str(return_evidence["status"] or "unavailable")
                                if return_evidence else "presence_confirmed"
                            ),
                            "storyboard_url": (
                                f"/api/activity/evidence/{window_id}/person"
                                if return_evidence and return_evidence["person_storyboard"] else ""
                            ),
                        })
        return events

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
        needle = " ".join(query.strip().casefold().split())
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT a.*, COALESCE(a.person_id, m.person_id) AS resolved_person_id, "
                "p.name AS person_name, p.active AS person_active "
                "FROM person_activity_intervals a "
                "LEFT JOIN person_track_map m ON m.track_id = a.track_id "
                "LEFT JOIN people p ON p.id = COALESCE(a.person_id, m.person_id) "
                "WHERE a.camera_id = ? AND a.started_at < ? "
                "AND COALESCE(a.ended_at, a.last_observed_at, ?) > ? "
                "ORDER BY a.started_at, a.id",
                (camera_id, end, effective_now, start),
            ).fetchall()
            lifecycle_events = self._person_lifecycle_events(
                connection, start, end, effective_now, person_id=person_id, needle=needle,
            )

        events: list[dict[str, Any]] = []
        for row in rows:
            resolved_person_id = int(row["resolved_person_id"]) if row["resolved_person_id"] is not None else None
            # Track IDs are short-lived implementation details. Publish an event only
            # after identity resolution has attached it to an active library person.
            if resolved_person_id is None or row["person_name"] is None or int(row["person_active"] or 0) != 1:
                continue
            if person_id is not None and resolved_person_id != person_id:
                continue
            raw_activity = str(row["activity"])
            description = clean_target_description(
                row["description"], ACTIVITY_LABELS.get(raw_activity, raw_activity),
            )
            # Apply the current workstation semantics while reading historical
            # rows as well, so existing desk-posture fragments immediately join
            # their surrounding computer-use segment without rewriting evidence.
            activity = infer_activity_from_actions(raw_activity, [description])
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

        with closing(self._connect()) as evidence_connection:
            for event in merged:
                placeholders = ",".join("?" for _ in event["source_ids"])
                observation = evidence_connection.execute(
                    "SELECT o.*, w.motion_summary, w.status AS evidence_status, w.id AS window_id, "
                    "w.person_storyboard, w.scene_storyboard "
                    "FROM person_activity_observations o "
                    "JOIN person_motion_windows w ON w.id = o.motion_window_id "
                    f"WHERE o.interval_id IN ({placeholders}) ORDER BY o.created_at DESC LIMIT 1",
                    tuple(event["source_ids"]),
                ).fetchone()
                event["motion_summary"] = str(observation["motion_summary"]) if observation else ""
                event["evidence_status"] = str(observation["evidence_status"]) if observation else "unavailable"
                event["storyboard_url"] = (
                    f"/api/activity/evidence/{int(observation['window_id'])}/person"
                    if observation and observation["person_storyboard"] else ""
                )

        merged.extend(lifecycle_events)
        merged.sort(key=lambda item: (item["started_at"], item["id"]))

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

    def _lifecycle_event_detail(self, event_id: int) -> dict[str, Any] | None:
        encoded = abs(event_id)
        kind = "returned_to_workstation" if encoded % 2 else "left_workstation"
        if encoded >= PRESENCE_LIFECYCLE_ID_BASE:
            seated_id = (encoded - PRESENCE_LIFECYCLE_ID_BASE) // 2
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT s.*, m.person_id AS resolved_person_id, p.name AS person_name "
                    "FROM person_seated_intervals s "
                    "JOIN person_track_map m ON m.track_id = s.track_id "
                    "JOIN people p ON p.id = m.person_id WHERE s.id = ?",
                    (seated_id,),
                ).fetchone()
            if not row:
                return None
            boundary = float(
                row["started_at"] if kind == "returned_to_workstation"
                else row["ended_at"] or row["last_seen_at"]
            )
            description = (
                "离开座位并离开工位" if kind == "left_workstation" else "返回工位并重新坐下"
            )
            evidence_status = (
                "presence_timeout" if kind == "left_workstation" else "presence_confirmed"
            )
            motion_summary = (
                "识别人员离开座椅区域并持续缺席"
                if kind == "left_workstation" else "识别人员重新进入座椅区域"
            )
            return {
                "id": event_id,
                "person_id": int(row["resolved_person_id"]),
                "person_name": str(row["person_name"]),
                "track_id": str(row["track_id"]),
                "category": kind,
                "description": description,
                "started_at": boundary,
                "ended_at": boundary + 1,
                "observations": [{
                    "id": 0,
                    "window_id": 0,
                    "window": {"start": boundary, "end": boundary + 1},
                    "category": kind,
                    "description": description,
                    "observed_actions": [motion_summary],
                    "continues_current": False,
                    "confidence": 1.0,
                    "uncertainty": "",
                    "motion_summary": motion_summary,
                    "motion_facts": {
                        "presence": {
                            "seated_interval_id": seated_id,
                            "started_at": float(row["started_at"]),
                            "last_seen_at": float(row["last_seen_at"]),
                            "ended_at": float(row["ended_at"] or row["last_seen_at"]),
                        },
                    },
                    "hand_motion": {},
                    "evidence_status": evidence_status,
                    "storyboards": {"person": "", "scene": ""},
                }],
            }
        window_id = encoded // 2
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT w.*, o.id AS observation_id, o.description AS observation_description, "
                "o.observed_actions_json, o.continues_current, o.confidence AS observation_confidence, "
                "o.uncertainty, COALESCE(w.person_id, m.person_id) AS resolved_person_id, "
                "p.name AS person_name FROM person_motion_windows w "
                "LEFT JOIN person_activity_observations o ON o.motion_window_id = w.id "
                "LEFT JOIN person_track_map m ON m.track_id = w.track_id "
                "LEFT JOIN people p ON p.id = COALESCE(w.person_id, m.person_id) "
                "WHERE w.id = ?",
                (window_id,),
            ).fetchone()
        if not row or row["resolved_person_id"] is None or row["person_name"] is None:
            return None
        description = (
            "离开座位并离开工位" if kind == "left_workstation" else "返回工位并重新坐下"
        )
        observed_description = clean_target_description(
            row["observation_description"] or row["motion_summary"], description,
        )
        try:
            observed_actions = clean_target_actions(json.loads(str(row["observed_actions_json"] or "[]")))
        except json.JSONDecodeError:
            observed_actions = []
        return {
            "id": event_id,
            "person_id": int(row["resolved_person_id"]),
            "person_name": str(row["person_name"]),
            "track_id": str(row["track_id"]),
            "category": kind,
            "description": description,
            "started_at": float(row["started_at"]),
            "ended_at": float(row["ended_at"]),
            "observations": [{
                "id": int(row["observation_id"] or 0),
                "window_id": window_id,
                "window": {"start": float(row["started_at"]), "end": float(row["ended_at"])},
                "category": kind,
                "description": observed_description,
                "observed_actions": observed_actions,
                "continues_current": bool(row["continues_current"] or False),
                "confidence": float(row["observation_confidence"] or row["quality"] or 1),
                "uncertainty": str(row["uncertainty"] or ""),
                "motion_summary": str(row["motion_summary"] or ""),
                "motion_facts": json.loads(str(row["facts_json"] or "{}")),
                "hand_motion": json.loads(str(row["hand_json"] or "{}")),
                "evidence_status": str(row["status"] or "unavailable"),
                "storyboards": {
                    "person": f"/api/activity/evidence/{window_id}/person" if row["person_storyboard"] else "",
                    "scene": "",
                },
            }],
        }

    def activity_event_detail(self, event_id: int) -> dict[str, Any] | None:
        """Return the observation trail and measurable evidence for one raw activity interval."""
        if event_id < 0:
            return self._lifecycle_event_detail(event_id)
        with closing(self._connect()) as connection:
            event = connection.execute(
                "SELECT a.*, COALESCE(a.person_id, m.person_id) AS resolved_person_id, p.name AS person_name "
                "FROM person_activity_intervals a "
                "LEFT JOIN person_track_map m ON m.track_id = a.track_id "
                "LEFT JOIN people p ON p.id = COALESCE(a.person_id, m.person_id) WHERE a.id = ?",
                (event_id,),
            ).fetchone()
            if not event:
                return None
            observations = connection.execute(
                "SELECT o.*, w.started_at AS window_started_at, w.ended_at AS window_ended_at, "
                "w.facts_json, w.hand_json, w.motion_summary, w.status AS evidence_status, "
                "w.id AS window_id, w.person_storyboard, w.scene_storyboard "
                "FROM person_activity_observations o JOIN person_motion_windows w ON w.id = o.motion_window_id "
                "WHERE o.interval_id = ? ORDER BY o.created_at",
                (event_id,),
            ).fetchall()
        return {
            "id": int(event["id"]),
            "person_id": int(event["resolved_person_id"]) if event["resolved_person_id"] is not None else None,
            "person_name": str(event["person_name"] or f"人员 {event['track_id']}"),
            "track_id": str(event["track_id"]),
            "category": str(event["activity"]),
            "description": clean_target_description(
                event["description"], ACTIVITY_LABELS.get(str(event["activity"]), ""),
            ),
            "started_at": float(event["started_at"]),
            "ended_at": float(event["ended_at"] or event["last_observed_at"] or event["started_at"]),
            "observations": [{
                "id": int(row["id"]),
                "window_id": int(row["window_id"]),
                "window": {"start": float(row["window_started_at"]), "end": float(row["window_ended_at"])},
                "category": str(row["category"]),
                "description": clean_target_description(row["description"]),
                "observed_actions": clean_target_actions(
                    json.loads(str(row["observed_actions_json"] or "[]")),
                ),
                "continues_current": bool(row["continues_current"]),
                "confidence": float(row["confidence"]),
                "uncertainty": str(row["uncertainty"]),
                "motion_summary": str(row["motion_summary"]),
                "motion_facts": json.loads(str(row["facts_json"])),
                "hand_motion": json.loads(str(row["hand_json"] or "{}")),
                "evidence_status": str(row["evidence_status"]),
                "storyboards": {
                    "person": f"/api/activity/evidence/{int(row['window_id'])}/person" if row["person_storyboard"] else "",
                    "scene": "",
                },
            } for row in observations],
        }

    def evidence_image(self, window_id: int, kind: str) -> Path | None:
        column = "person_storyboard" if kind == "person" else "scene_storyboard" if kind == "scene" else ""
        if not column:
            return None
        with closing(self._connect()) as connection:
            row = connection.execute(f"SELECT {column} FROM person_motion_windows WHERE id = ?", (window_id,)).fetchone()
        if not row or not row[0]:
            return None
        path = (self.database_file.parent / str(row[0])).resolve()
        root = self.database_file.parent.resolve()
        return path if path.is_file() and (path == root or root in path.parents) else None

    def motion_status(self, now: float | None = None) -> dict[str, Any]:
        current = time.time() if now is None else now
        with closing(self._connect()) as connection:
            last_seen = float(self._state(connection, "motion_worker_last_seen", "0") or 0)
            pending = connection.execute(
                "SELECT COUNT(*) FROM person_motion_windows WHERE status IN ('pending', 'retry', 'processing')"
            ).fetchone()[0]
        return {
            "enabled": bool(self.workstation.get("motion_pipeline", {}).get("enabled", True)),
            "healthy": bool(last_seen and current - last_seen < 15),
            "last_seen_at": last_seen or None,
            "worker_seen_within_seconds": round(current - last_seen, 3) if last_seen else None,
            "pending_windows": int(pending),
            "last_error": self.last_vlm_error,
        }

    def people_list(self, now: float | None = None) -> list[dict[str, Any]]:
        """已注册人员列表：名字、参考图（相对路径）、首见/末见时间、关联 track 数。"""
        current = time.time() if now is None else now
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT p.id, p.name, p.reference_image, p.first_seen_at, p.last_seen_at, "
                "(SELECT COUNT(*) FROM person_track_map m WHERE m.person_id = p.id) AS track_count, "
                "(SELECT COUNT(*) FROM person_reference_images r WHERE r.person_id = p.id AND r.active = 1) "
                "AS image_count "
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
                "image_count": int(row["image_count"]),
            }
            for row in rows
        ]

    def person_gallery(self, person_id: int) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, captured_at, quality_score, is_cover FROM person_reference_images "
                "WHERE person_id = ? AND active = 1 ORDER BY is_cover DESC, quality_score DESC",
                (person_id,),
            ).fetchall()
        return [{
            "image_id": int(row["id"]),
            "captured_at": float(row["captured_at"]),
            "quality_score": float(row["quality_score"]),
            "is_cover": bool(row["is_cover"]),
            "url": f"/api/people/{person_id}/images/{int(row['id'])}",
        } for row in rows]

    def person_gallery_image(self, person_id: int, image_id: int) -> Path | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT path FROM person_reference_images WHERE id = ? AND person_id = ? AND active = 1",
                (image_id, person_id),
            ).fetchone()
        if not row:
            return None
        path = (self.database_file.parent / str(row[0])).resolve()
        root = self.database_file.parent.resolve()
        return path if path.is_file() and root in path.parents else None

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
            connection.execute(
                "UPDATE person_reference_images SET active = 0, is_cover = 0 WHERE person_id = ?", (person_id,)
            )
        return True

    def merge_person(self, source_person_id: int, target_person_id: int) -> dict[str, Any] | None:
        """Merge a duplicate library person into the retained target person."""
        if source_person_id == target_person_id:
            raise ValueError("source and target person must be different")
        with self.lock, closing(self._connect()) as connection, connection:
            previous = connection.execute(
                "SELECT * FROM person_merge_history WHERE source_person_id = ?",
                (source_person_id,),
            ).fetchone()
            if previous:
                previous_target = int(previous["target_person_id"])
                if previous_target != target_person_id:
                    raise ValueError(
                        f"source person was already merged into person {previous_target}"
                    )
                try:
                    previous_counts = json.loads(str(previous["migrated_json"] or "{}"))
                except json.JSONDecodeError:
                    previous_counts = {}
                return {
                    "source_person_id": source_person_id,
                    "target_person_id": target_person_id,
                    "source_name": str(previous["source_name"]),
                    "target_name": str(previous["target_name"]),
                    "migrated": previous_counts,
                    "already_merged": True,
                }
            source = connection.execute(
                "SELECT * FROM people WHERE id = ? AND active = 1", (source_person_id,)
            ).fetchone()
            target = connection.execute(
                "SELECT * FROM people WHERE id = ? AND active = 1", (target_person_id,)
            ).fetchone()
            if not source or not target:
                return None
            target_cover = connection.execute(
                "SELECT id, path FROM person_reference_images "
                "WHERE person_id = ? AND active = 1 AND is_cover = 1 "
                "ORDER BY quality_score DESC, captured_at DESC LIMIT 1",
                (target_person_id,),
            ).fetchone()

            counts: dict[str, int] = {}

            def migrate(table: str, column: str = "person_id") -> None:
                cursor = connection.execute(
                    f"UPDATE {table} SET {column} = ? WHERE {column} = ?",
                    (target_person_id, source_person_id),
                )
                counts[f"{table}.{column}"] = max(0, int(cursor.rowcount))

            migrate("person_track_map")
            migrate("person_activity_intervals")
            migrate("person_motion_windows")
            migrate("person_reference_images")
            migrate("person_verifications")
            migrate("person_verifications", "matched_person_id")
            migrate("person_verifications", "candidate_person_id")
            flywheel_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'flywheel_candidates'"
            ).fetchone()
            if flywheel_exists:
                migrate("flywheel_candidates")

            connection.execute(
                "UPDATE people SET first_seen_at = MIN(first_seen_at, ?), "
                "last_seen_at = MAX(last_seen_at, ?) WHERE id = ?",
                (float(source["first_seen_at"]), float(source["last_seen_at"]), target_person_id),
            )
            connection.execute("UPDATE people SET active = 0 WHERE id = ?", (source_person_id,))

            cover = target_cover
            if not cover:
                cover = connection.execute(
                    "SELECT id, path FROM person_reference_images "
                    "WHERE person_id = ? AND active = 1 "
                    "ORDER BY quality_score DESC, captured_at DESC LIMIT 1",
                    (target_person_id,),
                ).fetchone()
            connection.execute(
                "UPDATE person_reference_images SET is_cover = 0 WHERE person_id = ?",
                (target_person_id,),
            )
            if cover:
                connection.execute(
                    "UPDATE person_reference_images SET is_cover = 1 WHERE id = ?", (int(cover["id"]),)
                )
                connection.execute(
                    "UPDATE people SET reference_image = ? WHERE id = ?",
                    (str(cover["path"]), target_person_id),
                )
            gallery_limit = max(1, int(self.workstation.get("identity", {}).get("gallery_limit", 5)))
            gallery = connection.execute(
                "SELECT id FROM person_reference_images WHERE person_id = ? AND active = 1 "
                "ORDER BY is_cover DESC, quality_score DESC, captured_at DESC",
                (target_person_id,),
            ).fetchall()
            for row in gallery[gallery_limit:]:
                connection.execute(
                    "UPDATE person_reference_images SET active = 0, is_cover = 0 WHERE id = ?",
                    (int(row["id"]),),
                )
            connection.execute(
                "INSERT INTO person_merge_history("
                "source_person_id, target_person_id, source_name, target_name, migrated_json, merged_at"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    source_person_id,
                    target_person_id,
                    str(source["name"]),
                    str(target["name"]),
                    json.dumps(counts, ensure_ascii=False),
                    time.time(),
                ),
            )

        return {
            "source_person_id": source_person_id,
            "target_person_id": target_person_id,
            "source_name": str(source["name"]),
            "target_name": str(target["name"]),
            "migrated": counts,
            "already_merged": False,
        }

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
            motion_enabled = engine.workstation.get("motion_pipeline", {}).get("enabled", True)
            if not motion_enabled and engine.needs_vlm_sample():
                engine.analyze_activity()
            if not motion_enabled:
                engine.sample_next_person()
            engine._drain_person_pending()  # noqa: SLF001 - worker 主循环异步消费新 track 判定
            engine.mark_overtime()
            engine.archive_pending_clips()
            if int(time.time()) % 3600 < max(1, int(poll_seconds)):
                engine.cleanup()
        except Exception as error:  # Worker must remain alive across dependent-service outages.
            print(f"[office-api] workstation worker failed: {error}", flush=True)
        time.sleep(max(0.5, poll_seconds))
