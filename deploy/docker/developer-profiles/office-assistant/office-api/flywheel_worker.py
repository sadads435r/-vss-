# SPDX-License-Identifier: Apache-2.0
"""Mine high-recall drinking candidates from mdx-office-pose and archive clips."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import subprocess
import tempfile
import time
from collections import deque
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from confluent_kafka import Consumer, KafkaError

from app import DATABASE_FILE
from flywheel import FlywheelStore, initialize_flywheel_schema
from gdino_client import GroundingDinoClient

LOGGER = logging.getLogger("office-flywheel-worker")
RULE_VERSION = "drink-wrist-mouth-v2"
OBJECT_DETECTOR_VERSION = "cup-bottle-v2"


def number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def fuse_object_scores(
    scores: list[float], threshold: float = 0.40, high_threshold: float = 0.65,
    required_frames: int = 2,
) -> tuple[bool, int, float]:
    best = max(scores, default=0.0)
    confirmed_frames = sum(score >= threshold for score in scores)
    return confirmed_frames >= required_frames or best >= high_threshold, confirmed_frames, best


def wrist_mouth_measurement(observation: dict[str, Any]) -> dict[str, Any] | None:
    pose = observation.get("pose") or {}
    points = pose.get("keypoints") or {}
    bbox = observation.get("bbox") or {}
    mouths = [points.get("left_mouth"), points.get("right_mouth"), points.get("nose")]
    mouths = [point for point in mouths if isinstance(point, dict) and number(point.get("confidence")) >= 0.55]
    if not mouths:
        return None
    mouth = {
        "x": sum(number(point.get("x")) for point in mouths) / len(mouths),
        "y": sum(number(point.get("y")) for point in mouths) / len(mouths),
    }
    height = max(0.08, number(bbox.get("bottom")) - number(bbox.get("top")))
    choices = []
    for side in ("left", "right"):
        wrist = points.get(f"{side}_wrist")
        minimum_wrist = number(os.environ.get("OFFICE_FLYWHEEL_MIN_WRIST_CONFIDENCE", "0.55"))
        if not isinstance(wrist, dict) or number(wrist.get("confidence")) < minimum_wrist:
            continue
        distance = math.hypot(number(wrist.get("x")) - mouth["x"], number(wrist.get("y")) - mouth["y"]) / height
        choices.append((distance, side, number(wrist.get("confidence"))))
    if not choices:
        return None
    distance, side, confidence = min(choices)
    return {
        "near": distance <= number(os.environ.get("OFFICE_FLYWHEEL_MOUTH_DISTANCE", "0.28")),
        "distance_bbox_heights": round(distance, 4),
        "side": side,
        "wrist_confidence": round(confidence, 4),
        "mouth_confidence": round(sum(number(point.get("confidence")) for point in mouths) / len(mouths), 4),
    }


class DrinkingCandidateMiner:
    def __init__(self, store: FlywheelStore, recordings: Path) -> None:
        self.store = store
        self.recordings = recordings
        self.active: dict[str, dict[str, Any]] = {}
        self.recent: dict[str, deque[dict[str, Any]]] = {}
        self.pending_archive: list[tuple[float, int]] = []
        self.cooldown: dict[str, float] = {}
        self.minimum_seconds = number(os.environ.get("OFFICE_FLYWHEEL_MIN_NEAR_SECONDS", "0.8"))
        self.exit_seconds = number(os.environ.get("OFFICE_FLYWHEEL_EXIT_SECONDS", "0.9"))
        self.cooldown_seconds = number(os.environ.get("OFFICE_FLYWHEEL_COOLDOWN_SECONDS", "30"))
        self.pre_roll = number(os.environ.get("OFFICE_FLYWHEEL_PRE_ROLL_SECONDS", "3"))
        self.post_roll = number(os.environ.get("OFFICE_FLYWHEEL_POST_ROLL_SECONDS", "3"))
        self.known_people_only = os.environ.get("OFFICE_FLYWHEEL_KNOWN_PEOPLE_ONLY", "true").casefold() in {
            "1", "true", "yes", "on",
        }
        self.object_detector = GroundingDinoClient(
            os.environ.get(
                "OFFICE_GDINO_URL",
                "http://127.0.0.1:18000/v2/models/ensemble_python_gdino/infer",
            ),
            os.environ.get("OFFICE_GDINO_PROMPT", "cup . mug . bottle . water bottle ."),
            number(os.environ.get("OFFICE_GDINO_THRESHOLD", "0.40")),
        )
        self.object_required_frames = max(1, int(os.environ.get("OFFICE_GDINO_REQUIRED_FRAMES", "2")))
        self.object_high_threshold = number(os.environ.get("OFFICE_GDINO_HIGH_THRESHOLD", "0.65"))
        self._last_object_backfill_at = 0.0
        with closing(self.store.connect()) as connection, connection:
            connection.execute(
                "UPDATE flywheel_candidates SET object_status='pending', object_error='' "
                "WHERE object_status='complete' AND "
                "COALESCE(json_extract(object_json, '$.detector_version'), '') != ?",
                (OBJECT_DETECTOR_VERSION,),
            )

    @staticmethod
    def key(observation: dict[str, Any]) -> str:
        person = observation.get("person_id")
        identity = f"person:{person}" if person is not None else f"track:{observation.get('track_id', '')}"
        return f"{observation.get('sensor_id', '')}:{identity}"

    def observe(self, observation: dict[str, Any]) -> None:
        if observation.get("type") != "office.pose.observation":
            return
        if self.known_people_only and observation.get("person_id") is None:
            return
        stamp = number(observation.get("timestamp"), time.time())
        key = self.key(observation)
        measurement = wrist_mouth_measurement(observation)
        history = self.recent.setdefault(key, deque(maxlen=30))
        history.append({
            "timestamp": stamp,
            "bbox": observation.get("bbox") or {},
            "measurement": measurement or {},
        })
        state = self.active.get(key)
        if measurement and measurement["near"]:
            if not state and stamp >= self.cooldown.get(key, 0):
                approached_from_far = any(
                    item.get("measurement")
                    and not item["measurement"].get("near", False)
                    and stamp - number(item.get("timestamp")) <= 4.0
                    for item in history
                )
                if not approached_from_far:
                    return
                state = {
                    "sensor_id": str(observation.get("sensor_id", "office-main")),
                    "track_id": str(observation.get("track_id", "")),
                    "person_id": observation.get("person_id"),
                    "started_at": stamp,
                    "last_near_at": stamp,
                    "near_count": 0,
                    "measurements": [],
                    "bboxes": [],
                }
                self.active[key] = state
            if state:
                state["last_near_at"] = stamp
                state["near_count"] += 1
                state["measurements"].append(measurement)
                state["bboxes"].append(observation.get("bbox") or {})
        elif state and stamp - state["last_near_at"] >= self.exit_seconds:
            self.finish(key, state)
        for stale_key, stale in list(self.active.items()):
            if stamp - stale["last_near_at"] > 5:
                self.finish(stale_key, stale)
        self.archive_due()

    def finish(self, key: str, state: dict[str, Any]) -> None:
        self.active.pop(key, None)
        duration = state["last_near_at"] - state["started_at"]
        if duration < self.minimum_seconds or int(state["near_count"]) < 2:
            return
        score = min(0.99, max(0.01, 1 - min(
            number(item.get("distance_bbox_heights"), 1) for item in state["measurements"]
        ) / 0.5))
        signature = (
            f"{state['sensor_id']}:{state['person_id']}:{state['track_id']}:"
            f"{state['started_at']:.2f}:{state['last_near_at']:.2f}:{RULE_VERSION}"
        )
        sample_id = hashlib.sha256(signature.encode()).hexdigest()[:24]
        now = time.time()
        with closing(self.store.connect()) as connection, connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO flywheel_candidates("
                "sample_id,sensor_id,track_id,person_id,activity,started_at,ended_at,rule_version,"
                "score,trigger_json,bbox_json,status,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sample_id, state["sensor_id"], state["track_id"], state["person_id"], "drinking",
                    state["started_at"], state["last_near_at"], RULE_VERSION, score,
                    json.dumps({"measurements": state["measurements"][-12:]}, ensure_ascii=False),
                    json.dumps(self.union_bbox(state["bboxes"]), ensure_ascii=False),
                    "pending_clip", now, now,
                ),
            )
            candidate_id = int(cursor.lastrowid) if cursor.lastrowid else 0
        if candidate_id:
            self.pending_archive.append((state["last_near_at"] + self.post_roll + 1.5, candidate_id))
            self.cooldown[key] = state["last_near_at"] + self.cooldown_seconds
            LOGGER.info("candidate id=%s person=%s score=%.3f", candidate_id, state["person_id"], score)

    @staticmethod
    def union_bbox(values: list[dict[str, Any]]) -> dict[str, float]:
        values = [value for value in values if value]
        if not values:
            return {"left": 0, "top": 0, "right": 1, "bottom": 1}
        left = min(number(value.get("left")) for value in values)
        top = min(number(value.get("top")) for value in values)
        right = max(number(value.get("right"), 1) for value in values)
        bottom = max(number(value.get("bottom"), 1) for value in values)
        margin_x = max(0.03, (right - left) * 0.28)
        margin_y = max(0.04, (bottom - top) * 0.18)
        return {
            "left": max(0, left - margin_x), "top": max(0, top - margin_y),
            "right": min(1, right + margin_x), "bottom": min(1, bottom + margin_y),
        }

    def archive_due(self) -> None:
        now = time.time()
        due = [item for item in self.pending_archive if item[0] <= now]
        self.pending_archive = [item for item in self.pending_archive if item[0] > now]
        for _, candidate_id in due:
            try:
                self.archive(candidate_id)
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
                LOGGER.warning("clip archive failed candidate=%s: %s", candidate_id, error)
                with closing(self.store.connect()) as connection, connection:
                    connection.execute(
                        "UPDATE flywheel_candidates SET status='clip_error',clip_error=?,updated_at=? WHERE id=?",
                        (str(error)[:1000], time.time(), candidate_id),
                    )
        if now - self._last_object_backfill_at >= 15:
            self._last_object_backfill_at = now
            self.backfill_objects(limit=1)

    def backfill_objects(self, limit: int = 1) -> None:
        with closing(self.store.connect()) as connection:
            rows = connection.execute(
                "SELECT c.id, c.clip_path FROM flywheel_candidates c "
                "LEFT JOIN flywheel_labels l ON l.candidate_id=c.id "
                "WHERE c.clip_path != '' AND c.object_status IN ('pending','retry') "
                "AND (c.person_id IS NOT NULL OR l.id IS NOT NULL) "
                "ORDER BY CASE l.label WHEN 'positive' THEN 0 WHEN 'negative' THEN 1 ELSE 2 END, "
                "c.started_at DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        for candidate_id, clip_path in rows:
            path = (self.store.data_dir / str(clip_path)).resolve()
            if path.is_file() and self.store.clip_dir.resolve() in path.parents:
                self.detect_objects(int(candidate_id), path)

    @staticmethod
    def video_duration(path: Path) -> float:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            check=True, capture_output=True, timeout=20,
        )
        return max(0.1, number(result.stdout.decode("utf-8").strip(), 0.1))

    @staticmethod
    def video_frame(path: Path, offset: float) -> bytes:
        result = subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-ss", f"{offset:.3f}", "-i", str(path),
                "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
            ],
            check=True, capture_output=True, timeout=30,
        )
        if not result.stdout:
            raise RuntimeError("ffmpeg returned an empty candidate frame")
        return result.stdout

    def detect_objects(self, candidate_id: int, clip: Path) -> None:
        now = time.time()
        try:
            duration = self.video_duration(clip)
            offsets = [duration * fraction for fraction in (0.2, 0.35, 0.5, 0.65, 0.8)]
            frames = []
            for index, offset in enumerate(offsets):
                evidence = self.object_detector.infer(self.video_frame(clip, offset))
                evidence["frame_index"] = index
                evidence["offset_seconds"] = round(offset, 3)
                frames.append(evidence)
            confirmed, confirmed_frames, best_score = fuse_object_scores(
                [number(frame.get("best_score")) for frame in frames],
                self.object_detector.threshold, self.object_high_threshold, self.object_required_frames,
            )
            evidence = {
                "detector": "grounding-dino",
                "detector_version": OBJECT_DETECTOR_VERSION,
                "prompt": self.object_detector.prompt,
                "threshold": self.object_detector.threshold,
                "high_threshold": self.object_high_threshold,
                "required_frames": self.object_required_frames,
                "confirmed_frames": confirmed_frames,
                "confirmed": confirmed,
                "best_score": best_score,
                "frames": frames,
            }
            with closing(self.store.connect()) as connection, connection:
                connection.execute(
                    "UPDATE flywheel_candidates SET object_json=?,object_status='complete',"
                    "object_error='',updated_at=? WHERE id=?",
                    (json.dumps(evidence, ensure_ascii=False), now, candidate_id),
                )
            LOGGER.info(
                "object evidence candidate=%s confirmed=%s frames=%s best=%.3f",
                candidate_id, evidence["confirmed"], confirmed_frames, evidence["best_score"],
            )
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError, KeyError) as error:
            LOGGER.warning("object detection failed candidate=%s: %s", candidate_id, error)
            with closing(self.store.connect()) as connection, connection:
                connection.execute(
                    "UPDATE flywheel_candidates SET object_status='retry',object_error=?,updated_at=? WHERE id=?",
                    (str(error)[:1000], now, candidate_id),
                )

    @staticmethod
    def segment_time(path: Path) -> float | None:
        try:
            value = path.stem[:26]
            return datetime.strptime(value, "%Y-%m-%d_%H-%M-%S-%f").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            return None

    def archive(self, candidate_id: int) -> None:
        sample = self.store.candidate(candidate_id)
        if not sample:
            return
        start = number(sample["started_at"]) - self.pre_roll
        end = number(sample["ended_at"]) + self.post_roll
        folder = self.recordings / str(sample["sensor_id"])
        segments = []
        for path in folder.glob("*.mp4"):
            stamp = self.segment_time(path)
            if stamp is not None and stamp <= end + 3 and stamp + 3 >= start:
                segments.append((stamp, path))
        segments.sort()
        if not segments:
            raise RuntimeError("rolling recording segments expired or unavailable")
        bbox = sample.get("bbox") or {}
        left = max(0, min(1918, round(number(bbox.get("left")) * 1920)))
        top = max(0, min(1078, round(number(bbox.get("top")) * 1080)))
        width = max(64, min(1920 - left, round((number(bbox.get("right"), 1) - number(bbox.get("left"))) * 1920)))
        height = max(64, min(1080 - top, round((number(bbox.get("bottom"), 1) - number(bbox.get("top"))) * 1080)))
        width -= width % 2
        height -= height % 2
        destination = self.store.clip_dir / f"{sample['sample_id']}.mp4"
        temporary = destination.with_suffix(".tmp.mp4")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as manifest:
            manifest_path = Path(manifest.name)
            for _, path in segments:
                manifest.write("file '" + str(path).replace("'", "'\\''") + "'\n")
        try:
            command = [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(manifest_path),
                "-vf", f"crop={width}:{height}:{left}:{top},scale='min(640,iw)':-2",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-movflags", "+faststart",
                str(temporary),
            ]
            subprocess.run(command, check=True, timeout=90)
            temporary.replace(destination)
        finally:
            manifest_path.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
        relative = str(destination.relative_to(self.store.data_dir))
        with closing(self.store.connect()) as connection, connection:
            connection.execute(
                "UPDATE flywheel_candidates SET clip_path=?,status='unlabeled',clip_error='',updated_at=? WHERE id=?",
                (relative, time.time(), candidate_id),
            )
        self.detect_objects(candidate_id, destination)


def main() -> None:
    logging.basicConfig(level=os.environ.get("OFFICE_LOG_LEVEL", "INFO"))
    initialize_flywheel_schema(DATABASE_FILE)
    store = FlywheelStore(DATABASE_FILE)
    miner = DrinkingCandidateMiner(store, Path(os.environ.get("OFFICE_RECORDINGS_DIR", "/recordings")))
    consumer = Consumer({
        "bootstrap.servers": os.environ.get("OFFICE_KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092"),
        "group.id": os.environ.get("OFFICE_FLYWHEEL_GROUP", "office-flywheel-v1"),
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    })
    topic = os.environ.get("OFFICE_POSE_TOPIC", "mdx-office-pose")
    consumer.subscribe([topic])
    LOGGER.info("flywheel subscribed topic=%s rule=%s", topic, RULE_VERSION)
    try:
        while True:
            message = consumer.poll(0.5)
            if message is None:
                miner.archive_due()
                continue
            if message.error():
                if message.error().code() != KafkaError._PARTITION_EOF:
                    LOGGER.warning("Kafka error: %s", message.error())
                continue
            try:
                payload = json.loads(message.value().decode("utf-8"))
                if isinstance(payload, dict):
                    miner.observe(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                LOGGER.warning("invalid pose message: %s", error)
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
