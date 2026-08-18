# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporal body/hand motion utilities for the office activity pipeline."""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


BODYPOSE34_NAMES = (
    "pelvis", "left_hip", "right_hip", "torso", "left_knee", "right_knee",
    "neck", "left_ankle", "right_ankle", "left_big_toe", "right_big_toe",
    "left_small_toe", "right_small_toe", "left_heel", "right_heel", "nose",
    "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb",
    "right_thumb", "head_top", "spine",
)


@dataclass(frozen=True)
class Keypoint:
    """One named pose point in image/local-body coordinates."""

    name: str
    x: float
    y: float
    z: float
    confidence: float


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_timestamp(value: Any) -> float:
    """Parse epoch seconds or an ISO-8601 timestamp without importing app.py."""
    if isinstance(value, (int, float)):
        return float(value)
    from datetime import datetime

    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def extract_embedding(item: dict[str, Any]) -> list[float]:
    """Read the ReID vector emitted by the different nvmsgconv schema variants."""
    raw = item.get("embedding") or item.get("embeddings") or {}
    if isinstance(raw, dict):
        raw = raw.get("vector", [])
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        raw = raw[0].get("vector", [])
    if not isinstance(raw, list):
        return []
    return [_number(value) for value in raw]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def parse_pose(item: dict[str, Any]) -> dict[str, Keypoint]:
    """Normalize NVS Pose, BodyPose JSON and flat pose25d payloads."""
    pose = item.get("pose") or item.get("poses") or {}
    if isinstance(pose, list) and pose and isinstance(pose[0], dict) and "keypoints" in pose[0]:
        pose = pose[0]
    points = pose.get("keypoints", []) if isinstance(pose, dict) else pose
    result: dict[str, Keypoint] = {}
    if isinstance(points, list) and points and isinstance(points[0], dict):
        for index, point in enumerate(points):
            fallback = BODYPOSE34_NAMES[index] if index < len(BODYPOSE34_NAMES) else str(index)
            name = str(point.get("name") or point.get("id") or fallback)
            coordinate = point.get("coordinate") or point.get("coordinates") or point
            result[name] = Keypoint(
                name,
                _number(coordinate.get("x")),
                _number(coordinate.get("y")),
                _number(coordinate.get("z")),
                _number(point.get("confidence", coordinate.get("confidence", point.get("score", 0.0)))),
            )
        return result
    flat = None
    if isinstance(pose, dict):
        flat = pose.get("pose25d") or pose.get("pose3d") or pose.get("data")
    if flat is None:
        flat = item.get("pose25d") or item.get("pose3d")
    if isinstance(flat, list):
        for index in range(min(len(flat) // 4, len(BODYPOSE34_NAMES))):
            x, y, z, confidence = flat[index * 4:index * 4 + 4]
            name = BODYPOSE34_NAMES[index]
            result[name] = Keypoint(name, _number(x), _number(y), _number(z), _number(confidence))
    return result


def joint_angle(first: Keypoint | None, center: Keypoint | None, last: Keypoint | None) -> float | None:
    """Return the 3-D angle ABC in degrees."""
    if not first or not center or not last or min(first.confidence, center.confidence, last.confidence) < 0.35:
        return None
    vector_a = (first.x - center.x, first.y - center.y, first.z - center.z)
    vector_b = (last.x - center.x, last.y - center.y, last.z - center.z)
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))
    if norm_a == 0 or norm_b == 0:
        return None
    cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(vector_a, vector_b, strict=True)) / (norm_a * norm_b)))
    return math.degrees(math.acos(cosine))


def _bbox_center(sample: dict[str, Any]) -> tuple[float, float]:
    bbox = sample.get("bbox") or {}
    return (
        (_number(bbox.get("leftX")) + _number(bbox.get("rightX"))) / 2,
        (_number(bbox.get("topY")) + _number(bbox.get("bottomY"))) / 2,
    )


def _point_delta(samples: list[dict[str, Any]], name: str) -> dict[str, float] | None:
    valid = [(sample["timestamp"], sample["pose"].get(name)) for sample in samples]
    valid = [(stamp, point) for stamp, point in valid if point and point.confidence >= 0.35]
    if len(valid) < 2:
        return None
    start_time, start = valid[0]
    end_time, end = valid[-1]
    duration = max(0.001, end_time - start_time)
    return {
        "dx": round(end.x - start.x, 4),
        "dy": round(end.y - start.y, 4),
        "dz_relative": round(end.z - start.z, 4),
        "speed": round(math.dist((start.x, start.y, start.z), (end.x, end.y, end.z)) / duration, 4),
    }


def summarize_motion(samples: list[dict[str, Any]], frame_width: float, frame_height: float) -> dict[str, Any]:
    """Aggregate synchronized BodyPose samples into measurable motion facts."""
    if not samples:
        return {"body_motion": {}, "posture_transitions": [], "quality": {"sample_count": 0}}
    samples = sorted(samples, key=lambda value: value["timestamp"])
    duration = max(0.001, samples[-1]["timestamp"] - samples[0]["timestamp"])
    start_center = _bbox_center(samples[0])
    end_center = _bbox_center(samples[-1])
    dx = (end_center[0] - start_center[0]) / max(1.0, frame_width)
    dy = (end_center[1] - start_center[1]) / max(1.0, frame_height)
    direction = "stationary"
    if abs(dx) >= 0.03 and abs(dx) >= abs(dy):
        direction = "right_in_image" if dx > 0 else "left_in_image"
    elif abs(dy) >= 0.03:
        direction = "down_in_image" if dy > 0 else "up_in_image"

    angle_specs = {
        "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
        "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
        "left_knee": ("left_hip", "left_knee", "left_ankle"),
        "right_knee": ("right_hip", "right_knee", "right_ankle"),
    }
    angles: dict[str, dict[str, float]] = {}
    for label, (first, center, last) in angle_specs.items():
        values = [joint_angle(sample["pose"].get(first), sample["pose"].get(center), sample["pose"].get(last)) for sample in samples]
        values = [value for value in values if value is not None]
        if values:
            angles[label] = {
                "start_deg": round(values[0], 1),
                "end_deg": round(values[-1], 1),
                "change_deg": round(values[-1] - values[0], 1),
                "angular_velocity_deg_sec": round((values[-1] - values[0]) / duration, 2),
            }

    visible_ratios = [sum(point.confidence >= 0.35 for point in sample["pose"].values()) / 34 for sample in samples]
    bbox_heights = [
        (_number(sample["bbox"].get("bottomY")) - _number(sample["bbox"].get("topY"))) / max(1.0, frame_height)
        for sample in samples
    ]
    transitions: list[dict[str, Any]] = []
    if len(bbox_heights) >= 2 and bbox_heights[-1] - bbox_heights[0] >= 0.12:
        transitions.append({"type": "stood_up_or_approached_camera", "confidence": 0.7})
    elif len(bbox_heights) >= 2 and bbox_heights[0] - bbox_heights[-1] >= 0.12:
        transitions.append({"type": "sat_down_or_moved_away", "confidence": 0.7})
    if direction in {"left_in_image", "right_in_image"}:
        transitions.append({"type": "translated_in_image", "direction": direction, "confidence": 0.8})

    return {
        "window": {"start": samples[0]["timestamp"], "end": samples[-1]["timestamp"], "duration_seconds": round(duration, 3)},
        "body_motion": {
            "bbox_center_dx_normalized": round(dx, 4),
            "bbox_center_dy_normalized": round(dy, 4),
            "bbox_speed_per_second": round(math.hypot(dx, dy) / duration, 4),
            "direction_in_image": direction,
            "joints": {name: delta for name in ("head_top", "nose", "left_wrist", "right_wrist", "pelvis") if (delta := _point_delta(samples, name))},
            "joint_angles": angles,
        },
        "posture_transitions": transitions,
        "quality": {
            "sample_count": len(samples),
            "visible_keypoints_ratio": round(statistics.fmean(visible_ratios), 3) if visible_ratios else 0.0,
            "missing_keypoints_ratio": round(1 - statistics.fmean(visible_ratios), 3) if visible_ratios else 1.0,
            "pose_confidence": round(statistics.fmean(
                [point.confidence for sample in samples for point in sample["pose"].values()]
            ), 3) if any(sample["pose"] for sample in samples) else 0.0,
            "z_is_relative": True,
        },
    }


def facts_json(facts: dict[str, Any]) -> str:
    return json.dumps(facts, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class RtspFrameBuffer:
    """Keep a bounded low-rate JPEG buffer from the same RTSP camera stream."""

    def __init__(self, rtsp_url: str, retention_seconds: float = 120, fps: float = 2) -> None:
        self.rtsp_url = rtsp_url
        self.retention_seconds = retention_seconds
        self.fps = fps
        self.frames: deque[tuple[float, bytes]] = deque()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, name="office-frame-buffer", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", self.rtsp_url,
                "-vf", f"fps={self.fps}", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
            ]
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                buffer = bytearray()
                assert process.stdout is not None
                while not self.stop_event.is_set():
                    chunk = process.stdout.read(65536)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    while True:
                        start = buffer.find(b"\xff\xd8")
                        end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                        if start < 0 or end < 0:
                            if len(buffer) > 8_000_000:
                                del buffer[:-2]
                            break
                        jpeg = bytes(buffer[start:end + 2])
                        del buffer[:end + 2]
                        stamp = time.time()
                        with self.lock:
                            self.frames.append((stamp, jpeg))
                            cutoff = stamp - self.retention_seconds
                            while self.frames and self.frames[0][0] < cutoff:
                                self.frames.popleft()
                process.terminate()
                process.wait(timeout=5)
            except (OSError, subprocess.SubprocessError):
                time.sleep(2)

    def between(self, start: float, end: float) -> list[tuple[float, bytes]]:
        with self.lock:
            return [(stamp, data) for stamp, data in self.frames if start - 0.5 <= stamp <= end + 0.5]


def select_storyboard_frames(
    frames: list[tuple[float, bytes]], count: int, samples: list[dict[str, Any]] | None = None,
) -> list[tuple[float, bytes]]:
    """Keep endpoints, largest pose changes, then fill remaining slots uniformly."""
    count = max(1, count)
    if len(frames) <= count:
        return frames
    indexes = {0, len(frames) - 1}
    changes: list[tuple[float, float]] = []
    if samples:
        ordered = sorted(samples, key=lambda value: float(value["timestamp"]))
        for previous, current in zip(ordered, ordered[1:]):
            px, py = _bbox_center(previous)
            cx, cy = _bbox_center(current)
            score = math.hypot(cx - px, cy - py)
            for name in ("left_wrist", "right_wrist", "head_top", "pelvis"):
                first = previous.get("pose", {}).get(name)
                second = current.get("pose", {}).get(name)
                if first and second and min(first.confidence, second.confidence) >= 0.35:
                    score += math.dist((first.x, first.y, first.z), (second.x, second.y, second.z))
            changes.append((score, float(current["timestamp"])))
        for _score, stamp in sorted(changes, reverse=True):
            indexes.add(min(range(len(frames)), key=lambda index: abs(frames[index][0] - stamp)))
            if len(indexes) >= count:
                break
    if len(indexes) < count:
        if count == 1:
            indexes = {len(frames) // 2}
        else:
            for index in range(count):
                indexes.add(round(index * (len(frames) - 1) / (count - 1)))
                if len(indexes) >= count:
                    break
    return [frames[index] for index in sorted(indexes)]


def nearest_bbox(samples: list[dict[str, Any]], stamp: float) -> dict[str, Any]:
    if not samples:
        return {}
    return min(samples, key=lambda sample: abs(float(sample["timestamp"]) - stamp)).get("bbox") or {}


def crop_person_frames(
    frames: list[tuple[float, bytes]], samples: list[dict[str, Any]], margin: float = 0.2,
) -> list[tuple[float, bytes]]:
    result: list[tuple[float, bytes]] = []
    for stamp, data in frames:
        bbox = nearest_bbox(samples, stamp)
        if not bbox:
            continue
        try:
            image = Image.open(BytesIO(data)).convert("RGB")
            width, height = image.size
            left = float(bbox.get("leftX", 0))
            top = float(bbox.get("topY", 0))
            right = float(bbox.get("rightX", width))
            bottom = float(bbox.get("bottomY", height))
            margin_x, margin_y = (right - left) * margin, (bottom - top) * margin
            cropped = image.crop((max(0, left - margin_x), max(0, top - margin_y), min(width, right + margin_x), min(height, bottom + margin_y)))
            output = BytesIO()
            cropped.save(output, format="JPEG", quality=85)
            result.append((stamp, output.getvalue()))
        except (OSError, TypeError, ValueError):
            continue
    return result


def build_storyboard(
    frames: list[tuple[float, bytes]],
    samples: list[dict[str, Any]],
    output: Path,
    *,
    person_crop: bool,
    columns: int = 3,
) -> str:
    """Create one labelled contact sheet; return an empty path when no synchronized frames exist."""
    if not frames:
        return ""
    panels: list[tuple[float, Image.Image]] = []
    for stamp, data in frames:
        try:
            image = Image.open(BytesIO(data)).convert("RGB")
        except OSError:
            continue
        if person_crop:
            bbox = nearest_bbox(samples, stamp)
            if bbox:
                width, height = image.size
                left = float(bbox.get("leftX", 0))
                top = float(bbox.get("topY", 0))
                right = float(bbox.get("rightX", width))
                bottom = float(bbox.get("bottomY", height))
                margin_x = (right - left) * 0.2
                margin_y = (bottom - top) * 0.2
                image = image.crop((max(0, left - margin_x), max(0, top - margin_y), min(width, right + margin_x), min(height, bottom + margin_y)))
        image.thumbnail((480, 320))
        panels.append((stamp, image.copy()))
    if not panels:
        return ""
    panel_width, panel_height = 480, 348
    rows = math.ceil(len(panels) / columns)
    sheet = Image.new("RGB", (panel_width * columns, panel_height * rows), "#10151d")
    draw = ImageDraw.Draw(sheet)
    origin = panels[0][0]
    for index, (stamp, panel) in enumerate(panels):
        x = (index % columns) * panel_width
        y = (index // columns) * panel_height
        sheet.paste(panel, (x + (panel_width - panel.width) // 2, y))
        draw.text((x + 8, y + 324), f"+{stamp - origin:.1f}s", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=85)
    return str(output)


class MediaPipeHandAnalyzer:
    """Optional MediaPipe Tasks adapter; absence degrades to explicit unavailable evidence."""

    def __init__(self, model_path: str = "") -> None:
        self.model_path = model_path
        self.landmarker: Any | None = None
        self.error = ""
        if not model_path or not Path(model_path).is_file():
            self.error = "hand model is not configured"
            return
        try:
            import mediapipe as mp
            import numpy as np

            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
            )
            self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
            self.mp = mp
            self.np = np
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            self.error = str(error)

    def analyze(self, frames: list[tuple[float, bytes]]) -> dict[str, Any]:
        if self.landmarker is None:
            return {"available": False, "reason": self.error, "observations": []}
        observations = []
        for stamp, data in frames:
            try:
                image = Image.open(BytesIO(data)).convert("RGB")
                result = self.landmarker.detect(self.mp.Image(
                    image_format=self.mp.ImageFormat.SRGB,
                    data=self.np.asarray(image),
                ))
            except (OSError, RuntimeError, ValueError) as error:
                self.error = str(error)
                continue
            hands = []
            for index, landmarks in enumerate(result.hand_landmarks):
                z_values = [float(point.z) for point in landmarks]
                handedness = "unknown"
                if index < len(result.handedness) and result.handedness[index]:
                    handedness = str(result.handedness[index][0].category_name or "unknown").casefold()
                hands.append({
                    "handedness": handedness,
                    "landmark_count": len(landmarks),
                    "wrist": [round(float(landmarks[0].x), 5), round(float(landmarks[0].y), 5), round(float(landmarks[0].z), 5)],
                    "fingertips": {
                        name: [round(float(landmarks[position].x), 5), round(float(landmarks[position].y), 5), round(float(landmarks[position].z), 5)]
                        for name, position in (("thumb", 4), ("index", 8), ("middle", 12), ("ring", 16), ("pinky", 20))
                    },
                    "relative_z_min": round(min(z_values), 5),
                    "relative_z_max": round(max(z_values), 5),
                    "relative_z_span": round(max(z_values) - min(z_values), 5),
                })
            observations.append({"timestamp": stamp, "hands": hands})
        motions = []
        by_hand: dict[str, list[tuple[float, list[float]]]] = {}
        for observation in observations:
            for hand in observation["hands"]:
                by_hand.setdefault(hand["handedness"], []).append((observation["timestamp"], hand["wrist"]))
        for handedness, points in by_hand.items():
            if len(points) < 2:
                continue
            elapsed = max(0.001, points[-1][0] - points[0][0])
            delta = [points[-1][1][axis] - points[0][1][axis] for axis in range(3)]
            motions.append({
                "handedness": handedness,
                "wrist_delta": [round(value, 5) for value in delta],
                "relative_speed": round(math.sqrt(sum(value * value for value in delta)) / elapsed, 5),
            })
        return {
            "available": True,
            "z_is_relative": True,
            "observations": observations,
            "motion": motions,
            "error": self.error,
        }
