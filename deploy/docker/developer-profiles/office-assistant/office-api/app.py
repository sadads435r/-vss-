# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small read-only office dashboard facade for the VSS alerts profile."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from datetime import datetime
from datetime import time as datetime_time
from datetime import timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from flywheel import FlywheelStore
from flywheel import initialize_flywheel_schema
from workstation import WorkstationEngine
from workstation import default_config as default_workstation_config
from workstation import initialize_schema as initialize_workstation_schema
from workstation import validate_workstation_config
from workstation import worker as workstation_worker

CONFIG_FILE = Path(os.environ.get("OFFICE_CONFIG_FILE", "/config/office-config.yaml"))
DATABASE_FILE = Path(os.environ.get("OFFICE_DATABASE_FILE", "/data/office.db"))
CLIP_DIR = Path(os.environ.get("OFFICE_CLIP_DIR", "/data/clips"))
ELASTICSEARCH_URL = os.environ.get("ELASTICSEARCH_URL", "http://127.0.0.1:9200").rstrip("/")
VSS_AGENT_URL = os.environ.get("VSS_AGENT_URL", "http://127.0.0.1:8000").rstrip("/")
VSS_AGENT_CHAT_URL = os.environ.get("VSS_AGENT_CHAT_URL", f"{VSS_AGENT_URL}/chat/stream")
VSS_VST_URL = os.environ.get("VSS_VST_URL", "http://127.0.0.1:30888").rstrip("/")
STATIC_DIR = Path(__file__).with_name("static")


class ConfigurationError(ValueError):
    """Raised when office-config.yaml is incomplete or unsafe."""


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration root must be a mapping")
    if not raw.get("timezone"):
        raise ConfigurationError("missing required field: timezone")
    defaults = default_workstation_config()
    configured_workstation = raw.get("workstation", {})
    if not isinstance(configured_workstation, dict):
        raise ConfigurationError("workstation must be a mapping")
    merged_workstation = {**defaults, **configured_workstation}
    for nested in ("motion_pipeline", "identity"):
        configured_nested = configured_workstation.get(nested, {})
        if not isinstance(configured_nested, dict):
            raise ConfigurationError(f"workstation.{nested} must be a mapping")
        merged_workstation[nested] = {**defaults[nested], **configured_nested}
    raw["workstation"] = merged_workstation
    try:
        ZoneInfo(str(raw["timezone"]))
    except (KeyError, ValueError) as error:
        raise ConfigurationError(f"invalid timezone: {raw['timezone']}") from error
    for section in ("camera", "occupancy", "schedule", "retention", "rules", "zones"):
        if section not in raw:
            raise ConfigurationError(f"missing required section: {section}")
    camera = raw["camera"]
    if not isinstance(camera, dict) or not str(camera.get("rtsp_url", "")).startswith("rtsp://"):
        raise ConfigurationError("camera.rtsp_url must use rtsp://")
    days = int(raw["retention"].get("event_days", 0))
    if days < 1 or days > 365:
        raise ConfigurationError("retention.event_days must be between 1 and 365")
    poll_seconds = float(raw["occupancy"].get("poll_seconds", 0))
    departure_timeout = float(raw["occupancy"].get("departure_timeout_seconds", 0))
    confidence = float(raw["occupancy"].get("minimum_person_confidence", -1))
    if poll_seconds < 0.5:
        raise ConfigurationError("occupancy.poll_seconds must be at least 0.5")
    if departure_timeout < poll_seconds:
        raise ConfigurationError("occupancy.departure_timeout_seconds must be at least poll_seconds")
    if not 0 <= confidence <= 1:
        raise ConfigurationError("occupancy.minimum_person_confidence must be between 0 and 1")
    for rule_name in ("after_hours_presence", "restricted_zone_entry", "dwell_time", "occupancy_limit"):
        if rule_name not in raw["rules"]:
            raise ConfigurationError(f"missing required rule: {rule_name}")
    for zone in raw["zones"]:
        polygon = zone.get("polygon", [])
        if len(polygon) < 3:
            raise ConfigurationError(f"zone {zone.get('id', '<unknown>')} needs at least three points")
        if any(len(point) != 2 or any(float(value) < 0 or float(value) > 1 for value in point) for point in polygon):
            raise ConfigurationError("zone coordinates must be normalized into [0, 1]")
    try:
        validate_workstation_config(raw)
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
    return raw


def initialize_database() -> None:
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DATABASE_FILE, timeout=30)) as connection, connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS acknowledgements "
            "(event_id TEXT PRIMARY KEY, acknowledged_at INTEGER NOT NULL, note TEXT NOT NULL DEFAULT '')"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS presence_sessions ("
            "session_id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL, track_id TEXT NOT NULL, "
            "arrived_at REAL NOT NULL, last_seen_at REAL NOT NULL, left_at REAL)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_presence_open "
            "ON presence_sessions(camera_id, track_id, left_at)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS occupancy_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
    initialize_workstation_schema(DATABASE_FILE)
    initialize_flywheel_schema(DATABASE_FILE)


def request_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method=method)
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=4) as response:
        return json.loads(response.read().decode("utf-8"))


def service_status(url: str) -> dict[str, Any]:
    started = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return {"healthy": response.status < 500, "latency_ms": round((time.monotonic() - started) * 1000)}
    except (OSError, urllib.error.URLError) as error:
        return {"healthy": False, "error": str(error)}


def parse_timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def parse_activity_window(
    query: dict[str, list[str]], timezone: ZoneInfo, now: float | None = None,
) -> tuple[float, float]:
    """Parse a local date or an ISO date/datetime range into epoch seconds."""
    current = time.time() if now is None else now
    if query.get("date"):
        try:
            day = datetime.fromisoformat(query["date"][0]).date()
        except ValueError as error:
            raise ValueError("date must use ISO format YYYY-MM-DD") from error
        start = datetime.combine(day, datetime_time.min, timezone)
        end = start + timedelta(days=1)
    else:
        today = datetime.fromtimestamp(current, timezone).date()
        start_value = query.get("start", [today.isoformat()])[0]
        end_value = query.get("end", [today.isoformat()])[0]

        def boundary(value: str, *, end_boundary: bool) -> datetime:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError("start and end must be ISO dates or datetimes") from error
            is_date = len(value) == 10
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone)
            if is_date and end_boundary:
                parsed += timedelta(days=1)
            return parsed

        start = boundary(start_value, end_boundary=False)
        end = boundary(end_value, end_boundary=True)
    if end <= start:
        raise ValueError("end must be after start")
    if end - start > timedelta(days=366):
        raise ValueError("activity query range cannot exceed 366 days")
    return start.timestamp(), end.timestamp()


def fetch_latest_frame(config: dict[str, Any]) -> dict[str, Any] | None:
    # mdx-raw 的 sensorId 是相机名（如 office-main）；vss_sensor_id 是 VST 的 UUID，不能用于 ES 过滤
    sensor_id = str(config["camera"].get("id", "")).strip()
    # 帧数据源为 mdx-raw-*（RT-CV 检测事件经 Kafka/ES 落库），排除 1970 占位空消息
    if sensor_id:
        query: dict[str, Any] = {
            "bool": {
                "must": [{"term": {"sensorId.keyword": sensor_id}}],
                "filter": [{"range": {"timestamp": {"gt": "2020-01-01T00:00:00Z"}}}],
            }
        }
    else:
        query = {
            "bool": {
                "must": [{"match_all": {}}],
                "filter": [{"range": {"timestamp": {"gt": "2020-01-01T00:00:00Z"}}}],
            }
        }
    payload = {
        "size": 8,
        "sort": [{"timestamp": {"order": "desc", "unmapped_type": "date"}}],
        "query": query,
    }
    result = request_json(f"{ELASTICSEARCH_URL}/mdx-raw-*/_search", method="POST", payload=payload)
    hits = result.get("hits", {}).get("hits", [])
    if not hits:
        return None
    # RT-CV 事件成对落库（sentinel conf=-0.1 帧 + 真实置信度帧），同刻多条排序不稳定。
    # 优先返回带真实 person 置信度（>0）的帧，避免随机取到 sentinel 帧导致占用判定抖动。
    for hit in hits:
        source = hit.get("_source", {})
        for obj in source.get("objects", []):
            if str(obj.get("type", "")).casefold() == "person":
                conf = float(obj.get("confidence", -1))
                bbox_conf = float((obj.get("bbox") or {}).get("confidence", -1))
                if max(conf, bbox_conf) > 0:
                    return dict(source)
    return dict(hits[0].get("_source", {}))


def extract_people(frame: dict[str, Any], minimum_confidence: float) -> dict[str, float]:
    people: dict[str, float] = {}
    for detected_object in frame.get("objects", []):
        if str(detected_object.get("type", "")).casefold() != "person":
            continue
        track_id = str(detected_object.get("id", "")).strip()
        confidence = float(detected_object.get("confidence", 0))
        if track_id and confidence >= minimum_confidence:
            people[track_id] = confidence
    return people


def update_presence(frame: dict[str, Any] | None, config: dict[str, Any], now: float | None = None) -> None:
    current_time = time.time() if now is None else now
    timeout = float(config["occupancy"].get("departure_timeout_seconds", 10))
    with closing(sqlite3.connect(DATABASE_FILE, timeout=30)) as connection, connection:
        if frame:
            frame_token = f"{frame.get('sensorId', '')}:{frame.get('id', '')}:{frame.get('timestamp', '')}"
            previous = connection.execute(
                "SELECT value FROM occupancy_state WHERE key = 'last_frame_token'"
            ).fetchone()
            if not previous or previous[0] != frame_token:
                camera_id = str(frame.get("sensorId") or config["camera"]["id"])
                seen_at = parse_timestamp(str(frame.get("timestamp"))) if frame.get("timestamp") else current_time
                people = extract_people(
                    frame,
                    float(config["occupancy"].get("minimum_person_confidence", 0.3)),
                )
                for track_id in people:
                    open_session = connection.execute(
                        "SELECT session_id FROM presence_sessions "
                        "WHERE camera_id = ? AND track_id = ? AND left_at IS NULL ORDER BY session_id DESC LIMIT 1",
                        (camera_id, track_id),
                    ).fetchone()
                    if open_session:
                        connection.execute(
                            "UPDATE presence_sessions SET last_seen_at = ? WHERE session_id = ?",
                            (seen_at, open_session[0]),
                        )
                    else:
                        connection.execute(
                            "INSERT INTO presence_sessions(camera_id, track_id, arrived_at, last_seen_at) "
                            "VALUES (?, ?, ?, ?)",
                            (camera_id, track_id, seen_at, seen_at),
                        )
                connection.execute(
                    "INSERT OR REPLACE INTO occupancy_state(key, value) VALUES ('last_frame_token', ?)",
                    (frame_token,),
                )
                connection.execute(
                    "INSERT OR REPLACE INTO occupancy_state(key, value) VALUES ('last_frame_timestamp', ?)",
                    (str(seen_at),),
                )
        connection.execute(
            "UPDATE presence_sessions SET left_at = last_seen_at "
            "WHERE left_at IS NULL AND last_seen_at < ?",
            (current_time - timeout,),
        )


def occupancy_snapshot(config: dict[str, Any], now: float | None = None) -> dict[str, Any]:
    current_time = time.time() if now is None else now
    timeout = float(config["occupancy"].get("departure_timeout_seconds", 10))
    timezone = ZoneInfo(str(config["timezone"]))
    local_now = datetime.fromtimestamp(current_time, timezone)
    day_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    history_limit = min(max(int(config["occupancy"].get("history_limit", 100)), 1), 500)
    with closing(sqlite3.connect(DATABASE_FILE, timeout=30)) as connection, connection:
        active_rows = connection.execute(
            "SELECT session_id, camera_id, track_id, arrived_at, last_seen_at "
            "FROM presence_sessions WHERE left_at IS NULL AND last_seen_at >= ? ORDER BY arrived_at",
            (current_time - timeout,),
        ).fetchall()
        history_rows = connection.execute(
            "SELECT session_id, camera_id, track_id, arrived_at, last_seen_at, left_at "
            "FROM presence_sessions WHERE arrived_at >= ? ORDER BY arrived_at DESC LIMIT ?",
            (day_start, history_limit),
        ).fetchall()
        frame_row = connection.execute(
            "SELECT value FROM occupancy_state WHERE key = 'last_frame_timestamp'"
        ).fetchone()
    last_frame_at = float(frame_row[0]) if frame_row else None
    active = [
        {
            "session_id": row[0],
            "camera_id": row[1],
            "track_id": row[2],
            "arrived_at": row[3],
            "last_seen_at": row[4],
        }
        for row in active_rows
    ]
    history = [
        {
            "session_id": row[0],
            "camera_id": row[1],
            "track_id": row[2],
            "arrived_at": row[3],
            "last_seen_at": row[4],
            "left_at": row[5],
            "duration_seconds": round((row[5] or current_time) - row[3]),
            "status": "working" if row[5] is None and row[4] >= current_time - timeout else "left",
        }
        for row in history_rows
    ]
    return {
        "current_count": len(active),
        "active_people": active,
        "today_session_count": len(history),
        "history": history,
        "last_frame_at": last_frame_at,
        "data_age_seconds": round(max(0, current_time - last_frame_at), 1) if last_frame_at else None,
        "camera_online": bool(last_frame_at and current_time - last_frame_at <= timeout * 2),
    }


def acknowledgements(event_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not event_ids:
        return {}
    placeholders = ",".join("?" for _ in event_ids)
    with closing(sqlite3.connect(DATABASE_FILE, timeout=30)) as connection, connection:
        rows = connection.execute(
            f"SELECT event_id, acknowledged_at, note FROM acknowledgements WHERE event_id IN ({placeholders})",
            event_ids,
        ).fetchall()
    return {row[0]: {"acknowledged_at": row[1], "note": row[2]} for row in rows}


def nested_value(source: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        value: Any = source
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            return value
    return None


def office_event_type(source: dict[str, Any], config: dict[str, Any]) -> str:
    """Classify a verified VSS event using privacy-preserving office rules."""
    zone_id = str(nested_value(source, "zone_id", "zoneId", "roi.id") or "")
    restricted_zones = {str(zone["id"]) for zone in config["zones"] if zone.get("restricted")}
    if config["rules"]["restricted_zone_entry"].get("enabled") and zone_id in restricted_zones:
        return "restricted_zone_entry"

    duration = float(nested_value(source, "duration_seconds", "duration", "metrics.duration") or 0)
    dwell_rule = config["rules"]["dwell_time"]
    if dwell_rule.get("enabled") and duration >= float(dwell_rule.get("seconds", 120)):
        return "dwell_time"

    objects = nested_value(source, "objects")
    count = nested_value(source, "person_count", "objectCount", "metrics.count", "metrics.personCount")
    if count is None and isinstance(objects, list):
        count = sum(1 for item in objects if not isinstance(item, dict) or item.get("type", "person") == "person")
    occupancy_rule = config["rules"]["occupancy_limit"]
    if occupancy_rule.get("enabled") and int(count or 0) > int(occupancy_rule.get("maximum_people", 10)):
        return "occupancy_limit"

    timestamp = str(nested_value(source, "@timestamp", "timestamp", "started_at") or "")
    after_hours_rule = config["rules"]["after_hours_presence"]
    if after_hours_rule.get("enabled") and timestamp:
        instant = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(ZoneInfo(config["timezone"]))
        schedule = config["schedule"]
        weekdays = {str(day).lower() for day in schedule["weekdays"]}
        start = datetime_time.fromisoformat(str(schedule["start"]))
        end = datetime_time.fromisoformat(str(schedule["end"]))
        holiday = instant.date().isoformat() in set(schedule.get("holidays", []))
        if holiday or instant.strftime("%A").lower() not in weekdays or not start <= instant.time().replace(tzinfo=None) < end:
            return "after_hours_presence"
    return "office_presence"


def search_events(query: dict[str, list[str]], config: dict[str, Any]) -> dict[str, Any]:
    size = min(max(int(query.get("limit", ["50"])[0]), 1), 200)
    filters: list[dict[str, Any]] = []
    if query.get("type"):
        filters.append({"term": {"category.keyword": query["type"][0]}})
    if query.get("camera"):
        filters.append({"term": {"sensorId.keyword": query["camera"][0]}})
    if query.get("start") or query.get("end"):
        time_range: dict[str, str] = {}
        if query.get("start"):
            time_range["gte"] = query["start"][0]
        if query.get("end"):
            time_range["lte"] = query["end"][0]
        filters.append({"range": {"@timestamp": time_range}})
    payload = {
        "size": size,
        "sort": [{"@timestamp": {"order": "desc", "unmapped_type": "date"}}],
        "query": {"bool": {"filter": filters}} if filters else {"match_all": {}},
    }
    result = request_json(f"{ELASTICSEARCH_URL}/mdx-*/_search", method="POST", payload=payload)
    hits = result.get("hits", {}).get("hits", [])
    ack = acknowledgements([str(hit.get("_id")) for hit in hits])
    events = []
    for hit in hits:
        event_id = str(hit.get("_id"))
        source = dict(hit.get("_source", {}))
        source["event_id"] = event_id
        source["acknowledgement"] = ack.get(event_id)
        source["office_event_type"] = office_event_type(source, config)
        events.append(source)
    return {"events": events, "count": len(events)}


def get_event(event_id: str) -> dict[str, Any] | None:
    result = request_json(
        f"{ELASTICSEARCH_URL}/mdx-*/_search",
        method="POST",
        payload={"size": 1, "query": {"ids": {"values": [event_id]}}},
    )
    hits = result.get("hits", {}).get("hits", [])
    if not hits:
        return None
    source = dict(hits[0].get("_source", {}))
    source["event_id"] = event_id
    source["acknowledgement"] = acknowledgements([event_id]).get(event_id)
    return source


def cleanup_expired_clips(config: dict[str, Any]) -> None:
    cutoff = time.time() - int(config["retention"]["event_days"]) * 86400
    for clip in CLIP_DIR.rglob("*"):
        if clip.is_file() and clip.stat().st_mtime < cutoff:
            clip.unlink(missing_ok=True)


def clip_path(event_id: str) -> Path:
    return CLIP_DIR / f"{hashlib.sha256(event_id.encode('utf-8')).hexdigest()}.mp4"


def event_clip_url(event: dict[str, Any]) -> str | None:
    value = nested_value(event, "clip_url", "video_url", "videoUrl", "videoPath", "metadata.videoUrl")
    return str(value) if value else None


def archive_event_clips(config: dict[str, Any]) -> None:
    result = search_events({"limit": ["200"]}, config)
    for event in result["events"]:
        status = str(nested_value(event, "verification_status", "verification.status", "verified") or "").lower()
        if status in ("false", "rejected", "no"):
            continue
        url = event_clip_url(event)
        destination = clip_path(str(event["event_id"]))
        if not url or destination.exists():
            continue
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            continue
        temporary = destination.with_suffix(".part")
        try:
            with urllib.request.urlopen(url, timeout=30) as source, temporary.open("wb") as output:
                remaining = 256 * 1024 * 1024
                while remaining > 0:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    output.write(chunk)
                    remaining -= len(chunk)
                if source.read(1):
                    raise ValueError("event clip exceeds 256 MB limit")
            temporary.replace(destination)
        except (OSError, urllib.error.URLError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            print(f"[office-api] could not archive event clip {event['event_id']}: {error}", flush=True)


def maintenance_worker(config: dict[str, Any]) -> None:
    cleanup_counter = 0
    while True:
        try:
            archive_event_clips(config)
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as error:
            print(f"[office-api] event archive pass failed: {error}", flush=True)
        if cleanup_counter % 60 == 0:
            cleanup_expired_clips(config)
        cleanup_counter += 1
        time.sleep(60)


def occupancy_worker(config: dict[str, Any]) -> None:
    poll_seconds = max(float(config["occupancy"].get("poll_seconds", 2)), 0.5)
    while True:
        frame = None
        try:
            frame = fetch_latest_frame(config)
            update_presence(frame, config)
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError, sqlite3.Error) as error:
            print(f"[office-api] occupancy frame poll failed: {error}", flush=True)
        time.sleep(poll_seconds)


class OfficeHandler(BaseHTTPRequestHandler):
    server_version = "VSSOfficeAssistant/0.1"

    def send_json(self, status: int, payload: Any) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            # The mutation may already be committed even if a browser navigates away
            # before reading the response. Avoid turning that disconnect into a
            # second error response and let idempotent callers safely retry.
            return

    def send_static(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self, maximum: int = 65536) -> dict[str, Any]:
        length = min(int(self.headers.get("Content-Length", "0")), maximum)
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/healthz":
                self.send_json(HTTPStatus.OK, {"status": "ok"})
            elif parsed.path in (
                "/office", "/office/", "/office/settings", "/office/settings/",
                "/office/agent", "/office/agent/", "/office/flywheel", "/office/flywheel/",
            ):
                self.send_static(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif parsed.path == "/api/config":
                self.send_json(HTTPStatus.OK, self.server.office_config)  # type: ignore[attr-defined]
            elif parsed.path == "/api/status":
                self.send_json(HTTPStatus.OK, {
                    "office_api": {"healthy": True},
                    "vss_agent": service_status(f"{VSS_AGENT_URL}/health"),
                    "vst": service_status(f"{VSS_VST_URL}/vst/api/v1/sensor/streams"),
                    "elasticsearch": service_status(ELASTICSEARCH_URL),
                    "rtsp_gateway": service_status("http://127.0.0.1:9997/v3/paths/list"),
                    "motion_pipeline": self.server.workstation.motion_status(),  # type: ignore[attr-defined]
                })
            elif parsed.path == "/api/occupancy/current":
                self.send_json(
                    HTTPStatus.OK,
                    occupancy_snapshot(self.server.office_config),  # type: ignore[attr-defined]
                )
            elif parsed.path == "/api/workstation/live":
                self.send_json(HTTPStatus.OK, self.server.workstation.live())  # type: ignore[attr-defined]
            elif parsed.path == "/api/person/activity/today":
                self.send_json(HTTPStatus.OK, self.server.workstation.person_activity_today())  # type: ignore[attr-defined]
            elif parsed.path == "/api/activity/events":
                query = urllib.parse.parse_qs(parsed.query)
                timezone = ZoneInfo(self.server.office_config["timezone"])  # type: ignore[attr-defined]
                start, end = parse_activity_window(query, timezone)
                person_id = int(query["person_id"][0]) if query.get("person_id") else None
                search = str(query.get("q", [""])[0])[:200]
                self.send_json(
                    HTTPStatus.OK,
                    self.server.workstation.activity_events(  # type: ignore[attr-defined]
                        start, end, person_id=person_id, query=search,
                    ),
                )
            elif parsed.path.startswith("/api/activity/events/"):
                event_id = int(parsed.path.removeprefix("/api/activity/events/").strip("/"))
                detail = self.server.workstation.activity_event_detail(event_id)  # type: ignore[attr-defined]
                self.send_json(HTTPStatus.OK if detail else HTTPStatus.NOT_FOUND, detail or {"error": "activity event not found"})
            elif parsed.path.startswith("/api/activity/evidence/"):
                parts = parsed.path.removeprefix("/api/activity/evidence/").strip("/").split("/")
                if len(parts) != 2:
                    raise ValueError("evidence path must contain window id and person|scene")
                evidence = self.server.workstation.evidence_image(int(parts[0]), parts[1])  # type: ignore[attr-defined]
                if evidence:
                    self.send_static(evidence, "image/jpeg")
                else:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "activity evidence not found"})
            elif parsed.path == "/api/motion/status":
                self.send_json(HTTPStatus.OK, self.server.workstation.motion_status())  # type: ignore[attr-defined]
            elif parsed.path == "/api/flywheel/status":
                self.send_json(HTTPStatus.OK, self.server.flywheel.status())  # type: ignore[attr-defined]
            elif parsed.path == "/api/flywheel/candidates":
                query = urllib.parse.parse_qs(parsed.query)
                review = str(query.get("review", ["all"])[0])
                person_id = int(query["person_id"][0]) if query.get("person_id") else None
                limit = int(query.get("limit", ["100"])[0])
                self.send_json(HTTPStatus.OK, {  # type: ignore[attr-defined]
                    "candidates": self.server.flywheel.candidates(
                        review=review, person_id=person_id, limit=limit,
                    ),
                })
            elif parsed.path == "/api/flywheel/export":
                destination, counts = self.server.flywheel.export_jsonl()  # type: ignore[attr-defined]
                body = destination.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{destination.name}"')
                self.send_header("X-Dataset-Splits", json.dumps(counts, separators=(",", ":")))
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path.startswith("/api/flywheel/candidates/") and parsed.path.endswith("/clip"):
                candidate_id = int(
                    parsed.path.removeprefix("/api/flywheel/candidates/").removesuffix("/clip").strip("/")
                )
                clip = self.server.flywheel.clip(candidate_id)  # type: ignore[attr-defined]
                if clip:
                    self.send_static(clip, "video/mp4")
                else:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "flywheel clip not ready"})
            elif parsed.path.startswith("/api/flywheel/candidates/") and parsed.path.endswith("/training-clip"):
                candidate_id = int(
                    parsed.path.removeprefix("/api/flywheel/candidates/")
                    .removesuffix("/training-clip").strip("/")
                )
                clip = self.server.flywheel.training_clip(candidate_id)  # type: ignore[attr-defined]
                if clip:
                    self.send_static(clip, "video/mp4")
                else:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "training clip not ready"})
            elif parsed.path == "/api/people":
                self.send_json(HTTPStatus.OK, {"people": self.server.workstation.people_list()})  # type: ignore[attr-defined]
            elif parsed.path.startswith("/api/people/") and "/images/" in parsed.path:
                person_value, image_value = parsed.path.removeprefix("/api/people/").split("/images/", 1)
                image = self.server.workstation.person_gallery_image(  # type: ignore[attr-defined]
                    int(person_value.strip("/")), int(image_value.strip("/")),
                )
                if image:
                    self.send_static(image, "image/jpeg")
                else:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "gallery image not found"})
            elif parsed.path.startswith("/api/people/") and parsed.path.endswith("/images"):
                person_id = int(parsed.path.removeprefix("/api/people/").removesuffix("/images").strip("/"))
                self.send_json(HTTPStatus.OK, {  # type: ignore[attr-defined]
                    "person_id": person_id,
                    "images": self.server.workstation.person_gallery(person_id),
                })
            elif parsed.path.startswith("/api/people/") and parsed.path.endswith("/image"):
                person_id = int(parsed.path.removeprefix("/api/people/").removesuffix("/image").strip("/"))
                image = self.server.workstation.person_image(person_id)  # type: ignore[attr-defined]
                if image:
                    self.send_static(image, "image/jpeg")
                else:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "person image not found"})
            elif parsed.path == "/api/workstation/reports":
                query = urllib.parse.parse_qs(parsed.query)
                timezone = ZoneInfo(self.server.office_config["timezone"])  # type: ignore[attr-defined]
                today = datetime.now(timezone).date()
                start = datetime.fromisoformat(query.get("start", [(today - timedelta(days=6)).isoformat()])[0]).date()
                end = datetime.fromisoformat(query.get("end", [today.isoformat()])[0]).date()
                self.send_json(HTTPStatus.OK, {"reports": self.server.workstation.reports(start, end)})  # type: ignore[attr-defined]
            elif parsed.path.startswith("/api/workstation/reports/"):
                report_date = datetime.fromisoformat(parsed.path.rsplit("/", 1)[-1]).date()
                self.send_json(HTTPStatus.OK, self.server.workstation.report(report_date))  # type: ignore[attr-defined]
            elif parsed.path == "/api/workstation/roi":
                self.send_json(HTTPStatus.OK, self.server.workstation.roi())  # type: ignore[attr-defined]
            elif parsed.path == "/api/workstation/frame":
                self.send_bytes(HTTPStatus.OK, self.server.workstation.current_picture(), "image/jpeg")  # type: ignore[attr-defined]
            elif parsed.path.startswith("/api/workstation/events/") and parsed.path.endswith("/clip"):
                event_id = int(parsed.path.removeprefix("/api/workstation/events/").removesuffix("/clip").strip("/"))
                clip = self.server.workstation.event_clip(event_id)  # type: ignore[attr-defined]
                if clip:
                    self.send_static(clip, "video/mp4")
                else:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "workstation event clip not ready"})
            elif parsed.path == "/api/events":
                self.send_json(
                    HTTPStatus.OK,
                    search_events(urllib.parse.parse_qs(parsed.query), self.server.office_config),  # type: ignore[attr-defined]
                )
            elif parsed.path.startswith("/api/events/") and parsed.path.endswith("/clip"):
                event_id = urllib.parse.unquote(
                    parsed.path.removeprefix("/api/events/").removesuffix("/clip").rstrip("/")
                )
                local_clip = clip_path(event_id)
                if local_clip.exists():
                    self.send_static(local_clip, "video/mp4")
                    return
                event = get_event(event_id)
                clip_url = event_clip_url(event or {})
                if not clip_url:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "event clip not found"})
                else:
                    target = urllib.parse.urlparse(str(clip_url))
                    if target.scheme not in ("http", "https"):
                        self.send_json(HTTPStatus.BAD_REQUEST, {"error": "unsupported clip URL"})
                    else:
                        self.send_response(HTTPStatus.FOUND)
                        self.send_header("Location", urllib.parse.urlunparse(("", "", target.path, "", target.query, "")))
                        self.end_headers()
            elif parsed.path.startswith("/api/events/"):
                event_id = urllib.parse.unquote(parsed.path.removeprefix("/api/events/").split("/")[0])
                event = get_event(event_id)
                self.send_json(HTTPStatus.OK if event else HTTPStatus.NOT_FOUND, event or {"error": "event not found"})
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (ConfigurationError, ValueError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except (OSError, urllib.error.URLError, json.JSONDecodeError, sqlite3.Error) as error:
            self.send_json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/agent/query":
            try:
                incoming = self.read_json(maximum=65536)
                raw_messages = incoming.get("messages", [])
                if not isinstance(raw_messages, list) or not raw_messages:
                    raise ValueError("messages must be a non-empty array")
                messages = []
                for item in raw_messages[-12:]:
                    if not isinstance(item, dict):
                        continue
                    role = str(item.get("role", "user"))
                    content = str(item.get("content", "")).strip()[:4000]
                    if role in {"user", "assistant"} and content:
                        messages.append({"role": role, "content": content})
                if not messages or messages[-1]["role"] != "user":
                    raise ValueError("the last message must be a user question")
                body = json.dumps({"messages": messages, "stream": False}).encode("utf-8")
                request = urllib.request.Request(VSS_AGENT_CHAT_URL, data=body, method="POST")
                request.add_header("Content-Type", "application/json")
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = response.read(10 * 1024 * 1024)
                self.send_bytes(HTTPStatus.OK, result, "text/event-stream; charset=utf-8")
            except ValueError as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except (OSError, urllib.error.URLError) as error:
                self.send_json(HTTPStatus.BAD_GATEWAY, {"error": f"VSS Agent unavailable: {error}"})
            return
        if not parsed.path.endswith("/acknowledge") or not parsed.path.startswith("/api/events/"):
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        event_id = urllib.parse.unquote(parsed.path.removeprefix("/api/events/").removesuffix("/acknowledge").rstrip("/"))
        length = min(int(self.headers.get("Content-Length", "0")), 4096)
        payload = json.loads(self.rfile.read(length) or b"{}")
        note = str(payload.get("note", ""))[:1000]
        acknowledged_at = int(time.time())
        with closing(sqlite3.connect(DATABASE_FILE, timeout=30)) as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO acknowledgements(event_id, acknowledged_at, note) VALUES (?, ?, ?)",
                (event_id, acknowledged_at, note),
            )
        self.send_json(HTTPStatus.OK, {"event_id": event_id, "acknowledged_at": acknowledged_at, "note": note})

    def do_PUT(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path.startswith("/api/flywheel/candidates/") and parsed.path.endswith("/trim"):
                candidate_id = int(
                    parsed.path.removeprefix("/api/flywheel/candidates/").removesuffix("/trim").strip("/")
                )
                payload = self.read_json()
                candidate = self.server.flywheel.trim_candidate(  # type: ignore[attr-defined]
                    candidate_id, float(payload.get("start", -1)), float(payload.get("end", -1)),
                )
                if not candidate:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "flywheel candidate not found"})
                    return
                self.send_json(HTTPStatus.OK, candidate)
                return
            if parsed.path.startswith("/api/flywheel/candidates/") and parsed.path.endswith("/label"):
                candidate_id = int(
                    parsed.path.removeprefix("/api/flywheel/candidates/").removesuffix("/label").strip("/")
                )
                payload = self.read_json()
                candidate = self.server.flywheel.label(  # type: ignore[attr-defined]
                    candidate_id,
                    str(payload.get("label", "")),
                    subtype=str(payload.get("subtype", "")),
                    note=str(payload.get("note", "")),
                    annotator=str(payload.get("annotator", "local-reviewer")),
                )
                if not candidate:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "flywheel candidate not found"})
                    return
                self.send_json(HTTPStatus.OK, candidate)
                return
            if parsed.path == "/api/workstation/roi":
                payload = self.read_json()
                polygon = payload.get("chair_roi")
                if not isinstance(polygon, list):
                    raise ValueError("chair_roi must be a polygon array")
                self.send_json(HTTPStatus.OK, self.server.workstation.save_roi(polygon))  # type: ignore[attr-defined]
                return
            if parsed.path.startswith("/api/people/") and parsed.path.endswith("/merge"):
                source_person_id = int(
                    parsed.path.removeprefix("/api/people/").removesuffix("/merge").strip("/")
                )
                payload = self.read_json()
                target_person_id = int(payload.get("target_person_id", 0))
                result = self.server.workstation.merge_person(  # type: ignore[attr-defined]
                    source_person_id, target_person_id,
                )
                if not result:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "source or target person not found"})
                    return
                self.send_json(HTTPStatus.OK, result)
                return
            if parsed.path.startswith("/api/people/") and parsed.path.endswith("/name"):
                person_id = int(parsed.path.removeprefix("/api/people/").removesuffix("/name").strip("/"))
                payload = self.read_json()
                name = str(payload.get("name", ""))
                if not self.server.workstation.rename_person(person_id, name):  # type: ignore[attr-defined]
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "person not found or empty name"})
                    return
                self.send_json(HTTPStatus.OK, {"person_id": person_id, "name": name})
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (ConfigurationError, ValueError, OSError, json.JSONDecodeError, sqlite3.Error) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/people/"):
            person_id = int(parsed.path.removeprefix("/api/people/").strip("/"))
            if self.server.workstation.delete_person(person_id):  # type: ignore[attr-defined]
                self.send_json(HTTPStatus.OK, {"person_id": person_id, "deleted": True})
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "person not found"})
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def log_message(self, message_format: str, *args: Any) -> None:
        print(f"[office-api] {self.address_string()} {message_format % args}", flush=True)


def main() -> None:
    config = load_config()
    initialize_database()
    cleanup_expired_clips(config)
    threading.Thread(target=maintenance_worker, args=(config,), daemon=True).start()
    threading.Thread(target=occupancy_worker, args=(config,), daemon=True).start()
    workstation = WorkstationEngine(config, DATABASE_FILE, CONFIG_FILE, ELASTICSEARCH_URL, VSS_VST_URL)
    flywheel = FlywheelStore(DATABASE_FILE)
    threading.Thread(
        target=workstation_worker,
        args=(workstation, lambda: fetch_latest_frame(config), float(config["occupancy"].get("poll_seconds", 2))),
        daemon=True,
    ).start()
    port = int(os.environ.get("OFFICE_API_PORT", "8090"))
    server = ThreadingHTTPServer(("127.0.0.1", port), OfficeHandler)
    server.office_config = config  # type: ignore[attr-defined]
    server.workstation = workstation  # type: ignore[attr-defined]
    server.flywheel = flywheel  # type: ignore[attr-defined]
    print(f"[office-api] listening on http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
