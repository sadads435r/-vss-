# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kafka consumer for synchronized RT-CV pose metadata."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from confluent_kafka import Consumer, KafkaError, Producer
from google.protobuf.json_format import MessageToDict

import nv_pb2
from app import CONFIG_FILE, DATABASE_FILE, ELASTICSEARCH_URL, VSS_VST_URL, load_config
from workstation import WorkstationEngine, initialize_schema

LOGGER = logging.getLogger("office-motion-worker")


def env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "true" if default else "false").strip().casefold() in {
        "1", "true", "yes", "on",
    }


class PoseTopicPublisher:
    """Best-effort JSON publisher isolated from the existing activity pipeline."""

    def __init__(self, bootstrap_servers: str, topic: str, enabled: bool) -> None:
        self.topic = topic
        self.enabled = enabled
        self.sent = 0
        self.failed = 0
        self.last_report_at = time.time()
        self.producer = Producer({
            "bootstrap.servers": bootstrap_servers,
            "client.id": "office-pose-collector",
            "compression.type": "zstd",
            "linger.ms": 50,
            "enable.idempotence": True,
        }) if enabled else None

    def _delivered(self, error: Any, _message: Any) -> None:
        if error is not None:
            self.failed += 1
            LOGGER.warning("pose delivery failed: %s", error)
        else:
            self.sent += 1

    def publish(self, observation: dict[str, Any]) -> None:
        if self.producer is None:
            return
        payload = json.dumps(observation, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        key = f"{observation.get('sensor_id', '')}:{observation.get('track_id', '')}"
        try:
            self.producer.produce(self.topic, key=key.encode("utf-8"), value=payload, on_delivery=self._delivered)
        except BufferError:
            self.producer.poll(0.25)
            self.producer.produce(self.topic, key=key.encode("utf-8"), value=payload, on_delivery=self._delivered)
        self.producer.poll(0)

    def report(self) -> None:
        if self.producer is None:
            return
        self.producer.poll(0)
        current = time.time()
        if current - self.last_report_at >= 60:
            LOGGER.info("pose collector topic=%s sent=%d failed=%d", self.topic, self.sent, self.failed)
            self.last_report_at = current

    def close(self) -> None:
        if self.producer is not None:
            remaining = self.producer.flush(10)
            if remaining:
                LOGGER.warning("pose collector closed with %d undelivered messages", remaining)


def normalize_timestamp(value: Any) -> str | float:
    """Normalize protobuf Timestamp dictionaries and preserve ISO/epoch values."""
    if isinstance(value, dict):
        if "seconds" in value:
            return float(value["seconds"]) + float(value.get("nanos", 0)) / 1_000_000_000
        required = {"year", "month", "day"}
        if required.issubset(value):
            return datetime(
                int(value["year"]), int(value["month"]), int(value["day"]),
                int(value.get("hour", 0)), int(value.get("minute", 0)), int(value.get("second", 0)),
                int(value.get("millisecond", 0)) * 1000, tzinfo=timezone.utc,
            ).isoformat()
    return value if isinstance(value, (str, int, float)) else time.time()


def normalize_message(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept both frame-level mdx-raw JSON and one-object NVS protobuf JSON."""
    if isinstance(payload.get("objects"), list):
        normalized = dict(payload)
        normalized["timestamp"] = normalize_timestamp(payload.get("timestamp") or time.time())
        return normalized
    detected = payload.get("object")
    sensor = payload.get("sensor") or {}
    return {
        "id": payload.get("messageid") or payload.get("id") or "",
        "sensorId": payload.get("sensorId") or sensor.get("id") or sensor.get("name") or "",
        "timestamp": normalize_timestamp(payload.get("timestamp") or time.time()),
        "objects": [detected] if isinstance(detected, dict) else [],
    }


def main() -> None:
    logging.basicConfig(level=os.environ.get("OFFICE_LOG_LEVEL", "INFO"))
    config = load_config(CONFIG_FILE)
    initialize_schema(DATABASE_FILE)
    engine = WorkstationEngine(config, DATABASE_FILE, CONFIG_FILE, ELASTICSEARCH_URL, VSS_VST_URL)
    engine.enable_motion_runtime()
    bootstrap_servers = os.environ.get("OFFICE_KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")
    pose_publisher = PoseTopicPublisher(
        bootstrap_servers,
        os.environ.get("OFFICE_POSE_TOPIC", "mdx-office-pose"),
        env_flag("OFFICE_POSE_PUBLISH_ENABLED", False),
    )
    if pose_publisher.enabled:
        engine.pose_observer = pose_publisher.publish
        engine.pose_observe_hands = env_flag("OFFICE_POSE_HANDS_ENABLED", True)
        engine.pose_observation_interval_seconds = max(
            0.1, float(os.environ.get("OFFICE_POSE_INTERVAL_SECONDS", "0.5")),
        )
    consumer = Consumer({
        "bootstrap.servers": bootstrap_servers,
        "group.id": os.environ.get("OFFICE_MOTION_CONSUMER_GROUP", "office-motion-v1"),
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    })
    topic = os.environ.get("OFFICE_RTCV_TOPIC", "mdx-raw")
    consumer.subscribe([topic])
    camera_ids = {
        str(value).strip() for value in (
            config["camera"].get("id", ""), config["camera"].get("vss_sensor_id", ""),
        ) if str(value).strip()
    }
    LOGGER.info(
        "motion worker subscribed topic=%s cameras=%s database=%s pose_topic=%s pose_enabled=%s",
        topic, sorted(camera_ids), Path(DATABASE_FILE), pose_publisher.topic, pose_publisher.enabled,
    )
    last_analysis = 0.0
    try:
        while True:
            message = consumer.poll(0.5)
            if message is not None:
                if message.error():
                    if message.error().code() != KafkaError._PARTITION_EOF:
                        LOGGER.warning("Kafka error: %s", message.error())
                else:
                    try:
                        # mdx-raw carries a binary nv.Frame protobuf emitted by RT-CV (not JSON).
                        nv_frame = nv_pb2.Frame()
                        nv_frame.ParseFromString(message.value())
                        decoded = MessageToDict(nv_frame, preserving_proto_field_name=True)
                        frame = normalize_message(decoded)
                        if not camera_ids or str(frame.get("sensorId", "")) in camera_ids:
                            engine.process_motion_frame(frame)
                    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                        LOGGER.warning("Ignoring invalid RT-CV message: %s", error)
            current = time.time()
            if current - last_analysis >= 0.5:
                engine.analyze_pending_motion(current)
                last_analysis = current
            pose_publisher.report()
    finally:
        consumer.close()
        pose_publisher.close()
        if engine.frame_buffer is not None:
            engine.frame_buffer.stop()


if __name__ == "__main__":
    main()
