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

from confluent_kafka import Consumer, KafkaError

from app import CONFIG_FILE, DATABASE_FILE, ELASTICSEARCH_URL, VSS_VST_URL, load_config
from workstation import WorkstationEngine, initialize_schema

LOGGER = logging.getLogger("office-motion-worker")


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
    consumer = Consumer({
        "bootstrap.servers": os.environ.get("OFFICE_KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092"),
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
    LOGGER.info("motion worker subscribed topic=%s cameras=%s database=%s", topic, sorted(camera_ids), Path(DATABASE_FILE))
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
                        frame = normalize_message(json.loads(message.value().decode("utf-8")))
                        if not camera_ids or str(frame.get("sensorId", "")) in camera_ids:
                            engine.process_motion_frame(frame)
                    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                        LOGGER.warning("Ignoring invalid RT-CV message: %s", error)
            current = time.time()
            if current - last_analysis >= 0.5:
                engine.analyze_pending_motion(current)
                last_analysis = current
    finally:
        consumer.close()
        if engine.frame_buffer is not None:
            engine.frame_buffer.stop()


if __name__ == "__main__":
    main()
