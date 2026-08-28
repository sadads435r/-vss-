# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated MediaPipe observation collector for mdx-office-pose."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from confluent_kafka import Consumer, KafkaError
from google.protobuf.json_format import MessageToDict

import nv_pb2
from app import CONFIG_FILE, DATABASE_FILE, ELASTICSEARCH_URL, VSS_VST_URL, load_config
from motion_worker import PoseTopicPublisher, env_flag, normalize_message
from workstation import WorkstationEngine, initialize_schema

LOGGER = logging.getLogger("office-pose-collector")


def main() -> None:
    logging.basicConfig(level=os.environ.get("OFFICE_LOG_LEVEL", "INFO"))
    config = load_config(CONFIG_FILE)
    initialize_schema(DATABASE_FILE)
    engine = WorkstationEngine(config, DATABASE_FILE, CONFIG_FILE, ELASTICSEARCH_URL, VSS_VST_URL)
    engine.enable_motion_runtime()

    bootstrap_servers = os.environ.get("OFFICE_KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")
    publisher = PoseTopicPublisher(
        bootstrap_servers,
        os.environ.get("OFFICE_POSE_TOPIC", "mdx-office-pose"),
        env_flag("OFFICE_POSE_PUBLISH_ENABLED", True),
    )
    engine.pose_observer = publisher.publish if publisher.enabled else None
    engine.pose_observe_hands = env_flag("OFFICE_POSE_HANDS_ENABLED", True)
    engine.pose_observation_interval_seconds = max(
        0.1, float(os.environ.get("OFFICE_POSE_INTERVAL_SECONDS", "0.5")),
    )

    consumer = Consumer({
        "bootstrap.servers": bootstrap_servers,
        "group.id": os.environ.get("OFFICE_POSE_CONSUMER_GROUP", "office-pose-collector-v1"),
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    })
    input_topic = os.environ.get("OFFICE_RTCV_TOPIC", "mdx-raw")
    consumer.subscribe([input_topic])
    camera_ids = {
        str(value).strip() for value in (
            config["camera"].get("id", ""), config["camera"].get("vss_sensor_id", ""),
        ) if str(value).strip()
    }
    LOGGER.info(
        "pose collector subscribed topic=%s cameras=%s output=%s interval=%ss hands=%s database=%s",
        input_topic, sorted(camera_ids), publisher.topic, engine.pose_observation_interval_seconds,
        engine.pose_observe_hands, Path(DATABASE_FILE),
    )
    try:
        while True:
            message = consumer.poll(0.5)
            if message is not None:
                if message.error():
                    if message.error().code() != KafkaError._PARTITION_EOF:
                        LOGGER.warning("Kafka error: %s", message.error())
                else:
                    try:
                        nv_frame = nv_pb2.Frame()
                        nv_frame.ParseFromString(message.value())
                        frame = normalize_message(MessageToDict(nv_frame, preserving_proto_field_name=True))
                        if not camera_ids or str(frame.get("sensorId", "")) in camera_ids:
                            engine.process_motion_frame(frame, observation_only=True)
                    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                        LOGGER.warning("Ignoring invalid RT-CV message: %s", error)
            publisher.report()
    finally:
        consumer.close()
        publisher.close()
        if engine.frame_buffer is not None:
            engine.frame_buffer.stop()


if __name__ == "__main__":
    main()
