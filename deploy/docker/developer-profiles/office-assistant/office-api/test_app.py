# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from app import ConfigurationError
from app import load_config
from app import office_event_type

VALID_CONFIG = """
timezone: Asia/Hong_Kong
camera: {rtsp_url: 'rtsp://127.0.0.1:8554/office-main'}
occupancy: {poll_seconds: 2, departure_timeout_seconds: 10, minimum_person_confidence: 0.3, history_limit: 100}
schedule: {weekdays: [monday], start: '09:00', end: '18:00'}
retention: {event_days: 7}
workstation:
  chair_roi: [[0.3, 0.2], [0.8, 0.2], [0.8, 0.95], [0.3, 0.95]]
  sample_seconds: 20
  departure_seconds: 60
  state_confirmation_samples: 2
  activities: [computer, reading, writing, phone, conversation, rest, unknown]
  focused_activities: [computer, reading, writing]
  report_retention_days: 365
  cosmos3_url: http://127.0.0.1:8018
rules:
  after_hours_presence: {enabled: true}
  restricted_zone_entry: {enabled: true}
  dwell_time: {enabled: true, seconds: 120}
  occupancy_limit: {enabled: true, maximum_people: 10}
zones:
  - id: restricted
    polygon: [[0, 0], [1, 0], [1, 1]]
"""


class ConfigTest(unittest.TestCase):
    def write_config(self, content: str) -> Path:
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        handle.write(content)
        handle.close()
        return Path(handle.name)

    def test_valid_config(self) -> None:
        self.assertEqual(load_config(self.write_config(VALID_CONFIG))["retention"]["event_days"], 7)

    def test_rejects_non_rtsp_camera(self) -> None:
        with self.assertRaises(ConfigurationError):
            load_config(self.write_config(VALID_CONFIG.replace("rtsp://", "https://")))

    def test_rejects_out_of_range_zone(self) -> None:
        with self.assertRaises(ConfigurationError):
            load_config(self.write_config(VALID_CONFIG.replace("[1, 1]", "[2, 1]")))

    def test_after_hours_event(self) -> None:
        config = load_config(self.write_config(VALID_CONFIG))
        config["rules"] = {
            "restricted_zone_entry": {"enabled": False},
            "dwell_time": {"enabled": False},
            "occupancy_limit": {"enabled": False},
            "after_hours_presence": {"enabled": True},
        }
        self.assertEqual(office_event_type({"@timestamp": "2026-08-10T20:00:00+08:00"}, config), "after_hours_presence")

    def test_restricted_zone_has_priority(self) -> None:
        config = load_config(self.write_config(VALID_CONFIG))
        config["zones"][0]["restricted"] = True
        config["rules"] = {
            "restricted_zone_entry": {"enabled": True},
            "dwell_time": {"enabled": False},
            "occupancy_limit": {"enabled": False},
            "after_hours_presence": {"enabled": False},
        }
        self.assertEqual(office_event_type({"zone_id": "restricted"}, config), "restricted_zone_entry")

    def test_occupancy_limit(self) -> None:
        config = load_config(self.write_config(VALID_CONFIG))
        config["rules"] = {
            "restricted_zone_entry": {"enabled": False},
            "dwell_time": {"enabled": False},
            "occupancy_limit": {"enabled": True, "maximum_people": 2},
            "after_hours_presence": {"enabled": False},
        }
        self.assertEqual(office_event_type({"person_count": 3}, config), "occupancy_limit")

    def test_dwell_time(self) -> None:
        config = load_config(self.write_config(VALID_CONFIG))
        config["rules"] = {
            "restricted_zone_entry": {"enabled": False},
            "dwell_time": {"enabled": True, "seconds": 120},
            "occupancy_limit": {"enabled": False},
            "after_hours_presence": {"enabled": False},
        }
        self.assertEqual(office_event_type({"duration_seconds": 121}, config), "dwell_time")

    def test_presence_arrival_and_departure(self) -> None:
        config = load_config(self.write_config(VALID_CONFIG))
        frame = {
            "id": "frame-1",
            "sensorId": "camera-1",
            "timestamp": "2026-08-10T01:00:00Z",
            "objects": [
                {"id": "track-1", "type": "Person", "confidence": 0.9},
                {"id": "track-2", "type": "Person", "confidence": 0.8},
                {"id": "chair-1", "type": "Chair", "confidence": 0.99},
            ],
        }
        observed_at = app.parse_timestamp(frame["timestamp"])
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "office.db"
            clips = Path(directory) / "clips"
            with patch.object(app, "DATABASE_FILE", database), patch.object(app, "CLIP_DIR", clips):
                app.initialize_database()
                app.update_presence(frame, config, now=observed_at)
                snapshot = app.occupancy_snapshot(config, now=observed_at)
                self.assertEqual(snapshot["current_count"], 2)
                self.assertEqual(snapshot["today_session_count"], 2)

                # Re-reading the same Elasticsearch frame must not create duplicate sessions.
                app.update_presence(frame, config, now=observed_at + 1)
                self.assertEqual(app.occupancy_snapshot(config, now=observed_at + 1)["today_session_count"], 2)

                app.update_presence(None, config, now=observed_at + 11)
                departed = app.occupancy_snapshot(config, now=observed_at + 11)
                self.assertEqual(departed["current_count"], 0)
                self.assertTrue(all(session["status"] == "left" for session in departed["history"]))

    def test_filters_low_confidence_people(self) -> None:
        frame = {
            "objects": [
                {"id": "accepted", "type": "Person", "confidence": 0.7},
                {"id": "rejected", "type": "Person", "confidence": 0.2},
            ]
        }
        self.assertEqual(app.extract_people(frame, 0.3), {"accepted": 0.7})


if __name__ == "__main__":
    unittest.main()
