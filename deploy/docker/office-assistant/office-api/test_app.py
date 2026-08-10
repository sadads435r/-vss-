# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

from app import ConfigurationError
from app import load_config
from app import office_event_type

VALID_CONFIG = """
timezone: Asia/Hong_Kong
camera: {rtsp_url: 'rtsp://127.0.0.1:8554/office-main'}
schedule: {weekdays: [monday], start: '09:00', end: '18:00'}
retention: {event_days: 7}
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


if __name__ == "__main__":
    unittest.main()
