# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from datetime import date
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from workstation import WorkstationEngine
from workstation import initialize_schema
from workstation import person_in_chair
from workstation import point_in_polygon


CONFIG = {
    "timezone": "Asia/Hong_Kong",
    "camera": {"resolution": [1000, 1000], "vss_sensor_id": "sensor-1"},
    "occupancy": {"minimum_person_confidence": 0.3},
    "schedule": {"weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday"], "end": "18:00", "holidays": []},
    "workstation": {
        "id": "desk-main", "chair_roi": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.9], [0.3, 0.9]],
        "sample_seconds": 20, "departure_seconds": 60, "state_confirmation_samples": 2,
        "frame_stale_seconds": 15, "focused_activities": ["computer", "reading", "writing"],
        "activities": ["computer", "reading", "writing", "phone", "conversation", "rest", "unknown"],
        "cosmos3_url": "http://127.0.0.1:8018", "cosmos3_model": "auto", "report_retention_days": 365,
    },
}


def frame(timestamp: str, *, inside: bool = True) -> dict:
    left, right = (400, 600) if inside else (50, 200)
    return {"id": timestamp, "sensorId": "sensor-1", "timestamp": timestamp, "objects": [{
        "id": "person-1", "type": "Person", "confidence": 0.9,
        "bbox": {"leftX": left, "rightX": right, "topY": 200, "bottomY": 800, "confidence": 0.9},
    }]}


class WorkstationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "office.db"
        self.config_file = Path(self.temp.name) / "office.yaml"
        self.config_file.write_text("workstation: {}", encoding="utf-8")
        initialize_schema(self.database)
        self.engine = WorkstationEngine(CONFIG, self.database, self.config_file, "http://es", "http://vst")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_polygon_and_chair_filter(self) -> None:
        self.assertTrue(point_in_polygon((0.5, 0.5), CONFIG["workstation"]["chair_roi"]))
        self.assertEqual(person_in_chair(frame("2026-08-10T01:00:00Z"), CONFIG), (True, "person-1"))
        self.assertEqual(person_in_chair(frame("2026-08-10T01:00:00Z", inside=False), CONFIG), (False, None))

    def test_departure_requires_sixty_seconds(self) -> None:
        observed = 1786323600.0
        self.engine.process_frame(frame("2026-08-10T01:00:00Z"), now=observed)
        self.engine.process_frame(frame("2026-08-10T01:00:20Z", inside=False), now=observed + 20)
        # Live presence drops immediately, but the short absence is not counted as a departure.
        self.assertFalse(self.engine.live(observed + 20)["occupied"])
        self.assertEqual(self.engine.live(observed + 20)["today"]["away_count"], 0)
        self.engine.process_frame(frame("2026-08-10T01:01:01Z", inside=False), now=observed + 61)
        self.assertFalse(self.engine.live(observed + 61)["occupied"])
        self.assertEqual(self.engine.live(observed + 61)["today"]["away_count"], 1)

    def test_activity_switch_requires_two_samples(self) -> None:
        observed = 1786323600.0
        self.engine.process_frame(frame("2026-08-10T01:00:00Z"), now=observed)
        self.engine._record_activity(observed + 20, "computer", 0.9, "typing")
        self.assertEqual(self.engine.live(observed + 20)["activity"], "unknown")
        self.engine._record_activity(observed + 40, "computer", 0.9, "typing")
        self.assertEqual(self.engine.live(observed + 40)["activity"], "computer")

    def test_cosmos_is_only_called_when_chair_is_currently_occupied(self) -> None:
        observed = 1786323600.0
        self.engine.process_frame(frame("2026-08-10T01:00:00Z"), now=observed)
        self.assertTrue(self.engine.needs_vlm_sample(observed + 20))
        self.engine.process_frame(frame("2026-08-10T01:00:20Z", inside=False), now=observed + 20)
        self.assertFalse(self.engine.needs_vlm_sample(observed + 40))

    def test_missing_interval_is_excluded_from_seated_time(self) -> None:
        timezone = ZoneInfo("Asia/Hong_Kong")
        observed = datetime(2026, 8, 10, 9, 0, tzinfo=timezone).timestamp()
        self.engine.process_frame(frame("2026-08-10T01:00:00Z"), now=observed)
        self.engine._record_missing(observed + 120, "rtcv_no_frame")
        report = self.engine.report(date(2026, 8, 10), now=observed + 3600)
        self.assertEqual(report["seated_seconds"], 120)

    def test_weekend_is_overtime(self) -> None:
        timezone = ZoneInfo("Asia/Hong_Kong")
        sunday = date(2026, 8, 9)
        start = datetime(2026, 8, 9, 9, 0, tzinfo=timezone).timestamp()
        self.assertEqual(self.engine._overtime_overlap(start, start + 3600, sunday, timezone), 3600)

    def test_weekday_after_end_is_overtime(self) -> None:
        timezone = ZoneInfo("Asia/Hong_Kong")
        monday = date(2026, 8, 10)
        start = datetime(2026, 8, 10, 17, 30, tzinfo=timezone).timestamp()
        end = datetime(2026, 8, 10, 19, 0, tzinfo=timezone).timestamp()
        self.assertEqual(self.engine._overtime_overlap(start, end, monday, timezone), 3600)


if __name__ == "__main__":
    unittest.main()
