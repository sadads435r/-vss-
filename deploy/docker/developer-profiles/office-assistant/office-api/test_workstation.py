# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import date
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from workstation import WorkstationEngine
from workstation import allow_fast_known_match
from workstation import clean_target_actions
from workstation import clean_target_description
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


def motion_frame(timestamp: float, center_x: float) -> dict:
    points = []
    for index, name in enumerate((
        "pelvis", "left_hip", "right_hip", "torso", "left_knee", "right_knee", "neck",
        "left_ankle", "right_ankle", "left_big_toe", "right_big_toe", "left_small_toe",
        "right_small_toe", "left_heel", "right_heel", "nose", "left_eye", "right_eye",
        "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb", "head_top", "spine",
    )):
        points.append({"name": name, "x": center_x + index, "y": 200 + index, "z": index / 100, "confidence": 0.95})
    return {
        "id": f"motion-{timestamp}", "sensorId": "sensor-1", "timestamp": timestamp,
        "objects": [{
            "id": "person-1", "type": "Person", "confidence": 0.95,
            "bbox": {"leftX": center_x - 50, "rightX": center_x + 50, "topY": 100, "bottomY": 800},
            "pose": {"keypoints": points}, "embedding": {"vector": [0.1, 0.2, 0.3]},
        }],
    }


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

    def test_identity_queue_prioritizes_chair_and_drops_stale_tracks(self) -> None:
        observed = 1786323600.0
        called = []
        self.engine._person_pending = {
            "stale": {"_identity_last_seen_at": observed - 20, "_identity_in_chair_roi": True},
            "background": {"_identity_last_seen_at": observed, "_identity_in_chair_roi": False},
            "desk": {"_identity_last_seen_at": observed, "_identity_in_chair_roi": True},
        }
        self.engine._verify_new_track = lambda track_id, item, now: called.append(track_id)
        self.engine._drain_person_pending(now=observed)
        self.assertEqual(called, ["desk"])
        self.assertNotIn("stale", self.engine._person_pending)

    def test_known_identity_candidate_gets_immediate_priority_retry(self) -> None:
        observed = 1786323600.0
        called = []
        self.engine._person_pending = {
            "desk-new": {"_identity_last_seen_at": observed, "_identity_in_chair_roi": True},
            "known": {"_identity_last_seen_at": observed, "_identity_in_chair_roi": False},
        }
        self.engine._person_match_priority["known"] = 3
        self.engine._person_last_verify_at["known"] = observed - 1
        self.engine._verify_new_track = lambda track_id, item, now: called.append(track_id)
        self.engine._drain_person_pending(now=observed)
        self.assertEqual(called, ["known"])

    def test_fast_known_match_requires_recent_high_quality_roi_evidence(self) -> None:
        identity = {
            "fast_known_match_confidence": 0.95,
            "fast_known_person_confidence": 0.95,
            "fast_known_minimum_quality": 0.65,
            "fast_known_recent_seconds": 1800,
        }
        evidence = dict(
            match_confidence=0.95,
            verdict_confidence=0.98,
            quality=0.69,
            in_chair_roi=True,
            seconds_since_target_seen=300,
            identity=identity,
        )
        self.assertTrue(allow_fast_known_match(**evidence))
        self.assertFalse(allow_fast_known_match(**{**evidence, "in_chair_roi": False}))
        self.assertFalse(allow_fast_known_match(**{**evidence, "quality": 0.60}))
        self.assertFalse(allow_fast_known_match(**{**evidence, "seconds_since_target_seen": 1900}))

    def test_target_action_filter_excludes_scene_people_and_appearance(self) -> None:
        actions = clean_target_actions([
            "右手托住下巴", "背景是开放式办公室", "另一名男子坐在旁边", "戴眼镜",
            "头部轻微移动", "右手托住下巴",
        ])
        self.assertEqual(actions, ["右手托住下巴", "头部轻微移动"])

    def test_legacy_description_is_sanitized_for_api_output(self) -> None:
        description = clean_target_description(
            "一名戴眼镜的男子坐在办公椅上，右手托着下巴，头部轻微移动。"
            "背景是开放式办公室，其他同事在后方工作。"
        )
        self.assertEqual(description, "右手托着下巴；头部轻微移动")

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

    def test_person_activity_event_merges_and_splits_after_confirmation(self) -> None:
        observed = datetime(2026, 8, 10, 9, 0, tzinfo=ZoneInfo("Asia/Hong_Kong")).timestamp()
        with closing(sqlite3.connect(self.database)) as connection, connection:
            person_id = connection.execute(
                "INSERT INTO people(name, first_seen_at, last_seen_at) VALUES ('张三', ?, ?)",
                (observed, observed),
            ).lastrowid
            connection.execute(
                "INSERT INTO person_track_map(track_id, person_id, matched_at) VALUES ('person-1', ?, ?)",
                (person_id, observed),
            )
            self.engine._record_person_activity(
                connection, "office-main", "person-1", "computer", "在电脑前工作", 0.9, False, observed,
            )
            self.engine._record_person_activity(
                connection, "office-main", "person-1", "computer", "在电脑前工作", 0.9, False, observed + 10,
            )
            self.engine._record_person_activity(
                connection, "office-main", "person-1", "computer", "在电脑前工作", 0.8, True, observed + 20,
            )
            self.engine._record_person_activity(
                connection, "office-main", "person-1", "reading", "阅读纸质材料", 0.3, False, observed + 30,
            )
            self.engine._record_person_activity(
                connection, "office-main", "person-1", "reading", "阅读纸质材料", 0.9, False, observed + 40,
            )
            self.engine._record_person_activity(
                connection, "office-main", "person-1", "reading", "阅读纸质材料", 0.9, False, observed + 50,
            )

        result = self.engine.activity_events(observed - 1, observed + 3600, now=observed + 60)
        self.assertEqual([event["activity"] for event in result["events"]], ["computer", "reading"])
        self.assertEqual(result["events"][0]["description"], "在电脑前工作")
        self.assertEqual(result["events"][0]["observation_count"], 3)
        self.assertEqual(result["events"][0]["ended_at"], observed + 40)
        self.assertTrue(result["events"][1]["ongoing"])

    def test_schema_upgrade_is_idempotent_and_keeps_legacy_rows(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute(
                "INSERT INTO person_activity_intervals("
                "camera_id, track_id, activity, confidence, started_at, ended_at) "
                "VALUES ('office-main', 'legacy', 'writing', 0.8, 10, 20)"
            )
        initialize_schema(self.database)
        initialize_schema(self.database)
        with closing(sqlite3.connect(self.database)) as connection, connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(person_activity_intervals)")}
            count = connection.execute("SELECT COUNT(*) FROM person_activity_intervals").fetchone()[0]
        self.assertTrue({"person_id", "description", "last_observed_at", "observation_count"} <= columns)
        self.assertEqual(count, 1)

    def test_legacy_cover_is_migrated_to_gallery_once(self) -> None:
        image = Path(self.temp.name) / "people" / "legacy.jpg"
        image.parent.mkdir(exist_ok=True)
        image.write_bytes(b"jpeg")
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute(
                "INSERT INTO people(name, first_seen_at, last_seen_at, reference_image) "
                "VALUES ('旧人员', 10, 20, 'people/legacy.jpg')"
            )
        initialize_schema(self.database)
        initialize_schema(self.database)
        with closing(sqlite3.connect(self.database)) as connection:
            rows = connection.execute(
                "SELECT person_id, path, is_cover FROM person_reference_images"
            ).fetchall()
        self.assertEqual(rows, [(1, "people/legacy.jpg", 1)])

    def test_activity_does_not_end_while_person_is_visible_outside_roi(self) -> None:
        observed = datetime(2026, 8, 10, 9, 0, tzinfo=ZoneInfo("Asia/Hong_Kong")).timestamp()
        with closing(self.engine._connect()) as connection, connection:
            self.engine._record_person_activity(
                connection, "office-main", "person-1", "computer", "在电脑前工作", 0.9, False, observed,
            )
            self.engine._record_person_activity(
                connection, "office-main", "person-1", "computer", "在电脑前工作", 0.9, False, observed + 1,
            )
        self.engine.process_frame(frame("2026-08-10T01:02:00Z", inside=False), now=observed + 120)
        with closing(sqlite3.connect(self.database)) as connection:
            ended_at = connection.execute(
                "SELECT ended_at FROM person_activity_intervals ORDER BY id DESC LIMIT 1"
            ).fetchone()[0]
        self.assertIsNone(ended_at)

    def test_motion_frames_create_facts_and_one_semantic_window(self) -> None:
        # This fixture injects RT-CV keypoints directly; production remains MediaPipe-only.
        self.engine.workstation.setdefault("motion_pipeline", {})["pose_source"] = "rtcv"
        started = 1_787_000_000.0
        for offset in range(9):
            self.engine.process_motion_frame(motion_frame(started + offset, 200 + offset * 20), now=started + offset)
        with closing(sqlite3.connect(self.database)) as connection:
            statuses = [row[0] for row in connection.execute(
                "SELECT status FROM person_motion_windows ORDER BY started_at"
            ).fetchall()]
            facts = connection.execute(
                "SELECT facts_json FROM person_motion_windows ORDER BY ended_at DESC LIMIT 1"
            ).fetchone()[0]
        self.assertIn("pending", statuses)
        self.assertIn("facts_only", statuses)
        self.assertIn('"direction_in_image":"right_in_image"', facts)

    def test_activity_query_filters_person_and_keyword(self) -> None:
        observed = datetime(2026, 8, 10, 9, 0, tzinfo=ZoneInfo("Asia/Hong_Kong")).timestamp()
        with closing(sqlite3.connect(self.database)) as connection, connection:
            person_id = connection.execute(
                "INSERT INTO people(name, first_seen_at, last_seen_at) VALUES ('李四', ?, ?)",
                (observed, observed),
            ).lastrowid
            connection.execute(
                "INSERT INTO person_activity_intervals("
                "camera_id, track_id, person_id, activity, description, confidence, "
                "started_at, ended_at, last_observed_at, observation_count) "
                "VALUES ('office-main', 'track-2', ?, 'writing', '在纸上书写', 0.91, ?, ?, ?, 4)",
                (person_id, observed, observed + 120, observed + 120),
            )
            connection.execute(
                "INSERT INTO person_activity_intervals("
                "camera_id, track_id, person_id, activity, description, confidence, "
                "started_at, ended_at, last_observed_at, observation_count) "
                "VALUES ('office-main', 'track-3', ?, 'writing', '在纸上书写', 0.89, ?, ?, ?, 2)",
                (person_id, observed + 130, observed + 180, observed + 180),
            )
        found = self.engine.activity_events(
            observed - 1, observed + 300, person_id=person_id, query="书写", now=observed + 300,
        )
        missing = self.engine.activity_events(
            observed - 1, observed + 300, person_id=person_id + 1, now=observed + 300,
        )
        self.assertEqual(found["event_count"], 1)
        self.assertEqual(found["events"][0]["duration_seconds"], 180)
        self.assertEqual(found["events"][0]["observation_count"], 6)
        self.assertEqual(found["people"][0]["categories"], {"writing": 180})
        self.assertEqual(missing["events"], [])

    def test_person_departure_uses_presence_timeout_without_vlm_text(self) -> None:
        observed = datetime(2026, 8, 10, 9, 0, tzinfo=ZoneInfo("Asia/Hong_Kong")).timestamp()
        with closing(sqlite3.connect(self.database)) as connection, connection:
            person_id = connection.execute(
                "INSERT INTO people(name, first_seen_at, last_seen_at) VALUES ('人员4', ?, ?)",
                (observed, observed + 120),
            ).lastrowid
            connection.execute(
                "INSERT INTO person_track_map(track_id, person_id, matched_at) VALUES ('track-4', ?, ?)",
                (person_id, observed),
            )
            seated_id = connection.execute(
                "INSERT INTO person_seated_intervals(camera_id, track_id, started_at, last_seen_at, ended_at) "
                "VALUES ('office-main', 'track-4', ?, ?, ?)",
                (observed, observed + 120, observed + 120),
            ).lastrowid

        result = self.engine.activity_events(
            observed - 1, observed + 3600, person_id=person_id, now=observed + 600,
        )
        self.assertEqual([event["activity"] for event in result["events"]], ["left_workstation"])
        departure = result["events"][0]
        self.assertEqual(departure["evidence_status"], "presence_timeout")
        self.assertEqual(departure["started_at"], observed + 120)
        self.assertTrue(departure["ongoing"])
        detail = self.engine.activity_event_detail(departure["id"])
        self.assertIsNotNone(detail)
        self.assertEqual(detail["observations"][0]["motion_facts"]["presence"]["seated_interval_id"], seated_id)

    def test_person_return_uses_new_seated_interval_without_motion_window(self) -> None:
        observed = datetime(2026, 8, 10, 9, 0, tzinfo=ZoneInfo("Asia/Hong_Kong")).timestamp()
        with closing(sqlite3.connect(self.database)) as connection, connection:
            person_id = connection.execute(
                "INSERT INTO people(name, first_seen_at, last_seen_at) VALUES ('人员4', ?, ?)",
                (observed, observed + 900),
            ).lastrowid
            for track_id, matched_at in (("track-4a", observed), ("track-4b", observed + 700)):
                connection.execute(
                    "INSERT INTO person_track_map(track_id, person_id, matched_at) VALUES (?, ?, ?)",
                    (track_id, person_id, matched_at),
                )
            connection.execute(
                "INSERT INTO person_seated_intervals(camera_id, track_id, started_at, last_seen_at, ended_at) "
                "VALUES ('office-main', 'track-4a', ?, ?, ?)",
                (observed, observed + 120, observed + 120),
            )
            connection.execute(
                "INSERT INTO person_seated_intervals(camera_id, track_id, started_at, last_seen_at, ended_at) "
                "VALUES ('office-main', 'track-4b', ?, ?, NULL)",
                (observed + 700, observed + 900),
            )

        result = self.engine.activity_events(
            observed - 1, observed + 3600, person_id=person_id, now=observed + 900,
        )
        self.assertEqual(
            [event["activity"] for event in result["events"]],
            ["left_workstation", "returned_to_workstation"],
        )
        self.assertEqual(result["events"][1]["evidence_status"], "presence_confirmed")
        self.assertEqual(result["events"][1]["started_at"], observed + 700)

    def test_merge_person_migrates_identity_history_and_gallery(self) -> None:
        observed = datetime(2026, 8, 10, 9, 0, tzinfo=ZoneInfo("Asia/Hong_Kong")).timestamp()
        with closing(sqlite3.connect(self.database)) as connection, connection:
            target_id = connection.execute(
                "INSERT INTO people(name, first_seen_at, last_seen_at, reference_image) "
                "VALUES ('shi', ?, ?, 'people/target.jpg')", (observed + 100, observed + 200),
            ).lastrowid
            source_id = connection.execute(
                "INSERT INTO people(name, first_seen_at, last_seen_at, reference_image) "
                "VALUES ('人员9', ?, ?, 'people/source.jpg')", (observed, observed + 300),
            ).lastrowid
            connection.execute(
                "INSERT INTO person_track_map(track_id, person_id, matched_at) VALUES ('duplicate-track', ?, ?)",
                (source_id, observed),
            )
            connection.execute(
                "INSERT INTO person_activity_intervals(camera_id, track_id, person_id, activity, "
                "description, confidence, started_at, ended_at, last_observed_at) "
                "VALUES ('office-main', 'duplicate-track', ?, 'computer', '使用电脑', 0.9, ?, ?, ?)",
                (source_id, observed, observed + 30, observed + 30),
            )
            connection.execute(
                "INSERT INTO person_motion_windows(camera_id, track_id, person_id, started_at, ended_at, "
                "facts_json, created_at) VALUES ('office-main', 'duplicate-track', ?, ?, ?, '{}', ?)",
                (source_id, observed, observed + 2, observed + 2),
            )
            connection.execute(
                "INSERT INTO person_reference_images(person_id, path, captured_at, quality_score, is_cover) "
                "VALUES (?, 'people/target.jpg', ?, 0.9, 1)", (target_id, observed + 100),
            )
            connection.execute(
                "INSERT INTO person_reference_images(person_id, path, captured_at, quality_score, is_cover) "
                "VALUES (?, 'people/source.jpg', ?, 0.8, 1)", (source_id, observed),
            )
            connection.execute(
                "INSERT INTO person_verifications(track_id, person_id, matched_person_id, "
                "candidate_person_id, is_person, created_at) VALUES ('duplicate-track', ?, ?, ?, 1, ?)",
                (source_id, source_id, source_id, observed),
            )
            connection.execute("CREATE TABLE flywheel_candidates(id INTEGER PRIMARY KEY, person_id INTEGER)")
            connection.execute("INSERT INTO flywheel_candidates(id, person_id) VALUES (1, ?)", (source_id,))

        result = self.engine.merge_person(source_id, target_id)
        self.assertIsNotNone(result)
        self.assertFalse(result["already_merged"])
        repeated = self.engine.merge_person(source_id, target_id)
        self.assertIsNotNone(repeated)
        self.assertTrue(repeated["already_merged"])
        self.assertEqual(repeated["migrated"], result["migrated"])
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute(
                "SELECT active FROM people WHERE id = ?", (source_id,)
            ).fetchone()[0], 0)
            target = connection.execute(
                "SELECT first_seen_at, last_seen_at, reference_image FROM people WHERE id = ?", (target_id,)
            ).fetchone()
            self.assertEqual(target, (observed, observed + 300, "people/target.jpg"))
            for table in (
                "person_track_map", "person_activity_intervals", "person_motion_windows",
                "person_reference_images", "flywheel_candidates",
            ):
                self.assertEqual(connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE person_id = ?", (source_id,)
                ).fetchone()[0], 0)
            verification = connection.execute(
                "SELECT person_id, matched_person_id, candidate_person_id FROM person_verifications"
            ).fetchone()
            self.assertEqual(verification, (target_id, target_id, target_id))
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM person_reference_images WHERE person_id = ? AND active = 1",
                (target_id,),
            ).fetchone()[0], 2)
            self.assertEqual(connection.execute(
                "SELECT target_person_id FROM person_merge_history WHERE source_person_id = ?",
                (source_id,),
            ).fetchone()[0], target_id)

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
