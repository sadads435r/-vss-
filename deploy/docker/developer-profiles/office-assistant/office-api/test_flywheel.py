from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flywheel import FlywheelStore
from flywheel_worker import DrinkingCandidateMiner, fuse_object_scores, wrist_mouth_measurement


def observation(stamp: float, wrist_x: float, wrist_y: float) -> dict:
    return {
        "type": "office.pose.observation",
        "sensor_id": "office-main",
        "track_id": "42",
        "person_id": 7,
        "timestamp": stamp,
        "bbox": {"left": 0.2, "top": 0.2, "right": 0.6, "bottom": 0.8},
        "pose": {"keypoints": {
            "left_mouth": {"x": 0.4, "y": 0.3, "confidence": 0.95},
            "right_mouth": {"x": 0.42, "y": 0.3, "confidence": 0.95},
            "right_wrist": {"x": wrist_x, "y": wrist_y, "confidence": 0.9},
        }},
    }


class FlywheelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.database = self.root / "office.db"
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "CREATE TABLE people(id INTEGER PRIMARY KEY, name TEXT NOT NULL)"
            )
            connection.execute("INSERT INTO people(id,name) VALUES(7,'shi')")
        self.store = FlywheelStore(self.database, self.root)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_wrist_mouth_measurement(self) -> None:
        near = wrist_mouth_measurement(observation(100, 0.42, 0.34))
        far = wrist_mouth_measurement(observation(101, 0.2, 0.75))
        self.assertTrue(near and near["near"])
        self.assertFalse(far and far["near"])

    def test_object_evidence_accepts_persistent_or_one_strong_frame(self) -> None:
        self.assertEqual(fuse_object_scores([0.41, 0.44, 0.2]), (True, 2, 0.44))
        self.assertEqual(fuse_object_scores([0.72, 0.2, 0.1]), (True, 1, 0.72))
        self.assertEqual(fuse_object_scores([0.39, 0.38, 0.2]), (False, 0, 0.39))

    def test_miner_creates_candidate_after_temporal_hold(self) -> None:
        miner = DrinkingCandidateMiner(self.store, self.root / "recordings")
        started = time.time()
        miner.observe(observation(started - 0.5, 0.2, 0.75))
        miner.observe(observation(started, 0.42, 0.34))
        miner.observe(observation(started + 1.0, 0.42, 0.34))
        miner.observe(observation(started + 2.0, 0.2, 0.75))
        candidates = self.store.candidates(review="unlabeled")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["person_name"], "shi")
        self.assertEqual(candidates[0]["status"], "pending_clip")

    def test_label_and_export(self) -> None:
        clip = self.store.clip_dir / "sample.mp4"
        clip.write_bytes(b"video")
        with self.store.connect() as connection, connection:
            connection.execute(
                "INSERT INTO flywheel_candidates(sample_id,sensor_id,track_id,person_id,activity,"
                "started_at,ended_at,rule_version,score,trigger_json,bbox_json,clip_path,status,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("sample", "office-main", "42", 7, "drinking", 100, 102, "v1", 0.9,
                 json.dumps({}), json.dumps({}), "flywheel/clips/sample.mp4", "unlabeled", 103, 103),
            )
        labeled = self.store.label(1, "positive", annotator="test")
        self.assertEqual(labeled["label"], "positive")
        def fake_run(command: list[str], **_: object) -> SimpleNamespace:
            if command[0] == "ffprobe":
                return SimpleNamespace(stdout=b"12.0\n")
            Path(command[-1]).write_bytes(b"trimmed-video")
            return SimpleNamespace(stdout=b"")
        with patch("flywheel.subprocess.run", side_effect=fake_run):
            trimmed = self.store.trim_candidate(1, 2.0, 6.5)
        self.assertEqual(trimmed["trim_start"], 2.0)
        self.assertEqual(trimmed["trim_end"], 6.5)
        self.assertTrue(self.store.training_clip(1).is_file())
        destination, counts = self.store.export_jsonl()
        self.assertEqual(sum(counts.values()), 1)
        record = json.loads(destination.read_text(encoding="utf-8").strip())
        self.assertTrue(record["response"]["confirmed"])
        self.assertEqual(record["metadata"]["person_id"], 7)
        self.assertTrue(record["metadata"]["is_trimmed"])
        self.assertEqual(record["metadata"]["trim_start"], 2.0)
        self.assertEqual(record["video"], "flywheel/training_clips/candidate-1.mp4")


if __name__ == "__main__":
    unittest.main()
