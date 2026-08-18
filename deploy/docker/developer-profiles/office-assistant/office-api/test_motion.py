# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

from motion import Keypoint
from motion import cosine_similarity
from motion import joint_angle
from motion import parse_pose
from motion import select_storyboard_frames
from motion import summarize_motion


def point(name: str, x: float, y: float, z: float = 0, confidence: float = 0.95) -> Keypoint:
    return Keypoint(name, x, y, z, confidence)


def sample(timestamp: float, center_x: float, knee_y: float = 0.7) -> dict:
    return {
        "timestamp": timestamp,
        "bbox": {"leftX": center_x - 50, "rightX": center_x + 50, "topY": 100, "bottomY": 800},
        "pose": {
            "left_shoulder": point("left_shoulder", center_x - 20, 250),
            "left_elbow": point("left_elbow", center_x - 20, 400),
            "left_wrist": point("left_wrist", center_x - 20, 550),
            "left_hip": point("left_hip", center_x - 15, 500),
            "left_knee": point("left_knee", center_x - 15, knee_y * 1000),
            "left_ankle": point("left_ankle", center_x - 15, 900),
            "head_top": point("head_top", center_x, 120),
            "pelvis": point("pelvis", center_x, 500),
        },
    }


class MotionTest(unittest.TestCase):
    def test_parses_flat_bodypose_25d(self) -> None:
        values = []
        for index in range(34):
            values.extend([index * 1.0, index * 2.0, index * -0.1, 0.9])
        pose = parse_pose({"pose": {"pose25d": values}})
        self.assertEqual(len(pose), 34)
        self.assertEqual(pose["left_wrist"].x, 24.0)
        self.assertAlmostEqual(pose["right_wrist"].z, -2.5)

    def test_joint_angle_uses_xyz(self) -> None:
        angle = joint_angle(point("a", 1, 0), point("b", 0, 0), point("c", 0, 1))
        self.assertAlmostEqual(angle or 0, 90.0)
        self.assertIsNone(joint_angle(point("a", 1, 0, confidence=0.1), point("b", 0, 0), point("c", 0, 1)))

    def test_motion_summary_reports_image_direction_and_relative_z(self) -> None:
        first = sample(0, 200)
        second = sample(2, 500)
        second["pose"]["left_wrist"] = point("left_wrist", 500, 550, 0.4)
        facts = summarize_motion([first, second], 1000, 1000)
        self.assertEqual(facts["body_motion"]["direction_in_image"], "right_in_image")
        self.assertAlmostEqual(facts["body_motion"]["joints"]["left_wrist"]["dz_relative"], 0.4)
        self.assertTrue(facts["quality"]["z_is_relative"])
        self.assertTrue(math.isfinite(facts["body_motion"]["bbox_speed_per_second"]))

    def test_storyboard_keeps_endpoints_and_motion_turn(self) -> None:
        frames = [(float(index), bytes([index])) for index in range(9)]
        samples = [sample(float(index), 100 if index < 4 else 700 if index == 4 else 710) for index in range(9)]
        selected = select_storyboard_frames(frames, 4, samples)
        stamps = [stamp for stamp, _data in selected]
        self.assertEqual(stamps[0], 0)
        self.assertEqual(stamps[-1], 8)
        self.assertIn(4, stamps)

    def test_cosine_similarity_rejects_shape_mismatch(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertEqual(cosine_similarity([1], [1, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
