from pathlib import Path

from PIL import Image

from motion import MediaPipePoseAnalyzer


image_path = Path("/tmp/test_person.jpg")
with Image.open(image_path) as image:
    width, height = image.size

analyzer = MediaPipePoseAnalyzer("/models/mediapipe/pose_landmarker_lite.task")
pose = analyzer.analyze(
    image_path.read_bytes(),
    {"leftX": 0, "topY": 0, "rightX": width, "bottomY": height},
)
print({
    "model_loaded": analyzer.landmarker is not None,
    "error": analyzer.error,
    "image_size": [width, height],
    "keypoint_count": len(pose),
    "keypoints": sorted(pose),
})
if analyzer.landmarker is None or not pose:
    raise SystemExit(1)
