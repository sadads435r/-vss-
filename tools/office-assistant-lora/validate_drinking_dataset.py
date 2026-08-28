import collections
import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("root", nargs="?", type=Path, default=Path("/home/shiyiming/cosmos3-lora/data/drinking-v0.1"))
root = parser.parse_args().root
rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
bad = []
present = 0
videos = 0
for row in rows:
    videos += (root / row["video"]).is_file()
    frame_file = root / row["frames"]
    if not frame_file.is_file():
        bad.append((str(frame_file), "missing"))
        continue
    present += 1
    frames = np.load(frame_file)["frames"]
    if frames.shape != (4, 224, 224, 3) or frames.dtype != np.uint8:
        bad.append((str(frame_file), frames.shape, str(frames.dtype)))

print(
    {
        "records": len(rows),
        "videos": videos,
        "frames": present,
        "bad": bad[:5],
        "splits": dict(collections.Counter(row["split"] for row in rows)),
        "augmented_in_eval": sum(
            row.get("is_augmented", False) and row["split"] != "train" for row in rows
        ),
    }
)
