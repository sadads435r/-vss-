from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np


DATA_ROOT = Path("/data")
OUTPUT = Path("/output")
PROMPT = "只判断视频中对应人物是否正在喝水。忽略环境和其他人，只输出JSON。"
HARD_NEGATIVE_TERMS = (
    "摸", "挠", "手机", "捂", "耳机", "交谈", "办公", "电脑", "工作",
)


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True, timeout=20,
    )
    return float(result.stdout.strip())


def encode_clip(source: Path, destination: Path, start: float, duration: float, vf: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(source)]
    command += ["-t", f"{duration:.3f}"]
    if vf:
        command += ["-vf", vf]
    command += ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-movflags", "+faststart", str(destination)]
    subprocess.run(command, check=True, timeout=90)


def extract_frames(source: Path, destination: Path, count: int = 4, size: int = 224) -> None:
    duration = max(0.1, probe_duration(source))
    result = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(source), "-vf",
         f"fps={count / duration:.8f},scale={size}:{size}:force_original_aspect_ratio=decrease,"
         f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2,format=rgb24",
         "-frames:v", str(count), "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
        check=True, capture_output=True, timeout=30,
    )
    frame_bytes = size * size * 3
    decoded = len(result.stdout) // frame_bytes
    if not decoded:
        raise RuntimeError(f"no decoded frames for {source}")
    frames = np.frombuffer(result.stdout[:decoded * frame_bytes], dtype=np.uint8).reshape(decoded, size, size, 3)
    if decoded < count:
        frames = np.concatenate([frames, np.repeat(frames[-1:], count - decoded, axis=0)])
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, frames=frames)


def split_rows(rows: list[sqlite3.Row]) -> dict[int, str]:
    ordered = sorted(rows, key=lambda row: (float(row["started_at"]), int(row["id"])))
    result = {}
    total = len(ordered)
    train_end = max(1, round(total * 0.8))
    validation_end = max(train_end + 1, round(total * 0.9)) if total >= 10 else train_end
    for index, row in enumerate(ordered):
        result[int(row["id"])] = "train" if index < train_end else "validation" if index < validation_end else "test"
    return result


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"output must be empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    database_uri = f"file:{DATA_ROOT / 'office.db'}?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT c.*,l.label,l.subtype FROM flywheel_candidates c "
            "JOIN flywheel_labels l ON l.candidate_id=c.id "
            "WHERE l.label IN ('positive','negative') AND c.clip_path != '' ORDER BY c.started_at"
        ).fetchall()
    positives = [row for row in rows if row["label"] == "positive" and row["training_clip_path"]]
    categorized = [
        row for row in rows if row["label"] == "negative"
        and any(term in (row["subtype"] or "") for term in HARD_NEGATIVE_TERMS)
    ]
    uncategorized = [row for row in rows if row["label"] == "negative" and row not in categorized]
    negatives = categorized[:90]
    if len(negatives) < 90:
        negatives.extend(uncategorized[:90 - len(negatives)])
    positive_splits = split_rows(positives)
    negative_splits = split_rows(negatives)
    records = []

    for row in positives + negatives:
        candidate_id = int(row["id"])
        positive = row["label"] == "positive"
        split = positive_splits[candidate_id] if positive else negative_splits[candidate_id]
        source_rel = row["training_clip_path"] if positive else row["clip_path"]
        source = DATA_ROOT / source_rel
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = OUTPUT / split / "videos" / f"candidate-{candidate_id}.mp4"
        if positive:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        else:
            total = probe_duration(source)
            length = min(6.0, total)
            event_duration = max(0.0, float(row["ended_at"]) - float(row["started_at"]))
            event_center = 3.0 + event_duration / 2.0
            start = max(0.0, min(max(0.0, total - length), event_center - length / 2.0))
            encode_clip(source, destination, start, length)
        frames_path = OUTPUT / split / "frames" / f"candidate-{candidate_id}.npz"
        extract_frames(destination, frames_path)
        records.append({
            "sample_id": f"candidate-{candidate_id}", "source_candidate_id": candidate_id,
            "source_sample_id": row["sample_id"], "split": split,
            "video": str(destination.relative_to(OUTPUT)),
            "frames": str(frames_path.relative_to(OUTPUT)), "prompt": PROMPT,
            "answer": json.dumps({"activity": "drinking", "confirmed": positive}, separators=(",", ":")),
            "confirmed": positive, "subtype": row["subtype"] or "",
            "is_augmented": False, "sample_weight": 1.0,
        })

    train_originals = [record for record in records if record["split"] == "train"]
    augment_sources = (
        [record for record in train_originals if record["confirmed"]][:16]
        + [record for record in train_originals if not record["confirmed"]][:16]
    )
    for index, record in enumerate(augment_sources):
        source = OUTPUT / record["video"]
        destination = OUTPUT / "train" / "augmented" / f"{record['sample_id']}-aug.mp4"
        brightness = 0.035 if index % 2 == 0 else -0.025
        contrast = 1.06 if index % 3 else 0.95
        encode_clip(
            source, destination, 0.0, probe_duration(source),
            f"setpts=PTS/1.05,eq=brightness={brightness}:contrast={contrast},noise=alls=2:allf=t",
        )
        augmented = dict(record)
        frames_path = OUTPUT / "train" / "frames" / f"{record['sample_id']}-aug.npz"
        extract_frames(destination, frames_path)
        augmented.update({
            "sample_id": record["sample_id"] + "-aug", "video": str(destination.relative_to(OUTPUT)),
            "frames": str(frames_path.relative_to(OUTPUT)),
            "is_augmented": True, "sample_weight": 0.4,
            "augmentation": {"speed": 1.05, "brightness": brightness, "contrast": contrast, "noise": 2},
        })
        records.append(augmented)

    with (OUTPUT / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "records": len(records), "originals": len(positives) + len(negatives),
        "augmented": sum(record["is_augmented"] for record in records),
        "by_split_label": Counter(
            f"{record['split']}:{'positive' if record['confirmed'] else 'negative'}"
            for record in records if not record["is_augmented"]
        ),
        "augmented_fraction_train": round(
            sum(record["is_augmented"] for record in records if record["split"] == "train")
            / sum(record["split"] == "train" for record in records), 4
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
