from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np


DATA_ROOT = Path("/data")
SEED_ROOT = Path("/seed")
OUTPUT = Path("/output")
PROMPT = "只判断视频中对应人物是否正在喝水。忽略环境和其他人，只输出JSON。"


def duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True, timeout=20,
    )
    return float(result.stdout.strip())


def encode(source: Path, destination: Path, start: float, length: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(source),
         "-t", f"{length:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "24", "-movflags", "+faststart", str(destination)],
        check=True, timeout=90,
    )


def frames(source: Path, destination: Path, count: int = 4, size: int = 224) -> None:
    clip_duration = max(0.1, duration(source))
    result = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(source), "-vf",
         f"fps={count / clip_duration:.8f},scale={size}:{size}:force_original_aspect_ratio=decrease,"
         f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2,format=rgb24",
         "-frames:v", str(count), "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
        check=True, capture_output=True, timeout=30,
    )
    frame_bytes = size * size * 3
    decoded = len(result.stdout) // frame_bytes
    if not decoded:
        raise RuntimeError(f"no frames: {source}")
    array = np.frombuffer(result.stdout[: decoded * frame_bytes], dtype=np.uint8).reshape(
        decoded, size, size, 3
    )
    if decoded < count:
        array = np.concatenate([array, np.repeat(array[-1:], count - decoded, axis=0)])
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, frames=array)


def record(candidate_id: int, *, confirmed: bool, subtype: str, video: Path, frame_file: Path) -> dict:
    return {
        "sample_id": f"candidate-{candidate_id}",
        "source_candidate_id": candidate_id,
        "split": "test",
        "video": str(video.relative_to(OUTPUT)),
        "frames": str(frame_file.relative_to(OUTPUT)),
        "prompt": PROMPT,
        "answer": json.dumps(
            {"activity": "drinking", "confirmed": confirmed}, separators=(",", ":")
        ),
        "confirmed": confirmed,
        "subtype": subtype,
        "is_augmented": False,
        "sample_weight": 1.0,
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"output must be empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_rows = [
        json.loads(line)
        for line in (SEED_ROOT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    used_ids = {
        int(row["source_candidate_id"])
        for row in seed_rows
        if not row.get("is_augmented", False)
    }
    seed_positives = [
        row
        for row in seed_rows
        if row["split"] == "test" and row["confirmed"] and not row.get("is_augmented", False)
    ]

    records = []
    for row in seed_positives:
        candidate_id = int(row["source_candidate_id"])
        video = OUTPUT / "test" / "videos" / f"candidate-{candidate_id}.mp4"
        frame_file = OUTPUT / "test" / "frames" / f"candidate-{candidate_id}.npz"
        video.parent.mkdir(parents=True, exist_ok=True)
        frame_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SEED_ROOT / row["video"], video)
        shutil.copy2(SEED_ROOT / row["frames"], frame_file)
        records.append(record(candidate_id, confirmed=True, subtype=row.get("subtype", ""), video=video, frame_file=frame_file))

    with sqlite3.connect(f"file:{DATA_ROOT / 'office.db'}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT c.id,c.started_at,c.ended_at,c.clip_path,l.label,l.subtype "
            "FROM flywheel_candidates c JOIN flywheel_labels l ON l.candidate_id=c.id "
            "WHERE l.label='negative' AND c.clip_path != '' ORDER BY c.started_at,c.id"
        ).fetchall()

    missing = []
    for row in rows:
        candidate_id = int(row["id"])
        if candidate_id in used_ids:
            continue
        source = DATA_ROOT / row["clip_path"]
        if not source.is_file():
            missing.append(candidate_id)
            continue
        video = OUTPUT / "test" / "videos" / f"candidate-{candidate_id}.mp4"
        frame_file = OUTPUT / "test" / "frames" / f"candidate-{candidate_id}.npz"
        total = duration(source)
        length = min(6.0, total)
        event_duration = max(0.0, float(row["ended_at"]) - float(row["started_at"]))
        center = 3.0 + event_duration / 2.0
        start = max(0.0, min(max(0.0, total - length), center - length / 2.0))
        encode(source, video, start, length)
        frames(video, frame_file)
        records.append(
            record(
                candidate_id,
                confirmed=False,
                subtype=row["subtype"] or "未分类",
                video=video,
                frame_file=frame_file,
            )
        )
        print(json.dumps({"built": len(records), "candidate_id": candidate_id}), flush=True)

    (OUTPUT / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )
    summary = {
        "records": len(records),
        "positive": sum(row["confirmed"] for row in records),
        "negative": sum(not row["confirmed"] for row in records),
        "subtypes": dict(Counter(row["subtype"] for row in records if not row["confirmed"])),
        "missing_ids": missing,
        "seed_used_ids": len(used_ids),
        "leaked_negative_ids": sorted(
            int(row["source_candidate_id"])
            for row in records
            if not row["confirmed"] and int(row["source_candidate_id"]) in used_ids
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
