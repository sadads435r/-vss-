import json
import sqlite3
from collections import Counter
from pathlib import Path


data_root = Path("/data")
dataset_root = Path("/dataset")
used = {
    json.loads(line)["source_candidate_id"]
    for line in (dataset_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip() and not json.loads(line).get("is_augmented", False)
}

with sqlite3.connect(f"file:{data_root / 'office.db'}?mode=ro", uri=True) as connection:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT c.id,c.started_at,c.clip_path,c.training_clip_path,l.label,l.subtype "
        "FROM flywheel_candidates c JOIN flywheel_labels l ON l.candidate_id=c.id "
        "WHERE l.label IN ('positive','negative') AND c.clip_path != '' ORDER BY c.started_at"
    ).fetchall()

unseen = [row for row in rows if int(row["id"]) not in used]
valid = []
missing = []
for row in unseen:
    relative = row["training_clip_path"] if row["label"] == "positive" else row["clip_path"]
    if relative and (data_root / relative).is_file():
        valid.append(row)
    else:
        missing.append(int(row["id"]))

print(
    json.dumps(
        {
            "labeled_total": len(rows),
            "used_originals": len(used),
            "unseen_total": len(unseen),
            "unseen_valid": len(valid),
            "unseen_by_label": dict(Counter(row["label"] for row in valid)),
            "unseen_negative_subtypes": dict(
                Counter((row["subtype"] or "未分类") for row in valid if row["label"] == "negative")
            ),
            "missing_ids": missing,
            "latest_unseen_started_at": max((row["started_at"] for row in valid), default=None),
        },
        ensure_ascii=False,
        indent=2,
    )
)
