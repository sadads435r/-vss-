import json
import sqlite3


connection = sqlite3.connect("/data/office.db")
connection.row_factory = sqlite3.Row
print("status_counts", connection.execute(
    "SELECT status, count(*) AS count FROM person_motion_windows GROUP BY status ORDER BY status"
).fetchall())
for row in connection.execute(
    "SELECT id, track_id, started_at, ended_at, facts_json, hand_json, quality, status, error "
    "FROM person_motion_windows ORDER BY id DESC LIMIT 3"
):
    facts = json.loads(row["facts_json"] or "{}")
    hands = json.loads(row["hand_json"] or "{}")
    print({
        "id": row["id"],
        "track_id": row["track_id"],
        "duration": round(float(row["ended_at"]) - float(row["started_at"]), 3),
        "sample_count": facts.get("quality", {}).get("sample_count"),
        "visible_ratio": facts.get("quality", {}).get("visible_keypoints_ratio"),
        "pose_confidence": facts.get("quality", {}).get("pose_confidence"),
        "pose_source": facts.get("quality", {}).get("pose_source"),
        "hand_available": hands.get("available"),
        "status": row["status"],
        "error": row["error"],
    })
