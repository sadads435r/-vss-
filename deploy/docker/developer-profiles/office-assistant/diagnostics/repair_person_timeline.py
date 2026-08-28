"""One-time scoped repair for strong identity matches created before one-shot matching."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


database = Path("/data/office.db")
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
backup = database.with_name(f"office.db.bak-codex-person-timeline-{stamp}")

with sqlite3.connect(database) as connection, sqlite3.connect(backup) as destination:
    connection.backup(destination)

with sqlite3.connect(database) as connection:
    candidates = connection.execute(
        "SELECT track_id, candidate_person_id, created_at FROM person_verifications "
        "WHERE decision = 'match_candidate' AND candidate_person_id IS NOT NULL "
        "AND confidence >= 0.95 AND reid_similarity >= 0.80 ORDER BY created_at"
    ).fetchall()
    interval_rows = 0
    window_rows = 0
    repaired: list[tuple[str, int, float]] = []
    for track_id, person_id, observed in candidates:
        lifecycle_start = float(observed) - 600
        lifecycle_end = float(observed) + 60
        cursor = connection.execute(
            "UPDATE person_activity_intervals SET person_id = ? "
            "WHERE track_id = ? AND person_id IS NULL "
            "AND COALESCE(last_observed_at, started_at) >= ? AND started_at <= ?",
            (int(person_id), str(track_id), lifecycle_start, lifecycle_end),
        )
        changed_intervals = cursor.rowcount
        cursor = connection.execute(
            "UPDATE person_motion_windows SET person_id = ? "
            "WHERE track_id = ? AND person_id IS NULL AND ended_at >= ? AND started_at <= ?",
            (int(person_id), str(track_id), lifecycle_start, lifecycle_end),
        )
        changed_windows = cursor.rowcount
        if changed_intervals or changed_windows:
            repaired.append((str(track_id), int(person_id), float(observed)))
            interval_rows += changed_intervals
            window_rows += changed_windows

print(f"backup={backup}")
print(f"repaired={repaired}")
print(f"activity_intervals={interval_rows} motion_windows={window_rows}")
