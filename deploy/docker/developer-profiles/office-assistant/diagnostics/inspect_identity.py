import sqlite3


connection = sqlite3.connect("/data/office.db")
connection.row_factory = sqlite3.Row
for title, query in (
    ("people", "SELECT id,name,active,first_seen_at,last_seen_at FROM people ORDER BY id"),
    ("track_map", "SELECT track_id,person_id,matched_at FROM person_track_map ORDER BY matched_at DESC LIMIT 50"),
    ("verifications", "SELECT track_id,decision,matched_person_id,candidate_person_id,confidence,reid_similarity,quality_score,created_at FROM person_verifications ORDER BY id DESC LIMIT 50"),
):
    print(title)
    for row in connection.execute(query):
        print(dict(row))
