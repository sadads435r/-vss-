from __future__ import annotations

import json
import time
from collections import defaultdict

from confluent_kafka import Consumer, TopicPartition

from flywheel_worker import wrist_mouth_measurement


topic = "mdx-office-pose"
consumer = Consumer({
    "bootstrap.servers": "127.0.0.1:9092",
    "group.id": f"office-pose-diagnostic-{time.time_ns()}",
    "enable.auto.commit": False,
    "auto.offset.reset": "latest",
})
metadata = consumer.list_topics(topic, timeout=10)
partitions = []
targets = {}
watermarks = {}
for partition_id in metadata.topics[topic].partitions:
    partition = TopicPartition(topic, partition_id)
    low, high = consumer.get_watermark_offsets(partition, timeout=10)
    watermarks[partition_id] = {"low": low, "high": high}
    partitions.append(TopicPartition(topic, partition_id, max(low, high - 1600)))
    targets[partition_id] = high
consumer.assign(partitions)

cutoff = time.time() - 600
rows = defaultdict(list)
message_count = 0
type_counts = defaultdict(int)
latest_stamp = 0.0
latest_type = ""
idle = 0
reached = set()
while idle < 5 and message_count < 5000 and len(reached) < len(targets):
    message = consumer.poll(0.5)
    if message is None:
        idle += 1
        continue
    idle = 0
    if message.error():
        continue
    if message.offset() >= targets.get(message.partition(), 0) - 1:
        reached.add(message.partition())
    try:
        observation = json.loads(message.value())
    except (TypeError, ValueError):
        continue
    message_count += 1
    latest_type = str(observation.get("type") or "")
    type_counts[latest_type] += 1
    stamp = float(observation.get("timestamp") or 0)
    latest_stamp = max(latest_stamp, stamp)
    if observation.get("type") != "office.pose.observation" or stamp < cutoff:
        continue
    measurement = wrist_mouth_measurement(observation)
    if not measurement:
        continue
    key = (str(observation.get("track_id")), observation.get("person_id"))
    rows[key].append({"t": stamp, **measurement})

consumer.close()
now = time.time()
print(json.dumps({
    "now": now,
    "watermarks": watermarks,
    "messages_read": message_count,
    "types": type_counts,
    "latest_type": latest_type,
    "latest_timestamp": latest_stamp,
    "latest_age": round(now - latest_stamp, 1) if latest_stamp else None,
}, ensure_ascii=False))
for (track_id, person_id), values in sorted(
    rows.items(), key=lambda item: item[1][-1]["t"], reverse=True
):
    recent = [value for value in values if now - value["t"] <= 300]
    if not recent:
        continue
    near = [value for value in recent if value["near"]]
    print(json.dumps({
        "track_id": track_id,
        "person_id": person_id,
        "last_age": round(now - recent[-1]["t"], 1),
        "observations": len(recent),
        "near_frames": len(near),
        "minimum_distance": min(value["distance_bbox_heights"] for value in recent),
        "last_near": [
            {"age": round(now - value["t"], 1), "d": value["distance_bbox_heights"],
             "w": value["wrist_confidence"]}
            for value in near[-12:]
        ],
    }, ensure_ascii=False))
    if track_id == "9488":
        print("TRACK_9488 " + json.dumps([
            {"age": round(now - value["t"], 1), "d": value["distance_bbox_heights"],
             "near": value["near"], "w": value["wrist_confidence"]}
            for value in recent if now - value["t"] <= 180
        ], ensure_ascii=False))
