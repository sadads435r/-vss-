#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/common.sh"

require_file "${OFFICE_ENV}"
require_file "${OFFICE_CONFIG}"

camera_name="$(read_env CAMERA_STREAM_NAME)"; camera_name="${camera_name:-office-main}"
rtcv_url="${RTCV_URL:-http://127.0.0.1:9010}"
vst_url="http://127.0.0.1:$(read_env VSS_VST_PORT)"
sensor_id="$(sed -n -E 's/^[[:space:]]+vss_sensor_id:[[:space:]]*["'"']?([^"'"']*)["'"']?[[:space:]]*$/\1/p' "${OFFICE_CONFIG}" | head -n1)"

if [[ -z "${sensor_id}" ]]; then
  sensor_id="$(curl --fail --silent --show-error --max-time 10 "${vst_url}/vst/api/v1/sensor/list" | \
    python3 -c 'import json,sys; name=sys.argv[1]; data=json.load(sys.stdin); print(next((str(x.get("sensorId","")) for x in data if isinstance(x,dict) and (x.get("name")==name or x.get("sensorName")==name)), ""))' "${camera_name}")"
fi

# Persist the resolved ID so snapshots, clips, and Elasticsearch filtering all use
# the same camera after an Office API restart.
sed -i -E "s|^([[:space:]]*vss_sensor_id:).*|\1 \"${sensor_id}\"|" "${OFFICE_CONFIG}"

if [[ -z "${sensor_id}" ]]; then
  echo "[ERROR] Could not resolve the VST sensor ID for ${camera_name}. Set camera.vss_sensor_id in config/office-config.yaml." >&2
  exit 1
fi

host_ip="$(ip route get 1.1.1.1 | awk '/src/ {for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}')"
camera_url="rtsp://${host_ip}:30554/live/${sensor_id}"
created_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
payload="$(printf '{"key":"sensor","value":{"camera_id":"%s","camera_name":"%s","camera_url":"%s","change":"camera_add","metadata":{"resolution":"1920x1080","codec":"h264","framerate":15}},"headers":{"source":"office-assistant","created_at":"%s"}}' "${sensor_id}" "${camera_name}" "${camera_url}" "${created_at}")"

for attempt in $(seq 1 30); do
  if response="$(curl --fail-with-body --silent --show-error --max-time 15 \
      -X POST "${rtcv_url}/api/v1/stream/add" -H 'Content-Type: application/json' \
      -H "x-stream-id: ${sensor_id}" -d "${payload}" 2>&1)"; then
    echo "[OK] RT-CV workstation stream registered: ${sensor_id}"
    exit 0
  fi
  if grep -qiE 'already|duplicate' <<<"${response}"; then
    echo "[OK] RT-CV workstation stream is already registered: ${sensor_id}"
    exit 0
  fi
  if grep -qi 'max-batch-size' <<<"${response}"; then
    if curl --fail --silent --show-error --max-time 5 \
      -H 'Content-Type: application/json' \
      -d "{\"size\":1,\"query\":{\"term\":{\"sensorId.keyword\":\"${sensor_id}\"}}}" \
      "http://127.0.0.1:$(read_env ELASTICSEARCH_PORT)/mdx-frames-*/_search" | grep -q '"_source"'; then
      echo "[OK] RT-CV already has its single active source: ${sensor_id}"
      exit 0
    fi
  fi
  [[ "${attempt}" -eq 30 ]] && { echo "[ERROR] RT-CV registration failed: ${response}" >&2; exit 1; }
  sleep 2
done
