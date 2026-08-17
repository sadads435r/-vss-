#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/common.sh"

require_file "${OFFICE_ENV}"
failures=0
probe() {
  local name="$1" url="$2"
  if curl --fail --silent --show-error --max-time 5 "${url}" >/dev/null; then echo "[PASS] ${name}"; else echo "[FAIL] ${name}" >&2; failures=$((failures + 1)); fi
}

probe "MediaMTX API" "http://127.0.0.1:9997/v3/paths/list"
probe "Office API" "http://127.0.0.1:$(read_env OFFICE_API_PORT)/healthz"
probe "Office status" "http://127.0.0.1:$(read_env OFFICE_API_PORT)/api/status"
probe "Live occupancy API" "http://127.0.0.1:$(read_env OFFICE_API_PORT)/api/occupancy/current"
probe "Workstation live API" "http://127.0.0.1:$(read_env OFFICE_API_PORT)/api/workstation/live"
probe "Workstation reports API" "http://127.0.0.1:$(read_env OFFICE_API_PORT)/api/workstation/reports"
probe "Daily activity log API" "http://127.0.0.1:$(read_env OFFICE_API_PORT)/api/activity/events"
probe "Cosmos3-Nano 16B API" "http://127.0.0.1:$(read_env COSMOS3_API_PORT)/health"
probe "VSS UI ingress" "http://127.0.0.1:$(read_env VSS_UI_PORT)/"
probe "VST sensor API" "http://127.0.0.1:$(read_env VSS_VST_PORT)/vst/api/v1/sensor/streams"

camera_name="$(read_env CAMERA_STREAM_NAME)"; camera_name="${camera_name:-office-main}"
curl --silent http://127.0.0.1:9997/v3/paths/list | grep -q "${camera_name}" \
  && echo "[PASS] RTSP camera path ${camera_name}" \
  || { echo "[FAIL] RTSP camera path ${camera_name}" >&2; failures=$((failures + 1)); }

if [[ "${failures}" -gt 0 ]]; then
  echo "[ERROR] Smoke test failed with ${failures} issue(s)." >&2
  exit 1
fi
echo "[OK] Office assistant smoke test passed."
