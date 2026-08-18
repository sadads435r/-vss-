#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/common.sh"

require_file "${OFFICE_ENV}"
require_file "${OFFICE_CONFIG}"
"${SCRIPT_DIR}/preflight.sh"

export NGC_CLI_API_KEY="$(read_env NGC_CLI_API_KEY)"
export HF_TOKEN="$(read_env HF_TOKEN)"
export ELASTICSEARCH_ILM_MIN_AGE="7d"
export VST_VIDEO_STORAGE_SIZE_MB="$(read_env VST_VIDEO_STORAGE_SIZE_MB)"
export VLM_AS_VERIFIER_ALERT_TYPE_CONFIG_FILE="${REPO_ROOT}/deploy/docker/developer-profiles/office-assistant/alert_type_config.json"
export OFFICE_UID="$(id -u)"
export OFFICE_GID="$(id -g)"
export NUM_SENSORS=1

if [[ "$(read_env COSMOS3_AUTO_DOWNLOAD)" != "false" ]]; then
  bash "${SCRIPT_DIR}/download-cosmos3-nano.sh"
fi
if [[ "$(read_env MOTION_MODELS_AUTO_DOWNLOAD)" != "false" ]]; then
  bash "${SCRIPT_DIR}/download-motion-models.sh"
fi

host_ip="$(ip route get 1.1.1.1 | awk '/src/ {for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}')"
if [[ -z "${host_ip}" ]]; then
  echo "[ERROR] Could not determine the DGX Spark LAN address." >&2
  exit 1
fi

echo "[INFO] Deploying NVIDIA VSS alerts profile on ${host_ip}..."
"${REPO_ROOT}/deploy/docker/scripts/dev-profile.sh" up \
  --profile alerts \
  --hardware-profile DGX-SPARK \
  --host-ip "${host_ip}" \
  --external-ip "${host_ip}" \
  --mode verification \
  --llm nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8 \
  --vlm nvidia/cosmos3-reasoner

generated_env="${REPO_ROOT}/deploy/docker/developer-profiles/dev-profile-alerts/generated.env"
if [[ -f "${generated_env}" ]]; then
  sed -i -E 's/^ELASTICSEARCH_ILM_MIN_AGE=.*/ELASTICSEARCH_ILM_MIN_AGE=7d/' "${generated_env}"
  sed -i -E "s|^VLM_AS_VERIFIER_ALERT_TYPE_CONFIG_FILE=.*|VLM_AS_VERIFIER_ALERT_TYPE_CONFIG_FILE=${VLM_AS_VERIFIER_ALERT_TYPE_CONFIG_FILE}|" "${generated_env}"
  if grep -q '^VST_VIDEO_STORAGE_SIZE_MB=' "${generated_env}"; then
    sed -i -E "s/^VST_VIDEO_STORAGE_SIZE_MB=.*/VST_VIDEO_STORAGE_SIZE_MB=${VST_VIDEO_STORAGE_SIZE_MB}/" "${generated_env}"
  else
    printf '\nVST_VIDEO_STORAGE_SIZE_MB=%s\n' "${VST_VIDEO_STORAGE_SIZE_MB}" >> "${generated_env}"
  fi
  if grep -q '^NUM_SENSORS=' "${generated_env}"; then
    sed -i -E 's/^NUM_SENSORS=.*/NUM_SENSORS=1/' "${generated_env}"
  else
    printf '\nNUM_SENSORS=1\n' >> "${generated_env}"
  fi
fi

mkdir -p \
  "${REPO_ROOT}/data/office-assistant/clips" \
  "${REPO_ROOT}/data/office-assistant/people" \
  "${REPO_ROOT}/data/office-assistant/storyboards" \
  "${REPO_ROOT}/data/office-assistant/rolling-recordings"
echo "[INFO] Stopping the legacy VSS RT-VLM; workstation classification uses Cosmos3-Nano 16B instead."
docker stop vss-rtvi-vlm >/dev/null 2>&1 || true
echo "[INFO] Starting USB camera gateway, office API, dashboard, and HTTPS proxy..."
office_compose up --detach --build

agent_url="http://127.0.0.1:8000/api/v1/rtsp-streams/add"
camera_name="$(read_env CAMERA_STREAM_NAME)"; camera_name="${camera_name:-office-main}"
payload="$(printf '{\"sensorUrl\":\"rtsp://127.0.0.1:8554/%s\",\"name\":\"%s\",\"location\":\"office\",\"tags\":\"anonymous,office\"}' "${camera_name}" "${camera_name}")"
for attempt in $(seq 1 20); do
  if response="$(curl --fail --silent --show-error -H 'Content-Type: application/json' -d "${payload}" "${agent_url}" 2>/dev/null)"; then
    echo "[INFO] Camera registration response: ${response}"
    break
  fi
  [[ "${attempt}" -eq 20 ]] && echo "[WARN] Automatic camera registration failed; add the RTSP URL from Video Management."
  sleep 3
done

echo "[INFO] Registering the office stream with RT-CV..."
if ! bash "${SCRIPT_DIR}/register-workstation-stream.sh"; then
  echo "[WARN] RT-CV is not ready yet. Re-run ./scripts/register-workstation-stream.sh after VSS becomes healthy."
else
  office_compose restart office-api >/dev/null
fi

echo "[OK] Office assistant is available at https://${host_ip}:$(read_env OFFICE_HTTPS_PORT)/office"
echo "[INFO] Trust the Caddy local CA from deploy/docker/developer-profiles/office-assistant/caddy-data/caddy/pki/authorities/local/root.crt on office clients."
