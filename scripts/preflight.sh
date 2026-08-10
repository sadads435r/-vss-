#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/common.sh"

errors=0
check() { "$@" || { echo "[FAIL] $*" >&2; errors=$((errors + 1)); }; }

for command_name in git git-lfs docker curl nvidia-smi v4l2-ctl python3; do
  check require_command "${command_name}"
done
require_file "${OFFICE_ENV}"
require_file "${OFFICE_CONFIG}"

arch="$(uname -m)"
[[ "${arch}" == "aarch64" ]] || { echo "[FAIL] DGX Spark requires aarch64; detected ${arch}" >&2; errors=$((errors + 1)); }
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi 'GB10' \
  || { echo "[FAIL] NVIDIA GB10 GPU was not detected" >&2; errors=$((errors + 1)); }
docker info >/dev/null 2>&1 || { echo "[FAIL] Docker daemon is unavailable" >&2; errors=$((errors + 1)); }
docker compose version >/dev/null 2>&1 || { echo "[FAIL] Docker Compose v2 is unavailable" >&2; errors=$((errors + 1)); }

camera_device="$(read_env CAMERA_DEVICE)"; camera_device="${camera_device:-/dev/video0}"
[[ -c "${camera_device}" ]] || { echo "[FAIL] Camera is not a character device: ${camera_device}" >&2; errors=$((errors + 1)); }
[[ -n "$(read_env NGC_CLI_API_KEY)" ]] || { echo "[FAIL] NGC_CLI_API_KEY is empty in .env" >&2; errors=$((errors + 1)); }
[[ -n "$(read_env OFFICE_PASSWORD_HASH)" ]] || { echo "[FAIL] OFFICE_PASSWORD_HASH is empty in .env" >&2; errors=$((errors + 1)); }

available_gb="$(df -Pk "${REPO_ROOT}" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
[[ "${available_gb}" -ge 30 ]] || { echo "[FAIL] At least 30 GB free disk is required; found ${available_gb} GB" >&2; errors=$((errors + 1)); }

docker compose --env-file "${OFFICE_ENV}" -f "${OFFICE_COMPOSE}" config --quiet \
  || { echo "[FAIL] Office Compose configuration is invalid" >&2; errors=$((errors + 1)); }

if [[ "${errors}" -gt 0 ]]; then
  echo "[ERROR] Preflight failed with ${errors} issue(s)." >&2
  exit 1
fi
echo "[OK] DGX Spark office assistant preflight passed."
