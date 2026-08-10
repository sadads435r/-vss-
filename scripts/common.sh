#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
OFFICE_COMPOSE="${REPO_ROOT}/deploy/docker/developer-profiles/office-assistant/compose.yml"
OFFICE_ENV="${REPO_ROOT}/.env"
OFFICE_CONFIG="${REPO_ROOT}/config/office-config.yaml"

read_env() {
  local key="$1"
  local value
  value="$(sed -n -E "s/^${key}=(.*)$/\1/p" "${OFFICE_ENV}" | tail -n1)"
  value="${value%\"}"; value="${value#\"}"
  value="${value%\'}"; value="${value#\'}"
  printf '%s' "${value}"
}

require_file() {
  [[ -f "$1" ]] || { echo "[ERROR] Missing file: $1" >&2; exit 1; }
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] Missing command: $1" >&2; exit 1; }
}

office_compose() {
  docker compose --env-file "${OFFICE_ENV}" -f "${OFFICE_COMPOSE}" "$@"
}
