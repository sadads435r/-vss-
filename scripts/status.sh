#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/common.sh"

require_file "${OFFICE_ENV}"
echo "== Office services =="
office_compose ps
echo "== VSS services =="
docker ps --filter 'name=vss-' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
echo "== GPU =="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
echo "== Camera stream =="
curl --silent --show-error http://127.0.0.1:9997/v3/paths/list || true
echo
echo "== Office API =="
curl --silent --show-error "http://127.0.0.1:$(read_env OFFICE_API_PORT)/api/status" || true
echo
echo "== Live occupancy =="
curl --silent --show-error "http://127.0.0.1:$(read_env OFFICE_API_PORT)/api/occupancy/current" || true
echo
echo "== Workstation analytics =="
curl --silent --show-error "http://127.0.0.1:$(read_env OFFICE_API_PORT)/api/workstation/live" || true
echo
echo "== Cosmos3-Nano 16B =="
curl --silent --show-error "http://127.0.0.1:$(read_env COSMOS3_API_PORT)/v1/models" || true
echo
