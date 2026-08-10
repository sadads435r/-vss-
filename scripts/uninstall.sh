#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/common.sh"

require_file "${OFFICE_ENV}"
office_compose down --remove-orphans
"${REPO_ROOT}/deploy/docker/scripts/dev-profile.sh" down
echo "[OK] Services stopped. Configuration and data under data/office-assistant were preserved."
