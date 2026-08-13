#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/common.sh"

require_file "${OFFICE_ENV}"
require_command python3

model_repo="$(read_env COSMOS3_MODEL_REPO)"; model_repo="${model_repo:-nvidia/Cosmos3-Nano}"
model_revision="$(read_env COSMOS3_MODEL_REVISION)"; model_revision="${model_revision:-main}"
model_dir="$(read_env COSMOS3_MODEL_DIR)"; model_dir="${model_dir:-${HOME}/models/Cosmos3-Nano}"
hf_token="$(read_env HF_TOKEN)"

available_kib="$(df -Pk "$(dirname -- "${model_dir}")" 2>/dev/null | awk 'NR==2 {print $4}')"
if [[ -n "${available_kib}" && "${available_kib}" -lt 41943040 ]]; then
  echo "[ERROR] Cosmos3-Nano needs about 32.6 GiB; keep at least 40 GiB free in $(dirname -- "${model_dir}")." >&2
  exit 1
fi

venv="${REPO_ROOT}/data/.hf-download-venv"
if [[ ! -x "${venv}/bin/hf" ]]; then
  echo "[INFO] Installing the Hugging Face download client in ${venv}..."
  python3 -m venv "${venv}"
  "${venv}/bin/pip" install --disable-pip-version-check 'huggingface_hub[cli]>=0.34,<2'
fi

mkdir -p "${model_dir}"
echo "[INFO] Downloading ${model_repo}@${model_revision} to ${model_dir} (resume is automatic)..."
HF_TOKEN="${hf_token}" "${venv}/bin/hf" download "${model_repo}" \
  --revision "${model_revision}" \
  --local-dir "${model_dir}"

[[ -f "${model_dir}/config.json" ]] || { echo "[ERROR] Missing ${model_dir}/config.json after download." >&2; exit 1; }
weight_count="$(find "${model_dir}" -type f -name '*.safetensors' | wc -l)"
[[ "${weight_count}" -gt 0 ]] || { echo "[ERROR] No safetensors weights found after download." >&2; exit 1; }
echo "[OK] Cosmos3-Nano is ready: ${weight_count} weight file(s), $(du -sh "${model_dir}" | awk '{print $1}')."
