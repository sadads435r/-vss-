#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/common.sh"

require_command curl
require_command md5sum

bodypose_dir="${REPO_ROOT}/data/models/bodypose3dnet"
mediapipe_dir="${REPO_ROOT}/data/models/mediapipe"
bodypose_file="${bodypose_dir}/bodypose3dnet_accuracy.onnx"
hand_file="${mediapipe_dir}/hand_landmarker.task"

# Pinned immutable model versions. Checksums are the official object ETags.
bodypose_url="https://api.ngc.nvidia.com/v2/models/nvidia/tao/bodypose3dnet/versions/deployable_accuracy_onnx_1.0/files/bodypose3dnet_accuracy.onnx"
bodypose_md5="691aac6a25e2eaeaf86d38724b8d2052"
hand_url="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
hand_md5="15318430ea3851670fe9914116a9cfad"

download_and_verify() {
  local url="$1" destination="$2" expected="$3"
  local actual
  mkdir -p "$(dirname -- "${destination}")"
  if [[ ! -f "${destination}" || "$(md5sum "${destination}" | awk '{print $1}')" != "${expected}" ]]; then
    echo "[INFO] Downloading $(basename -- "${destination}")..."
    curl --fail --location --retry 5 --continue-at - --output "${destination}.part" "${url}"
    mv "${destination}.part" "${destination}"
  fi
  actual="$(md5sum "${destination}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "[ERROR] Checksum mismatch for ${destination}: expected ${expected}, got ${actual}" >&2
    exit 1
  }
  echo "[OK] $(basename -- "${destination}") verified (${actual})."
}

download_and_verify "${bodypose_url}" "${bodypose_file}" "${bodypose_md5}"
download_and_verify "${hand_url}" "${hand_file}" "${hand_md5}"
