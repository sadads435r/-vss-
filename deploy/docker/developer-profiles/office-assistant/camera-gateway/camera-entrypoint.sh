#!/usr/bin/env bash
set -euo pipefail

device="${CAMERA_DEVICE:-/dev/video0}"
width="${CAMERA_WIDTH:-1920}"
height="${CAMERA_HEIGHT:-1080}"
fps="${CAMERA_FPS:-10}"
bitrate="${CAMERA_BITRATE_KBPS:-4000}"
format="${CAMERA_INPUT_FORMAT:-auto}"
publish_url="${RTSP_PUBLISH_URL:-rtsp://127.0.0.1:8554/office-main}"

if [[ ! -c "${device}" ]]; then
  echo "[camera-gateway] Camera device not found: ${device}" >&2
  exit 1
fi

if [[ "${format}" == "auto" ]]; then
  if v4l2-ctl --device "${device}" --list-formats-ext 2>/dev/null | grep -qi "MJPG"; then
    format="mjpeg"
  else
    format="raw"
  fi
fi

if [[ "${format}" == "mjpeg" ]]; then
  source_chain=(v4l2src "device=${device}" io-mode=2 ! "image/jpeg,width=${width},height=${height},framerate=${fps}/1" ! jpegdec)
elif [[ "${format}" == "raw" ]]; then
  source_chain=(v4l2src "device=${device}" io-mode=2 ! "video/x-raw,width=${width},height=${height},framerate=${fps}/1")
else
  echo "[camera-gateway] CAMERA_INPUT_FORMAT must be auto, mjpeg, or raw" >&2
  exit 2
fi

echo "[camera-gateway] Publishing ${device} as ${publish_url} (${width}x${height}@${fps}, single-slice H.264)"
exec gst-launch-1.0 -e "${source_chain[@]}" \
  ! videoconvert \
  ! "video/x-raw,format=I420" \
  ! x264enc tune=zerolatency speed-preset=ultrafast bitrate="${bitrate}" bframes=0 key-int-max="${fps}" sliced-threads=false threads=1 \
  ! h264parse config-interval=-1 \
  ! rtspclientsink location="${publish_url}" protocols=tcp
