# SPDX-License-Identifier: Apache-2.0
"""Minimal binary HTTP client for the deployed Triton Grounding DINO ensemble."""

from __future__ import annotations

import json
import urllib.request
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


class GroundingDinoClient:
    def __init__(
        self,
        url: str = "http://127.0.0.1:18000/v2/models/ensemble_python_gdino/infer",
        prompt: str = "cup . mug . bottle . water bottle .",
        threshold: float = 0.40,
    ) -> None:
        self.url = url
        self.categories = [part.strip().rstrip(".") for part in prompt.split(" . ") if part.strip().rstrip(".")]
        self.prompt = " . ".join(self.categories) + " ."
        self.threshold = max(0.0, min(1.0, float(threshold)))

    @staticmethod
    def preprocess(image_bytes: bytes) -> tuple[np.ndarray, dict[str, float]]:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        source_width, source_height = image.size
        scale = min(960 / max(1, source_width), 540 / max(1, source_height))
        target_width = max(1, round(source_width * scale))
        target_height = max(1, round(source_height * scale))
        left = (960 - target_width) // 2
        top = (544 - target_height) // 2
        canvas = Image.new("RGB", (960, 544))
        canvas.paste(image.resize((target_width, target_height), Image.Resampling.BILINEAR), (left, top))
        pixels = np.asarray(canvas, dtype=np.float32)
        tensor = ((pixels - np.asarray([123.675, 116.280, 103.530], dtype=np.float32)) * 0.017507)
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None], dtype=np.float32)
        return tensor, {
            "source_width": float(source_width), "source_height": float(source_height),
            "scale": float(scale), "left": float(left), "top": float(top),
        }

    @staticmethod
    def parse_response(payload: bytes, header_length: int) -> dict[str, np.ndarray]:
        metadata = json.loads(payload[:header_length])
        binary = memoryview(payload)[header_length:]
        offset = 0
        dtypes = {"FP32": np.float32, "INT32": np.int32}
        outputs: dict[str, np.ndarray] = {}
        for output in metadata.get("outputs", []):
            datatype = str(output["datatype"])
            if datatype not in dtypes:
                raise ValueError(f"unsupported Triton output datatype: {datatype}")
            size = int(output.get("parameters", {}).get("binary_data_size", 0))
            outputs[str(output["name"])] = np.frombuffer(
                binary[offset:offset + size], dtype=dtypes[datatype],
            ).reshape(output["shape"])
            offset += size
        return outputs

    def infer(self, image_bytes: bytes, *, timeout: float = 30.0) -> dict[str, Any]:
        tensor, transform = self.preprocess(image_bytes)
        prompt = np.zeros((1, 2048), dtype=np.uint8)
        encoded = self.prompt.encode("utf-8")[:2048]
        prompt[0, :len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
        header = {
            "inputs": [
                {
                    "name": "inputs", "shape": list(tensor.shape), "datatype": "FP32",
                    "parameters": {"binary_data_size": tensor.nbytes},
                },
                {
                    "name": "PROMPT", "shape": list(prompt.shape), "datatype": "UINT8",
                    "parameters": {"binary_data_size": prompt.nbytes},
                },
            ],
            "outputs": [
                {"name": "scores", "parameters": {"binary_data": True}},
                {"name": "boxes", "parameters": {"binary_data": True}},
                {"name": "labels", "parameters": {"binary_data": True}},
            ],
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=header_bytes + tensor.tobytes() + prompt.tobytes(),
            headers={
                "Content-Type": "application/octet-stream",
                "Inference-Header-Content-Length": str(len(header_bytes)),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_header_length = int(response.headers["Inference-Header-Content-Length"])
            outputs = self.parse_response(response.read(), response_header_length)
        scores = outputs["scores"].reshape(-1)
        boxes = outputs["boxes"].reshape(-1, 4)
        labels = outputs["labels"].reshape(-1)
        best_by_label: dict[int, dict[str, Any]] = {}
        for index in np.argsort(scores)[::-1]:
            label_id = int(labels[index])
            if label_id < 0 or label_id >= len(self.categories) or label_id in best_by_label:
                continue
            box = boxes[index]
            best_by_label[label_id] = {
                "label": self.categories[label_id],
                "label_id": label_id,
                "score": round(float(scores[index]), 6),
                "box_canvas": [round(float(value), 2) for value in box],
            }
            if len(best_by_label) == len(self.categories):
                break
        detections = sorted(best_by_label.values(), key=lambda value: value["score"], reverse=True)
        best_score = max((item["score"] for item in detections), default=0.0)
        return {
            "prompt": self.prompt,
            "threshold": self.threshold,
            "confirmed": best_score >= self.threshold,
            "best_score": best_score,
            "detections": detections,
            "transform": transform,
        }
