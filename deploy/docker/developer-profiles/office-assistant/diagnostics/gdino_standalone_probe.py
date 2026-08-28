import json
import urllib.request

import numpy as np
from PIL import Image


URL = "http://127.0.0.1:18000/v2/models/ensemble_python_gdino/infer"
IMAGE = "/tmp/roi-current-check.jpg"


def resize_and_pad(rgb):
    image = Image.fromarray(rgb).resize((960, 540), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (960, 544))
    canvas.paste(image, (0, 2))
    return np.asarray(canvas, dtype=np.float32)


def infer(name, tensor):
    prompt = np.zeros((1, 2048), dtype=np.uint8)
    encoded = b"person ."
    prompt[0, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    tensor = np.ascontiguousarray(tensor[None], dtype=np.float32)
    header = {
        "inputs": [
            {
                "name": "inputs",
                "shape": list(tensor.shape),
                "datatype": "FP32",
                "parameters": {"binary_data_size": tensor.nbytes},
            },
            {
                "name": "PROMPT",
                "shape": list(prompt.shape),
                "datatype": "UINT8",
                "parameters": {"binary_data_size": prompt.nbytes},
            },
        ],
        "outputs": [
            {"name": "scores", "parameters": {"binary_data": True}},
            {"name": "boxes", "parameters": {"binary_data": True}},
            {"name": "labels", "parameters": {"binary_data": True}},
        ],
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    request = urllib.request.Request(
        URL,
        data=header_bytes + tensor.tobytes() + prompt.tobytes(),
        headers={
            "Content-Type": "application/octet-stream",
            "Inference-Header-Content-Length": str(len(header_bytes)),
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        response_header_len = int(response.headers["Inference-Header-Content-Length"])
        payload = response.read()
    metadata = json.loads(payload[:response_header_len])
    binary = memoryview(payload)[response_header_len:]
    offset = 0
    outputs = {}
    dtypes = {"FP32": np.float32, "INT32": np.int32}
    for output in metadata["outputs"]:
        size = output.get("parameters", {}).get("binary_data_size", 0)
        outputs[output["name"]] = np.frombuffer(
            binary[offset : offset + size], dtype=dtypes[output["datatype"]]
        ).reshape(output["shape"])
        offset += size
    scores = outputs["scores"].reshape(-1)
    boxes = outputs["boxes"].reshape(-1, 4)
    top = np.argsort(scores)[::-1][:10]
    print(
        name,
        "max=", f"{scores[top[0]]:.6f}",
        "top10=", ",".join(f"{scores[i]:.6f}" for i in top),
        "box=", ",".join(f"{value:.1f}" for value in boxes[top[0]]),
        flush=True,
    )


rgb = np.asarray(Image.open(IMAGE).convert("RGB"), dtype=np.uint8)
pixels = resize_and_pad(rgb)
infer("ds_rgb_shared_std", ((pixels - [123.675, 116.280, 103.530]) * 0.017507).transpose(2, 0, 1))
infer("ds_bgr_shared_std", ((pixels[..., ::-1] - [123.675, 116.280, 103.530]) * 0.017507).transpose(2, 0, 1))
infer(
    "standard_rgb_imagenet",
    (((pixels / 255.0) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1),
)
infer("rgb_zero_one", (pixels / 255.0).transpose(2, 0, 1))
