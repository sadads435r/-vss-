from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, Qwen3VLForConditionalGeneration


root = Path("/model")
config = AutoConfig.from_pretrained(root, local_files_only=True)
with torch.device("meta"):
    model = Qwen3VLForConditionalGeneration(config)
expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
index = json.loads((root / "model.safetensors.index.json").read_text())
actual = {}
for filename in sorted(set(index["weight_map"].values())):
    with safe_open(root / filename, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            actual[name] = tuple(handle.get_slice(name).get_shape())
missing = sorted(expected.keys() - actual.keys())
unexpected = sorted(actual.keys() - expected.keys())
shape_mismatch = sorted(
    (name, expected[name], actual[name]) for name in expected.keys() & actual.keys()
    if expected[name] != actual[name]
)
report = {
    "expected": len(expected), "actual": len(actual), "missing": missing,
    "unexpected": unexpected, "shape_mismatch": shape_mismatch,
}
print(json.dumps(report, indent=2))
if missing or unexpected or shape_mismatch:
    raise SystemExit(1)
