from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoConfig, Qwen3VLForConditionalGeneration


root = Path("/model")
index = json.loads((root / "model.safetensors.index.json").read_text())
source_keys = sorted(index["weight_map"])
print("source_count", len(source_keys))
print("source_prefixes", Counter(key.split(".")[0] for key in source_keys))
for pattern in ("embed", "norm", "lm_head", "visual", "layers.0.", "layers.35."):
    print("SOURCE", pattern, [key for key in source_keys if pattern in key][:20])

config = AutoConfig.from_pretrained(root, local_files_only=True)
config.architectures = ["Qwen3VLForConditionalGeneration"]
with torch.device("meta"):
    model = Qwen3VLForConditionalGeneration(config)
target_keys = sorted(model.state_dict())
print("target_count", len(target_keys))
print("target_prefixes", Counter(key.split(".")[0] for key in target_keys))
for pattern in ("embed", "norm", "lm_head", "visual", "layers.0.", "layers.35."):
    print("TARGET", pattern, [key for key in target_keys if pattern in key][:20])
