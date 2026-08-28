import importlib.util

import torch
import transformers
import inspect

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("qwen3_vl", hasattr(transformers, "Qwen3VLForConditionalGeneration"))
for package in ("peft", "trl", "accelerate", "cosmos_framework", "vllm_cosmos3"):
    print(package, bool(importlib.util.find_spec(package)))
for package in ("qwen_vl_utils", "av", "cv2", "torchcodec"):
    print(package, bool(importlib.util.find_spec(package)))
print("processor_call", inspect.signature(transformers.Qwen3VLProcessor.__call__))
