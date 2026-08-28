from __future__ import annotations

import argparse
import gc
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, Qwen3VLForConditionalGeneration


VISUAL_PREFIXES = (
    "blocks.", "deepstack_merger_list.", "merger.", "patch_embed.", "pos_embed.",
)
COPY_FILES = (
    "chat_template.json", "generation_config.json", "merges.txt", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json",
)


def remap_key(key: str) -> str | None:
    if key == "lm_head.weight":
        return key
    if key == "embed_tokens.weight":
        return "model.language_model.embed_tokens.weight"
    if key == "norm.weight":
        return "model.language_model.norm.weight"
    if key.startswith(VISUAL_PREFIXES):
        return "model.visual." + key
    if not key.startswith("layers.") or "_moe_gen" in key:
        return None
    if (
        ".self_attn.add_" in key
        or ".self_attn.norm_added_" in key
        or ".self_attn.to_add_out." in key
    ):
        return None
    key = "model.language_model." + key
    replacements = {
        ".self_attn.to_q.weight": ".self_attn.q_proj.weight",
        ".self_attn.to_k.weight": ".self_attn.k_proj.weight",
        ".self_attn.to_v.weight": ".self_attn.v_proj.weight",
        ".self_attn.to_out.weight": ".self_attn.o_proj.weight",
        ".self_attn.norm_q.weight": ".self_attn.q_norm.weight",
        ".self_attn.norm_k.weight": ".self_attn.k_norm.weight",
    }
    for old, new in replacements.items():
        if key.endswith(old):
            return key.removesuffix(old) + new
    return key


def target_keys(source: Path) -> set[str]:
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    config.architectures = ["Qwen3VLForConditionalGeneration"]
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config)
    return set(model.state_dict())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_index = json.loads((source / "model.safetensors.index.json").read_text())
    mapped: dict[str, tuple[str, str]] = {}
    duplicates = []
    for source_key, filename in source_index["weight_map"].items():
        destination_key = remap_key(source_key)
        if destination_key is None:
            continue
        if destination_key in mapped:
            duplicates.append(destination_key)
        mapped[destination_key] = (filename, source_key)
    expected = target_keys(source)
    missing = sorted(expected - mapped.keys())
    unexpected = sorted(mapped.keys() - expected)
    if duplicates or missing or unexpected:
        raise SystemExit(json.dumps({
            "duplicates": duplicates[:20], "missing": missing[:50], "unexpected": unexpected[:50],
        }, indent=2))

    by_file: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for destination_key, (filename, source_key) in mapped.items():
        by_file[filename].append((destination_key, source_key))
    output_map = {}
    total_size = 0
    ordered_files = sorted(by_file)
    for index, source_filename in enumerate(ordered_files, 1):
        output_filename = f"model-{index:05d}-of-{len(ordered_files):05d}.safetensors"
        tensors = {}
        with safe_open(source / source_filename, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for destination_key, source_key in by_file[source_filename]:
                if source_key not in available:
                    raise KeyError(f"{source_key} is absent from {source_filename}")
                tensor = handle.get_tensor(source_key)
                tensors[destination_key] = tensor.contiguous()
                total_size += tensor.numel() * tensor.element_size()
                output_map[destination_key] = output_filename
        save_file(tensors, output / output_filename, metadata={"format": "pt"})
        print(f"wrote {output_filename}: {len(tensors)} tensors")
        del tensors
        gc.collect()

    config = json.loads((source / "config.json").read_text())
    config["architectures"] = ["Qwen3VLForConditionalGeneration"]
    config.pop("allow_patterns_overrides", None)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for filename in COPY_FILES:
        path = source / filename
        if path.is_file():
            shutil.copy2(path, output / filename)
    (output / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_size}, "weight_map": output_map,
    }, indent=2) + "\n")
    print(json.dumps({
        "status": "complete", "tensors": len(output_map), "bytes": total_size,
        "output": str(output),
    }))


if __name__ == "__main__":
    main()
