from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from train_cosmos_drinking_lora import prepare, read_manifest


def semantic_prediction(text: str) -> bool:
    compact = re.sub(r"\s+", "", text).lower()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key, value in payload.items():
                normalized = str(key).lower()
                if isinstance(value, bool) and (
                    normalized == "confirmed"
                    or any(token in normalized for token in ("drink", "water", "sipping", "喝水", "饮水"))
                ):
                    return value
            boolean_values = [value for value in payload.values() if isinstance(value, bool)]
            if len(boolean_values) == 1:
                return boolean_values[0]
    if '"confirmed":true' in compact:
        return True
    if '"confirmed":false' in compact:
        return False
    return "true" in compact and "false" not in compact


def metrics(predictions: list[dict], key: str) -> dict:
    tp = sum(item["expected"] and item[key] for item in predictions)
    fp = sum(not item["expected"] and item[key] for item in predictions)
    fn = sum(item["expected"] and not item[key] for item in predictions)
    tn = sum(not item["expected"] and not item[key] for item in predictions)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / max(1, len(predictions)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    args = parser.parse_args()

    records = [
        record
        for record in read_manifest(args.dataset / "manifest.jsonl")
        if record["split"] == args.split and not record.get("is_augmented", False)
    ]
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        device_map={"": 0},
        local_files_only=True,
    ).eval()

    predictions = []
    for index, record in enumerate(records, start=1):
        batch = prepare(processor, args.dataset, record, labels=False, device="cuda:0")
        generated = model.generate(**batch, max_new_tokens=32, do_sample=False)
        suffix = generated[:, batch["input_ids"].shape[1] :]
        text = processor.batch_decode(suffix, skip_special_tokens=True)[0].strip()
        compact = re.sub(r"\s+", "", text).lower()
        prediction = {
            "sample_id": record["sample_id"],
            "subtype": record.get("subtype"),
            "expected": bool(record["confirmed"]),
            "strict_predicted": '"confirmed":true' in compact,
            "semantic_predicted": semantic_prediction(text),
            "output": text,
        }
        predictions.append(prediction)
        print(json.dumps({"progress": f"{index}/{len(records)}", **prediction}, ensure_ascii=False), flush=True)

    report = {
        "model": "Cosmos3-Nano-Reasoner-BF16 base (no LoRA)",
        "split": args.split,
        "samples": len(records),
        "strict": metrics(predictions, "strict_predicted"),
        "semantic": metrics(predictions, "semantic_predicted"),
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
