from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from evaluate_cosmos_base import metrics, semantic_prediction
from train_cosmos_drinking_lora import prepare, read_manifest


@torch.inference_mode()
def evaluate(name, model, processor, dataset, records):
    predictions = []
    for index, row in enumerate(records, start=1):
        batch = prepare(processor, dataset, row, labels=False, device="cuda:0")
        generated = model.generate(**batch, max_new_tokens=32, do_sample=False)
        suffix = generated[:, batch["input_ids"].shape[1] :]
        text = processor.batch_decode(suffix, skip_special_tokens=True)[0].strip()
        compact = re.sub(r"\s+", "", text).lower()
        item = {
            "sample_id": row["sample_id"],
            "subtype": row.get("subtype", ""),
            "expected": bool(row["confirmed"]),
            "strict_predicted": '"confirmed":true' in compact,
            "semantic_predicted": semantic_prediction(text),
            "schema_compliant": bool(re.search(r'"confirmed":(?:true|false)', compact)),
            "output": text,
        }
        predictions.append(item)
        print(
            json.dumps(
                {
                    "model": name,
                    "progress": f"{index}/{len(records)}",
                    "sample_id": item["sample_id"],
                    "expected": item["expected"],
                    "predicted": item["semantic_predicted"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    false_positives = [item for item in predictions if not item["expected"] and item["semantic_predicted"]]
    false_negatives = [item for item in predictions if item["expected"] and not item["semantic_predicted"]]
    return {
        "semantic": metrics(predictions, "semantic_predicted"),
        "strict": metrics(predictions, "strict_predicted"),
        "schema_compliance": sum(item["schema_compliant"] for item in predictions) / len(predictions),
        "false_positive_subtypes": dict(Counter(item["subtype"] for item in false_positives)),
        "false_positive_ids": [item["sample_id"] for item in false_positives],
        "false_negative_ids": [item["sample_id"] for item in false_negatives],
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [
        row
        for row in read_manifest(args.dataset / "manifest.jsonl")
        if row["split"] == "test" and not row.get("is_augmented", False)
    ]
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        device_map={"": 0},
        local_files_only=True,
    ).eval()
    base_result = evaluate("base", base, processor, args.dataset, records)

    lora = PeftModel.from_pretrained(base, args.adapter, is_trainable=False, local_files_only=True).eval()
    lora_result = evaluate("lora", lora, processor, args.dataset, records)

    base_by_id = {row["sample_id"]: row for row in base_result["predictions"]}
    lora_by_id = {row["sample_id"]: row for row in lora_result["predictions"]}
    fixed = []
    regressed = []
    changed = []
    for row in records:
        sample_id = row["sample_id"]
        expected = bool(row["confirmed"])
        before = base_by_id[sample_id]["semantic_predicted"]
        after = lora_by_id[sample_id]["semantic_predicted"]
        if before != after:
            changed.append(sample_id)
        if before != expected and after == expected:
            fixed.append(sample_id)
        if before == expected and after != expected:
            regressed.append(sample_id)

    report = {
        "samples": len(records),
        "positive": sum(row["confirmed"] for row in records),
        "negative": sum(not row["confirmed"] for row in records),
        "base": base_result,
        "lora": lora_result,
        "pairwise": {"changed": changed, "fixed": fixed, "regressed": regressed},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "samples": report["samples"],
        "positive": report["positive"],
        "negative": report["negative"],
        "base": {key: value for key, value in base_result.items() if key != "predictions"},
        "lora": {key: value for key, value in lora_result.items() if key != "predictions"},
        "pairwise": report["pairwise"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
