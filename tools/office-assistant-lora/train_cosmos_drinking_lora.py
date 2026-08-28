from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# cuDNN 9 on GB10 cannot select a BF16 engine for Qwen3-VL's temporal
# patch-embedding Conv3D. The native CUDA convolution path supports it.
torch.backends.cudnn.enabled = False


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def decode_frames(path: Path, count: int = 4, max_side: int = 224) -> np.ndarray:
    ffmpeg = "/opt/ffmpeg-safe/bin/ffmpeg"
    ffprobe = "/opt/ffmpeg-safe/bin/ffprobe"
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True, timeout=20,
    )
    duration = max(0.1, float(probe.stdout.strip()))
    fps = count / duration
    result = subprocess.run(
        [ffmpeg, "-loglevel", "error", "-i", str(path), "-vf",
         f"fps={fps:.8f},scale={max_side}:{max_side}:force_original_aspect_ratio=decrease,"
         f"pad={max_side}:{max_side}:(ow-iw)/2:(oh-ih)/2,format=rgb24",
         "-frames:v", str(count), "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
        check=False, capture_output=True, timeout=30,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-2000:])
    frame_bytes = max_side * max_side * 3
    decoded = len(result.stdout) // frame_bytes
    if decoded == 0:
        raise RuntimeError(f"ffmpeg returned no frames for {path}")
    frames = np.frombuffer(result.stdout[:decoded * frame_bytes], dtype=np.uint8).reshape(
        decoded, max_side, max_side, 3,
    )
    if decoded < count:
        frames = np.concatenate([frames, np.repeat(frames[-1:], count - decoded, axis=0)])
    return frames


def messages(record: dict, include_answer: bool) -> list[dict]:
    result = [{
        "role": "user",
        "content": [{"type": "video"}, {"type": "text", "text": record["prompt"]}],
    }]
    if include_answer:
        result.append({"role": "assistant", "content": record["answer"]})
    return result


def prepare(
    processor: AutoProcessor, dataset_root: Path, record: dict, *, labels: bool, device: str,
) -> dict[str, torch.Tensor]:
    frames_path = dataset_root / record.get("frames", "")
    frames = np.load(frames_path)["frames"] if frames_path.is_file() else decode_frames(dataset_root / record["video"])
    full_text = processor.apply_chat_template(
        messages(record, labels), tokenize=False, add_generation_prompt=not labels,
    )
    batch = processor(text=[full_text], videos=[frames], return_tensors="pt", padding=True)
    if labels:
        prompt_text = processor.apply_chat_template(
            messages(record, False), tokenize=False, add_generation_prompt=True,
        )
        prompt_batch = processor(text=[prompt_text], videos=[frames], return_tensors="pt", padding=True)
        target = batch["input_ids"].clone()
        prompt_length = prompt_batch["input_ids"].shape[1]
        target[:, :prompt_length] = -100
        if processor.tokenizer.pad_token_id is not None:
            target[target == processor.tokenizer.pad_token_id] = -100
        if not torch.any(target != -100):
            raise RuntimeError(f"no assistant tokens remain for {record['sample_id']}")
        batch["labels"] = target
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
    )


def dry_run(model_path: Path, dataset_root: Path, records: list[dict]) -> None:
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    samples = []
    for split in ("train", "validation", "test"):
        record = next(item for item in records if item["split"] == split and not item["is_augmented"])
        batch = prepare(processor, dataset_root, record, labels=True, device="cpu")
        samples.append({
            "split": split, "sample_id": record["sample_id"],
            "input_tokens": int(batch["input_ids"].numel()),
            "target_tokens": int(torch.sum(batch["labels"] != -100)),
            "pixel_shape": list(batch["pixel_values_videos"].shape),
        })
    config = model_path / "config.json"
    model_config = json.loads(config.read_text())
    model_config["use_cache"] = False
    from transformers import AutoConfig
    config_object = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config_object.use_cache = False
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config_object)
        model = get_peft_model(model, lora_config())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({
        "status": "dry_run_ok", "samples": samples, "records": len(records),
        "trainable_parameters": trainable, "total_parameters": total,
        "trainable_percent": round(trainable / total * 100, 4),
    }, ensure_ascii=False, indent=2))


@torch.no_grad()
def evaluate(model, processor, dataset_root: Path, records: list[dict], device: str) -> dict:
    model.eval()
    predictions = []
    for record in records:
        batch = prepare(processor, dataset_root, record, labels=False, device=device)
        generated = model.generate(**batch, max_new_tokens=32, do_sample=False)
        suffix = generated[:, batch["input_ids"].shape[1]:]
        text = processor.batch_decode(suffix, skip_special_tokens=True)[0].strip()
        compact = text.replace(" ", "").lower()
        predicted = '"confirmed":true' in compact
        predictions.append({
            "sample_id": record["sample_id"], "expected": bool(record["confirmed"]),
            "predicted": predicted, "output": text,
        })
    tp = sum(item["expected"] and item["predicted"] for item in predictions)
    fp = sum(not item["expected"] and item["predicted"] for item in predictions)
    fn = sum(item["expected"] and not item["predicted"] for item in predictions)
    tn = sum(not item["expected"] and not item["predicted"] for item in predictions)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / max(1, len(predictions)),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "predictions": predictions,
    }


def train(args, records: list[dict]) -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA with BF16 support is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda:0"
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa", device_map={"": 0}, local_files_only=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config())
    model.print_trainable_parameters()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "validation" and not record["is_augmented"]]
    test_records = [record for record in records if record["split"] == "test" and not record["is_augmented"]]
    before = {} if args.probe_only else evaluate(model, processor, args.dataset, validation_records, device)
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    micro_steps = 0
    losses = []
    started = time.time()
    epoch = 0
    while optimizer_steps < args.max_steps:
        epoch += 1
        random.Random(args.seed + epoch).shuffle(train_records)
        model.train()
        for record in train_records:
            batch = prepare(processor, args.dataset, record, labels=True, device=device)
            output = model(**batch)
            weighted_loss = output.loss * float(record.get("sample_weight", 1.0))
            (weighted_loss / args.gradient_accumulation).backward()
            losses.append(float(output.loss.detach().cpu()))
            micro_steps += 1
            if micro_steps % args.gradient_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            print(json.dumps({
                "step": optimizer_steps, "loss": round(sum(losses[-args.gradient_accumulation:]) / args.gradient_accumulation, 5),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            }), flush=True)
            if optimizer_steps >= args.max_steps:
                break
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    after_validation = {} if args.probe_only else evaluate(model, processor, args.dataset, validation_records, device)
    after_test = {} if args.probe_only else evaluate(model, processor, args.dataset, test_records, device)
    report = {
        "status": "probe_complete" if args.probe_only else "training_complete",
        "max_steps": args.max_steps, "micro_steps": micro_steps,
        "elapsed_seconds": round(time.time() - started, 1),
        "mean_loss": sum(losses) / max(1, len(losses)),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "validation_before": before, "validation_after": after_validation, "test_after": after_test,
    }
    (args.output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    records = read_manifest(args.dataset / "manifest.jsonl")
    if args.dry_run:
        dry_run(args.model, args.dataset, records)
    else:
        train(args, records)


if __name__ == "__main__":
    main()
