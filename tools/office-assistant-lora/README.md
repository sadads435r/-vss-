# Cosmos3 Office Behavior LoRA Toolkit

This directory contains the reproducible dataset, training, evaluation, and audit utilities for the Office Assistant behavior-analysis data flywheel. The current experiment targets drinking-action verification, but the pipeline can be extended to other person-centric actions with separate label specifications and held-out tests.

For the complete system overview, architecture, UI, deployment flow, privacy constraints, and benchmark interpretation, see [`docs/OFFICE_ASSISTANT.md`](../../docs/OFFICE_ASSISTANT.md).

## Workflow

1. Review and trim candidate clips in the Office Assistant data-flywheel UI.
2. Export human-confirmed positives, hard negatives, and uncertain samples.
3. Build group-separated train, validation, and test manifests.
4. Apply light augmentation only to the training split and mark it with `is_augmented=true`.
5. Validate media paths, labels, split leakage, and GQA/SFT message structure.
6. Probe Cosmos3 compatibility and available GB10 memory before training.
7. Fine-tune Cosmos3-Nano Reasoner with BF16 LoRA.
8. Compare Base and LoRA on the same held-out clips and audit each regression.

## Main tools

- `build_drinking_lora_dataset.py`: builds a leakage-aware SFT dataset from reviewed clips.
- `build_drinking_expanded_test.py`: creates a larger held-out stress-test set.
- `validate_drinking_dataset.py`: validates labels, media, split membership, and manifest structure.
- `train_cosmos_drinking_lora.py`: runs BF16 PEFT LoRA supervised fine-tuning.
- `evaluate_cosmos_base.py`: evaluates the unmodified base model.
- `compare_cosmos_base_lora.py`: performs paired Base/LoRA evaluation.
- `audit_flywheel_training.py`: audits candidate provenance and training inclusion.
- `audit_unseen_drinking_samples.py`: verifies that test samples are unseen.
- `probe_cosmos_train_env.py`: checks the training environment before allocating model memory.
- `probe_cosmos_weight_map.py`: inspects the Cosmos3 weight map and module layout.
- `verify_cosmos_reasoner.py`: verifies model loading and structured response behavior.
- `convert_cosmos_reasoner_streaming.py`: adapts Reasoner checkpoints for the streaming workflow.
- `diagnose_recent_pose.py`: inspects recent pose evidence for missed candidates.
- `Dockerfile`: reproducible Cosmos3 LoRA training environment.

## Current benchmark

The expanded paired benchmark contains 137 clips: 5 positive drinking clips and 132 held-out negative clips.

| Metric | Base | LoRA |
| --- | ---: | ---: |
| False positives | 17 | 6 |
| Negative false-positive rate | 12.9% | 4.5% |
| Precision | 22.7% | 45.5% |
| Recall | 100% | 100% |
| F1 | 0.370 | 0.625 |
| JSON schema compliance | 0% | 100% |

The LoRA reduced false positives by 64.7% relative while preserving 5/5 detection on the small positive test subset. Because the test set contains only five unseen positives, the recall result is preliminary and must not be treated as production-grade evidence.

## Data rules

- Human review is the source of truth; a rule or VLM prediction is not a training label by itself.
- Split by original event before augmentation or alternate trimming.
- Do not place variants of one source event in different splits.
- Keep augmented samples below 30% of the training set and down-weight them during training.
- Do not augment validation or test clips.
- Preserve source event ID, crop boundaries, rule version, object evidence, label author, and augmentation metadata.
- Never commit recordings, frames, manifests containing private paths, model weights, LoRA adapters, databases, tokens, or live RTSP URLs.

## Reproducibility notes

The completed baseline used BF16 LoRA rank 16 with approximately 43.6 million trainable parameters, 50 optimizer steps, 400 micro-steps, and 17.51 GiB peak memory on NVIDIA GB10. Exact dataset paths, output directories, model revisions, seeds, and hashes should be recorded in the experiment report rather than hard-coded into source files.
