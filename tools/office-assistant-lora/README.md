# Office Multimodal Behavior Analytics

An experimental person-centric office behavior-analysis system built with NVIDIA VSS, DeepStream, GDINO, MediaPipe, ReID, and Cosmos3.

The project connects real-time perception with a data flywheel:

1. Detect and track people and relevant objects.
2. Publish independent MediaPipe pose features through `mdx-office-pose`.
3. Mine behavior candidates using pose, temporal rules, and object evidence.
4. Extract person-level event clips for human review.
5. Build leakage-aware SFT datasets and fine-tune a Cosmos3-Nano Reasoner with BF16 LoRA.
6. Compare the base model and LoRA on paired, held-out video clips.

## Repository layout

- `flywheel-patch/`: office API, candidate miner, GDINO client, UI, and tests.
- `pose-collector-patch/`: pose collector, motion worker, VSS configuration, and tests.
- `spark-mediapipe-patch/`: DeepStream/MediaPipe integration and diagnostic utilities.
- `build_drinking_lora_dataset.py`: leakage-aware SFT dataset builder.
- `train_cosmos_drinking_lora.py`: BF16 PEFT LoRA training entry point.
- `compare_cosmos_base_lora.py`: paired Base/LoRA evaluator.
- `PROJECT_ACHIEVEMENTS.md`: engineering notes, benchmark results, and resume material.

## Latest offline benchmark

The expanded paired benchmark contains 137 clips: 5 positive drinking clips and 132 held-out negative clips.

| Metric | Base | LoRA |
| --- | ---: | ---: |
| False positives | 17 | 6 |
| Negative false-positive rate | 12.9% | 4.5% |
| Precision | 22.7% | 45.5% |
| Recall | 100% | 100% |
| F1 | 0.370 | 0.625 |
| JSON schema compliance | 0% | 100% |

The positive test set is still small, so recall is not yet a production-grade estimate. See `PROJECT_ACHIEVEMENTS.md` for limitations and the full engineering history.

## Security and data policy

Live RTSP URLs, deployment-specific configuration, recordings, extracted frames, databases, model weights, adapters, and training outputs are intentionally excluded from Git. Use the checked-in example configuration as a starting point.
