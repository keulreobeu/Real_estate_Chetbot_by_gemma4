# 06_finetuning Runbook

## Goal

Run one schema-first context-aware finetuning attempt for `baseline-gemma4-2b-r3`
without mutating stage 05 canonical artifacts.

## Pre-Run Sequence

1. confirm the latest readiness verdict is still `GO`
2. lock `context-aware input schema v1`
3. build the run-local contextual train/valid view
4. verify contextual row counts, excluded no-context rows, and token budget
5. create the run manifest
6. request user approval for the actual GPU training command

## Baseline Recommendation

Stay on the proven local path:

- one base model
- one run id
- one unified compressed schema
- one train split
- one valid split
- no sweep
- `gates_and_norms` partial finetune
- `max_seq_length = 512`
- accepted r3 compression defaults:
  - `max_docs = 1`
  - `max_description_chars = 12`

## Required Run-Local Artifacts

Before training, the run directory should contain:

- `schema_v1.md`
- `train_contextual.jsonl`
- `valid_contextual.jsonl`
- `context_build_summary.json`
- `context_build_summary.json` must show:
  - `builder_pass = true`
  - `schema_status = accepted`
  - `validation_version = r3_full_sequence_v3`
- unresolved doc rows may be excluded, but the summary must record those exclusions explicitly
- `manifest.json`
- `config.json`

## Post-Run Sequence

1. verify `train.log`, `checkpoints/`, and `final/` exist
2. generate `valid_predictions.csv`
3. generate `grounded_holdout_predictions.csv`
4. generate `edge_safety_holdout_predictions.csv`
5. run `evaluate_post_finetuning_run.py`
6. summarize valid quality
7. summarize grounded holdout quality
8. summarize edge safety quality
9. compare with:
   - pre-finetuning baseline
   - `baseline-gemma4-2b-r2`
10. classify:
   - `reject`
   - `experiment_only`
   - `next_baseline`

## Stop Conditions

Stop and report if:

- contextual JSONL row counts do not match the frozen split
- excluded no-context rows are not reported clearly in the summary
- any contextual row has empty input or empty output
- full training sequence budget is exceeded
- contextual summary is missing or not accepted
- the output directory already exists for another run
- training cannot write logs or checkpoints
- post-train predictions cannot be produced

## Approval Prompts

Training:

`baseline-gemma4-2b-r3`를 context-aware schema v1 기준으로 학습해도 될까요? frozen stage05 inputs는 유지하고, stage06 run-local training view만 새로 만들겠습니다.

Prediction generation:

`r3` final 모델로 valid, grounded holdout, edge safety holdout prediction을 생성해도 될까요? row metadata와 cited doc context를 포함한 context-aware prompt를 사용하겠습니다.

Gate evaluation:

`r3` post-train gate를 실행해도 될까요? 기존 evaluator는 유지하고, 이번 라운드는 verdict와 blocker만 판정하겠습니다.
