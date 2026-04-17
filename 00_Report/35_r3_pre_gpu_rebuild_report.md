# 35 R3 Pre-GPU Rebuild Report

## Purpose

Regenerate `baseline-gemma4-2b-r3` run-local contextual artifacts after the
schema-validation drift fixes, then confirm the run is back to a valid
pre-GPU state.

## Input Files

- `data/qa/finetuning_prep/training_candidates_gemma4_2b.csv`
- `data/apartment_chatbot_v3.csv`
- `data/qa/finetuning_prep/train_gemma4_2b.jsonl`
- `data/qa/finetuning_prep/valid_gemma4_2b.jsonl`

## Output Files

- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/schema_v1.md`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/train_contextual.jsonl`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/valid_contextual.jsonl`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/context_build_summary.json`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/manifest.json`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/config.json`

## Final Accepted Schema Budget

- `validation_version = r3_full_sequence_v3`
- `max_docs = 1`
- `max_description_chars = 12`
- `max_seq_length = 512`

## Rebuild Summary

- The first rebuild attempt under the stricter full-sequence validator failed as designed.
- Iterative prompt compression reduced the overflow count from `831` to `38`, then `14`, then `4`.
- The accepted compact contract uses:
  - one constant short instruction
  - fixed compact row-contract keys
  - one-doc context by default
  - very short description clipping
  - disclosure-only date/scope lines
  - no repeated doc id inside each context line

## Validation Results

- `builder_pass = true`
- `schema_status = accepted`
- `rows_over_full_sequence_budget_count = 0` for both train and valid
- `contextual_selection_mode = accepted_contextual`
- manifest selected:
  - `data\\qa\\finetuning_runs\\baseline-gemma4-2b-r3\\train_contextual.jsonl`
  - `data\\qa\\finetuning_runs\\baseline-gemma4-2b-r3\\valid_contextual.jsonl`

### Token Stats

- Train prompt tokens: `min 61 / max 129 / avg 99.47`
- Train full-sequence tokens: `min 133 / max 511 / avg 379.71`
- Valid prompt tokens: `min 63 / max 128 / avg 99.48`
- Valid full-sequence tokens: `min 136 / max 496 / avg 377.86`

### Row Counts

- Train rows: `889`
- Valid rows: `99`

## Warnings

- Some rows still have no resolved doc context:
  - train: `133`
  - valid: `15`
- This no longer blocks training, but it can still cap answer quality and should be kept in mind during post-train review.

## Commands Run

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' -m py_compile .\06_finetuning\common.py .\06_finetuning\build_contextual_training_view.py .\06_finetuning\create_run_manifest.py .\06_finetuning\train_finetuning_baseline.py .\06_finetuning\generate_post_train_prediction_sets.py
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\build_contextual_training_view.py --run-id baseline-gemma4-2b-r3 --model gemma4_2b
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\create_run_manifest.py --run-id baseline-gemma4-2b-r3 --model gemma4_2b
```

## Current Status

`baseline-gemma4-2b-r3` is now back to a valid pre-GPU state.

The next step is a fresh training-time review or direct GPU training approval,
depending on the desired workflow.
