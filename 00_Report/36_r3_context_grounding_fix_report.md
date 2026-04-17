# 36 R3 Context Grounding Fix Report

## Purpose

Close the remaining pre-GPU review findings for `baseline-gemma4-2b-r3` by:

- removing `ctx=none` rows from the contextual train/valid build
- aligning prediction-time schema selection to the accepted run summary
- keeping compact cited-doc metadata consistent with the actual context selection

## Input Files

- `data/qa/finetuning_prep/training_candidates_gemma4_2b.csv`
- `data/apartment_chatbot_v3.csv`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/context_build_summary.json`

## Output Files

- `06_finetuning/common.py`
- `06_finetuning/build_contextual_training_view.py`
- `06_finetuning/generate_post_train_prediction_sets.py`
- `06_finetuning/README.md`
- `06_finetuning/RUNBOOK.md`
- `06_finetuning/CONTRACT.md`
- `06_finetuning/CHECKLIST.md`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/train_contextual.jsonl`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/valid_contextual.jsonl`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/context_build_summary.json`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/manifest.json`

## What Changed

### 1. No-context rows are excluded from contextual train/valid

- The builder now resolves cited docs before writing each contextual record.
- If a row has no resolvable doc context, it is excluded from the contextual view.
- Exclusions are reported explicitly in `rows_excluded_no_context` and counted per split.

### 2. Prediction uses the accepted run budget

- Prediction generation now reads `context_build_summary.json` when it is accepted.
- If no explicit CLI override is provided, prediction uses the accepted run budget instead of only global defaults.

### 3. Cited-doc metadata is normalized once

- `c=` now comes from the same normalized doc-id selection logic used to build actual context.
- This removes the previous mismatch between `top_doc_id`, compact cited-doc metadata, and injected doc context.

## Validation Results

- `builder_pass = true`
- `schema_status = accepted`
- `validation_version = r3_full_sequence_v3`
- `max_docs = 1`
- `max_description_chars = 12`
- `max_seq_length = 512`

### Frozen vs Contextual Counts

- Frozen train rows: `889`
- Contextual train rows: `756`
- Excluded no-context train rows: `133`
- Reconciled train total: `889`

- Frozen valid rows: `99`
- Contextual valid rows: `84`
- Excluded no-context valid rows: `15`
- Reconciled valid total: `99`

### Budget Check

- Train over full-sequence budget: `0`
- Valid over full-sequence budget: `0`

### Manifest Selection

- `contextual_selection_mode = accepted_contextual`
- selected train file:
  - `data\\qa\\finetuning_runs\\baseline-gemma4-2b-r3\\train_contextual.jsonl`
- selected valid file:
  - `data\\qa\\finetuning_runs\\baseline-gemma4-2b-r3\\valid_contextual.jsonl`

## Sample Excluded Rows

Excluded rows mostly had empty `top_doc_id` and `cited_doc_ids`, for example:

- `광주시에서 지하철 300m 이내 아파트 추천해줘`
- `화성시에서 중형 조건 맞는 단지 찾아줘`
- `오산시에서 공원 기준 아파트 비교해줘`

These are not recoverable from the existing stage 06 inputs without inventing context.

## Commands Run

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' -m py_compile .\06_finetuning\common.py .\06_finetuning\build_contextual_training_view.py .\06_finetuning\create_run_manifest.py .\06_finetuning\train_finetuning_baseline.py .\06_finetuning\generate_post_train_prediction_sets.py
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\build_contextual_training_view.py --run-id baseline-gemma4-2b-r3 --model gemma4_2b
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\create_run_manifest.py --run-id baseline-gemma4-2b-r3 --model gemma4_2b
```

## Current Status

`baseline-gemma4-2b-r3` remains in a valid pre-GPU state.

The remaining step is GPU training approval, or one more pre-train review if we want a final sanity pass before spending GPU time.
