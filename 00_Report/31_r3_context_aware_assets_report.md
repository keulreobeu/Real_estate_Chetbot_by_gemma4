# 31 R3 Context-Aware Assets Report

## Purpose

Implement the stage 06 code and documentation needed for the `baseline-gemma4-2b-r3`
context-aware second run plan without executing the actual run.

## Input Files

- `06_finetuning/common.py`
- `06_finetuning/create_run_manifest.py`
- `06_finetuning/train_finetuning_baseline.py`
- `06_finetuning/generate_post_train_prediction_sets.py`
- `06_finetuning/README.md`
- `06_finetuning/RUNBOOK.md`
- `06_finetuning/CONTRACT.md`
- `06_finetuning/CHECKLIST.md`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r2/post_train_summary.json`
- `data/qa/finetuning_prep/training_candidates_gemma4_2b.csv`
- `data/apartment_chatbot_v3.csv`

## Output Files

- `06_finetuning/build_contextual_training_view.py`
- updated stage 06 scripts and docs listed above

## Original Row And Column Counts

Reference counts from the frozen stage 05 summary:

- train rows: `889`
- valid rows: `99`
- grounded holdout rows: `200`
- edge safety holdout rows: `150`

No dataset rows were regenerated in this implementation step.

## Final Row And Column Counts

No run-local datasets were generated in this implementation step because execution
that creates or updates run directories remains approval-gated.

## Removed Exact Duplicate Count

- none

## Newly Generated Columns

- none in committed repository artifacts
- the new contextual builder is designed to emit JSONL with:
  - `instruction`
  - `input`
  - `output`

## Processing Steps Summary

1. Added shared stage 06 helpers for context-aware schema construction and apartment document lookup.
2. Added `build_contextual_training_view.py` to create run-local `train_contextual.jsonl`,
   `valid_contextual.jsonl`, `schema_v1.md`, and `context_build_summary.json`.
3. Updated `create_run_manifest.py` to auto-select run-local contextual train/valid files
   when they exist and record them in the manifest.
4. Updated `train_finetuning_baseline.py` so both `instruction` and `input` are used in the
   actual SFT prompt and training text.
5. Rewrote `generate_post_train_prediction_sets.py` to align post-train prompting with the
   unified context-aware schema and keep deterministic fallback behavior.
6. Updated stage 06 docs to reflect the new schema-first r3 flow and approval boundaries.

## Major Warnings Or Exceptions

- actual run-local contextual datasets were not created
- actual manifest creation was not executed
- actual GPU training was not executed
- actual prediction generation was not executed
- actual gate evaluation was not executed
- root `README.md` and root `AGENTS.md` were not edited in this step because stage-level docs
  were sufficient for the stage 06 contract and the root docs currently show encoding issues

## Validation Performed

- `python -m py_compile` on:
  - `06_finetuning/common.py`
  - `06_finetuning/build_contextual_training_view.py`
  - `06_finetuning/create_run_manifest.py`
  - `06_finetuning/train_finetuning_baseline.py`
  - `06_finetuning/generate_post_train_prediction_sets.py`
- `--help` verified for:
  - `06_finetuning/build_contextual_training_view.py`
  - `06_finetuning/create_run_manifest.py`
  - `06_finetuning/train_finetuning_baseline.py`
  - `06_finetuning/generate_post_train_prediction_sets.py`

## Sample Outputs

Expected new run-local artifacts for `baseline-gemma4-2b-r3` once execution is approved:

- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/schema_v1.md`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/train_contextual.jsonl`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/valid_contextual.jsonl`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/context_build_summary.json`

## Next Action

The next step is execution-time approval for:

1. run-local contextual view generation
2. manifest creation
3. `gates_and_norms` training with `max_seq_length=512`
4. post-train prediction generation
5. post-train gate evaluation
