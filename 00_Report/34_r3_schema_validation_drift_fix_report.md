# 34 R3 Schema Validation Drift Fix Report

## Purpose

Fix the pre-GPU schema-validation drift in stage 06 before allowing `baseline-gemma4-2b-r3`
to proceed to GPU training.

## Input Files

- `06_finetuning/common.py`
- `06_finetuning/build_contextual_training_view.py`
- `06_finetuning/create_run_manifest.py`
- `06_finetuning/train_finetuning_baseline.py`
- `06_finetuning/generate_post_train_prediction_sets.py`
- `06_finetuning/README.md`
- `06_finetuning/RUNBOOK.md`
- `06_finetuning/CONTRACT.md`
- `06_finetuning/CHECKLIST.md`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/context_build_summary.json`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/train_contextual.jsonl`

## Output Files

- updated stage 06 scripts and stage docs listed above

## Original Row And Column Counts

Reference counts from frozen stage 05 summary:

- train rows: `889`
- valid rows: `99`
- grounded holdout rows: `200`
- edge safety holdout rows: `150`

No row counts were changed in this implementation step.

## Final Row And Column Counts

No run artifacts were regenerated in this implementation step.

## Removed Exact Duplicate Count

- none

## Newly Generated Columns

- none in repository data artifacts
- new summary metadata is now expected when the builder is rerun:
  - `schema_status`
  - `validation_version`
  - `selected_schema_budget`
  - `builder_pass`
  - `accepted_at` or `rejected_at`
  - split-level prompt/full-sequence token stats and overflow counts

## Processing Steps Summary

1. Added shared schema defaults and shared token counting helpers to `common.py`.
2. Updated the contextual builder to validate full training sequence length instead of prompt-only length.
3. Changed the builder to reject overflow builds before publishing accepted canonical artifacts.
4. Added accepted-summary guard logic to manifest creation so contextual files are only selected when the contextual build is explicitly accepted.
5. Added trainer-side sanity check that refuses contextual training when the accepted summary is missing or the contextual `max_seq_length` disagrees with the runtime argument.
6. Aligned prediction generator defaults with the accepted r3 schema budget (`max_docs=1`, `max_description_chars=96`).
7. Updated stage 06 docs to reflect full-sequence validation and accepted-summary gating.

## Major Warnings Or Exceptions

- existing `baseline-gemma4-2b-r3` artifacts remain stale after this code-only step
- current stale summary still evaluates as not accepted under the new guard
- contextual rebuild, manifest refresh, GPU training, prediction generation, and gate execution were not run in this step

## Validation Performed

- `python -m py_compile` on:
  - `06_finetuning/common.py`
  - `06_finetuning/build_contextual_training_view.py`
  - `06_finetuning/create_run_manifest.py`
  - `06_finetuning/train_finetuning_baseline.py`
  - `06_finetuning/generate_post_train_prediction_sets.py`
- `--help` verified for:
  - `06_finetuning/build_contextual_training_view.py`
  - `06_finetuning/generate_post_train_prediction_sets.py`
- read-only sanity script verified:
  - `current_r3_summary_accepted = False`
  - shared schema defaults resolve to:
    - `max_docs = 1`
    - `max_description_chars = 96`
    - `max_seq_length = 512`
    - `validation_version = r3_full_sequence_v1`
  - sample current stale row still shows prompt/full mismatch:
    - prompt tokens: `419`
    - full training tokens: `665`

## Sample Outputs

Expected accepted builder outputs after the next approved rebuild:

- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/train_contextual.jsonl`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/valid_contextual.jsonl`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/schema_v1.md`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/context_build_summary.json`

## Next Action

The next step is approval to regenerate the existing `baseline-gemma4-2b-r3` contextual
artifacts and refresh its manifest under the new validation rules. Only after that should
GPU training be considered.
