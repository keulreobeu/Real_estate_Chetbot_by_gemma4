## Purpose

- Add the minimum executable stage 06 assets needed to run the first baseline finetuning job
- Connect the full stage 06 loop end-to-end: manifest -> training -> post-train prediction generation -> post-train gate
- Keep the first run conservative and compatible with the current local environment

## Input Files

- `data/qa/finetuning_prep/train_gemma4_2b.jsonl`
- `data/qa/finetuning_prep/valid_gemma4_2b.jsonl`
- `data/qa/finetuning_prep/training_candidates_gemma4_2b.csv`
- `data/qa/finetuning_prep/holdout_grounded_generation.csv`
- `data/qa/finetuning_prep/holdout_edge_safety.csv`
- `data/qa/finetuning_prep/dataset_summary_gemma4_2b.json`
- `data/qa/finetuning_prep/stage06_readiness_gemma4_2b.json`
- `02_gemma4_generation/config/models.local.json`

## Output Files

- `06_finetuning/common.py`
- `06_finetuning/train_finetuning_baseline.py`
- `06_finetuning/generate_post_train_prediction_sets.py`
- `06_finetuning/README.md`
- `06_finetuning/RUNBOOK.md`
- `README.md`
- `AGENTS.md`

## Original Row And Column Counts

- `train_gemma4_2b.jsonl`: 889 rows
- `valid_gemma4_2b.jsonl`: 99 rows
- `training_candidates_gemma4_2b.csv`: 3000 rows, 29 columns
- `holdout_grounded_generation.csv`: 200 rows, 29 columns
- `holdout_edge_safety.csv`: 150 rows, 29 columns

## Final Row And Column Counts

- no data row counts changed in this round
- added 2 executable stage 06 scripts
- updated 4 documentation files

## Removed Exact Duplicate Count

- no duplicate removal step was applied in this round

## Newly Generated Columns

- no dataset columns were added
- the new stage 06 scripts generate run-specific artifacts only at execution time

## Processing Steps Summary

1. Inspected the current environment and confirmed that `torch`, `transformers`, and `accelerate` are installed locally
2. Confirmed that no real finetuning execution script existed yet in `06_finetuning`
3. Added `train_finetuning_baseline.py`
   - reads the frozen manifest
   - loads the local Gemma model from `models.local.json`
   - trains from stage 05 JSONL using a custom torch dataset and transformers `Trainer`
   - writes logs, checkpoints, final model, and a training result summary into the run directory
4. Added `generate_post_train_prediction_sets.py`
   - loads the completed finetuned model from the run's `final/` directory
   - generates `valid_predictions.csv`, `grounded_holdout_predictions.csv`, and `edge_safety_holdout_predictions.csv`
   - preserves source row metadata needed by the post-train gate
5. Updated stage 06 docs and root docs to document the full command sequence
6. Verified compile and `--help` execution for all stage 06 scripts

## Major Warnings Or Exceptions

- no actual finetuning training run was started in this round
- the first baseline trainer uses `transformers + torch` only, not PEFT/LoRA, because `peft`, `datasets`, and `trl` are not currently installed in the local environment
- this is the safest zero-new-dependency baseline, but it may be slower or heavier than a later adapter-based run

## Sample Outputs

- training command:
  - `python .\06_finetuning\train_finetuning_baseline.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b`
- prediction generation command:
  - `python .\06_finetuning\generate_post_train_prediction_sets.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b`
- full stage 06 order:
  - `create_run_manifest.py`
  - `train_finetuning_baseline.py`
  - `generate_post_train_prediction_sets.py`
  - `evaluate_post_finetuning_run.py`
