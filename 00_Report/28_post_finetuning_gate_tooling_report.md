## Purpose

- Add the minimum stage 06 tooling needed to evaluate a completed finetuning run against the frozen pre-finetuning GO snapshot
- Lock in run manifest creation, integrity checks, subset evaluation, baseline comparison, and verdict generation
- Keep the implementation generic enough to work with future finetuning frameworks without changing the stage 05 contracts

## Input Files

- `data/qa/finetuning_prep/train_gemma4_2b.jsonl`
- `data/qa/finetuning_prep/valid_gemma4_2b.jsonl`
- `data/qa/finetuning_prep/training_candidates_gemma4_2b.csv`
- `data/qa/finetuning_prep/holdout_grounded_generation.csv`
- `data/qa/finetuning_prep/holdout_edge_safety.csv`
- `data/qa/finetuning_prep/dataset_summary_gemma4_2b.json`
- `data/qa/finetuning_prep/stage06_readiness_gemma4_2b.json`
- `data/eval/gemma4_generation_eval_predictions_gemma4_2b.csv`
- `data/eval/gemma4_generation_edge_predictions_gemma4_2b.csv`
- `data/eval/gemma4_generation_edge_metrics_gemma4_2b.json`
- `data/eval/generation_optimization/generation_optimization_summary_gemma4_2b.json`

## Output Files

- `06_finetuning/common.py`
- `06_finetuning/create_run_manifest.py`
- `06_finetuning/evaluate_post_finetuning_run.py`
- `06_finetuning/README.md`
- `06_finetuning/CONTRACT.md`
- `06_finetuning/RUNBOOK.md`
- `README.md`
- `AGENTS.md`

## Original Row And Column Counts

- `training_candidates_gemma4_2b.csv`: 3000 rows, 29 columns
- `holdout_grounded_generation.csv`: 200 rows, 29 columns
- `holdout_edge_safety.csv`: 150 rows, 29 columns
- `gemma4_generation_eval_predictions_gemma4_2b.csv`: 1000 rows, 41 columns
- `gemma4_generation_edge_predictions_gemma4_2b.csv`: 2000 rows, 41 columns

## Final Row And Column Counts

- no dataset row counts were changed in this tooling round
- added 3 new stage 06 python files
- updated 5 documentation files

## Removed Exact Duplicate Count

- no duplicate removal step was applied in this round

## Newly Generated Columns

- no repository data columns were added
- the new tooling reads existing contracts and writes run-specific JSON outputs at execution time

## Processing Steps Summary

1. Added shared stage 06 helpers in `06_finetuning/common.py`
2. Added `create_run_manifest.py`
   - freezes the current GO snapshot
   - records input file hashes
   - writes `manifest.json` and a run-local `config.json` template
3. Added `evaluate_post_finetuning_run.py`
   - verifies manifest/config/log/final artifact presence
   - checks frozen input hash integrity
   - evaluates `valid`, `grounded_holdout`, and `edge_safety_holdout`
   - compares each scope against the frozen pre-finetuning baseline subset
   - emits `valid_eval.json`, `grounded_holdout_eval.json`, `edge_safety_holdout_eval.json`, `post_train_summary.json`, and `notes.md`
4. Updated stage 06 docs and root docs with the new commands and prediction CSV contract
5. Ran script compile checks and command help validation
6. Ran a smoke test in a temporary workspace-backed directory, confirmed end-to-end flow, then removed the temporary artifacts to avoid confusion with a real run

## Major Warnings Or Exceptions

- no actual finetuning training run was started
- no real post-train predictions were generated in this round
- the post-train evaluator assumes run predictions preserve `source_row_index` and the generation-style output contract fields

## Sample Outputs

- manifest creation command:
  - `python .\06_finetuning\create_run_manifest.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b`
- post-train gate command:
  - `python .\06_finetuning\evaluate_post_finetuning_run.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b`
- verdict classes supported by the gate:
  - `reject`
  - `experiment_only`
  - `next_baseline`
  - `deployment_candidate`
