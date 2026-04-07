# 13. Edge Generation Stall Investigation and Fast Resume Policy

## Purpose

- Investigate why edge generation looked stalled.
- Implement a root-cause fix path focused on faster resume from current progress.
- Add prevention policy so long runs are observable and recoverable.

## Input files

- `data/qa/edge_case_eval.csv`
- `data/eval/gemma4_generation_edge_predictions_gemma4_2b.csv` (existing in-progress output)
- `logs/edge_2b_run_20260406-045314.out.log`
- `logs/edge_2b_run_20260406-045314.err.log`
- `02_gemma4_generation/run_generation_mvp.py`
- `02_gemma4_generation/edge_runbook_2b.ps1`
- `02_gemma4_generation/README.md`

## Output files

- Updated:
  - `02_gemma4_generation/run_generation_mvp.py`
  - `02_gemma4_generation/edge_runbook_2b.ps1`
  - `02_gemma4_generation/README.md`
- New:
  - `00_Report/13_edge_generation_stall_investigation_and_fast_resume_policy.md`

## Original and final counts

- `edge_case_eval.csv`: rows `2000`, columns `3` (unchanged)
- `gemma4_generation_edge_predictions_gemma4_2b.csv`: rows `771`, columns `17` at investigation time (no rewrite performed in this task)

## Removed exact duplicate count

- `0` in this task (no dataset rewrite run performed).

## Newly generated columns

- None. Output schema was not changed.

## Root cause summary

1. Process was not blocked:
   - Python generation process stayed alive and CPU usage kept increasing.
   - GPU utilization was at 100% (`nvidia-smi`), so compute was active.
2. Main reason for "stalled" perception:
   - Long per-row inference latency with default long-run settings.
   - Limited visible progress feedback during long runs.
3. Supporting evidence:
   - Existing output latency distribution (current file): `p50 ~26.5s`, `p90 ~36.8s`, max outlier much higher.
   - Checkpoint logs showed progress but with large wall-clock gaps.

## Processing steps summary

1. Confirmed runtime state:
   - Checked live process + CPU delta over time.
   - Checked GPU utilization and active process mapping.
2. Confirmed generation progress:
   - Verified output CSV row count and last write timestamp.
3. Implemented fast-resume controls:
   - Added `--profile fast_edge` in `run_generation_mvp.py`.
   - Added runtime override flags for generation config.
   - Added periodic progress/ETA logging (`--log-every`).
   - Made debug-heavy columns optional (`--save-debug-columns`) for long runs.
4. Hardened runbook defaults for long execution:
   - `-Limit 0` => process all remaining rows.
   - Raised default checkpoint interval to reduce write overhead.
   - Added fast resume preset path.
5. Updated stage docs with prevention policy and recommended commands.

## Major warnings or exceptions

- PowerShell execution policy blocks direct `.ps1` execution in current environment.
  - Workaround validated: `powershell -ExecutionPolicy Bypass -File ...`
- Existing output includes some rows without `source_row_index` from legacy runs.
  - Resume logic already supports legacy question matching fallback.

## Sample outputs

- Runbook fast resume command preview:
  - `python ...run_generation_mvp.py --mode edge --backend transformers --model gemma4_2b --offset 0 --checkpoint-every 25 --log-every 10 --no-startup-check --resume --append --profile fast_edge`
- Existing output sample rows were inspected from:
  - `data/eval/gemma4_generation_edge_predictions_gemma4_2b.csv`

## Validation performed

- Syntax validation:
  - `python -m py_compile 02_gemma4_generation/run_generation_mvp.py ...`
- Runbook command validation:
  - `edge_runbook_2b.ps1 -Action print` (bypass mode)
  - `edge_runbook_2b.ps1 -Action resume -FastProfile -NoStartupCheck` (preview mode)
- No full generation rerun was executed in this report step.

## Remaining follow-up

- Stop old long-running process and restart with fast resume profile to realize the speedup.
- After completion, run:
  - `validate_generation_outputs.py --mode edge --model gemma4_2b`
  - `evaluate_generation_mvp.py --mode edge --model gemma4_2b`
