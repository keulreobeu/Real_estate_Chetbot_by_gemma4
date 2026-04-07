# 14. Edge Stall Detector Hardening Report

## Purpose

- Harden long-run edge generation observability.
- Distinguish `BLOCKED_BY_GATE` from real stalled runs.
- Add multi-signal status detection using output, log, heartbeat, and benchmark state.

## Input files

- `02_gemma4_generation/run_generation_mvp.py`
- `02_gemma4_generation/monitor_edge_progress.py`
- `02_gemma4_generation/edge_runbook_2b.ps1`
- `02_gemma4_generation/run_edge_2b_until_1400.ps1`
- `02_gemma4_generation/README.md`
- `data/eval/gemma4_generation_edge_predictions_gemma4_2b.csv`
- `logs/edge_2b_resume_20260406-143542.log`
- `logs/edge_2b_benchmark_20260406-140151.json`

## Output files

- Updated:
  - `02_gemma4_generation/run_generation_mvp.py`
  - `02_gemma4_generation/monitor_edge_progress.py`
  - `02_gemma4_generation/edge_runbook_2b.ps1`
  - `02_gemma4_generation/run_edge_2b_until_1400.ps1`
  - `02_gemma4_generation/README.md`
- New:
  - `00_Report/14_edge_stall_detector_hardening_report.md`

## Original and final counts

- `data/eval/gemma4_generation_edge_predictions_gemma4_2b.csv`
  - original rows: `971`
  - final rows: `971`
  - original columns: `17`
  - final columns: `17`
- No dataset rewrite was performed in this task.

## Removed exact duplicate count

- `0`

## Newly generated columns

- None in committed dataset outputs.
- New runtime heartbeat artifact path introduced:
  - `data/eval/gemma4_generation_edge_predictions_gemma4_2b.heartbeat.json`

## Processing steps summary

1. Verified current edge output progress directly from the prediction CSV.
2. Confirmed the current run was no longer active and that previous logs mixed gate-fail and resume scenarios.
3. Added heartbeat emission to the generation runner for PID-based freshness checks.
4. Replaced single-signal monitor logic with multi-signal detector logic.
5. Extended the runbook status view to print detector verdict and remaining progress.
6. Hardened worker logs with explicit `RUN_STATE` markers for gate-blocked vs run-exit cases.
7. Updated stage documentation for new detector behavior and heartbeat usage.

## Root cause hypothesis

- Confirmed:
  - Resume and checkpoint logic were already present.
  - The main gap was observability, not output append safety.
- Inferred:
  - Users could not quickly tell whether a slow run was still alive, already stopped, or blocked by the benchmark gate because status signals were split across separate files without a single verdict layer.

## Major warnings or exceptions

- Windows process command-line inspection is access-restricted in this environment.
- To avoid that limitation, the detector now relies on a runner-written heartbeat file with PID and last-event timestamps.
- Current status on the real artifact still indicates an incomplete run:
  - verdict: `STALL_CONFIRMED`
  - max `source_row_index`: `969`
  - remaining rows: `1029`

## Sample outputs

Real current status sample:

```text
VERDICT: STALL_CONFIRMED | rows=971 | max_source_row_index=969 | remaining=1029 | process_alive=false | output_fresh=false | log_fresh=false | heartbeat_fresh=false
```

Synthetic gate-block sample:

```text
VERDICT: BLOCKED_BY_GATE | rows=971 | max_source_row_index=969 | remaining=1029 | process_alive=false | output_fresh=false | log_fresh=true | heartbeat_fresh=false
```

## Validation performed

- Syntax validation:
  - `python -m py_compile 02_gemma4_generation/run_generation_mvp.py`
  - `python -m py_compile 02_gemma4_generation/monitor_edge_progress.py`
- Runbook status validation:
  - `powershell -ExecutionPolicy Bypass -File .\02_gemma4_generation\edge_runbook_2b.ps1 -Action status`
- Detector verdict validation:
  - confirmed real artifact returns `STALL_CONFIRMED`
  - confirmed synthetic benchmark-block sample returns `BLOCKED_BY_GATE`

## Remaining follow-up

- The new heartbeat file will only appear after the next generation run starts with the updated runner.
- A future improvement could add detector regression tests around verdict classification logic.
