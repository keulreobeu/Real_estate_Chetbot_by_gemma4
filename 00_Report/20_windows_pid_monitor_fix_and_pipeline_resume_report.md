# 20. Windows PID Monitor Fix And Pipeline Resume Report

## purpose

Fix the Windows-side monitoring failure discovered while resuming the long-running edge pipeline to 14:00.

## input files

- `02_gemma4_generation/run_pipeline_until_1400.ps1`
- `02_gemma4_generation/edge_runbook_2b.ps1`
- `02_gemma4_generation/monitor_edge_progress.py`
- `logs/pipeline_until_1400_20260408-051511.log`
- `data/eval/gemma4_generation_edge_predictions_gemma4_2b.heartbeat.json`

## output files

- updated `02_gemma4_generation/run_pipeline_until_1400.ps1`
- updated `02_gemma4_generation/edge_runbook_2b.ps1`
- updated `02_gemma4_generation/monitor_edge_progress.py`
- `00_Report/20_windows_pid_monitor_fix_and_pipeline_resume_report.md`

## original row and column counts

- edge predictions before live resume confirmation: `970` rows

## final row and column counts

- edge predictions reached live progress beyond `1020` rows during this run

## removed exact duplicate count

- no dataset rewrite or duplicate removal performed in this run

## newly generated columns

- none

## processing steps summary

1. Investigated why the `14:00` worker failed immediately after launch.
2. Confirmed root cause:
   - PowerShell helper functions used `param([string[]]$Args)` and then splatted `@Args`
   - that conflicted with PowerShell automatic argument handling and launched `python` without the target script
3. Fixed the helper variable name in:
   - `run_pipeline_until_1400.ps1`
   - `edge_runbook_2b.ps1`
4. Re-ran benchmark and relaunched the `14:00` worker.
5. Confirmed real resume progress:
   - edge rows increased from `970` to `1020+`
   - heartbeat reported active progress with `current_source_row_index` advancing
6. Investigated why `edge_runbook_2b.ps1 -Action status` still threw a traceback on Windows.
7. Confirmed second root cause:
   - `monitor_edge_progress.py` used `os.kill(pid, 0)` for PID liveness checks
   - on this Windows environment that raised `WinError 87`
8. Switched Windows PID liveness checks to `tasklist`, keeping the existing non-Windows path unchanged.

## validation method

- live benchmark execution:
  - `powershell -ExecutionPolicy Bypass -File .\02_gemma4_generation\edge_runbook_2b.ps1 -Action benchmark -SampleSize 5 -Execute`
- live worker launch:
  - background `run_pipeline_until_1400.ps1`
- live evidence check:
  - pipeline log tail
  - heartbeat json inspection
  - edge CSV row count increase

## major warnings or exceptions

- benchmark gate still reports `rows_per_hour=0.0` in deterministic-heavy samples, so the gate is currently conservative and noisy
- runbook `status` also points at an older `edge_2b_*.log` naming family while the new orchestrator writes `pipeline_until_1400_*.log`
- the generation worker itself is progressing; the remaining gap is monitoring polish, not generation correctness

## sample outputs

Observed live progress after relaunch:

```text
rows=1020
max_source_row_index=1019
remaining_rows=980
```

Observed heartbeat excerpt:

```json
{
  "state": "running",
  "event": "row_started",
  "processed_count": 59,
  "current_total_rows": 1029,
  "current_source_row_index": 1029
}
```
