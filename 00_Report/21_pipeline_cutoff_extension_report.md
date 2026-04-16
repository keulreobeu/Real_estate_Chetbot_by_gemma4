# 21. Pipeline Cutoff Extension Report

## purpose

Extend the long-running orchestration worker so it can run beyond the old `14:00` cutoff and continue the post-edge flow automatically.

## input files

- `02_gemma4_generation/run_pipeline_until_1400.ps1`
- `02_gemma4_generation/check_stage04_readiness.py`

## output files

- updated `02_gemma4_generation/run_pipeline_until_1400.ps1`
- `00_Report/21_pipeline_cutoff_extension_report.md`

## original row and column counts

- edge predictions at extension time: `1841` rows
- eval predictions at extension time: `5` rows

## final row and column counts

- no dataset regeneration performed in this change

## removed exact duplicate count

- none

## newly generated columns

- none

## processing steps summary

1. Added `CutoffHour` and `CutoffMinute` parameters to the orchestration worker.
2. Switched log naming from fixed `1400` to cutoff-aware naming.
3. Kept the main generation order:
   - edge resume
   - eval resume
   - stage 04 metrics
4. Added post-stage-04 readiness gate execution:
   - `check_stage04_readiness.py --model gemma4_2b`
5. Kept post-04 validation steps in place so the worker can leave a clearer end state after inference.

## validation method

- static script inspection
- parameter syntax sanity via PowerShell invocation in the next worker launch

## major warnings or exceptions

- the worker still uses the same benchmark gate and log streaming behavior as before
- `05~09` fully autonomous execution is not implemented because those stages are mostly decision and document driven, not single-command batch work

## sample outputs

Expected launch pattern after this change:

```powershell
powershell -ExecutionPolicy Bypass -File .\02_gemma4_generation\run_pipeline_until_1400.ps1 -CheckpointEvery 10 -CutoffHour 23 -CutoffMinute 0
```
