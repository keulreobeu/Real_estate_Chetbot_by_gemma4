# 18 Resume Safeguards And Edge Sanitize Report

## Purpose

Review the current GPU inference safeguards, explain the `rows` vs `max_source_row_index` mismatch, and complete non-GPU preparation work before the next resume run.

## Input files

- `02_gemma4_generation/run_generation_mvp.py`
- `02_gemma4_generation/edge_runbook_2b.ps1`
- `02_gemma4_generation/run_pipeline_until_1400.ps1`
- `data/eval/gemma4_generation_edge_predictions_gemma4_2b.csv`
- `data/qa/edge_case_eval.csv`

## Output files

- new `02_gemma4_generation/sanitize_generation_outputs.py`
- updated `02_gemma4_generation/README.md`
- updated `02_gemma4_generation/run_pipeline_until_1400.ps1`
- sanitized `data/eval/gemma4_generation_edge_predictions_gemma4_2b.csv`

## Original row and column counts

- edge predictions before sanitize: 971 rows
- current edge input: 2000 rows

## Final row and column counts

- edge predictions after sanitize: 970 rows
- current edge input: 2000 rows

## Removed exact duplicate count

- removed stale/unmappable legacy row count: 1

## Newly generated columns

- none in this run

## Processing steps summary

1. Reviewed current safeguard coverage in `run_generation_mvp.py` and `edge_runbook_2b.ps1`.
2. Confirmed the previous `971 rows / max_source_row_index 969` mismatch was caused by one legacy row without `source_row_index`.
3. Added `sanitize_generation_outputs.py` to:
   - remove rows unmappable to the current input dataset
   - backfill `source_row_index` when exact question match is unique
   - deduplicate by `source_row_index`
4. Applied sanitize to the edge 2b output.
5. Updated `run_pipeline_until_1400.ps1` to align with the current runbook:
   - `fast_edge` profile
   - `--no-startup-check`
   - `--log-every`
   - stop-signal and heartbeat paths
   - benchmark gate before edge resume

## Major warnings or exceptions

- edge output remains incomplete after sanitize:
  - `970 / 2000`
- eval output remains incomplete:
  - `5 / 1000`
- current edge output is now structurally cleaner:
  - `duplicate_question_rows=0`
  - `duplicate_source_index_rows=0`
  - `has_source_row_index=True`

## Sample outputs

Validation excerpt after sanitize:

```text
actual_rows=970
missing_rows=1030
duplicate_question_rows=0
duplicate_source_index_rows=0
has_source_row_index=True
STATUS=WARN_INCOMPLETE
```

## Processing log

- no separate processing log generated beyond standard status and validation output

