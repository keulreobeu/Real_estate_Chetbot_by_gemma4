# 09 Non-GPU Validation And Metrics Report

## Purpose

Run CPU-only progress tasks while GPU inference is paused:

- add generation output completeness validator
- validate current `eval/edge` prediction artifacts
- regenerate missing edge metrics for `gemma4_4b`

## Input files

- `data/qa/edge_case_eval.csv`
- `data/qa/evaluation_dataset.csv`
- `data/eval/gemma4_generation_edge_predictions_gemma4_2b.csv`
- `data/eval/gemma4_generation_eval_predictions_gemma4_2b.csv`
- `data/eval/gemma4_generation_edge_predictions_gemma4_4b.csv`

## Output files

- `02_gemma4_generation/validate_generation_outputs.py` (new)
- updated `02_gemma4_generation/README.md`
- `data/eval/gemma4_generation_edge_metrics_gemma4_4b.json` (generated)

## Original row and column counts

- edge input: 2000 rows, 3 columns
- eval input: 1000 rows, 3 columns
- edge predictions (2b): 20 rows, 16 columns
- eval predictions (2b): 100 rows, 16 columns
- edge predictions (4b): 20 rows, 16 columns

## Final row and column counts

- no preprocessing transformation executed in this run
- prediction artifact sizes are unchanged except regenerated edge metrics for `gemma4_4b`

## Removed exact duplicate count

- not applicable (validation/metrics task only)

## Newly generated columns

- none

## Processing steps summary

1. Added `validate_generation_outputs.py` to check:
   - expected vs actual row count
   - required columns per mode
   - duplicate questions
   - empty answers
2. Added README commands for non-GPU validation workflow.
3. Executed validation for:
   - `edge + gemma4_2b`
   - `eval + gemma4_2b`
   - `edge + gemma4_4b`
4. Regenerated `edge` metrics for `gemma4_4b` using existing prediction CSV.

## Major warnings or exceptions

- `edge + gemma4_2b`: incomplete (`expected_rows=2000`, `actual_rows=20`, `missing_rows=1980`)
- `eval + gemma4_2b`: incomplete (`expected_rows=1000`, `actual_rows=100`, `missing_rows=900`)
- `edge + gemma4_4b`: incomplete (`expected_rows=2000`, `actual_rows=20`, `missing_rows=1980`)
- No schema failures were detected in validated files (`missing_columns=[]`).

## Sample outputs

Validation output excerpt:

```text
=== GENERATION OUTPUT VALIDATION ===
mode=edge
model_id=gemma4_2b
expected_rows=2000
actual_rows=20
missing_rows=1980
STATUS=WARN_INCOMPLETE
```

Generated metrics excerpt:

```json
{
  "model_id": "gemma4_4b",
  "total_questions": 20,
  "doc_hit_rate": 0.1,
  "field_hit_rate": 0.4,
  "insufficient_context_rate": 0.0,
  "avg_latency_ms": 0.0
}
```

## Processing log

- no separate log file was generated for this validation run

