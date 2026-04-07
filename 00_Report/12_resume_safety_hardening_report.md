# 12 Resume Safety Hardening Report

## Purpose

Strengthen interrupted-run recovery safety for generation jobs by replacing fragile question-text resume matching with safer row-index based matching.

## Input files

- `02_gemma4_generation/run_generation_mvp.py`
- `02_gemma4_generation/validate_generation_outputs.py`
- `02_gemma4_generation/README.md`

## Output files

- updated `02_gemma4_generation/run_generation_mvp.py`
- updated `02_gemma4_generation/validate_generation_outputs.py`
- updated `02_gemma4_generation/README.md`

## Original row and column counts

- not applicable (logic hardening only)

## Final row and column counts

- no dataset transformation executed

## Removed exact duplicate count

- not applicable

## Newly generated columns

- new prediction output column: `source_row_index` (for newly generated outputs)

## Processing steps summary

1. Added safer resume path in generation runner:
   - preferred resume key: `source_row_index`
   - legacy fallback: `question` matching only when old outputs do not have index key
2. Added write-time deduplication by `source_row_index` to prevent accidental duplicate append.
3. Added `source_row_index` to each generated prediction row.
4. Updated validation script to report:
   - `duplicate_source_index_rows`
   - `has_source_row_index`
   - legacy warning status when old outputs have no index key.
5. Updated README with resume safety behavior notes.

## Major warnings or exceptions

- Existing historical prediction CSV files do not contain `source_row_index`.
- Those files are now reported as `WARN_LEGACY_NO_SOURCE_ROW_INDEX` until regenerated.

## Sample outputs

Validation excerpt on legacy file:

```text
has_source_row_index=False
STATUS=WARN_LEGACY_NO_SOURCE_ROW_INDEX
```

## Processing log

- no separate log file generated

