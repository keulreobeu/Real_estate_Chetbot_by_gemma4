# 03_generation_optimization Contract

## Stage purpose

Reduce edge-eval label noise and recommendation-safety failures before the first SFT run.

## Required inputs

- completed edge predictions CSV
- completed edge metrics JSON
- edge-case evaluation CSV
- current `02_gemma4_generation` runtime code

## Required analysis outputs

### `edge_failure_buckets_<model_id>.csv`

Required columns:

- `bucket`
- `rows`

### `hard_negative_review_queue_<model_id>.csv`

Required columns:

- `source_row_index`
- `question`
- `answer`
- `answer_type`
- `match_status`
- `query_type`
- `expected_router_type`
- `expected_match_status`
- `must_not_recommend`
- `must_disclose_limit`
- `top_doc_id`
- `used_fields`
- `area_band_unknown_candidate`
- `recommended_action`

### `legacy_edge_regeneration_plan_<model_id>.csv`

Required columns:

- `source_row_index`
- `question`
- `recommended_action`

### `generation_optimization_summary_<model_id>.json`

Required keys:

- `model_id`
- `input_rows`
- `buckets`
- `recommended_next_step`

## Policy

- hard-negative rows are review queue items, not SFT rows
- legacy rows must be regenerated with the current prediction schema before they can enter SFT
- safety/routing changes must be evaluated before `05_finetuning_prep` is regenerated
