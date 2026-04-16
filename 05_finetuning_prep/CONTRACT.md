# 05_finetuning_prep Contract

## Stage purpose

Convert completed stage 02 and 04 artifacts into a reproducible SFT candidate pool, holdout sets, and JSONL training files.

## Required input files

- `data/eval/gemma4_generation_edge_predictions_<model_id>.csv`
- `data/eval/gemma4_generation_eval_predictions_<model_id>.csv`
- `data/qa/edge_case_eval.csv`
- `data/qa/evaluation_dataset.csv`

## Required candidate columns

- `source_dataset`
- `source_row_index`
- `question`
- `answer`
- `answer_type`
- `match_status`
- `query_type`
- `top_doc_id`
- `cited_doc_ids`
- `used_fields`
- `retrieval_score`
- `insufficient_context`
- `latency_ms`
- `expected_*`
- `must_*`
- `include_in_sft`
- `exclusion_reason`
- `sft_bucket`
- `split`

## Output files

All generated data artifacts are written to `data/qa/finetuning_prep/`.

### `training_candidates_<model_id>.csv`

Contains the full candidate universe plus:

- inclusion decision
- exclusion reason
- selected SFT bucket
- split assignment

### `training_rejected_<model_id>.csv`

Contains rows excluded from the first-cycle SFT pool.

### `holdout_grounded_generation.csv`

Contains grounded-generation quality holdout rows reserved for later comparison.

### `holdout_edge_safety.csv`

Contains safety-sensitive edge holdout rows reserved for later regression checks.

### `train_<model_id>.jsonl`

One JSON object per line:

```json
{"instruction":"질문","input":"","output":"답변"}
```

### `valid_<model_id>.jsonl`

Uses the same JSONL contract as the train split.

## First-cycle exclusion rules

- exclude rows with missing `answer_type`, `match_status`, or `query_type`
- exclude empty `question` or empty `answer`
- exclude deterministic `apartment_fact_lookup`
- exclude `meta_answer`, `knowledge_answer`, and `fallback_answer`
- exclude grounded-generation rows with `insufficient_context=True`
- exclude recommendation rows unless:
  - `match_status=EXACT_MATCH`
  - `must_not_recommend != Y`
  - disclosure text exists when `must_disclose_limit=Y`

## Split policy

- reserve grounded-generation holdout before safety holdout
- reserve safety holdout before train/valid selection
- downsample non-generation classes to keep train/valid centered on grounded generation
- use deterministic hash-based selection so reruns are reproducible
