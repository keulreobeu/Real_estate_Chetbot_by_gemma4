## Purpose

- Lock the post-v9 routing correction round in active stage `03_generation_optimization`
- Reduce remaining router mismatch families before `06` readiness
- Re-run the canonical edge snapshot and downstream `03 -> 05 -> 06` pipeline on the same regenerated output

## Input Files

- `data/eval/gemma4_generation_edge_predictions_gemma4_2b.csv` (pre-v10 canonical backup source)
- `data/eval/generation_optimization/edge_regenerated_gemma4_2b_v10.csv`
- `data/eval/gemma4_generation_eval_predictions_gemma4_2b.csv`
- `data/qa/edge_case_eval.csv`
- `data/qa/eval_dataset.csv`

## Output Files

- `data/eval/gemma4_generation_edge_predictions_gemma4_2b.csv` (promoted from v10)
- `data/eval/gemma4_generation_edge_metrics_gemma4_2b.json`
- `data/eval/generation_optimization/generation_optimization_summary_gemma4_2b.json`
- `data/eval/generation_optimization/router_mismatch_review_queue_gemma4_2b.csv`
- `data/eval/generation_optimization/router_mismatch_family_counts_gemma4_2b.csv`
- `data/qa/finetuning_prep/training_candidates_gemma4_2b.csv`
- `data/qa/finetuning_prep/training_rejected_gemma4_2b.csv`
- `data/qa/finetuning_prep/train_gemma4_2b.jsonl`
- `data/qa/finetuning_prep/valid_gemma4_2b.jsonl`
- `data/qa/finetuning_prep/dataset_summary_gemma4_2b.json`
- `data/qa/finetuning_prep/stage06_readiness_gemma4_2b.json`
- `data/eval/backups/edge_promotion_v10_20260414_134140/`

## Original Row And Column Counts

- pre-v10 canonical edge predictions: 2000 rows
- promoted v10 canonical edge predictions: 2000 rows, 41 columns
- router mismatch review queue: 79 rows, 9 columns
- training candidates: 3000 rows, 29 columns
- training rejected: 2790 rows, 29 columns
- train JSONL: 50 rows
- valid JSONL: 10 rows

## Final Row And Column Counts

- edge predictions: 2000 rows, 41 columns
- router mismatch review queue: 79 rows, 9 columns
- train JSONL: 50 rows
- valid JSONL: 10 rows
- holdout edge safety rows: 150
- holdout grounded generation rows: 0

## Removed Exact Duplicate Count

- edge prediction duplicate `source_row_index`: 0
- no exact duplicate removal step was applied in this round

## Newly Generated Columns

- `router_mismatch_family` added to router mismatch review output

## Processing Steps Summary

1. Added router family taxonomy generation in `03_generation_optimization/analyze_edge_failures.py`
2. Tightened routing in `02_gemma4_generation/query_service.py`
   - subjective quality requests such as `괜찮은`, `무난한`, `추천할 만한` are treated as unsupported comparative requests
   - explicit comparative phrasing continues to route to `RECOMMEND_COMPARATIVE`
   - implicit `역 가까운` bounded filter behavior remains in place
3. Raised low-supply `05` floor in `05_finetuning_prep/prepare_sft_dataset.py`
   - minimum train rows: `50`
   - minimum valid rows: `10`
4. Promoted `edge_regenerated_gemma4_2b_v10.csv` to canonical edge predictions after backup
5. Re-ran:
   - `02_gemma4_generation/evaluate_generation_mvp.py --mode edge --model gemma4_2b`
   - `03_generation_optimization/analyze_edge_failures.py --model gemma4_2b`
   - `05_finetuning_prep/prepare_sft_dataset.py --model gemma4_2b`
   - `05_finetuning_prep/validate_06_readiness.py --model gemma4_2b`

## Major Warnings Or Exceptions

- `06` readiness is still `NO_GO`
- router target is now cleared, but grounded generation supply is still absent
- current train and valid rows are recovered only by low-supply floor logic and remain entirely `safety_refusal`
- top remaining router mismatch family is now dataset-contract misalignment on prompts like `가격 괜찮은 아파트 추천해줘`
- `data/qa/edge_case_eval.csv` currently contains `RECOMMEND_STRUCTURED=1331`, `RECOMMEND_COMPARATIVE=669`, and no `GENERAL_RETRIEVAL_QA` expected-router rows

## Key Metrics

- edge metrics
  - `router_match_rate`: `0.9605`
  - `match_status_match_rate`: `0.9395`
  - `must_not_recommend_pass_rate`: `0.9807467911318553`
  - `must_disclose_limit_pass_rate`: `1.0`
  - `insufficient_context_rate`: `0.5125`
- generation optimization buckets
  - `unsafe_recommendation`: `33`
  - `router_mismatch`: `79`
  - `match_status_mismatch`: `121`
  - `disclosure_miss`: `0`
- finetuning prep
  - `train_rows`: `50`
  - `valid_rows`: `10`
  - `holdout_grounded_generation_rows`: `0`
  - `holdout_edge_safety_rows`: `150`
- candidate answer type mix
  - `unknown_response`: `1025`
  - `apartment_fact_lookup`: `1000`
  - `unsupported_comparative_response`: `688`
  - `recommendation`: `183`
  - `comparison_recommendation`: `58`
  - `no_match_response`: `46`

## Sample Outputs

- router mismatch family counts
  - `structured_expected_but_comparative_actual`: `65`
  - `area_band_structured_but_comparative_actual`: `13`
  - `subjective_quality_structured_instead_of_comparative`: `1`
- representative router mismatch rows
  - `화성시에서 가격 괜찮은 아파트 추천해줘`
  - `연수구에서 가격 괜찮은 아파트 추천해줘`
  - `평택시에서 가격 괜찮은 아파트 추천해줘`
- boundary controls preserved
  - `674`: `NO_MATCH`
  - `675`: `EXACT_MATCH`
  - `686`: `NO_MATCH`
  - `694`: `NO_MATCH`

## Why This Matters

- the plan goal to clear the router threshold succeeded in this round
- the next blocker is no longer router safety but dataset composition
- the next correction pass should focus on grounded-generation contract recovery rather than reopening must-not routing
