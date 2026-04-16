# 23. Generation Optimization Gate Report

## purpose

Implement the non-GPU preparation work required before `06_finetuning`.

This run focused on:

- creating `03_generation_optimization`
- analyzing edge failure buckets
- producing a hard-negative review queue
- producing a legacy edge regeneration plan
- adding a `06` readiness gate
- fixing one deterministic recommendation-safety gap for unsupported area-band terms

## input files

- `data/eval/gemma4_generation_edge_predictions_gemma4_2b.csv`
- `data/eval/gemma4_generation_edge_metrics_gemma4_2b.json`
- `data/eval/gemma4_generation_eval_metrics_gemma4_2b.json`
- `data/qa/edge_case_eval.csv`
- `data/qa/finetuning_prep/dataset_summary_gemma4_2b.json`
- `02_gemma4_generation/query_service.py`
- `05_finetuning_prep/prepare_sft_dataset.py`

## output files

New code and docs:

- `03_generation_optimization/analyze_edge_failures.py`
- `03_generation_optimization/README.md`
- `03_generation_optimization/CONTRACT.md`
- `05_finetuning_prep/validate_06_readiness.py`

Generated artifacts:

- `data/eval/generation_optimization/edge_failure_buckets_gemma4_2b.csv`
- `data/eval/generation_optimization/hard_negative_review_queue_gemma4_2b.csv`
- `data/eval/generation_optimization/legacy_edge_regeneration_plan_gemma4_2b.csv`
- `data/eval/generation_optimization/generation_optimization_summary_gemma4_2b.json`
- `data/eval/generation_optimization/area_band_safety_smoke_gemma4_2b.csv`
- `data/qa/finetuning_prep/stage06_readiness_gemma4_2b.json`

Updated docs:

- `README.md`
- `AGENTS.md`
- `05_finetuning_prep/README.md`

## original row and column counts

- edge predictions: `2000` rows
- edge dataset: `2000` rows
- stage 05 candidate universe: `3000` rows
- stage 05 train rows: `85`
- stage 05 valid rows: `8`

## final row and column counts

- `edge_failure_buckets_gemma4_2b.csv`: bucket summary rows
- `hard_negative_review_queue_gemma4_2b.csv`: current-schema unsafe recommendation review rows
- `legacy_edge_regeneration_plan_gemma4_2b.csv`: `970` legacy rows
- `area_band_safety_smoke_gemma4_2b.csv`: `42` smoke rows
- `stage06_readiness_gemma4_2b.json`: `NO_GO`

## removed exact duplicate count

- no source dataset rows were removed in this run
- no canonical prediction CSV was rewritten

## newly generated columns

`analyze_edge_failures.py` generates these analysis flags:

- `is_legacy_schema`
- `router_mismatch`
- `match_status_mismatch`
- `unsafe_recommendation`
- `disclosure_required`
- `has_limit_disclosure`
- `disclosure_miss`
- `area_band_unknown_candidate`
- `recommended_action`

## processing steps summary

### 1. Added `03_generation_optimization`

The new stage now records edge-failure buckets before finetuning. It produces:

- failure bucket counts
- hard-negative review queue
- legacy-row regeneration plan
- JSON summary for follow-up automation

### 2. Added `06` readiness gate

`validate_06_readiness.py` checks:

- train rows `>= 500`
- valid rows `>= 50`
- missing contract fields `<= 5`
- safety holdout rows `>= 150`
- grounded holdout rows `>= 200`
- `must_not_recommend_pass_rate >= 0.90`
- `must_disclose_limit_pass_rate >= 0.90`
- `router_match_rate >= 0.85`
- `match_status_match_rate >= 0.85`
- latency no worse than baseline + 20%
- eval deterministic metrics preserved

Current verdict:

```text
NO_GO
```

### 3. Fixed one deterministic safety gap

`query_service.py` now treats unsupported area-band terms as insufficiently structured recommendation filters:

- `소형`
- `중소형`
- `중형`
- `중대형`
- `대형`

These now return `unknown_response / UNKNOWN` instead of arbitrary apartment recommendations when no supported numeric area filter exists.

## validation method

Syntax checks:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' -m compileall .\03_generation_optimization\analyze_edge_failures.py .\05_finetuning_prep\validate_06_readiness.py .\02_gemma4_generation\query_service.py
```

Analysis run:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\03_generation_optimization\analyze_edge_failures.py --model gemma4_2b
```

Readiness gate:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\05_finetuning_prep\validate_06_readiness.py --model gemma4_2b
```

Area-band smoke check:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\run_generation_mvp.py --mode edge --backend mock --model gemma4_2b --offset 960 --limit 42 --output-path .\data\eval\generation_optimization\area_band_safety_smoke_gemma4_2b.csv --no-startup-check --checkpoint-every 0 --log-every 10
```

Smoke result:

- rows `960-1000`: `unknown_response / UNKNOWN / RECOMMEND_STRUCTURED`
- row `1001`: remains `recommendation / EXACT_MATCH / RECOMMEND_STRUCTURED`

## major warnings or exceptions

- `git status` could not run because the repo is still blocked by Git dubious ownership detection.
- canonical edge predictions were not rewritten in this run.
- `05_finetuning_prep` was not regenerated because canonical predictions still contain `970` legacy rows.
- GPU-backed regeneration is still needed before `06_finetuning` can be considered.

## sample outputs

Current `06` readiness:

```json
{
  "ready_for_06": false,
  "verdict": "NO_GO"
}
```

Area-band smoke sample:

```text
평택시에서 중형이면서 가격 괜찮은 단지 찾아줘 -> unknown_response / UNKNOWN / RECOMMEND_STRUCTURED
평택시에서 지하철 300m 이내 아파트 추천해줘 -> recommendation / EXACT_MATCH / RECOMMEND_STRUCTURED
강동구에서 지하철 300m 이내 아파트 추천해줘 -> no_match_response / NO_MATCH / RECOMMEND_STRUCTURED
```
