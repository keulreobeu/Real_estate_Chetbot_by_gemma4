# 03_generation_optimization

## Purpose

`03_generation_optimization` is the preparation gate before `06_finetuning`.

It focuses on fixing label noise and contract failures before training:

- recommendation safety
- disclosure behavior
- routing accuracy
- match-status accuracy
- legacy edge regeneration planning
- ambiguous structured scope handling such as `주변`, `라인`, `권역`, `쪽`
- area-band recommendation alignment for supported objective filters such as `중형이면서 역 가까운`
- unsupported structured recommendation handling for vague qualifiers such as `가성비`, `규제 적은곳`, `대단지`
- unsupported comparative handling for abstract ranking syntax such as `기준`, `좋은 순`, `후보 비교`

## Inputs

- `data/eval/gemma4_generation_edge_predictions_gemma4_2b.csv`
- `data/eval/gemma4_generation_edge_metrics_gemma4_2b.json`
- `data/qa/edge_case_eval.csv`
- `02_gemma4_generation/query_service.py`

## Outputs

- `data/eval/generation_optimization/edge_failure_buckets_gemma4_2b.csv`
- `data/eval/generation_optimization/hard_negative_review_queue_gemma4_2b.csv`
- `data/eval/generation_optimization/router_mismatch_review_queue_gemma4_2b.csv`
- `data/eval/generation_optimization/router_mismatch_family_counts_gemma4_2b.csv`
- `data/eval/generation_optimization/legacy_edge_regeneration_plan_gemma4_2b.csv`
- `data/eval/generation_optimization/generation_optimization_summary_gemma4_2b.json`

## Command

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\03_generation_optimization\analyze_edge_failures.py --model gemma4_2b
```

## 06 entry targets

- `must_not_recommend_pass_rate >= 0.90`
- `must_disclose_limit_pass_rate >= 0.90`
- `router_match_rate >= 0.85`
- `match_status_match_rate >= 0.85`
- regenerated `05_finetuning_prep` train rows `>= 500`
- regenerated `05_finetuning_prep` valid rows `>= 50`
