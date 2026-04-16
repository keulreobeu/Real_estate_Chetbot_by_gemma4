# Stage 05 Dataset Contract

## First-cycle intent

The first `05_finetuning_prep` cycle is intentionally narrow.

- train the model on `grounded_generation` behavior first
- keep a smaller curated subset for refusal and safety consistency
- keep only contract-safe recommendation examples
- leave deterministic fact/meta/knowledge behavior in the runtime contract path

## Selected buckets

- `grounded_generation`
- `safety_refusal`
- `safe_recommendation`

## Rejected buckets

- `apartment_fact_lookup`
- `meta_answer`
- `knowledge_answer`
- `fallback_answer`
- legacy rows missing `answer_type`, `match_status`, or `query_type`
- recommendation rows that violate safety flags

## Holdout policy

- `holdout_grounded_generation.csv`
  - reserved for generation-quality comparison after optimization and tuning
- `holdout_edge_safety.csv`
  - reserved for safety and contract regression checks after optimization and tuning

## JSONL contract

```json
{"instruction":"질문","input":"","output":"답변"}
```
