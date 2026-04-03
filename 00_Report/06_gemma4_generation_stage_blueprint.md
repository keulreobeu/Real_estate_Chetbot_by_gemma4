# 06 Gemma4 Generation Stage Blueprint

## Purpose

Define the smallest executable draft for `02_gemma4_generation`.

This stage exists to prove a retrieval-first, grounded answer path before any fine-tuning work starts.

## Input files

- `data/apartment_chatbot_v3.csv`
- `data/qa/evaluation_dataset.csv`
- `data/qa/edge_case_eval.csv`

## Planned output files

- `data/eval/gemma4_generation_source_index.csv`
- `data/eval/gemma4_generation_eval_predictions_<model_id>.csv`
- `data/eval/gemma4_generation_edge_predictions_<model_id>.csv`
- `data/eval/gemma4_generation_eval_metrics_<model_id>.json`
- `data/eval/gemma4_generation_edge_metrics_<model_id>.json`
- `00_Report/07_gemma4_generation_model_comparison.md`

## Original row and column counts

Not executed in this blueprint document. These counts are produced by `02_gemma4_generation/build_generation_assets.py` during validation runs.

## Final row and column counts

Not executed in this blueprint document. Output counts are produced during validation runs.

## Removed exact duplicate count

No dataset transformation was executed in this planning change.

## Newly generated columns

Planned generated columns:

- `검색텍스트`
- `top_doc_id`
- `cited_doc_ids`
- `used_fields`
- `retrieval_score`
- `insufficient_context`
- `model_id`
- `runtime`
- `latency_ms`
- `finish_reason`
- `prompt_text`

## Processing steps summary

1. Validate the main RAG dataset and both evaluation datasets.
2. Build a retrieval-ready source index.
3. Retrieve top-k candidate rows per question.
4. Build a grounded prompt.
5. Run a backend adapter.
6. Save predictions and evaluate them offline.

## Major warnings or exceptions

- The included backend is `mock`, not a real Gemma 4 runtime.
- The `gemma4` backend path is intentionally left unimplemented until local inference setup is chosen.
- Existing repository README content was not re-encoded or broadly edited in this change.

## Sample outputs

Expected prediction columns:

- `question`
- `answer`
- `top_doc_id`
- `cited_doc_ids`
- `used_fields`
- `retrieval_score`
- `insufficient_context`
- `backend`
- `model_id`
- `runtime`
- `latency_ms`
- `finish_reason`
- `prompt_text`
