# 19. Non-GPU Stage04 Gate And Stage05-09 Blueprint

## purpose

Lock the non-GPU planning and readiness work needed before the next GPU resume window.

This run focused on:

- `02b_contract_alignment_gate`
- `04_evaluation_readiness_gate`
- `05_finetuning_prep` policy draft
- `07_serving_inference` boundary cleanup
- `08_monitoring_feedback` taxonomy draft
- `09_release_governance` checklist draft

## input files

- `02_gemma4_generation/CONTRACT.md`
- `02_gemma4_generation/evaluate_generation_mvp.py`
- `02_gemma4_generation/run_generation_mvp.py`
- `02_gemma4_generation/query_service.py`
- `02_gemma4_generation/README.md`
- `data/eval/gemma4_generation_edge_predictions_gemma4_2b.csv`
- `data/eval/gemma4_generation_eval_predictions_gemma4_2b.csv`
- `data/eval/gemma4_generation_edge_metrics_gemma4_2b.json`
- `data/eval/gemma4_generation_eval_metrics_gemma4_2b.json`

## output files

- `02_gemma4_generation/check_stage04_readiness.py`
- `02_gemma4_generation/README.md`
- `00_Report/19_non_gpu_stage04_gate_and_stage05_09_blueprint.md`

## original row and column counts

- edge predictions: `970` rows, `20` columns
- eval predictions: `5` rows, `27` columns
- edge input dataset: `2000` rows
- eval input dataset: `1000` rows

## final row and column counts

- no prediction regeneration performed in this run
- edge predictions remain `970` rows
- eval predictions remain `5` rows

## removed exact duplicate count

- no dataset rewrite performed in this run

## newly generated columns

- none in existing datasets
- one new non-GPU readiness gate script was added

## processing steps summary

### 1. 02b contract alignment gate

Current stage 04 readiness is **NO**.

Why:

- edge output is incomplete: `970 / 2000`
- eval output is incomplete: `5 / 1000`
- current edge prediction CSV is a legacy-format file
- `evaluate_generation_mvp.py` expects edge columns that do not exist in the current edge CSV

Current edge mismatch summary:

| category | columns |
|---|---|
| evaluator only | `expected_router_type`, `expected_match_status`, `must_not_recommend`, `must_disclose_limit` |
| evaluator missing in actual | `expected_match_status`, `expected_router_type`, `match_status`, `must_disclose_limit`, `must_not_recommend`, `query_type` |
| contract missing in actual | `answer_type`, `decode_ms`, `device_map_requested`, `generate_ms`, `hf_device_map`, `input_prepare_ms`, `load_runtime_ms`, `local_files_only`, `match_status`, `model_device`, `model_load_ms`, `model_source`, `processor_load_ms`, `processor_source`, `prompt_render_ms`, `query_type`, `to_device_ms` |
| actual only | `source_row_index`, `raw_response` |

Current eval mismatch summary:

| category | columns |
|---|---|
| evaluator missing in actual | none |
| contract missing in actual | `decode_ms`, `device_map_requested`, `generate_ms`, `hf_device_map`, `input_prepare_ms`, `load_runtime_ms`, `local_files_only`, `model_device`, `model_load_ms`, `model_source`, `processor_load_ms`, `processor_source`, `prompt_render_ms`, `to_device_ms` |
| actual only | `source_row_index`, `raw_response` |

Metric contract summary:

| mode | missing required metric keys |
|---|---|
| edge | `match_status_match_rate`, `must_disclose_limit_pass_rate`, `must_not_recommend_pass_rate`, `router_match_rate` |
| eval | none |

Important note:

- edge has a three-way drift right now:
  - current output CSV is legacy-format
  - evaluator expects safety-routing fields
  - contract edge prediction required columns do not yet list every evaluator-required safety field
- stage 04 should not be considered ready until this drift is resolved in the next GPU-backed regeneration cycle and contract/doc expectations stay aligned.

### 2. 04 evaluation readiness gate

Stage 04 completion is now defined as:

```text
required rows complete
and contract-aligned outputs present
and evaluator-required columns present
and required metric keys valid
```

Important clarification:

- `metrics json exists` is not completion
- partial metrics are untrusted when row completeness is not reached
- partial metrics are also untrusted when evaluator-required columns are absent from the actual CSV

### 3. 05 finetuning prep policy draft

Default policy for the next stage:

- deterministic-only responses are lower priority for SFT
- generation-path failures are the primary collection target
- recommendation/comparison/fact/meta should not be mixed blindly into one training pool

Recommended split:

| bucket | training priority | note |
|---|---|---|
| grounded generation failures | high | most direct tuning target |
| grounded generation successes | high | preserves desired response style |
| deterministic fact/meta answers | low | better treated as routing/contract logic first |
| unsupported or safety refusal answers | medium | useful for safety consistency, but curated carefully |

Recommended JSONL contract:

```json
{"instruction":"질문","input":"","output":"답변"}
```

Recommended split policy:

- train: majority grounded-generation rows
- valid: representative sample by query type and failure class
- holdout: safety-sensitive recommendation/refusal cases

### 4. 07 serving inference structure

Stage 07 should be framed as a promotion of existing 02 assets, not a greenfield build.

Current layer split:

| layer | current assets | role |
|---|---|---|
| runtime core | `query_service.py`, `run_generation_mvp.py`, inference adapters | routing, retrieval, answer contract, runtime metadata |
| local demo | `demo_chatbot_mvp.py`, `demo_chatbot_web_mvp.py` | smoke/demo/inspection path |
| future service | to be promoted from runtime core | stable API, timeout policy, error surface, monitoring hooks |

### 5. 08 monitoring feedback taxonomy

Recommended initial failure buckets:

| bucket | meaning | minimum fields |
|---|---|---|
| retrieval_miss | retrieved docs do not support the answer | question, query_type, top_doc_id, cited_doc_ids, retrieval_score |
| router_mismatch | predicted query routing does not match expectation | question, expected_router_type, query_type, match_status |
| unsafe_recommendation | recommendation shown when it should not be | question, match_status, top_doc_id, answer_type |
| timeout | response too slow or soft-timeout limitation triggered | question, latency_ms, generate_ms, finish_reason |
| incomplete_disclosure | limits/caveats should have been disclosed but were not | question, answer, must_disclose_limit, finish_reason |
| hallucination_like_answer | answer sounds grounded but cannot be tied to cited evidence | question, answer, cited_doc_ids, used_fields |

Recommended feedback artifact shape:

- CSV for structured review queue
- markdown incident note for major failures

### 6. 09 release governance draft

Recommended release go/no-go checkpoints:

1. edge `2000 / 2000`
2. eval `1000 / 1000`
3. stage04 contract gate `YES`
4. required metric keys present
5. no unresolved blocker in latest report

Recommended rollback triggers:

- new run produces contract regression
- evaluator-required columns disappear
- duplicate `source_row_index` rows reappear
- recommendation safety behavior regresses

Recommended recurring regression checks:

- eval contract fields present
- edge safety fields present
- resume safety by `source_row_index`
- heartbeat/status artifacts update during long run

Required report set before release-style handoff:

- latest generation runtime report
- latest readiness gate report
- latest evaluation metrics report
- latest stall/investigation report if a failure occurred

## validation method

- static inspection of:
  - `CONTRACT.md`
  - `evaluate_generation_mvp.py`
  - `run_generation_mvp.py`
  - `query_service.py`
- non-GPU readiness gate:
  - `python .\02_gemma4_generation\check_stage04_readiness.py --model gemma4_2b`
- syntax check:
  - `python -m compileall .\02_gemma4_generation\check_stage04_readiness.py`

## major warnings or exceptions

- root-level docs still present some older stage framing and should not be treated as the runtime source of truth
- current edge CSV is still a legacy-format artifact relative to the now-expanded contract
- current eval CSV is contract-closer than edge, but still incomplete and missing newer runtime debug fields

## sample outputs

Expected gate conclusion right now:

```text
stage04_ready=False
CONCLUSION=NO
```

Expected completion definition:

```text
required_rows_complete + contract_aligned + evaluator_columns_present + required_metric_keys_valid
```
