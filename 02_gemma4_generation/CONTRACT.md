# 02_gemma4_generation Contract

## Stage purpose

Build a retrieval-first generation stage that answers apartment questions using repository data and produces auditable outputs for later Gemma 4 evaluation and tuning.

## Upstream dependencies

### Required input files

- `data/apartment_chatbot_v3.csv`
- `data/qa/evaluation_dataset.csv`
- `data/qa/edge_case_eval.csv`

### Required columns in `data/apartment_chatbot_v3.csv`

- `문서ID`
- `아파트명`
- `시도`
- `시군구`
- `동`
- `전용면적`
- `공급면적`
- `공급액(만원)`
- `평당_공급액`
- `가장가까운역`
- `거리_m`
- `가장가까운역_호선요약`
- `환승역여부`
- `description`
- `검색키워드`
- `데이터기준일`
- `질의매칭태그`
- `공원_비교요약`
- `병원_비교요약`
- `교통_비교요약`

### Required columns in `data/qa/evaluation_dataset.csv`

- `question`
- `expected_answer`
- `expected_doc_id`
- `expected_answer_type`
- `expected_match_status`
- `must_include`
- `must_not_include`

### Required columns in `data/qa/edge_case_eval.csv`

- `question`
- `expected_doc`
- `expected_field`
- `expected_router_type`
- `expected_match_status`
- `must_not_recommend`
- `must_disclose_limit`

## Core outputs

### `data/eval/gemma4_generation_source_index.csv`

Required columns:

- `문서ID`
- `아파트명`
- `시도`
- `시군구`
- `동`
- `가장가까운역`
- `거리_m`
- `가장가까운역_호선요약`
- `환승역여부`
- `공급액(만원)`
- `평당_공급액`
- `description`
- `검색키워드`
- `검색텍스트`

### `data/eval/gemma4_generation_eval_predictions_<model_id>.csv`

Required columns:

- `question`
- `answer`
- `answer_type`
- `match_status`
- `query_type`
- `expected_answer`
- `expected_doc_id`
- `top_doc_id`
- `cited_doc_ids`
- `used_fields`
- `retrieval_score`
- `insufficient_context`
- `backend`
- `model_id`
- `runtime`
- `latency_ms`
- `load_runtime_ms`
- `processor_load_ms`
- `model_load_ms`
- `prompt_render_ms`
- `input_prepare_ms`
- `to_device_ms`
- `generate_ms`
- `decode_ms`
- `model_device`
- `hf_device_map`
- `device_map_requested`
- `model_source`
- `processor_source`
- `local_files_only`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `finish_reason`
- `prompt_text`

### `data/eval/gemma4_generation_edge_predictions_<model_id>.csv`

Required columns:

- `question`
- `answer`
- `answer_type`
- `match_status`
- `query_type`
- `expected_doc`
- `expected_field`
- `top_doc_id`
- `cited_doc_ids`
- `used_fields`
- `retrieval_score`
- `insufficient_context`
- `backend`
- `model_id`
- `runtime`
- `latency_ms`
- `load_runtime_ms`
- `processor_load_ms`
- `model_load_ms`
- `prompt_render_ms`
- `input_prepare_ms`
- `to_device_ms`
- `generate_ms`
- `decode_ms`
- `model_device`
- `hf_device_map`
- `device_map_requested`
- `model_source`
- `processor_source`
- `local_files_only`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `finish_reason`
- `prompt_text`

### `data/eval/gemma4_generation_eval_metrics_<model_id>.json`

Required keys:

- `total_questions`
- `retrieval_hit_rate`
- `exact_match_rate`
- `contains_expected_answer_rate`
- `answer_type_match_rate`
- `match_status_match_rate`
- `must_include_pass_rate`
- `must_not_include_pass_rate`
- `insufficient_context_rate`
- `avg_latency_ms`

### `data/eval/gemma4_generation_edge_metrics_<model_id>.json`

Required keys:

- `total_questions`
- `doc_hit_rate`
- `field_hit_rate`
- `router_match_rate`
- `match_status_match_rate`
- `must_not_recommend_pass_rate`
- `must_disclose_limit_pass_rate`
- `insufficient_context_rate`
- `avg_latency_ms`

## Local inference config

### Required config files

- `02_gemma4_generation/config/generation_defaults.json`
- `02_gemma4_generation/config/models.local.json` or `02_gemma4_generation/config/models.local.example.json`

### Shared generation keys

- `backend`
- `prompt_template`
- `top_k`
- `max_context_docs`
- `max_output_tokens`
- `temperature`
- `top_p`
- `repeat_penalty`
- `stop_sequences`
- `fallback_on_low_retrieval_score`
- `retrieval_score_threshold`
- `request_timeout_seconds`
- `web_timeout_seconds`

### Per-model keys for the primary runtime

- `model_id`
- `display_name`
- `runtime`
- `hf_model_id`
- `local_dir`
- `processor_id`
- `torch_dtype`
- `device_map`
- `attn_implementation`
- `max_input_tokens`
- `max_output_tokens`
- `supports_gpu`
- `cpu_threads` (optional, CPU inference intra-op threads; default: all logical cores)
- `cpu_interop_threads` (optional, CPU inference inter-op threads; default: 1)

### Experimental per-model keys for `llama_cpp`

- `model_path`
- `n_ctx`
- `n_gpu_layers`
- `n_threads` (optional, default: all logical cores)
- `chat_format`
- `quantization`

## Inference adapter contract

Input:

- `prompt_text: str`
- `model_config: dict`
- `generation_config: dict`

Output:

- `text: str`
- `backend: str`
- `model_id: str`
- `finish_reason: str`
- `token_usage: dict | None`
- `latency_ms: int`
- `raw_response: dict | None`

When `raw_response` is present for the `transformers` backend, it should expose:

- `timing.load_runtime_ms`
- `timing.processor_load_ms`
- `timing.model_load_ms`
- `timing.prompt_render_ms`
- `timing.input_prepare_ms`
- `timing.to_device_ms`
- `timing.generate_ms`
- `timing.decode_ms`
- `runtime_info.model_device`
- `runtime_info.hf_device_map`
- `runtime_info.device_map_requested`
- `runtime_info.model_source`
- `runtime_info.processor_source`
- `runtime_info.local_files_only`

## Generation behavior contract

The generation stage must:

- retrieve evidence before generation
- route queries into recommendation, comparative, fact lookup, knowledge, and meta answer types before fallback generation
- enforce `NO_MATCH` and `UNKNOWN` recommendation contracts without returning arbitrary apartments
- answer in Korean
- avoid inventing missing values
- expose which documents were cited
- explicitly say evidence is insufficient when grounding is weak
- keep retrieval-first as the default path
- support `gemma4_2b` and `gemma4_4b` through configuration, not separate scripts
- support graceful file-signal stop that checkpoints completed in-memory rows before exit
- expose `answer_type`, `match_status`, and `query_type` in CLI/web demo responses
- expose `data_cutoff` and `limitations` in CLI/web demo responses
- keep startup probes split into `load_only` by default and `full` only when explicitly requested

The generation stage must not:

- guess unknown values
- use data outside repository inputs
- silently drop missing required columns
- let generated text override `NO_MATCH` or `UNKNOWN` recommendation contracts

## Failure rules

Stop and report when:

- a required input file is missing
- required columns are missing
- no retrieval source rows are available
- backend and model configuration do not match
- a configured local snapshot path does not exist

Continue with warning when:

- a question has no strong retrieval hit
- an answer is forced to use the fallback insufficient-context response
- the adapter returns an empty answer
- `llama_cpp` remains unavailable on the local machine

## Validation rules

Minimum validation for this stage:

1. build source index successfully
2. run `mock` generation on `eval` and `edge`
3. run `transformers` generation on `eval` for `gemma4_2b`
4. produce model-specific metrics JSON files
5. confirm output files exist at documented paths
