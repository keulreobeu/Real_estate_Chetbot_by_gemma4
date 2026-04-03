# 02_gemma4_generation Contract

## Stage purpose

Build a retrieval-first generation stage that answers apartment questions using repository data and produces auditable outputs for later Gemma 4 integration.

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

### Required columns in `data/qa/evaluation_dataset.csv`

- `question`
- `expected_answer`
- `문서ID`

### Required columns in `data/qa/edge_case_eval.csv`

- `question`
- `expected_doc`
- `expected_field`

## Core outputs

### `data/eval/gemma4_generation_source_index.csv`

Purpose:
- retrieval-ready source index derived from the main RAG dataset

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

Purpose:
- model-ready prediction log for the standard evaluation set

Required columns:
- `question`
- `answer`
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
- `finish_reason`
- `prompt_text`

### `data/eval/gemma4_generation_edge_predictions_<model_id>.csv`

Purpose:
- prediction log for edge-case retrieval and grounding checks

Required columns:
- `question`
- `answer`
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
- `finish_reason`
- `prompt_text`

### `data/eval/gemma4_generation_eval_metrics_<model_id>.json`

Purpose:
- summary metrics for the standard evaluation set

Required keys:
- `total_questions`
- `retrieval_hit_rate`
- `exact_match_rate`
- `contains_expected_answer_rate`
- `insufficient_context_rate`
- `avg_latency_ms`

### `data/eval/gemma4_generation_edge_metrics_<model_id>.json`

Purpose:
- summary metrics for edge-case evaluation

Required keys:
- `total_questions`
- `doc_hit_rate`
- `field_hit_rate`
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

### Per-model keys

- `model_id`
- `display_name`
- `runtime`
- `model_path`
- `n_ctx`
- `n_gpu_layers`
- `n_threads`
- `chat_format`
- `supports_gpu`
- `quantization`
- `recommended_batch_size`

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

## Generation behavior contract

The generation stage must:

- retrieve evidence before generation
- answer in Korean
- avoid inventing missing values
- expose which documents were cited
- explicitly say evidence is insufficient when grounding is weak
- keep retrieval-first as the default path
- support `gemma4_2b` and `gemma4_4b` through configuration, not separate scripts

The generation stage must not:

- guess unknown values
- use data outside repository inputs
- silently drop missing required columns

## Failure rules

Stop and report when:

- a required input file is missing
- required columns are missing
- no retrieval source rows are available
- model file is missing for `llama_cpp`
- backend and model configuration do not match

Continue with warning when:

- a question has no strong retrieval hit
- an answer is forced to use the fallback insufficient-context response
- the adapter returns an empty answer

## Validation rules

Minimum validation for this stage:

1. build source index successfully
2. run `mock` generation on `eval` and `edge`
3. run `llama_cpp` generation on `eval` for `gemma4_2b` and `gemma4_4b` when model files are available
4. produce model-specific metrics JSON files
5. confirm output files exist at documented paths
