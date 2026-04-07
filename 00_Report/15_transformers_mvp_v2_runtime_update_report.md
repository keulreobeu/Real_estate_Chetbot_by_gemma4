# 15. Transformers MVP V2 Runtime Update Report

## purpose

Implement the MVP V2 runtime plan for [`02_gemma4_generation`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation) so the default demo path uses `transformers + gemma4_2b` while preserving the existing recommendation safety contract.

## input files

- `data/apartment_chatbot_v3.csv`
- `data/qa/real_estate_knowledge_base.csv`
- `data/eval/gemma4_generation_source_index.csv`
- `02_gemma4_generation/config/generation_defaults.json`
- `02_gemma4_generation/config/models.local.json`

## output files

- `data/eval/gemma4_generation_eval_predictions_gemma4_2b.csv` (5-row smoke output refreshed)
- `data/eval/gemma4_generation_eval_metrics_gemma4_2b.json`
- `00_Report/15_transformers_mvp_v2_runtime_update_report.md`

## original row and column counts

- `data/apartment_chatbot_v3.csv`: 3239 rows, 87 columns
- `data/eval/gemma4_generation_source_index.csv`: 3239 rows, 21 columns

## final row and column counts

- `data/apartment_chatbot_v3.csv`: 3239 rows, 87 columns
- `data/eval/gemma4_generation_source_index.csv`: 3239 rows, 21 columns
- `data/eval/gemma4_generation_eval_predictions_gemma4_2b.csv`: 5 rows, 27 columns

## removed exact duplicate count

- Dataset rewrite not performed in this run
- Exact duplicate removal count: 0

## newly generated columns

No preprocessing schema change was made in this run.

New runtime response fields exposed by the CLI/web demo layer:

- `answer_type`
- `match_status`
- `query_type`
- `used_fields`
- `data_cutoff`
- `limitations`

## processing steps summary

1. Changed runtime defaults in [`generation_defaults.json`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\config\generation_defaults.json) to V2-oriented values:
   - backend `transformers`
   - model via existing default `gemma4_2b`
   - `max_output_tokens=96`
   - `temperature=0.0`
   - `top_p=1.0`
   - `repeat_penalty=1.05`
   - request/web timeout settings added
2. Switched [`common.py`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\common.py) default backend constant to `transformers`.
3. Rebuilt [`query_service.py`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\query_service.py) with clean deterministic routing and Korean field handling.
4. Added apartment-level deduplication for structured and comparative recommendation answers.
5. Exposed contract metadata and safety metadata in:
   - [`demo_chatbot_mvp.py`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\demo_chatbot_mvp.py)
   - [`demo_chatbot_web_mvp.py`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\demo_chatbot_web_mvp.py)
6. Updated stage docs:
   - [`README.md`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\README.md)
   - [`CONTRACT.md`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\CONTRACT.md)

## validation method

- `python -m py_compile` on modified runtime files
- `verify_local_inference_setup.py --model gemma4_2b`
- `demo_chatbot_mvp.py --backend transformers --model gemma4_2b --question ...`
- temporary local web startup + `Invoke-WebRequest` root/API check
- `run_generation_mvp.py --mode eval --backend transformers --model gemma4_2b --limit 5`
- `evaluate_generation_mvp.py --mode eval --model gemma4_2b`

## major warnings or exceptions

- The 5-row `run_generation_mvp.py` smoke run saved output successfully but exceeded the shell timeout because the transformers startup probe was very slow in this environment.
- The startup probe printed `latency_ms=1218682`, so long-run docs should continue to mention `--no-startup-check` as a deliberate recovery option after setup verification.
- Root `README.md` was not safely rewritten in this run because the file is currently stored/displayed with mojibake-like encoding in the shell view. Stage-local docs were updated instead.

## sample outputs

### no-match structured recommendation

Question:

`서울에서 1억 이하이면서 지하철 100m 이내 아파트 추천해줘`

Observed contract:

- `answer_type=no_match_response`
- `match_status=NO_MATCH`
- `query_type=RECOMMEND_STRUCTURED`
- `cited_doc_ids=[]`

### supported comparative recommendation

Question:

`공원 가까운 아파트 비교해줘`

Observed contract:

- `answer_type=comparison_recommendation`
- `match_status=EXACT_MATCH`
- `query_type=RECOMMEND_COMPARATIVE`
- top 3 apartments were unique by apartment name

### unsupported comparative recommendation

Question:

`아이 키우기 좋은 아파트 추천해줘`

Observed contract:

- `answer_type=unsupported_comparative_response`
- `match_status=UNKNOWN`
- `query_type=RECOMMEND_COMPARATIVE`
- response includes limitation guidance and example prompts
