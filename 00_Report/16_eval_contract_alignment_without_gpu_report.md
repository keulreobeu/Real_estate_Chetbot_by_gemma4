# 16. Eval Contract Alignment Without GPU Report

## purpose

Align the evaluation contract for MVP V2 without loading Gemma or using GPU resources.

## input files

- `data/qa/evaluation_dataset.csv`
- `data/eval/gemma4_generation_eval_predictions_gemma4_2b.csv`
- `02_gemma4_generation/query_service.py`
- `02_gemma4_generation/evaluate_generation_mvp.py`

## output files

- `data/eval/gemma4_generation_eval_predictions_gemma4_2b.csv`
- `data/eval/gemma4_generation_eval_metrics_gemma4_2b.json`
- `00_Report/16_eval_contract_alignment_without_gpu_report.md`

## original row and column counts

- previous prediction file state before rerun: 5 rows, 27 columns

## final row and column counts

- `data/eval/gemma4_generation_eval_predictions_gemma4_2b.csv`: 1000 rows, 27 columns

## removed exact duplicate count

- duplicate-removal work not performed in this run
- exact duplicate removal count: 0

## newly generated columns

- no schema change in this run

## processing steps summary

1. Investigated why `must_include_pass_rate` was `0.0` without using the model runtime.
2. Confirmed the failure cause:
   - evaluation rows required apartment name in `must_include`
   - deterministic `APARTMENT_FACT_LOOKUP` answers did not include apartment name
3. Updated [`query_service.py`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\query_service.py) so fact lookup answers start with `<아파트명> 정보입니다.`
4. Updated [`evaluate_generation_mvp.py`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\evaluate_generation_mvp.py) so `must_include` and `must_not_include` treat empty/NaN values correctly via `safe_text`.
5. Re-ran eval generation with `--backend mock` so only deterministic contract logic executed and no GPU/model load was required.
6. Re-ran eval metrics generation.

## validation method

- `python -m py_compile` on modified files
- deterministic sample check:
  - `demo_chatbot_mvp.py --backend mock --model gemma4_2b --question "e편한세상남양뉴타운 ... 병원 접근성은 어때"`
- full CPU-only eval regeneration:
  - `run_generation_mvp.py --mode eval --backend mock --model gemma4_2b`
- metrics refresh:
  - `evaluate_generation_mvp.py --mode eval --model gemma4_2b`

## major warnings or exceptions

- This run only aligned deterministic fact lookup contract and evaluation matching.
- `contains_expected_answer_rate` remains below 1.0 because current evaluation still compares concise expected answers against richer field-oriented deterministic responses.
- Retrieval hit rate remains unchanged because this run did not change retrieval ranking or expected document contracts.

## sample outputs

### deterministic fact lookup after alignment

Question:

`e편한세상남양뉴타운 전용 84.93㎡ 공급 114.90㎡ 타입 8 병원 접근성은 어때`

Observed answer prefix:

`e편한세상남양뉴타운 정보입니다.`

### updated eval metrics

- `total_questions`: 1000
- `answer_type_match_rate`: 1.0
- `match_status_match_rate`: 1.0
- `must_include_pass_rate`: 1.0
- `must_not_include_pass_rate`: 1.0
