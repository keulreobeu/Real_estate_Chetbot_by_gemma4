# Real Estate Chatbot By Gemma 4

부동산 AI 챗봇을 위한 stage 기반 데이터 파이프라인 저장소입니다.  
현재 저장소는 아파트 원천 CSV를 전처리하고, RAG용 데이터셋과 QA/eval/edge 질문 세트를 만들며, 이후 Gemma 4 기반 생성·평가 단계로 이어질 수 있도록 계약을 정리하는 데 초점을 두고 있습니다.

현재 활성 stage:

- `01_preprocessing`

현재 generation 실험 stage:

- `02_gemma4_generation`

## Current Status

지금까지 정리된 핵심 상태는 아래와 같습니다.

- 추천형 질문은 검색 hit와 조건 match를 분리합니다.
- `NO_MATCH` 또는 `UNKNOWN`이면 임의 아파트 추천을 하지 않습니다.
- 질의는 최소 다음 타입으로 분기됩니다.
  - `RECOMMEND_STRUCTURED`
  - `RECOMMEND_COMPARATIVE`
  - `APARTMENT_FACT_LOOKUP`
  - `REAL_ESTATE_KNOWLEDGE`
  - `DATA_SCOPE_META`
- 일반 부동산 지식과 데이터셋 기반 사실 응답을 분리합니다.
- `01_preprocessing` 산출물에는 답변 계약용 컬럼과 비교용 파생 컬럼이 포함됩니다.
- `02_gemma4_generation`은 deterministic contract path를 먼저 적용하고, 필요한 경우에만 retrieval-first generation으로 내려갑니다.

최근 진행 사항:

- QA/eval 정합성 보정
  - `must_include`, `must_not_include` 계약을 deterministic 응답 포맷에 맞게 정리
- Transformers 기반 MVP V2 정리
  - 기본 backend를 `transformers` 기준으로 문서화
  - 기본 모델은 `gemma4_2b`
- runtime 병목 완화 1차 반영
  - `device_map="auto"`를 기본값으로 사용
  - startup check를 `load_only`와 `full generation probe`로 분리
  - `transformers` runtime timing/debug 메타를 예측 CSV와 웹 응답에 노출

주의:

- GPU 기반 실제 `transformers` 추론 검증은 현재 보류 상태입니다.
- 따라서 현재는 계약/문서/비GPU 검증까지 반영된 상태이며, GPU 재개 시 runtime 성능 smoke test를 다시 수행해야 합니다.

## Repository Structure

```text
00_Report/
  처리 보고서와 로그

01_preprocessing/
  현재 활성 stage
  전처리, RAG 준비, QA 생성, edge 질문 생성

02_gemma4_generation/
  Gemma 4 generation MVP, query routing, evaluation

data/original/
  원본 CSV

data/
  전처리된 메인 데이터셋, RAG 준비 산출물

data/qa/
  QA base, QA dataset, evaluation, edge dataset, knowledge base

data/eval/
  generation source index, prediction CSV, metrics JSON
```

## Main Artifacts

핵심 산출물:

- `data/apartment_chatbot_v3.csv`
  - 메인 RAG/답변용 아파트 데이터셋
- `data/qa/evaluation_dataset.csv`
  - generation eval 질문 세트
- `data/qa/edge_case_eval.csv`
  - edge/제한 응답 검증용 질문 세트
- `data/qa/real_estate_knowledge_base.csv`
  - 일반 부동산 지식 사전
- `data/eval/gemma4_generation_source_index.csv`
  - generation retrieval source index

`apartment_chatbot_v3.csv`에는 현재 아래 성격의 컬럼이 포함됩니다.

- 기본 단지 정보
  - `문서ID`, `아파트명`, `시도`, `시군구`, `동`
- 가격/면적 정보
  - `전용면적`, `공급면적`, `공급액(만원)`, `평당_공급액`
- 교통 정보
  - `가장가까운역`, `거리_m`, `환승역여부`, `호선수`, `가장가까운역_호선요약`
- 계약/답변 정보
  - `데이터기준일`, `추천가능여부`, `확인불가항목`, `질의매칭태그`, `답변가능범위`, `source_flags`
- 비교/추천용 파생 필드
  - `공원_접근지표`, `병원_접근지표`, `학교_접근지표`, `지하철_접근지표`
  - `공원_비교요약`, `병원_비교요약`, `학교_비교요약`, `교통_비교요약`

## Pipeline Overview

### 1. Preprocessing

`01_preprocessing`에서는 원본 아파트 CSV를 읽고 아래를 수행합니다.

- 주소 분리
- 정책 필드 정규화
- 교통 정보 구조화
- RAG용 summary/description/검색키워드 생성
- 추천/비교/지식 응답용 파생 컬럼 생성
- QA / evaluation / edge 질문 세트 생성

주요 스크립트:

- [`preprocess_apartment_pipeline.py`](G:\GitProjects\New_Local_GPT_Chetbot\01_preprocessing\preprocess_apartment_pipeline.py)
- [`generate_apartment_qa_dataset.py`](G:\GitProjects\New_Local_GPT_Chetbot\01_preprocessing\generate_apartment_qa_dataset.py)
- [`generate_edge_questions.py`](G:\GitProjects\New_Local_GPT_Chetbot\01_preprocessing\generate_edge_questions.py)

### 2. Generation MVP

`02_gemma4_generation`은 retrieval-first 구조를 사용합니다.

흐름:

1. source index build
2. query routing
3. deterministic contract evaluation
4. 필요 시 grounded generation
5. prediction/eval 저장

핵심 정책:

- 추천형 질문은 deterministic contract path 우선
- `NO_MATCH`와 `UNKNOWN`이면 추천 차단
- 일반 지식은 knowledge base 기반
- 자유 질의만 generation path 사용

관련 파일:

- [`query_service.py`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\query_service.py)
- [`run_generation_mvp.py`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\run_generation_mvp.py)
- [`demo_chatbot_mvp.py`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\demo_chatbot_mvp.py)
- [`demo_chatbot_web_mvp.py`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\demo_chatbot_web_mvp.py)

## Runtime Notes

현재 문서화된 기본 runtime 정책:

- backend: `transformers`
- model: `gemma4_2b`
- `device_map="auto"`
- startup probe 기본값: `load_only`
- `--startup-check-full`일 때만 짧은 generation probe 수행

예측 CSV와 web demo는 아래 런타임 메타를 다룰 수 있습니다.

- `load_runtime_ms`
- `processor_load_ms`
- `model_load_ms`
- `prompt_render_ms`
- `input_prepare_ms`
- `to_device_ms`
- `generate_ms`
- `decode_ms`
- `device_map`
- `model_source`
- `last_load_ms`
- `last_generate_ms`

## Common Commands

### Preprocessing

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\01_preprocessing\preprocess_apartment_pipeline.py
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\01_preprocessing\generate_apartment_qa_dataset.py
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\01_preprocessing\generate_edge_questions.py
```

### Build generation assets

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\build_generation_assets.py
```

### CLI demo

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\demo_chatbot_mvp.py --backend mock --model gemma4_2b
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\demo_chatbot_mvp.py --backend transformers --model gemma4_2b --question "용적률이 뭐야"
```

### Web demo

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\demo_chatbot_web_mvp.py --backend mock --model gemma4_2b --host 127.0.0.1 --port 8787
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\demo_chatbot_web_mvp.py --backend transformers --model gemma4_2b --host 127.0.0.1 --port 8787
```

### Eval run

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\run_generation_mvp.py --mode eval --backend mock --model gemma4_2b
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\evaluate_generation_mvp.py --mode eval --model gemma4_2b
```

## Recent Reports

최근 관련 보고서:

- [`15_transformers_mvp_v2_runtime_update_report.md`](G:\GitProjects\New_Local_GPT_Chetbot\00_Report\15_transformers_mvp_v2_runtime_update_report.md)
- [`16_eval_contract_alignment_without_gpu_report.md`](G:\GitProjects\New_Local_GPT_Chetbot\00_Report\16_eval_contract_alignment_without_gpu_report.md)
- [`17_runtime_probe_split_and_cpu_only_alignment_report.md`](G:\GitProjects\New_Local_GPT_Chetbot\00_Report\17_runtime_probe_split_and_cpu_only_alignment_report.md)

## Next Recommended Steps

GPU 작업이 재개되면 아래 순서가 권장됩니다.

1. `transformers + gemma4_2b + device_map=auto` startup probe 재확인
2. warm web request 2회 이상으로 load/generate timing 비교
3. `GENERAL_RETRIEVAL_QA` 실제 generation latency 점검
4. 필요 시 양자화 또는 `llama_cpp` fallback 검토
