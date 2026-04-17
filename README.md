# Real Estate Chatbot By Gemma 4

부동산 AI 챗봇을 위한 stage 기반 데이터 파이프라인 저장소입니다.  
현재 저장소는 아파트 원천 CSV를 전처리하고, RAG용 데이터셋과 QA/eval/edge 질문 세트를 만들며, 이후 Gemma 4 기반 생성·평가 단계로 이어질 수 있도록 계약을 정리하는 데 초점을 두고 있습니다.

현재 활성 stage:

- `06_finetuning`

현재 generation 실험 stage:

- `02_gemma4_generation`

## Serviceization Direction

This repository should be readable not only as a data pipeline project, but also as an
`AI Application Engineer / AI Service Engineer` portfolio project.

The current strengths are already clear:

- data collection and preprocessing contracts
- retrieval-first real estate QA structure
- evaluation and edge-case coverage
- finetuning readiness and post-train comparison flow

The main gap is not another modeling stage. The main gap is product evidence:

- API-facing usage path
- deployable demo flow
- logging and traceability
- reviewer or admin evaluation loop
- operator-facing runbooks for failure handling

When extending this repository, prefer changes that preserve the current stage-based
data contracts while making the system easier to present as a service:

- keep the current preprocessing, generation, optimization, and finetuning stages
- add service-facing wrappers around existing query and evaluation logic
- make logs, reports, and run artifacts easy to inspect
- document request and response contracts plus deployment steps
- treat post-train validation as both model evaluation and service-readiness evidence

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
- stage `02_gemma4_generation` 완료
  - edge predictions `2000/2000`
  - eval predictions `1000/1000`
  - stage 04 metrics와 readiness gate 완료
- stage `05_finetuning_prep` 부트스트랩 완료
  - `data/qa/finetuning_prep/`에 candidate/rejected/holdout/train-valid JSONL 생성
- stage `03_generation_optimization` 준비 게이트 추가
  - recommendation safety / routing / match-status 실패 버킷을 분석
  - 면적대(`소형`, `중형`, `중소형`, `중대형`, `대형`) 조건은 추천 대신 `UNKNOWN`으로 처리
- 현재 `06_finetuning` 진입 판정은 `GO`

주의:

- GPU 기반 실제 `transformers` 추론 검증은 현재 보류 상태입니다.
- 따라서 현재는 계약/문서/비GPU 검증까지 반영된 상태이며, GPU 재개 시 runtime 성능 smoke test를 다시 수행해야 합니다.

## Repository Structure

```text
00_Report/
  처리 보고서와 로그

01_preprocessing/
  전처리, RAG 준비, QA 생성, edge 질문 생성

02_gemma4_generation/
  Gemma 4 generation MVP, query routing, evaluation

03_generation_optimization/
  upstream optimization gate before finetuning
  recommendation safety, routing, match-status optimization gate

05_finetuning_prep/
  SFT 후보 분리, holdout 구성, train/valid JSONL 준비

06_finetuning/
  active stage after readiness GO
  finetuning contract, runbook, checklist, and post-train evaluation plan

data/original/
  원본 CSV

data/
  전처리된 메인 데이터셋, RAG 준비 산출물

data/qa/
  QA base, QA dataset, evaluation, edge dataset, knowledge base

data/qa/finetuning_prep/
  stage 05 candidate/rejected/holdout/train-valid artifacts

data/eval/
  generation source index, prediction CSV, metrics JSON

data/eval/generation_optimization/
  stage 03 failure buckets, hard-negative review queue, regeneration plan
```

## Main Artifacts

핵심 산출물:

- `data/apartment_chatbot_v3.csv`
  - 메인 RAG/답변용 아파트 데이터셋
- `data/qa/evaluation_dataset.csv`
  - generation eval 질문 세트
- `data/qa/edge_case_eval.csv`
  - edge/제한 응답 검증용 질문 세트
  - `RECOMMEND_STRUCTURED`, `RECOMMEND_COMPARATIVE`, `GENERAL_RETRIEVAL_QA` 기대 라우터를 함께 포함
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

## Docker Environment

현재 작업 폴더를 그대로 마운트하는 재현 가능한 Docker 개발 환경을 추가했습니다.

- 기본 범위:
  - `01_preprocessing`
  - `02_gemma4_generation`의 `transformers` 기반 경로
- 기본 이미지 제외 항목:
  - `llama_cpp`
  - 로컬 GGUF 모델 파일

빌드:

```powershell
docker compose build
```

셸 진입:

```powershell
docker compose run --rm chatbot-dev
```

컨테이너 내부 실행 예시:

```bash
python ./01_preprocessing/preprocess_apartment_pipeline.py
python ./01_preprocessing/generate_apartment_qa_dataset.py
python ./01_preprocessing/generate_edge_questions.py
python ./02_gemma4_generation/build_generation_assets.py
python ./02_gemma4_generation/verify_local_inference_setup.py
```

메모:

- `data/`와 `00_Report/`는 이미지에 복사하지 않고 워크스페이스 볼륨으로 사용합니다.
- Hugging Face 캐시는 `huggingface-cache` Docker 볼륨에 저장됩니다.
- GPU runtime이 필요하면 NVIDIA Container Toolkit 기준의 추가 설정이 필요합니다.

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

### Prepare finetuning datasets

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\05_finetuning_prep\prepare_sft_dataset.py --model gemma4_2b
```

### Prepare a finetuning run manifest

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\create_run_manifest.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b --context-mode auto
```

### Prepare an unattended finetuning run

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\prepare_unattended_finetuning_run.py --run-id baseline-gemma4-2b-r5 --model gemma4_2b --context-mode contextual --training-scope gates_and_norms --max-seq-length 512
```

This preflight command refreshes readiness, validates the frozen JSONL contract, prepares
optional run-local contextual assets, creates `manifest.json`, checks the local environment,
and writes launch commands plus a numbered report before the long unattended run.

Stage 06 run outputs are project-local only. Keep `--run-dir` and `--output-dir` inside this
repository when using the manifest or unattended preflight tooling.

### Run the baseline finetuning job

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\train_finetuning_baseline.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b --max-seq-length 512 --training-scope gates_and_norms
```

### Generate post-train prediction sets

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\generate_post_train_prediction_sets.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b
```

### Evaluate a completed finetuning run

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\evaluate_post_finetuning_run.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b
```

### Analyze generation optimization blockers

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\03_generation_optimization\analyze_edge_failures.py --model gemma4_2b
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\05_finetuning_prep\validate_06_readiness.py --model gemma4_2b
```

### Overnight stage 05 to 06 pipeline

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run_overnight_stage05_to_06.ps1
```

This worker resumes the interrupted staging edge regeneration, promotes the validated staging output to canonical,
re-runs stage 04 metrics, regenerates stage 05 finetuning prep outputs, and refreshes the stage 06 readiness verdict.

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

### FastAPI service

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\fastapi_app.py --backend mock --model gemma4_2b --host 127.0.0.1 --port 8788
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\fastapi_app.py --backend transformers --model gemma4_2b --host 127.0.0.1 --port 8788
```

- FastAPI is the service-facing API path for the current grounded QA runtime.
- Public v1 endpoints are `GET /api/status`, `POST /api/ask`, `POST /api/check-rule`, and `POST /api/check-generation-ready`.
- JSONL request logs are written to `logs/api_requests/fastapi_YYYYMMDD.jsonl`.
- The existing web demo remains available as a separate local UI path.

- Web demo 상단 카드에서 데이터 포함 지역(`서울권`, `경기권`, `인천권`)과 포함된 `시군구`를 확인할 수 있습니다.
- `규칙기반 답변 확인`은 고정 질문 `데이터 기준 알려줘`로 빠른 계약 기반 응답 경로를 점검합니다.
- `Gemma 생성 가능 상태 확인`은 장문 답변 대신 짧은 readiness probe만 실행합니다.
- 중복 실행으로 포트 충돌이나 모델 중복 로딩이 생기지 않도록, 기존 서버가 실행 중인지 먼저 확인하고 같은 포트에는 한 서버만 유지하세요.

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
