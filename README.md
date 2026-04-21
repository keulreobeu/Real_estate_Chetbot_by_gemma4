# Gemma 4 기반 부동산 챗봇

## 프로젝트 개요

이 프로젝트는 아파트 원천 CSV 데이터를 기반으로 부동산 AI 챗봇을 만들기 위한 단계형 데이터 파이프라인입니다.

원천 데이터를 전처리해 RAG용 아파트 데이터셋을 만들고, 질문·답변 데이터셋, 평가 데이터셋, 엣지 케이스 검증 세트를 생성합니다. 이후 Gemma 4 기반 생성 실험, 추천 안전성 점검, SFT 데이터 준비, 파인튜닝 실행 및 학습 후 평가로 이어질 수 있도록 단계별 산출물과 데이터 계약을 정리합니다.

이 저장소는 모델 호출 코드만 두는 방식이 아니라, 데이터 구조, 검색 기반 응답 흐름, 추천 제한 규칙, 평가 기준, 실행 보고서를 함께 관리하는 방식으로 구성되어 있습니다.

## 현재 단계

현재 활성 단계:

- `06_finetuning`

현재 생성 실험 단계:

- `02_gemma4_generation`

## 서비스화 방향

현재 구조는 전처리, 검색 기반 응답, 평가, 파인튜닝 준비까지 단계별 계약과 산출물로 정리되어 있습니다.

다음 확장 방향은 이 파이프라인을 실제 서비스 형태로 검증할 수 있게 만드는 것입니다.

- API 기반 사용 경로
- 배포 가능한 데모 흐름 구성
- 요청/응답 로그와 추적 가능성 강화
- 리뷰어 또는 관리자 평가 루프 추가
- 운영자 관점의 장애 대응 절차 정리

확장 시에는 현재 단계 기반 데이터 계약을 유지하면서, 기존 질의 및 평가 로직 위에 서비스용 API, 로그, 실행 보고서, 배포 절차를 얹는 방향을 우선합니다.

## 현재 상태

지금까지 정리된 핵심 상태는 아래와 같습니다.

- 추천형 질문은 검색 적중과 조건 일치를 분리합니다.
- `NO_MATCH` 또는 `UNKNOWN`이면 임의 아파트 추천을 하지 않습니다.
- 질의는 최소 다음 타입으로 분기됩니다.
  - `RECOMMEND_STRUCTURED`
  - `RECOMMEND_COMPARATIVE`
  - `APARTMENT_FACT_LOOKUP`
  - `REAL_ESTATE_KNOWLEDGE`
  - `DATA_SCOPE_META`
- 일반 부동산 지식과 데이터셋 기반 사실 응답을 분리합니다.
- `01_preprocessing` 산출물에는 답변 계약용 컬럼과 비교용 파생 컬럼이 포함됩니다.
- `02_gemma4_generation`은 결정적 계약 경로를 먼저 적용하고, 필요한 경우에만 검색 우선 생성 경로로 내려갑니다.

최근 진행 사항:

- QA/평가 정합성 보정
  - `must_include`, `must_not_include` 계약을 deterministic 응답 포맷에 맞게 정리
- Transformers 기반 MVP V2 정리
  - 기본 백엔드를 `transformers` 기준으로 문서화
  - 기본 모델은 `gemma4_2b`
- 런타임 병목 완화 1차 반영
  - `device_map="auto"`를 기본값으로 사용
  - 시작 점검을 `load_only`와 전체 생성 점검으로 분리
  - `transformers` 런타임 타이밍/디버그 메타를 예측 CSV와 웹 응답에 노출
- `02_gemma4_generation` 단계 완료
  - 엣지 예측 `2000/2000`
  - 평가 예측 `1000/1000`
  - 단계 04 지표와 준비도 게이트 완료
- `05_finetuning_prep` 단계 부트스트랩 완료
  - `data/qa/finetuning_prep/`에 후보/제외/holdout/train-valid JSONL 생성
- `03_generation_optimization` 단계 준비 게이트 추가
  - 추천 안전성 / 라우팅 / 일치 상태 실패 버킷을 분석
  - 면적대(`소형`, `중형`, `중소형`, `중대형`, `대형`) 조건은 추천 대신 `UNKNOWN`으로 처리
- 현재 `06_finetuning` 진입 판정은 `GO`

주의:

- GPU 기반 실제 `transformers` 추론 검증은 현재 보류 상태입니다.
- 따라서 현재는 계약/문서/비GPU 검증까지 반영된 상태이며, GPU 재개 시 런타임 성능 스모크 테스트를 다시 수행해야 합니다.

## 저장소 구조

```text
00_Report/
  처리 보고서와 로그

01_preprocessing/
  전처리, RAG 준비, QA 생성, 엣지 질문 생성

02_gemma4_generation/
  Gemma 4 생성 MVP, 질의 라우팅, 평가

03_generation_optimization/
  파인튜닝 전 상위 최적화 게이트
  추천 안전성, 라우팅, 일치 상태 최적화 게이트

05_finetuning_prep/
  SFT 후보 분리, holdout 구성, 학습/검증 JSONL 준비

06_finetuning/
  준비도 GO 이후 활성 단계
  파인튜닝 계약, 운영 절차서, 점검표, 학습 후 평가 계획

data/original/
  원본 CSV

data/
  전처리된 메인 데이터셋, RAG 준비 산출물

data/qa/
  QA 기반 자료, QA 데이터셋, 평가 데이터셋, 엣지 데이터셋, 지식 기반 자료

data/qa/finetuning_prep/
  단계 05 후보/제외/홀드아웃/학습-검증 산출물

data/eval/
  생성 소스 인덱스, 예측 CSV, 지표 JSON

data/eval/generation_optimization/
  단계 03 실패 버킷, 고난도 부정 예시 검토 큐, 재생성 계획
```

## 주요 산출물

핵심 산출물:

- `data/apartment_chatbot_v3.csv`
  - 메인 RAG/답변용 아파트 데이터셋
- `data/qa/evaluation_dataset.csv`
  - 생성 평가 질문 세트
- `data/qa/edge_case_eval.csv`
  - 엣지/제한 응답 검증용 질문 세트
  - `RECOMMEND_STRUCTURED`, `RECOMMEND_COMPARATIVE`, `GENERAL_RETRIEVAL_QA` 기대 라우터를 함께 포함
- `data/qa/real_estate_knowledge_base.csv`
  - 일반 부동산 지식 사전
- `data/eval/gemma4_generation_source_index.csv`
  - 생성 검색 소스 인덱스

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

## 파이프라인 개요

### 1. 전처리

`01_preprocessing`에서는 원본 아파트 CSV를 읽고 아래를 수행합니다.

- 주소 분리
- 정책 필드 정규화
- 교통 정보 구조화
- RAG용 요약/설명/검색키워드 생성
- 추천/비교/지식 응답용 파생 컬럼 생성
- QA / 평가 / 엣지 질문 세트 생성

주요 스크립트:

- [`preprocess_apartment_pipeline.py`](G:\GitProjects\New_Local_GPT_Chetbot\01_preprocessing\preprocess_apartment_pipeline.py)
- [`generate_apartment_qa_dataset.py`](G:\GitProjects\New_Local_GPT_Chetbot\01_preprocessing\generate_apartment_qa_dataset.py)
- [`generate_edge_questions.py`](G:\GitProjects\New_Local_GPT_Chetbot\01_preprocessing\generate_edge_questions.py)

### 2. 생성 MVP

`02_gemma4_generation`은 검색 우선 구조를 사용합니다.

흐름:

1. 소스 인덱스 생성
2. 질의 라우팅
3. 결정적 계약 평가
4. 필요 시 근거 기반 생성
5. 예측/평가 저장

핵심 정책:

- 추천형 질문은 결정적 계약 경로 우선
- `NO_MATCH`와 `UNKNOWN`이면 추천 차단
- 일반 지식은 지식 기반 자료 기반
- 자유 질의만 생성 경로 사용

관련 파일:

- [`query_service.py`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\query_service.py)
- [`run_generation_mvp.py`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\run_generation_mvp.py)
- [`demo_chatbot_mvp.py`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\demo_chatbot_mvp.py)
- [`demo_chatbot_web_mvp.py`](G:\GitProjects\New_Local_GPT_Chetbot\02_gemma4_generation\demo_chatbot_web_mvp.py)

## 런타임 메모

현재 문서화된 기본 런타임 정책은 아래와 같습니다.

- 백엔드: `transformers`
- model: `gemma4_2b`
- `device_map="auto"`
- 시작 점검 기본값: `load_only`
- `--startup-check-full`일 때만 짧은 생성 점검 수행

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

## Docker 환경

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
- GPU 런타임이 필요하면 NVIDIA Container Toolkit 기준의 추가 설정이 필요합니다.

## 공통 명령어

### 전처리

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\01_preprocessing\preprocess_apartment_pipeline.py
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\01_preprocessing\generate_apartment_qa_dataset.py
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\01_preprocessing\generate_edge_questions.py
```

### 생성 산출물 생성

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\build_generation_assets.py
```

### 파인튜닝 데이터셋 준비

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\05_finetuning_prep\prepare_sft_dataset.py --model gemma4_2b
```

### 파인튜닝 실행 명세 준비

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\create_run_manifest.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b --context-mode auto
```

### 무인 파인튜닝 실행 준비

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\prepare_unattended_finetuning_run.py --run-id baseline-gemma4-2b-r5 --model gemma4_2b --context-mode contextual --training-scope gates_and_norms --max-seq-length 512
```

이 사전 점검 명령은 장시간 무인 실행 전에 준비도를 갱신하고, 고정된 JSONL 계약을 검증하며,
선택적인 실행별 맥락 산출물을 준비하고, `manifest.json`을 생성합니다.
또한 로컬 환경을 점검하고 실행 명령과 번호가 붙은 보고서를 작성합니다.

단계 06 실행 산출물은 프로젝트 내부에만 둡니다.
실행 명세 또는 무인 사전 점검 도구를 사용할 때 `--run-dir`와 `--output-dir`는 이 저장소 내부 경로로 유지하세요.

### 베이스라인 파인튜닝 작업 실행

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\train_finetuning_baseline.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b --max-seq-length 512 --training-scope gates_and_norms
```

### 학습 후 예측 세트 생성

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\generate_post_train_prediction_sets.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b
```

### 완료된 파인튜닝 실행 평가

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\evaluate_post_finetuning_run.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b
```

### 생성 최적화 차단 요인 분석

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\03_generation_optimization\analyze_edge_failures.py --model gemma4_2b
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\05_finetuning_prep\validate_06_readiness.py --model gemma4_2b
```

### 야간 단계 05에서 06까지의 파이프라인

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run_overnight_stage05_to_06.ps1
```

이 작업자는 중단된 스테이징 엣지 재생성을 재개하고, 검증된 스테이징 산출물을 표준 산출물로 승격합니다.
또한 단계 04 지표를 다시 실행하고, 단계 05 파인튜닝 준비 산출물을 재생성하며,
단계 06 준비도 판정을 갱신합니다.

### CLI 데모

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\demo_chatbot_mvp.py --backend mock --model gemma4_2b
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\demo_chatbot_mvp.py --backend transformers --model gemma4_2b --question "용적률이 뭐야"
```

### 웹 데모

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\demo_chatbot_web_mvp.py --backend mock --model gemma4_2b --host 127.0.0.1 --port 8787
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\demo_chatbot_web_mvp.py --backend transformers --model gemma4_2b --host 127.0.0.1 --port 8787
```

### FastAPI 서비스

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\fastapi_app.py --backend mock --model gemma4_2b --host 127.0.0.1 --port 8788
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\fastapi_app.py --backend transformers --model gemma4_2b --host 127.0.0.1 --port 8788
```

- FastAPI는 현재 근거 기반 QA 런타임의 서비스용 API 경로입니다.
- 공개 v1 엔드포인트는 `GET /api/status`, `POST /api/ask`, `POST /api/check-rule`, `POST /api/check-generation-ready`입니다.
- JSONL 요청 로그는 `logs/api_requests/fastapi_YYYYMMDD.jsonl`에 기록됩니다.
- 기존 웹 데모는 별도 로컬 UI 경로로 계속 사용할 수 있습니다.

- 웹 데모 상단 카드에서 데이터 포함 지역(`서울권`, `경기권`, `인천권`)과 포함된 `시군구`를 확인할 수 있습니다.
- `규칙기반 답변 확인`은 고정 질문 `데이터 기준 알려줘`로 빠른 계약 기반 응답 경로를 점검합니다.
- `Gemma 생성 가능 상태 확인`은 장문 답변 대신 짧은 준비도 점검만 실행합니다.
- 중복 실행으로 포트 충돌이나 모델 중복 로딩이 생기지 않도록, 기존 서버가 실행 중인지 먼저 확인하고 같은 포트에는 한 서버만 유지하세요.

### 평가 실행

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\run_generation_mvp.py --mode eval --backend mock --model gemma4_2b
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\02_gemma4_generation\evaluate_generation_mvp.py --mode eval --model gemma4_2b
```

## 최근 보고서

최근 관련 보고서:

- [`15_transformers_mvp_v2_runtime_update_report.md`](G:\GitProjects\New_Local_GPT_Chetbot\00_Report\15_transformers_mvp_v2_runtime_update_report.md)
- [`16_eval_contract_alignment_without_gpu_report.md`](G:\GitProjects\New_Local_GPT_Chetbot\00_Report\16_eval_contract_alignment_without_gpu_report.md)
- [`17_runtime_probe_split_and_cpu_only_alignment_report.md`](G:\GitProjects\New_Local_GPT_Chetbot\00_Report\17_runtime_probe_split_and_cpu_only_alignment_report.md)

## 다음 권장 단계

GPU 작업이 재개되면 아래 순서가 권장됩니다.

1. `transformers + gemma4_2b + device_map=auto` 시작 점검 재확인
2. 웜업 웹 요청 2회 이상으로 로드/생성 시간 비교
3. `GENERAL_RETRIEVAL_QA` 실제 생성 지연 시간 점검
4. 필요 시 양자화 또는 `llama_cpp` 대체 경로 검토
