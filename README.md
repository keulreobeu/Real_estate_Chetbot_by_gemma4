# Real_estate_Chetbot_by_gemma4

Gemma 4 기반 AI 부동산 챗봇 프로젝트를 위한 아파트 데이터 전처리, RAG 확장, QA 데이터 생성, 특이 질문 평가셋 생성을 정리한 저장소입니다.

현재 활성 단계는 `01_preprocessing`입니다.

이 프로젝트는 청약 아파트 원천 CSV를 입력으로 받아 다음 단계의 데이터셋을 생성합니다.

- 1차 전처리: 중복 제거, 정책 컬럼명 정리, 주소 분리
- 2차 전처리: 지하철역/호선 구조화, 역-호선 매핑, 아파트-역 연결 테이블 생성
- RAG 데이터 확장: 의료, 생활 인프라, 통근통학, 구조, 가격, 정책 요약 컬럼 생성
- QA 데이터 생성: 학습용 QA CSV, 파인튜닝용 JSONL, 평가용 샘플 생성
- Edge Case 질문 생성: 조건형, 비교형, 복합형, 지역형, 모호형, 구어체 질문 생성

## 프로젝트 구조

### 원본 데이터

- `data/original/apartment_20230905.csv`

### 전처리 스크립트

- `01_preprocessing/preprocess_apartment_csv.py`
  - 초기 cleaned CSV, QA base CSV, 전처리 보고서 생성
- `01_preprocessing/preprocess_apartment_pipeline.py`
  - 전체 파이프라인 버전
  - 주소 분리, 정책 컬럼 정리, 교통 정규화, RAG 컬럼 확장, QA base 데이터 생성
- `01_preprocessing/generate_apartment_qa_dataset.py`
  - `apartment_chatbot_v3.csv` 기반 5만 건 이상 QA 데이터셋 생성
- `01_preprocessing/generate_edge_questions.py`
  - 평가용 특이 질문 2000개 생성

### 생성 데이터

- `data/apartment_chatbot_cleaned.csv`
- `data/apartment_chatbot_v3.csv`
- `data/station_line_map.csv`
- `data/apartment_station_map.csv`

### QA 데이터

- `data/qa/apartment_chatbot_qa_base.csv`
- `data/qa/apartment_chatbot_qa_base_v3.csv`
- `data/qa/apartment_qa_dataset.csv`
- `data/qa/apartment_finetune_dataset.jsonl`
- `data/qa/evaluation_dataset.csv`
- `data/qa/edge_case_questions.csv`
- `data/qa/edge_case_eval.csv`

### 보고서 및 매핑 파일

- `data/apartment_column_mapping.csv`
- `00_Report/01_preprocessing_report.md`
- `00_Report/02_preprocessing_full_report.md`
- `00_Report/03_qa_generation_report.md`
- `00_Report/04_edge_question_report.md`
- `00_Report/05_preprocessing_log.txt`

## 데이터 파이프라인 요약

### 1. 초기 전처리

원본 CSV를 로드한 뒤 전체 컬럼, 결측치, 중복 여부를 분석합니다. 정책 관련 컬럼은 삭제하지 않고 아래와 같이 사람이 이해하기 쉬운 이름으로 변경합니다.

- `투기과열지구_before` → `분양당시_투기과열지구`
- `투기과열지구_after` → `현재_투기과열지구`
- `분양가상한제_before` → `분양당시_분양가상한제`
- `분양가상한제_after` → `현재_분양가상한제`

또한 `법정동주소`를 기준으로 `시도`, `시군구`, `동`, `상세주소`를 분리합니다.

### 2. 교통 데이터 정규화

지하철 관련 정보를 다음 구조로 확장합니다.

- `가장가까운역`
- `거리_m`
- `환승역여부`
- `호선수`
- `가장가까운역_호선요약`

추가로 다음 보조 테이블을 생성합니다.

- `station_line_map.csv`: 역명-호선 매핑
- `apartment_station_map.csv`: 아파트-역 연결 테이블

### 3. RAG 확장 컬럼 생성

검색 품질과 응답 근거 강화를 위해 다음 컬럼을 생성합니다.

- `의료시설_요약`
- `생활인프라_요약`
- `통근통학_요약`
- `구조요약`
- `면적대`
- `가격요약`
- `건설사_요약`
- `정책특이사항_설명`
- `검색키워드`
- `description`
- `문서ID`

### 4. QA 데이터 생성

`apartment_chatbot_v3.csv`를 기반으로 행당 다수의 질문 템플릿을 적용해 QA 데이터를 생성합니다.

- 출력 CSV: `apartment_qa_dataset.csv`
- 파인튜닝용 JSONL: `apartment_finetune_dataset.jsonl`
- 평가용 샘플: `evaluation_dataset.csv`

카테고리는 다음과 같습니다.

- `fact`
- `location`
- `transport`
- `price`
- `lifestyle`
- `policy`

### 5. Edge Case 질문 생성

모델 평가를 위해 일반 QA와 별도로 특이 질문 세트를 생성합니다.

- 출력 CSV: `edge_case_questions.csv`
- 평가 매핑: `edge_case_eval.csv`

질문 유형은 다음 6가지입니다.

- `condition`
- `comparison`
- `multi_condition`
- `region`
- `vague`
- `colloquial`

## 실행 방법

Python과 pandas가 설치된 환경에서 아래 순서로 실행하면 됩니다.

```powershell
python .\01_preprocessing\preprocess_apartment_csv.py
python .\01_preprocessing\preprocess_apartment_pipeline.py
python .\01_preprocessing\generate_apartment_qa_dataset.py
python .\01_preprocessing\generate_edge_questions.py
```

현재 작업 환경에서는 `C:\Users\lwwde\miniconda3\envs\py312\python.exe` 인터프리터로 실행했습니다.

예시:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\01_preprocessing\preprocess_apartment_pipeline.py
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\01_preprocessing\generate_apartment_qa_dataset.py
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\01_preprocessing\generate_edge_questions.py
```

## 주요 산출물 요약

- `data/apartment_chatbot_v3.csv`
  - 챗봇 RAG 검색용 메인 데이터셋
- `data/qa/apartment_chatbot_qa_base_v3.csv`
  - QA 생성용 핵심 필드 베이스
- `data/qa/apartment_qa_dataset.csv`
  - 자동 생성 QA 데이터셋
- `data/qa/apartment_finetune_dataset.jsonl`
  - instruction tuning용 데이터셋
- `data/qa/edge_case_questions.csv`
  - 평가용 특이 질문 세트

## 업로드 전 체크 포인트

- 원본 데이터와 생성 데이터가 모두 포함되어 저장소 용량이 커질 수 있습니다.
- 대용량 CSV/JSONL 파일을 계속 누적할 계획이면 Git LFS 사용도 검토할 수 있습니다.
- GitHub에 새 저장소를 만들 때 권장 이름은 `Real_estate_Chetbot_by_gemma4`입니다.

## 향후 확장 아이디어

- 질의응답 정답 근거 컬럼 추가
- 지역/가격대별 추천형 QA 강화
- 정책 변화 시점별 버전 데이터셋 관리
- 파인튜닝용 instruction 스타일 다양화
