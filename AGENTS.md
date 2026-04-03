# AGENTS.md

## 1. Project purpose

이 저장소는 부동산 AI 챗봇용 데이터 파이프라인 프로젝트다.  
주요 목적은 원본 아파트 CSV를 전처리하고, 교통 데이터를 정규화하며, RAG 검색용 메인 데이터셋과 QA 및 평가 데이터셋을 생성하는 것이다.

이 문서는 사람이 읽는 안내문이 아니라 Codex와 에이전트가 그대로 따르는 작업 표준서다.  
모호한 판단보다 명시된 입력, 출력, 실패 조건, 보고 규칙을 우선하라.

## 2. Global working rules

- 원본 raw CSV는 절대 수정하지 마라.
- 원본 파일은 항상 `data/original` 아래에 둬라.
- 결과 파일은 항상 새 이름 또는 명시된 버전 파일명으로 저장하라.
- 사용자의 명시적 지시가 없으면 기존 결과 파일을 덮어쓰지 마라.
- 모든 major run은 결과 파일과 markdown report를 함께 생성하라.
- 필요한 경우 processing log를 남겨라.
- 로그와 리포트는 `00_Report` 아래에 저장하라.
- 산출물은 지정된 폴더에만 저장하라. 루트에 임시 CSV를 만들지 마라.
- 인코딩은 기본적으로 `utf-8-sig`를 사용하라.
- CSV는 헤더를 포함해 저장하라.
- 결과가 비정상적으로 감소하거나 필수 컬럼이 없으면 조용히 진행하지 말고 중단 또는 warning 규칙을 따르라.

## 3. Standard folder rules

- 이 저장소는 기능 중심이 아니라 단계 중심으로 폴더를 정리하라.
- 폴더 번호는 버전이 아니라 작업 단계 순서를 의미하게 유지하라.
- 단계 폴더는 `NN_name` 형식을 사용하라.
- `NN`은 증가하는 번호를 사용하라.
- 이미 사용한 번호는 재사용하지 마라.
- 폴더명은 짧고 명확한 역할명으로 작성하라.
- `pipeline`, `new`, `test`, `final` 같은 모호한 이름은 사용하지 마라.

현재 기준 구조는 다음을 따른다.

- `00_Report`
  - 실행 리포트와 로그
- `01_preprocessing`
  - 현재 활성 단계
  - 원본 CSV 전처리, RAG 데이터 생성, QA 생성, edge question 생성 스크립트
- `data/original`
  - 원본 CSV
- `data`
  - 메인 전처리 결과와 RAG 메인 CSV
- `data/qa`
  - QA base, QA dataset, finetune JSONL, evaluation, edge question 관련 파일

새 폴더가 필요하면 아래 공용 폴더를 우선 고려하라.

- `shared/`
  - 두 개 이상 단계에서 공통으로 사용하는 유틸
- `common/`
  - 재사용 로직
- `config/`
  - 공통 설정
- `data/mapping`
  - 컬럼 매핑, 역-호선 매핑 같은 참조 테이블
- `data/eval`
  - 평가 전용 산출물
- `logs`
  - 장기 보관용 실행 로그

## 4. Folder stage rules

- 새 단계 폴더는 아래 조건 중 하나를 만족할 때만 생성하라.
  - 사용 모델이 바뀐다.
  - 작업 목적이 바뀐다.
  - 필수 입력 파일 종류가 바뀐다.
  - 필수 출력 파일 종류가 바뀐다.
  - 필수 컬럼 또는 필드가 추가, 삭제, 이름 변경된다.
  - 후속 스크립트 수정이 필요한 입출력 구조 변경이 발생한다.
  - 학습, 추론, 평가처럼 작업 성격이 바뀐다.
- 같은 목적 안에서의 소규모 수정, 리팩터링, 템플릿 추가, 경미한 로직 개선만 있는 경우에는 새 단계 폴더를 만들지 말고 기존 단계 폴더를 유지하라.
- 단계 폴더는 단계 기준으로만 나눠라.
- 기능 기준과 단계 기준을 섞지 마라.
- 전처리, RAG 생성, 추론, 평가처럼 단계 흐름이 바뀔 때만 폴더를 나눠라.
- 하나의 단계 폴더는 하나의 주요 목적만 갖게 하라.
- 단계 폴더의 역할을 나중에 다른 의미로 바꾸지 마라.
- 단계 폴더에는 해당 단계에 특화된 코드와 설정만 둬라.
- 두 개 이상 단계에서 공통으로 사용하는 유틸, 템플릿, 설정은 단계 폴더에 복사하지 말고 공용 디렉토리로 분리하라.
- 현재 기본 실행 대상 단계는 `README.md`와 `AGENTS.md`에 명시하라.
- 여러 단계가 공존하더라도 기본 기준이 되는 활성 단계는 하나로 정하라.
- 더 이상 사용하지 않는 단계 폴더는 삭제하지 말고 `deprecated` 또는 `archived` 상태로 문서에 표시하라.
- 폐기 단계는 폐기 사유와 대체 단계를 함께 기록하라.
- 새 단계 폴더를 만들면 `README.md`와 `AGENTS.md`의 폴더 구조 설명을 함께 갱신하라.
- 각 단계는 최소한 다음 정보를 문서에 남겨라.
  - 단계 목적
  - 입력 파일
  - 출력 파일
  - 실행 스크립트
  - 주요 설정 파일
  - 의존 관계
  - 후속 단계 연결 방식

권장 예:

- `01_preprocessing`
- `02_rag_build`
- `03_gemma4_generation`
- `04_evaluation`

## 5. Major runs

다음 실행 단위를 major run으로 간주하라.

### Run: preprocess_apartment_csv

- 목적: 원본 CSV의 1차 정제, 주소 분리, 정책 컬럼 정규화, cleaned CSV 생성
- 기본 입력: `data/original/apartment_20230905.csv`
- 기본 출력:
  - `data/apartment_chatbot_cleaned.csv`
  - `data/qa/apartment_chatbot_qa_base.csv`
  - `data/apartment_column_mapping.csv`
  - `00_Report/01_preprocessing_report.md`
  - `00_Report/05_preprocessing_log.txt`

### Run: preprocess_apartment_pipeline

- 목적: 전체 전처리 파이프라인 실행, 교통 정규화, RAG 컬럼 확장, QA base 생성
- 기본 입력: `data/original/apartment_20230905.csv`
- 기본 출력:
  - `data/apartment_chatbot_v3.csv`
  - `data/qa/apartment_chatbot_qa_base_v3.csv`
  - `data/station_line_map.csv`
  - `data/apartment_station_map.csv`
  - `data/apartment_column_mapping.csv`
  - `00_Report/02_preprocessing_full_report.md`

### Run: generate_qa

- 목적: 메인 RAG CSV 기반 QA, finetune, evaluation 데이터 생성
- 기본 입력: `data/apartment_chatbot_v3.csv`
- 기본 출력:
  - `data/qa/apartment_qa_dataset.csv`
  - `data/qa/apartment_finetune_dataset.jsonl`
  - `data/qa/evaluation_dataset.csv`
  - `00_Report/03_qa_generation_report.md`

### Run: generate_edge_questions

- 목적: 평가용 edge case 질문 생성
- 기본 입력: `data/apartment_chatbot_v3.csv`
- 기본 출력:
  - `data/qa/edge_case_questions.csv`
  - `data/qa/edge_case_eval.csv`
  - `00_Report/04_edge_question_report.md`

## 6. CSV preprocessing rules

- 먼저 입력 CSV의 행 수, 열 수, 컬럼 목록, dtype, 결측치 현황을 분석하라.
- 완전 동일 행만 제거하라.
- 완전 동일 행 이외의 중복 가능성은 제거하지 말고 리포트에 남겨라.
- 주소 컬럼이 있으면 `시도`, `시군구`, `동`, `상세주소`로 분리하라.
- 주소 분리 성공률을 계산하라.
- 주소 분리 실패율이 10%를 초과하면 warning을 기록하라.
- 숫자 컬럼은 비교와 정렬이 가능하도록 정제하라.
- 결측치는 무리하게 0으로 채우지 마라.
- 문자열 결측 대체가 필요하면 `"정보 없음"`을 쓸 수 있지만, 원본 의미를 바꾸지 마라.

## 7. Policy field rules

- 정책 컬럼은 삭제하지 마라.
- 정책 컬럼이 있으면 반드시 다음 이름으로 정규화하라.
  - `투기과열지구_before` → `분양당시_투기과열지구`
  - `투기과열지구_after` → `현재_투기과열지구`
  - `분양가상한제_before` → `분양당시_분양가상한제`
  - `분양가상한제_after` → `현재_분양가상한제`
- 정책 컬럼 일부가 없으면 warning을 남기고 계속하라.
- 정책 컬럼 전체가 없으면 warning을 남기고 계속하되, report에 명시하라.
- 정책 정보는 메인 CSV와 설명형 컬럼 양쪽에 유지하라.
- 정책 설명 컬럼은 사람이 읽기 쉬운 한국어 문장으로 작성하라.

## 8. Transport rules

- 교통 데이터가 있으면 `가장가까운역`, 거리, 환승 여부, 호선 요약을 메인 CSV에 유지하라.
- `지하철역_거리`가 km 계열이면 `거리_m`를 추가 생성하라.
- 역명과 호선 정보가 있으면 `station_line_map.csv`를 생성하라.
- 문서ID와 역명 연결이 가능하면 `apartment_station_map.csv`를 생성하라.
- 환승역 여부는 기본적으로 `호선수 >= 2` 규칙을 사용하라.
- 교통 관련 필수 컬럼이 일부 누락되면 warning을 남기고 가능한 범위까지 처리하라.

## 9. RAG rules

- 메인 CSV에는 사람이 읽기 쉬운 요약 컬럼을 유지하라.
- `description`은 반드시 한국어로 작성하라.
- `description`은 3~6문장으로 작성하라.
- `description` 최대 길이는 300자로 유지하라. 초과 시 정보 우선순위 기준으로 줄여라.
- `description`에는 가능한 범위에서 다음 정보를 포함하라.
  - 위치
  - 규모
  - 면적
  - 교통
  - 가격
  - 생활 인프라
  - 정책
- 값이 없으면 억지로 넣지 마라.
- 금지 표현:
  - 추측성 표현
  - 데이터에 없는 추천 단정
  - 과장 표현
- `검색키워드`에는 다음 요소를 포함하도록 구성하라.
  - 아파트명
  - 지역명
  - 역명
  - 호선
  - 건설사
  - 면적대
  - 정책정보

## 10. QA generation rules

- QA 답변은 반드시 CSV 기반 정보만 사용하라.
- QA 생성 시 질문 다양성을 확보하라.
- 질문 중복은 제거하라.
- 빈 질문과 빈 답변은 제거하라.
- QA category는 최소한 사실형, 위치형, 교통형, 가격형, 생활환경형, 정책형을 포함하라.
- category별 비중은 한쪽으로 과도하게 치우치지 않게 유지하라.
- category별 비중이 전체의 10% 미만이면 report에 warning을 남겨라.
- 답변은 짧고 명확하게 작성하라.
- 답변은 가능하면 1~3문장으로 작성하라.
- QA 산출물은 `data/qa` 아래에 저장하라.
- evaluation 데이터와 edge question 데이터는 메인 QA와 분리하라.

권장 QA 출력 스펙:

- QA CSV 필수 컬럼:
  - `question`
  - `answer`
  - `아파트명`
  - `문서ID`
  - `category`
- evaluation CSV 필수 컬럼:
  - `question`
  - `expected_answer`
  - `문서ID`
- edge question CSV 필수 컬럼:
  - `question`
  - `type`

## 11. Output contracts

- CSV는 `utf-8-sig`, header 포함으로 저장하라.
- 메인 RAG CSV는 최소한 다음 컬럼을 포함하도록 유지하라.
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
  - `가장가까운역_호선요약`
  - `description`
  - `검색키워드`
- JSONL이 필요한 경우 한 줄당 하나의 JSON object로 저장하라.
- finetune JSONL은 기본적으로 다음 구조를 사용하라.

```json
{"instruction":"질문","input":"","output":"답변"}
```

## 12. Failure and stop rules

- 다음 경우에는 즉시 중단하고 report를 생성하라.
  - 입력 파일이 없음
  - 필수 입력 컬럼이 모두 누락됨
  - 결과 행 수가 원본 대비 30% 이상 감소했는데 명확한 사유를 설명할 수 없음
  - 문서ID 생성 실패
- 다음 경우에는 warning을 남기고 계속하라.
  - 정책 컬럼 일부 누락
  - 주소 분리 실패율 10% 초과
  - 교통 컬럼 일부 누락
  - category 비중 불균형
- 조용히 실패하지 마라.
- 예외를 무시하지 말고 report나 log에 남겨라.

## 13. Report requirements

- 모든 major run은 markdown report를 생성하라.
- report 파일은 숫자 접두어를 사용해 순서를 유지하라.
- report는 최소한 다음 섹션을 포함하라.
  - 목적
  - 입력 파일
  - 출력 파일
  - 원본 데이터 크기
  - 최종 데이터 크기
  - 제거된 완전 중복 수
  - 새로 생성된 컬럼
  - 처리 단계 요약
  - 주요 예외 및 warning
  - 샘플 데이터
- processing log가 있으면 report에서 로그 파일 위치를 함께 적어라.

## 14. Communication / coding style

- 설명은 짧고 작업 중심으로 작성하라.
- 명령형 문장을 사용하라.
- 경로와 산출물 위치를 항상 명확히 적어라.
- 스크립트는 재실행 가능하게 작성하라.
- 추측보다 명시된 규칙을 우선하라.
- 애매한 표현보다 수치 기준과 출력 계약을 우선하라.
