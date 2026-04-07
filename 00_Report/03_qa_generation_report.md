# QA Generation Report

## 개요
- 입력 파일: `apartment_chatbot_v3.csv`
- 입력 인코딩: `utf-8-sig`
- 입력 아파트 행 수: 3239
- 필터 전 QA 수: 61560
- 필터 후 QA 수: 56392
- null 응답으로 제외된 수: 0
- 질문 중복 제거 수: 5168
- 평가셋 크기: 1000

## 생성 규칙 요약
- 아파트별 사실 QA를 기본으로 생성했습니다.
- 일반 부동산 지식 QA와 데이터 범위 안내 QA를 추가했습니다.
- no-match 응답과 지원/미지원 비교 질문을 별도 카테고리로 추가했습니다.

## 카테고리별 QA 수
| category | count |
| --- | --- |
| fact | 17802 |
| lifestyle | 11868 |
| transport | 8901 |
| price | 8901 |
| location | 5934 |
| policy | 2967 |
| knowledge_term | 12 |
| scope_meta | 2 |
| recommend_no_match | 2 |
| comparative_supported | 2 |
| comparative_unsupported | 1 |

## 샘플 QA 12개
| question | answer | 아파트명 | 문서ID | category | answer_type | expected_answer_type | expected_match_status | must_include | must_not_include |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 올림픽파크포레온 전용 84.99㎡ 공급 114.26㎡ 타입 16 전용면적은 얼마야 | 올림픽파크포레온의 전용면적은 84.99㎡입니다. | 올림픽파크포레온 | APT_000001 | fact | apartment_fact_lookup | apartment_fact_lookup | EXACT_MATCH | 올림픽파크포레온 |  |
| 올림픽파크포레온 전용 84.99㎡ 공급 114.26㎡ 타입 16 공급면적은 얼마야 | 올림픽파크포레온의 공급면적은 114.26㎡입니다. | 올림픽파크포레온 | APT_000001 | fact | apartment_fact_lookup | apartment_fact_lookup | EXACT_MATCH | 올림픽파크포레온 |  |
| 올림픽파크포레온 전용 84.99㎡ 공급 114.26㎡ 타입 16 세대수는 몇 세대야 | 올림픽파크포레온은 총 12,032세대 규모입니다. | 올림픽파크포레온 | APT_000001 | fact | apartment_fact_lookup | apartment_fact_lookup | EXACT_MATCH | 올림픽파크포레온 |  |
| 올림픽파크포레온 전용 84.99㎡ 공급 114.26㎡ 타입 16 구조는 어떻게 돼 | 올림픽파크포레온의 구조 요약은 전용면적 85㎡ 기준 방 3개 욕실 2개 혼합식 현관구조 구조입니다. | 올림픽파크포레온 | APT_000001 | fact | apartment_fact_lookup | apartment_fact_lookup | EXACT_MATCH | 올림픽파크포레온 |  |
| 올림픽파크포레온 전용 84.99㎡ 공급 114.26㎡ 타입 16 면적대는 뭐야 | 올림픽파크포레온은 중형 면적대로 분류됩니다. | 올림픽파크포레온 | APT_000001 | fact | apartment_fact_lookup | apartment_fact_lookup | EXACT_MATCH | 올림픽파크포레온 |  |
| 올림픽파크포레온 전용 84.99㎡ 공급 114.26㎡ 타입 16 위치가 어디야 | 올림픽파크포레온은 서울특별시 강동구 둔촌1동에 위치합니다. | 올림픽파크포레온 | APT_000001 | location | apartment_fact_lookup | apartment_fact_lookup | EXACT_MATCH | 올림픽파크포레온 |  |
| 올림픽파크포레온 전용 84.99㎡ 공급 114.26㎡ 타입 16 주소 알려줘 | 올림픽파크포레온의 주소는 서울특별시 강동구 둔촌1동 170-1입니다. | 올림픽파크포레온 | APT_000001 | location | apartment_fact_lookup | apartment_fact_lookup | EXACT_MATCH | 올림픽파크포레온 |  |
| 올림픽파크포레온 전용 84.99㎡ 공급 114.26㎡ 타입 16 근처 지하철역은 어디야 | 가장 가까운 역은 둔촌동역이며 약 664m 거리입니다. | 올림픽파크포레온 | APT_000001 | transport | apartment_fact_lookup | apartment_fact_lookup | EXACT_MATCH | 올림픽파크포레온 |  |
| 올림픽파크포레온 전용 84.99㎡ 공급 114.26㎡ 타입 16 지하철 접근성 어때 | 둔촌동역 이용 가능, 도보 거리 약 664m, 1개 노선 접근 | 올림픽파크포레온 | APT_000001 | transport | apartment_fact_lookup | apartment_fact_lookup | EXACT_MATCH | 올림픽파크포레온 |  |
| 올림픽파크포레온 전용 84.99㎡ 공급 114.26㎡ 타입 16 가까운 역까지 거리는 얼마야 | 가장 가까운 역까지 거리는 약 664m입니다. | 올림픽파크포레온 | APT_000001 | transport | apartment_fact_lookup | apartment_fact_lookup | EXACT_MATCH | 올림픽파크포레온 |  |
| 올림픽파크포레온 전용 84.99㎡ 공급 114.26㎡ 타입 16 분양가는 얼마야 | 올림픽파크포레온의 공급액은 132,040만원입니다. | 올림픽파크포레온 | APT_000001 | price | apartment_fact_lookup | apartment_fact_lookup | EXACT_MATCH | 올림픽파크포레온 |  |
| 올림픽파크포레온 전용 84.99㎡ 공급 114.26㎡ 타입 16 평당 가격은 | 평당 공급액은 3,813.51만원입니다. | 올림픽파크포레온 | APT_000001 | price | apartment_fact_lookup | apartment_fact_lookup | EXACT_MATCH | 올림픽파크포레온 |  |

## 생성 파일
- `apartment_qa_dataset.csv`
- `apartment_finetune_dataset.jsonl`
- `evaluation_dataset.csv`
- `03_qa_generation_report.md`