# Edge Question Report

## 개요
- 입력 파일: `apartment_chatbot_v3.csv`
- 입력 인코딩: `utf-8-sig`
- 총 생성 질문 수: 2000

## 유형별 개수
| type | count |
| --- | --- |
| condition | 334 |
| comparison | 334 |
| multi_condition | 333 |
| region | 333 |
| vague | 333 |
| colloquial | 333 |

## 생성 규칙 요약
- 추천형 질문에는 expected_router_type과 expected_match_status를 함께 기록했습니다.
- no-match와 unsupported comparative 질문은 must_not_recommend=Y로 표시했습니다.
- 한계 고지가 필요한 질문은 must_disclose_limit=Y로 표시했습니다.

## 샘플 12개
| question | type | expected_doc | expected_field | expected_router_type | expected_match_status | must_not_recommend | must_disclose_limit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 화성시에서 지하철 300m 이내 아파트 추천해줘 | condition | APT_002263 | 거리_m | RECOMMEND_STRUCTURED | EXACT_MATCH | N | N |
| 화성시에서 지하철 500m 이내 아파트 추천해줘 | condition | APT_002267 | 거리_m | RECOMMEND_STRUCTURED | EXACT_MATCH | N | N |
| 화성시에서 지하철 700m 이내 아파트 추천해줘 | condition | APT_002263 | 거리_m | RECOMMEND_STRUCTURED | EXACT_MATCH | N | N |
| 화성시에서 지하철 1000m 이내 아파트 추천해줘 | condition | APT_002263 | 거리_m | RECOMMEND_STRUCTURED | EXACT_MATCH | N | N |
| 화성시에서 가격 괜찮은 아파트 추천해줘 | condition | APT_001874 | 가격요약 | RECOMMEND_STRUCTURED | EXACT_MATCH | N | N |
| 연수구에서 지하철 300m 이내 아파트 추천해줘 | condition | APT_001421 | 거리_m | RECOMMEND_STRUCTURED | EXACT_MATCH | N | N |
| 연수구에서 지하철 500m 이내 아파트 추천해줘 | condition | APT_001421 | 거리_m | RECOMMEND_STRUCTURED | EXACT_MATCH | N | N |
| 연수구에서 지하철 700m 이내 아파트 추천해줘 | condition | APT_001424 | 거리_m | RECOMMEND_STRUCTURED | EXACT_MATCH | N | N |
| 연수구에서 지하철 1000m 이내 아파트 추천해줘 | condition | APT_001423 | 거리_m | RECOMMEND_STRUCTURED | EXACT_MATCH | N | N |
| 연수구에서 가격 괜찮은 아파트 추천해줘 | condition | APT_001406 | 가격요약 | RECOMMEND_STRUCTURED | EXACT_MATCH | N | N |
| 평택시에서 지하철 300m 이내 아파트 추천해줘 | condition | APT_000851 | 거리_m | RECOMMEND_STRUCTURED | EXACT_MATCH | N | N |
| 평택시에서 지하철 500m 이내 아파트 추천해줘 | condition | APT_000854 | 거리_m | RECOMMEND_STRUCTURED | EXACT_MATCH | N | N |

## 생성 파일
- `edge_case_questions.csv`
- `edge_case_eval.csv`
- `04_edge_question_report.md`