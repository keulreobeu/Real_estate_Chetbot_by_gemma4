# Edge Question Report

## 개요
- 입력 파일: `apartment_chatbot_v3.csv`
- 감지 인코딩: `utf-8-sig`
- 총 생성 질문 수: 2000

## 유형별 개수
| type | count |
| --- | --- |
| condition | 335 |
| comparison | 335 |
| multi_condition | 334 |
| vague | 334 |
| region | 333 |
| colloquial | 329 |

## 생성 규칙 요약
- 조건 질문: 거리, 가격, 병원, 정책, 건설사, 면적대 조건을 반영했습니다.
- 비교 질문: 세대수, 가격, 면적, 역 접근성 기준 최댓값/최솟값 질문을 만들었습니다.
- 복합 조건 질문: 지역 + 교통 + 가격 + 정책 + 인프라 조합을 사용했습니다.
- 지역 질문: 시도, 시군구, 동 단위 질문을 생성했습니다.
- 모호 질문: 추천/살기 좋음/생활 편의 같은 추상 표현을 사용했습니다.
- 구어체 질문: 오타/축약/구어형 표현을 반영했습니다.

## 샘플 12개
| question | type | expected_doc | expected_field |
| --- | --- | --- | --- |
| 지하철 300m 이내 아파트 알려줘 | condition | APT_002261 | 거리_m |
| 역까지 300m 안쪽 아파트 뭐 있어 | condition | APT_002261 | 거리_m |
| 지하철 500m 이내 아파트 알려줘 | condition | APT_002267 | 거리_m |
| 역까지 500m 안쪽 아파트 뭐 있어 | condition | APT_002267 | 거리_m |
| 지하철 700m 이내 아파트 알려줘 | condition | APT_002262 | 거리_m |
| 역까지 700m 안쪽 아파트 뭐 있어 | condition | APT_002262 | 거리_m |
| 지하철 1000m 이내 아파트 알려줘 | condition | APT_002267 | 거리_m |
| 역까지 1000m 안쪽 아파트 뭐 있어 | condition | APT_002267 | 거리_m |
| 중형 아파트 알려줘 | condition | APT_003149 | 면적대 |
| 중형 면적대 단지 뭐 있어 | condition | APT_003149 | 면적대 |
| 중소형 아파트 알려줘 | condition | APT_003065 | 면적대 |
| 중소형 면적대 단지 뭐 있어 | condition | APT_003065 | 면적대 |

## 생성 파일
- `edge_case_questions.csv`
- `edge_case_eval.csv`
- `edge_question_report.md`