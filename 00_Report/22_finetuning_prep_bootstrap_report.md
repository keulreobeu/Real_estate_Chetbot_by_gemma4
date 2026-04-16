# 22. Finetuning Prep Bootstrap Report

## purpose

Bootstrap stage `05_finetuning_prep` so the repository can move from completed generation/evaluation outputs into a reproducible first-cycle SFT dataset workflow.

This run focused on:

- creating the new stage `05_finetuning_prep`
- separating trainable vs rejected outputs from completed stage 02 artifacts
- generating holdout sets for grounded generation and edge safety checks
- writing first-cycle `train/valid` JSONL files
- updating repository docs to include the new stage and command

## input files

- `data/eval/gemma4_generation_edge_predictions_gemma4_2b.csv`
- `data/eval/gemma4_generation_eval_predictions_gemma4_2b.csv`
- `data/qa/edge_case_eval.csv`
- `data/qa/evaluation_dataset.csv`
- `README.md`
- `AGENTS.md`

## output files

Code and stage docs:

- `05_finetuning_prep/prepare_sft_dataset.py`
- `05_finetuning_prep/README.md`
- `05_finetuning_prep/CONTRACT.md`
- `05_finetuning_prep/dataset_contract.md`

Generated artifacts:

- `data/qa/finetuning_prep/training_candidates_gemma4_2b.csv`
- `data/qa/finetuning_prep/training_rejected_gemma4_2b.csv`
- `data/qa/finetuning_prep/holdout_grounded_generation.csv`
- `data/qa/finetuning_prep/holdout_edge_safety.csv`
- `data/qa/finetuning_prep/train_gemma4_2b.jsonl`
- `data/qa/finetuning_prep/valid_gemma4_2b.jsonl`
- `data/qa/finetuning_prep/dataset_summary_gemma4_2b.json`

Documentation updates:

- `README.md`
- `AGENTS.md`

## original row and column counts

- edge predictions: `2000` rows, `41` columns
- eval predictions: `1000` rows, `41` columns
- edge dataset: `2000` rows
- eval dataset: `1000` rows

## final row and column counts

- combined candidate universe: `3000` rows
- rejected rows: `2557`
- grounded-generation holdout: `200`
- edge-safety holdout: `150`
- train JSONL rows: `85`
- valid JSONL rows: `8`

Train bucket distribution:

- `grounded_generation`: `59`
- `safe_recommendation`: `13`
- `safety_refusal`: `13`

Valid bucket distribution:

- `grounded_generation`: `6`
- `safe_recommendation`: `1`
- `safety_refusal`: `1`

## removed exact duplicate count

- exact duplicate `(question, answer)` rows removed: `0`
- duplicate `(source_dataset, source_row_index)` rows removed: `0`

## newly generated columns

New stage-05 candidate annotations:

- `source_dataset`
- `include_in_sft`
- `exclusion_reason`
- `sft_bucket`
- `split`
- `stable_id`

## processing steps summary

### 1. Added stage 05 preparation script

`prepare_sft_dataset.py` now:

- loads completed edge/eval prediction CSVs
- normalizes missing contract fields
- rejects deterministic and legacy rows according to the approved first-cycle policy
- reserves `grounded_generation` holdout before safety holdout
- down-samples non-generation buckets to keep train/valid centered on grounded generation
- emits reproducible JSONL outputs using deterministic hash-based selection

### 2. Added stage 05 contracts and usage docs

Added:

- stage README
- stage contract
- dataset contract

These docs describe:

- first-cycle inclusion/exclusion policy
- output paths
- JSONL contract
- holdout strategy

### 3. Updated root docs

`README.md` and `AGENTS.md` now include:

- active stage `05_finetuning_prep`
- new stage folder in repository structure
- `data/qa/finetuning_prep/` artifact path
- stage 05 command

## major warnings or exceptions

- the train split is intentionally small because the current repository truth contains:
  - `1000` deterministic eval rows excluded by policy
  - `970` legacy edge rows missing `answer_type/match_status/query_type`
  - `539` recommendation rows flagged with `must_not_recommend=Y`
- `holdout_edge_safety.csv` is built from the broader safety universe, not only from trainable rows, so it can be used as a stricter regression set later
- current `grounded_generation` answers are still weak in several samples, so stage `03_generation_optimization` should improve routing/safety/quality before `06_finetuning`

## sample outputs

Sample `train_gemma4_2b.jsonl` rows:

```json
{"instruction":"경기도에서 추천 가능한 아파트 알려줘","input":"","output":"조건에 맞는 아파트 후보입니다.\n- 힐스테이트리슈빌강일: 역 강일역, 0.0m, 공급액 69980만원, 생활 편의시설이 무난하게 형성된 지역입니다.\n- 주안극동스타클래스더로얄: 역 시민공원역, 27.0m, 공급액 39730만원, 생활 편의시설이 무난하게 형성된 지역입니다.\n- 부평역해링턴플레이스: 역 부평역, 37.8m, 공급액 49800만원, 생활 편의시설이 무난하게 형성된 지역입니다.\n데이터 기준일: 2023-09-05\n답변 가능 범위: 아파트 기본정보, 가격, 교통, 정책, 공원/병원 기반 비교 근거 문서: APT_002271, APT_000409, APT_000853"}
{"instruction":"서울시에서 추천 가능한 아파트 알려줘","input":"","output":"조건에 맞는 아파트 후보입니다.\n- 덕은삼정그린코아더베스트: 역 화전역, 93.4m, 공급액 48700만원, 주변 상권과 생활 편의시설이 비교적 풍부한 지역입니다.\n- 쌍용더플래티넘오목천역: 역 오목천역, 113.7m, 공급액 25200만원, 주변 상권과 생활 편의시설이 비교적 풍부한 지역입니다.\n- 군포당동그랑시티2차: 역 군포역, 164.9m, 공급액 28000만원, 생활 인프라는 제한적인 편입니다.\n데이터 기준일: 2023-09-05\n답변 가능 범위: 아파트 기본정보, 가격, 교통, 정책, 공원/병원 기반 비교 근거 문서: APT_002728, APT_001900, APT_002860"}
```

Sample `valid_gemma4_2b.jsonl` row:

```json
{"instruction":"서구 주변 아파트 알려줘","input":"","output":"인천광역시 서구 불로동에 위치한 아파트는 인덕원자이SKVIEW가 있습니다. 이 아파트는 총 2,633세대 규모이며 전용면적은 60.00㎡ 또는 50.00㎡입니다. 가장 가까운 지하 근거 문서: APT_000196, APT_000197"}
```

## validation method

- syntax check:
  - `& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' -m compileall .\05_finetuning_prep\prepare_sft_dataset.py`
- stage 05 generation run:
  - `& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\05_finetuning_prep\prepare_sft_dataset.py --model gemma4_2b`
- output existence check:
  - `Get-ChildItem .\data\qa\finetuning_prep`

## reference artifacts

- summary: `data/qa/finetuning_prep/dataset_summary_gemma4_2b.json`
