# Web GPT Handoff

## Purpose

This repository is a stage-based real estate AI chatbot pipeline project.

Main goals:
- preprocess raw apartment CSV data
- normalize transport and policy-related fields
- build RAG-ready apartment datasets
- generate QA, evaluation, and edge-case datasets
- prepare safer generation and future finetuning with Gemma 4

Current active stage:
- `03_generation_optimization`

## Current status

The project has already completed:
- apartment preprocessing and main RAG dataset build
- QA / evaluation / edge-case dataset generation
- Gemma 4 generation MVP structure
- stage `05_finetuning_prep` bootstrap outputs

The current focus is:
- recommendation safety
- routing accuracy
- match-status accuracy
- reducing edge-eval label noise before finetuning

Current gate status for stage `06_finetuning`:
- `NO_GO`

Main reason:
- there are still legacy edge-prediction rows and optimization issues that must be fixed before regenerating the finetuning dataset

## Important folders

- `00_Report`
  - run reports and processing summaries
- `01_preprocessing`
  - preprocessing, QA generation, edge-question generation
- `02_gemma4_generation`
  - retrieval-first generation MVP and demo apps
- `03_generation_optimization`
  - current active stage
  - edge failure analysis, safety/routing optimization gate
- `05_finetuning_prep`
  - SFT candidate filtering, holdout, train/valid JSONL prep
- `data`
  - processed main datasets
- `data/qa`
  - QA, evaluation, edge, finetuning-ready artifacts
- `data/eval/generation_optimization`
  - failure buckets, review queues, stage-gate outputs

## Main artifacts

- `data/apartment_chatbot_v3.csv`
  - main RAG-ready apartment dataset
- `data/qa/apartment_qa_dataset.csv`
  - generated QA dataset
- `data/qa/evaluation_dataset.csv`
  - evaluation questions
- `data/qa/edge_case_eval.csv`
  - edge-case evaluation set
- `data/qa/real_estate_knowledge_base.csv`
  - general real estate knowledge base
- `data/eval/generation_optimization/edge_failure_buckets_gemma4_2b.csv`
  - failure bucket counts
- `data/eval/generation_optimization/hard_negative_review_queue_gemma4_2b.csv`
  - review queue for unsafe or mismatched cases
- `data/qa/finetuning_prep/stage06_readiness_gemma4_2b.json`
  - current finetuning readiness gate result

## Important scripts

- `01_preprocessing/preprocess_apartment_pipeline.py`
  - builds the cleaned/main dataset
- `01_preprocessing/generate_apartment_qa_dataset.py`
  - builds the QA dataset
- `01_preprocessing/generate_edge_questions.py`
  - builds edge-case questions
- `02_gemma4_generation/query_service.py`
  - core routing and answer contract logic
- `02_gemma4_generation/run_generation_mvp.py`
  - eval/edge run script
- `02_gemma4_generation/demo_chatbot_web_mvp.py`
  - web demo app
- `03_generation_optimization/analyze_edge_failures.py`
  - analyzes edge failures and produces optimization artifacts
- `05_finetuning_prep/prepare_sft_dataset.py`
  - prepares train/valid JSONL for SFT
- `05_finetuning_prep/validate_06_readiness.py`
  - checks whether stage `06_finetuning` is allowed

## Core data / behavior contracts

- Raw source files in `data/original` must not be modified directly.
- Main CSV outputs should be saved as `utf-8-sig` with headers.
- Recommendation safety matters more than aggressive recommendation behavior.
- If match status is `NO_MATCH` or `UNKNOWN`, the system should not force apartment recommendations.
- Unsupported area-band words should be treated as insufficiently structured filters unless supported by explicit numeric area conditions.
- Schema changes must be treated as contract changes and reflected in both code and docs.

## Current known constraints

- Active model line is centered on `gemma4_2b`.
- Generation MVP is retrieval-first with deterministic contract handling first.
- GPU-backed full validation is still pending for some flows.
- Stage `06_finetuning` is currently blocked until generation optimization issues are addressed and finetuning prep is regenerated.

## Good questions to ask Web GPT

- Review the pipeline architecture and suggest simplifications without changing current data contracts.
- Propose ways to improve recommendation safety and routing accuracy.
- Suggest a strategy to regenerate and relabel legacy edge rows.
- Review the readiness criteria for moving from `03_generation_optimization` to `06_finetuning`.
- Help design better hard-negative sampling or edge-case taxonomy.
- Suggest better reporting or artifact tracking for this repository.
- Help draft stage-specific documentation for `03_generation_optimization`.

## Copy-paste prompt for Web GPT

Use this project context when answering:

I am working on a stage-based real estate AI chatbot pipeline project.

Project goals:
- preprocess raw apartment CSV data
- normalize transport and policy data
- build RAG-ready datasets
- generate QA / evaluation / edge-case datasets
- prepare safer generation and later finetuning with Gemma 4

Current active stage:
- 03_generation_optimization

Important facts:
- main RAG dataset: `data/apartment_chatbot_v3.csv`
- QA dataset: `data/qa/apartment_qa_dataset.csv`
- edge eval dataset: `data/qa/edge_case_eval.csv`
- generation optimization outputs are under `data/eval/generation_optimization`
- finetuning readiness output is `data/qa/finetuning_prep/stage06_readiness_gemma4_2b.json`
- current stage 06 verdict is `NO_GO`
- recommendation safety is critical
- if match status is `NO_MATCH` or `UNKNOWN`, the system should avoid forced apartment recommendations
- current focus is fixing safety/routing/match-status failures before finetuning

Important scripts:
- `01_preprocessing/preprocess_apartment_pipeline.py`
- `01_preprocessing/generate_apartment_qa_dataset.py`
- `01_preprocessing/generate_edge_questions.py`
- `02_gemma4_generation/query_service.py`
- `03_generation_optimization/analyze_edge_failures.py`
- `05_finetuning_prep/prepare_sft_dataset.py`
- `05_finetuning_prep/validate_06_readiness.py`

Please answer my next question with this context in mind. If you suggest changes, preserve existing data contracts and stage-based repository structure unless there is a strong reason to change them.

## Recommended reference order

If Web GPT needs the most useful reading order, use this:
1. `AGENTS.md`
2. `03_generation_optimization/README.md`
3. `03_generation_optimization/CONTRACT.md`
4. `00_Report/23_generation_optimization_gate_report.md`
5. `README.md`
