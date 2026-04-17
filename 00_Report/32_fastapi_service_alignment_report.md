# 32 FastAPI Service Alignment Report

## Purpose

Add a formal FastAPI service layer for the real estate grounded QA runtime while preserving
the existing stage-based repository structure and current response semantics.

## Input Files

- `02_gemma4_generation/demo_chatbot_web_mvp.py`
- `02_gemma4_generation/query_service.py`
- `02_gemma4_generation/web_demo_support.py`
- `README.md`
- `02_gemma4_generation/README.md`
- `06_finetuning/README.md`

## Output Files

- `02_gemma4_generation/api_runtime.py`
- `02_gemma4_generation/api_models.py`
- `02_gemma4_generation/api_logging.py`
- `02_gemma4_generation/fastapi_app.py`
- `02_gemma4_generation/tests/test_fastapi_app.py`
- `README.md`
- `02_gemma4_generation/README.md`
- `06_finetuning/README.md`
- `requirements.txt`
- `00_Report/32_fastapi_service_alignment_report.md`

## Original Row and Column Counts

- Not applicable for this service-layer task.
- No dataset rows or columns were modified.

## Final Row and Column Counts

- Not applicable for this service-layer task.
- No dataset rows or columns were modified.

## Removed Exact Duplicate Count

- `0`

## Newly Generated Columns

- None

## Processing Steps Summary

1. Added a shared runtime module for grounded QA requests and context loading.
2. Added Pydantic request and response models for the FastAPI contract.
3. Added a JSONL request logger under `logs/api_requests`.
4. Added a formal FastAPI app with four v1 endpoints.
5. Connected the existing web MVP to the shared runtime context and shared ask path.
6. Added endpoint-level contract and status tests.
7. Updated top-level and stage-level documentation.

## Major Warnings or Exceptions

- This change adds new Python dependencies for the FastAPI service path.
- The existing web MVP remains in place; the repository now has both a demo UI path and a formal API path.
- No authentication or deployment-grade health endpoint was added in this v1 scope.

## Sample Outputs

- FastAPI endpoint surface:
  - `GET /api/status`
  - `POST /api/ask`
  - `POST /api/check-rule`
  - `POST /api/check-generation-ready`
- Log output path:
  - `logs/api_requests/fastapi_YYYYMMDD.jsonl`

## Validation Performed

- Added endpoint contract tests for meta, knowledge, fact lookup, structured recommendation,
  comparative recommendation, unsupported comparative, `NO_MATCH`, status, rule check, and mock readiness.
- Reused the existing runtime safety semantics for `NO_MATCH`, `UNKNOWN`, and busy generation state.

## Remaining Follow-Up Items

- Add a deployment-oriented `/health` endpoint in a later version if needed.
- Add operator-facing review endpoints only after the current v1 API path is validated.
- Consider moving the remaining duplicate helper code out of the web MVP if a second refactor pass is needed.
