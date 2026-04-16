# 02_gemma4_generation

## Purpose

`02_gemma4_generation` is the retrieval-first Gemma 4 generation stage after `01_preprocessing`.

This stage keeps the pipeline intentionally small:

- single-turn grounded QA
- retrieval-first prompt assembly
- query routing and match-status evaluation before fallback generation
- local Gemma 4 inference
- offline evaluation on existing eval and edge datasets

Primary local runtime:

- `transformers` + `torch` + `accelerate`

Experimental runtime:

- `llama_cpp`

## MVP V2 defaults

The current MVP V2 runs with these defaults:

- default backend: `transformers`
- default model: `gemma4_2b`
- deterministic contract path first for recommendation, comparison, fact, knowledge, and meta queries
- generation path only for `GENERAL_RETRIEVAL_QA`
- recommendation safety rule: `NO_MATCH` and `UNKNOWN` never show arbitrary apartment candidates
- structured recommendation supports objective filters such as price, area band, subway distance, and park/hospital access
- implicit `역 가까운` requests are treated as a bounded near-subway filter (default `500m`) so far-away stations do not produce false exact-match recommendations
- vague recommendation phrases such as broad region-only asks or unsupported qualifier asks are routed to safe `UNKNOWN`
- subjective quality asks such as `괜찮은`, `무난한`, `추천할 만한`, and `실거주 괜찮은` are treated as unsupported comparative requests rather than structured recommendation
- region-scoped explanation prompts such as `화성시 아파트 뭐 있어` and `동탄동 대표 단지 특징 설명해줘` are routed to `GENERAL_RETRIEVAL_QA` so grounded generation remains covered in edge evaluation
- public demo recommendation: `transformers --model gemma4_2b`

## Inputs

- `data/apartment_chatbot_v3.csv`
- `data/qa/evaluation_dataset.csv`
- `data/qa/edge_case_eval.csv`

## Outputs

- `data/eval/gemma4_generation_source_index.csv`
- `data/eval/gemma4_generation_eval_predictions_<model_id>.csv`
- `data/eval/gemma4_generation_edge_predictions_<model_id>.csv`
- `data/eval/gemma4_generation_eval_metrics_<model_id>.json`
- `data/eval/gemma4_generation_edge_metrics_<model_id>.json`
- `00_Report/07_gemma4_generation_model_comparison.md`

## Folder structure

```text
02_gemma4_generation/
  README.md
  CONTRACT.md
  common.py
  build_generation_assets.py
  verify_local_inference_setup.py
  run_generation_mvp.py
  demo_chatbot_mvp.py
  demo_chatbot_web_mvp.py
  benchmark_edge_2b.py
  monitor_edge_progress.py
  edge_preflight_2b.ps1
  evaluate_generation_mvp.py
  compare_generation_runs.py
  inference/
    base.py
    mock_adapter.py
    transformers_adapter.py
    llama_cpp_adapter.py
    registry.py
  config/
    generation_defaults.json
    models.local.example.json
  prompts/
    grounded_answer_prompt.txt
```

## Pipeline

1. Validate upstream datasets from `01_preprocessing`.
2. Build a retrieval-ready source index from the main RAG dataset.
3. Retrieve top-k rows for each question.
4. Route the question into recommendation, comparison, fact lookup, knowledge, or meta response flow.
5. For recommendation questions, enforce `EXACT_MATCH / NO_MATCH / UNKNOWN` before any fallback generation.
6. Build a grounded prompt from retrieved evidence only when the query is not fully answered by the deterministic contract path.
7. Generate an answer through a backend adapter.
8. Save predictions and evaluate them offline.

## Prerequisites

- Python environment with `pandas`
- `torch`
- `transformers`
- `accelerate`

## Docker

프로젝트 루트 기준 Docker 개발 환경을 사용할 수 있습니다.

빌드:

```powershell
docker compose build
```

셸 진입:

```powershell
docker compose run --rm chatbot-dev
```

컨테이너 내부 readiness check:

```bash
python ./02_gemma4_generation/verify_local_inference_setup.py
python ./02_gemma4_generation/verify_local_inference_setup.py --model gemma4_2b
```

주의:

- Docker 기본 이미지는 `transformers` 경로를 기준으로 구성했습니다.
- `llama_cpp`와 GGUF 모델 파일은 기본 이미지에 포함하지 않습니다.
- `models.local.json`과 Hugging Face 캐시는 로컬 파일/볼륨으로 관리해야 합니다.

Recommended install command in the active Python environment:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' -m pip install -U transformers accelerate
```

Gemma 4 model loading uses the official Hugging Face model ids by default:

- `google/gemma-4-E2B-it`
- `google/gemma-4-E4B-it`

If you already have a local Hugging Face snapshot, point `local_dir` to that folder in `models.local.json`.

## Local model setup

1. Copy `02_gemma4_generation/config/models.local.example.json` to `02_gemma4_generation/config/models.local.json`
2. Keep `hf_model_id` as-is, or replace `local_dir` with a local snapshot path
3. Keep `models.local.json` uncommitted

```powershell
Copy-Item .\02_gemma4_generation\config\models.local.example.json .\02_gemma4_generation\config\models.local.json
```

## Runtime verification

Run readiness checks before smoke tests:

```powershell
python .\02_gemma4_generation\verify_local_inference_setup.py
python .\02_gemma4_generation\verify_local_inference_setup.py --model gemma4_2b
python .\02_gemma4_generation\verify_local_inference_setup.py --model gemma4_4b
```

The verifier checks:

- `transformers`, `torch`, `accelerate` import availability
- `models.local.json` presence
- `gemma4_2b` and `gemma4_4b` config resolution
- optional `local_dir` existence when configured

## Scripts

### 1. Build retrieval assets

```powershell
python .\02_gemma4_generation\build_generation_assets.py
```

### 2. Run the MVP generation flow

```powershell
python .\02_gemma4_generation\run_generation_mvp.py --mode eval --backend mock --model gemma4_2b
python .\02_gemma4_generation\run_generation_mvp.py --mode eval --backend transformers --model gemma4_2b --limit 5
python .\02_gemma4_generation\run_generation_mvp.py --mode eval --backend transformers --model gemma4_4b --limit 5
```

### 2-1. Run local demo chatbot MVP

```powershell
python .\02_gemma4_generation\demo_chatbot_mvp.py --backend transformers --model gemma4_2b --question "지하철 500m 이내 아파트 알려줘"
python .\02_gemma4_generation\demo_chatbot_mvp.py --backend mock --model gemma4_2b
```

Demo output includes:
- answer text
- `answer_type`
- `match_status`
- `query_type`
- `cited_doc_ids`
- `top_doc_id`
- `retrieval_score`
- `used_fields`
- `data_cutoff`
- `limitations`

Demo scope and limitation:
- This is a local demonstration path for pre-finetuning validation.
- It is not yet a production-grade service endpoint.

### 2-2. Run local web demo chatbot MVP

```powershell
python .\02_gemma4_generation\demo_chatbot_web_mvp.py --backend transformers --model gemma4_2b --host 127.0.0.1 --port 8787
```

Then open:
- `http://127.0.0.1:8787`
- Web page header shows current `backend/model_id`, confirm it is `transformers` for real generation quality.
- Web page top cards now show included regions grouped as `서울권`, `경기권`, and `인천권`.
- The UI now separates `규칙기반 답변 확인` from `Gemma 생성 가능 상태 확인`.
- `규칙기반 답변 확인` runs a fixed fast contract question: `데이터 기준 알려줘`
- `Gemma 생성 가능 상태 확인` runs a short readiness probe instead of a full user-facing long answer.
- Web response panel now shows `answer_type`, `match_status`, `query_type`, `data_cutoff`, `limitations`, and `used_fields`.
- Web response panel also shows `device_map`, `model_source`, `last_load_ms`, and `last_generate_ms`.
- Current status card also shows `pid`, `port`, startup time, and latest readiness state.
- For LAN access from another device, bind host to `0.0.0.0` and open `http://<your-local-ip>:8787`.
- For access from completely different networks, use Cloudflare quick tunnel launcher:
  - `run_web_demo_public_double_click.bat` (double-click)
  - it prints a public `https://...trycloudflare.com` URL you can open from phone/mobile network.

The `mock` backend remains for regression checks. The primary runtime for real inference is now `transformers`.

Usage notes:

- Keep one web demo server per port to avoid port conflicts and duplicate model loading.
- The readiness button is for runtime verification only, not for judging answer quality.
- If a readiness probe or another generation request is already running, the UI reports the runtime as busy instead of starting another generation task.

Runtime defaults for V2:

- `max_output_tokens=96`
- `temperature=0.0`
- `top_p=1.0`
- `repeat_penalty=1.05`
- `request_timeout_seconds=20`
- `web_timeout_seconds=25`
- `device_map=auto` for the default local `gemma4_2b` runtime profile on constrained GPUs

Timeout behavior:

- deterministic contract path returns immediately without model generation
- generation path records a soft-timeout limitation when latency exceeds the configured request timeout
- web demo keeps serving the response but exposes the timeout state in `limitations` and `finish_reason`
- transformers runtime now records split timing debug fields for load, prompt preparation, device transfer, generation, and decode

Chunked execution options for large runs:

- `--offset`: start row index from the selected question dataset
- `--limit`: number of rows to process from `offset` (`None` means all remaining rows)
- `--append`: append new rows to an existing prediction CSV instead of overwrite
- `--resume`: skip already saved questions from the current output CSV and continue
- `--checkpoint-every N`: save partial progress every N rows during a long run
- `--log-every N`: print progress/ETA every N rows
- `--startup-check` (default on): run a short startup probe before the long run
- default startup probe is now `load_only` and does not force a generation call
- `--startup-check-full`: run the full startup probe with a short generation sample
- `--no-startup-check`: skip startup probe intentionally
- `--profile fast_edge`: speed-oriented edge profile (shorter outputs + deterministic decoding)
- `--max-output-tokens`, `--temperature`, `--top-p`, `--repeat-penalty`: runtime generation overrides
- `--save-debug-columns`: persist `prompt_text`/`raw_response` in output CSV (off by default for long runs)
- `--output-path`: write predictions to a custom CSV path (used by benchmark runs)
- `--stop-signal-path`: file-based graceful stop signal path
- `--heartbeat-path`: JSON heartbeat path used by status/monitor detection

CPU inference behavior:

- `transformers` runtime now auto-uses all logical CPU cores by default during CPU-only runs.
- Optional per-model overrides in `models.local.json`:
  - `cpu_threads`: intra-op CPU threads
  - `cpu_interop_threads`: inter-op CPU threads (default `1`)
- `llama_cpp` runtime now defaults `n_threads` to all logical CPU cores when not configured.

Non-GPU stage 04 readiness gate:

- before resuming GPU work, run the contract/readiness check:

```powershell
python .\02_gemma4_generation\check_stage04_readiness.py --model gemma4_2b
```

- this gate compares:
  - contract-required prediction columns
  - evaluator-read columns
  - actual current prediction CSV columns
  - metric json required keys
- `CONCLUSION=YES` means the current files are structurally ready for stage 04
- `CONCLUSION=NO` means stage 04 metrics should not be treated as release-grade yet

Long-running orchestration worker:

```powershell
powershell -ExecutionPolicy Bypass -File .\02_gemma4_generation\run_pipeline_until_1400.ps1 -CheckpointEvery 10 -CutoffHour 23 -CutoffMinute 0
```

- resumes `edge`, then `eval`, then runs stage 04 metrics
- after stage 04 it also runs the stage 04 readiness gate and non-GPU validation steps

Fast-edge tuning (current):

- `--profile fast_edge` uses `max_output_tokens=64`, `temperature=0`, `top_p=1.0`
- fast-edge retrieval uses top-2 context docs for lower prompt cost

Resume safety behavior:

- preferred: `source_row_index` matching (safe against duplicate question text)
- legacy fallback: question-text matching only when old output files have no `source_row_index`
- during save, duplicate `source_row_index` rows are automatically deduplicated

Example for 50-row chunk execution:

```powershell
python .\02_gemma4_generation\run_generation_mvp.py --mode edge --backend transformers --model gemma4_2b --offset 0 --limit 50
python .\02_gemma4_generation\run_generation_mvp.py --mode edge --backend transformers --model gemma4_2b --offset 50 --limit 50 --append
python .\02_gemma4_generation\run_generation_mvp.py --mode edge --backend transformers --model gemma4_2b --offset 100 --limit 50 --append
```

Resume with checkpoint example:

```powershell
python .\02_gemma4_generation\run_generation_mvp.py --mode edge --backend transformers --model gemma4_2b --offset 0 --limit 50 --checkpoint-every 10
python .\02_gemma4_generation\run_generation_mvp.py --mode edge --backend transformers --model gemma4_2b --offset 0 --limit 50 --resume --append --checkpoint-every 10
```

Fast resume example for long edge runs:

```powershell
python .\02_gemma4_generation\run_generation_mvp.py --mode edge --backend transformers --model gemma4_2b --resume --append --profile fast_edge --no-startup-check --checkpoint-every 25 --log-every 10
```

### 3. Evaluate generation outputs

```powershell
python .\02_gemma4_generation\evaluate_generation_mvp.py --mode eval --model gemma4_2b
python .\02_gemma4_generation\evaluate_generation_mvp.py --mode edge --model gemma4_2b
```

### 3-1. Validate output completeness (no GPU)

```powershell
python .\02_gemma4_generation\validate_generation_outputs.py --mode edge --model gemma4_2b
python .\02_gemma4_generation\validate_generation_outputs.py --mode eval --model gemma4_2b
```

Optional output sanitization against the current input dataset:

```powershell
python .\02_gemma4_generation\sanitize_generation_outputs.py --mode edge --model gemma4_2b
python .\02_gemma4_generation\sanitize_generation_outputs.py --mode edge --model gemma4_2b --write
```

### 3-2. Edge runbook helper (prepared, no auto GPU execution)

The helper script prints safe command previews by default.
Actual generation/evaluation runs only happen when `-Execute` is explicitly set.

```powershell
.\02_gemma4_generation\edge_runbook_2b.ps1 -Action print
.\02_gemma4_generation\edge_runbook_2b.ps1 -Action precheck
.\02_gemma4_generation\edge_runbook_2b.ps1 -Action benchmark -SampleSize 20
.\02_gemma4_generation\edge_runbook_2b.ps1 -Action start -Offset 0 -Limit 0 -CheckpointEvery 25 -LogEvery 10
.\02_gemma4_generation\edge_runbook_2b.ps1 -Action resume -Offset 0 -Limit 0 -CheckpointEvery 25 -LogEvery 10
.\\02_gemma4_generation\\edge_runbook_2b.ps1 -Action stop
.\\02_gemma4_generation\\edge_runbook_2b.ps1 -Action monitor -MonitorIntervalMinutes 10 -TargetRows 2000
.\02_gemma4_generation\edge_runbook_2b.ps1 -Action status
.\02_gemma4_generation\edge_runbook_2b.ps1 -Action finalize
```

Run with execution enabled only when you are ready:

```powershell
.\02_gemma4_generation\edge_runbook_2b.ps1 -Action start -Offset 0 -Limit 0 -CheckpointEvery 25 -LogEvery 10 -Execute
.\02_gemma4_generation\edge_runbook_2b.ps1 -Action resume -Offset 0 -Limit 0 -CheckpointEvery 25 -LogEvery 10 -Execute
.\\02_gemma4_generation\\edge_runbook_2b.ps1 -Action stop -Execute
.\02_gemma4_generation\edge_runbook_2b.ps1 -Action resume -FastProfile -NoStartupCheck -Execute
.\\02_gemma4_generation\\edge_runbook_2b.ps1 -Action monitor -MonitorIntervalMinutes 10 -TargetRows 2000 -Execute
.\02_gemma4_generation\edge_runbook_2b.ps1 -Action finalize -Execute
```

Graceful stop behavior:

- Stop signal file is watched between rows.
- On stop detection, completed rows currently in memory are checkpoint-saved immediately.
- The in-flight row may be dropped, then process exits cleanly.
- Default stop signal path for edge 2b runbook:
  - `data/eval/gemma4_generation_edge_predictions_gemma4_2b.stop`
- Default heartbeat path for edge 2b runbook:
  - `data/eval/gemma4_generation_edge_predictions_gemma4_2b.heartbeat.json`

Direct stop-signal command:

```powershell
New-Item -ItemType File -Path .\data\eval\gemma4_generation_edge_predictions_gemma4_2b.stop -Force
```

Resume command after graceful stop:

```powershell
python .\02_gemma4_generation\run_generation_mvp.py --mode edge --backend transformers --model gemma4_2b --resume --append --profile fast_edge --no-startup-check --checkpoint-every 25 --log-every 10
```

Benchmark gate command:

```powershell
python .\02_gemma4_generation\benchmark_edge_2b.py --sample-size 20
```

Monitor command:

```powershell
python .\02_gemma4_generation\monitor_edge_progress.py --output-csv .\data\eval\gemma4_generation_edge_predictions_gemma4_2b.csv --target-rows 2000 --interval-minutes 10 --main-log-path .\logs\edge_2b_resume_20260406-143542.log --heartbeat-path .\data\eval\gemma4_generation_edge_predictions_gemma4_2b.heartbeat.json --benchmark-json-path .\logs\edge_2b_benchmark_20260406-140151.json
```

## Long-run prevention policy

To avoid "stuck" misreads and wasted GPU time on long edge generation:

1. Run benchmark gate first (`benchmark_edge_2b.py --sample-size 20`).
2. Use `--profile fast_edge` for full edge-set runs unless quality validation requires default profile.
3. Keep `--checkpoint-every` between 20 and 50 for resume safety with lower write overhead.
4. Keep `--log-every` enabled so progress and ETA stay visible during long runs.
5. Use `--resume --append` for every recovery run. Do not restart from scratch unless output contract changed.
6. Keep 10-minute monitor active during long runs to track output freshness, heartbeat freshness, benchmark state, recent latency window, and GPU utilization.

Detector verdicts:

- `RUNNING_HEALTHY`: process alive, freshness signals healthy
- `RUNNING_SLOW`: process alive, but recent latency window breached thresholds
- `STALL_WARNING`: process missing or stale signals suggest trouble, but not enough evidence to confirm a stall
- `STALL_CONFIRMED`: process missing and freshness signals stale beyond the confirmed threshold
- `BLOCKED_BY_GATE`: benchmark gate blocked the full run before generation started

Status command now reports:

- total rows
- max `source_row_index`
- remaining rows
- output/log/heartbeat freshness
- benchmark pass/fail
- detector verdict in one line

### 4. Compare 2B and 4B

```powershell
python .\02_gemma4_generation\compare_generation_runs.py --mode eval --left-model gemma4_2b --right-model gemma4_4b
```

## Smoke test commands

Recommended execution order:

```powershell
python .\02_gemma4_generation\build_generation_assets.py
python .\02_gemma4_generation\verify_local_inference_setup.py
python .\02_gemma4_generation\run_generation_mvp.py --backend transformers --model gemma4_2b --mode eval --limit 5
python .\02_gemma4_generation\evaluate_generation_mvp.py --mode eval --model gemma4_2b
python .\02_gemma4_generation\run_generation_mvp.py --backend transformers --model gemma4_4b --mode eval --limit 5
python .\02_gemma4_generation\evaluate_generation_mvp.py --mode eval --model gemma4_4b
python .\02_gemma4_generation\compare_generation_runs.py --mode eval --left-model gemma4_2b --right-model gemma4_4b
```

## Common failures

- `transformers` or `accelerate` missing
  - install the required packages in the active Python environment
- Hugging Face access or cache issue
  - confirm `hf_model_id` or `local_dir`
- first-token latency spike on small VRAM GPUs
  - confirm the runtime is using `device_map=auto`
  - use the default load-only startup probe first
  - inspect split timing fields before changing backend/runtime
- GPU memory pressure on `gemma4_4b`
  - validate `gemma4_2b` first
  - keep `gemma4_4b` as the second smoke target
- `llama_cpp` instability
  - `llama_cpp` is now experimental and is no longer the default runtime path

## Runtime assumptions

- Primary hardware target: NVIDIA GPU on Windows
- Primary runtime: `transformers`
- Default model IDs:
  - `gemma4_2b`
  - `gemma4_4b`
- `llama_cpp` remains available only as an experimental backend
