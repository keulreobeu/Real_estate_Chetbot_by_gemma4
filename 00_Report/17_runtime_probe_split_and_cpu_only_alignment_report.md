# 17. Runtime Probe Split And CPU-Only Alignment Report

## Purpose

Implement the first non-GPU portion of the RTX 3060 Ti Gemma 4 E2B runtime bottleneck mitigation plan.

This run focused on:

- splitting runtime timing inside the `transformers` adapter
- separating startup health checks into load-only vs full probe semantics
- exposing runtime debug metadata in the web MVP and prediction outputs
- updating stage documentation without running new GPU workloads

## Input Files

- `02_gemma4_generation/inference/transformers_adapter.py`
- `02_gemma4_generation/run_generation_mvp.py`
- `02_gemma4_generation/demo_chatbot_web_mvp.py`
- `02_gemma4_generation/README.md`
- `02_gemma4_generation/CONTRACT.md`
- `02_gemma4_generation/config/models.local.json`

## Output Files

- `02_gemma4_generation/inference/transformers_adapter.py`
- `02_gemma4_generation/run_generation_mvp.py`
- `02_gemma4_generation/demo_chatbot_web_mvp.py`
- `02_gemma4_generation/README.md`
- `02_gemma4_generation/CONTRACT.md`
- `00_Report/17_runtime_probe_split_and_cpu_only_alignment_report.md`

## Original Row And Column Counts

Not applicable for this run because no preprocessing dataset regeneration was performed.

## Final Row And Column Counts

Not applicable for this run because no preprocessing dataset regeneration was performed.

## Removed Exact Duplicate Count

Not applicable.

## Newly Generated Columns

Prediction CSV runtime debug columns added by `run_generation_mvp.py`:

- `load_runtime_ms`
- `processor_load_ms`
- `model_load_ms`
- `prompt_render_ms`
- `input_prepare_ms`
- `to_device_ms`
- `generate_ms`
- `decode_ms`
- `model_device`
- `hf_device_map`
- `device_map_requested`
- `model_source`
- `processor_source`
- `local_files_only`

## Processing Steps Summary

1. Kept the default local `gemma4_2b` runtime on `device_map="auto"`.
2. Extended the `transformers` adapter to emit split timing and runtime metadata in `raw_response`.
3. Updated `run_generation_mvp.py` to:
   - treat `--startup-check` as load-only by default
   - support `--startup-check-full` for a short generation probe
   - persist runtime timing/debug fields into prediction CSV rows
4. Reworked `demo_chatbot_web_mvp.py` so the web demo:
   - exposes `device_map`, `model_source`, `last_load_ms`, `last_generate_ms`
   - performs a load-only runtime probe on startup when supported
5. Updated stage documentation and contract notes for the new probe semantics and runtime debug fields.

## Major Warnings Or Exceptions

- GPU-backed validation was intentionally skipped in this run because GPU work was paused by request.
- As a result, this run confirms code structure and non-GPU safety only. It does not newly confirm live `transformers` runtime behavior.
- Console rendering on this Windows environment may still display some Korean strings as mojibake even when the underlying UTF-8 file content is correct.

## Validation Method

Validation performed without GPU execution:

- `python -m py_compile` on:
  - `02_gemma4_generation/inference/transformers_adapter.py`
  - `02_gemma4_generation/run_generation_mvp.py`
  - `02_gemma4_generation/demo_chatbot_web_mvp.py`
- manual source inspection of:
  - startup probe path
  - prediction record schema
  - web response schema
  - updated README/CONTRACT sections

## Sample Outputs

Expected startup probe modes after this change:

- default:
  - `--startup-check` -> `load_only`
- explicit generation probe:
  - `--startup-check --startup-check-full` -> `full`

Expected new web response fields:

- `device_map`
- `model_source`
- `last_load_ms`
- `last_generate_ms`

Expected new prediction CSV fields:

- `load_runtime_ms`
- `processor_load_ms`
- `model_load_ms`
- `prompt_render_ms`
- `input_prepare_ms`
- `to_device_ms`
- `generate_ms`
- `decode_ms`

## Follow-up Items

1. Run non-mock `transformers` smoke validation once GPU work resumes.
2. Confirm the load-only startup probe reduces cold-start pain in the real runtime path.
3. Compare warm web requests against one-shot CLI requests after the GPU path is re-enabled.
