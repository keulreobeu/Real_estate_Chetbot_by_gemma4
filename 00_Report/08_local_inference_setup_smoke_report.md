# 08 Local Inference Setup Smoke Report

## Purpose

Validate that `02_gemma4_generation` is ready for real local Gemma 4 inference setup and smoke testing.

## Input files

- `02_gemma4_generation/config/generation_defaults.json`
- `02_gemma4_generation/config/models.local.example.json`
- `02_gemma4_generation/config/models.local.json`
- `data/apartment_chatbot_v3.csv`
- `data/qa/evaluation_dataset.csv`

## Output files

- `02_gemma4_generation/verify_local_inference_setup.py`
- updated `02_gemma4_generation/README.md`

## Original row and column counts

- main dataset: not re-counted in this setup validation step
- eval dataset: not re-counted in this setup validation step

## Final row and column counts

- no dataset transformation executed

## Removed exact duplicate count

- not applicable

## Newly generated columns

- none

## Processing steps summary

1. Added a local inference readiness checker.
2. Added a local `models.local.json` file from the example contract for runtime setup.
3. Improved `run_generation_mvp.py` runtime error messages for config, runtime, import, and model file failures.
4. Updated setup documentation with prerequisites, readiness checks, smoke commands, and common failures.
5. Ran compile validation and readiness checks.

## Validation performed

- `python -m compileall .\02_gemma4_generation`
- `python .\02_gemma4_generation\verify_local_inference_setup.py`
- `python .\02_gemma4_generation\verify_local_inference_setup.py --model gemma4_2b`
- `python .\02_gemma4_generation\run_generation_mvp.py --backend llama_cpp --model gemma4_2b --mode eval --limit 1`

## Major warnings or exceptions

- `llama-cpp-python` is not installed in the current `py312` environment.
- Attempting to install `llama-cpp-python` failed because Windows build tools were not available, with `nmake` and C/C++ compiler errors.
- `gemma4_2b` GGUF file was not found at `%LOCALAPPDATA%\llm_models\gemma4\2b\model.gguf`.
- `gemma4_4b` GGUF file was not found at `%LOCALAPPDATA%\llm_models\gemma4\4b\model.gguf`.
- Real `llama_cpp` smoke execution is still blocked by local environment setup, not by repository code structure.

## Sample outputs

Readiness checker result excerpt:

```text
FAIL llama-cpp-python is not installed.
PASS local model config found.
FAIL gemma4_2b model file not found.
FAIL gemma4_4b model file not found.
NOT_READY fix the failed items above before running llama_cpp smoke tests.
```
