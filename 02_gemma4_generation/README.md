# 02_gemma4_generation

## Purpose

`02_gemma4_generation` is the next planned stage after `01_preprocessing`.

This stage builds the smallest practical MVP for a retrieval-first Gemma 4 real-estate chatbot. The goal is to answer apartment questions using only repository data, return grounded Korean answers, and avoid hallucination when evidence is weak.

This stage is intentionally narrow:

- single-turn question answering
- retrieval-first pipeline
- grounded answer generation
- offline evaluation against existing evaluation assets

Out of scope for this stage:

- fine-tuning
- web serving
- multi-turn memory
- external live data
- recommendation or price prediction

## Inputs

Primary inputs from `01_preprocessing`:

- `data/apartment_chatbot_v3.csv`
- `data/qa/evaluation_dataset.csv`
- `data/qa/edge_case_eval.csv`

## Outputs

Default stage outputs:

- `data/eval/gemma4_generation_source_index.csv`
- `data/eval/gemma4_generation_eval_predictions_<model_id>.csv`
- `data/eval/gemma4_generation_edge_predictions_<model_id>.csv`
- `data/eval/gemma4_generation_eval_metrics_<model_id>.json`
- `data/eval/gemma4_generation_edge_metrics_<model_id>.json`
- `00_Report/06_gemma4_generation_stage_blueprint.md`
- `00_Report/07_gemma4_generation_model_comparison.md`

## Folder structure

```text
02_gemma4_generation/
  README.md
  CONTRACT.md
  common.py
  build_generation_assets.py
  run_generation_mvp.py
  evaluate_generation_mvp.py
  compare_generation_runs.py
  inference/
    base.py
    mock_adapter.py
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
3. Retrieve top-k rows for each question using a lightweight lexical baseline.
4. Build a grounded prompt from retrieved evidence.
5. Generate an answer through a backend adapter.
6. Save predictions and evaluate them offline.

## Scripts

### 1. Build retrieval assets

```powershell
python .\02_gemma4_generation\build_generation_assets.py
```

What it does:

- validates required upstream files and columns
- creates a retrieval-ready source index
- writes `data/eval/gemma4_generation_source_index.csv`

### 2. Run the MVP generation flow

```powershell
python .\02_gemma4_generation\run_generation_mvp.py --mode eval --backend mock --model gemma4_2b
python .\02_gemma4_generation\run_generation_mvp.py --mode edge --backend mock --model gemma4_2b
python .\02_gemma4_generation\run_generation_mvp.py --mode eval --backend llama_cpp --model gemma4_2b
python .\02_gemma4_generation\run_generation_mvp.py --mode eval --backend llama_cpp --model gemma4_4b
```

What it does:

- loads the source index
- retrieves top-k evidence rows for each question
- builds prompts
- produces grounded answer candidates

The default `mock` backend is included for regression checks and path validation.

The local runtime backend is `llama_cpp` and expects GGUF model files outside the repository.

Package note:

- local inference requires `llama-cpp-python`
- the package is imported lazily, so `mock` runs still work without it

Recommended local config flow:

1. Copy `02_gemma4_generation/config/models.local.example.json` to `02_gemma4_generation/config/models.local.json`
2. Update each `model_path`
3. Keep the real file uncommitted

### 3. Evaluate generation outputs

```powershell
python .\02_gemma4_generation\evaluate_generation_mvp.py --mode eval --model gemma4_2b
python .\02_gemma4_generation\evaluate_generation_mvp.py --mode edge --model gemma4_2b
```

What it does:

- checks retrieval hit rate
- checks simple answer overlap metrics
- writes metrics JSON files for both eval sets

### 4. Compare 2B and 4B

```powershell
python .\02_gemma4_generation\compare_generation_runs.py --mode eval --left-model gemma4_2b --right-model gemma4_4b
```

What it does:

- loads two metrics files
- writes a small markdown comparison report

## Backend plan

Recommended backend sequence:

1. `mock`
2. `llama_cpp` local Gemma 4 adapter
3. later, optional prompt and retrieval improvements

The point of the `mock` backend is not realism. It is to prove the stage contract, file paths, retrieval flow, and evaluation loop before adding model runtime complexity.

## Runtime assumptions

- Primary hardware target: NVIDIA GPU
- Model format: GGUF
- Runtime: `llama.cpp` via `llama-cpp-python`
- Default model IDs:
  - `gemma4_2b`
  - `gemma4_4b`
