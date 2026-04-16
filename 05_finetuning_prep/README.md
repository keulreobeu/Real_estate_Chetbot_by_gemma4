# 05_finetuning_prep

## Purpose

`05_finetuning_prep` prepares the first SFT-ready dataset from the completed `02_gemma4_generation` and `04_evaluation` outputs.

This stage does not retrain the model yet. It separates:

- weakly useful deterministic outputs
- grounded generation outputs
- safety/refusal examples
- recommendation examples that satisfy the contract

## Inputs

- `data/eval/gemma4_generation_edge_predictions_<model_id>.csv`
- `data/eval/gemma4_generation_eval_predictions_<model_id>.csv`
- `data/qa/edge_case_eval.csv`
- `data/qa/evaluation_dataset.csv`

## Outputs

Generated artifacts are written to `data/qa/finetuning_prep/`.

- `training_candidates_<model_id>.csv`
- `training_rejected_<model_id>.csv`
- `holdout_grounded_generation.csv`
- `holdout_edge_safety.csv`
- `train_<model_id>.jsonl`
- `valid_<model_id>.jsonl`
- `dataset_summary_<model_id>.json`

## Current first-cycle policy

- exclude deterministic `apartment_fact_lookup`
- exclude `meta_answer`, `knowledge_answer`, and `fallback_answer`
- exclude legacy rows that do not have `answer_type`, `match_status`, and `query_type`
- keep `grounded_generation` as the core train bucket
- keep refusal/safety rows in a smaller curated subset
- keep recommendation rows only when they satisfy the exact-match and safety constraints
- shrink grounded/safety holdouts dynamically when the candidate pool is small
- preserve a minimum train/valid pool before final holdout allocation
- current low-supply floor keeps at least `train>=50`, `valid>=10` whenever the surviving candidate pool allows it

## Command

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\05_finetuning_prep\prepare_sft_dataset.py --model gemma4_2b
```

### 06 readiness gate

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\05_finetuning_prep\validate_06_readiness.py --model gemma4_2b
```

### Overnight resume to readiness

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run_overnight_stage05_to_06.ps1
```

This unattended worker resumes the staging edge regeneration checkpoint, promotes the completed output to canonical,
then re-runs evaluation, finetuning prep, and the 06 readiness gate in one pass.

## Validation

Recommended minimal checks after a run:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' -m compileall .\05_finetuning_prep\prepare_sft_dataset.py
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\05_finetuning_prep\prepare_sft_dataset.py --model gemma4_2b
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\05_finetuning_prep\validate_06_readiness.py --model gemma4_2b
```
