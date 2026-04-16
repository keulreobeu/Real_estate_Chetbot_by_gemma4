# 06_finetuning

## Purpose

This stage owns the first real finetuning run after `stage06_readiness = GO`.

The goal of this folder is not to regenerate data. Upstream data generation and gating stay in:

- `03_generation_optimization`
- `05_finetuning_prep`

This stage starts only after the latest readiness snapshot is `GO` and the input artifacts are frozen.

## Entry Gate

Start this stage only when all of the following are true:

- `data/qa/finetuning_prep/stage06_readiness_gemma4_2b.json` has `verdict = GO`
- `data/qa/finetuning_prep/train_gemma4_2b.jsonl` exists
- `data/qa/finetuning_prep/valid_gemma4_2b.jsonl` exists
- `data/qa/finetuning_prep/holdout_grounded_generation.csv` exists
- `data/qa/finetuning_prep/holdout_edge_safety.csv` exists
- the canonical edge snapshot and stage 05 outputs come from the same evaluation cycle

## Inputs

- `data/qa/finetuning_prep/train_gemma4_2b.jsonl`
- `data/qa/finetuning_prep/valid_gemma4_2b.jsonl`
- `data/qa/finetuning_prep/holdout_grounded_generation.csv`
- `data/qa/finetuning_prep/holdout_edge_safety.csv`
- `data/qa/finetuning_prep/dataset_summary_gemma4_2b.json`
- `data/qa/finetuning_prep/stage06_readiness_gemma4_2b.json`
- `data/eval/gemma4_generation_edge_metrics_gemma4_2b.json`

## Outputs

Expected run artifacts for each finetuning attempt should live under a run-specific directory such as:

- `data/qa/finetuning_runs/<run_id>/config.json`
- `data/qa/finetuning_runs/<run_id>/train.log`
- `data/qa/finetuning_runs/<run_id>/eval_summary.json`
- `data/qa/finetuning_runs/<run_id>/post_train_holdout_results.json`
- `data/qa/finetuning_runs/<run_id>/notes.md`

Do not overwrite one finetuning run with another. Each run should use a unique run id.

## Stage Commands

Create a frozen run manifest before training:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\create_run_manifest.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b
```

Run the first baseline finetuning job:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\train_finetuning_baseline.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b --max-seq-length 512 --training-scope gates_and_norms
```

Generate post-train valid and holdout prediction sets:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\generate_post_train_prediction_sets.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b
```

Evaluate a completed post-train run against the frozen baseline:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\evaluate_post_finetuning_run.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b
```

## Dependencies

- upstream readiness must remain `GO`
- train/valid/holdout artifacts must remain frozen during the run
- model choice, adapter method, and hyperparameters must be recorded before execution

## Validation Method

Before a real run:

1. confirm the readiness file is still `GO`
2. confirm train/valid row counts match the frozen dataset summary
3. record the source artifact paths in the run manifest
4. request user approval before starting the actual training command

After a real run:

1. evaluate on valid
2. evaluate on grounded holdout
3. evaluate on edge safety holdout
4. compare against the pre-finetuning baseline
5. write a report to `00_Report`

The post-train evaluator expects these run artifacts:

- `valid_predictions.csv`
- `grounded_holdout_predictions.csv`
- `edge_safety_holdout_predictions.csv`
- `train.log`
- `config.json`
- `manifest.json`
- `checkpoints/`
- `final/`

The baseline trainer assumes:

- local Gemma model files are already available through `02_gemma4_generation/config/models.local.json`
- training uses `transformers` + `torch` only
- the first run is a conservative memory-adaptive partial finetune on `gates_and_norms`, not a sweep
- post-train prediction generation uses row metadata plus cited apartment document context from `data/apartment_chatbot_v3.csv`

## Current Status

- stage entry is now unlocked because the latest readiness snapshot is `GO`
- actual training execution is not started in this folder yet
- next work should add the first run-specific config and execution wrapper only after explicit user approval
