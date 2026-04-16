# 06_finetuning Runbook

## Goal

Run one baseline finetuning job from the frozen stage 05 dataset and evaluate it before any sweep or second run.

## Pre-Run Sequence

1. confirm the latest readiness verdict is still `GO`
2. copy the exact source artifact paths into the run manifest
3. create the run manifest
4. choose one baseline training method
5. choose one output directory
6. request user approval for the actual training command

## Baseline Recommendation

Use one conservative baseline only:

- one base model
- one adapter strategy
- one train split
- one validation split
- no multi-run sweep
- on the local 8GB GPU, default to `gates_and_norms` partial finetuning instead of full finetuning

This keeps the first run attributable and easy to compare against the pre-finetuning baseline.

## Suggested Run Manifest

Create a run-specific note with:

- run id
- date
- source dataset summary path
- readiness path
- chosen model
- chosen training method
- output directory
- expected evaluation outputs

## Post-Run Sequence

1. collect the train log
2. save `valid_predictions.csv`, `grounded_holdout_predictions.csv`, and `edge_safety_holdout_predictions.csv`
3. run `generate_post_train_prediction_sets.py`
4. make sure prediction generation used row metadata and cited document context, not just the raw question
5. run `evaluate_post_finetuning_run.py`
6. summarize valid performance
7. summarize grounded holdout performance
8. summarize edge safety holdout performance
9. compare with pre-finetuning baseline
10. decide one of:
   - promote this run as the new baseline
   - keep as experiment only
   - return to `03/05` if safety or grounded quality regresses

## Stop Conditions

Stop the run and report if:

- the frozen input files do not match the manifest
- the output directory already exists with another run id
- the training process cannot write logs or checkpoints
- post-train evaluation cannot be produced
