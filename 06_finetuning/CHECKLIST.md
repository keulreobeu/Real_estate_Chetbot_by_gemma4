# 06_finetuning Checklist

## Before Schema Lock

- unified schema choice is confirmed
- stage 06 only scope is confirmed
- `max_seq_length = 512` is confirmed
- stage 05 canonical artifacts remain untouched

## Before Run

- readiness verdict is `GO`
- training candidates file exists
- grounded holdout exists
- edge safety holdout exists
- apartment doc source exists
- output directory for this run is unique

## Before Training

- `schema_v1.md` exists
- `train_contextual.jsonl` exists
- `valid_contextual.jsonl` exists
- `context_build_summary.json` exists
- train and valid contextual row counts plus excluded no-context rows reconcile to the frozen split
- no empty input rows
- no empty output rows
- no contextual row exceeds the full training sequence budget
- excluded no-context rows are recorded in the summary
- `builder_pass = true`
- `schema_status = accepted`
- `validation_version = r3_full_sequence_v3`
- user approval for the training command is recorded

## During Run

- training log is being written
- checkpoint path is stable
- no upstream stage 03 or 05 regeneration is modifying frozen inputs

## Before Prediction Generation

- `final/` exists
- prediction prompt uses the context-aware schema family
- disclosure rules are present for required rows
- user approval for prediction generation is recorded

## After Run

- `valid_predictions.csv` exists
- `grounded_holdout_predictions.csv` exists
- `edge_safety_holdout_predictions.csv` exists
- `valid_eval.json` exists
- `grounded_holdout_eval.json` exists
- `edge_safety_holdout_eval.json` exists
- `post_train_summary.json` exists
- run note is written
- next action is decided and recorded
