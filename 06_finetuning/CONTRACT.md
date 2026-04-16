# 06_finetuning Contract

## Frozen Inputs

The first finetuning run must record these exact inputs:

- canonical edge predictions path
- `train_gemma4_2b.jsonl`
- `valid_gemma4_2b.jsonl`
- `holdout_grounded_generation.csv`
- `holdout_edge_safety.csv`
- `dataset_summary_gemma4_2b.json`
- `stage06_readiness_gemma4_2b.json`

## Required Metadata

Each finetuning run must capture:

- `run_id`
- `base_model_id`
- `training_method`
- `adapter_or_full_finetune`
- `train_file`
- `valid_file`
- `grounded_holdout_file`
- `edge_safety_holdout_file`
- `start_time`
- `end_time`
- `operator_note`

## Required Safety Rules

- never mutate stage 05 input JSONL files in place
- never replace holdout files with post-train outputs
- never evaluate the trained model against train rows and call it a gate pass
- never start a run without a run-specific output directory
- never start a training command without explicit user approval

## Minimum Post-Run Outputs

Each run must produce:

- a saved config snapshot
- a training log
- a validation summary
- a grounded holdout summary
- an edge safety holdout summary
- a short markdown note describing result quality and next action

Prediction CSVs must keep the generation-style contract fields required for comparison:

- `source_row_index`
- `question`
- `answer`
- `answer_type`
- `match_status`
- `query_type`

Recommended additional fields for subset comparison:

- `top_doc_id`
- `cited_doc_ids`
- `used_fields`
- `insufficient_context`

## Gate To Continue

The first real training run can begin only after:

- this contract is accepted
- the runbook is filled with the chosen method and command
- the output directory is defined
- the user approves the run
