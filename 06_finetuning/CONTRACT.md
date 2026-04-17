# 06_finetuning Contract

## Frozen Inputs

Each stage 06 run must record these canonical frozen inputs:

- `train_gemma4_2b.jsonl`
- `valid_gemma4_2b.jsonl`
- `training_candidates_gemma4_2b.csv`
- `holdout_grounded_generation.csv`
- `holdout_edge_safety.csv`
- `dataset_summary_gemma4_2b.json`
- `stage06_readiness_gemma4_2b.json`
- `data/apartment_chatbot_v3.csv`
- baseline evaluation and edge metrics JSON

## Run-Local Inputs

`baseline-gemma4-2b-r3` may additionally use run-local derived inputs:

- `schema_v1.md`
- `train_contextual.jsonl`
- `valid_contextual.jsonl`
- `context_build_summary.json`

These are derived artifacts. They do not replace or rewrite the frozen stage 05 files.

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
- `operator_note`

Recommended additional metadata:

- `schema_version`
- `max_docs`
- `max_description_chars`
- `prompt_token_budget`

## Context-Aware Schema v1

The unified contextual SFT record format is:

```json
{"instruction":"...","input":"...","output":"..."}
```

Field expectations:

- `instruction`
  - role and high-level behavioral instruction
  - must reflect row contract and safety/disclosure intent
- `input`
  - `[ROW CONTRACT]`
  - `[QUESTION]`
  - `[CONTEXT]`
  - `[OUTPUT RULES]`
- `output`
  - the frozen answer text from the selected row

Schema limits:

- at most 2 cited docs
- clipped grounded descriptions
- no raw full apartment row dump
- no long keyword fields
- accepted r3 defaults are `max_docs = 1`, `max_description_chars = 12`
- full training sequence budget must fit within `max_seq_length = 512`
- rows with no resolvable cited document are excluded from the contextual train/valid build and must be reported in the summary

## Required Safety Rules

- never mutate stage 05 canonical JSONL or CSV files in place
- never replace holdout files with post-train outputs
- never evaluate train rows and call it a gate pass
- never start a run without a run-specific output directory
- never start GPU training without explicit user approval
- never overwrite an existing run directory without explicit user approval

## Prediction Contract

Prediction CSVs must preserve the comparison fields used by the post-train gate:

- `source_row_index`
- `question`
- `answer`
- `answer_type`
- `match_status`
- `query_type`

Recommended additional fields:

- `top_doc_id`
- `cited_doc_ids`
- `used_fields`
- `must_not_recommend`
- `must_disclose_limit`
- `insufficient_context`

Prediction prompts must remain context-aware:

- include row contract
- include cited doc context from `data/apartment_chatbot_v3.csv`
- include disclosure instruction when required
- block recommendation behavior when required
- use the same accepted schema budget defaults as the contextual builder unless explicitly overridden

## Minimum Post-Run Outputs

Each run must produce:

- `config.json`
- `manifest.json`
- `train.log`
- `checkpoints/`
- `final/`
- `valid_predictions.csv`
- `grounded_holdout_predictions.csv`
- `edge_safety_holdout_predictions.csv`
- `valid_eval.json`
- `grounded_holdout_eval.json`
- `edge_safety_holdout_eval.json`
- `post_train_summary.json`
- `notes.md`

## Gate To Continue

The next run can begin only after:

- the schema is locked
- the run-local contextual view validates
- the manifest records the selected train/valid paths
- the output directory is unique
- the user approves the run
