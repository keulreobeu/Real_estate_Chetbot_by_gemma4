# 06_finetuning

## Purpose

This stage owns post-readiness finetuning runs after `stage06_readiness = GO`.

The current active loop is:

```text
schema lock
-> run-local contextual view build
-> manifest creation
-> memory-safe finetune
-> post-train prediction generation
-> post-train gate
```

Stage 06 does not regenerate canonical stage 05 artifacts. It reads frozen inputs and
builds run-local derivatives when needed.

For `r3`, rows with no resolvable cited apartment document are excluded from the
run-local contextual train/valid view instead of training on `ctx=none`.

## Entry Gate

Start this stage only when all of the following are true:

- `data/qa/finetuning_prep/stage06_readiness_gemma4_2b.json` has `verdict = GO`
- `data/qa/finetuning_prep/training_candidates_gemma4_2b.csv` exists
- `data/qa/finetuning_prep/holdout_grounded_generation.csv` exists
- `data/qa/finetuning_prep/holdout_edge_safety.csv` exists
- `data/apartment_chatbot_v3.csv` exists
- the canonical baseline metrics and stage 05 artifacts come from the same frozen cycle

## R3 Context-Aware Direction

`baseline-gemma4-2b-r3` is a schema-first round.

Key decisions:

- one unified compressed schema
- run-local contextual JSONL inside the run directory
- `gates_and_norms` partial finetune as the default local path
- `max_seq_length = 512`
- default contextual compression:
  - `max_docs = 1`
  - `max_description_chars = 12`
- evaluator stays unchanged for this round

## Inputs

Canonical frozen inputs:

- `data/qa/finetuning_prep/train_gemma4_2b.jsonl`
- `data/qa/finetuning_prep/valid_gemma4_2b.jsonl`
- `data/qa/finetuning_prep/training_candidates_gemma4_2b.csv`
- `data/qa/finetuning_prep/holdout_grounded_generation.csv`
- `data/qa/finetuning_prep/holdout_edge_safety.csv`
- `data/qa/finetuning_prep/dataset_summary_gemma4_2b.json`
- `data/qa/finetuning_prep/stage06_readiness_gemma4_2b.json`
- `data/apartment_chatbot_v3.csv`

Optional run-local contextual inputs:

- `data/qa/finetuning_runs/<run_id>/schema_v1.md`
- `data/qa/finetuning_runs/<run_id>/train_contextual.jsonl`
- `data/qa/finetuning_runs/<run_id>/valid_contextual.jsonl`
- `data/qa/finetuning_runs/<run_id>/context_build_summary.json`

## Outputs

Each run keeps its own directory:

- `data/qa/finetuning_runs/<run_id>/manifest.json`
- `data/qa/finetuning_runs/<run_id>/config.json`
- `data/qa/finetuning_runs/<run_id>/train.log`
- `data/qa/finetuning_runs/<run_id>/checkpoints/`
- `data/qa/finetuning_runs/<run_id>/final/`
- `data/qa/finetuning_runs/<run_id>/valid_predictions.csv`
- `data/qa/finetuning_runs/<run_id>/grounded_holdout_predictions.csv`
- `data/qa/finetuning_runs/<run_id>/edge_safety_holdout_predictions.csv`
- `data/qa/finetuning_runs/<run_id>/valid_eval.json`
- `data/qa/finetuning_runs/<run_id>/grounded_holdout_eval.json`
- `data/qa/finetuning_runs/<run_id>/edge_safety_holdout_eval.json`
- `data/qa/finetuning_runs/<run_id>/post_train_summary.json`
- `data/qa/finetuning_runs/<run_id>/notes.md`

## Stage Commands

Prepare a new unattended finetuning run with readiness refresh, input contract checks,
optional contextual asset build, manifest creation, environment checks, and launch commands:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\prepare_unattended_finetuning_run.py --run-id baseline-gemma4-2b-r5 --model gemma4_2b --context-mode contextual --training-scope gates_and_norms --max-seq-length 512
```

Stage 06 unattended tooling currently supports only project-local run directories under
the repository root. Do not point `--run-dir` or `--output-dir` outside this repository.

This writes:

- `data/qa/finetuning_runs/<run_id>/preflight_summary.json`
- `data/qa/finetuning_runs/<run_id>/launch_commands.md`
- a numbered preflight report in `00_Report/`

Build the run-local context-aware train/valid view:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\build_contextual_training_view.py --run-id baseline-gemma4-2b-r3 --model gemma4_2b
```

Create a run manifest. Use `--context-mode auto` to preserve the existing selection behavior,
`--context-mode contextual` to require accepted run-local contextual assets, or `--context-mode frozen`
to force the stage-05 frozen JSONL. Manifest output paths are also restricted to project-local
run directories:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\create_run_manifest.py --run-id baseline-gemma4-2b-r3 --model gemma4_2b --context-mode contextual
```

Run the memory-safe finetuning job:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\train_finetuning_baseline.py --run-id baseline-gemma4-2b-r3 --model gemma4_2b --max-seq-length 512 --training-scope gates_and_norms
```

Generate context-aware valid and holdout prediction sets:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\generate_post_train_prediction_sets.py --run-id baseline-gemma4-2b-r3 --model gemma4_2b
```

To stop prediction generation safely mid-run, create:

- `data/qa/finetuning_runs/<run_id>/prediction_generation.stop`

Evaluate the completed run:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\evaluate_post_finetuning_run.py --run-id baseline-gemma4-2b-r3 --model gemma4_2b
```

To stop gate evaluation between subsets, create:

- `data/qa/finetuning_runs/<run_id>/post_train_gate.stop`

## Validation Method

Before a real run:

1. confirm readiness is still `GO`
2. lock `schema_v1.md`
3. build run-local contextual train/valid JSONL
4. verify contextual row counts plus excluded no-context rows reconcile to the frozen split
5. verify no empty input or output rows
6. verify full training sequence budget is respected
7. verify `context_build_summary.json` reports `builder_pass = true` and `schema_status = accepted`
8. request user approval before training

After a real run:

1. verify final artifact exists
2. generate valid, grounded holdout, and edge safety predictions
3. run the post-train gate
4. compare against both the pre-finetuning baseline and `r2`
5. write a run report to `00_Report`

## Approval Boundary

Execution-time approval is still required before:

- creating or updating run output directories
- GPU training
- prediction generation
- post-train gate execution
- deleting or overwriting existing run artifacts

## Current Status

- `baseline-gemma4-2b-r2` remains `experiment_only`
- the next active target is `baseline-gemma4-2b-r3`
- the current bottleneck is stage 06 input contract quality, not basic training viability
