# Unattended Finetuning Preflight Report For baseline-gemma4-2b-r5

## Purpose
- Prepare a new stage 06 unattended finetuning run for `baseline-gemma4-2b-r5`.
- Validate readiness, frozen input alignment, run-id hygiene, optional contextual assets, manifest creation, and local environment checks before training.

## Input Files
- `data\qa\finetuning_prep\train_gemma4_2b.jsonl`
- `data\qa\finetuning_prep\valid_gemma4_2b.jsonl`
- `data\qa\finetuning_prep\training_candidates_gemma4_2b.csv`
- `data\qa\finetuning_prep\dataset_summary_gemma4_2b.json`
- `data\qa\finetuning_prep\stage06_readiness_gemma4_2b.json`
- `data\qa\finetuning_prep\holdout_grounded_generation.csv`
- `data\qa\finetuning_prep\holdout_edge_safety.csv`

## Output Files
- `data\qa\finetuning_runs\baseline-gemma4-2b-r5\preflight_summary.json`
- `data\qa\finetuning_runs\baseline-gemma4-2b-r5\launch_commands.md`
- `data\qa\finetuning_runs\baseline-gemma4-2b-r5\manifest.json`
- `data\qa\finetuning_runs\baseline-gemma4-2b-r5\config.json`
- `data\qa\finetuning_runs\baseline-gemma4-2b-r5\schema_v1.md`
- `data\qa\finetuning_runs\baseline-gemma4-2b-r5\train_contextual.jsonl`
- `data\qa\finetuning_runs\baseline-gemma4-2b-r5\valid_contextual.jsonl`
- `data\qa\finetuning_runs\baseline-gemma4-2b-r5\context_build_summary.json`

## Original Row And Column Counts
- `training_candidates_gemma4_2b.csv`: rows=3000, columns=29
- `holdout_grounded_generation.csv`: rows=200, columns=29
- `holdout_edge_safety.csv`: rows=150, columns=29

## Final Row And Column Counts
- `train_gemma4_2b.jsonl`: rows=889
- `valid_gemma4_2b.jsonl`: rows=99
- `train_contextual.jsonl`: rows=756
- `valid_contextual.jsonl`: rows=84

## Removed Exact Duplicate Count
- Not applicable in this preflight step. No train/valid/holdout source datasets were rewritten.

## Newly Generated Columns
- None. This step validates contracts and prepares run-local artifacts only.

## Processing Steps Summary
- Refreshed the stage 06 readiness verdict and rewrote `stage06_readiness_<model>.json` in the canonical stage-05 output directory.
- Verified train/valid JSONL schema plus content-level alignment against the current `training_candidates` train/valid splits.
- Prepared run-local assets using context mode `contextual`.
- Probed both tokenizer load and training-compatible model load before reserving the run with a manifest.
- Validated write access against the actual target run directory `data\qa\finetuning_runs\baseline-gemma4-2b-r5`.
- Stage 06 unattended tooling supports only project-local run directories under the repository root.
- Created a new run manifest, config template, summary, launch commands, and report only after all blocking checks passed.

## Major Warnings Or Exceptions
- Manual check required: Disable sleep or hibernation before unattended training.
- Manual check required: Prevent automatic restart from Windows Update for the unattended window.
- Manual check required: Keep the terminal session pinned to a stable shell or runner.

## Recovery Notes
- Run-id reusable now: `False`
- Blocking reason: `reserved_run_artifacts_present:config.json,manifest.json`
- Current run-dir artifacts: `config.json, context_build_summary.json, manifest.json, schema_v1.md, train_contextual.jsonl, valid_contextual.jsonl`
- Validated write target: `data\qa\finetuning_runs\baseline-gemma4-2b-r5`
- Environment failure before reservation: Fix the local environment and rerun the same run-id. No manifest or training reservation should exist yet.
- Stale contextual assets in frozen mode: Choose a new run-id or remove the contextual preflight-owned artifacts before retrying frozen mode.
- Reserved run artifacts already present: Do not reuse this run-id. Pick a new run-id for a new unattended run.

## Sample Outputs
- Readiness verdict: `GO`
- Manifest input mode: `accepted_contextual`
- GPU available: `True`
- GPU free memory (GB): `6.96`
- Tokenizer probe: `True`
- Model probe: `True`
- Train command: `C:\Users\lwwde\miniconda3\envs\py312\python.exe G:\GitProjects\New_Local_GPT_Chetbot\06_finetuning\train_finetuning_baseline.py --run-id baseline-gemma4-2b-r5 --model gemma4_2b --max-seq-length 512 --training-scope gates_and_norms`
