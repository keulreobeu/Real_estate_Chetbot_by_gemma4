# 33 R3 Pre-GPU Setup Report

## Purpose

Prepare `baseline-gemma4-2b-r3` up to the last non-GPU step:

- build the run-local context-aware train/valid view
- validate prompt budget
- create the run manifest and config

## Input Files

- `data/qa/finetuning_prep/training_candidates_gemma4_2b.csv`
- `data/qa/finetuning_prep/dataset_summary_gemma4_2b.json`
- `data/qa/finetuning_prep/stage06_readiness_gemma4_2b.json`
- `data/qa/finetuning_prep/holdout_grounded_generation.csv`
- `data/qa/finetuning_prep/holdout_edge_safety.csv`
- `data/apartment_chatbot_v3.csv`

## Output Files

- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/schema_v1.md`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/train_contextual.jsonl`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/valid_contextual.jsonl`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/context_build_summary.json`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/manifest.json`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/config.json`

## Original Row And Column Counts

Frozen stage 05 reference counts:

- train rows: `889`
- valid rows: `99`
- grounded holdout rows: `200`
- edge safety holdout rows: `150`

## Final Row And Column Counts

Run-local contextual views:

- `train_contextual.jsonl`: `889` rows
- `valid_contextual.jsonl`: `99` rows

## Removed Exact Duplicate Count

- none

## Newly Generated Columns

Generated JSONL keys:

- `instruction`
- `input`
- `output`

## Processing Steps Summary

1. Tried the initial schema-default compression:
   - `max_docs=2`
   - `max_description_chars=180`
2. Validation failed because prompt budget overflowed heavily:
   - train prompt average: `530.37`
   - train prompt max: `681`
   - rows over budget: `818`
3. Tightened the pre-GPU compression for the run-local view:
   - `max_docs=1`
   - `max_description_chars=96`
4. Rebuilt the contextual train/valid JSONL.
5. Recreated the manifest so it selects the contextual JSONL instead of canonical stage 05 JSONL.

## Major Warnings Or Exceptions

- The first builder attempt exceeded the `512` token prompt budget and was intentionally rejected.
- The accepted pre-GPU setup uses a more aggressive compression profile than the original schema draft.
- `doc_count_distribution` still shows missing doc context on some rows:
  - train: `133` rows with `0` resolved docs
  - valid: `15` rows with `0` resolved docs
- GPU training, prediction generation, and gate evaluation were not run in this step.

## Validation Performed

- Context build summary reviewed after rebuild
- Run-local file existence checked
- Manifest selection checked

Validated final prompt stats:

- train:
  - prompt min: `206`
  - prompt max: `437`
  - prompt avg: `360.23`
  - rows over budget: `0`
- valid:
  - prompt min: `230`
  - prompt max: `430`
  - prompt avg: `359.65`
  - rows over budget: `0`

Manifest selection validated:

- selected train file: `data/qa/finetuning_runs/baseline-gemma4-2b-r3/train_contextual.jsonl`
- selected valid file: `data/qa/finetuning_runs/baseline-gemma4-2b-r3/valid_contextual.jsonl`

## Sample Outputs

Run directory:

- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/manifest.json`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/config.json`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/train_contextual.jsonl`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/valid_contextual.jsonl`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/schema_v1.md`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r3/context_build_summary.json`

## Next Action

The next step is explicit approval for GPU training:

```powershell
& 'C:\Users\lwwde\miniconda3\envs\py312\python.exe' .\06_finetuning\train_finetuning_baseline.py --run-id baseline-gemma4-2b-r3 --model gemma4_2b --max-seq-length 512 --training-scope gates_and_norms
```
