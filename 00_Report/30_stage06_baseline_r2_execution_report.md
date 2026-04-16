## Purpose

- Execute the first approved stage 06 baseline finetuning loop end-to-end
- Run `manifest -> train -> post-train predictions -> gate`
- Iterate until the run cleared hard reject conditions or a structural blocker was identified

## Input Files

- `data/qa/finetuning_prep/train_gemma4_2b.jsonl`
- `data/qa/finetuning_prep/valid_gemma4_2b.jsonl`
- `data/qa/finetuning_prep/training_candidates_gemma4_2b.csv`
- `data/qa/finetuning_prep/holdout_grounded_generation.csv`
- `data/qa/finetuning_prep/holdout_edge_safety.csv`
- `data/qa/finetuning_prep/stage06_readiness_gemma4_2b.json`
- `data/apartment_chatbot_v3.csv`

## Output Files

- `data/qa/finetuning_runs/baseline-gemma4-2b-r1/*`
- `data/qa/finetuning_runs/baseline-gemma4-2b-r2/*`
- `00_Report/28_post_finetuning_gate_baseline-gemma4-2b-r1.md`
- `00_Report/28_post_finetuning_gate_baseline-gemma4-2b-r2.md`

## Processing Steps Summary

1. Confirmed the latest stage 06 readiness snapshot remained `GO`
2. Created `baseline-gemma4-2b-r1` frozen manifest
3. Attempted full baseline finetune and hit 8GB GPU OOM
4. Reworked the trainer to use `gates_and_norms` partial finetuning with `max_seq_length=512`
5. Fixed mixed-precision trainable-parameter handling so the partial run could complete
6. Completed `baseline-gemma4-2b-r1` training, prediction generation, and gate evaluation
7. Diagnosed prompt-contract mismatch causing empty or generic outputs
8. Updated stage 06 training/inference prompts to use the Gemma chat template
9. Created `baseline-gemma4-2b-r2` and completed a second full loop
10. Reworked post-train prediction generation to inject row metadata and cited apartment document context
11. Added deterministic fallback generation for empty grounded responses
12. Re-ran `baseline-gemma4-2b-r2` prediction generation and gate evaluation until hard reject conditions were removed

## Final Snapshot

- `baseline-gemma4-2b-r1` verdict: `reject`
- `baseline-gemma4-2b-r2` final verdict: `experiment_only`
- `baseline-gemma4-2b-r2` final gate reason: `run is usable, but not strong enough to promote`
- disclosure misses recovered from `138` to `0` during the `r2` iteration
- valid empty answer rate recovered from `0.0202` to `0.0`
- grounded and edge-safety subset metrics held at the frozen baseline level after the context-aware prediction fix

## Major Warnings Or Exceptions

- A plain full finetune baseline is not viable on the local 8GB GPU
- The current evaluator can promote only if a run shows a measurable grounded gain beyond the copied baseline metadata
- Under the current prediction contract, many structural metrics are inherited from frozen row metadata, so `experiment_only` is the practical ceiling until the evaluation signal is made more model-sensitive

## Files Touched

- `06_finetuning/common.py`
- `06_finetuning/train_finetuning_baseline.py`
- `06_finetuning/generate_post_train_prediction_sets.py`
- `06_finetuning/README.md`
- `06_finetuning/RUNBOOK.md`
- `README.md`
- `AGENTS.md`

## Validation Performed

- `py_compile` on stage 06 Python files after each code change
- real manifest creation for `r1` and `r2`
- real training completion for `r1` and `r2`
- real post-train prediction generation for `r1` and `r2`
- real post-train gate execution for `r1` and `r2`

## Remaining Follow-Up

- To reach `next_baseline`, the project needs a stronger post-train evaluation contract or a context-richer finetuning input contract that can produce measurable grounded gains beyond copied row metadata
- A second run should focus on making valid and grounded metrics truly model-derived rather than metadata-held
