# 06_finetuning Checklist

## Before Run

- readiness verdict is `GO`
- train file exists
- valid file exists
- grounded holdout exists
- edge safety holdout exists
- dataset summary exists
- output directory for this run is unique
- user approval for the training command is recorded

## During Run

- training log is being written
- checkpoint path is stable
- no upstream stage 03 or 05 regeneration is modifying the frozen inputs

## After Run

- validation summary exists
- grounded holdout summary exists
- edge safety holdout summary exists
- run note is written
- next action is decided and recorded
