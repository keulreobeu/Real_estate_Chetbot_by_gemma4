## Purpose

- Evaluate the post-finetuning run `baseline-gemma4-2b-r1` against the frozen pre-finetuning GO snapshot
- Verify run integrity before judging model quality
- Compare valid, grounded holdout, and edge safety holdout before any baseline promotion

## Input Files

- `data\qa\finetuning_runs\baseline-gemma4-2b-r1\manifest.json`
- `data\qa\finetuning_runs\baseline-gemma4-2b-r1\config.json`
- `data\qa\finetuning_runs\baseline-gemma4-2b-r1\train.log`
- `data\qa\finetuning_runs\baseline-gemma4-2b-r1\valid_predictions.csv`
- `data\qa\finetuning_runs\baseline-gemma4-2b-r1\grounded_holdout_predictions.csv`
- `data\qa\finetuning_runs\baseline-gemma4-2b-r1\edge_safety_holdout_predictions.csv`
- `data\qa\finetuning_prep\training_candidates_gemma4_2b.csv`
- `data\qa\finetuning_prep\holdout_grounded_generation.csv`
- `data\qa\finetuning_prep\holdout_edge_safety.csv`

## Output Files

- `data\qa\finetuning_runs\baseline-gemma4-2b-r1\valid_eval.json`
- `data\qa\finetuning_runs\baseline-gemma4-2b-r1\grounded_holdout_eval.json`
- `data\qa\finetuning_runs\baseline-gemma4-2b-r1\edge_safety_holdout_eval.json`
- `data\qa\finetuning_runs\baseline-gemma4-2b-r1\post_train_summary.json`
- `data\qa\finetuning_runs\baseline-gemma4-2b-r1\notes.md`

## Final Row And Column Counts

- valid predictions: 99 rows
- grounded holdout predictions: 200 rows
- edge safety predictions: 150 rows

## Processing Steps Summary

1. Verified run integrity against manifest/config/log/final artifact requirements
2. Verified frozen input hashes still matched the pre-finetuning snapshot
3. Evaluated valid predictions against the frozen valid subset
4. Evaluated grounded holdout predictions against the frozen grounded holdout subset
5. Evaluated edge safety predictions against the frozen edge safety holdout subset
6. Compared all three scopes against the pre-finetuning baseline and assigned a verdict

## Major Warnings Or Exceptions

- no integrity issues

## Final Snapshot

- verdict: `reject`
- verdict reasons: disclosure miss reappeared in edge safety holdout
- valid current: `{"rows": 99, "answer_type_match_rate": 0.0, "match_status_match_rate": 0.9797979797979798, "must_include_pass_rate": 1.0, "must_not_include_pass_rate": 1.0, "empty_answer_rate": 0.8484848484848485, "short_output_rate": 0.0}`
- grounded current: `{"rows": 200, "doc_hit_rate": 0.03, "field_hit_rate": 0.805, "router_match_rate": 1.0, "match_status_match_rate": 1.0, "must_not_recommend_pass_rate": 1.0, "must_disclose_limit_pass_rate": 1.0, "unsafe_recommendation": 0, "disclosure_miss": 0, "insufficient_context_rate": 0.0}`
- edge safety current: `{"rows": 150, "doc_hit_rate": 0.0, "field_hit_rate": 0.0, "router_match_rate": 0.9733333333333334, "match_status_match_rate": 0.9133333333333333, "must_not_recommend_pass_rate": 0.9652777777777778, "must_disclose_limit_pass_rate": 0.0, "unsafe_recommendation": 5, "disclosure_miss": 140, "insufficient_context_rate": 0.4666666666666667}`