# AGENTS.md

## Project purpose

This repository is a stage-based real estate AI chatbot data pipeline project.

Use Codex as a practical engineering assistant. Build maintainable code, explicit data contracts, reproducible outputs, and synchronized documentation.

Primary project goals:
- preprocess raw apartment CSV data
- normalize transport-related data
- build RAG-ready real estate datasets
- generate QA, evaluation, and edge-case question datasets
- prepare the project for future Gemma 4 generation and evaluation stages

Current active stage:
- `06_finetuning`

## Repository structure

Use the current repository structure as the default standard.

- `00_Report`
  - reports and processing logs
- `01_preprocessing`
  - preprocessing, RAG build, QA generation, edge question generation scripts
- `03_generation_optimization`
  - upstream optimization gate before finetuning
  - recommendation safety, routing, match-status optimization gate before finetuning
- `05_finetuning_prep`
  - SFT candidate separation, holdout generation, train-valid JSONL preparation
- `06_finetuning`
  - active stage after `stage06_readiness = GO`
  - finetuning run contract, runbook, and post-train evaluation gate
- `data/original`
  - raw source CSV files
- `data`
  - processed and RAG-ready main datasets
- `data/qa`
  - QA base, QA dataset, finetune JSONL, evaluation, edge questions
- `data/qa/finetuning_prep`
  - stage 05 candidate, rejected, holdout, and train-valid artifacts

Optional shared directories when needed:
- `shared/`
- `common/`
- `config/`
- `data/mapping`
- `data/eval`
- `logs`
- `data/eval/generation_optimization`

## Working style

- Plan first for non-trivial tasks.
- Prefer small, reviewable changes over broad rewrites.
- Preserve existing repository conventions unless there is a clear reason to change them.
- State assumptions explicitly when they affect outputs.
- Do not make unrelated refactors.
- Do not silently change schema, architecture, folder structure, or file naming rules.
- Read relevant files before editing them.
- Prefer targeted edits over full-file rewrites.
- Do not claim execution you did not perform.
- Do not claim success without verification.

## Mandatory workflow

Use this sequence for meaningful work:
1. Inspect the request and the relevant files.
2. Briefly summarize the intended change.
3. Identify affected scripts, outputs, docs, and downstream dependencies.
4. Implement the smallest correct change.
5. Validate the result with the smallest relevant check.
6. Update docs if commands, structure, outputs, or behavior changed.
7. Report what changed, what was verified, and what still needs attention.

## gstack usage policy

Use gstack when the task matches a specialized workflow.

Recommended mapping:
- product scoping or MVP refinement → `office-hours`
- scope challenge or value check → `plan-ceo-review`
- implementation planning or architecture review → `plan-eng-review`
- code review or regression scan → `review`
- browser verification or UI checks → `qa` or `browse`
- release notes or ship readiness → `ship`
- documentation sync → `document-release`
- post-task reflection → `retro`

Use this default sequence for non-trivial feature work:
1. planning skill first
2. implementation second
3. `review` before finalizing
4. `document-release` if documentation changed

Do not use gstack just to add ceremony. Use it when the task benefits from specialized review, planning, QA, or release support.

## Folder stage rules

- This repository is organized by work stage, not by version history.
- Use stage folders in `NN_name` format.
- Never reuse an old stage number.
- Use short, specific names.
- Do not use vague names such as `pipeline`, `new`, `test`, or `final`.

Good examples:
- `01_preprocessing`
- `02_rag_build`
- `03_gemma4_generation`
- `04_evaluation`

Create a new stage folder only when one of the following changes materially:
- model choice
- pipeline purpose
- required input file types
- required output file types
- required schema fields or contracts
- downstream scripts must change because of the new structure
- work type changes, such as preprocessing, generation, training, inference, or evaluation

Do not create a new stage folder for:
- small refactors
- minor bug fixes
- prompt wording tweaks
- local cleanup
- template additions
- small validation improvements

Rules for stage folders:
- Keep each stage folder focused on one major purpose.
- Do not mix stage-based and function-based folder logic.
- Do not change the role of an existing stage folder after it is established.
- Keep stage-specific code in the stage folder.
- Move cross-stage reusable code into `shared/`, `common/`, or `config/`.

Stage lifecycle rules:
- Mark one active stage in `README.md` and `AGENTS.md`.
- Do not delete old stages just because they are no longer primary.
- Mark old stages as `deprecated` or `archived` in documentation when needed.
- Record the reason and replacement stage when a stage becomes deprecated.

## Repository change rules

- Do not change repository structure unless the task explicitly requires it.
- Do not create new top-level folders casually.
- Do not rename outputs without updating all dependent scripts and docs.
- Do not add new dependencies unless necessary.
- If you add a dependency, explain why and update setup documentation.
- If a generated file is committed, make that explicit in the report.

## Data and preprocessing rules

- Never modify raw source CSV files in `data/original`.
- Always write outputs as new files or approved versioned files.
- Analyze input data before major preprocessing:
  - row count
  - column count
  - column list
  - dtype summary
  - missing-value summary
- Remove only exact duplicate rows.
- Do not remove non-exact duplicates unless explicitly instructed.
- If non-exact duplicates are detected, report them.
- Normalize numeric fields so they can be filtered, sorted, and compared.
- Do not fill numeric missing values with `0` unless explicitly justified.
- If a string placeholder is needed, prefer `"정보 없음"` and avoid changing original meaning.
- If an address column exists, try to split it into:
  - `시도`
  - `시군구`
  - `동`
  - `상세주소`
- Report address split success and failure counts.
- If address split failure exceeds 10%, add a warning to the report.

## Policy field rules

- Never delete policy-related fields.
- If the following fields exist, normalize them exactly as follows:
  - `투기과열지구_before` → `분양당시_투기과열지구`
  - `투기과열지구_after` → `현재_투기과열지구`
  - `분양가상한제_before` → `분양당시_분양가상한제`
  - `분양가상한제_after` → `현재_분양가상한제`
- If some policy fields are missing, continue with a warning.
- If all policy fields are missing, continue with a warning and report it clearly.
- Keep normalized policy fields in the main dataset.
- Also generate a readable Korean policy summary field when appropriate.

## Transport rules

- Keep transport information structured when available.
- Preferred fields:
  - `가장가까운역`
  - `거리_m`
  - `환승역여부`
  - `호선수`
  - `가장가까운역_호선요약`
- If distance is given in km-like units, also generate a meter-based field when useful.
- If station and line information exist, normalize into:
  - `station_line_map.csv`
  - `apartment_station_map.csv`
- Default transfer rule:
  - `호선수 >= 2` means transfer station
- If transport fields are partially missing, continue with a warning and process as much as possible.

## RAG rules

- Keep the main RAG CSV readable for both search and answer generation.
- Maintain both structured fields and readable summary fields.
- `description` must be written in Korean.
- `description` should usually be 3 to 6 sentences.
- `description` should stay under about 300 characters when practical.
- Include as many of the following as available:
  - location
  - scale
  - area
  - transport
  - price
  - living infrastructure
  - policy
- Do not invent missing data.
- Do not use speculative or promotional wording.
- `검색키워드` should be built from relevant available values such as:
  - apartment name
  - region
  - station name
  - line
  - builder
  - area band
  - policy information

## QA generation rules

- Answers must be grounded only in repository data outputs.
- Maximize question diversity.
- Remove duplicate questions.
- Remove empty questions and empty answers.
- Keep QA outputs in `data/qa`.
- Keep evaluation and edge-case outputs separate from the main QA dataset.

Minimum QA categories:
- fact
- location
- transport
- price
- lifestyle
- policy

QA quality rules:
- Keep category distribution reasonably balanced.
- If any major category falls below 10% of the dataset, add a warning to the report.
- Prefer concise answers.
- Prefer 1 to 3 sentence answers unless a richer description is explicitly required.

Preferred QA output contracts:
- QA CSV required columns:
  - `question`
  - `answer`
  - `아파트명`
  - `문서ID`
  - `category`
- Evaluation CSV required columns:
  - `question`
  - `expected_answer`
  - `문서ID`
- Edge question CSV required columns:
  - `question`
  - `type`

## Pipeline rules

For pipeline-related work:
- define each stage clearly
- state inputs, outputs, and failure points
- keep contracts explicit
- do not mix collection, transformation, retrieval, inference, and presentation without reason
- check downstream compatibility when changing any step

When documenting or building a pipeline step, prefer this structure:
- purpose
- inputs
- outputs
- dependencies
- validation method

## Data contract rules

- Do not change schema names lightly.
- If schema changes, update dependent code and docs together.
- Call out backward compatibility impact explicitly.
- Treat renaming, required field changes, and file path changes as contract changes.

## Output contracts

- Save CSV outputs as `utf-8-sig` with headers.
- Keep generated artifacts separate from source code.
- Name artifacts consistently.
- Document how each major artifact is produced.

Main RAG CSV should preserve or generate at least the following kinds of fields when available:
- `문서ID`
- `아파트명`
- `시도`
- `시군구`
- `동`
- `전용면적`
- `공급면적`
- `공급액(만원)`
- `평당_공급액`
- `가장가까운역`
- `가장가까운역_호선요약`
- `description`
- `검색키워드`

Finetune JSONL should use one JSON object per line.
Preferred format:

```json
{"instruction":"질문","input":"","output":"답변"}
```

## Documentation rules

Update documentation whenever any of the following changes:
- setup steps
- commands
- environment requirements
- folder structure
- pipeline flow
- model behavior
- output files
- artifact locations

Preferred documentation targets:
- `README.md` for setup, usage, active stage, and repository structure
- `AGENTS.md` for Codex working rules and operational constraints
- stage-level notes when stage-specific workflow is needed
- inline comments only when code-level clarification is necessary

## Environment rules

- Never hardcode secrets.
- Follow existing environment file conventions.
- If a new environment variable is required, document it.
- Do not silently introduce configuration that users cannot discover from docs.

## Testing and validation

Before finalizing:
- run the smallest relevant validation available
- check broken imports
- check invalid paths
- check stale names
- check mismatched contracts
- verify outputs exist where expected

If tests are unavailable:
- say what was manually verified
- do not overclaim confidence

## Failure and stop rules

Stop and report when:
- the required input file does not exist
- required input columns are missing and the task cannot proceed
- the output row count drops by more than 30% without a justified reason
- document IDs cannot be generated when required

Continue with warning when:
- some policy fields are missing
- address split failure exceeds 10%
- some transport fields are missing
- QA category balance is weak
- a non-critical validation step cannot be completed

Never fail silently.

## Report requirements

Every major run must generate a markdown report.

Report files should stay in `00_Report` and use ordered numeric prefixes.

Each major run report must include:
- purpose
- input files
- output files
- original row and column counts
- final row and column counts
- removed exact duplicate count
- newly generated columns
- processing steps summary
- major warnings or exceptions
- sample outputs

If a processing log exists, reference its location in the report.

## Common commands

Use the current repository commands as documented. If new commands are introduced, update `README.md`.

Current common command pattern in this repository:

- preprocessing:
  - `python .\01_preprocessing\preprocess_apartment_csv.py`
  - `python .\01_preprocessing\preprocess_apartment_pipeline.py`
- QA generation:
  - `python .\01_preprocessing\generate_apartment_qa_dataset.py`
- edge question generation:
  - `python .\01_preprocessing\generate_edge_questions.py`
- finetuning prep:
  - `python .\05_finetuning_prep\prepare_sft_dataset.py --model gemma4_2b`
- finetuning:
  - `python .\06_finetuning\create_run_manifest.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b`
  - `python .\06_finetuning\train_finetuning_baseline.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b`
  - `python .\06_finetuning\train_finetuning_baseline.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b --max-seq-length 512 --training-scope gates_and_norms`
  - `python .\06_finetuning\generate_post_train_prediction_sets.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b`
  - `python .\06_finetuning\evaluate_post_finetuning_run.py --run-id baseline-gemma4-2b-r1 --model gemma4_2b`
- generation optimization:
  - `python .\03_generation_optimization\analyze_edge_failures.py --model gemma4_2b`
  - `python .\05_finetuning_prep\validate_06_readiness.py --model gemma4_2b`

## Reporting format

At the end of a task, report:
- what changed
- why it changed
- files touched
- validation performed
- remaining risks or follow-up items

## Constraints

- Do not fabricate results, metrics, or logs.
- Do not claim success without verification.
- Do not delete user content unless explicitly requested.
- Do not perform broad cleanup unless explicitly requested.
- Keep outputs practical and implementation-oriented.

## Done definition

A task is done when:
- the requested change is implemented or clearly bounded
- impacted files are consistent
- obvious regressions were checked
- documentation is updated when needed
- the final report states remaining limits honestly
