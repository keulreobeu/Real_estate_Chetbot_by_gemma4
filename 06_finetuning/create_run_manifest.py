from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from common import (
    CONTEXTUAL_VALIDATION_VERSION,
    DEFAULT_MODEL_ID,
    PROJECT_ROOT,
    build_run_dir,
    contextual_summary_is_accepted,
    describe_file,
    ensure_parent,
    get_default_frozen_paths,
    load_json,
    save_json,
)


def resolve_project_local_output_dir(run_id: str, output_dir: str | None) -> Path:
    run_dir = Path(output_dir).resolve() if output_dir else build_run_dir(run_id).resolve()
    if not run_dir.is_relative_to(PROJECT_ROOT):
        raise RuntimeError(
            "Stage 06 tooling currently supports only project-local run directories under the repository root. "
            "External --output-dir paths are not supported without additional implementation."
        )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a stage 06 run manifest from frozen inputs plus optional run-local contextual JSONL."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--training-method", default="pending")
    parser.add_argument("--adapter-or-full-finetune", default="pending")
    parser.add_argument("--operator-note", default="baseline run manifest created before training")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-file", default=None, help="Optional run-local train JSONL override.")
    parser.add_argument("--valid-file", default=None, help="Optional run-local valid JSONL override.")
    parser.add_argument(
        "--context-mode",
        choices=["auto", "contextual", "frozen"],
        default="auto",
        help="auto preserves current selection behavior, contextual requires accepted run-local assets, frozen forces stage-05 JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_project_local_output_dir(args.run_id, args.output_dir)
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.json"

    frozen_paths = get_default_frozen_paths(args.model)
    missing = [name for name, path in frozen_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required frozen inputs are missing: {missing}")

    contextual_train_path = Path(args.train_file) if args.train_file else run_dir / "train_contextual.jsonl"
    contextual_valid_path = Path(args.valid_file) if args.valid_file else run_dir / "valid_contextual.jsonl"
    contextual_train_path = contextual_train_path if contextual_train_path.is_absolute() else PROJECT_ROOT / contextual_train_path
    contextual_valid_path = contextual_valid_path if contextual_valid_path.is_absolute() else PROJECT_ROOT / contextual_valid_path
    contextual_schema_path = run_dir / "schema_v1.md"
    contextual_summary_path = run_dir / "context_build_summary.json"

    if args.train_file and not contextual_train_path.exists():
        raise FileNotFoundError(f"Explicit train file does not exist: {contextual_train_path}")
    if args.valid_file and not contextual_valid_path.exists():
        raise FileNotFoundError(f"Explicit valid file does not exist: {contextual_valid_path}")
    if args.context_mode == "frozen" and (args.train_file or args.valid_file):
        raise RuntimeError("Frozen context mode cannot be combined with explicit run-local train or valid overrides.")

    contextual_candidates_exist = any(
        path.exists() for path in (contextual_train_path, contextual_valid_path, contextual_schema_path, contextual_summary_path)
    )
    contextual_summary = load_json(contextual_summary_path) if contextual_summary_path.exists() else {}
    contextual_summary_accepted = contextual_summary_is_accepted(contextual_summary)

    if contextual_candidates_exist and not contextual_summary_accepted and args.context_mode in {"auto", "contextual"}:
        raise RuntimeError(
            "Contextual artifacts are present but not accepted. "
            f"Expected an accepted summary with validation_version={CONTEXTUAL_VALIDATION_VERSION} before manifest selection."
        )

    contextual_assets_complete = (
        contextual_summary_accepted
        and contextual_train_path.exists()
        and contextual_valid_path.exists()
        and contextual_schema_path.exists()
    )

    if args.context_mode == "contextual":
        if not contextual_assets_complete:
            raise RuntimeError(
                "Contextual mode requires accepted contextual assets: "
                "train_contextual.jsonl, valid_contextual.jsonl, schema_v1.md, and context_build_summary.json."
            )
        use_contextual_selection = True
    elif args.context_mode == "frozen":
        use_contextual_selection = False
    else:
        use_contextual_selection = contextual_assets_complete

    selected_train_path = contextual_train_path if use_contextual_selection else frozen_paths["train_file"]
    selected_valid_path = contextual_valid_path if use_contextual_selection else frozen_paths["valid_file"]

    readiness = load_json(frozen_paths["readiness_file"])
    if readiness.get("verdict") != "GO":
        raise RuntimeError("Stage 06 readiness is not GO, refusing to create a training run manifest.")

    dataset_summary = load_json(frozen_paths["dataset_summary_file"])
    edge_metrics = load_json(frozen_paths["baseline_edge_metrics_file"])
    generation_summary = load_json(frozen_paths["baseline_generation_summary_file"])

    manifest = {
        "run_id": args.run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "model_id": args.model,
        "base_model_id": args.base_model_id,
        "training_method": args.training_method,
        "adapter_or_full_finetune": args.adapter_or_full_finetune,
        "operator_note": args.operator_note,
        "pre_finetuning_baseline": {
            "readiness_verdict": readiness.get("verdict"),
            "train_rows": dataset_summary.get("train_rows"),
            "valid_rows": dataset_summary.get("valid_rows"),
            "holdout_grounded_generation_rows": dataset_summary.get("holdout_grounded_generation_rows"),
            "holdout_edge_safety_rows": dataset_summary.get("holdout_edge_safety_rows"),
            "router_match_rate": edge_metrics.get("router_match_rate"),
            "match_status_match_rate": edge_metrics.get("match_status_match_rate"),
            "must_not_recommend_pass_rate": edge_metrics.get("must_not_recommend_pass_rate"),
            "must_disclose_limit_pass_rate": edge_metrics.get("must_disclose_limit_pass_rate"),
            "unsafe_recommendation": generation_summary.get("buckets", {}).get("unsafe_recommendation"),
            "router_mismatch": generation_summary.get("buckets", {}).get("router_mismatch"),
            "match_status_mismatch": generation_summary.get("buckets", {}).get("match_status_mismatch"),
            "disclosure_miss": generation_summary.get("buckets", {}).get("disclosure_miss"),
        },
        "frozen_inputs": {name: describe_file(path) for name, path in frozen_paths.items()},
        "run_local_inputs": {
            "selected_train_file": describe_file(selected_train_path),
            "selected_valid_file": describe_file(selected_valid_path),
            "contextual_train_file": describe_file(contextual_train_path),
            "contextual_valid_file": describe_file(contextual_valid_path),
            "schema_v1": describe_file(contextual_schema_path),
            "context_build_summary": describe_file(contextual_summary_path),
            "requested_context_mode": args.context_mode,
            "contextual_selection_mode": "accepted_contextual" if use_contextual_selection else "frozen_stage05",
            "contextual_summary_accepted": contextual_summary_accepted,
            "contextual_validation_version": contextual_summary.get("validation_version", ""),
        },
        "expected_outputs": {
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "train_log": str((run_dir / "train.log").relative_to(PROJECT_ROOT)),
            "checkpoints_dir": str((run_dir / "checkpoints").relative_to(PROJECT_ROOT)),
            "final_dir": str((run_dir / "final").relative_to(PROJECT_ROOT)),
            "schema_v1": str((run_dir / "schema_v1.md").relative_to(PROJECT_ROOT)),
            "train_contextual": str((run_dir / "train_contextual.jsonl").relative_to(PROJECT_ROOT)),
            "valid_contextual": str((run_dir / "valid_contextual.jsonl").relative_to(PROJECT_ROOT)),
            "context_build_summary": str((run_dir / "context_build_summary.json").relative_to(PROJECT_ROOT)),
            "valid_predictions": str((run_dir / "valid_predictions.csv").relative_to(PROJECT_ROOT)),
            "grounded_holdout_predictions": str((run_dir / "grounded_holdout_predictions.csv").relative_to(PROJECT_ROOT)),
            "edge_safety_holdout_predictions": str((run_dir / "edge_safety_holdout_predictions.csv").relative_to(PROJECT_ROOT)),
            "valid_eval": str((run_dir / "valid_eval.json").relative_to(PROJECT_ROOT)),
            "grounded_holdout_eval": str((run_dir / "grounded_holdout_eval.json").relative_to(PROJECT_ROOT)),
            "edge_safety_holdout_eval": str((run_dir / "edge_safety_holdout_eval.json").relative_to(PROJECT_ROOT)),
            "post_train_summary": str((run_dir / "post_train_summary.json").relative_to(PROJECT_ROOT)),
            "notes": str((run_dir / "notes.md").relative_to(PROJECT_ROOT)),
        },
    }

    config = {
        "run_id": args.run_id,
        "base_model_id": args.base_model_id,
        "training_method": args.training_method,
        "adapter_or_full_finetune": args.adapter_or_full_finetune,
        "context_mode": args.context_mode,
        "train_file": manifest["run_local_inputs"]["selected_train_file"]["path"],
        "valid_file": manifest["run_local_inputs"]["selected_valid_file"]["path"],
        "grounded_holdout_file": manifest["frozen_inputs"]["grounded_holdout_file"]["path"],
        "edge_safety_holdout_file": manifest["frozen_inputs"]["edge_safety_holdout_file"]["path"],
        "hyperparameters": {},
    }

    ensure_parent(manifest_path)
    save_json(manifest, manifest_path)
    save_json(config, config_path)
    print(f"Saved run manifest to {manifest_path}")
    print(f"Saved run config template to {config_path}")


if __name__ == "__main__":
    main()
