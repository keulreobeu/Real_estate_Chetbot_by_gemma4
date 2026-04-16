from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from common import (
    DEFAULT_MODEL_ID,
    PROJECT_ROOT,
    build_run_dir,
    describe_file,
    ensure_parent,
    get_default_frozen_paths,
    load_json,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a frozen run manifest for the first finetuning run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--training-method", default="pending")
    parser.add_argument("--adapter-or-full-finetune", default="pending")
    parser.add_argument("--operator-note", default="baseline run manifest created before training")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_dir) if args.output_dir else build_run_dir(args.run_id)
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.json"

    frozen_paths = get_default_frozen_paths(args.model)
    missing = [name for name, path in frozen_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required frozen inputs are missing: {missing}")

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
        "expected_outputs": {
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "train_log": str((run_dir / "train.log").relative_to(PROJECT_ROOT)),
            "checkpoints_dir": str((run_dir / "checkpoints").relative_to(PROJECT_ROOT)),
            "final_dir": str((run_dir / "final").relative_to(PROJECT_ROOT)),
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
        "train_file": manifest["frozen_inputs"]["train_file"]["path"],
        "valid_file": manifest["frozen_inputs"]["valid_file"]["path"],
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
