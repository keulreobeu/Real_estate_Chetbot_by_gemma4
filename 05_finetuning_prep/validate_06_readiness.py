from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
QA_PREP_DIR = DATA_DIR / "qa" / "finetuning_prep"
EVAL_DIR = DATA_DIR / "eval"
DEFAULT_MODEL_ID = "gemma4_2b"

THRESHOLDS = {
    "min_train_rows": 500,
    "min_valid_rows": 50,
    "max_missing_contract_fields": 5,
    "min_holdout_edge_safety_rows": 150,
    "min_holdout_grounded_generation_rows": 200,
    "min_must_not_recommend_pass_rate": 0.90,
    "min_must_disclose_limit_pass_rate": 0.90,
    "min_router_match_rate": 0.85,
    "min_match_status_match_rate": 0.85,
    "max_avg_latency_ms": 24411.1685 * 1.20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate whether stage 05 outputs are ready for 06 finetuning.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check(condition: bool, name: str, actual: Any, expected: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(condition), "actual": actual, "expected": expected}


def main() -> None:
    args = parse_args()
    summary_path = QA_PREP_DIR / f"dataset_summary_{args.model}.json"
    edge_metrics_path = EVAL_DIR / f"gemma4_generation_edge_metrics_{args.model}.json"
    eval_metrics_path = EVAL_DIR / f"gemma4_generation_eval_metrics_{args.model}.json"

    dataset_summary = load_json(summary_path)
    edge_metrics = load_json(edge_metrics_path)
    eval_metrics = load_json(eval_metrics_path)
    rejected_reasons = dataset_summary.get("rejected_reason_counts", {})

    checks = [
        check(dataset_summary.get("train_rows", 0) >= THRESHOLDS["min_train_rows"], "train_rows", dataset_summary.get("train_rows"), f">= {THRESHOLDS['min_train_rows']}"),
        check(dataset_summary.get("valid_rows", 0) >= THRESHOLDS["min_valid_rows"], "valid_rows", dataset_summary.get("valid_rows"), f">= {THRESHOLDS['min_valid_rows']}"),
        check(
            rejected_reasons.get("missing_contract_fields", 0) <= THRESHOLDS["max_missing_contract_fields"],
            "missing_contract_fields",
            rejected_reasons.get("missing_contract_fields", 0),
            f"<= {THRESHOLDS['max_missing_contract_fields']}",
        ),
        check(
            dataset_summary.get("holdout_edge_safety_rows", 0) >= THRESHOLDS["min_holdout_edge_safety_rows"],
            "holdout_edge_safety_rows",
            dataset_summary.get("holdout_edge_safety_rows"),
            f">= {THRESHOLDS['min_holdout_edge_safety_rows']}",
        ),
        check(
            dataset_summary.get("holdout_grounded_generation_rows", 0) >= THRESHOLDS["min_holdout_grounded_generation_rows"],
            "holdout_grounded_generation_rows",
            dataset_summary.get("holdout_grounded_generation_rows"),
            f">= {THRESHOLDS['min_holdout_grounded_generation_rows']}",
        ),
        check(
            edge_metrics.get("must_not_recommend_pass_rate", 0) >= THRESHOLDS["min_must_not_recommend_pass_rate"],
            "must_not_recommend_pass_rate",
            edge_metrics.get("must_not_recommend_pass_rate"),
            f">= {THRESHOLDS['min_must_not_recommend_pass_rate']}",
        ),
        check(
            edge_metrics.get("must_disclose_limit_pass_rate", 0) >= THRESHOLDS["min_must_disclose_limit_pass_rate"],
            "must_disclose_limit_pass_rate",
            edge_metrics.get("must_disclose_limit_pass_rate"),
            f">= {THRESHOLDS['min_must_disclose_limit_pass_rate']}",
        ),
        check(edge_metrics.get("router_match_rate", 0) >= THRESHOLDS["min_router_match_rate"], "router_match_rate", edge_metrics.get("router_match_rate"), f">= {THRESHOLDS['min_router_match_rate']}"),
        check(
            edge_metrics.get("match_status_match_rate", 0) >= THRESHOLDS["min_match_status_match_rate"],
            "match_status_match_rate",
            edge_metrics.get("match_status_match_rate"),
            f">= {THRESHOLDS['min_match_status_match_rate']}",
        ),
        check(edge_metrics.get("avg_latency_ms", 0) <= THRESHOLDS["max_avg_latency_ms"], "avg_latency_ms", edge_metrics.get("avg_latency_ms"), f"<= {THRESHOLDS['max_avg_latency_ms']:.1f}"),
        check(eval_metrics.get("answer_type_match_rate", 0) >= 1.0, "eval_answer_type_match_rate", eval_metrics.get("answer_type_match_rate"), ">= 1.0"),
        check(eval_metrics.get("match_status_match_rate", 0) >= 1.0, "eval_match_status_match_rate", eval_metrics.get("match_status_match_rate"), ">= 1.0"),
    ]
    ready = all(item["pass"] for item in checks)
    payload = {
        "model_id": args.model,
        "ready_for_06": ready,
        "verdict": "GO" if ready else "NO_GO",
        "checks": checks,
    }
    output_path = QA_PREP_DIR / f"stage06_readiness_{args.model}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
