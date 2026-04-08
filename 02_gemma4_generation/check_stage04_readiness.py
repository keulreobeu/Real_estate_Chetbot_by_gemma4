from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from common import INPUT_EDGE, INPUT_EVAL, get_metrics_output_path, get_prediction_output_path, pick_default_model_id


SECTION_PATTERN = r"### `{heading}`\r?\n\r?\nRequired {kind}:\r?\n\r?\n(?P<body>.*?)(?=\r?\n### |\Z)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether stage 04 evaluation is contract-aligned and ready without running GPU work."
    )
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def extract_contract_list(contract_text: str, heading: str, kind: str) -> list[str]:
    pattern = SECTION_PATTERN.format(heading=re.escape(heading), kind=re.escape(kind))
    match = re.search(pattern, contract_text, flags=re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not find contract section for {heading}")
    return re.findall(r"- `([^`]+)`", match.group("body"))


def extract_function_body(source_text: str, func_name: str) -> str:
    pattern = rf"def {func_name}\(.*?\) -> None:\r?\n(?P<body>.*?)(?=\r?\ndef |\Z)"
    match = re.search(pattern, source_text, flags=re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not find function body for {func_name}")
    return match.group("body")


def extract_evaluator_columns(source_text: str, func_name: str) -> list[str]:
    body = extract_function_body(source_text, func_name)
    seen: list[str] = []
    for column in re.findall(r'row\.get\("([^"]+)"', body):
        if column not in seen:
            seen.append(column)
    return seen


def read_csv_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.columns.tolist()


def read_csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    df = pd.read_csv(path, encoding="utf-8-sig")
    return len(df)


def read_json_keys(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(data.keys())


def classify_columns(contract_cols: list[str], evaluator_cols: list[str], actual_cols: list[str]) -> dict[str, list[str]]:
    contract_set = set(contract_cols)
    evaluator_set = set(evaluator_cols)
    actual_set = set(actual_cols)
    return {
        "evaluator_only": sorted(evaluator_set - contract_set - actual_set),
        "contract_only": sorted(contract_set - evaluator_set - actual_set),
        "actual_only": sorted(actual_set - contract_set - evaluator_set),
        "evaluator_missing_in_actual": sorted(evaluator_set - actual_set),
        "contract_missing_in_actual": sorted(contract_set - actual_set),
        "contract_and_actual_but_not_evaluator": sorted((contract_set & actual_set) - evaluator_set),
        "evaluator_and_actual_but_not_contract": sorted((evaluator_set & actual_set) - contract_set),
    }


def print_mode_report(
    *,
    mode: str,
    expected_rows: int,
    actual_rows: int,
    contract_cols: list[str],
    evaluator_cols: list[str],
    actual_cols: list[str],
    contract_metric_keys: list[str],
    actual_metric_keys: list[str],
) -> tuple[bool, list[str]]:
    summary = classify_columns(contract_cols, evaluator_cols, actual_cols)
    print(f"=== {mode.upper()} STAGE04 READINESS ===")
    print(f"expected_rows={expected_rows}")
    print(f"actual_rows={actual_rows}")
    print(f"rows_complete={actual_rows == expected_rows}")
    print(f"contract_required_columns={contract_cols}")
    print(f"evaluator_read_columns={evaluator_cols}")
    print(f"actual_output_columns={actual_cols}")
    for key, values in summary.items():
        print(f"{key}={values}")
    print(f"contract_metric_keys={contract_metric_keys}")
    print(f"actual_metric_keys={actual_metric_keys}")
    print(f"metric_keys_missing={sorted(set(contract_metric_keys) - set(actual_metric_keys))}")

    blockers: list[str] = []
    if actual_rows != expected_rows:
        blockers.append(f"{mode}: rows incomplete ({actual_rows}/{expected_rows})")
    if summary["evaluator_missing_in_actual"]:
        blockers.append(f"{mode}: evaluator columns missing in actual output")
    if summary["contract_missing_in_actual"]:
        blockers.append(f"{mode}: contract columns missing in actual output")
    if actual_metric_keys and sorted(set(contract_metric_keys) - set(actual_metric_keys)):
        blockers.append(f"{mode}: metric json missing required contract keys")

    ready = not blockers
    print(f"stage04_ready_for_{mode}={ready}")
    if blockers:
        print(f"blockers={blockers}")
    print("")
    return ready, blockers


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    model_id = args.model or pick_default_model_id()

    contract_path = repo_root / "02_gemma4_generation" / "CONTRACT.md"
    evaluator_path = repo_root / "02_gemma4_generation" / "evaluate_generation_mvp.py"
    contract_text = contract_path.read_text(encoding="utf-8")
    evaluator_text = evaluator_path.read_text(encoding="utf-8")

    edge_prediction_contract = extract_contract_list(
        contract_text, "data/eval/gemma4_generation_edge_predictions_<model_id>.csv", "columns"
    )
    eval_prediction_contract = extract_contract_list(
        contract_text, "data/eval/gemma4_generation_eval_predictions_<model_id>.csv", "columns"
    )
    edge_metrics_contract = extract_contract_list(
        contract_text, "data/eval/gemma4_generation_edge_metrics_<model_id>.json", "keys"
    )
    eval_metrics_contract = extract_contract_list(
        contract_text, "data/eval/gemma4_generation_eval_metrics_<model_id>.json", "keys"
    )

    edge_evaluator_cols = extract_evaluator_columns(evaluator_text, "evaluate_edge")
    eval_evaluator_cols = extract_evaluator_columns(evaluator_text, "evaluate_eval")

    edge_output_path = get_prediction_output_path("edge", model_id)
    eval_output_path = get_prediction_output_path("eval", model_id)
    edge_metrics_path = get_metrics_output_path("edge", model_id)
    eval_metrics_path = get_metrics_output_path("eval", model_id)

    edge_ready, edge_blockers = print_mode_report(
        mode="edge",
        expected_rows=read_csv_row_count(INPUT_EDGE),
        actual_rows=read_csv_row_count(edge_output_path),
        contract_cols=edge_prediction_contract,
        evaluator_cols=edge_evaluator_cols,
        actual_cols=read_csv_columns(edge_output_path),
        contract_metric_keys=edge_metrics_contract,
        actual_metric_keys=read_json_keys(edge_metrics_path),
    )
    eval_ready, eval_blockers = print_mode_report(
        mode="eval",
        expected_rows=read_csv_row_count(INPUT_EVAL),
        actual_rows=read_csv_row_count(eval_output_path),
        contract_cols=eval_prediction_contract,
        evaluator_cols=eval_evaluator_cols,
        actual_cols=read_csv_columns(eval_output_path),
        contract_metric_keys=eval_metrics_contract,
        actual_metric_keys=read_json_keys(eval_metrics_path),
    )

    overall_ready = edge_ready and eval_ready
    print("=== OVERALL STAGE04 GATE ===")
    print(f"model_id={model_id}")
    print(f"stage04_ready={overall_ready}")
    print(
        "completion_definition="
        "required_rows_complete + contract_aligned + evaluator_columns_present + required_metric_keys_valid"
    )
    print(
        "partial_metrics_not_trustworthy_when="
        "rows_incomplete or evaluator_missing_in_actual or contract_missing_in_actual"
    )
    all_blockers = edge_blockers + eval_blockers
    print(f"blockers={all_blockers if all_blockers else []}")
    if overall_ready:
        print("CONCLUSION=YES")
        raise SystemExit(0)
    print("CONCLUSION=NO")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
