from __future__ import annotations

import argparse
from typing import Any

from common import (
    INPUT_EDGE,
    INPUT_EVAL,
    get_prediction_output_path,
    load_csv,
    pick_default_model_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generation output completeness and integrity.")
    parser.add_argument("--mode", choices=["eval", "edge"], default="edge")
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def required_columns_for_mode(mode: str) -> list[str]:
    common_cols = [
        "question",
        "answer",
        "cited_doc_ids",
        "used_fields",
        "insufficient_context",
        "latency_ms",
    ]
    if mode == "eval":
        return common_cols + ["expected_answer", "expected_doc_id"]
    return common_cols + ["expected_doc", "expected_field"]


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def validate(mode: str, model_id: str) -> int:
    input_path = INPUT_EVAL if mode == "eval" else INPUT_EDGE
    output_path = get_prediction_output_path(mode, model_id)

    input_df, _ = load_csv(input_path)
    output_df, _ = load_csv(output_path)

    expected_rows = len(input_df)
    actual_rows = len(output_df)
    missing_rows = max(expected_rows - actual_rows, 0)

    missing_columns = [c for c in required_columns_for_mode(mode) if c not in output_df.columns]
    has_source_row_index = "source_row_index" in output_df.columns
    duplicate_question_rows = int(output_df.duplicated(subset=["question"]).sum()) if "question" in output_df.columns else 0
    duplicate_source_index_rows = (
        int(output_df.duplicated(subset=["source_row_index"]).sum()) if "source_row_index" in output_df.columns else 0
    )
    empty_answer_rows = int(output_df["answer"].astype(str).str.strip().eq("").sum()) if "answer" in output_df.columns else 0

    print("=== GENERATION OUTPUT VALIDATION ===")
    print(f"mode={mode}")
    print(f"model_id={model_id}")
    print(f"input_path={input_path}")
    print(f"output_path={output_path}")
    print(f"expected_rows={expected_rows}")
    print(f"actual_rows={actual_rows}")
    print(f"missing_rows={missing_rows}")
    print(f"duplicate_question_rows={duplicate_question_rows}")
    print(f"duplicate_source_index_rows={duplicate_source_index_rows}")
    print(f"empty_answer_rows={empty_answer_rows}")
    print(f"missing_columns={missing_columns if missing_columns else '[]'}")
    print(f"has_source_row_index={has_source_row_index}")

    if "question" in output_df.columns:
        first_q = safe_str(output_df.iloc[0]["question"]) if actual_rows else ""
        last_q = safe_str(output_df.iloc[-1]["question"]) if actual_rows else ""
        print(f"first_question={first_q}")
        print(f"last_question={last_q}")

    has_critical_issue = False
    if missing_columns:
        print("STATUS=FAIL")
        has_critical_issue = True
    elif not has_source_row_index:
        print("STATUS=WARN_LEGACY_NO_SOURCE_ROW_INDEX")
    elif missing_rows > 0:
        print("STATUS=WARN_INCOMPLETE")
    else:
        print("STATUS=PASS")

    return 1 if has_critical_issue else 0


def main() -> None:
    args = parse_args()
    model_id = args.model or pick_default_model_id()
    raise SystemExit(validate(args.mode, model_id))


if __name__ == "__main__":
    main()
