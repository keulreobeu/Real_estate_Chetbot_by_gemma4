from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import DEFAULT_MODEL_ID, build_run_dir, load_csv, save_json, safe_text
from evaluate_post_finetuning_run import has_limit_disclosure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit disclosure-required prediction rows before full post-train gate execution.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--run-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else build_run_dir(args.run_id)
    prediction_path = run_dir / "edge_safety_holdout_predictions.csv"
    audit_path = run_dir / "disclosure_audit.json"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

    prediction_df = load_csv(prediction_path)
    must_disclose_df = prediction_df[prediction_df["must_disclose_limit"].fillna("").astype(str) == "Y"].copy()
    missing_rows = []
    for _, row in must_disclose_df.iterrows():
        answer = safe_text(row.get("answer"))
        if has_limit_disclosure(answer):
            continue
        missing_rows.append(
            {
                "source_row_index": int(row["source_row_index"]) if safe_text(row.get("source_row_index")) else None,
                "question": safe_text(row.get("question")),
                "top_doc_id": safe_text(row.get("top_doc_id")),
                "cited_doc_ids": safe_text(row.get("cited_doc_ids")),
                "answer": answer,
            }
        )

    payload = {
        "run_id": args.run_id,
        "rows": int(len(prediction_df)),
        "must_disclose_rows": int(len(must_disclose_df)),
        "disclosure_miss": int(len(missing_rows)),
        "missing_rows": missing_rows,
    }
    save_json(payload, audit_path)
    print(audit_path)
    print(payload)


if __name__ == "__main__":
    main()
