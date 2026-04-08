from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from common import INPUT_EDGE, INPUT_EVAL, get_prediction_output_path, load_csv, save_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize generation outputs against the current input dataset.")
    parser.add_argument("--mode", choices=["eval", "edge"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def get_input_path(mode: str) -> Path:
    return INPUT_EVAL if mode == "eval" else INPUT_EDGE


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def sanitize(mode: str, model: str, write: bool) -> int:
    input_path = get_input_path(mode)
    output_path = get_prediction_output_path(mode, model)

    input_df, _ = load_csv(input_path)
    output_df, _ = load_csv(output_path)

    original_rows = len(output_df)
    changes: list[str] = []

    if "source_row_index" in output_df.columns:
        source_idx = pd.to_numeric(output_df["source_row_index"], errors="coerce")
        valid_index_set = set(range(len(input_df)))

        stale_values = source_idx.dropna().astype(int)
        stale_mask = source_idx.notna() & ~stale_values.reindex(source_idx.index).isin(valid_index_set)
        stale_count = int(stale_mask.sum())
        if stale_count:
            output_df = output_df.loc[~stale_mask].copy()
            changes.append(f"removed_stale_source_row_index={stale_count}")

        missing_source_mask = source_idx.isna()
        missing_source_count = int(missing_source_mask.sum())
        if missing_source_count:
            removable_rows: list[int] = []
            backfilled = 0
            for idx, row in output_df.loc[missing_source_mask].iterrows():
                question = safe_text(row.get("question"))
                if not question:
                    removable_rows.append(idx)
                    continue

                matches = input_df.index[input_df["question"].fillna("").astype(str).str.strip() == question].tolist()
                if len(matches) == 1:
                    output_df.at[idx, "source_row_index"] = int(matches[0])
                    backfilled += 1
                else:
                    removable_rows.append(idx)

            if backfilled:
                changes.append(f"backfilled_source_row_index={backfilled}")
            if removable_rows:
                output_df = output_df.drop(index=removable_rows).reset_index(drop=True)
                changes.append(f"removed_unmappable_rows={len(removable_rows)}")

        if "source_row_index" in output_df.columns:
            before = len(output_df)
            output_df = output_df.drop_duplicates(subset=["source_row_index"], keep="last").reset_index(drop=True)
            removed = before - len(output_df)
            if removed:
                changes.append(f"removed_duplicate_source_row_index={removed}")

    before_question_dedup = len(output_df)
    output_df = output_df.drop_duplicates(subset=["question", "expected_doc", "expected_field"], keep="last").reset_index(
        drop=True
    ) if mode == "edge" and {"question", "expected_doc", "expected_field"}.issubset(output_df.columns) else output_df
    removed_question_dups = before_question_dedup - len(output_df)
    if removed_question_dups:
        changes.append(f"removed_duplicate_question_contract_rows={removed_question_dups}")

    final_rows = len(output_df)
    print(f"mode={mode}")
    print(f"model={model}")
    print(f"input_rows={len(input_df)}")
    print(f"original_rows={original_rows}")
    print(f"final_rows={final_rows}")
    print(f"changes={changes if changes else '[]'}")

    if write and changes:
        save_csv(output_df, output_path)
        print(f"wrote={output_path}")
    elif write:
        print("wrote=no_changes")

    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(sanitize(args.mode, args.model, args.write))


if __name__ == "__main__":
    main()
