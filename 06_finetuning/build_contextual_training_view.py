from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from common import (
    DEFAULT_MODEL_ID,
    CONTEXTUAL_SCHEMA_VERSION,
    CONTEXTUAL_VALIDATION_VERSION,
    PROJECT_ROOT,
    build_contextual_input,
    build_contextual_instruction,
    build_contextual_schema_markdown,
    build_run_dir,
    compute_full_training_token_count,
    compute_prompt_token_count,
    ensure_parent,
    get_default_frozen_paths,
    get_context_schema_defaults,
    load_apartment_doc_lookup,
    load_csv,
    load_model_config,
    normalize_text,
    parse_pipe_values,
    remove_file_if_exists,
    resolve_context_docs,
    replace_file_atomically,
    safe_text,
    save_json,
    summarize_counts,
)


def parse_args() -> argparse.Namespace:
    defaults = get_context_schema_defaults()
    parser = argparse.ArgumentParser(
        description="Build a run-local context-aware train/valid JSONL view for stage 06 finetuning."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--max-docs", type=int, default=int(defaults["max_docs"]))
    parser.add_argument("--max-description-chars", type=int, default=int(defaults["max_description_chars"]))
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=int(defaults["max_seq_length"]),
        help="Full training sequence budget, including prompt, output, and EOS when present.",
    )
    return parser.parse_args()


def resolve_tokenizer(model_id: str) -> AutoTokenizer:
    model_config = load_model_config(model_id)
    model_source = model_config.get("local_dir") or model_config.get("hf_model_id")
    if not model_source:
        raise RuntimeError(f"Unable to resolve model source for tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_jsonl_record(
    row: dict[str, Any],
    doc_lookup: dict[str, dict[str, str]],
    *,
    max_docs: int,
    max_description_chars: int,
) -> dict[str, str]:
    return {
        "instruction": build_contextual_instruction(row),
        "input": build_contextual_input(
            row,
            doc_lookup,
            max_docs=max_docs,
            max_description_chars=max_description_chars,
        ),
        "output": safe_text(row.get("answer")),
    }


def write_jsonl(records: list[dict[str, str]], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_split_records(
    split_df: Any,
    doc_lookup: dict[str, dict[str, str]],
    tokenizer: AutoTokenizer,
    *,
    max_docs: int,
    max_description_chars: int,
    max_seq_length: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    records: list[dict[str, str]] = []
    prompt_token_counts: list[int] = []
    full_sequence_token_counts: list[int] = []
    rows_over_prompt_budget: list[dict[str, Any]] = []
    rows_over_full_sequence_budget: list[dict[str, Any]] = []
    doc_count_distribution: dict[str, int] = {"0": 0, "1": 0, "2": 0}
    rows_excluded_no_context: list[dict[str, Any]] = []

    for _, row in split_df.iterrows():
        row_dict = row.to_dict()
        resolved_docs = resolve_context_docs(row_dict, doc_lookup, max_docs=max_docs)
        if not resolved_docs:
            rows_excluded_no_context.append(
                {
                    "source_row_index": int(row_dict.get("source_row_index"))
                    if safe_text(row_dict.get("source_row_index"))
                    else None,
                    "stable_id": safe_text(row_dict.get("stable_id")),
                    "question": safe_text(row_dict.get("question")),
                    "query_type": safe_text(row_dict.get("query_type")),
                    "top_doc_id": safe_text(row_dict.get("top_doc_id")),
                    "cited_doc_ids": safe_text(row_dict.get("cited_doc_ids")),
                }
            )
            doc_count_distribution["0"] = doc_count_distribution.get("0", 0) + 1
            continue

        record = build_jsonl_record(
            row_dict,
            doc_lookup,
            max_docs=max_docs,
            max_description_chars=max_description_chars,
        )
        if not normalize_text(record["input"]):
            raise ValueError(f"Encountered empty contextual input for source_row_index={row_dict.get('source_row_index')}")
        if not normalize_text(record["output"]):
            raise ValueError(f"Encountered empty output for source_row_index={row_dict.get('source_row_index')}")

        doc_bucket = str(min(len(resolved_docs), 2))
        doc_count_distribution[doc_bucket] = doc_count_distribution.get(doc_bucket, 0) + 1

        prompt_token_count = compute_prompt_token_count(
            record.get("instruction", ""),
            record.get("input", ""),
            tokenizer=tokenizer,
        )
        full_sequence_token_count = compute_full_training_token_count(
            record.get("instruction", ""),
            record.get("input", ""),
            record.get("output", ""),
            tokenizer=tokenizer,
        )
        prompt_token_counts.append(prompt_token_count)
        full_sequence_token_counts.append(full_sequence_token_count)
        if prompt_token_count > max_seq_length:
            rows_over_prompt_budget.append(
                {
                    "source_row_index": int(row_dict.get("source_row_index"))
                    if safe_text(row_dict.get("source_row_index"))
                    else None,
                    "stable_id": safe_text(row_dict.get("stable_id")),
                    "question": safe_text(row_dict.get("question")),
                    "prompt_tokens": prompt_token_count,
                }
            )
        if full_sequence_token_count > max_seq_length:
            rows_over_full_sequence_budget.append(
                {
                    "source_row_index": int(row_dict.get("source_row_index"))
                    if safe_text(row_dict.get("source_row_index"))
                    else None,
                    "stable_id": safe_text(row_dict.get("stable_id")),
                    "question": safe_text(row_dict.get("question")),
                    "prompt_tokens": prompt_token_count,
                    "full_sequence_tokens": full_sequence_token_count,
                }
            )

        records.append(record)

    stats = {
        "row_count": len(records),
        "prompt_token_stats": {
            "min": min(prompt_token_counts) if prompt_token_counts else 0,
            "max": max(prompt_token_counts) if prompt_token_counts else 0,
            "avg": round(sum(prompt_token_counts) / len(prompt_token_counts), 2) if prompt_token_counts else 0.0,
        },
        "full_sequence_token_stats": {
            "min": min(full_sequence_token_counts) if full_sequence_token_counts else 0,
            "max": max(full_sequence_token_counts) if full_sequence_token_counts else 0,
            "avg": round(sum(full_sequence_token_counts) / len(full_sequence_token_counts), 2)
            if full_sequence_token_counts
            else 0.0,
        },
        "rows_over_prompt_budget_count": len(rows_over_prompt_budget),
        "rows_over_prompt_budget": rows_over_prompt_budget,
        "rows_over_full_sequence_budget_count": len(rows_over_full_sequence_budget),
        "rows_over_full_sequence_budget": rows_over_full_sequence_budget,
        "doc_count_distribution": doc_count_distribution,
        "rows_excluded_no_context_count": len(rows_excluded_no_context),
        "rows_excluded_no_context": rows_excluded_no_context,
    }
    return records, stats


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else build_run_dir(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    frozen_paths = get_default_frozen_paths(args.model)
    candidates_path = frozen_paths["training_candidates_file"]
    if not candidates_path.exists():
        raise FileNotFoundError(f"Training candidates file not found: {candidates_path}")

    candidates_df = load_csv(candidates_path)
    if "split" not in candidates_df.columns:
        raise KeyError("training_candidates file must contain a split column")

    train_df = candidates_df[candidates_df["split"] == "train"].copy()
    valid_df = candidates_df[candidates_df["split"] == "valid"].copy()
    if train_df.empty or valid_df.empty:
        raise RuntimeError("Frozen training candidates must contain both train and valid rows")

    tokenizer = resolve_tokenizer(args.model)
    doc_lookup = load_apartment_doc_lookup()

    train_records, train_stats = build_split_records(
        train_df,
        doc_lookup,
        tokenizer,
        max_docs=args.max_docs,
        max_description_chars=args.max_description_chars,
        max_seq_length=args.max_seq_length,
    )
    valid_records, valid_stats = build_split_records(
        valid_df,
        doc_lookup,
        tokenizer,
        max_docs=args.max_docs,
        max_description_chars=args.max_description_chars,
        max_seq_length=args.max_seq_length,
    )

    summary = {
        "run_id": args.run_id,
        "project_root": str(PROJECT_ROOT),
        "model_id": args.model,
        "schema_version": CONTEXTUAL_SCHEMA_VERSION,
        "validation_version": CONTEXTUAL_VALIDATION_VERSION,
        "sources": {
            "training_candidates_file": str(candidates_path.relative_to(PROJECT_ROOT)),
            "apartment_docs_file": "data/apartment_chatbot_v3.csv",
        },
        "selected_schema_budget": {
            "max_docs": args.max_docs,
            "max_description_chars": args.max_description_chars,
            "max_seq_length": args.max_seq_length,
        },
        "frozen_split_counts": {
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
        },
        "split_counts": {
            "train_rows": int(len(train_records)),
            "valid_rows": int(len(valid_records)),
        },
        "query_type_counts": {
            "train": summarize_counts(train_df, "query_type"),
            "valid": summarize_counts(valid_df, "query_type"),
        },
        "split_stats": {
            "train": train_stats,
            "valid": valid_stats,
        },
    }

    train_out = run_dir / "train_contextual.jsonl"
    valid_out = run_dir / "valid_contextual.jsonl"
    schema_out = run_dir / "schema_v1.md"
    summary_out = run_dir / "context_build_summary.json"
    rejected_summary_out = run_dir / "context_build_summary.rejected.json"
    train_out_tmp = run_dir / "train_contextual.jsonl.tmp"
    valid_out_tmp = run_dir / "valid_contextual.jsonl.tmp"
    schema_out_tmp = run_dir / "schema_v1.md.tmp"
    summary_out_tmp = run_dir / "context_build_summary.json.tmp"

    total_prompt_over_budget = (
        train_stats["rows_over_prompt_budget_count"] + valid_stats["rows_over_prompt_budget_count"]
    )
    total_full_sequence_over_budget = (
        train_stats["rows_over_full_sequence_budget_count"] + valid_stats["rows_over_full_sequence_budget_count"]
    )
    builder_pass = total_full_sequence_over_budget == 0
    summary["builder_pass"] = builder_pass
    summary["schema_status"] = "accepted" if builder_pass else "rejected"
    if builder_pass:
        summary["accepted_at"] = datetime.now().astimezone().isoformat()
    else:
        summary["rejected_at"] = datetime.now().astimezone().isoformat()

    write_jsonl(train_records, train_out_tmp)
    write_jsonl(valid_records, valid_out_tmp)
    schema_out_tmp.write_text(build_contextual_schema_markdown(), encoding="utf-8")
    save_json(summary, summary_out_tmp)

    if not builder_pass:
        save_json(summary, rejected_summary_out)
        for temp_path in (train_out_tmp, valid_out_tmp, schema_out_tmp, summary_out_tmp):
            remove_file_if_exists(temp_path)
        raise RuntimeError(
            "Contextual rows exceeded full training sequence budget: "
            f"{total_full_sequence_over_budget} rows. Prompt over-budget rows: {total_prompt_over_budget}."
        )

    replace_file_atomically(train_out_tmp, train_out)
    replace_file_atomically(valid_out_tmp, valid_out)
    replace_file_atomically(schema_out_tmp, schema_out)
    replace_file_atomically(summary_out_tmp, summary_out)
    remove_file_if_exists(rejected_summary_out)

    print(f"Saved contextual train view to {train_out}")
    print(f"Saved contextual valid view to {valid_out}")
    print(f"Saved schema markdown to {schema_out}")
    print(f"Saved context build summary to {summary_out}")


if __name__ == "__main__":
    main()
