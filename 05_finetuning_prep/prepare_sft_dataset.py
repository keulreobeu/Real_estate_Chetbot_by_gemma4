from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
QA_DIR = DATA_DIR / "qa"
EVAL_DIR = DATA_DIR / "eval"
OUTPUT_DIR = QA_DIR / "finetuning_prep"

DEFAULT_MODEL_ID = "gemma4_2b"
DISCLOSURE_HINTS = (
    "답변 가능 범위",
    "데이터 기준일",
    "비교 기준",
    "근거 문서",
    "현재 데이터로는",
    "판단할 수 없습니다",
    "판단 근거는 데이터셋에 없습니다",
)
TARGET_MIX = {
    "grounded_generation": 0.70,
    "safety_refusal": 0.15,
    "safe_recommendation": 0.15,
}
MIN_GROUNDED_HOLDOUT = 200
MIN_SAFETY_HOLDOUT = 150
MIN_TRAIN_ROWS = 50
MIN_VALID_ROWS = 10
MIN_TRAIN_VALID_POOL = MIN_TRAIN_ROWS + MIN_VALID_ROWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare stage 05 SFT datasets from stage 02/04 outputs.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--edge-predictions", default=str(EVAL_DIR / f"gemma4_generation_edge_predictions_{DEFAULT_MODEL_ID}.csv"))
    parser.add_argument("--eval-predictions", default=str(EVAL_DIR / f"gemma4_generation_eval_predictions_{DEFAULT_MODEL_ID}.csv"))
    parser.add_argument("--edge-dataset", default=str(QA_DIR / "edge_case_eval.csv"))
    parser.add_argument("--eval-dataset", default=str(QA_DIR / "evaluation_dataset.csv"))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def deterministic_hash(*parts: object) -> str:
    payload = "||".join(safe_text(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_flag(value: object) -> str:
    text = safe_text(value).upper()
    if text in {"Y", "N"}:
        return text
    return ""


def bool_from_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = safe_text(value).lower()
    return text in {"1", "true", "t", "yes", "y"}


def has_required_contract_fields(row: pd.Series) -> bool:
    return all(safe_text(row.get(column)) for column in ("answer_type", "match_status", "query_type"))


def has_disclosure_language(answer: str) -> bool:
    normalized = answer.replace(" ", "")
    return any(hint.replace(" ", "") in normalized for hint in DISCLOSURE_HINTS)


def build_base_frame(edge_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    edge = edge_df.copy()
    eval_predictions = eval_df.copy()
    edge["source_dataset"] = "edge"
    eval_predictions["source_dataset"] = "eval"
    combined = pd.concat([edge, eval_predictions], ignore_index=True, sort=False)

    for column in (
        "source_row_index",
        "question",
        "answer",
        "answer_type",
        "match_status",
        "query_type",
        "top_doc_id",
        "cited_doc_ids",
        "used_fields",
        "retrieval_score",
        "insufficient_context",
        "latency_ms",
        "expected_answer",
        "expected_doc_id",
        "expected_answer_type",
        "expected_match_status",
        "expected_doc",
        "expected_field",
        "expected_router_type",
        "must_include",
        "must_not_include",
        "must_not_recommend",
        "must_disclose_limit",
    ):
        if column not in combined.columns:
            combined[column] = ""

    combined["question"] = combined["question"].map(safe_text)
    combined["answer"] = combined["answer"].map(safe_text)
    combined["answer_type"] = combined["answer_type"].map(safe_text)
    combined["match_status"] = combined["match_status"].map(safe_text)
    combined["query_type"] = combined["query_type"].map(safe_text)
    combined["must_not_recommend"] = combined["must_not_recommend"].map(normalize_flag)
    combined["must_disclose_limit"] = combined["must_disclose_limit"].map(normalize_flag)
    combined["insufficient_context"] = combined["insufficient_context"].map(bool_from_value)
    combined["source_row_index"] = combined["source_row_index"].map(safe_text)
    combined["stable_id"] = combined.apply(
        lambda row: deterministic_hash(
            row.get("source_dataset"),
            row.get("source_row_index"),
            row.get("question"),
            row.get("answer_type"),
        ),
        axis=1,
    )
    combined["source_key"] = combined.apply(
        lambda row: f"{safe_text(row.get('source_dataset'))}|{safe_text(row.get('source_row_index'))}",
        axis=1,
    )
    return combined


def classify_row(row: pd.Series) -> tuple[bool, str, str]:
    question = safe_text(row.get("question"))
    answer = safe_text(row.get("answer"))
    answer_type = safe_text(row.get("answer_type"))
    match_status = safe_text(row.get("match_status"))
    must_not_recommend = normalize_flag(row.get("must_not_recommend"))
    must_disclose_limit = normalize_flag(row.get("must_disclose_limit"))

    if not question:
        return False, "missing_question", "excluded"
    if not answer:
        return False, "missing_answer", "excluded"
    if not has_required_contract_fields(row):
        return False, "missing_contract_fields", "excluded"

    if answer_type == "grounded_generation":
        if bool_from_value(row.get("insufficient_context")):
            return False, "grounded_generation_insufficient_context", "excluded"
        return True, "", "grounded_generation"

    if answer_type in {"unsupported_comparative_response", "no_match_response", "unknown_response"}:
        return True, "", "safety_refusal"

    if answer_type in {"recommendation", "comparison_recommendation"}:
        if match_status != "EXACT_MATCH":
            return False, "recommendation_not_exact_match", "excluded"
        if must_not_recommend == "Y":
            return False, "recommendation_flagged_must_not_recommend", "excluded"
        if must_disclose_limit == "Y" and not has_disclosure_language(answer):
            return False, "recommendation_missing_disclosure_language", "excluded"
        return True, "", "safe_recommendation"

    if answer_type == "apartment_fact_lookup":
        return False, "deterministic_fact_lookup_excluded", "excluded"
    if answer_type == "meta_answer":
        return False, "deterministic_meta_excluded", "excluded"
    if answer_type == "knowledge_answer":
        return False, "deterministic_knowledge_excluded", "excluded"
    if answer_type == "fallback_answer":
        return False, "fallback_answer_excluded", "excluded"

    return False, f"unsupported_answer_type:{answer_type or 'empty'}", "excluded"


def pick_deterministic(df: pd.DataFrame, count: int, salt: str) -> pd.DataFrame:
    if count <= 0 or df.empty:
        return df.head(0).copy()
    ranked = df.copy()
    ranked["_rank"] = ranked["stable_id"].map(lambda value: deterministic_hash(salt, value))
    ranked = ranked.sort_values("_rank", kind="stable")
    return ranked.head(count).drop(columns=["_rank"])


def compute_grounded_holdout_n(pool_size: int) -> int:
    if pool_size <= 0:
        return 0
    if pool_size < 80:
        return max(0, min(pool_size, int(pool_size * 0.20)))
    return min(MIN_GROUNDED_HOLDOUT, max(20, int(pool_size * 0.25)))


def compute_safety_holdout_n(pool_size: int, included_size: int) -> int:
    if pool_size <= 0:
        return 0
    ratio = 0.25 if included_size < 250 else 0.35
    minimum = 0 if included_size < MIN_TRAIN_VALID_POOL else 30
    return min(MIN_SAFETY_HOLDOUT, max(minimum, int(pool_size * ratio)))


def release_holdout_rows(df: pd.DataFrame, count: int, salt: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if count <= 0 or df.empty:
        return df.copy(), df.head(0).copy()
    ranked = df.copy()
    ranked["_rank"] = ranked["stable_id"].map(lambda value: deterministic_hash(salt, value))
    ranked = ranked.sort_values("_rank", kind="stable")
    keep_n = max(0, len(ranked) - count)
    kept = ranked.head(keep_n).drop(columns=["_rank"])
    released = ranked.tail(count).drop(columns=["_rank"])
    return kept, released


def stratified_valid_split(df: pd.DataFrame, ratio: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_parts: list[pd.DataFrame] = []
    train_parts: list[pd.DataFrame] = []
    for bucket, bucket_df in df.groupby("sft_bucket", sort=False):
        size = len(bucket_df)
        if size <= 1:
            train_parts.append(bucket_df)
            continue
        valid_n = int(round(size * ratio))
        if valid_n == 0:
            valid_n = 1 if size >= 5 else 0
        if valid_n >= size:
            valid_n = size - 1
        valid_df = pick_deterministic(bucket_df, valid_n, f"valid::{bucket}")
        train_df = bucket_df[~bucket_df["stable_id"].isin(valid_df["stable_id"])]
        valid_parts.append(valid_df)
        train_parts.append(train_df)
    valid = pd.concat(valid_parts, ignore_index=True, sort=False) if valid_parts else df.head(0).copy()
    train = pd.concat(train_parts, ignore_index=True, sort=False) if train_parts else df.head(0).copy()
    return train, valid


def build_final_selection(
    included_df: pd.DataFrame,
    safety_universe_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grounded_pool = included_df[included_df["sft_bucket"] == "grounded_generation"].copy()
    grounded_holdout_n = compute_grounded_holdout_n(len(grounded_pool))
    grounded_holdout = pick_deterministic(grounded_pool, grounded_holdout_n, "holdout::grounded_generation")

    remaining = included_df[~included_df["stable_id"].isin(grounded_holdout["stable_id"])].copy()
    safety_pool = safety_universe_df[
        ~safety_universe_df["stable_id"].isin(grounded_holdout["stable_id"])
        & (
            (safety_universe_df["must_not_recommend"] == "Y")
            | (safety_universe_df["must_disclose_limit"] == "Y")
            | (safety_universe_df["sft_bucket"] == "safety_refusal")
        )
    ].copy()
    safety_holdout_n = compute_safety_holdout_n(len(safety_pool), len(included_df))
    safety_holdout = pick_deterministic(safety_pool, safety_holdout_n, "holdout::edge_safety")

    remaining = remaining[~remaining["stable_id"].isin(safety_holdout["stable_id"])].copy()

    train_valid_room = len(remaining)
    if train_valid_room < MIN_TRAIN_VALID_POOL:
        release_needed = MIN_TRAIN_VALID_POOL - train_valid_room
        grounded_holdout, grounded_released = release_holdout_rows(
            grounded_holdout,
            min(release_needed, len(grounded_holdout)),
            "release::grounded_holdout",
        )
        if not grounded_released.empty:
            remaining = pd.concat([remaining, grounded_released], ignore_index=True, sort=False)
        train_valid_room = len(remaining)
        if train_valid_room < MIN_TRAIN_VALID_POOL:
            release_needed = MIN_TRAIN_VALID_POOL - train_valid_room
            safety_holdout, safety_released = release_holdout_rows(
                safety_holdout,
                min(release_needed, len(safety_holdout)),
                "release::safety_holdout",
            )
            if not safety_released.empty:
                remaining = pd.concat([remaining, safety_released], ignore_index=True, sort=False)

    grounded_train_pool = remaining[remaining["sft_bucket"] == "grounded_generation"].copy()
    safety_train_pool = remaining[remaining["sft_bucket"] == "safety_refusal"].copy()
    recommendation_train_pool = remaining[remaining["sft_bucket"] == "safe_recommendation"].copy()

    grounded_count = len(grounded_train_pool)
    target_safety = min(len(safety_train_pool), round((grounded_count * TARGET_MIX["safety_refusal"]) / TARGET_MIX["grounded_generation"])) if grounded_count else 0
    target_recommendation = min(
        len(recommendation_train_pool),
        round((grounded_count * TARGET_MIX["safe_recommendation"]) / TARGET_MIX["grounded_generation"]),
    ) if grounded_count else 0

    baseline_pool = grounded_count + target_safety + target_recommendation
    if baseline_pool < MIN_TRAIN_VALID_POOL:
        remaining_capacity = MIN_TRAIN_VALID_POOL - baseline_pool
        safety_headroom = max(0, len(safety_train_pool) - target_safety)
        safety_boost = min(safety_headroom, remaining_capacity)
        target_safety += safety_boost
        remaining_capacity -= safety_boost
        if remaining_capacity > 0:
            recommendation_headroom = max(0, len(recommendation_train_pool) - target_recommendation)
            recommendation_boost = min(recommendation_headroom, remaining_capacity)
            target_recommendation += recommendation_boost

    safety_selected = pick_deterministic(safety_train_pool, target_safety, "train_pool::safety_refusal")
    recommendation_selected = pick_deterministic(
        recommendation_train_pool,
        target_recommendation,
        "train_pool::safe_recommendation",
    )

    train_valid_pool = pd.concat(
        [grounded_train_pool, safety_selected, recommendation_selected],
        ignore_index=True,
        sort=False,
    )
    train_df, valid_df = stratified_valid_split(train_valid_pool)

    if len(valid_df) < MIN_VALID_ROWS and len(train_df) > MIN_TRAIN_ROWS:
        promote_n = min(MIN_VALID_ROWS - len(valid_df), len(train_df) - MIN_TRAIN_ROWS)
        promote_df = pick_deterministic(train_df, promote_n, "valid::global_topup")
        if not promote_df.empty:
            train_df = train_df[~train_df["stable_id"].isin(promote_df["stable_id"])].copy()
            valid_df = pd.concat([valid_df, promote_df], ignore_index=True, sort=False)

    downsampled_ids = set(safety_train_pool["stable_id"]) - set(safety_selected["stable_id"])
    downsampled_ids |= set(recommendation_train_pool["stable_id"]) - set(recommendation_selected["stable_id"])
    downsampled_df = remaining[remaining["stable_id"].isin(downsampled_ids)].copy()

    return grounded_holdout, safety_holdout, train_df, valid_df, downsampled_df


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in df.itertuples(index=False):
            payload = {
                "instruction": safe_text(getattr(row, "question", "")),
                "input": "",
                "output": safe_text(getattr(row, "answer", "")),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def summarize_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in df[column].value_counts(dropna=False).to_dict().items()}


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edge_predictions = load_csv(Path(args.edge_predictions))
    eval_predictions = load_csv(Path(args.eval_predictions))
    edge_dataset = load_csv(Path(args.edge_dataset))
    eval_dataset = load_csv(Path(args.eval_dataset))

    combined = build_base_frame(edge_predictions, eval_predictions)
    combined["include_in_sft"] = False
    combined["exclusion_reason"] = ""
    combined["sft_bucket"] = "excluded"
    combined["split"] = "excluded"

    duplicate_source_mask = combined.duplicated(subset=["source_key"], keep="first")
    duplicate_qa_mask = combined.duplicated(subset=["question", "answer"], keep="first")

    for index, row in combined.iterrows():
        if duplicate_source_mask.iloc[index] and safe_text(row.get("source_row_index")):
            combined.at[index, "exclusion_reason"] = "duplicate_source_dataset_row"
            continue
        if duplicate_qa_mask.iloc[index]:
            combined.at[index, "exclusion_reason"] = "duplicate_question_answer"
            continue
        include, reason, bucket = classify_row(row)
        combined.at[index, "include_in_sft"] = include
        combined.at[index, "exclusion_reason"] = reason
        combined.at[index, "sft_bucket"] = bucket

    included = combined[combined["include_in_sft"]].copy()
    safety_universe = combined[
        combined["exclusion_reason"].isin(
            {
                "",
                "recommendation_flagged_must_not_recommend",
                "recommendation_missing_disclosure_language",
                "recommendation_not_exact_match",
            }
        )
    ].copy()
    grounded_holdout, safety_holdout, train_df, valid_df, downsampled_df = build_final_selection(included, safety_universe)

    if not downsampled_df.empty:
        combined.loc[combined["stable_id"].isin(downsampled_df["stable_id"]), "include_in_sft"] = False
        combined.loc[combined["stable_id"].isin(downsampled_df["stable_id"]), "exclusion_reason"] = "downsampled_outside_target_mix"
        combined.loc[combined["stable_id"].isin(downsampled_df["stable_id"]), "sft_bucket"] = "excluded"

    combined.loc[combined["stable_id"].isin(train_df["stable_id"]), "split"] = "train"
    combined.loc[combined["stable_id"].isin(valid_df["stable_id"]), "split"] = "valid"
    combined.loc[combined["stable_id"].isin(grounded_holdout["stable_id"]), "split"] = "holdout_grounded_generation"
    combined.loc[combined["stable_id"].isin(safety_holdout["stable_id"]), "split"] = "holdout_edge_safety"

    final_train = combined[combined["split"] == "train"].copy()
    final_valid = combined[combined["split"] == "valid"].copy()
    final_grounded_holdout = combined[combined["split"] == "holdout_grounded_generation"].copy()
    final_safety_holdout = combined[combined["split"] == "holdout_edge_safety"].copy()
    rejected = combined[combined["split"] == "excluded"].copy()

    preferred_columns = [
        "source_dataset",
        "source_row_index",
        "question",
        "answer",
        "answer_type",
        "match_status",
        "query_type",
        "top_doc_id",
        "cited_doc_ids",
        "used_fields",
        "retrieval_score",
        "insufficient_context",
        "latency_ms",
        "expected_answer",
        "expected_doc_id",
        "expected_answer_type",
        "expected_match_status",
        "expected_doc",
        "expected_field",
        "expected_router_type",
        "must_include",
        "must_not_include",
        "must_not_recommend",
        "must_disclose_limit",
        "include_in_sft",
        "exclusion_reason",
        "sft_bucket",
        "split",
        "stable_id",
    ]
    existing_columns = [column for column in preferred_columns if column in combined.columns]

    candidate_path = output_dir / f"training_candidates_{args.model}.csv"
    rejected_path = output_dir / f"training_rejected_{args.model}.csv"
    grounded_holdout_path = output_dir / "holdout_grounded_generation.csv"
    safety_holdout_path = output_dir / "holdout_edge_safety.csv"
    train_jsonl_path = output_dir / f"train_{args.model}.jsonl"
    valid_jsonl_path = output_dir / f"valid_{args.model}.jsonl"
    summary_path = output_dir / f"dataset_summary_{args.model}.json"

    save_csv(combined[existing_columns], candidate_path)
    save_csv(rejected[existing_columns], rejected_path)
    save_csv(final_grounded_holdout[existing_columns], grounded_holdout_path)
    save_csv(final_safety_holdout[existing_columns], safety_holdout_path)
    write_jsonl(final_train, train_jsonl_path)
    write_jsonl(final_valid, valid_jsonl_path)

    summary = {
        "model_id": args.model,
        "input_rows": {
            "edge_predictions": int(len(edge_predictions)),
            "eval_predictions": int(len(eval_predictions)),
            "edge_dataset": int(len(edge_dataset)),
            "eval_dataset": int(len(eval_dataset)),
        },
        "candidate_rows": int(len(combined)),
        "rejected_rows": int(len(rejected)),
        "train_rows": int(len(final_train)),
        "valid_rows": int(len(final_valid)),
        "holdout_grounded_generation_rows": int(len(final_grounded_holdout)),
        "holdout_edge_safety_rows": int(len(final_safety_holdout)),
        "train_bucket_counts": summarize_counts(final_train, "sft_bucket"),
        "valid_bucket_counts": summarize_counts(final_valid, "sft_bucket"),
        "rejected_reason_counts": summarize_counts(rejected, "exclusion_reason"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
