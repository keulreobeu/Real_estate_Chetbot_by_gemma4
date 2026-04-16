from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = DATA_DIR / "eval"
QA_DIR = DATA_DIR / "qa"
OUTPUT_DIR = EVAL_DIR / "generation_optimization"
DEFAULT_MODEL_ID = "gemma4_2b"

SUBJECTIVE_COMPARATIVE_HINTS = (
    "괜찮은",
    "무난한",
    "추천할 만한",
    "편한",
    "실거주 괜찮은",
    "실거주 좋은",
    "살기 좋은",
)
AREA_BAND_HINTS = ("소형", "중소형", "중형", "중대형", "대형")
DISCLOSURE_HINTS = (
    "답변 가능 범위",
    "데이터 기준",
    "비교 기준",
    "근거 문서",
    "현재 데이터로는",
    "판단 보류",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze edge failures before 06 finetuning.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--edge-predictions",
        default=str(EVAL_DIR / f"gemma4_generation_edge_predictions_{DEFAULT_MODEL_ID}.csv"),
    )
    parser.add_argument("--edge-dataset", default=str(QA_DIR / "edge_case_eval.csv"))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def has_limit_disclosure(answer: str) -> bool:
    normalized = answer.replace(" ", "")
    return any(hint.replace(" ", "") in normalized for hint in DISCLOSURE_HINTS)


def classify_router_mismatch_family(row: pd.Series) -> str:
    expected_router = safe_text(row.get("expected_router_type"))
    actual_router = safe_text(row.get("query_type"))
    question = safe_text(row.get("question"))

    if not expected_router or expected_router == actual_router:
        return ""
    if expected_router == "RECOMMEND_COMPARATIVE" and actual_router == "RECOMMEND_STRUCTURED":
        if any(hint in question for hint in SUBJECTIVE_COMPARATIVE_HINTS):
            return "subjective_quality_structured_instead_of_comparative"
        if "비교" in question and any(hint in question for hint in AREA_BAND_HINTS):
            return "area_band_comparative_structured_instead_of_comparative"
        return "comparative_expected_but_structured_actual"
    if expected_router == "RECOMMEND_STRUCTURED" and actual_router == "RECOMMEND_COMPARATIVE":
        if "비교" in question and any(hint in question for hint in AREA_BAND_HINTS):
            return "area_band_structured_but_comparative_actual"
        return "structured_expected_but_comparative_actual"
    if actual_router == "GENERAL_RETRIEVAL_QA":
        return "fell_through_to_grounded_generation"
    return f"{expected_router.lower()}_to_{actual_router.lower()}"


def add_failure_flags(df: pd.DataFrame) -> pd.DataFrame:
    analyzed = df.copy()
    for column in [
        "question",
        "answer_type",
        "match_status",
        "query_type",
        "expected_router_type",
        "expected_match_status",
        "must_not_recommend",
        "must_disclose_limit",
        "top_doc_id",
        "answer",
    ]:
        if column not in analyzed.columns:
            analyzed[column] = ""
        analyzed[column] = analyzed[column].map(safe_text)

    analyzed["is_legacy_schema"] = (
        analyzed["answer_type"].eq("") | analyzed["match_status"].eq("") | analyzed["query_type"].eq("")
    )
    analyzed["router_mismatch"] = (
        analyzed["expected_router_type"].ne("") & analyzed["query_type"].ne(analyzed["expected_router_type"])
    )
    analyzed["match_status_mismatch"] = (
        analyzed["expected_match_status"].ne("")
        & analyzed["match_status"].ne(analyzed["expected_match_status"])
    )
    analyzed["unsafe_recommendation"] = analyzed["must_not_recommend"].eq("Y") & analyzed["top_doc_id"].ne("")
    analyzed["disclosure_required"] = analyzed["must_disclose_limit"].eq("Y")
    analyzed["has_limit_disclosure"] = analyzed["answer"].map(has_limit_disclosure)
    analyzed["disclosure_miss"] = analyzed["disclosure_required"] & ~analyzed["has_limit_disclosure"]
    analyzed["area_band_unknown_candidate"] = (
        analyzed["question"].map(lambda q: any(token in safe_text(q) for token in AREA_BAND_HINTS))
        & analyzed["expected_match_status"].eq("UNKNOWN")
    )
    analyzed["router_mismatch_family"] = analyzed.apply(classify_router_mismatch_family, axis=1)
    return analyzed


def build_router_family_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["router_mismatch_family", "rows"])
    return (
        df["router_mismatch_family"]
        .fillna("")
        .map(safe_text)
        .replace("", "unclassified_router_mismatch")
        .value_counts()
        .rename_axis("router_mismatch_family")
        .reset_index(name="rows")
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    edge_predictions = load_csv(Path(args.edge_predictions))
    edge_dataset = load_csv(Path(args.edge_dataset))

    analyzed = add_failure_flags(edge_predictions)
    current_schema = analyzed[~analyzed["is_legacy_schema"]].copy()

    unsafe_queue = current_schema[current_schema["unsafe_recommendation"]].copy()
    hard_negative_queue = unsafe_queue[
        [
            "source_row_index",
            "question",
            "answer",
            "answer_type",
            "match_status",
            "query_type",
            "expected_router_type",
            "expected_match_status",
            "must_not_recommend",
            "must_disclose_limit",
            "top_doc_id",
            "used_fields",
            "area_band_unknown_candidate",
        ]
    ].copy()
    hard_negative_queue["recommended_action"] = hard_negative_queue["area_band_unknown_candidate"].map(
        lambda value: "route_to_unknown_before_recommendation" if value else "review_safety_rule"
    )

    router_mismatch_queue = current_schema[current_schema["router_mismatch"]].copy()
    router_mismatch_review = router_mismatch_queue[
        [
            "source_row_index",
            "question",
            "query_type",
            "expected_router_type",
            "answer_type",
            "match_status",
            "expected_match_status",
            "top_doc_id",
            "router_mismatch_family",
        ]
    ].copy()
    router_family_counts = build_router_family_counts(router_mismatch_queue)

    legacy_rows = analyzed[analyzed["is_legacy_schema"]].copy()
    legacy_plan = legacy_rows[["source_row_index", "question"]].copy()
    legacy_plan["recommended_action"] = "regenerate_with_current_schema"

    bucket_rows = []
    for name, mask in {
        "legacy_schema": analyzed["is_legacy_schema"],
        "unsafe_recommendation": analyzed["unsafe_recommendation"],
        "unsafe_recommendation_current_schema": ~analyzed["is_legacy_schema"] & analyzed["unsafe_recommendation"],
        "router_mismatch": analyzed["router_mismatch"],
        "router_mismatch_current_schema": ~analyzed["is_legacy_schema"] & analyzed["router_mismatch"],
        "match_status_mismatch": analyzed["match_status_mismatch"],
        "match_status_mismatch_current_schema": ~analyzed["is_legacy_schema"] & analyzed["match_status_mismatch"],
        "disclosure_miss": analyzed["disclosure_miss"],
        "disclosure_miss_current_schema": ~analyzed["is_legacy_schema"] & analyzed["disclosure_miss"],
        "area_band_unknown_candidate": analyzed["area_band_unknown_candidate"],
        "area_band_unknown_candidate_current_schema": ~analyzed["is_legacy_schema"] & analyzed["area_band_unknown_candidate"],
    }.items():
        bucket_rows.append({"bucket": name, "rows": int(mask.sum())})
    bucket_df = pd.DataFrame(bucket_rows)

    summary = {
        "model_id": args.model,
        "input_rows": {
            "edge_predictions": int(len(edge_predictions)),
            "edge_dataset": int(len(edge_dataset)),
        },
        "buckets": {row["bucket"]: int(row["rows"]) for row in bucket_rows},
        "top_router_mismatch_families": router_family_counts.head(5).to_dict(orient="records"),
        "legacy_source_row_index_min": safe_text(legacy_rows["source_row_index"].min()) if not legacy_rows.empty else "",
        "legacy_source_row_index_max": safe_text(legacy_rows["source_row_index"].max()) if not legacy_rows.empty else "",
        "recommended_next_step": "reduce router mismatch families first, then recover grounded-generation supply before 06",
    }

    save_csv(bucket_df, output_dir / f"edge_failure_buckets_{args.model}.csv")
    save_csv(hard_negative_queue, output_dir / f"hard_negative_review_queue_{args.model}.csv")
    save_csv(router_mismatch_review, output_dir / f"router_mismatch_review_queue_{args.model}.csv")
    save_csv(router_family_counts, output_dir / f"router_mismatch_family_counts_{args.model}.csv")
    save_csv(legacy_plan, output_dir / f"legacy_edge_regeneration_plan_{args.model}.csv")
    save_json(summary, output_dir / f"generation_optimization_summary_{args.model}.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
