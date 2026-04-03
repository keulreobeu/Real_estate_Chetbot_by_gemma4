from __future__ import annotations

import argparse

from common import (
    get_metrics_output_path,
    get_prediction_output_path,
    load_csv,
    normalize_text,
    pick_default_model_id,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the 02_gemma4_generation MVP outputs.")
    parser.add_argument("--mode", choices=["eval", "edge"], default="eval")
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def contains_expected(answer: str, expected_answer: str) -> bool:
    if not answer or not expected_answer:
        return False
    return normalize_text(expected_answer) in normalize_text(answer)


def evaluate_eval(model_id: str) -> None:
    predictions_df, _ = load_csv(get_prediction_output_path("eval", model_id))

    total_questions = len(predictions_df)
    exact_matches = 0
    contains_matches = 0
    retrieval_hits = 0
    insufficient_context = 0
    latency_total = 0.0

    for _, row in predictions_df.iterrows():
        answer = str(row.get("answer", ""))
        expected_answer = str(row.get("expected_answer", ""))
        expected_doc = str(row.get("expected_doc_id", ""))
        cited_doc_ids = str(row.get("cited_doc_ids", "")).split("|")
        normalized_answer = normalize_text(answer)
        normalized_expected = normalize_text(expected_answer)

        if normalized_answer == normalized_expected:
            exact_matches += 1
        if contains_expected(answer, expected_answer):
            contains_matches += 1
        if expected_doc and expected_doc in cited_doc_ids:
            retrieval_hits += 1
        if bool(row.get("insufficient_context")):
            insufficient_context += 1
        latency_total += float(row.get("latency_ms", 0) or 0)

    metrics = {
        "model_id": model_id,
        "total_questions": total_questions,
        "retrieval_hit_rate": retrieval_hits / total_questions if total_questions else 0.0,
        "exact_match_rate": exact_matches / total_questions if total_questions else 0.0,
        "contains_expected_answer_rate": contains_matches / total_questions if total_questions else 0.0,
        "insufficient_context_rate": insufficient_context / total_questions if total_questions else 0.0,
        "avg_latency_ms": latency_total / total_questions if total_questions else 0.0,
    }
    output_path = get_metrics_output_path("eval", model_id)
    save_json(metrics, output_path)
    print(f"Saved eval metrics to {output_path}")


def evaluate_edge(model_id: str) -> None:
    predictions_df, _ = load_csv(get_prediction_output_path("edge", model_id))

    total_questions = len(predictions_df)
    doc_hits = 0
    field_hits = 0
    insufficient_context = 0
    latency_total = 0.0

    for _, row in predictions_df.iterrows():
        expected_doc = str(row.get("expected_doc", ""))
        expected_field = str(row.get("expected_field", ""))
        cited_doc_ids = str(row.get("cited_doc_ids", "")).split("|")
        used_fields = str(row.get("used_fields", "")).split("|")

        if expected_doc and expected_doc in cited_doc_ids:
            doc_hits += 1
        if expected_field and expected_field in used_fields:
            field_hits += 1
        if bool(row.get("insufficient_context")):
            insufficient_context += 1
        latency_total += float(row.get("latency_ms", 0) or 0)

    metrics = {
        "model_id": model_id,
        "total_questions": total_questions,
        "doc_hit_rate": doc_hits / total_questions if total_questions else 0.0,
        "field_hit_rate": field_hits / total_questions if total_questions else 0.0,
        "insufficient_context_rate": insufficient_context / total_questions if total_questions else 0.0,
        "avg_latency_ms": latency_total / total_questions if total_questions else 0.0,
    }
    output_path = get_metrics_output_path("edge", model_id)
    save_json(metrics, output_path)
    print(f"Saved edge metrics to {output_path}")


def main() -> None:
    args = parse_args()
    model_id = args.model or pick_default_model_id()
    if args.mode == "eval":
        evaluate_eval(model_id)
    else:
        evaluate_edge(model_id)


if __name__ == "__main__":
    main()
