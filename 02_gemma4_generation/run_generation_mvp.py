from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from common import (
    DEFAULT_BACKEND,
    INPUT_EDGE,
    INPUT_EVAL,
    OUTPUT_SOURCE_INDEX,
    append_citation,
    build_prompt,
    fallback_answer,
    get_prediction_output_path,
    infer_used_fields,
    load_csv,
    load_generation_config,
    pick_default_model_id,
    resolve_model_config,
    retrieve_top_k,
    save_csv,
    safe_text,
)
from inference.registry import get_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 02_gemma4_generation MVP pipeline.")
    parser.add_argument("--mode", choices=["eval", "edge"], default="eval")
    parser.add_argument("--backend", choices=["mock", "llama_cpp"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k", type=int, default=None, dest="top_k")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_questions(mode: str) -> pd.DataFrame:
    source = INPUT_EVAL if mode == "eval" else INPUT_EDGE
    df, _ = load_csv(source)
    return df


def run_backend(
    backend: str,
    prompt_text: str,
    model_config: dict[str, Any],
    generation_config: dict[str, Any],
) -> Any:
    adapter = get_adapter(backend)
    return adapter.generate(
        prompt_text=prompt_text,
        model_config=model_config,
        generation_config=generation_config,
    )


def main() -> None:
    args = parse_args()
    generation_config = load_generation_config()
    backend = args.backend or str(generation_config.get("backend", DEFAULT_BACKEND))
    model_id = args.model or pick_default_model_id()
    model_config = resolve_model_config(model_id)
    top_k = args.top_k or int(generation_config.get("top_k", 3))
    retrieval_threshold = float(generation_config.get("retrieval_score_threshold", 1.0))
    fallback_on_low_score = bool(generation_config.get("fallback_on_low_retrieval_score", True))

    if backend == "llama_cpp" and str(model_config.get("runtime")) != "llama_cpp":
        raise ValueError(f"{model_id} 설정은 llama_cpp runtime이 아닙니다: {model_config.get('runtime')}")

    if not OUTPUT_SOURCE_INDEX.exists():
        raise FileNotFoundError(
            f"소스 인덱스가 없습니다: {OUTPUT_SOURCE_INDEX}. 먼저 build_generation_assets.py를 실행하세요."
        )

    source_df, _ = load_csv(OUTPUT_SOURCE_INDEX)
    questions_df = load_questions(args.mode)

    if args.limit is not None:
        questions_df = questions_df.head(args.limit).copy()

    rows: list[dict[str, Any]] = []

    for _, question_row in questions_df.iterrows():
        question = safe_text(question_row.get("question"))
        retrieved_df = retrieve_top_k(question, source_df, top_k=top_k)
        prompt_text = build_prompt(question, retrieved_df)
        cited_doc_ids = [safe_text(value) for value in retrieved_df["문서ID"].tolist() if safe_text(value)]
        top_doc_id = cited_doc_ids[0] if cited_doc_ids else ""
        retrieval_score = float(retrieved_df.iloc[0]["retrieval_score"]) if not retrieved_df.empty else 0.0
        used_fields = infer_used_fields(question)

        if fallback_on_low_score and retrieval_score < retrieval_threshold:
            answer = fallback_answer(cited_doc_ids)
            insufficient_context = True
            finish_reason = "retrieval_threshold"
            latency_ms = 0
            raw_response = None
        else:
            inference_result = run_backend(
                backend=backend,
                prompt_text=prompt_text,
                model_config=model_config,
                generation_config=generation_config,
            )
            if safe_text(inference_result.text):
                answer = append_citation(safe_text(inference_result.text), cited_doc_ids)
                insufficient_context = False
            else:
                answer = fallback_answer(cited_doc_ids)
                insufficient_context = True
            finish_reason = inference_result.finish_reason
            latency_ms = inference_result.latency_ms
            raw_response = inference_result.raw_response

        record: dict[str, Any] = {
            "question": question,
            "answer": answer,
            "top_doc_id": top_doc_id,
            "cited_doc_ids": "|".join(cited_doc_ids),
            "used_fields": "|".join(used_fields),
            "retrieval_score": retrieval_score,
            "insufficient_context": insufficient_context,
            "backend": backend,
            "model_id": model_id,
            "runtime": backend if backend == "mock" else str(model_config.get("runtime", backend)),
            "latency_ms": latency_ms,
            "finish_reason": finish_reason,
            "prompt_text": prompt_text,
            "raw_response": "" if raw_response is None else str(raw_response),
        }

        if args.mode == "eval":
            record["expected_answer"] = safe_text(question_row.get("expected_answer"))
            record["expected_doc_id"] = safe_text(question_row.get("문서ID"))
        else:
            record["expected_doc"] = safe_text(question_row.get("expected_doc"))
            record["expected_field"] = safe_text(question_row.get("expected_field"))

        rows.append(record)

    output_df = pd.DataFrame(rows)
    output_path = get_prediction_output_path(args.mode, model_id)
    save_csv(output_df, output_path)
    print(f"Saved {args.mode} predictions to {output_path}")


if __name__ == "__main__":
    main()
