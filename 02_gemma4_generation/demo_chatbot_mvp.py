from __future__ import annotations

import argparse
import re
from typing import Any

from common import (
    DEFAULT_BACKEND,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    INPUT_KNOWLEDGE,
    INPUT_MAIN,
    OUTPUT_SOURCE_INDEX,
    append_citation,
    build_prompt,
    fallback_answer,
    load_csv,
    load_generation_config,
    pick_default_model_id,
    resolve_model_config,
    retrieve_top_k,
    safe_text,
)
from inference.registry import get_adapter
from query_service import answer_query, determine_data_cutoff, route_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local demo chatbot MVP for 02_gemma4_generation.")
    parser.add_argument("--backend", choices=["mock", "transformers", "llama_cpp"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k", type=int, default=None, dest="top_k")
    parser.add_argument("--retrieval-threshold", type=float, default=None)
    parser.add_argument("--question", default=None, help="Single question mode. If omitted, interactive mode starts.")
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repeat-penalty", type=float, default=None)
    return parser.parse_args()


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


def normalize_generated_answer(text: str) -> str:
    normalized = text.strip()
    normalized = re.sub(r"^\s*[-*]?\s*answer\s*:\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.split(r"\n\s*[-*]?\s*cited_doc_ids\s*:", normalized, maxsplit=1, flags=re.IGNORECASE)[0]
    normalized = re.split(r"<turn\|>", normalized, maxsplit=1, flags=re.IGNORECASE)[0]
    normalized = re.split(r"근거\s*문서\s*:", normalized, maxsplit=1, flags=re.IGNORECASE)[0]
    return normalized.strip()


def ask_once(
    question: str,
    source_df: Any,
    detailed_df: Any,
    knowledge_df: Any,
    backend: str,
    model_id: str,
    model_config: dict[str, Any],
    generation_config: dict[str, Any],
    top_k: int,
    retrieval_threshold: float,
    fallback_on_low_score: bool,
) -> dict[str, Any]:
    request_timeout_seconds = int(generation_config.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS))
    retrieved_df = retrieve_top_k(question, source_df, top_k=top_k)
    cited_doc_ids = [safe_text(value) for value in retrieved_df["문서ID"].tolist() if safe_text(value)]
    top_doc_id = cited_doc_ids[0] if cited_doc_ids else ""
    retrieval_score = float(retrieved_df.iloc[0]["retrieval_score"]) if not retrieved_df.empty else 0.0

    routed_result = answer_query(question, detailed_df=detailed_df, knowledge_df=knowledge_df)
    if routed_result is not None:
        answer = append_citation(routed_result["answer"], routed_result["cited_doc_ids"])
        finish_reason = "query_contract"
        latency_ms = 0
        cited_doc_ids = routed_result["cited_doc_ids"]
        top_doc_id = routed_result["top_doc_id"]
        answer_type = routed_result["answer_type"]
        match_status = routed_result["match_status"]
        query_type = routed_result["query_type"]
        used_fields = routed_result.get("used_fields", [])
        data_cutoff = routed_result.get("data_cutoff", determine_data_cutoff(detailed_df))
        limitations = routed_result.get("limitations", [])
    elif fallback_on_low_score and retrieval_score < retrieval_threshold:
        answer = fallback_answer(cited_doc_ids)
        finish_reason = "retrieval_threshold"
        latency_ms = 0
        answer_type = "insufficient_context_response"
        match_status = "UNKNOWN"
        query_type = route_query(question, detailed_df, knowledge_df)["query_type"]
        used_fields = []
        data_cutoff = determine_data_cutoff(detailed_df)
        limitations = ["retrieval_score_below_threshold"]
    else:
        prompt_text = build_prompt(question, retrieved_df)
        inference_result = run_backend(
            backend=backend,
            prompt_text=prompt_text,
            model_config=model_config,
            generation_config=generation_config,
        )
        answer_text = normalize_generated_answer(safe_text(inference_result.text))
        answer = append_citation(answer_text, cited_doc_ids) if answer_text else fallback_answer(cited_doc_ids)
        finish_reason = inference_result.finish_reason
        latency_ms = inference_result.latency_ms
        answer_type = "generation_answer"
        match_status = "N/A"
        query_type = "GENERAL_RETRIEVAL_QA"
        used_fields = []
        data_cutoff = determine_data_cutoff(detailed_df)
        limitations = []
        if latency_ms > request_timeout_seconds * 1000:
            limitations.append(f"soft_timeout_exceeded:{request_timeout_seconds}s")
            finish_reason = f"{finish_reason}+soft_timeout"

    return {
        "question": question,
        "answer": answer,
        "answer_type": answer_type,
        "match_status": match_status,
        "query_type": query_type,
        "cited_doc_ids": cited_doc_ids,
        "top_doc_id": top_doc_id,
        "retrieval_score": retrieval_score,
        "used_fields": used_fields,
        "data_cutoff": data_cutoff,
        "limitations": limitations,
        "backend": backend,
        "model_id": model_id,
        "runtime": backend if backend == "mock" else str(model_config.get("runtime", backend)),
        "latency_ms": latency_ms,
        "finish_reason": finish_reason,
    }


def print_result(result: dict[str, Any]) -> None:
    print("")
    print("=== Chatbot MVP Result ===")
    print(f"question: {result['question']}")
    print(f"answer: {result['answer']}")
    print(f"answer_type: {result['answer_type']}")
    print(f"match_status: {result['match_status']}")
    print(f"query_type: {result['query_type']}")
    print(f"cited_doc_ids: {result['cited_doc_ids']}")
    print(f"top_doc_id: {result['top_doc_id']}")
    print(f"retrieval_score: {result['retrieval_score']:.3f}")
    print(f"used_fields: {result['used_fields']}")
    print(f"data_cutoff: {result['data_cutoff']}")
    print(f"limitations: {result['limitations']}")
    print(f"backend: {result['backend']}")
    print(f"model_id: {result['model_id']}")
    print(f"runtime: {result['runtime']}")
    print(f"latency_ms: {result['latency_ms']}")
    print(f"finish_reason: {result['finish_reason']}")
    print("==========================")
    print("")


def main() -> None:
    args = parse_args()
    try:
        generation_config = load_generation_config()
    except Exception as exc:
        raise RuntimeError(
            "Failed to load generation config. Check 02_gemma4_generation/config/generation_defaults.json"
        ) from exc

    if args.max_output_tokens is not None:
        generation_config["max_output_tokens"] = int(args.max_output_tokens)
    if args.temperature is not None:
        generation_config["temperature"] = float(args.temperature)
    if args.top_p is not None:
        generation_config["top_p"] = float(args.top_p)
    if args.repeat_penalty is not None:
        generation_config["repeat_penalty"] = float(args.repeat_penalty)

    backend = args.backend or str(generation_config.get("backend", DEFAULT_BACKEND))
    model_id = args.model or pick_default_model_id()

    try:
        model_config = resolve_model_config(model_id)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load model config. Check 02_gemma4_generation/config/models.local.json and model_id."
        ) from exc

    if backend != "mock" and str(model_config.get("runtime")) != backend:
        raise RuntimeError(
            f"Model runtime and requested backend mismatch: runtime={model_config.get('runtime')} backend={backend}."
        )

    if not OUTPUT_SOURCE_INDEX.exists():
        raise FileNotFoundError(
            f"Source index not found: {OUTPUT_SOURCE_INDEX}. Run build_generation_assets.py first."
        )
    if not INPUT_MAIN.exists():
        raise FileNotFoundError(f"Main dataset not found: {INPUT_MAIN}")
    if not INPUT_KNOWLEDGE.exists():
        raise FileNotFoundError(f"Knowledge base not found: {INPUT_KNOWLEDGE}")

    source_df, _ = load_csv(OUTPUT_SOURCE_INDEX)
    detailed_df, _ = load_csv(INPUT_MAIN)
    knowledge_df, _ = load_csv(INPUT_KNOWLEDGE)
    top_k = args.top_k or int(generation_config.get("top_k", 3))
    retrieval_threshold = float(
        args.retrieval_threshold
        if args.retrieval_threshold is not None
        else generation_config.get("retrieval_score_threshold", 1.0)
    )
    fallback_on_low_score = bool(generation_config.get("fallback_on_low_retrieval_score", True))

    print(
        "Demo chatbot started. "
        f"backend={backend} model_id={model_id} top_k={top_k} retrieval_threshold={retrieval_threshold}"
    )
    print("Type 'exit' or 'quit' to stop.")

    if args.question is not None:
        question = safe_text(args.question)
        if not question:
            raise ValueError("--question must not be empty.")
        result = ask_once(
            question=question,
            source_df=source_df,
            detailed_df=detailed_df,
            knowledge_df=knowledge_df,
            backend=backend,
            model_id=model_id,
            model_config=model_config,
            generation_config=generation_config,
            top_k=top_k,
            retrieval_threshold=retrieval_threshold,
            fallback_on_low_score=fallback_on_low_score,
        )
        print_result(result)
        return

    while True:
        try:
            question = input("Q> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nDemo chatbot stopped.")
            return
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Demo chatbot stopped.")
            return
        result = ask_once(
            question=question,
            source_df=source_df,
            detailed_df=detailed_df,
            knowledge_df=knowledge_df,
            backend=backend,
            model_id=model_id,
            model_config=model_config,
            generation_config=generation_config,
            top_k=top_k,
            retrieval_threshold=retrieval_threshold,
            fallback_on_low_score=fallback_on_low_score,
        )
        print_result(result)


if __name__ == "__main__":
    main()
