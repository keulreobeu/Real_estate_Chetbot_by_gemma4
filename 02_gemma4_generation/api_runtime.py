from __future__ import annotations

import os
import re
import threading
import time
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
from web_demo_support import build_region_groups, build_runtime_meta, build_runtime_probe_summary


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


def extract_runtime_debug(raw_response: Any, model_config: dict[str, Any]) -> dict[str, Any]:
    debug = {
        "device_map": safe_text(model_config.get("device_map")),
        "model_source": safe_text(model_config.get("local_dir") or model_config.get("hf_model_id")),
        "model_device": "",
        "last_load_ms": 0,
        "last_generate_ms": 0,
        "local_files_only": bool(model_config.get("local_dir")),
    }
    if not isinstance(raw_response, dict):
        return debug

    timing = raw_response.get("timing")
    runtime_info = raw_response.get("runtime_info")
    if isinstance(timing, dict):
        debug["last_load_ms"] = int(timing.get("load_runtime_ms", 0) or 0)
        debug["last_generate_ms"] = int(timing.get("generate_ms", 0) or 0)
    if isinstance(runtime_info, dict):
        debug["device_map"] = safe_text(runtime_info.get("device_map_requested")) or debug["device_map"]
        debug["model_source"] = safe_text(runtime_info.get("model_source")) or debug["model_source"]
        debug["model_device"] = safe_text(runtime_info.get("model_device"))
        debug["local_files_only"] = bool(runtime_info.get("local_files_only", debug["local_files_only"]))
    return debug


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
    generation_lock: threading.Lock | None = None,
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
        runtime_debug = extract_runtime_debug(None, model_config)
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
        runtime_debug = extract_runtime_debug(None, model_config)
    else:
        if generation_lock is not None and not generation_lock.acquire(blocking=False):
            return {
                "question": question,
                "answer": "현재 모델 진단 또는 다른 생성 요청이 진행 중입니다. 잠시 후 다시 시도해 주세요.",
                "answer_type": "system_status_response",
                "match_status": "N/A",
                "query_type": "GENERAL_RETRIEVAL_QA",
                "cited_doc_ids": cited_doc_ids,
                "top_doc_id": top_doc_id,
                "retrieval_score": round(retrieval_score, 3),
                "used_fields": [],
                "data_cutoff": determine_data_cutoff(detailed_df),
                "limitations": ["generation_runtime_busy"],
                "backend": backend,
                "model_id": model_id,
                "runtime": backend if backend == "mock" else str(model_config.get("runtime", backend)),
                "latency_ms": 0,
                "finish_reason": "runtime_busy",
                "device_map": safe_text(model_config.get("device_map")),
                "model_source": safe_text(model_config.get("local_dir") or model_config.get("hf_model_id")),
                "model_device": "",
                "last_load_ms": 0,
                "last_generate_ms": 0,
                "local_files_only": bool(model_config.get("local_dir")),
            }
        try:
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
            runtime_debug = extract_runtime_debug(inference_result.raw_response, model_config)
            if latency_ms > request_timeout_seconds * 1000:
                limitations.append(f"soft_timeout_exceeded:{request_timeout_seconds}s")
                finish_reason = f"{finish_reason}+soft_timeout"
        finally:
            if generation_lock is not None:
                generation_lock.release()

    return {
        "question": question,
        "answer": answer,
        "answer_type": answer_type,
        "match_status": match_status,
        "query_type": query_type,
        "cited_doc_ids": cited_doc_ids,
        "top_doc_id": top_doc_id,
        "retrieval_score": round(retrieval_score, 3),
        "used_fields": used_fields,
        "data_cutoff": data_cutoff,
        "limitations": limitations,
        "backend": backend,
        "model_id": model_id,
        "runtime": backend if backend == "mock" else str(model_config.get("runtime", backend)),
        "latency_ms": latency_ms,
        "finish_reason": finish_reason,
        "device_map": runtime_debug["device_map"],
        "model_source": runtime_debug["model_source"],
        "model_device": runtime_debug["model_device"],
        "last_load_ms": runtime_debug["last_load_ms"],
        "last_generate_ms": runtime_debug["last_generate_ms"],
        "local_files_only": runtime_debug["local_files_only"],
    }


def ask_from_context(context: dict[str, Any], question: str) -> dict[str, Any]:
    return ask_once(
        question=question,
        source_df=context["source_df"],
        detailed_df=context["detailed_df"],
        knowledge_df=context["knowledge_df"],
        backend=context["backend"],
        model_id=context["model_id"],
        model_config=context["model_config"],
        generation_config=context["generation_config"],
        top_k=context["top_k"],
        retrieval_threshold=context["retrieval_threshold"],
        fallback_on_low_score=context["fallback_on_low_score"],
        generation_lock=context["generation_lock"],
    )


def build_runtime_context(
    *,
    backend: str | None = None,
    model_id: str | None = None,
    top_k: int | None = None,
    retrieval_threshold: float | None = None,
    port: int,
    server_started_at: str | None = None,
) -> dict[str, Any]:
    generation_config = load_generation_config()
    resolved_backend = backend or str(generation_config.get("backend", DEFAULT_BACKEND))
    resolved_model_id = model_id or pick_default_model_id()
    model_config = resolve_model_config(resolved_model_id)

    if resolved_backend != "mock" and str(model_config.get("runtime")) != resolved_backend:
        raise RuntimeError(
            f"Model runtime and requested backend mismatch: runtime={model_config.get('runtime')} backend={resolved_backend}."
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
    resolved_top_k = top_k or int(generation_config.get("top_k", 3))
    resolved_retrieval_threshold = float(
        retrieval_threshold
        if retrieval_threshold is not None
        else generation_config.get("retrieval_score_threshold", 1.0)
    )
    runtime_probe = build_runtime_probe_summary(
        backend=resolved_backend,
        model_id=resolved_model_id,
        model_config=model_config,
        generation_config=generation_config,
    )
    context = {
        "source_df": source_df,
        "detailed_df": detailed_df,
        "knowledge_df": knowledge_df,
        "backend": resolved_backend,
        "model_id": resolved_model_id,
        "model_config": model_config,
        "generation_config": generation_config,
        "top_k": resolved_top_k,
        "retrieval_threshold": resolved_retrieval_threshold,
        "fallback_on_low_score": bool(generation_config.get("fallback_on_low_retrieval_score", True)),
        "runtime_probe": runtime_probe,
        "runtime_meta": "",
        "region_groups": build_region_groups(detailed_df),
        "last_generation_probe": {
            "ready": False if resolved_backend == "mock" else not bool(runtime_probe.get("probe_error")),
            "status": "not_run",
            "backend": resolved_backend,
            "model_id": resolved_model_id,
            "device_map": runtime_probe.get("device_map", ""),
            "model_source": runtime_probe.get("model_source", ""),
            "load_runtime_ms": int(runtime_probe.get("last_load_ms", 0) or 0),
            "generate_ms": 0,
            "probe_total_ms": 0,
            "text_preview": "",
            "error": runtime_probe.get("probe_error", ""),
            "checked_at": 0,
        },
        "generation_probe_in_progress": False,
        "generation_lock": threading.Lock(),
        "server_started_at": server_started_at or time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
        "port": port,
    }
    context["runtime_meta"] = build_runtime_meta(context)
    return context
