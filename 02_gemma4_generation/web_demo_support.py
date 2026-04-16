from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd

from common import DEFAULT_REQUEST_TIMEOUT_SECONDS, safe_text
from inference.registry import get_adapter


REGION_GROUP_LABELS = {
    "서울": "서울권",
    "서울시": "서울권",
    "서울특별시": "서울권",
    "경기": "경기권",
    "경기도": "경기권",
    "인천": "인천권",
    "인천시": "인천권",
    "인천광역시": "인천권",
}

DEFAULT_RULE_CHECK_QUESTION = "데이터 기준 알려줘"


def build_region_groups(detailed_df: pd.DataFrame) -> list[dict[str, Any]]:
    grouped: dict[str, set[str]] = {"서울권": set(), "경기권": set(), "인천권": set()}
    if "시도" not in detailed_df.columns or "시군구" not in detailed_df.columns:
        return [{"label": label, "districts": []} for label in grouped]

    for _, row in detailed_df[["시도", "시군구"]].dropna().iterrows():
        region_label = REGION_GROUP_LABELS.get(safe_text(row["시도"]))
        district = safe_text(row["시군구"])
        if region_label and district:
            grouped[region_label].add(district)

    return [
        {"label": label, "districts": sorted(districts)}
        for label, districts in grouped.items()
    ]


def build_runtime_probe_summary(
    backend: str,
    model_id: str,
    model_config: dict[str, Any],
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "backend": backend,
        "model_id": model_id,
        "runtime": backend if backend == "mock" else str(model_config.get("runtime", backend)),
        "device_map": safe_text(model_config.get("device_map")),
        "model_source": safe_text(model_config.get("local_dir") or model_config.get("hf_model_id")),
        "last_load_ms": 0,
        "last_generate_ms": 0,
        "probe_error": "",
        "request_timeout_seconds": int(
            generation_config.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS)
        ),
    }
    if backend == "mock":
        return summary

    adapter = get_adapter(backend)
    if not hasattr(adapter, "probe_runtime"):
        summary["probe_error"] = f"backend {backend} does not support probe_runtime"
        return summary

    try:
        probe_result = adapter.probe_runtime(
            model_config=model_config,
            generation_config=generation_config,
            include_generation=False,
        )
        timing = probe_result.get("timing", {})
        runtime_info = probe_result.get("runtime_info", {})
        summary["last_load_ms"] = int(timing.get("load_runtime_ms", 0) or 0)
        summary["device_map"] = safe_text(runtime_info.get("device_map_requested")) or summary["device_map"]
        summary["model_source"] = safe_text(runtime_info.get("model_source")) or summary["model_source"]
    except Exception as exc:
        summary["probe_error"] = safe_text(exc)
    return summary


def build_status_payload(context: dict[str, Any]) -> dict[str, Any]:
    runtime_probe = context["runtime_probe"]
    readiness_probe = context.get("last_generation_probe") or {}
    return {
        "backend": context["backend"],
        "model_id": context["model_id"],
        "runtime": runtime_probe.get("runtime", context["backend"]),
        "device_map": runtime_probe.get("device_map", ""),
        "model_source": runtime_probe.get("model_source", ""),
        "last_load_ms": int(runtime_probe.get("last_load_ms", 0) or 0),
        "request_timeout_seconds": int(runtime_probe.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS)),
        "probe_error": runtime_probe.get("probe_error", ""),
        "rule_check_question": DEFAULT_RULE_CHECK_QUESTION,
        "rule_check_description": "규칙기반 테스트는 실제 답변을 빠르게 반환합니다.",
        "generation_check_description": "Gemma readiness 테스트는 장문 답변 대신 짧은 생성 probe만 수행합니다.",
        "generation_ready": bool(readiness_probe.get("ready", not runtime_probe.get("probe_error"))),
        "generation_probe_supported": context["backend"] != "mock",
        "generation_probe_in_progress": bool(context.get("generation_probe_in_progress", False)),
        "last_generation_probe": readiness_probe,
        "regions": context["region_groups"],
        "server_started_at": context["server_started_at"],
        "pid": context["pid"],
        "port": context["port"],
    }


def run_generation_readiness_probe(context: dict[str, Any]) -> dict[str, Any]:
    backend = context["backend"]
    if backend == "mock":
        return {
            "ready": False,
            "status": "unsupported",
            "backend": backend,
            "model_id": context["model_id"],
            "error": "mock backend does not support Gemma readiness probe",
            "device_map": context["runtime_probe"].get("device_map", ""),
            "model_source": context["runtime_probe"].get("model_source", ""),
            "load_runtime_ms": 0,
            "generate_ms": 0,
            "probe_total_ms": 0,
        }

    if context.get("generation_probe_in_progress"):
        return {
            "ready": False,
            "status": "already_running",
            "backend": backend,
            "model_id": context["model_id"],
            "error": "generation readiness probe is already running",
            "device_map": context["runtime_probe"].get("device_map", ""),
            "model_source": context["runtime_probe"].get("model_source", ""),
            "load_runtime_ms": 0,
            "generate_ms": 0,
            "probe_total_ms": 0,
        }

    generation_lock = context["generation_lock"]
    if not generation_lock.acquire(blocking=False):
        return {
            "ready": False,
            "status": "busy",
            "backend": backend,
            "model_id": context["model_id"],
            "error": "generation runtime is busy with another request",
            "device_map": context["runtime_probe"].get("device_map", ""),
            "model_source": context["runtime_probe"].get("model_source", ""),
            "load_runtime_ms": 0,
            "generate_ms": 0,
            "probe_total_ms": 0,
        }

    context["generation_probe_in_progress"] = True
    try:
        adapter = get_adapter(backend)
        started_at = time.perf_counter()
        probe_result = adapter.probe_runtime(
            model_config=context["model_config"],
            generation_config={**context["generation_config"], "max_output_tokens": 8},
            include_generation=True,
            prompt_text="위 준비 상태를 한 줄로만 답해줘.",
        )
        timing = probe_result.get("timing", {})
        runtime_info = probe_result.get("runtime_info", {})
        result = {
            "ready": True,
            "status": "ready",
            "backend": backend,
            "model_id": context["model_id"],
            "device_map": safe_text(runtime_info.get("device_map_requested"))
            or context["runtime_probe"].get("device_map", ""),
            "model_source": safe_text(runtime_info.get("model_source"))
            or context["runtime_probe"].get("model_source", ""),
            "load_runtime_ms": int(timing.get("load_runtime_ms", 0) or 0),
            "generate_ms": int(timing.get("generate_ms", 0) or 0),
            "probe_total_ms": int(probe_result.get("probe_total_ms", 0) or 0),
            "text_preview": safe_text(probe_result.get("text_preview")),
            "error": "",
            "checked_at": int(time.time()),
        }
        if result["probe_total_ms"] <= 0:
            result["probe_total_ms"] = int((time.perf_counter() - started_at) * 1000)
        context["last_generation_probe"] = result
        return result
    except Exception as exc:
        result = {
            "ready": False,
            "status": "error",
            "backend": backend,
            "model_id": context["model_id"],
            "device_map": context["runtime_probe"].get("device_map", ""),
            "model_source": context["runtime_probe"].get("model_source", ""),
            "load_runtime_ms": 0,
            "generate_ms": 0,
            "probe_total_ms": 0,
            "text_preview": "",
            "error": safe_text(exc),
            "checked_at": int(time.time()),
        }
        context["last_generation_probe"] = result
        return result
    finally:
        context["generation_probe_in_progress"] = False
        generation_lock.release()


def build_runtime_meta(context: dict[str, Any]) -> str:
    runtime_probe = context["runtime_probe"]
    return (
        f"backend={context['backend']} model_id={context['model_id']} top_k={context['top_k']} "
        f"device_map={runtime_probe.get('device_map')} "
        f"last_load_ms={int(runtime_probe.get('last_load_ms', 0) or 0)} "
        f"request_timeout={int(runtime_probe.get('request_timeout_seconds', DEFAULT_REQUEST_TIMEOUT_SECONDS))}s "
        f"pid={os.getpid()} port={context['port']}"
    )
