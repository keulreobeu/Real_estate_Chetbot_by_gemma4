from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from common import PROJECT_ROOT, ensure_parent


_LOG_LOCK = threading.Lock()


def build_log_path(now: datetime | None = None) -> Path:
    timestamp = now or datetime.now()
    return PROJECT_ROOT / "logs" / "api_requests" / f"fastapi_{timestamp.strftime('%Y%m%d')}.jsonl"


def write_api_log(
    *,
    path: str,
    status_code: int,
    question: str = "",
    result: dict[str, Any] | None = None,
    error: str = "",
) -> Path:
    log_path = build_log_path()
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "path": path,
        "question": question,
        "query_type": (result or {}).get("query_type", ""),
        "answer_type": (result or {}).get("answer_type", ""),
        "match_status": (result or {}).get("match_status", ""),
        "top_doc_id": (result or {}).get("top_doc_id", ""),
        "retrieval_score": (result or {}).get("retrieval_score", 0),
        "latency_ms": (result or {}).get("latency_ms", 0),
        "finish_reason": (result or {}).get("finish_reason", ""),
        "backend": (result or {}).get("backend", ""),
        "model_id": (result or {}).get("model_id", ""),
        "limitations": (result or {}).get("limitations", []),
        "status_code": status_code,
        "device_map": (result or {}).get("device_map", ""),
        "last_load_ms": (result or {}).get("last_load_ms", 0),
        "last_generate_ms": (result or {}).get("last_generate_ms", 0),
        "error": error,
    }
    ensure_parent(log_path)
    encoded = json.dumps(payload, ensure_ascii=False)
    with _LOG_LOCK:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    return log_path
