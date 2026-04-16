from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
from web_demo_support import (
    DEFAULT_RULE_CHECK_QUESTION,
    build_region_groups,
    build_runtime_meta,
    build_runtime_probe_summary,
    build_status_payload,
    run_generation_readiness_probe,
)


INDEX_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Real Estate Chatbot MVP</title>
  <style>
    :root { --bg:#f4f6f8; --card:#ffffff; --ink:#1f2937; --muted:#6b7280; --line:#d1d5db; --accent:#0f766e; --warn:#9a3412; }
    body { margin:0; background:linear-gradient(180deg,#eef2f7,#f9fafb); color:var(--ink); font-family:Segoe UI, Malgun Gothic, sans-serif; }
    .wrap { max-width:1120px; margin:24px auto; padding:0 16px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:16px; margin-bottom:16px; }
    .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; box-shadow:0 6px 20px rgba(15,23,42,0.06); }
    h1 { margin:0 0 8px; font-size:22px; }
    h2 { margin:0 0 10px; font-size:16px; }
    .meta { color:var(--muted); margin:0 0 14px; font-size:13px; }
    .stack { display:flex; flex-direction:column; gap:8px; }
    .chips { display:flex; flex-wrap:wrap; gap:6px; }
    .chip { background:#ecfeff; color:#155e75; border:1px solid #bae6fd; border-radius:999px; padding:4px 8px; font-size:12px; }
    .status-list { display:grid; grid-template-columns:auto 1fr; gap:6px 10px; font-size:13px; }
    .status-list strong { color:var(--muted); font-weight:600; }
    .hint { font-size:13px; color:var(--muted); margin:0; }
    textarea { width:100%; min-height:90px; border:1px solid var(--line); border-radius:10px; padding:10px; resize:vertical; font-size:14px; }
    button { margin-top:10px; background:var(--accent); color:white; border:0; border-radius:10px; padding:10px 14px; font-weight:600; cursor:pointer; }
    button:disabled { opacity:.5; cursor:not-allowed; }
    .row { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
    .button-secondary { background:#0f172a; }
    .out { margin-top:16px; border-top:1px solid var(--line); padding-top:14px; }
    .label { color:var(--muted); font-size:12px; margin-top:8px; }
    .box { background:#f8fafc; border:1px solid #e5e7eb; border-radius:10px; padding:10px; white-space:pre-wrap; }
    .notice { color:var(--warn); font-size:12px; margin:8px 0 0; }
    .region-block + .region-block { margin-top:12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="grid">
      <div class="card">
        <h2>데이터 포함 지역</h2>
        <p class="meta">메인 RAG CSV 기준 포함된 시군구 범위입니다.</p>
        <div id="regions" class="stack"></div>
      </div>
      <div class="card">
        <h2>빠른 확인</h2>
        <p class="hint" id="ruleCheckDescription"></p>
        <div class="box" id="ruleQuestionBox"></div>
        <button id="ruleCheckBtn" class="button-secondary">규칙기반 답변 확인</button>
        <p class="hint" id="generationCheckDescription" style="margin-top:12px;"></p>
        <div class="box">Gemma readiness 확인은 실제 장문 답변 대신 짧은 probe만 실행합니다.</div>
        <button id="generationCheckBtn">Gemma 생성 가능 상태 확인</button>
        <p class="notice" id="quickCheckNotice"></p>
      </div>
      <div class="card">
        <h2>현재 상태</h2>
        <div class="status-list">
          <strong>backend</strong><span id="statusBackend"></span>
          <strong>model_id</strong><span id="statusModel"></span>
          <strong>runtime</strong><span id="statusRuntime"></span>
          <strong>device_map</strong><span id="statusDeviceMap"></span>
          <strong>last_load_ms</strong><span id="statusLoadMs"></span>
          <strong>request_timeout</strong><span id="statusTimeout"></span>
          <strong>generation_ready</strong><span id="statusReady"></span>
          <strong>server</strong><span id="statusServer"></span>
        </div>
        <p class="notice" id="statusProbeError"></p>
      </div>
    </div>
    <div class="card">
      <h1>Real Estate Chatbot MVP</h1>
      <p class="meta">로컬 테스트용 UI이며, 규칙기반 응답과 Gemma generation readiness를 분리해서 확인할 수 있습니다.</p>
      <p class="meta"><strong>runtime:</strong> {{RUNTIME_META}}</p>
      <textarea id="q" placeholder="질문을 입력하세요. 예: 송파구 아파트들의 특징을 요약해줘"></textarea>
      <div class="row">
        <button id="askBtn">질문하기</button>
      </div>
      <div class="out" id="out" style="display:none;">
        <div class="label">answer</div><div class="box" id="answer"></div>
        <div class="label">answer_type / match_status / query_type</div><div class="box" id="contract"></div>
        <div class="label">cited_doc_ids</div><div class="box" id="cited"></div>
        <div class="label">top_doc_id / retrieval_score</div><div class="box" id="meta"></div>
        <div class="label">data_cutoff / limitations / used_fields</div><div class="box" id="safety"></div>
        <div class="label">backend / model / runtime / latency / finish_reason</div><div class="box" id="runtime"></div>
        <div class="label">device_map / model_source / last_load / last_generate</div><div class="box" id="runtimeDebug"></div>
      </div>
    </div>
  </div>
  <script>
    const btn = document.getElementById("askBtn");
    const ruleBtn = document.getElementById("ruleCheckBtn");
    const generationBtn = document.getElementById("generationCheckBtn");
    const q = document.getElementById("q");
    const out = document.getElementById("out");
    const quickCheckNotice = document.getElementById("quickCheckNotice");

    function renderRegions(regions) {
      const el = document.getElementById("regions");
      el.innerHTML = "";
      (regions || []).forEach(region => {
        const wrap = document.createElement("div");
        wrap.className = "region-block";
        const title = document.createElement("div");
        title.className = "label";
        title.textContent = region.label;
        wrap.appendChild(title);
        const chips = document.createElement("div");
        chips.className = "chips";
        (region.districts || []).forEach(name => {
          const chip = document.createElement("span");
          chip.className = "chip";
          chip.textContent = name;
          chips.appendChild(chip);
        });
        wrap.appendChild(chips);
        el.appendChild(wrap);
      });
    }

    function renderStatus(status) {
      document.getElementById("statusBackend").textContent = status.backend || "";
      document.getElementById("statusModel").textContent = status.model_id || "";
      document.getElementById("statusRuntime").textContent = status.runtime || "";
      document.getElementById("statusDeviceMap").textContent = status.device_map || "";
      document.getElementById("statusLoadMs").textContent = `${status.last_load_ms || 0}ms`;
      document.getElementById("statusTimeout").textContent = `${status.request_timeout_seconds || 0}s`;
      document.getElementById("statusReady").textContent = status.generation_ready ? "ready" : "not ready";
      document.getElementById("statusServer").textContent = `pid=${status.pid} port=${status.port} started_at=${status.server_started_at}`;
      document.getElementById("statusProbeError").textContent = status.probe_error || "";
      document.getElementById("ruleQuestionBox").textContent = status.rule_check_question || "";
      document.getElementById("ruleCheckDescription").textContent = status.rule_check_description || "";
      document.getElementById("generationCheckDescription").textContent = status.generation_check_description || "";
      generationBtn.disabled = !status.generation_probe_supported || status.generation_probe_in_progress;
      renderRegions(status.regions || []);
    }

    async function loadStatus() {
      const res = await fetch("/api/status");
      const data = await res.json();
      renderStatus(data);
      return data;
    }

    function renderAskResult(data) {
      out.style.display = "block";
      document.getElementById("answer").textContent = data.answer;
      document.getElementById("contract").textContent = `${data.answer_type} / ${data.match_status} / ${data.query_type}`;
      document.getElementById("cited").textContent = JSON.stringify(data.cited_doc_ids);
      document.getElementById("meta").textContent = `${data.top_doc_id} / ${data.retrieval_score}`;
      document.getElementById("safety").textContent = `${data.data_cutoff} / ${JSON.stringify(data.limitations || [])} / ${JSON.stringify(data.used_fields || [])}`;
      document.getElementById("runtime").textContent = `${data.backend} / ${data.model_id} / ${data.runtime} / ${data.latency_ms}ms / ${data.finish_reason}`;
      document.getElementById("runtimeDebug").textContent = `${data.device_map || ""} / ${data.model_source || ""} / ${data.last_load_ms || 0}ms / ${data.last_generate_ms || 0}ms`;
    }

    btn.onclick = async () => {
      const question = q.value.trim();
      if (!question) return;
      btn.disabled = true;
      try {
        const res = await fetch("/api/ask", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({question})
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "요청 실패");
        renderAskResult(data);
      } catch (e) {
        alert(String(e.message || e));
      } finally {
        btn.disabled = false;
        await loadStatus();
      }
    };

    ruleBtn.onclick = async () => {
      quickCheckNotice.textContent = "";
      ruleBtn.disabled = true;
      try {
        const res = await fetch("/api/check-rule", { method: "POST" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "규칙기반 확인 실패");
        renderAskResult(data);
        quickCheckNotice.textContent = "규칙기반 전용 질문이 즉시 처리되었습니다.";
      } catch (e) {
        quickCheckNotice.textContent = String(e.message || e);
      } finally {
        ruleBtn.disabled = false;
        await loadStatus();
      }
    };

    generationBtn.onclick = async () => {
      quickCheckNotice.textContent = "";
      generationBtn.disabled = true;
      try {
        const res = await fetch("/api/check-generation-ready", { method: "POST" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Gemma readiness 확인 실패");
        quickCheckNotice.textContent = data.ready
          ? `Gemma readiness OK: load=${data.load_runtime_ms}ms generate=${data.generate_ms}ms total=${data.probe_total_ms}ms`
          : `Gemma readiness NOT_READY: ${data.error || data.status}`;
      } catch (e) {
        quickCheckNotice.textContent = String(e.message || e);
      } finally {
        await loadStatus();
      }
    };

    loadStatus().catch(err => {
      quickCheckNotice.textContent = String(err.message || err);
    });
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local web chatbot MVP for 02_gemma4_generation.")
    parser.add_argument("--backend", choices=["mock", "transformers", "llama_cpp"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k", type=int, default=None, dest="top_k")
    parser.add_argument("--retrieval-threshold", type=float, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    return parser.parse_args()


def run_backend(
    backend: str,
    prompt_text: str,
    model_config: dict[str, Any],
    generation_config: dict[str, Any],
) -> Any:
    adapter = get_adapter(backend)
    return adapter.generate(prompt_text=prompt_text, model_config=model_config, generation_config=generation_config)


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


class ChatHandler(BaseHTTPRequestHandler):
    context: dict[str, Any] = {}

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/status":
            self._send_json(200, build_status_payload(self.context))
            return
        if path != "/":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        runtime_meta = str(self.context.get("runtime_meta", "unknown"))
        page_html = INDEX_HTML.replace("{{RUNTIME_META}}", runtime_meta)
        data = page_html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in {"/api/ask", "/api/check-rule", "/api/check-generation-ready"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            ctx = self.context
            if path == "/api/check-generation-ready":
                result = run_generation_readiness_probe(ctx)
                self._send_json(200, result)
                return

            if path == "/api/check-rule":
                question = DEFAULT_RULE_CHECK_QUESTION
            else:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8")) if raw else {}
                question = safe_text(payload.get("question"))
                if not question:
                    self._send_json(400, {"error": "question must not be empty"})
                    return

            result = ask_once(
                question=question,
                source_df=ctx["source_df"],
                detailed_df=ctx["detailed_df"],
                knowledge_df=ctx["knowledge_df"],
                backend=ctx["backend"],
                model_id=ctx["model_id"],
                model_config=ctx["model_config"],
                generation_config=ctx["generation_config"],
                top_k=ctx["top_k"],
                retrieval_threshold=ctx["retrieval_threshold"],
                fallback_on_low_score=ctx["fallback_on_low_score"],
                generation_lock=ctx["generation_lock"],
            )
            self._send_json(200, result)
        except Exception as exc:
            self._send_json(500, {"error": f"request failed: {exc}"})


def main() -> None:
    args = parse_args()
    generation_config = load_generation_config()
    backend = args.backend or str(generation_config.get("backend", DEFAULT_BACKEND))
    model_id = args.model or pick_default_model_id()
    model_config = resolve_model_config(model_id)

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

    runtime_probe = build_runtime_probe_summary(
        backend=backend,
        model_id=model_id,
        model_config=model_config,
        generation_config=generation_config,
    )
    generation_lock = threading.Lock()
    region_groups = build_region_groups(detailed_df)
    server_started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    ChatHandler.context = {
        "source_df": source_df,
        "detailed_df": detailed_df,
        "knowledge_df": knowledge_df,
        "backend": backend,
        "model_id": model_id,
        "model_config": model_config,
        "generation_config": generation_config,
        "top_k": top_k,
        "retrieval_threshold": retrieval_threshold,
        "fallback_on_low_score": bool(generation_config.get("fallback_on_low_retrieval_score", True)),
        "runtime_probe": runtime_probe,
        "runtime_meta": "",
        "region_groups": region_groups,
        "last_generation_probe": {
            "ready": False if backend == "mock" else not bool(runtime_probe.get("probe_error")),
            "status": "not_run",
            "backend": backend,
            "model_id": model_id,
            "device_map": runtime_probe.get("device_map", ""),
            "model_source": runtime_probe.get("model_source", ""),
            "load_runtime_ms": int(runtime_probe.get("last_load_ms", 0) or 0),
            "generate_ms": 0,
            "probe_total_ms": 0,
            "text_preview": "",
            "error": runtime_probe.get("probe_error", ""),
        },
        "generation_probe_in_progress": False,
        "generation_lock": generation_lock,
        "server_started_at": server_started_at,
        "pid": os.getpid(),
        "port": args.port,
    }
    ChatHandler.context["runtime_meta"] = build_runtime_meta(ChatHandler.context)

    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(
        f"Web demo started: http://{args.host}:{args.port} "
        f"(backend={backend}, model_id={model_id}, top_k={top_k}, retrieval_threshold={retrieval_threshold})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Web demo stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
