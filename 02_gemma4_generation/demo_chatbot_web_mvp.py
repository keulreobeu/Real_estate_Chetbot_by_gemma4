from __future__ import annotations

import argparse
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from common import DEFAULT_BACKEND, load_generation_config, pick_default_model_id, safe_text
from api_runtime import ask_from_context, build_runtime_context
from web_demo_support import DEFAULT_RULE_CHECK_QUESTION, build_status_payload, run_generation_readiness_probe


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

            result = ask_from_context(ctx, question)
            self._send_json(200, result)
        except Exception as exc:
            self._send_json(500, {"error": f"request failed: {exc}"})


def main() -> None:
    args = parse_args()
    generation_config = load_generation_config()
    backend = args.backend or str(generation_config.get("backend", DEFAULT_BACKEND))
    model_id = args.model or pick_default_model_id()
    ChatHandler.context = build_runtime_context(
        backend=backend,
        model_id=model_id,
        top_k=args.top_k,
        retrieval_threshold=args.retrieval_threshold,
        port=args.port,
        server_started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(
        f"Web demo started: http://{args.host}:{args.port} "
        f"(backend={backend}, model_id={model_id}, top_k={ChatHandler.context['top_k']}, "
        f"retrieval_threshold={ChatHandler.context['retrieval_threshold']})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Web demo stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
