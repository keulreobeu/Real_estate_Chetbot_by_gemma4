from __future__ import annotations

import argparse
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api_logging import write_api_log
from api_models import AskRequest, AskResponse, ErrorResponse, GenerationReadyResponse, StatusResponse
from api_runtime import ask_from_context, build_runtime_context
from web_demo_support import DEFAULT_RULE_CHECK_QUESTION, build_status_payload, run_generation_readiness_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastAPI service for 02_gemma4_generation.")
    parser.add_argument("--backend", choices=["mock", "transformers", "llama_cpp"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k", type=int, default=None, dest="top_k")
    parser.add_argument("--retrieval-threshold", type=float, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    return parser.parse_args()


def create_app(context: dict[str, Any]) -> FastAPI:
    app = FastAPI(
        title="Real Estate Chatbot FastAPI",
        version="0.1.0",
        description="Thin FastAPI adapter around the 02_gemma4_generation grounded QA runtime.",
    )
    app.state.context = context

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        question = ""
        try:
            payload = await request.json()
            if isinstance(payload, dict):
                question = str(payload.get("question", "") or "")
        except Exception:
            question = ""
        write_api_log(path=request.url.path, status_code=500, question=question, error=str(exc))
        return JSONResponse(status_code=500, content={"error": f"request failed: {exc}"})

    @app.get("/api/status", response_model=StatusResponse)
    async def get_status() -> dict[str, Any]:
        result = build_status_payload(app.state.context)
        write_api_log(path="/api/status", status_code=200, result={"backend": result["backend"], "model_id": result["model_id"]})
        return result

    @app.post("/api/ask", response_model=AskResponse, responses={500: {"model": ErrorResponse}})
    async def post_ask(payload: AskRequest) -> dict[str, Any]:
        result = ask_from_context(app.state.context, payload.question)
        write_api_log(path="/api/ask", status_code=200, question=payload.question, result=result)
        return result

    @app.post("/api/check-rule", response_model=AskResponse, responses={500: {"model": ErrorResponse}})
    async def post_check_rule() -> dict[str, Any]:
        result = ask_from_context(app.state.context, DEFAULT_RULE_CHECK_QUESTION)
        write_api_log(path="/api/check-rule", status_code=200, question=DEFAULT_RULE_CHECK_QUESTION, result=result)
        return result

    @app.post(
        "/api/check-generation-ready",
        response_model=GenerationReadyResponse,
        responses={500: {"model": ErrorResponse}},
    )
    async def post_check_generation_ready() -> dict[str, Any]:
        result = run_generation_readiness_probe(app.state.context)
        write_api_log(
            path="/api/check-generation-ready",
            status_code=200,
            result={
                "backend": result.get("backend", ""),
                "model_id": result.get("model_id", ""),
                "limitations": [result.get("status", "")] if result.get("status") else [],
                "device_map": result.get("device_map", ""),
                "last_load_ms": result.get("load_runtime_ms", 0),
                "last_generate_ms": result.get("generate_ms", 0),
                "finish_reason": result.get("status", ""),
            },
        )
        return result

    return app


def main() -> None:
    args = parse_args()
    context = build_runtime_context(
        backend=args.backend,
        model_id=args.model,
        top_k=args.top_k,
        retrieval_threshold=args.retrieval_threshold,
        port=args.port,
    )
    app = create_app(context)
    print(
        f"FastAPI service started: http://{args.host}:{args.port} "
        f"(backend={context['backend']}, model_id={context['model_id']}, top_k={context['top_k']}, "
        f"retrieval_threshold={context['retrieval_threshold']})"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
