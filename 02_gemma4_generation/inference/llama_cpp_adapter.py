from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from inference.base import InferenceAdapter, InferenceResult


class LlamaCppAdapter(InferenceAdapter):
    _instances: dict[str, Any] = {}

    def _resolve_thread_count(self, model_config: dict[str, Any]) -> int:
        logical_cores = max(int(os.cpu_count() or 1), 1)
        configured = int(model_config.get("n_threads", logical_cores))
        if configured <= 0:
            configured = logical_cores
        return min(configured, logical_cores)

    def generate(
        self,
        prompt_text: str,
        model_config: dict[str, Any],
        generation_config: dict[str, Any],
    ) -> InferenceResult:
        model_path = Path(str(model_config["model_path"]))
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama_cpp backend를 사용하려면 `llama-cpp-python` 패키지가 필요합니다."
            ) from exc

        cache_key = f"{model_path}|{model_config.get('n_ctx')}|{model_config.get('n_gpu_layers')}"
        if cache_key not in self._instances:
            self._instances[cache_key] = Llama(
                model_path=str(model_path),
                n_ctx=int(model_config.get("n_ctx", 4096)),
                n_gpu_layers=int(model_config.get("n_gpu_layers", -1)),
                n_threads=self._resolve_thread_count(model_config),
                chat_format=str(model_config.get("chat_format", "gemma")),
                verbose=False,
            )

        client = self._instances[cache_key]
        started_at = time.perf_counter()
        response = client.create_completion(
            prompt=prompt_text,
            max_tokens=int(generation_config.get("max_output_tokens", 256)),
            temperature=float(generation_config.get("temperature", 0.1)),
            top_p=float(generation_config.get("top_p", 0.9)),
            repeat_penalty=float(generation_config.get("repeat_penalty", 1.1)),
            stop=generation_config.get("stop_sequences", []),
        )
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        choice = response["choices"][0]
        usage = response.get("usage")

        return InferenceResult(
            text=str(choice.get("text", "")).strip(),
            backend="llama_cpp",
            model_id=str(model_config["model_id"]),
            finish_reason=str(choice.get("finish_reason", "unknown")),
            token_usage=usage if isinstance(usage, dict) else None,
            latency_ms=latency_ms,
            raw_response=response,
        )
