from __future__ import annotations

import os
import time
from typing import Any

from inference.base import InferenceAdapter, InferenceResult


class TransformersAdapter(InferenceAdapter):
    _instances: dict[str, tuple[Any, Any, Any]] = {}

    def _load_runtime_with_timing(
        self,
        model_config: dict[str, Any],
    ) -> tuple[Any, Any, Any, dict[str, int]]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers backend를 사용하려면 `transformers`, `torch`, `accelerate` 패키지가 필요합니다."
            ) from exc

        model_source = str(model_config.get("local_dir") or model_config["hf_model_id"])
        cache_key = "|".join(
            [
                model_source,
                str(model_config.get("processor_id") or model_config["hf_model_id"]),
                str(model_config.get("torch_dtype", "bfloat16")),
                str(model_config.get("device_map", "auto")),
                str(model_config.get("attn_implementation", "")),
            ]
        )
        if cache_key in self._instances:
            return torch, *self._instances[cache_key], {
                "load_runtime_ms": 0,
                "processor_load_ms": 0,
                "model_load_ms": 0,
            }

        local_files_only = bool(model_config.get("local_dir"))
        processor_source = str(model_config.get("local_dir") or model_config.get("processor_id") or model_config["hf_model_id"])
        processor_started_at = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            processor_source,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        processor_load_ms = int((time.perf_counter() - processor_started_at) * 1000)

        model_kwargs: dict[str, Any] = {
            "device_map": model_config.get("device_map", "auto"),
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        torch_dtype = self._resolve_torch_dtype(torch, str(model_config.get("torch_dtype", "bfloat16")))
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        attn_implementation = str(model_config.get("attn_implementation", "")).strip()
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        model_started_at = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
        model_load_ms = int((time.perf_counter() - model_started_at) * 1000)
        model.eval()
        self._configure_cpu_threads(torch, model, model_config)

        self._instances[cache_key] = (processor, model)
        return torch, processor, model, {
            "load_runtime_ms": processor_load_ms + model_load_ms,
            "processor_load_ms": processor_load_ms,
            "model_load_ms": model_load_ms,
        }

    def probe_runtime(
        self,
        model_config: dict[str, Any],
        generation_config: dict[str, Any] | None = None,
        *,
        include_generation: bool = False,
        prompt_text: str = "점검용 짧은 응답을 생성해 주세요.",
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        torch, processor, model, load_timing = self._load_runtime_with_timing(model_config)

        timing: dict[str, int] = {
            "load_runtime_ms": load_timing["load_runtime_ms"],
            "processor_load_ms": load_timing["processor_load_ms"],
            "model_load_ms": load_timing["model_load_ms"],
            "prompt_render_ms": 0,
            "input_prepare_ms": 0,
            "to_device_ms": 0,
            "generate_ms": 0,
            "decode_ms": 0,
        }
        runtime_info = self._build_runtime_info(torch, model, model_config)
        text_preview = ""

        if include_generation:
            probe_generation_config = dict(generation_config or {})
            messages = [{"role": "user", "content": prompt_text}]

            render_started_at = time.perf_counter()
            rendered_prompt = self._render_prompt(processor, messages, prompt_text)
            timing["prompt_render_ms"] = int((time.perf_counter() - render_started_at) * 1000)

            max_input_tokens = int(model_config.get("max_input_tokens", 4096))
            input_started_at = time.perf_counter()
            inputs = processor(
                text=rendered_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            )
            timing["input_prepare_ms"] = int((time.perf_counter() - input_started_at) * 1000)

            move_started_at = time.perf_counter()
            inputs = self._move_inputs_to_device(inputs, model)
            timing["to_device_ms"] = int((time.perf_counter() - move_started_at) * 1000)

            input_len = int(inputs["input_ids"].shape[-1])
            max_new_tokens = min(int(probe_generation_config.get("max_output_tokens", 16) or 16), 16)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": float(probe_generation_config.get("repeat_penalty", 1.05)),
                "do_sample": False,
                "use_cache": True,
            }

            generate_started_at = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            timing["generate_ms"] = int((time.perf_counter() - generate_started_at) * 1000)

            decode_started_at = time.perf_counter()
            generated_ids = outputs[0][input_len:]
            response = processor.decode(generated_ids, skip_special_tokens=False)
            text_preview = self._parse_response(processor, response).strip()[:120]
            timing["decode_ms"] = int((time.perf_counter() - decode_started_at) * 1000)

        return {
            "include_generation": include_generation,
            "timing": timing,
            "runtime_info": runtime_info,
            "text_preview": text_preview,
            "probe_total_ms": int((time.perf_counter() - started_at) * 1000),
        }

    def generate(
        self,
        prompt_text: str,
        model_config: dict[str, Any],
        generation_config: dict[str, Any],
    ) -> InferenceResult:
        torch, processor, model, load_timing = self._load_runtime_with_timing(model_config)

        messages = [{"role": "user", "content": prompt_text}]
        render_started_at = time.perf_counter()
        rendered_prompt = self._render_prompt(processor, messages, prompt_text)
        prompt_render_ms = int((time.perf_counter() - render_started_at) * 1000)

        max_input_tokens = int(model_config.get("max_input_tokens", 4096))
        input_started_at = time.perf_counter()
        inputs = processor(
            text=rendered_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        input_prepare_ms = int((time.perf_counter() - input_started_at) * 1000)

        move_started_at = time.perf_counter()
        inputs = self._move_inputs_to_device(inputs, model)
        to_device_ms = int((time.perf_counter() - move_started_at) * 1000)
        input_len = int(inputs["input_ids"].shape[-1])
        max_new_tokens = int(generation_config.get("max_output_tokens", model_config.get("max_output_tokens", 256)))
        temperature = float(generation_config.get("temperature", 0.1))
        do_sample = temperature > 0.0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": float(generation_config.get("repeat_penalty", 1.1)),
            "do_sample": do_sample,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = float(generation_config.get("top_p", 0.9))

        generate_started_at = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                **generation_kwargs,
            )
        generate_ms = int((time.perf_counter() - generate_started_at) * 1000)

        generated_ids = outputs[0][input_len:]
        decode_started_at = time.perf_counter()
        response = processor.decode(generated_ids, skip_special_tokens=False)
        text = self._parse_response(processor, response)
        text = self._apply_stop_sequences(text, generation_config.get("stop_sequences", []))
        decode_ms = int((time.perf_counter() - decode_started_at) * 1000)

        completion_tokens = int(generated_ids.shape[-1]) if hasattr(generated_ids, "shape") else 0
        token_usage = {
            "prompt_tokens": input_len,
            "completion_tokens": completion_tokens,
            "total_tokens": input_len + completion_tokens,
        }

        return InferenceResult(
            text=text.strip(),
            backend="transformers",
            model_id=str(model_config["model_id"]),
            finish_reason="stop",
            token_usage=token_usage,
            latency_ms=generate_ms,
            raw_response={
                "response": response,
                "rendered_prompt_preview": rendered_prompt[:400],
                "timing": {
                    "load_runtime_ms": load_timing["load_runtime_ms"],
                    "processor_load_ms": load_timing["processor_load_ms"],
                    "model_load_ms": load_timing["model_load_ms"],
                    "prompt_render_ms": prompt_render_ms,
                    "input_prepare_ms": input_prepare_ms,
                    "to_device_ms": to_device_ms,
                    "generate_ms": generate_ms,
                    "decode_ms": decode_ms,
                },
                "runtime_info": self._build_runtime_info(torch, model, model_config),
            },
        )

    def _get_or_load_runtime(self, model_config: dict[str, Any]) -> tuple[Any, Any, Any]:
        torch, processor, model, _ = self._load_runtime_with_timing(model_config)
        return torch, processor, model

    def _render_prompt(self, processor: Any, messages: list[dict[str, Any]], fallback_prompt: str) -> str:
        apply_chat_template = getattr(processor, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                return apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return fallback_prompt

    def _move_inputs_to_device(self, inputs: Any, model: Any) -> Any:
        target_device = getattr(model, "device", None)
        if target_device is None:
            return inputs
        return {key: value.to(target_device) if hasattr(value, "to") else value for key, value in inputs.items()}

    def _build_runtime_info(self, torch: Any, model: Any, model_config: dict[str, Any]) -> dict[str, Any]:
        runtime_info: dict[str, Any] = {
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "model_device": str(getattr(model, "device", "unknown")),
            "hf_device_map": getattr(model, "hf_device_map", None),
            "device_map_requested": model_config.get("device_map"),
            "model_source": str(model_config.get("local_dir") or model_config.get("hf_model_id", "")),
            "processor_source": str(model_config.get("local_dir") or model_config.get("processor_id") or model_config.get("hf_model_id", "")),
            "local_files_only": bool(model_config.get("local_dir")),
        }
        if torch.cuda.is_available():
            try:
                runtime_info["cuda_memory_allocated"] = int(torch.cuda.memory_allocated())
                runtime_info["cuda_memory_reserved"] = int(torch.cuda.memory_reserved())
            except Exception:
                pass
        return runtime_info

    def _parse_response(self, processor: Any, response: str) -> str:
        parse_response = getattr(processor, "parse_response", None)
        if callable(parse_response):
            try:
                parsed = parse_response(response)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                text = parsed.get("text")
                if isinstance(text, str) and text.strip():
                    return text
        return response

    def _apply_stop_sequences(self, text: str, stop_sequences: Any) -> str:
        if not isinstance(stop_sequences, list):
            return text
        trimmed = text
        for stop_sequence in stop_sequences:
            if not isinstance(stop_sequence, str) or not stop_sequence:
                continue
            if stop_sequence in trimmed:
                trimmed = trimmed.split(stop_sequence, maxsplit=1)[0]
        return trimmed

    def _resolve_torch_dtype(self, torch: Any, raw_dtype: str) -> Any:
        dtype_name = raw_dtype.strip().lower()
        if not dtype_name or dtype_name == "auto":
            return None
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported torch_dtype: {raw_dtype}")
        return mapping[dtype_name]

    def _configure_cpu_threads(self, torch: Any, model: Any, model_config: dict[str, Any]) -> None:
        if not self._is_cpu_execution(model):
            return

        logical_cores = max(int(os.cpu_count() or 1), 1)
        intra_threads = int(model_config.get("cpu_threads", logical_cores))
        if intra_threads <= 0:
            intra_threads = logical_cores
        intra_threads = min(intra_threads, logical_cores)

        interop_threads = int(model_config.get("cpu_interop_threads", 1))
        if interop_threads <= 0:
            interop_threads = 1

        torch.set_num_threads(intra_threads)
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            pass

    def _is_cpu_execution(self, model: Any) -> bool:
        device = getattr(model, "device", None)
        if device is not None:
            device_text = str(getattr(device, "type", device)).lower()
            if "cuda" in device_text or "mps" in device_text or "xpu" in device_text:
                return False
            if "cpu" in device_text:
                return True

        hf_device_map = getattr(model, "hf_device_map", None)
        if isinstance(hf_device_map, dict) and hf_device_map:
            values = [str(value).lower() for value in hf_device_map.values()]
            if any(("cuda" in value) or ("mps" in value) or ("xpu" in value) for value in values):
                return False
            return any("cpu" in value for value in values)

        return True
