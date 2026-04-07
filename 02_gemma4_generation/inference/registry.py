from __future__ import annotations

from inference.base import InferenceAdapter
from inference.llama_cpp_adapter import LlamaCppAdapter
from inference.mock_adapter import MockAdapter
from inference.transformers_adapter import TransformersAdapter


def get_adapter(backend: str) -> InferenceAdapter:
    if backend == "mock":
        return MockAdapter()
    if backend == "transformers":
        return TransformersAdapter()
    if backend == "llama_cpp":
        return LlamaCppAdapter()
    raise ValueError(f"지원하지 않는 backend 입니다: {backend}")
