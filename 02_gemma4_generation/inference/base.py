from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class InferenceResult:
    text: str
    backend: str
    model_id: str
    finish_reason: str
    token_usage: dict[str, Any] | None
    latency_ms: int
    raw_response: dict[str, Any] | None


class InferenceAdapter(ABC):
    @abstractmethod
    def generate(
        self,
        prompt_text: str,
        model_config: dict[str, Any],
        generation_config: dict[str, Any],
    ) -> InferenceResult:
        raise NotImplementedError
