from __future__ import annotations

import re
import time

from inference.base import InferenceAdapter, InferenceResult


class MockAdapter(InferenceAdapter):
    def generate(
        self,
        prompt_text: str,
        model_config: dict[str, object],
        generation_config: dict[str, object],
    ) -> InferenceResult:
        started_at = time.perf_counter()
        question = self._extract_question(prompt_text)
        documents = self._extract_documents(prompt_text)
        text = self._build_answer(question, documents)
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        return InferenceResult(
            text=text,
            backend="mock",
            model_id=str(model_config.get("model_id", "mock")),
            finish_reason="stop",
            token_usage=None,
            latency_ms=latency_ms,
            raw_response={"documents": documents[:1]},
        )

    def _extract_question(self, prompt_text: str) -> str:
        match = re.search(r"질문:\s*(.*?)\s*근거 문서:", prompt_text, flags=re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_documents(self, prompt_text: str) -> list[dict[str, str]]:
        if "근거 문서:" not in prompt_text:
            return []
        context = prompt_text.split("근거 문서:", maxsplit=1)[1]
        if "출력 형식:" in context:
            context = context.split("출력 형식:", maxsplit=1)[0]
        raw_docs = []
        for chunk in re.split(r"\n\s*\n(?=문서ID:)", context.strip()):
            chunk = chunk.strip()
            if chunk:
                raw_docs.append(chunk)
        documents: list[dict[str, str]] = []
        for raw_doc in raw_docs:
            parsed: dict[str, str] = {}
            for line in raw_doc.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", maxsplit=1)
                parsed[key.strip()] = value.strip()
            if parsed:
                documents.append(parsed)
        return documents

    def _build_answer(self, question: str, documents: list[dict[str, str]]) -> str:
        if not documents:
            return "데이터에서 확인되지 않습니다."

        top_doc = documents[0]
        apartment_name = top_doc.get("아파트명", "")
        transport = top_doc.get("교통", "")
        price = top_doc.get("가격", "")
        location = top_doc.get("위치", "")
        description = top_doc.get("설명", "")
        cited_doc = top_doc.get("문서ID", "")

        normalized = question.lower()
        if any(keyword in normalized for keyword in ["역", "지하철", "노선", "환승"]):
            return f"{apartment_name}의 교통 정보는 다음과 같습니다. {transport} 근거 문서: {cited_doc}".strip()
        if any(keyword in normalized for keyword in ["가격", "공급가", "분양가", "평당"]):
            return f"{apartment_name}의 가격 정보는 다음과 같습니다. {price} 근거 문서: {cited_doc}".strip()
        if any(keyword in normalized for keyword in ["위치", "주소", "어디"]):
            return f"{apartment_name}는 {location}에 위치합니다. 근거 문서: {cited_doc}".strip()
        return f"{description} 근거 문서: {cited_doc}".strip()
