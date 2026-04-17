from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi_app import create_app  # noqa: E402
from query_service import (  # noqa: E402
    APARTMENT_NAME_COL,
    DATA_CUTOFF_COL,
    DESCRIPTION_COL,
    DOC_ID_COL,
    EXCLUSIVE_AREA_COL,
    HOUSEHOLDS_COL,
    LIFESTYLE_SUMMARY_COL,
    MOVE_IN_YEAR_COL,
    POLICY_SUMMARY_COL,
    PRICE_COL,
    REGION_COLUMNS,
    SUBWAY_DISTANCE_COL,
    SUBWAY_NAME_COL,
    SUBWAY_SUMMARY_COL,
)


def build_context(*, fallback_on_low_score: bool = True) -> dict:
    detailed_row = {
        DOC_ID_COL: "APT-001",
        APARTMENT_NAME_COL: "헬리오시티",
        REGION_COLUMNS[0]: "서울",
        REGION_COLUMNS[1]: "송파구",
        REGION_COLUMNS[2]: "가락동",
        EXCLUSIVE_AREA_COL: 84.0,
        PRICE_COL: 120000,
        HOUSEHOLDS_COL: 1000,
        MOVE_IN_YEAR_COL: 2018,
        SUBWAY_NAME_COL: "송파역",
        SUBWAY_DISTANCE_COL: 300,
        SUBWAY_SUMMARY_COL: "8호선",
        LIFESTYLE_SUMMARY_COL: "생활 인프라 우수",
        POLICY_SUMMARY_COL: "정책 특이사항 없음",
        DESCRIPTION_COL: "송파구 대표 단지입니다.",
        DATA_CUTOFF_COL: "2026-04-01",
        "병원_접근지표": 9,
        "병원_비교요약": "병원 접근성이 좋은 편입니다.",
        "공원_접근지표": 8,
        "공원_비교요약": "공원 접근성이 좋은 편입니다.",
        "학교_접근지표": 7,
        "학교_비교요약": "학교 접근성이 무난합니다.",
        "교통_비교요약": "지하철 접근성이 좋은 편입니다.",
        "의료시설_요약": "대형 병원 접근 가능",
        "통근통학_요약": "통근 여건 양호",
        "면적대": "중소형",
    }
    source_df = pd.DataFrame([{DOC_ID_COL: "APT-001"}])
    detailed_df = pd.DataFrame([detailed_row])
    knowledge_df = pd.DataFrame(
        [
            {
                "term": "분양가상한제",
                "definition": "분양가를 일정 기준에 따라 제한하는 제도입니다.",
                "related_dataset_fields": POLICY_SUMMARY_COL,
                "caution": "지역별 적용 여부를 함께 확인해야 합니다.",
            }
        ]
    )
    return {
        "source_df": source_df,
        "detailed_df": detailed_df,
        "knowledge_df": knowledge_df,
        "backend": "mock",
        "model_id": "gemma4_2b",
        "model_config": {"runtime": "mock", "device_map": "cpu", "hf_model_id": "mock-model"},
        "generation_config": {"request_timeout_seconds": 20},
        "top_k": 3,
        "retrieval_threshold": 1.0,
        "fallback_on_low_score": fallback_on_low_score,
        "runtime_probe": {
            "runtime": "mock",
            "device_map": "cpu",
            "model_source": "mock-model",
            "last_load_ms": 0,
            "request_timeout_seconds": 20,
            "probe_error": "",
        },
        "runtime_meta": "backend=mock model_id=gemma4_2b",
        "region_groups": [{"label": "서울권", "districts": ["송파구"]}],
        "last_generation_probe": {
            "ready": False,
            "status": "not_run",
            "backend": "mock",
            "model_id": "gemma4_2b",
            "device_map": "cpu",
            "model_source": "mock-model",
            "load_runtime_ms": 0,
            "generate_ms": 0,
            "probe_total_ms": 0,
            "text_preview": "",
            "error": "",
            "checked_at": 0,
        },
        "generation_probe_in_progress": False,
        "generation_lock": threading.Lock(),
        "server_started_at": "2026-04-17 00:00:00",
        "pid": 12345,
        "port": 8788,
    }


class FastAPITest(unittest.TestCase):
    def make_client(self, *, fallback_on_low_score: bool = True) -> TestClient:
        app = create_app(build_context(fallback_on_low_score=fallback_on_low_score))
        return TestClient(app)

    def test_empty_question_returns_422(self) -> None:
        with self.make_client() as client:
            response = client.post("/api/ask", json={"question": "   "})
        self.assertEqual(response.status_code, 422)

    def test_meta_query_contract(self) -> None:
        with self.make_client() as client:
            response = client.post("/api/ask", json={"question": "데이터 기준 알려줘"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query_type"], "DATA_SCOPE_META")
        self.assertEqual(payload["match_status"], "KNOWN")

    def test_knowledge_query_contract(self) -> None:
        with self.make_client() as client:
            response = client.post("/api/ask", json={"question": "분양가상한제 뜻이 뭐야"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query_type"], "REAL_ESTATE_KNOWLEDGE")
        self.assertEqual(payload["match_status"], "KNOWN")

    def test_apartment_fact_lookup_contract(self) -> None:
        with self.make_client() as client:
            response = client.post("/api/ask", json={"question": "헬리오시티 지하철 정보 알려줘"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query_type"], "APARTMENT_FACT_LOOKUP")
        self.assertEqual(payload["match_status"], "EXACT_MATCH")

    def test_structured_recommendation_contract(self) -> None:
        with self.make_client() as client:
            response = client.post("/api/ask", json={"question": "송파구에서 13억 이하 아파트 추천해줘"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query_type"], "RECOMMEND_STRUCTURED")
        self.assertEqual(payload["match_status"], "EXACT_MATCH")

    def test_comparative_recommendation_contract(self) -> None:
        with self.make_client() as client:
            response = client.post("/api/ask", json={"question": "병원 비교해줘"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query_type"], "RECOMMEND_COMPARATIVE")
        self.assertEqual(payload["match_status"], "EXACT_MATCH")

    def test_unsupported_comparative_keeps_unknown(self) -> None:
        with self.make_client() as client:
            response = client.post("/api/ask", json={"question": "병원 좋은 곳 비교해줘"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query_type"], "RECOMMEND_COMPARATIVE")
        self.assertEqual(payload["match_status"], "UNKNOWN")

    def test_no_match_contract(self) -> None:
        with self.make_client() as client:
            response = client.post("/api/ask", json={"question": "송파구에서 1억 이하 아파트 추천해줘"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query_type"], "RECOMMEND_STRUCTURED")
        self.assertEqual(payload["match_status"], "NO_MATCH")

    def test_status_endpoint(self) -> None:
        with self.make_client() as client:
            response = client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("backend", payload)
        self.assertIn("generation_ready", payload)
        self.assertIn("regions", payload)

    def test_check_rule_endpoint(self) -> None:
        with self.make_client() as client:
            response = client.post("/api/check-rule")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("answer_type", payload)
        self.assertIn("match_status", payload)

    def test_generation_ready_mock_backend(self) -> None:
        with self.make_client() as client:
            response = client.post("/api/check-generation-ready")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["status"], "unsupported")

    def test_busy_semantics_preserved(self) -> None:
        context = build_context(fallback_on_low_score=False)
        context["generation_lock"].acquire()
        try:
            app = create_app(context)
            with TestClient(app) as client:
                response = client.post("/api/ask", json={"question": "송파구 대표 아파트 특징 설명해줘"})
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["finish_reason"], "runtime_busy")
            self.assertIn("generation_runtime_busy", payload["limitations"])
        finally:
            context["generation_lock"].release()


if __name__ == "__main__":
    unittest.main()
