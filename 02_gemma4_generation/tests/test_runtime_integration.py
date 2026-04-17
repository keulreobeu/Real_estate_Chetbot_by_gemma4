from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_runtime import build_runtime_context  # noqa: E402
from fastapi_app import create_app  # noqa: E402


class RuntimeIntegrationTest(unittest.TestCase):
    def temp_dir(self) -> tempfile.TemporaryDirectory[str]:
        return tempfile.TemporaryDirectory(dir=ROOT / "tests")

    def write_csv(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8-sig")

    def test_build_runtime_context_with_real_files(self) -> None:
        context = build_runtime_context(backend="mock", model_id="gemma4_2b", port=8788)
        self.assertEqual(context["backend"], "mock")
        self.assertEqual(context["model_id"], "gemma4_2b")
        self.assertIn("source_df", context)
        self.assertIn("detailed_df", context)
        self.assertIn("knowledge_df", context)

    def test_fastapi_status_with_real_runtime_context(self) -> None:
        app = create_app(build_runtime_context(backend="mock", model_id="gemma4_2b", port=8788))
        with TestClient(app) as client:
            response = client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["backend"], "mock")
        self.assertEqual(payload["model_id"], "gemma4_2b")

    def test_missing_source_index_fails_fast(self) -> None:
        with self.temp_dir() as temp_dir:
            missing = Path(temp_dir) / "missing_source.csv"
            with patch("api_runtime.OUTPUT_SOURCE_INDEX", missing):
                with self.assertRaises(FileNotFoundError):
                    build_runtime_context(backend="mock", model_id="gemma4_2b", port=8788)

    def test_missing_main_dataset_fails_fast(self) -> None:
        with self.temp_dir() as temp_dir:
            temp_root = Path(temp_dir)
            existing_source = temp_root / "source.csv"
            missing_main = temp_root / "missing_main.csv"
            self.write_csv(existing_source, "문서ID\nAPT-001\n")
            with patch("api_runtime.OUTPUT_SOURCE_INDEX", existing_source), patch("api_runtime.INPUT_MAIN", missing_main):
                with self.assertRaises(FileNotFoundError):
                    build_runtime_context(backend="mock", model_id="gemma4_2b", port=8788)

    def test_missing_knowledge_base_fails_fast(self) -> None:
        with self.temp_dir() as temp_dir:
            temp_root = Path(temp_dir)
            existing_source = temp_root / "source.csv"
            existing_main = temp_root / "main.csv"
            missing_knowledge = temp_root / "missing_knowledge.csv"
            self.write_csv(existing_source, "문서ID\nAPT-001\n")
            self.write_csv(existing_main, "문서ID\nAPT-001\n")
            with (
                patch("api_runtime.OUTPUT_SOURCE_INDEX", existing_source),
                patch("api_runtime.INPUT_MAIN", existing_main),
                patch("api_runtime.INPUT_KNOWLEDGE", missing_knowledge),
            ):
                with self.assertRaises(FileNotFoundError):
                    build_runtime_context(backend="mock", model_id="gemma4_2b", port=8788)


if __name__ == "__main__":
    unittest.main()
