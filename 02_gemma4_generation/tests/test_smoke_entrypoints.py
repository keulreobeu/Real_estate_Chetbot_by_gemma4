from __future__ import annotations

import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def wait_for_json(url: str, timeout_seconds: int = 10) -> str:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return response.read().decode("utf-8")
        except Exception as exc:  # pragma: no cover
            last_error = str(exc)
            time.sleep(0.5)
    raise AssertionError(f"failed to fetch {url}: {last_error}")


class SmokeEntrypointsTest(unittest.TestCase):
    def spawn_and_check(self, command: list[str], url: str) -> str:
        process = subprocess.Popen(command, cwd=ROOT)
        try:
            return wait_for_json(url)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def test_fastapi_mock_status_smoke(self) -> None:
        port = pick_free_port()
        content = self.spawn_and_check(
            [
                PYTHON,
                ".\\02_gemma4_generation\\fastapi_app.py",
                "--backend",
                "mock",
                "--model",
                "gemma4_2b",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            f"http://127.0.0.1:{port}/api/status",
        )
        self.assertIn('"backend":"mock"', content.replace(" ", ""))

    def test_web_mvp_mock_status_smoke(self) -> None:
        port = pick_free_port()
        content = self.spawn_and_check(
            [
                PYTHON,
                ".\\02_gemma4_generation\\demo_chatbot_web_mvp.py",
                "--backend",
                "mock",
                "--model",
                "gemma4_2b",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            f"http://127.0.0.1:{port}/api/status",
        )
        self.assertIn('"backend"', content)


if __name__ == "__main__":
    unittest.main()
