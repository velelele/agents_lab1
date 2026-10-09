"""Maple Preview provider using a local llama.cpp HTTP server."""
import atexit
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional

import requests

from src.config import BASE_DIR
from src.providers.base import BaseLLMProvider


class MapleServerProvider(BaseLLMProvider):
    """Run Maple-Preview through the official Windows CPU llama.cpp runtime."""

    def __init__(
        self,
        model_path: Path,
        model_alias: str = "maple",
        n_ctx: int = 4096,
        n_threads: int = 8,
        port: int = 18501,
        startup_timeout: int = 600,
        max_tokens: int = 1024,
        reasoning_budget: int = 128,
    ):
        self.model_path = Path(model_path)
        self.model_alias = model_alias
        self.max_tokens = max_tokens
        runtime_dir = BASE_DIR / ".runtime" / "llama-b11429"
        self.server_path = runtime_dir / "llama-server.exe"
        self.log_path = BASE_DIR / ".runtime" / "maple-server.log"
        self.base_url = f"http://127.0.0.1:{port}"
        self._log_file = None
        self._process = None
        self._closed = False

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Maple GGUF не найден: {self.model_path}")
        if not self.server_path.is_file():
            raise FileNotFoundError(
                f"llama-server не найден: {self.server_path}. "
                "Распакуйте официальный Windows CPU runtime llama.cpp в .runtime/llama-b11429."
            )

        command = [
            str(self.server_path), "--model", str(self.model_path),
            "--host", "127.0.0.1", "--port", str(port),
            "--ctx-size", str(n_ctx), "--threads", str(n_threads),
            "--n-gpu-layers", "0", "--jinja", "--reasoning-budget", str(reasoning_budget), "--seed", "42",
            "--alias", model_alias,
        ]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            command,
            cwd=str(runtime_dir),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                details = self._tail_log()
                self.close()
                raise RuntimeError(f"llama-server завершился при загрузке Maple. Лог:\n{details}")
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            details = self._tail_log()
            self.close()
            raise TimeoutError(f"llama-server не загрузил Maple за {startup_timeout} с. Лог:\n{details}")

        print(f"[OK] Maple-Preview загружен локально: {self.model_path.name}")
        atexit.register(self.close)

    def _tail_log(self) -> str:
        try:
            return "\n".join(
                self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            )
        except OSError:
            return "(лог недоступен)"

    @property
    def model_name(self) -> str:
        return f"llama.cpp-server:{self.model_alias} ({self.model_path.name})"

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> str:
        return "".join(self.stream_generate(messages, temperature, max_tokens, stop))

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        payload = {
            "model": self.model_alias,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop

        with requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=(10, None),
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=False):
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8", errors="replace")
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._log_file is not None:
            self._log_file.close()
