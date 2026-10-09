"""
Local Llama.cpp provider using llama-cpp-python.
Supports:
- Running GGUF models directly on CPU / GPU
- Synchronous generation and token-by-token streaming
- Custom stop tokens (e.g. 'Observation:' to prevent hallucinating tool outputs)
"""
from pathlib import Path
from typing import Generator, List, Dict, Optional, Union
import llama_cpp
from src.providers.base import BaseLLMProvider
from src.config import DEFAULT_CTX_SIZE, DEFAULT_N_THREADS, DEFAULT_TEMPERATURE


class LlamaCppProvider(BaseLLMProvider):
    def __init__(
        self,
        model_path: Union[str, Path],
        model_alias: str = "local-llama",
        n_ctx: int = DEFAULT_CTX_SIZE,
        n_threads: int = DEFAULT_N_THREADS,
        chat_format: Optional[str] = None,
        verbose: bool = False
    ):
        self._model_alias = model_alias
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Файл модели не найден по пути: {self.model_path}\n"
                f"Запустите скрипт загрузки: python scripts/download_models.py"
            )

        kwargs = {
            "model_path": str(self.model_path),
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "verbose": verbose
        }
        if chat_format:
            kwargs["chat_format"] = chat_format

        print(f"[+] Инициализация llama-cpp модели '{self._model_alias}' ({self.model_path.name})...")
        self.llm = llama_cpp.Llama(**kwargs)
        print(f"[OK] Модель '{self._model_alias}' успешно загружена в память.")

    @property
    def model_name(self) -> str:
        return f"LlamaCpp:{self._model_alias} ({self.model_path.name})"

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> str:
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            stream=False
        )
        return response["choices"][0]["message"].get("content", "") or ""

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        stream = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            stream=True
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {}).get("content", "")
            if delta:
                yield delta
