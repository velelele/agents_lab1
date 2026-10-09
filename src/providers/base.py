"""
Abstract base class for LLM providers.
"""
from abc import ABC, abstractmethod
from typing import Generator, List, Dict, Any, Optional


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate complete response synchronously."""
        pass

    @abstractmethod
    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """Stream response tokens sequentially."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the loaded model."""
        pass
