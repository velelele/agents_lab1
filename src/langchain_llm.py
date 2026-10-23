import os
from pathlib import Path
from typing import List, Optional, Any, Dict, Iterator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

class LocalLlamaChatModel(BaseChatModel):
    model_path: str = ""
    model_alias: str = "qwen"
    temperature: float = 0.2
    max_tokens: int = 512
    n_ctx: int = 2048
    n_threads: int = 4
    llm_instance: Optional[Any] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.llm_instance and self.model_path and os.path.exists(self.model_path):
            import llama_cpp
            self.llm_instance = llama_cpp.Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False
            )

    @property
    def _llm_type(self) -> str:
        return f"llama_cpp_{self.model_alias}"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        formatted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted.append({"role": "assistant", "content": msg.content})
            else:
                formatted.append({"role": "user", "content": str(msg.content)})
        return formatted

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        if not self.llm_instance:
            last_msg = messages[-1].content if messages else ""
            reply = f"Привет! Я принял твоё сообщение: '{last_msg}'."
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

        dict_msgs = self._convert_messages(messages)
        res = self.llm_instance.create_chat_completion(
            messages=dict_msgs,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            stream=False
        )
        content = res["choices"][0]["message"].get("content", "") or ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        if not self.llm_instance:
            yield ChatGenerationChunk(message=AIMessageChunk(content="Ответ локальной модели."))
            return

        dict_msgs = self._convert_messages(messages)
        stream = self.llm_instance.create_chat_completion(
            messages=dict_msgs,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            stream=True
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {}).get("content", "")
            if delta:
                if run_manager:
                    run_manager.on_llm_new_token(delta)
                yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
