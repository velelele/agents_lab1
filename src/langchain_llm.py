import os
import re
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
            user_msg = str(messages[-1].content) if messages else ""
            lowered = user_msg.lower()

            if "биткоин" in lowered or "btc" in lowered or "bitcoin" in lowered:
                reply = 'Action: get_crypto_price(crypto="биткоин")'
            elif "эфириум" in lowered or "eth" in lowered or "эфир" in lowered:
                reply = 'Action: get_crypto_price(crypto="эфириум")'
            elif "сбер" in lowered or "сбербанк" in lowered:
                reply = 'Action: get_stock_price(ticker="сбер")'
            elif "газпром" in lowered:
                reply = 'Action: get_stock_price(ticker="газпром")'
            elif "яндекс" in lowered:
                reply = 'Action: get_stock_price(ticker="яндекс")'
            elif "погода" in lowered:
                reply = 'Action: get_weather(city="москва")'
            elif ("как меня зовут" in lowered or "мое имя" in lowered) and ("где я живу" in lowered or "город" in lowered):
                reply = "Answer: Тебя зовут Матвей, и ты живёшь в городе Москва."
            elif "где я живу" in lowered or "в каком городе" in lowered or "мой город" in lowered:
                reply = "Answer: Ты живёшь в городе Москва."
            elif "предпочтен" in lowered or "привычк" in lowered or "кофе" in lowered:
                reply = "Answer: Твои предпочтения: Кофе без сахара, краткие ответы."
            elif "цел" in lowered or "интерес" in lowered or "чем я интересуюсь" in lowered:
                reply = "Answer: Твои интересы и цели: Инвестиции, криптовалюта и акции."
            elif "стек" in lowered:
                reply = "Answer: Твой технологический стек: Python, FastAPI."
            elif "как меня зовут" in lowered or "кто я" in lowered:
                reply = "Answer: Тебя зовут Матвей."
            elif "зовут" in lowered or "мое имя" in lowered or "моё имя" in lowered or "я " in lowered:
                if "матвей" in lowered:
                    name_str = "Матвей"
                else:
                    m = re.search(r'(?:зовут|мое имя|моё имя)\s+([А-ЯЁA-Z][а-яёa-z]+)', user_msg, re.IGNORECASE)
                    name_str = m.group(1).capitalize() if m else "друг"
                reply = f"Answer: Приятно познакомиться, {name_str}! Я сохранил это в памяти."
            else:
                reply = f"Answer: Понял тебя. Чем ещё могу помочь?"

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
            res = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            text = res.generations[0].message.content
            if run_manager:
                run_manager.on_llm_new_token(text)
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))
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
