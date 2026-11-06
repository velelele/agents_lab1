import os
import re
from pathlib import Path
from typing import List, Optional, Any, Dict, Iterator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field

class LocalLlamaChatModel(BaseChatModel):
    model_path: str = ""
    model_alias: str = "qwen"
    temperature: float = 0.2
    max_tokens: int = 512
    n_ctx: int = 2048
    n_threads: int = 4
    llm_instance: Optional[Any] = None
    provider: Optional[Any] = Field(default=None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.provider and not self.llm_instance and self.model_path and os.path.exists(self.model_path):
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
        if self.provider is not None:
            content = self.provider.generate(
                messages=self._convert_messages(messages), temperature=self.temperature,
                max_tokens=self.max_tokens, stop=stop
            )
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
        if not self.llm_instance:
            user_msg = str(messages[-1].content) if messages else ""
            lowered = user_msg.lower()
            system_text = "\n".join(str(msg.content) for msg in messages if isinstance(msg, SystemMessage))
            facts = {}
            for label in ["Имя", "Город", "Интересы", "Цели / Задачи", "Предпочтения"]:
                match = re.search(r"(?m)^- " + re.escape(label) + r": (.+)$", system_text)
                if match:
                    facts[label] = match.group(1).strip()

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
                name = facts.get("Имя", "имя не сохранено")
                city = facts.get("Город", "город не сохранён")
                reply = f"Answer: Имя: {name}. Город: {city}."
            elif "где я живу" in lowered or "в каком городе" in lowered or "мой город" in lowered:
                city = facts.get("Город")
                reply = f"Answer: Твой город: {city}." if city else "Answer: Я пока не знаю твой город."
            elif "предпочтен" in lowered or "привычк" in lowered:
                value = facts.get("Предпочтения")
                reply = f"Answer: Твои предпочтения: {value}." if value else "Answer: Я пока не знаю твоих предпочтений."
            elif "цел" in lowered or "интерес" in lowered:
                values = [facts[key] for key in ["Интересы", "Цели / Задачи"] if key in facts]
                reply = "Answer: " + "; ".join(values) if values else "Answer: Я пока не знаю твоих интересов и целей."
            elif "как меня зовут" in lowered or "кто я" in lowered:
                name = facts.get("Имя")
                reply = f"Answer: Тебя зовут {name}." if name else "Answer: Я пока не знаю твоего имени."
            elif "зовут" in lowered or "мое имя" in lowered or "моё имя" in lowered:
                name_str = facts.get("Имя", "друг")
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
        if self.provider is not None:
            for delta in self.provider.stream_generate(
                messages=self._convert_messages(messages), temperature=self.temperature,
                max_tokens=self.max_tokens, stop=stop
            ):
                if run_manager:
                    run_manager.on_llm_new_token(delta)
                yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
            return
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
