import time
import re
import json
from typing import List, Dict, Any, Optional, Callable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from src.providers.base import BaseLLMProvider
from src.tools import execute_tool
from src.parser import ToolCallParser
from src.prompt_templates import build_system_prompt
from src.file_fact_memory import FileFactMemory
from src.display_cell import render_cell
from src.langchain_llm import LocalLlamaChatModel


class ToolAgent:
    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        max_steps: int = 5,
        system_prompt: Optional[str] = None,
        verbose: bool = True,
        session_id: str = "default_user",
        file_path: str = "memory.json",
        memory: Optional[FileFactMemory] = None,
        show_cell: bool = True
    ):
        self.provider = provider
        self.max_steps = max_steps
        self.base_system_prompt = system_prompt or build_system_prompt()
        self.verbose = verbose
        self.show_cell = show_cell

        if memory is not None:
            self.memory = memory
        else:
            self.memory = FileFactMemory(session_id=session_id, file_path=file_path, show_service_cell=show_cell)

        if provider is not None and hasattr(provider, "model_path"):
            self.llm = LocalLlamaChatModel(
                model_path=str(provider.model_path),
                model_alias=getattr(provider, "_model_alias", "local"),
                temperature=0.2
            )
        else:
            self.llm = LocalLlamaChatModel(model_alias="qwen")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}\n\nФакты о пользователе из памяти:\n{facts}\n\n{context}"),
            ("human", "{user_message}")
        ])

        self.chain = (
            {
                "facts": RunnableLambda(lambda _: self.memory.format_facts_string()),
                "system_prompt": RunnableLambda(lambda _: self.base_system_prompt),
                "context": RunnableLambda(lambda x: x.get("context", "")),
                "user_message": RunnableLambda(lambda x: x["user_message"]),
            }
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

    def get_greeting(self) -> str:
        known = self.memory.facts
        name = known.get("name")

        summary_payload = {
            "Сессия": self.memory.session_id,
            "Файл памяти": self.memory.file_path,
            "Пользователь": name if name else "Не указан",
            "Сохранённые факты": self.memory.format_facts_string()
        }
        if self.show_cell:
            render_cell(
                title="СТАРТ СЕССИИ: ЗАГРУЗКА ПАМЯТИ ИЗ ФАЙЛА",
                content=summary_payload,
                border_style="bold blue",
                subtitle=f"Файл: {self.memory.file_path}"
            )

        if name:
            known_items = []
            for k, v in known.items():
                if k != "name":
                    known_items.append(f"{k}: {v}")
            details = f" ({', '.join(known_items)})" if known_items else ""
            return f"Привет, {name}! Рад снова быть на связи{details}. Чем могу помочь тебе сегодня?"
        else:
            return "Привет! Я твой личный ассистент. Я запоминаю важные факты о тебе в файл memory.json, умею узнавать цены криптовалют (BTC, ETH), акций российских компаний (Сбер, Газпром, Яндекс) и помогать с задачами. Как к тебе обращаться?"

    def run(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        on_token_callback: Optional[Callable[[str], None]] = None,
        on_tool_callback: Optional[Callable[[str, Dict[str, Any], str], None]] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()

        self.memory.save_context({"input": user_message}, {"output": ""})

        context_str = ""
        if conversation_history:
            history_lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in conversation_history[-4:]]
            context_str = "Предыдущий контекст:\n" + "\n".join(history_lines)

        step_logs = []
        final_answer = ""
        current_step = 0
        current_input = user_message
        current_context = context_str

        while current_step < self.max_steps:
            current_step += 1

            raw_response = self.chain.invoke({
                "user_message": current_input,
                "context": current_context
            })

            func_name, kwargs, thought, answer = ToolCallParser.parse(raw_response)

            step_record = {
                "step": current_step,
                "action": func_name,
                "args": kwargs,
                "observation": None,
                "raw_response": raw_response
            }

            if func_name:
                if "_arg0" in kwargs and len(kwargs) == 1:
                    clean_arg = kwargs["_arg0"]
                    if func_name == "get_weather":
                        kwargs = {"city": clean_arg}
                    elif func_name == "get_exchange_rate":
                        kwargs = {"pair": clean_arg}
                    elif func_name == "get_crypto_price":
                        kwargs = {"crypto": clean_arg}
                    elif func_name == "get_stock_price":
                        kwargs = {"ticker": clean_arg}
                    elif func_name == "calculate":
                        kwargs = {"expression": clean_arg}
                    elif func_name == "get_device_status":
                        kwargs = {"device": clean_arg}

                step_record["args"] = kwargs
                observation = execute_tool(func_name, **kwargs)
                step_record["observation"] = observation

                if self.show_cell:
                    render_cell(
                        title=f"ВЫЗОВ ИНСТРУМЕНТА: {func_name.upper()}",
                        content={"Инструмент": func_name, "Параметры": kwargs, "Результат": observation},
                        border_style="bold yellow"
                    )

                if on_tool_callback:
                    on_tool_callback(func_name, kwargs, observation)

                step_logs.append(step_record)
                final_answer = observation
                break
            else:
                step_logs.append(step_record)
                if answer:
                    final_answer = answer
                else:
                    final_answer = raw_response.strip()
                break

        if on_token_callback and final_answer:
            on_token_callback(final_answer)

        elapsed = time.time() - start_time
        return {
            "user_message": user_message,
            "steps": step_logs,
            "final_answer": final_answer,
            "total_time": elapsed,
            "model": self.llm._llm_type if hasattr(self, "llm") else "LangChainChain"
        }
