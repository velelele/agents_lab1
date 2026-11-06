import time
import re
import json
from typing import List, Dict, Any, Optional, Callable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch

from src.providers.base import BaseLLMProvider
from src.tools import execute_tool
from src.parser import ToolCallParser
from src.prompt_templates import build_system_prompt
from src.file_fact_memory import FileFactMemory, FACT_KEY_LABELS
from src.display_cell import render_cell
from src.langchain_llm import LocalLlamaChatModel
from src.dialogue_memory import DialogueMemory
from src.city_tools import SaveUserCity, extract_requested_city
from src.request_router import classify_request, weather_city


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
        show_cell: bool = True,
        window_size: int = 2
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

        if provider is not None:
            self.llm = LocalLlamaChatModel(
                model_alias=getattr(provider, "_model_alias", "local"),
                provider=provider,
                temperature=0.2
            )
        else:
            self.llm = LocalLlamaChatModel(model_alias="qwen")

        self.dialogue_memory = DialogueMemory(self.llm if provider is not None else None, window_size=window_size)
        self.city_tool = SaveUserCity(memory=self.memory)
        self.request_branch = RunnableBranch(
            (lambda x: x.get("request_type") == "weather", RunnableLambda(lambda x: f"Запрос о погоде. Используй get_weather и город: {x.get('city', '')}.")),
            (lambda x: x.get("request_type") == "finance", RunnableLambda(lambda _: "Запрос о финансах. Используй подходящий инструмент для акций, криптовалюты или курса.")),
            (lambda x: x.get("request_type") == "news", RunnableLambda(lambda _: "Для новостей нужен подключённый источник.")),
            (lambda x: x.get("request_type") == "mixed", RunnableLambda(lambda _: "Запрос содержит несколько категорий данных.")),
            RunnableLambda(lambda _: "Ответь на запрос, используя сохранённые факты, если они нужны.")
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}\n\nФакты о пользователе из памяти:\n{facts}\n\nМаршрут запроса: {guidance}\n\n{context}"),
            ("human", "{user_message}")
        ])

        self.chain = (
            {
                "facts": RunnableLambda(lambda _: self.memory.format_facts_string()),
                "system_prompt": RunnableLambda(lambda _: self.base_system_prompt),
                "context": RunnableLambda(lambda x: x.get("context", "")),
                "guidance": RunnableLambda(lambda x: x["guidance"]),
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
                if k != "name" and k in FACT_KEY_LABELS:
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

        request_type = classify_request(user_message)
        requested_city = extract_requested_city(user_message)
        if request_type == "general" and not requested_city and "?" not in user_message:
            self.memory.save_context({"input": user_message}, {"output": ""})
        self.memory.load_from_file()
        city = weather_city(user_message, self.memory.facts.get("city", "")) if request_type == "weather" else ""
        guidance = self.request_branch.invoke({"request_type": request_type, "city": city})

        early_answer = None
        if requested_city:
            try:
                early_answer = self.city_tool.invoke({"city": requested_city})
            except Exception as exc:
                early_answer = f"Не удалось сохранить город: {exc}"
        elif request_type == "news":
            early_answer = "Источник новостей пока не подключён, поэтому получить актуальные новости я не могу."
        elif request_type == "mixed":
            early_answer = "Пожалуйста, задайте вопросы о погоде, финансах и новостях отдельно: сейчас я обрабатываю одну категорию за раз."
        elif request_type == "weather" and not city:
            early_answer = "Укажите город или сохраните его командой: Запомни мой город: Екатеринбург."

        if early_answer is not None:
            if on_token_callback:
                on_token_callback(early_answer)
            self.dialogue_memory.save_context(user_message, early_answer)
            return {
                "user_message": user_message,
                "steps": [],
                "final_answer": early_answer,
                "total_time": time.time() - start_time,
                "request_type": request_type,
                "memory": self.dialogue_memory.load_memory_variables(),
                "model": self.llm._llm_type,
            }

        context_str = self.dialogue_memory.format_context()
        if conversation_history and not context_str:
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
                "context": current_context,
                "guidance": guidance
            })

            func_name, kwargs, thought, answer = ToolCallParser.parse(raw_response)

            step_record = {
                "step": current_step,
                "action": func_name,
                "args": kwargs,
                "observation": None,
                "raw_response": raw_response
            }

            allowed_tools = {
                "weather": {"get_weather"},
                "finance": {"get_stock_price", "get_crypto_price", "get_exchange_rate"},
            }
            if request_type in allowed_tools and func_name not in allowed_tools[request_type]:
                step_record["action"] = None
                step_logs.append(step_record)
                final_answer = "Не удалось получить данные для этого запроса. Уточните вопрос и попробуйте ещё раз."
                break

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
                if func_name == "save_user_city":
                    observation = "Чтобы сохранить город, напишите: Запомни мой город: название города."
                else:
                    if func_name == "get_weather" and request_type == "weather":
                        kwargs = {"city": city}
                        step_record["args"] = kwargs
                    observation = execute_tool(func_name, **kwargs)
                    if func_name in ("get_weather", "get_exchange_rate", "get_crypto_price", "get_stock_price"):
                        observation += " (демонстрационные данные)"
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

        self.dialogue_memory.save_context(user_message, final_answer)

        elapsed = time.time() - start_time
        return {
            "user_message": user_message,
            "steps": step_logs,
            "final_answer": final_answer,
            "total_time": elapsed,
            "request_type": request_type,
            "memory": self.dialogue_memory.load_memory_variables(),
            "model": self.llm._llm_type if hasattr(self, "llm") else "LangChainChain"
        }
