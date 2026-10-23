import time
import re
import json
from typing import List, Dict, Any, Optional, Callable, Generator
from src.providers.base import BaseLLMProvider
from src.tools import execute_tool
from src.parser import ToolCallParser
from src.prompt_templates import build_system_prompt
from src.file_fact_memory import FileFactMemory
from src.display_cell import render_cell

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

    def build_context_system_prompt(self) -> str:
        facts_text = self.memory.format_facts_string()
        prompt = self.base_system_prompt
        prompt += f"\n\nИзвестные долговременные факты о пользователе из файла памяти:\n{facts_text}\n"
        prompt += "Учитывай эти факты при ответах. Если пользователь спрашивает о себе, используй эти данные точно.\n"
        return prompt

    def run(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        on_token_callback: Optional[Callable[[str], None]] = None,
        on_tool_callback: Optional[Callable[[str, Dict[str, Any], str], None]] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        history = list(conversation_history or [])

        self.memory.save_context({"input": user_message}, {"output": ""})

        system_prompt = self.build_context_system_prompt()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        step_logs = []
        final_answer = ""
        current_step = 0
        stop_tokens = ["Observation:", "\nObservation:", "[Observation:"]

        while current_step < self.max_steps:
            current_step += 1
            step_record = {
                "step": current_step,
                "input_messages_count": len(messages),
                "action": None,
                "args": {},
                "observation": None,
                "raw_response": ""
            }

            raw_chunks = []
            stream_buffer = ""
            answer_streaming = False
            action_before_answer = False
            answer_marker = re.compile(r"(?im)^[ \t]*Answer[ \t]*:[ \t]*")
            action_marker = re.compile(r"(?im)^[ \t]*(?:Action|Вызов|Tool)[ \t]*:")

            if self.provider:
                for chunk in self.provider.stream_generate(
                    messages=messages,
                    temperature=0.2,
                    max_tokens=getattr(self.provider, "max_tokens", 600),
                    stop=stop_tokens
                ):
                    raw_chunks.append(chunk)
                    if not on_token_callback:
                        continue
                    if answer_streaming:
                        on_token_callback(chunk)
                        continue

                    stream_buffer += chunk
                    answer_match = answer_marker.search(stream_buffer)
                    action_match = action_marker.search(stream_buffer)
                    if action_match and (not answer_match or action_match.start() < answer_match.start()):
                        action_before_answer = True
                    if answer_match and not action_before_answer:
                        answer_streaming = True
                        answer_text = stream_buffer[answer_match.end():]
                        if answer_text:
                            on_token_callback(answer_text)

                raw_response = "".join(raw_chunks)
            else:
                lowered = user_message.lower()
                name = self.memory.facts.get("name", "")
                greeting_name = f", {name}" if name else ""

                if "биткоин" in lowered or "btc" in lowered or "bitcoin" in lowered:
                    raw_response = 'Action: get_crypto_price(crypto="биткоин")'
                elif "эфириум" in lowered or "eth" in lowered or "эфир" in lowered:
                    raw_response = 'Action: get_crypto_price(crypto="эфириум")'
                elif "сбер" in lowered or "сбербанк" in lowered:
                    raw_response = 'Action: get_stock_price(ticker="сбер")'
                elif "газпром" in lowered:
                    raw_response = 'Action: get_stock_price(ticker="газпром")'
                elif "яндекс" in lowered:
                    raw_response = 'Action: get_stock_price(ticker="яндекс")'
                elif "погода" in lowered:
                    raw_response = 'Action: get_weather(city="москва")'
                elif ("как меня зовут" in lowered or "мое имя" in lowered) and ("где я живу" in lowered or "город" in lowered):
                    city = self.memory.facts.get("city", "Москва")
                    raw_response = f"Answer: Тебя зовут {name}, и ты живёшь в городе {city}."
                elif "где я живу" in lowered or "в каком городе" in lowered or "мой город" in lowered:
                    city = self.memory.facts.get("city", "Москва")
                    raw_response = f"Answer: Ты живёшь в городе {city}{greeting_name}."
                elif "предпочтен" in lowered or "привычк" in lowered or "кофе" in lowered:
                    prefs = self.memory.facts.get("preferences", "не указаны")
                    raw_response = f"Answer: Твои предпочтения{greeting_name}: {prefs}."
                elif "цел" in lowered or "интерес" in lowered or "чем я интересуюсь" in lowered:
                    goals = self.memory.facts.get("goals", "не указаны")
                    raw_response = f"Answer: Твои интересы и цели{greeting_name}: {goals}."
                elif "стек" in lowered:
                    stack = self.memory.facts.get("stack", "")
                    raw_response = f"Answer: Твой стек{greeting_name}: {stack}."
                elif "как меня зовут" in lowered or "кто я" in lowered:
                    raw_response = f"Answer: Тебя зовут {name}." if name else "Answer: Я пока не знаю твоего имени."
                elif "зовут" in lowered:
                    raw_response = f"Answer: Приятно познакомиться{greeting_name}! Я сохранил это в памяти."
                else:
                    raw_response = f"Answer: Понял тебя{greeting_name}. Чем ещё могу помочь?"

            step_record["raw_response"] = raw_response

            func_name, kwargs, thought, answer = ToolCallParser.parse(raw_response)

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

                step_record["action"] = func_name
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

                if not self.provider:
                    final_answer = observation
                    break

                rendered_args = ", ".join(
                    f"{key}={json.dumps(value, ensure_ascii=False)}"
                    for key, value in kwargs.items()
                )
                assistant_msg = f"Action: {func_name}({rendered_args})"

                messages.append({"role": "assistant", "content": assistant_msg})
                completed = "\n".join(
                    f"- {item['action']}({json.dumps(item['args'], ensure_ascii=False, sort_keys=True)}): {item['observation']}"
                    for item in step_logs if item.get("action")
                )
                messages.append({
                    "role": "user",
                    "content": (
                        f"Исходный запрос пользователя: {user_message}\n"
                        f"Уже выполненные действия и результаты:\n{completed}\n"
                        "Не повторяй завершённые вызовы. Если все запрошенные факты уже получены — сразу дай короткий Answer и перенеси значения из результатов точно."
                    ),
                })
                continue
            else:
                step_logs.append(step_record)
                if answer:
                    final_answer = answer
                else:
                    clean_text = raw_response.strip()
                    if "Thought:" in clean_text and "\n" in clean_text:
                        lines = clean_text.splitlines()
                        filtered = [l for l in lines if not l.strip().startswith("Thought:")]
                        final_answer = "\n".join(filtered).strip() or clean_text
                    else:
                        final_answer = clean_text

                if on_token_callback and not answer_streaming and final_answer:
                    on_token_callback(final_answer)
                break

        if not final_answer and step_logs:
            last_resp = step_logs[-1]["raw_response"]
            final_answer = last_resp.strip()

        elapsed = time.time() - start_time
        model_name = self.provider.model_name if self.provider else "LocalFastFallback"

        return {
            "user_message": user_message,
            "steps": step_logs,
            "final_answer": final_answer,
            "total_time": elapsed,
            "model": model_name
        }
