"""
ReAct Tool Agent implementation.
Requirements:
- На базе чата сделать агента, который парсит вызовы функций прямо из тела ответа.
- Без function calling, tools calling, MCP.
- Функции — захардкоженные константы (погода, время, курсы и т.п.).
- Поддержка стриминга и подробного логирования шагов.
"""
import time
import re
import json
from typing import List, Dict, Any, Optional, Callable, Generator
from src.providers.base import BaseLLMProvider
from src.tools import execute_tool
from src.parser import ToolCallParser
from src.prompt_templates import build_system_prompt


class ToolAgent:
    def __init__(
        self,
        provider: BaseLLMProvider,
        max_steps: int = 5,
        system_prompt: Optional[str] = None,
        verbose: bool = True
    ):
        self.provider = provider
        self.max_steps = max_steps
        self.system_prompt = system_prompt or build_system_prompt()
        self.verbose = verbose

    def run(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        on_token_callback: Optional[Callable[[str], None]] = None,
        on_tool_callback: Optional[Callable[[str, Dict[str, Any], str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the ReAct loop for a user query.
        Returns detailed execution log dictionary:
        {
            "user_message": ...,
            "steps": [ ... ],
            "final_answer": ...,
            "total_time": ...,
            "model": ...
        }
        """
        start_time = time.time()
        history = list(conversation_history or [])
        # Build working messages for the current ReAct cycle
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        # Include past turns
        messages.extend(history)
        # Add current user query
        messages.append({"role": "user", "content": user_message})

        step_logs = []
        final_answer = ""
        current_step = 0

        # Stop words to prevent model from hallucinating observations
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

            # Stream each model turn, but show only the final Answer to the user.
            # Action/Thought text stays hidden while the agent is deciding or calling tools.
            raw_chunks = []
            stream_buffer = ""
            answer_streaming = False
            action_before_answer = False
            answer_marker = re.compile(r"(?im)^[ \t]*Answer[ \t]*:[ \t]*")
            action_marker = re.compile(r"(?im)^[ \t]*(?:Action|Вызов|Tool)[ \t]*:")

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
            step_record["raw_response"] = raw_response

            # Parse action and answer
            func_name, kwargs, thought, answer = ToolCallParser.parse(raw_response)

            if func_name:
                # Handle positional args fallback
                if "_arg0" in kwargs and len(kwargs) == 1:
                    clean_arg = kwargs["_arg0"]
                    if func_name == "get_weather":
                        kwargs = {"city": clean_arg}
                    elif func_name == "get_exchange_rate":
                        kwargs = {"pair": clean_arg}
                    elif func_name == "calculate":
                        kwargs = {"expression": clean_arg}
                    elif func_name == "get_device_status":
                        kwargs = {"device": clean_arg}

                step_record["action"] = func_name
                step_record["args"] = kwargs

                # Execute tool
                observation = execute_tool(func_name, **kwargs)
                step_record["observation"] = observation

                if on_tool_callback:
                    on_tool_callback(func_name, kwargs, observation)

                # Record the observation before building the next model prompt.
                step_logs.append(step_record)

                # Append assistant action and observation to working dialog
                # We format this cleanly into the context
                # Keep only the parsed call in history if a model emitted extra lines/actions.
                rendered_args = ", ".join(
                    f"{key}={json.dumps(value, ensure_ascii=False)}"
                    for key, value in kwargs.items()
                )
                assistant_msg = f"Action: {func_name}({rendered_args})"

                # Append the single executed action and explicit progress to the ReAct dialog.
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
                        "Не повторяй завершённые вызовы. Если исходный запрос содержит другую явно запрошенную категорию, которой ещё нет в результатах, вызови только её инструмент. "
                        "Если все запрошенные факты уже получены — сразу дай короткий Answer и перенеси значения из результатов точно."
                    ),
                })

                continue

            else:
                # No tool called -> Model produced direct reasoning and answer
                step_logs.append(step_record)
                if answer:
                    final_answer = answer
                else:
                    # Clean up "Thought: ..." if present, or use full text
                    clean_text = raw_response.strip()
                    if "Thought:" in clean_text and "\n" in clean_text:
                        lines = clean_text.splitlines()
                        filtered = [l for l in lines if not l.strip().startswith("Thought:")]
                        final_answer = "\n".join(filtered).strip() or clean_text
                    else:
                        final_answer = clean_text

                # Models that omit the Answer: label still get a visible final reply.
                if on_token_callback and not answer_streaming and final_answer:
                    on_token_callback(final_answer)
                break

        if not final_answer and step_logs:
            # Fallback if loop ended after tools without explicit Answer prefix
            last_resp = step_logs[-1]["raw_response"]
            final_answer = last_resp.strip()

        elapsed = time.time() - start_time

        return {
            "user_message": user_message,
            "steps": step_logs,
            "final_answer": final_answer,
            "total_time": elapsed,
            "model": self.provider.model_name
        }
