import re
import json
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

FACT_KEYS = {"name", "city", "interests", "goals", "preferences"}

EXTRACTION_SYSTEM_PROMPT = """Ты — модуль долговременной памяти личного ассистента.
Извлеки только факты, которые пользователь явно сообщил о себе:
- name: имя
- city: город проживания
- interests: интересы пользователя
- goals: цели пользователя
- preferences: привычки и предпочтения

Не делай предположений и не запоминай факты из вопросов как сведения о пользователе.
Верни только JSON с найденными фактами или пустой объект {}.
"""


def rule_based_extract(text: str) -> Dict[str, Any]:
    extracted = {}
    cleaned = text.strip()

    name_match = re.search(r"(?:меня зовут|мо[её] имя|зовут меня)\s+([А-ЯЁA-Z][а-яёa-z-]+)", cleaned, re.IGNORECASE)
    if name_match:
        extracted["name"] = name_match.group(1).capitalize()
    elif re.fullmatch(r"[А-ЯЁA-Z][а-яёa-z-]+", cleaned) and cleaned.casefold() not in {"привет", "здравствуйте", "спасибо", "пока", "да", "нет"}:
        extracted["name"] = cleaned.capitalize()

    city_match = re.search(
        r"(?:живу|проживаю)\s+в\s+([А-ЯЁA-Zа-яёa-z]+(?:-[А-ЯЁA-Zа-яёa-z]+)*)",
        cleaned,
        re.IGNORECASE,
    )
    if city_match:
        city = city_match.group(1)
        aliases = {"москве": "Москва", "москвы": "Москва", "петербурге": "Санкт-Петербург", "санкт-петербурге": "Санкт-Петербург", "казани": "Казань", "екатеринбурге": "Екатеринбург"}
        extracted["city"] = aliases.get(city.casefold(), city.capitalize())

    patterns = [
        ("interests", r"(?:я\s+)?(интересуюсь|мне\s+интересны|мне\s+интересна|мне\s+интересен)\s+([^.!?;]+)"),
        ("goals", r"(?:моя\s+цель|я\s+планирую|я\s+хочу)\s*(?:это|[:—-])?\s+([^.!?;]+)"),
        ("preferences", r"(?:я\s+)?(люблю|предпочитаю|мне\s+нравится)\s+([^.!?;]+)"),
    ]
    for key, pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            extracted[key] = match.group(0).strip().rstrip(", ")

    return extracted


class FactExtractor:
    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm

    def extract(self, user_text: str) -> Dict[str, Any]:
        result = {}
        if self.llm is not None:
            try:
                response = self.llm.invoke([
                    SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
                    HumanMessage(content=f"Сообщение пользователя:\n{user_text}"),
                ])
                raw_text = response.content if hasattr(response, "content") else str(response)
                json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, dict):
                        for key, value in parsed.items():
                            if key in FACT_KEYS and value and isinstance(value, (str, int, float)):
                                value_str = str(value).strip()
                                if value_str and value_str.casefold() not in ["null", "none", "unknown", "нет"]:
                                    result[key] = value_str
            except Exception:
                pass

        for key, value in rule_based_extract(user_text).items():
            if not result.get(key):
                result[key] = value
        return result
