import re
import json
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

EXTRACTION_SYSTEM_PROMPT = """Ты — модуль извлечения долговременной памяти для личного ассистента.
Твоя задача — извлечь из сообщения пользователя любые новые факты о нём.

Возможные факты:
- name: имя пользователя (например "Матвей")
- role: профессия или род деятельности (например "Backend разработчик", "ML-инженер")
- stack: технологии, языки программирования, инструменты (например "Python, FastAPI, Docker")
- level: уровень или грейд (Junior, Middle, Senior, Lead)
- city: город проживания (например "Москва", "Санкт-Петербург")
- goals: цели или текущие задачи пользователя
- preferences: предпочтения пользователя или привычки

Верни ТОЛЬКО валидный JSON-объект с обнаруженными фактами.
Если фактов нет, верни пустой JSON-объект: {}
"""

def rule_based_extract(text: str) -> Dict[str, Any]:
    extracted = {}
    cleaned = text.strip()

    name_patterns = [
        r'(?:меня зовут|мое имя|моё имя|зовут|я)\s+([А-ЯЁA-Z][а-яёa-z]+)',
        r'^([А-ЯЁA-Z][а-яёa-z]+)$'
    ]
    for pattern in name_patterns:
        match = re.search(pattern, cleaned)
        if match:
            extracted["name"] = match.group(1).capitalize()
            break

    if "матвей" in cleaned.lower():
        extracted["name"] = "Матвей"

    level_patterns = [
        (r'\b(junior\s*\+|джуниор\s*\+|джун\s*\+)\b', "Junior+"),
        (r'\b(junior|джуниор|джун)\b', "Junior"),
        (r'\b(middle\s*\+|мидл\s*\+|миддл\s*\+)\b', "Middle+"),
        (r'\b(middle|мидл|миддл)\b', "Middle"),
        (r'\b(senior|сеньор|синиор|сеньер)\b', "Senior"),
        (r'\b(lead|тимлид|лид)\b', "Lead")
    ]
    for pattern, label in level_patterns:
        if re.search(pattern, cleaned, re.IGNORECASE):
            extracted["level"] = label
            break

    role_patterns = [
        (r'\b(backend|бэкенд|бекенд|бек)\b', "Backend разработчик"),
        (r'\b(frontend|фронтенд|фронт)\b', "Frontend разработчик"),
        (r'\b(fullstack|фулстек|фуллстек)\b', "Fullstack разработчик"),
        (r'\b(data\s*scientist|дата\s*саентист|ds)\b', "Data Scientist"),
        (r'\b(ml\s*engineer|ml-инженер|ml)\b', "ML Engineer"),
        (r'\b(devops|девопс)\b', "DevOps Engineer")
    ]
    for pattern, label in role_patterns:
        if re.search(pattern, cleaned, re.IGNORECASE):
            extracted["role"] = label
            break

    stack_keywords = ["python", "fastapi", "django", "flask", "docker", "postgres", "postgresql", "redis", "langchain", "sql", "git", "linux", "golang", "go", "react"]
    found_tech = []
    for tech in stack_keywords:
        if re.search(r'\b' + re.escape(tech) + r'\b', cleaned, re.IGNORECASE):
            found_tech.append(tech.title() if tech not in ["sql", "git"] else tech.upper())
    if found_tech:
        extracted["stack"] = ", ".join(found_tech)

    city_patterns = [
        r'(?:живу в|город|из)\s+([А-ЯЁ][а-яё\-]+)'
    ]
    for pattern in city_patterns:
        m = re.search(pattern, cleaned, re.IGNORECASE)
        if m:
            extracted["city"] = m.group(1).capitalize()
            break

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
                    HumanMessage(content=f"Сообщение пользователя:\n{user_text}")
                ])
                raw_text = response.content if hasattr(response, "content") else str(response)
                json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, dict):
                        for k, v in parsed.items():
                            if v and isinstance(v, (str, int, float)):
                                val_str = str(v).strip()
                                if val_str and val_str.lower() not in ["null", "none", "unknown", "нет"]:
                                    result[k] = val_str
            except Exception:
                pass

        rule_facts = rule_based_extract(user_text)
        for k, v in rule_facts.items():
            if k not in result or not result[k]:
                result[k] = v

        return result
