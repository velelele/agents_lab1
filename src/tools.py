"""
Hardcoded tool implementations and registry for the agent.
Requirements:
- Без function calling / tools calling / MCP API.
- Функции — захардкоженные константы (погода, время, курсы и т.п.).
"""
import ast
import operator
import re
from typing import Callable, Dict, Any

# Hardcoded constants database
WEATHER_DATABASE = {
    "москва": "+14°C, переменная облачность, ветер 4 м/с",
    "санкт-петербург": "+11°C, небольшой дождь, ветер 6 м/с",
    "новосибирск": "+7°C, пасмурно, без осадков",
    "екатеринбург": "+9°C, ясно, сухо",
    "сочи": "+21°C, солнечно, штиль",
    "казань": "+12°C, облачно, ветер 3 м/с",
    "лондон": "+15°C, туман, моросит дождь",
    "париж": "+18°C, ясно, тепло",
    "токио": "+22°C, солнечно",
    "нью-йорк": "+17°C, ветрено, солнечно"
}

RATES_DATABASE = {
    "usd": "1 USD = 92.40 RUB",
    "usd/rub": "1 USD = 92.40 RUB",
    "доллар": "1 USD = 92.40 RUB",
    "eur": "1 EUR = 100.85 RUB",
    "eur/rub": "1 EUR = 100.85 RUB",
    "евро": "1 EUR = 100.85 RUB",
    "cny": "1 CNY = 12.75 RUB",
    "cny/rub": "1 CNY = 12.75 RUB",
    "юань": "1 CNY = 12.75 RUB"
}

DEVICES_DATABASE = {
    "свет": "Свет в гостиной: ВКЛЮЧЕН, яркость 75%",
    "освещение": "Свет в гостиной: ВКЛЮЧЕН, яркость 75%",
    "кондиционер": "Кондиционер: ВКЛЮЧЕН, режим Охлаждение, целевая температура 21°C",
    "чайник": "Чайник: ВЫКЛЮЧЕН, температура воды 40°C, уровень воды 1.2 л",
    "пылесос": "Робот-пылесос: На зарядной станции, батарея 100%"
}


def get_weather(city: str) -> str:
    """Возвращает текущую погоду для указанного города."""
    normalized = city.strip().lower()
    for key, value in WEATHER_DATABASE.items():
        if key in normalized or normalized in key:
            return f"Погода в городе {city.strip()}: {value}"
    return f"Погода в городе {city.strip()}: +16°C, переменная облачность, давление 750 мм рт. ст."


def get_current_time(timezone: str = "Europe/Moscow") -> str:
    """Возвращает текущее системное время."""
    # Захардкоженное фиксированное эталонное время для стабильности тестов/инференса
    return "Текущее время: 12:30:00 (MSK, UTC+3), дата: 09 октября 2026 года"


def get_exchange_rate(pair: str) -> str:
    """Возвращает захардкоженный курс валюты (например: USD, EUR, CNY)."""
    normalized = pair.strip().lower()
    for key, val in RATES_DATABASE.items():
        if key in normalized:
            return val
    return f"Курс для пары '{pair}': 1 {pair.upper()} = 90.00 RUB"


def calculate(expression: str) -> str:
    """Безопасный калькулятор базовых арифметических выражений."""
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in allowed_operators:
                left = _eval(node.left)
                right = _eval(node.right)
                return allowed_operators[op_type](left, right)
            raise ValueError(f"Оператор {op_type} не поддерживается")
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in allowed_operators:
                return allowed_operators[op_type](_eval(node.operand))
            raise ValueError(f"Оператор {op_type} не поддерживается")
        else:
            raise ValueError("Недопустимое выражение")

    clean_expr = expression.strip().replace("^", "**").replace(",", ".")
    clean_expr = re.sub(r"[^\d+\-*/().% ]", "", clean_expr)
    try:
        parsed = ast.parse(clean_expr, mode='eval')
        res = _eval(parsed.body)
        return f"{clean_expr} = {res}"
    except Exception as e:
        return f"Ошибка вычисления выражения '{expression}': {e}"


def get_device_status(device: str) -> str:
    """Возвращает состояние устройства умного дома."""
    normalized = device.strip().lower()
    for key, val in DEVICES_DATABASE.items():
        if key in normalized:
            return val
    return f"Статус устройства '{device}': Устройство в сети, работает нормально"


# Реестр доступных инструментов
TOOLS_REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_weather": {
        "func": get_weather,
        "description": "get_weather(city=\"Название города\") — узнать погоду в городе",
        "params": ["city"]
    },
    "get_current_time": {
        "func": get_current_time,
        "description": "get_current_time() — узнать текущее время и дату",
        "params": []
    },
    "get_exchange_rate": {
        "func": get_exchange_rate,
        "description": "get_exchange_rate(pair=\"USD\") — получить курс валюты к рублю (USD, EUR, CNY)",
        "params": ["pair"]
    },
    "calculate": {
        "func": calculate,
        "description": "calculate(expression=\"2 + 2 * 2\") — посчитать математическое выражение",
        "params": ["expression"]
    },
    "get_device_status": {
        "func": get_device_status,
        "description": "get_device_status(device=\"кондиционер\") — проверить состояние прибора умного дома (передавай название устройства из запроса, например кондиционер, свет, чайник)",
        "params": ["device"]
    }
}


def get_tools_prompt_text() -> str:
    """Формирует текстовое описание инструментов для системного промта."""
    lines = ["Тебе доступны следующие инструменты (функции):"]
    for name, tool_info in TOOLS_REGISTRY.items():
        lines.append(f"- {tool_info['description']}")
    return "\n".join(lines)


def execute_tool(func_name: str, **kwargs) -> str:
    """Выполняет инструмент по имени и возвращает результат в виде строки."""
    if func_name not in TOOLS_REGISTRY:
        return f"Ошибка: неизвестная функция '{func_name}'. Доступны: {', '.join(TOOLS_REGISTRY.keys())}"

    func = TOOLS_REGISTRY[func_name]["func"]
    try:
        result = func(**kwargs)
        return str(result)
    except TypeError as e:
        # Если аргументы переданы не под теми именами или без имен
        try:
            # Попробуем передать первый позиционный аргумент
            if kwargs and len(kwargs) == 1:
                val = next(iter(kwargs.values()))
                return str(func(val))
            elif not kwargs:
                return str(func())
        except Exception:
            pass
        return f"Ошибка вызова {func_name}: неверные параметры {kwargs} ({e})"
    except Exception as e:
        return f"Ошибка при выполнении {func_name}: {e}"
